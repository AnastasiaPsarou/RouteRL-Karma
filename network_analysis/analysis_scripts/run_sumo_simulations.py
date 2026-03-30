#!/usr/bin/env python3
"""
Parallel SUMO simulations (parameter block defined on top, no CLI args)

Author: Anastasia Psarou
Date: 2025-10-20
"""

import time
import socket
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import xml.etree.ElementTree as ET
import shutil

# ============================================================
#                  EXPERIMENT PARAMETERS
# ============================================================

# Path to your SUMO configuration file (.sumocfg)
CONFIG_FILE = "../network_base/handmade_simulation.sumocfg"

# Template .rou file to modify per run
ROU_TEMPLATE = "../network_base/handmade_routes.rou.xml"

# Flow ID to modify in the .rou file
FLOW_ID = "flow_am_peak"

# Vehicle counts for each run (cycled if fewer than RUNS)
COUNTS = [1, 10, 20, 50, 70, 80, 100, 150, 180, 200, 220, 230, 240, 250, 300]#, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000]

# SUMO / execution settings
RUNS = 15           # how many total runs
PARALLEL = 15       # how many in parallel
STEP_LIMIT = None  # optional cap on simulation steps (None = until finished)
USE_GUI = False     # False = headless (recommended)
SEED_START = None  # None = SUMO default seed
EXTRA_SUMO_ARGS: List[str] = []  # e.g. ["--scale", "1.0"]
SIM_END = 1500
DEPARTURE_TILL_EPISODE = 100

# --- Route variants you want to test (label -> edges string) ---
ROUTE_VARIANTS = {
    "route0": "E0 E20 E20.444 E20.444.91 E17 E17.200 E17.400 E17.600",
    "route1": "E0 E2 E18 E18.118 E19 E17 E17.200 E17.400 E17.600",
    "route2": "E0 E7 E8 E9 E10 E11 E12 E13 E14 E15 E16 E17 E17.200 E17.400 E17.600",
}

# Choose which route each run uses (cycled if fewer than RUNS)
ROUTE_SEQUENCE = ["route2"]   # will cycle for RUNS > 3

# ============================================================

try:
    import traci
    from sumolib import checkBinary
except Exception:
    raise RuntimeError(
        "Could not import 'traci' or 'sumolib'. "
        "Install them via `pip install traci sumolib` and ensure SUMO is installed."
    )

# ---------- Helpers ----------

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def parse_summary_loaded(summary_path: Path) -> Optional[int]:
    if not summary_path.exists():
        return None
    loaded = None
    for _, elem in ET.iterparse(str(summary_path)):
        if elem.tag == "step":
            val = elem.get("loaded") or elem.get("inserted")
            if val is not None:
                loaded = int(float(val))
        elem.clear()
    return loaded


def get_sumo_binary(gui: bool) -> str:
    return checkBinary("sumo-gui" if gui else "sumo")


def read_sim_duration_from_cfg(cfg_path: Path) -> float:
    """Return (end - begin) in seconds from the .sumocfg file."""
    cfg_tree = ET.parse(str(cfg_path))
    cfg_root = cfg_tree.getroot()
    dur = None
    for time_tag in cfg_root.iter("time"):
        b = float(time_tag.find("begin").get("value", 0))
        e = float(time_tag.find("end").get("value", 0))
        dur = e - b
    return dur if (dur is not None and dur > 0) else 1500.0


def write_routes_variant(rou_template: Path, out_path: Path,
                         flow_id: str, number_value: int,
                         route_key: str, route_edges: str,
                         cfg_path: Path):
    """
    Create a per-run .rou from a template:

      - Remove ALL existing <route> elements.
      - Insert a single <route id=route_key edges="..."> BEFORE the first <flow>.
      - Set <flow id=flow_id> to use that route.
      - If target/duration <= 1 veh/s -> set probability.
        Else -> set vehsPerHour, removing conflicting attrs.
      - Align flow begin/end to match the sim time window from .sumocfg.
    """
    # ---- read sim duration (seconds) from .sumocfg ----
    cfg_tree = ET.parse(str(cfg_path))
    cfg_root = cfg_tree.getroot()
    sim_dur = None
    for time_tag in cfg_root.iter("time"):
        b = float(time_tag.find("begin").get("value", 0))
        e = float(time_tag.find("end").get("value", 0))
        sim_dur = e - b
    if not sim_dur or sim_dur <= 0:
        sim_dur = 100.0  # fallback

    sim_dur = DEPARTURE_TILL_EPISODE

    rate_per_sec = max(0.0, float(number_value) / float(sim_dur))
    use_probability = rate_per_sec <= 1.0
    vehs_per_hour = rate_per_sec * 3600.0

    # ---- load template ----
    tree = ET.parse(str(rou_template))
    root = tree.getroot()

    # 1) Remove ALL existing <route> elements (including '0_0_0')
    for child in list(root):
        if child.tag == "route":
            root.remove(child)

    # 2) Insert our route BEFORE the first <flow>
    new_route = ET.Element("route", id=route_key, edges=route_edges)
    insert_idx = 0
    found_flow = False
    for idx, child in enumerate(list(root)):
        if child.tag == "flow":
            insert_idx = idx
            found_flow = True
            break
    if found_flow:
        root.insert(insert_idx, new_route)
    else:
        root.append(new_route)

    # 3) Adjust the target flow
    flows = [el for el in root.iter("flow") if el.get("id") == flow_id]
    if not flows:
        raise ValueError(f"Flow id '{flow_id}' not found in {rou_template}")

    for f in flows:
        f.set("route", route_key)
        # Remove conflicting volume attributes
        for attr in ("number", "vehsPerHour", "period", "probability"):
            if attr in f.attrib:
                del f.attrib[attr]
        # Set appropriate volume attribute
        if use_probability:
            f.set("probability", f"{rate_per_sec:.6f}")
        else:
            f.set("vehsPerHour", f"{vehs_per_hour:.6f}")
        # Keep timing consistent
        f.set("begin", "0")
        f.set("end", str(int(sim_dur)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)


def write_per_run_sumocfg(base_cfg: Path, out_cfg: Path, extra_route_file: Optional[Path]) -> None:
    """
    Write a per-run .sumocfg into `out_cfg`, starting from `base_cfg`.
    We REPLACE <input><route-files value="..."> with only the generated route file
    and we also ABSOLUTIZE all input file paths (net-file, additional-files, etc.)
    so the per-run config is self-contained no matter the working directory.
    """
    REPLACE_ROUTE_FILES = True  # set False to append instead

    base_cfg = base_cfg.resolve()
    tree = ET.parse(str(base_cfg))
    root = tree.getroot()

    # Ensure <input> exists
    input_el = root.find("input")
    if input_el is None:
        input_el = ET.SubElement(root, "input")

    base_dir = base_cfg.parent

    def absolutize_attr(elem: Optional[ET.Element], attr_name: str, base_dir: Path):
        """
        Absolutize file path(s) in an attribute. Only split on commas.
        If there are no commas, treat the entire string as one path (spaces allowed).
        Writes back using commas.
        """
        if elem is None or attr_name not in elem.attrib:
            return
        val = elem.get(attr_name, "") or ""
        if "," in val:
            raw_tokens = val.split(",")
        else:
            raw_tokens = [val]
        abs_paths = []
        for t in raw_tokens:
            t = t.strip()
            if not t:
                continue
            p = Path(t)
            if not p.is_absolute():
                p = (base_dir / t).resolve()
            abs_paths.append(str(p))
        elem.set(attr_name, ",".join(abs_paths))

    # 1) route-files: replace and absolutize
    rf_el = input_el.find("route-files")
    if rf_el is None:
        rf_el = ET.SubElement(input_el, "route-files")

    if extra_route_file is not None:
        abs_route = str(extra_route_file.resolve())
        if REPLACE_ROUTE_FILES:
            rf_el.set("value", abs_route)
        else:
            current = rf_el.get("value", "") or ""
            toks = [current] if "," not in current else current.split(",")
            toks = [t for t in [*toks, abs_route] if t and t.strip()]
            rf_el.set("value", ",".join(dict.fromkeys([t.strip() for t in toks])))
    # Always normalize/absolutize whatever is there
    absolutize_attr(rf_el, "value", base_dir)

    # 2) Make ALL other input-file attributes absolute
    for tag in [
        "net-file",
        "additional-files",
        "gui-settings-file",
        "meandata-files",
        "taz-files",
        "weights-files",
        "polygon-files",
        "edge-files",
        "node-files",
        "type-files",
        "tls-files",
        # keep route-files too (already handled)
        "route-files",
    ]:
        el = input_el.find(tag)
        absolutize_attr(el, "value", base_dir)

    # (Optional) absolutize any gui/other top-level file refs people sometimes place outside <input>
    # Example: <gui-settings value="..."> in some configs
    for tag in ["gui-settings"]:
        el = root.find(tag)
        absolutize_attr(el, "value", base_dir)

    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_cfg), encoding="utf-8", xml_declaration=True)


def start_sumo_process(
    sumo_binary: str,
    config_file: str,
    port: int,
    seed: Optional[int],
    step_limit: Optional[int],
    extra_args: Optional[list] = None,
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    cmd = [
        sumo_binary,
        "-c", config_file,
        "--remote-port", str(port),
        "--duration-log.disable", "true",
        "--no-warnings", "true",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if step_limit is not None:
        cmd += ["--step-log.period", "0"]
    if extra_args:
        cmd += extra_args
    # Write the command into cwd or into the run folder if we can deduce it
    try:
        run_dir = Path(config_file).parent
        with open(run_dir / "sumo_command.txt", "w", encoding="utf-8") as fcmd:
            fcmd.write(" ".join(cmd) + "\n")
    except Exception:
        pass
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)


def parse_tripinfo_count(tripinfo_path: Path) -> int:
    if not tripinfo_path.exists():
        return 0
    count = 0
    for _, elem in ET.iterparse(str(tripinfo_path)):
        if elem.tag == "tripinfo":
            count += 1
        elem.clear()
    return count


# ---------- Core worker ----------

def run_single_simulation(
    run_id: int,
    config_file: str,
    gui: bool,
    seed: Optional[int],
    step_limit: Optional[int],
    extra_sumo_args: Optional[list],
    rou_template: Optional[str],
    flow_id: Optional[str],
    number_value: Optional[int],
    route_key_arg: Optional[str],   # which route to use for this run
) -> Dict[str, Any]:
    """Run one SUMO instance and return result summary."""
    if route_key_arg is None:
        route_key_arg = "route0"
    if route_key_arg not in ROUTE_VARIANTS:
        raise ValueError(f"Unknown route key '{route_key_arg}'. Known: {list(ROUTE_VARIANTS)}")

    label = f"run{run_id}"
    port = find_free_port()
    base_cfg_path = Path(config_file).resolve()
    base_cfg_dir = base_cfg_path.parent
    working_dir = str(base_cfg_dir)  # keep base cfg folder as CWD for relative paths in cfg

    # Per-run folder named by veh count and route key
    folder_name = f"veh{number_value}_{route_key_arg}" if number_value is not None else label
    # >>> Make absolute to avoid .. segments confusion
    run_out = (base_cfg_dir / f"../runs_out/{folder_name}").resolve()
    prefix_dir = run_out
    prefix_dir.mkdir(parents=True, exist_ok=True)
    # >>> Absolute output prefix
    output_prefix = prefix_dir.as_posix() + "/"

    # Copy the base .sumocfg for reference (unaltered)
    try:
        shutil.copy2(str(base_cfg_path), str(prefix_dir / "simulation_base.sumocfg"))
    except Exception:
        pass

    # Generate per-run .rou with chosen route + probability
    generated_rou = None
    if rou_template and flow_id and number_value is not None:
        rou_template_p = Path(rou_template).resolve()
        generated_rou = prefix_dir / f"{Path(rou_template).stem}_{number_value}_{route_key_arg}.rou.xml"
        write_routes_variant(
            rou_template_p, generated_rou, FLOW_ID, int(number_value),
            route_key=route_key_arg,
            route_edges=ROUTE_VARIANTS[route_key_arg],
            cfg_path=base_cfg_path
        )

    # Build a per-run .sumocfg that injects the generated route file (absolute path)
    per_run_cfg = prefix_dir / f"{base_cfg_path.stem}_{number_value}_{route_key_arg}.sumocfg"
    write_per_run_sumocfg(base_cfg_path, per_run_cfg, generated_rou if generated_rou else None)

    # Build SUMO args
    extra = list(extra_sumo_args or [])
    extra += [
        "--output-prefix", output_prefix,
        "--tripinfo-output", "tripinfo.xml",
        "--device.tripinfo.probability", "1.0",
        "--tripinfo-output.write-unfinished", "true",
        "--statistic-output", "stats.xml",
        "--summary-output", "summary.xml",
    ]
    # NOTE: we do NOT pass --route-files here anymore; it is in the per-run sumocfg

    result = {
        "run_id": run_id,
        "seed": seed,
        "veh_target": number_value,
        "veh_count": None,
        "steps": 0,
        "duration_s": None,
        "ok": False,
        "error": None,
        "route_key": route_key_arg,
    }

    try:
        sumo_binary = get_sumo_binary(gui)
        # Use per-run config file
        proc = start_sumo_process(
            sumo_binary, str(per_run_cfg), port, seed, step_limit, extra, cwd=working_dir
        )

        # Connect TraCI (wait up to ~15s)
        connected = False
        for _ in range(150):
            time.sleep(0.1)
            if proc.poll() is not None:
                raise RuntimeError(proc.stderr.read())
            try:
                traci_connection = traci.connect(port=port, label=label)
                traci.switch(label)
                connected = True
                break
            except Exception:
                continue
        if not connected:
            raise RuntimeError("Timeout waiting for TraCI to connect")

        # Step simulation
        t0 = time.time()
        steps = 0
        try:
            while (step_limit is None or steps < step_limit) and traci.simulation.getTime() < SIM_END:
                traci.simulationStep()
                steps += 1
        finally:
            try:
                traci.close(False)
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()

        veh_count = parse_tripinfo_count(prefix_dir / "tripinfo.xml")
        loaded = parse_summary_loaded(prefix_dir / "summary.xml")

        result.update(
            veh_count=veh_count,
            steps=steps,
            duration_s=round(time.time() - t0, 3),
            ok=True,
        )

        # Also store the exact SUMO command line used (already written by start_sumo_process into the run folder)
        with open(prefix_dir / "run_summary.txt", "w", encoding="utf-8") as f:
            f.write(
                f"target={number_value}\n"
                f"route_key={route_key_arg}\n"
                f"route_edges={ROUTE_VARIANTS[route_key_arg]}\n"
                f"spawned_total={veh_count}  (tripinfo incl. unfinished)\n"
                f"loaded_total={loaded}      (summary.xml)\n"
                f"steps={steps}\n"
                f"duration_s={result['duration_s']}\n"
                f"per_run_sumocfg={per_run_cfg}\n"
                f"base_sumocfg_copy={prefix_dir / 'simulation_base.sumocfg'}\n"
            )

        return result

    except Exception as e:
        result["error"] = str(e)
        # Try to still write a summary with whatever exists
        try:
            veh_count = parse_tripinfo_count(prefix_dir / "tripinfo.xml")
            loaded = parse_summary_loaded(prefix_dir / "summary.xml")
            with open(prefix_dir / "run_summary.txt", "w", encoding="utf-8") as f:
                f.write(
                    f"target={number_value}\n"
                    f"route_key={route_key_arg}\n"
                    f"route_edges={ROUTE_VARIANTS[route_key_arg]}\n"
                    f"spawned_total={veh_count}\n"
                    f"loaded_total={loaded}\n"
                    f"per_run_sumocfg={per_run_cfg}\n"
                    f"base_sumocfg_copy={prefix_dir / 'simulation_base.sumocfg'}\n"
                    f"error={result['error']}\n"
                )
        except Exception:
            pass
        return result


# ---------- Main ----------

def main():
    jobs = []
    for i in range(RUNS):
        seed = (SEED_START + i) if SEED_START is not None else None
        number_value = COUNTS[i % len(COUNTS)]
        route_key = ROUTE_SEQUENCE[i % len(ROUTE_SEQUENCE)]
        jobs.append(dict(
            run_id=i,
            config_file=CONFIG_FILE,
            gui=USE_GUI,
            seed=seed,
            step_limit=STEP_LIMIT,
            extra_sumo_args=EXTRA_SUMO_ARGS,
            rou_template=ROU_TEMPLATE,
            flow_id=FLOW_ID,
            number_value=number_value,
            route_key_arg=route_key,
        ))

    print(f"Starting {RUNS} SUMO runs with up to {min(PARALLEL, RUNS)} in parallel...")
    results = []
    with ProcessPoolExecutor(max_workers=min(PARALLEL, RUNS)) as pool:
        futures = [pool.submit(run_single_simulation, **job) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            status = "OK" if res["ok"] else "FAIL"
            msg = (f"[{status}] run={res['run_id']} "
                   f"veh={res['veh_target']} route={res.get('route_key')} "
                   f"actual={res['veh_count']} steps={res['steps']} duration={res['duration_s']}s")
            if not res["ok"]:
                msg += f" error={res['error']}"
            print(msg)

    ok_count = sum(1 for r in results if r["ok"])
    print(f"\nCompleted {len(results)} runs: {ok_count} OK, {len(results) - ok_count} failed.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
