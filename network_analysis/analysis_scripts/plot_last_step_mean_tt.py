#!/usr/bin/env python3
"""
Plot mean travel time (last timestep) vs. number of vehicles,
with one color per route_key (using your custom colors).

Outputs:
  BASE_DIR/last_step_mean_travel_time_by_route.csv
  BASE_DIR/last_step_mean_travel_time_by_route.png
  BASE_DIR/last_step_mean_travel_time_by_route_xlim2000.png
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import csv
import re
from collections import defaultdict
from math import inf

# ===========================
#          PARAMS
# ===========================
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
BASE_DIR = (SCRIPT_DIR / "../runs_out").resolve()

RUN_SUMMARY_FILE = "run_summary.txt"
SUMMARY_FILE = "summary.xml"

PNG_PATH = BASE_DIR / "last_step_mean_travel_time_by_route.png"
PNG_PATH_ZOOM = BASE_DIR / "last_step_mean_travel_time_by_route_xlim2000.png"
CSV_PATH = BASE_DIR / "last_step_mean_travel_time_by_route.csv"

# ===========================
#       CUSTOM COLORS
# ===========================
CUSTOM_COLORS = [
    "firebrick",  # red-brown
    "teal",       # blue-green
    "peru",       # earthy orange-brown
]

# ===========================
#      LABEL UTILITIES
# ===========================
def read_label_from_run_summary(run_folder: Path):
    """Extract target and route_key from run_summary.txt if available."""
    p = run_folder / RUN_SUMMARY_FILE
    if not p.exists():
        return None, None
    target, route = None, None
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("target="):
                try:
                    target = str(int(s.split("=", 1)[1]))
                except Exception:
                    pass
            elif s.startswith("route_key="):
                route = s.split("=", 1)[1].strip()
    except Exception:
        pass
    return (target, route) if (target and route) else (None, None)


def infer_label_from_folder_name(name: str):
    m = re.search(r"veh\s*(\d+)", name, flags=re.IGNORECASE)
    n = m.group(1) if m else None
    r = None
    m2 = re.search(r"(route\s*\d+)", name, flags=re.IGNORECASE)
    if m2:
        r = m2.group(1).lower().replace(" ", "")
    return (n, r) if (n and r) else (None, None)


def _extract_route_num(route_key: str | None):
    if not route_key:
        return inf
    m = re.search(r"(\d+)", route_key)
    return int(m.group(1)) if m else inf


# ===========================
#      SUMMARY PARSER
# ===========================
def last_step_mean_tt(summary_path: Path):
    """Return (last_timestep, meanTravelTime) from summary.xml"""
    if not summary_path.exists():
        return None, None
    last_t, last_m = None, None
    try:
        for _, elem in ET.iterparse(str(summary_path)):
            if elem.tag == "step":
                t = elem.get("time")
                mtt = elem.get("meanTravelTime")
                if t is not None and mtt is not None:
                    try:
                        last_t = float(t)
                        last_m = float(mtt)
                    except ValueError:
                        pass
            elem.clear()
    except ET.ParseError:
        return None, None
    return (last_t, last_m) if last_t is not None and last_m is not None else (None, None)


# ===========================
#         MAIN LOGIC
# ===========================
def collect_data(base_dir: Path):
    """Collect final timestep mean travel time per (veh, route)."""
    data = []
    subdirs = [p for p in sorted(base_dir.iterdir()) if p.is_dir()]

    for sub in subdirs:
        t_str, r_key = read_label_from_run_summary(sub)
        if not (t_str and r_key):
            t2, r2 = infer_label_from_folder_name(sub.name)
            t_str = t_str or t2
            r_key = r_key or r2

        summary_path = sub / SUMMARY_FILE
        last_t, last_m = last_step_mean_tt(summary_path)
        if last_t is None or last_m is None:
            print(f"[SKIP] {sub.name}: missing summary.xml or no valid data")
            continue

        veh = int(t_str) if (t_str and t_str.isdigit()) else None
        route = r_key or "unknown"
        data.append((veh, route, last_m))
        print(f"[OK] {sub.name} → veh={veh}, route={route}, meanTT={last_m:.2f}")

    return data


def write_csv(data, path: Path):
    if not data:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vehicles", "route_key", "mean_travel_time_last_step_s"])
        for veh, route, mean_tt in sorted(data, key=lambda x: (x[0], route_key_sort(x[1]))):
            w.writerow([veh, route, f"{mean_tt:.6f}"])
    print(f"[OK] CSV written: {path}")


def route_key_sort(route_key: str):
    m = re.search(r"(\d+)", route_key)
    return int(m.group(1)) if m else 9999


# ===========================
#         PLOT
# ===========================
def set_paper_style():
    """Matplotlib settings for cleaner, paper-ready figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_data(data, path_full: Path, path_zoom: Path):
    if not data:
        print("[WARN] No data to plot.")
        return

    set_paper_style()

    # Group by route
    grouped = defaultdict(list)
    for veh, route, mean_tt in data:
        grouped[route].append((veh, mean_tt))

    for route in grouped:
        grouped[route].sort(key=lambda x: x[0])

    routes = sorted(grouped.keys(), key=route_key_sort)

    def make_plot(xlim=None, save_path=None):
        fig, ax = plt.subplots(figsize=(6.6, 4.4))

        for i, route in enumerate(routes):
            color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
            xs = [v for v, _ in grouped[route]]
            ys = [m for _, m in grouped[route]]

            ax.plot(
                xs, ys,
                marker="o",
                label=route,
                color=color,
                linewidth=1.8,
                markersize=4.5,
            )

        ax.set_xlabel("Number of vehicles")
        ax.set_ylabel("Mean travel time at final timestep [s]")
        #ax.set_title("Final-step mean travel time vs. demand", pad=10)

        # Cleaner grid
        ax.grid(True, which="major")
        ax.minorticks_on()
        ax.grid(True, which="minor", linewidth=0.3, alpha=0.18)

        # Remove top/right spines for a cleaner paper style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend
        leg = ax.legend(
            title="Route",
            frameon=False,
            loc="best",
            ncol=1,
            handlelength=2.2,
        )

        if xlim is not None:
            ax.set_xlim(0, xlim)

        # Optional: add a small margin so points do not touch borders
        ax.margins(x=0.02, y=0.05)

        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Plot written: {save_path}")

    make_plot(xlim=None, save_path=path_full)
    make_plot(xlim=300, save_path=path_zoom)

# ===========================
#         RUN
# ===========================
def main():
    print(f"[INFO] Using BASE_DIR: {BASE_DIR}")
    data = collect_data(BASE_DIR)
    write_csv(data, CSV_PATH)
    plot_data(data, PNG_PATH, PNG_PATH_ZOOM)


if __name__ == "__main__":
    main()
