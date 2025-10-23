#!/usr/bin/env python3
"""
Plot mean travel time vs timestep for each run folder under a generic 'data' directory,
labeling each series with "<N> veh – <route_key>".

- Looks for run folders directly inside BASE_DIR (no deep recursion).
- For labels, prefers run_summary.txt:
    target=<N>
    route_key=<routeX>
  Falls back to parsing folder names like: veh<N>_route<k>
- Reads per-step meanTravelTime from summary.xml; if absent, derives per-step mean
  travel time by grouping tripinfo.xml arrivals.

Outputs:
  BASE_DIR/mean_travel_time_by_step.png
  BASE_DIR/mean_travel_time_by_step.csv

Author: ChatGPT (for Anastasia)
Date: 2025-10-21
"""

from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import csv
import re
from math import inf, ceil

# ===========================
#          PARAMS
# ===========================
BASE_DIR = Path("../runs_out")

RUN_SUMMARY_FILE = "run_summary.txt"
SUMMARY_FILE = "summary.xml"
TRIPINFO_FILE = "tripinfo.xml"

PLOT_PATH = BASE_DIR / "mean_travel_time_by_step.png"
CSV_PATH = BASE_DIR / "mean_travel_time_by_step.csv"
ALIGN_STEPS = True   # align x-axis across runs

# Colormap (fallback if custom colors not defined)
COLORMAP_NAME = "Set1"

# ===========================
#       CUSTOM COLORS
# ===========================
CUSTOM_COLORS = [
    "firebrick",  # red-brown
    "teal",       # blue-green
    "peru",       # earthy orange-brown
]

# Subplot layout preferences
SUBPLOT_MAX_COLS = 3
FIGSIZE_PER_COL = 5.0
FIGSIZE_PER_ROW = 3.5

# ===========================
#       PARSERS
# ===========================

def read_label_from_run_summary(run_folder: Path) -> tuple[str, str] | tuple[None, None]:
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
        return None, None
    return (target, route) if (target is not None and route is not None) else (None, None)


def infer_label_from_folder_name(name: str) -> tuple[str, str] | tuple[None, None]:
    m = re.search(r"veh\s*(\d+)", name, flags=re.IGNORECASE)
    n = m.group(1) if m else None
    r = None
    m2 = re.search(r"(route\s*\d+)", name, flags=re.IGNORECASE)
    if m2:
        r = m2.group(1).lower().replace(" ", "")
    return (n, r) if (n and r) else (None, None)


def make_series_label(target_str: str | None, route_key: str | None, folder_name: str) -> str:
    if target_str and route_key:
        return f"{target_str} veh – {route_key}"
    if target_str:
        return f"{target_str} veh"
    if route_key:
        return f"{route_key}"
    return folder_name


def parse_summary_mean_tt(summary_path: Path):
    if not summary_path.exists():
        return [], []
    steps, mean_tt = [], []
    for _, elem in ET.iterparse(str(summary_path)):
        if elem.tag == "step":
            t = elem.get("time")
            mtt = elem.get("meanTravelTime")
            if t is not None and mtt is not None:
                try:
                    steps.append(float(t))
                    mean_tt.append(float(mtt))
                except ValueError:
                    pass
        elem.clear()
    return steps, mean_tt


def parse_tripinfo_mean_tt_per_arrival(tripinfo_path: Path):
    if not tripinfo_path.exists():
        return [], []
    sums, counts = defaultdict(float), defaultdict(int)
    for _, elem in ET.iterparse(str(tripinfo_path)):
        if elem.tag == "tripinfo":
            dep, arr, dur = elem.get("depart"), elem.get("arrival"), elem.get("duration")
            try:
                travel = float(dur) if dur is not None else (float(arr) - float(dep))
                bin_t = int(round(float(arr)))
                sums[bin_t] += travel
                counts[bin_t] += 1
            except Exception:
                pass
        elem.clear()
    if not counts:
        return [], []
    times_sorted = sorted(counts.keys())
    steps, mean_tt = [], []
    for t in times_sorted:
        steps.append(t)
        mean_tt.append(sums[t] / counts[t])
    return steps, mean_tt


# ===========================
#        MAIN LOGIC
# ===========================

def _extract_route_num(route_key: str | None) -> float:
    if not route_key:
        return inf
    m = re.search(r"(\d+)", route_key)
    return int(m.group(1)) if m else inf


def collect_run_series(base_dir: Path):
    series = OrderedDict()
    if not base_dir.exists():
        print(f"[WARN] Base dir not found: {base_dir.resolve()}")
        return series

    subdirs = [p for p in sorted(base_dir.iterdir()) if p.is_dir()]
    if not subdirs:
        print(f"[WARN] No subfolders under: {base_dir.resolve()}")
        return series

    for sub in subdirs:
        target_str, route_key = read_label_from_run_summary(sub)
        if not (target_str and route_key):
            t2, r2 = infer_label_from_folder_name(sub.name)
            target_str = target_str or t2
            route_key = route_key or r2
        label = make_series_label(target_str, route_key, sub.name)

        summary_path = sub / SUMMARY_FILE
        tripinfo_path = sub / TRIPINFO_FILE

        steps, mean_tt = parse_summary_mean_tt(summary_path)
        source = "summary.xml"
        if not steps:
            steps, mean_tt = parse_tripinfo_mean_tt_per_arrival(tripinfo_path)
            source = "tripinfo.xml (derived)"

        if steps:
            veh_num = int(target_str) if (target_str and target_str.isdigit()) else inf
            route_num = _extract_route_num(route_key)
            series[label] = (steps, mean_tt, veh_num, route_num)
            print(f"[OK] {sub.name} → {label}: {len(steps)} points from {source}")
        else:
            print(f"[SKIP] {sub.name}: no usable data (missing/empty {SUMMARY_FILE} and {TRIPINFO_FILE})")

    return series


def _sorted_labels(series: OrderedDict[str, tuple]) -> list[str]:
    return sorted(series.keys(), key=lambda k: (series[k][2], series[k][3], k))


def write_csv(series: OrderedDict, csv_path: Path):
    if not series:
        return
    labels_sorted = _sorted_labels(series)

    if ALIGN_STEPS:
        all_steps = sorted(set(t for (steps, _, _, _) in series.values() for t in steps))
    else:
        all_steps = None

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if ALIGN_STEPS:
            header = ["timestep"] + labels_sorted
            writer.writerow(header)
            run_maps = {
                name: {int(round(t)): v for t, v in zip(series[name][0], series[name][1])}
                for name in labels_sorted
            }
            for t in all_steps:
                row = [int(round(t))]
                for name in labels_sorted:
                    v = run_maps[name].get(int(round(t)))
                    row.append("" if v is None else f"{v:.6f}")
                writer.writerow(row)
        else:
            writer.writerow(["run_label", "timestep", "mean_travel_time"])
            for name in labels_sorted:
                steps, vals = series[name][0], series[name][1]
                for t, v in zip(steps, vals):
                    writer.writerow([name, int(round(t)), f"{v:.6f}"])
    print(f"[OK] CSV written: {csv_path.resolve()}")


# ===========================
#        SUBPLOT PLOTTER
# ===========================

def plot_series_subplots(series: OrderedDict, plot_path: Path):
    if not series:
        print("[WARN] Nothing to plot.")
        return

    groups: dict[float | None, list[str]] = defaultdict(list)
    for label in series:
        veh_num = series[label][2]
        key = None if veh_num == inf else veh_num
        groups[key].append(label)

    known_counts = sorted(k for k in groups.keys() if k is not None)
    group_keys = known_counts + ([None] if None in groups else [])

    n_groups = len(group_keys)
    n_cols = min(SUBPLOT_MAX_COLS, n_groups) if n_groups > 0 else 1
    n_rows = ceil(n_groups / n_cols) if n_groups > 0 else 1
    figsize = (FIGSIZE_PER_COL * n_cols, FIGSIZE_PER_ROW * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True)
    axes = axes.ravel()

    x_min, x_max = None, None
    if ALIGN_STEPS:
        all_steps = [t for (steps, _, _, _) in series.values() for t in steps]
        if all_steps:
            x_min, x_max = (min(all_steps), max(all_steps))

    for ax_idx, key in enumerate(group_keys):
        ax = axes[ax_idx]
        labels_in_group = sorted(
            groups[key],
            key=lambda k: (series[k][3], k)
        )

        # >>> UPDATED: use custom colors if defined
        if 'CUSTOM_COLORS' in globals() and CUSTOM_COLORS:
            colors = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(labels_in_group))]
        else:
            cmap = cm.get_cmap(COLORMAP_NAME, len(labels_in_group) or 1)
            colors = [cmap(i) for i in range(len(labels_in_group))]

        for i, label in enumerate(labels_in_group):
            steps, vals = series[label][0], series[label][1]
            if not steps:
                continue
            xs, ys = zip(*sorted(zip(steps, vals)))
            ax.plot(xs, ys, label=label, color=colors[i])

        title = f"{int(key)} veh" if key is not None else "Unknown veh"
        ax.set_title(title)
        ax.set_xlabel("Simulation timestep (s)")
        ax.set_ylabel("Mean travel time (s)")
        ax.grid(True, linewidth=0.3)
        if labels_in_group:
            ax.legend(title="Run", loc="best", fontsize=8)

        if ALIGN_STEPS and x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)

    for j in range(len(group_keys), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Mean travel time per timestep — grouped by vehicle count", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plot_path, dpi=150)
    print(f"[OK] Plot written: {plot_path.resolve()}")


def main():
    series = collect_run_series(BASE_DIR)
    write_csv(series, CSV_PATH)
    plot_series_subplots(series, PLOT_PATH)


if __name__ == "__main__":
    main()
