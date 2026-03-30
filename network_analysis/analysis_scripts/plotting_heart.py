#!/usr/bin/env python3
"""
Side-by-side publication-style figure with:

LEFT:
    Mean travel time at the final timestep vs. number of vehicles,
    one line per route_key.

RIGHT:
    SUMO network visualization where
    - color encodes lane length
    - line width encodes lane speed

Output:
    ../imgs/combined_travel_time_and_network.png
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import numpy as np
import re
from collections import defaultdict
from math import inf


# =========================================================
# USER PARAMETERS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# ----- Input paths -----
RUNS_BASE_DIR = (SCRIPT_DIR / "../runs_out").resolve()
NET_FILE = (SCRIPT_DIR / "../network_base/network.net.xml").resolve()

# ----- Output path -----
OUT_DIR = (SCRIPT_DIR / "../imgs").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "combined_travel_time_and_network.png"

# ----- Files inside each run folder -----
RUN_SUMMARY_FILE = "run_summary.txt"
SUMMARY_FILE = "summary.xml"

# ----- Left panel: travel-time plot -----
LEFT_XMAX = 305      # e.g. 300 for zoom, or None for full range
LEFT_TITLE = None       # e.g. "Travel time vs demand", or None
CUSTOM_COLORS = [
    "firebrick",  # red-brown
    "teal",       # blue-green
    "peru",       # earthy orange-brown
]

# ----- Right panel: network plot -----
INCLUDE_INTERNAL = True
SHOW_NETWORK_TITLE = None   # e.g. "Network", or None
LENGTH_CMAP = "viridis"

MIN_LINE_WIDTH = 0.8
MAX_LINE_WIDTH = 7.0

INTERNAL_LINE_WIDTH_SCALE = 0.8
INTERNAL_ALPHA = 0.25
NORMAL_ALPHA = 0.95

SHOW_END_CIRCLES = False
END_CIRCLE_SIZE = 2

# Optional display warping (turn off for faithful geometry)
WARP_LOWER_NETWORK = False
Y_SPLIT = 200.0
LOWER_X_SCALE = 0.65
LOWER_Y_SCALE = 0.75

# ----- Combined figure layout -----
FIG_WIDTH = 6
FIG_HEIGHT = 1.8
DPI = 400
WSPACE = 0.28
BOTTOM_MARGIN = 0.20

# ----- Typography / style -----
FONT_FAMILY = "DejaVu Sans"
BASE_FONT_SIZE = 8
TITLE_FONTSIZE = 8
LABEL_FONTSIZE = 6
TICK_FONTSIZE = 5
LEGEND_FONTSIZE = 5

# ----- Route labels on selected junctions -----
SHOW_ROUTE_LABELS = True
ROUTE_LABEL_FONTSIZE = 6
ROUTE_LABEL_WEIGHT = "bold"
ROUTE_LABEL_COLOR = "black"
ROUTE_LABEL_X_OFFSET_FRAC = 0.0
ROUTE_LABEL_Y_OFFSET_FRAC = 0.02

# placement: "above" or "below"
ROUTE_LABELS = {
    "J10": {"text": "Route 2", "placement": "above"},
    "J5":  {"text": "Route 0", "placement": "below"},
    "J4":  {"text": "Route 1", "placement": "above"},
}


# ----- Origin / destination markers on network -----
SHOW_OD_MARKERS = True
OD_MARKER_SIZE = 18
OD_MARKER_COLOR = "black"
OD_TEXT_FONTSIZE = 6
OD_TEXT_WEIGHT = "bold"
OD_TEXT_X_OFFSET_FRAC = 0.01
OD_TEXT_Y_OFFSET_FRAC = 0.02


# =========================================================
# GLOBAL MATPLOTLIB STYLE
# =========================================================
mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "legend.title_fontsize": LEGEND_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1,
    "lines.markersize": 3,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================
# LEFT PANEL HELPERS
# =========================================================
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


def route_key_sort(route_key: str):
    m = re.search(r"(\d+)", route_key)
    return int(m.group(1)) if m else 9999


def last_step_mean_tt(summary_path: Path):
    """Return (last_timestep, meanTravelTime) from summary.xml."""
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

    return (last_t, last_m) if (last_t is not None and last_m is not None) else (None, None)


def collect_travel_time_data(base_dir: Path):
    """Collect final timestep mean travel time per (veh, route)."""
    data = []
    if not base_dir.exists():
        raise FileNotFoundError(f"RUNS_BASE_DIR not found: {base_dir}")

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
        print(f"[OK] {sub.name} -> veh={veh}, route={route}, meanTT={last_m:.2f}")

    if not data:
        raise RuntimeError("No valid travel-time data found in RUNS_BASE_DIR.")

    return data


def plot_travel_time_panel(ax, data, xlim=None):
    grouped = defaultdict(list)
    for veh, route, mean_tt in data:
        if veh is None:
            continue
        grouped[route].append((veh, mean_tt))

    if not grouped:
        raise RuntimeError("No plottable travel-time data after grouping.")

    for route in grouped:
        grouped[route].sort(key=lambda x: x[0])

    routes = sorted(grouped.keys(), key=route_key_sort)

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
    ax.set_ylabel("Mean travel time \nat final timestep [s]")

    if LEFT_TITLE:
        ax.set_title(LEFT_TITLE, pad=8)

    ax.grid(True, which="major")
    ax.minorticks_on()
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Route",
        frameon=False,
        loc="best",
        ncol=1,
        handlelength=2.2,
    )

    if xlim is not None:
        ax.set_xlim(0, xlim)

    ax.margins(x=0.02, y=0.05)


# =========================================================
# RIGHT PANEL HELPERS
# =========================================================
def _dist2d(p0, p1):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return (dx * dx + dy * dy) ** 0.5


def add_route_labels(ax, junctions, x_center_global):
    """
    Add bold route labels at selected junctions.
    """
    if not SHOW_ROUTE_LABELS or not junctions:
        return

    # Quick lookup by junction id
    junction_by_id = {j["id"]: j for j in junctions}

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xoff = ROUTE_LABEL_X_OFFSET_FRAC * (xmax - xmin)
    yoff = ROUTE_LABEL_Y_OFFSET_FRAC * (ymax - ymin)

    for jid, spec in ROUTE_LABELS.items():
        if jid not in junction_by_id:
            print(f"[WARN] Route label skipped: junction id not found: {jid}")
            continue

        j = junction_by_id[jid]
        x, y = j["x"], j["y"]

        if WARP_LOWER_NETWORK:
            x, y = transform_point_for_display(
                x, y,
                y_split=Y_SPLIT,
                x_center_global=x_center_global,
                lower_x_scale=LOWER_X_SCALE,
                lower_y_scale=LOWER_Y_SCALE,
            )

        text = spec["text"]
        placement = spec.get("placement", "above").lower()

        if placement == "below":
            tx, ty = x + xoff, y - yoff
            va = "top"
        else:
            tx, ty = x + xoff, y + yoff
            va = "bottom"

        if text == "Route 0":

            ax.text(
                tx+55, ty-30,
                text,
                fontsize=ROUTE_LABEL_FONTSIZE,
                fontweight=ROUTE_LABEL_WEIGHT,
                color=ROUTE_LABEL_COLOR,
                ha="center",
                va=va,
                zorder=15,
                # optional white background for readability:
                # bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3),
            )
        if text == "Route 1":

            ax.text(
                tx-75, ty-12,
                text,
                fontsize=ROUTE_LABEL_FONTSIZE,
                fontweight=ROUTE_LABEL_WEIGHT,
                color=ROUTE_LABEL_COLOR,
                ha="center",
                va=va,
                zorder=15,
                # optional white background for readability:
                # bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3),
            )
        if text == "Route 2":

            ax.text(
                tx-4, ty+120,
                text,
                fontsize=ROUTE_LABEL_FONTSIZE,
                fontweight=ROUTE_LABEL_WEIGHT,
                color=ROUTE_LABEL_COLOR,
                ha="center",
                va=va,
                zorder=15,
                # optional white background for readability:
                # bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3),
            )


def add_origin_destination_markers(ax, junctions, x_center_global):
    """
    Mark the leftmost junction as Origin and the rightmost junction as Destination.
    """
    if not SHOW_OD_MARKERS or not junctions:
        return

    # Transform points if display warping is enabled
    pts = []
    for j in junctions:
        x, y = j["x"], j["y"]
        if WARP_LOWER_NETWORK:
            x, y = transform_point_for_display(
                x, y,
                y_split=Y_SPLIT,
                x_center_global=x_center_global,
                lower_x_scale=LOWER_X_SCALE,
                lower_y_scale=LOWER_Y_SCALE,
            )
        pts.append((j["id"], x, y))

    # Leftmost and rightmost by displayed x-coordinate
    origin = min(pts, key=lambda t: t[1])
    destination = max(pts, key=lambda t: t[1])

    _, ox, oy = origin
    _, dx, dy = destination

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xoff = OD_TEXT_X_OFFSET_FRAC * (xmax - xmin)
    yoff = OD_TEXT_Y_OFFSET_FRAC * (-130*ymax - ymin)

    # Origin
    ax.scatter(
        [ox], [oy],
        s=OD_MARKER_SIZE,
        c=OD_MARKER_COLOR,
        zorder=14,
        clip_on=False,
    )
    ax.text(
        ox + xoff, oy - yoff,
        "Origin",
        fontsize=OD_TEXT_FONTSIZE,
        fontweight=OD_TEXT_WEIGHT,
        ha="center",
        va="top",
        color=OD_MARKER_COLOR,
        zorder=11,
    )

    # Destination
    ax.scatter(
        [dx], [dy],
        s=OD_MARKER_SIZE,
        c=OD_MARKER_COLOR,
        zorder=14,
        clip_on=False,
    )
    ax.text(
        dx - xoff, dy - yoff,
        "Destination",
        fontsize=OD_TEXT_FONTSIZE,
        fontweight=OD_TEXT_WEIGHT,
        ha="center",
        va="top",
        color=OD_MARKER_COLOR,
        zorder=11,
    )


def read_junctions(net_path: Path):
    """
    Read junction (node) coordinates from a SUMO network.
    Excludes internal junctions.
    """
    root = ET.parse(net_path).getroot()
    junctions = []

    for j in root.findall("junction"):
        jid = j.get("id", "")
        jtype = j.get("type", "")

        # Skip internal junctions
        if jid.startswith(":") or jtype == "internal":
            continue

        x = j.get("x")
        y = j.get("y")
        if x is None or y is None:
            continue

        try:
            junctions.append({
                "id": jid,
                "x": float(x),
                "y": float(y),
            })
        except Exception:
            pass

    return junctions


def _parse_type_speeds(root):
    type_speed = {}
    for t in list(root.findall("type")) + list(root.findall("edgeType")):
        tid = t.get("id")
        sp = t.get("speed")
        if tid and sp:
            try:
                type_speed[tid] = float(sp)
            except Exception:
                pass
    return type_speed


def read_lanes_with_speed_and_length(net_path: Path, include_internal: bool = True):
    lanes_out = []
    root = ET.parse(net_path).getroot()
    type_speed = _parse_type_speeds(root)

    for edge in root.findall("edge"):
        is_internal = (edge.get("function") == "internal")
        if is_internal and not include_internal:
            continue

        edge_speed = None
        if edge.get("speed") is not None:
            try:
                edge_speed = float(edge.get("speed"))
            except Exception:
                edge_speed = None

        if edge_speed is None:
            etype = edge.get("type")
            if etype in type_speed:
                edge_speed = type_speed[etype]

        for lane in edge.findall("lane"):
            lane_id = lane.get("id", "")

            sp = lane.get("speed")
            if sp is not None:
                try:
                    speed = float(sp)
                except Exception:
                    speed = edge_speed if edge_speed is not None else 0.0
            else:
                speed = edge_speed if edge_speed is not None else 0.0

            pts = []
            shape = lane.get("shape", "")
            if shape:
                for p in shape.split():
                    try:
                        x, y = map(float, p.split(","))
                        pts.append((x, y))
                    except Exception:
                        pass

            ln_attr = lane.get("length")
            if ln_attr is not None:
                try:
                    length = float(ln_attr)
                except Exception:
                    length = 0.0
            else:
                length = 0.0
                if len(pts) >= 2:
                    for i in range(1, len(pts)):
                        length += _dist2d(pts[i - 1], pts[i])

            if len(pts) >= 2:
                lanes_out.append({
                    "points": pts,
                    "speed": float(speed),   # m/s
                    "length": float(length), # m
                    "edge_id": edge.get("id", ""),
                    "lane_id": lane_id,
                    "internal": is_internal,
                })

    return lanes_out


def compute_global_x_center(lanes, y_split):
    xs = []
    for ln in lanes:
        for x, y in ln["points"]:
            if y < y_split:
                xs.append(x)
    if not xs:
        return 0.0
    return float(np.mean(xs))


def transform_point_for_display(x, y, y_split, x_center_global,
                                lower_x_scale=1.0, lower_y_scale=1.0):
    if y >= y_split:
        return x, y
    x_new = x_center_global + lower_x_scale * (x - x_center_global)
    y_new = y_split - lower_y_scale * (y_split - y)
    return x_new, y_new


def transform_points_for_display(points, y_split, x_center_global,
                                 lower_x_scale=1.0, lower_y_scale=1.0):
    return [
        transform_point_for_display(
            x, y,
            y_split=y_split,
            x_center_global=x_center_global,
            lower_x_scale=lower_x_scale,
            lower_y_scale=lower_y_scale,
        )
        for x, y in points
    ]


def make_linewidth_mapper(speed_vals_ms: np.ndarray):
    smin, smax = float(np.min(speed_vals_ms)), float(np.max(speed_vals_ms))
    if smin == smax:
        smax = smin + 1e-9

    def _f(S):
        t = (S - smin) / (smax - smin)
        t = max(0.0, min(1.0, t))
        return MIN_LINE_WIDTH + t * (MAX_LINE_WIDTH - MIN_LINE_WIDTH)

    return _f, smin, smax


def get_unique_speed_values_ms(lanes, include_internal=False, round_digits=2):
    speeds = []
    for ln in lanes:
        if (not include_internal) and ln["internal"]:
            continue
        speeds.append(round(ln["speed"], round_digits))
    return sorted(set(speeds))


def _all_bounds(lanes, x_center_global):
    all_pts = []
    for ln in lanes:
        pts = ln["points"]
        if WARP_LOWER_NETWORK:
            pts = transform_points_for_display(
                pts,
                y_split=Y_SPLIT,
                x_center_global=x_center_global,
                lower_x_scale=LOWER_X_SCALE,
                lower_y_scale=LOWER_Y_SCALE,
            )
        all_pts.extend(pts)

    xs = [pt[0] for pt in all_pts]
    ys = [pt[1] for pt in all_pts]
    return min(xs), max(xs), min(ys), max(ys)


def add_network_legends(fig, ax, cmap, norm, linewidth_map, speed_values_ms):
    """Add length colorbar and speed legend below the network subplot."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Horizontal colorbar for length
    cax = inset_axes(
        ax,
        width="55%",
        height="4.5%",
        loc="lower left",
        bbox_to_anchor=(0.02, -0.10, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

    tick_values = np.linspace(norm.vmin, norm.vmax, 5)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{t:.0f}" for t in tick_values])
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE, length=1 )

    ax.text(
        0.32, -0.29,
        "Length (m)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=LABEL_FONTSIZE,
    )

    # Speed legend
    handles = []
    labels = []
    for s in speed_values_ms:
        lw = linewidth_map(s)
        handles.append(
            mlines.Line2D([], [], color="0.25", linewidth=lw, solid_capstyle="round")
        )
        if float(s).is_integer():
            labels.append(f"{int(s)}")
        else:
            labels.append(f"{s:.2f}")

    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.67, -0.24),
        frameon=False,
        fontsize=TICK_FONTSIZE,
        handlelength=3.4,
        borderpad=0.2,
        labelspacing=0.5,
        ncol=1,
    )

    ax.text(
        0.86, -0.26,
        "Speed (m/s)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=LABEL_FONTSIZE,
    )


def plot_network_panel(fig, ax, lanes, junctions, x_center_global):
    speed_vals_ms = np.array([ln["speed"] for ln in lanes], dtype=float)
    length_vals = np.array([ln["length"] for ln in lanes], dtype=float)

    cmap = mpl.colormaps[LENGTH_CMAP]
    lmin = float(length_vals.min())
    lmax = float(length_vals.max())
    if lmin == lmax:
        lmax = lmin + 1e-9
    norm = mpl.colors.Normalize(vmin=lmin, vmax=lmax)

    linewidth_map, _, _ = make_linewidth_mapper(speed_vals_ms)
    speed_values_ms = get_unique_speed_values_ms(
        lanes,
        include_internal=False,
        round_digits=2
    )

    xmin, xmax, ymin, ymax = _all_bounds(lanes, x_center_global)
    xr = xmax - xmin
    yr = ymax - ymin
    pad_x = xr * 0.02 if xr > 0 else 1.0
    pad_y = yr * 0.04 if yr > 0 else 1.0

    for ln in lanes:
        plot_points = ln["points"]
        if WARP_LOWER_NETWORK:
            plot_points = transform_points_for_display(
                plot_points,
                y_split=Y_SPLIT,
                x_center_global=x_center_global,
                lower_x_scale=LOWER_X_SCALE,
                lower_y_scale=LOWER_Y_SCALE,
            )

        xs, ys = zip(*plot_points)
        color = cmap(norm(ln["length"]))
        lw = linewidth_map(ln["speed"])
        alpha = NORMAL_ALPHA

        if ln["internal"]:
            lw *= INTERNAL_LINE_WIDTH_SCALE
            alpha = INTERNAL_ALPHA

        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")

        if SHOW_END_CIRCLES:
            ex, ey = plot_points[-1]
            ax.scatter([ex], [ey], s=END_CIRCLE_SIZE, c=[color], alpha=alpha, zorder=5)

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    if SHOW_NETWORK_TITLE:
        ax.set_title(SHOW_NETWORK_TITLE, pad=8)

    add_origin_destination_markers(ax, junctions, x_center_global)
    add_route_labels(ax, junctions, x_center_global)
    add_network_legends(fig, ax, cmap, norm, linewidth_map, speed_values_ms)


# =========================================================
# MAIN
# =========================================================
def main():
    if not RUNS_BASE_DIR.exists():
        raise FileNotFoundError(f"RUNS_BASE_DIR not found: {RUNS_BASE_DIR}")

    if not NET_FILE.exists():
        raise FileNotFoundError(f"NET_FILE not found: {NET_FILE}")

    print(f"[INFO] Reading travel-time runs from: {RUNS_BASE_DIR}")
    travel_data = collect_travel_time_data(RUNS_BASE_DIR)

    print(f"[INFO] Reading network from: {NET_FILE}")
    lanes = read_lanes_with_speed_and_length(NET_FILE, include_internal=INCLUDE_INTERNAL)
    if not lanes:
        raise RuntimeError("No drawable lanes found in network.")

    x_center_global = compute_global_x_center(lanes, Y_SPLIT)
    junctions = read_junctions(NET_FILE)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        gridspec_kw={"width_ratios": [1.05, 1.0]}
    )

    plot_network_panel(fig, ax_left, lanes, junctions, x_center_global)
    plot_travel_time_panel(ax_right, travel_data, xlim=LEFT_XMAX)


    fig.subplots_adjust(wspace=WSPACE, bottom=BOTTOM_MARGIN)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved combined figure to: {OUT_PATH}")


if __name__ == "__main__":
    main()