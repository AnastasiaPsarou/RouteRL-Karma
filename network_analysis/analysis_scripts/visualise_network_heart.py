#!/usr/bin/env python3
"""
Publication-style visualisation of a SUMO network where:
- color encodes length
- line width encodes speed

Outputs:
  ../imgs/network_length_and_speed_publication.png
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import numpy as np

# =========================================================
# User parameters
# =========================================================
NET_FILE = "../network_base/network.net.xml"
OUT_DIR = Path("../imgs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INCLUDE_INTERNAL = True

# =========================================================
# Figure settings
# =========================================================
FIG_WIDTH = 4.0
DPI = 500

SHOW_TITLE = False
TITLE_FONTSIZE = 11
AXIS_OFF = True

# Colormap for length
LENGTH_CMAP = "viridis"

# Line width range used to encode speed
MIN_LINE_WIDTH = 0.8
MAX_LINE_WIDTH = 8.0

# Internal lanes drawn slightly thinner/fainter
INTERNAL_LINE_WIDTH_SCALE = 0.8
INTERNAL_ALPHA = 0.35
NORMAL_ALPHA = 0.95

# Optional endpoint circles
SHOW_END_CIRCLES = False
END_CIRCLE_SIZE = 10

# Colorbar / legend text sizes
COLORBAR_FRACTION = 0.035
COLORBAR_PAD = 0.03
COLORBAR_LABELSIZE = 11
COLORBAR_TICKSIZE = 9

FONT_FAMILY = "DejaVu Sans"

# =========================================================
# Optional display warping (turn off for faithful geometry)
# =========================================================
WARP_LOWER_NETWORK = False
Y_SPLIT = 200.0
LOWER_X_SCALE = 0.65
LOWER_Y_SCALE = 0.75

# =========================================================
# Matplotlib style
# =========================================================
mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 10,
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

# =========================================================
# Parsing helpers
# =========================================================
def _dist2d(p0, p1):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return (dx * dx + dy * dy) ** 0.5

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

def read_lanes_with_speed_and_length(net_path: str, include_internal: bool = True):
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

# =========================================================
# Optional global display transform
# =========================================================
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

# =========================================================
# Plot helpers
# =========================================================
def make_linewidth_mapper(speed_vals_kmh: np.ndarray):
    smin, smax = float(np.min(speed_vals_kmh)), float(np.max(speed_vals_kmh))
    if smin == smax:
        smax = smin + 1e-9

    def _f(S):
        t = (S - smin) / (smax - smin)
        t = max(0.0, min(1.0, t))
        return MIN_LINE_WIDTH + t * (MAX_LINE_WIDTH - MIN_LINE_WIDTH)

    return _f, smin, smax

def get_unique_speed_values_ms(lanes, include_internal=False, round_digits=2):
    """
    Return sorted unique speed values (in m/s) present in the network.
    Optionally exclude internal lanes from the legend.
    """
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

def add_combined_legends(fig, ax, cmap, norm, linewidth_map, speed_values_ms):
    """
    Add a length colorbar on the left with its label on the same horizontal line,
    and a vertical speed-width legend on the right with its label below.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # --- Length colorbar on the left ---
    cax = inset_axes(
        ax,
        width="58%",
        height="4%",
        loc="lower left",
        bbox_to_anchor=(0.04, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

    tick_values = np.linspace(norm.vmin, norm.vmax, 6)   # or 6, 7, etc.
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{t:.0f}" for t in tick_values])

    cbar.ax.tick_params(labelsize=COLORBAR_TICKSIZE)

    # Label for length
    ax.text(
        0.20, -0.22,
        "Length (m)",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=COLORBAR_LABELSIZE,
    )

    # --- Speed legend on the right ---
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
        bbox_to_anchor=(0.68, -0.165),
        frameon=False,
        fontsize=9,
        handlelength=4.0,
        borderpad=0.2,
        labelspacing=0.6,
        ncol=1,
    )

    ax.text(
        0.88, -0.18,
        "Speed (m/s)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=COLORBAR_LABELSIZE,
    )

def plot_length_and_speed(lanes, out_path, x_center_global):
    speed_vals_ms = np.array([ln["speed"] for ln in lanes], dtype=float)
    length_vals = np.array([ln["length"] for ln in lanes], dtype=float)

    cmap = mpl.colormaps[LENGTH_CMAP]
    lmin = float(length_vals.min())
    lmax = float(length_vals.max())
    if lmin == lmax:
        lmax = lmin + 1e-9
    norm = mpl.colors.Normalize(vmin=lmin, vmax=lmax)

    linewidth_map, smin, smax = make_linewidth_mapper(speed_vals_ms)

    # All unique speeds in the network, excluding internal lanes from the legend
    speed_values_ms = get_unique_speed_values_ms(
        lanes,
        include_internal=False,
        round_digits=2
    )
    print("[INFO] Unique speed values in legend (m/s):", speed_values_ms)

    xmin, xmax, ymin, ymax = _all_bounds(lanes, x_center_global)
    xr = xmax - xmin
    yr = ymax - ymin
    pad_x = xr * 0.02 if xr > 0 else 1.0
    pad_y = yr * 0.04 if yr > 0 else 1.0

    data_aspect = (yr + 2 * pad_y) / (xr + 2 * pad_x) if (xr + 2 * pad_x) > 0 else 1.0
    fig_height = max(2.5, FIG_WIDTH * data_aspect)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

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

    if AXIS_OFF:
        ax.set_axis_off()

    if SHOW_TITLE:
        ax.set_title("Network: color = length, width = speed", pad=6)

    add_combined_legends(fig, ax, cmap, norm, linewidth_map, speed_values_ms)

    fig.savefig(out_path, dpi=DPI, pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Saved: {out_path.resolve()}")

def main():
    net_path = Path(NET_FILE)
    if not net_path.exists():
        print(f"[ERROR] NET_FILE not found: {net_path.resolve()}")
        return

    print(f"[INFO] Reading network: {net_path.resolve()}")
    lanes = read_lanes_with_speed_and_length(str(net_path), include_internal=INCLUDE_INTERNAL)
    if not lanes:
        print("[WARN] No drawable lanes found.")
        return

    x_center_global = compute_global_x_center(lanes, Y_SPLIT)

    out_path = OUT_DIR / "network_length_and_speed_publication.png"
    plot_length_and_speed(lanes, out_path, x_center_global)

if __name__ == "__main__":
    main()