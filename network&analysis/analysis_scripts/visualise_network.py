#!/usr/bin/env python3
"""
Visualise a SUMO network with per-lane plotting:
- Each LANE is drawn separately and colored by speed (file 1) or length (file 2)
- Small circles at each lane's FINISH point
- Horizontal colorbar (below or above the plot)

Outputs:
  images/network_by_speed.png
  images/network_by_length.png
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# =========================================================
# User parameters
# =========================================================
NET_FILE = "../network_base/Anastasianetwork.net.xml"   # <-- your network path
OUT_DIR = Path("../imgs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Include lanes from internal edges (function="internal")
INCLUDE_INTERNAL = True

# Visual styling
LINE_WIDTH = 2.0

SHOW_END_CIRCLES = True
END_CIRCLE_EDGE = "black"
END_CIRCLE_EDGEWIDTH = 0.35
END_CIRCLE_ALPHA = 0.9
END_CIRCLE_MIN = 40    # smallest circle area (pts^2)
END_CIRCLE_MAX = 160   # largest circle area (pts^2)

# Colorbar placement: "bottom" or "top"
COLORBAR_POSITION = "bottom"

# Colormaps
SPEED_CMAP = "plasma"
LENGTH_CMAP = "viridis"

# =========================================================
# Robust parsing helpers
# =========================================================
def _dist2d(p0, p1):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return (dx * dx + dy * dy) ** 0.5

def _parse_type_speeds(root):
    """Collect speeds defined on <type> (and legacy <edgeType>) elements."""
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
    """Return list of dicts: {points, speed, length, edge_id, lane_id} for drawable lanes."""
    lanes_out = []
    root = ET.parse(net_path).getroot()
    type_speed = _parse_type_speeds(root)

    for edge in root.findall("edge"):
        is_internal = (edge.get("function") == "internal")
        if is_internal and not include_internal:
            continue

        # Resolve edge-level speed via edge or type
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
            # Resolve lane speed: lane > edge > type
            sp = lane.get("speed")
            if sp is not None:
                try:
                    speed = float(sp)
                except Exception:
                    speed = edge_speed if edge_speed is not None else 0.0
            else:
                speed = edge_speed if edge_speed is not None else 0.0

            # Parse lane geometry
            shape = lane.get("shape", "")
            pts = []
            if shape:
                for p in shape.split():
                    try:
                        x, y = map(float, p.split(","))
                        pts.append((x, y))
                    except Exception:
                        pass

            # Resolve length: attribute or compute from shape
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
                    "speed": float(speed),
                    "length": float(length),
                    "edge_id": edge.get("id", ""),
                    "lane_id": lane_id,
                    "internal": is_internal
                })

    return lanes_out

# =========================================================
# Plotting helpers
# =========================================================
def make_norm(vals: np.ndarray) -> mpl.colors.Normalize:
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmin == vmax:
        vmax = vmin + 1e-9
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)

def map_length_to_size(length_vals: np.ndarray):
    """Map edge length to scatter area (pts^2) for finish circles."""
    lmin, lmax = float(np.min(length_vals)), float(np.max(length_vals))
    if lmin == lmax:
        lmax = lmin + 1e-9
    def _f(L):
        t = (L - lmin) / (lmax - lmin)
        t = max(0.0, min(1.0, t))
        return END_CIRCLE_MIN + t * (END_CIRCLE_MAX - END_CIRCLE_MIN)
    return _f

def add_horizontal_colorbar(fig, ax, mappable, label: str):
    """Add a horizontal colorbar below or above the axis using constrained layout."""
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="horizontal",
        location="bottom" if COLORBAR_POSITION.lower() == "bottom" else "top",
        fraction=0.045,
        pad=0.06,
    )
    cbar.set_label(label)
    return cbar

def _all_bounds(lanes):
    xs = [pt[0] for ln in lanes for pt in ln["points"]]
    ys = [pt[1] for ln in lanes for pt in ln["points"]]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, xmax, ymin, ymax

def plot_attribute(lanes, attr_name: str, cmap_name: str, out_path: Path, title_prefix: str):
    """Plot the network colored by a given attribute ('speed' or 'length'), per-lane."""
    vals = np.array([ln[attr_name] for ln in lanes], dtype=float)
    all_lengths = np.array([ln["length"] for ln in lanes], dtype=float)

    print(f"[INFO] Lanes: {len(lanes)}  |  {attr_name} min={vals.min():.4f}, max={vals.max():.4f}")

    cmap = mpl.cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()) if float(vals.max()) > float(vals.min()) else float(vals.min()) + 1e-9)
    size_map = map_length_to_size(all_lengths)

    # ---- Compute data bounds and figure aspect ----
    xmin, xmax, ymin, ymax = _all_bounds(lanes)
    xr = xmax - xmin
    yr = ymax - ymin
    # small padding (1%)
    pad_x = xr * 0.01 if xr > 0 else 1.0
    pad_y = yr * 0.01 if yr > 0 else 1.0

    data_aspect = (yr + 2 * pad_y) / (xr + 2 * pad_x) if (xr + 2 * pad_x) > 0 else 1.0

    # Choose a base width and derive height to match data aspect (so 'equal' doesn't add huge whitespace)
    base_width = 12.0
    fig_height = max(6.0, base_width * data_aspect)

    # Use constrained_layout to avoid colorbar/tight_layout conflicts
    fig, ax = plt.subplots(figsize=(base_width, fig_height), constrained_layout=True)

    # ----- draw lanes -----
    for ln, val in zip(lanes, vals):
        xs, ys = zip(*ln["points"])
        color = cmap(norm(val))
        ax.plot(xs, ys, color=color, linewidth=LINE_WIDTH)

        if SHOW_END_CIRCLES:
            ex, ey = ln["points"][-1]
            s_area = size_map(ln["length"])
            ax.scatter([ex], [ey], s=s_area, c=[color],
                       edgecolors=END_CIRCLE_EDGE, linewidths=END_CIRCLE_EDGEWIDTH,
                       alpha=END_CIRCLE_ALPHA, zorder=5)

    # Lock limits tightly around data (with small padding), then set equal aspect
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linewidth=0.3)
    ax.set_title(f"{title_prefix} — lanes colored by {attr_name}", pad=5)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    label = "Speed (m/s)" if attr_name == "speed" else "Length (m)"
    fig.colorbar(
        sm, ax=ax, orientation="horizontal",
        location="bottom", fraction=0.045, pad=0.06
    ).set_label(label)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] Saved: {out_path.resolve()}")
