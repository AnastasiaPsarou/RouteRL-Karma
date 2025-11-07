import re
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def compare_avg_delay_per_episode_by_route(
    base_dirs,
    labels=None,
    last_n=None,
    route_order="appearance",        # or "alphabetical" (by 'path' if present)
    plot_style="lines",               # "lines" or "heatmap"
    figsize_per_panel=(6, 4),
    save_fig_path=None,
    show_plot=False,
    share_y_limits=True,
    cmap_name="tab20",
    # line style
    line_width=2.0,
    marker_size=0,
    # fonts
    title_fontsize=14,
    axis_label_fontsize=11,
    tick_fontsize=9,
    legend_fontsize=10,
    suptitle="Avg Delay per Episode by Route",
):
    """
    For each run (base_dir), compute avg delay per episode for each route (action)
    and plot:
      - lines: one subplot per run, multiple lines (one per route)
      - heatmap: one subplot per run, routes x episodes colored by avg delay

    Returns: list of per-run DataFrames with columns:
      ['episode', 'route_index', 'avg_delay']
    """

    base_dirs = [Path(b) for b in base_dirs]
    n = len(base_dirs)
    if labels is None:
        labels = [b.name for b in base_dirs]
    if len(labels) != n:
        raise ValueError("labels must be same length as base_dirs")

    # --- helper: compute per-run per-episode per-route avg delay
    def compute_ep_route_df(base_dir):
        episodes_dir = base_dir / "episodes"
        free_flow_csv = base_dir / "paths.csv"
        if not free_flow_csv.exists():
            raise FileNotFoundError(f"Expected free-flow file not found: {free_flow_csv}")
        if not episodes_dir.exists():
            raise FileNotFoundError(f"Expected episodes directory not found: {episodes_dir}")

        # free-flow
        df_ff = pd.read_csv(free_flow_csv).dropna(how="all")
        df_ff = df_ff.rename(columns={
            "origins": "origin",
            "destinations": "destination",
            "free_flow_time": "ff_time"
        })
        df_ff = df_ff.dropna(subset=["origin", "destination", "ff_time"])
        df_ff["origin"] = pd.to_numeric(df_ff["origin"], errors="coerce").astype("Int64")
        df_ff["destination"] = pd.to_numeric(df_ff["destination"], errors="coerce").astype("Int64")

        if route_order == "alphabetical":
            sort_cols = ["origin", "destination"]
            if "path" in df_ff.columns:
                sort_cols.append("path")
            df_ff = df_ff.sort_values(sort_cols, kind="stable")
        elif route_order != "appearance":
            raise ValueError("route_order must be 'appearance' or 'alphabetical'")

        df_ff["route_index"] = df_ff.groupby(["origin", "destination"]).cumcount()
        ff_time_map = df_ff.set_index(["origin", "destination", "route_index"])["ff_time"].to_dict()

        # episodes
        episode_files = []
        for p in glob.glob(str(episodes_dir / "ep*.csv")):
            m = re.search(r"ep(\d+)\.csv$", Path(p).name)
            if m:
                episode_files.append((int(m.group(1)), Path(p)))
        if not episode_files:
            raise FileNotFoundError(f"No episode CSVs matching 'ep{{number}}.csv' found in {episodes_dir}")

        episode_files.sort(key=lambda x: x[0])
        if last_n and last_n > 0:
            episode_files = episode_files[-last_n:]

        dfs = []
        for ep_num, path in episode_files:
            df = pd.read_csv(path, on_bad_lines="skip")
            df["episode"] = ep_num
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True).dropna(how="all")

        needed = ["travel_time", "id", "action", "origin", "destination", "episode"]
        missing = [c for c in needed if c not in df_all.columns]
        if missing:
            raise ValueError(f"Missing required columns in episode CSVs: {missing}")

        for c in ["id", "action", "origin", "destination", "episode"]:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce").astype("Int64")
        df_all["travel_time"] = pd.to_numeric(df_all["travel_time"], errors="coerce")
        df_all = df_all.dropna(subset=["id", "action", "origin", "destination", "travel_time", "episode"])
        df_all[["id", "action", "origin", "destination", "episode"]] = df_all[
            ["id", "action", "origin", "destination", "episode"]
        ].astype(int)

        # attach ff and compute delay
        df_all["ff_time"] = df_all.apply(
            lambda r: ff_time_map.get((r["origin"], r["destination"], r["action"])), axis=1
        )
        unmatched = int(df_all["ff_time"].isna().sum())
        if unmatched > 0:
            print(f"Warning: {unmatched} rows had no free-flow match. Dropping them.")
        df_all = df_all.dropna(subset=["ff_time"])

        df_all["delay"] = df_all["travel_time"] - df_all["ff_time"]

        # average by (episode, route_index)
        df_ep_route = (
            df_all.groupby(["episode", "action"], as_index=False)["delay"]
                  .mean()
                  .rename(columns={"action": "route_index", "delay": "avg_delay"})
                  .sort_values(["episode", "route_index"])
        )
        return df_ep_route

    # --- compute per-run data
    per_run = []
    all_routes = set()
    for b in base_dirs:
        df_ep_route = compute_ep_route_df(b)
        per_run.append(df_ep_route)
        all_routes.update(df_ep_route["route_index"].unique())

    # stable route order across panels
    sorted_routes = sorted(int(r) for r in all_routes)
    cmap = plt.get_cmap(cmap_name)
    palette = [cmap(i % cmap.N) for i in range(max(1, len(sorted_routes)))]
    route_to_color = {r: palette[i] for i, r in enumerate(sorted_routes)}  # for "lines" style

    # layout
    fig_width = max(6, figsize_per_panel[0] * n)
    fig_height = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes[0]

    # optional shared y-lims (compute across all runs/routes)
    y_limits = None
    if share_y_limits:
        ys = []
        for df in per_run:
            ys.append(df["avg_delay"].to_numpy(dtype=float))
        if ys:
            y_all = np.concatenate(ys)
            if len(y_all):
                y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
                pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
                y_limits = (y_min - pad, y_max + pad)

    # plot per panel
    for i, (ax, label, df) in enumerate(zip(axes, labels, per_run)):
        ax.set_title(label, fontsize=title_fontsize, pad=8)

        # Y label only on leftmost plot
        if i == 0:
            ax.set_ylabel("Avg delay", fontsize=axis_label_fontsize)
        else:
            ax.set_ylabel("")

        # X label only on leftmost plot (if you want)
        if i == 0:
            ax.set_xlabel("Episode", fontsize=axis_label_fontsize)
        else:
            ax.set_xlabel("Episode", fontsize=axis_label_fontsize)

        # --- tick font sizes ---
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        #ax.set_xlim(0, 410)

        # Hide ticks on all but leftmost plot
        if i != 0:
            ax.tick_params(axis='y', labelleft=False)

        ax.grid(True, linestyle="--", linewidth=1, alpha=0.5)

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        if plot_style == "lines":
            # plot one line per route
            for r in sorted_routes:
                sub = df[df["route_index"] == r]
                if sub.empty:
                    continue
                x = sub["episode"].to_numpy()
                y = sub["avg_delay"].to_numpy()
                ax.plot(
                    x, y,
                    label=f"Route {r}",
                    color=route_to_color[r],
                    linewidth=line_width,
                    marker='o' if marker_size > 0 else None,
                    markersize=marker_size,
                )

        elif plot_style == "heatmap":
            # pivot to routes x episodes
            pivot = df.pivot(index="route_index", columns="episode", values="avg_delay")
            pivot = pivot.reindex(index=sorted_routes)  # ensure route order
            im = ax.imshow(
                pivot.to_numpy(),
                aspect="auto",
                origin="lower",
                interpolation="nearest",
            )
            # axis ticks
            ax.set_yticks(range(len(sorted_routes)))
            ax.set_yticklabels([str(r) for r in sorted_routes], fontsize=tick_fontsize)
            episodes_sorted = sorted(pivot.columns.tolist())
            ax.set_xticks(range(len(episodes_sorted)))
            ax.set_xticklabels([str(e) for e in episodes_sorted], rotation=45, ha="right", fontsize=tick_fontsize)
            # colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=tick_fontsize)
            ax.set_ylabel("Route index", fontsize=axis_label_fontsize)

        else:
            raise ValueError("plot_style must be 'lines' or 'heatmap'")

        if y_limits and plot_style == "lines":
            ax.set_ylim(y_limits)

    # legend (for line style) – one shared legend at the top
    if plot_style == "lines" and len(sorted_routes):
        legend_items = [
            Line2D([0], [0], color=route_to_color[r], linewidth=line_width * 3,
                marker='o' if marker_size > 0 else None, markersize=marker_size,
                label=f"Route {r}")
            for r in sorted_routes
        ]
        # Leave room for both suptitle and legend
        plt.subplots_adjust(top=0.78)
        # Legend a bit lower
        fig.legend(
            handles=legend_items,
            loc="upper center",
            ncol=min(len(sorted_routes), 6),
            frameon=False,
            fontsize=legend_fontsize,
            bbox_to_anchor=(0.5, 0.92),
        )

    # Place suptitle slightly higher (above legend)
    fig.suptitle(suptitle, y=0.99, fontsize=title_fontsize + 1, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.85] if plot_style == "lines" else [0, 0, 1, 0.90])


    if save_fig_path:
        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure to: {save_fig_path}")

    if show_plot:
        plt.show()
    plt.close(fig)

    return per_run, fig



BASE_DIRS = [
    r"../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_4_long",
    r"../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_5_long",
    r"../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_6_long",
    r"../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_7_long",
    r"../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_8_long",
]
LABELS = ["Run 1", "Run 2", "Run 3", "Run 4", "Run 5"]

_, _ = compare_avg_delay_per_episode_by_route(
    base_dirs=BASE_DIRS,
    labels=LABELS,
    last_n=410,
    route_order="appearance",
    plot_style="lines",                 # try "heatmap" for a heatmap view
    figsize_per_panel=(6, 7),
    line_width=4.0,
    marker_size=0,                      # e.g., 4 for small markers
    cmap_name="tab20",
    title_fontsize=34,
    axis_label_fontsize=28,
    tick_fontsize=26,
    legend_fontsize=28,
    suptitle="Avg Delay per Episode by Route (lines)",
    save_fig_path="imgs/avg_delay_per_episode_by_route_lines.png",
    show_plot=False,
)
