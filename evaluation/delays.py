import re
import glob
from pathlib import Path
import pandas as pd

def compute_agent_delays(
    base_dir: str,
    last_n: int = 10,
    output_csv: str | None = None,
    route_order: str = "appearance",  # or "alphabetical"
    income_csv: str | None = None,    # optional; defaults to base_dir/agents_income.csv if present
):
    """
    Compute per-agent average delay over the last N episodes and include:
      - chosen route (index + path)
      - income (from file or episodes)
    
    base_dir: Path containing both:
        base_dir/episodes/   → episode CSVs (ep*.csv)
        base_dir/paths.csv   → free-flow reference
    
    Optional:
        base_dir/agents_income.csv → agent incomes (id,income)

    Output columns include:
      - id
      - income
      - episodes_used
      - records_used
      - avg_travel_time
      - free_flow_time
      - avg_delay
      - chosen_route_index
      - chosen_route_path
    """

    base_dir = Path(base_dir)
    episodes_dir = base_dir / "episodes"
    free_flow_csv = base_dir / "paths.csv"

    if not free_flow_csv.exists():
        raise FileNotFoundError(f"Expected free-flow file not found: {free_flow_csv}")
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Expected episodes directory not found: {episodes_dir}")

    # Auto-locate income file if not provided
    if income_csv is None:
        default_income = base_dir / "agents_income.csv"
        if default_income.exists():
            income_csv = str(default_income)

    # --- 1) Load free-flow times and build mappings
    df_ff = pd.read_csv(free_flow_csv).dropna(how="all")
    df_ff = df_ff.rename(columns={
        "origins": "origin",
        "destinations": "destination",
        "free_flow_time": "ff_time"
    })
    df_ff = df_ff.dropna(subset=["origin", "destination", "ff_time"])
    df_ff["origin"] = df_ff["origin"].astype(int)
    df_ff["destination"] = df_ff["destination"].astype(int)

    if route_order == "alphabetical":
        df_ff = df_ff.sort_values(["origin", "destination", "path"], kind="stable")
    elif route_order != "appearance":
        raise ValueError("route_order must be 'appearance' or 'alphabetical'")

    # Assign route indices per OD
    df_ff["route_index"] = df_ff.groupby(["origin", "destination"]).cumcount()

    ff_time_map = df_ff.set_index(["origin", "destination", "route_index"])["ff_time"].to_dict()
    path_map = df_ff.set_index(["origin", "destination", "route_index"])["path"].to_dict()

    # --- 2) Load and sort episode files
    episode_files = []
    for p in glob.glob(str(episodes_dir / "ep*.csv")):
        m = re.search(r"ep(\d+)\.csv$", Path(p).name)
        if m:
            ep = int(m.group(1))
            episode_files.append((ep, Path(p)))

    if not episode_files:
        raise FileNotFoundError(f"No episode CSVs matching 'ep{{number}}.csv' found in {episodes_dir}")

    episode_files.sort(key=lambda x: x[0])
    selected = episode_files[-last_n:]

    # --- 3) Load episodes
    dfs = []
    for ep_num, path in selected:
        df = pd.read_csv(path, on_bad_lines="skip")
        df["episode"] = ep_num
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True).dropna(how="all")

    needed_cols = ["travel_time", "id", "action", "origin", "destination"]
    missing = [c for c in needed_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing required columns in episode CSVs: {missing}")

    # Normalize dtypes
    for c in ["id", "action", "origin", "destination"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce").astype("Int64")
    df_all["travel_time"] = pd.to_numeric(df_all["travel_time"], errors="coerce")
    df_all = df_all.dropna(subset=["id", "action", "origin", "destination", "travel_time"])

    df_all[["id", "action", "origin", "destination"]] = df_all[["id", "action", "origin", "destination"]].astype(int)

    # --- 3a) Load/derive income
    income_df = None
    if income_csv:
        tmp = pd.read_csv(income_csv)
        tmp_cols = {c.lower(): c for c in tmp.columns}
        id_col = tmp_cols.get("id")
        income_col = tmp_cols.get("income") or tmp_cols.get("agent_income")
        if id_col is None or income_col is None:
            raise ValueError("income_csv must contain columns 'id' and 'income' (or 'agent_income').")
        income_df = (
            tmp.rename(columns={id_col: "id", income_col: "income"})[["id", "income"]]
              .dropna(subset=["id", "income"])
        )
        income_df["id"] = pd.to_numeric(income_df["id"], errors="coerce").astype("Int64")
        income_df["income"] = pd.to_numeric(income_df["income"], errors="coerce")
        income_df = income_df.dropna(subset=["id", "income"]).astype({"id": int})
        income_df = income_df.drop_duplicates(subset=["id"], keep="first")
    elif "income" in df_all.columns:
        tmp = df_all[["id", "income"]].copy()
        tmp["income"] = pd.to_numeric(tmp["income"], errors="coerce")
        tmp = tmp.dropna(subset=["income"])
        if not tmp.empty:
            def mode_income(s):
                vc = s.value_counts(dropna=True)
                if vc.empty:
                    return None
                top = vc.iloc[0]
                cands = vc[vc == top].index.tolist()
                return min(cands)
            income_df = tmp.groupby("id", as_index=False)["income"].apply(mode_income)
            if "id" not in income_df.columns or "income" not in income_df.columns:
                income_df = income_df.reset_index(names=["id"]).rename(columns={0: "income"})
            income_df["id"] = income_df["id"].astype(int)

    # --- 4) Attach free-flow times and paths
    df_all["ff_time"] = df_all.apply(
        lambda r: ff_time_map.get((r["origin"], r["destination"], r["action"])), axis=1
    )
    df_all["route_path"] = df_all.apply(
        lambda r: path_map.get((r["origin"], r["destination"], r["action"])), axis=1
    )

    unmatched = df_all["ff_time"].isna().sum()
    if unmatched > 0:
        print(f"Warning: {unmatched} rows had no free-flow match. Dropping them.")
    df_all = df_all.dropna(subset=["ff_time"])

    # --- 5) Compute delay
    df_all["delay"] = df_all["travel_time"] - df_all["ff_time"]

    # --- 6) Determine chosen route
    def mode_with_tiebreak(series):
        vc = series.value_counts()
        if vc.empty:
            return None
        max_count = vc.iloc[0]
        candidates = vc[vc == max_count].index.tolist()
        return min(candidates)

    chosen_idx = (
        df_all.groupby("id")["action"]
        .apply(mode_with_tiebreak)
        .rename("chosen_route_index")
        .reset_index()
    )

    def chosen_path_for_agent(sub):
        idx = sub["chosen_route_index"].iloc[0]
        paths = df_all.loc[(df_all["id"] == sub["id"].iloc[0]) & (df_all["action"] == idx), "route_path"]
        if paths.empty:
            return None
        vc = paths.value_counts()
        top_count = vc.iloc[0]
        candidates = vc[vc == top_count].index.tolist()
        return sorted(candidates)[0]

    chosen_paths = (
        chosen_idx.merge(df_all[["id"]].drop_duplicates(), on="id", how="left")
        .groupby("id", as_index=False)
        .apply(chosen_path_for_agent)
        .rename(columns={None: "chosen_route_path"})
    )

    if "id" not in chosen_paths.columns or "chosen_route_path" not in chosen_paths.columns:
        chosen_paths = chosen_paths.reset_index(names=["id"]).rename(columns={0: "chosen_route_path"})

    chosen = chosen_idx.merge(chosen_paths, on="id", how="left")

    # --- 7) Aggregate
    agg = (
        df_all.groupby("id", as_index=False)
        .agg(
            episodes_used=("episode", "nunique"),
            records_used=("id", "size"),
            avg_travel_time=("travel_time", "mean"),
            free_flow_time=("ff_time", "mean"),
            avg_delay=("delay", "mean"),
        )
        .sort_values("id")
    )

    agg = agg.merge(chosen, on="id", how="left")

    # --- 8) Merge income
    if income_df is not None and not income_df.empty:
        agg = agg.merge(income_df, on="id", how="left")
        cols = agg.columns.tolist()
        cols.remove("income")
        cols.insert(1, "income")
        agg = agg[cols]
    else:
        agg["income"] = pd.NA
        cols = agg.columns.tolist()
        cols.remove("income")
        cols.insert(1, "income")
        agg = agg[cols]

    # --- 9) Output
    for c in ["avg_travel_time", "free_flow_time", "avg_delay", "income"]:
        if c in agg.columns:
            agg[c] = pd.to_numeric(agg[c], errors="coerce")

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("\nPer-agent average delay (with chosen route & income):")
    print(agg.to_string(index=False))

    # --- 10) Save
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(output_csv, index=False)
        print(f"\nSaved results to: {output_csv}")

    return agg


import os
from pathlib import Path
import contextlib
import io
import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path
import contextlib
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D

import os
from pathlib import Path
import contextlib
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def compare_agent_delays_side_by_side(
    base_dirs,
    labels=None,
    last_n=100,
    route_order="appearance",
    income_csv=None,
    output_csvs=None,
    figsize_per_panel=(5, 4),
    figure_title="Income vs Avg Delay (per run)",
    save_fig_path=None,
    share_limits=True,
    cmap_name="tab10",
    jitter_if_same_income=True,
    jitter_frac=0.01,
    show_plot=False,
):
    """
    Scatter per panel: x=income, y=avg_delay, colored by chosen_route_index.
    Uses discrete colors per route, saves figure correctly before showing.
    """
    base_dirs = [Path(b) for b in base_dirs]
    n = len(base_dirs)
    if labels is None:
        labels = [b.name for b in base_dirs]
    if len(labels) != n:
        raise ValueError("labels must be same length as base_dirs")

    # Normalize income_csv handling
    if isinstance(income_csv, (str, Path)) or income_csv is None:
        income_csvs = [income_csv] * n
    else:
        if len(income_csv) != n:
            raise ValueError("income_csv list must match base_dirs length")
        income_csvs = income_csv

    if output_csvs is not None and len(output_csvs) != n:
        raise ValueError("output_csvs list must match base_dirs length")

    # Run aggregations
    results_per_run = []
    for i, (bdir, inc_csv) in enumerate(zip(base_dirs, income_csvs)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            agg = compute_agent_delays(
                base_dir=str(bdir),
                last_n=last_n,
                output_csv=output_csvs[i] if output_csvs else None,
                route_order=route_order,
                income_csv=str(inc_csv) if inc_csv else None,
            )
        results_per_run.append(agg)

    # Prepare data
    per_run_clean = []
    all_x, all_y = [], []
    global_routes = set()

    for label, agg in zip(labels, results_per_run):
        tmp = agg.copy()
        tmp["income"] = pd.to_numeric(tmp["income"], errors="coerce")
        tmp["avg_delay"] = pd.to_numeric(tmp["avg_delay"], errors="coerce")
        tmp["chosen_route_index"] = pd.to_numeric(tmp.get("chosen_route_index", np.nan), errors="coerce")
        tmp = tmp.dropna(subset=["income", "avg_delay"])
        per_run_clean.append(tmp)
        if not tmp.empty:
            all_x.append(tmp["income"].to_numpy())
            all_y.append(tmp["avg_delay"].to_numpy())
            routes_present = sorted(r for r in tmp["chosen_route_index"].dropna().unique())
            print(f"[{label}] usable points: {len(tmp)} | routes present: {routes_present}")
            global_routes.update(routes_present)
        else:
            print(f"[{label}] usable points: 0")

    # Shared axis limits
    shared_xlim = shared_ylim = None
    if share_limits and len(all_x):
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        x_min, x_max = float(np.nanmin(all_x)), float(np.nanmax(all_x))
        y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
        x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
        y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
        shared_xlim = (x_min - x_pad, x_max + x_pad)
        shared_ylim = (y_min - y_pad, y_max + y_pad)

    # Build route → color mapping
    sorted_routes = sorted(global_routes)
    base_cmap = plt.get_cmap(cmap_name)
    palette = [base_cmap(i % base_cmap.N) for i in range(max(1, len(sorted_routes)))]
    route_to_color = {route: palette[i] for i, route in enumerate(sorted_routes)} if sorted_routes else {}

    # Create figure
    fig_width = max(6, figsize_per_panel[0] * n)
    fig_height = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes[0]

    for ax, label, tmp in zip(axes, labels, per_run_clean):
        ax.set_title(label, pad=8)
        ax.set_xlabel("Income")
        ax.set_ylabel("Avg delay")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        if tmp.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            x = tmp["income"].to_numpy(dtype=float)
            y = tmp["avg_delay"].to_numpy(dtype=float)
            routes = tmp["chosen_route_index"].to_numpy(dtype=float)

            # choose per-point colors
            colors = [
                route_to_color.get(int(r), palette[0]) if not np.isnan(r) else palette[0]
                for r in routes
            ]

            # Jitter if needed
            if jitter_if_same_income and np.nanmax(x) == np.nanmin(x):
                span = (shared_xlim[1] - shared_xlim[0]) if shared_xlim else 1.0
                eps = max(span * jitter_frac, 1e-9)
                x = x + (np.random.rand(len(x)) - 0.5) * 2 * eps

            ax.scatter(x, y, c=colors)

        if shared_xlim and shared_ylim:
            ax.set_xlim(shared_xlim)
            ax.set_ylim(shared_ylim)

    # Legend (below suptitle, not overlapping it)
    if sorted_routes:
        legend_elements = [
            Line2D([0], [0], marker='o', linestyle='', color=route_to_color[r], label=f"Route {r}")
            for r in sorted_routes
        ]
        # Adjust top margin to make room for legend + title
        plt.subplots_adjust(top=0.80)
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=min(len(sorted_routes), 6),
            frameon=False,
            bbox_to_anchor=(0.5, 0.95)
        )

    fig.suptitle(figure_title, y=0.98, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    # Save before showing
    if save_fig_path:
        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        fig.savefig(save_fig_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure to: {save_fig_path}")

    if show_plot:
        plt.show()

    plt.close(fig)
    return results_per_run



"""BASE_DIRS = [
    "../data/training_records_300_agents/training_records_monetary_pricing_300_agents_long",
    "../data/training_records_300_agents/training_records_monetary_pricing_300_agents_long_1",
    "../data/training_records_300_agents/training_records_monetary_pricing_300_agents_long_2",
    "../data/training_records_300_agents/training_records_monetary_pricing_300_agents_long_3",
    "../data/training_records_300_agents/training_records_monetary_pricing_300_agents_long_4",
]
LABELS = ["Run A", "Run B", "Run C", "Run D", "Run E"]"""

BASE_DIRS = [
    r"../scenarios/monetary_pricing/training_records_monetary_pricing_300_agents_fee_0_25",
    r"../scenarios/monetary_pricing/training_records_monetary_pricing_300_agents_fee_0_5",
    r"../scenarios/monetary_pricing/training_records_monetary_pricing_300_agents_fee_1",
]

LABELS = ["Run A", "Run B", "Run C"]

_ = compare_agent_delays_side_by_side(
    base_dirs=BASE_DIRS,
    labels=LABELS,
    last_n=10,
    route_order="appearance",
    save_fig_path="income_delay_by_route.png",
    cmap_name="tab20",
    show_plot=False,   # save first; show if you want
)

