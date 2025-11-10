import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import contextlib, io

# assumes your compute_agent_delays(base_dir, ...) function is already defined/imported
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
    


def plot_avg_delay_by_income_quintile(
    base_dirs,
    labels=None,
    last_n=100,
    income_csv=None,
    output_csvs=None,
    global_quintiles=True,
    use_observed_range_on_xlabel=True,
    figsize=(9, 4.5),
    save_fig_path=None,      # if split_by_quintile=False: single figure path (as before)
    save_fig_dir=None,       # if split_by_quintile=True: directory to save Q-specific figures
    fname_template="avg_delay_{quintile}.png",  # filename per quintile when split
    show_plot=False,
    # bar styling
    bar_width=0.6,
    error_bars=False,
    # fonts
    title="Average Delay by Income Quintile",
    title_fontsize=14,
    axis_label_fontsize=11,
    tick_fontsize=9,
    legend_fontsize=10,
    income_decimals=0,
    # NEW:
    split_by_quintile=True,  # <-- make one bar plot per quintile
):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import contextlib, io

    # ---------- same data prep as your version ----------
    base_dirs = [Path(b) for b in (base_dirs if isinstance(base_dirs, (list, tuple)) else [base_dirs])]
    n_runs = len(base_dirs)
    if labels is None:
        labels = [b.name for b in base_dirs]
    if len(labels) != n_runs:
        raise ValueError("labels must be same length as base_dirs")

    if income_csv is None or isinstance(income_csv, (str, Path)):
        income_csvs = [income_csv] * n_runs
    else:
        if len(income_csv) != n_runs:
            raise ValueError("income_csv list must match base_dirs length")
        income_csvs = income_csv

    per_run_agent = []
    for i, (bdir, inc_csv) in enumerate(zip(base_dirs, income_csvs)):
        """buf = io.StringIO()
        with contextlib.redirect_stdout(buf):"""
        df = compute_agent_delays(
            base_dir=str(bdir),
            last_n=last_n,
            output_csv=output_csvs[i] if output_csvs else None,
            route_order="appearance",
            income_csv=str(inc_csv) if inc_csv else None,
        )
        df = df[["id", "income", "avg_delay"]].copy()
        df["income"] = pd.to_numeric(df["income"], errors="coerce")
        df["avg_delay"] = pd.to_numeric(df["avg_delay"], errors="coerce")
        df = df.dropna(subset=["income", "avg_delay"])
        if df.empty:
            raise ValueError(f"No usable (income, avg_delay) data for run: {labels[i]}")
        per_run_agent.append(df)

    def compute_bins(series):
        try:
            _, bins = pd.qcut(series, q=5, retbins=True, duplicates="drop")
            bins = np.unique(bins)
        except ValueError:
            bins = np.linspace(series.min(), series.max(), 6)
        if len(bins) < 6:
            bins = np.linspace(series.min(), series.max(), 6)
        return bins

    if global_quintiles:
        all_incomes = pd.concat(per_run_agent, ignore_index=True)["income"]
        global_bins = compute_bins(all_incomes)
        bins_per_run = [global_bins] * n_runs
    else:
        bins_per_run = [compute_bins(df["income"]) for df in per_run_agent]

    def aggregate_one(df, bins):
        labels_q = [f"Q{i}" for i in range(1, len(bins))]
        cats = pd.cut(df["income"], bins=bins, include_lowest=True, labels=labels_q, right=True, duplicates="drop")
        tmp = df.assign(income_quintile=cats).dropna(subset=["income_quintile"])
        grp = tmp.groupby("income_quintile", observed=True)
        out = grp["avg_delay"].mean().to_frame("avg_delay")
        out["n_agents"] = grp.size()
        if error_bars:
            out["se"] = grp["avg_delay"].std(ddof=1) / np.sqrt(out["n_agents"])
        out["income_min"] = grp["income"].min()
        out["income_max"] = grp["income"].max()
        out = out.reset_index()
        out["order"] = out["income_quintile"].str.extract(r"Q(\d+)").astype(int)
        out = out.sort_values("order").drop(columns=["order"])
        return out

    results = []
    for df, bins in zip(per_run_agent, bins_per_run):
        results.append(aggregate_one(df, bins))

    quintiles = [f"Q{i}" for i in range(1, 6)]
    for i in range(n_runs):
        results[i] = (results[i].set_index("income_quintile").reindex(quintiles).reset_index())

    # label helper
    def fmt_income(x):
        if pd.isna(x):
            return "NA"
        return f"{x:.0f}" if income_decimals == 0 else f"{x:.{income_decimals}f}"

    # Build x-axis labels for the quintile captions (min–max across runs)
    x_labels_for_quintile = {}
    for q in quintiles:
        mins = [r.loc[r["income_quintile"] == q, "income_min"].values[0] for r in results]
        maxs = [r.loc[r["income_quintile"] == q, "income_max"].values[0] for r in results]
        mn = np.nanmin(mins)
        mx = np.nanmax(maxs)
        x_labels_for_quintile[q] = f"{q} [{fmt_income(mn)} – {fmt_income(mx)}]"

    figs = []

    if split_by_quintile:
        # one separate figure per quintile: bars = runs
        if save_fig_dir is not None:
            Path(save_fig_dir).mkdir(parents=True, exist_ok=True)

        for qi, q in enumerate(quintiles):
            # collect heights & (optional) within-quintile SE for each run
            heights = []
            yerrs = []
            for r in results:
                row = r.loc[r["income_quintile"] == q]
                val = row["avg_delay"].values[0] if len(row) else np.nan
                heights.append(val)
                if error_bars and "se" in r.columns:
                    yerrs.append(row["se"].values[0] if len(row) else np.nan)

            x = np.arange(n_runs)
            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(x, heights, width=bar_width)
            if error_bars and len(yerrs) == n_runs:
                ax.errorbar(x, heights, yerr=yerrs, fmt="none", capsize=3, linewidth=1)

            ax.set_title(f"{title} — {x_labels_for_quintile[q]}", fontsize=title_fontsize, pad=8)
            ax.set_xlabel("Run", fontsize=axis_label_fontsize)
            ax.set_ylabel("Average delay", fontsize=axis_label_fontsize)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=tick_fontsize)
            ax.tick_params(axis="y", labelsize=tick_fontsize)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

            fig.tight_layout()

            if save_fig_dir is not None:
                out_path = Path(save_fig_dir) / fname_template.format(quintile=q.lower())
                fig.savefig(out_path, bbox_inches="tight", dpi=150)
                print(f"Saved figure: {out_path}")

            if show_plot:
                plt.show()
            plt.close(fig)
            figs.append(fig)

        # still optionally export per-run CSVs
        if output_csvs:
            for path, df_out in zip(output_csvs, results):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                df_out.to_csv(path, index=False)

        return results, figs

    # fallback: original grouped chart across all quintiles (unchanged)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(quintiles))
    total_width = min(0.9, 0.22 * n_runs)
    offsets = np.linspace(-total_width/2, total_width/2, n_runs) if n_runs > 1 else [0.0]
    for (label, res, dx) in zip(labels, results, offsets):
        heights = res["avg_delay"].to_numpy(dtype=float)
        ax.bar(x + dx, heights, width=0.22, label=label)
        if error_bars and "se" in res.columns:
            yerr = res["se"].to_numpy(dtype=float)
            ax.errorbar(x + dx, heights, yerr=yerr, fmt="none", capsize=3, linewidth=1)

    ax.set_title(title, fontsize=title_fontsize, pad=8)
    ax.set_xlabel("Income group (quintiles)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Average delay", fontsize=axis_label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels([x_labels_for_quintile[q] for q in quintiles], fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    if n_runs > 1:
        ax.legend(frameon=False, fontsize=legend_fontsize, loc="best")
    fig.tight_layout()
    if save_fig_path:
        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure to: {save_fig_path}")
    if show_plot:
        plt.show()
    plt.close(fig)

    if output_csvs:
        for path, df_out in zip(output_csvs, results):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(path, index=False)

    return results, fig


def plot_crossrun_mean_std_by_quintile(
    base_dirs,
    labels=None,
    last_n=100,
    income_csv=None,
    global_quintiles=True,
    use_observed_range_on_xlabel=True,
    income_decimals=0,
    figsize=(8, 4.5),
    bar_width=0.6,
    title="Average Delay by Income Quintile (mean ± std across runs)",
    title_fontsize=14,
    axis_label_fontsize=11,
    tick_fontsize=9,
    save_fig_path=None,
    show_plot=False,
    # --- NEW visual tweaks ---
    error_color="black",         # color of error bars
    error_linewidth=2,           # thickness of error bar lines
    error_capsize=6,             # cap size on error bars
    bar_color="skyblue",         # fill color for bars
    bar_edgecolor="black",       # outline color for bars
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    import contextlib, io

    # ---------- collect per-run per-agent (income, avg_delay) ----------
    base_dirs = [Path(b) for b in (base_dirs if isinstance(base_dirs, (list, tuple)) else [base_dirs])]
    n_runs = len(base_dirs)
    if labels is None:
        labels = [b.name for b in base_dirs]
    if len(labels) != n_runs:
        raise ValueError("labels must be same length as base_dirs")

    if income_csv is None or isinstance(income_csv, (str, Path)):
        income_csvs = [income_csv] * n_runs
    else:
        if len(income_csv) != n_runs:
            raise ValueError("income_csv list must match base_dirs length")
        income_csvs = income_csv

    per_run_agent = []
    for (bdir, inc_csv) in zip(base_dirs, income_csvs):
        """buf = io.StringIO()
        with contextlib.redirect_stdout(buf):"""
        df = compute_agent_delays(
            base_dir=str(bdir),
            last_n=last_n,
            route_order="appearance",
            income_csv=str(inc_csv) if inc_csv else None,
        )
        df = df[["id", "income", "avg_delay"]].copy()
        df["income"] = pd.to_numeric(df["income"], errors="coerce")
        df["avg_delay"] = pd.to_numeric(df["avg_delay"], errors="coerce")
        df = df.dropna(subset=["income", "avg_delay"])
        if df.empty:
            raise ValueError(f"No usable (income, avg_delay) data for run: {bdir}")
        per_run_agent.append(df)

    # ---------- build quintiles ----------
    def compute_bins(series):
        try:
            _, bins = pd.qcut(series, q=5, retbins=True, duplicates="drop")
            bins = np.unique(bins)
        except ValueError:
            bins = np.linspace(series.min(), series.max(), 6)
        if len(bins) < 6:
            bins = np.linspace(series.min(), series.max(), 6)
        return bins

    if global_quintiles:
        all_incomes = pd.concat(per_run_agent, ignore_index=True)["income"]
        global_bins = compute_bins(all_incomes)
        bins_per_run = [global_bins] * n_runs
    else:
        bins_per_run = [compute_bins(df["income"]) for df in per_run_agent]

    quintiles = [f"Q{i}" for i in range(1, 6)]
    per_run_means = []
    per_run_minmax = []

    for df, bins in zip(per_run_agent, bins_per_run):
        labels_q = [f"Q{i}" for i in range(1, len(bins))]
        cats = pd.cut(df["income"], bins=bins, include_lowest=True, labels=labels_q, right=True, duplicates="drop")
        tmp = df.assign(income_quintile=cats).dropna(subset=["income_quintile"])
        grp = tmp.groupby("income_quintile", observed=True)

        run_mean = grp["avg_delay"].mean().reindex(quintiles)
        per_run_means.append(run_mean.to_numpy(dtype=float))

        per_run_minmax.append(pd.DataFrame({
            "income_min": grp["income"].min().reindex(quintiles),
            "income_max": grp["income"].max().reindex(quintiles),
        }))

    # ---------- collapse across runs ----------
    mat = np.vstack(per_run_means)
    mean_across_runs = np.nanmean(mat, axis=0)
    std_across_runs = np.nanstd(mat, axis=0, ddof=1 if n_runs > 1 else 0)

    # ---------- x-tick labels ----------
    def fmt_income(x):
        if pd.isna(x):
            return "NA"
        return f"{x:.0f}" if income_decimals == 0 else f"{x:.{income_decimals}f}"

    if use_observed_range_on_xlabel:
        mins, maxs = [], []
        for q_idx, q in enumerate(quintiles):
            q_mins = [df.iloc[q_idx]["income_min"] for df in per_run_minmax]
            q_maxs = [df.iloc[q_idx]["income_max"] for df in per_run_minmax]
            mins.append(np.nanmin(q_mins))
            maxs.append(np.nanmax(q_maxs))
        x_labels = [f"{q}\n[{fmt_income(mn)} – {fmt_income(mx)}]" for q, mn, mx in zip(quintiles, mins, maxs)]
    else:
        x_labels = quintiles

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(quintiles))

    bars = ax.bar(
        x, mean_across_runs,
        width=bar_width,
        color=bar_color,
        edgecolor=bar_edgecolor,
        linewidth=1.2
    )

    # error bars (±1 std across runs)
    ax.errorbar(
        x, mean_across_runs, yerr=std_across_runs,
        fmt="none",
        ecolor=error_color,
        elinewidth=error_linewidth,
        capsize=error_capsize,
        capthick=error_linewidth
    )

    ax.set_title(title, fontsize=title_fontsize, pad=8)
    ax.set_xlabel("Income group (quintiles)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Average delay (mean ± std across runs)", fontsize=axis_label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    if save_fig_path:
        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure to: {save_fig_path}")
    if show_plot:
        plt.show()
    plt.close(fig)

    return {
        "mean": mean_across_runs,
        "std": std_across_runs,
        "matrix": mat,
        "x_labels": x_labels
    }, fig


BASE_DIRS = [
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_5",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_6",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_8",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_9",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_10",
]
LABELS = ["Run 1", "Run 2", "Run 3", "Run 4", "Run 5"]

_, _ = plot_crossrun_mean_std_by_quintile(
    base_dirs=BASE_DIRS,
    labels=LABELS,
    last_n=10,
    global_quintiles=True,
    use_observed_range_on_xlabel=True,
    income_decimals=0,
    save_fig_path="imgs/avg_delay_by_income_quintile_mean_std.png",
    show_plot=True,
    # emphasize error bars
    error_color="black",
    error_linewidth=2.5,
    error_capsize=8,
)
