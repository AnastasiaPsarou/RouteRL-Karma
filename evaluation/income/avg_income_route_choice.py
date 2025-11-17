import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, re
from pathlib import Path

def compute_avg_income_by_action(base_dir, last_n=10, income_csv=None):
    """Compute average income per route (action) for the last N episodes in one run."""
    base_dir = Path(base_dir)
    episodes_dir = base_dir / "episodes"

    # --- 1) Find last N episode files ---
    episode_files = []
    for p in glob.glob(str(episodes_dir / "ep*.csv")):
        m = re.search(r"ep(\d+)\.csv$", Path(p).name)
        if m:
            episode_files.append((int(m.group(1)), Path(p)))
    if not episode_files:
        raise FileNotFoundError(f"No episode files found in {episodes_dir}")

    episode_files.sort(key=lambda x: x[0])
    selected = episode_files[-last_n:]
    dfs = []
    for ep, path in selected:
        df = pd.read_csv(path, on_bad_lines="skip")
        df["episode"] = ep
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True).dropna(how="all")

    if "id" not in df_all.columns or "action" not in df_all.columns:
        raise ValueError("Episode files must contain 'id' and 'action' columns")

    # --- 2) Handle income ---
    if "income" in df_all.columns:
        df_all["income"] = pd.to_numeric(df_all["income"], errors="coerce")
    elif income_csv:
        df_income = pd.read_csv(income_csv)
        lower = {c.lower(): c for c in df_income.columns}
        id_col = lower.get("id")
        inc_col = lower.get("income") or lower.get("agent_income")
        df_income = df_income.rename(columns={id_col: "id", inc_col: "income"})[["id", "income"]]
        df_all = df_all.merge(df_income, on="id", how="left")
    else:
        raise ValueError("No income information found (provide income_csv or include 'income' in episodes).")

    df_all = df_all.dropna(subset=["income", "action"])
    df_all["action"] = df_all["action"].astype(int)

    # --- 3) Compute mean income per action ---
    df_summary = (
        df_all.groupby("action", as_index=False)["income"]
              .mean()
              .rename(columns={"income": "avg_income"})
    )

    # Ensure actions 0, 1, 2 appear even if missing
    all_actions = pd.DataFrame({"action": [0, 1, 2]})
    df_summary = all_actions.merge(df_summary, on="action", how="left")

    return df_summary


def plot_avg_income_by_action_across_runs(
    base_dirs,
    labels=None,
    last_n=10,
    income_csvs=None,
    figsize=(6, 4.5),
    bar_color="skyblue",
    edge_color="black",
    bar_width=0.6,
    title="Average Income by Route (mean ± std across replications)",
    title_fontsize=14,
    axis_label_fontsize=12,
    tick_fontsize=10,
    save_fig_path=None,
    show_plot=True,
):
    """Compute and plot mean ± std of average income per route across replications."""

    base_dirs = [Path(b) for b in base_dirs]
    n = len(base_dirs)
    if labels is None:
        labels = [b.name for b in base_dirs]
    if income_csvs is None:
        income_csvs = [None] * n

    per_run_results = []
    for bdir, inc_csv, label in zip(base_dirs, income_csvs, labels):
        df = compute_avg_income_by_action(bdir, last_n=last_n, income_csv=inc_csv)
        df["run"] = label
        per_run_results.append(df)

    # --- Combine results ---
    df_all = pd.concat(per_run_results, ignore_index=True)

    # --- Compute mean ± std across runs ---
    stats = (
        df_all.groupby("action", as_index=False)["avg_income"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_income", "std": "std_income"})
    )

    print("\nMean ± Std of average income by action across runs:")
    print(stats.to_string(index=False))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(stats))
    
    # --- Bars ---
    bars = ax.bar(
        x,
        stats["mean_income"],
        color=bar_color,
        edgecolor=edge_color,
        width=bar_width,
    )

    # --- Error bars (± std) ---
    ax.errorbar(
        x,
        stats["mean_income"],
        yerr=stats["std_income"],
        fmt="none",
        ecolor="black",
        elinewidth=2,
        capsize=8,
        capthick=2,
    )


    ax.set_xticks(x)
    ax.set_xticklabels(stats["action"].astype(str), fontsize=tick_fontsize)
    ax.set_xlabel("Route (action)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Average income", fontsize=axis_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=8)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    if save_fig_path:
        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure to: {save_fig_path}")

    if show_plot:
        plt.show()
    plt.close(fig)

    return stats, df_all


BASE_DIRS = [
    r"../../data/training_records_400_agents/training_records_monetary_pricing_400_agents_fee_0_05_urgency_obs_seed_7",
    r"../../data/training_records_400_agents/training_records_monetary_pricing_400_agents_fee_0_05_urgency_obs_seed_8",
    r"../../data/training_records_400_agents/training_records_monetary_pricing_400_agents_fee_0_05_urgency_obs_seed_9",
]
LABELS = ["Run 1", "Run 2", "Run 3"]

plot_avg_income_by_action_across_runs(
    base_dirs=BASE_DIRS,
    labels=LABELS,
    last_n=100,
    save_fig_path="imgs/avg_income_by_route_mean_std.png",
    show_plot=True,
    title_fontsize=18,
    axis_label_fontsize=14,
    tick_fontsize=12,
)
