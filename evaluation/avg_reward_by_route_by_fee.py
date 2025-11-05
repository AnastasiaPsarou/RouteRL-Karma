#!/usr/bin/env python3
"""
Compare multiple fee folders:

Row 1: Average reward per episode by route (one subplot per fee)
Row 2: Counts of agents whose highest average reward is on each route (bar plot per fee),
       annotated with:
         - exact income values of agents whose max-avg-reward route is 0
         - their percentile (%) relative to all agents' incomes in that fee

Outputs:
  - imgs/avg_reward_by_route_by_fee_with_bars.png
  - imgs/income_route0_fee_<fee>.csv  (per fee: id, income, percentile)
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

# === USER PARAMETERS ===
FOLDER_PATHS = [
    r"../scenarios/monetary_pricing/training_records_monetary_pricing_300_agents_fee_0_1/episodes",
    r"../scenarios/monetary_pricing/training_records_monetary_pricing_300_agents_fee_0_25/episodes",
    r"../scenarios/monetary_pricing/training_records_monetary_pricing_300_agents_fee_0_5/episodes",
    r"../scenarios/monetary_pricing/training_records_monetary_pricing_300_agents_fee_1/episodes",
]
RECURSIVE = False
ROUTES = [0, 1, 2]

# === PLOT STYLE PARAMETERS ===
TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 14
ANNOTATION_FONTSIZE = 14
ANNOTATION_FONTWEIGHT = 'bold'

# place the Route 0 income/percentile box below the barplot
ANNOTATION_BELOW = True
ANNOTATION_BELOW_Y = -0.25   # how far below (axes fraction; negative goes outside)



def find_csvs(folder: str, recursive: bool) -> list[str]:
    if recursive:
        return glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    return glob.glob(os.path.join(folder, "*.csv"))


def read_needed_columns(path: str) -> pd.DataFrame | None:
    """Read CSV and extract needed columns with resilience to quirks."""
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return None

    df.columns = [str(c).strip() for c in df.columns]

    # Fallback renaming by index (based on known layout)
    # travel_time=0, id=1, action=3, income=7, reward=8
    if "travel_time" not in df.columns and df.shape[1] > 0:
        df = df.rename(columns={df.columns[0]: "travel_time"})
    if "id" not in df.columns and df.shape[1] > 1:
        df = df.rename(columns={df.columns[1]: "id"})
    if "action" not in df.columns and df.shape[1] > 3:
        df = df.rename(columns={df.columns[3]: "action"})
    if "income" not in df.columns and df.shape[1] > 7:
        df = df.rename(columns={df.columns[7]: "income"})
    if "reward" not in df.columns and df.shape[1] > 8:
        df = df.rename(columns={df.columns[8]: "reward"})

    needed = {"id", "action", "reward", "travel_time", "income"}
    if not needed.issubset(df.columns):
        print(f"Warning: skipping {path} (missing one of {needed}).")
        return None

    out = df[["id", "action", "reward", "travel_time", "income"]].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["id", "action", "reward", "travel_time", "income"])
    out = out[out["action"].isin(ROUTES)]
    out["id"] = out["id"].astype(int)
    out["action"] = out["action"].astype(int)
    return out


def extract_fee_from_path(folder_path: str) -> float | None:
    m = re.search(r"fee_([0-9_]+)", folder_path)
    if not m:
        return None
    fee_str = m.group(1).replace("_", ".")
    try:
        return float(fee_str)
    except ValueError:
        return None


def load_folder_data(folder: str):
    """Return per-episode rewards, agent counts, and route-0 incomes/percentiles."""
    files = find_csvs(folder, RECURSIVE)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder} (toggle RECURSIVE if needed).")

    per_episode_reward_rows = []
    frames = []

    for f in files:
        sub = read_needed_columns(f)
        if sub is None or sub.empty:
            continue
        frames.append(sub)

        reward_means = sub.groupby("action")["reward"].mean().reindex(ROUTES)
        per_episode_reward_rows.append({
            "episode": os.path.basename(f),
            **{f"avg_reward_route_{r}": (None if pd.isna(reward_means.loc[r]) else float(reward_means.loc[r]))
               for r in ROUTES}
        })

    if not frames:
        raise ValueError(f"No usable rows found in {folder}.")

    reward_df = pd.DataFrame(per_episode_reward_rows).sort_values("episode").reset_index(drop=True)
    all_df = pd.concat(frames, ignore_index=True)

    # Per-agent averages and income
    per_agent_avg = (
        all_df.groupby(["id", "action"])["reward"]
        .mean().unstack("action").reindex(columns=ROUTES)
    ).dropna(how="all")

    per_agent_income = all_df.groupby("id")["income"].median()
    income_percentiles = per_agent_income.rank(pct=True, method="average") * 100.0
    max_route_per_agent = per_agent_avg.idxmax(axis=1)

    counts_by_max_route = (
        max_route_per_agent.value_counts().reindex(ROUTES, fill_value=0).astype(int)
    )

    ids_route0 = max_route_per_agent[max_route_per_agent == 0].index
    route0_df = pd.DataFrame({
        "id": ids_route0,
        "income": per_agent_income.reindex(ids_route0).values,
        "percentile": income_percentiles.reindex(ids_route0).values
    }).dropna().sort_values("income").reset_index(drop=True)

    return reward_df, counts_by_max_route, route0_df


def main():
    if not FOLDER_PATHS:
        raise ValueError("Please provide at least one folder in FOLDER_PATHS.")

    os.makedirs("imgs", exist_ok=True)

    items = []
    for folder in FOLDER_PATHS:
        if not os.path.isdir(folder):
            raise NotADirectoryError(f"Not a folder: {folder}")
        fee = extract_fee_from_path(folder)
        fee_label = "unknown" if fee is None else fee
        reward_df, counts_by_max_route, route0_df = load_folder_data(folder)
        items.append((fee, fee_label, folder, reward_df, counts_by_max_route, route0_df))

    items.sort(key=lambda t: (float("inf") if t[0] is None else t[0]))

    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10), sharey="row")
    if n == 1:
        axes = axes.reshape(2, 1)

    global_max_count = max(int(counts.max()) for *_, counts, _ in items if len(counts) > 0)

    # --- Row 1: line plots ---
    for col, (fee, fee_label, _, reward_df, _, _) in enumerate(items):
        ax = axes[0, col]
        episodes = reward_df["episode"].tolist()
        x = range(len(episodes))
        for r in ROUTES:
            y = reward_df[f"avg_reward_route_{r}"].tolist()
            ax.plot(x, y, marker="o", label=f"Route {r}")
        title_fee = "unknown" if fee is None else f"{fee_label}"
        ax.set_title(f"Average Reward per Episode — Fee = {title_fee}", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Episode (CSV file)", fontsize=AXIS_LABEL_FONTSIZE)
        
        # Show every 2nd tick to avoid clutter
        skip = 2
        xticks = list(range(0, len(episodes), skip))
        ax.set_xticks(xticks)
        ax.set_xticklabels([episodes[i] for i in xticks], rotation=0, ha="center", fontsize=TICK_FONTSIZE)

        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    axes[0, 0].set_ylabel("Average Reward", fontsize=AXIS_LABEL_FONTSIZE)

    handles, labels = axes[0, min(n - 1, 0)].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ROUTES),
               bbox_to_anchor=(0.5, 0.995), fontsize=LEGEND_FONTSIZE)

    # --- Row 2: bar plots + Route 0 annotation ---
    for col, (fee, fee_label, folder, _, counts, route0_df) in enumerate(items):
        ax = axes[1, col]
        x = list(range(len(ROUTES)))
        heights = [int(counts.get(r, 0)) for r in ROUTES]
        bars = ax.bar(x, heights)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in ROUTES], fontsize=TICK_FONTSIZE)
        title_fee = "unknown" if fee is None else f"{fee_label}"
        ax.set_title(f"Agents by Highest Avg Reward — Fee = {title_fee}", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Route", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
        ax.set_ylim(0, max(global_max_count * 1.1, 1))
        for rect, h in zip(bars, heights):
            ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=ANNOTATION_FONTSIZE)

        # --- annotate Route 0 with exact incomes + percentiles ---
        # --- annotate Route 0 with exact incomes + percentiles (one per line) ---
        # --- annotate Route 0 with exact incomes + percentiles (3 per line) ---
        if not route0_df.empty:
            fee_str = "unknown" if fee is None else str(fee).replace(".", "_")
            csv_path = f"imgs/income_route0_fee_{fee_str}.csv"
            route0_df.to_csv(csv_path, index=False)

            # Build entries like "42.0 (85.3%)"
            entries = [f"{inc:.1f} ({pct:.1f}%)"
                    for inc, pct in zip(route0_df["income"], route0_df["percentile"])]

            # Group entries into lines of 3
            grouped_lines = [", ".join(entries[i:i+3]) for i in range(0, len(entries), 3)]
            list_text = "\n".join(grouped_lines)

            text = (f"Route 0 incomes ({len(entries)} agents):\n{list_text}")

            if ANNOTATION_BELOW:
                # Place the box centered below the plot
                ax.annotate(text,
                            xy=(0.5, ANNOTATION_BELOW_Y), xycoords="axes fraction",
                            ha="center", va="top",
                            fontsize=ANNOTATION_FONTSIZE, fontweight=ANNOTATION_FONTWEIGHT,
                            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
                            annotation_clip=False)
            else:
                # Original: above the Route 0 bar
                idx0 = ROUTES.index(0)
                rect0 = bars[idx0]
                y0 = rect0.get_height()
                ax.annotate(text,
                            xy=(rect0.get_x() + rect0.get_width()/2, y0),
                            xytext=(0, 34), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=ANNOTATION_FONTSIZE, fontweight=ANNOTATION_FONTWEIGHT,
                            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2))

    axes[1, 0].set_ylabel("Number of Agents", fontsize=AXIS_LABEL_FONTSIZE)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = "imgs/avg_reward_by_route_by_fee_with_bars.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    #plt.show()
    print(f"\nSaved combined figure to: {out_path}")


if __name__ == "__main__":
    main()
