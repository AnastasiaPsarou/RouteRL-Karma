#!/usr/bin/env python3
"""
Compute average rewards for routes 0, 1, and 2 across all CSV files in a folder,
and plot both:
  - Agents by route with highest average reward
  - Average reward per episode by route

How to use:
    1. Set FOLDER_PATH below to your folder containing CSV files.
    2. Run:  python avg_rewards_by_route.py
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# === USER PARAMETERS ===
FOLDER_PATH = r"../scenarios/monetary_pricing/training_records_monetary_pricing_300_agents_fee_0_5/episodes"   # <-- CHANGE THIS to your folder path
#FOLDER_PATH = r"../data/training_records_300_agents/training_records_monetary_pricing_300_agents_long/episodes"   # <-- CHANGE THIS to your folder path
RECURSIVE = False                       # set True if you want to include subfolders
ROUTES = [0, 1, 2]                      # Routes to track


def find_csvs(folder: str, recursive: bool) -> list[str]:
    if recursive:
        return glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    return glob.glob(os.path.join(folder, "*.csv"))


def read_needed_columns(path: str) -> pd.DataFrame | None:
    """
    Read CSV and extract 'id', 'action', 'reward', 'travel_time' with resilience to
    extra columns, bad lines, and header quirks.
    """
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return None

    df.columns = [str(c).strip() for c in df.columns]

    # Fallback renaming by index
    if "travel_time" not in df.columns and df.shape[1] > 0:
        df = df.rename(columns={df.columns[0]: "travel_time"})
    if "id" not in df.columns and df.shape[1] > 1:
        df = df.rename(columns={df.columns[1]: "id"})
    if "action" not in df.columns and df.shape[1] > 3:
        df = df.rename(columns={df.columns[3]: "action"})
    if "reward" not in df.columns and df.shape[1] > 8:
        df = df.rename(columns={df.columns[8]: "reward"})

    needed = {"id", "action", "reward", "travel_time"}
    if not needed.issubset(df.columns):
        print(f"Warning: skipping {path} (missing one of {needed}).")
        return None

    out = df[["id", "action", "reward", "travel_time"]].copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out["action"] = pd.to_numeric(out["action"], errors="coerce")
    out["reward"] = pd.to_numeric(out["reward"], errors="coerce")
    out["travel_time"] = pd.to_numeric(out["travel_time"], errors="coerce")
    out = out.dropna(subset=["id", "action", "reward", "travel_time"])
    out = out[out["action"].isin(ROUTES)]
    out["id"] = out["id"].astype(int)
    out["action"] = out["action"].astype(int)
    return out


def main():
    if not os.path.isdir(FOLDER_PATH):
        raise NotADirectoryError(f"Not a folder: {FOLDER_PATH}")

    files = find_csvs(FOLDER_PATH, RECURSIVE)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {FOLDER_PATH} (toggle RECURSIVE if needed).")

    per_episode_rows = []
    per_episode_reward_rows = []  # per-episode avg reward by route
    frames = []

    for f in files:
        sub = read_needed_columns(f)
        if sub is None or sub.empty:
            continue
        frames.append(sub)

        # per-episode counts by route
        counts = sub["action"].value_counts().reindex(ROUTES, fill_value=0).astype(int)
        per_episode_rows.append({
            "episode": os.path.basename(f),
            **{f"count_route_{r}": int(counts.loc[r]) for r in ROUTES}
        })

        # per-episode average REWARD by route (this replaces travel time plot)
        reward_means = (
            sub.groupby("action")["reward"]
               .mean()
               .reindex(ROUTES)
        )
        per_episode_reward_rows.append({
            "episode": os.path.basename(f),
            **{f"avg_reward_route_{r}": (None if pd.isna(reward_means.loc[r]) else float(reward_means.loc[r])) for r in ROUTES}
        })

    if not frames:
        raise ValueError("No usable rows with needed columns found.")

    all_df = pd.concat(frames, ignore_index=True)

    # 1) Overall average rewards by route
    avg_rewards = (
        all_df.groupby("action", as_index=True)["reward"]
              .mean()
              .rename("avg_reward")
              .to_frame()
              .reindex(ROUTES)
    )

    # 2) Per-episode counts
    counts_df = pd.DataFrame(per_episode_rows).sort_values("episode").reset_index(drop=True)

    # 3) Per-agent average reward per route
    per_agent_avg = (
        all_df.groupby(["id", "action"])["reward"]
              .mean()
              .rename("avg_reward")
              .reset_index()
    )
    per_agent_pivot = per_agent_avg.pivot(index="id", columns="action", values="avg_reward")
    per_agent_pivot = per_agent_pivot.reindex(columns=ROUTES)
    per_agent_pivot.columns = [f"avg_reward_route_{r}" for r in ROUTES]

    # 4) Route with HIGHEST average reward per agent
    max_route = per_agent_pivot.idxmax(axis=1).str.replace("avg_reward_route_", "", regex=False).astype(int)
    per_agent_max = pd.DataFrame({
        "id": per_agent_pivot.index,
        "max_avg_reward_route": max_route
    }).reset_index(drop=True)

    # 5) Per-episode average reward table
    reward_df = pd.DataFrame(per_episode_reward_rows).sort_values("episode").reset_index(drop=True)

    # ---- Print summaries ----
    print("\nAverage rewards by route (across ALL episodes):\n")
    print(avg_rewards.to_string())

    print("\nNumber of agents choosing each route BY EPISODE:\n")
    print(counts_df.to_string(index=False))

    print("\nPer-agent route with the HIGHEST average reward (ALL agents):\n")
    print(per_agent_max.to_string(index=False))

    # ---- Figure with two subplots ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # A) Bar chart: agents by route with highest avg reward
    counts_by_max_route = per_agent_max["max_avg_reward_route"].value_counts().reindex(ROUTES, fill_value=0)
    axes[0].bar(range(len(ROUTES)), counts_by_max_route.values)
    axes[0].set_title("Agents by Route with Highest Average Reward")
    axes[0].set_xlabel("Route")
    axes[0].set_ylabel("Number of Agents")
    axes[0].set_xticks(range(len(ROUTES)))
    axes[0].set_xticklabels([str(r) for r in ROUTES])

    # B) Line plot: per-episode average REWARD per route
    episodes = reward_df["episode"].tolist()
    x = range(len(episodes))
    for r in ROUTES:
        y = reward_df[f"avg_reward_route_{r}"].tolist()
        axes[1].plot(x, y, marker="o", label=f"Route {r}")
    axes[1].set_title("Average Reward per Episode by Route")
    axes[1].set_xlabel("Episode (CSV file)")
    axes[1].set_ylabel("Average Reward")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(episodes, rotation=45, ha="right")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("imgs/avg_rewards_summary.png", dpi=300)  # <-- saves the figure
    plt.show()  # display only

if __name__ == "__main__":
    main()
