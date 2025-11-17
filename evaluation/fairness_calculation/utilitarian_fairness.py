#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import glob
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# ------------------- USER PARAMETERS ------------------------
# ============================================================

BASE_DIRS = [
    r"../../data/training_records_400_agents/training_records_monetary_pricing_400_agents_fee_0_05_urgency_obs_seed_7",
    r"../../data/training_records_400_agents/training_records_monetary_pricing_400_agents_fee_0_05_urgency_obs_seed_8",
    r"../../data/training_records_400_agents/training_records_monetary_pricing_400_agents_fee_0_05_urgency_obs_seed_9",
]
LABELS = ["Run 1", "Run 2", "Run 3"]

LAST_N_EPISODES = 10  # how many latest episodes to use per run

# Optional CSV outputs (set to None to skip saving)
SAVE_EPISODE_TOTALS_CSV = "csvs/episode_total_travel_times_last10.csv"        # per-run, per-episode totals
SAVE_REPLICATION_SUMMARY_CSV = "csvs/replication_summary_by_relative_episode.csv"  # across runs mean/std per relative episode


# ============================================================
# -------------------- CORE FUNCTIONS ------------------------
# ============================================================

def list_last_n_episode_files(episodes_dir: Path, last_n: int) -> list[tuple[int, Path]]:
    """Return [(episode_number, csv_path)] for the last_n episodes under episodes_dir."""
    episodes_dir = Path(episodes_dir)
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Expected episodes directory not found: {episodes_dir}")

    episode_files = []
    for p in glob.glob(str(episodes_dir / "ep*.csv")):
        m = re.search(r"ep(\d+)\.csv$", Path(p).name)
        if m:
            episode_files.append((int(m.group(1)), Path(p)))

    if not episode_files:
        raise FileNotFoundError(f"No episode CSVs matching 'ep{{number}}.csv' found in {episodes_dir}")

    episode_files.sort(key=lambda x: x[0])
    return episode_files[-last_n:] if last_n and last_n > 0 else episode_files


def total_travel_time_for_episode(episode_csv: Path) -> float:
    """Sum of travel_time for a single episode CSV."""
    df = pd.read_csv(episode_csv, on_bad_lines="skip")
    if "travel_time" not in df.columns:
        raise ValueError(f"'travel_time' column not found in {episode_csv}")
    tt = pd.to_numeric(df["travel_time"], errors="coerce")
    return float(tt.dropna().sum())


# ============================================================
# ----------------------- MAIN SCRIPT ------------------------
# ============================================================

if __name__ == "__main__":

    per_run_episode_totals = []   # rows: run, episode, total_travel_time, pos_from_end

    for base_dir, label in zip(BASE_DIRS, LABELS):
        episode_files = list_last_n_episode_files(Path(base_dir) / "episodes", LAST_N_EPISODES)

        # compute totals for these last N episodes
        rows = []
        for ep_num, ep_path in episode_files:
            total_tt = total_travel_time_for_episode(ep_path)
            rows.append({"episode": ep_num, "total_travel_time": total_tt})

        df = pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)

        # add relative position from the end:
        # 1 = last episode, 2 = second last, ..., LAST_N_EPISODES = earliest among the selected
        df["pos_from_end"] = (LAST_N_EPISODES - np.arange(len(df))[::-1]).astype(int)
        # After sorting ascending by episode, above formula gives 1..N incorrectly; fix with:
        # recompute by ranking from the end explicitly
        df = df.sort_values("episode").reset_index(drop=True)
        df["pos_from_end"] = (len(df) - df.reset_index().index).astype(int)

        df.insert(0, "run", label)
        per_run_episode_totals.append(df)

    # Combine all runs
    episode_totals_df = pd.concat(per_run_episode_totals, ignore_index=True)

    # --- Replication summary: for each relative position, mean/std across runs ---
    replication_summary = (
        episode_totals_df
        .groupby("pos_from_end", as_index=False)
        .agg(
            mean_total_travel_time=("total_travel_time", "mean"),
            std_total_travel_time=("total_travel_time", "std"),
            n_runs=("total_travel_time", "size"),
        )
        .sort_values("pos_from_end", ascending=False)  # show from earliest among selected to last
        .reset_index(drop=True)
    )

    # ---- Print results ----
    print("\n=== Per-run episode totals (last N episodes) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.6g}"):
        for label in LABELS:
            sub = episode_totals_df[episode_totals_df["run"] == label].copy()
            sub = sub.sort_values("episode")
            print(f"\n{label}")
            print(sub[["episode", "pos_from_end", "total_travel_time"]].to_string(index=False))

    print("\n=== Replication summary (by relative episode position) ===")
    print("pos_from_end: 1 = last, 2 = second last, ..., "
          f"{LAST_N_EPISODES} = earliest among the selected.")
    with pd.option_context("display.float_format", lambda v: f"{v:.6g}"):
        print(replication_summary.to_string(index=False))

    # ---- Optional CSV saves ----
    if SAVE_EPISODE_TOTALS_CSV:
        Path(SAVE_EPISODE_TOTALS_CSV).parent.mkdir(parents=True, exist_ok=True)
        episode_totals_df.to_csv(SAVE_EPISODE_TOTALS_CSV, index=False)
        print(f"\nSaved per-episode totals to: {SAVE_EPISODE_TOTALS_CSV}")

    if SAVE_REPLICATION_SUMMARY_CSV:
        Path(SAVE_REPLICATION_SUMMARY_CSV).parent.mkdir(parents=True, exist_ok=True)
        replication_summary.to_csv(SAVE_REPLICATION_SUMMARY_CSV, index=False)
        print(f"Saved replication summary to: {SAVE_REPLICATION_SUMMARY_CSV}")
