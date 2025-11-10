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
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_5",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_6",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_7",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_8",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_9",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_10",
]

LABELS = ["Run 1", "Run 2", "Run 3", "Run 4", "Run 5"]

LAST_N_EPISODES = 10  # number of last episodes to consider

# If you know the agent column name, set it here; otherwise leave as None to auto-detect.
AGENT_COL = None  # e.g., "agent_id", "agent", or "id"


# ============================================================
# -------------------- CORE FUNCTIONS ------------------------
# ============================================================

def list_last_n_episode_files(episodes_dir: Path, last_n: int) -> list[tuple[int, Path]]:
    """Return (episode_number, csv_path) for the last N episodes."""
    episodes_dir = Path(episodes_dir)
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Expected episodes directory not found: {episodes_dir}")

    episode_files = []
    for p in glob.glob(str(episodes_dir / "ep*.csv")):
        m = re.search(r"ep(\d+)\.csv$", Path(p).name)
        if m:
            episode_files.append((int(m.group(1)), Path(p)))

    if not episode_files:
        raise FileNotFoundError(f"No episode CSVs found in {episodes_dir}")

    episode_files.sort(key=lambda x: x[0])
    return episode_files[-last_n:] if last_n and last_n > 0 else episode_files


def detect_agent_column(df: pd.DataFrame, forced: str | None = None) -> str:
    """Return the agent id column name. Try a few common options if not forced."""
    if forced:
        if forced not in df.columns:
            raise ValueError(f"Forced AGENT_COL='{forced}' not found in columns: {list(df.columns)}")
        return forced
    candidates = ["agent_id", "agent", "id"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not detect agent id column. Tried: {candidates}. Columns: {list(df.columns)}")


def read_episode_subset(episode_csv: Path, agent_col_hint: str | None = None) -> pd.DataFrame:
    """
    Read one episode CSV and return DataFrame with [agent_col, travel_time].
    Coerces travel_time to numeric and drops rows missing agent or travel_time.
    """
    df = pd.read_csv(episode_csv, on_bad_lines="skip")
    if "travel_time" not in df.columns:
        raise ValueError(f"'travel_time' column not found in {episode_csv}")
    agent_col = detect_agent_column(df, forced=agent_col_hint)
    sub = df[[agent_col, "travel_time"]].copy()
    sub.rename(columns={agent_col: "agent_id"}, inplace=True)  # unify name
    sub["travel_time"] = pd.to_numeric(sub["travel_time"], errors="coerce")
    sub = sub.dropna(subset=["agent_id", "travel_time"])
    return sub


# ============================================================
# ----------------------- MAIN SCRIPT ------------------------
# ============================================================

if __name__ == "__main__":

    # Define output directory relative to this script’s location
    SCRIPT_DIR = Path(__file__).resolve().parent
    CSV_DIR = SCRIPT_DIR / "csvs"
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Define save paths inside csvs/
    SAVE_PER_RUN_AGENT_MEANS_CSV = CSV_DIR / "per_run_agent_avg_travel_time_last10.csv"
    SAVE_AGENT_COMPARISON_WIDE_CSV = CSV_DIR / "agent_avg_travel_time_across_replications_last10.csv"
    SAVE_AGENT_SUMMARY_CSV = CSV_DIR / "agent_across_replications_summary_last10.csv"

    # Will store, for each run, a DataFrame: [run, agent_id, mean_travel_time_lastN]
    per_run_agent_means = []

    # Also keep track of which episodes were used per run (for info/printing)
    run_to_episode_range = {}

    for base_dir, label in zip(BASE_DIRS, LABELS):
        print(f"\nProcessing {label} ...")
        episode_files = list_last_n_episode_files(Path(base_dir) / "episodes", LAST_N_EPISODES)

        # Read and stack per-episode [agent_id, travel_time]
        per_episode_frames = []
        for ep_num, ep_path in episode_files:
            ep_df = read_episode_subset(ep_path, agent_col_hint=AGENT_COL)
            ep_df["episode"] = ep_num
            per_episode_frames.append(ep_df)

        if not per_episode_frames:
            raise RuntimeError(f"No valid episode data for {label}")

        run_df = pd.concat(per_episode_frames, ignore_index=True)

        # Per-agent mean travel_time across the selected episodes
        agent_means = (
            run_df.groupby("agent_id", as_index=False)["travel_time"]
                 .mean()
                 .rename(columns={"travel_time": "mean_travel_time_lastN"})
        )
        agent_means.insert(0, "run", label)  # add run label
        per_run_agent_means.append(agent_means)

        # Track episode range used
        ep_nums = sorted(set(run_df["episode"]))
        run_to_episode_range[label] = (min(ep_nums), max(ep_nums), len(ep_nums))

    # Combine per-run agent means (long form)
    all_runs_long = pd.concat(per_run_agent_means, ignore_index=True)

    # Build wide comparison table: rows = agent_id, columns = run labels
    comparison_wide = all_runs_long.pivot(index="agent_id", columns="run", values="mean_travel_time_lastN")

    # Build per-agent across-replications summary
    summary = pd.DataFrame(index=comparison_wide.index)
    summary["mean_across_runs"] = comparison_wide.mean(axis=1, skipna=True)
    summary["std_across_runs"]  = comparison_wide.std(axis=1, ddof=1, skipna=True)
    summary["min_across_runs"]  = comparison_wide.min(axis=1, skipna=True)
    summary["max_across_runs"]  = comparison_wide.max(axis=1, skipna=True)
    summary["best_run"]  = comparison_wide.idxmin(axis=1, skipna=True)
    summary["worst_run"] = comparison_wide.idxmax(axis=1, skipna=True)

    # -------- Print results --------
    with pd.option_context("display.float_format", lambda v: f"{v:.6g}"):
        print("\n=== PER-RUN: mean travel time per agent over last N episodes ===")
        for label in LABELS:
            sub = all_runs_long[all_runs_long["run"] == label].copy()
            if sub.empty:
                print(f"\n{label}: (no data)")
                continue
            a, b, k = run_to_episode_range[label]
            print(f"\n{label}  [episodes used: {k} | range: {a}..{b}]")
            print(sub.sort_values("agent_id")[["agent_id", "mean_travel_time_lastN"]].to_string(index=False))

        print("\n=== COMPARISON: per-agent mean travel time across replications (wide) ===")
        print(comparison_wide.sort_index().to_string())

        print("\n=== SUMMARY: per-agent stats across replications ===")
        out = summary.reset_index().rename(columns={"index": "agent_id"})
        print(out.sort_values("agent_id").to_string(index=False))

    # -------- Save outputs --------
    all_runs_long.to_csv(SAVE_PER_RUN_AGENT_MEANS_CSV, index=False)
    print(f"\nSaved per-run agent means to: {SAVE_PER_RUN_AGENT_MEANS_CSV}")

    comparison_wide.to_csv(SAVE_AGENT_COMPARISON_WIDE_CSV)
    print(f"Saved agent comparison (wide) to: {SAVE_AGENT_COMPARISON_WIDE_CSV}")

    summary.reset_index().to_csv(SAVE_AGENT_SUMMARY_CSV, index=False)
    print(f"Saved per-agent across-replications summary to: {SAVE_AGENT_SUMMARY_CSV}")

    # -------- Overall std among agent mean travel times --------
    overall_std_among_agents = summary["mean_across_runs"].std(ddof=1)
    print(f"\n=== OVERALL STD among mean travel times of all agents: {overall_std_among_agents:.6f} ===")
