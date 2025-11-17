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

LAST_N_EPISODES = 1010  # how many of the latest episodes to analyze

SAVE_DETAILED_CSV = "csvs/agent_avg_delay_traveltime_last10.csv"
SAVE_RUN_SUMMARY_CSV = "csvs/per_run_avg_summary_last10.csv"

# ============================================================
# -------------------- CORE FUNCTIONS ------------------------
# ============================================================

def build_ff_time_map(free_flow_csv: Path) -> dict:
    """Read paths.csv and return mapping (origin, destination, route_index) -> ff_time."""
    if not free_flow_csv.exists():
        raise FileNotFoundError(f"Expected free-flow file not found: {free_flow_csv}")

    df_ff = pd.read_csv(free_flow_csv).dropna(how="all")
    rename_map = {}
    if "origins" in df_ff.columns:
        rename_map["origins"] = "origin"
    if "destinations" in df_ff.columns:
        rename_map["destinations"] = "destination"
    if "free_flow_time" in df_ff.columns:
        rename_map["free_flow_time"] = "ff_time"
    df_ff = df_ff.rename(columns=rename_map)

    needed = ["origin", "destination", "ff_time"]
    missing = [c for c in needed if c not in df_ff.columns]
    if missing:
        raise ValueError(f"paths.csv missing required columns: {missing}")

    df_ff = df_ff.dropna(subset=["origin", "destination", "ff_time"])
    df_ff["origin"] = pd.to_numeric(df_ff["origin"], errors="coerce").astype("Int64")
    df_ff["destination"] = pd.to_numeric(df_ff["destination"], errors="coerce").astype("Int64")
    df_ff["route_index"] = df_ff.groupby(["origin", "destination"]).cumcount()
    return df_ff.set_index(["origin", "destination", "route_index"])["ff_time"].to_dict()


def list_last_n_episode_files(episodes_dir: Path, last_n: int) -> list[tuple[int, Path]]:
    """Return a list of (episode_number, csv_path) for the last_n episodes found."""
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
    if last_n and last_n > 0:
        episode_files = episode_files[-last_n:]
    return episode_files


def load_concat_episodes(episode_files: list[tuple[int, Path]]) -> pd.DataFrame:
    """Read and concatenate episode CSVs, adding 'episode' column."""
    dfs = []
    for ep_num, path in episode_files:
        df = pd.read_csv(path, on_bad_lines="skip")
        df["episode"] = ep_num
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True).dropna(how="all")

    needed = ["travel_time", "id", "action", "origin", "destination", "episode"]
    missing = [c for c in needed if c not in df_all.columns]
    if missing:
        raise ValueError(f"Episode CSVs missing required columns: {missing}")

    for c in ["id", "action", "origin", "destination", "episode"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce").astype("Int64")
    df_all["travel_time"] = pd.to_numeric(df_all["travel_time"], errors="coerce")

    df_all = df_all.dropna(subset=["id", "action", "origin", "destination", "travel_time", "episode"])
    df_all[["id", "action", "origin", "destination", "episode"]] = df_all[
        ["id", "action", "origin", "destination", "episode"]
    ].astype(int)
    return df_all


def compute_agent_avg_delay_traveltime(base_dir: Path, last_n: int = 10) -> pd.DataFrame:
    """
    For the last N episodes:
      - compute per-agent average delay and average travel time.
    """
    base_dir = Path(base_dir)
    ff_time_map = build_ff_time_map(base_dir / "paths.csv")
    episode_files = list_last_n_episode_files(base_dir / "episodes", last_n)
    episodes_considered = [ep for ep, _ in episode_files]
    df_all = load_concat_episodes(episode_files)

    has_income = "income" in df_all.columns
    if has_income:
        df_all["income"] = pd.to_numeric(df_all["income"], errors="coerce")

    # attach ff_time and compute delay
    df_all["ff_time"] = df_all.apply(
        lambda r: ff_time_map.get((r["origin"], r["destination"], r["action"])), axis=1
    )
    unmatched_rows = int(df_all["ff_time"].isna().sum())
    df_all = df_all.dropna(subset=["ff_time"])
    df_all["delay"] = df_all["travel_time"] - df_all["ff_time"]

    # per-agent averages (+ average income if present)
    agg_ops = {
        "delay": "mean",
        "travel_time": "mean"
    }
    if has_income:
        agg_ops["income"] = "mean"

    agg = df_all.groupby("id", as_index=False).agg(agg_ops).rename(columns={
        "id": "agent_id",
        "delay": "avg_delay",
        "travel_time": "avg_travel_time"
    })

    agg["episodes_considered"] = [tuple(episodes_considered)] * len(agg)
    agg["unmatched_rows"] = unmatched_rows
    return agg


# ============================================================
# ----------------------- MAIN SCRIPT ------------------------
# ============================================================

if __name__ == "__main__":

    all_results = []          # per-agent rows for each run
    per_run_summary = []      # per-run averages over agents

    for base_dir, label in zip(BASE_DIRS, LABELS):
        print(f"\nProcessing {label} ...")
        df_agents = compute_agent_avg_delay_traveltime(base_dir, last_n=LAST_N_EPISODES)
        df_agents.insert(0, "run", label)
        all_results.append(df_agents)

        # Run-level means across agents (i.e., average delay of all agents in that run)
        run_avg_delay_all_agents = df_agents["avg_delay"].mean()
        run_avg_tt_all_agents = df_agents["avg_travel_time"].mean()

        # (Optional) run-level income mean if present
        run_avg_income = float(df_agents["income"].mean()) if "income" in df_agents.columns else np.nan

        per_run_summary.append({
            "run": label,
            "avg_delay_all_agents": run_avg_delay_all_agents,
            "avg_travel_time_all_agents": run_avg_tt_all_agents,
            "avg_income_all_agents": run_avg_income,
        })

    # Combine per-agent detailed results across runs
    results = pd.concat(all_results, ignore_index=True)

    # Per-run summary dataframe
    per_run_summary_df = pd.DataFrame(per_run_summary)

    print("\n=== PER-RUN AVERAGES (mean over agents) ===")
    cols = [
        "run",
        "avg_delay_all_agents",
        "avg_travel_time_all_agents",
        "avg_income_all_agents"
    ]
    # Only include income column if it exists in at least one run
    cols = [c for c in cols if c in per_run_summary_df.columns]
    print(per_run_summary_df[cols].to_string(index=False, float_format="%.6g"))

    # --------------------------------------------------------
    # Across-replications statistics (pooling all agents)
    # --------------------------------------------------------
    # "avg and std of all agents among replications":
    # pool all per-agent avg_delays from every run, compute mean & std
    pooled_agent_avg_delay_mean = results["avg_delay"].mean()
    pooled_agent_avg_delay_std = results["avg_delay"].std(ddof=1)

    # Also compute pooled mean travel time (per-agent averages pooled)
    pooled_agent_avg_tt_mean = results["avg_travel_time"].mean()

    # Final ratio: (avg delay of all agents among all replications) / (avg travel time of all agents among all replications)
    ratio_pooled = pooled_agent_avg_delay_mean / pooled_agent_avg_tt_mean if pooled_agent_avg_tt_mean > 0 else np.nan

    print("\n=== ACROSS-REPLICATION (POOLED OVER ALL AGENTS) ===")
    print(f"Mean of per-agent average delays:          {pooled_agent_avg_delay_mean:.6g}")
    print(f"Std  of per-agent average delays:          {pooled_agent_avg_delay_std:.6g}")
    print(f"Mean of per-agent average travel times:    {pooled_agent_avg_tt_mean:.6g}")
    print(f"\nRatio (mean(avg_delay) / mean(avg_travel_time)) = {ratio_pooled:.6g}")

    # Save outputs
    if SAVE_DETAILED_CSV:
        results_out = results.copy()
        # Serialize episodes_considered tuple
        results_out["episodes_considered"] = results_out["episodes_considered"].apply(
            lambda t: ",".join(map(str, t)) if isinstance(t, (list, tuple)) else str(t)
        )
        Path(SAVE_DETAILED_CSV).parent.mkdir(parents=True, exist_ok=True)
        results_out.to_csv(SAVE_DETAILED_CSV, index=False)
        print(f"\nSaved detailed per-agent results to: {SAVE_DETAILED_CSV}")

    if SAVE_RUN_SUMMARY_CSV:
        Path(SAVE_RUN_SUMMARY_CSV).parent.mkdir(parents=True, exist_ok=True)
        per_run_summary_df.to_csv(SAVE_RUN_SUMMARY_CSV, index=False)
        print(f"Saved per-run summary to: {SAVE_RUN_SUMMARY_CSV}")
