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
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_8",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_9",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_10",

]
LABELS = ["Run 1", "Run 2", "Run 3", "Run 4", "Run 5"]

LAST_N_EPISODES = 10  # how many of the latest episodes to analyze

SAVE_DETAILED_CSV = "csvs/agent_max_delay_traveltime_last10.csv"
SAVE_RUN_SUMMARY_CSV = "csvs/per_run_summary_last10.csv"

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


def compute_agent_max_delay_traveltime(base_dir: Path, last_n: int = 10) -> pd.DataFrame:
    """Compute per-agent max delay, max travel time, and ratio for last N episodes."""
    base_dir = Path(base_dir)
    ff_time_map = build_ff_time_map(base_dir / "paths.csv")
    episode_files = list_last_n_episode_files(base_dir / "episodes", last_n)
    episodes_considered = [ep for ep, _ in episode_files]
    df_all = load_concat_episodes(episode_files)

    # Optional: income column if exists
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

    # per-agent maxima (+ average income if present)
    agg_ops = {
        "delay": "max",
        "travel_time": "max"
    }
    if has_income:
        agg_ops["income"] = "mean"  # average income across episodes

    agg = df_all.groupby("id", as_index=False).agg(agg_ops).rename(columns={
        "id": "agent_id",
        "delay": "max_delay",
        "travel_time": "max_travel_time"
    })

    agg["ratio"] = np.where(agg["max_travel_time"] > 0,
                            agg["max_delay"] / agg["max_travel_time"],
                            np.nan)
    agg["episodes_considered"] = [tuple(episodes_considered)] * len(agg)
    agg["unmatched_rows"] = unmatched_rows
    return agg


# ============================================================
# ----------------------- MAIN SCRIPT ------------------------
# ============================================================

if __name__ == "__main__":

    all_results = []
    per_run_summary = []

    for base_dir, label in zip(BASE_DIRS, LABELS):
        print(f"\nProcessing {label} ...")
        df_agents = compute_agent_max_delay_traveltime(base_dir, last_n=LAST_N_EPISODES)
        df_agents.insert(0, "run", label)
        all_results.append(df_agents)

        # Identify run-level maxima and the agents (and incomes)
        idx_delay = df_agents["max_delay"].idxmax()
        idx_tt = df_agents["max_travel_time"].idxmax()

        run_max_delay = df_agents.loc[idx_delay, "max_delay"]
        run_max_delay_agent = int(df_agents.loc[idx_delay, "agent_id"])
        run_max_delay_income = float(df_agents.loc[idx_delay, "income"]) if "income" in df_agents.columns else np.nan

        run_max_tt = df_agents.loc[idx_tt, "max_travel_time"]
        run_max_tt_agent = int(df_agents.loc[idx_tt, "agent_id"])
        run_max_tt_income = float(df_agents.loc[idx_tt, "income"]) if "income" in df_agents.columns else np.nan

        per_run_summary.append({
            "run": label,
            "max_delay": run_max_delay,
            "max_delay_agent": run_max_delay_agent,
            "max_delay_agent_income": run_max_delay_income,
            "max_travel_time": run_max_tt,
            "max_travel_time_agent": run_max_tt_agent,
            "max_travel_time_agent_income": run_max_tt_income,
        })

    # Combine per-agent detailed results
    results = pd.concat(all_results, ignore_index=True)

    # Per-run summary dataframe
    per_run_summary_df = pd.DataFrame(per_run_summary)

    print("\n=== PER-RUN MAX VALUES (with agents & income) ===")
    cols = [
        "run",
        "max_delay", "max_delay_agent", "max_delay_agent_income",
        "max_travel_time", "max_travel_time_agent", "max_travel_time_agent_income"
    ]
    print(per_run_summary_df[cols].to_string(index=False, float_format="%.6g"))

    # Mean & std across runs (for the numeric values)
    mean_delay = per_run_summary_df["max_delay"].mean()
    std_delay = per_run_summary_df["max_delay"].std(ddof=1)
    mean_tt = per_run_summary_df["max_travel_time"].mean()
    std_tt = per_run_summary_df["max_travel_time"].std(ddof=1)

    # Ratio of means
    ratio_mean_delay_traveltime = mean_delay / mean_tt if mean_tt > 0 else np.nan

    print("\n=== ACROSS-RUN STATISTICS ===")
    print(f"Mean of run-level max delays:         {mean_delay:.6g}")
    print(f"Std  of run-level max delays:         {std_delay:.6g}")
    print(f"Mean of run-level max travel times:   {mean_tt:.6g}")
    print(f"Std  of run-level max travel times:   {std_tt:.6g}")
    print(f"\nRatio (mean(max_delay) / mean(max_travel_time)) = {ratio_mean_delay_traveltime:.6g}")

    # Save outputs
    if SAVE_DETAILED_CSV:
        results_out = results.copy()
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
