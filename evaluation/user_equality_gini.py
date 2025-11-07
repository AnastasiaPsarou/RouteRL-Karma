#!/usr/bin/env python3
import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd

# ==========================================================
# ---- USER PARAMETERS (edit these) ----
# ==========================================================
folders = [
    "../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_4_long/episodes",
    "../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_5_long/episodes",
    "../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_6_long/episodes",
    "../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_7_long/episodes",
    "../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_8_long/episodes",
    # "../data/training_records/training_records_monetary_pricing_150_agents/episodes",
    # add more folders if needed
]

"""folders = [
    "../data/training_records_10_agents/training_records_monetary_pricing_10_agents_long_1/episodes",
    "../data/training_records_10_agents/training_records_monetary_pricing_10_agents_long_2/episodes",
    "../data/training_records_10_agents/training_records_monetary_pricing_10_agents_long_3/episodes",
    "../data/training_records_10_agents/training_records_monetary_pricing_10_agents_long_4/episodes",
    "../data/training_records_10_agents/training_records_monetary_pricing_10_agents_long_5/episodes",
    # "../data/training_records/training_records_monetary_pricing_150_agents/episodes",
    # add more folders if needed
]"""
last_n = 10                  # Number of last episodes to include per folder
metric = "travel_time"            # "income" or "travel_time"
print_table = False          # Print per-agent averages table?
# ==========================================================


def gini(x: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a 1D array.
    G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mean(x))
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]  # remove NaNs and infs
    n = x.size
    if n < 2 or x.mean() == 0:
        return float("nan")
    diff_sum = np.abs(x[:, None] - x[None, :]).sum()
    return diff_sum / (2 * (n ** 2) * x.mean())


def compute_gini_for_dir(episodes_dir: Path, last_n: int, metric: str, print_table: bool):
    # --- Load and combine the last N episodes
    episode_files = []
    for p in glob.glob(str(episodes_dir / "ep*.csv")):
        m = re.search(r"ep(\d+)\.csv$", Path(p).name)
        if m:
            episode_files.append((int(m.group(1)), Path(p)))

    if not episode_files:
        raise FileNotFoundError(f"No episode CSVs found in {episodes_dir}")

    episode_files.sort(key=lambda t: t[0])
    selected = episode_files[-last_n:]

    dfs = []
    for ep, path in selected:
        df = pd.read_csv(path, on_bad_lines="skip")
        df["episode"] = ep
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True).dropna(how="all")

    # --- Check required columns
    needed = {"id", metric}
    missing = [c for c in needed if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing required columns in episode CSVs: {missing} (dir: {episodes_dir})")

    # --- Clean and aggregate
    df_all["id"] = pd.to_numeric(df_all["id"], errors="coerce").astype("Int64")
    df_all[metric] = pd.to_numeric(df_all[metric], errors="coerce")
    df_all = df_all.dropna(subset=["id", metric])
    df_all["id"] = df_all["id"].astype(int)

    per_agent = (
        df_all.groupby("id", as_index=False)
        .agg(
            episodes_seen=("episode", "nunique"),
            records_used=("id", "size"),
            value=(metric, "mean"),
        )
        .sort_values("id")
    )

    values = per_agent["value"].to_numpy()
    g = gini(values)

    if print_table:
        per_agent = per_agent.rename(columns={"value": f"avg_{metric}"})
        pd.set_option("display.float_format", lambda x: f"{x:.6f}")
        print("\nPer-agent averages used for Gini:")
        print(per_agent.to_string(index=False))

    return g


def main():
    expanded_folders = []
    for f in folders:
        hits = glob.glob(f)
        expanded_folders.extend(hits if hits else [f])

    results = []
    for f in expanded_folders:
        episodes_dir = Path(f)
        print("\n" + "="*72)
        print(f"Folder: {episodes_dir}")
        print("="*72)
        try:
            g = compute_gini_for_dir(episodes_dir, last_n=last_n, metric=metric, print_table=print_table)
            print(f"Gini coefficient over per-agent average {metric} (last {last_n} episodes): {g:.6f}")
            results.append(g)
        except Exception as e:
            print(f"⚠️  Skipped {episodes_dir}: {e}")
            results.append(np.nan)

    # --- Aggregate across folders
    ginis = np.array(results, dtype=float)
    mean = np.nanmean(ginis) if np.any(np.isfinite(ginis)) else np.nan
    finite = np.isfinite(ginis)
    if finite.sum() >= 2:
        std = np.nanstd(ginis, ddof=1)
    else:
        std = np.nanstd(ginis, ddof=0)

    print("\n" + "-"*72)
    print("Summary across folders")
    print("-"*72)
    for f, g in zip(expanded_folders, ginis):
        gstr = f"{g:.6f}" if np.isfinite(g) else "NaN"
        print(f"{f}: {gstr}")
    print("-"*72)
    print(f"Mean Gini: {mean:.6f}" if np.isfinite(mean) else "Mean Gini: NaN")
    print(f"Std  Gini: {std:.6f}"  if np.isfinite(std)  else "Std  Gini: NaN")


if __name__ == "__main__":
    main()
