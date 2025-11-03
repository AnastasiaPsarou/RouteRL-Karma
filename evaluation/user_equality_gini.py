import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd

# ==========================================================
# ---- USER PARAMETERS ----
# ==========================================================
episodes_dir = Path("../scenarios/monetary_pricing/training_records_monetary_pricing_10_agents/episodes")  # Folder containing ep{number}.csv files
last_n = 10                                           # Number of last episodes to include
metric = "income"                                     # or "travel_time"
print_table = True                                    # Print per-agent averages table?
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
    raise ValueError(f"Missing required columns in episode CSVs: {missing}")

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

# --- Compute Gini
values = per_agent["value"].to_numpy()
g = gini(values)

print(f"\nGini coefficient over per-agent average {metric.replace('_', ' ')} "
      f"(last {last_n} episodes): {g:.6f}")

if print_table:
    per_agent = per_agent.rename(columns={"value": f"avg_{metric}"})
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("\nPer-agent averages used for Gini:")
    print(per_agent.to_string(index=False))
