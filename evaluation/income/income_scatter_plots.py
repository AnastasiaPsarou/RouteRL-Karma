"""A scatter plot of the income values of each agent for the different runs of the exp."""

#!/usr/bin/env python3
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Times New Roman']

# ==========================================================
# ---- USER PARAMETERS ----
# ==========================================================
folders = [
    r"../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_4_long/episodes",
    r"../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_5_long/episodes",
    r"../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_6_long/episodes",
    r"../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_7_long/episodes",
    r"../data/training_records_300_agents_fee_0_1/training_records_monetary_pricing_300_agents_fee_0_1_seed_8_long/episodes",
]
metric = "income"          # column to plot
SAVE_PNG = "imgs/income_scatter_last_episode.png"
FIGSIZE = (10, 5)
# ==========================================================


def get_last_csv(folder):
    """Return the path to the last episode CSV in a folder."""
    episode_files = []
    for fn in os.listdir(folder):
        if fn.startswith("ep") and fn.endswith(".csv"):
            m = re.search(r"ep(\d+)\.csv$", fn)
            if m:
                episode_files.append((int(m.group(1)), fn))
    if not episode_files:
        return None
    episode_files.sort(key=lambda t: t[0])
    return os.path.join(folder, episode_files[-1][1])  # last one only


def load_last_income(folder, metric="income"):
    """Load the income values from the last episode CSV in a folder."""
    last_csv = get_last_csv(folder)
    if last_csv is None:
        print(f"[WARN] No episode CSVs found in {folder}")
        return np.array([])
    try:
        df = pd.read_csv(last_csv, on_bad_lines="skip")
    except Exception as e:
        print(f"[WARN] Failed to read {last_csv}: {e}")
        return np.array([])
    if metric not in df.columns:
        print(f"[WARN] Missing '{metric}' in {last_csv}")
        return np.array([])
    return pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()


# ==========================================================
# ---- MAIN PLOTTING ----
# ==========================================================
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)
colors = plt.cm.tab10.colors

all_incomes = []  # we'll store all for computing y-axis limits

for i, folder in enumerate(folders):
    if not os.path.isdir(folder):
        print(f"[WARN] Not a directory: {folder}")
        continue

    incomes = load_last_income(folder, metric=metric)
    if len(incomes) == 0:
        continue

    all_incomes.extend(incomes)

    # Scatter per run with small horizontal jitter to prevent overlap
    x = np.full_like(incomes, i + 1, dtype=float)
    x += np.random.uniform(-0.1, 0.1, size=len(incomes))
    ax.scatter(x, incomes, s=8, alpha=0.7,
               color=colors[i % len(colors)],
               label=os.path.basename(os.path.dirname(folder)))

# --- Adjust y-axis range for clarity ---
if all_incomes:
    ymin, ymax = np.min(all_incomes), np.max(all_incomes)
    yrange = ymax - ymin
    ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.05 * yrange)  # small padding
    print(f"[INFO] y-axis set from {ymin - 0.05 * yrange:.2f} to {ymax + 0.05 * yrange:.2f}")

ax.set_title(f"Distribution of '{metric}' across replications",
             fontsize=14)
ax.set_xlabel("Replication (folder index)", fontsize=14)
ax.set_ylabel(metric.capitalize(), fontsize=14)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xticks(range(1, len(folders) + 1))
ax.set_xticklabels([f"Run {i+1}" for i in range(len(folders))], fontsize=12)
ax.tick_params(axis="both", labelsize=12)
#ax.legend(fontsize=10, loc="best")

plt.tight_layout()

if SAVE_PNG:
    os.makedirs(os.path.dirname(SAVE_PNG), exist_ok=True)
    plt.savefig(SAVE_PNG, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved scatter plot to: {SAVE_PNG}")

#plt.show()
