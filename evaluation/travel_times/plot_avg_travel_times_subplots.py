#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']

# ==========================================================
# ---- USER PARAMETERS ----
# ==========================================================
data_paths = [
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_5/episodes",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_6/episodes",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_8/episodes",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_9/episodes",
    r"../../data/training_records_300_agents_fee_0_1_long/training_records_monetary_pricing_300_agents_fee_0_1_seed_10/episodes",

]
PLOT_STAT = "travel_time"   # column in CSV
KIND_COL  = "kind"          # filter column (if present)
PLOT_KIND = "AV"            # "AV", "Human", or "All"
SMOOTH    = 20               # moving average window; 0/1 disables
MAX_EPISODE = 110           # None = no cap
XLIM = (0, 110)           # None = auto
SAVE_PNG = "imgs/mean_travel_time_across_replications.png"  # None to skip saving

# Dynamically set title and labels based on the chosen metric
if PLOT_STAT.lower() == "reward":
    TITLE = "Mean reward per episode"
    YLABEL = "Mean reward"
elif PLOT_STAT.lower() == "travel_time":
    TITLE = "Mean travel time per episode"
    YLABEL = "Mean travel time (min)"
else:
    TITLE = f"Mean {PLOT_STAT} per episode"
    YLABEL = f"Mean {PLOT_STAT}"

XLABEL = "Episodes (days)"

FIGSIZE = (3.5, 3.5)
COLOR = "navy"
BAND_ALPHA = 0.25

MEDIAN_COLOR = "darkorange"   # color for median line
MEDIAN_LW    = 1.8            # linewidth for median
MEDIAN_LS    = "--"           # linestyle for median
# ==========================================================

def smooth_ma(y, w):
    y = np.asarray(y, dtype=float)
    if w is None or w <= 1 or len(y) < w: 
        return y
    half = w // 2
    pad = np.pad(y, (half, half), mode='edge')
    out = np.convolve(pad, np.ones(w)/w, mode='valid')
    return out[:len(y)]

def get_episode_files(folder):
    if not os.path.isdir(folder):
        print(f"[WARN] not a folder: {folder}")
        return []
    files = []
    for fn in os.listdir(folder):
        if fn.startswith("ep") and fn.endswith(".csv"):
            try:
                ep = int(fn[2:-4])
                files.append((ep, fn))
            except ValueError:
                pass
    files.sort(key=lambda t: t[0])
    if MAX_EPISODE is not None:
        files = [(ep, fn) for ep, fn in files if ep <= MAX_EPISODE]
    return files

def per_episode_mean_for_folder(folder, metric=PLOT_STAT, kind=PLOT_KIND, kind_col=KIND_COL):
    """
    Returns DataFrame: episode, mean   (mean across agents/rows in that episode)
    No std here (we’ll compute std ACROSS folders later).
    """
    rows = []
    for ep, fn in get_episode_files(folder):
        path = os.path.join(folder, fn)
        if not os.path.getsize(path):
            continue
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
        except Exception as e:
            print(f"[WARN] read failed {path}: {e}")
            continue

        if metric not in df.columns:
            continue

        if kind != "All" and kind_col in df.columns:
            df = df[df[kind_col] == kind]

        vals = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(vals) == 0:
            continue

        rows.append((ep, float(vals.mean())))

    if not rows:
        return pd.DataFrame(columns=["episode", "mean"])
    out = pd.DataFrame(rows, columns=["episode", "mean"]).sort_values("episode").reset_index(drop=True)
    return out

def align_and_aggregate_across_folders(series_list):
    """
    Input: list of (label, df) where df has columns ['episode','mean'] for each folder.
    Output: DataFrame with columns: episode, mean_across_rep, std_across_rep, n_rep
    """
    # Union of all episode indices
    all_eps = sorted(set().union(*[set(df["episode"]) for _, df in series_list if not df.empty]))
    if not all_eps:
        return pd.DataFrame(columns=["episode","mean_across_rep","std_across_rep","n_rep"])

    Y = np.full((len(series_list), len(all_eps)), np.nan, dtype=float)
    for r, (_, df) in enumerate(series_list):
        if df.empty:
            continue
        m = dict(zip(df["episode"].to_numpy(), df["mean"].to_numpy()))
        for c, ep in enumerate(all_eps):
            if ep in m:
                Y[r, c] = m[ep]

    n_rep = np.sum(~np.isnan(Y), axis=0)
    mean_across   = np.nanmean(Y, axis=0)
    median_across = np.nanmedian(Y, axis=0)  # <-- NEW

    std_across = np.array([
        np.nanstd(Y[:, i], ddof=1) if n_rep[i] >= 2 else (0.0 if n_rep[i] == 1 else np.nan)
        for i in range(len(all_eps))
    ], dtype=float)

    out = pd.DataFrame({
        "episode": all_eps,
        "mean_across_rep": mean_across,
        "median_across_rep": median_across,  # <-- NEW
        "std_across_rep": std_across,
        "n_rep": n_rep
    })
    return out

def main():
    # 1) Build per-folder per-episode means (each folder = one replication)
    series = []
    for folder in data_paths:
        print(f"[INFO] loading {folder}")
        df = per_episode_mean_for_folder(folder)
        if df.empty:
            print(f"[WARN] no usable episodes in {folder} for kind='{PLOT_KIND}'")
        series.append((os.path.basename(folder) or folder, df))

    # Filter out empty datasets
    series = [(label, df) for label, df in series if not df.empty]
    n_rep = len(series)
    if n_rep == 0:
        raise ValueError("No usable data across replications. Check paths or filters.")

    # 2) Set up subplots: one per replication (vertical stack)
    fig, axes = plt.subplots(
        nrows=1, ncols=n_rep, figsize=(FIGSIZE[0] * n_rep, FIGSIZE[1]),
        sharey=True
    )


    if n_rep == 1:
        axes = [axes]  # make iterable if only one

    # 3) Plot each replication separately
    for ax, (label, df) in zip(axes, series):
        x = df["episode"].to_numpy()
        y = df["mean"].to_numpy(dtype=float)
        y_plot = smooth_ma(y, SMOOTH)

        ax.plot(x, y_plot, color=COLOR, lw=1.8)
        ax.set_title(f"{label}", fontsize=12)
        ax.grid(True, which="both")

        if XLIM is not None:
            ax.set_xlim(*XLIM)

        ax.set_ylabel(YLABEL, fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

    # Label the bottom plot’s x-axis
    axes[-1].set_xlabel(XLABEL, fontsize=12)

    # 4) Add overall figure title
    fig.suptitle(f"{TITLE} | kind={PLOT_KIND}", fontsize=14, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if SAVE_PNG:
        os.makedirs(os.path.dirname(SAVE_PNG), exist_ok=True)
        plt.savefig(SAVE_PNG, dpi=300, bbox_inches="tight")
        print(f"[INFO] saved multi-panel plot to: {SAVE_PNG}")

    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
