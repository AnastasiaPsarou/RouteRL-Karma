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
MAX_EPISODE = 1010           # None = no cap
XLIM = (0, 1010)           # None = auto
SAVE_PNG = "imgs/mean_travel_time_across_replications.png"  # None to skip saving
TITLE = "Mean travel time per episode (mean ± std across replications)"
XLABEL = "Episodes (days)"
YLABEL = "Mean travel time (min)"
FIGSIZE = (7.5, 3.5)
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

    # 2) Aggregate across folders (replications)
    agg = align_and_aggregate_across_folders(series)
    if agg.empty:
        raise ValueError("No usable data across replications. Check paths, kind filter, or MAX_EPISODE.")

    x = agg["episode"].to_numpy()
    y = agg["mean_across_rep"].to_numpy(dtype=float)
    s = agg["std_across_rep"].to_numpy(dtype=float)
    n = agg["n_rep"].to_numpy(dtype=int)

    med = agg["median_across_rep"].to_numpy(dtype=float)

    # Smooth median too (consistent with mean/std)
    med_plot = smooth_ma(med, SMOOTH)

    # Optional smoothing (apply to mean and std consistently)
    y_plot = smooth_ma(y, SMOOTH)
    s_plot = smooth_ma(s, SMOOTH)

    # 3) Plot single series: mean across replications with std band across replications
    plt.figure(figsize=FIGSIZE)
    plt.plot(x, y_plot, color=COLOR, lw=1.8, label=f"Mean across {np.nanmax(n):.0f} replications")
    
    # Shade only where we have at least 2 replications
    mask = n >= 2
    plt.fill_between(x[mask], (y_plot - s_plot)[mask], (y_plot + s_plot)[mask],
                     color=COLOR, alpha=BAND_ALPHA, linewidth=0, label="±1 std across replications")

    plt.plot(
        x, med_plot,
        linestyle=MEDIAN_LS, linewidth=MEDIAN_LW, color=MEDIAN_COLOR,
        label=f"Median across {np.nanmax(n):.0f} replications"
    )

    if XLIM is not None:
        plt.xlim(*XLIM)
    plt.xlabel(XLABEL, fontsize=24)
    plt.ylabel(YLABEL, fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(which='both')
    plt.title(TITLE + f" | kind={PLOT_KIND}", fontsize=14)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()

    if SAVE_PNG:
        os.makedirs(os.path.dirname(SAVE_PNG), exist_ok=True)
        plt.savefig(SAVE_PNG, dpi=300, bbox_inches="tight")
        print(f"[INFO] saved plot to: {SAVE_PNG}")

    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
