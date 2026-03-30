# =======================
# INCOME-BIN vs AVG TRAVEL TIME (minutes)
# =======================
# Plot:
#   x-axis: income bins (built like your bar-code logic)
#   y-axis: average travel time in minutes
# Average is computed over the selected episode range (or last_n episodes).
#
# Notes:
# - Uses income from episode CSVs (no agents_income.csv).
# - If your travel_time is already in minutes, keep travel_time_unit="minutes".
#   If it's in seconds, set travel_time_unit="seconds" (it will divide by 60).

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_EP_RE = re.compile(r"^ep(\d+)\.csv$")

def smooth_1d(y: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Simple moving-average smoothing with edge padding.
    window must be odd (3,5,7,...) and <= len(y).
    NaNs are handled by smoothing only valid values via weights.
    """
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < 2:
        return y
    if window % 2 == 0:
        raise ValueError("Smoothing window must be odd (e.g., 3, 5).")
    if window > len(y):
        window = len(y) if len(y) % 2 == 1 else len(y) - 1
        window = max(window, 1)

    k = window // 2
    y_pad = np.pad(y, (k, k), mode="edge")

    valid = ~np.isnan(y_pad)
    y_filled = np.where(valid, y_pad, 0.0)
    w = valid.astype(float)

    kernel = np.ones(window, dtype=float)

    num = np.convolve(y_filled, kernel, mode="valid")
    den = np.convolve(w, kernel, mode="valid")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out

def _episodes_dir(p: str) -> str:
    ep = os.path.join(p, "episodes")
    return ep if os.path.isdir(ep) else p

def list_episode_files(folder: str):
    folder = _episodes_dir(folder)
    if not os.path.isdir(folder):
        return []
    out = []
    for fn in os.listdir(folder):
        m = _EP_RE.match(fn)
        if m:
            out.append((int(m.group(1)), os.path.join(folder, fn)))
    out.sort(key=lambda t: t[0])
    return out

def load_episodes(folder: str, n_last: int = 30, episode_range=None, episodes=None):
    folder = _episodes_dir(folder)
    eps = list_episode_files(folder)
    
    if not eps:
        return None
    eps_map = {ep: path for ep, path in eps}

    if episodes is not None:
        wanted = sorted(set(int(e) for e in episodes))
        chosen = [(ep, eps_map[ep]) for ep in wanted if ep in eps_map]
    elif episode_range is not None:
        start, end = map(int, episode_range)
        chosen = [(ep, path) for ep, path in eps if start <= ep <= end]
    else:
        chosen = eps[-n_last:] if len(eps) >= n_last else eps

    if not chosen:
        return None

    dfs = []
    for ep, path in chosen:
        df = pd.read_csv(path, on_bad_lines="skip")
        df["episode"] = ep
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def derive_agent_income_from_episodes(
    run_dir,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
):
    df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
    if df is None or df.empty or id_col not in df.columns or income_col not in df.columns:
        return pd.DataFrame(columns=[id_col, "income"])

    tmp = df[[id_col, income_col]].copy()
    tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
    tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
    tmp = tmp.dropna(subset=[id_col, income_col])
    if tmp.empty:
        return pd.DataFrame(columns=[id_col, "income"])
    tmp[id_col] = tmp[id_col].astype(int)

    def mode_income(s):
        vc = s.value_counts(dropna=True)
        if vc.empty:
            return np.nan
        top = vc.iloc[0]
        cands = vc[vc == top].index.tolist()
        return float(min(cands))

    return (
        tmp.groupby(id_col)[income_col]
           .apply(mode_income)
           .reset_index(name="income")
           .dropna(subset=["income"])
    )



def build_vot_bins_like_your_code(
    run_dirs,
    high_vot_threshold=None,     # e.g. 40, or None for pure quantile bins
    n_equal_bins=4,
    episode_range=None,
    episodes=None,
    last_n=30,
    income_col="income",         # monthly income
    urgency_col="urgency",
    work_hours_per_month=160.0,
):
    """
    Build global VoT bins across all runs.

    VoT = (monthly_income / work_hours_per_month) * urgency

    If high_vot_threshold is given:
      - VoT <= threshold is split into quantile bins
      - one extra top bin is added: > high_vot_threshold

    If high_vot_threshold is None:
      - all VoT values are split into quantile bins only
    """
    pooled_vot = []

    for rd in run_dirs:
        df = load_episodes(rd, n_last=last_n, episode_range=episode_range, episodes=episodes)
        if df is None or df.empty:
            continue

        needed = [income_col, urgency_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] Could not derive VoT from {rd}. Missing {missing}. Skipping run.")
            continue

        tmp = df[[income_col, urgency_col]].copy()
        tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
        tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
        tmp = tmp.dropna(subset=[income_col, urgency_col])
        if tmp.empty:
            continue

        tmp["vot"] = (tmp[income_col] / float(work_hours_per_month)) * tmp[urgency_col]
        vot = pd.to_numeric(tmp["vot"], errors="coerce").dropna().to_numpy()
        if vot.size:
            pooled_vot.append(vot)

    if not pooled_vot:
        raise ValueError("No valid VoT values found across runs.")

    pooled_vot = np.concatenate(pooled_vot)

    def fmt(x):
        return f"{x:.2f}" if abs(x) < 100 else f"{x:.0f}"

    if high_vot_threshold is None:
        qs = np.linspace(0, 1, n_equal_bins + 1)
        edges = np.unique(np.quantile(pooled_vot, qs))
        if len(edges) < 2:
            raise ValueError("Quantile edges collapsed for VoT bins.")
        labels = [f"{fmt(edges[i])}–{fmt(edges[i+1])}" for i in range(len(edges) - 1)]
        return edges, labels, False

    low = pooled_vot[pooled_vot <= high_vot_threshold]
    if low.size == 0:
        raise ValueError("No VoT values <= high_vot_threshold found. Increase threshold or remove it.")

    qs = np.linspace(0, 1, n_equal_bins + 1)
    edges = np.unique(np.quantile(low, qs))
    if len(edges) < 2:
        raise ValueError("Quantile edges collapsed for low-VoT bins.")

    low_labels = [f"{fmt(edges[i])}–{fmt(edges[i+1])}" for i in range(len(edges) - 1)]
    high_label = f">{fmt(high_vot_threshold)}"
    bins = np.concatenate([edges, [np.inf]])
    labels = low_labels + [high_label]
    return bins, labels, True


def assign_equal_sized_daily_vot_bins(
    df: pd.DataFrame,
    id_col="id",
    episode_col="episode",
    income_col="income",
    urgency_col="urgency",
    work_hours_per_month=240.0,
    n_bins=5,
):
    """
    Assign VoT bins separately within each episode, so each day has
    approximately equal numbers of individuals in each bin.

    An agent may fall into different bins on different days.
    """
    labels = [f"Q{i}" for i in range(1, n_bins + 1)]

    tmp = df[[episode_col, id_col, income_col, urgency_col]].copy()
    tmp[episode_col] = pd.to_numeric(tmp[episode_col], errors="coerce")
    tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
    tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
    tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
    tmp = tmp.dropna(subset=[episode_col, id_col, income_col, urgency_col])

    if tmp.empty:
        return pd.Series(pd.NA, index=df.index, dtype="object")

    tmp[episode_col] = tmp[episode_col].astype(int)
    tmp[id_col] = tmp[id_col].astype(int)

    # daily VoT
    tmp["vot"] = (tmp[income_col] / float(work_hours_per_month)) * tmp[urgency_col]

    # if there are accidental duplicate rows per (episode, id), collapse first
    per_agent_day = (
        tmp.groupby([episode_col, id_col], observed=True)["vot"]
           .mean()
           .reset_index()
    )

    def _bin_one_episode(s: pd.Series) -> pd.Series:
        if s.notna().sum() == 0:
            return pd.Series(pd.NA, index=s.index, dtype="object")

        # if too few distinct values, qcut may fail; use rank fallback
        try:
            q = pd.qcut(s, q=n_bins, labels=labels, duplicates="drop")
            return q.astype(object)
        except Exception:
            ranks = s.rank(method="first")
            q = pd.qcut(ranks, q=n_bins, labels=labels, duplicates="drop")
            return q.astype(object)

    per_agent_day["vot_bin"] = (
        per_agent_day.groupby(episode_col, observed=True)["vot"]
                     .transform(_bin_one_episode)
    )

    # map back to original rows
    key_to_bin = per_agent_day.set_index([episode_col, id_col])["vot_bin"]

    out_keys = pd.MultiIndex.from_arrays(
        [
            pd.to_numeric(df[episode_col], errors="coerce"),
            pd.to_numeric(df[id_col], errors="coerce"),
        ]
    )
    out = pd.Series(out_keys.map(key_to_bin), index=df.index, dtype="object")
    return out


def plot_avg_travel_time_by_vot_quantile_lines(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    kind_col="kind",
    plot_kind="AV",
    work_hours_per_month=160.0,
    n_vot_bins=5,
    travel_time_unit="minutes",
    avg_mode="row",
    show_std_band=True,
    figsize=(6, 5.2),
    save_path=None,
    show=True,
    smooth_lines=True,
    smooth_window=1,
    show_bin_n=True,
    bin_n_agg="mean",
):
    scenarios = list(algo_runs.keys())
    vot_labels = [f"Q{i}" for i in range(1, n_vot_bins + 1)]
    x = np.arange(len(vot_labels))

    def to_minutes(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if travel_time_unit.lower().startswith("sec"):
            return s / 60.0
        return s

    fig, ax = plt.subplots(figsize=figsize)
    pooled_n_across_scenarios = []

    for sc in scenarios:
        sc_cfg = algo_runs[sc]

        if isinstance(sc_cfg, dict):
            run_dirs = sc_cfg["runs"]
            sc_episode_range = sc_cfg.get("episode_range", episode_range)
            sc_episodes = sc_cfg.get("episodes", episodes)
            sc_last_n = sc_cfg.get("last_n", last_n)
        else:
            run_dirs = sc_cfg
            sc_episode_range = episode_range
            sc_episodes = episodes
            sc_last_n = last_n

        per_run_vecs = []
        per_run_n = []

        for run_dir in run_dirs:
            df = load_episodes(
                run_dir,
                n_last=sc_last_n,
                episode_range=sc_episode_range,
                episodes=sc_episodes,
            )
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = ["episode", id_col, income_col, urgency_col, travel_time_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[["episode", id_col, income_col, urgency_col, travel_time_col]].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
            tmp[travel_time_col] = to_minutes(tmp[travel_time_col])
            tmp = tmp.dropna(subset=[id_col, income_col, urgency_col, travel_time_col])

            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)

            tmp["vot_bin"] = assign_equal_sized_daily_vot_bins(
                tmp,
                id_col=id_col,
                episode_col="episode",
                income_col=income_col,
                urgency_col=urgency_col,
                work_hours_per_month=work_hours_per_month,
                n_bins=n_vot_bins,
            )
            tmp = tmp.dropna(subset=["vot_bin"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "vot_bin"], observed=True)[travel_time_col]
                    .mean()
                    .reset_index()
                )
                series = per_agent.groupby("vot_bin", observed=True)[travel_time_col].mean()
            else:
                series = tmp.groupby("vot_bin", observed=True)[travel_time_col].mean()

            vec = series.reindex(vot_labels).to_numpy(dtype=float)
            per_run_vecs.append(vec)

            n_series = (
                tmp.groupby(["episode", "vot_bin"], observed=True)[id_col]
                .nunique()
                .groupby("vot_bin", observed=True)
                .mean()
            )
            n_vec = n_series.reindex(vot_labels).to_numpy(dtype=float)
            per_run_n.append(n_vec)

        if not per_run_vecs:
            print(f"[WARN] Scenario '{sc}' had no usable runs. Skipping line.")
            continue

        arr = np.vstack(per_run_vecs)
        n_arr = np.vstack(per_run_n)
        pooled_n_across_scenarios.append(n_arr)

        mean_vec = np.nanmean(arr, axis=0)
        std_vec = np.nanstd(arr, axis=0, ddof=1 if arr.shape[0] > 1 else 0) if arr.shape[0] > 1 else None

        mean_plot = smooth_1d(mean_vec, window=smooth_window) if smooth_lines else mean_vec
        std_plot = smooth_1d(std_vec, window=smooth_window) if (smooth_lines and std_vec is not None) else std_vec

        ax.plot(x, mean_plot, marker="o", linewidth=2, label=f"{sc}")
        if show_std_band and std_plot is not None:
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(vot_labels, rotation=25, ha="right", fontsize=12)
    ax.set_xlabel("VoT quantile", fontsize=16)
    ax.set_ylabel("Avg travel time (minutes)", fontsize=16)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return fig

ALGO_RUNS = {

     "monetary pricing 0.5":{
        "runs":[
        #r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward/episodes",
        ],
        "episode_range": (200, 500),
     },
    "karma pricing":{
        "runs":[
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        ],
        "episode_range": (750, 900),
    }
        


}

"""       "monetary pricing fee 0.5 longer": [
       r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
    
    ],"""

plot_avg_travel_time_by_vot_quantile_lines(
    ALGO_RUNS,
    episode_range=(800, 1000),
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    travel_time_unit="minutes",
    work_hours_per_month=240.0,
    n_vot_bins=5,
    avg_mode="row",
    plot_kind="AV",
    smooth_window=1,
    show_std_band=True,
    save_path="imgs/avg_travel_time_by_vot_quantile_lines.png",
    show=True,
)