
import numpy as np

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

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


def build_income_bins_like_your_code(
    run_dirs,
    high_income_threshold=6000,
    n_equal_bins=4,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
):
    pooled_low = []

    for rd in run_dirs:
        inc_df = derive_agent_income_from_episodes(
            rd,
            episode_range=episode_range,
            episodes=episodes,
            last_n=last_n,
            id_col=id_col,
            income_col=income_col,
        )
        if inc_df.empty:
            print(f"[WARN] Could not derive income from episodes for {rd}. Skipping run.")
            continue

        incomes = pd.to_numeric(inc_df["income"], errors="coerce").dropna().to_numpy()
        low = incomes[incomes <= high_income_threshold]
        if low.size:
            pooled_low.append(low)

    if not pooled_low:
        raise ValueError(
            "No incomes <= high_income_threshold found across runs. "
            "Either increase high_income_threshold or check that episode CSVs contain a numeric 'income' column."
        )

    pooled_low = np.concatenate(pooled_low)
    qs = np.linspace(0, 1, n_equal_bins + 1)
    edges = np.unique(np.quantile(pooled_low, qs))

    if len(edges) < 2:
        raise ValueError("Quantile edges collapsed (all low incomes identical). Cannot create bins.")

    def fmt(x):
        return f"{x:.0f}" if abs(x) >= 100 else f"{x:.2f}"

    low_labels = [f"{fmt(edges[i])}–{fmt(edges[i+1])}" for i in range(len(edges) - 1)]
    high_label = f">{high_income_threshold}"

    bins = np.concatenate([edges, [np.inf]])
    labels = low_labels + [high_label]
    return bins, labels


def parse_observations_to_array(x):
    """
    Robustly parse an observations cell into a 1D numpy array.
    Supports:
      - Python-list strings like "[0.1, 0.2, 0.3]"
      - numpy-like strings
      - actual list / tuple / np.ndarray
    Returns np.ndarray of dtype float, or None if parsing fails.
    """
    if isinstance(x, np.ndarray):
        arr = x.astype(float).ravel()
        return arr

    if isinstance(x, (list, tuple)):
        try:
            return np.asarray(x, dtype=float).ravel()
        except Exception:
            return None

    if pd.isna(x):
        return None

    s = str(x).strip()
    if not s:
        return None

    # First try literal_eval
    try:
        val = ast.literal_eval(s)
        arr = np.asarray(val, dtype=float).ravel()
        return arr
    except Exception:
        pass

    # Fallback: extract numbers with regex
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        return None
    try:
        return np.asarray([float(v) for v in nums], dtype=float)
    except Exception:
        return None


def compute_tt_minus_best_observed_route(
    df: pd.DataFrame,
    travel_time_col="travel_time",
    observations_col="observations",
    obs_tt_scale=100.0,
    travel_time_unit="minutes",   # "minutes" or "seconds"
    output_col="tt_diff",
):
    """
    Creates a new column:
        experienced_travel_time - min(first 3 values of observations) * obs_tt_scale

    Assumes:
      - first 3 entries of observations are normalized travel times
      - multiplying by obs_tt_scale de-normalizes them
      - result is in same unit as travel_time

    Rows with invalid observations become NaN.
    """
    if travel_time_col not in df.columns or observations_col not in df.columns:
        df[output_col] = np.nan
        return df

    travel_time = pd.to_numeric(df[travel_time_col], errors="coerce")
    if travel_time_unit.lower().startswith("sec"):
        travel_time = travel_time / 60.0

    mins = []
    for obs in df[observations_col]:
        arr = parse_observations_to_array(obs)
        if arr is None or len(arr) < 3:
            mins.append(np.nan)
            continue

        first3 = arr[:3].astype(float)
        if np.any(~np.isfinite(first3)):
            mins.append(np.nan)
            continue

        best_tt = np.min(first3) * obs_tt_scale
        if not np.isfinite(best_tt):
            mins.append(np.nan)
            continue

        mins.append(best_tt)

    best_route_tt = pd.Series(mins, index=df.index, dtype=float)
    df[output_col] = travel_time - best_route_tt
    return df



def plot_tt_minus_best_observed_route_by_income_bin_seed_pairs(
    algo_runs,
    monetary_key="monetary pricing",
    karma_key="karma pricing",

    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    observations_col="observations",
    kind_col="kind",
    plot_kind="AV",

    high_income_threshold=6000,
    n_equal_bins=4,

    obs_tt_scale=100.0,
    travel_time_unit="minutes",
    avg_mode="row",          # "row" or "agent"

    center_stat="mean",      # "mean" or "median"
    show_percentile_band=True,
    band_percentiles=(25, 75),   # e.g. (25,75) or (10,90)
    figsize_per_panel=(5.2, 4.4),
    save_path=None,
    show=True,

    smooth_lines=True,
    smooth_window=1,

    show_bin_n=True,
    bin_n_agg="mean",        # "mean" or "minmax"
):
    """
    Creates one subplot per paired seed:
      subplot i compares:
        monetary pricing run i
        karma pricing run i

    Y-axis:
      mean or median of (experienced TT - best observed TT)

    Band:
      percentile band across individuals/rows within each income bin
      (depending on avg_mode)
    """

    if monetary_key not in algo_runs or karma_key not in algo_runs:
        raise KeyError(f"algo_runs must contain '{monetary_key}' and '{karma_key}'.")

    if center_stat not in {"mean", "median"}:
        raise ValueError("center_stat must be 'mean' or 'median'.")

    monetary_runs = algo_runs[monetary_key]
    karma_runs = algo_runs[karma_key]

    n_pairs = min(len(monetary_runs), len(karma_runs))
    if n_pairs == 0:
        raise ValueError("No paired runs found.")

    if len(monetary_runs) != len(karma_runs):
        print(f"[WARN] Number of runs differs: "
              f"{monetary_key}={len(monetary_runs)}, {karma_key}={len(karma_runs)}. "
              f"Using first {n_pairs} paired runs.")

    paired_scenarios = [
        (f"seed {i+1}", monetary_runs[i], karma_runs[i])
        for i in range(n_pairs)
    ]

    all_run_dirs = [rd for _, mrd, krd in paired_scenarios for rd in [mrd, krd]]
    bins_edges, bin_labels = build_income_bins_like_your_code(
        all_run_dirs,
        high_income_threshold=high_income_threshold,
        n_equal_bins=n_equal_bins,
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        id_col=id_col,
        income_col=income_col,
    )
    n_bins = len(bin_labels)
    x = np.arange(n_bins)

    p_lo, p_hi = band_percentiles
    if not (0 <= p_lo < p_hi <= 100):
        raise ValueError("band_percentiles must satisfy 0 <= low < high <= 100.")

    def assign_income_group(income_series: pd.Series) -> pd.Series:
        out = pd.Series(pd.NA, index=income_series.index, dtype="object")
        hi = income_series > high_income_threshold
        out.loc[hi] = f">{high_income_threshold}"
        lo = ~hi
        if lo.any():
            out.loc[lo] = pd.cut(
                income_series.loc[lo],
                bins=bins_edges[:-1],
                labels=bin_labels[:-1],
                include_lowest=True,
                right=True,
            ).astype(object)
        return out

    ncols = min(3, n_pairs)
    nrows = int(np.ceil(n_pairs / ncols))
    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    global_ymins = []
    global_ymaxs = []

    for ax, (seed_name, monetary_run, karma_run) in zip(axes, paired_scenarios):
        scenario_specs = [
            (monetary_key, monetary_run),
            (karma_key, karma_run),
        ]

        pooled_n_across_scenarios = []

        for sc_label, run_dir in scenario_specs:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                print(f"[WARN] {seed_name} / {sc_label}: no data.")
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                print(f"[WARN] {seed_name} / {sc_label}: empty after kind filter.")
                continue

            needed = [id_col, income_col, travel_time_col, observations_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {seed_name} / {sc_label}: missing {missing}. Skipping.")
                continue

            tmp = df[[id_col, income_col, travel_time_col, observations_col]].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[travel_time_col] = pd.to_numeric(tmp[travel_time_col], errors="coerce")

            tmp = compute_tt_minus_best_observed_route(
                tmp,
                travel_time_col=travel_time_col,
                observations_col=observations_col,
                obs_tt_scale=obs_tt_scale,
                travel_time_unit=travel_time_unit,
                output_col="tt_diff",
            )

            tmp = tmp.dropna(subset=[id_col, income_col, "tt_diff"])
            if tmp.empty:
                print(f"[WARN] {seed_name} / {sc_label}: no valid rows after tt_diff.")
                continue

            tmp[id_col] = tmp[id_col].astype(int)
            tmp["income_bin"] = assign_income_group(tmp[income_col])
            tmp = tmp.dropna(subset=["income_bin"])
            if tmp.empty:
                print(f"[WARN] {seed_name} / {sc_label}: no valid rows after binning.")
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "income_bin"], observed=True)["tt_diff"]
                       .mean()
                       .reset_index()
                )
                grouped = per_agent.groupby("income_bin", observed=True)["tt_diff"]
                n_series = per_agent.groupby("income_bin", observed=True)[id_col].nunique()
            else:
                grouped = tmp.groupby("income_bin", observed=True)["tt_diff"]
                n_series = tmp.groupby("income_bin", observed=True)[id_col].nunique()

            if center_stat == "median":
                center_series = grouped.median()
            else:
                center_series = grouped.mean()

            q_lo_series = grouped.quantile(p_lo / 100.0)
            q_hi_series = grouped.quantile(p_hi / 100.0)

            center_vec = center_series.reindex(bin_labels).to_numpy(dtype=float)
            q_lo_vec = q_lo_series.reindex(bin_labels).to_numpy(dtype=float)
            q_hi_vec = q_hi_series.reindex(bin_labels).to_numpy(dtype=float)
            n_vec = n_series.reindex(bin_labels).to_numpy(dtype=float)

            pooled_n_across_scenarios.append(n_vec)

            center_plot = center_vec.copy()
            q_lo_plot = q_lo_vec.copy()
            q_hi_plot = q_hi_vec.copy()

            if smooth_lines:
                center_plot = smooth_1d(center_plot, window=smooth_window)
                q_lo_plot = smooth_1d(q_lo_plot, window=smooth_window)
                q_hi_plot = smooth_1d(q_hi_plot, window=smooth_window)

            ax.plot(x, center_plot, marker="o", linewidth=2, label=sc_label)

            if show_percentile_band:
                ax.fill_between(x, q_lo_plot, q_hi_plot, alpha=0.15)

            valid = np.isfinite(center_plot)
            if np.any(valid):
                ymin = np.nanmin(q_lo_plot[np.isfinite(q_lo_plot)]) if np.any(np.isfinite(q_lo_plot)) else np.nanmin(center_plot[valid])
                ymax = np.nanmax(q_hi_plot[np.isfinite(q_hi_plot)]) if np.any(np.isfinite(q_hi_plot)) else np.nanmax(center_plot[valid])
                global_ymins.append(ymin)
                global_ymaxs.append(ymax)

        ax.set_title(seed_name, fontsize=14)
        ax.set_xticks(x)

        if show_bin_n and pooled_n_across_scenarios:
            pooled_n_across_scenarios = np.vstack(pooled_n_across_scenarios)

            if bin_n_agg == "minmax":
                pooled_min = np.nanmin(pooled_n_across_scenarios, axis=0)
                pooled_max = np.nanmax(pooled_n_across_scenarios, axis=0)
                tick_labels = [
                    f"{lab}\n(n={int(round(a))}–{int(round(b))})"
                    for lab, a, b in zip(bin_labels, pooled_min, pooled_max)
                ]
            else:
                pooled = np.nanmean(pooled_n_across_scenarios, axis=0)
                tick_labels = [
                    f"{lab}\n(n≈{int(round(n))})"
                    for lab, n in zip(bin_labels, pooled)
                ]
            ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=10)
        else:
            ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=10)

        ax.tick_params(axis="y", labelsize=11)
        ax.set_xlabel("Income bin", fontsize=12)
        ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=10)

    for ax in axes[n_pairs:]:
        ax.axis("off")

    if global_ymins and global_ymaxs:
        ymin = min(global_ymins)
        ymax = max(global_ymaxs)
        for ax in axes[:n_pairs]:
            ax.set_ylim(ymin, ymax)

    ylabel_stat = "median" if center_stat == "median" else "mean"
    for i, ax in enumerate(axes[:n_pairs]):
        if i % ncols == 0:
            ax.set_ylabel(f"{ylabel_stat} experienced TT - best observed TT", fontsize=14)

    fig.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig

ALGO_RUNS = {   


    "monetary pricing":[
        #r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_10_heart_exps/episodes",
   
    ],

    "karma pricing":[
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
    ],


}


plot_tt_minus_best_observed_route_by_income_bin_seed_pairs(
    ALGO_RUNS,
    monetary_key="monetary pricing",
    karma_key="karma pricing",
    episode_range=(1320, 1320),
    travel_time_col="travel_time",
    observations_col="observation",
    plot_kind="AV",
    high_income_threshold=5000,
    n_equal_bins=5,
    obs_tt_scale=100.0,
    travel_time_unit="minutes",
    avg_mode="row",
    center_stat="mean",
    show_percentile_band=False,
    band_percentiles=(25, 75),
    smooth_window=3,
    save_path="imgs/tt_minus_best_observed_seed_pairs_percentiles_median.png",
    show=True,
)