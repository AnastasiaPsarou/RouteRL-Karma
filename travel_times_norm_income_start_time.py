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


def plot_tt_minus_best_observed_by_income_bin_and_start_quantile_bars(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    observations_col="observations",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",

    # income binning
    high_income_threshold=6000,
    n_equal_bins=4,

    # units / normalization
    obs_tt_scale=100.0,
    travel_time_unit="minutes",

    # start-time quantiles
    start_time_unit="minutes",   # only used for quantile ordering if conversion is needed later
    n_start_quantiles=5,

    # aggregation
    avg_mode="row",              # "row" or "agent"

    # across-runs error bars
    show_std=True,

    figsize_per_panel=(6.5, 5.0),
    save_path=None,
    show=True,
):
    """
    Grouped barplot:
      x = income bins
      within each income bin = start-time quantiles (Q1..Q5)
      y = avg(experienced travel time - best observed travel time)

    Q1 = earliest 20% of start times
    Q5 = latest 20% of start times
    """

    scenarios = list(algo_runs.keys())
    n_panels = len(scenarios)

    # consistent income bins across all runs
    all_run_dirs = [rd for runs in algo_runs.values() for rd in runs]
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
    n_income_bins = len(bin_labels)

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

    def assign_start_quantiles(start_series: pd.Series, n_q: int = 5) -> pd.Series:
        s = pd.to_numeric(start_series, errors="coerce")
        out = pd.Series(pd.NA, index=s.index, dtype="object")
        valid = s.notna()
        if valid.sum() == 0:
            return out

        labels = [f"Q{i}" for i in range(1, n_q + 1)]
        try:
            q = pd.qcut(s[valid], q=n_q, labels=labels, duplicates="drop")
            out.loc[valid] = q.astype(object)
        except Exception:
            ranks = s[valid].rank(method="first")
            q = pd.qcut(ranks, q=n_q, labels=labels, duplicates="drop")
            out.loc[valid] = q.astype(object)
        return out

    fig_w = figsize_per_panel[0] * n_panels
    fig_h = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), sharey=True)
    if n_panels == 1:
        axes = [axes]

    x = np.arange(n_income_bins)
    bar_width = 0.14 if n_start_quantiles == 5 else 0.8 / max(n_start_quantiles, 1)
    offsets = (np.arange(n_start_quantiles) - (n_start_quantiles - 1) / 2.0) * bar_width
    quantile_labels = [f"Q{i}" for i in range(1, n_start_quantiles + 1)]

    for ax, sc in zip(axes, scenarios):
        per_run_mats = []

        for run_dir in algo_runs[sc]:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, travel_time_col, observations_col, start_time_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[[id_col, income_col, travel_time_col, observations_col, start_time_col]].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[travel_time_col] = pd.to_numeric(tmp[travel_time_col], errors="coerce")
            tmp[start_time_col] = pd.to_numeric(tmp[start_time_col], errors="coerce")

            # compute travel_time - best_observed_travel_time
            tmp = compute_tt_minus_best_observed_route(
                tmp,
                travel_time_col=travel_time_col,
                observations_col=observations_col,
                obs_tt_scale=obs_tt_scale,
                travel_time_unit=travel_time_unit,
                output_col="tt_diff",
            )

            tmp = tmp.dropna(subset=[id_col, income_col, start_time_col, "tt_diff"])
            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)

            # income bins
            tmp["income_bin"] = assign_income_group(tmp[income_col])
            tmp = tmp.dropna(subset=["income_bin"])
            if tmp.empty:
                continue

            # start-time quantiles
            tmp["start_q"] = assign_start_quantiles(tmp[start_time_col], n_q=n_start_quantiles)
            tmp = tmp.dropna(subset=["start_q"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "income_bin", "start_q"], observed=True)["tt_diff"]
                       .mean()
                       .reset_index()
                )
                series = (
                    per_agent.groupby(["income_bin", "start_q"], observed=True)["tt_diff"]
                    .mean()
                )
            else:
                series = (
                    tmp.groupby(["income_bin", "start_q"], observed=True)["tt_diff"]
                    .mean()
                )

            mat = np.full((n_income_bins, n_start_quantiles), np.nan, dtype=float)
            for i_bin, inc_lab in enumerate(bin_labels):
                for j_q, q_lab in enumerate(quantile_labels):
                    key = (inc_lab, q_lab)
                    if key in series.index:
                        mat[i_bin, j_q] = float(series.loc[key])

            per_run_mats.append(mat)

        if not per_run_mats:
            ax.set_title(sc + " (no data)")
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, rotation=25, ha="right")
            ax.grid(True, axis="y", alpha=0.25)
            continue

        arr = np.stack(per_run_mats, axis=0)   # (n_runs, n_income_bins, n_start_quantiles)
        mean_mat = np.nanmean(arr, axis=0)

        if show_std and arr.shape[0] > 1:
            std_mat = np.nanstd(arr, axis=0, ddof=1)
        else:
            std_mat = None

        for j_q, q_lab in enumerate(quantile_labels):
            heights = mean_mat[:, j_q]
            yerr = std_mat[:, j_q] if std_mat is not None else None

            ax.bar(
                x + offsets[j_q],
                heights,
                width=bar_width,
                yerr=yerr,
                capsize=3 if yerr is not None else 0,
                alpha=0.85,
                label=q_lab,
            )

        ax.set_title(sc)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=25, ha="right")
        ax.set_xlabel("Income bin")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="Start-time quantile", frameon=False)

    axes[0].set_ylabel("experienced TT - best observed TT")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig

def build_vot_bins(
    run_dirs,
    high_vot_threshold=None,   # e.g. 200, or None to use only quantile bins
    n_equal_bins=4,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    urgency_col="urgency",
    work_hours_per_day=8.0,
):
    """
    Build global VoT bins across all runs.

    VoT = urgency * (daily_income / work_hours_per_day)

    If high_vot_threshold is given:
      - values <= threshold are split into quantile bins
      - one extra top bin is added: > high_vot_threshold

    If high_vot_threshold is None:
      - all values are split into n_equal_bins quantile bins
    """
    pooled_vot = []

    for rd in run_dirs:
        df = load_episodes(rd, n_last=last_n, episode_range=episode_range, episodes=episodes)
        if df is None or df.empty:
            continue

        needed = [income_col, urgency_col]
        if any(c not in df.columns for c in needed):
            print(f"[WARN] Could not derive VoT from {rd}. Missing one of {needed}. Skipping run.")
            continue

        tmp = df[[income_col, urgency_col]].copy()
        tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
        tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
        tmp = tmp.dropna(subset=[income_col, urgency_col])
        if tmp.empty:
            continue

        tmp["vot"] = tmp[urgency_col] * (tmp[income_col] / float(work_hours_per_day))
        vot = pd.to_numeric(tmp["vot"], errors="coerce").dropna().to_numpy()
        if vot.size:
            pooled_vot.append(vot)

    if not pooled_vot:
        raise ValueError("No valid VoT values found across runs.")

    pooled_vot = np.concatenate(pooled_vot)

    def fmt(x):
        return f"{x:.0f}" if abs(x) >= 100 else f"{x:.2f}"

    if high_vot_threshold is None:
        qs = np.linspace(0, 1, n_equal_bins + 1)
        edges = np.unique(np.quantile(pooled_vot, qs))
        if len(edges) < 2:
            raise ValueError("Quantile edges collapsed for VoT bins.")
        labels = [f"{fmt(edges[i])}–{fmt(edges[i+1])}" for i in range(len(edges) - 1)]
        bins = edges
        return bins, labels, False

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

def plot_tt_minus_best_observed_by_vot_bin_and_start_quantile_bars(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    observations_col="observation",   # change if your CSV uses "observations"
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",

    # VoT binning
    high_vot_threshold=None,   # e.g. 200, or None
    n_equal_bins=4,
    work_hours_per_day=8.0,

    # units / normalization
    obs_tt_scale=100.0,
    travel_time_unit="minutes",

    # start-time quantiles
    n_start_quantiles=5,

    # aggregation
    avg_mode="row",   # "row" or "agent"

    # across-runs error bars
    show_std=True,

    figsize_per_panel=(6.8, 5.0),
    save_path=None,
    show=True,
):
    """
    Grouped barplot:
      x = VoT bins
      within each VoT bin = start-time quantiles (Q1..Q5)
      y = avg(experienced travel time - best observed travel time)

    VoT = urgency * (daily_income / work_hours_per_day)
    """

    scenarios = list(algo_runs.keys())
    n_panels = len(scenarios)

    # consistent VoT bins across all runs
    all_run_dirs = [rd for runs in algo_runs.values() for rd in runs]
    bins_edges, bin_labels, has_top_bin = build_vot_bins(
        all_run_dirs,
        high_vot_threshold=high_vot_threshold,
        n_equal_bins=n_equal_bins,
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        id_col=id_col,
        income_col=income_col,
        urgency_col=urgency_col,
        work_hours_per_day=work_hours_per_day,
    )
    n_vot_bins = len(bin_labels)

    def assign_vot_group(vot_series: pd.Series) -> pd.Series:
        out = pd.Series(pd.NA, index=vot_series.index, dtype="object")

        if has_top_bin:
            hi = vot_series > high_vot_threshold
            out.loc[hi] = bin_labels[-1]
            lo = ~hi
            if lo.any():
                out.loc[lo] = pd.cut(
                    vot_series.loc[lo],
                    bins=bins_edges[:-1],
                    labels=bin_labels[:-1],
                    include_lowest=True,
                    right=True,
                ).astype(object)
        else:
            out.loc[:] = pd.cut(
                vot_series,
                bins=bins_edges,
                labels=bin_labels,
                include_lowest=True,
                right=True,
            ).astype(object)

        return out

    def assign_start_quantiles(start_series: pd.Series, n_q: int = 5) -> pd.Series:
        s = pd.to_numeric(start_series, errors="coerce")
        out = pd.Series(pd.NA, index=s.index, dtype="object")
        valid = s.notna()
        if valid.sum() == 0:
            return out

        labels = [f"Q{i}" for i in range(1, n_q + 1)]
        try:
            q = pd.qcut(s[valid], q=n_q, labels=labels, duplicates="drop")
            out.loc[valid] = q.astype(object)
        except Exception:
            ranks = s[valid].rank(method="first")
            q = pd.qcut(ranks, q=n_q, labels=labels, duplicates="drop")
            out.loc[valid] = q.astype(object)
        return out

    fig_w = figsize_per_panel[0] * n_panels
    fig_h = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), sharey=True)
    if n_panels == 1:
        axes = [axes]

    x = np.arange(n_vot_bins)
    bar_width = 0.14 if n_start_quantiles == 5 else 0.8 / max(n_start_quantiles, 1)
    offsets = (np.arange(n_start_quantiles) - (n_start_quantiles - 1) / 2.0) * bar_width
    quantile_labels = [f"Q{i}" for i in range(1, n_start_quantiles + 1)]

    for ax, sc in zip(axes, scenarios):
        per_run_mats = []

        for run_dir in algo_runs[sc]:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, urgency_col, travel_time_col, observations_col, start_time_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[[id_col, income_col, urgency_col, travel_time_col, observations_col, start_time_col]].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
            tmp[travel_time_col] = pd.to_numeric(tmp[travel_time_col], errors="coerce")
            tmp[start_time_col] = pd.to_numeric(tmp[start_time_col], errors="coerce")

            # compute TT - best observed TT
            tmp = compute_tt_minus_best_observed_route(
                tmp,
                travel_time_col=travel_time_col,
                observations_col=observations_col,
                obs_tt_scale=obs_tt_scale,
                travel_time_unit=travel_time_unit,
                output_col="tt_diff",
            )

            tmp = tmp.dropna(subset=[id_col, income_col, urgency_col, start_time_col, "tt_diff"])
            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)

            # VoT
            tmp["vot"] = tmp[urgency_col] * (tmp[income_col] / float(work_hours_per_day))

            # VoT bins
            tmp["vot_bin"] = assign_vot_group(tmp["vot"])
            tmp = tmp.dropna(subset=["vot_bin"])
            if tmp.empty:
                continue

            # start-time quantiles
            tmp["start_q"] = assign_start_quantiles(tmp[start_time_col], n_q=n_start_quantiles)
            tmp = tmp.dropna(subset=["start_q"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "vot_bin", "start_q"], observed=True)["tt_diff"]
                       .mean()
                       .reset_index()
                )
                series = (
                    per_agent.groupby(["vot_bin", "start_q"], observed=True)["tt_diff"]
                    .mean()
                )
            else:
                series = (
                    tmp.groupby(["vot_bin", "start_q"], observed=True)["tt_diff"]
                    .mean()
                )

            mat = np.full((n_vot_bins, n_start_quantiles), np.nan, dtype=float)
            for i_bin, vot_lab in enumerate(bin_labels):
                for j_q, q_lab in enumerate(quantile_labels):
                    key = (vot_lab, q_lab)
                    if key in series.index:
                        mat[i_bin, j_q] = float(series.loc[key])

            per_run_mats.append(mat)

        if not per_run_mats:
            ax.set_title(sc + " (no data)")
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, rotation=25, ha="right")
            ax.grid(True, axis="y", alpha=0.25)
            continue

        arr = np.stack(per_run_mats, axis=0)   # (n_runs, n_vot_bins, n_start_quantiles)
        mean_mat = np.nanmean(arr, axis=0)

        if show_std and arr.shape[0] > 1:
            std_mat = np.nanstd(arr, axis=0, ddof=1)
        else:
            std_mat = None

        for j_q, q_lab in enumerate(quantile_labels):
            heights = mean_mat[:, j_q]
            yerr = std_mat[:, j_q] if std_mat is not None else None

            ax.bar(
                x + offsets[j_q],
                heights,
                width=bar_width,
                yerr=yerr,
                capsize=3 if yerr is not None else 0,
                alpha=0.85,
                label=q_lab,
            )

        ax.set_title(sc)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=25, ha="right")
        ax.set_xlabel("VoT bin")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="Start-time quantile", frameon=False)

    axes[0].set_ylabel("experienced TT - best observed TT")
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


    "monetary pricing":[
        #r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        ##r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_2_5_heart_exps/episodes",
    
    ],
    "monetary-pricing 1":[
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_1_heart_exps_longer/episodes",
    ],

    "karma pricing":[
            r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_norm_reward/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_norm_reward/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_norm_reward/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_norm_reward/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_norm_reward/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_norm_reward/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_norm_reward/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_norm_reward/episodes",
        ],


    }

"""plot_tt_minus_best_observed_by_income_bin_and_start_quantile_bars(
    ALGO_RUNS,
    episode_range=(300, 300),
    travel_time_col="travel_time",
    observations_col="observation",   # or "observations" if that is your real column name
    start_time_col="start_time",
    plot_kind="AV",
    high_income_threshold=4700,
    n_equal_bins=4,
    obs_tt_scale=100.0,
    travel_time_unit="minutes",
    n_start_quantiles=5,
    avg_mode="row",   # or "agent"
    show_std=True,
    save_path="imgs/avg_tt_minus_best_observed_by_income_bin_start_quantile_bars.png",
    show=True,
)"""
plot_tt_minus_best_observed_by_vot_bin_and_start_quantile_bars(
    ALGO_RUNS,
    episode_range=(1320, 1320),
    travel_time_col="travel_time",
    observations_col="observation",   # change to "observations" if needed
    start_time_col="start_time",
    urgency_col="urgency",
    plot_kind="AV",

    high_vot_threshold=None,   # or set something like 250 if you want a top-coded last bin
    n_equal_bins=5,            # total number of VoT bins if high_vot_threshold=None
    work_hours_per_day=8.0,

    obs_tt_scale=100.0,
    travel_time_unit="minutes",

    n_start_quantiles=5,
    avg_mode="row",            # or "agent"
    show_std=True,

    save_path="imgs/avg_tt_minus_best_observed_by_vot_bin_start_quantile_bars.png",
    show=True,
)