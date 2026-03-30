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


def plot_avg_travel_time_by_income_bin(
    algo_runs,                      # dict scenario -> list of run dirs
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    kind_col="kind",
    plot_kind="AV",                 # "AV", "Human", "All"

    # binning
    high_income_threshold=6000,
    n_equal_bins=4,

    # units
    travel_time_unit="minutes",     # "minutes" or "seconds"
    # aggregation:
    # "row" = average travel_time across all rows in selected episodes
    # "agent" = average travel_time per agent first, then average agents (each agent equal weight)
    avg_mode="row",

    # across-runs error bars
    show_std=True,

    figsize_per_panel=(4.0, 4.0),
    save_path=None,
    show=True,
):
    """
    x: income bins
    y: average travel_time (minutes), computed over selected episodes.
    One panel per scenario. Error bars are std across runs.
    """

    scenarios = list(algo_runs.keys())
    n_panels = len(scenarios)

    # build global bins across all runs (consistent bins)
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
    n_bins = len(bin_labels)

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

    def to_minutes(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        if travel_time_unit.lower().startswith("sec"):
            return x / 60.0
        return x

    fig_w = figsize_per_panel[0] * n_panels
    fig_h = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), sharey=True)
    if n_panels == 1:
        axes = [axes]

    x = np.arange(n_bins)

    for ax, sc in zip(axes, scenarios):
        per_run_vecs = []

        for run_dir in algo_runs[sc]:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, travel_time_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            df = df[[id_col, income_col, travel_time_col]].copy()
            df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
            df[income_col] = pd.to_numeric(df[income_col], errors="coerce")
            df[travel_time_col] = to_minutes(df[travel_time_col])

            df = df.dropna(subset=[id_col, income_col, travel_time_col])
            if df.empty:
                continue

            df[id_col] = df[id_col].astype(int)

            # income bins
            df["income_bin"] = assign_income_group(df[income_col])
            df = df.dropna(subset=["income_bin"])
            if df.empty:
                continue

            if avg_mode == "agent":
                # average per agent first (equal weight per agent)
                per_agent = (
                    df.groupby([id_col, "income_bin"], observed=True)[travel_time_col]
                      .mean()
                      .reset_index()
                )
                series = per_agent.groupby("income_bin", observed=True)[travel_time_col].mean()
            else:
                # average across all rows (each row equal weight)
                series = df.groupby("income_bin", observed=True)[travel_time_col].mean()

            vec = series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_vecs.append(vec)

        if not per_run_vecs:
            ax.set_title(sc + " (no data)")
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, rotation=25, ha="right")
            ax.grid(True, axis="y", alpha=0.25)
            continue

        arr = np.vstack(per_run_vecs)
        mean_vec = np.nanmean(arr, axis=0)

        ddof = 1 if arr.shape[0] > 1 else 0
        std_vec = np.nanstd(arr, axis=0, ddof=ddof) if (show_std and arr.shape[0] > 1) else None

        ax.bar(x, mean_vec, alpha=0.8, yerr=std_vec, capsize=4)
        ax.set_title(sc)
        ax.set_xlabel("Income bin")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel(f"Avg travel time (minutes)")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig

def plot_avg_travel_time_by_income_bin_lines(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    kind_col="kind",
    plot_kind="AV",

    high_income_threshold=6000,
    n_equal_bins=4,

    travel_time_unit="minutes",   # "minutes" or "seconds"
    avg_mode="row",               # "row" or "agent"

    show_std_band=True,           # shaded band = ± std across runs
    figsize=(6, 5.2),
    save_path=None,
    show=True,

    smooth_lines=True,
    smooth_window=1,

    show_bin_n=True,
    bin_n_agg="mean",   # "mean" or "minmax"
):
    """
    ONE plot:
      x = income bins
      y = avg travel time (minutes)
      one LINE per scenario
      optional shaded band = ± std across runs for each scenario
    """

    # --- build global bins across all runs (consistent bins) ---
    scenarios = list(algo_runs.keys())
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
    n_bins = len(bin_labels)
    x = np.arange(n_bins)

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

    def to_minutes(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if travel_time_unit.lower().startswith("sec"):
            return s / 60.0
        return s

    fig, ax = plt.subplots(figsize=figsize)
    pooled_n_across_scenarios = []

    for sc in scenarios:
        run_dirs = algo_runs[sc]
        per_run_vecs = []
        per_run_n = []

        for run_dir in run_dirs:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, travel_time_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[[id_col, income_col, travel_time_col]].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[travel_time_col] = to_minutes(tmp[travel_time_col])
            tmp = tmp.dropna(subset=[id_col, income_col, travel_time_col])
            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)
            tmp["income_bin"] = assign_income_group(tmp[income_col])
            tmp = tmp.dropna(subset=["income_bin"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "income_bin"], observed=True)[travel_time_col]
                       .mean()
                       .reset_index()
                )
                series = per_agent.groupby("income_bin", observed=True)[travel_time_col].mean()
            else:
                series = tmp.groupby("income_bin", observed=True)[travel_time_col].mean()

            vec = series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_vecs.append(vec)

            n_series = tmp.groupby("income_bin", observed=True)[id_col].nunique()
            n_vec = n_series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_n.append(n_vec)

        if not per_run_vecs:
            print(f"[WARN] Scenario '{sc}' had no usable runs. Skipping line.")
            continue

        arr = np.vstack(per_run_vecs)  # (n_runs, n_bins)
        n_arr = np.vstack(per_run_n)   # (n_runs, n_bins)
        pooled_n_across_scenarios.append(n_arr)
        mean_vec = np.nanmean(arr, axis=0)

        ddof = 1 if arr.shape[0] > 1 else 0
        std_vec = np.nanstd(arr, axis=0, ddof=ddof) if (arr.shape[0] > 1) else None

        mean_plot = mean_vec.copy()
        std_plot = std_vec.copy() if std_vec is not None else None

        if smooth_lines:
            mean_plot = smooth_1d(mean_plot, window=smooth_window)
            if std_plot is not None:
                std_plot = smooth_1d(std_plot, window=smooth_window)

        # line with markers
        ax.plot(x, mean_plot, marker="o", linewidth=2, label=f"{sc}")

        # optional std band (smoothed consistently)
        if show_std_band and std_plot is not None:
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)


    ax.set_xticks(x)
    # ---- NEW: show number of individuals per bin on x tick labels ----
    if show_bin_n and pooled_n_across_scenarios:
        # average counts across runs within each scenario, then average across scenarios
        scen_means = [np.nanmean(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
        pooled = np.nanmean(np.vstack(scen_means), axis=0)

        if bin_n_agg == "minmax":
            scen_mins = [np.nanmin(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            scen_maxs = [np.nanmax(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            pooled_min = np.nanmin(np.vstack(scen_mins), axis=0)
            pooled_max = np.nanmax(np.vstack(scen_maxs), axis=0)
            tick_labels = [
                f"{lab}\n(n={int(round(a))}–{int(round(b))})"
                for lab, a, b in zip(bin_labels, pooled_min, pooled_max)
            ]
        else:
            tick_labels = [
                f"{lab}\n(n≈{int(round(n))})"
                for lab, n in zip(bin_labels, pooled)
            ]

        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=12)
    else:
        ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=12)
    #ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    #ax.set_ylim(27, 30.5)
    ax.set_xlabel("Income bin", fontsize=16)
    ax.set_ylabel(f"Avg travel time (minutes)", fontsize=16)
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

def compute_tt_minus_start_time(
    df: pd.DataFrame,
    travel_time_col="travel_time",
    start_time_col="start_time",
    travel_time_unit="minutes",   # "minutes" or "seconds"
    start_time_unit="minutes",    # "minutes", "seconds", or "steps"
    step_to_minutes=1.0,          # used only if start_time_unit == "steps"
    output_col="tt_minus_start",
):
    """
    Creates a new column:
        experienced_travel_time - start_time

    Units are aligned to minutes if needed.

    Parameters
    ----------
    travel_time_col : str
        Column with experienced travel time.
    start_time_col : str
        Column with the agent's start/departure time.
    travel_time_unit : str
        Unit of travel_time_col: "minutes" or "seconds".
    start_time_unit : str
        Unit of start_time_col: "minutes", "seconds", or "steps".
    step_to_minutes : float
        Conversion factor if start_time is stored in simulation steps.
    output_col : str
        Name of output column.
    """
    if travel_time_col not in df.columns or start_time_col not in df.columns:
        df[output_col] = np.nan
        return df

    travel_time = pd.to_numeric(df[travel_time_col], errors="coerce")
    start_time = pd.to_numeric(df[start_time_col], errors="coerce")

    # convert travel time to minutes
    if travel_time_unit.lower().startswith("sec"):
        travel_time = travel_time / 60.0

    # convert start time to minutes
    unit = start_time_unit.lower()
    if unit.startswith("sec"):
        start_time = start_time / 60.0
    elif unit.startswith("step"):
        start_time = start_time * step_to_minutes
    # else assume already minutes

    df[output_col] = travel_time - start_time
    return df


def plot_tt_minus_start_time_by_income_bin_lines(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    start_time_col="start_time",   # <-- change if your column has another name
    kind_col="kind",
    plot_kind="AV",

    high_income_threshold=6000,
    n_equal_bins=4,

    travel_time_unit="minutes",
    start_time_unit="minutes",     # "minutes", "seconds", or "steps"
    step_to_minutes=1.0,
    avg_mode="row",                # "row" or "agent"

    show_std_band=True,
    figsize=(6, 5.2),
    save_path=None,
    show=True,

    smooth_lines=True,
    smooth_window=1,

    show_bin_n=True,
    bin_n_agg="mean",
):
    """
    ONE plot:
      x = income bins
      y = avg( experienced_travel_time - start_time )
    """

    scenarios = list(algo_runs.keys())
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
    n_bins = len(bin_labels)
    x = np.arange(n_bins)

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

    fig, ax = plt.subplots(figsize=figsize)
    pooled_n_across_scenarios = []

    for sc in scenarios:
        run_dirs = algo_runs[sc]
        per_run_vecs = []
        per_run_n = []

        for run_dir in run_dirs:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, travel_time_col, start_time_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[[id_col, income_col, travel_time_col, start_time_col]].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[travel_time_col] = pd.to_numeric(tmp[travel_time_col], errors="coerce")
            tmp[start_time_col] = pd.to_numeric(tmp[start_time_col], errors="coerce")

            tmp = compute_tt_minus_start_time(
                tmp,
                travel_time_col=travel_time_col,
                start_time_col=start_time_col,
                travel_time_unit=travel_time_unit,
                start_time_unit=start_time_unit,
                step_to_minutes=step_to_minutes,
                output_col="tt_minus_start",
            )

            tmp = tmp.dropna(subset=[id_col, income_col, "tt_minus_start"])
            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)
            tmp["income_bin"] = assign_income_group(tmp[income_col])
            tmp = tmp.dropna(subset=["income_bin"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "income_bin"], observed=True)["tt_minus_start"]
                       .mean()
                       .reset_index()
                )
                series = per_agent.groupby("income_bin", observed=True)["tt_minus_start"].mean()
            else:
                series = tmp.groupby("income_bin", observed=True)["tt_minus_start"].mean()

            vec = series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_vecs.append(vec)

            n_series = tmp.groupby("income_bin", observed=True)[id_col].nunique()
            n_vec = n_series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_n.append(n_vec)

        if not per_run_vecs:
            print(f"[WARN] Scenario '{sc}' had no usable runs. Skipping line.")
            continue

        arr = np.vstack(per_run_vecs)
        n_arr = np.vstack(per_run_n)
        pooled_n_across_scenarios.append(n_arr)

        mean_vec = np.nanmean(arr, axis=0)
        ddof = 1 if arr.shape[0] > 1 else 0
        std_vec = np.nanstd(arr, axis=0, ddof=ddof) if (arr.shape[0] > 1) else None

        mean_plot = mean_vec.copy()
        std_plot = std_vec.copy() if std_vec is not None else None

        if smooth_lines:
            mean_plot = smooth_1d(mean_plot, window=smooth_window)
            if std_plot is not None:
                std_plot = smooth_1d(std_plot, window=smooth_window)

        ax.plot(x, mean_plot, marker="o", linewidth=2, label=f"{sc}")

        if show_std_band and std_plot is not None:
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)

    ax.set_xticks(x)

    if show_bin_n and pooled_n_across_scenarios:
        scen_means = [np.nanmean(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
        pooled = np.nanmean(np.vstack(scen_means), axis=0)

        if bin_n_agg == "minmax":
            scen_mins = [np.nanmin(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            scen_maxs = [np.nanmax(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            pooled_min = np.nanmin(np.vstack(scen_mins), axis=0)
            pooled_max = np.nanmax(np.vstack(scen_maxs), axis=0)
            tick_labels = [
                f"{lab}\n(n={int(round(a))}–{int(round(b))})"
                for lab, a, b in zip(bin_labels, pooled_min, pooled_max)
            ]
        else:
            tick_labels = [
                f"{lab}\n(n≈{int(round(n))})"
                for lab, n in zip(bin_labels, pooled)
            ]
        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=12)
    else:
        ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=12)

    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("Income bin", fontsize=16)
    ax.set_ylabel("experienced TT - start time", fontsize=16)
    ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.7)
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

def plot_avg_vot_by_income_bin_lines(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",          # monthly income in CSV
    urgency_col="urgency",        # urgency column in CSV
    kind_col="kind",
    plot_kind="AV",

    high_income_threshold=6000,
    n_equal_bins=4,

    # monthly -> hourly conversion
    work_hours_per_month=160.0,

    # aggregation:
    # "row" = average VOT across all rows
    # "agent" = average VOT per agent first, then average agents
    avg_mode="row",

    show_std_band=True,
    figsize=(6, 5.2),
    save_path=None,
    show=True,

    smooth_lines=True,
    smooth_window=1,

    show_bin_n=True,
    bin_n_agg="mean",   # "mean" or "minmax"
):
    """
    ONE plot:
      x = income bins
      y = avg value of time
      one LINE per scenario
      optional shaded band = ± std across runs for each scenario

    VOT = hourly_income * urgency
        = (monthly_income / work_hours_per_month) * urgency
    """

    scenarios = list(algo_runs.keys())
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
    n_bins = len(bin_labels)
    x = np.arange(n_bins)

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

    fig, ax = plt.subplots(figsize=figsize)
    pooled_n_across_scenarios = []

    for sc in scenarios:
        run_dirs = algo_runs[sc]
        per_run_vecs = []
        per_run_n = []

        for run_dir in run_dirs:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, urgency_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[[id_col, income_col, urgency_col]].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
            tmp = tmp.dropna(subset=[id_col, income_col, urgency_col])
            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)

            # monthly income -> hourly income
            tmp["hourly_income"] = tmp[income_col] / float(work_hours_per_month)

            # value of time
            tmp["vot"] = tmp["hourly_income"] * tmp[urgency_col]

            # income bins still based on monthly income
            tmp["income_bin"] = assign_income_group(tmp[income_col])
            tmp = tmp.dropna(subset=["income_bin", "vot"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "income_bin"], observed=True)["vot"]
                       .mean()
                       .reset_index()
                )
                series = per_agent.groupby("income_bin", observed=True)["vot"].mean()
            else:
                series = tmp.groupby("income_bin", observed=True)["vot"].mean()

            vec = series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_vecs.append(vec)

            n_series = tmp.groupby("income_bin", observed=True)[id_col].nunique()
            n_vec = n_series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_n.append(n_vec)

        if not per_run_vecs:
            print(f"[WARN] Scenario '{sc}' had no usable runs. Skipping line.")
            continue

        arr = np.vstack(per_run_vecs)
        n_arr = np.vstack(per_run_n)
        pooled_n_across_scenarios.append(n_arr)

        mean_vec = np.nanmean(arr, axis=0)
        ddof = 1 if arr.shape[0] > 1 else 0
        std_vec = np.nanstd(arr, axis=0, ddof=ddof) if (arr.shape[0] > 1) else None

        mean_plot = mean_vec.copy()
        std_plot = std_vec.copy() if std_vec is not None else None

        if smooth_lines:
            mean_plot = smooth_1d(mean_plot, window=smooth_window)
            if std_plot is not None:
                std_plot = smooth_1d(std_plot, window=smooth_window)

        ax.plot(x, mean_plot, marker="o", linewidth=2, label=f"{sc}")

        if show_std_band and std_plot is not None:
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)

    ax.set_xticks(x)

    if show_bin_n and pooled_n_across_scenarios:
        scen_means = [np.nanmean(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
        pooled = np.nanmean(np.vstack(scen_means), axis=0)

        if bin_n_agg == "minmax":
            scen_mins = [np.nanmin(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            scen_maxs = [np.nanmax(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            pooled_min = np.nanmin(np.vstack(scen_mins), axis=0)
            pooled_max = np.nanmax(np.vstack(scen_maxs), axis=0)
            tick_labels = [
                f"{lab}\n(n={int(round(a))}–{int(round(b))})"
                for lab, a, b in zip(bin_labels, pooled_min, pooled_max)
            ]
        else:
            tick_labels = [
                f"{lab}\n(n≈{int(round(n))})"
                for lab, n in zip(bin_labels, pooled)
            ]

        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=12)
    else:
        ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=12)

    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Income bin", fontsize=16)
    ax.set_ylabel("Avg value of time", fontsize=16)
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


def plot_avg_vot_times_travel_time_by_income_bin_lines(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",          # monthly income in CSV
    urgency_col="urgency",        # urgency column in CSV
    travel_time_col="travel_time",
    kind_col="kind",
    plot_kind="AV",

    high_income_threshold=6000,
    n_equal_bins=4,

    # income conversion
    work_hours_per_month=160.0,

    # travel time unit in CSV
    travel_time_unit="minutes",   # "minutes", "seconds", or "hours"

    # aggregation:
    # "row" = average across all rows
    # "agent" = average per agent first, then average agents
    avg_mode="row",

    show_std_band=True,
    figsize=(6, 5.2),
    save_path=None,
    show=True,

    smooth_lines=True,
    smooth_window=1,

    show_bin_n=True,
    bin_n_agg="mean",   # "mean" or "minmax"
):
    """
    ONE plot:
      x = income bins
      y = avg (VOT * travel time)
      one LINE per scenario
      optional shaded band = ± std across runs

    VOT = hourly_income * urgency
    VOT-cost = VOT * travel_time_hours
    """

    scenarios = list(algo_runs.keys())
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
    n_bins = len(bin_labels)
    x = np.arange(n_bins)

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

    def to_hours(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        unit = travel_time_unit.lower()
        if unit.startswith("sec"):
            return s / 3600.0
        if unit.startswith("min"):
            return s / 60.0
        return s  # already hours

    fig, ax = plt.subplots(figsize=figsize)
    pooled_n_across_scenarios = []

    for sc in scenarios:
        run_dirs = algo_runs[sc]
        per_run_vecs = []
        per_run_n = []

        for run_dir in run_dirs:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, urgency_col, travel_time_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[[id_col, income_col, urgency_col, travel_time_col]].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
            tmp["travel_time_hours"] = to_hours(tmp[travel_time_col])

            tmp = tmp.dropna(subset=[id_col, income_col, urgency_col, "travel_time_hours"])
            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)

            # hourly income from monthly income
            tmp["hourly_income"] = tmp[income_col] / float(work_hours_per_month)

            # value of time
            tmp["vot"] = tmp["hourly_income"] * tmp[urgency_col]

            # generalized time cost proxy
            tmp["vot_x_travel_time"] = tmp["vot"] * tmp["travel_time_hours"]

            # bins are still based on monthly income
            tmp["income_bin"] = assign_income_group(tmp[income_col])
            tmp = tmp.dropna(subset=["income_bin", "vot_x_travel_time"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "income_bin"], observed=True)["vot_x_travel_time"]
                       .mean()
                       .reset_index()
                )
                series = per_agent.groupby("income_bin", observed=True)["vot_x_travel_time"].mean()
            else:
                series = tmp.groupby("income_bin", observed=True)["vot_x_travel_time"].mean()

            vec = series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_vecs.append(vec)

            n_series = tmp.groupby("income_bin", observed=True)[id_col].nunique()
            n_vec = n_series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_n.append(n_vec)

        if not per_run_vecs:
            print(f"[WARN] Scenario '{sc}' had no usable runs. Skipping line.")
            continue

        arr = np.vstack(per_run_vecs)
        n_arr = np.vstack(per_run_n)
        pooled_n_across_scenarios.append(n_arr)

        mean_vec = np.nanmean(arr, axis=0)
        ddof = 1 if arr.shape[0] > 1 else 0
        std_vec = np.nanstd(arr, axis=0, ddof=ddof) if (arr.shape[0] > 1) else None

        mean_plot = mean_vec.copy()
        std_plot = std_vec.copy() if std_vec is not None else None

        if smooth_lines:
            mean_plot = smooth_1d(mean_plot, window=smooth_window)
            if std_plot is not None:
                std_plot = smooth_1d(std_plot, window=smooth_window)

        ax.plot(x, mean_plot, marker="o", linewidth=2, label=f"{sc}")

        if show_std_band and std_plot is not None:
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)

    ax.set_xticks(x)

    if show_bin_n and pooled_n_across_scenarios:
        scen_means = [np.nanmean(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
        pooled = np.nanmean(np.vstack(scen_means), axis=0)

        if bin_n_agg == "minmax":
            scen_mins = [np.nanmin(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            scen_maxs = [np.nanmax(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            pooled_min = np.nanmin(np.vstack(scen_mins), axis=0)
            pooled_max = np.nanmax(np.vstack(scen_maxs), axis=0)
            tick_labels = [
                f"{lab}\n(n={int(round(a))}–{int(round(b))})"
                for lab, a, b in zip(bin_labels, pooled_min, pooled_max)
            ]
        else:
            tick_labels = [
                f"{lab}\n(n≈{int(round(n))})"
                for lab, n in zip(bin_labels, pooled)
            ]
        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=12)
    else:
        ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=12)

    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Income bin", fontsize=16)
    ax.set_ylabel("Avg VOT × travel time", fontsize=16)
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
import ast

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


def compute_tt_over_best_observed_route(
    df: pd.DataFrame,
    travel_time_col="travel_time",
    observations_col="observations",
    obs_tt_scale=100.0,
    output_col="tt_over_best_obs_route",
):
    """
    Creates a new column:
        experienced_travel_time / min(first 3 values of observations * obs_tt_scale)

    Assumes:
      - first 3 entries of observations are route travel times normalized by `obs_tt_scale`
      - experienced travel time is in the same unit as the de-normalized route travel times

    Rows with invalid observations or non-positive denominator become NaN.
    """
    if travel_time_col not in df.columns or observations_col not in df.columns:
        df[output_col] = np.nan
        return df

    travel_time = pd.to_numeric(df[travel_time_col], errors="coerce")

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
        if not np.isfinite(best_tt) or best_tt <= 0:
            mins.append(np.nan)
            continue

        mins.append(best_tt)

    best_route_tt = pd.Series(mins, index=df.index, dtype=float)
    df[output_col] = travel_time / best_route_tt
    return df




def plot_tt_over_best_observed_route_by_income_bin_lines(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    observations_col="observation",
    kind_col="kind",
    plot_kind="AV",

    high_income_threshold=6000,
    n_equal_bins=4,

    obs_tt_scale=100.0,   # because observations[:3] are normalized by 100
    avg_mode="row",       # "row" or "agent"

    show_std_band=True,
    figsize=(6, 5.2),
    save_path=None,
    show=True,

    smooth_lines=True,
    smooth_window=1,

    show_bin_n=True,
    bin_n_agg="mean",
):
    """
    ONE plot:
      x = income bins
      y = avg( experienced_travel_time / best_observed_route_travel_time )

    best_observed_route_travel_time is computed from:
      min(observations[:3]) * obs_tt_scale
    """

    scenarios = list(algo_runs.keys())
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
    n_bins = len(bin_labels)
    x = np.arange(n_bins)

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

    fig, ax = plt.subplots(figsize=figsize)
    pooled_n_across_scenarios = []

    for sc in scenarios:
        run_dirs = algo_runs[sc]
        per_run_vecs = []
        per_run_n = []

        for run_dir in run_dirs:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, travel_time_col, observations_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[[id_col, income_col, travel_time_col, observations_col]].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[travel_time_col] = pd.to_numeric(tmp[travel_time_col], errors="coerce")

            tmp = compute_tt_over_best_observed_route(
                tmp,
                travel_time_col=travel_time_col,
                observations_col=observations_col,
                obs_tt_scale=obs_tt_scale,
                output_col="tt_ratio",
            )

            tmp = tmp.dropna(subset=[id_col, income_col, "tt_ratio"])
            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)
            tmp["income_bin"] = assign_income_group(tmp[income_col])
            tmp = tmp.dropna(subset=["income_bin"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "income_bin"], observed=True)["tt_ratio"]
                       .mean()
                       .reset_index()
                )
                series = per_agent.groupby("income_bin", observed=True)["tt_ratio"].mean()
            else:
                series = tmp.groupby("income_bin", observed=True)["tt_ratio"].mean()

            vec = series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_vecs.append(vec)

            n_series = tmp.groupby("income_bin", observed=True)[id_col].nunique()
            n_vec = n_series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_n.append(n_vec)

        if not per_run_vecs:
            print(f"[WARN] Scenario '{sc}' had no usable runs. Skipping line.")
            continue

        arr = np.vstack(per_run_vecs)
        n_arr = np.vstack(per_run_n)
        pooled_n_across_scenarios.append(n_arr)

        mean_vec = np.nanmean(arr, axis=0)
        ddof = 1 if arr.shape[0] > 1 else 0
        std_vec = np.nanstd(arr, axis=0, ddof=ddof) if (arr.shape[0] > 1) else None

        mean_plot = mean_vec.copy()
        std_plot = std_vec.copy() if std_vec is not None else None

        if smooth_lines:
            mean_plot = smooth_1d(mean_plot, window=smooth_window)
            if std_plot is not None:
                std_plot = smooth_1d(std_plot, window=smooth_window)

        ax.plot(x, mean_plot, marker="o", linewidth=2, label=f"{sc}")

        if show_std_band and std_plot is not None:
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)

    ax.set_xticks(x)

    if show_bin_n and pooled_n_across_scenarios:
        scen_means = [np.nanmean(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
        pooled = np.nanmean(np.vstack(scen_means), axis=0)

        if bin_n_agg == "minmax":
            scen_mins = [np.nanmin(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            scen_maxs = [np.nanmax(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            pooled_min = np.nanmin(np.vstack(scen_mins), axis=0)
            pooled_max = np.nanmax(np.vstack(scen_maxs), axis=0)
            tick_labels = [
                f"{lab}\n(n={int(round(a))}–{int(round(b))})"
                for lab, a, b in zip(bin_labels, pooled_min, pooled_max)
            ]
        else:
            tick_labels = [
                f"{lab}\n(n≈{int(round(n))})"
                for lab, n in zip(bin_labels, pooled)
            ]
        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=12)
    else:
        ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=12)

    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Income bin", fontsize=16)
    ax.set_ylabel("Avg experienced TT / best observed TT", fontsize=16)
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


def plot_tt_minus_best_observed_route_by_income_bin_lines(
    algo_runs,
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
    avg_mode="row",       # "row" or "agent"

    show_std_band=True,
    figsize=(6, 5.2),
    save_path=None,
    show=True,

    smooth_lines=True,
    smooth_window=1,

    show_bin_n=True,
    bin_n_agg="mean",
):
    """
    ONE plot:
      x = income bins
      y = avg( experienced_travel_time - best_observed_route_travel_time )

    best_observed_route_travel_time:
      min(observations[:3]) * obs_tt_scale
    """

    scenarios = list(algo_runs.keys())
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
    n_bins = len(bin_labels)
    x = np.arange(n_bins)

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

    fig, ax = plt.subplots(figsize=figsize)
    pooled_n_across_scenarios = []

    for sc in scenarios:
        run_dirs = algo_runs[sc]
        per_run_vecs = []
        per_run_n = []

        for run_dir in run_dirs:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, travel_time_col, observations_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
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
                continue

            tmp[id_col] = tmp[id_col].astype(int)
            tmp["income_bin"] = assign_income_group(tmp[income_col])
            tmp = tmp.dropna(subset=["income_bin"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "income_bin"], observed=True)["tt_diff"]
                       .mean()
                       .reset_index()
                )
                series = per_agent.groupby("income_bin", observed=True)["tt_diff"].mean()
            else:
                series = tmp.groupby("income_bin", observed=True)["tt_diff"].mean()

            vec = series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_vecs.append(vec)

            n_series = tmp.groupby("income_bin", observed=True)[id_col].nunique()
            n_vec = n_series.reindex(bin_labels).to_numpy(dtype=float)
            per_run_n.append(n_vec)

        if not per_run_vecs:
            print(f"[WARN] Scenario '{sc}' had no usable runs. Skipping line.")
            continue

        arr = np.vstack(per_run_vecs)
        n_arr = np.vstack(per_run_n)
        pooled_n_across_scenarios.append(n_arr)

        mean_vec = np.nanmean(arr, axis=0)
        ddof = 1 if arr.shape[0] > 1 else 0
        std_vec = np.nanstd(arr, axis=0, ddof=ddof) if (arr.shape[0] > 1) else None

        mean_plot = mean_vec.copy()
        std_plot = std_vec.copy() if std_vec is not None else None

        if smooth_lines:
            mean_plot = smooth_1d(mean_plot, window=smooth_window)
            if std_plot is not None:
                std_plot = smooth_1d(std_plot, window=smooth_window)

        ax.plot(x, mean_plot, marker="o", linewidth=2, label=f"{sc}")

        if show_std_band and std_plot is not None:
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)

    ax.set_xticks(x)

    if show_bin_n and pooled_n_across_scenarios:
        scen_means = [np.nanmean(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
        pooled = np.nanmean(np.vstack(scen_means), axis=0)

        if bin_n_agg == "minmax":
            scen_mins = [np.nanmin(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            scen_maxs = [np.nanmax(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            pooled_min = np.nanmin(np.vstack(scen_mins), axis=0)
            pooled_max = np.nanmax(np.vstack(scen_maxs), axis=0)
            tick_labels = [
                f"{lab}\n(n={int(round(a))}–{int(round(b))})"
                for lab, a, b in zip(bin_labels, pooled_min, pooled_max)
            ]
        else:
            tick_labels = [
                f"{lab}\n(n≈{int(round(n))})"
                for lab, n in zip(bin_labels, pooled)
            ]
        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=12)
    else:
        ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=12)

    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("Income bin", fontsize=16)
    ax.set_ylabel("experienced TT - best observed TT", fontsize=16)
    ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=12)
    #ax.set_ylim(1, 7)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# =======================
# EXAMPLE USAGE
# =======================
if __name__ == "__main__":

    """ALGO_RUNS = {
        "Karma": [
            r"../../data/18000_iters/training_records_karma_dqn_factorized_300_agents_12_600_iter/episodes",
            r"../../data/18000_iters/training_records_karma_dqn_factorized_300_agents_12_800_iter/episodes",
        ],
        "Monetary-pricing - fee 10": [
            r"../../data/18000_iters/training_records_monetary_pricing_dqn_300_agents_14_fee_10_iter_600/episodes",
            r"../../data/18000_iters/training_records_monetary_pricing_dqn_300_agents_15_fee_10_iter_600/episodes",
        ]
    }"""
    """  "Karma 7": [
            r"training_records_karma_300_agents_9_mean_field_new_net_centr_price_7/episodes",
            r"training_records_karma_300_agents_10_mean_field_new_net_centr_price_7/episodes",
            r"training_records_karma_300_agents_11_mean_field_new_net_centr_price_7/episodes",
            r"training_records_karma_300_agents_12_mean_field_new_net_centr_price_7/episodes",
            r"training_records_karma_300_agents_13_mean_field_new_net_centr_price_7/episodes",
        ],
        "Karma 8": [
            r"training_records_karma_300_agents_9_mean_field_new_net_centr_price_8/episodes",
            r"training_records_karma_300_agents_10_mean_field_new_net_centr_price_8/episodes",
            r"training_records_karma_300_agents_11_mean_field_new_net_centr_price_8/episodes",
            r"training_records_karma_300_agents_12_mean_field_new_net_centr_price_8/episodes",
            r"training_records_karma_300_agents_13_mean_field_new_net_centr_price_8/episodes",
        ],"""

    """ALGO_RUNS = {
    "Karma 8": [
        r"training_records_karma_300_agents_9_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_10_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_11_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_12_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_13_mean_field_new_net_centr_price_8/episodes",
    ],
    "Karma 7": [
        r"training_records_karma_300_agents_9_mean_field_new_net_centr_price_7/episodes",
        r"training_records_karma_300_agents_10_mean_field_new_net_centr_price_7/episodes",
        r"training_records_karma_300_agents_11_mean_field_new_net_centr_price_7/episodes",
        r"training_records_karma_300_agents_12_mean_field_new_net_centr_price_7/episodes",
        r"training_records_karma_300_agents_13_mean_field_new_net_centr_price_7/episodes",
    ],

    "Monetary 20":[
        r"training_records_monetary_pricing_300_agents_9_fee_20/episodes",
        r"training_records_monetary_pricing_300_agents_10_fee_20/episodes",
        r"training_records_monetary_pricing_300_agents_11_fee_20/episodes",
        r"training_records_monetary_pricing_300_agents_12_fee_20/episodes",
        r"training_records_monetary_pricing_300_agents_13_fee_20/episodes",
        #r"training_records_monetary_pricing_300_agents_14_fee_20/episodes",
        #r"training_records_monetary_pricing_300_agents_15_fee_20/episodes",

    ],
    

    }"""

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
        #r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_10_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_10_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_10_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_10_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_10_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_10_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_10_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_route1_fee_1/episodes",
    
    ],

        "karma pricing":[
            #r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
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

    """ALGO_RUNS = {
        "Hypernet server": [
            #r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing/episodes",
        ],
        "hypernet 15":[
            r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_15/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_15/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_15/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_15/episodes"
        ]
    }"""
    """ALGO_RUNS = {

    "monetary 50 norm":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_50_norm_ff/episodes",
    ],
    "monetary 50 norm greedy":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_50_norm_ff_greedy/episodes",
    ],
        "monetary 15 norm":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_15_norm_ff/episodes",
    ],
    "monetary 15 norm greedy":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_15_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_15_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_15_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_15_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_15_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_15_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_15_norm_ff_greedy/episodes",
    ],
    }"""

    """ALGO_RUNS = {
        "Monetary-pricing - fee 8 ": [
            r"../../data/24000_iters/training_records_monetary_pricing_dqn_300_agents_14_fee_8_iter_800/episodes",
            r"../../data/24000_iters/training_records_monetary_pricing_dqn_300_agents_15_fee_8_iter_800/episodes",
            r"../../data/24000_iters/training_records_monetary_pricing_dqn_300_agents_16_fee_8_iter_800/episodes",
        ],
        "Monetary-pricing - fee 8.5": [
            r"../../data/24000_iters/monetary_pricing_different_fees/training_records_monetary_pricing_dqn_300_agents_18_fee_8_5_iter_800/episodes",
        ],
        "Monetary-pricing - fee 9": [
            r"../../data/24000_iters/monetary_pricing_different_fees/training_records_monetary_pricing_dqn_300_agents_18_fee_9_iter_800/episodes",
        ],
        "Monetary-pricing - fee 9.5": [
            r"../../data/24000_iters/monetary_pricing_different_fees/training_records_monetary_pricing_dqn_300_agents_18_fee_9_5_iter_800/episodes",
        ],
        "Monetary-pricing - fee 10": [
            r"../../data/24000_iters/monetary_pricing_different_fees/training_records_monetary_pricing_dqn_300_agents_18_fee_10_iter_800/episodes",
        ],        
    }"""

    """plot_avg_travel_time_by_income_bin(
        ALGO_RUNS,
        episode_range=(18000, 18200),   # <-- your chosen range
        travel_time_col="travel_time",
        travel_time_unit="minutes",     # set to "seconds" if needed
        avg_mode="row",                 # "row" or "agent"
        plot_kind="AV",
        high_income_threshold=6000,
        n_equal_bins=4,
        show_std=True,
        save_path="imgs/avg_travel_time_by_income_bin.png",
        show=True,
    )"""
    """plot_avg_travel_time_by_income_bin_lines(
        ALGO_RUNS,
        episode_range=(1300, 1340),
        travel_time_col="travel_time",
        travel_time_unit="minutes",   # or "seconds"
        avg_mode="row",               # or "agent"
        plot_kind="AV",
        high_income_threshold=6000,
        n_equal_bins=4,
        show_std_band=True,
        save_path="imgs/avg_travel_time_by_income_bin_lines.png",
        show=True,
    )"""

    """plot_avg_vot_by_income_bin_lines(
        ALGO_RUNS,
        episode_range=(2300, 2340),
        plot_kind="AV",
        high_income_threshold=6000,
        n_equal_bins=4,
        work_hours_per_month=160,   # adjust if needed
        avg_mode="row",
        show_std_band=True,
        save_path="imgs/avg_vot_by_income_bin_lines.png",
        show=True,
    )"""

    """plot_avg_vot_times_travel_time_by_income_bin_lines(
        ALGO_RUNS,
        episode_range=(2300, 2340),
        travel_time_col="travel_time",
        travel_time_unit="minutes",   # change if needed
        plot_kind="AV",
        high_income_threshold=6000,
        n_equal_bins=4,
        work_hours_per_month=160,
        avg_mode="row",
        show_std_band=True,
        save_path="imgs/avg_vot_x_travel_time_by_income_bin_lines.png",
        show=True,
    )"""

    """plot_tt_over_best_observed_route_by_income_bin_lines(
        ALGO_RUNS,
        episode_range=(1300, 1340),
        travel_time_col="travel_time",
        observations_col="observation",
        plot_kind="AV",
        high_income_threshold=6000,
        n_equal_bins=4,
        obs_tt_scale=100.0,   # because obs[:3] are normalized by 100
        avg_mode="row",
        show_std_band=True,
        save_path="imgs/avg_tt_over_best_observed_route_by_income_bin_lines.png",
        show=True,
    )"""
    plot_tt_minus_best_observed_route_by_income_bin_lines(
        ALGO_RUNS,
        episode_range=(1300, 1300),
        travel_time_col="travel_time",
        observations_col="observation", 
        plot_kind="AV",
        high_income_threshold=4700,
        n_equal_bins=4,
        obs_tt_scale=100.0,
        travel_time_unit="minutes",
        avg_mode="row",
        show_std_band=True,
        smooth_window=1,
        save_path="imgs/avg_tt_minus_best_observed_route_by_income_bin_lines.png",
        show=True,
    )

    plot_tt_minus_start_time_by_income_bin_lines(
        ALGO_RUNS,
        episode_range=(1300, 1320),
        travel_time_col="travel_time",
        start_time_col="start_time",   # <-- change this if your column name differs
        plot_kind="AV",
        high_income_threshold=4700,
        n_equal_bins=4,
        travel_time_unit="minutes",
        start_time_unit="minutes",     # or "seconds" / "steps"
        step_to_minutes=1.0,           # only used if start_time_unit="steps"
        avg_mode="row",
        show_std_band=True,
        smooth_window=1,
        save_path="imgs/avg_tt_minus_start_time_by_income_bin_lines.png",
        show=True,
    )

