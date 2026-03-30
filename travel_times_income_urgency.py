#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# TWO-PANEL FIGURE:
#   (LEFT)  Urgency (x) vs Avg travel time (y)
#   (RIGHT) Income-bin (x) vs Avg travel time (y) + N individuals per bin
# Legend appears ONLY on the LEFT plot.
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_EP_RE = re.compile(r"^ep(\d+)\.csv$")


# ============================================================
# ----------------------- HELPERS ----------------------------
# ============================================================

def _episodes_dir(p: str) -> str:
    """Allow passing either base_dir or base_dir/episodes."""
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


def load_episodes(
    folder: str,
    n_last: int = 30,
    episode_range=None,   # tuple (start, end) inclusive
    episodes=None,        # explicit list/iterable of episode numbers
):
    """
    Load episodes from folder (or folder/episodes).

    Selection priority:
      1) episodes=[...]
      2) episode_range=(start, end) inclusive
      3) last n_last episodes
    """
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


# ============================================================
# ----------- INCOME-BIN HELPERS (from episodes) -------------
# ============================================================

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


# ============================================================
# ------------------- PLOT: URGENCY LINES --------------------
# ============================================================

def plot_avg_travel_time_by_urgency_lines(
    algo_runs,                      # dict scenario -> list of run dirs
    episode_range=None,
    episodes=None,
    last_n=30,
    travel_time_col="travel_time",
    urgency_col="urgency",
    kind_col="kind",
    plot_kind="AV",                 # "AV", "Human", "All"
    travel_time_unit="minutes",     # "minutes" or "seconds"
    avg_mode="row",                 # "row" or "agent"
    id_col="id",                    # used only if avg_mode="agent"
    urgency_round=None,             # None or e.g. 1, 5, 10
    min_n_per_urgency=1,
    show_std_band=True,
    show_percentile_band=False,
    lower_percentile=25,
    upper_percentile=75,
    smooth_window=None,             # rolling smoothing window (int) or None
    smooth_center=True,
    ax=None,
    figsize=(10.5, 5.2),
    title=None,
    show_legend=True,
    save_path=None,
    show=True,
):
    scenarios = list(algo_runs.keys())

    def to_minutes(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if travel_time_unit.lower().startswith("sec"):
            return s / 60.0
        return s

    def round_urg(u: pd.Series) -> pd.Series:
        u = pd.to_numeric(u, errors="coerce")
        if urgency_round is None:
            return u
        step = float(urgency_round)
        return np.round(u / step) * step

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    for sc in scenarios:
        per_run_series = []

        for run_dir in algo_runs[sc]:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [urgency_col, travel_time_col]
            if avg_mode == "agent":
                needed.append(id_col)

            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[needed].copy()
            tmp[urgency_col] = round_urg(tmp[urgency_col])
            tmp[travel_time_col] = to_minutes(tmp[travel_time_col])
            tmp = tmp.dropna(subset=[urgency_col, travel_time_col])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
                tmp = tmp.dropna(subset=[id_col])
                if tmp.empty:
                    continue
                tmp[id_col] = tmp[id_col].astype(int)

                per_agent = (
                    tmp.groupby([id_col, urgency_col], observed=True)[travel_time_col]
                       .mean()
                       .reset_index()
                )
                g = per_agent.groupby(urgency_col, observed=True)[travel_time_col].mean()
                n = per_agent.groupby(urgency_col, observed=True)[id_col].nunique()
            else:
                g = tmp.groupby(urgency_col, observed=True)[travel_time_col].mean()
                n = tmp.groupby(urgency_col, observed=True)[travel_time_col].size()

            if min_n_per_urgency > 1:
                g = g[n >= min_n_per_urgency]

            if g.empty:
                continue

            per_run_series.append(g)

        if not per_run_series:
            print(f"[WARN] Scenario '{sc}' had no usable runs. Skipping line.")
            continue

        all_u = sorted(set(np.concatenate([s.index.to_numpy(dtype=float) for s in per_run_series])))
        mat = np.full((len(per_run_series), len(all_u)), np.nan, dtype=float)
        for i, s in enumerate(per_run_series):
            mat[i, :] = s.reindex(all_u).to_numpy(dtype=float)

        all_u = np.asarray(all_u, dtype=float)
        mean_vec = np.nanmean(mat, axis=0)

        std_vec = None
        if mat.shape[0] > 1:
            std_vec = np.nanstd(mat, axis=0, ddof=1)

        lower_vec = np.nanpercentile(mat, lower_percentile, axis=0)
        upper_vec = np.nanpercentile(mat, upper_percentile, axis=0)

        order = np.argsort(all_u)
        all_u = all_u[order]
        mean_vec = mean_vec[order]
        lower_vec = lower_vec[order]
        upper_vec = upper_vec[order]
        if std_vec is not None:
            std_vec = std_vec[order]

        if smooth_window is not None and smooth_window > 1:
            mean_vec = (
                pd.Series(mean_vec)
                .rolling(window=smooth_window, center=smooth_center, min_periods=1)
                .mean()
                .to_numpy()
            )
            lower_vec = (
                pd.Series(lower_vec)
                .rolling(window=smooth_window, center=smooth_center, min_periods=1)
                .mean()
                .to_numpy()
            )
            upper_vec = (
                pd.Series(upper_vec)
                .rolling(window=smooth_window, center=smooth_center, min_periods=1)
                .mean()
                .to_numpy()
            )
            if std_vec is not None:
                std_vec = (
                    pd.Series(std_vec)
                    .rolling(window=smooth_window, center=smooth_center, min_periods=1)
                    .mean()
                    .to_numpy()
                )

        ax.plot(all_u, mean_vec, marker="o", markersize=3, linewidth=2, label=f"{sc}")

        if show_std_band and std_vec is not None:
            ax.fill_between(all_u, mean_vec - std_vec, mean_vec + std_vec, alpha=0.15)

        if show_percentile_band:
            ax.fill_between(all_u, lower_vec, upper_vec, alpha=0.18)

    if title is not None:
        ax.set_title(title)

    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("Urgency" + (f" (rounded to {urgency_round})" if urgency_round is not None else ""), fontsize=16)
    ax.set_ylabel("Avg travel time (minutes)", fontsize=16)
    ax.set_ylim(27.2, 30.5)
    ax.grid(True, axis="y", alpha=0.25)
    if show_legend:
        ax.legend(frameon=False, fontsize=12)

    if created_fig:
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    return fig


# ============================================================
# ----------------- PLOT: INCOME-BIN LINES -------------------
# ============================================================

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
    travel_time_unit="minutes",
    avg_mode="row",               # "row" or "agent"
    show_std_band=True,
    smooth_lines=True,
    smooth_window=3,
    # ---- NEW: show N individuals in xtick labels ----
    show_bin_n=True,
    bin_n_mode="overall_unique",  # "overall_unique" or "mean_per_run"
    ax=None,
    figsize=(8, 5.2),
    title=None,
    show_legend=True,
    save_path=None,
    show=True,
):
    scenarios = list(algo_runs.keys())

    def assign_income_group(income_series: pd.Series, bins_edges, bin_labels) -> pd.Series:
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

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # global bins across all runs for consistency
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

    # We'll compute "N individuals per bin" across all runs+scenarios
    # in one of two modes:
    # - overall_unique: count unique IDs pooled across ALL selected episodes across ALL runs (big pooled dataset)
    # - mean_per_run: compute N unique IDs per run, then average across runs (more comparable if IDs reset per run)
    pooled_tmp_for_counts = []   # used by overall_unique
    per_run_n_vecs = []          # used by mean_per_run

    for sc in scenarios:
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

            tmp = df[[id_col, income_col, travel_time_col]].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[travel_time_col] = to_minutes(tmp[travel_time_col])
            tmp = tmp.dropna(subset=[id_col, income_col, travel_time_col])
            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)
            tmp["income_bin"] = assign_income_group(tmp[income_col], bins_edges, bin_labels)
            tmp = tmp.dropna(subset=["income_bin"])
            if tmp.empty:
                continue

            # ---- mean travel time per bin (your existing logic) ----
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

            # ---- counts per bin (individuals) ----
            n_series = tmp.groupby("income_bin", observed=True)[id_col].nunique()
            n_vec = n_series.reindex(bin_labels).to_numpy(dtype=float)

            if show_bin_n:
                if bin_n_mode == "mean_per_run":
                    per_run_n_vecs.append(n_vec)
                else:
                    # overall_unique mode: pool data (but avoid double counting within the same run by dropping duplicates)
                    pooled_tmp_for_counts.append(tmp[[id_col, "income_bin"]].drop_duplicates())

        if not per_run_vecs:
            print(f"[WARN] Scenario '{sc}' had no usable runs. Skipping line.")
            continue

        arr = np.vstack(per_run_vecs)  # (n_runs, n_bins)
        mean_vec = np.nanmean(arr, axis=0)

        std_vec = None
        if arr.shape[0] > 1:
            std_vec = np.nanstd(arr, axis=0, ddof=1)

        mean_plot = mean_vec.copy()
        std_plot = std_vec.copy() if std_vec is not None else None

        if smooth_lines:
            mean_plot = smooth_1d(mean_plot, window=smooth_window)
            if std_plot is not None:
                std_plot = smooth_1d(std_plot, window=smooth_window)

        ax.plot(x, mean_plot, marker="o", linewidth=2, label=f"{sc}")

        if show_std_band and std_plot is not None:
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)

    if title is not None:
        ax.set_title(title)

    # ---- Build xtick labels with n individuals per bin ----
    tick_labels = list(bin_labels)
    if show_bin_n:
        n_display = np.full(n_bins, np.nan, dtype=float)

        if bin_n_mode == "mean_per_run" and per_run_n_vecs:
            n_mat = np.vstack(per_run_n_vecs)  # (n_runs_total, n_bins)
            n_display = np.nanmean(n_mat, axis=0)
            tick_labels = [
                f"{lab}\n(n≈{int(round(n))})" if np.isfinite(n) else f"{lab}\n(n=0)"
                for lab, n in zip(bin_labels, n_display)
            ]
        else:
            # overall_unique
            if pooled_tmp_for_counts:
                pooled = pd.concat(pooled_tmp_for_counts, ignore_index=True)
                pooled = pooled.drop_duplicates(subset=[id_col, "income_bin"])
                n_series = pooled.groupby("income_bin", observed=True)[id_col].nunique()
                n_display = n_series.reindex(bin_labels).to_numpy(dtype=float)
                tick_labels = [
                    f"{lab}\n(n={int(n)})" if np.isfinite(n) else f"{lab}\n(n=0)"
                    for lab, n in zip(bin_labels, n_display)
                ]

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Income bin", fontsize=16)
    #ax.set_ylabel("Avg travel time (minutes)", fontsize=16)
    ax.set_ylim(27.2, 30.5)
    ax.grid(True, axis="y", alpha=0.25)
    if show_legend:
        ax.legend(frameon=False, fontsize=12)

    if created_fig:
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    return fig


# ============================================================
# ------------------------- MAIN -----------------------------
# ============================================================

if __name__ == "__main__":

    ALGO_RUNS = {
        "monetary pricing": [
            r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_48_no_urgency_smaller_training/episodes",
        ],
        "karma pricing": [
            r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_15_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_16_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_17_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4/episodes",
        ],
             "karma fee 5 longer": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_15_karma_pricing_minimum_price_5_longer/episodes",

    ],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    # LEFT: urgency (legend ON)
    plot_avg_travel_time_by_urgency_lines(
        ALGO_RUNS,
        episode_range=(2100, 2320),
        travel_time_col="travel_time",
        urgency_col="urgency",
        travel_time_unit="minutes",
        avg_mode="row",
        plot_kind="AV",
        urgency_round=None,
        min_n_per_urgency=1,
        show_std_band=True,
        show_percentile_band=False,
        smooth_window=3,
        smooth_center=True,
        ax=ax1,
        title="a)",
        show_legend=True,
        show=False,
    )

    # RIGHT: income bins + N (legend OFF)
    plot_avg_travel_time_by_income_bin_lines(
        ALGO_RUNS,
        episode_range=(1200, 1320),
        travel_time_col="travel_time",
        travel_time_unit="minutes",
        avg_mode="row",
        plot_kind="AV",
        high_income_threshold=6000,
        n_equal_bins=4,
        show_std_band=True,
        smooth_lines=True,
        smooth_window=3,
        show_bin_n=True,              # <-- ON
        bin_n_mode="mean_per_run",    # "mean_per_run" is usually safest across independent runs
        ax=ax2,
        title="b)",
        show_legend=False,
        show=False,
    )

    fig.tight_layout()
    os.makedirs("imgs", exist_ok=True)
    fig.savefig("imgs/two_panel_travel_time.png", dpi=200, bbox_inches="tight")
    plt.show()