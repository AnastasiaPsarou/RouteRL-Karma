import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# SHARED HELPERS
# ==========================================================
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


def safe_nanmean_1d_over_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape[1], np.nan, dtype=float)
    counts = np.sum(np.isfinite(arr), axis=0)
    mask = counts > 0
    if np.any(mask):
        out[mask] = np.nansum(arr[:, mask], axis=0) / counts[mask]
    return out


def safe_nanstd_1d_over_rows(arr: np.ndarray, ddof: int = 1) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape[1], np.nan, dtype=float)
    counts = np.sum(np.isfinite(arr), axis=0)
    mask = counts > ddof
    if not np.any(mask):
        return out

    sub = arr[:, mask]
    means = np.nansum(sub, axis=0) / counts[mask]
    centered = np.where(np.isfinite(sub), sub - means, 0.0)
    ss = np.sum(centered ** 2, axis=0)
    out[mask] = np.sqrt(ss / (counts[mask] - ddof))
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
    show_hourly_labels=False,          # NEW
    work_hours_per_month=160.0,        # NEW
):
    pooled_low = []
    pooled_high = []

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
        high = incomes[incomes > high_income_threshold]

        if low.size:
            pooled_low.append(low)
        if high.size:
            pooled_high.append(high)

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

    def convert_for_label(x):
        if show_hourly_labels:
            return x / float(work_hours_per_month)
        return x

    def fmt(x):
        x = convert_for_label(x)
        return f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"

    low_labels = [f"{fmt(edges[i])}–{fmt(edges[i+1])}" for i in range(len(edges) - 1)]

    if pooled_high:
        pooled_high = np.concatenate(pooled_high)
        max_high = np.max(pooled_high)
        high_label = f"{fmt(high_income_threshold)}–{fmt(max_high)}"
    else:
        high_label = f">{fmt(high_income_threshold)}"

    bins = np.concatenate([edges, [np.inf]])
    labels = low_labels + [high_label]
    return bins, labels

def scenario_color(name: str) -> str:
    s = name.lower()
    if "monetary" in s:
        return "forestgreen"
    if "karma" in s:
        return "rebeccapurple"
    return "tab:blue"


# ==========================================================
# COMPUTE SERIES FOR LEFT PANEL: INCOME BIN -> AVG TT
# ==========================================================
def compute_income_bin_curves(
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
    avg_mode="row",
    show_hourly_income_labels=False,
    work_hours_per_month=160.0,
):
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
        show_hourly_labels=show_hourly_income_labels,
        work_hours_per_month=work_hours_per_month,
    )

    def assign_income_group(income_series: pd.Series) -> pd.Series:
        out = pd.Series(pd.NA, index=income_series.index, dtype="object")
        hi = income_series > high_income_threshold
        out.loc[hi] = bin_labels[-1]   # use the real last-bin label
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

    results = {}
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
                n_series = per_agent.groupby("income_bin", observed=True)[id_col].nunique()
            else:
                series = tmp.groupby("income_bin", observed=True)[travel_time_col].mean()
                n_series = tmp.groupby("income_bin", observed=True)[id_col].nunique()

            vec = series.reindex(bin_labels).to_numpy(dtype=float)
            n_vec = n_series.reindex(bin_labels).to_numpy(dtype=float)

            per_run_vecs.append(vec)
            per_run_n.append(n_vec)

        if not per_run_vecs:
            print(f"[WARN] Scenario '{sc}' had no usable runs for income-bin panel.")
            continue

        arr = np.vstack(per_run_vecs)
        n_arr = np.vstack(per_run_n)
        pooled_n_across_scenarios.append(n_arr)

        mean_vec = safe_nanmean_1d_over_rows(arr)
        std_vec = safe_nanstd_1d_over_rows(arr, ddof=1) if arr.shape[0] > 1 else None

        results[sc] = {
            "x": np.arange(len(bin_labels)),
            "labels": bin_labels,
            "mean": mean_vec,
            "std": std_vec,
            "n_arr": n_arr,
        }

    pooled_n = None
    if pooled_n_across_scenarios:
        scen_means = [safe_nanmean_1d_over_rows(n_arr) for n_arr in pooled_n_across_scenarios]
        pooled_n = safe_nanmean_1d_over_rows(np.vstack(scen_means))

    return results, bin_labels, pooled_n


# ==========================================================
# COMPUTE SERIES FOR RIGHT PANEL: URGENCY -> AVG TT
# ==========================================================
def compute_urgency_curves(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    travel_time_col="travel_time",
    urgency_col="urgency",
    kind_col="kind",
    plot_kind="AV",
    travel_time_unit="minutes",
    avg_mode="row",
    id_col="id",
    urgency_round=None,
    min_n_per_urgency=1,
    lower_percentile=25,
    upper_percentile=75,
):
    scenarios = list(algo_runs.keys())
    results = {}

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
            print(f"[WARN] Scenario '{sc}' had no usable runs for urgency panel.")
            continue

        all_u = sorted(set(np.concatenate([s.index.to_numpy(dtype=float) for s in per_run_series])))
        mat = np.full((len(per_run_series), len(all_u)), np.nan, dtype=float)

        for i, s in enumerate(per_run_series):
            mat[i, :] = s.reindex(all_u).to_numpy(dtype=float)

        mean_vec = safe_nanmean_1d_over_rows(mat)
        std_vec = safe_nanstd_1d_over_rows(mat, ddof=1) if mat.shape[0] > 1 else None

        lower_vec = np.full(mat.shape[1], np.nan, dtype=float)
        upper_vec = np.full(mat.shape[1], np.nan, dtype=float)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            col = col[np.isfinite(col)]
            if col.size > 0:
                lower_vec[j] = np.percentile(col, lower_percentile)
                upper_vec[j] = np.percentile(col, upper_percentile)

        results[sc] = {
            "x": np.asarray(all_u, dtype=float),
            "mean": mean_vec,
            "std": std_vec,
            "lower": lower_vec,
            "upper": upper_vec,
        }

    return results


# ==========================================================
# COMBINED SIDE-BY-SIDE PLOT
# ==========================================================
def plot_income_bin_and_urgency_side_by_side(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,

    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    urgency_col="urgency",
    kind_col="kind",
    plot_kind="AV",

    high_income_threshold=6000,
    n_equal_bins=4,
    income_avg_mode="row",
    show_income_std_band=True,
    show_bin_n=True,

    urgency_avg_mode="row",
    urgency_round=None,
    min_n_per_urgency=1,
    show_urgency_std_band=True,
    show_urgency_percentile_band=False,
    lower_percentile=25,
    upper_percentile=75,

    travel_time_unit="minutes",
    figsize=(14, 5.5),
    save_path=None,
    show=True,

    smooth_income_lines=True,
    smooth_income_window=3,
    smooth_urgency_lines=True,
    smooth_urgency_window=3,

    show_hourly_income_labels=True,    # NEW
    work_hours_per_month=160.0,        # NEW
):
    income_results, bin_labels, pooled_bin_n = compute_income_bin_curves(
        algo_runs=algo_runs,
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        id_col=id_col,
        income_col=income_col,
        travel_time_col=travel_time_col,
        kind_col=kind_col,
        plot_kind=plot_kind,
        high_income_threshold=high_income_threshold,
        n_equal_bins=n_equal_bins,
        travel_time_unit=travel_time_unit,
        avg_mode=income_avg_mode,
        show_hourly_income_labels=show_hourly_income_labels,   # NEW
        work_hours_per_month=work_hours_per_month,             # NEW
    )

    urgency_results = compute_urgency_curves(
        algo_runs=algo_runs,
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        travel_time_col=travel_time_col,
        urgency_col=urgency_col,
        kind_col=kind_col,
        plot_kind=plot_kind,
        travel_time_unit=travel_time_unit,
        avg_mode=urgency_avg_mode,
        id_col=id_col,
        urgency_round=urgency_round,
        min_n_per_urgency=min_n_per_urgency,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # ---------------- LEFT: income bin ----------------
    for sc, res in income_results.items():
        color = scenario_color(sc)
        x = res["x"]
        mean_plot = res["mean"].copy()
        std_plot = res["std"].copy() if res["std"] is not None else None

        if smooth_income_lines:
            mean_plot = smooth_1d(mean_plot, window=smooth_income_window)
            if std_plot is not None:
                std_plot = smooth_1d(std_plot, window=smooth_income_window)

        ax_left.plot(x, mean_plot, marker="o", linewidth=2.5, color=color, label=sc)

        if show_income_std_band and std_plot is not None:
            ax_left.fill_between(
                x,
                mean_plot - std_plot,
                mean_plot + std_plot,
                color=color,
                alpha=0.15,
            )

    ax_left.set_xticks(np.arange(len(bin_labels)))
    if show_bin_n and pooled_bin_n is not None:
        tick_labels = [
            f"{lab}"
            for lab, n in zip(bin_labels, pooled_bin_n)
        ]
        ax_left.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=11)
    else:
        ax_left.set_xticklabels(bin_labels, rotation=0, ha="center", fontsize=11)

    ax_left.tick_params(axis="y", labelsize=12)
    ax_left.set_xlabel("Hourly income bin", fontsize=14, labelpad=10)
    ax_left.set_ylabel("Average travel time (minutes)", fontsize=14)
    ax_left.set_title("", fontsize=15)
    ax_left.grid(True, axis="y", alpha=0.25)
    ax_left.legend(loc="upper left",frameon=False, fontsize=11)

    # ---------------- RIGHT: urgency ----------------
    for sc, res in urgency_results.items():
        color = scenario_color(sc)
        x = res["x"].copy()
        mean_plot = res["mean"].copy()
        std_plot = res["std"].copy() if res["std"] is not None else None
        lower_plot = res["lower"].copy()
        upper_plot = res["upper"].copy()

        order = np.argsort(x)
        x = x[order]
        mean_plot = mean_plot[order]
        lower_plot = lower_plot[order]
        upper_plot = upper_plot[order]
        if std_plot is not None:
            std_plot = std_plot[order]

        if smooth_urgency_lines and smooth_urgency_window is not None and smooth_urgency_window > 1:
            mean_plot = (
                pd.Series(mean_plot)
                .rolling(window=smooth_urgency_window, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
            lower_plot = (
                pd.Series(lower_plot)
                .rolling(window=smooth_urgency_window, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
            upper_plot = (
                pd.Series(upper_plot)
                .rolling(window=smooth_urgency_window, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
            if std_plot is not None:
                std_plot = (
                    pd.Series(std_plot)
                    .rolling(window=smooth_urgency_window, center=True, min_periods=1)
                    .mean()
                    .to_numpy()
                )

        ax_right.plot(
            x,
            mean_plot,
            marker="o",
            markersize=3,
            linewidth=2.5,
            color=color,
            label=sc,
        )

        if show_urgency_std_band and std_plot is not None:
            ax_right.fill_between(
                x,
                mean_plot - std_plot,
                mean_plot + std_plot,
                color=color,
                alpha=0.15,
            )

        if show_urgency_percentile_band:
            ax_right.fill_between(
                x,
                lower_plot,
                upper_plot,
                color=color,
                alpha=0.10,
            )

    ax_right.tick_params(axis="both", labelsize=12)
    ax_right.set_xlabel("Urgency", fontsize=14, labelpad=10)
    ax_right.set_ylabel("", fontsize=14)
    ax_right.set_title("", fontsize=15)
    ax_right.grid(True, axis="y", alpha=0.25)
    ax_left.tick_params(axis="x", pad=6)
    ax_right.tick_params(axis="x", pad=6)

    ax_left.xaxis.set_label_coords(0.5, -0.14)
    ax_right.xaxis.set_label_coords(0.5, -0.14)
    #ax_right.legend(frameon=False, fontsize=11)

    fig.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ==========================================================
# EXAMPLE USAGE
# ==========================================================
if __name__ == "__main__":
    ALGO_RUNS = {
        "monetary pricing": [
            r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        ],
        "karma pricing": [
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

    plot_income_bin_and_urgency_side_by_side(
        ALGO_RUNS,
        episode_range=(1200, 1320),
        travel_time_col="travel_time",
        urgency_col="urgency",
        travel_time_unit="minutes",
        plot_kind="AV",

        high_income_threshold=4700,
        n_equal_bins=4,
        income_avg_mode="row",
        show_income_std_band=True,
        show_bin_n=True,

        urgency_avg_mode="row",
        urgency_round=None,
        min_n_per_urgency=1,
        show_urgency_std_band=True,
        show_urgency_percentile_band=False,

        figsize=(12, 4),
        save_path="imgs/income_and_urgency_tt_side_by_side.png",
        show=True,

        smooth_income_lines=False,
        smooth_income_window=3,
        smooth_urgency_lines=True,
        smooth_urgency_window=3,

        show_hourly_income_labels=True,
        work_hours_per_month=160.0,
    )