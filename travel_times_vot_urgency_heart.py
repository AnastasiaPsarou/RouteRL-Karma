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


def unpack_scenario_cfg(algo_runs, scenario, episode_range=None, episodes=None, last_n=30):
    """
    Supports both:
      old format:
        "scenario": [run1, run2, ...]
      new format:
        "scenario": {
            "runs": [...],
            "episode_range": (...),
            "episodes": [...],
            "last_n": ...
        }
    """
    sc_cfg = algo_runs[scenario]

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

    return run_dirs, sc_episode_range, sc_episodes, sc_last_n


def scenario_color(name: str) -> str:
    s = name.lower()
    if "monetary" in s:
        return "forestgreen"
    if "karma" in s:
        return "rebeccapurple"
    return "tab:blue"


# ==========================================================
# VOT BINNING
# ==========================================================
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

    VoT = (monthly_income / work_hours_per_month) * urgency
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

    tmp["vot"] = (tmp[income_col] / float(work_hours_per_month)) * tmp[urgency_col]

    per_agent_day = (
        tmp.groupby([episode_col, id_col], observed=True)["vot"]
        .mean()
        .reset_index()
    )

    def _bin_one_episode(s: pd.Series) -> pd.Series:
        if s.notna().sum() == 0:
            return pd.Series(pd.NA, index=s.index, dtype="object")

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

    key_to_bin = per_agent_day.set_index([episode_col, id_col])["vot_bin"]

    out_keys = pd.MultiIndex.from_arrays(
        [
            pd.to_numeric(df[episode_col], errors="coerce"),
            pd.to_numeric(df[id_col], errors="coerce"),
        ]
    )
    out = pd.Series(out_keys.map(key_to_bin), index=df.index, dtype="object")
    return out


# ==========================================================
# LEFT PANEL: VOT QUANTILE -> AVG TT
# ==========================================================
def compute_vot_quantile_curves(
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
    work_hours_per_month=240.0,
    n_vot_bins=5,
    travel_time_unit="minutes",
    avg_mode="row",
):
    scenarios = list(algo_runs.keys())
    vot_labels = [f"Q{i}" for i in range(1, n_vot_bins + 1)]
    results = {}
    pooled_n_across_scenarios = []

    def to_minutes(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if travel_time_unit.lower().startswith("sec"):
            return s / 60.0
        return s

    for sc in scenarios:
        run_dirs, sc_episode_range, sc_episodes, sc_last_n = unpack_scenario_cfg(
            algo_runs, sc, episode_range=episode_range, episodes=episodes, last_n=last_n
        )

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
            print(f"[WARN] Scenario '{sc}' had no usable runs for VoT panel.")
            continue

        arr = np.vstack(per_run_vecs)
        n_arr = np.vstack(per_run_n)
        pooled_n_across_scenarios.append(n_arr)

        mean_vec = safe_nanmean_1d_over_rows(arr)
        std_vec = safe_nanstd_1d_over_rows(arr, ddof=1) if arr.shape[0] > 1 else None

        results[sc] = {
            "x": np.arange(len(vot_labels)),
            "labels": vot_labels,
            "mean": mean_vec,
            "std": std_vec,
            "n_arr": n_arr,
        }

    pooled_n = None
    if pooled_n_across_scenarios:
        scen_means = [safe_nanmean_1d_over_rows(n_arr) for n_arr in pooled_n_across_scenarios]
        pooled_n = safe_nanmean_1d_over_rows(np.vstack(scen_means))

    return results, vot_labels, pooled_n


# ==========================================================
# RIGHT PANEL: URGENCY -> AVG TT
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
        run_dirs, sc_episode_range, sc_episodes, sc_last_n = unpack_scenario_cfg(
            algo_runs, sc, episode_range=episode_range, episodes=episodes, last_n=last_n
        )

        per_run_series = []

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



def set_paper_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2.2,
        "lines.markersize": 5.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def scenario_style(name: str) -> dict:
    s = name.lower()

    if "monetary" in s:
        return {
            "color": "#1b9e77",
            "marker": "o",
            "linestyle": "-",
        }
    if "karma" in s:
        return {
            "color": "#7570b3",
            "marker": "s",
            "linestyle": "-",
        }

    return {
        "color": "#4d4d4d",
        "marker": "^",
        "linestyle": "-",
    }


# ==========================================================
# COMBINED SIDE-BY-SIDE PLOT
# ==========================================================
def plot_vot_quantile_and_urgency_side_by_side(
    algo_runs_left,
    algo_runs_right,
    episode_range_left=None,
    episodes_left=None,
    last_n_left=30,
    episode_range_right=None,
    episodes_right=None,
    last_n_right=30,

    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    urgency_col="urgency",
    kind_col="kind",
    plot_kind_left="AV",
    plot_kind_right="AV",

    work_hours_per_month=240.0,
    n_vot_bins=5,
    vot_avg_mode="row",
    show_vot_std_band=True,
    show_vot_bin_n=False,

    urgency_avg_mode="row",
    urgency_round=None,
    min_n_per_urgency=1,
    show_urgency_std_band=True,
    show_urgency_percentile_band=False,
    lower_percentile=25,
    upper_percentile=75,

    travel_time_unit="minutes",
    figsize=(11, 4.2),
    save_path=None,
    show=True,

    smooth_vot_lines=False,
    smooth_vot_window=3,
    smooth_urgency_lines=True,
    smooth_urgency_window=3,

    panel_labels=True,
    left_title=None,
    right_title=None,
):
    set_paper_style()

    vot_results, vot_labels, pooled_vot_n = compute_vot_quantile_curves(
        algo_runs=algo_runs_left,
        episode_range=episode_range_left,
        episodes=episodes_left,
        last_n=last_n_left,
        id_col=id_col,
        income_col=income_col,
        urgency_col=urgency_col,
        travel_time_col=travel_time_col,
        kind_col=kind_col,
        plot_kind=plot_kind_left,
        work_hours_per_month=work_hours_per_month,
        n_vot_bins=n_vot_bins,
        travel_time_unit=travel_time_unit,
        avg_mode=vot_avg_mode,
    )

    urgency_results = compute_urgency_curves(
        algo_runs=algo_runs_right,
        episode_range=episode_range_right,
        episodes=episodes_right,
        last_n=last_n_right,
        travel_time_col=travel_time_col,
        urgency_col=urgency_col,
        kind_col=kind_col,
        plot_kind=plot_kind_right,
        travel_time_unit=travel_time_unit,
        avg_mode=urgency_avg_mode,
        id_col=id_col,
        urgency_round=urgency_round,
        min_n_per_urgency=min_n_per_urgency,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # ---------- LEFT: VoT quantiles ----------
    for sc, res in vot_results.items():
        sty = scenario_style(sc)
        x = res["x"]
        mean_plot = res["mean"].copy()
        std_plot = res["std"].copy() if res["std"] is not None else None

        if smooth_vot_lines:
            mean_plot = smooth_1d(mean_plot, window=smooth_vot_window)
            if std_plot is not None:
                std_plot = smooth_1d(std_plot, window=smooth_vot_window)

        ax_left.plot(
            x,
            mean_plot,
            color=sty["color"],
            marker=sty["marker"],
            linestyle=sty["linestyle"],
            linewidth=2.2,
            markersize=5,
            label=sc,
            zorder=3,
        )

        if show_vot_std_band and std_plot is not None:
            ax_left.fill_between(
                x,
                mean_plot - std_plot,
                mean_plot + std_plot,
                color=sty["color"],
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )

    ax_left.set_xticks(np.arange(len(vot_labels)))
    if show_vot_bin_n and pooled_vot_n is not None:
        tick_labels = [f"{lab}\n(n≈{int(round(n))})" for lab, n in zip(vot_labels, pooled_vot_n)]
        ax_left.set_xticklabels(tick_labels, rotation=0, ha="center")
    else:
        ax_left.set_xticklabels(vot_labels, rotation=0, ha="center")

    ax_left.set_xlabel("VoT quantile")
    ax_left.set_ylabel("Average travel time (minutes)")
    if left_title is not None:
        ax_left.set_title(left_title, pad=8)

    ax_left.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax_left.grid(False, axis="x")
    ax_left.legend(
        loc="upper left",
        frameon=False,
        handlelength=2.0,
        borderpad=0.2,
        labelspacing=0.35,
    )

    # ---------- RIGHT: Urgency ----------
    for sc, res in urgency_results.items():
        sty = scenario_style(sc)
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
            color=sty["color"],
            marker=sty["marker"],
            linestyle=sty["linestyle"],
            linewidth=2.2,
            markersize=4.2,
            label=sc,
            zorder=3,
        )

        if show_urgency_std_band and std_plot is not None:
            ax_right.fill_between(
                x,
                mean_plot - std_plot,
                mean_plot + std_plot,
                color=sty["color"],
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )

        if show_urgency_percentile_band:
            ax_right.fill_between(
                x,
                lower_plot,
                upper_plot,
                color=sty["color"],
                alpha=0.08,
                linewidth=0,
                zorder=1,
            )

    ax_right.set_xlabel("Urgency")
    if right_title is not None:
        ax_right.set_title(right_title, pad=8)

    ax_right.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax_right.grid(False, axis="x")

    # Optional: keep y-axis formatting consistent and clean
    for ax in (ax_left, ax_right):
        ax.margins(x=0.03)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    # panel labels
    if panel_labels:
        ax_left.text(0.12, 1.03, "a)", transform=ax_left.transAxes,
                     fontsize=12, va="bottom")
        ax_right.text(0.12, 1.03, "b)", transform=ax_right.transAxes,
                      fontsize=12, va="bottom")

    fig.tight_layout(pad=1.0, w_pad=1.6)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

        root, ext = os.path.splitext(save_path)
        if ext.lower() != ".pdf":
            pdf_path = root + ".pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"Saved: {pdf_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ==========================================================
# EXAMPLE USAGE
# ==========================================================
if __name__ == "__main__":


    ALGO_RUNS_LEFT = {
        "monetary pricing": {
            "runs": [
                r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_route1_fee_1_5/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_route1_fee_1_5/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_route1_fee_1_5/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_route1_fee_1_5/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_route1_fee_1_5/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_route1_fee_1_5/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_route1_fee_1_5/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_route1_fee_1_5/episodes",
            ],
            "episode_range": (1000, 1320),
        },

        "karma pricing": {
            "runs": [
                r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_norm_reward/episodes",],
            "episode_range": (1000, 1320),
        },
    }


    ALGO_RUNS_RIGHT = {
        "monetary pricing": {
            "runs": [
                r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_no_dist_cost/episodes",
                r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_no_dist_cost/episodes",
                r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_no_dist_cost/episodes",
                r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_no_dist_cost/episodes",
                r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_no_dist_cost/episodes",
                r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_no_dist_cost/episodes",
                r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_no_dist_cost/episodes",
                r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_no_dist_cost/episodes",
            ],
            "episode_range": (1200, 1320),
        },

        "karma pricing": {
            "runs": [
                 r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_norm_reward/episodes",
                r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_norm_reward/episodes",
            ],
            "episode_range": (1200, 1320),
        },
    }

    plot_vot_quantile_and_urgency_side_by_side(
    algo_runs_left=ALGO_RUNS_LEFT,
    algo_runs_right=ALGO_RUNS_RIGHT,

    travel_time_col="travel_time",
    urgency_col="urgency",
    travel_time_unit="minutes",

    plot_kind_left="AV",
    plot_kind_right="AV",

    work_hours_per_month=240.0,
    n_vot_bins=5,

    vot_avg_mode="row",
    urgency_avg_mode="row",

    show_vot_std_band=True,
    show_urgency_std_band=True,
    show_urgency_percentile_band=False,

    figsize=(10.5, 4.0),
    save_path="imgs/vot_left_urgency_right_different_data.pdf",
    show=True,

    smooth_vot_lines=False,
    smooth_urgency_lines=True,
    smooth_urgency_window=3,

    panel_labels=True,
    left_title=None,
    right_title=None,
)