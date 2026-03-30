import os
import re
import ast
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_EP_RE = re.compile(r"^ep(\d+)\.csv$")


# =========================================================
# BASIC IO
# =========================================================
def _episodes_dir(p: str) -> str:
    ep = os.path.join(p, "episodes")
    return ep if os.path.isdir(ep) else p


def _run_label_from_path(run_dir: str) -> str:
    m = re.search(r"seed[_-](\d+)", str(run_dir))
    if m:
        return f"seed {m.group(1)}"

    p = os.path.normpath(run_dir)
    base = os.path.basename(p)
    if base == "episodes":
        return os.path.basename(os.path.dirname(p))
    return base


def _ensure_parent_dir(path: str):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


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


# =========================================================
# PARSING OBSERVATIONS + ABS GAP
# =========================================================
def parse_observations_to_array(x):
    """
    Robustly parse an observations cell into a 1D numpy array.
    Supports:
      - Python-list strings like "[0.1, 0.2, 0.3]"
      - actual list / tuple / np.ndarray
      - loose numeric strings via regex fallback
    """
    if isinstance(x, np.ndarray):
        try:
            return x.astype(float).ravel()
        except Exception:
            return None

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

    try:
        val = ast.literal_eval(s)
        return np.asarray(val, dtype=float).ravel()
    except Exception:
        pass

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
    observations_col="observation",   # or "observations"
    obs_tt_scale=100.0,
    travel_time_unit="minutes",       # "minutes" or "seconds"
    output_col="tt_diff",
    use_absolute=True,
):
    """
    Computes:
        travel_time - min(first 3 entries of observations) * obs_tt_scale

    If use_absolute=True:
        |travel_time - best_observed_travel_time|

    Assumes first 3 observation values are normalized route travel times.
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
    diff = travel_time - best_route_tt

    if use_absolute:
        diff = diff.abs()

    df[output_col] = diff
    return df


# =========================================================
# GLOBAL INCOME QUANTILE BINS
# =========================================================
def build_global_income_quantile_bins(
    run_dirs,
    n_income_bins=5,
    episode_range=None,
    episodes=None,
    last_n=30,
    income_col="income",
):
    """
    Build one shared set of income quantile bins across ALL runs.
    These bins are computed once globally, not per seed.

    Returns:
        income_edges, income_labels
    """
    pooled = []

    for rd in run_dirs:
        df = load_episodes(rd, n_last=last_n, episode_range=episode_range, episodes=episodes)
        if df is None or df.empty or income_col not in df.columns:
            print(f"[WARN] Could not derive income bins from {rd}. Skipping run.")
            continue

        vals = pd.to_numeric(df[income_col], errors="coerce").dropna().to_numpy()
        if vals.size:
            pooled.append(vals)

    if not pooled:
        raise ValueError("No valid income values found across runs.")

    pooled = np.concatenate(pooled)

    qs = np.linspace(0, 1, n_income_bins + 1)
    edges = np.unique(np.quantile(pooled, qs))

    if len(edges) < 2:
        raise ValueError("Income quantile edges collapsed. Cannot build bins.")

    labels = [f"Q{i}" for i in range(1, len(edges))]
    return edges, labels


# =========================================================
# FIXED START-TIME BINS
# =========================================================
def _prepare_cut_edges(edges, right=False):
    """
    Slightly nudges the last finite edge upward when right=False,
    so a value exactly equal to the last edge is still included.
    Example:
      bins [0,20,40,60,80,100], right=False
      then start_time == 100 still falls into the last bin.
    """
    edges = np.asarray(edges, dtype=float).copy()
    if not right and np.isfinite(edges[-1]):
        edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def assign_fixed_bins(series: pd.Series, bin_edges, labels, right=False, include_lowest=True):
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="object")

    valid = s.notna()
    if valid.sum() == 0:
        return out

    cut_edges = _prepare_cut_edges(bin_edges, right=right)
    out.loc[valid] = pd.cut(
        s[valid],
        bins=cut_edges,
        labels=labels,
        right=right,
        include_lowest=include_lowest,
    ).astype(object)

    return out


def assign_income_bins(series: pd.Series, income_edges, income_labels):
    s = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        s,
        bins=income_edges,
        labels=income_labels,
        include_lowest=True,
        right=True,
    ).astype(object)


# =========================================================
# CORE PER-RUN MATRIX
# =========================================================
def build_run_matrix_abs_tt_gap(
    run_dir,
    start_time_bins,
    start_time_labels,
    income_edges,
    income_labels,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    observations_col="observation",   # or "observations"
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    obs_tt_scale=100.0,
    travel_time_unit="minutes",
    avg_mode="row",                   # "row" or "agent"
):
    """
    Returns a matrix of shape:
        (n_start_bins, n_income_bins)

    where each cell is:
        mean( |experienced TT - best observed TT| )
    """
    df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
    if df is None or df.empty:
        return None, "no data"

    if plot_kind != "All" and kind_col in df.columns:
        df = df[df[kind_col] == plot_kind]
    if df.empty:
        return None, "no matching rows"

    needed = [id_col, income_col, travel_time_col, observations_col, start_time_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[WARN] {run_dir}: missing {missing}.")
        return None, f"missing {missing}"

    tmp = df[[id_col, income_col, travel_time_col, observations_col, start_time_col]].copy()
    tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
    tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
    tmp[travel_time_col] = pd.to_numeric(tmp[travel_time_col], errors="coerce")
    tmp[start_time_col] = pd.to_numeric(tmp[start_time_col], errors="coerce")

    tmp = compute_tt_minus_best_observed_route(
        tmp,
        travel_time_col=travel_time_col,
        observations_col=observations_col,
        obs_tt_scale=obs_tt_scale,
        travel_time_unit=travel_time_unit,
        output_col="tt_diff",
        use_absolute=True,
    )

    tmp = tmp.dropna(subset=[id_col, income_col, start_time_col, "tt_diff"])
    if tmp.empty:
        return None, "no valid rows"

    tmp[id_col] = tmp[id_col].astype(int)

    # fixed start-time bins
    tmp["start_bin"] = assign_fixed_bins(
        tmp[start_time_col],
        bin_edges=start_time_bins,
        labels=start_time_labels,
        right=False,              # [0,20), [20,40), ...
        include_lowest=True,
    )
    tmp = tmp.dropna(subset=["start_bin"])
    if tmp.empty:
        return None, "no rows in start bins"

    # global income quantiles
    tmp["income_bin"] = assign_income_bins(tmp[income_col], income_edges, income_labels)
    tmp = tmp.dropna(subset=["income_bin"])
    if tmp.empty:
        return None, "no rows in income bins"

    if avg_mode == "agent":
        per_agent = (
            tmp.groupby([id_col, "start_bin", "income_bin"], observed=True)["tt_diff"]
               .mean()
               .reset_index()
        )
        series = (
            per_agent.groupby(["start_bin", "income_bin"], observed=True)["tt_diff"]
            .mean()
        )
    else:
        series = (
            tmp.groupby(["start_bin", "income_bin"], observed=True)["tt_diff"]
            .mean()
        )

    mat = np.full((len(start_time_labels), len(income_labels)), np.nan, dtype=float)
    for i_s, s_lab in enumerate(start_time_labels):
        for j_i, i_lab in enumerate(income_labels):
            key = (s_lab, i_lab)
            if key in series.index:
                mat[i_s, j_i] = float(series.loc[key])

    return mat, None


# =========================================================
# PLOT 1: AVERAGE ACROSS REPLICATIONS
# =========================================================
def plot_abs_tt_gap_avg_across_replications(
    algo_runs,
    start_time_bins,
    start_time_labels,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    observations_col="observation",   # or "observations"
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    n_income_bins=5,
    obs_tt_scale=100.0,
    travel_time_unit="minutes",
    avg_mode="row",                   # "row" or "agent"
    show_std=True,
    figsize_per_panel=(7.0, 5.0),
    save_path=None,
    show=True,
):
    """
    One subplot per scenario.
    Grouped bars:
      x = fixed start-time bins
      colors = global income quantiles
      y = mean absolute TT gap averaged across replications

    Error bars = std across replications if show_std=True.
    """
    scenarios = list(algo_runs.keys())
    n_panels = len(scenarios)

    all_run_dirs = [rd for runs in algo_runs.values() for rd in runs]
    income_edges, income_labels = build_global_income_quantile_bins(
        all_run_dirs,
        n_income_bins=n_income_bins,
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        income_col=income_col,
    )

    fig_w = figsize_per_panel[0] * n_panels
    fig_h = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), sharey=True)
    if n_panels == 1:
        axes = [axes]

    x = np.arange(len(start_time_labels))
    n_inc = len(income_labels)
    bar_width = 0.15 if n_inc == 5 else 0.8 / max(n_inc, 1)
    offsets = (np.arange(n_inc) - (n_inc - 1) / 2.0) * bar_width

    for ax, sc in zip(axes, scenarios):
        per_run_mats = []

        for run_dir in algo_runs[sc]:
            mat, reason = build_run_matrix_abs_tt_gap(
                run_dir=run_dir,
                start_time_bins=start_time_bins,
                start_time_labels=start_time_labels,
                income_edges=income_edges,
                income_labels=income_labels,
                episode_range=episode_range,
                episodes=episodes,
                last_n=last_n,
                id_col=id_col,
                income_col=income_col,
                travel_time_col=travel_time_col,
                observations_col=observations_col,
                start_time_col=start_time_col,
                kind_col=kind_col,
                plot_kind=plot_kind,
                obs_tt_scale=obs_tt_scale,
                travel_time_unit=travel_time_unit,
                avg_mode=avg_mode,
            )
            if mat is not None:
                per_run_mats.append(mat)
            else:
                print(f"[WARN] {sc} / {_run_label_from_path(run_dir)} skipped: {reason}")

        if not per_run_mats:
            ax.set_title(f"{sc} (no data)")
            ax.set_xticks(x)
            ax.set_xticklabels(start_time_labels, rotation=25, ha="right")
            ax.grid(True, axis="y", alpha=0.25)
            continue

        arr = np.stack(per_run_mats, axis=0)   # (n_runs, n_start_bins, n_income_bins)
        mean_mat = np.nanmean(arr, axis=0)

        if show_std and arr.shape[0] > 1:
            std_mat = np.nanstd(arr, axis=0, ddof=1)
        else:
            std_mat = None

        for j_inc, inc_lab in enumerate(income_labels):
            heights = mean_mat[:, j_inc]
            yerr = std_mat[:, j_inc] if std_mat is not None else None

            ax.bar(
                x + offsets[j_inc],
                heights,
                width=bar_width,
                yerr=yerr,
                capsize=3 if yerr is not None else 0,
                alpha=0.85,
                label=inc_lab,
            )

        ax.set_title(sc)
        ax.set_xticks(x)
        ax.set_xticklabels(start_time_labels, rotation=25, ha="right")
        ax.set_xlabel("Start-time bin")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="Income quantile", frameon=False)

    axes[0].set_ylabel("|experienced TT - best observed TT|")
    fig.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# =========================================================
# PLOT 2: EACH SEED SEPARATELY
# =========================================================
def plot_abs_tt_gap_per_seed(
    algo_runs,
    start_time_bins,
    start_time_labels,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    observations_col="observation",   # or "observations"
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    n_income_bins=5,
    obs_tt_scale=100.0,
    travel_time_unit="minutes",
    avg_mode="row",                   # "row" or "agent"
    ncols=4,
    figsize_per_subplot=(4.2, 3.6),
    save_dir=None,
    show=True,
):
    """
    One figure per scenario.
    Inside each figure:
      - one subplot per seed/run folder
      - grouped bars identical in style to the average plot
      - x = fixed start-time bins
      - colors = global income quantiles
      - y = mean absolute TT gap for that seed only
    """
    scenarios = list(algo_runs.keys())

    all_run_dirs = [rd for runs in algo_runs.values() for rd in runs]
    income_edges, income_labels = build_global_income_quantile_bins(
        all_run_dirs,
        n_income_bins=n_income_bins,
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        income_col=income_col,
    )

    x = np.arange(len(start_time_labels))
    n_inc = len(income_labels)
    bar_width = 0.15 if n_inc == 5 else 0.8 / max(n_inc, 1)
    offsets = (np.arange(n_inc) - (n_inc - 1) / 2.0) * bar_width

    figs = {}

    for sc in scenarios:
        run_dirs = algo_runs[sc]
        n_runs = len(run_dirs)

        ncols_eff = min(ncols, max(n_runs, 1))
        nrows_eff = int(math.ceil(n_runs / ncols_eff))

        fig, axes = plt.subplots(
            nrows_eff,
            ncols_eff,
            figsize=(figsize_per_subplot[0] * ncols_eff, figsize_per_subplot[1] * nrows_eff),
            sharey=True,
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for ax_idx, run_dir in enumerate(run_dirs):
            ax = axes_flat[ax_idx]
            run_label = _run_label_from_path(run_dir)

            mat, reason = build_run_matrix_abs_tt_gap(
                run_dir=run_dir,
                start_time_bins=start_time_bins,
                start_time_labels=start_time_labels,
                income_edges=income_edges,
                income_labels=income_labels,
                episode_range=episode_range,
                episodes=episodes,
                last_n=last_n,
                id_col=id_col,
                income_col=income_col,
                travel_time_col=travel_time_col,
                observations_col=observations_col,
                start_time_col=start_time_col,
                kind_col=kind_col,
                plot_kind=plot_kind,
                obs_tt_scale=obs_tt_scale,
                travel_time_unit=travel_time_unit,
                avg_mode=avg_mode,
            )

            if mat is None:
                ax.set_title(f"{run_label}\n({reason})")
                ax.grid(True, axis="y", alpha=0.25)
                continue

            for j_inc, inc_lab in enumerate(income_labels):
                ax.bar(
                    x + offsets[j_inc],
                    mat[:, j_inc],
                    width=bar_width,
                    alpha=0.85,
                    label=inc_lab,
                )

            ax.set_title(run_label)
            ax.set_xticks(x)
            ax.set_xticklabels(start_time_labels, rotation=25, ha="right")
            ax.set_xlabel("Start-time bin")
            ax.grid(True, axis="y", alpha=0.25)

            if ax_idx % ncols_eff == 0:
                ax.set_ylabel("|experienced TT - best observed TT|")

        # hide unused axes
        for j in range(n_runs, len(axes_flat)):
            axes_flat[j].axis("off")

        # legend
        handles = labels = None
        for ax in axes_flat:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break

        if handles:
            fig.legend(
                handles,
                labels,
                title="Income quantile",
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
            )
            fig.tight_layout(rect=(0, 0, 0.88, 1))
        else:
            fig.tight_layout()

        figs[sc] = fig

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_sc = re.sub(r"[^A-Za-z0-9._-]+", "_", sc)
            out_path = os.path.join(save_dir, f"{safe_sc}_per_seed_abs_tt_gap.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return figs


def plot_abs_tt_gap_compare_scenarios_by_start_bin(
    algo_runs,
    start_time_bins,
    start_time_labels,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    travel_time_col="travel_time",
    observations_col="observation",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    n_income_bins=5,
    obs_tt_scale=100.0,
    travel_time_unit="minutes",
    avg_mode="row",
    show_std=True,
    figsize_per_subplot=(4.8, 4.4),
    ylims_by_start_bin=None,   # <- NEW
    save_path=None,
    show=True,
):
    """
    5 subplots (one per fixed start-time bin).
    In each subplot:
      x = scenario
      bars/colors = income quantiles
      y = mean absolute TT gap averaged across replications
    """
    scenarios = list(algo_runs.keys())
    n_scenarios = len(scenarios)
    n_start_bins = len(start_time_labels)

    all_run_dirs = [rd for runs in algo_runs.values() for rd in runs]
    income_edges, income_labels = build_global_income_quantile_bins(
        all_run_dirs,
        n_income_bins=n_income_bins,
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        income_col=income_col,
    )
    n_inc = len(income_labels)

    # collect per-scenario mean/std matrices
    scenario_stats = {}

    for sc in scenarios:
        per_run_mats = []

        for run_dir in algo_runs[sc]:
            mat, reason = build_run_matrix_abs_tt_gap(
                run_dir=run_dir,
                start_time_bins=start_time_bins,
                start_time_labels=start_time_labels,
                income_edges=income_edges,
                income_labels=income_labels,
                episode_range=episode_range,
                episodes=episodes,
                last_n=last_n,
                id_col=id_col,
                income_col=income_col,
                travel_time_col=travel_time_col,
                observations_col=observations_col,
                start_time_col=start_time_col,
                kind_col=kind_col,
                plot_kind=plot_kind,
                obs_tt_scale=obs_tt_scale,
                travel_time_unit=travel_time_unit,
                avg_mode=avg_mode,
            )
            if mat is not None:
                per_run_mats.append(mat)
            else:
                print(f"[WARN] {sc} / {_run_label_from_path(run_dir)} skipped: {reason}")

        if not per_run_mats:
            scenario_stats[sc] = {
                "mean_mat": None,
                "std_mat": None,
                "n_runs": 0,
            }
            continue

        arr = np.stack(per_run_mats, axis=0)   # (n_runs, n_start_bins, n_income_bins)
        mean_mat = np.nanmean(arr, axis=0)

        if show_std and arr.shape[0] > 1:
            std_mat = np.nanstd(arr, axis=0, ddof=1)
        else:
            std_mat = None

        scenario_stats[sc] = {
            "mean_mat": mean_mat,
            "std_mat": std_mat,
            "n_runs": arr.shape[0],
        }

    fig_w = figsize_per_subplot[0] * n_start_bins
    fig_h = figsize_per_subplot[1]
    fig, axes = plt.subplots(1, n_start_bins, figsize=(fig_w, fig_h), sharey=False)
    if n_start_bins == 1:
        axes = [axes]

    x = np.arange(n_scenarios)
    bar_width = 0.15 if n_inc == 5 else 0.8 / max(n_inc, 1)
    offsets = (np.arange(n_inc) - (n_inc - 1) / 2.0) * bar_width

    for i_s, ax in enumerate(axes):
        for j_inc, inc_lab in enumerate(income_labels):
            heights = []
            yerr_vals = []
            any_err = False

            for sc in scenarios:
                stats = scenario_stats[sc]
                if stats["mean_mat"] is None:
                    heights.append(np.nan)
                    yerr_vals.append(0.0)
                    continue

                heights.append(float(stats["mean_mat"][i_s, j_inc]))

                if stats["std_mat"] is not None and np.isfinite(stats["std_mat"][i_s, j_inc]):
                    yerr_vals.append(float(stats["std_mat"][i_s, j_inc]))
                    any_err = True
                else:
                    yerr_vals.append(0.0)

            ax.bar(
                x + offsets[j_inc],
                heights,
                width=bar_width,
                yerr=yerr_vals if any_err else None,
                capsize=3 if any_err else 0,
                alpha=0.85,
                label=inc_lab,
            )

        ax.set_title(f"Start time {start_time_labels[i_s]}")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right")
        ax.set_xlabel("Scenario")
        ax.grid(True, axis="y", alpha=0.25)

        if ylims_by_start_bin is not None:
            ylim = ylims_by_start_bin[i_s]
            if ylim is not None:
                ax.set_ylim(*ylim)

    axes[0].set_ylabel("|experienced TT - best observed TT|")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Income quantile",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 0.88, 1))
    else:
        fig.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# Fixed start-time bins shared by every plot and every seed.
# With right=False these mean:
#   [0,20), [20,40), [40,60), [60,80), [80,100]
# and the helper nudges the final edge so 100 is still included.
START_TIME_BINS = [0, 20, 40, 60, 80, 100]
START_TIME_LABELS = ["0–20", "20–40", "40–60", "60–80", "80–100"]

ALGO_RUNS = {   


    #"monetary pricing":[
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
        #r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_2_5_heart_exps/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_2_5_heart_exps/episodes",
    
    #],
    #"monetary-pricing 1":[
    #    r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #    r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_1_heart_exps_longer/episodes",
    #],

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

        "monetary pricing 0.5":[
        #r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_no_dist_cost/episodes",
    ],

     "monetary pricing fee 1 longer": [
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





    }


plot_abs_tt_gap_compare_scenarios_by_start_bin(
    ALGO_RUNS,
    start_time_bins=START_TIME_BINS,
    start_time_labels=START_TIME_LABELS,
    episode_range=(1000, 1320),
    travel_time_col="travel_time",
    observations_col="observation",
    start_time_col="start_time",
    income_col="income",
    plot_kind="AV",
    n_income_bins=5,
    obs_tt_scale=0,
    travel_time_unit="minutes",
    avg_mode="row",
    show_std=True,
    ylims_by_start_bin=[
        (12, 22),    # 0–20
        (18, 28),   # 20–40
        (24, 34),   # 40–60
        (30, 40),   # 60–80
        (35, 45),   # 80–100
    ],
    save_path="imgs/avg_abs_tt_gap_compare_scenarios_by_start_bin.png",
    show=True,
)
# ---------------------------------------------------------
# 1) AVERAGE ACROSS REPLICATIONS
# ---------------------------------------------------------
"""plot_abs_tt_gap_avg_across_replications(
    ALGO_RUNS,
    start_time_bins=START_TIME_BINS,
    start_time_labels=START_TIME_LABELS,
    episode_range=(1315, 1315),
    travel_time_col="travel_time",
    observations_col="observation",   # change to "observations" if needed
    start_time_col="start_time",
    income_col="income",
    plot_kind="AV",
    n_income_bins=5,                  # global income quintiles
    obs_tt_scale=0,
    travel_time_unit="minutes",
    avg_mode="row",                   # or "agent"
    show_std=True,
    save_path="imgs/avg_abs_tt_gap_by_start_bin_income_quintile.png",
    show=True,
)"""


# ---------------------------------------------------------
# 2) EACH SEED SEPARATELY
# ---------------------------------------------------------
plot_abs_tt_gap_per_seed(
    ALGO_RUNS,
    start_time_bins=START_TIME_BINS,
    start_time_labels=START_TIME_LABELS,
    episode_range=(1315, 1315),
    travel_time_col="travel_time",
    observations_col="observation",   # change to "observations" if needed
    start_time_col="start_time",
    income_col="income",
    plot_kind="AV",
    n_income_bins=5,                  # same global income quintiles
    obs_tt_scale=100.0,
    travel_time_unit="minutes",
    avg_mode="row",                   # or "agent"
    ncols=4,
    save_dir="imgs/per_seed_abs_tt_gap",
    show=True,
)