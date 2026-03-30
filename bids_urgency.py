# ==========================================================
# URGENCY (x) vs AVG BID POINTS PER ROUTE — 3 BARPLOT PANELS
# ==========================================================
# Reuses:
# - _episodes_dir
# - list_episode_files
# - load_episodes

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Episode loading helpers
# -----------------------
_EP_RE = re.compile(r"^ep(\d+)\.csv$")

smooth_window=3      # None or odd integer (e.g. 3,5,7,11)
smooth_center=True   # center the window

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


def plot_avg_bids_per_route_by_urgency_bars(
    algo_runs,                          # dict: scenario -> list of run dirs
    episode_range=None,
    episodes=None,
    last_n=30,

    bid_cols=("bid_route_0", "bid_route_1", "bid_route_2"),
    route_titles=("Route 0", "Route 1", "Route 2"),

    urgency_col="urgency",
    kind_col="kind",
    plot_kind="AV",                     # "AV", "Human", "All"

    # aggregation:
    # "row"   = average over all rows at each urgency
    # "agent" = average per agent at each urgency, then average agents
    avg_mode="row",
    id_col="id",

    urgency_round=None,                 # None or 1/5/10/...
    min_n_per_urgency=1,
    show_error_bars=True,               # std across runs
    figsize=(18, 5.5),
    save_path=None,
    show=True,
    rotation=0,
):
    scenarios = list(algo_runs.keys())

    def round_urg(u: pd.Series) -> pd.Series:
        u = pd.to_numeric(u, errors="coerce")
        if urgency_round is None:
            return u
        step = float(urgency_round)
        return np.round(u / step) * step

    # -------------------------------------------------
    # Collect results:
    # scenario_stats[scenario][route] = {"urgencies", "mean", "std"}
    # -------------------------------------------------
    scenario_stats = {sc: {} for sc in scenarios}
    all_urgencies = set()

    for sc in scenarios:
        for bid_col in bid_cols:
            per_run_series = []

            for run_dir in algo_runs[sc]:
                df = load_episodes(
                    run_dir,
                    n_last=last_n,
                    episode_range=episode_range,
                    episodes=episodes,
                )
                if df is None or df.empty:
                    continue

                if plot_kind != "All" and kind_col in df.columns:
                    df = df[df[kind_col] == plot_kind]
                if df.empty:
                    continue

                needed = [urgency_col, bid_col]
                if avg_mode == "agent":
                    needed.append(id_col)

                missing = [c for c in needed if c not in df.columns]
                if missing:
                    print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run for {bid_col}.")
                    continue

                tmp = df[needed].copy()
                tmp[urgency_col] = round_urg(tmp[urgency_col])
                tmp[bid_col] = pd.to_numeric(tmp[bid_col], errors="coerce")
                tmp = tmp.dropna(subset=[urgency_col, bid_col])

                if tmp.empty:
                    continue

                if avg_mode == "agent":
                    tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
                    tmp = tmp.dropna(subset=[id_col])
                    if tmp.empty:
                        continue
                    tmp[id_col] = tmp[id_col].astype(int)

                    # mean bid per agent at each urgency
                    per_agent = (
                        tmp.groupby([id_col, urgency_col], observed=True)[bid_col]
                        .mean()
                        .reset_index()
                    )

                    # then mean across agents
                    g = per_agent.groupby(urgency_col, observed=True)[bid_col].mean()
                    n = per_agent.groupby(urgency_col, observed=True)[id_col].nunique()
                else:
                    # mean bid over all rows at each urgency
                    g = tmp.groupby(urgency_col, observed=True)[bid_col].mean()
                    n = tmp.groupby(urgency_col, observed=True)[bid_col].size()

                if min_n_per_urgency > 1:
                    g = g[n >= min_n_per_urgency]

                if g.empty:
                    continue

                per_run_series.append(g)

            if not per_run_series:
                print(f"[WARN] Scenario '{sc}' had no usable runs for {bid_col}.")
                continue

            urgs = sorted(set(np.concatenate([s.index.to_numpy(dtype=float) for s in per_run_series])))
            mat = np.full((len(per_run_series), len(urgs)), np.nan, dtype=float)

            for i, s in enumerate(per_run_series):
                mat[i, :] = s.reindex(urgs).to_numpy(dtype=float)

            mean_vec = np.nanmean(mat, axis=0)
            std_vec = (
                np.nanstd(mat, axis=0, ddof=1 if mat.shape[0] > 1 else 0)
                if mat.shape[0] > 1 else np.zeros(len(urgs))
            )

            scenario_stats[sc][bid_col] = {
                "urgencies": np.asarray(urgs, dtype=float),
                "mean": mean_vec,
                "std": std_vec,
            }
            all_urgencies.update(urgs)

    if not all_urgencies:
        print("[WARN] No usable data found.")
        return None

    all_urgencies = np.array(sorted(all_urgencies), dtype=float)

    # -------------
    # Create figure
    # -------------
    n_routes = len(bid_cols)
    fig, axes = plt.subplots(1, n_routes, figsize=figsize, sharey=True)
    if n_routes == 1:
        axes = [axes]

    x = np.arange(len(all_urgencies))
    n_scen = len(scenarios)
    total_width = 0.82
    bar_width = total_width / max(n_scen, 1)

    for ax, bid_col, title in zip(axes, bid_cols, route_titles):
        for i, sc in enumerate(scenarios):
            if bid_col not in scenario_stats[sc]:
                continue

            stats = scenario_stats[sc][bid_col]

            mean_aligned = (
                pd.Series(stats["mean"], index=stats["urgencies"])
                .reindex(all_urgencies)
                .to_numpy(dtype=float)
            )
            std_aligned = (
                pd.Series(stats["std"], index=stats["urgencies"])
                .reindex(all_urgencies)
                .to_numpy(dtype=float)
            )

            offset = (i - (n_scen - 1) / 2) * bar_width
            mask = ~np.isnan(mean_aligned)

            ax.bar(
                x[mask] + offset,
                mean_aligned[mask],
                width=bar_width,
                label=sc,
                yerr=std_aligned[mask] if show_error_bars else None,
                capsize=3 if show_error_bars else 0,
                alpha=0.9,
            )

        ax.set_title(title, fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{int(u)}" if float(u).is_integer() else f"{u:g}" for u in all_urgencies],
            rotation=rotation
        )
        ax.set_xlabel("Urgency", fontsize=15)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=13)

    axes[0].set_ylabel("Average bid points", fontsize=15)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_avg_bids_per_route_by_urgency_lines(
    algo_runs,                          # dict: scenario -> list of run dirs
    episode_range=None,
    episodes=None,
    last_n=30,

    bid_cols=("bid_route_0", "bid_route_1", "bid_route_2"),
    route_titles=("Route 0", "Route 1", "Route 2"),

    urgency_col="urgency",
    kind_col="kind",
    plot_kind="AV",                     # "AV", "Human", "All"

    # aggregation:
    # "row"   = average over all rows at each urgency
    # "agent" = average per agent at each urgency, then average agents
    avg_mode="row",
    id_col="id",

    urgency_round=None,                 # None or 1/5/10/...
    min_n_per_urgency=1,

    show_std_band=True,                 # shaded band = ± std across runs
    show_percentile_band=False,         # shaded percentile band across runs
    lower_percentile=25,
    upper_percentile=75,

    smooth_window=None,                 # None or odd integer (e.g. 3,5,7)
    smooth_center=True,

    figsize=(18, 5.5),
    save_path=None,
    show=True,
):
    scenarios = list(algo_runs.keys())

    def round_urg(u: pd.Series) -> pd.Series:
        u = pd.to_numeric(u, errors="coerce")
        if urgency_round is None:
            return u
        step = float(urgency_round)
        return np.round(u / step) * step

    # -------------------------------------------------
    # Collect results:
    # scenario_stats[scenario][route] = {"urgencies", "mean", "std", "lower", "upper"}
    # -------------------------------------------------
    scenario_stats = {sc: {} for sc in scenarios}
    all_urgencies = set()

    for sc in scenarios:
        for bid_col in bid_cols:
            per_run_series = []

            for run_dir in algo_runs[sc]:
                df = load_episodes(
                    run_dir,
                    n_last=last_n,
                    episode_range=episode_range,
                    episodes=episodes,
                )
                if df is None or df.empty:
                    continue

                if plot_kind != "All" and kind_col in df.columns:
                    df = df[df[kind_col] == plot_kind]
                if df.empty:
                    continue

                needed = [urgency_col, bid_col]
                if avg_mode == "agent":
                    needed.append(id_col)

                missing = [c for c in needed if c not in df.columns]
                if missing:
                    print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run for {bid_col}.")
                    continue

                tmp = df[needed].copy()
                tmp[urgency_col] = round_urg(tmp[urgency_col])
                tmp[bid_col] = pd.to_numeric(tmp[bid_col], errors="coerce")
                tmp = tmp.dropna(subset=[urgency_col, bid_col])

                if tmp.empty:
                    continue

                if avg_mode == "agent":
                    tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
                    tmp = tmp.dropna(subset=[id_col])
                    if tmp.empty:
                        continue
                    tmp[id_col] = tmp[id_col].astype(int)

                    # mean bid per agent at each urgency
                    per_agent = (
                        tmp.groupby([id_col, urgency_col], observed=True)[bid_col]
                        .mean()
                        .reset_index()
                    )

                    # then mean across agents
                    g = per_agent.groupby(urgency_col, observed=True)[bid_col].mean()
                    n = per_agent.groupby(urgency_col, observed=True)[id_col].nunique()
                else:
                    # mean bid over all rows at each urgency
                    g = tmp.groupby(urgency_col, observed=True)[bid_col].mean()
                    n = tmp.groupby(urgency_col, observed=True)[bid_col].size()

                if min_n_per_urgency > 1:
                    g = g[n >= min_n_per_urgency]

                if g.empty:
                    continue

                per_run_series.append(g)

            if not per_run_series:
                print(f"[WARN] Scenario '{sc}' had no usable runs for {bid_col}.")
                continue

            urgs = sorted(set(np.concatenate([s.index.to_numpy(dtype=float) for s in per_run_series])))
            mat = np.full((len(per_run_series), len(urgs)), np.nan, dtype=float)

            for i, s in enumerate(per_run_series):
                mat[i, :] = s.reindex(urgs).to_numpy(dtype=float)

            mean_vec = np.nanmean(mat, axis=0)
            std_vec = (
                np.nanstd(mat, axis=0, ddof=1 if mat.shape[0] > 1 else 0)
                if mat.shape[0] > 1 else np.zeros(len(urgs))
            )
            lower_vec = np.nanpercentile(mat, lower_percentile, axis=0)
            upper_vec = np.nanpercentile(mat, upper_percentile, axis=0)

            urgs = np.asarray(urgs, dtype=float)

            # optional smoothing
            if smooth_window is not None and smooth_window > 1:
                order = np.argsort(urgs)
                urgs = urgs[order]
                mean_vec = mean_vec[order]
                std_vec = std_vec[order]
                lower_vec = lower_vec[order]
                upper_vec = upper_vec[order]

                mean_vec = (
                    pd.Series(mean_vec)
                    .rolling(window=smooth_window, center=smooth_center, min_periods=1)
                    .mean()
                    .to_numpy()
                )
                std_vec = (
                    pd.Series(std_vec)
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

            scenario_stats[sc][bid_col] = {
                "urgencies": urgs,
                "mean": mean_vec,
                "std": std_vec,
                "lower": lower_vec,
                "upper": upper_vec,
            }
            all_urgencies.update(urgs.tolist())

    if not all_urgencies:
        print("[WARN] No usable data found.")
        return None

    # -------------
    # Create figure
    # -------------
    n_routes = len(bid_cols)
    fig, axes = plt.subplots(1, n_routes, figsize=figsize, sharey=True)
    if n_routes == 1:
        axes = [axes]

    for ax, bid_col, title in zip(axes, bid_cols, route_titles):
        for sc in scenarios:
            if bid_col not in scenario_stats[sc]:
                continue

            stats = scenario_stats[sc][bid_col]
            x = stats["urgencies"]
            y = stats["mean"]
            y_std = stats["std"]
            y_low = stats["lower"]
            y_up = stats["upper"]

            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            y_std = y_std[mask]
            y_low = y_low[mask]
            y_up = y_up[mask]

            if len(x) == 0:
                continue

            ax.plot(
                x,
                y,
                marker="o",
                markersize=3,
                linewidth=2,
                label=sc,
            )

            if show_std_band:
                ax.fill_between(x, y - y_std, y + y_std, alpha=0.15)

            if show_percentile_band:
                ax.fill_between(x, y_low, y_up, alpha=0.18)

        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Urgency", fontsize=15)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=13)

    axes[0].set_ylabel("Average bid points", fontsize=15)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_avg_bids_per_route_by_urgency_scatter(
    algo_runs,                          # dict: scenario -> list of run dirs
    episode_range=None,
    episodes=None,
    last_n=30,

    bid_cols=("bid_route_0", "bid_route_1", "bid_route_2"),
    route_titles=("Route 0", "Route 1", "Route 2"),

    urgency_col="urgency",
    kind_col="kind",
    plot_kind="AV",                     # "AV", "Human", "All"

    # aggregation:
    # "row"   = average over all rows at each urgency
    # "agent" = average per agent at each urgency, then average agents
    avg_mode="row",
    id_col="id",

    urgency_round=None,                 # None or 1/5/10/...
    min_n_per_urgency=1,

    show_error_bars=True,               # vertical std bars across runs
    point_size=36,
    point_alpha=0.9,

    figsize=(18, 5.5),
    save_path=None,
    show=True,
):
    scenarios = list(algo_runs.keys())

    def round_urg(u: pd.Series) -> pd.Series:
        u = pd.to_numeric(u, errors="coerce")
        if urgency_round is None:
            return u
        step = float(urgency_round)
        return np.round(u / step) * step

    # scenario_stats[scenario][route] = {"urgencies", "mean", "std"}
    scenario_stats = {sc: {} for sc in scenarios}

    for sc in scenarios:
        for bid_col in bid_cols:
            per_run_series = []

            for run_dir in algo_runs[sc]:
                df = load_episodes(
                    run_dir,
                    n_last=last_n,
                    episode_range=episode_range,
                    episodes=episodes,
                )
                if df is None or df.empty:
                    continue

                if plot_kind != "All" and kind_col in df.columns:
                    df = df[df[kind_col] == plot_kind]
                if df.empty:
                    continue

                needed = [urgency_col, bid_col]
                if avg_mode == "agent":
                    needed.append(id_col)

                missing = [c for c in needed if c not in df.columns]
                if missing:
                    print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run for {bid_col}.")
                    continue

                tmp = df[needed].copy()
                tmp[urgency_col] = round_urg(tmp[urgency_col])
                tmp[bid_col] = pd.to_numeric(tmp[bid_col], errors="coerce")
                tmp = tmp.dropna(subset=[urgency_col, bid_col])

                if tmp.empty:
                    continue

                if avg_mode == "agent":
                    tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
                    tmp = tmp.dropna(subset=[id_col])
                    if tmp.empty:
                        continue
                    tmp[id_col] = tmp[id_col].astype(int)

                    per_agent = (
                        tmp.groupby([id_col, urgency_col], observed=True)[bid_col]
                        .mean()
                        .reset_index()
                    )
                    g = per_agent.groupby(urgency_col, observed=True)[bid_col].mean()
                    n = per_agent.groupby(urgency_col, observed=True)[id_col].nunique()
                else:
                    g = tmp.groupby(urgency_col, observed=True)[bid_col].mean()
                    n = tmp.groupby(urgency_col, observed=True)[bid_col].size()

                if min_n_per_urgency > 1:
                    g = g[n >= min_n_per_urgency]

                if g.empty:
                    continue

                per_run_series.append(g)

            if not per_run_series:
                print(f"[WARN] Scenario '{sc}' had no usable runs for {bid_col}.")
                continue

            urgs = sorted(set(np.concatenate([s.index.to_numpy(dtype=float) for s in per_run_series])))
            mat = np.full((len(per_run_series), len(urgs)), np.nan, dtype=float)

            for i, s in enumerate(per_run_series):
                mat[i, :] = s.reindex(urgs).to_numpy(dtype=float)

            mean_vec = np.nanmean(mat, axis=0)
            std_vec = (
                np.nanstd(mat, axis=0, ddof=1 if mat.shape[0] > 1 else 0)
                if mat.shape[0] > 1 else np.zeros(len(urgs))
            )

            scenario_stats[sc][bid_col] = {
                "urgencies": np.asarray(urgs, dtype=float),
                "mean": mean_vec,
                "std": std_vec,
            }

    # check if anything was collected
    has_any = any(len(v) > 0 for v in scenario_stats.values())
    if not has_any:
        print("[WARN] No usable data found.")
        return None

    n_routes = len(bid_cols)
    fig, axes = plt.subplots(1, n_routes, figsize=figsize, sharey=True)
    if n_routes == 1:
        axes = [axes]

    for ax, bid_col, title in zip(axes, bid_cols, route_titles):
        for sc in scenarios:
            if bid_col not in scenario_stats[sc]:
                continue

            stats = scenario_stats[sc][bid_col]
            x = stats["urgencies"]
            y = stats["mean"]
            yerr = stats["std"]

            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            yerr = yerr[mask]

            if len(x) == 0:
                continue

            # scatter points
            ax.scatter(
                x,
                y,
                s=point_size,
                alpha=point_alpha,
                label=sc,
            )

            # optional vertical error bars
            if show_error_bars:
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt="none",
                    alpha=0.45,
                    capsize=2,
                    linewidth=1,
                )

        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Urgency", fontsize=15)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=13)

    axes[0].set_ylabel("Average bid points", fontsize=15)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig

def plot_avg_bids_per_route_by_urgency_lines_per_seed(
    algo_runs,                          # dict: scenario -> list of run dirs
    episode_range=None,
    episodes=None,
    last_n=30,

    bid_cols=("bid_route_0", "bid_route_1", "bid_route_2"),
    route_titles=("Route 0", "Route 1", "Route 2"),

    urgency_col="urgency",
    kind_col="kind",
    plot_kind="AV",                     # "AV", "Human", "All"

    # aggregation inside each run
    # "row"   = average over all rows at each urgency
    # "agent" = average per agent at each urgency, then average agents
    avg_mode="row",
    id_col="id",

    urgency_round=None,                 # None or 1/5/10/...
    min_n_per_urgency=1,

    smooth_window=None,                 # None or odd integer (e.g. 3,5,7)
    smooth_center=True,

    figsize_per_row=3.8,                # figure height per seed row
    figsize_per_col=5.5,                # figure width per route col
    save_path=None,
    show=True,
):
    import re

    def round_urg(u: pd.Series) -> pd.Series:
        u = pd.to_numeric(u, errors="coerce")
        if urgency_round is None:
            return u
        step = float(urgency_round)
        return np.round(u / step) * step

    def extract_run_label(run_dir: str) -> str:
        # try to extract seed number from path, else fallback to basename
        m = re.search(r"seed[_\-]?(\d+)", run_dir)
        if m:
            return f"Seed {m.group(1)}"
        return os.path.basename(os.path.normpath(run_dir))

    # Flatten all runs into [(scenario, run_dir, label), ...]
    run_entries = []
    for sc, run_dirs in algo_runs.items():
        for run_dir in run_dirs:
            run_entries.append((sc, run_dir, extract_run_label(run_dir)))

    if not run_entries:
        print("[WARN] No runs found.")
        return None

    n_rows = len(run_entries)
    n_cols = len(bid_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_col * n_cols, figsize_per_row * n_rows),
        sharex=False,
        sharey=True,
        squeeze=False,
    )

    for row_idx, (sc, run_dir, run_label) in enumerate(run_entries):
        df = load_episodes(
            run_dir,
            n_last=last_n,
            episode_range=episode_range,
            episodes=episodes,
        )

        if df is None or df.empty:
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].text(
                    0.5, 0.5, "No data",
                    ha="center", va="center",
                    transform=axes[row_idx, col_idx].transAxes,
                    fontsize=12
                )
                axes[row_idx, col_idx].set_axis_off()
            continue

        if plot_kind != "All" and kind_col in df.columns:
            df = df[df[kind_col] == plot_kind]

        if df.empty:
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].text(
                    0.5, 0.5, "No matching rows",
                    ha="center", va="center",
                    transform=axes[row_idx, col_idx].transAxes,
                    fontsize=12
                )
                axes[row_idx, col_idx].set_axis_off()
            continue

        for col_idx, (bid_col, route_title) in enumerate(zip(bid_cols, route_titles)):
            ax = axes[row_idx, col_idx]

            needed = [urgency_col, bid_col]
            if avg_mode == "agent":
                needed.append(id_col)

            missing = [c for c in needed if c not in df.columns]
            if missing:
                ax.text(
                    0.5, 0.5, f"Missing {missing}",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=11
                )
                ax.set_axis_off()
                continue

            tmp = df[needed].copy()
            tmp[urgency_col] = round_urg(tmp[urgency_col])
            tmp[bid_col] = pd.to_numeric(tmp[bid_col], errors="coerce")
            tmp = tmp.dropna(subset=[urgency_col, bid_col])

            if tmp.empty:
                ax.text(
                    0.5, 0.5, "No valid data",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=11
                )
                ax.set_axis_off()
                continue

            if avg_mode == "agent":
                tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
                tmp = tmp.dropna(subset=[id_col])
                if tmp.empty:
                    ax.text(
                        0.5, 0.5, "No valid agent IDs",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=11
                    )
                    ax.set_axis_off()
                    continue
                tmp[id_col] = tmp[id_col].astype(int)

                per_agent = (
                    tmp.groupby([id_col, urgency_col], observed=True)[bid_col]
                    .mean()
                    .reset_index()
                )
                g = per_agent.groupby(urgency_col, observed=True)[bid_col].mean()
                n = per_agent.groupby(urgency_col, observed=True)[id_col].nunique()
            else:
                g = tmp.groupby(urgency_col, observed=True)[bid_col].mean()
                n = tmp.groupby(urgency_col, observed=True)[bid_col].size()

            if min_n_per_urgency > 1:
                g = g[n >= min_n_per_urgency]

            if g.empty:
                ax.text(
                    0.5, 0.5, "No data after filtering",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=11
                )
                ax.set_axis_off()
                continue

            x = g.index.to_numpy(dtype=float)
            y = g.to_numpy(dtype=float)

            order = np.argsort(x)
            x = x[order]
            y = y[order]

            if smooth_window is not None and smooth_window > 1:
                y = (
                    pd.Series(y)
                    .rolling(window=smooth_window, center=smooth_center, min_periods=1)
                    .mean()
                    .to_numpy()
                )

            ax.plot(
                x,
                y,
                marker="o",
                markersize=3,
                linewidth=2,
            )

            if row_idx == 0:
                ax.set_title(route_title, fontsize=14)

            if col_idx == 0:
                ax.set_ylabel(f"{run_label}\nAvg bid points", fontsize=12)
            else:
                ax.set_ylabel("")

            ax.set_xlabel("Urgency", fontsize=12)
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)

            # optional subtitle with scenario name on left-most plot
            if col_idx == 0:
                ax.text(
                    0.02, 0.98,
                    sc,
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=10,
                    alpha=0.8,
                )

    fig.tight_layout()

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_avg_bid_route0_by_urgency_lines_per_seed_horizontal(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    bid_col="bid_route_0",
    route_title="Route 0",
    urgency_col="urgency",
    kind_col="kind",
    plot_kind="AV",
    avg_mode="agent",
    id_col="id",
    urgency_round=None,
    min_n_per_urgency=1,
    smooth_window=None,
    smooth_center=True,
    figsize_per_seed=4.5,
    height=3.8,
    save_path=None,
    show=True,
):
    import re

    def round_urg(u: pd.Series) -> pd.Series:
        u = pd.to_numeric(u, errors="coerce")
        if urgency_round is None:
            return u
        step = float(urgency_round)
        return np.round(u / step) * step

    def extract_run_label(run_dir: str) -> str:
        m = re.search(r"seed[_\-]?(\d+)", run_dir)
        if m:
            return f"Seed {m.group(1)}"
        return os.path.basename(os.path.normpath(run_dir))

    run_entries = []
    for sc, run_dirs in algo_runs.items():
        for run_dir in run_dirs:
            run_entries.append((sc, run_dir, extract_run_label(run_dir)))

    if not run_entries:
        print("[WARN] No runs found.")
        return None

    n_cols = len(run_entries)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(figsize_per_seed * n_cols, height),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for col_idx, (sc, run_dir, run_label) in enumerate(run_entries):
        ax = axes[col_idx]

        df = load_episodes(
            run_dir,
            n_last=last_n,
            episode_range=episode_range,
            episodes=episodes,
        )

        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        if plot_kind != "All" and kind_col in df.columns:
            df = df[df[kind_col] == plot_kind]

        if df.empty:
            ax.text(0.5, 0.5, "No matching rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        needed = [urgency_col, bid_col]
        if avg_mode == "agent":
            needed.append(id_col)

        missing = [c for c in needed if c not in df.columns]
        if missing:
            ax.text(0.5, 0.5, f"Missing {missing}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        tmp = df[needed].copy()
        tmp[urgency_col] = round_urg(tmp[urgency_col])
        tmp[bid_col] = pd.to_numeric(tmp[bid_col], errors="coerce")
        tmp = tmp.dropna(subset=[urgency_col, bid_col])

        if tmp.empty:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        if avg_mode == "agent":
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp = tmp.dropna(subset=[id_col])
            if tmp.empty:
                ax.text(0.5, 0.5, "No valid agent IDs", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            tmp[id_col] = tmp[id_col].astype(int)

            per_agent = (
                tmp.groupby([id_col, urgency_col], observed=True)[bid_col]
                .mean()
                .reset_index()
            )
            g = per_agent.groupby(urgency_col, observed=True)[bid_col].mean()
            n = per_agent.groupby(urgency_col, observed=True)[id_col].nunique()
        else:
            g = tmp.groupby(urgency_col, observed=True)[bid_col].mean()
            n = tmp.groupby(urgency_col, observed=True)[bid_col].size()

        if min_n_per_urgency > 1:
            g = g[n >= min_n_per_urgency]

        if g.empty:
            ax.text(0.5, 0.5, "No data after filtering", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        x = g.index.to_numpy(dtype=float)
        y = g.to_numpy(dtype=float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        if smooth_window is not None and smooth_window > 1:
            y = (
                pd.Series(y)
                .rolling(window=smooth_window, center=smooth_center, min_periods=1)
                .mean()
                .to_numpy()
            )

        ax.plot(x, y, marker="o", markersize=3, linewidth=2)
        ax.set_title(run_label, fontsize=13)
        ax.set_xlabel("Urgency", fontsize=12)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.text(0.02, 0.98, sc, transform=ax.transAxes, ha="left", va="top", fontsize=10, alpha=0.8)

    axes[0].set_ylabel(f"{route_title}\nAvg bid points", fontsize=12)

    fig.tight_layout()

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


ALGO_RUNS = {   

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

"""plot_avg_bids_per_route_by_urgency_bars(
    ALGO_RUNS,
    episode_range=(1320, 1320),
    bid_cols=("bid_route_0", "bid_route_1", "bid_route_2"),
    route_titles=("Route 0", "Route 1", "Route 2"),
    urgency_col="urgency",
    plot_kind="AV",
    avg_mode="row",              # or "agent"
    urgency_round=None,          # set to 5 or 10 if urgency is dense
    min_n_per_urgency=1,
    show_error_bars=True,
    save_path="imgs/avg_bid_points_per_route_by_urgency.png",
    show=True,
)"""


"""plot_avg_bids_per_route_by_urgency_lines(
    ALGO_RUNS,
    episode_range=(1300, 1320),
    bid_cols=("bid_route_0", "bid_route_1", "bid_route_2"),
    route_titles=("Route 0", "Route 1", "Route 2"),
    urgency_col="urgency",
    plot_kind="AV",
    avg_mode="agent",          # better if you want equal weight per agent
    urgency_round=None,        # or 5 / 10 if urgency is dense
    min_n_per_urgency=1,
    show_std_band=True,
    show_percentile_band=False,
    smooth_window=5,
    save_path="imgs/avg_bid_points_per_route_by_urgency_lines.png",
    show=True,
)"""

"""plot_avg_bids_per_route_by_urgency_lines_per_seed(
    ALGO_RUNS,
    episode_range=(1300, 1320),
    bid_cols=("bid_route_0"),#", "bid_route_1", "bid_route_2"
    route_titles=("Route 0"),#, "Route 1", "Route 2"),
    urgency_col="urgency",
    plot_kind="AV",
    avg_mode="agent",
    urgency_round=None,
    min_n_per_urgency=1,
    smooth_window=5,
    save_path="imgs/avg_bid_points_per_route_by_urgency_per_seed.png",
    show=True,
)"""


plot_avg_bid_route0_by_urgency_lines_per_seed_horizontal(
    ALGO_RUNS,
    episode_range=(1300, 1320),
    bid_col="bid_route_0",
    route_title="Route 0",
    urgency_col="urgency",
    plot_kind="AV",
    avg_mode="agent",
    smooth_window=3,
    save_path="imgs/route0_bid_by_urgency_per_seed_horizontal.png",
    show=True,
)