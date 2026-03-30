import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_EP_RE = re.compile(r"^ep(\d+)\.csv$")


def smooth_1d(y: np.ndarray, window: int = 3) -> np.ndarray:
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


def plot_bin_normalized_generalized_cost_by_income_bin_lines(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    payment_col="payment",
    kind_col="kind",
    plot_kind="AV",

    high_income_threshold=6000,
    n_equal_bins=4,

    # travel time unit used in CSV
    travel_time_unit="minutes",   # "minutes", "seconds", or "hours"

    # if True, generalized cost = urgency * travel_time_used + payment
    # if False, generalized cost = travel_time_used + payment
    use_urgency_weight=True,

    # "row" = average across all rows
    # "agent" = average per agent first, then average agents
    avg_mode="row",

    show_std_band=True,
    figsize=(7, 5.2),
    save_path=None,
    show=True,

    smooth_lines=True,
    smooth_window=1,

    show_bin_n=True,
    bin_n_agg="mean",

    # optional explicit mapping, e.g.
    # {"karma pricing": "karma", "monetary pricing": "monetary"}
    scenario_type_map=None,
):
    """
    Plots, for each scenario and income bin:

        E[generalized_cost | income_bin] / E[generalized_cost]

    where:
        generalized_cost =
            urgency * travel_time + payment   (if use_urgency_weight=True)
            travel_time + payment             (if use_urgency_weight=False)

    For karma scenarios, payment is set to 0.
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

    def convert_travel_time(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        unit = travel_time_unit.lower()
        if unit.startswith("sec"):
            return s / 3600.0
        if unit.startswith("min"):
            return s / 60.0
        return s

    def infer_scenario_type(name: str) -> str:
        if scenario_type_map is not None and name in scenario_type_map:
            return scenario_type_map[name].lower()

        low = name.lower()
        if "karma" in low:
            return "karma"
        if "monetary" in low:
            return "monetary"
        raise ValueError(
            f"Could not infer scenario type for '{name}'. "
            "Use scenario_type_map to specify 'karma' or 'monetary'."
        )

    fig, ax = plt.subplots(figsize=figsize)
    pooled_n_across_scenarios = []

    for sc in scenarios:
        sc_type = infer_scenario_type(sc)
        per_run_vecs = []
        per_run_n = []

        for run_dir in algo_runs[sc]:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, travel_time_col]
            if use_urgency_weight:
                needed.append(urgency_col)
            if sc_type == "monetary":
                needed.append(payment_col)

            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            cols = [id_col, income_col, travel_time_col]
            if use_urgency_weight:
                cols.append(urgency_col)
            if sc_type == "monetary":
                cols.append(payment_col)

            tmp = df[cols].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp["travel_time_used"] = convert_travel_time(tmp[travel_time_col])

            if use_urgency_weight:
                tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")

            if sc_type == "monetary":
                tmp[payment_col] = pd.to_numeric(tmp[payment_col], errors="coerce")
            else:
                tmp[payment_col] = 0.0

            drop_cols = [id_col, income_col, "travel_time_used", payment_col]
            if use_urgency_weight:
                drop_cols.append(urgency_col)

            tmp = tmp.dropna(subset=drop_cols)
            tmp = tmp[tmp[income_col] > 0]
            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)

            if use_urgency_weight:
                tmp["generalized_cost"] = tmp[urgency_col] * tmp["travel_time_used"] + tmp[payment_col]
            else:
                tmp["generalized_cost"] = tmp["travel_time_used"] + tmp[payment_col]

            tmp["income_bin"] = assign_income_group(tmp[income_col])
            tmp = tmp.dropna(subset=["income_bin", "generalized_cost"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "income_bin"], observed=True)["generalized_cost"]
                       .mean()
                       .reset_index()
                )
                bin_series = per_agent.groupby("income_bin", observed=True)["generalized_cost"].mean()
                overall_mean = per_agent["generalized_cost"].mean()
            else:
                bin_series = tmp.groupby("income_bin", observed=True)["generalized_cost"].mean()
                overall_mean = tmp["generalized_cost"].mean()

            if pd.isna(overall_mean) or overall_mean == 0:
                print(f"[WARN] {sc} / {run_dir}: overall mean generalized cost is zero/NaN. Skipping run.")
                continue

            normalized_series = bin_series / overall_mean
            vec = normalized_series.reindex(bin_labels).to_numpy(dtype=float)
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

        ax.plot(x, mean_plot, marker="o", linewidth=2, label=sc)

        if show_std_band and std_plot is not None:
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)

    ax.axhline(1.0, linestyle="--", linewidth=1, alpha=0.7)

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
        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=11)
    else:
        ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=11)

    ax.set_xlabel("Income bin", fontsize=13)
    ax.set_ylabel("Bin-normalized generalized cost", fontsize=13)
    
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=11)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig

def plot_bin_normalized_generalized_cost_by_income_bin_lines(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    action_col="action",
    monetary_fee=0.5,
    kind_col="kind",
    plot_kind="AV",

    high_income_threshold=6000,
    n_equal_bins=4,

    # travel time unit used in CSV
    travel_time_unit="minutes",   # "minutes", "seconds", or "hours"

    # if True, generalized cost = urgency * travel_time_used + payment
    # if False, generalized cost = travel_time_used + payment
    use_urgency_weight=True,

    # "row" = average across all rows
    # "agent" = average per agent first, then average agents
    avg_mode="row",

    show_std_band=True,
    figsize=(7, 5.2),
    save_path=None,
    show=True,

    smooth_lines=True,
    smooth_window=1,

    show_bin_n=True,
    bin_n_agg="mean",

    # optional explicit mapping, e.g.
    # {"karma pricing": "karma", "monetary pricing": "monetary"}
    scenario_type_map=None,
):
    """
    Plots, for each scenario and income bin:

        E[generalized_cost | income_bin] / E[generalized_cost]

    where:
        generalized_cost =
            urgency * travel_time + payment   (if use_urgency_weight=True)
            travel_time + payment             (if use_urgency_weight=False)

    Payment is computed as:
        payment = monetary_fee if action == 0 else 0

    For karma scenarios, payment is set to 0.
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

    def convert_travel_time(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        unit = travel_time_unit.lower()
        if unit.startswith("sec"):
            return s / 3600.0
        if unit.startswith("min"):
            return s / 60.0
        return s

    def infer_scenario_type(name: str) -> str:
        if scenario_type_map is not None and name in scenario_type_map:
            return scenario_type_map[name].lower()

        low = name.lower()
        if "karma" in low:
            return "karma"
        if "monetary" in low:
            return "monetary"
        raise ValueError(
            f"Could not infer scenario type for '{name}'. "
            "Use scenario_type_map to specify 'karma' or 'monetary'."
        )

    fig, ax = plt.subplots(figsize=figsize)
    pooled_n_across_scenarios = []

    for sc in scenarios:
        sc_type = infer_scenario_type(sc)
        per_run_vecs = []
        per_run_n = []

        for run_dir in algo_runs[sc]:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, travel_time_col]
            if use_urgency_weight:
                needed.append(urgency_col)
            if sc_type == "monetary":
                needed.append(action_col)

            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            cols = [id_col, income_col, travel_time_col]
            if use_urgency_weight:
                cols.append(urgency_col)
            if sc_type == "monetary":
                cols.append(action_col)

            tmp = df[cols].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp["travel_time_used"] = convert_travel_time(tmp[travel_time_col])

            if use_urgency_weight:
                tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")

            if sc_type == "monetary":
                tmp[action_col] = pd.to_numeric(tmp[action_col], errors="coerce")
                tmp["payment_used"] = np.where(tmp[action_col] == 0, monetary_fee, 0.0)
            else:
                tmp["payment_used"] = 0.0

            drop_cols = [id_col, income_col, "travel_time_used", "payment_used"]
            if use_urgency_weight:
                drop_cols.append(urgency_col)
            if sc_type == "monetary":
                drop_cols.append(action_col)

            tmp = tmp.dropna(subset=drop_cols)
            tmp = tmp[tmp[income_col] > 0]
            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)

            if use_urgency_weight:
                tmp["generalized_cost"] = tmp[urgency_col] * tmp["travel_time_used"] + tmp["payment_used"]
            else:
                tmp["generalized_cost"] = tmp["travel_time_used"] + tmp["payment_used"]

            tmp["income_bin"] = assign_income_group(tmp[income_col])
            tmp = tmp.dropna(subset=["income_bin", "generalized_cost"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, "income_bin"], observed=True)["generalized_cost"]
                       .mean()
                       .reset_index()
                )
                bin_series = per_agent.groupby("income_bin", observed=True)["generalized_cost"].mean()
                overall_mean = per_agent["generalized_cost"].mean()
            else:
                bin_series = tmp.groupby("income_bin", observed=True)["generalized_cost"].mean()
                overall_mean = tmp["generalized_cost"].mean()

            if pd.isna(overall_mean) or overall_mean == 0:
                print(f"[WARN] {sc} / {run_dir}: overall mean generalized cost is zero/NaN. Skipping run.")
                continue

            normalized_series = bin_series / overall_mean
            vec = normalized_series.reindex(bin_labels).to_numpy(dtype=float)
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

        ax.plot(x, mean_plot, marker="o", linewidth=2, label=sc)

        if show_std_band and std_plot is not None:
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)

    ax.axhline(1.0, linestyle="--", linewidth=1, alpha=0.7)

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
        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=11)
    else:
        ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=11)

    ax.set_xlabel("Income bin", fontsize=13)
    ax.set_ylabel("Bin-normalized generalized cost", fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=11)
    #ax.set_ylim(0.8, 1.2)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig

def plot_bin_normalized_generalized_cost_by_urgency_lines(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    action_col="action",
    monetary_fee=0.5,
    kind_col="kind",
    plot_kind="AV",

    travel_time_unit="minutes",   # "minutes", "seconds", or "hours"
    use_urgency_weight=True,

    # "row" = average across all rows
    # "agent" = average per agent first, then average agents
    avg_mode="row",

    show_std_band=True,
    figsize=(7, 5.2),
    save_path=None,
    show=True,

    smooth_lines=False,
    smooth_window=1,

    show_urgency_n=True,
    urgency_n_agg="mean",   # "mean" or "minmax"

    scenario_type_map=None,
):
    """
    Plots, for each scenario and urgency value:

        E[generalized_cost | urgency = u] / E[generalized_cost]

    where:
        generalized_cost =
            urgency * travel_time + payment   (if use_urgency_weight=True)
            travel_time + payment             (if use_urgency_weight=False)

    Payment is computed as:
        payment = monetary_fee if action == 0 else 0

    For karma scenarios, payment is set to 0.
    """

    scenarios = list(algo_runs.keys())

    def convert_travel_time(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        unit = travel_time_unit.lower()
        if unit.startswith("sec"):
            return s / 3600.0
        if unit.startswith("min"):
            return s / 60.0
        return s

    def infer_scenario_type(name: str) -> str:
        if scenario_type_map is not None and name in scenario_type_map:
            return scenario_type_map[name].lower()

        low = name.lower()
        if "karma" in low:
            return "karma"
        if "monetary" in low:
            return "monetary"
        raise ValueError(
            f"Could not infer scenario type for '{name}'. "
            "Use scenario_type_map to specify 'karma' or 'monetary'."
        )

    # collect all urgency values across all runs so x-axis is consistent
    all_urgencies = set()
    for sc in scenarios:
        for run_dir in algo_runs[sc]:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty or urgency_col not in df.columns:
                continue
            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            vals = pd.to_numeric(df[urgency_col], errors="coerce").dropna().unique().tolist()
            all_urgencies.update(vals)

    if not all_urgencies:
        raise ValueError(f"No valid urgency values found in column '{urgency_col}'.")

    urgency_values = sorted(all_urgencies)
    x = np.arange(len(urgency_values))

    fig, ax = plt.subplots(figsize=figsize)
    pooled_n_across_scenarios = []

    for sc in scenarios:
        sc_type = infer_scenario_type(sc)
        per_run_vecs = []
        per_run_n = []

        for run_dir in algo_runs[sc]:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [id_col, income_col, urgency_col, travel_time_col]
            if sc_type == "monetary":
                needed.append(action_col)

            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            cols = [id_col, income_col, urgency_col, travel_time_col]
            if sc_type == "monetary":
                cols.append(action_col)

            tmp = df[cols].copy()
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
            tmp["travel_time_used"] = convert_travel_time(tmp[travel_time_col])

            if sc_type == "monetary":
                tmp[action_col] = pd.to_numeric(tmp[action_col], errors="coerce")
                tmp["payment_used"] = np.where(tmp[action_col] == 0, monetary_fee, 0.0)
            else:
                tmp["payment_used"] = 0.0

            drop_cols = [id_col, income_col, urgency_col, "travel_time_used", "payment_used"]
            if sc_type == "monetary":
                drop_cols.append(action_col)

            tmp = tmp.dropna(subset=drop_cols)
            tmp = tmp[tmp[income_col] > 0]
            if tmp.empty:
                continue

            tmp[id_col] = tmp[id_col].astype(int)

            if use_urgency_weight:
                tmp["generalized_cost"] = tmp[urgency_col] * tmp["travel_time_used"] + tmp["payment_used"]
            else:
                tmp["generalized_cost"] = tmp["travel_time_used"] + tmp["payment_used"]

            tmp = tmp.dropna(subset=["generalized_cost", urgency_col])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                per_agent = (
                    tmp.groupby([id_col, urgency_col], observed=True)["generalized_cost"]
                       .mean()
                       .reset_index()
                )
                urg_series = per_agent.groupby(urgency_col, observed=True)["generalized_cost"].mean()
                overall_mean = per_agent["generalized_cost"].mean()
            else:
                urg_series = tmp.groupby(urgency_col, observed=True)["generalized_cost"].mean()
                overall_mean = tmp["generalized_cost"].mean()

            if pd.isna(overall_mean) or overall_mean == 0:
                print(f"[WARN] {sc} / {run_dir}: overall mean generalized cost is zero/NaN. Skipping run.")
                continue

            normalized_series = urg_series / overall_mean
            vec = normalized_series.reindex(urgency_values).to_numpy(dtype=float)
            per_run_vecs.append(vec)

            n_series = tmp.groupby(urgency_col, observed=True)[id_col].nunique()
            n_vec = n_series.reindex(urgency_values).to_numpy(dtype=float)
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

        ax.plot(x, mean_plot, marker="o", linewidth=2, label=sc)

        if show_std_band and std_plot is not None:
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)

    ax.axhline(1.0, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(x)

    if show_urgency_n and pooled_n_across_scenarios:
        scen_means = [np.nanmean(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
        pooled = np.nanmean(np.vstack(scen_means), axis=0)

        if urgency_n_agg == "minmax":
            scen_mins = [np.nanmin(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            scen_maxs = [np.nanmax(n_arr, axis=0) for n_arr in pooled_n_across_scenarios]
            pooled_min = np.nanmin(np.vstack(scen_mins), axis=0)
            pooled_max = np.nanmax(np.vstack(scen_maxs), axis=0)
            tick_labels = [
                f"{u:g}\n(n={int(round(a))}–{int(round(b))})"
                for u, a, b in zip(urgency_values, pooled_min, pooled_max)
            ]
        else:
            tick_labels = [
                f"{u:g}\n(n≈{int(round(n))})"
                for u, n in zip(urgency_values, pooled)
            ]
        ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=11)
    else:
        ax.set_xticklabels([f"{u:g}" for u in urgency_values], fontsize=11)

    ax.set_xlabel("Urgency", fontsize=13)
    ax.set_ylabel("Bin-normalized generalized cost", fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=11)
    #ax.set_ylim(0.8, 1.2)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


ALGO_RUNS = {   

    "karma pricing": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_15_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_16_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_17_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_longer/episodes",
    ],

        
    "monetary pricing":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
    ], 
}

"""plot_bin_normalized_generalized_cost_by_income_bin_lines(
    ALGO_RUNS,
    episode_range=(1300, 1320),
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    payment_col="payment",   # change if your toll column has another name
    travel_time_unit="minutes",
    plot_kind="AV",
    high_income_threshold=6000,
    n_equal_bins=4,
    use_urgency_weight=True,
    avg_mode="row",
    show_std_band=True,
    save_path="imgs/bin_normalized_generalized_cost_by_income_bin.png",
    show=True,
)"""

plot_bin_normalized_generalized_cost_by_urgency_lines(
    ALGO_RUNS,
    episode_range=(2300, 2320),
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    action_col="action",
    monetary_fee=0.5,
    travel_time_unit="minutes",
    plot_kind="AV",
    use_urgency_weight=True,
    avg_mode="row",
    show_std_band=True,
    save_path="imgs/bin_normalized_generalized_cost_by_urgency.png",
    show=True,
)