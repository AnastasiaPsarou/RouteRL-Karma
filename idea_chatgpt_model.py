import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_EP_RE = re.compile(r"^ep(\d+)\.csv$")

smooth_window = 1
smooth_center = True

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

def load_episodes(
    folder: str,
    n_last: int = 30,
    episode_range=None,
    episodes=None,
):
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

def parse_observations_to_array(x):
    if isinstance(x, np.ndarray):
        return x.astype(float).ravel()

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
    

def _run_label_from_path(run_dir: str) -> str:
    m = re.search(r"seed[_-](\d+)", str(run_dir))
    if m:
        return f"seed {m.group(1)}"

    p = os.path.normpath(run_dir)
    base = os.path.basename(p)
    if base == "episodes":
        return os.path.basename(os.path.dirname(p))
    return base

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


def build_analysis_df_for_vot_gradient(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",          # monthly income
    urgency_col="urgency",
    travel_time_col="travel_time",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    travel_time_unit="minutes",
    work_hours_per_day=8.0,
    work_days_per_month=30.0,
    start_time_bins=None,
    start_time_labels=None,
):
    """
    Build one pooled dataframe across all scenarios/runs.
    """
    if start_time_bins is None:
        start_time_bins = [0, 20, 40, 60, 80, 100]
    if start_time_labels is None:
        start_time_labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]

    rows = []

    def to_minutes(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if travel_time_unit.lower().startswith("sec"):
            return s / 60.0
        return s

    for scenario, run_dirs in algo_runs.items():
        for run_dir in run_dirs:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = ["episode", id_col, income_col, urgency_col, travel_time_col, start_time_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {scenario} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[["episode", id_col, income_col, urgency_col, travel_time_col, start_time_col]].copy()
            tmp["episode"] = pd.to_numeric(tmp["episode"], errors="coerce")
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
            tmp["travel_time_min"] = to_minutes(tmp[travel_time_col])
            tmp[start_time_col] = pd.to_numeric(tmp[start_time_col], errors="coerce")

            tmp = tmp.dropna(subset=["episode", id_col, income_col, urgency_col, "travel_time_min", start_time_col])
            if tmp.empty:
                continue

            tmp["episode"] = tmp["episode"].astype(int)
            tmp[id_col] = tmp[id_col].astype(int)

            tmp["scenario"] = scenario
            tmp["run_id"] = f"{scenario}__{_run_label_from_path(run_dir)}"

            # VoT = hourly income * urgency
            tmp["hourly_income"] = tmp[income_col] / float(work_days_per_month * work_hours_per_day)
            tmp["vot"] = tmp["hourly_income"] * tmp[urgency_col]

            tmp["start_bin"] = assign_fixed_bins(
                tmp[start_time_col],
                bin_edges=start_time_bins,
                labels=start_time_labels,
                right=False,
                include_lowest=True,
            )

            tmp = tmp.dropna(subset=["vot", "start_bin"])
            if tmp.empty:
                continue

            rows.append(
                tmp[["scenario", "run_id", "episode", id_col, "travel_time_min", "vot", "start_bin", urgency_col]]
            )

    if not rows:
        raise ValueError("No usable rows found.")

    out = pd.concat(rows, ignore_index=True)
    return out


def fit_vot_gradient_model_and_plot(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    travel_time_unit="minutes",
    work_hours_per_day=8.0,
    work_days_per_month=30.0,
    start_time_bins=None,
    start_time_labels=None,
    save_path=None,
    show=True,
):
    """
    1) Build pooled agent-day dataset
    2) Residualize TT within (scenario, run, episode, start_bin)
    3) Fit regression: tt_resid ~ log_vot * scenario
    4) Plot predicted lines with 95% CIs
    """
    df = build_analysis_df_for_vot_gradient(
        algo_runs=algo_runs,
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        id_col=id_col,
        income_col=income_col,
        urgency_col=urgency_col,
        travel_time_col=travel_time_col,
        start_time_col=start_time_col,
        kind_col=kind_col,
        plot_kind=plot_kind,
        travel_time_unit=travel_time_unit,
        work_hours_per_day=work_hours_per_day,
        work_days_per_month=work_days_per_month,
        start_time_bins=start_time_bins,
        start_time_labels=start_time_labels,
    )

    # residualize within scenario/run/day/start-time-bin
    group_cols = ["scenario", "run_id", "episode", "start_bin"]
    df["tt_resid"] = df["travel_time_min"] - df.groupby(group_cols)["travel_time_min"].transform("mean")

    # log-transform VoT for stability
    df["log_vot"] = np.log1p(df["vot"])

    # standardize so slopes are easier to read
    mu = df["log_vot"].mean()
    sd = df["log_vot"].std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        raise ValueError("VoT has no variation after preprocessing.")
    df["z_log_vot"] = (df["log_vot"] - mu) / sd

    # regression with scenario interaction
    # cluster SE by run_id
    model = smf.ols(
        "tt_resid ~ z_log_vot * C(scenario)",
        data=df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": pd.factorize(df["run_id"])[0]}
    )

    print(model.summary())

    # prediction grid
    scenarios = list(df["scenario"].unique())
    grid = np.linspace(df["z_log_vot"].quantile(0.05), df["z_log_vot"].quantile(0.95), 80)

    pred_frames = []
    for sc in scenarios:
        tmp = pd.DataFrame({
            "z_log_vot": grid,
            "scenario": sc,
        })
        pred = model.get_prediction(tmp).summary_frame()
        tmp["pred"] = pred["mean"].to_numpy()
        tmp["low"] = pred["mean_ci_lower"].to_numpy()
        tmp["high"] = pred["mean_ci_upper"].to_numpy()
        pred_frames.append(tmp)

    pred_df = pd.concat(pred_frames, ignore_index=True)

    # plot
    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    for sc in scenarios:
        sub = pred_df[pred_df["scenario"] == sc].copy()
        ax.plot(sub["z_log_vot"], sub["pred"], marker=None, linewidth=2, label=sc)
        ax.fill_between(sub["z_log_vot"], sub["low"], sub["high"], alpha=0.18)

    ax.axhline(0.0, linestyle="--", linewidth=1)

    ax.set_xlabel("Standardized log(VoT)", fontsize=14)
    ax.set_ylabel("Residual travel time (minutes)", fontsize=14)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return model, df, pred_df

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def build_analysis_df_for_income_gradient(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",          # monthly income
    urgency_col="urgency",
    travel_time_col="travel_time",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    travel_time_unit="minutes",
    start_time_bins=None,
    start_time_labels=None,
):
    """
    Build one pooled dataframe across all scenarios/runs for income-gradient analysis.

    Output columns:
      - scenario
      - run_id
      - episode
      - id
      - travel_time_min
      - income
      - start_bin
      - urgency
    """
    if start_time_bins is None:
        start_time_bins = [0, 20, 40, 60, 80, 100]
    if start_time_labels is None:
        start_time_labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]

    rows = []

    def to_minutes(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if travel_time_unit.lower().startswith("sec"):
            return s / 60.0
        return s

    for scenario, run_dirs in algo_runs.items():
        for run_dir in run_dirs:
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

            needed = ["episode", id_col, income_col, urgency_col, travel_time_col, start_time_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {scenario} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[["episode", id_col, income_col, urgency_col, travel_time_col, start_time_col]].copy()

            tmp["episode"] = pd.to_numeric(tmp["episode"], errors="coerce")
            tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
            tmp["travel_time_min"] = to_minutes(tmp[travel_time_col])
            tmp[start_time_col] = pd.to_numeric(tmp[start_time_col], errors="coerce")

            tmp = tmp.dropna(
                subset=["episode", id_col, income_col, urgency_col, "travel_time_min", start_time_col]
            )
            if tmp.empty:
                continue

            tmp["episode"] = tmp["episode"].astype(int)
            tmp[id_col] = tmp[id_col].astype(int)

            tmp["scenario"] = scenario
            tmp["run_id"] = f"{scenario}__{_run_label_from_path(run_dir)}"

            tmp["start_bin"] = assign_fixed_bins(
                tmp[start_time_col],
                bin_edges=start_time_bins,
                labels=start_time_labels,
                right=False,
                include_lowest=True,
            )

            tmp = tmp.dropna(subset=[income_col, "start_bin"])
            if tmp.empty:
                continue

            rows.append(
                tmp[
                    [
                        "scenario",
                        "run_id",
                        "episode",
                        id_col,
                        "travel_time_min",
                        income_col,
                        "start_bin",
                        urgency_col,
                    ]
                ].copy()
            )

    if not rows:
        raise ValueError("No usable rows found.")

    out = pd.concat(rows, ignore_index=True)
    return out

def fit_income_gradient_model_and_plot_raw_income_axis(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    travel_time_unit="minutes",
    start_time_bins=None,
    start_time_labels=None,
    reference_start_bin="40–60",
    save_path=None,
    show=True,
):
    """
    Plot predicted NORMAL travel time (minutes) against NORMAL monthly income.
    The model still uses log(income) internally, but the x-axis is monthly income.
    """
    df = build_analysis_df_for_income_gradient(
        algo_runs=algo_runs,
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        id_col=id_col,
        income_col=income_col,
        urgency_col=urgency_col,
        travel_time_col=travel_time_col,
        start_time_col=start_time_col,
        kind_col=kind_col,
        plot_kind=plot_kind,
        travel_time_unit=travel_time_unit,
        start_time_bins=start_time_bins,
        start_time_labels=start_time_labels,
    )

    # log-transform income for the model
    df["log_income"] = np.log1p(df[income_col])

    mu = df["log_income"].mean()
    sd = df["log_income"].std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        raise ValueError("Income has no variation after preprocessing.")

    df["z_log_income"] = (df["log_income"] - mu) / sd

    ref_urgency = float(df[urgency_col].mean())

    model = smf.ols(
        f"travel_time_min ~ z_log_income * C(scenario) + C(start_bin) + {urgency_col}",
        data=df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": pd.factorize(df['run_id'])[0]}
    )

    print(model.summary())

    scenarios = list(df["scenario"].unique())

    # grid in NORMAL monthly income units
    income_grid = np.linspace(
        float(df[income_col].quantile(0.05)),
        float(df[income_col].quantile(0.95)),
        80
    )

    log_income_grid = np.log1p(income_grid)
    z_log_income_grid = (log_income_grid - mu) / sd

    pred_frames = []
    for sc in scenarios:
        tmp = pd.DataFrame({
            "z_log_income": z_log_income_grid,
            "scenario": sc,
            "start_bin": reference_start_bin,
            urgency_col: ref_urgency,
            income_col: income_grid,
        })
        pred = model.get_prediction(tmp).summary_frame()
        tmp["pred"] = pred["mean"].to_numpy()
        tmp["low"] = pred["mean_ci_lower"].to_numpy()
        tmp["high"] = pred["mean_ci_upper"].to_numpy()
        pred_frames.append(tmp)

    pred_df = pd.concat(pred_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    for sc in scenarios:
        sub = pred_df[pred_df["scenario"] == sc].copy()
        ax.plot(sub[income_col], sub["pred"], linewidth=2, label=sc)
        ax.fill_between(sub[income_col], sub["low"], sub["high"], alpha=0.18)

    ax.set_xlabel("Monthly income", fontsize=14)
    ax.set_ylabel("Predicted travel time (minutes)", fontsize=14)
    ax.set_title(f"Predicted travel time at start-time bin {reference_start_bin}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return model, df, pred_df

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def fit_income_gradient_model_and_plot_all_start_bins(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    travel_time_unit="minutes",
    start_time_bins=None,
    start_time_labels=None,
    ncols=3,
    sharey=True,
    save_path=None,
    show=True,
):
    """
    Plot predicted NORMAL travel time (minutes) against NORMAL monthly income,
    with one subplot per start-time bin.

    The model uses log(income) internally, but the x-axis is monthly income.

    This version allows the income gradient to vary by:
      - scenario
      - start-time bin
    """
    df = build_analysis_df_for_income_gradient(
        algo_runs=algo_runs,
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        id_col=id_col,
        income_col=income_col,
        urgency_col=urgency_col,
        travel_time_col=travel_time_col,
        start_time_col=start_time_col,
        kind_col=kind_col,
        plot_kind=plot_kind,
        travel_time_unit=travel_time_unit,
        start_time_bins=start_time_bins,
        start_time_labels=start_time_labels,
    )

    if start_time_labels is None:
        start_time_labels = [x for x in df["start_bin"].dropna().unique().tolist()]

    # log-transform income for the model
    df["log_income"] = np.log1p(df[income_col])

    mu = df["log_income"].mean()
    sd = df["log_income"].std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        raise ValueError("Income has no variation after preprocessing.")

    df["z_log_income"] = (df["log_income"] - mu) / sd

    # reference urgency for plotting
    ref_urgency = float(df[urgency_col].mean())

    # Full interaction: income gradient can differ by scenario and start-time bin
    model = smf.ols(
        f"travel_time_min ~ z_log_income * C(scenario) * C(start_bin) + {urgency_col}",
        data=df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": pd.factorize(df['run_id'])[0]}
    )

    print(model.summary())

    scenarios = list(df["scenario"].unique())

    # grid in NORMAL monthly income units
    income_grid = np.linspace(
        float(df[income_col].quantile(0.05)),
        float(df[income_col].quantile(0.95)),
        80
    )

    log_income_grid = np.log1p(income_grid)
    z_log_income_grid = (log_income_grid - mu) / sd

    # prediction dataframe
    pred_frames = []
    for sb in start_time_labels:
        for sc in scenarios:
            tmp = pd.DataFrame({
                "z_log_income": z_log_income_grid,
                "scenario": sc,
                "start_bin": sb,
                urgency_col: ref_urgency,
                income_col: income_grid,
            })
            pred = model.get_prediction(tmp).summary_frame()
            tmp["pred"] = pred["mean"].to_numpy()
            tmp["low"] = pred["mean_ci_lower"].to_numpy()
            tmp["high"] = pred["mean_ci_upper"].to_numpy()
            pred_frames.append(tmp)

    pred_df = pd.concat(pred_frames, ignore_index=True)

    # subplot layout
    n_bins = len(start_time_labels)
    ncols_eff = min(ncols, n_bins)
    nrows_eff = int(math.ceil(n_bins / ncols_eff))

    fig, axes = plt.subplots(
        nrows_eff,
        ncols_eff,
        figsize=(5.2 * ncols_eff, 4.4 * nrows_eff),
        sharey=sharey,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    legend_handles = {}

    for i, sb in enumerate(start_time_labels):
        ax = axes_flat[i]
        sub_bin = pred_df[pred_df["start_bin"] == sb]

        for sc in scenarios:
            sub = sub_bin[sub_bin["scenario"] == sc].copy()
            if sub.empty:
                continue

            line, = ax.plot(
                sub[income_col],
                sub["pred"],
                linewidth=2,
                label=sc,
            )
            ax.fill_between(
                sub[income_col],
                sub["low"],
                sub["high"],
                alpha=0.18,
            )

            if sc not in legend_handles:
                legend_handles[sc] = line

        ax.set_title(f"Start time {sb}")
        ax.set_xlabel("Monthly income")
        ax.grid(True, axis="y", alpha=0.25)

        if i % ncols_eff == 0:
            ax.set_ylabel("Predicted travel time (minutes)")

    # hide unused axes
    for j in range(n_bins, len(axes_flat)):
        axes_flat[j].axis("off")

    # Put the legend INSIDE the figure at the top
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(legend_handles),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return model, df, pred_df


def fit_income_gradient_model_and_plot_no_start_time(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    id_col="id",
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    travel_time_unit="minutes",
    save_path=None,
    show=True,
):
    """
    Plot predicted NORMAL travel time (minutes) against NORMAL monthly income,
    without taking start time into account.

    The model uses log(income) internally, but the x-axis is monthly income.
    Start time is ignored in both the model and the plotting.
    """

    df = build_analysis_df_for_income_gradient(
        algo_runs=algo_runs,
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        id_col=id_col,
        income_col=income_col,
        urgency_col=urgency_col,
        travel_time_col=travel_time_col,
        start_time_col=start_time_col,
        kind_col=kind_col,
        plot_kind=plot_kind,
        travel_time_unit=travel_time_unit,
    )

    # log-transform income for the model
    df["log_income"] = np.log1p(df[income_col])

    mu = df["log_income"].mean()
    sd = df["log_income"].std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        raise ValueError("Income has no variation after preprocessing.")

    df["z_log_income"] = (df["log_income"] - mu) / sd

    # use mean urgency as the reference for predictions
    ref_urgency = float(df[urgency_col].mean())

    # NO start-time term here
    model = smf.ols(
        f"travel_time_min ~ z_log_income * C(scenario) + {urgency_col}",
        data=df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": pd.factorize(df['run_id'])[0]}
    )

    print(model.summary())

    scenarios = list(df["scenario"].unique())

    # grid in NORMAL monthly income units
    income_grid = np.linspace(
        float(df[income_col].quantile(0.05)),
        float(df[income_col].quantile(0.95)),
        80
    )

    log_income_grid = np.log1p(income_grid)
    z_log_income_grid = (log_income_grid - mu) / sd

    pred_frames = []
    for sc in scenarios:
        tmp = pd.DataFrame({
            "z_log_income": z_log_income_grid,
            "scenario": sc,
            urgency_col: ref_urgency,
            income_col: income_grid,
        })

        pred = model.get_prediction(tmp).summary_frame()
        tmp["pred"] = pred["mean"].to_numpy()
        tmp["low"] = pred["mean_ci_lower"].to_numpy()
        tmp["high"] = pred["mean_ci_upper"].to_numpy()
        pred_frames.append(tmp)

    pred_df = pd.concat(pred_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    for sc in scenarios:
        sub = pred_df[pred_df["scenario"] == sc].copy()
        ax.plot(sub[income_col], sub["pred"], linewidth=2, label=sc)
        ax.fill_between(sub[income_col], sub["low"], sub["high"], alpha=0.18)

    ax.set_xlabel("Monthly income", fontsize=14)
    ax.set_ylabel("Predicted travel time (minutes)", fontsize=14)
    ax.set_title("Predicted travel time vs monthly income")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return model, df, pred_df


ALGO_RUNS = {

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

model, analysis_df, pred_df = fit_income_gradient_model_and_plot_all_start_bins(
    ALGO_RUNS,
    episode_range=(1200, 1320),
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    start_time_col="start_time",
    plot_kind="AV",
    travel_time_unit="minutes",
    start_time_bins=[0, 20, 40, 60, 80, 100],
    start_time_labels=["0–20", "20–40", "40–60", "60–80", "80–100"],
    ncols=3,
    sharey=True,
    save_path="imgs/model_based_tt_vs_income_all_start_bins.png",
    show=True,
)

"""model, analysis_df, pred_df = fit_income_gradient_model_and_plot_no_start_time(
    ALGO_RUNS,
    episode_range=(1200, 1320),
    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",
    start_time_col="start_time",
    plot_kind="AV",
    travel_time_unit="minutes",
    save_path="imgs/model_based_tt_vs_income_no_start_time.png",
    show=True,
)"""