
import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Episode loading helpers
# -----------------------
_EP_RE = re.compile(r"^ep(\d+)\.csv$")

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


# -----------------------
# Observation parsing helpers
# -----------------------
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


def add_travel_time_over_best_observed(
    df: pd.DataFrame,
    travel_time_col="travel_time",
    observations_col="observation",   # <- matches your current usage
    output_col="travel_time_over_best_observed",
    obs_scale=100.0,
    travel_time_unit="minutes",
):
    out = df.copy()

    tt = pd.to_numeric(out[travel_time_col], errors="coerce")
    if travel_time_unit.lower().startswith("sec"):
        tt = tt / 60.0

    best_vals = []
    for obs in out[observations_col]:
        arr = parse_observations_to_array(obs)
        if arr is None or len(arr) < 3:
            best_vals.append(np.nan)
            continue

        first3 = arr[:3]
        if np.any(~np.isfinite(first3)):
            best_vals.append(np.nan)
            continue

        best_tt = np.min(first3) * obs_scale
        if not np.isfinite(best_tt) or best_tt <= 0:
            best_vals.append(np.nan)
            continue

        best_vals.append(best_tt)

    out["_best_observed_tt"] = pd.Series(best_vals, index=out.index, dtype=float)
    out[output_col] = tt - out["_best_observed_tt"]
    return out


# -----------------------
# Helpers for left plot
# -----------------------
def _collect_line_data_for_scenario(
    run_dirs,
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
):
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

    per_run_series = []

    for run_dir in run_dirs:
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
            print(f"[WARN] {run_dir}: missing {missing}. Skipping run.")
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

        if not g.empty:
            per_run_series.append(g)

    return per_run_series


# -----------------------
# Helpers for scatter
# -----------------------
def _collect_xy_for_scenario(
    run_dirs,
    episode_range=None,
    episodes=None,
    last_n=30,
    x_col="income",
    y_col="travel_time",
    color_col="urgency",
    kind_col="kind",
    plot_kind="AV",
    travel_time_unit="minutes",
    x_decimals=None,
    observations_col="observation",
    obs_scale=100.0,
):
    def to_minutes(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        return s / 60.0 if travel_time_unit.lower().startswith("sec") else s

    parts = []
    for run_dir in run_dirs:
        df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
        if df is None or df.empty:
            continue

        if plot_kind != "All" and kind_col in df.columns:
            df = df[df[kind_col] == plot_kind]
        if df.empty:
            continue

        base_needed = ["income", color_col]
        if y_col == "travel_time_over_best_observed":
            needed = base_needed + ["travel_time", observations_col]
        else:
            needed = base_needed + [y_col]

        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] {run_dir}: missing {missing}. Skipping run.")
            continue

        tmp_cols = list(dict.fromkeys(
            ["income", color_col] +
            (["travel_time", observations_col] if y_col == "travel_time_over_best_observed" else [y_col])
        ))
        tmp = df[tmp_cols].copy()

        tmp["income"] = pd.to_numeric(tmp["income"], errors="coerce")
        tmp["income"] = tmp["income"] / 240.0
        tmp[color_col] = pd.to_numeric(tmp[color_col], errors="coerce")

        if y_col == "travel_time_over_best_observed":
            tmp = add_travel_time_over_best_observed(
                tmp,
                travel_time_col="travel_time",
                observations_col=observations_col,
                output_col=y_col,
                obs_scale=obs_scale,
                travel_time_unit=travel_time_unit,
            )
        else:
            tmp[y_col] = to_minutes(tmp[y_col])

        tmp["value_of_time"] = tmp["income"] * tmp[color_col]

        if x_col in {"value_of_time", "hourly_salary"}:
            if x_decimals is not None:
                tmp[x_col] = tmp[x_col].round(int(x_decimals))
            tmp = tmp.dropna(subset=[x_col, y_col, color_col])
        else:
            if x_col != "income":
                tmp[x_col] = pd.to_numeric(df[x_col], errors="coerce")
                if x_decimals is not None:
                    tmp[x_col] = tmp[x_col].round(int(x_decimals))
            else:
                if x_decimals is not None:
                    tmp["income"] = tmp["income"].round(int(x_decimals))
            tmp = tmp.dropna(subset=[x_col, y_col, color_col])

        if not tmp.empty:
            parts.append(tmp[[x_col, y_col, color_col]])

    if not parts:
        return None

    return pd.concat(parts, ignore_index=True)


# -----------------------
# Combined figure
# -----------------------
def plot_urgency_line_and_two_panel_scatter(
    algo_runs,

    # left panel: line plot settings
    line_episode_range=None,
    line_episodes=None,
    line_last_n=30,
    line_travel_time_col="travel_time",
    line_urgency_col="urgency",
    line_kind_col="kind",
    line_plot_kind="AV",
    line_travel_time_unit="minutes",
    line_avg_mode="row",
    line_id_col="id",
    urgency_round=None,
    min_n_per_urgency=1,
    show_std_band=True,
    show_percentile_band=False,
    lower_percentile=25,
    upper_percentile=75,
    smooth_window=3,
    smooth_center=True,

    # right two panels: scatter settings
    left_scenario="monetary pricing",
    right_scenario="karma pricing",
    scatter_episode_range=None,
    scatter_episodes=None,
    scatter_last_n=30,
    scatter_x_col="income",
    scatter_y_col="travel_time_over_best_observed",
    scatter_color_col="urgency",
    scatter_kind_col="kind",
    scatter_plot_kind="AV",
    scatter_travel_time_unit="minutes",
    scatter_x_decimals=None,
    jitter_x=0.0,
    jitter_seed=0,
    max_points=None,
    observations_col="observation",
    obs_scale=100.0,

    # figure styling
    figsize=(19, 5.4),
    scatter_s=22,
    scatter_alpha=0.35,
    cmap="viridis",
    save_path=None,
    show=True,
):
    rng = np.random.default_rng(jitter_seed)
    scenarios = list(algo_runs.keys())

    fig, axes = plt.subplots(
        1, 3, figsize=figsize,
        gridspec_kw={"width_ratios": [1.35, 1, 1]},
        sharey=False
    )
    ax_line, ax_sc1, ax_sc2 = axes

    # -----------------------
    # LEFT: urgency line plot
    # -----------------------
    for sc in scenarios:
        per_run_series = _collect_line_data_for_scenario(
            algo_runs[sc],
            episode_range=line_episode_range,
            episodes=line_episodes,
            last_n=line_last_n,
            travel_time_col=line_travel_time_col,
            urgency_col=line_urgency_col,
            kind_col=line_kind_col,
            plot_kind=line_plot_kind,
            travel_time_unit=line_travel_time_unit,
            avg_mode=line_avg_mode,
            id_col=line_id_col,
            urgency_round=urgency_round,
            min_n_per_urgency=min_n_per_urgency,
        )

        if not per_run_series:
            print(f"[WARN] Scenario '{sc}' had no usable runs for line plot. Skipping.")
            continue

        all_u = sorted(set(np.concatenate([s.index.to_numpy(dtype=float) for s in per_run_series])))
        mat = np.full((len(per_run_series), len(all_u)), np.nan, dtype=float)

        for i, s in enumerate(per_run_series):
            mat[i, :] = s.reindex(all_u).to_numpy(dtype=float)

        mean_vec = np.nanmean(mat, axis=0)
        std_vec = np.nanstd(mat, axis=0, ddof=1 if mat.shape[0] > 1 else 0) if mat.shape[0] > 1 else None
        lower_vec = np.nanpercentile(mat, lower_percentile, axis=0)
        upper_vec = np.nanpercentile(mat, upper_percentile, axis=0)

        all_u = np.asarray(all_u, dtype=float)
        order = np.argsort(all_u)
        all_u = all_u[order]
        mean_vec = mean_vec[order]
        lower_vec = lower_vec[order]
        upper_vec = upper_vec[order]
        if std_vec is not None:
            std_vec = std_vec[order]

        if smooth_window is not None and smooth_window > 1:
            mean_vec = pd.Series(mean_vec).rolling(
                window=smooth_window, center=smooth_center, min_periods=1
            ).mean().to_numpy()

            lower_vec = pd.Series(lower_vec).rolling(
                window=smooth_window, center=smooth_center, min_periods=1
            ).mean().to_numpy()

            upper_vec = pd.Series(upper_vec).rolling(
                window=smooth_window, center=smooth_center, min_periods=1
            ).mean().to_numpy()

            if std_vec is not None:
                std_vec = pd.Series(std_vec).rolling(
                    window=smooth_window, center=smooth_center, min_periods=1
                ).mean().to_numpy()

        ax_line.plot(all_u, mean_vec, marker="o", markersize=3, linewidth=2, label=sc)

        if show_std_band and std_vec is not None:
            ax_line.fill_between(all_u, mean_vec - std_vec, mean_vec + std_vec, alpha=0.15)

        if show_percentile_band:
            ax_line.fill_between(all_u, lower_vec, upper_vec, alpha=0.18)

    ax_line.set_title("Urgency vs avg travel time", fontsize=16)
    ax_line.set_xlabel("Urgency", fontsize=14)
    ax_line.set_ylabel("Avg travel time (minutes)", fontsize=14)
    ax_line.grid(True, axis="y", alpha=0.25)
    ax_line.legend(frameon=False, fontsize=11)
    ax_line.tick_params(axis="both", labelsize=12)

    # -----------------------
    # RIGHT: two scatter panels
    # -----------------------
    left_data = _collect_xy_for_scenario(
        algo_runs[left_scenario],
        episode_range=scatter_episode_range,
        episodes=scatter_episodes,
        last_n=scatter_last_n,
        x_col=scatter_x_col,
        y_col=scatter_y_col,
        color_col=scatter_color_col,
        kind_col=scatter_kind_col,
        plot_kind=scatter_plot_kind,
        travel_time_unit=scatter_travel_time_unit,
        x_decimals=scatter_x_decimals,
        observations_col=observations_col,
        obs_scale=obs_scale,
    )

    right_data = _collect_xy_for_scenario(
        algo_runs[right_scenario],
        episode_range=scatter_episode_range,
        episodes=scatter_episodes,
        last_n=scatter_last_n,
        x_col=scatter_x_col,
        y_col=scatter_y_col,
        color_col=scatter_color_col,
        kind_col=scatter_kind_col,
        plot_kind=scatter_plot_kind,
        travel_time_unit=scatter_travel_time_unit,
        x_decimals=scatter_x_decimals,
        observations_col=observations_col,
        obs_scale=obs_scale,
    )

    all_color_vals = []
    for d in [left_data, right_data]:
        if d is not None and not d.empty:
            all_color_vals.append(d[scatter_color_col].to_numpy(dtype=float))

    if not all_color_vals:
        raise RuntimeError(f"No valid data found for color column '{scatter_color_col}'.")

    vmin = 0.1
    vmax = 1.0
    cmap_obj = plt.get_cmap(cmap)
    scatter_artist = None

    def pretty_label(col):
        if col == "value_of_time":
            return "Value of time"
        if col == "travel_time":
            return "Travel time (minutes)"
        if col == "travel_time_over_best_observed":
            return "Experienced TT - best observed TT"
        if col == "income":
            return "Income"
        if col == "urgency":
            return "Urgency"
        if col == "start_time":
            return "Start time"
        return col

    def draw_scatter_panel(ax, data, title):
        nonlocal scatter_artist

        if data is None or data.empty:
            ax.set_title(f"{title} (no data)", fontsize=16)
            ax.grid(True, axis="y", alpha=0.25)
            return None

        if max_points is not None and len(data) > max_points:
            data = data.sample(n=max_points, random_state=jitter_seed).reset_index(drop=True)

        x = data[scatter_x_col].to_numpy(dtype=float)
        y = data[scatter_y_col].to_numpy(dtype=float)
        c = data[scatter_color_col].to_numpy(dtype=float)

        if jitter_x and jitter_x > 0:
            x = x + rng.uniform(-jitter_x, jitter_x, size=len(x))

        sc = ax.scatter(
            x, y, c=c, cmap=cmap_obj, vmin=vmin, vmax=vmax,
            s=scatter_s, alpha=scatter_alpha, marker="o", edgecolors="none"
        )
        scatter_artist = sc

        ax.set_title(title, fontsize=16)
        ax.set_xlabel(pretty_label(scatter_x_col), fontsize=14)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=12)
        return (np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y))

    lims = []
    lims.append(draw_scatter_panel(ax_sc1, left_data, left_scenario))
    lims.append(draw_scatter_panel(ax_sc2, right_data, right_scenario))

    ax_sc1.set_ylabel(pretty_label(scatter_y_col), fontsize=14)

    lims = [t for t in lims if t is not None]
    if lims:
        xmin = min(t[0] for t in lims)
        xmax = max(t[1] for t in lims)
        ymin = min(t[2] for t in lims)
        ymax = max(t[3] for t in lims)
        ax_sc1.set_xlim(xmin, xmax)
        ax_sc2.set_xlim(xmin, xmax)
        ax_sc1.set_ylim(ymin, ymax)
        ax_sc2.set_ylim(ymin, ymax)

    cbar = fig.colorbar(scatter_artist, ax=[ax_sc1, ax_sc2], fraction=0.035, pad=0.03)
    cbar.set_label(pretty_label(scatter_color_col), fontsize=13)
    cbar.ax.tick_params(labelsize=11)

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

plot_urgency_line_and_two_panel_scatter(
    ALGO_RUNS,

    # left panel
    line_episode_range=(1300, 1340),
    line_travel_time_col="travel_time",
    line_urgency_col="urgency",
    line_travel_time_unit="minutes",
    line_avg_mode="row",
    line_plot_kind="AV",
    urgency_round=None,
    min_n_per_urgency=1,
    show_std_band=True,
    show_percentile_band=False,
    smooth_window=3,

    # right two panels
    left_scenario="monetary pricing",
    right_scenario="karma pricing",
    scatter_episode_range=(1000, 1320),
    scatter_x_col="income",
    scatter_y_col="travel_time_over_best_observed",
    scatter_color_col="urgency",
    scatter_plot_kind="AV",
    scatter_travel_time_unit="minutes",
    observations_col="observation",
    obs_scale=100.0,
    jitter_x=0.01,
    scatter_alpha=0.85,

    save_path="imgs/combined_urgency_and_scatter.png",
    show=True,
)