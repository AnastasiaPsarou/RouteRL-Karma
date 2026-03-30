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

def compute_tt_minus_best_observed_route(
    df: pd.DataFrame,
    travel_time_col="travel_time",
    observations_col="observation",
    obs_tt_scale=100.0,
    travel_time_unit="minutes",   # "minutes" or "seconds"
    output_col="tt_diff",
):
    """
    Creates:
        tt_diff = experienced_travel_time - min(observations[:3]) * obs_tt_scale

    Assumes first 3 observation values are normalized route travel times.
    """
    if travel_time_col not in df.columns or observations_col not in df.columns:
        df[output_col] = np.nan
        return df

    travel_time = pd.to_numeric(df[travel_time_col], errors="coerce")
    if travel_time_unit.lower().startswith("sec"):
        travel_time = travel_time / 60.0   # convert to minutes

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

def plot_tt_minus_best_observed_route_by_urgency_lines(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,
    travel_time_col="travel_time",
    observations_col="observation",
    urgency_col="urgency",
    kind_col="kind",
    plot_kind="AV",                 # "AV", "Human", "All"

    travel_time_unit="minutes",     # "minutes" or "seconds"
    obs_tt_scale=100.0,

    # aggregation:
    # "row"   = average tt_diff across all rows at that urgency
    # "agent" = average per agent at that urgency, then average agents
    avg_mode="row",
    id_col="id",

    urgency_round=None,
    min_n_per_urgency=1,
    show_std_band=True,
    figsize=(10.5, 5.2),
    save_path=None,
    show=True,
    show_percentile_band=False,
    lower_percentile=25,
    upper_percentile=75,
):
    """
    ONE plot:
      x = urgency
      y = avg( experienced TT - best observed TT )
      one line per scenario
      optional shaded band across runs
    """
    scenarios = list(algo_runs.keys())

    def round_urg(u: pd.Series) -> pd.Series:
        u = pd.to_numeric(u, errors="coerce")
        if urgency_round is None:
            return u
        step = float(urgency_round)
        return np.round(u / step) * step

    fig, ax = plt.subplots(figsize=figsize)

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

            needed = [urgency_col, travel_time_col, observations_col]
            if avg_mode == "agent":
                needed.append(id_col)

            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[needed].copy()

            tmp = compute_tt_minus_best_observed_route(
                tmp,
                travel_time_col=travel_time_col,
                observations_col=observations_col,
                obs_tt_scale=obs_tt_scale,
                travel_time_unit=travel_time_unit,
                output_col="tt_diff",
            )

            tmp[urgency_col] = round_urg(tmp[urgency_col])
            tmp = tmp.dropna(subset=[urgency_col, "tt_diff"])
            if tmp.empty:
                continue

            if avg_mode == "agent":
                tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
                tmp = tmp.dropna(subset=[id_col])
                if tmp.empty:
                    continue
                tmp[id_col] = tmp[id_col].astype(int)

                per_agent = (
                    tmp.groupby([id_col, urgency_col], observed=True)["tt_diff"]
                       .mean()
                       .reset_index()
                )
                g = per_agent.groupby(urgency_col, observed=True)["tt_diff"].mean()
                n = per_agent.groupby(urgency_col, observed=True)[id_col].nunique()
            else:
                g = tmp.groupby(urgency_col, observed=True)["tt_diff"].mean()
                n = tmp.groupby(urgency_col, observed=True)["tt_diff"].size()

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

        mean_vec = np.nanmean(mat, axis=0)
        std_vec = (
            np.nanstd(mat, axis=0, ddof=1 if mat.shape[0] > 1 else 0)
            if mat.shape[0] > 1 else None
        )
        lower_vec = np.nanpercentile(mat, lower_percentile, axis=0)
        upper_vec = np.nanpercentile(mat, upper_percentile, axis=0)

        all_u = np.asarray(all_u, dtype=float)

        if smooth_window is not None and smooth_window > 1:
            order = np.argsort(all_u)
            all_u = all_u[order]
            mean_vec = mean_vec[order]
            lower_vec = lower_vec[order]
            upper_vec = upper_vec[order]

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

        ax.plot(
            all_u,
            mean_vec,
            marker="o",
            markersize=3,
            linewidth=2,
            label=f"{sc}",
        )

        if show_std_band and std_vec is not None:
            ax.fill_between(all_u, mean_vec - std_vec, mean_vec + std_vec, alpha=0.15)

        if show_percentile_band:
            ax.fill_between(all_u, lower_vec, upper_vec, alpha=0.18)

    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("Urgency", fontsize=16)
    ax.set_ylabel("Avg experienced TT - best observed TT (minutes)", fontsize=16)
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

plot_tt_minus_best_observed_route_by_urgency_lines(
    ALGO_RUNS,
    episode_range=(1300, 1320),
    travel_time_col="travel_time",
    observations_col="observation",   # important
    urgency_col="urgency",
    travel_time_unit="minutes",
    obs_tt_scale=100.0,
    avg_mode="row",
    plot_kind="AV",
    urgency_round=None,
    min_n_per_urgency=1,
    show_std_band=True,
    save_path="imgs/avg_tt_minus_best_observed_route_by_urgency_lines.png",
    show=True,
    show_percentile_band=False,
)