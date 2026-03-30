# =======================
# URGENCY (x) vs AVG TRAVEL TIME (minutes) — LINE PLOT
# =======================
# What this does:
# - Loads episode CSVs from each run folder (or run/episodes).
# - Uses urgency as the x-axis (NO income bins).
# - For each urgency value, computes the average travel_time over the selected episode range.
# - Plots ONE line per scenario (key in ALGO_RUNS).
# - Optional: shaded band = ± std across runs for each urgency value.
#
# Practical note:
# - If urgency has many unique values, the x-axis will be dense.
#   You can set urgency_round=1/5/10/... to round urgency values and reduce noise.

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Episode loading helpers
# -----------------------
_EP_RE = re.compile(r"^ep(\d+)\.csv$")

smooth_window=3     # None or odd integer (e.g. 3,5,7,11)
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


# ------------------------------------------
# LINE PLOT: x = urgency, y = avg travel time
# ------------------------------------------
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
    # aggregation:
    # "row"   = average travel_time across all rows at that urgency
    # "agent" = average per agent at that urgency, then average agents (equal agent weight)
    avg_mode="row",
    id_col="id",                    # used only if avg_mode="agent"

    urgency_round=None,             # None or e.g. 1, 5, 10
    min_n_per_urgency=1,            # per-run minimum points (or agents) to keep an urgency value
    show_std_band=True,             # shaded band = ± std across runs
    figsize=(10.5, 5.2),
    save_path=None,
    show=True,
    show_percentile_band=False,
    lower_percentile=25,
    upper_percentile=75,
):
    """
    ONE plot:
      x = urgency values (no grouping unless urgency_round is set)
      y = average travel time (minutes)
      one line per scenario
      optional shaded band = ± std across runs
    """

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

    fig, ax = plt.subplots(figsize=figsize)

    for sc in scenarios:
        per_run_series = []

        for run_dir in algo_runs[sc]:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            # kind filter (if present)
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

                # mean travel time per agent at each urgency
                per_agent = (
                    tmp.groupby([id_col, urgency_col], observed=True)[travel_time_col]
                       .mean()
                       .reset_index()
                )
                # then mean across agents for each urgency
                g = per_agent.groupby(urgency_col, observed=True)[travel_time_col].mean()
                n = per_agent.groupby(urgency_col, observed=True)[id_col].nunique()  # agents per urgency
            else:
                g = tmp.groupby(urgency_col, observed=True)[travel_time_col].mean()
                n = tmp.groupby(urgency_col, observed=True)[travel_time_col].size()  # rows per urgency

            if min_n_per_urgency > 1:
                g = g[n >= min_n_per_urgency]

            if g.empty:
                continue

            per_run_series.append(g)

        if not per_run_series:
            print(f"[WARN] Scenario '{sc}' had no usable runs. Skipping line.")
            continue

        # Align runs on the union of urgency values
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

        # -------------------
        # SMOOTHING SECTION
        # -------------------
        if smooth_window is not None and smooth_window > 1:
            order = np.argsort(all_u)
            all_u = all_u[order]
            mean_vec = mean_vec[order]
            lower_vec = lower_vec[order]
            upper_vec = upper_vec[order]

            mean_vec = (
                pd.Series(mean_vec)
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )

            lower_vec = (
                pd.Series(lower_vec)
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )

            upper_vec = (
                pd.Series(upper_vec)
                .rolling(window=smooth_window, center=True, min_periods=1)
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
            ax.fill_between(
                all_u,
                lower_vec,
                upper_vec,
                alpha=0.18,
            )

    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("Urgency", fontsize=16)
    ax.set_ylabel("Average travel time (minutes)", fontsize=16)
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


# =======================
# EXAMPLE USAGE
# =======================
if __name__ == "__main__":

    """ALGO_RUNS = {

        "Karma 7": [
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
        ],
        "Monetary 20":[
            r"training_records_monetary_pricing_300_agents_9_fee_20/episodes",
            r"training_records_monetary_pricing_300_agents_10_fee_20/episodes",
            r"training_records_monetary_pricing_300_agents_11_fee_20/episodes",
            r"training_records_monetary_pricing_300_agents_12_fee_20/episodes",
            r"training_records_monetary_pricing_300_agents_13_fee_20/episodes",
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
    """"UE": [
            r"training_records_user_equilibrium_300_agents_10/episodes",
            r"training_records_user_equilibrium_300_agents_11/episodes",
            r"training_records_user_equilibrium_300_agents_12/episodes",
        ],"""
    """ALGO_RUNS = {

           "fee 10":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_dd_sched_reward_zero/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_dd_sched_reward_zero/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_dd_sched_reward_zero/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_dd_sched_reward_zero/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_dd_sched_reward_zero/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_dd_sched_reward_zero/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_dd_sched_reward_zero/episodes",

    ],
    "fee 11":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_11/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_11/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_11/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_11/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_11/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_11/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_11/episodes",
    ],
    "fee 12":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_12/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_12/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_12/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_12/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_12/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_12/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_12/episodes",

    ],
    "fee 13":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_13/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_13/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_13/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_13/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_13/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_13/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_13/episodes",
    ],
    "fee 14":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_14/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_14/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_14/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_14/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_14/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_14/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_14/episodes",

    ],
    "fee 15":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_15/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_15/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_15/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_15/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_15/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_15/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_15/episodes",
    ],
    }"""
    """ALGO_RUNS = {

    "karma fee 6": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_6/episodes",
    ],
    "karma fee 7": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_7_norm/episodes",
    ],
    "karma fee 8": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_8_norm/episodes",
    ],

    }"""

    """    "karma fee 5": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5/episodes",
    ],
    "karma fee 5 longer": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5_longer/episodes",
    ],
    "karma fee 5 longer norm": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5_norm/episodes",
    ],"""

    plot_avg_travel_time_by_urgency_lines(
        ALGO_RUNS,
        episode_range=(1200, 1320),
        travel_time_col="travel_time",
        urgency_col="urgency",
        travel_time_unit="minutes",  # or "seconds"
        avg_mode="row",              # or "agent"
        plot_kind="AV",

        urgency_round=None,          # set e.g. 5 or 10 if too dense
        min_n_per_urgency=1,
        show_std_band=True,
        save_path="imgs/avg_travel_time_by_urgency_lines.png",
        show=True,
        show_percentile_band=False,
    )
