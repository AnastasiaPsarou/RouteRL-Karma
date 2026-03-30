import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

"""ALGO_RUNS = {

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
}"""
ALGO_RUNS = {





    "monetary diverse start times smaller training":[
        #r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
    ],

    "karma diverse start times smaller training":[
        #r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        #r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        #r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        #r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        ##r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        #r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        #r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        #r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
    ],


}

# --- selection (same priority rules as your other script) ---
LAST_N_FILES = 10           # used only if EPISODES and EPISODE_RANGE are None
EPISODE_RANGE = (1200, 1320)  # inclusive; set None to disable
EPISODES = None               # e.g. [5800, 5801, 5900]; set None to disable

KIND_FILTER = "AV"          # "AV", "Human", or None
BINS = 50
DENSITY = False
OUT_PATH = "imgs/grouped_travel_time_distributions.png"

ACTIONS_TO_PLOT = None      # e.g. [0,1,2] or None for all

# ============================================================
# Helpers
# ============================================================

_EP_RE = re.compile(r"^ep(\d+)\.csv$", re.IGNORECASE)

def _episodes_dir(p: str) -> str:
    """Allow passing either base_dir or base_dir/episodes."""
    ep = os.path.join(p, "episodes")
    return ep if os.path.isdir(ep) else p

def list_episode_files(folder: str):
    """Return list of (episode_number, fullpath) for epXXXX.csv style files."""
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

def load_selected_csvs(
    folder: str,
    last_n: int | None,
    kind_filter: str | None,
    episode_range: tuple[int, int] | None = None,   # (start, end) inclusive
    episodes: list[int] | None = None,              # explicit episode numbers
):
    """
    Read selected episode CSV files and return:
      travel_by_action: dict[action -> list of travel_time]
      selected_files: list of processed filepaths (or filenames)

    Selection priority:
      1) episodes=[...]
      2) episode_range=(start, end) inclusive
      3) last_n episodes (or all if last_n is None)
    """
    eps = list_episode_files(folder)
    if not eps:
        print(f"[WARN] No episode CSV files found in: {folder}")
        return {}, []

    eps_map = {ep: path for ep, path in eps}

    if episodes is not None:
        wanted = sorted(set(int(e) for e in episodes))
        chosen = [(ep, eps_map[ep]) for ep in wanted if ep in eps_map]
    elif episode_range is not None:
        start, end = map(int, episode_range)
        chosen = [(ep, path) for ep, path in eps if start <= ep <= end]
    else:
        chosen = eps if (last_n is None) else (eps[-last_n:] if len(eps) >= last_n else eps)

    if not chosen:
        print(f"[WARN] No episodes matched selection in: {folder}")
        return {}, []

    travel_by_action: dict[int, list[float]] = {}

    for ep, path in chosen:
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
        except Exception as e:
            print(f"[WARN] Failed reading {path}: {e}")
            continue

        if "travel_time" not in df.columns or "action" not in df.columns:
            continue

        df = df.dropna(how="all")
        df["travel_time"] = pd.to_numeric(df["travel_time"], errors="coerce")
        df["action"] = pd.to_numeric(df["action"], errors="coerce")
        df = df.dropna(subset=["travel_time", "action"])
        df["action"] = df["action"].astype(int)

        if kind_filter is not None and "kind" in df.columns:
            df = df[df["kind"] == kind_filter]

        for a, sub in df.groupby("action"):
            travel_by_action.setdefault(int(a), []).extend(sub["travel_time"].to_list())

    selected_paths = [p for _, p in chosen]
    return travel_by_action, selected_paths

def merge_travel_dicts(dicts: list[dict[int, list[float]]]) -> dict[int, list[float]]:
    merged: dict[int, list[float]] = {}
    for d in dicts:
        for a, vals in d.items():
            merged.setdefault(a, []).extend(vals)
    return merged

# ============================================================
# Plotting
# ============================================================

def plot_grouped_distributions(
    algo_runs: dict[str, list[str]],
    last_n_files: int | None,
    kind_filter: str | None,
    bins: int,
    density: bool,
    out_path: str,
    title: str | None = None,
    actions_to_plot: list[int] | None = None,
    episode_range: tuple[int, int] | None = None,
    episodes: list[int] | None = None,
):
    # ---- Load + aggregate per group ----
    group_data: dict[str, dict[int, list[float]]] = {}

    for group_name, folders in algo_runs.items():
        per_folder = []

        for folder in folders:
            d, selected = load_selected_csvs(
                folder,
                last_n=last_n_files,
                kind_filter=kind_filter,
                episode_range=episode_range,
                episodes=episodes,
            )
            if d:
                per_folder.append(d)

        merged = merge_travel_dicts(per_folder)
        if not merged:
            print(f"[WARN] No usable data for group '{group_name}'. Skipping.")
            continue

        group_data[group_name] = merged

    if not group_data:
        raise ValueError("No usable data found across groups.")

    # ---- Determine actions to plot ----
    all_actions = sorted(set().union(*[set(d.keys()) for d in group_data.values()]))
    if actions_to_plot is None:
        actions = all_actions
    else:
        actions = [a for a in actions_to_plot if a in all_actions]

    if not actions:
        raise ValueError("No actions available to plot after filtering.")

    # ---- Shared bin edges per action ----
    bin_edges_by_action: dict[int, np.ndarray] = {}
    for a in actions:
        vals_all = []
        for g in group_data:
            if a in group_data[g]:
                vals_all.append(np.asarray(group_data[g][a], dtype=float))
        vals_all = np.concatenate(vals_all)
        vmin, vmax = float(np.min(vals_all)), float(np.max(vals_all))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        bin_edges_by_action[a] = np.linspace(vmin, vmax, bins + 1)

    # ---- Plot ----
    ncols = len(actions)
    fig, axs = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.3), sharey=True)
    if ncols == 1:
        axs = [axs]

    group_names = list(group_data.keys())

    for ax, a in zip(axs, actions):
        edges = bin_edges_by_action[a]

        for g in group_names:
            if a not in group_data[g]:
                continue

            vals = np.asarray(group_data[g][a], dtype=float)

            ax.hist(
                vals,
                bins=edges,
                density=density,
                alpha=0.35,
                label=g,
                edgecolor="none"
            )

        ax.set_title(f"Action {a}")
        ax.set_xlabel("travel_time")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    axs[0].set_ylabel("Density" if density else "Count")

    if title is None:
        title = "Travel-time distributions by action"
       

    fig.suptitle(title, y=1.03)

    # ---- Single legend ----
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=max(1, len(labels)),
        frameon=True,
        fontsize=9
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved plot to: {out_path}")

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    plot_grouped_distributions(
        algo_runs=ALGO_RUNS,
        last_n_files=LAST_N_FILES,
        kind_filter=KIND_FILTER,
        bins=BINS,
        density=DENSITY,
        out_path=OUT_PATH,
        actions_to_plot=ACTIONS_TO_PLOT,
        episode_range=EPISODE_RANGE,
        episodes=EPISODES,
    )