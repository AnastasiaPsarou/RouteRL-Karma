import os
import re
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
# Data preparation
# -----------------------
def get_filtered_data_for_scenario(
    algo_runs,
    scenario,
    episode_range=None,
    episodes=None,
    last_n=30,
    action_col="action",
    urgency_col="urgency",
    kind_col="kind",
    plot_kind="AV",
    income_col="income",
    income_threshold=None,
    income_filter_mode="ge",
):
    if scenario not in algo_runs:
        raise ValueError(f"Scenario '{scenario}' not found in algo_runs.")

    if income_filter_mode not in {"ge", "le"}:
        raise ValueError("income_filter_mode must be 'ge' or 'le'.")

    run_dirs = algo_runs[scenario]
    parts = []

    for run_dir in run_dirs:
        df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
        if df is None or df.empty:
            continue

        if plot_kind != "All" and kind_col in df.columns:
            df = df[df[kind_col] == plot_kind]
        if df.empty:
            continue

        needed = [action_col, urgency_col]
        if income_threshold is not None:
            needed.append(income_col)

        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] {run_dir}: missing {missing}. Skipping run.")
            continue

        keep_cols = [action_col, urgency_col]
        if income_threshold is not None:
            keep_cols.append(income_col)

        tmp = df[keep_cols].copy()
        tmp[action_col] = pd.to_numeric(tmp[action_col], errors="coerce")
        tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")

        if income_threshold is not None:
            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp = tmp.dropna(subset=[action_col, urgency_col, income_col])

            if income_filter_mode == "ge":
                tmp = tmp[tmp[income_col] >= income_threshold]
            else:
                tmp = tmp[tmp[income_col] <= income_threshold]
        else:
            tmp = tmp.dropna(subset=[action_col, urgency_col])

        if not tmp.empty:
            parts.append(tmp)

    if not parts:
        raise RuntimeError(f"No valid data found for scenario '{scenario}' after applying filters.")

    return pd.concat(parts, ignore_index=True)


# -----------------------
# Plotting helpers
# -----------------------
def plot_route_choice_counts_by_urgency_on_axes(
    data,
    axes,
    scenario_title,
    actions=(0, 1, 2),
    action_col="action",
    urgency_col="urgency",
    bins=12,
):
    all_counts = []

    urg = data[urgency_col].to_numpy(dtype=float)
    urg = urg[np.isfinite(urg)]
    if len(urg) == 0:
        raise RuntimeError(f"No valid urgency values for scenario '{scenario_title}'.")

    bin_edges = np.linspace(np.nanmin(urg), np.nanmax(urg), bins + 1)

    for ax, act in zip(axes, actions):
        sub = data[data[action_col] == act].copy()

        if sub.empty:
            ax.set_title(f"{scenario_title} — Action {act} (no data)", fontsize=13)
            ax.grid(True, alpha=0.25)
            continue

        counts, edges = np.histogram(sub[urgency_col].to_numpy(dtype=float), bins=bin_edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        ax.bar(centers, counts, width=widths, align="center", alpha=0.8)
        ax.set_title(f"{scenario_title} — Route {act}", fontsize=13)
        ax.grid(True, alpha=0.25)

        all_counts.append(np.nanmax(counts) if len(counts) else 0)

    ymax = max(all_counts) if all_counts else 0
    return bin_edges, ymax


def plot_two_scenarios_route_choice_vs_urgency(
    algo_runs,
    scenarios=("monetary pricing", "karma pricing"),
    episode_range=None,
    episodes=None,
    last_n=30,
    action_col="action",
    urgency_col="urgency",
    kind_col="kind",
    plot_kind="AV",
    actions=(0, 1, 2),
    income_col="income",
    income_threshold=None,
    income_filter_mode="ge",
    bins=12,
    figsize=(16, 10),
    save_path=None,
    show=True,
):
    if len(scenarios) != 2:
        raise ValueError("Please provide exactly 2 scenarios.")

    scenario_data = {}
    global_min_urg = np.inf
    global_max_urg = -np.inf

    for scenario in scenarios:
        data = get_filtered_data_for_scenario(
            algo_runs=algo_runs,
            scenario=scenario,
            episode_range=episode_range,
            episodes=episodes,
            last_n=last_n,
            action_col=action_col,
            urgency_col=urgency_col,
            kind_col=kind_col,
            plot_kind=plot_kind,
            income_col=income_col,
            income_threshold=income_threshold,
            income_filter_mode=income_filter_mode,
        )
        scenario_data[scenario] = data

        urg = data[urgency_col].to_numpy(dtype=float)
        urg = urg[np.isfinite(urg)]
        global_min_urg = min(global_min_urg, np.nanmin(urg))
        global_max_urg = max(global_max_urg, np.nanmax(urg))

    fig, axes = plt.subplots(2, len(actions), figsize=figsize, sharex=True, sharey=True)
    if len(actions) == 1:
        axes = np.array(axes).reshape(2, 1)

    bin_edges = np.linspace(global_min_urg, global_max_urg, bins + 1)
    row_maxes = []

    for row_idx, scenario in enumerate(scenarios):
        data = scenario_data[scenario]

        max_count_this_row = 0
        for ax, act in zip(axes[row_idx], actions):
            sub = data[data[action_col] == act].copy()

            if sub.empty:
                ax.set_title(f"{scenario} — Route {act} (no data)", fontsize=13)
                ax.grid(True, alpha=0.25)
                continue

            counts, edges = np.histogram(sub[urgency_col].to_numpy(dtype=float), bins=bin_edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)

            ax.bar(centers, counts, width=widths, align="center", alpha=0.8)
            ax.set_title(f"{scenario} — Route {act}", fontsize=13)
            ax.grid(True, alpha=0.25)

            max_count_this_row = max(max_count_this_row, np.nanmax(counts) if len(counts) else 0)

        row_maxes.append(max_count_this_row)

    ymax = max(row_maxes) if row_maxes else 0
    for r in range(2):
        for c in range(len(actions)):
            axes[r, c].set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
            axes[r, c].set_xlim(bin_edges[0], bin_edges[-1])

    for r in range(2):
        axes[r, 0].set_ylabel("Number of people", fontsize=14)
        for c in range(len(actions)):
            axes[r, c].set_xlabel("Urgency", fontsize=14)
            axes[r, c].tick_params(axis="both", labelsize=11)

    title = "Number of people choosing each route by urgency"
    if income_threshold is not None:
        op = ">=" if income_filter_mode == "ge" else "<="
        title += f" (income {op} {income_threshold})"

    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


# -----------------------
# Example input
# -----------------------
ALGO_RUNS = {
    "monetary pricing": [
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
    ],
    "karma pricing": [
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
    ],
}


plot_two_scenarios_route_choice_vs_urgency(
    ALGO_RUNS,
    scenarios=("monetary pricing", "karma pricing"),
    episode_range=(900, 1320),
    action_col="action",
    urgency_col="urgency",
    plot_kind="AV",
    actions=(0, 1, 2),
    income_threshold=0,
    income_filter_mode="ge",
    bins=12,
    figsize=(16, 10),
    save_path="imgs/route_choice_counts_by_urgency.png",
    show=True,
)