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
    income_col="income",
    kind_col="kind",
    plot_kind="AV",
    urgency_col="urgency",
    urgency_threshold=None,
):
    if scenario not in algo_runs:
        raise ValueError(f"Scenario '{scenario}' not found in algo_runs.")

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

        needed = [action_col, income_col]
        if urgency_threshold is not None:
            needed.append(urgency_col)

        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] {run_dir}: missing {missing}. Skipping run.")
            continue

        keep_cols = [action_col, income_col]
        if urgency_threshold is not None:
            keep_cols.append(urgency_col)

        tmp = df[keep_cols].copy()
        tmp[action_col] = pd.to_numeric(tmp[action_col], errors="coerce")
        tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")

        if urgency_threshold is not None:
            tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
            tmp = tmp.dropna(subset=[action_col, income_col, urgency_col])
            tmp = tmp[tmp[urgency_col] > urgency_threshold]
        else:
            tmp = tmp.dropna(subset=[action_col, income_col])

        if not tmp.empty:
            parts.append(tmp)

    if not parts:
        raise RuntimeError(f"No valid data found for scenario '{scenario}' after applying filters.")

    return pd.concat(parts, ignore_index=True)


# -----------------------
# Plot
# -----------------------
def plot_two_scenarios_route_choice_vs_income(
    algo_runs,
    scenarios=("monetary pricing", "karma pricing"),
    episode_range=None,
    episodes=None,
    last_n=30,
    action_col="action",
    income_col="income",
    kind_col="kind",
    plot_kind="AV",
    actions=(0, 1, 2),
    urgency_col="urgency",
    urgency_threshold=None,
    bins=12,
    figsize=(16, 5),
    save_path=None,
    show=True,
):
    if len(scenarios) != 2:
        raise ValueError("Please provide exactly 2 scenarios.")

    scenario_colors = {
        "monetary pricing": "forestgreen",
        "karma pricing": "mediumpurple",
    }

    scenario_data = {}
    global_min_income = np.inf
    global_max_income = -np.inf

    for scenario in scenarios:
        data = get_filtered_data_for_scenario(
            algo_runs=algo_runs,
            scenario=scenario,
            episode_range=episode_range,
            episodes=episodes,
            last_n=last_n,
            action_col=action_col,
            income_col=income_col,
            kind_col=kind_col,
            plot_kind=plot_kind,
            urgency_col=urgency_col,
            urgency_threshold=urgency_threshold,
        ).copy()

        # Convert income to hourly salary
        data[income_col] = pd.to_numeric(data[income_col], errors="coerce") / 240.0
        data = data.dropna(subset=[income_col])

        scenario_data[scenario] = data

        inc = data[income_col].to_numpy(dtype=float)
        inc = inc[np.isfinite(inc)]
        global_min_income = min(global_min_income, np.nanmin(inc))
        global_max_income = max(global_max_income, np.nanmax(inc))

    fig, axes = plt.subplots(1, len(actions), figsize=figsize, sharex=True, sharey=False)
    if len(actions) == 1:
        axes = [axes]

    bin_edges = np.linspace(global_min_income, global_max_income, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    # narrower bars so the two scenarios fit side by side in each income bin
    bar_width = 0.42 * bin_widths

    global_ymax = 0

    for ax, act in zip(axes, actions):
        local_ymax = 0

        for i, scenario in enumerate(scenarios):
            data = scenario_data[scenario]
            sub = data[data[action_col] == act].copy()

            counts, _ = np.histogram(sub[income_col].to_numpy(dtype=float), bins=bin_edges)

            # shift bars left/right within each bin
            offset = (-0.5 + i) * bar_width
            x_positions = bin_centers + offset

            ax.bar(
                x_positions,
                counts,
                width=bar_width,
                align="center",
                alpha=0.85,
                color=scenario_colors.get(scenario, None),
                label=scenario,
                edgecolor="black",
                linewidth=0.4,
            )

            if len(counts):
                local_ymax = max(local_ymax, np.nanmax(counts))

        ymax = local_ymax * 1.05 if local_ymax > 0 else 1
        ax.set_ylim(0, ymax)
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_title(f"Route {act}", fontsize=13)
        ax.set_xlabel("Hourly salary", fontsize=14)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.25)

    """ymax = global_ymax * 1.05 if global_ymax > 0 else 1
    for ax in axes:
        ax.set_ylim(0, ymax)
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_xlabel("Hourly salary", fontsize=14)
        ax.tick_params(axis="both", labelsize=11)"""

    axes[0].set_ylabel("Number of people", fontsize=14)

    title = "Number of people choosing each route by hourly salary"
    if urgency_threshold is not None:
        title += f" (urgency > {urgency_threshold})"

    fig.suptitle(title, fontsize=17)

    # one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))

    fig.tight_layout(rect=[0, 0, 1, 0.90])

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

plot_two_scenarios_route_choice_vs_income(
    ALGO_RUNS,
    scenarios=("monetary pricing", "karma pricing"),
    episode_range=(1000, 1320),
    action_col="action",
    income_col="income",
    plot_kind="AV",
    actions=(0, 1, 2),
    urgency_threshold=0,
    bins=12,
    figsize=(16, 5),
    save_path="imgs/route_choice_counts_by_hourly_salary.png",
    show=True,
)