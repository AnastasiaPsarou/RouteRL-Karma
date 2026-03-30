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

def plot_urgency_vs_start_time_by_action(
    algo_runs,
    scenario,
    episode_range=None,
    episodes=None,
    last_n=30,
    action_col="action",       # change if your CSV uses another name
    urgency_col="urgency",
    start_time_col="start_time",
    kind_col="kind",
    plot_kind="AV",
    actions=(0, 1, 2),         # three actions / three panels
    max_points=None,
    jitter_x=0.0,
    jitter_y=0.0,
    jitter_seed=0,
    alpha=0.35,
    s=20,
    figsize=(15, 4.8),
    sharex=True,
    sharey=True,
    save_path=None,
    show=True,
):
    """
    Create 3 subplots, one for each action.
    In each subplot:
      x = start_time
      y = urgency
      points = individuals whose chosen action equals that subplot action

    Parameters
    ----------
    algo_runs : dict
        Mapping scenario -> list of run directories.
    scenario : str
        Which scenario from algo_runs to plot.
    episode_range, episodes, last_n :
        Same semantics as in your existing loading code.
    action_col, urgency_col, start_time_col : str
        Column names in episode CSVs.
    kind_col, plot_kind :
        Optional filter, e.g. keep only AV agents.
    actions : tuple/list
        Which actions to plot as separate subplots.
    max_points : int or None
        Optional subsample per action panel.
    jitter_x, jitter_y : float
        Optional random jitter for overplotting.
    """

    if scenario not in algo_runs:
        raise ValueError(f"Scenario '{scenario}' not found in algo_runs.")

    rng = np.random.default_rng(jitter_seed)
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

        needed = [action_col, urgency_col, start_time_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] {run_dir}: missing {missing}. Skipping run.")
            continue

        tmp = df[[action_col, urgency_col, start_time_col]].copy()
        tmp[action_col] = pd.to_numeric(tmp[action_col], errors="coerce")
        tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
        tmp[start_time_col] = pd.to_numeric(tmp[start_time_col], errors="coerce")
        tmp = tmp.dropna(subset=[action_col, urgency_col, start_time_col])

        if not tmp.empty:
            parts.append(tmp)

    if not parts:
        raise RuntimeError(f"No valid data found for scenario '{scenario}'.")

    data = pd.concat(parts, ignore_index=True)

    fig, axes = plt.subplots(1, len(actions), figsize=figsize, sharex=sharex, sharey=sharey)
    if len(actions) == 1:
        axes = [axes]

    all_limits = []

    for ax, act in zip(axes, actions):
        sub = data[data[action_col] == act].copy()

        if sub.empty:
            ax.set_title(f"Action {act} (no data)", fontsize=16)
            ax.grid(True, alpha=0.25)
            continue

        if max_points is not None and len(sub) > max_points:
            sub = sub.sample(n=max_points, random_state=jitter_seed).reset_index(drop=True)

        x = sub[start_time_col].to_numpy(dtype=float)
        y = sub[urgency_col].to_numpy(dtype=float)

        if jitter_x > 0:
            x = x + rng.uniform(-jitter_x, jitter_x, size=len(x))
        if jitter_y > 0:
            y = y + rng.uniform(-jitter_y, jitter_y, size=len(y))

        ax.scatter(
            x,
            y,
            s=s,
            alpha=alpha,
            marker="o",
            edgecolors="none",
        )

        ax.set_title(f"Action {act}", fontsize=16)
        ax.grid(True, alpha=0.25)

        all_limits.append((
            np.nanmin(x), np.nanmax(x),
            np.nanmin(y), np.nanmax(y)
        ))

    # common labels
    axes[0].set_ylabel("Urgency", fontsize=15)
    for ax in axes:
        ax.set_xlabel("Start time", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)

    # common axis limits
    if all_limits:
        xmin = min(t[0] for t in all_limits)
        xmax = max(t[1] for t in all_limits)
        ymin = min(t[2] for t in all_limits)
        ymax = max(t[3] for t in all_limits)
        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    fig.suptitle(f"{scenario}: urgency vs start time by action", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

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


    "monetary pricing":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
        #r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
    ],

        "karma pricing":[
            r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_variable_start_times_smaller/episodes",
        ],


    }

plot_urgency_vs_start_time_by_action(
    ALGO_RUNS,
    scenario="monetary pricing",
    episode_range=(1320, 1320),
    action_col="action",
    urgency_col="urgency",
    start_time_col="start_time",
    plot_kind="AV",
    actions=(0, 1, 2),
    jitter_x=0.02,
    jitter_y=0.005,
    alpha=0.35,
    s=18,
    save_path="imgs/monetary_urgency_vs_start_time_by_action.png",
    show=True,
)