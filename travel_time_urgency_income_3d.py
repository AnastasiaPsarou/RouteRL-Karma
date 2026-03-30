from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

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

def plot_3d_scatter_income_urgency_travel(
    algo_runs,
    episode_range=None,
    episodes=None,
    last_n=30,

    income_col="income",
    urgency_col="urgency",
    travel_time_col="travel_time",

    kind_col="kind",
    plot_kind="AV",

    travel_time_unit="minutes",

    s=6,
    alpha=0.15,
    figsize=(9,7),
    save_path=None,
    show=True
):

    def to_minutes(x):
        x = pd.to_numeric(x, errors="coerce")
        if travel_time_unit.lower().startswith("sec"):
            return x / 60.0
        return x

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for sc, run_dirs in algo_runs.items():

        parts = []

        for run_dir in run_dirs:

            df = load_episodes(
                run_dir,
                n_last=last_n,
                episode_range=episode_range,
                episodes=episodes
            )

            if df is None or df.empty:
                continue

            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]

            needed = [income_col, urgency_col, travel_time_col]

            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}")
                continue

            tmp = df[needed].copy()

            tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
            tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
            tmp[travel_time_col] = to_minutes(tmp[travel_time_col])

            tmp = tmp.dropna()

            if not tmp.empty:
                parts.append(tmp)

        if not parts:
            print(f"[WARN] No data for scenario {sc}")
            continue

        data = pd.concat(parts, ignore_index=True)

        x = data[income_col].to_numpy()
        y = data[travel_time_col].to_numpy()
        z = data[urgency_col].to_numpy()

        ax.scatter(
            x, y, z,
            s=s,
            alpha=alpha,
            label=sc
        )

    ax.set_xlabel("Income")
    ax.set_ylabel("Travel time (minutes)")
    ax.set_zlabel("Urgency")

    ax.legend(frameon=False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print("Saved:", save_path)

    if show:
        plt.show()

    plt.close()

ALGO_RUNS = {



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

    

    

}

plot_3d_scatter_income_urgency_travel(
    ALGO_RUNS,
    episode_range=(1000,1320),
    plot_kind="AV",
    save_path="imgs/3d_income_urgency_travel.png"
)