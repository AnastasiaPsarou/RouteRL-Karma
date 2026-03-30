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
# Core: collect rows for one scenario
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

        needed = ["income", y_col, color_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] {run_dir}: missing {missing}. Skipping run.")
            continue

        tmp = df[["income", y_col, color_col]].copy()

        tmp["income"] = pd.to_numeric(tmp["income"], errors="coerce")
        tmp[color_col] = pd.to_numeric(tmp[color_col], errors="coerce")
        tmp[y_col] = to_minutes(tmp[y_col])

        # Derived x-axis: value of time = urgency * income
        tmp["income"] = tmp["income"] * tmp[color_col]

        if x_col == "income":
            if x_decimals is not None:
                tmp["income"] = tmp["income"].round(int(x_decimals))
            tmp = tmp.dropna(subset=["income", y_col, color_col])
            tmp = tmp.rename(columns={"income": x_col})
        else:
            tmp[x_col] = pd.to_numeric(df[x_col], errors="coerce")
            if x_decimals is not None:
                tmp[x_col] = tmp[x_col].round(int(x_decimals))
            tmp = tmp.dropna(subset=[x_col, y_col, color_col])

        if not tmp.empty:
            parts.append(tmp[[x_col, y_col, color_col]])

    if not parts:
        return None

    data = pd.concat(parts, ignore_index=True)
    return data

def plot_two_panel_scatter(
    algo_runs,
    left_scenario,
    right_scenario,
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
    jitter_x=0.0,
    jitter_seed=0,
    max_points=None,
    figsize=(14.5, 5.2),
    s=40,
    alpha=0.20,
    sharey=True,
    sharex=True,
    cmap="viridis",
    save_path=None,
    show=True,
):
    rng = np.random.default_rng(jitter_seed)

    left_data = _collect_xy_for_scenario(
        algo_runs[left_scenario],
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        kind_col=kind_col,
        plot_kind=plot_kind,
        travel_time_unit=travel_time_unit,
        x_decimals=x_decimals,
    )

    right_data = _collect_xy_for_scenario(
        algo_runs[right_scenario],
        episode_range=episode_range,
        episodes=episodes,
        last_n=last_n,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        kind_col=kind_col,
        plot_kind=plot_kind,
        travel_time_unit=travel_time_unit,
        x_decimals=x_decimals,
    )

    all_color_vals = []
    for d in [left_data, right_data]:
        if d is not None and not d.empty:
            all_color_vals.append(d[color_col].to_numpy(dtype=float))

    if not all_color_vals:
        raise RuntimeError(f"No valid data found for color column '{color_col}'.")

    all_color_vals = np.concatenate(all_color_vals)
    all_color_vals = all_color_vals[np.isfinite(all_color_vals)]

    vmin = 0.1
    vmax = 1.0
    cmap_obj = plt.get_cmap(cmap)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=sharey, sharex=sharex)
    fig.subplots_adjust(right=0.88, wspace=0.18)
    axL, axR = axes

    scatter_artist = None

    def _panel(ax, data, title):
        nonlocal scatter_artist

        if data is None or data.empty:
            ax.set_title(f"{title} (no data)", fontsize=18)
            ax.grid(True, axis="y", alpha=0.25)
            return None

        if max_points is not None and len(data) > max_points:
            data = data.sample(n=max_points, random_state=jitter_seed).reset_index(drop=True)

        x = data[x_col].to_numpy(dtype=float)
        y = data[y_col].to_numpy(dtype=float)
        c = data[color_col].to_numpy(dtype=float)

        if jitter_x and jitter_x > 0:
            x = x + rng.uniform(-jitter_x, jitter_x, size=len(x))

        sc = ax.scatter(
            x,
            y,
            c=c,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            s=s,
            alpha=alpha,
            marker="o",
            edgecolors="none",
        )

        scatter_artist = sc
        ax.set_title(title, fontsize=18)
        ax.grid(True, axis="y", alpha=0.25)
        return (np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y))

    lims = []
    lims.append(_panel(axL, left_data, left_scenario))
    lims.append(_panel(axR, right_data, right_scenario))

    def pretty_label(col):
        if col == "value_of_time":
            return "Value of time"
        if col == "travel_time":
            return "Travel time (minutes)"
        if col == "income":
            return "Income"
        if col == "urgency":
            return "Urgency"
        return col

    axL.set_xlabel(pretty_label(x_col), fontsize=16)
    axR.set_xlabel(pretty_label(x_col), fontsize=16)
    axL.set_ylabel(pretty_label(y_col), fontsize=16)
    axL.tick_params(axis="both", labelsize=14)
    axR.tick_params(axis="both", labelsize=14)

    lims = [t for t in lims if t is not None]
    if lims:
        xmin = min(t[0] for t in lims)
        xmax = max(t[1] for t in lims)
        ymin = min(t[2] for t in lims)
        ymax = max(t[3] for t in lims)
        axL.set_xlim(xmin, xmax)
        axR.set_xlim(xmin, xmax)
        axL.set_ylim(ymin, ymax)

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.72])
    cbar = fig.colorbar(scatter_artist, cax=cax)
    cbar.set_label("Urgency", fontsize=15)
    cbar.set_ticks(np.arange(0.1, 1.01, 0.1))
    cbar.ax.tick_params(labelsize=13)

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


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    """"karma pricing": [
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
"""
       

    
    ALGO_RUNS = {   

        "monetary pricing":[
            #r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
            #r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
            #r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
            #r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
            #r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
            #r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
            #r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
            #r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_new_net_no_dist_cost/episodes",
        ],

            "karma pricing":[
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

    plot_two_panel_scatter(
        ALGO_RUNS,
        left_scenario="monetary pricing",
        right_scenario="karma pricing",
        episode_range=(1000, 1320),
        x_col="income",
        y_col="travel_time",
        color_col="start_time",
        plot_kind="AV",
        travel_time_unit="minutes",
        jitter_x=0.01,
        alpha=0.85,
        save_path="imgs/value_of_time_vs_travel_time_two_panels.png",
        show=True,
    )