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
# Income binning helper
# -----------------------
def add_income_bins(
    df: pd.DataFrame,
    income_col: str,
    out_col: str = "income_bin",
    n_bins: int = 5,
    method: str = "quantile",      # "quantile" (qcut) or "equal_width" (cut)
    labels: str = "range",         # "range" -> "lo–hi", "index" -> "Bin 1..5"
):
    """
    Adds an income bin column based on df[income_col].

    - quantile: bins contain (roughly) equal counts. Robust to skew.
    - equal_width: bins have equal numeric width.

    Handles duplicate edges by dropping bins if needed (e.g., many identical incomes).
    """
    if income_col not in df.columns:
        raise KeyError(f"Missing income column '{income_col}'")

    inc = pd.to_numeric(df[income_col], errors="coerce")
    if inc.notna().sum() == 0:
        df[out_col] = np.nan
        return df

    if method == "quantile":
        # qcut can fail if there aren't enough unique values; duplicates='drop' helps.
        try:
            cats = pd.qcut(inc, q=n_bins, duplicates="drop")
        except ValueError:
            # fallback if qcut still can't bin (e.g. all equal)
            cats = pd.cut(inc, bins=min(n_bins, max(1, inc.nunique())), include_lowest=True)
    elif method == "equal_width":
        bins = min(n_bins, max(1, inc.nunique()))
        cats = pd.cut(inc, bins=bins, include_lowest=True)
    else:
        raise ValueError("method must be 'quantile' or 'equal_width'")

    # Convert to nice labels
    if labels == "index":
        # Bin 1..K based on category order
        if hasattr(cats, "cat"):
            k = len(cats.cat.categories)
            mapping = {cat: f"Bin {i+1}" for i, cat in enumerate(cats.cat.categories)}
            df[out_col] = cats.map(mapping)
        else:
            df[out_col] = cats.astype(str)
    else:
        # range labels
        df[out_col] = cats.astype(str)

    return df


# ------------------------------------------
# SCATTER: x=urgency, y=avg travel time, color=income_bin (derived)
# ------------------------------------------
# ------------------------------------------
# SCATTER: x=urgency, y=avg travel time
# color = scenario, no income bins
# ------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_raw_individual_scatter_by_scenario(
    algo_runs,                      # dict scenario -> list of run dirs
    episode_range=None,
    episodes=None,
    last_n=30,

    urgency_col="urgency",
    travel_time_col="travel_time",
    kind_col="kind",
    plot_kind="AV",                 # "AV", "Human", "All"

    travel_time_unit="minutes",     # "minutes" or "seconds"
    urgency_decimals=1,             # set to 1 for 0.1,0.2,...,1.0 (helps avoid float noise)

    # optional: jitter to reduce overlap when urgency is discrete
    jitter_x=0.0,                   # e.g. 0.01; set 0.0 for none
    jitter_seed=0,

    # optional downsampling for huge datasets (keeps plot responsive)
    max_points_per_scenario=None,   # e.g. 200000 or None

    figsize=(10.5, 5.2),
    s=8,
    alpha=0.01,
    save_path=None,
    show=True,
):
    def to_minutes(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        return s / 60.0 if travel_time_unit.lower().startswith("sec") else s

    rng = np.random.default_rng(jitter_seed)

    fig, ax = plt.subplots(figsize=figsize)

    for sc, run_dirs in algo_runs.items():
        parts = []

        for run_dir in run_dirs:
            df = load_episodes(run_dir, n_last=last_n, episode_range=episode_range, episodes=episodes)
            if df is None or df.empty:
                continue

            # filter kind if column exists
            if plot_kind != "All" and kind_col in df.columns:
                df = df[df[kind_col] == plot_kind]
            if df.empty:
                continue

            needed = [urgency_col, travel_time_col]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"[WARN] {sc} / {run_dir}: missing {missing}. Skipping run.")
                continue

            tmp = df[needed].copy()
            tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce").round(urgency_decimals)
            tmp[travel_time_col] = to_minutes(tmp[travel_time_col])
            tmp = tmp.dropna(subset=[urgency_col, travel_time_col])

            if not tmp.empty:
                parts.append(tmp)

        if not parts:
            print(f"[WARN] Scenario '{sc}': no usable rows.")
            continue

        data = pd.concat(parts, ignore_index=True)

        # optional downsample for speed
        if max_points_per_scenario is not None and len(data) > max_points_per_scenario:
            data = data.sample(n=max_points_per_scenario, random_state=jitter_seed).reset_index(drop=True)

        x = data[urgency_col].to_numpy(dtype=float)
        y = data[travel_time_col].to_numpy(dtype=float)

        if jitter_x and jitter_x > 0:
            x = x + rng.uniform(-jitter_x, jitter_x, size=len(x))

        ax.scatter(
            x, y,
            s=s,
            alpha=alpha,
            marker="o",
            label=f"{sc}",
        )

    ax.set_xlabel(f"Income", fontsize=16)
    ax.set_ylabel("Travel time (minutes)", fontsize=16)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_ylim(50, 65)

    # If you know urgency should be 0.1..1.0 you can uncomment these:
    # ticks = np.round(np.arange(0.1, 1.01, 0.1), 1)
    # ax.set_xticks(ticks)
    # ax.set_xlim(0.05, 1.05)

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

        
        "karma pricing": [
            r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_4/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_15_karma_pricing_minimum_price_4/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_16_karma_pricing_minimum_price_4/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_17_karma_pricing_minimum_price_4/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4/episodes",
            #r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4/episodes",
        ],

        



        "monetary pricing 47": [
            r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_2_vot_reward/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_2_vot_reward/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_2_vot_reward/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_2_vot_reward/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_2_vot_reward/episodes",
    ],


    }

    plot_raw_individual_scatter_by_scenario(
        ALGO_RUNS,
        episode_range=(1000, 1320),   # <-- multiple iterations/episodes
        urgency_col="income",
        travel_time_col="travel_time",
        travel_time_unit="minutes",
        plot_kind="AV",
        urgency_decimals=1,
        jitter_x=0.01,                # optional, helps when urgency is discrete
        s=8,
        alpha=0.4,
        save_path="imgs/raw_individual_scatter.png",
        show=True,
    )



