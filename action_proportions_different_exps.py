

#!/usr/bin/env python3
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ==========================================================
# ---- MATPLOTLIB STYLE (paper-ready) ----
# ==========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.22,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})

# ==========================================================
# ---- USER PARAMETERS ----
# ==========================================================

SYSTEM_OPTIMUM = {
    0: {"mean": 39.750, "std": 2.861},
    1: {"mean": 176.250, "std": 6.098},
    2: {"mean": 84.000, "std": 4.472},
}

SYSTEM_OPT_LINESTYLE = (0, (5, 2))   # cleaner than "--"
SYSTEM_OPT_LINEWIDTH = 1.8
SYSTEM_OPT_COLOR = "black"
SYSTEM_OPT_BAND_COLOR = "0.6"
SYSTEM_OPT_BAND_ALPHA = 0.12

ALGO_RUNS = {
    # ---- KEEP YOUR CURRENT ALGO_RUNS BLOCK HERE AS-IS ----
}

# --- what to count ---
ACTION_COL = "action"
ACTIONS = [0, 1, 2]
KIND_COL = "kind"
PLOT_KIND = "AV"          # "AV", "Human", or "All"

# --- plotting ---
SMOOTH = 20
MAX_EPISODE = 1320
XLIM = (0, MAX_EPISODE)
YLIM = (0, 300)           # since total is 300 agents, this helps comparability
SAVE_PNG = "imgs/action_counts_subplots.png"
SAVE_PDF = "imgs/action_counts_subplots.pdf"

# For paper figures, usually omit a large internal title and use the caption instead.
TITLE = None

XLABEL = "Training iteration"
YLABEL = "Number of AVs choosing action"

# Better proportions for a 3-panel paper figure
FIGSIZE = (12, 3.2)

# Lighter uncertainty band for learned policies
BAND_ALPHA = 0.10
LINE_LW = 2.0

YLIMS = {
    0: (0, 80),
    1: (120, 280),
    2: (0, 120),
}


"""    "monetary pricing":[
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
            r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
            r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_variable_start_times_smaller_obs/episodes",
        ],"""

ALGO_RUNS = {   




    "fee route 1":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
    ],
    "fee route 1 0.5":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_route1_fee_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_route1_fee_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_route1_fee_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_route1_fee_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_route1_fee_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_route1_fee_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_route1_fee_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_route1_fee_0_5/episodes",
    ],
    "fee route 1 1":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_route1_fee_1/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_route1_fee_1/episodes",
    ],
    "fee route 1 1.5":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_route1_fee_1_5/episodes",
    ],
    "fee route 1 2":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_route1_fee_2/episodes",
    ],
    "fee route 1 5":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_route1_fee_5/episodes",
    ],
    "fee route 1 1.5 1.5":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_route1_fee_1_5_route0_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_route1_fee_1_5_route0_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_route1_fee_1_5_route0_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_route1_fee_1_5_route0_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_route1_fee_1_5_route0_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_route1_fee_1_5_route0_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_route1_fee_1_5_route0_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_route1_fee_1_5_route0_fee_1_5/episodes",
    ],

    }



# ==========================================================
# ---- HELPERS ----
# ==========================================================
def smooth_ma(y, w):
    y = np.asarray(y, dtype=float)
    if w is None or w <= 1 or len(y) < w:
        return y
    half = w // 2
    pad = np.pad(y, (half, half), mode="edge")
    out = np.convolve(pad, np.ones(w) / w, mode="valid")
    return out[:len(y)]

def get_episode_files(folder):
    if not os.path.isdir(folder):
        print(f"[WARN] not a folder: {folder}")
        return []
    files = []
    for fn in os.listdir(folder):
        if fn.startswith("ep") and fn.endswith(".csv"):
            try:
                ep = int(fn[2:-4])
                files.append((ep, fn))
            except ValueError:
                pass
    files.sort(key=lambda t: t[0])
    if MAX_EPISODE is not None:
        files = [(ep, fn) for ep, fn in files if ep <= MAX_EPISODE]
    return files

def per_episode_action_counts_for_folder(folder, action_col=ACTION_COL, actions=ACTIONS,
                                        kind=PLOT_KIND, kind_col=KIND_COL):
    rows = []
    out_cols = ["episode"] + [f"a{a}" for a in actions]

    for ep, fn in get_episode_files(folder):
        path = os.path.join(folder, fn)
        if not os.path.getsize(path):
            continue
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
        except Exception as e:
            print(f"[WARN] read failed {path}: {e}")
            continue

        if action_col not in df.columns:
            continue

        if kind != "All" and kind_col in df.columns:
            df = df[df[kind_col] == kind]

        acts = pd.to_numeric(df[action_col], errors="coerce").dropna().astype(int)
        if len(acts) == 0:
            continue

        counts = acts.value_counts()
        row = [ep] + [int(counts.get(a, 0)) for a in actions]
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=out_cols)

    return (
        pd.DataFrame(rows, columns=out_cols)
        .sort_values("episode")
        .reset_index(drop=True)
    )

def align_and_aggregate_counts_across_folders(dfs, actions=ACTIONS):
    all_eps = sorted(set().union(*[set(df["episode"]) for df in dfs if not df.empty]))
    if not all_eps:
        return pd.DataFrame()

    Y = {a: np.full((len(dfs), len(all_eps)), np.nan, dtype=float) for a in actions}

    for r, df in enumerate(dfs):
        if df.empty:
            continue
        df_map = df.set_index("episode")
        for c, ep in enumerate(all_eps):
            if ep not in df_map.index:
                continue
            for a in actions:
                Y[a][r, c] = float(df_map.loc[ep, f"a{a}"])

    out = {"episode": np.array(all_eps, dtype=int)}

    for a in actions:
        Ya = Y[a]
        n = np.sum(~np.isnan(Ya), axis=0).astype(int)
        mean = np.nanmean(Ya, axis=0)
        std = np.array(
            [np.nanstd(Ya[:, i], ddof=1) if n[i] >= 2 else (0.0 if n[i] == 1 else np.nan)
             for i in range(len(all_eps))],
            dtype=float
        )
        out[f"mean_a{a}"] = mean
        out[f"std_a{a}"] = std
        out[f"n_a{a}"] = n

    return pd.DataFrame(out)

def load_and_aggregate_for_algorithm(algo_name, folders):
    per_seed = []
    for folder in folders:
        print(f"[INFO] {algo_name}: loading {folder}")
        df = per_episode_action_counts_for_folder(folder)
        if df.empty:
            print(f"[WARN] {algo_name}: no usable episodes in {folder} for kind='{PLOT_KIND}'")
        per_seed.append(df)

    agg = align_and_aggregate_counts_across_folders(per_seed)
    if agg.empty:
        print(f"[WARN] {algo_name}: no usable data across replications.")
        return algo_name, agg, 0

    ncols = [c for c in agg.columns if c.startswith("n_a")]
    nmax = int(np.nanmax(agg[ncols].to_numpy())) if ncols else 0
    return algo_name, agg, nmax

def extract_fee_value(name):
    m = re.search(r"fee\s+(\d+)", name.lower())
    return int(m.group(1)) if m else 10**9

def make_algorithm_colors(names):
    """
    Assign consistent colors in fee order.
    """
    ordered = sorted(names, key=extract_fee_value)
    cmap = plt.get_cmap("viridis", len(ordered))
    return {name: cmap(i) for i, name in enumerate(ordered)}

def short_algo_label(name):
    fee = extract_fee_value(name)
    if fee != 10**9:
        return rf"$c(k)={fee}$"
    return name

# ==========================================================
# ---- MAIN ----
# ==========================================================
def main():
    algo_aggs = []
    for algo_name, folders in sorted(ALGO_RUNS.items(), key=lambda kv: extract_fee_value(kv[0])):
        algo_aggs.append(load_and_aggregate_for_algorithm(algo_name, folders))

    algo_aggs = [(name, agg, nmax) for (name, agg, nmax) in algo_aggs if not agg.empty]
    if not algo_aggs:
        raise ValueError("No usable data for any algorithm. Check paths/kind/MAX_EPISODE/ACTION_COL.")

    algo_colors = make_algorithm_colors([name for name, _, _ in algo_aggs])

    n_actions = len(ACTIONS)
    fig, axes = plt.subplots(
        1, n_actions,
        figsize=FIGSIZE,
        sharex=True,
        sharey=False
    )
    if n_actions == 1:
        axes = [axes]

    for ax, a in zip(axes, ACTIONS):
        # --- learned policies ---
        for algo_name, agg, nmax in algo_aggs:
            x = agg["episode"].to_numpy()
            y = agg[f"mean_a{a}"].to_numpy(dtype=float)
            s = agg[f"std_a{a}"].to_numpy(dtype=float)
            n = agg[f"n_a{a}"].to_numpy(dtype=int)

            y_plot = smooth_ma(y, SMOOTH)
            s_plot = smooth_ma(s, SMOOTH)

            color = algo_colors[algo_name]
            label = f"{short_algo_label(algo_name)}"

            (line,) = ax.plot(
                x, y_plot,
                lw=LINE_LW,
                color=color,
                label=label,
                zorder=3
            )
            lc = line.get_color()

            mask = n >= 2
            if np.any(mask):
                ax.fill_between(
                    x[mask],
                    (y_plot - s_plot)[mask],
                    (y_plot + s_plot)[mask],
                    color=lc,
                    alpha=BAND_ALPHA,
                    linewidth=0,
                    zorder=2
                )

        # --- system optimum mean ± std ---
        if a in SYSTEM_OPTIMUM:
            so_mean = SYSTEM_OPTIMUM[a]["mean"]
            so_std = SYSTEM_OPTIMUM[a]["std"]

            ax.axhspan(
                so_mean - so_std,
                so_mean + so_std,
                color=SYSTEM_OPT_BAND_COLOR,
                alpha=SYSTEM_OPT_BAND_ALPHA,
                label="System optimum ±1 std" if a == ACTIONS[0] else None,
                zorder=0,
            )

            ax.axhline(
                y=so_mean,
                color=SYSTEM_OPT_COLOR,
                linestyle=SYSTEM_OPT_LINESTYLE,
                linewidth=SYSTEM_OPT_LINEWIDTH,
                label="System optimum mean" if a == ACTIONS[0] else None,
                zorder=4
            )

        # --- subplot cosmetics ---
        panel_labels = ["a)", "b)", "c)"]
        ax.set_title(f"{panel_labels[a]} Action {a}", pad=4)

        ax.grid(axis="y")
        ax.grid(axis="x", alpha=0.10)

        if XLIM is not None:
            ax.set_xlim(*XLIM)
        if a in YLIMS:
            ax.set_ylim(*YLIMS[a])

        ax.set_xticks([0, 300, 600, 900, 1200])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.tick_params(axis="both", which="major", length=3.5, width=0.8)

    # --- global labels ---
    if TITLE is not None:
        fig.suptitle(TITLE, y=0.98)

    fig.supxlabel(XLABEL, y=0.06)
    fig.supylabel(YLABEL, x=0.02)

    # --- shared legend ---
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.2,
        handletextpad=0.5,
        #title="Monetary fee",
        title_fontsize=12.5
    )

    fig.subplots_adjust(
        top=0.60,
        bottom=0.20,
        left=0.08,
        right=0.995,
        wspace=0.20
    )

    # --- save ---
    if SAVE_PNG:
        os.makedirs(os.path.dirname(SAVE_PNG), exist_ok=True)
        plt.savefig(SAVE_PNG, bbox_inches="tight")
        print(f"[INFO] saved PNG to: {SAVE_PNG}")

    if SAVE_PDF:
        os.makedirs(os.path.dirname(SAVE_PDF), exist_ok=True)
        plt.savefig(SAVE_PDF, bbox_inches="tight")
        print(f"[INFO] saved PDF to: {SAVE_PDF}")

    plt.show()
    plt.close()

if __name__ == "__main__":
    main()