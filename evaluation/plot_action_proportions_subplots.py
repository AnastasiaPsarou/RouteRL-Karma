import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']

# ---- FONT SIZE CONSTANTS (match previous plot) ----
TITLE_FONTSIZE = 10
LABEL_FONTSIZE = 10
TICK_FONTSIZE  = 10
LEGEND_FONTSIZE = 10

def process_action_counts(directory, step=10, kind_filter="AV"):
    """
    Read ep*.csv in `directory`, sample every `step`-th file,
    and return:
      episodes:  sorted list of episode numbers (aligned)
      action_counts: dict[action] -> list of counts aligned to `episodes`
    Only rows with df['kind'] == kind_filter are used, if that column is present.
    If `kind` is missing, all rows are counted.
    """
    if not os.path.exists(directory):
        print(f"[WARN] Directory does not exist: {directory}")
        return [], {}

    csv_files = sorted(
        [fn for fn in os.listdir(directory) if fn.endswith(".csv") and fn.startswith("ep")],
        key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))),
    )
    csv_files = csv_files[::step]  # sample
    if not csv_files:
        print(f"[WARN] No CSV files found in: {directory}")
        return [], {}

    # First pass: collect all actions present (to keep columns consistent)
    actions_seen = set()
    eps_order = []
    per_file_counts = []
    for fn in csv_files:
        ep = int(''.join(filter(str.isdigit, os.path.splitext(fn)[0])))
        path = os.path.join(directory, fn)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Error reading {path}: {e}")
            continue

        if kind_filter is not None and "kind" in df.columns:
            df = df[df["kind"] == kind_filter]

        if "action" not in df.columns:
            print(f"[WARN] Missing 'action' in {path}")
            continue

        counts = df["action"].value_counts()
        if counts.empty:
            continue

        eps_order.append(ep)
        per_file_counts.append(counts)
        actions_seen.update(counts.index.tolist())

    if not eps_order:
        return [], {}

    actions_sorted = sorted(actions_seen)
    # Align into arrays
    episodes = []
    action_counts = {a: [] for a in actions_sorted}

    for ep, counts in sorted(zip(eps_order, per_file_counts), key=lambda t: t[0]):
        episodes.append(ep)
        for a in actions_sorted:
            action_counts[a].append(int(counts.get(a, 0)))

    return episodes, action_counts

def plot_action_counts(
    ax, directories, algorithm_names, process_function=process_action_counts,
    kind_filter="AV", step=10, smoothing_window=1, show_legend=True,
    legend_location="best"
):
    """
    Group directories by algorithm name (replications).
    For each algorithm:
      - compute per-episode action counts for each replication,
      - align on common episode numbers,
      - average counts across replications (mean),
      - plot one line per action of the replication-mean counts.
    """
    grouped = {}
    for d, algo in zip(directories, algorithm_names):
        eps, act_counts = process_function(d, step=step, kind_filter=kind_filter)
        if not eps or not act_counts:
            print(f"[INFO] Skipping empty result for: {d}")
            continue
        grouped.setdefault(algo, []).append((eps, act_counts))

    if not grouped:
        raise ValueError("No usable data to plot. Check paths, CSVs, or columns.")

    for algo, rep_list in grouped.items():
        all_eps = sorted(set().union(*[set(eps) for eps, _ in rep_list]))
        if not all_eps:
            continue

        all_actions = sorted(set().union(*[
            set(act_counts.keys()) for _, act_counts in rep_list
        ]))

        means_per_action = {}
        for a in all_actions:
            M = np.full((len(rep_list), len(all_eps)), np.nan, dtype=float)
            for r, (eps, act_counts) in enumerate(rep_list):
                cnts = act_counts.get(a, [])
                if not cnts:
                    continue
                rep_map = dict(zip(eps, cnts))
                for c, ep in enumerate(all_eps):
                    if ep in rep_map:
                        M[r, c] = rep_map[ep]
            mean_counts = np.nanmean(M, axis=0)
            if smoothing_window and smoothing_window > 1:
                mean_counts = smooth_data(mean_counts, smoothing_window)
            means_per_action[a] = mean_counts

        linestyle = ":" if ("adapt" in algo or "centralized" in algo) else "-"
        action_palette = ["firebrick", "teal", "peru", "navy", "salmon", "slategray", "darkviolet",
                          "goldenrod", "olive", "tab:pink", "tab:purple", "tab:cyan"]

        for idx, a in enumerate(sorted(means_per_action.keys())):
            y = means_per_action[a]
            ax.plot(
                all_eps, y,
                label=f"{algo} — action {a}",
                linewidth=2.0,
                linestyle=linestyle,
                color=action_palette[idx % len(action_palette)],
            )

    # Ax cosmetics (font sizes aligned with your previous plot)
    ax.set_xlabel("Episodes (days)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Number of agents choosing each action", fontsize=LABEL_FONTSIZE)
    ax.set_title("Counts per action (averaged across replications with same name)",
                 fontsize=TITLE_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    if show_legend:
        ax.legend(fontsize=LEGEND_FONTSIZE, loc=legend_location, ncol=3)


# ==========================================================
# ---- MULTI-SUBFIGURE VERSION (one per folder) ----
# ==========================================================
rcParams['font.family'] = 'Times New Roman'

directories = [
    "../data/training_records_10_agents/training_records_monetary_pricing_10_agents_long_1/episodes",
    "../data/training_records_10_agents/training_records_monetary_pricing_10_agents_long_2/episodes",
    "../data/training_records_10_agents/training_records_monetary_pricing_10_agents_long_3/episodes",
    "../data/training_records_10_agents/training_records_monetary_pricing_10_agents_long_4/episodes",
    "../data/training_records_10_agents/training_records_monetary_pricing_10_agents_long_5/episodes",
]
algorithm_names = ["MAPPO"] * len(directories)

SAVE_PNG = "imgs/action_counts_per_replication_10_agents.png"  # <- will be created

n_dirs = len(directories)

# Create subplots horizontally (1 row per replication)
fig, axes = plt.subplots(
    nrows=1,
    ncols=n_dirs,
    figsize=(3.5 * n_dirs, 3.5),
    dpi=300,
    sharey=True
)

if n_dirs == 1:
    axes = [axes]

# Loop over folders and plot each one separately
for i, (ax, directory, algo_name) in enumerate(zip(axes, directories, algorithm_names)):
    print(f"[INFO] Plotting {directory}")
    try:
        plot_action_counts(
            ax,
            directories=[directory],
            algorithm_names=[algo_name],
            process_function=process_action_counts,
            kind_filter="AV",
            step=10,
            smoothing_window=1,
            show_legend=(i == n_dirs - 1),   # show legend only on last subplot
            legend_location="best"
        )
        ax.set_title(f"{algo_name} (Run {i+1})", fontsize=TITLE_FONTSIZE)
    except Exception as e:
        print(f"[WARN] Skipped {directory}: {e}")
        ax.set_visible(False)

# Shared labels and layout (tight first, then suptitle)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave a bit more top margin

# Add shared title after tight layout so it doesn’t overlap
fig.suptitle(
    "Action counts per replication (one subplot per folder)",
    fontsize=TITLE_FONTSIZE,
    y=1.02  # <-- slightly higher so it clears the top subplot titles
)

fig.supxlabel("Episodes (days)", fontsize=LABEL_FONTSIZE)
fig.supylabel("Number of agents choosing each action", fontsize=LABEL_FONTSIZE)

# Compact layout: less white space between plots
plt.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.88,    # <-- leaves room for the suptitle above
    bottom=0.15,
    wspace=0.15
)

# Save to imgs/ with safe directory creation
if SAVE_PNG:
    os.makedirs(os.path.dirname(SAVE_PNG), exist_ok=True)
    fig.savefig(SAVE_PNG, dpi=300, bbox_inches="tight")
    print(f"[INFO] saved figure to: {SAVE_PNG}")

#plt.show()

