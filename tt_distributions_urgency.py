import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

ALGO_RUNS = {
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
}

# --- selection (same priority rules as your other script) ---
LAST_N_FILES = 10             # used only if EPISODES and EPISODE_RANGE are None
EPISODE_RANGE = (1200, 1320)  # inclusive; set None to disable
EPISODES = None               # e.g. [5800, 5801, 5900]; set None to disable

KIND_FILTER = "AV"            # "AV", "Human", or None

# --- urgency plotting ---
URGENCY_COL = "urgency"       # column name in CSV
URGENCY_ROUND_DECIMALS = 1    # urgency values are 0.1,0.2,...,1.0 -> round(1) is ideal
URGENCIES_TO_PLOT = None      # e.g. [0.1, 0.5, 1.0] or None for all

# Optional: restrict which actions are included in the urgency distributions
ACTIONS_FILTER = None         # e.g. [0, 1] or None for all

# --- plot style ---
BINS = 50
DENSITY = False
OUT_PATH = "imgs/grouped_travel_time_distributions_by_urgency.png"

# Layout: avoid 1x10 tiny subplots if you have many urgencies
N_COLS = 5  # number of subplot columns in the grid

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

def load_selected_csvs_by_urgency(
    folder: str,
    last_n: int | None,
    kind_filter: str | None,
    urgency_col: str,
    urgency_round_decimals: int = 1,
    episode_range: tuple[int, int] | None = None,   # (start, end) inclusive
    episodes: list[int] | None = None,              # explicit episode numbers
    actions_filter: list[int] | None = None,
):
    """
    Read selected episode CSV files and return:
      travel_by_urgency: dict[urgency(float) -> list of travel_time]
      selected_files: list of processed filepaths

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

    travel_by_urgency: dict[float, list[float]] = {}

    for ep, path in chosen:
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
        except Exception as e:
            print(f"[WARN] Failed reading {path}: {e}")
            continue

        needed = {"travel_time", urgency_col}
        if not needed.issubset(df.columns):
            continue

        df = df.dropna(how="all")
        df["travel_time"] = pd.to_numeric(df["travel_time"], errors="coerce")
        df[urgency_col] = pd.to_numeric(df[urgency_col], errors="coerce")
        df = df.dropna(subset=["travel_time", urgency_col])

        if kind_filter is not None and "kind" in df.columns:
            df = df[df["kind"] == kind_filter]

        if actions_filter is not None and "action" in df.columns:
            df["action"] = pd.to_numeric(df["action"], errors="coerce")
            df = df.dropna(subset=["action"])
            df["action"] = df["action"].astype(int)
            df = df[df["action"].isin(actions_filter)]

        # Avoid float precision splitting by rounding (0.30000004 -> 0.3)
        urg_key = df[urgency_col].round(urgency_round_decimals)

        for u, sub in df.groupby(urg_key):
            travel_by_urgency.setdefault(float(u), []).extend(sub["travel_time"].to_list())

    selected_paths = [p for _, p in chosen]
    return travel_by_urgency, selected_paths

def merge_travel_dicts(dicts: list[dict[float, list[float]]]) -> dict[float, list[float]]:
    merged: dict[float, list[float]] = {}
    for d in dicts:
        for k, vals in d.items():
            merged.setdefault(k, []).extend(vals)
    return merged

# ============================================================
# Plotting (by urgency)
# ============================================================

def plot_grouped_distributions_by_urgency(
    algo_runs: dict[str, list[str]],
    last_n_files: int | None,
    kind_filter: str | None,
    urgency_col: str,
    urgency_round_decimals: int,
    bins: int,
    density: bool,
    out_path: str,
    title: str | None = None,
    urgencies_to_plot: list[float] | None = None,
    episode_range: tuple[int, int] | None = None,
    episodes: list[int] | None = None,
    actions_filter: list[int] | None = None,
    ncols: int = 5,
):
    # ---- Load + aggregate per group ----
    group_data: dict[str, dict[float, list[float]]] = {}

    for group_name, folders in algo_runs.items():
        per_folder = []
        for folder in folders:
            d, _selected = load_selected_csvs_by_urgency(
                folder,
                last_n=last_n_files,
                kind_filter=kind_filter,
                urgency_col=urgency_col,
                urgency_round_decimals=urgency_round_decimals,
                episode_range=episode_range,
                episodes=episodes,
                actions_filter=actions_filter,
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

    # ---- Determine urgencies to plot ----
    all_urgencies = sorted(set().union(*[set(d.keys()) for d in group_data.values()]))

    if urgencies_to_plot is None:
        urgencies = all_urgencies
    else:
        wanted = [round(float(u), urgency_round_decimals) for u in urgencies_to_plot]
        urgencies = [u for u in wanted if u in all_urgencies]

    if not urgencies:
        raise ValueError("No urgencies available to plot after filtering.")

    # ---- Shared bin edges per urgency (so overlays are comparable) ----
    bin_edges_by_urgency: dict[float, np.ndarray] = {}
    for u in urgencies:
        vals_all = []
        for g in group_data:
            if u in group_data[g]:
                vals_all.append(np.asarray(group_data[g][u], dtype=float))
        vals_all = np.concatenate(vals_all)
        vmin, vmax = float(np.min(vals_all)), float(np.max(vals_all))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        bin_edges_by_urgency[u] = np.linspace(vmin, vmax, bins + 1)

    # ---- Grid layout ----
    n = len(urgencies)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.3 * nrows),
        sharey=True
    )
    axs = np.array(axs).reshape(-1)

    # Hide unused axes
    for ax in axs[n:]:
        ax.axis("off")

    group_names = list(group_data.keys())

    for ax, u in zip(axs[:n], urgencies):
        edges = bin_edges_by_urgency[u]

        for g in group_names:
            if u not in group_data[g]:
                continue
            vals = np.asarray(group_data[g][u], dtype=float)
            ax.hist(
                vals,
                bins=edges,
                density=density,
                alpha=0.35,
                label=g,
                edgecolor="none"
            )

        ax.set_title(f"Urgency {u:.{urgency_round_decimals}f}")
        ax.set_xlabel("travel_time")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # One y-label (left-most visible axis)
    # Find first axis that is on and set its ylabel
    for ax in axs:
        if ax.axison:
            ax.set_ylabel("Density" if density else "Count")
            break

    if title is None:
        base = "Travel-time distributions by urgency"
        if kind_filter is not None:
            base += f" (kind={kind_filter})"
        if actions_filter is not None:
            base += f" (actions={actions_filter})"
        title = base

    fig.suptitle(title, y=1.02)

    # ---- Single legend ----
    # Use handles from first plotted axis
    first_on = next(ax for ax in axs if ax.axison)
    handles, labels = first_on.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
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
    plot_grouped_distributions_by_urgency(
        algo_runs=ALGO_RUNS,
        last_n_files=LAST_N_FILES,
        kind_filter=KIND_FILTER,
        urgency_col=URGENCY_COL,
        urgency_round_decimals=URGENCY_ROUND_DECIMALS,
        bins=BINS,
        density=DENSITY,
        out_path=OUT_PATH,
        title=None,
        urgencies_to_plot=URGENCIES_TO_PLOT,
        episode_range=EPISODE_RANGE,
        episodes=EPISODES,
        actions_filter=ACTIONS_FILTER,
        ncols=N_COLS,
    )