import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

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

def load_episodes(
    folder: str,
    n_last: int = 30,
    episode_range=None,
    episodes=None,
):
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


def plot_population_share_income_hourlysalary_urgency_vot_paper(
    run_sources,
    episode_range=None,
    episodes=None,
    last_n=1,
    income_col="income",
    urgency_col="urgency",
    id_col="id",
    kind_col="kind",
    plot_kind="AV",
    dedupe_agents=True,
    hourly_salary_divisor=(30.0 * 8.0),

    # binning
    hourly_salary_bin_width=None,
    urgency_bin_width=None,
    vot_bin_width=2,
    hourly_salary_n_bins=18,
    vot_n_bins=18,

    # figure options
    figsize=(10, 4),
    dpi=300,
    show_median=True,
    color_salary="#4C78A8",
    color_urgency="#F58518",
    color_vot="#54A24B",
    edgecolor="white",
    linewidth=0.8,

    save_path=None,
    show=True,
):
    """
    Produces 3 publication-style subplots:
      (a) Share of population by hourly salary = income / 30 / 8
      (b) Share of population by urgency
      (c) Share of population by value of time = urgency * hourly salary
    """

    # -------------------------
    # Flatten run_sources input
    # -------------------------
    if isinstance(run_sources, dict):
        run_dirs = []
        for _, dirs in run_sources.items():
            run_dirs.extend(dirs)
    else:
        run_dirs = list(run_sources)

    if not run_dirs:
        print("[WARN] No run directories provided.")
        return None

    # -------------------------
    # Load and concatenate data
    # -------------------------
    dfs = []
    for run_dir in run_dirs:
        df = load_episodes(
            run_dir,
            n_last=last_n,
            episode_range=episode_range,
            episodes=episodes,
        )
        if df is None or df.empty:
            continue
        dfs.append(df)

    if not dfs:
        print("[WARN] No episode data found.")
        return None

    df = pd.concat(dfs, ignore_index=True)

    # -------------------------
    # Filter kind if requested
    # -------------------------
    if plot_kind != "All" and kind_col in df.columns:
        df = df[df[kind_col] == plot_kind]

    if df.empty:
        print("[WARN] No rows left after filtering.")
        return None

    needed = [income_col, urgency_col]
    if dedupe_agents:
        needed.append(id_col)
    if dedupe_agents and "episode" in df.columns:
        needed.append("episode")

    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[WARN] Missing required columns: {missing}")
        return None

    tmp = df[needed].copy()
    tmp[income_col] = pd.to_numeric(tmp[income_col], errors="coerce")
    tmp[urgency_col] = pd.to_numeric(tmp[urgency_col], errors="coerce")
    tmp = tmp.dropna(subset=[income_col, urgency_col])

    if tmp.empty:
        print("[WARN] No valid income/urgency data after numeric conversion.")
        return None

    # -------------------------
    # Optional deduplication
    # -------------------------
    if dedupe_agents:
        tmp[id_col] = pd.to_numeric(tmp[id_col], errors="coerce")
        tmp = tmp.dropna(subset=[id_col])
        tmp[id_col] = tmp[id_col].astype(int)

        if "episode" in tmp.columns:
            tmp = (
                tmp.sort_values(["id", "episode"])
                   .groupby(id_col, as_index=False)
                   .tail(1)
            )
        else:
            tmp = tmp.groupby(id_col, as_index=False).tail(1)

    # -------------------------
    # Derived quantities
    # -------------------------
    tmp["hourly_salary"] = tmp[income_col] / hourly_salary_divisor
    tmp["value_of_time"] = tmp[urgency_col] * tmp["hourly_salary"]

    # -------------------------
    # Plot styling
    # -------------------------
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    def percent_fmt(x, pos):
        return f"{x:.0f}%"

    def make_bins(s, bin_width=None, n_bins=None):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return None
        if bin_width is not None:
            x_min = np.floor(s.min() / bin_width) * bin_width
            x_max = np.ceil(s.max() / bin_width) * bin_width
            if x_max == x_min:
                x_max = x_min + bin_width
            return np.arange(x_min, x_max + bin_width, bin_width)
        return n_bins

    def plot_share_distribution(ax, series, title, xlabel, color, bin_width=None, n_bins=None, discrete=False):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Population share (%)")
            return

        # Exact bar plot for discrete urgency or low-cardinality support
        if discrete or (bin_width is None and s.nunique() <= 20):
            counts = s.value_counts(normalize=True).sort_index() * 100.0
            x = counts.index.to_numpy(dtype=float)
            y = counts.to_numpy(dtype=float)

            if len(x) > 1:
                widths = np.diff(x)
                width = np.median(widths) * 0.8
            else:
                width = 0.8

            ax.bar(
                x, y,
                width=width,
                color=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=0.95,
            )
        else:
            bins = make_bins(s, bin_width=bin_width, n_bins=n_bins)
            counts, edges = np.histogram(s, bins=bins)
            shares = (counts / counts.sum()) * 100.0 if counts.sum() > 0 else counts
            centers = (edges[:-1] + edges[1:]) / 2
            widths = np.diff(edges)

            ax.bar(
                centers, shares,
                width=widths * 0.9,
                align="center",
                color=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=0.95,
            )

        if show_median:
            med = np.median(s)
            ax.axvline(med, linestyle="--", linewidth=1.2, color="black", alpha=0.8)
            ax.text(
                med, ax.get_ylim()[1] * 0.92,
                "median",
                rotation=90,
                ha="leftgit",
                va="top",
                fontsize=12,
                color="black",
            )

        ax.set_title(title, pad=8)
        ax.set_xlabel(xlabel)
        #ax.set_ylabel("Population share (%)")
        ax.yaxis.set_major_formatter(FuncFormatter(percent_fmt))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

    # -------------------------
    # Create figure
    # -------------------------
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    plot_share_distribution(
        axes[0],
        tmp["hourly_salary"],
        title="a) Hourly salary",
        xlabel="Hourly salary",
        color=color_salary,
        bin_width=hourly_salary_bin_width,
        n_bins=hourly_salary_n_bins,
        discrete=False,
    )

    plot_share_distribution(
        axes[1],
        tmp[urgency_col],
        title="b) Urgency",
        xlabel="Urgency",
        color=color_urgency,
        bin_width=urgency_bin_width,
        n_bins=None,
        discrete=True,
    )

    plot_share_distribution(
        axes[2],
        tmp["value_of_time"],
        title="c) Value of time",
        xlabel="Value of time",
        color=color_vot,
        bin_width=vot_bin_width,
        n_bins=vot_n_bins,
        discrete=False,
    )

    axes[0].set_ylabel("Population share (%)")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.tight_layout(w_pad=2.0)

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig

ALGO_RUNS = {   


    "monetary pricing":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer_diverse_start_times_smaller_training_obs/episodes",
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
        ],


    }

plot_population_share_income_hourlysalary_urgency_vot_paper(
    ALGO_RUNS,
    episode_range=(0, 1320),   # better for a true cross-sectional snapshot
    income_col="income",
    urgency_col="urgency",
    plot_kind="AV",
    dedupe_agents=False,
    hourly_salary_bin_width=2,
    urgency_bin_width=None,
    vot_bin_width=2,
    save_path="imgs/pop_share_hourlysalary_urgency_vot_paper.pdf",
    show=True,
)


