#!/usr/bin/env python3
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# 🔧 USER SETTINGS — adjust these and just run the script
# ==========================================================
CSV_PATH = "../../scenarios/monetary_pricing/training_records_monetary_pricing_300_agents_fee_0_1/episodes/ep1.csv"        # path to your CSV file
OUTPUT_DIR = "quintile_plots"            # folder for saved plots
SHOW_PLOTS = True                       # set to True to open plots interactively
N_BINS = 30                              # histogram bins if not using integer bins
# ==========================================================


def read_agents_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV and keep only income & start_time columns."""
    df = pd.read_csv(csv_path, engine="python")
    for col in ("income", "start_time"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found. Columns: {list(df.columns)}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["income", "start_time"]).copy()
    return df


def add_income_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'income_quintile' categorical column (5 quantile bins)."""
    try:
        quintiles = pd.qcut(
            df["income"], 5,
            labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]
        )
    except ValueError:
        # fallback when identical income values prevent 5 unique cut points
        quintiles = pd.qcut(df["income"], 5, labels=None, duplicates="drop")
        n_bins = quintiles.cat.categories.size
        labels = [f"Q{i+1}" for i in range(n_bins)]
        quintiles = pd.qcut(df["income"], n_bins, labels=labels)
    df = df.copy()
    df["income_quintile"] = quintiles
    return df


def choose_bins(all_values: np.ndarray, n_bins: int):
    """Choose consistent bins across groups (prefer integer bins if reasonable)."""
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        raise ValueError("No finite start_time values found.")
    min_t, max_t = float(np.nanmin(finite)), float(np.nanmax(finite))
    span = max_t - min_t
    # Use integer bins if values look integer-like and span isn't huge
    use_int_bins = span <= 500 and np.allclose(finite, np.rint(finite), atol=1e-6)
    if use_int_bins:
        return np.arange(np.floor(min_t), np.ceil(max_t) + 1, 1, dtype=int)
    return n_bins


def js_distance(p: np.ndarray, q: np.ndarray, base: int = 2) -> float:
    """
    Jensen–Shannon distance between two discrete probability vectors p, q.
    Returns a value in [0, 1] when base=2.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # normalize to probabilities (sum to 1). If a group is empty, this will raise.
    ps, qs = p.sum(), q.sum()
    if ps == 0 or qs == 0:
        raise ValueError("One of the distributions has zero total mass.")
    p /= ps
    q /= qs
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    js_div = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    if base == 2:
        js_div /= np.log(2)
    return float(np.sqrt(js_div))


def pairwise_js_distance(groups: dict, bins) -> pd.DataFrame:
    """
    Compute pairwise Jensen–Shannon distance for each group’s start_time histogram.
    groups: mapping {label -> 1D array of start_time}
    bins: histogram bin edges or an int (consistent across groups)
    Returns a pandas DataFrame (square matrix).
    """
    labels = list(groups.keys())

    # Build *counts* histograms on shared bins, then normalize to probabilities
    hists = {}
    for label in labels:
        counts, edges = np.histogram(groups[label], bins=bins)
        hists[label] = counts.astype(float)

    # Compute distances
    dist_mat = np.zeros((len(labels), len(labels)), dtype=float)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i == j:
                dist_mat[i, j] = 0.0
            elif j < i:
                dist_mat[i, j] = dist_mat[j, i]  # symmetry
            else:
                dist_mat[i, j] = js_distance(hists[li], hists[lj])

    return pd.DataFrame(dist_mat, index=labels, columns=labels)


def plot_side_by_side_histograms(df: pd.DataFrame, bins, out_file: pathlib.Path, show: bool = False):
    """Plot all quintile histograms side-by-side with shared y-axis."""
    quintile_groups = df.groupby("income_quintile", observed=False)
    labels = [str(name) for name, _ in quintile_groups]
    n_groups = len(labels)

    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4), sharey=True)
    if n_groups == 1:
        axes = [axes]

    for ax, (group_name, group_df) in zip(axes, quintile_groups):
        ax.hist(group_df["start_time"], bins=bins)
        ax.set_title(str(group_name))
        ax.set_xlabel("start_time")
        ax.set_ylabel("count")

    plt.suptitle("Start Time Distributions by Income Quintile", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_js_heatmap(js_df: pd.DataFrame, out_file: pathlib.Path, show: bool = False):
    """Plot a heatmap of the pairwise Jensen–Shannon distance matrix."""
    fig = plt.figure(figsize=(5 + 0.5 * len(js_df), 4 + 0.5 * len(js_df)))
    ax = plt.gca()
    im = ax.imshow(js_df.values, aspect="auto")
    # default colormap; don’t set any specific colors
    ax.set_xticks(range(len(js_df.columns)))
    ax.set_xticklabels(js_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(js_df.index)))
    ax.set_yticklabels(js_df.index)
    ax.set_title("Jensen–Shannon Distance (0 = identical, 1 = very different)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    csv_path = pathlib.Path(CSV_PATH)
    out_dir = pathlib.Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path.resolve()}")

    # Load & bin
    df = read_agents_csv(str(csv_path))
    df = add_income_quintiles(df)
    bins = choose_bins(df["start_time"].to_numpy(), N_BINS)

    # Combined side-by-side histograms
    hist_out = out_dir / "start_time_distributions_by_quintile.png"
    plot_side_by_side_histograms(df, bins=bins, out_file=hist_out, show=SHOW_PLOTS)
    print(f"✅ Combined histogram saved to: {hist_out.resolve()}")

    # Prepare groups for JS distance
    groups = {str(name): group_df["start_time"].to_numpy()
              for name, group_df in df.groupby("income_quintile", observed=False)}

    # Pairwise JS distance matrix (0..1)
    js_df = pairwise_js_distance(groups, bins=bins)

    # Save CSV + heatmap
    js_csv = out_dir / "js_distance_matrix.csv"
    js_df.to_csv(js_csv, index=True)
    print(f"✅ Jensen–Shannon distance matrix saved to: {js_csv.resolve()}")
    print("\nPairwise JS distances (0 = identical, 1 = very different):\n")
    print(js_df.round(3).to_string())

    heatmap_out = out_dir / "js_distance_heatmap.png"
    plot_js_heatmap(js_df, out_file=heatmap_out, show=SHOW_PLOTS)
    print(f"✅ Heatmap saved to: {heatmap_out.resolve()}")


if __name__ == "__main__":
    main()
