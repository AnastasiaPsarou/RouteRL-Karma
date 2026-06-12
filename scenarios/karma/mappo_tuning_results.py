#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG — EDIT THESE VALUES ONLY
# ============================================================

RECORDS_ROOT = Path("tuning_records")
RESULTS_CSV = Path("mappo_tuning_results.csv")

OUTPUT_DIR = Path("mappo_heatmaps")

LAST_N_EPISODES = 20

# Put the exact column name from ep*.csv here.
# If None, the script will try to auto-detect it.
METRIC_COLUMN = None
# Example:
# METRIC_COLUMN = "avg_travel_time"

# Heatmap layout
X_PARAM = "param_lr_actor"
Y_PARAM = "param_lr_critic"

# Makes one heatmap per entropy value.
# Set to None if you want only one heatmap.
FACET_PARAM = "param_entropy_coef"

# ============================================================


EP_RE = re.compile(r"^ep(\d+)\.csv$", re.IGNORECASE)
SEED_RE = re.compile(r"^(?P<run_name>.+)_seed_(?P<env_seed>\d+)$")

DEFAULT_METRIC_CANDIDATES = [
    "avg_travel_time",
    "average_travel_time",
    "mean_travel_time",
    "avg_tt",
    "mean_tt",
    "travel_time",
    "total_average_travel_time",
    "system_avg_travel_time",
]


def numeric_ep_files(episodes_dir: Path) -> List[Tuple[int, Path]]:
    files = []
    for p in episodes_dir.glob("ep*.csv"):
        m = EP_RE.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return sorted(files, key=lambda x: x[0])


def detect_metric_column(csv_file: Path, candidates: Iterable[str]) -> str:
    df = pd.read_csv(csv_file, nrows=5)
    cols = list(df.columns)

    for c in candidates:
        if c in cols:
            return c

    for c in cols:
        low = c.lower()
        if ("travel" in low and "time" in low) or low in {"tt", "avg_tt", "mean_tt"}:
            return c

    raise ValueError(
        f"Could not auto-detect travel-time column in {csv_file}\n"
        f"Available columns: {cols}\n"
        f"Set METRIC_COLUMN manually at the top of the script."
    )


def mean_metric_in_file(csv_file: Path, metric_column: str) -> float:
    df = pd.read_csv(csv_file)

    if metric_column not in df.columns:
        raise KeyError(
            f"Column {metric_column!r} not found in {csv_file}\n"
            f"Available columns: {list(df.columns)}"
        )

    return float(pd.to_numeric(df[metric_column], errors="coerce").mean())


def score_seed_folder(run_folder: Path) -> Optional[Dict[str, object]]:
    episodes_dir = run_folder / "episodes"

    if not episodes_dir.exists():
        return None

    files = numeric_ep_files(episodes_dir)

    if not files:
        return None

    selected = files[-LAST_N_EPISODES:]
    selected_paths = [p for _, p in selected]

    metric_col = METRIC_COLUMN or detect_metric_column(
        selected_paths[-1],
        DEFAULT_METRIC_CANDIDATES,
    )

    file_means = [
        mean_metric_in_file(p, metric_col)
        for p in selected_paths
    ]

    m = SEED_RE.match(run_folder.name)

    if m:
        run_name = m.group("run_name")
        env_seed = int(m.group("env_seed"))
    else:
        run_name = run_folder.name
        env_seed = np.nan

    return {
        "records_folder_name": run_folder.name,
        "run_name": run_name,
        "env_seed": env_seed,
        "metric_column": metric_col,
        "first_episode_used": selected[0][0],
        "last_episode_used": selected[-1][0],
        "num_episode_files_used": len(selected),
        "seed_score": float(np.nanmean(file_means)),
        "records_folder": str(run_folder),
    }


def load_results_csv() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)

    if "env_seed" not in df.columns and "seed" in df.columns:
        df["env_seed"] = df["seed"]

    if "trial_name" not in df.columns and "run_name" not in df.columns:
        df["run_name"] = [f"grid_{i:04d}" for i in range(1, len(df) + 1)]
    elif "trial_name" in df.columns and "run_name" not in df.columns:
        df["run_name"] = df["trial_name"]

    return df


def merge_scores_with_params(scores: pd.DataFrame) -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        print(f"Warning: {RESULTS_CSV} not found. Heatmaps may be limited.")
        return scores

    params = load_results_csv()

    join_cols = ["run_name"]

    if "env_seed" in params.columns and "env_seed" in scores.columns:
        join_cols.append("env_seed")

    return scores.merge(
        params,
        on=join_cols,
        how="left",
        suffixes=("", "_result"),
    )


def make_heatmaps(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    value_col = "mean_last20_travel_time"

    required = [X_PARAM, Y_PARAM, value_col]

    if FACET_PARAM is not None:
        required.append(FACET_PARAM)

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if FACET_PARAM is None:
        groups = [(None, df)]
    else:
        groups = list(df.groupby(FACET_PARAM, dropna=False))

    for facet_value, sub in groups:
        pivot = sub.pivot_table(
            index=Y_PARAM,
            columns=X_PARAM,
            values=value_col,
            aggfunc="mean",
        )

        if pivot.empty:
            continue

        fig_width = max(7, 1.2 * len(pivot.columns))
        fig_height = max(5, 0.8 * len(pivot.index))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        im = ax.imshow(pivot.values, aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean avg travel time")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])

        ax.set_xlabel(X_PARAM)
        ax.set_ylabel(Y_PARAM)

        title = f"Mean travel time over last {LAST_N_EPISODES} episodes"

        if FACET_PARAM is not None:
            title += f"\n{FACET_PARAM} = {facet_value}"

        ax.set_title(title)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3g}", ha="center", va="center")

        fig.tight_layout()

        if FACET_PARAM is None:
            suffix = "all"
        else:
            suffix = f"{FACET_PARAM}_{facet_value}"

        safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(suffix))

        out_path = OUTPUT_DIR / f"heatmap_{Y_PARAM}_vs_{X_PARAM}_{safe_suffix}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"Saved: {out_path}")


def main() -> None:
    rows = []

    for run_folder in sorted(RECORDS_ROOT.iterdir()):
        if not run_folder.is_dir():
            continue

        scored = score_seed_folder(run_folder)

        if scored is not None:
            rows.append(scored)

    if not rows:
        raise RuntimeError(f"No valid episode folders found under {RECORDS_ROOT}")

    seed_scores = pd.DataFrame(rows)
    merged = merge_scores_with_params(seed_scores)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_scores_path = OUTPUT_DIR / "seed_level_scores.csv"
    merged.to_csv(seed_scores_path, index=False)
    print(f"Saved: {seed_scores_path}")

    param_cols = [c for c in merged.columns if c.startswith("param_")]

    if param_cols:
        combo_cols = param_cols
    else:
        combo_cols = ["run_name"]

    combo_scores = (
        merged.groupby(combo_cols, dropna=False)["seed_score"]
        .agg(
            mean_last20_travel_time="mean",
            std_last20_travel_time="std",
            n_seed_runs="count",
        )
        .reset_index()
    )

    combo_scores_path = OUTPUT_DIR / "combination_scores_averaged_across_seeds.csv"
    combo_scores.to_csv(combo_scores_path, index=False)
    print(f"Saved: {combo_scores_path}")

    make_heatmaps(combo_scores)


if __name__ == "__main__":
    main()