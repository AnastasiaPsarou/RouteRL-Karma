#!/usr/bin/env python3
"""
Traverse provided folders, read all CSV files inside them, collect travel_time
(column 0) from every valid agent row, then compute mean/std across *all*
collected travel times.

- Skips empty rows like ",,,,,,,,"
- Skips headers automatically
- Saves results to a CSV next to this script by default

Run:
    python travel_time_stats.py
"""

from __future__ import annotations

import csv
import os
from glob import glob
from typing import List, Optional, Tuple

import numpy as np


# =========================
# USER SETTINGS (EDIT THIS)
# =========================
FOLDERS = [
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_9_fee_0_25/episodes",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_10_fee_0_25/episodes",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_11_fee_0_25/episodes",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_12_fee_0_25/episodes",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_13_fee_0_25/episodes",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_14_fee_0_25/episodes",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_15_fee_0_25/episodes",
]

OUTPUT_CSV = None  # None -> saves to same folder as this script, as "travel_time_summary.csv"
USE_SAMPLE_STD = False  # False -> population std (ddof=0), True -> sample std (ddof=1)
# =========================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def is_header_row(row: List[str]) -> bool:
    """Heuristic: if first cell can't be parsed as float, assume header."""
    if not row:
        return True
    try:
        float(row[0])
        return False
    except ValueError:
        return True


def extract_travel_times_from_csv(path: str) -> List[float]:
    """
    Extract travel_time values from a CSV.
    travel_time is expected at column 0.
    """
    values: List[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            return values

        # decide if header
        rows_iter = reader
        if first_row and not is_header_row(first_row):
            # treat first row as data
            rows = [first_row]
            rows.extend(rows_iter)
        else:
            rows = rows_iter

        for row in rows:
            # skip empty rows like ",,,,,,,,"
            if not row or all(c.strip() == "" for c in row):
                continue

            # need at least column 0
            if len(row) < 1:
                continue

            try:
                travel_time = float(row[0].strip())
            except ValueError:
                continue

            if np.isnan(travel_time):
                continue

            values.append(travel_time)

    return values


def mean_std(values: List[float], use_sample_std: bool) -> Optional[Tuple[float, float, int]]:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    ddof = 1 if use_sample_std and arr.size > 1 else 0
    return float(arr.mean()), float(arr.std(ddof=ddof)), int(arr.size)


def main() -> None:
    if not FOLDERS:
        raise SystemExit("FOLDERS is empty. Add at least one folder path.")

    # validate folders
    bad = [d for d in FOLDERS if not os.path.isdir(d)]
    if bad:
        raise SystemExit("These folders are not valid directories:\n" + "\n".join(bad))

    # gather CSVs
    csv_files: List[str] = []
    for folder in FOLDERS:
        csv_files.extend(glob(os.path.join(folder, "*.csv")))

    if not csv_files:
        raise SystemExit("No CSV files found in provided folders.")

    all_travel_times: List[float] = []
    per_file_rows = []

    for p in sorted(csv_files):
        vals = extract_travel_times_from_csv(p)
        all_travel_times.extend(vals)
        per_file_rows.append((p, len(vals)))
        print(f"{os.path.basename(p):>30s} | travel_time rows: {len(vals)}")

    stats = mean_std(all_travel_times, USE_SAMPLE_STD)
    if stats is None:
        raise SystemExit("No valid travel_time values found across all files.")

    mean_val, std_val, n = stats
    print("\n" + "=" * 60)
    print(f"Total CSV files: {len(csv_files)}")
    print(f"Total travel_time samples: {n}")
    print(f"Mean(travel_time): {mean_val:.10f}")
    print(f"Std(travel_time):  {std_val:.10f}")
    print("=" * 60)

    out_path = OUTPUT_CSV or os.path.join(SCRIPT_DIR, "travel_time_summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder_count", len(FOLDERS)])
        w.writerow(["csv_file_count", len(csv_files)])
        w.writerow(["total_samples", n])
        w.writerow(["mean_travel_time", mean_val])
        w.writerow(["std_travel_time", std_val])
        w.writerow([])
        w.writerow(["file_path", "n_travel_time_rows"])
        for p, cnt in per_file_rows:
            w.writerow([p, cnt])

    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
