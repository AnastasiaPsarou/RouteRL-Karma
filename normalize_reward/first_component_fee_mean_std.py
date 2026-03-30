#!/usr/bin/env python3
"""
Read multiple CSV files and for each file compute fee / income_value per agent,
then report mean and standard deviation across agents (per file),
and finally compute:
  - mean of file means
  - average of within-file stds (instead of std across file means)
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional

import numpy as np


# =========================
# USER SETTINGS (EDIT THIS)
# =========================
CSV_FILES = [
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_9_fee_0_25/episodes/ep2.csv",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_10_fee_0_25/episodes/ep2.csv",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_11_fee_0_25/episodes/ep2.csv",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_12_fee_0_25/episodes/ep2.csv",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_13_fee_0_25/episodes/ep2.csv",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_14_fee_0_25/episodes/ep2.csv",
    "../data/training_records_monetary_pricing_300_agents_fee_0_25_long/training_records_monetary_pricing_300_agents_15_fee_0_25/episodes/ep2.csv",
]

FEE = 5.0
USE_SAMPLE_STD = False  # False -> population std (ddof=0), True -> sample std (ddof=1)
# =========================


def compute_fee_over_income_values(csv_path: str) -> List[float]:
    values: List[float] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)

        first_row = next(reader, None)
        if first_row is None:
            return values

        # Detect header
        is_header = True
        if len(first_row) > 7:
            try:
                float(first_row[7])
                is_header = False
            except ValueError:
                is_header = True

        data_rows = [first_row] if not is_header else []
        data_rows.extend(reader)

        for row in data_rows:
            if not row or all(c.strip() == "" for c in row):
                continue
            if len(row) <= 7:
                continue

            try:
                income = float(row[7].strip())
            except ValueError:
                continue

            if income == 0 or np.isnan(income):
                continue

            income_value = income / 30.0  # daily income
            values.append(FEE / income_value)

    return values


def mean_std(values: List[float]) -> Optional[tuple[float, float, int]]:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    ddof = 1 if USE_SAMPLE_STD and arr.size > 1 else 0
    return float(arr.mean()), float(arr.std(ddof=ddof)), int(arr.size)


def main():
    if not CSV_FILES:
        raise SystemExit("CSV_FILES is empty. Add at least one path.")

    missing = [p for p in CSV_FILES if not os.path.isfile(p)]
    if missing:
        raise SystemExit("These CSV files were not found:\n" + "\n".join(missing))

    file_means: List[float] = []
    file_stds: List[float] = []

    for path in CSV_FILES:
        values = compute_fee_over_income_values(path)
        stats = mean_std(values)
        if stats is None:
            print(f"[skip] {os.path.basename(path)} (no valid agent rows)")
            continue

        mean_val, std_val, n = stats
        file_means.append(mean_val)
        file_stds.append(std_val)

        print(f"File: {os.path.basename(path)}")
        print(f"Agents: {n}")
        print(f"Mean(fee / income_value): {mean_val:.10f}")
        print(f"Std(fee / income_value):  {std_val:.10f}")
        print("-" * 60)

    # ===== Aggregate across files =====
    if not file_means:
        raise SystemExit("No valid files to aggregate.")

    overall_mean_of_means = float(np.mean(file_means))
    avg_std_across_files = float(np.mean(file_stds)) if file_stds else float("nan")

    print("\nAGGREGATE ACROSS FILES")
    print(f"Files used: {len(file_means)}")
    print(f"Mean of file means:       {overall_mean_of_means:.10f}")
    print(f"Average within-file std:  {avg_std_across_files:.10f}")


if __name__ == "__main__":
    main()
