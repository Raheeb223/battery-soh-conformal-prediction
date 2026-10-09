"""
Multi-vehicle field-data pooling test
======================================

Extends field_data_exploration.py (already tested against the n=1 GitHub sample
-- see that script's docstring for the honest single-vehicle findings) to test
the actual open question: does pooling across MULTIPLE vehicles reduce the
noise/confound problem found at n=1, or does the same dip-then-recover pattern
show up in every vehicle at correlated points (which would mean it's a common-
mode confound -- e.g. genuinely seasonal -- that pooling alone cannot fix)?

THIS SCRIPT IS UNTESTED against real multi-vehicle data, because I do not yet
have more than one vehicle's data. The single-vehicle functions it imports
(load_telemetry, extract_charge_segments, estimate_capacity_per_segment,
temperature_correct) ARE tested -- see field_data_exploration.py.

WHAT I DON'T KNOW YET AND THIS SCRIPT MAY NEED ADJUSTING FOR:
  - The folder/file structure of the full 300-vehicle download is unknown to me.
    This script assumes one subfolder per vehicle, each containing that
    vehicle's data_*.csv files (or their extracted .rar contents), e.g.:
        <data_root>/vehicle_001/data_1.csv, data_2.csv, ...
        <data_root>/vehicle_002/data_1.csv, ...
    Adjust `discover_vehicle_dirs()` below if the real structure differs
    (e.g. one file per vehicle instead of one folder per vehicle).
  - `terminaltime` in the n=1 sample looks like seconds RELATIVE to each
    vehicle's own logging start (values like 56, 493880, ... 114215913 -- not
    a Unix timestamp, which would be ~1.7 billion for 2023-2026). This means
    calendar month/season CANNOT be recovered from terminaltime alone unless
    the full dataset also provides an absolute start date per vehicle, or an
    absolute timestamp column this sample didn't have. If the full dataset
    includes such a column, use it -- true calendar alignment is the correct
    fix for a seasonal confound; this script's fallback (below) is a weaker
    substitute that only checks for correlated shapes across vehicles, it does
    not truly deseasonalize.

USAGE: python multi_vehicle_pooling_test.py --data-root <folder with per-vehicle subfolders>
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

# Reuse the tested single-vehicle functions rather than duplicate them.
from field_data_exploration import (
    load_telemetry,
    extract_charge_segments,
    estimate_capacity_per_segment,
    temperature_correct,
)


def discover_vehicle_dirs(data_root: str) -> list:
    """Find per-vehicle subdirectories. ADJUST if the real download's
    structure differs (see module docstring)."""
    dirs = sorted([d for d in glob.glob(f"{data_root}/*") if os.path.isdir(d)])
    if not dirs:
        raise FileNotFoundError(
            f"No subdirectories found under {data_root}. If the full dataset "
            "instead gives one CSV/archive per vehicle at the top level, tell "
            "me the actual structure and I'll adjust this function rather "
            "than guess further."
        )
    return dirs


def process_one_vehicle(vehicle_dir: str, vehicle_id: str) -> pd.DataFrame:
    """Run the tested single-vehicle pipeline on one vehicle's data. Returns
    a DataFrame of temperature-corrected capacity estimates with a
    'vehicle_id' column, or an empty DataFrame if extraction failed/found
    nothing usable (logged, not silently dropped)."""
    try:
        df = load_telemetry(vehicle_dir)
        df = extract_charge_segments(df)
        res = estimate_capacity_per_segment(df)
        if len(res) < 10:
            print(f"  [{vehicle_id}] only {len(res)} usable charge segments -- skipping "
                  f"(need >=10 for a meaningful per-vehicle trend).")
            return pd.DataFrame()
        res = temperature_correct(res)
        res["vehicle_id"] = vehicle_id
        # Normalize each vehicle's own trend to its own early-life mean, since
        # different vehicles/battery packs have different absolute capacities
        # and cannot be compared on the Ah scale directly.
        early_life_mean = res["capacity_corrected"].iloc[:5].mean()
        res["capacity_normalized"] = res["capacity_corrected"] / early_life_mean
        return res
    except Exception as e:
        print(f"  [{vehicle_id}] FAILED: {e}")
        return pd.DataFrame()


def pool_and_diagnose(all_results: pd.DataFrame):
    """The actual test this script exists for: does the population show a
    cleaner (closer to monotonically non-increasing) trend than any single
    vehicle did, or does the same dip-then-recover shape appear at similar
    RELATIVE odometer points across vehicles (suggesting a shared confound
    that pooling doesn't fix)?"""
    n_vehicles = all_results.vehicle_id.nunique()
    print(f"\n=== Pooled diagnostic across {n_vehicles} vehicles ===")

    # Bin by odometer (relative, since vehicles start at different points) and
    # take the population mean/median per bin -- this is the pooled trend.
    all_results = all_results.copy()
    all_results["odometer_bin"] = (all_results.odometer_km // 5000) * 5000
    pooled = all_results.groupby("odometer_bin")["capacity_normalized"].agg(["mean", "median", "std", "count"])
    print("\nPooled capacity_normalized by 5,000 km odometer bin:")
    print(pooled.to_string())

    # Key check: is the pooled trend monotonically non-increasing (as genuine
    # fade should be), and is its bin-to-bin variation smaller than any single
    # vehicle's raw variation was? Report both, don't just assert one.
    bins_sorted = pooled.sort_index()
    diffs = bins_sorted["mean"].diff().dropna()
    n_increasing = (diffs > 0.002).sum()  # allow small noise tolerance
    print(f"\nBins where pooled mean capacity INCREASED vs. previous bin "
          f"(tolerance 0.2%): {n_increasing} of {len(diffs)}")
    if n_increasing > len(diffs) * 0.3:
        print("WARNING: pooled trend still shows substantial non-monotonic "
              "behavior. This suggests the confound found at n=1 (Section: "
              "field_data_exploration.py docstring) is likely COMMON-MODE "
              "across vehicles (e.g. genuinely seasonal/calendar), not random "
              "per-vehicle noise -- pooling more vehicles alone will not fix "
              "a common-mode confound. Explicit calendar-time alignment would "
              "be needed instead (see module docstring's caveat on "
              "terminaltime not being an absolute timestamp).")
    else:
        print("Pooled trend is largely monotonic -- this is a positive sign "
              "that per-vehicle noise, not a shared confound, was the main "
              "problem at n=1, and cross-vehicle pooling is a valid fix.")

    return pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                         help="Folder containing one subfolder per vehicle")
    parser.add_argument("--max-vehicles", type=int, default=None,
                         help="Optional cap for a quick test run")
    args = parser.parse_args()

    vehicle_dirs = discover_vehicle_dirs(args.data_root)
    if args.max_vehicles:
        vehicle_dirs = vehicle_dirs[:args.max_vehicles]
    print(f"Found {len(vehicle_dirs)} vehicle directories.")

    all_results = []
    for vdir in vehicle_dirs:
        vid = os.path.basename(vdir)
        print(f"Processing {vid}...")
        res = process_one_vehicle(vdir, vid)
        if len(res) > 0:
            all_results.append(res)

    if not all_results:
        print("\nNo vehicles produced usable data. Check --data-root structure "
              "matches what discover_vehicle_dirs() expects (see docstring).")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("multi_vehicle_capacity_trends.csv", index=False)
    print(f"\nSaved {len(combined)} rows across {combined.vehicle_id.nunique()} "
          f"vehicles to multi_vehicle_capacity_trends.csv")

    pool_and_diagnose(combined)


if __name__ == "__main__":
    main()
