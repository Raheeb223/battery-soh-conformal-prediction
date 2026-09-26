"""
Field-data exploration + multi-vehicle pooling test (SELF-CONTAINED, single file)
===================================================================================

Combines what was previously two files (field_data_exploration.py and
multi_vehicle_pooling_test.py) into one script with no cross-file import, so
"ModuleNotFoundError" from running them out of the same folder cannot happen.

WHAT THIS SCRIPT ACTUALLY KNOWS TO BE TRUE (tested against the real n=1 GitHub
sample, github.com/HoraceLiu1010/Multi-modal-SOH-estimation-framework):
  - Expected raw columns: terminaltime, soc, chargestatus, totalcurrent,
    totalodometer, mintemperaturevalue, maxtemperaturevalue.
  - chargestatus: 1=charging, 3=driving/idle, 4=charge-complete, 255=invalid
    (empirically determined, not documented in the source repo).
  - Naive Coulomb-counted capacity, even after temperature correction, showed
    a dip-then-recover pattern inconsistent with genuine monotonic fade, with
    noise (~25-40 Ah spread) several times larger than plausible true fade
    (~3-8 Ah) over the odometer range tested. See the pool_and_diagnose()
    function below for how this script checks whether pooling more vehicles
    fixes this or whether it's a shared (e.g. seasonal) confound instead.

SCOPE NOTE: other exports of this dataset, or other field EV telemetry
sources entirely, may use a different column schema than the GitHub sample
above (e.g. a different portal export). Run with --inspect-only first on any
new file to check its actual columns before trusting the rest of the
pipeline, and edit COLUMN_MAP below if the names differ.

USAGE:
    First, inspect an unfamiliar file before processing it:
        python field_experiment.py --path <file_or_folder> --inspect-only

    Single-vehicle extraction (one CSV or one folder of data_*.csv files):
        python field_experiment.py --path <file_or_folder> --mode single

    Multi-vehicle pooling (a folder containing one subfolder OR one CSV per vehicle):
        python field_experiment.py --path <folder> --mode multi
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd


def trapezoidal_integral(y, x):
    """Trapezoidal rule, implemented manually so this works regardless of
    NumPy version (np.trapz is deprecated/removed in NumPy>=2.0, and
    np.trapezoid, its replacement, does not exist in NumPy<2.0 -- relying on
    either name directly breaks on some environment)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x))


# ---------------------------------------------------------------------------
# If a given file's column names differ from the GitHub sample, edit this
# map: {actual_column_name: expected_name}. Leave as {} if names already match.
# ---------------------------------------------------------------------------
COLUMN_MAP = {}

REQUIRED_COLUMNS = [
    "terminaltime", "soc", "chargestatus", "totalcurrent",
    "totalodometer", "mintemperaturevalue", "maxtemperaturevalue",
]


def inspect_file(path: str):
    """Print the actual columns/dtypes/sample rows of a file so a schema
    mismatch can be diagnosed and fixed in COLUMN_MAP before running anything
    that assumes a specific structure."""
    print(f"Inspecting: {path}")
    df = pd.read_csv(path, nrows=20)
    print(f"\nColumns found ({len(df.columns)}):")
    for c in df.columns:
        print(f"  - {c}  (dtype: {df[c].dtype})")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns and c not in COLUMN_MAP.values()]
    if missing:
        print(f"\n*** MISSING EXPECTED COLUMNS: {missing} ***")
        print("Edit COLUMN_MAP at the top of this script to map the actual "
              "column names to these, then rerun with --mode single/multi.")
    else:
        print("\nAll required columns present (or mapped). Safe to proceed.")


# ---------------------------------------------------------------------------
# Phase 1: Load and clean raw telemetry
# ---------------------------------------------------------------------------

def load_telemetry(path: str) -> pd.DataFrame:
    """Load telemetry from either a single CSV file or a folder of
    data_*.csv files (the GitHub sample's format)."""
    if os.path.isdir(path):
        files = sorted(glob.glob(f"{path}/*.csv"))
        if not files:
            raise FileNotFoundError(f"No .csv files found in folder: {path}")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(path)

    if COLUMN_MAP:
        df = df.rename(columns=COLUMN_MAP)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns {missing} in {path}. Run with --inspect-only "
            "first and edit COLUMN_MAP at the top of this script to match "
            "the file's actual column names."
        )

    df = df.sort_values("terminaltime").reset_index(drop=True)
    n_before = len(df)
    df = df[df.chargestatus != 255].reset_index(drop=True)
    if n_before - len(df) > 0:
        print(f"  Dropped {n_before - len(df)} invalid (chargestatus=255) rows.")
    return df


# ---------------------------------------------------------------------------
# Phase 2: Identify charge segments and estimate capacity
# ---------------------------------------------------------------------------

def extract_charge_segments(df: pd.DataFrame, gap_seconds: float = 600.0) -> pd.DataFrame:
    is_charge = (df.chargestatus == 1).values
    time_vals = df.terminaltime.values
    seg_id = np.zeros(len(df), dtype=int)
    current_seg, in_seg = 0, False
    for i in range(len(df)):
        if is_charge[i]:
            if not in_seg:
                current_seg += 1
                in_seg = True
            elif (time_vals[i] - time_vals[i - 1]) > gap_seconds:
                current_seg += 1
            seg_id[i] = current_seg
        else:
            in_seg = False
    df = df.copy()
    df["seg_id"] = seg_id
    return df


def estimate_capacity_per_segment(
    df: pd.DataFrame,
    soc_start_max: int = 30,
    soc_end_min: int = 90,
    min_samples: int = 10,
) -> pd.DataFrame:
    results = []
    for sid, g in df[df.seg_id > 0].groupby("seg_id"):
        g = g.sort_values("terminaltime")
        soc_start, soc_end = g.soc.iloc[0], g.soc.iloc[-1]
        if soc_start <= soc_start_max and soc_end >= soc_end_min and len(g) >= min_samples:
            t = g.terminaltime.values.astype(float)
            I = g.totalcurrent.values.astype(float)
            amp_seconds = trapezoidal_integral(np.abs(I), t)
            Ah = amp_seconds / 3600.0
            capacity_est = Ah / (soc_end - soc_start) * 100.0
            avg_temp = g[["mintemperaturevalue", "maxtemperaturevalue"]].mean(axis=1).mean()
            results.append({
                "seg_id": sid,
                "odometer_km": g.totalodometer.iloc[-1],
                "terminaltime": g.terminaltime.iloc[-1],
                "capacity_Ah_est": capacity_est,
                "soc_start": soc_start,
                "soc_end": soc_end,
                "avg_temp_C": avg_temp,
                "n_samples": len(g),
            })
    return pd.DataFrame(results).sort_values("odometer_km").reset_index(drop=True)


def temperature_correct(res_df: pd.DataFrame) -> pd.DataFrame:
    res_df = res_df.copy()
    coeffs = np.polyfit(res_df.avg_temp_C, res_df.capacity_Ah_est, 1)
    res_df["capacity_corrected"] = (
        res_df.capacity_Ah_est
        - np.polyval(coeffs, res_df.avg_temp_C)
        + res_df.capacity_Ah_est.mean()
    )
    return res_df


def diagnose_signal_vs_noise(res_df: pd.DataFrame):
    spread = res_df["capacity_corrected"].max() - res_df["capacity_corrected"].min()
    initial_cap = res_df["capacity_corrected"].iloc[:5].mean()
    plausible_fade_Ah = (0.02 * initial_cap, 0.06 * initial_cap)
    corr = res_df.odometer_km.corr(res_df["capacity_corrected"])
    print(f"\n--- Signal-vs-noise diagnostic ---")
    print(f"Estimated capacity spread: {spread:.1f} Ah")
    print(f"Plausible true fade over this odometer range: "
          f"{plausible_fade_Ah[0]:.1f}-{plausible_fade_Ah[1]:.1f} Ah")
    print(f"Correlation(odometer, corrected capacity): {corr:.3f}")
    if spread > 2 * plausible_fade_Ah[1]:
        warnings.warn(
            "Estimated noise/spread exceeds 2x the plausible true fade signal. "
            "Do not report this proxy's trend as a reliable degradation curve "
            "without further correction."
        )


def run_single_vehicle(path: str):
    print(f"\n=== Single-vehicle extraction: {path} ===")
    df = load_telemetry(path)
    df = extract_charge_segments(df)
    res = estimate_capacity_per_segment(df)
    print(f"Found {len(res)} near-full charge segments "
          f"(odometer {res.odometer_km.min():.0f}-{res.odometer_km.max():.0f} km)"
          if len(res) else "Found 0 usable charge segments.")
    if len(res) < 10:
        print("Too few segments for a meaningful trend (need >=10).")
        return None
    res = temperature_correct(res)
    diagnose_signal_vs_noise(res)
    res.to_csv("field_capacity_trend.csv", index=False)
    print("Saved to field_capacity_trend.csv")
    return res


# ---------------------------------------------------------------------------
# Phase 3: Multi-vehicle pooling
# ---------------------------------------------------------------------------

def discover_vehicle_paths(data_root: str) -> list:
    """Each vehicle may be a subfolder OR a single CSV directly under
    data_root. For a dataset organized differently, adjust this function to
    match the actual folder structure."""
    subdirs = sorted([d for d in glob.glob(f"{data_root}/*") if os.path.isdir(d)])
    if subdirs:
        return subdirs
    csvs = sorted(glob.glob(f"{data_root}/*.csv"))
    if csvs:
        return csvs
    raise FileNotFoundError(
        f"No per-vehicle subfolders or CSV files found directly under {data_root}."
    )


def process_one_vehicle(vehicle_path: str, vehicle_id: str):
    try:
        df = load_telemetry(vehicle_path)
        df = extract_charge_segments(df)
        res = estimate_capacity_per_segment(df)
        if len(res) < 10:
            print(f"  [{vehicle_id}] only {len(res)} usable segments -- skipping")
            return pd.DataFrame()
        res = temperature_correct(res)
        res["vehicle_id"] = vehicle_id
        early_life_mean = res["capacity_corrected"].iloc[:5].mean()
        res["capacity_normalized"] = res["capacity_corrected"] / early_life_mean
        return res
    except Exception as e:
        print(f"  [{vehicle_id}] FAILED: {e}")
        return pd.DataFrame()


def pool_and_diagnose(all_results: pd.DataFrame):
    n_vehicles = all_results.vehicle_id.nunique()
    print(f"\n=== Pooled diagnostic across {n_vehicles} vehicles ===")
    all_results = all_results.copy()
    all_results["odometer_bin"] = (all_results.odometer_km // 5000) * 5000
    pooled = all_results.groupby("odometer_bin")["capacity_normalized"].agg(["mean", "median", "std", "count"])
    print("\nPooled capacity_normalized by 5,000 km odometer bin:")
    print(pooled.to_string())

    diffs = pooled.sort_index()["mean"].diff().dropna()
    n_increasing = (diffs > 0.002).sum()
    print(f"\nBins where pooled mean capacity INCREASED vs. previous bin: "
          f"{n_increasing} of {len(diffs)}")
    if n_increasing > len(diffs) * 0.3:
        print("WARNING: pooled trend still non-monotonic. Likely a common-mode "
              "confound (e.g. seasonal), not per-vehicle noise -- more vehicles "
              "alone will not fix this.")
    else:
        print("Pooled trend is largely monotonic -- pooling appears to help.")
    return pooled


def run_multi_vehicle(data_root: str, max_vehicles: int = None):
    vehicle_paths = discover_vehicle_paths(data_root)
    if max_vehicles:
        vehicle_paths = vehicle_paths[:max_vehicles]
    print(f"Found {len(vehicle_paths)} vehicles.")

    all_results = []
    for vpath in vehicle_paths:
        vid = os.path.splitext(os.path.basename(vpath))[0]
        print(f"Processing {vid}...")
        res = process_one_vehicle(vpath, vid)
        if len(res) > 0:
            all_results.append(res)

    if not all_results:
        print("No vehicles produced usable data.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("multi_vehicle_capacity_trends.csv", index=False)
    print(f"\nSaved {len(combined)} rows across {combined.vehicle_id.nunique()} vehicles.")
    pool_and_diagnose(combined)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                         help="A single CSV file, a folder of one vehicle's CSVs, "
                              "or (for --mode multi) a folder containing multiple vehicles")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--inspect-only", action="store_true",
                         help="Print column names/sample rows and exit -- run this "
                              "first on any file you haven't used with this script before")
    parser.add_argument("--max-vehicles", type=int, default=None)
    args = parser.parse_args()

    if args.inspect_only:
        target = args.path
        if os.path.isdir(target):
            csvs = glob.glob(f"{target}/*.csv")
            if not csvs:
                print(f"No CSVs found in {target}")
                return
            target = csvs[0]
        inspect_file(target)
        return

    if args.mode == "single":
        run_single_vehicle(args.path)
    else:
        run_multi_vehicle(args.path, args.max_vehicles)


if __name__ == "__main__":
    main()
