"""
Inspect the raw per-cycle capacity sequence of the files that
verify_calce_v2.py flagged as containing an outlier row.

WHY. verify_calce_v2.py classifies each workbook as cumulative or not using
one rule for the WHOLE file (>=90% of consecutive groups increasing). Two
patterns can survive that: (a) the file has a real mid-session reset -- the
counter genuinely restarts partway through -- which the current code already
handles at the file level but can mishandle if the reset lands somewhere
other than the file boundary; (b) only PART of a "max"-classified file
behaves as a cumulative counter, so that segment is left undifferenced. This
prints the raw (cycle_raw, capacity-before-differencing, capacity-as-loaded,
voltage) sequence for each flagged file so both can be told apart by eye:
a single sharp drop then normal small steps is (a); a run of several
steadily increasing values by roughly one full cycle's worth each time,
followed by a drop, is (b).

INPUT. outputs/calce_v2_verification.csv (from verify_calce_v2.py).
Run from the project root:  python inspect_calce_outliers.py
Output: printed tables only (nothing is written).
"""

import os

import numpy as np
import pandas as pd

import calce_loader_v2 as ld
import config

TOP_N_FILES = 12   # print full detail for at most this many (cell, file) pairs, ranked by outlier count


def cell_dir_map():
    out = {}
    for dirs in config.CALCE_DIRS.values():
        for d in dirs:
            out[os.path.basename(d.rstrip("/\\"))] = d
    return out


def find_file(cell_dir, fname):
    for dirpath, _, filenames in os.walk(cell_dir):
        if fname in filenames:
            return os.path.join(dirpath, fname)
    return None


def main():
    csv_path = os.path.join(config.OUTPUT_DIR, "calce_v2_verification.csv")
    outliers = pd.read_csv(csv_path)
    if outliers.empty:
        print("no outlier rows recorded -- nothing to inspect")
        return

    ranked = (outliers.groupby(["cell", "file"]).size()
              .sort_values(ascending=False).reset_index(name="n_outliers"))
    dirs = cell_dir_map()
    print(f"{len(ranked)} distinct (cell, file) pairs contain at least one outlier; showing the top {TOP_N_FILES}\n")

    for _, row in ranked.head(TOP_N_FILES).iterrows():
        cell, fname, n = row["cell"], row["file"], row["n_outliers"]
        cell_dir = dirs.get(cell)
        path = find_file(cell_dir, fname) if cell_dir else None
        print(f"=== {cell} / {fname}  ({n} outlier row(s)) ===")
        if path is None:
            print("  file not found under the configured directory -- skipping")
            continue
        try:
            groups = ld._file_groups(path)
        except Exception as e:
            print(f"  could not read: {e}")
            continue
        caps = [g["capacity"] for g in groups]
        loaded = [g["capacity"] for g in ld._apply_capacity_mode(groups, fname)]
        inc = np.mean([caps[k + 1] > caps[k] for k in range(len(caps) - 1)]) if len(caps) > 1 else float("nan")
        cumulative_flag = len(caps) >= 4 and inc >= 0.90
        print(f"  {len(caps)} groups; {100*inc:.0f}% of consecutive groups increase "
              f"-> classified {'CUMULATIVE (differenced)' if cumulative_flag else 'not cumulative (kept as-is)'}")
        # Flag rows whose as-loaded capacity is far from this file's own median -- an
        # approximate, file-local re-derivation of the outlier test (the CSV's local_median
        # was computed over a cell-wide window, so its row index does not map 1:1 to this
        # file's local group index; this re-check is what lines up with the printout below).
        med = float(np.median(loaded)) if loaded else float("nan")
        print(f"  {'idx':>4s} {'cycle_raw':>10s} {'raw (undifferenced)':>20s} {'as-loaded':>12s} {'voltage':>9s}")
        for i, g in enumerate(groups):
            mark = " <-- far from this file's median" if med and not (0.7 * med <= loaded[i] <= 1.3 * med) else ""
            v = g["v_at_max"]
            print(f"  {i:4d} {g['cycle_raw']:>10} {caps[i]:20.4f} {loaded[i]:12.4f} "
                  f"{v if not np.isnan(v) else float('nan'):9.2f}{mark}")
        print()


if __name__ == "__main__":
    main()
