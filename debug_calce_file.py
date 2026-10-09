"""
Debug one specific CALCE file: prints every raw group's (cycle_raw, capacity,
voltage), then exactly what _flag_truncated_final_cycle computes from it (the
median of all groups except the last, the threshold, and the decision).

WHY. verify_calce_v2.py reported truncated_final == 0.0% for every real cell,
which does not match earlier evidence (84 of 141 low-side outlier rows were
each file's literal last group, with capacity far below the rest of that
file). This prints the ground truth for one such file so the mismatch can be
diagnosed from real data instead of guessed at.

Run:  python debug_calce_file.py CS2_33 CS2_33_10_04_10.xlsx
(cell id and file name from a row of outputs/calce_v2_verification.csv)
"""

import os
import sys

import numpy as np

import calce_loader_v2 as ld
import config


def cell_dir(cell_id):
    for dirs in config.CALCE_DIRS.values():
        for d in dirs:
            if os.path.basename(d.rstrip("/\\")) == cell_id:
                return d
    return None


def find_file(root, fname):
    for dirpath, _, filenames in os.walk(root):
        if fname in filenames:
            return os.path.join(dirpath, fname)
    return None


def main():
    if len(sys.argv) != 3:
        print("usage: python debug_calce_file.py <CELL_ID> <FILE_NAME.xlsx>")
        return
    cell_id, fname = sys.argv[1], sys.argv[2]
    root = cell_dir(cell_id)
    if root is None:
        print(f"cell '{cell_id}' not found in config.CALCE_DIRS"); return
    path = find_file(root, fname)
    if path is None:
        print(f"file '{fname}' not found under {root}"); return

    groups = ld._file_groups(path)
    print(f"{path}\n{len(groups)} groups (raw, before any differencing)\n")
    print(f"{'idx':>4s} {'cycle_raw':>10s} {'capacity':>12s} {'voltage':>9s}")
    for i, g in enumerate(groups):
        v = g["v_at_max"]
        print(f"{i:4d} {g['cycle_raw']:>10} {g['capacity']:12.6f} {v if not np.isnan(v) else float('nan'):9.2f}")

    print(f"\nTRUNCATED_FINAL_CYCLE_MIN_GROUPS = {ld.TRUNCATED_FINAL_CYCLE_MIN_GROUPS}")
    print(f"TRUNCATED_FINAL_CYCLE_RATIO      = {ld.TRUNCATED_FINAL_CYCLE_RATIO}")
    n = len(groups)
    if n < ld.TRUNCATED_FINAL_CYCLE_MIN_GROUPS:
        print(f"n={n} < MIN_GROUPS -> rule never runs for this file")
    else:
        others = [g["capacity"] for g in groups[:-1]]
        med = float(np.median(others))
        last = groups[-1]["capacity"]
        print(f"median of the other {len(others)} groups' capacity: {med:.6f}")
        print(f"last group's capacity: {last:.6f}")
        print(f"threshold (ratio * median): {ld.TRUNCATED_FINAL_CYCLE_RATIO * med:.6f}")
        print(f"med > 0: {med > 0}   last < threshold: {last < ld.TRUNCATED_FINAL_CYCLE_RATIO * med}")
        flags = ld._flag_truncated_final_cycle(groups)
        print(f"flag result: {flags[-1]}")

    # also show what CAPACITY_MODE classification decided for this file, since a
    # cumulative-counter classification changes every group's capacity before this
    # flag would ever see it in the real load_calce_cell() pipeline
    mode_before = [g["capacity"] for g in groups]
    after = ld._apply_capacity_mode(groups, fname)
    mode_after = [g["capacity"] for g in after]
    print(f"\ncapacity BEFORE _apply_capacity_mode (last 3): {[round(x,4) for x in mode_before[-3:]]}")
    print(f"capacity AFTER  _apply_capacity_mode (last 3): {[round(x,4) for x in mode_after[-3:]]}")
    print("(if these differ, the file was classified as a cumulative counter and differenced --"
          " _flag_truncated_final_cycle is computed on the BEFORE values, by design, but the row that"
          " reaches the output has the AFTER capacity, which is what verify_calce_v2.py's outlier test sees)")


if __name__ == "__main__":
    main()
