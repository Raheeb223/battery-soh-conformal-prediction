"""
diagnose_rwth_double_track.py

01_prediction_vs_actual.png and 03_per_cell_curves.png both show a
"double-track" pattern in RWTH data: a secondary diagonal branch
peeling off the main scatter cloud (overall + RWTH panels), and
RWTH_ep_sanyo_002's per-cell curve shows two parallel bands connected
by vertical stems -- not noisy scatter, but a structured pattern. This
looks like the same CLASS of issue as the XJTU sentinel-value bug
(periodic non-degradation readings mixed into the real signal), but a
different mechanism: RWTH's raw data comes from 1,642 session files
across 48 cells, spanning what look like different test-protocol types
(based on earlier loader output: "TBA_Zyk", "TBA_CUv2", "TBA_CHA",
"TBA_BOL_P1" -- likely cycling, capacity-check, characterization, and
beginning-of-life session types). If capacity readings from DIFFERENT
session types get concatenated under a single, possibly
file-parse-order-based cycle counter rather than a true chronological
one, you'd get exactly two parallel "tracks" -- one from routine
cycling estimates, one from periodic reference-performance-test
readings -- plotted against a shared but misaligned cycle axis.

This script:
  1. Reloads RWTH fresh via rwth_loader.load_all() and prints every
     column available (don't assume -- inspect).
  2. For every cell, flags a "sawtooth" signature: repeated large
     up-then-down (or down-then-up) jumps in capacity/SOH across
     consecutive rows, which is what a genuinely interleaved dual-track
     signal looks like (a real degrading cell's capacity should be
     roughly monotonic decreasing with only small noise, not large
     repeated reversals).
  3. For the most-flagged cell(s), prints the raw rows (every column
     available, including any session-file/type identifier if the
     loader kept one) so we can see the actual mechanism, not guess at
     it.

Run:
    python diagnose_rwth_double_track.py
"""

import numpy as np
import pandas as pd
import rwth_loader

SAWTOOTH_JUMP_THRESHOLD = 0.05  # fraction-of-capacity jump considered "large"


def sawtooth_score(soh_or_capacity):
    """Counts how many consecutive-pair jumps exceed the threshold AND
    reverse direction from the previous jump -- the signature of two
    interleaved tracks rather than one noisy-but-monotonic curve."""
    vals = np.asarray(soh_or_capacity, dtype=float)
    if len(vals) < 3:
        return 0, 0
    diffs = np.diff(vals)
    reversals = 0
    for i in range(1, len(diffs)):
        big_jump_now = abs(diffs[i]) > SAWTOOTH_JUMP_THRESHOLD
        big_jump_prev = abs(diffs[i - 1]) > SAWTOOTH_JUMP_THRESHOLD
        opposite_sign = (diffs[i] * diffs[i - 1]) < 0
        if big_jump_now and big_jump_prev and opposite_sign:
            reversals += 1
    return reversals, len(diffs)


def main():
    print("Reloading RWTH fresh via rwth_loader.load_all()...")
    df = rwth_loader.load_all()

    print(f"\nColumns available: {list(df.columns)}")
    print(f"Total rows: {len(df)}, total cells: {df['cell_id'].nunique()}\n")

    value_col = "capacity" if "capacity" in df.columns else (
        "soh" if "soh" in df.columns else None
    )
    if value_col is None:
        print("[!] Neither 'capacity' nor 'soh' column found -- adjust "
              f"this script to use the correct column from: {list(df.columns)}")
        return

    print(f"Using '{value_col}' column for sawtooth detection.\n")

    scores = []
    for cell_id, group in df.groupby("cell_id"):
        g = group.sort_values("cycle")
        reversals, n_diffs = sawtooth_score(g[value_col].to_numpy())
        rate = reversals / n_diffs if n_diffs > 0 else 0
        scores.append((cell_id, reversals, n_diffs, rate, len(g)))

    scores.sort(key=lambda t: -t[3])  # sort by reversal RATE, descending

    print(f"{'Cell':25s} {'reversals':>10s} {'n_diffs':>10s} {'rate':>8s} {'n_rows':>8s}")
    for cell_id, reversals, n_diffs, rate, n_rows in scores[:15]:
        flag = "  <-- HIGH sawtooth rate" if rate > 0.05 else ""
        print(f"{cell_id:25s} {reversals:10d} {n_diffs:10d} {rate:8.3f} {n_rows:8d}{flag}")

    if not scores or scores[0][3] < 0.02:
        print("\nNo cell shows a meaningfully high sawtooth rate -- the "
              "double-track pattern in the figures may come from a "
              "different, smaller number of cells than expected, or from "
              "a different mechanism entirely. Consider lowering "
              "SAWTOOTH_JUMP_THRESHOLD and re-running.")
        return

    worst_cell = scores[0][0]
    print(f"\n{'='*70}")
    print(f"Inspecting the worst-offending cell: {worst_cell}")
    print(f"{'='*70}")
    worst = df[df["cell_id"] == worst_cell].sort_values("cycle")
    # print first 40 rows in full, every column, so we can see everything
    # the loader actually captured -- not guessing at column meaning
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(worst.head(40).to_string())

    print(f"\n[!] Look at the columns above for anything indicating session "
          f"file name, test-protocol type, or session index -- if such a "
          f"column exists, group by it and check whether each group's "
          f"cycle range OVERLAPS with another group's for this same cell. "
          f"That would confirm the interleaved-session-type theory. If no "
          f"such column survived into this DataFrame, it needs to be added "
          f"in rwth_loader.py (e.g. preserving the session filename or "
          f"protocol code as a new column) before this can be fully "
          f"root-caused rather than just located.")


if __name__ == "__main__":
    main()
