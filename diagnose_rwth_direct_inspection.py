"""
diagnose_rwth_direct_inspection.py

diagnose_rwth_double_track.py's sawtooth heuristic came back negative
(max rate 0.009, nowhere near the 50-99% a genuinely interleaved
dual-track signal produces) -- so raw per-cell capacity sequences are
NOT duplicated/interleaved at the row level. The visual double-track
pattern in 01_prediction_vs_actual.png and 03_per_cell_curves.png must
have a different cause. Two direct checks, no heuristic guessing:

  1. Duplicate CYCLE numbers per cell -- a different signature than
     sawtooth jumps: if two session files independently number their
     own cycles starting from 1 (rather than continuing a running
     count), concatenating them would produce repeated cycle numbers
     with DIFFERENT capacity values at each duplicate -- which
     wouldn't necessarily show up as adjacent-row sawtooth jumps if
     the rows aren't sorted the way we'd expect, but WOULD show up as
     duplicate cycle numbers directly.
  2. Print RWTH_ep_sanyo_002's full raw (cycle, capacity) sequence,
     sorted by cycle -- direct visual inspection, since that's the
     specific cell 03_per_cell_curves.png flagged, rather than relying
     on a heuristic score.

Run:
    python diagnose_rwth_direct_inspection.py
"""

import numpy as np
import pandas as pd
import rwth_loader

TARGET_CELL = "RWTH_ep_sanyo_002"


def main():
    print("Reloading RWTH fresh via rwth_loader.load_all()...")
    df = rwth_loader.load_all()

    # --- Check 1: duplicate cycle numbers per cell ---
    print(f"\n{'='*70}")
    print("Check 1: duplicate cycle numbers per cell")
    print(f"{'='*70}")
    dup_summary = []
    for cell_id, group in df.groupby("cell_id"):
        n_total = len(group)
        n_unique_cycles = group["cycle"].nunique()
        n_dupes = n_total - n_unique_cycles
        if n_dupes > 0:
            dup_summary.append((cell_id, n_total, n_unique_cycles, n_dupes))

    if not dup_summary:
        print("No cell has any duplicate cycle numbers -- this rules out "
              "the 'two independently-numbered session types concatenated' "
              "theory entirely. The cause is something else.")
    else:
        dup_summary.sort(key=lambda t: -t[3])
        print(f"{'Cell':25s} {'total rows':>12s} {'unique cycles':>14s} {'duplicates':>12s}")
        for cell_id, n_total, n_unique, n_dupes in dup_summary[:15]:
            print(f"{cell_id:25s} {n_total:12d} {n_unique:14d} {n_dupes:12d}")
        print(f"\n{len(dup_summary)} of {df['cell_id'].nunique()} cells have "
              f"duplicate cycle numbers.")

    # --- Check 2: direct inspection of the target cell ---
    print(f"\n{'='*70}")
    print(f"Check 2: direct inspection of {TARGET_CELL}")
    print(f"{'='*70}")
    target = df[df["cell_id"] == TARGET_CELL].sort_values("cycle")
    if target.empty:
        print(f"[!] {TARGET_CELL} not found -- check the exact cell_id "
              f"string (case/format may differ from this script's guess).")
        print(f"Available cell_ids sample: {sorted(df['cell_id'].unique())[:10]}")
        return

    print(f"Total rows for {TARGET_CELL}: {len(target)}")
    print(f"Unique cycle numbers: {target['cycle'].nunique()}")
    print(f"Cycle range: {target['cycle'].min()} to {target['cycle'].max()}")
    print(f"Capacity range: {target['capacity'].min():.4f} to {target['capacity'].max():.4f}")

    with pd.option_context("display.max_rows", 60, "display.width", 200):
        print("\nFirst 60 rows (cycle, capacity), sorted by cycle:")
        print(target[["cycle", "capacity"]].head(60).to_string(index=False))

    # Look specifically for the same cycle number appearing with very
    # different capacity values, which is the most direct "two tracks"
    # signature possible
    dupe_cycles = target[target.duplicated(subset="cycle", keep=False)]
    if not dupe_cycles.empty:
        print(f"\n[!] {TARGET_CELL} has {len(dupe_cycles)} rows sharing a "
              f"duplicate cycle number with at least one other row. "
              f"Showing them grouped by cycle:")
        for cyc, grp in dupe_cycles.groupby("cycle"):
            vals = grp["capacity"].tolist()
            spread = max(vals) - min(vals)
            flag = "  <-- LARGE spread, likely two different tracks" if spread > 0.1 else ""
            print(f"  cycle={cyc}: capacity values = {vals}{flag}")
    else:
        print(f"\n{TARGET_CELL} has no duplicate cycle numbers either. "
              f"If the visual double-track pattern is real, it may be an "
              f"artifact of how make_windows() orders windows for this "
              f"cell, or of this script's own plotting code in "
              f"generate_all_figures.py -- worth re-examining "
              f"fig_per_cell_curves()'s indexing logic next, rather than "
              f"the raw data itself.")


if __name__ == "__main__":
    main()
