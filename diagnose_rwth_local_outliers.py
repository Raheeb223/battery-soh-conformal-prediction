"""
diagnose_rwth_local_outliers.py

diagnose_rwth_direct_inspection.py found the actual mechanism:
RWTH_ep_sanyo_002's cycle=1 has capacity=0.294 while cycle=2 onward
sits smoothly around 1.077 -- a single isolated sensor/logging glitch,
not degradation. This is a different bug class than XJTU's periodic
sentinel values: scattered, per-cell, single-row local outliers rather
than a systematic repeating pattern. The stated capacity range
"0.0000 to 1.0793" also confirms at least one row hits exactly zero
somewhere beyond the first 60 rows shown.

This scans every RWTH cell's FULL capacity sequence for local outliers:
points that deviate sharply from BOTH immediate neighbors, where the
neighbors themselves are close to each other (the signature of a single
bad reading dropped into an otherwise-smooth trend, as opposed to a
genuine but fast-changing degradation region).

Run:
    python diagnose_rwth_local_outliers.py
"""

import numpy as np
import pandas as pd
import rwth_loader

# A point is flagged if it differs from BOTH neighbors by more than this
# fraction of the local capacity level, while the neighbors differ from
# EACH OTHER by less than half that -- i.e. a real single-point spike,
# not a genuinely fast-moving 3-point stretch.
LOCAL_OUTLIER_THRESHOLD = 0.15


def find_local_outliers(capacity, threshold=LOCAL_OUTLIER_THRESHOLD):
    vals = np.asarray(capacity, dtype=float)
    n = len(vals)
    outlier_idx = []

    # interior points: compare against both neighbors
    for i in range(1, n - 1):
        left, mid, right = vals[i - 1], vals[i], vals[i + 1]
        neighbor_diff = abs(left - right)
        mid_dev_left = abs(mid - left)
        mid_dev_right = abs(mid - right)
        scale = max(abs(left), abs(right), 1e-9)
        if (mid_dev_left / scale > threshold and
                mid_dev_right / scale > threshold and
                neighbor_diff / scale < threshold / 2):
            outlier_idx.append(i)

    # edge points (first/last row) have only ONE neighbor -- this is
    # exactly the case that bit us: RWTH_ep_sanyo_002's anomaly was at
    # cycle=1, the very first row, which the interior-only loop above
    # cannot see. Flag an edge point if it differs sharply from its one
    # available neighbor AND that neighbor is reasonably close to ITS
    # OWN next-door neighbor (confirming the edge point, not the
    # neighbor, is the odd one out).
    if n >= 3:
        first, second, third = vals[0], vals[1], vals[2]
        scale = max(abs(second), abs(third), 1e-9)
        if (abs(first - second) / scale > threshold and
                abs(second - third) / scale < threshold / 2):
            outlier_idx.append(0)

        last, second_last, third_last = vals[-1], vals[-2], vals[-3]
        scale = max(abs(second_last), abs(third_last), 1e-9)
        if (abs(last - second_last) / scale > threshold and
                abs(second_last - third_last) / scale < threshold / 2):
            outlier_idx.append(n - 1)

    # exact zeros/negatives flagged regardless of local shape --
    # unambiguous regardless of context
    zero_idx = [i for i in range(n) if vals[i] <= 0]
    return sorted(set(outlier_idx) | set(zero_idx))


def main():
    print("Reloading RWTH fresh via rwth_loader.load_all()...")
    df = rwth_loader.load_all()

    print(f"\n{'='*70}")
    print("Scanning every RWTH cell for local single-row outliers")
    print(f"{'='*70}")

    total_outliers = 0
    per_cell_counts = []
    for cell_id, group in df.groupby("cell_id"):
        g = group.sort_values("cycle").reset_index(drop=True)
        outliers = find_local_outliers(g["capacity"].to_numpy())
        if outliers:
            per_cell_counts.append((cell_id, len(outliers), len(g)))
            total_outliers += len(outliers)

    per_cell_counts.sort(key=lambda t: -t[1])

    print(f"\n{'Cell':25s} {'n_outliers':>12s} {'n_rows':>10s} {'rate':>8s}")
    for cell_id, n_out, n_rows in per_cell_counts[:20]:
        print(f"{cell_id:25s} {n_out:12d} {n_rows:10d} {n_out/n_rows:8.4f}")

    n_affected_cells = len(per_cell_counts)
    n_total_cells = df["cell_id"].nunique()
    print(f"\n{n_affected_cells} of {n_total_cells} cells have at least one "
          f"local outlier / zero-or-negative capacity row.")
    print(f"Total flagged rows across all RWTH: {total_outliers} "
          f"({100*total_outliers/len(df):.3f}% of all RWTH rows)")

    # show a few concrete examples with surrounding context
    print(f"\n{'='*70}")
    print("Example outliers with surrounding context (up to 5 shown)")
    print(f"{'='*70}")
    shown = 0
    for cell_id, n_out, n_rows in per_cell_counts:
        if shown >= 5:
            break
        g = df[df["cell_id"] == cell_id].sort_values("cycle").reset_index(drop=True)
        outliers = find_local_outliers(g["capacity"].to_numpy())
        for idx in outliers[:1]:  # one example per cell, up to 5 cells
            lo, hi = max(0, idx - 2), min(len(g), idx + 3)
            print(f"\n{cell_id}, around row index {idx}:")
            print(g.iloc[lo:hi][["cycle", "capacity"]].to_string(index=False))
            shown += 1
            if shown >= 5:
                break

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print("If total flagged rows are a small fraction of all RWTH rows "
          "(as expected for scattered sensor glitches rather than a "
          "systematic pattern), the fix is a straightforward local-outlier "
          "filter added to rwth_loader.py: drop any row that is a local "
          "spike/dip meeting this script's criteria (or the simpler "
          "capacity <= 0 subset, which is unambiguous) before returning "
          "the DataFrame. This is a much smaller, more surgical fix than "
          "the XJTU case -- no periodicity or dataset-wide constant value "
          "involved, just per-row validation.")


if __name__ == "__main__":
    main()
