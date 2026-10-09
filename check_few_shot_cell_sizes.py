"""
Checks whether few_shot_calibration.py's small-budget results could have hit
hgc_cp.py's (now-fixed) inf-quantile edge case.

WHY. At a small cell budget (e.g. 1 cell), the sampled target cell(s) get
assigned distance=1e-9 (near the closest possible) in evaluate_budget().
Whether that triggers the inf condition depends on how many calibration
WINDOWS that specific cell contributes: too few windows combined with a
dominant weight share can push the per-point test_weight above the
alpha/(1-alpha) threshold (0.111 at alpha=0.1). This script reports each
dataset's per-cell window counts directly, so that can be checked rather
than guessed at.

This does NOT reproduce the exact weight computation (that depends on the
specific MMD distances to the other 6 sources, which vary by run) -- it
reports a simpler, conservative proxy: for a given cell's window count n and
an assumed self-weight share (printed for a few plausible values), whether
the n alone is small enough to make triggering possible regardless of what
the other distances happen to be.

Run from the project root:  python check_few_shot_cell_sizes.py
"""

import numpy as np

from hgc_cp_evaluation_harness import DATASETS, VAL_PREDICTIONS_NPZ_PATH

ALPHA = 0.1
THRESHOLD = ALPHA / (1 - ALPHA)  # 0.111 at alpha=0.1
# self-weight shares worth checking: a near-zero-distance pseudo-source
# typically dominates the softmax heavily once any real source distances
# exist, so these represent a plausible low/mid/high range, not a guess at
# one specific number.
SELF_WEIGHT_SHARES_TO_CHECK = [0.5, 0.7, 0.9, 0.99]


def main():
    val_pred = np.load(VAL_PREDICTIONS_NPZ_PATH, allow_pickle=True)
    lab_val = val_pred["lab_val"]
    cell_val = val_pred["cell_val"]

    print(f"{'Dataset':8s}{'n cells':>8s}{'min n':>8s}{'median n':>10s}{'max n':>8s}  cells at risk (n < threshold for self-weight share):")
    print(f"{'':8s}{'':8s}{'':8s}{'':10s}{'':8s}  " + "  ".join(f"share={s}" for s in SELF_WEIGHT_SHARES_TO_CHECK))

    for d in DATASETS:
        mask = lab_val == d
        if not mask.any():
            print(f"{d:8s}  no validation rows found")
            continue
        cells = cell_val[mask]
        unique_cells, counts = np.unique(cells, return_counts=True)

        at_risk_counts = []
        for share in SELF_WEIGHT_SHARES_TO_CHECK:
            # triggers if share / n > THRESHOLD  <=>  n < share / THRESHOLD
            risk_n = share / THRESHOLD
            n_at_risk = int(np.sum(counts < risk_n))
            at_risk_counts.append(f"{n_at_risk}/{len(counts)} (n<{risk_n:.1f})")

        print(f"{d:8s}{len(unique_cells):8d}{counts.min():8d}{int(np.median(counts)):10d}{counts.max():8d}  "
              + "  ".join(at_risk_counts))

    print("\nReading this: for a given assumed self-weight share, 'n_at_risk/n_cells' counts how many")
    print("of this dataset's validation cells have FEWER windows than the trigger threshold at that")
    print("share -- i.e. how many single-cell draws in few_shot_calibration.py's 1-cell budget could")
    print("plausibly have hit the (now-fixed) inf-quantile bug. 0/n means this dataset's cells are all")
    print("large enough to be safe regardless of how dominant the self-weight ends up being.")


if __name__ == "__main__":
    main()
