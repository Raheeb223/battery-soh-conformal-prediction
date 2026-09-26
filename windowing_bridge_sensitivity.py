"""
Sensitivity of conformal coverage to windows that span gaps in the cycle
numbering.

Sliding windows can span cycles that were removed during the XJTU/RWTH data
corrections (or gaps that are native to a dataset's measurement schedule).
This script measures how often that happens and whether it changes coverage.

Method: using the per-window raw cycle indices saved in cache/processed.npz,
a test window is flagged as "bridging" if any two consecutive cycle indices in
it differ by more than 1. Coverage of the groupwise-calibrated full model is
then recomputed on the non-bridging subset and compared with the full test set.

Decision criteria (fixed before the analysis):
  - Effect bounded: non-bridging coverage within ~1-2 pp of the full-set value
    for every dataset, in particular XJTU and RWTH.
  - Effect not bounded: a shift > 3-5 pp, or outside the dataset's
    Clopper-Pearson CI, for XJTU or RWTH.
  - Ambiguous: shifts concentrated in low-n datasets (Oxford, CALCE, NASA),
    where finite-sample noise alone can produce 1-2 pp; CI overlap is reported.
  - A near-zero (<1%) bridging fraction for XJTU and RWTH bounds the effect
    regardless of the coverage shift.

Run:
    python windowing_bridge_sensitivity.py
Output:
    outputs/windowing_bridge_results.json
"""

import os
import json

import numpy as np

import config


def _load_cache():
    path = os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(path):
        print(f"  [!] {path} not found -- run preprocessing.py first.")
        return None
    data = np.load(path, allow_pickle=True)
    required = {"cell_test", "lab_test", "window_cycle_indices_test", "y_test"}
    missing = required - set(data.files)
    if missing:
        print(f"  [!] processed.npz is missing {missing}. Re-run the "
              f"updated preprocessing.py first.")
        return None
    return data


def _load_groupwise_predictions():
    """The groupwise-calibration result comes from
    outputs/conformal_groupwise_predictions.npz, not
    ablation_full_predictions.npz (a separate run for the
    architecture-ablation table)."""
    path = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz")
    if not os.path.exists(path):
        print(f"  [!] {path} not found.")
        return None
    data = np.load(path, allow_pickle=True)
    required = {"y_test", "lower", "upper", "lab_test"}
    missing = required - set(data.files)
    if missing:
        print(f"  [!] {path} is missing {missing}.")
        return None
    return data


def find_bridging_windows(window_cycle_indices):
    """window_cycle_indices: array of shape (n_windows, window_len), the
    raw cycle-number sequence each window was built from. Returns a
    boolean mask, True where the window bridges a gap (non-contiguous
    cycle numbers -- i.e. at least one cycle was removed from inside
    this window's span)."""
    diffs = np.diff(window_cycle_indices, axis=1)
    # A clean, non-bridging window has all consecutive-cycle steps == 1.
    bridging = np.any(diffs != 1, axis=1)
    return bridging


def main():
    cache = _load_cache()
    if cache is None:
        return
    preds = _load_groupwise_predictions()
    if preds is None:
        return

    lab_test = cache["lab_test"]
    cell_test = cache["cell_test"]
    window_cycles = cache["window_cycle_indices_test"]
    y_test_cache = cache["y_test"]

    y_test = preds["y_test"]
    lower = preds["lower"]
    upper = preds["upper"]
    lab_test_preds = preds["lab_test"]

    # CRITICAL CHECK: cache/processed.npz and conformal_groupwise_predictions.npz
    # are two SEPARATE files. This script assumes their test-set rows are in
    # the SAME order (window i in one = window i in the other). Verify that
    # assumption explicitly rather than silently trust it -- a silent
    # misalignment would make every downstream number meaningless.
    if len(y_test_cache) != len(y_test):
        print(f"  [!] ROW COUNT MISMATCH: cache/processed.npz has "
              f"{len(y_test_cache)} test rows, but "
              f"conformal_groupwise_predictions.npz has {len(y_test)}. "
              f"These cannot be the same test set -- stopping rather than "
              f"produce a meaningless result. Check whether one of these "
              f"was generated with a different window size or a re-run "
              f"that changed the split.")
        return
    if not np.allclose(y_test_cache, y_test, atol=1e-5):
        print(f"  [!] ALIGNMENT CHECK FAILED: y_test values differ between "
              f"cache/processed.npz and conformal_groupwise_predictions.npz "
              f"at matching row positions. They are NOT in the same order -- "
              f"stopping rather than silently combine misaligned arrays. "
              f"You will need to re-derive a shared row key (e.g. re-run "
              f"inference directly from cache/processed.npz's X_test rather "
              f"than relying on two independently-saved files) before this "
              f"check can proceed safely.")
        return
    if not np.array_equal(lab_test, lab_test_preds):
        print(f"  [!] ALIGNMENT CHECK FAILED: lab_test arrays differ between "
              f"the two files even though y_test matched. Stopping.")
        return
    print("  [OK] Alignment check passed: y_test and lab_test match exactly "
          "between cache/processed.npz and conformal_groupwise_predictions.npz.\n")

    bridging_mask = find_bridging_windows(window_cycles)

    print(f"{'='*78}")
    print("D1: WINDOWING-BRIDGE SENSITIVITY CHECK")
    print(f"{'='*78}\n")

    results = {}
    datasets = sorted(set(np.unique(lab_test)))
    for ds in datasets:
        ds_mask = lab_test == ds
        n_total = int(ds_mask.sum())
        n_bridging = int((ds_mask & bridging_mask).sum())
        frac_bridging = n_bridging / n_total if n_total > 0 else 0.0

        # Full-set coverage (should match Table 6's existing groupwise number)
        full_cov = float(np.mean((y_test[ds_mask] >= lower[ds_mask]) &
                                   (y_test[ds_mask] <= upper[ds_mask])))

        # Non-bridging-subset coverage
        clean_mask = ds_mask & ~bridging_mask
        n_clean = int(clean_mask.sum())
        if n_clean > 0:
            clean_cov = float(np.mean((y_test[clean_mask] >= lower[clean_mask]) &
                                        (y_test[clean_mask] <= upper[clean_mask])))
        else:
            clean_cov = float("nan")

        delta_pp = (clean_cov - full_cov) * 100 if n_clean > 0 else float("nan")

        results[ds] = {
            "n_total": n_total,
            "n_bridging": n_bridging,
            "frac_bridging": frac_bridging,
            "full_set_coverage": full_cov,
            "non_bridging_coverage": clean_cov,
            "delta_pp": delta_pp,
        }
        print(f"{ds:8s}  bridging: {n_bridging:6d}/{n_total:6d} ({frac_bridging*100:.2f}%)  "
              f"full-set coverage: {full_cov*100:.1f}%  "
              f"non-bridging coverage: {clean_cov*100:.1f}%  "
              f"delta: {delta_pp:+.2f}pp")

    out_path = os.path.join(config.OUTPUT_DIR, "windowing_bridge_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved {out_path}")

    print(f"\n{'='*78}")
    print("INTERPRETATION (apply the pre-registered criteria above -- do "
          "not just eyeball this): for XJTU and RWTH specifically, is the "
          "delta within ~1-2pp AND within that dataset's own Clopper-Pearson "
          "CI half-width (Table 7)? If yes for both, the limitation is "
          "empirically bounded. If no for either, report the discrepancy "
          "honestly rather than keep the current 'unlikely to matter' "
          "language unchanged.")


if __name__ == "__main__":
    main()