"""
calce_output_range_extrapolation.py  (D2-followup)

MOTIVATION: calce_propensity_overlap.py (D2) found that CALCE's INPUT
windows are NOT distinguishable from the pooled training pool in
feature space (AUC=0.5015, 1.13% at propensity>0.9) -- refuting the
naive "no similar input examples exist" version of the manuscript's
support-gap claim. But Table 15's marginal-statistics finding (CALCE
reaches far lower absolute SOH, in far fewer cycles, than the pool)
is still real and unrefuted. The natural reconciliation: CALCE's local
INPUT window shapes are unremarkable (hence D2's good overlap), but its
TARGET (output) values may extend below anything the training pool's
targets ever reach -- output-range extrapolation, not input-space
unfamiliarity. This script tests that directly and cheaply.

METHOD: compare the minimum target SOH value in CALCE's test set against
the minimum target SOH value across the pooled OTHER 6 datasets' training
targets. If CALCE's test targets extend meaningfully below the pool's
training target range, the model is being asked to extrapolate outside
its trained output range for a real fraction of CALCE's test samples --
directly testable, no model retraining needed.

PRE-REGISTERED CRITERIA:
  - SUPPORTS output-range extrapolation as the mechanism: a large
    fraction of CALCE's test targets fall below the pool's training
    target minimum (or below some low percentile of it, e.g. 1st).
  - REFUTES it: CALCE's test targets sit comfortably within the pool's
    training target range -- would mean output-range extrapolation is
    NOT the mechanism either, and the LODO failure needs a still-
    different explanation than either hypothesis tested so far.
  - AMBIGUOUS: a moderate fraction (e.g. 10-40%) below range -- report
    the fraction and the distribution, not a forced binary conclusion.

IMPLICATION FOR EXPERIMENT B: if this CONFIRMS output-range
extrapolation, importance weighting (which reweights existing input-
target PAIRS by input similarity) should NOT be expected to help, even
though D2 found good input overlap -- reweighting cannot manufacture
training pairs with target values that do not exist in the pool at
all. Report this explicitly if it happens, rather than treating
Experiment B's outcome in isolation from this result.

Run from your project root:
    python calce_output_range_extrapolation.py
Output:
    outputs/calce_output_range_results.json
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
    return np.load(path, allow_pickle=True)


def main():
    cache = _load_cache()
    if cache is None:
        return

    y_train, lab_train = cache["y_train"], cache["lab_train"]
    y_test, lab_test = cache["y_test"], cache["lab_test"]

    pool_train_mask = lab_train != "CALCE"
    y_pool_train = y_train[pool_train_mask]

    calce_test_mask = lab_test == "CALCE"
    y_calce_test = y_test[calce_test_mask]

    pool_min = float(y_pool_train.min())
    pool_p1 = float(np.percentile(y_pool_train, 1))
    pool_p5 = float(np.percentile(y_pool_train, 5))

    frac_below_pool_min = float(np.mean(y_calce_test < pool_min))
    frac_below_pool_p1 = float(np.mean(y_calce_test < pool_p1))
    frac_below_pool_p5 = float(np.mean(y_calce_test < pool_p5))

    print(f"{'='*78}")
    print("D2-FOLLOWUP: OUTPUT-RANGE EXTRAPOLATION CHECK FOR CALCE")
    print(f"{'='*78}\n")
    print(f"Pooled training targets (other 6 datasets): "
          f"min={pool_min:.4f}, 1st pct={pool_p1:.4f}, 5th pct={pool_p5:.4f}")
    print(f"CALCE test targets: min={y_calce_test.min():.4f}, "
          f"mean={y_calce_test.mean():.4f}, n={len(y_calce_test)}")
    print()
    print(f"Fraction of CALCE test targets BELOW pool training minimum: "
          f"{frac_below_pool_min:.4f}")
    print(f"Fraction below pool's 1st percentile: {frac_below_pool_p1:.4f}")
    print(f"Fraction below pool's 5th percentile: {frac_below_pool_p5:.4f}")

    result = {
        "pool_train_min": pool_min, "pool_train_p1": pool_p1, "pool_train_p5": pool_p5,
        "calce_test_min": float(y_calce_test.min()), "calce_test_mean": float(y_calce_test.mean()),
        "n_calce_test": int(len(y_calce_test)),
        "frac_below_pool_min": frac_below_pool_min,
        "frac_below_pool_p1": frac_below_pool_p1,
        "frac_below_pool_p5": frac_below_pool_p5,
    }
    out_path = os.path.join(config.OUTPUT_DIR, "calce_output_range_results.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nsaved {out_path}")

    print(f"\n{'='*78}")
    print("INTERPRETATION: a large frac_below_pool_min/p1/p5 supports "
          "output-range extrapolation as CALCE's LODO failure mechanism, "
          "refining (not contradicting) Table 15's marginal-statistics "
          "finding. If this fraction is small, BOTH the input-space "
          "support-gap hypothesis (refuted by D2) AND this output-range "
          "hypothesis are ruled out, and the mechanism remains genuinely "
          "open -- report that honestly rather than defaulting back to "
          "the original 'support gap' language without evidence for it.")


if __name__ == "__main__":
    main()
