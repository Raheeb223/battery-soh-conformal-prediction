"""
CALCE target-range extrapolation check (leave-one-dataset-out setting).

Complements calce_propensity_overlap.py, which tests whether CALCE's INPUT
windows are distinguishable from the pooled training windows of the other six
datasets. This script tests the output side: whether CALCE's TARGET SOH
values fall below the range of targets seen in the pooled training data, in
which case the model would be asked to extrapolate outside its trained output
range.

Method: de-normalise the per-dataset z-scored targets back to raw SOH (using
the soh_mean / soh_std saved in cache/processed.npz), then compare CALCE's
test-set targets against the minimum and low percentiles of the pooled
training targets of the other six datasets.

Decision criteria (fixed before the analysis):
  - Supports output-range extrapolation: a large fraction of CALCE test targets
    fall below the pool's training-target minimum (or its 1st percentile).
  - Refutes it: CALCE test targets lie within the pool's training-target range.
  - Ambiguous: a moderate fraction (e.g. 10-40%) below range; the fraction and
    distribution are reported rather than a binary conclusion.

Run:
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

    if "soh_mean" not in cache.files or "soh_std" not in cache.files:
        print("  [!] cache is missing soh_mean/soh_std -- cannot de-normalize. "
              "These are saved by preprocessing.py's run() function as "
              "per-dataset dicts; re-run preprocessing.py if this cache "
              "predates that.")
        return
    soh_mean = cache["soh_mean"].item()  # saved as a 0-d object array holding a dict
    soh_std = cache["soh_std"].item()

    print("De-normalizing z-scored SOH back to raw units before comparing "
          "-- y_train/y_test are z-scored PER DATASET (each using its own "
          "mean/std, per preprocessing.py), so comparing them directly "
          "across datasets does not test absolute SOH range at all; it was "
          "actively hiding the cross-dataset signal Table 15's finding "
          "depends on. Raw SOH is bounded to roughly (0, 1.10) by this "
          "pipeline's own filter -- a z-scored value like -4.5 is not a "
          "real SOH value, and treating it as one would have been wrong.\n")

    def denormalize(y, lab, means, stds):
        raw = np.zeros_like(y)
        for ds in np.unique(lab):
            mask = lab == ds
            raw[mask] = y[mask] * stds[ds] + means[ds]
        return raw

    y_train_raw = denormalize(y_train, lab_train, soh_mean, soh_std)
    y_test_raw = denormalize(y_test, lab_test, soh_mean, soh_std)

    pool_train_mask = lab_train != "CALCE"
    y_pool_train_raw = y_train_raw[pool_train_mask]

    calce_test_mask = lab_test == "CALCE"
    y_calce_test_raw = y_test_raw[calce_test_mask]

    pool_min = float(y_pool_train_raw.min())
    pool_p1 = float(np.percentile(y_pool_train_raw, 1))
    pool_p5 = float(np.percentile(y_pool_train_raw, 5))

    frac_below_pool_min = float(np.mean(y_calce_test_raw < pool_min))
    frac_below_pool_p1 = float(np.mean(y_calce_test_raw < pool_p1))
    frac_below_pool_p5 = float(np.mean(y_calce_test_raw < pool_p5))

    print(f"{'='*78}")
    print("D2-FOLLOWUP: OUTPUT-RANGE EXTRAPOLATION CHECK FOR CALCE (raw SOH units)")
    print(f"{'='*78}\n")
    print(f"Pooled training targets (other 6 datasets), RAW SOH: "
          f"min={pool_min:.4f}, 1st pct={pool_p1:.4f}, 5th pct={pool_p5:.4f}")
    print(f"CALCE test targets, RAW SOH: min={y_calce_test_raw.min():.4f}, "
          f"mean={y_calce_test_raw.mean():.4f}, n={len(y_calce_test_raw)}")
    print()
    print(f"Fraction of CALCE test targets BELOW pool training minimum: "
          f"{frac_below_pool_min:.4f}")
    print(f"Fraction below pool's 1st percentile: {frac_below_pool_p1:.4f}")
    print(f"Fraction below pool's 5th percentile: {frac_below_pool_p5:.4f}")

    result = {
        "pool_train_min": pool_min, "pool_train_p1": pool_p1, "pool_train_p5": pool_p5,
        "calce_test_min": float(y_calce_test_raw.min()), "calce_test_mean": float(y_calce_test_raw.mean()),
        "n_calce_test": int(len(y_calce_test_raw)),
        "frac_below_pool_min": frac_below_pool_min,
        "frac_below_pool_p1": frac_below_pool_p1,
        "frac_below_pool_p5": frac_below_pool_p5,
        "units": "raw SOH (de-normalized)",
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