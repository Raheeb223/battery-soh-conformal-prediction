"""
diagnose_xjtu_value_by_cell.py

The definitive check. ablation_full_predictions.npz and
conformal_normalized_groupwise_predictions.npz are now CONFIRMED fresh
(file size jumped from 1,514,696 to 5,366,738 bytes -- the direct
fingerprint of the new train.py's cell_test field actually being saved).
Yet XJTU's value range is still exactly [-7.224, 1.033] with 2387 samples.

Rather than keep inferring from aggregate stats, use the now-available
cell_test field to answer directly:
  1. Exactly how many samples equal the minimum value, and how precisely
     identical are they (the original anomaly signature: bit-identical
     to 4 decimal places)?
  2. WHICH cell(s) do those samples belong to?
  3. Are those cells Sim_satellite_* (the ones the fix targets) or a
     completely different XJTU cell/batch (meaning this was never the
     bug we thought it was, and is a separate, real, legitimate --
     or differently-broken -- data point)?

Run:
    python diagnose_xjtu_value_by_cell.py
"""

import os
import glob
import numpy as np
import config

RUN_NAMES = ["ablation_full", "conformal_normalized_groupwise"]


def _load(run_name):
    candidates = glob.glob(os.path.join(config.OUTPUT_DIR, f"*{run_name}*predictions.npz"))
    if not candidates:
        print(f"  [skip] no predictions file found for {run_name}")
        return None
    return np.load(candidates[0], allow_pickle=True)


def main():
    for run_name in RUN_NAMES:
        print(f"\n=== {run_name} ===")
        data = _load(run_name)
        if data is None:
            continue

        if "cell_test" not in data.files:
            print(f"  [!] this file has no cell_test field -- it's still an "
                  f"OLD predictions file despite what the size suggested. "
                  f"Available fields: {data.files}")
            continue

        y_test = data["y_test"]
        lab_test = data["lab_test"]
        cell_test = data["cell_test"]

        mask = lab_test == "XJTU"
        yt = y_test[mask]
        ct = cell_test[mask]
        n = len(yt)
        print(f"  XJTU test samples: {n}")

        min_val = yt.min()
        # exact-match count at the minimum (the original anomaly was
        # "identical to 4 decimal places", so round before comparing to
        # avoid float noise hiding an otherwise-exact cluster)
        exact_match = np.isclose(yt, min_val, atol=1e-4)
        n_exact = int(exact_match.sum())
        print(f"  Minimum value: {min_val:.6f}")
        print(f"  Samples within 1e-4 of the minimum: {n_exact} "
              f"({100*n_exact/n:.2f}%)")

        if n_exact > 0:
            cells_at_min = ct[exact_match]
            uniq_cells, counts = np.unique(cells_at_min, return_counts=True)
            print(f"  Cell(s) responsible for these {n_exact} samples:")
            for cell_id, count in sorted(zip(uniq_cells, counts), key=lambda t: -t[1]):
                is_satellite = "Sim_satellite" in str(cell_id)
                tag = "SATELLITE (should be fixed)" if is_satellite else "NOT SATELLITE -- different cause"
                print(f"    {cell_id}: {count} samples  <-- {tag}")

            all_satellite = all("Sim_satellite" in str(c) for c in uniq_cells)
            if all_satellite:
                print(f"\n  [!!!] All samples at the minimum come from Sim_satellite_* "
                      f"cells. If this count is still ~79 and still forms a tight "
                      f"repeated cluster, the calibration-block fix has NOT "
                      f"actually reached these specific windows -- worth checking "
                      f"whether these particular satellite cells landed in the "
                      f"TEST split specifically (not train/val), and whether their "
                      f"post-strip row count still produces windows that bridge "
                      f"across where a block was removed (the known, documented "
                      f"windowing-discontinuity limitation).")
            else:
                print(f"\n  [!] These samples are NOT all from satellite cells. "
                      f"This was very likely never the calibration-block bug at "
                      f"all -- it's either a different, legitimate XJTU data "
                      f"point (e.g. genuine end-of-life samples clustering near "
                      f"a normalization floor) or a separate issue entirely. "
                      f"Do not assume the earlier fix is supposed to touch this.")
        else:
            print("  No tight cluster at the minimum -- looks like ordinary "
                  "data now, not a repeated-value artifact.")


if __name__ == "__main__":
    main()
