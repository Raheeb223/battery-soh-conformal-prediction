"""
Evaluation on cycling protocols closest to real-world operation.

Scope: this is not field validation. It isolates the two protocols in the
existing laboratory datasets that are documented as closer to real-world
operation than constant-current cycling, and compares the model's accuracy
and calibration on them with the remaining protocols of the same dataset:

  - XJTU Batch-5 ("RW", real-world profile) vs the other XJTU batches
    (2C/3C constant current, randomised R2.5/R3 profiles).
  - BIT "arbitrary" current-profile cells vs BIT "fixed" current-profile cells.

Comparable performance on the irregular protocols is supporting evidence of
robustness to realistic load profiles, not proof of field performance.

Run:
    python field_proxy_validation.py
Output:
    outputs/field_proxy_validation.json
    outputs/figures/field_proxy_comparison.png
"""

import os
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def _load_predictions(run_name="ablation_full"):
    candidates = glob.glob(os.path.join(config.OUTPUT_DIR, f"*{run_name}*predictions.npz"))
    if not candidates:
        print(f"  [!] no predictions file found for {run_name}")
        return None
    data = np.load(candidates[0], allow_pickle=True)
    required = {"y_test", "mean_pred", "lower", "upper", "lab_test", "cell_test"}
    missing = required - set(data.files)
    if missing:
        print(f"  [!] {candidates[0]} missing {missing} -- need cell_test to "
              f"identify specific cells by protocol. Was this saved by the "
              f"current train.py?")
        return None
    return data


def evaluate_subset(y_true, mean_pred, lower, upper, label):
    mae = float(np.mean(np.abs(mean_pred - y_true)))
    rmse = float(np.sqrt(np.mean((mean_pred - y_true) ** 2)))
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    n = len(y_true)
    print(f"    {label:30s}  n={n:6d}  MAE={mae:.5f}  RMSE={rmse:.5f}  coverage={coverage:.4f}")
    return {"n": n, "mae": mae, "rmse": rmse, "coverage": coverage}


def main():
    data = _load_predictions("ablation_full")
    if data is None:
        return

    y_true = data["y_test"]
    mean_pred = data["mean_pred"]
    lower = data["lower"]
    upper = data["upper"]
    lab_test = data["lab_test"]
    cell_test = data["cell_test"]

    results = {}

    # --- XJTU: Batch-5 (real-world profile) vs other batches ---
    print(f"\n{'='*78}")
    print("XJTU: Batch-5 (real-world profile) vs other batches (idealized protocols)")
    print(f"{'='*78}")
    xjtu_mask = lab_test == "XJTU"
    if xjtu_mask.sum() == 0:
        print("  [skip] no XJTU samples in this predictions file.")
    else:
        cells_xjtu = cell_test[xjtu_mask]
        y_xjtu, pred_xjtu = y_true[xjtu_mask], mean_pred[xjtu_mask]
        lo_xjtu, hi_xjtu = lower[xjtu_mask], upper[xjtu_mask]

        rw_mask = np.array(["Batch-5" in c for c in cells_xjtu])
        other_mask = ~rw_mask

        if rw_mask.sum() == 0:
            print("  [skip] no Batch-5 (RW) cells found in the test split for "
                  "this run -- try a different predictions file, or note "
                  "that Batch-5 cells simply didn't land in this particular "
                  "test split (cell-level random assignment).")
        else:
            results["xjtu_batch5_rw"] = evaluate_subset(
                y_xjtu[rw_mask], pred_xjtu[rw_mask], lo_xjtu[rw_mask], hi_xjtu[rw_mask],
                "XJTU Batch-5 (real-world profile)")
            results["xjtu_other_batches"] = evaluate_subset(
                y_xjtu[other_mask], pred_xjtu[other_mask], lo_xjtu[other_mask], hi_xjtu[other_mask],
                "XJTU other batches (idealized)")

    # --- BIT: arbitrary vs fixed current profile ---
    print(f"\n{'='*78}")
    print("BIT: 'arbitrary' current-profile cells vs 'fixed' current-profile cells")
    print(f"{'='*78}")
    bit_mask = lab_test == "BIT"
    if bit_mask.sum() == 0:
        print("  [skip] no BIT samples in this predictions file.")
    else:
        cells_bit = cell_test[bit_mask]
        y_bit, pred_bit = y_true[bit_mask], mean_pred[bit_mask]
        lo_bit, hi_bit = lower[bit_mask], upper[bit_mask]

        print(f"    (sample BIT cell IDs in this test split: {list(np.unique(cells_bit))[:5]})")
        arbitrary_mask = np.array(["arbitrary" in c.lower() for c in cells_bit])
        fixed_mask = np.array(["fixed" in c.lower() for c in cells_bit])

        if arbitrary_mask.sum() == 0 or fixed_mask.sum() == 0:
            print("  [!] could not cleanly split BIT cells by 'arbitrary'/'fixed' "
                  "in the cell_id string -- check the printed sample IDs above "
                  "and adjust the substring match in this script if your "
                  "bit_loader.py uses different naming.")
        else:
            results["bit_arbitrary"] = evaluate_subset(
                y_bit[arbitrary_mask], pred_bit[arbitrary_mask], lo_bit[arbitrary_mask], hi_bit[arbitrary_mask],
                "BIT arbitrary profile (more field-like)")
            results["bit_fixed"] = evaluate_subset(
                y_bit[fixed_mask], pred_bit[fixed_mask], lo_bit[fixed_mask], hi_bit[fixed_mask],
                "BIT fixed profile (idealized)")

    if not results:
        print("\n[!] No comparisons could be computed -- check the warnings "
              "above (likely a test-split composition issue, not a script bug).")
        return

    out_path = os.path.join(config.OUTPUT_DIR, "field_proxy_validation.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved {out_path}")

    labels = list(results.keys())
    maes = [results[k]["mae"] for k in labels]
    coverages = [results[k]["coverage"] * 100 for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(labels))
    axes[0].bar(x, maes, color="#4C72B0")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("MAE: field-like vs idealized protocols")

    axes[1].bar(x, coverages, color="#C44E52")
    axes[1].axhline(90, color="k", linestyle="--", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_title("Coverage: field-like vs idealized protocols")

    fig.suptitle("Field-Proxy Validation (partial, within-dataset only -- see script docstring)", fontsize=11)
    fig.tight_layout()
    fig_dir = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_fig = os.path.join(fig_dir, "field_proxy_comparison.png")
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"saved {out_fig}")

    print(f"\n{'='*78}")
    print("REMINDER: this is a WITHIN-DATASET proxy comparison using existing "
          "lab data with more irregular cycling protocols. It is NOT field "
          "validation. Report it in the manuscript with this scope explicitly "
          "stated, not as evidence of real-world deployment readiness.")


if __name__ == "__main__":
    main()
