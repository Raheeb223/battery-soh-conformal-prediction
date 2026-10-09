"""
Compares MC-Dropout uncertainty against DEEP ENSEMBLE uncertainty (using
your 3 already-trained multi-seed models — no retraining needed) to check
whether ensembling gives a better-calibrated / more useful uncertainty
estimate than MC-Dropout, particularly for small datasets like CALCE where
MC-Dropout's AUSE was poor (0.30).

Ensemble uncertainty = the variance across the 3 independently-initialized
models' point predictions for the same input. This is a genuinely different
signal from MC-Dropout (which measures a SINGLE model's internal dropout
noise) — ensemble disagreement instead captures how sensitive the learned
function itself is to random initialization/training dynamics, which is a
more direct measure of "how much do we actually know" for a given input.

Honest expectation: this MIGHT improve on MC-Dropout, or it might not —
report whichever happens. Only 3 ensemble members is a small ensemble
(5-10 is more typical), so treat this as a preliminary comparison, not a
definitive one.

Run:
    python ensemble_uncertainty_comparison.py
Requires: multiseed_42/123/2024_model.pt (already trained) and
cache/processed.npz (for X_test, y_test, lab_test).
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from model import build_model, predict_in_batches
from ause_analysis import sparsification_curves, ause_score, FIG_DIR

SEEDS = [42, 123, 2024]


def load_ensemble_predictions(cache_path=None):
    cache_path = cache_path or os.path.join(config.CACHE_DIR, "processed.npz")
    cache = np.load(cache_path, allow_pickle=True)
    X_test, y_test, lab_test = cache["X_test"], cache["y_test"], cache["lab_test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = torch.tensor(X_test).to(device)

    all_preds = []
    for seed in SEEDS:
        model_path = os.path.join(config.OUTPUT_DIR, f"multiseed_{seed}_model.pt")
        if not os.path.exists(model_path):
            print(f"  [!] missing {model_path} — cannot build ensemble without all 3 seeds")
            return None
        model = build_model("full").to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        pred = predict_in_batches(model, x_tensor).cpu().numpy()
        all_preds.append(pred)

    all_preds = np.stack(all_preds, axis=0)  # (3, N)
    ensemble_mean = all_preds.mean(axis=0)
    ensemble_std = all_preds.std(axis=0)
    return ensemble_mean, ensemble_std, y_test, lab_test


def compare(cache_path=None):
    result = load_ensemble_predictions(cache_path)
    if result is None:
        return None
    ens_mean, ens_std, y_test, lab_test = result

    # load the existing MC-Dropout results for direct comparison
    import glob
    candidates = glob.glob(os.path.join(config.OUTPUT_DIR, "*conformal_normalized_groupwise*predictions.npz"))
    if not candidates:
        print("  [!] no conformal_normalized_groupwise predictions found for comparison")
        return None
    mcd = np.load(candidates[0], allow_pickle=True)
    mcd_mean, mcd_std = mcd["mean_pred"], mcd["mc_std"]

    print(f"[ensemble] built {len(SEEDS)}-model ensemble uncertainty")

    # overall AUSE comparison
    frac, ens_curve, oracle_curve, rand_curve = sparsification_curves(y_test, ens_mean, ens_std)
    ause_ensemble = ause_score(frac, ens_curve, oracle_curve)
    frac2, mcd_curve, oracle_curve2, rand_curve2 = sparsification_curves(y_test, mcd_mean, mcd_std)
    ause_mcd = ause_score(frac2, mcd_curve, oracle_curve2)

    print(f"  AUSE (ensemble uncertainty, vs oracle) = {ause_ensemble:.5f}")
    print(f"  AUSE (MC-Dropout uncertainty, vs oracle) = {ause_mcd:.5f}")
    better = "ensemble" if ause_ensemble < ause_mcd else "MC-Dropout"
    print(f"  -> {better} uncertainty is more useful for ranking samples by difficulty "
          f"({'lower' if better=='ensemble' else 'higher'} AUSE = better)")

    corr_ens = float(np.corrcoef(ens_std, np.abs(y_test - ens_mean))[0, 1])
    corr_mcd = float(np.corrcoef(mcd_std, np.abs(y_test - mcd_mean))[0, 1])
    print(f"\n  correlation(uncertainty, |error|): ensemble={corr_ens:.4f}  MC-Dropout={corr_mcd:.4f}")

    print("\n  Per-dataset AUSE comparison:")
    per_dataset = {}
    for ds in np.unique(lab_test):
        mask = lab_test == ds
        if mask.sum() < 10:
            continue
        f_e, c_e, o_e, r_e = sparsification_curves(y_test[mask], ens_mean[mask], ens_std[mask])
        ause_e = ause_score(f_e, c_e, o_e)
        f_m, c_m, o_m, r_m = sparsification_curves(y_test[mask], mcd_mean[mask], mcd_std[mask])
        ause_m = ause_score(f_m, c_m, o_m)
        per_dataset[ds] = {"ause_ensemble": ause_e, "ause_mcdropout": ause_m}
        print(f"    {ds}: ensemble={ause_e:.5f}  MC-Dropout={ause_m:.5f}  "
              f"{'ensemble better' if ause_e < ause_m else 'MC-Dropout better'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(frac, ens_curve, "o-", label="ensemble uncertainty", color="#4c72b0")
    ax.plot(frac2, mcd_curve, "s-", label="MC-Dropout uncertainty", color="#dd8452")
    ax.plot(frac, oracle_curve, "--", label="oracle", color="#55a868")
    ax.set_xlabel("Fraction of samples removed (most uncertain first)")
    ax.set_ylabel("MAE on remaining samples")
    ax.set_title("Ensemble vs. MC-Dropout uncertainty: sparsification comparison")
    ax.legend()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "ensemble_vs_mcdropout_sparsification.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {os.path.join(FIG_DIR, 'ensemble_vs_mcdropout_sparsification.png')}")

    results = {
        "ause_ensemble_overall": ause_ensemble,
        "ause_mcdropout_overall": ause_mcd,
        "correlation_ensemble": corr_ens,
        "correlation_mcdropout": corr_mcd,
        "per_dataset": per_dataset,
        "note": "Ensemble uses only 3 members (your multi-seed runs) — a small "
                "ensemble; treat as a preliminary comparison, not definitive.",
    }
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "ensemble_uncertainty_comparison.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[ensemble] saved to {config.OUTPUT_DIR}/ensemble_uncertainty_comparison.json")
    return results


if __name__ == "__main__":
    compare()
