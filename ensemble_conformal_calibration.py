"""
Ensemble-normalised groupwise conformal calibration.

MC-Dropout captures uncertainty within a single trained model. It does not
capture variability across independently trained models (different random
seeds), which the multi-seed runs show to be substantial for some datasets
(notably MIT and RWTH).

This script uses the disagreement (prediction spread) across the three seed
models as the nonconformity normaliser, and then applies the same groupwise
(Mondrian) conformal calibration used elsewhere. The calibration framework is
unchanged; only the uncertainty source differs (cross-seed ensemble spread
instead of single-model MC-Dropout std).

Requires the three multi-seed checkpoints (outputs/multiseed_{42,123,2024}_
model.pt) and cache/processed.npz (the validation data used for calibration
is not stored in the *_predictions.npz files).

Run:
    python ensemble_conformal_calibration.py
Output:
    outputs/ensemble_conformal_results.json
    outputs/figures/ensemble_conformal_comparison.png
"""

import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from model import build_model

SEEDS = [42, 123, 2024]
ALPHA = getattr(config, "CONFORMAL_ALPHA", 0.10)
DATASET_ORDER = ["MIT", "BIT", "XJTU", "RWTH", "CALCE", "Oxford", "NASA"]
DATASET_COLORS = {
    "MIT": "#4C72B0", "BIT": "#DD8452", "XJTU": "#55A868", "RWTH": "#C44E52",
    "CALCE": "#8172B2", "Oxford": "#937860", "NASA": "#DA8BC3",
}


def _load_cache():
    path = os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(path):
        print(f"  [!] {path} not found -- run preprocessing.py first.")
        return None
    return np.load(path, allow_pickle=True)


def _predict_all_seeds(X, device):
    """Runs all 3 seed models (deterministic, dropout OFF -- eval mode) on
    the same input X, returning an array of shape (3, n_samples) of point
    predictions. This is the ensemble -- disagreement across these 3 rows
    is the new uncertainty source."""
    all_preds = []
    for seed in SEEDS:
        ckpt_path = os.path.join(config.OUTPUT_DIR, f"multiseed_{seed}_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [!] {ckpt_path} not found -- need all 3 multiseed "
                  f"checkpoints (saved automatically by train_one() during "
                  f"the multiseed section of experiments.py).")
            return None
        model = build_model("full").to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        with torch.no_grad():
            pred, _ = model(torch.tensor(X).to(device))
        all_preds.append(pred.cpu().numpy().flatten())
    return np.array(all_preds)  # shape (3, n_samples)


def groupwise_ensemble_calibrate(y_val, ensemble_mean_val, ensemble_std_val, lab_val, alpha=ALPHA):
    scores = np.abs(y_val - ensemble_mean_val) / (ensemble_std_val + 1e-6)
    qhat_per_group = {}
    for group in np.unique(lab_val):
        mask = lab_val == group
        group_scores = scores[mask]
        n = len(group_scores)
        if n == 0:
            continue
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        qhat_per_group[group] = float(np.quantile(group_scores, q_level))
    return qhat_per_group


def apply_intervals(ensemble_mean, ensemble_std, lab, qhat_per_group):
    lower = np.zeros_like(ensemble_mean)
    upper = np.zeros_like(ensemble_mean)
    for group, qhat in qhat_per_group.items():
        mask = lab == group
        lower[mask] = ensemble_mean[mask] - qhat * ensemble_std[mask]
        upper[mask] = ensemble_mean[mask] + qhat * ensemble_std[mask]
    return lower, upper


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache = _load_cache()
    if cache is None:
        return

    X_val, y_val, lab_val = cache["X_val"], cache["y_val"], cache["lab_val"]
    X_test, y_test, lab_test = cache["X_test"], cache["y_test"], cache["lab_test"]

    print("\nRunning all 3 seed models on validation set (for calibration)...")
    val_preds = _predict_all_seeds(X_val, device)
    if val_preds is None:
        return
    ensemble_mean_val = val_preds.mean(axis=0)
    ensemble_std_val = val_preds.std(axis=0)

    print("Running all 3 seed models on test set (for evaluation)...")
    test_preds = _predict_all_seeds(X_test, device)
    ensemble_mean_test = test_preds.mean(axis=0)
    ensemble_std_test = test_preds.std(axis=0)

    print("\nCalibrating groupwise conformal intervals using ENSEMBLE "
          "disagreement (not MC-Dropout) as the nonconformity normalizer...")
    qhat_per_group = groupwise_ensemble_calibrate(y_val, ensemble_mean_val, ensemble_std_val, lab_val)
    lower, upper = apply_intervals(ensemble_mean_test, ensemble_std_test, lab_test, qhat_per_group)

    overall_coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    overall_mae = float(np.mean(np.abs(ensemble_mean_test - y_test)))
    print(f"\nOverall ensemble-based coverage: {overall_coverage:.4f} (target {1-ALPHA:.2f})")
    print(f"Overall ensemble-mean MAE: {overall_mae:.5f}")

    print(f"\n{'='*78}")
    print("Per-dataset comparison: ensemble-based vs your existing single-model results")
    print(f"{'='*78}")
    print(f"{'Dataset':10s} {'Ensemble coverage':>18s} {'Ensemble MAE':>14s} {'Mean ensemble_std':>18s}")
    per_dataset = {}
    datasets_present = [d for d in DATASET_ORDER if d in set(np.unique(lab_test))]
    for ds in datasets_present:
        mask = lab_test == ds
        cov = float(np.mean((y_test[mask] >= lower[mask]) & (y_test[mask] <= upper[mask])))
        mae = float(np.mean(np.abs(ensemble_mean_test[mask] - y_test[mask])))
        mean_std = float(ensemble_std_test[mask].mean())
        per_dataset[ds] = {"coverage": cov, "mae": mae, "mean_ensemble_std": mean_std}
        print(f"{ds:10s} {cov:17.4f}  {mae:13.5f}  {mean_std:17.5f}")

    print(f"\nCompare the 'Ensemble coverage' column above directly against "
          f"Table 6's single-model groupwise coverage for MIT and RWTH "
          f"specifically -- these are the two datasets the ensemble "
          f"approach is intended to help, since their coverage instability "
          f"(Table 8) was NOT explained by finite-sample size (Table 7).")

    out = {
        "overall_coverage": overall_coverage,
        "overall_mae": overall_mae,
        "qhat_per_group": qhat_per_group,
        "per_dataset": per_dataset,
    }
    out_path = os.path.join(config.OUTPUT_DIR, "ensemble_conformal_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {out_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(datasets_present))
    covs = [per_dataset[d]["coverage"] * 100 for d in datasets_present]
    colors = [DATASET_COLORS.get(d, "#999999") for d in datasets_present]
    ax.bar(x, covs, color=colors)
    ax.axhline((1 - ALPHA) * 100, color="k", linestyle="--", linewidth=1, label=f"{(1-ALPHA)*100:.0f}% target")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_present, rotation=30, ha="right")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Ensemble-Augmented Groupwise Conformal Coverage\n(cross-seed disagreement as uncertainty source)")
    ax.legend()
    fig.tight_layout()
    fig_dir = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_fig = os.path.join(fig_dir, "ensemble_conformal_comparison.png")
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
