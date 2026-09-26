"""
CALCE input-space overlap with the pooled training data (propensity analysis).

Tests whether CALCE's poor leave-one-dataset-out transfer can be explained by
a support gap in the joint input-feature space the model conditions on
(rather than only in marginal SOH statistics).

Method: fit a classifier to distinguish CALCE windows from pooled training
windows of the other six datasets, using the same input features the BiLSTM
sees. If CALCE were out of support, the classifier would separate the classes
easily (most CALCE windows with propensity > 0.9); if CALCE is well mixed with
the pool, propensities spread across [0, 1] and the AUC is close to 0.5.

Decision criteria (fixed before the analysis):
  - Supports a support gap: a high fraction of CALCE windows with propensity > 0.9.
  - Refutes it: well-mixed propensities (AUC close to 0.5).
  - Ambiguous: moderate separation without a clear bimodal split; the full
    histogram is reported.

Run:
    python calce_propensity_overlap.py
Output:
    outputs/calce_propensity_overlap.json
    outputs/figures/calce_propensity_overlap.png
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

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

    X_parts, lab_parts = [], []
    for split in ("train", "val", "test"):
        X_key, lab_key = f"X_{split}", f"lab_{split}"
        if X_key not in cache.files or lab_key not in cache.files:
            print(f"  [!] cache missing {X_key}/{lab_key}")
            return
        X_parts.append(cache[X_key].reshape(cache[X_key].shape[0], -1))
        lab_parts.append(cache[lab_key])
    X_all = np.concatenate(X_parts, axis=0)
    lab_all = np.concatenate(lab_parts, axis=0)

    is_calce = (lab_all == "CALCE").astype(int)
    n_calce = int(is_calce.sum())
    n_pool = int((1 - is_calce).sum())
    print(f"CALCE windows: {n_calce}, pooled other-dataset windows: {n_pool}")
    if n_calce < 10:
        print("  [!] Very few CALCE windows -- propensity estimates will be noisy. "
              "Proceed, but treat the result cautiously and report the raw counts.")

    print("\nFitting logistic regression classifier (is_CALCE vs. pooled)...")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    propensities = cross_val_predict(clf, X_all, is_calce, cv=5, method="predict_proba")[:, 1]

    auc = roc_auc_score(is_calce, propensities)
    calce_propensities = propensities[is_calce == 1]
    pool_propensities = propensities[is_calce == 0]

    frac_high_confidence = float(np.mean(calce_propensities > 0.9))

    print(f"\n{'='*78}")
    print("D2: PROPENSITY-OVERLAP DIAGNOSTIC FOR CALCE")
    print(f"{'='*78}")
    print(f"Classifier AUC (is_CALCE vs. pooled): {auc:.4f}")
    print(f"  (AUC near 0.5 = poor separation/good overlap; AUC near 1.0 = "
          f"easy separation/poor overlap)")
    print(f"Fraction of CALCE windows with propensity > 0.9: {frac_high_confidence:.4f}")
    print(f"Mean CALCE propensity: {calce_propensities.mean():.4f}")
    print(f"Mean pool propensity: {pool_propensities.mean():.4f}")

    result = {
        "auc": float(auc),
        "frac_calce_propensity_gt_0.9": frac_high_confidence,
        "mean_calce_propensity": float(calce_propensities.mean()),
        "mean_pool_propensity": float(pool_propensities.mean()),
        "n_calce": n_calce,
        "n_pool": n_pool,
    }
    out_path = os.path.join(config.OUTPUT_DIR, "calce_propensity_overlap.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nsaved {out_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pool_propensities, bins=40, alpha=0.6, label="Pooled (other 6 datasets)", color="#4C72B0", density=True)
    ax.hist(calce_propensities, bins=40, alpha=0.6, label="CALCE", color="#C44E52", density=True)
    ax.axvline(0.9, color="k", linestyle="--", linewidth=1, label="propensity = 0.9")
    ax.set_xlabel("Propensity score (P(is_CALCE))")
    ax.set_ylabel("Density")
    ax.set_title(f"CALCE vs. pooled propensity overlap (AUC={auc:.3f})")
    ax.legend()
    fig.tight_layout()
    fig_dir = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_fig = os.path.join(fig_dir, "calce_propensity_overlap.png")
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"saved {out_fig}")

    print(f"\n{'='*78}")
    print("INTERPRETATION: high AUC + high frac(propensity>0.9) supports the "
          "support-gap claim already made in the manuscript from marginal "
          "statistics alone (Table 15), now with joint-feature-space "
          "evidence. Low AUC would mean that claim needs revisiting BEFORE "
          "running importance_weighted_lodo.py, since a null result there "
          "would then be uninformative rather than confirmatory.")


if __name__ == "__main__":
    main()
