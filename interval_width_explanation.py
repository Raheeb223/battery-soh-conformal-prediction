"""
Explains WHY each test sample's conformal interval is wide or narrow —
addressing the literature-review finding (Task 6) that battery XAI work
explains POINT predictions (e.g. SHAP on SOH value) but not interval width.

This ONLY works meaningfully with the 'normalized_groupwise' conformal mode
(conformal.py), since plain groupwise/pooled conformal gives every sample in
a dataset the exact same width — there's nothing per-sample to explain.

Method: engineer a handful of interpretable features from each test
sample's INPUT WINDOW (the 20 past SOH values the model saw), then fit a
small, transparent model (Random Forest) to predict that sample's interval
width from these features. Feature importances (permutation importance,
and SHAP if installed) tell you which properties of a cell's recent history
drive wider/narrower intervals — e.g. "cells with more volatile recent SOH
get wider intervals" is a physically interpretable, checkable claim.

Features (per sample, computed from its own input window):
  - window_std     : volatility of the recent SOH trace
  - window_slope    : linear trend (degradation rate) over the window
  - window_last     : most recent SOH value (roughly "how degraded is it now")
  - window_mean      : average SOH level over the window
  - window_range     : max-min over the window

Run:
    python interval_width_explanation.py
Requires: cache/processed.npz (for X_test, to recompute window features)
and a normalized_groupwise predictions.npz (for interval width / mc_std).
"""

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import config

FIG_DIR = os.path.join(config.OUTPUT_DIR, "figures")


def engineer_window_features(X: np.ndarray) -> dict:
    """X: (N, window, 1) normalized SOH windows -> dict of per-sample feature arrays."""
    seq = X[:, :, 0]  # (N, window)
    t = np.arange(seq.shape[1])
    features = {}
    features["window_std"] = seq.std(axis=1)
    features["window_last"] = seq[:, -1]
    features["window_mean"] = seq.mean(axis=1)
    features["window_range"] = seq.max(axis=1) - seq.min(axis=1)
    # per-sample linear slope via least-squares on each row
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    slopes = ((seq - seq.mean(axis=1, keepdims=True)) * (t - t_mean)).sum(axis=1) / t_var
    features["window_slope"] = slopes
    return features


def _find_predictions(name_substring):
    candidates = glob.glob(os.path.join(config.OUTPUT_DIR, f"*{name_substring}*predictions.npz"))
    return candidates[0] if candidates else None


def analyze(cache_path=None, run_name_substring="conformal_normalized_groupwise"):
    cache_path = cache_path or os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(cache_path):
        print(f"[interval_width] cache not found: {cache_path} — run preprocessing.py first")
        return None

    pred_path = _find_predictions(run_name_substring)
    if pred_path is None:
        print(f"[interval_width] no '{run_name_substring}' predictions found. "
              f"This analysis requires the normalized_groupwise conformal run "
              f"(plain groupwise/pooled give constant width per dataset — nothing to explain). "
              f"Run experiments.py (which now includes this mode) first.")
        return None

    cache = np.load(cache_path, allow_pickle=True)
    preds = np.load(pred_path, allow_pickle=True)

    X_test = cache["X_test"]
    y_test, mean_pred = preds["y_test"], preds["mean_pred"]
    lower, upper = preds["lower"], preds["upper"]
    lab_test = preds["lab_test"]
    width = upper - lower

    if len(X_test) != len(width):
        print(f"[interval_width] SIZE MISMATCH: cache X_test has {len(X_test)} samples, "
              f"predictions has {len(width)}. These must come from the SAME cache/run — "
              f"if you changed window size or re-ran preprocessing since this prediction "
              f"file was generated, re-run the normalized_groupwise conformal training first.")
        return None

    feats = engineer_window_features(X_test)
    feature_names = list(feats.keys())
    Xf = np.column_stack([feats[k] for k in feature_names])

    # fit a simple RF to predict interval width from engineered features
    Xf_train, Xf_test, w_train, w_test = train_test_split(
        Xf, width, test_size=0.3, random_state=config.RANDOM_SEED
    )
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=config.RANDOM_SEED, n_jobs=-1)
    model.fit(Xf_train, w_train)
    r2 = model.score(Xf_test, w_test)
    print(f"[interval_width] RF explaining interval width from window features: "
          f"held-out R^2 = {r2:.3f}")
    if r2 < 0.1:
        print("  -> low R^2: these simple engineered features explain little of the "
          "interval-width variation; width may be dominated by group membership "
          "and/or model internals not captured by these features. Report this honestly "
          "as a limitation rather than forcing a narrative.")

    perm = permutation_importance(model, Xf_test, w_test, n_repeats=20,
                                   random_state=config.RANDOM_SEED, n_jobs=-1)
    importances = dict(zip(feature_names, perm.importances_mean))
    print("\n  Permutation importances (higher = more explains interval width):")
    for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {name}: {imp:.5f}")

    # try SHAP if available, for a richer per-sample explanation
    shap_summary = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xf_test)
        mean_abs_shap = dict(zip(feature_names, np.abs(shap_values).mean(axis=0)))
        print("\n  Mean |SHAP value| per feature:")
        for name, val in sorted(mean_abs_shap.items(), key=lambda x: -x[1]):
            print(f"    {name}: {val:.5f}")
        shap_summary = mean_abs_shap

        fig, ax = plt.subplots(figsize=(7, 5))
        shap.summary_plot(shap_values, Xf_test, feature_names=feature_names, show=False)
        os.makedirs(FIG_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIG_DIR, "interval_width_shap_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved {os.path.join(FIG_DIR, 'interval_width_shap_summary.png')}")
    except ImportError:
        print("\n  [note] shap not installed (pip install shap) — reporting permutation "
              "importance only, which is a valid alternative but less detailed per-sample.")

    # bar chart of permutation importances (always produced, no extra dependency)
    fig, ax = plt.subplots(figsize=(7, 5))
    names_sorted = sorted(importances, key=lambda k: -importances[k])
    ax.barh(names_sorted, [importances[n] for n in names_sorted], color="#4c72b0")
    ax.set_xlabel("Permutation importance (mean decrease in R^2 when shuffled)")
    ax.set_title("What explains conformal interval width?")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "interval_width_importance.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {os.path.join(FIG_DIR, 'interval_width_importance.png')}")

    results = {
        "held_out_r2": float(r2),
        "permutation_importance": {k: float(v) for k, v in importances.items()},
        "shap_mean_abs": {k: float(v) for k, v in shap_summary.items()} if shap_summary else None,
    }
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "interval_width_explanation.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[interval_width] saved to {config.OUTPUT_DIR}/interval_width_explanation.json")
    return results


if __name__ == "__main__":
    analyze()