"""
Quality of the MC-Dropout uncertainty estimate.

Computed from the final prediction files; nothing is retrained.

1. Reliability: conformal calibration guarantees coverage only at the
   calibrated level (90%). This checks whether the MC-Dropout uncertainty
   (mean_pred, mc_std) is calibrated in a Gaussian sense across several
   nominal levels (50/70/80/90/95%), using z-scaled intervals from mc_std.
   This is an approximation (one mc_std scaled by different z-values), not an
   exact ECE.

2. AUSE (area under the sparsification error): whether mc_std ranks samples
   by error. The most uncertain samples are removed progressively and the MAE
   of the remainder is tracked, compared with the oracle curve (removal by
   true error) and random removal. Lower AUSE means better ranking.

Run:
    python uq_quality_analysis.py
Output:
    outputs/uq_quality_analysis.json
    outputs/figures/uq_reliability_and_sparsification.png
"""

import os
import glob
import json

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# np.trapz was removed in NumPy 2.0 (renamed np.trapezoid); this works
# correctly against both old and new NumPy installations.
_trapz = getattr(np, "trapezoid", None) or np.trapz

CONFIDENCE_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]
DATASET_ORDER = ["MIT", "BIT", "XJTU", "RWTH", "CALCE", "Oxford", "NASA"]
DATASET_COLORS = {
    "MIT": "#4C72B0", "BIT": "#DD8452", "XJTU": "#55A868", "RWTH": "#C44E52",
    "CALCE": "#8172B2", "Oxford": "#937860", "NASA": "#DA8BC3",
}


def _load_predictions(run_name="ablation_full"):
    candidates = glob.glob(os.path.join(config.OUTPUT_DIR, f"*{run_name}*predictions.npz"))
    if not candidates:
        print(f"  [!] no predictions file found for {run_name}")
        return None
    data = np.load(candidates[0], allow_pickle=True)
    required = {"y_test", "mean_pred", "mc_std"}
    missing = required - set(data.files)
    if missing:
        print(f"  [!] {candidates[0]} is missing required keys {missing} -- "
              f"cannot run this analysis without mc_std (MC-Dropout "
              f"uncertainty). Was this predictions file saved by the "
              f"current train.py?")
        return None
    return data


# ---------------------------------------------------------------------------
# 1. Reliability / approximate ECE
# ---------------------------------------------------------------------------

def reliability_analysis(y_true, mean_pred, mc_std):
    results = []
    for level in CONFIDENCE_LEVELS:
        z = norm.ppf((1 + level) / 2)
        lo = mean_pred - z * mc_std
        hi = mean_pred + z * mc_std
        actual_coverage = float(np.mean((y_true >= lo) & (y_true <= hi)))
        abs_error = abs(level - actual_coverage)
        results.append({
            "target": level,
            "actual_coverage": actual_coverage,
            "abs_calibration_error": abs_error,
        })
        print(f"    target={level:.2f}  actual={actual_coverage:.4f}  "
              f"|error|={abs_error:.4f}")
    ece_approx = float(np.mean([r["abs_calibration_error"] for r in results]))
    return results, ece_approx


# ---------------------------------------------------------------------------
# 2. AUSE (Area Under Sparsification Error)
# ---------------------------------------------------------------------------

def compute_sparsification_curve(errors, sort_key, n_points=20):
    """Removes the top-k fraction of samples (ranked by sort_key,
    descending) and computes mean error on the remainder, for k in a
    grid of fractions from 0 to ~0.95."""
    n = len(errors)
    order = np.argsort(-sort_key)  # descending: most "bad" first
    sorted_errors = errors[order]

    fractions = np.linspace(0, 0.95, n_points)
    curve = []
    for frac in fractions:
        keep_from = int(frac * n)
        remaining = sorted_errors[keep_from:]
        curve.append(remaining.mean() if len(remaining) > 0 else np.nan)
    return fractions, np.array(curve)


def ause_analysis(y_true, mean_pred, mc_std, rng_seed=42):
    abs_err = np.abs(mean_pred - y_true)

    # Curve 1: remove by predicted uncertainty (mc_std), descending
    frac, curve_uncertainty = compute_sparsification_curve(abs_err, mc_std)
    # Curve 2 (oracle): remove by TRUE error, descending -- best possible
    frac_o, curve_oracle = compute_sparsification_curve(abs_err, abs_err)
    # Curve 3 (random baseline): average over several random orderings
    rng = np.random.default_rng(rng_seed)
    random_curves = []
    for _ in range(10):
        random_key = rng.random(len(abs_err))
        _, c = compute_sparsification_curve(abs_err, random_key)
        random_curves.append(c)
    curve_random = np.mean(random_curves, axis=0)

    # AUSE vs oracle: area between uncertainty-sparsification and oracle
    ause_vs_oracle = float(_trapz(curve_uncertainty - curve_oracle, frac))
    # AUSE vs random: how much better than random removal (should be negative
    # if uncertainty is informative -- i.e. removing high-uncertainty samples
    # reduces error MORE than random removal does)
    ause_vs_random_gap = float(_trapz(curve_uncertainty - curve_random, frac))

    correlation = float(np.corrcoef(mc_std, abs_err)[0, 1])

    return {
        "ause_vs_oracle": ause_vs_oracle,
        "ause_vs_random_gap": ause_vs_random_gap,
        "uncertainty_error_correlation": correlation,
        "fractions": frac.tolist(),
        "curve_uncertainty": curve_uncertainty.tolist(),
        "curve_oracle": curve_oracle.tolist(),
        "curve_random": curve_random.tolist(),
    }


def main():
    print("Loading current predictions (ablation_full)...")
    data = _load_predictions("ablation_full")
    if data is None:
        return

    y_true = data["y_test"]
    mean_pred = data["mean_pred"]
    mc_std = data["mc_std"]
    lab_test = data["lab_test"]

    print(f"\n{'='*70}")
    print("1. RELIABILITY / APPROXIMATE ECE (overall)")
    print(f"{'='*70}")
    reliability, ece_approx = reliability_analysis(y_true, mean_pred, mc_std)
    print(f"  Approximate ECE (mean |calibration error| across levels): {ece_approx:.4f}")

    print(f"\n{'='*70}")
    print("2. AUSE / SPARSIFICATION (overall)")
    print(f"{'='*70}")
    ause_overall = ause_analysis(y_true, mean_pred, mc_std)
    print(f"  AUSE vs oracle: {ause_overall['ause_vs_oracle']:.5f}")
    print(f"  AUSE vs random baseline (gap): {ause_overall['ause_vs_random_gap']:.5f}")
    print(f"  Uncertainty-error correlation: {ause_overall['uncertainty_error_correlation']:.4f}")

    print(f"\n{'='*70}")
    print("Per-dataset AUSE")
    print(f"{'='*70}")
    per_dataset = {}
    datasets_present = [d for d in DATASET_ORDER if d in set(np.unique(lab_test))]
    for ds in datasets_present:
        mask = lab_test == ds
        if mask.sum() < 20:
            print(f"  [skip] {ds}: only {mask.sum()} samples, too few for a "
                  f"stable sparsification curve")
            continue
        ds_result = ause_analysis(y_true[mask], mean_pred[mask], mc_std[mask])
        per_dataset[ds] = ds_result
        print(f"  {ds:8s}  AUSE vs oracle={ds_result['ause_vs_oracle']:.5f}  "
              f"corr={ds_result['uncertainty_error_correlation']:.4f}")

    # Save results
    out = {
        "reliability": reliability,
        "ece_approx": ece_approx,
        "ece_note": ("Approximated by Gaussian z-scaling of a single MC-Dropout "
                    "std per sample across multiple nominal levels, not "
                    "independently recalibrated at each level -- report as "
                    "an approximation, not an exact ECE."),
        "ause_overall": {k: v for k, v in ause_overall.items()
                         if k not in ("fractions", "curve_uncertainty",
                                      "curve_oracle", "curve_random")},
        "ause_per_dataset": {
            ds: {k: v for k, v in r.items()
                 if k not in ("fractions", "curve_uncertainty",
                              "curve_oracle", "curve_random")}
            for ds, r in per_dataset.items()
        },
    }
    out_path = os.path.join(config.OUTPUT_DIR, "uq_quality_analysis.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {out_path}")

    # Figure: reliability diagram + sparsification curve side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    targets = [r["target"] for r in reliability]
    actuals = [r["actual_coverage"] for r in reliability]
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    axes[0].plot(targets, actuals, "o-", color="#4C72B0", label="Observed")
    axes[0].set_xlabel("Target confidence level")
    axes[0].set_ylabel("Actual empirical coverage")
    axes[0].set_title(f"Reliability diagram (approx. ECE={ece_approx:.3f})")
    axes[0].legend()
    axes[0].set_xlim(0.4, 1.0)
    axes[0].set_ylim(0.4, 1.0)

    frac = ause_overall["fractions"]
    axes[1].plot(frac, ause_overall["curve_random"], "--", color="#999999", label="Random removal")
    axes[1].plot(frac, ause_overall["curve_uncertainty"], "-", color="#C44E52", label="Remove by uncertainty")
    axes[1].plot(frac, ause_overall["curve_oracle"], "-", color="#55A868", label="Oracle (remove by true error)")
    axes[1].set_xlabel("Fraction of samples removed")
    axes[1].set_ylabel("MAE on remaining samples")
    axes[1].set_title(f"Sparsification curve (AUSE vs oracle={ause_overall['ause_vs_oracle']:.4f})")
    axes[1].legend()

    fig.suptitle("Uncertainty Quality: Reliability and Sparsification", fontsize=13)
    fig.tight_layout()

    fig_dir = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_fig = os.path.join(fig_dir, "uq_reliability_and_sparsification.png")
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
