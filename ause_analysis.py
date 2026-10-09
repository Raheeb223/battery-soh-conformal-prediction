"""
Sparsification (AUSE) analysis — quantifies whether MC-Dropout uncertainty
is actually USEFUL for ranking samples by difficulty, addressing the
literature-review finding (Task 7) that no battery SOH paper explicitly
reports this despite it being standard practice in general ML.

Method: sort test samples by predicted uncertainty (mc_std), descending.
Progressively remove the most-uncertain fraction and recompute MAE on the
remaining samples. Compare against an ORACLE curve that instead sorts by
actual |error| (the best possible ranking). The gap between the two curves
— Area Under the Sparsification Error (AUSE) — quantifies how much
"discriminative value" the uncertainty estimate has: AUSE=0 would mean
mc_std ranks samples exactly as well as knowing the true error in advance
(impossible in practice, but the benchmark); a large AUSE means mc_std is
close to uninformative for ranking.

This directly operationalizes the weak correlation (r=0.189) found earlier
into something concrete: does the uncertainty estimate at least rank
samples in roughly the right order, even if the correlation coefficient
alone looked weak?

Run:
    python ause_analysis.py
Requires *_predictions.npz files with mc_std saved (train_one saves this
by default) — compares the 'conformal_groupwise' and
'conformal_normalized_groupwise' runs from experiments.py if both exist,
falls back to 'best' otherwise.
"""

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

FIG_DIR = os.path.join(config.OUTPUT_DIR, "figures")
FRACTIONS = np.linspace(0.0, 0.95, 20)  # fraction of samples removed (most uncertain first)


def _find_predictions(name_substring):
    candidates = glob.glob(os.path.join(config.OUTPUT_DIR, f"*{name_substring}*predictions.npz"))
    return candidates[0] if candidates else None


def sparsification_curves(y_true, mean_pred, mc_std):
    """Returns (fractions, uncertainty_curve, oracle_curve, random_curve)."""
    abs_err = np.abs(y_true - mean_pred)
    n = len(abs_err)

    order_uncertainty = np.argsort(-mc_std)      # most uncertain first
    order_oracle = np.argsort(-abs_err)          # actual worst first (best possible ranking)
    rng = np.random.RandomState(config.RANDOM_SEED)
    order_random = rng.permutation(n)

    def curve(order):
        vals = []
        for frac in FRACTIONS:
            keep_from = int(np.ceil(frac * n))
            remaining_idx = order[keep_from:]
            if len(remaining_idx) == 0:
                vals.append(0.0)
            else:
                vals.append(float(np.mean(abs_err[remaining_idx])))
        return np.array(vals)

    return FRACTIONS, curve(order_uncertainty), curve(order_oracle), curve(order_random)


def ause_score(fractions, curve_a, curve_b):
    """Area between two sparsification curves (trapezoidal). Lower = closer
    to the reference curve (typically the oracle)."""
    return float(np.trapz(np.abs(curve_a - curve_b), fractions))


def analyze(run_name_substring="conformal_normalized_groupwise", label=None):
    label = label or run_name_substring
    path = _find_predictions(run_name_substring)
    if path is None:
        path = _find_predictions("best")
        label = "best (fallback — normalized run not found)"
    if path is None:
        print(f"[ause] no predictions file found for '{run_name_substring}' or 'best' — "
              f"run experiments.py or train.py first")
        return None

    data = np.load(path, allow_pickle=True)
    y_true, mean_pred, mc_std = data["y_test"], data["mean_pred"], data["mc_std"]
    lab_test = data["lab_test"]

    print(f"[ause] analyzing: {label} ({os.path.basename(path)})")
    fractions, unc_curve, oracle_curve, rand_curve = sparsification_curves(y_true, mean_pred, mc_std)

    ause_vs_oracle = ause_score(fractions, unc_curve, oracle_curve)
    ause_vs_random = ause_score(fractions, unc_curve, rand_curve)
    print(f"  AUSE (uncertainty vs. oracle, lower=better)  = {ause_vs_oracle:.5f}")
    print(f"  Gap vs. random ranking (higher=more useful)  = {ause_vs_random:.5f}")
    if ause_vs_random < 0.001:
        print("  -> uncertainty ranking is barely better than RANDOM ranking — "
              "MC-Dropout uncertainty provides little practical ranking value here.")
    else:
        print("  -> uncertainty ranking outperforms random, i.e. mc_std has SOME "
              "genuine discriminative value even though the raw correlation was weak.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fractions, unc_curve, "o-", label="sorted by MC-Dropout uncertainty", color="#4c72b0")
    ax.plot(fractions, oracle_curve, "s--", label="oracle (sorted by true error)", color="#55a868")
    ax.plot(fractions, rand_curve, ":", label="random order", color="#999999")
    ax.set_xlabel("Fraction of samples removed (most uncertain / worst first)")
    ax.set_ylabel("Mean absolute error on remaining samples")
    ax.set_title(f"Sparsification plot ({label})\nAUSE vs oracle = {ause_vs_oracle:.4f}")
    ax.legend()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "sparsification_curve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {os.path.join(FIG_DIR, 'sparsification_curve.png')}")

    # per-dataset AUSE, since heterogeneity is the whole point of this project
    per_dataset = {}
    for ds in np.unique(lab_test):
        mask = lab_test == ds
        if mask.sum() < 10:
            continue
        f2, u2, o2, r2 = sparsification_curves(y_true[mask], mean_pred[mask], mc_std[mask])
        per_dataset[ds] = {
            "ause_vs_oracle": ause_score(f2, u2, o2),
            "ause_vs_random_gap": ause_score(f2, u2, r2),
        }
    print("\n  Per-dataset AUSE (vs. oracle, lower=better):")
    for ds, v in per_dataset.items():
        print(f"    {ds}: {v['ause_vs_oracle']:.5f}")

    results = {
        "run_analyzed": label,
        "ause_vs_oracle": ause_vs_oracle,
        "ause_vs_random_gap": ause_vs_random,
        "per_dataset": per_dataset,
    }
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "ause_analysis.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[ause] saved to {config.OUTPUT_DIR}/ause_analysis.json")
    return results


if __name__ == "__main__":
    analyze()