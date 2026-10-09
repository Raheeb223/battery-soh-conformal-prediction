"""
Statistical significance testing: is your model's improvement over the
baselines/ablations real, or could it be explained by run-to-run noise?

This is the legitimate version of "more proof" — it doesn't add a new
number, it tells you whether the numbers you ALREADY HAVE are reliable
enough to make a comparative claim from.

Method: paired Wilcoxon signed-rank test on per-sample absolute errors
between two models on the SAME test set. Paired because both models were
evaluated on identical test windows — this is more powerful than an
unpaired test and is the standard choice for this kind of comparison.

Requires: the *_predictions.npz files saved by train_one() for each variant
you want to compare (these already exist for every run in your ablation and
baseline sections, IF save_outputs was True — which it is by default).

Run:
    python significance.py
"""

import os
import glob
import numpy as np
from scipy.stats import wilcoxon

import config


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size — chosen over Cohen's d because prediction
    errors are typically not normally distributed (often right-skewed with
    a long tail of hard samples), and Cliff's delta makes no distributional
    assumption. Ranges -1 to 1; |delta| < 0.147 negligible, < 0.33 small,
    < 0.474 medium, else large (Romano et al. 2006 thresholds)."""
    # efficient O(n log n) computation via ranking rather than the naive O(n*m)
    n, m = len(a), len(b)
    all_vals = np.concatenate([a, b])
    ranks = np.argsort(np.argsort(all_vals))
    rank_a = ranks[:n]
    # sum of ranks -> Mann-Whitney U -> Cliff's delta
    r_a_sum = np.sum(rank_a) + n  # +n to convert 0-indexed ranks to 1-indexed
    u_a = r_a_sum - n * (n + 1) / 2
    delta = (2 * u_a / (n * m)) - 1
    return float(delta)


def _delta_magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def bootstrap_mae_diff_ci(err_a: np.ndarray, err_b: np.ndarray, n_boot=2000, seed=None):
    """Bootstrap 95% CI for the difference in mean absolute error (err_a - err_b),
    resampling PAIRS (same indices for both) since the two error arrays come
    from the same test samples."""
    rng = np.random.RandomState(seed or config.RANDOM_SEED)
    n = len(err_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[i] = np.mean(err_a[idx]) - np.mean(err_b[idx])
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return float(lower), float(upper)


def _load_errors(run_name_substring):
    """Loads |error| per test sample for a run whose predictions file
    contains run_name_substring (e.g. 'ablation_full', 'ablation_gru')."""
    candidates = glob.glob(os.path.join(config.OUTPUT_DIR, f"*{run_name_substring}*predictions.npz"))
    if not candidates:
        return None, None
    data = np.load(candidates[0], allow_pickle=True)
    errors = np.abs(data["mean_pred"] - data["y_test"])
    return errors, data["lab_test"]


def compare(name_a: str, name_b: str, label_a=None, label_b=None):
    """Paired Wilcoxon test comparing two runs' per-sample absolute errors.
    Both runs MUST have been evaluated on the exact same test set (true for
    any two runs from the same cache_path — e.g. all the 'ablation_*' runs
    share the same test set, so they're directly comparable this way)."""
    label_a = label_a or name_a
    label_b = label_b or name_b
    err_a, lab_a = _load_errors(name_a)
    err_b, lab_b = _load_errors(name_b)

    if err_a is None or err_b is None:
        print(f"[significance] SKIPPED {label_a} vs {label_b} — predictions file(s) not found. "
              f"(Need *{name_a}*predictions.npz and *{name_b}*predictions.npz in outputs/)")
        return None
    if len(err_a) != len(err_b):
        print(f"[significance] SKIPPED {label_a} vs {label_b} — different test set sizes "
              f"({len(err_a)} vs {len(err_b)}), likely from different cache/window settings. "
              f"A paired test requires the identical test samples for both.")
        return None

    stat, p_value = wilcoxon(err_a, err_b)
    mean_a, mean_b = float(np.mean(err_a)), float(np.mean(err_b))
    better = label_a if mean_a < mean_b else label_b
    significant = p_value < 0.05
    delta = cliffs_delta(err_a, err_b)
    ci_lower, ci_upper = bootstrap_mae_diff_ci(err_a, err_b)

    print(f"[significance] {label_a} (MAE={mean_a:.5f}) vs {label_b} (MAE={mean_b:.5f})")
    print(f"  Wilcoxon p-value = {p_value:.6f}  "
          f"({'SIGNIFICANT at p<0.05' if significant else 'NOT significant at p<0.05'})")
    print(f"  Cliff's delta = {delta:.3f} ({_delta_magnitude(delta)} effect size)")
    print(f"  95% bootstrap CI for MAE difference ({label_a} - {label_b}): "
          f"[{ci_lower:.5f}, {ci_upper:.5f}]"
          f"{' (excludes 0 -> consistent with significance)' if (ci_lower > 0 or ci_upper < 0) else ' (includes 0 -> consistent with non-significance)'}")
    if significant:
        print(f"  -> {better} is significantly better on this test set.")
    else:
        print(f"  -> Difference could plausibly be noise — do not claim one model is "
              f"reliably better based on this comparison alone.")

    return {
        "comparison": f"{label_a}_vs_{label_b}",
        "mae_a": mean_a, "mae_b": mean_b,
        "p_value": float(p_value),
        "significant": bool(significant),
        "cliffs_delta": delta,
        "effect_size_magnitude": _delta_magnitude(delta),
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "better": better if significant else None,
    }


def run_all_comparisons():
    """The comparisons that actually matter for your paper's claims:
    proposed model vs each ablation variant, vs GRU, and vs the sklearn
    baselines (linear/SVR/XGBoost) — this last group is what actually
    tells you whether 'SVR beats us on MAE' is a real, reliable difference
    or within noise."""
    print("=" * 60)
    print("Statistical significance: proposed model vs. ablation variants")
    print("=" * 60)
    results = []
    for variant in ["no_attention", "no_multiscale", "plain_lstm", "gru"]:
        r = compare("ablation_full", f"ablation_{variant}",
                    label_a="full (proposed)", label_b=variant)
        if r:
            results.append(r)

    print("\n" + "=" * 60)
    print("Statistical significance: proposed model vs. classical baselines")
    print("=" * 60)
    for baseline in ["baseline_linear", "baseline_svr", "baseline_xgboost"]:
        r = compare("ablation_full", baseline,
                    label_a="full (proposed)", label_b=baseline.replace("baseline_", ""))
        if r:
            results.append(r)
    return results


if __name__ == "__main__":
    run_all_comparisons()