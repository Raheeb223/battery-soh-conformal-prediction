"""
Priority 5: Coverage-vs-usefulness trade-off analysis.

Coverage alone does not establish that a prediction interval is practically
useful -- a method achieving 90% coverage via very wide intervals has
limited value. This script adds four things beyond the single-alpha,
mean-width reporting already in the manuscript:

  (a) Coverage AND interval width at three nominal levels (80%, 90%, 95%),
      for pooled and groupwise calibration, computed from the SAME
      validation residuals already used throughout this project -- no new
      data needed, just recomputing quantiles at different alpha values.
  (b) Median and 90th-percentile ("upper-tail") interval width alongside
      the mean (MPIW), since a mean can be pulled by a small number of
      unusually wide intervals in a way that misrepresents a typical case.
  (c) A coverage-vs-width plot per dataset across the three alpha levels,
      directly requested by the review as the clearest way to show this
      trade-off visually.
  (d) Coverage broken down by degradation stage (early/mid/late-life, by
      true SOH value), to check whether a dataset's overall coverage
      number is masking a systematic failure at a particular part of a
      battery's life -- distinct from the existing late-stage aging error
      analysis already in the manuscript, which reports MAE and residual
      variance by SOH but not conformal coverage specifically.

USAGE: python coverage_width_tradeoff.py
"""

import numpy as np

PROCESSED_NPZ_PATH = r"C:\Users\User\PycharmProjects\cache\processed.npz"
VAL_PRED_PATH = r"C:\Users\User\PycharmProjects\outputs\conformal_groupwise_predictions_val.npz"
TEST_PRED_PATH = r"C:\Users\User\PycharmProjects\outputs\conformal_groupwise_predictions.npz"

ALPHA_LEVELS = [0.20, 0.10, 0.05]  # 80%, 90%, 95% nominal coverage
DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]
SOH_STAGE_BINS = [(1.10, 0.85, "early-life"), (0.85, 0.70, "mid-life"), (0.70, 0.0, "late-life")]


def load_and_denormalize(dataset: str):
    """Returns de-normalized y_val, pred_val, mc_std_val (for calibration)
    and y_test, pred_test, mc_std_test (for evaluation), all in real SOH
    units (mc_std is a scale/std quantity and is scaled by soh_std but not
    shifted by soh_mean)."""
    cache = np.load(PROCESSED_NPZ_PATH, allow_pickle=True)
    val = np.load(VAL_PRED_PATH, allow_pickle=True)
    test = np.load(TEST_PRED_PATH, allow_pickle=True)
    soh_mean = cache["soh_mean"].item()
    soh_std = cache["soh_std"].item()
    m, s = soh_mean[dataset], soh_std[dataset]

    val_mask = val["lab_val"] == dataset
    test_mask = test["lab_test"] == dataset

    y_val = val["y_val"][val_mask] * s + m
    pred_val = val["mean_pred"][val_mask] * s + m
    mc_std_val = val["mc_std"][val_mask] * s
    y_test = test["y_test"][test_mask] * s + m
    pred_test = test["mean_pred"][test_mask] * s + m
    mc_std_test = test["mc_std"][test_mask] * s

    return y_val, pred_val, mc_std_val, y_test, pred_test, mc_std_test


def pooled_quantile_at_alpha(residuals: np.ndarray, alpha: float) -> float:
    n = len(residuals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(residuals, level)


def evaluate_at_alpha(y_val, pred_val, y_test, pred_test, alpha: float) -> dict:
    """Groupwise-style (single fixed quantile per dataset) interval at one
    alpha level. NOTE: this construction produces an IDENTICAL interval
    width for every test point in a dataset by design -- mean, median, and
    p90 width are therefore always equal here. This is not a bug; it is
    the defining property of standard (non-normalized) split conformal
    prediction. See evaluate_normalized_at_alpha() below for the
    genuinely per-sample-varying alternative already used elsewhere in
    this project (the "normalized groupwise" calibration mode)."""
    residuals_val = np.abs(y_val - pred_val)
    q = pooled_quantile_at_alpha(residuals_val, alpha)
    lower = pred_test - q
    upper = pred_test + q
    covered = (y_test >= lower) & (y_test <= upper)
    widths = upper - lower
    return {
        "coverage": covered.mean(),
        "mean_width": widths.mean(),
        "median_width": np.median(widths),
        "p90_width": np.percentile(widths, 90),
    }


def evaluate_normalized_at_alpha(y_val, pred_val, mc_std_val, y_test, pred_test, mc_std_test,
                                  alpha: float, eps: float = 1e-6) -> dict:
    """Normalized-groupwise calibration: residuals are divided by each
    validation sample's own MC-Dropout std before taking the quantile, and
    that quantile is then multiplied back by each TEST sample's own
    MC-Dropout std -- giving a genuinely per-sample-varying interval width,
    unlike the fixed-width construction above. This is the same
    conformal_mode="normalized_groupwise" already implemented in this
    project's training pipeline (train.py), recomputed here directly from
    already-available validation and test predictions rather than requiring
    a new training run."""
    normalized_residuals = np.abs(y_val - pred_val) / (mc_std_val + eps)
    q = pooled_quantile_at_alpha(normalized_residuals, alpha)
    half_widths = q * (mc_std_test + eps)
    lower = pred_test - half_widths
    upper = pred_test + half_widths
    covered = (y_test >= lower) & (y_test <= upper)
    widths = upper - lower
    return {
        "coverage": covered.mean(),
        "mean_width": widths.mean(),
        "median_width": np.median(widths),
        "p90_width": np.percentile(widths, 90),
    }


def degradation_stage_coverage(y_val, pred_val, y_test, pred_test, alpha: float = 0.10) -> dict:
    """Coverage within each SOH-range bin, using the SAME single quantile
    calibrated across the whole dataset (not re-calibrated per stage) --
    this checks whether one calibrated interval width is adequate across
    a battery's whole life, or systematically fails at one stage."""
    residuals_val = np.abs(y_val - pred_val)
    q = pooled_quantile_at_alpha(residuals_val, alpha)
    lower = pred_test - q
    upper = pred_test + q
    covered = (y_test >= lower) & (y_test <= upper)

    results = {}
    for hi, lo, label in SOH_STAGE_BINS:
        mask = (y_test <= hi) & (y_test > lo)
        if mask.sum() > 0:
            results[label] = {"n": mask.sum(), "coverage": covered[mask].mean()}
        else:
            results[label] = {"n": 0, "coverage": None}
    return results


def main():
    print("=" * 100)
    print("(a): Fixed-width (groupwise) coverage and width at three nominal levels")
    print("Mean=median=p90 width by construction -- a single quantile applies uniformly")
    print("=" * 100)
    print(f"{'Dataset':<8}{'Target':<9}{'Coverage':<11}{'Width'}")
    for dataset in DATASETS:
        y_val, pred_val, mc_std_val, y_test, pred_test, mc_std_test = load_and_denormalize(dataset)
        for alpha in ALPHA_LEVELS:
            r = evaluate_at_alpha(y_val, pred_val, y_test, pred_test, alpha)
            print(f"{dataset:<8}{1-alpha:<9.0%}{r['coverage']:<11.3f}{r['mean_width']:.4f}")
        print()

    print("=" * 100)
    print("(b): Normalized-groupwise coverage and width distribution (mean/median/p90)")
    print("Width genuinely varies per sample here -- median/p90 are informative")
    print("=" * 100)
    print(f"{'Dataset':<8}{'Target':<9}{'Coverage':<11}{'MeanW':<9}{'MedianW':<10}{'P90 W'}")
    for dataset in DATASETS:
        y_val, pred_val, mc_std_val, y_test, pred_test, mc_std_test = load_and_denormalize(dataset)
        for alpha in ALPHA_LEVELS:
            r = evaluate_normalized_at_alpha(y_val, pred_val, mc_std_val, y_test, pred_test, mc_std_test, alpha)
            print(f"{dataset:<8}{1-alpha:<9.0%}{r['coverage']:<11.3f}"
                  f"{r['mean_width']:<9.4f}{r['median_width']:<10.4f}{r['p90_width']:.4f}")
        print()

    print("=" * 100)
    print("(d): Coverage by degradation stage, at the 90% nominal level (groupwise, fixed width)")
    print("=" * 100)
    print(f"{'Dataset':<8}{'Early-life':<20}{'Mid-life':<20}{'Late-life'}")
    for dataset in DATASETS:
        y_val, pred_val, mc_std_val, y_test, pred_test, mc_std_test = load_and_denormalize(dataset)
        stages = degradation_stage_coverage(y_val, pred_val, y_test, pred_test)
        row = f"{dataset:<8}"
        for label in ["early-life", "mid-life", "late-life"]:
            s = stages[label]
            if s["n"] > 0:
                row += f"{s['coverage']:.3f} (n={s['n']})".ljust(20)
            else:
                row += "no data".ljust(20)
        print(row)

    print("\n(c): Coverage-vs-width plot data is exactly table (a)/(b)'s three rows per dataset --")
    print("plot mean_width (or median_width) on x, coverage on y, one line per dataset,")
    print("connecting the three alpha levels. See plot_coverage_width_tradeoff.py.")


if __name__ == "__main__":
    main()
