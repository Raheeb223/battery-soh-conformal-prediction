"""
Split conformal prediction on top of the point predictions.

Why: your earlier MC-Dropout intervals were found severely overconfident.
Conformal prediction fixes this by calibrating interval width on a held-out
calibration set so the interval has a PROVABLE (1 - alpha) coverage
guarantee, regardless of whether the model's internal uncertainty estimate
was well-calibrated to begin with.

Usage:
    conformal = SplitConformal(alpha=config.CONFORMAL_ALPHA)
    conformal.calibrate(y_true_cal, y_pred_cal)          # run once on val set
    lower, upper = conformal.predict_interval(y_pred_test)
"""

import numpy as np


class SplitConformal:
    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.qhat = None

    def calibrate(self, y_true_cal: np.ndarray, y_pred_cal: np.ndarray):
        """y_true_cal, y_pred_cal: 1D arrays from a held-out calibration set
        (use your VALIDATION set for this — never the test set)."""
        residuals = np.abs(y_true_cal - y_pred_cal)
        n = len(residuals)
        # the (1-alpha) empirical quantile, with the standard finite-sample correction
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        # NOTE (confirmed by direct test, tests/stage4_conformal_unit_tests.py):
        # np.quantile's default linear interpolation does not land exactly on
        # the textbook discrete order statistic residuals_sorted[ceil((n+1)
        # (1-alpha))-1] -- it linearly blends that order statistic with the
        # next one up, which (since residuals are sorted ascending) makes
        # this qhat PROVABLY >= the discrete order statistic, i.e. at least
        # as conservative, never less. Coverage validity is unaffected (if
        # anything strengthened slightly); this is a precision note, not a
        # correctness concern.
        self.qhat = np.quantile(residuals, q_level)
        return self.qhat

    def predict_interval(self, y_pred: np.ndarray):
        if self.qhat is None:
            raise RuntimeError("Call calibrate() before predict_interval().")
        return y_pred - self.qhat, y_pred + self.qhat

    def coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Empirical coverage on a test set — report this number in your paper
        alongside the nominal (1 - alpha) target."""
        lower, upper = self.predict_interval(y_pred)
        return float(np.mean((y_true >= lower) & (y_true <= upper)))


class NormalizedSplitConformal:
    """
    Normalized (studentized) conformal prediction: instead of one fixed
    half-width for every sample, the interval width for each sample scales
    by that sample's OWN MC-Dropout uncertainty estimate. A single scalar
    qhat is still calibrated on the validation set, but it multiplies each
    sample's mc_std rather than being added directly:

        interval_i = pred_i +/- qhat * mc_std_i

    Why this matters: plain SplitConformal gives every sample the SAME
    width, so per-sample uncertainty (MC-Dropout) has no effect on interval
    size at all — the model's uncertainty estimate is computed but never
    actually used to size the interval. This wastes exactly the information
    needed to explain (Task 6) why some samples get wider/narrower intervals,
    and it never allows you to test whether MC-Dropout uncertainty is USEFUL
    for anything (Task 7) beyond a diagnostic correlation number, since it
    doesn't affect any downstream output. This class fixes that: coverage
    validity still holds (it's a standard, textbook conformal variant —
    Papadopoulos et al. 2008), while intervals now genuinely vary per
    sample based on the model's own uncertainty, giving you something for
    feature-attribution methods (interval_width_explanation.py) to explain.

    mc_std values of 0 (or very close to 0) are clipped to a small epsilon
    to avoid division-by-zero-equivalent blowups in the calibration quantile.
    """

    def __init__(self, alpha=0.10, eps=1e-4):
        self.alpha = alpha
        self.eps = eps
        self.qhat = None

    def calibrate(self, y_true_cal: np.ndarray, y_pred_cal: np.ndarray, mc_std_cal: np.ndarray):
        mc_std_safe = np.maximum(mc_std_cal, self.eps)
        normalized_residuals = np.abs(y_true_cal - y_pred_cal) / mc_std_safe
        n = len(normalized_residuals)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.qhat = np.quantile(normalized_residuals, q_level)
        return self.qhat

    def predict_interval(self, y_pred: np.ndarray, mc_std: np.ndarray):
        if self.qhat is None:
            raise RuntimeError("Call calibrate() before predict_interval().")
        mc_std_safe = np.maximum(mc_std, self.eps)
        half_width = self.qhat * mc_std_safe
        return y_pred - half_width, y_pred + half_width

    def coverage(self, y_true, y_pred, mc_std) -> float:
        lower, upper = self.predict_interval(y_pred, mc_std)
        return float(np.mean((y_true >= lower) & (y_true <= upper)))


class NormalizedGroupwiseConformal:
    """Same idea as GroupwiseConformal (separate calibration per dataset),
    but each group's calibration is NORMALIZED conformal (per-sample width
    scaled by that sample's own MC-Dropout std) rather than a single fixed
    width. This is the version actually used for the final results — it
    combines the fix for small-group coverage failure (per-group qhat) with
    genuinely informative, per-sample-varying interval widths."""

    def __init__(self, alpha=0.10, eps=1e-4):
        self.alpha = alpha
        self.eps = eps
        self.group_conformals = {}

    def calibrate(self, y_true_cal, y_pred_cal, mc_std_cal, labels_cal):
        for group in np.unique(labels_cal):
            mask = labels_cal == group
            conformal = NormalizedSplitConformal(alpha=self.alpha, eps=self.eps)
            conformal.calibrate(y_true_cal[mask], y_pred_cal[mask], mc_std_cal[mask])
            self.group_conformals[group] = conformal
        return {g: c.qhat for g, c in self.group_conformals.items()}

    def predict_interval(self, y_pred, mc_std, labels):
        lower = np.full_like(y_pred, np.nan)
        upper = np.full_like(y_pred, np.nan)
        for group, conformal in self.group_conformals.items():
            mask = labels == group
            if not mask.any():
                continue
            lo, hi = conformal.predict_interval(y_pred[mask], mc_std[mask])
            lower[mask] = lo
            upper[mask] = hi

        missing_mask = np.isnan(lower)
        if missing_mask.any():
            missing_groups = sorted(set(labels[missing_mask]))
            print(f"[normalized conformal] WARNING: no calibration data for {missing_groups} "
                  f"— falling back to a pooled qhat across all calibrated groups.")
            pooled_qhat = np.mean([c.qhat for c in self.group_conformals.values()])
            mc_std_safe = np.maximum(mc_std[missing_mask], self.eps)
            half_width = pooled_qhat * mc_std_safe
            lower[missing_mask] = y_pred[missing_mask] - half_width
            upper[missing_mask] = y_pred[missing_mask] + half_width
        return lower, upper

    def coverage_by_group(self, y_true, y_pred, mc_std, labels):
        lower, upper = self.predict_interval(y_pred, mc_std, labels)
        results = {}
        for group in np.unique(labels):
            mask = labels == group
            results[group] = float(np.mean((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])))
        return results


class GroupwiseConformal:
    """
    Calibrates a SEPARATE interval width per dataset/group, instead of one
    pooled width across all groups (the original, non-normalized version —
    kept for the conformal ablation comparing groupwise vs pooled).

    Why this matters: split conformal's (1-alpha) coverage guarantee relies
    on calibration and test residuals being exchangeable. Pooling groups of
    very different sizes (e.g. 15,952 MIT windows vs 42 CS2 windows) means
    the pooled quantile is dominated by whichever group has the most
    samples — so small groups get an interval width calibrated on data that
    doesn't represent their own error distribution, and their coverage can
    come out far from the target even though the pooled/global number looks
    fine. Calibrating per-group fixes this at the cost of needing enough
    calibration samples in EACH group (a group with very few validation
    samples will still have a noisy qhat — that's a real limitation to
    mention in your paper if any group's cell count stays small).
    """

    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.group_conformals = {}

    def calibrate(self, y_true_cal: np.ndarray, y_pred_cal: np.ndarray, labels_cal: np.ndarray):
        for group in np.unique(labels_cal):
            mask = labels_cal == group
            conformal = SplitConformal(alpha=self.alpha)
            conformal.calibrate(y_true_cal[mask], y_pred_cal[mask])
            self.group_conformals[group] = conformal
        return {g: c.qhat for g, c in self.group_conformals.items()}

    def predict_interval(self, y_pred: np.ndarray, labels: np.ndarray):
        lower = np.full_like(y_pred, np.nan)
        upper = np.full_like(y_pred, np.nan)
        for group, conformal in self.group_conformals.items():
            mask = labels == group
            if not mask.any():
                continue
            lo, hi = conformal.predict_interval(y_pred[mask])
            lower[mask] = lo
            upper[mask] = hi

        missing_mask = np.isnan(lower)
        if missing_mask.any():
            missing_groups = sorted(set(labels[missing_mask]))
            print(f"[conformal] WARNING: no calibration data for {missing_groups} — "
                  f"this means that group had 0 validation cells (a split bug, not "
                  f"expected behavior). Falling back to a pooled qhat across all "
                  f"calibrated groups so results aren't silently wrong, but you should "
                  f"fix the split so every dataset gets validation cells.")
            pooled_qhat = np.mean([c.qhat for c in self.group_conformals.values()])
            lower[missing_mask] = y_pred[missing_mask] - pooled_qhat
            upper[missing_mask] = y_pred[missing_mask] + pooled_qhat
        return lower, upper

    def coverage_by_group(self, y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray):
        """Returns {group_name: coverage} — report per-group coverage in your
        paper, not just one pooled number, since that's the whole point of
        calibrating per group."""
        lower, upper = self.predict_interval(y_pred, labels)
        results = {}
        for group in np.unique(labels):
            mask = labels == group
            results[group] = float(np.mean((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])))
        return results