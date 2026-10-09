"""
Hierarchical partial-pooling calibration (v2) -- replaces
hierarchical_calibration.py / hierarchical_calibration_corrected.py.

WHAT WAS WRONG WITH THE PREVIOUS VERSION (see audit register E1, E3, E6, E13):
  * Shrinkage weight. It used w = n_g / (n_g + tau2/sigma2_g). With sigma2_g
    the sampling variance of the group quantile estimate (the quantity the
    DerSimonian-Laird step requires), the empirical-Bayes weight on the group's
    own estimate is tau2 / (tau2 + sigma2_g). The previous expression moves in
    the opposite direction: a MORE precisely estimated group (smaller sigma2_g)
    was shrunk MORE. Weights and every interpretation built on them (for
    example RWTH's 0.537) were artifacts.
  * Variance input. A test-coverage confidence-interval width was used as a
    proxy for the variance of a calibration QUANTILE. Different quantity,
    different dimension. It is not used here.
  * Mixed inputs. q_g came from the main run (deterministic-calibration) and
    sigma2_g from a regenerated MC-Dropout validation file. Here q_g and
    sigma2_g both come from the SAME validation residuals, and the test
    intervals use the SAME predictor (MC-Dropout mean).
  * Units. q_g was in per-dataset z-scored units, which are not comparable
    across datasets and would require the target's own SOH mean/std to apply
    the prior to a new source. Everything here is in physical SOH units.

METHOD.
  For each dataset g with >= 2 validation cells: q_g = finite-sample conformal
  quantile of |y - yhat| (real SOH units) on that dataset's validation set;
  sigma2_g = variance of q_g under a cell-clustered bootstrap (whole cells
  resampled with replacement). Oxford and NASA each have a single validation
  cell, so a cell-level variance is undefined; they are excluded from the
  random-effects fit (Oxford is still evaluated as a held-out target).
  mu_hat, tau2_hat: DerSimonian-Laird on {(q_g, sigma2_g)}.
  Held-out evaluation (calibration-level leave-one-dataset-out): for target t,
  refit mu_hat, tau2_hat WITHOUT t, then evaluate on t's test set:
    prior-only     : half-width mu_hat_{-t}
    EB-shrunk      : w q_t + (1-w) mu_hat_{-t},  w = tau2_{-t}/(tau2_{-t}+sigma2_t)
    own quantile   : q_t (groupwise ceiling)
    cross-source pooled : finite-sample quantile of the OTHER datasets' pooled
                          validation residuals
  tau2_hat is estimated from at most five groups; its group-level bootstrap
  interval is reported and is very wide. Treat all weights as descriptive.

Requires: cache/processed.npz, outputs/conformal_groupwise_predictions_val.npz,
outputs/conformal_groupwise_predictions.npz, config.py.
Run:  python hierarchical_calibration_v2.py
Out:  outputs/hierarchical_calibration_v2.json
"""

import json
import os

import numpy as np

import config

ALPHA = getattr(config, "CONFORMAL_ALPHA", 0.10)
N_BOOT_CELLS = 2000
N_BOOT_GROUPS = 2000
SEED = 0
ALL = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]
TARGETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford"]


def fsq(r, alpha=ALPHA):
    n = len(r)
    return float(np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))


def cell_bootstrap_variance(residuals, cell_ids, rng):
    cells = np.unique(cell_ids)
    if len(cells) < 2:
        return None
    idx = {c: np.where(cell_ids == c)[0] for c in cells}
    qs = np.empty(N_BOOT_CELLS)
    for b in range(N_BOOT_CELLS):
        pick = rng.choice(cells, size=len(cells), replace=True)
        qs[b] = fsq(residuals[np.concatenate([idx[c] for c in pick])])
    return float(np.var(qs))


def dersimonian_laird(q, s2):
    q, s2 = np.asarray(q, float), np.asarray(s2, float)
    w = 1.0 / s2
    mu_fe = np.sum(w * q) / np.sum(w)
    Q = np.sum(w * (q - mu_fe) ** 2)
    k = len(q)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    wr = 1.0 / (s2 + tau2)
    return float(np.sum(wr * q) / np.sum(wr)), float(tau2)


def group_bootstrap_tau2(q, s2, rng):
    q, s2 = np.asarray(q), np.asarray(s2)
    k = len(q)
    vals = []
    for _ in range(N_BOOT_GROUPS):
        i = rng.choice(k, size=k, replace=True)
        if len(set(i)) < 2:
            continue
        vals.append(dersimonian_laird(q[i], s2[i])[1])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    rng = np.random.default_rng(SEED)
    cache = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    val = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz"), allow_pickle=True)
    test = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"), allow_pickle=True)
    std = cache["soh_std"].item()

    rv = {}
    cells = {}
    for d in ALL:
        m = val["lab_val"] == d
        rv[d] = np.abs(val["y_val"][m] - val["mean_pred"][m]) * std[d]
        cells[d] = val["cell_val"][m]
    rt = {}
    for d in ALL:
        m = test["lab_test"] == d
        rt[d] = np.abs(test["y_test"][m] - test["mean_pred"][m]) * std[d]

    q, s2, n_cells = {}, {}, {}
    for d in ALL:
        q[d] = fsq(rv[d])
        n_cells[d] = int(len(np.unique(cells[d])))
        s2[d] = cell_bootstrap_variance(rv[d], cells[d], rng)
    fit = [d for d in ALL if s2[d] is not None]

    print("Inputs (real SOH units)")
    print(f"{'Dataset':8s}{'val cells':>10s}{'q_g':>10s}{'sigma2_g (cell bootstrap)':>28s}  in RE fit")
    for d in ALL:
        s = "undefined" if s2[d] is None else f"{s2[d]:.3e}"
        print(f"{d:8s}{n_cells[d]:10d}{q[d]:10.5f}{s:>28s}  {'yes' if d in fit else 'no'}")

    mu, tau2 = dersimonian_laird([q[d] for d in fit], [s2[d] for d in fit])
    ci = group_bootstrap_tau2([q[d] for d in fit], [s2[d] for d in fit], rng)
    fit_nc = [d for d in fit if d != "CALCE"]
    mu_nc, tau2_nc = dersimonian_laird([q[d] for d in fit_nc], [s2[d] for d in fit_nc])
    print(f"\nAll {len(fit)} groups   : mu_hat={mu:.5f}  tau2_hat={tau2:.3e}  group-bootstrap 95% CI for tau2: [{ci[0]:.2e}, {ci[1]:.2e}]")
    print(f"Excluding CALCE ({len(fit_nc)}): mu_hat={mu_nc:.5f}  tau2_hat={tau2_nc:.3e}   (sensitivity to the unresolved CALCE series issue)")

    out = {"alpha": ALPHA, "fit_groups": fit,
           "global": {"mu_hat": mu, "tau2_hat": tau2, "tau2_group_bootstrap_ci95": ci},
           "global_excluding_calce": {"mu_hat": mu_nc, "tau2_hat": tau2_nc},
           "q_g": q, "sigma2_g": s2, "per_target": {}}

    print("\nCalibration-level leave-one-dataset-out, coverage on the target's test set (90% nominal)")
    print(f"{'Target':8s}{'mu_-t':>9s}{'tau2_-t':>10s}{'w_EB':>7s}{'prior-only':>12s}{'EB-shrunk':>11s}{'own q':>8s}{'x-src pooled':>14s}{'width p/o':>11s}")
    for t in TARGETS:
        others_fit = [d for d in fit if d != t]
        mu_t, tau2_t = dersimonian_laird([q[d] for d in others_fit], [s2[d] for d in others_fit])
        cov = lambda h: float(np.mean(rt[t] <= h))
        prior = cov(mu_t)
        own = cov(q[t])
        pooled_other = fsq(np.concatenate([rv[d] for d in ALL if d != t]))
        xpool = cov(pooled_other)
        if s2[t] is not None:
            w = tau2_t / (tau2_t + s2[t]) if tau2_t > 0 else 0.0
            shrunk = cov(w * q[t] + (1 - w) * mu_t)
            ws, ss = f"{w:7.3f}", f"{100*shrunk:10.1f}%"
        else:
            w, shrunk, ws, ss = None, None, "    n/a", "       n/a"
        width_ratio = mu_t / q[t]
        out["per_target"][t] = {"n_test": int(len(rt[t])), "mu_hat_loo": mu_t, "tau2_loo": tau2_t,
                                "w_eb": w, "cov_prior_only": prior, "cov_eb_shrunk": shrunk,
                                "cov_own_quantile": own, "cov_cross_source_pooled": xpool,
                                "prior_only_width_over_own_width": width_ratio}
        print(f"{t:8s}{mu_t:9.5f}{tau2_t:10.2e}{ws}{100*prior:11.1f}%{ss}{100*own:7.1f}%{100*xpool:13.1f}%{width_ratio:11.2f}x")

    path = os.path.join(config.OUTPUT_DIR, "hierarchical_calibration_v2.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
