"""
Stronger borrowing baselines: is "borrowing does not help" a property of the data or of our
shrinkage estimator?

hierarchical_calibration_v2.py shrinks each dataset's conformal quantile toward the others with a
DerSimonian-Laird (DL) random-effects fit on a linear scale, with the between-dataset variance
estimated WITHOUT the target. Three objections apply to that design: DL is unreliable with four
or five groups; BIT's quantile is several times the others', so a linear scale is a poor fit for
a scale parameter; and leaving the target out of the variance estimate hides exactly the
heterogeneity the target brings. This script fits stronger versions, all in log units of the
quantile, on every split, with the design fixed before the first run:

  With the target's own validation residuals (the target is part of the fit):
    eb_linear_joint   DL on the linear scale, target included in the fit.
    eb_log_joint      DL on log q_g, target included.
    bayes_log         Full Bayesian normal-normal model on log q_g: flat prior on the mean,
                      half-Cauchy(0, A) prior on tau with A = 1 (a factor of e between datasets),
                      tau integrated out on a grid. Interval half-width = exp(posterior mean of
                      the target's log quantile). A in {0.5, 2} is reported as a sensitivity check.
    bayes_log_noise   bayes_log with a meta-regression on a label-free covariate: the log of the
                      median absolute cycle-to-cycle SOH change inside the dataset's validation
                      input windows (how noisy its SOH histories are).
    bayes_log_spread  the same with the log of the mean MC-Dropout spread on the validation
                      windows (how uncertain the model itself is on the dataset).
  Without the target's own residuals (the target is left out of the fit; its covariate, which
  needs no calibration labels, is still used):
    prior_log, prior_log_noise, prior_log_spread
                      posterior mean of a NEW dataset's log quantile under the three models.

Reference rows (recomputed here with the same residuals): own (groupwise), hier_linear_loo and
prior_linear_loo (the published hierarchical method and its prior).

Inputs per split are the same files as hierarchical_calibration_v2.py and split_replication.py.
Run:  python borrowing_variants.py
Out:  outputs/borrowing_variants.json
"""
import json
import os

import numpy as np

import config
from hierarchical_calibration_v2 import dersimonian_laird, fsq

ALPHA = config.CONFORMAL_ALPHA
FIVE = ["MIT", "RWTH", "XJTU", "BIT", "CALCE"]
ALL = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]
SPLITS = [42, 43, 44, 45, 46]
N_BOOT = 2000
TAU_GRID = np.linspace(1e-4, 6.0, 3000)
PRIOR_SCALE = 1.0
SENSITIVITY_SCALES = [0.5, 2.0]
WITH_OWN = ["own", "hier_linear_loo", "eb_linear_joint", "eb_log_joint", "bayes_log",
            "bayes_log_noise", "bayes_log_spread"]
WITHOUT_OWN = ["prior_linear_loo", "prior_log", "prior_log_noise", "prior_log_spread"]


def paths(split):
    if split == 42:
        return (os.path.join(config.CACHE_DIR, "processed.npz"),
                os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz"),
                os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"))
    return (os.path.join(config.CACHE_DIR, f"processed_split{split}.npz"),
            os.path.join(config.OUTPUT_DIR, f"split{split}_groupwise_predictions_val.npz"),
            os.path.join(config.OUTPUT_DIR, f"split{split}_groupwise_predictions.npz"))


def bootstrap_variances(r, cells, rng):
    """Cell-clustered bootstrap variance of q and of log q (None with fewer than two cells)."""
    uniq = np.unique(cells)
    if len(uniq) < 2:
        return None, None
    idx = {c: np.where(cells == c)[0] for c in uniq}
    qs = np.array([fsq(r[np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])])
                   for _ in range(N_BOOT)])
    return float(np.var(qs)), float(np.var(np.log(qs)))


def tau_posterior(y, s2, X, scale):
    """Posterior weights of tau on TAU_GRID for y_g = X_g b + u_g + e_g, u_g ~ N(0, tau^2),
    e_g ~ N(0, s2_g), flat prior on b, half-Cauchy(0, scale) prior on tau. Returns the weights
    and, per grid point, the GLS estimate of b."""
    logp, betas = np.empty(len(TAU_GRID)), []
    for i, tau in enumerate(TAU_GRID):
        vinv = 1.0 / (s2 + tau ** 2)
        XtV = X.T * vinv
        A = XtV @ X
        b = np.linalg.solve(A, XtV @ y)
        resid = y - X @ b
        logp[i] = (np.log(1.0 / (1.0 + (tau / scale) ** 2)) - 0.5 * np.linalg.slogdet(A)[1]
                   + 0.5 * np.sum(np.log(vinv)) - 0.5 * np.sum(vinv * resid ** 2))
        betas.append(b)
    w = np.exp(logp - logp.max())
    return w / w.sum(), np.array(betas)


def bayes_own(y, s2, X, t, scale):
    """Posterior mean of the target's log quantile (target t is in the fit) and the posterior
    mean weight on its own estimate."""
    w, betas = tau_posterior(y, s2, X, scale)
    tau2 = TAU_GRID ** 2
    shrink = tau2 / (tau2 + s2[t])
    theta = shrink * y[t] + (1 - shrink) * (betas @ X[t])
    return float(w @ theta), float(w @ shrink)


def bayes_new(y, s2, X, x_new, scale):
    """Posterior mean of a new dataset's log quantile (not in the fit), given its covariates."""
    w, betas = tau_posterior(y, s2, X, scale)
    return float(w @ (betas @ x_new))


def analyse(split):
    # A fresh generator per split, seeded and consumed in the same order as
    # hierarchical_calibration_v2.analyse, so the linear rows reproduce its published values exactly.
    rng = np.random.default_rng(0)
    cache_path, val_path, test_path = paths(split)
    cache = np.load(cache_path, allow_pickle=True)
    val = np.load(val_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    std = cache["soh_std"].item()
    assert np.array_equal(cache["y_val"], val["y_val"])

    rv, rt, q, s2, s2log, noise, spread = {}, {}, {}, {}, {}, {}, {}
    for d in ALL:
        mv, mt = val["lab_val"] == d, test["lab_test"] == d
        if not mv.any():
            continue
        rv[d] = np.abs(val["y_val"][mv] - val["mean_pred"][mv]) * std[d]
        rt[d] = np.abs(test["y_test"][mt] - test["mean_pred"][mt]) * std[d]
        q[d] = fsq(rv[d])
        s2[d], s2log[d] = bootstrap_variances(rv[d], val["cell_val"][mv], rng)
        x_in = cache["X_val"][mv, :, 0] * std[d]
        noise[d] = float(np.median(np.abs(np.diff(x_in, axis=1))))
        spread[d] = float(np.mean(val["mc_std"][mv]) * std[d])
    fit = [d for d in q if s2[d] is not None]
    targets = [t for t in FIVE if t in fit and len(rt[t])]

    def design(ds, cov):
        if cov is None:
            return np.ones((len(ds), 1))
        return np.column_stack([np.ones(len(ds)), np.log([cov[d] for d in ds])])

    per = {}
    for t in targets:
        cov = lambda h: float(np.mean(rt[t] <= h))
        others = [d for d in fit if d != t]
        ylog = np.log([q[d] for d in fit])
        vlog = np.array([s2log[d] for d in fit])
        ti = fit.index(t)
        half = {"own": q[t]}

        mu_o, tau2_o = dersimonian_laird([q[d] for d in others], [s2[d] for d in others])
        w_loo = tau2_o / (tau2_o + s2[t]) if tau2_o > 0 else 0.0
        half["hier_linear_loo"] = w_loo * q[t] + (1 - w_loo) * mu_o
        half["prior_linear_loo"] = mu_o

        mu_j, tau2_j = dersimonian_laird([q[d] for d in fit], [s2[d] for d in fit])
        w_lin = tau2_j / (tau2_j + s2[t]) if tau2_j > 0 else 0.0
        half["eb_linear_joint"] = w_lin * q[t] + (1 - w_lin) * mu_j
        mu_l, tau2_l = dersimonian_laird(ylog, vlog)
        w_log = tau2_l / (tau2_l + s2log[t]) if tau2_l > 0 else 0.0
        half["eb_log_joint"] = float(np.exp(w_log * ylog[ti] + (1 - w_log) * mu_l))

        weights = {"hier_linear_loo": w_loo, "eb_linear_joint": w_lin, "eb_log_joint": w_log}
        for name, covar in (("bayes_log", None), ("bayes_log_noise", noise), ("bayes_log_spread", spread)):
            m, wb = bayes_own(ylog, vlog, design(fit, covar), ti, PRIOR_SCALE)
            half[name], weights[name] = float(np.exp(m)), wb
        yo = np.log([q[d] for d in others])
        vo = np.array([s2log[d] for d in others])
        for name, covar in (("prior_log", None), ("prior_log_noise", noise), ("prior_log_spread", spread)):
            x_new = design([t], covar)[0]
            half[name] = float(np.exp(bayes_new(yo, vo, design(others, covar), x_new, PRIOR_SCALE)))
        sens = {}
        for a in SENSITIVITY_SCALES:
            m, wb = bayes_own(ylog, vlog, design(fit, None), ti, a)
            sens[str(a)] = {"coverage": cov(float(np.exp(m))), "weight_own": wb,
                            "prior_coverage": cov(float(np.exp(bayes_new(yo, vo, design(others, None),
                                                                          design([t], None)[0], a))))}
        per[t] = {"coverage": {k: cov(v) for k, v in half.items()},
                  "half_width_over_own": {k: v / q[t] for k, v in half.items()},
                  "weight_own": weights, "prior_scale_sensitivity": sens}

    dev = {m: float(np.mean([abs(per[t]["coverage"][m] - 0.9) for t in targets])) for m in WITH_OWN + WITHOUT_OWN}
    dev_sens = {a: {"bayes_log": float(np.mean([abs(per[t]["prior_scale_sensitivity"][a]["coverage"] - 0.9)
                                                 for t in targets])),
                    "prior_log": float(np.mean([abs(per[t]["prior_scale_sensitivity"][a]["prior_coverage"] - 0.9)
                                                 for t in targets]))}
                for a in map(str, SENSITIVITY_SCALES)}
    return {"fit_groups": fit, "targets": targets, "q": q, "covariate_noise": noise, "covariate_spread": spread,
            "per_target": per, "mean_abs_dev": dev, "mean_abs_dev_prior_scale": dev_sens}


def main():
    results = {"prior_scale": PRIOR_SCALE, "splits": {}}
    for split in SPLITS:
        r = analyse(split)
        results["splits"][str(split)] = r
        print(f"\n=== split {split} (fit groups: {', '.join(r['fit_groups'])})")
        print(f"{'method':18s}" + "".join(f"{t:>8s}" for t in r["targets"]) + f"{'dev':>8s}")
        for m in WITH_OWN + WITHOUT_OWN:
            print(f"{m:18s}" + "".join(f"{100 * r['per_target'][t]['coverage'][m]:8.1f}" for t in r["targets"])
                  + f"{100 * r['mean_abs_dev'][m]:8.2f}")
        if "BIT" in r["per_target"]:
            print("BIT weight on own quantile: " + ", ".join(
                f"{k} {v:.3f}" for k, v in r["per_target"]["BIT"]["weight_own"].items()))

    splits = list(results["splits"])
    methods = WITH_OWN + WITHOUT_OWN
    devs = {m: [results["splits"][s]["mean_abs_dev"][m] for s in splits] for m in methods}
    own = devs["own"]
    summary = {m: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                   "beats_own_splits": int(sum(x < o for x, o in zip(v, own))),
                   "ties_or_beats_own_splits": int(sum(x <= o + 1e-12 for x, o in zip(v, own)))}
               for m, v in devs.items()}
    bit = {m: [results["splits"][s]["per_target"]["BIT"]["coverage"][m] for s in splits
               if "BIT" in results["splits"][s]["per_target"]] for m in methods}
    summary_bit = {m: {"min": float(np.min(v)), "max": float(np.max(v))} for m, v in bit.items()}
    bit_w = {m: [results["splits"][s]["per_target"]["BIT"]["weight_own"][m] for s in splits]
             for m in ["hier_linear_loo", "eb_linear_joint", "eb_log_joint", "bayes_log",
                       "bayes_log_noise", "bayes_log_spread"]}
    results["summary"] = {"mean_abs_dev": summary, "bit_coverage": summary_bit,
                          "bit_weight_own": {m: {"min": float(np.min(v)), "max": float(np.max(v))}
                                             for m, v in bit_w.items()},
                          "prior_scale_sensitivity": {
                              a: {k: float(np.mean([results["splits"][s]["mean_abs_dev_prior_scale"][a][k]
                                                    for s in splits])) for k in ("bayes_log", "prior_log")}
                              for a in map(str, SENSITIVITY_SCALES)}}
    with open(os.path.join(config.OUTPUT_DIR, "borrowing_variants.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'method':18s}{'mean':>8s}{'min':>8s}{'max':>8s}{'beats own':>11s}{'BIT min':>9s}{'BIT max':>9s}")
    for m in methods:
        s, b = summary[m], summary_bit[m]
        print(f"{m:18s}{100 * s['mean']:8.2f}{100 * s['min']:8.2f}{100 * s['max']:8.2f}"
              f"{s['beats_own_splits']:11d}{100 * b['min']:9.1f}{100 * b['max']:9.1f}")
    print("saved outputs/borrowing_variants.json")


if __name__ == "__main__":
    main()
