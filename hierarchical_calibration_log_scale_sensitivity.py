"""
Log-scale sensitivity check for the hierarchical calibration model
(hierarchical_calibration_v2.py).

WHY. That model treats q_g ~ Normal(theta_g, sigma_g^2) with theta_g ~
Normal(mu, tau^2) on the ADDITIVE scale. Conformal quantiles across datasets
span nearly two orders of magnitude (0.0019 to 0.10 in physical SOH units,
Section 5.3.7), so a common additive tau^2 may not be appropriate: the
between-group spread of a quantity that varies multiplicatively is often
better modeled on the log scale. This script refits the identical
DerSimonian-Laird random-effects model on log(q_g) instead, with the
sampling variance of log(q_g) obtained by the delta method
(Var(log X) ~ Var(X)/X^2), exponentiates the shrunk estimate back, and
reports the SAME calibration-level LODO coverage comparison as the additive
model, so the two can be compared on equal footing.

This is a robustness check on the STATISTICAL MODEL choice. It does not
depend on the CALCE loader fix (Section on CALCE series structure) and uses
the same as-run inputs as hierarchical_calibration_v2.py.

Requires: cache/processed.npz, outputs/conformal_groupwise_predictions_val.npz,
outputs/conformal_groupwise_predictions.npz, config.py.
Run:  python hierarchical_calibration_log_scale_sensitivity.py
Out:  outputs/hierarchical_calibration_log_scale.json
"""

import json
import os

import numpy as np

import config
from hierarchical_calibration_v2 import (ALL, TARGETS, cell_bootstrap_variance,
                                          dersimonian_laird, fsq)

ALPHA = getattr(config, "CONFORMAL_ALPHA", 0.10)
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    cache = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    val = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz"), allow_pickle=True)
    test = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"), allow_pickle=True)
    std = cache["soh_std"].item()

    rv, cells, rt = {}, {}, {}
    for d in ALL:
        mv = val["lab_val"] == d
        rv[d] = np.abs(val["y_val"][mv] - val["mean_pred"][mv]) * std[d]
        cells[d] = val["cell_val"][mv]
        mt = test["lab_test"] == d
        rt[d] = np.abs(test["y_test"][mt] - test["mean_pred"][mt]) * std[d]

    q, s2, log_q, log_s2 = {}, {}, {}, {}
    for d in ALL:
        q[d] = fsq(rv[d])
        s2[d] = cell_bootstrap_variance(rv[d], cells[d], rng)
        log_q[d] = float(np.log(q[d]))
        log_s2[d] = (s2[d] / q[d] ** 2) if s2[d] is not None else None   # delta method

    fit = [d for d in ALL if s2[d] is not None]
    mu_log, tau2_log = dersimonian_laird([log_q[d] for d in fit], [log_s2[d] for d in fit])
    mu_add, tau2_add = dersimonian_laird([q[d] for d in fit], [s2[d] for d in fit])
    print(f"Additive scale:  mu_hat={mu_add:.5f}  tau2_hat={tau2_add:.3e}")
    print(f"Log scale:       mu_hat(log)={mu_log:.4f} -> exp={np.exp(mu_log):.5f}  tau2_hat(log)={tau2_log:.4f}\n")

    out = {"alpha": ALPHA, "additive": {"mu_hat": mu_add, "tau2_hat": tau2_add},
           "log_scale": {"mu_hat_log": mu_log, "mu_hat_exp": float(np.exp(mu_log)), "tau2_hat_log": tau2_log},
           "per_target": {}}

    print(f"{'Target':8s}{'EB-shrunk (additive)':>22s}{'EB-shrunk (log-scale)':>23s}{'own quantile':>14s}")
    for t in TARGETS:
        others = [d for d in fit if d != t]
        mu_a, tau2_a = dersimonian_laird([q[d] for d in others], [s2[d] for d in others])
        mu_l, tau2_l = dersimonian_laird([log_q[d] for d in others], [log_s2[d] for d in others])
        cov = lambda h: float(np.mean(rt[t] <= h))
        own = cov(q[t])

        if s2[t] is not None:
            w_a = tau2_a / (tau2_a + s2[t]) if tau2_a > 0 else 0.0
            shrunk_add = cov(w_a * q[t] + (1 - w_a) * mu_a)
            w_l = tau2_l / (tau2_l + log_s2[t]) if tau2_l > 0 else 0.0
            shrunk_log_h = np.exp(w_l * log_q[t] + (1 - w_l) * mu_l)
            shrunk_log = cov(shrunk_log_h)
        else:
            shrunk_add = shrunk_log = None

        out["per_target"][t] = {"cov_eb_shrunk_additive": shrunk_add, "cov_eb_shrunk_log_scale": shrunk_log,
                                "cov_own_quantile": own}
        sa = f"{100*shrunk_add:20.1f}%" if shrunk_add is not None else "                 n/a"
        sl = f"{100*shrunk_log:21.1f}%" if shrunk_log is not None else "                  n/a"
        print(f"{t:8s}{sa}{sl}{100*own:13.1f}%")

    path = os.path.join(config.OUTPUT_DIR, "hierarchical_calibration_log_scale.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
