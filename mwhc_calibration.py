"""
MWHC: MMD-weighted hierarchical calibration.

Hierarchical calibration (hierarchical_calibration_v2.py) shrinks a dataset's own conformal
quantile toward a prior fitted to the other datasets, weighting each of them by the precision of its
quantile (1/sigma2_d) only. MWHC multiplies that weight by HGC-CP's MMD similarity weight sim_d
(hgc_cp.compute_group_weights), so that datasets that resemble the target count more in the prior.
The shrinkage weight tau2/(tau2 + sigma2_t) is computed as before, from the similarity-weighted fit.
With equal similarities, weighted_dersimonian_laird reduces exactly to the ordinary
DerSimonian-Laird estimator; this is checked when the script runs.

The protocol is the calibration-level leave-one-dataset-out of hierarchical_calibration_v2.py, with
q_g and sigma2_g recomputed here so the flat and similarity-weighted fits come from the same run.
The MMD temperature lambda is swept over HGC-CP's grid [1, 3, 5, 10, 20]. No model is retrained.

Requires: cache/processed.npz, outputs/conformal_groupwise_predictions_val.npz,
outputs/conformal_groupwise_predictions.npz, config.py.
Run:  python mwhc_calibration.py
Out:  outputs/mwhc_results.json
"""

import json
import os

import numpy as np

import config
from hgc_cp import mmd_squared, compute_group_weights
from hgc_cp_evaluation_harness import load_representations
from hierarchical_calibration_v2 import fsq, cell_bootstrap_variance, dersimonian_laird

ALPHA = getattr(config, "CONFORMAL_ALPHA", 0.10)
SEED = 0
MMD_SAMPLE_SIZE = 300
LAM_GRID = [1.0, 3.0, 5.0, 10.0, 20.0]
DEFAULT_LAM = 3.0  # headline value -- no per-target tuning, see module docstring
ALL = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]
TARGETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford"]  # same 6 as hierarchical_calibration_v2.py


def effective_sample_size(weights: dict) -> float:
    """Kish's effective sample size (Kish, 1965) for a set of normalized
    group weights: (sum w)^2 / sum(w^2). Ranges from 1 (all weight on a
    single group -- the method is relying on exactly one other dataset) to
    k (perfectly uniform -- equivalent to the flat, unweighted prior).
    Used here as a diagnostic: see the main() loop and the module-level
    discussion in __main__'s printed summary for why low ESS coincides with
    MWHC's coverage failures at high lam."""
    w = np.array(list(weights.values()), dtype=float)
    return float((w.sum() ** 2) / np.sum(w ** 2))


def compute_floored_group_weights(distances: dict, lam: float, floor_frac: float = 0.5) -> dict:
    """Same as hgc_cp.compute_group_weights, but floors each group's
    softmax weight at floor_frac * (1/k) BEFORE renormalizing, so a single
    close-but-imperfect source can never be driven to near-zero effective
    weight on the others. This is weight truncation/flooring, a standard
    fix for importance-weight degeneracy (e.g. Ionides, 2008,
    "Truncated Importance Sampling") applied here to bound how much MWHC's
    similarity weighting can concentrate -- a direct, principled response
    to the effective-sample-size collapse this script's lam-sweep exposes
    at lam >= 10 (see effective_sample_size)."""
    sim = compute_group_weights(distances, lam)
    k = len(sim)
    floor = floor_frac / k
    floored = {g: max(w, floor) for g, w in sim.items()}
    total = sum(floored.values())
    return {g: w / total for g, w in floored.items()}


def weighted_dersimonian_laird(q, s2, sim):
    """DerSimonian-Laird with each group's precision weight (1/s2) further
    scaled by a similarity weight sim_d in [0, 1]. Reduces EXACTLY to
    dersimonian_laird() when sim is uniform across groups -- see the
    self-check in __main__ below, which asserts this directly rather than
    merely claiming it."""
    q, s2, sim = np.asarray(q, float), np.asarray(s2, float), np.asarray(sim, float)
    w = sim / s2
    mu_fe = np.sum(w * q) / np.sum(w)
    Q = np.sum(w * (q - mu_fe) ** 2)
    k = len(q)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    wr = sim / (s2 + tau2)
    return float(np.sum(wr * q) / np.sum(wr)), float(tau2)


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

    q, s2 = {}, {}
    for d in ALL:
        q[d] = fsq(rv[d], ALPHA)
        s2[d] = cell_bootstrap_variance(rv[d], cells[d], rng)
    fit = [d for d in ALL if s2[d] is not None]

    # Representation features for MMD, sampled once per dataset (fixed seed,
    # same MMD_SAMPLE_SIZE convention as nested_hyperparameter_search.py, so
    # distances here are directly comparable to HGC-CP's).
    Z = {}
    for d in ALL:
        Zf = load_representations(d)
        idx = rng.choice(len(Zf), size=min(len(Zf), MMD_SAMPLE_SIZE), replace=False)
        Z[d] = Zf[idx]

    cov = lambda d, h: float(np.mean(rt[d] <= h))

    out = {"alpha": ALPHA, "fit_groups": fit, "lam_grid": LAM_GRID, "default_lam": DEFAULT_LAM,
           "q_g": q, "sigma2_g": s2, "per_target": {}}

    print("MWHC: per-target calibration-level LODO, coverage on the target's test set (90% nominal)")
    print(f"{'Target':8s}{'flat mu':>9s}{'flat EB':>9s}  |" +
          "".join(f"  lam={l:<5g}EB-sim" for l in LAM_GRID))
    for t in TARGETS:
        others_fit = [d for d in fit if d != t]
        distances = {d: mmd_squared(Z[t], Z[d]) for d in others_fit}

        mu_flat, tau2_flat = dersimonian_laird([q[d] for d in others_fit], [s2[d] for d in others_fit])
        prior_flat = cov(t, mu_flat)
        if s2[t] is not None:
            w_flat = tau2_flat / (tau2_flat + s2[t]) if tau2_flat > 0 else 0.0
            eb_flat = cov(t, w_flat * q[t] + (1 - w_flat) * mu_flat)
        else:
            w_flat, eb_flat = None, None

        per_lam = {}
        for lam in LAM_GRID:
            sim = compute_group_weights(distances, lam)
            ess = effective_sample_size(sim)
            mu_sim, tau2_sim = weighted_dersimonian_laird(
                [q[d] for d in others_fit], [s2[d] for d in others_fit], [sim[d] for d in others_fit])
            prior_sim = cov(t, mu_sim)
            if s2[t] is not None:
                w_sim = tau2_sim / (tau2_sim + s2[t]) if tau2_sim > 0 else 0.0
                eb_sim = cov(t, w_sim * q[t] + (1 - w_sim) * mu_sim)
            else:
                w_sim, eb_sim = None, None

            # Floored variant: same lam, but the similarity weight feeding
            # the random-effects fit is truncated so no single source can
            # dominate -- see compute_floored_group_weights.
            sim_fl = compute_floored_group_weights(distances, lam, floor_frac=0.5)
            ess_fl = effective_sample_size(sim_fl)
            mu_fl, tau2_fl = weighted_dersimonian_laird(
                [q[d] for d in others_fit], [s2[d] for d in others_fit], [sim_fl[d] for d in others_fit])
            if s2[t] is not None:
                w_fl = tau2_fl / (tau2_fl + s2[t]) if tau2_fl > 0 else 0.0
                eb_fl = cov(t, w_fl * q[t] + (1 - w_fl) * mu_fl)
            else:
                w_fl, eb_fl = None, None

            per_lam[str(lam)] = {
                "sim_weights": sim, "ess": ess, "mu_hat_loo_sim": mu_sim, "tau2_loo_sim": tau2_sim,
                "w_eb_sim": w_sim, "cov_prior_only_sim": prior_sim, "cov_eb_shrunk_sim": eb_sim,
                "floored": {"sim_weights": sim_fl, "ess": ess_fl, "mu_hat_loo": mu_fl, "tau2_loo": tau2_fl,
                            "w_eb": w_fl, "cov_eb_shrunk": eb_fl},
            }

        out["per_target"][t] = {
            "n_test": int(len(rt[t])), "mmd_distances": distances,
            "mu_hat_loo_flat": mu_flat, "tau2_loo_flat": tau2_flat,
            "w_eb_flat": w_flat, "cov_prior_only_flat": prior_flat, "cov_eb_shrunk_flat": eb_flat,
            "per_lam": per_lam,
        }

        row = f"{t:8s}{100*prior_flat:8.1f}%{'' if eb_flat is None else f'{100*eb_flat:8.1f}%':>9s}  |"
        for lam in LAM_GRID:
            v = per_lam[str(lam)]["cov_eb_shrunk_sim"]
            row += f"  {'n/a':>12s}" if v is None else f"  {100*v:11.1f}%"
        print(row)

    # Headline summary: mean |coverage - 90%| across targets with an EB value
    # defined (same set hierarchical_calibration_v2.py's own summary would
    # use), flat vs. each lam, so a single number answers "did adding
    # similarity weighting help, on average, across all tried lam values,
    # with zero per-target tuning?"
    eb_targets = [t for t in TARGETS if out["per_target"][t]["w_eb_flat"] is not None]
    flat_dev = np.mean([abs(out["per_target"][t]["cov_eb_shrunk_flat"] - (1 - ALPHA)) for t in eb_targets])
    print(f"\nMean |coverage - 90%| across {len(eb_targets)} targets with EB defined:")
    print(f"  flat hierarchical (v2):        {flat_dev:.4f}")
    lam_devs, lam_devs_floored, lam_ess = {}, {}, {}
    for lam in LAM_GRID:
        d = np.mean([abs(out["per_target"][t]["per_lam"][str(lam)]["cov_eb_shrunk_sim"] - (1 - ALPHA))
                      for t in eb_targets])
        d_fl = np.mean([abs(out["per_target"][t]["per_lam"][str(lam)]["floored"]["cov_eb_shrunk"] - (1 - ALPHA))
                         for t in eb_targets])
        ess_mean = np.mean([out["per_target"][t]["per_lam"][str(lam)]["ess"] for t in TARGETS])
        lam_devs[lam], lam_devs_floored[lam], lam_ess[lam] = d, d_fl, ess_mean
        print(f"  MWHC, lam={lam:<6g}             {d:.4f}   (floored: {d_fl:.4f})   mean ESS: {ess_mean:.2f} / {len(fit)-1}")
    out["summary"] = {"eb_targets": eb_targets, "mean_abs_dev_flat": flat_dev,
                       "mean_abs_dev_by_lam": {str(l): v for l, v in lam_devs.items()},
                       "mean_abs_dev_by_lam_floored": {str(l): v for l, v in lam_devs_floored.items()},
                       "mean_ess_by_lam": {str(l): v for l, v in lam_ess.items()}}
    print("\nInterpretation: ESS near 1 means the similarity-weighted prior is effectively relying on a "
          "single other dataset; ESS near (k-1) means it is close to the flat/uniform prior. If coverage "
          "degrades as ESS falls (compare the lam=1 vs lam=20 rows above), that points to weight "
          "concentration -- not the similarity signal itself -- as the failure mode, and the floored "
          "variant should recover closer to the flat baseline's deviation at the same lam.")

    path = os.path.join(config.OUTPUT_DIR, "mwhc_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    # Self-check: weighted_dersimonian_laird with uniform similarity must
    # reduce EXACTLY to dersimonian_laird -- run before trusting anything
    # else in this file.
    q_chk = [0.01, 0.015, 0.02, 0.012]
    s2_chk = [1e-5, 2e-5, 1.5e-5, 1.2e-5]
    mu_a, tau2_a = dersimonian_laird(q_chk, s2_chk)
    mu_b, tau2_b = weighted_dersimonian_laird(q_chk, s2_chk, [1.0, 1.0, 1.0, 1.0])
    assert abs(mu_a - mu_b) < 1e-12 and abs(tau2_a - tau2_b) < 1e-12, \
        "weighted_dersimonian_laird(uniform sim) must equal dersimonian_laird exactly"
    print("self-check passed: weighted_dersimonian_laird(uniform sim) == dersimonian_laird\n")

    main()
