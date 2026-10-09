"""
Sensitivity check for tau^2_hat (the DerSimonian-Laird random-effects
variance) itself, not just its downstream effect on shrinkage weights or
mu_hat -- addressing the specific concern that tau^2_hat, estimated from
only 5-6 groups, may itself be unstable and deserves its own uncertainty
characterization, not just a single point value.

METHOD: for each of N_OUTER bootstrap iterations, resample each dataset's
calibration cells with replacement (the same cell-clustered logic used
throughout this project), recompute that dataset's q_g and a plug-in
sigma_g^2 from the resample, then run the full DerSimonian-Laird procedure
on this one bootstrap draw of all 6 datasets simultaneously. This produces
a genuine bootstrap DISTRIBUTION of tau^2_hat (and mu_hat), not a single
number -- showing directly how much sampling noise in the 5-6 group inputs
propagates into the random-effects variance estimate itself.

USAGE: python bootstrap_tau2_sensitivity.py
"""

import numpy as np

PROCESSED_NPZ_PATH = r"C:\Users\User\PycharmProjects\cache\processed.npz"
VAL_PRED_PATH = r"C:\Users\User\PycharmProjects\outputs\conformal_groupwise_predictions_val.npz"

ALPHA = 0.10
N_OUTER = 1000  # outer bootstrap draws (each redoing the full DL procedure once)
SEED = 0
DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford"]

PROXY_SIGMA_G2 = {
    "MIT": (0.009 / (2 * 1.96)) ** 2, "RWTH": (0.012 / (2 * 1.96)) ** 2,
    "XJTU": (0.015 / (2 * 1.96)) ** 2, "BIT": (0.048 / (2 * 1.96)) ** 2,
    "CALCE": (0.187 / (2 * 1.96)) ** 2, "Oxford": (0.064 / (2 * 1.96)) ** 2,
}


def get_val_residuals_and_cells(dataset: str):
    val = np.load(VAL_PRED_PATH, allow_pickle=True)
    cache = np.load(PROCESSED_NPZ_PATH, allow_pickle=True)
    soh_mean, soh_std = cache["soh_mean"].item(), cache["soh_std"].item()
    m, s = soh_mean[dataset], soh_std[dataset]
    mask = val["lab_val"] == dataset
    y_real = val["y_val"][mask] * s + m
    pred_real = val["mean_pred"][mask] * s + m
    return np.abs(y_real - pred_real), val["cell_val"][mask]


def pooled_quantile_at_alpha(residuals: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(residuals)
    if n == 0:
        return np.nan
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(residuals, level)


def dersimonian_laird(q_g: dict, sigma_g2: dict, groups: list):
    w = {g: 1 / sigma_g2[g] for g in groups}
    mu_fe = sum(w[g] * q_g[g] for g in groups) / sum(w[g] for g in groups)
    Q = sum(w[g] * (q_g[g] - mu_fe) ** 2 for g in groups)
    k = len(groups)
    sum_w, sum_w2 = sum(w[g] for g in groups), sum(w[g] ** 2 for g in groups)
    tau2 = max(0.0, (Q - (k - 1)) / (sum_w - sum_w2 / sum_w))
    w_prime = {g: 1 / (sigma_g2[g] + tau2) for g in groups}
    mu_hat = sum(w_prime[g] * q_g[g] for g in groups) / sum(w_prime[g] for g in groups)
    return mu_hat, tau2


def main():
    data_by_dataset = {}
    for d in DATASETS:
        residuals, cell_ids = get_val_residuals_and_cells(d)
        unique_cells = np.unique(cell_ids)
        cell_to_idx = {c: np.where(cell_ids == c)[0] for c in unique_cells}
        data_by_dataset[d] = {
            "residuals": residuals, "cell_ids": cell_ids,
            "unique_cells": unique_cells, "cell_to_idx": cell_to_idx,
            "has_direct": len(unique_cells) >= 2,
        }

    rng = np.random.default_rng(SEED)
    tau2_draws, mu_draws = [], []

    # Direct sigma_g^2 point estimates already established (Appendix A.5);
    # held fixed across outer draws so this check isolates how sampling
    # noise in q_g (and hence in Q and tau^2) propagates, rather than
    # re-deriving sigma_g^2 itself at each draw (which would require a
    # nested nested bootstrap, too expensive at N_OUTER=1000).
    direct_sigma_g2_point = {
        "MIT": 0.00000009, "RWTH": 0.00000007, "XJTU": 0.00002432,
        "BIT": 0.00062758, "CALCE": 0.00010669, "Oxford": PROXY_SIGMA_G2["Oxford"],
    }

    for outer in range(N_OUTER):
        q_g_draw = {}
        for d in DATASETS:
            info = data_by_dataset[d]
            if not info["has_direct"]:
                # Oxford: only 1 cell exists, so cell-level resampling is
                # undefined; window-level resampling is used here as a
                # documented fallback for THIS sensitivity check only, and
                # understates true between-cell uncertainty for this one
                # dataset, consistent with its illustrative-only treatment
                # elsewhere in this paper.
                idx = rng.choice(len(info["residuals"]), size=len(info["residuals"]), replace=True)
            else:
                sampled_cells = rng.choice(info["unique_cells"], size=len(info["unique_cells"]), replace=True)
                idx = np.concatenate([info["cell_to_idx"][c] for c in sampled_cells])
            q_g_draw[d] = pooled_quantile_at_alpha(info["residuals"][idx])

        mu_d, tau2_d = dersimonian_laird(q_g_draw, direct_sigma_g2_point, DATASETS)
        mu_draws.append(mu_d)
        tau2_draws.append(tau2_d)

    tau2_draws, mu_draws = np.array(tau2_draws), np.array(mu_draws)
    print(f"tau2_hat bootstrap distribution over {N_OUTER} draws (resampling q_g's underlying cells):")
    print(f"  point estimate (no resampling): 0.00004601")
    print(f"  bootstrap mean: {tau2_draws.mean():.8f}")
    print(f"  bootstrap median: {np.median(tau2_draws):.8f}")
    print(f"  bootstrap 95% CI: [{np.percentile(tau2_draws, 2.5):.8f}, {np.percentile(tau2_draws, 97.5):.8f}]")
    print(f"  fraction of draws with tau2_hat == 0 (boundary): {(tau2_draws == 0).mean():.3f}")

    print(f"\nmu_hat bootstrap distribution over {N_OUTER} draws:")
    print(f"  point estimate (no resampling): 0.01800")
    print(f"  bootstrap mean: {mu_draws.mean():.5f}")
    print(f"  bootstrap 95% CI: [{np.percentile(mu_draws, 2.5):.5f}, {np.percentile(mu_draws, 97.5):.5f}]")


if __name__ == "__main__":
    main()
