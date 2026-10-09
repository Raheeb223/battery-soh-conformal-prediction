"""
Corrected hierarchical calibration: direct cell-clustered sigma_g^2 as the
PRIMARY variance input, with the original coverage-CI-derived proxy
retained only as a secondary sensitivity comparison -- reversing which
estimate this paper treats as primary, per review.

HONEST LIMITATION STATED UP FRONT: Oxford has exactly 1 validation cell.
Direct cell-clustered bootstrap is mathematically undefined with a single
cell (no between-cell variability exists to resample), the same limitation
already established for NASA elsewhere in this paper. Oxford's sigma_g^2
in the "corrected, primary" analysis below therefore necessarily falls
back to the original proxy, explicitly flagged as such at every place it
is used -- this is not silently glossed over.

USAGE: python corrected_hierarchical_calibration.py
"""

import numpy as np

PROCESSED_NPZ_PATH = r"C:\Users\User\PycharmProjects\cache\processed.npz"
VAL_PRED_PATH = r"C:\Users\User\PycharmProjects\outputs\conformal_groupwise_predictions_val.npz"
TEST_PRED_PATH = r"C:\Users\User\PycharmProjects\outputs\conformal_groupwise_predictions.npz"

ALPHA = 0.10
N_BOOTSTRAP = 2000
SEED = 0
DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford"]  # the 6 used in hierarchical calibration (NASA excluded there already)

# Original proxy, from Table 7's per-window Clopper-Pearson CI widths --
# retained here ONLY as the fallback for Oxford and as the sensitivity
# comparison, not as the primary estimate.
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
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(residuals, level)


def direct_sigma_g2(residuals: np.ndarray, cell_ids: np.ndarray, seed: int = SEED):
    unique_cells = np.unique(cell_ids)
    if len(unique_cells) < 2:
        return None
    rng = np.random.default_rng(seed)
    cell_to_idx = {c: np.where(cell_ids == c)[0] for c in unique_cells}
    boot_q = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sampled = rng.choice(unique_cells, size=len(unique_cells), replace=True)
        idx = np.concatenate([cell_to_idx[c] for c in sampled])
        boot_q[b] = pooled_quantile_at_alpha(residuals[idx])
    return np.var(boot_q)


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
    q_g, n_g, sigma_direct, sigma_used_flag = {}, {}, {}, {}
    for dataset in DATASETS:
        residuals, cell_ids = get_val_residuals_and_cells(dataset)
        q_g[dataset] = pooled_quantile_at_alpha(residuals)
        n_g[dataset] = len(residuals)
        direct = direct_sigma_g2(residuals, cell_ids)
        if direct is None:
            sigma_direct[dataset] = PROXY_SIGMA_G2[dataset]
            sigma_used_flag[dataset] = "PROXY (direct undefined: <2 cells)"
        else:
            sigma_direct[dataset] = direct
            sigma_used_flag[dataset] = "direct bootstrap"

    print(f"{'Dataset':<8}{'n_g':<8}{'q_g':<9}{'sigma_g2 used':<16}{'Source'}")
    for d in DATASETS:
        print(f"{d:<8}{n_g[d]:<8}{q_g[d]:<9.4f}{sigma_direct[d]:<16.8f}{sigma_used_flag[d]}")

    mu_primary, tau2_primary = dersimonian_laird(q_g, sigma_direct, DATASETS)
    mu_proxy, tau2_proxy = dersimonian_laird(q_g, PROXY_SIGMA_G2, DATASETS)

    print(f"\nPRIMARY (direct, Oxford falls back to proxy): mu_hat={mu_primary:.5f}, tau2_hat={tau2_primary:.8f}")
    print(f"SENSITIVITY (original proxy, all datasets):    mu_hat={mu_proxy:.5f}, tau2_hat={tau2_proxy:.8f}")

    print(f"\n{'Dataset':<8}{'w_g (primary)':<16}{'w_g (proxy, sensitivity)'}")
    for d in DATASETS:
        w_primary = n_g[d] / (n_g[d] + tau2_primary / sigma_direct[d])
        w_proxy = n_g[d] / (n_g[d] + tau2_proxy / PROXY_SIGMA_G2[d])
        print(f"{d:<8}{w_primary:<16.4f}{w_proxy:.4f}")

    # ---- Downstream: prior-only and prior+100 for CALCE, the case that matters most ----
    print("\n--- CALCE prior-only and prior+100, under PRIMARY (direct) inputs ---")
    target = "CALCE"
    test_pred = np.load(TEST_PRED_PATH, allow_pickle=True)
    cache = np.load(PROCESSED_NPZ_PATH, allow_pickle=True)
    m, s = cache["soh_mean"].item()[target], cache["soh_std"].item()[target]
    mask = test_pred["lab_test"] == target
    y_true = test_pred["y_test"][mask] * s + m
    yhat = test_pred["mean_pred"][mask] * s + m

    # prior-only: apply mu_primary as the quantile directly (zero target data)
    lower, upper = yhat - mu_primary, yhat + mu_primary
    cov_prior_only = np.mean((y_true >= lower) & (y_true <= upper))
    print(f"Prior-only (mu_hat={mu_primary:.4f} applied directly): coverage={cov_prior_only:.3f}")

    # prior+100: shrink using w_g computed with n_g=100 target samples added
    residuals_target, cell_ids_target = get_val_residuals_and_cells(target)
    idx100 = np.random.default_rng(SEED).choice(len(residuals_target), size=min(100, len(residuals_target)), replace=False)
    own_q_100 = pooled_quantile_at_alpha(residuals_target[idx100])
    w_100 = 100 / (100 + tau2_primary / sigma_direct[target])
    q_prior_plus_100 = w_100 * own_q_100 + (1 - w_100) * mu_primary
    lower2, upper2 = yhat - q_prior_plus_100, yhat + q_prior_plus_100
    cov_prior_plus_100 = np.mean((y_true >= lower2) & (y_true <= upper2))
    print(f"Prior+100 (w_g={w_100:.4f}): coverage={cov_prior_plus_100:.3f}")


if __name__ == "__main__":
    main()
