"""
Drop-in replacement for hierarchical_calibration.py's sigma_g^2 computation.

WHY THIS IS A PATCH, NOT A STANDALONE SCRIPT: reconstructing Table 16 from
outside the original pipeline risks subtle methodological drift (confirmed
in this project's earlier validation: even a quantity that should not
depend on sigma_g^2 at all showed unexplained differences from the
original, most likely from a difference in how q_g itself was derived).
The only way to guarantee Table 16's other columns -- (i) n-weighted pooled
approx, (iii) prior+100, (iv) true groupwise -- stay exactly consistent
with the paper's already-published methodology is to keep everything about
this pipeline the same EXCEPT the one input being corrected. This file
provides that one replacement function; it is not meant to be run on its
own.

INTEGRATION STEPS:
1. In your hierarchical_calibration.py, find wherever sigma_g^2 is currently
   computed (likely derived from Table 7's per-window Clopper-Pearson CI
   width, e.g. something like `sigma_g2 = (ci_width / (2*1.96))**2`).
2. Import or paste compute_direct_sigma_g2() below into that file.
3. Replace the sigma_g^2 assignment with a call to this function, passing
   each dataset's validation residuals and cell IDs (already loaded
   elsewhere in your pipeline for q_g's own computation -- reuse the same
   arrays rather than reload them separately, to guarantee q_g and sigma_g^2
   are computed from IDENTICAL underlying data).
4. For Oxford specifically (and any other dataset with fewer than 2 unique
   validation cells), direct estimation is undefined -- keep your existing
   proxy computation as an explicit fallback for that case only, and print
   or log which datasets used which method so this is auditable in the
   output, not silent.
5. Re-run your existing pipeline unmodified otherwise. Every other column
   in Table 16 (n-weighted pooled approx, prior+100, true groupwise) uses
   your existing, already-correct q_g and n_g computations and does not
   need to change -- only this one function's output feeds into them
   differently now.

VERIFICATION BEFORE TRUSTING THE OUTPUT: this function has already been
validated against your real data in this project (see
validate_sigma_g_proxy.py and corrected_hierarchical_calibration.py,
already confirmed to reproduce identical numbers on your machine). Before
trusting a full pipeline re-run, sanity-check that plugging this function
in reproduces the SAME sigma_g^2 values already reported in the manuscript
appendix for at least MIT and CALCE:
    MIT:   sigma_g^2 ~= 0.00000009
    CALCE: sigma_g^2 ~= 0.00010669
If your pipeline's own re-run of this function gives different numbers for
these two, something about how residuals/cell_ids are being passed in
differs from what this patch assumes -- stop and check before trusting the
rest of the table.
"""

import numpy as np

N_BOOTSTRAP = 2000
ALPHA = 0.10  # match whatever alpha your pipeline already uses for q_g


def compute_direct_sigma_g2(residuals: np.ndarray, cell_ids: np.ndarray,
                             alpha: float = ALPHA, n_bootstrap: int = N_BOOTSTRAP,
                             seed: int = 0):
    """Direct, cell-clustered bootstrap estimate of calibration-quantile
    sampling variance. Returns (sigma_g2, method_used_string).

    residuals: this dataset's |y - yhat| validation residuals -- the SAME
        array your pipeline already uses to compute q_g, not a separately
        reloaded copy.
    cell_ids: the corresponding per-residual cell identifiers, same length
        and same order as residuals.

    Returns None for sigma_g2 (not a fallback value) when fewer than 2
    unique cells exist, so the caller must explicitly decide the fallback
    (e.g. the existing proxy) rather than this function silently choosing
    one -- this keeps the fallback decision visible in your pipeline's own
    code, not hidden inside this utility.
    """
    unique_cells = np.unique(cell_ids)
    if len(unique_cells) < 2:
        return None, f"undefined ({len(unique_cells)} unique cell(s), need >=2)"

    rng = np.random.default_rng(seed)
    cell_to_idx = {c: np.where(cell_ids == c)[0] for c in unique_cells}

    def pooled_quantile(r, a):
        n = len(r)
        level = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
        return np.quantile(r, level)

    boot_q = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled_cells = rng.choice(unique_cells, size=len(unique_cells), replace=True)
        idx = np.concatenate([cell_to_idx[c] for c in sampled_cells])
        boot_q[b] = pooled_quantile(residuals[idx], alpha)

    return np.var(boot_q), "direct cell-clustered bootstrap"


# ---------------------------------------------------------------------------
# Example of how this slots into your existing per-dataset loop -- adapt
# variable names to match your actual pipeline, this is illustrative only.
# ---------------------------------------------------------------------------
EXAMPLE_INTEGRATION = '''
for dataset in datasets:
    residuals_g = ...   # your existing per-dataset validation residuals
    cell_ids_g = ...    # your existing per-dataset validation cell IDs
    q_g[dataset] = ...  # your existing, UNCHANGED q_g computation

    sigma_g2_direct, method = compute_direct_sigma_g2(residuals_g, cell_ids_g)
    if sigma_g2_direct is not None:
        sigma_g2[dataset] = sigma_g2_direct
    else:
        sigma_g2[dataset] = your_existing_proxy_computation(dataset)
        method = "PROXY fallback (" + method + ")"
    print(f"{dataset}: sigma_g2={sigma_g2[dataset]:.8f} [{method}]")

# The rest of your pipeline -- DerSimonian-Laird tau^2/mu_hat estimation,
# shrinkage weight computation, and Table 16's four columns -- runs exactly
# as before, using this corrected sigma_g2 dict as its only changed input.
'''

if __name__ == "__main__":
    print(__doc__)
    print("\nThis file defines compute_direct_sigma_g2() for you to import or paste")
    print("into your existing hierarchical_calibration.py -- it is not meant to be")
    print("run standalone. See EXAMPLE_INTEGRATION above for the integration pattern.")
