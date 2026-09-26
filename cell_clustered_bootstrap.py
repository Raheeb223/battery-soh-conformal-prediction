"""
Cell-clustered bootstrap confidence intervals for coverage (Table 8,
Section 5.3.1).

WHY THIS MATTERS: Table 7's Clopper-Pearson intervals treat every TEST
WINDOW as an independent Bernoulli trial. But windows from the same cell are
correlated (overlapping input history, shared cell-level noise) -- if that
correlation is real, the existing intervals are narrower than they should
be, meaning some "statistically well-powered" claims may overstate
precision. This script tests that directly: resample whole CELLS (not
windows) with replacement, and compare the resulting coverage confidence
intervals against the existing per-window ones.

Two datasets have only ONE unique test cell (NASA, Oxford). Cell-clustered
bootstrap is mathematically undefined for a single cell -- resampling one
cell repeatedly captures zero between-cell variability, so this script
explicitly reports those two as undefined rather than compute a number that
would look precise but isn't meaningful. CALCE has only 2 test cells,
severely limiting how much between-cell variability the bootstrap can
express; its result should be read with that in mind, not treated as equally
reliable as MIT's (21 cells).

USAGE: python cell_clustered_bootstrap.py
"""

import os

import numpy as np

import config

N_BOOTSTRAP = 2000
ALPHA = 0.1  # matches the paper's 90% nominal coverage target
SEED = 0


def load_data(pred_path=None):
    if pred_path is None:
        pred_path = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz")
    return np.load(pred_path, allow_pickle=True)


def cell_clustered_bootstrap_coverage(y_true, lower, upper, cell_ids, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Resample whole cells with replacement (same count as original unique
    cells), pool all windows from the resampled cells, compute coverage on
    each bootstrap draw. Returns the array of bootstrap coverage estimates."""
    rng = np.random.default_rng(seed)
    unique_cells = np.unique(cell_ids)
    n_cells = len(unique_cells)

    if n_cells < 2:
        return None  # undefined -- see module docstring

    covered = (y_true >= lower) & (y_true <= upper)
    # Precompute each cell's window indices once, for speed across bootstrap draws.
    cell_to_indices = {c: np.where(cell_ids == c)[0] for c in unique_cells}

    boot_coverages = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled_cells = rng.choice(unique_cells, size=n_cells, replace=True)
        idx = np.concatenate([cell_to_indices[c] for c in sampled_cells])
        boot_coverages[b] = covered[idx].mean()
    return boot_coverages


def clopper_pearson(k, n, alpha=ALPHA):
    """Exact per-window interval, for direct comparison -- same formula
    already used for Table 7, reimplemented here only for side-by-side
    printing rather than re-deriving anything new."""
    from scipy.stats import beta
    if k == 0:
        lo = 0.0
    else:
        lo = beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def main():
    data = load_data()
    datasets = np.unique(data["lab_test"])

    print(f"{'Dataset':<8}{'n_cells':<9}{'n_windows':<11}{'Per-window CI (width)':<24}{'Cell-boot 95% CI (width)'}")
    for d in datasets:
        mask = data["lab_test"] == d
        y_true = data["y_test"][mask]
        lower = data["lower"][mask]
        upper = data["upper"][mask]
        cell_ids = data["cell_test"][mask]
        n_windows = mask.sum()
        n_cells = len(np.unique(cell_ids))

        covered = (y_true >= lower) & (y_true <= upper)
        k = covered.sum()
        cp_lo, cp_hi = clopper_pearson(k, n_windows)
        cp_width = cp_hi - cp_lo

        if n_cells < 2:
            print(f"{d:<8}{n_cells:<9}{n_windows:<11}"
                  f"[{cp_lo:.3f}, {cp_hi:.3f}] ({cp_width:.3f})   "
                  f"UNDEFINED (only {n_cells} unique test cell)")
            continue

        boot = cell_clustered_bootstrap_coverage(y_true, lower, upper, cell_ids)
        boot_lo, boot_hi = np.percentile(boot, [2.5, 97.5])
        boot_width = boot_hi - boot_lo

        flag = ""
        if n_cells <= 3:
            flag = f"  (only {n_cells} cells -- limited)"

        print(f"{d:<8}{n_cells:<9}{n_windows:<11}"
              f"[{cp_lo:.3f}, {cp_hi:.3f}] ({cp_width:.3f})   "
              f"[{boot_lo:.3f}, {boot_hi:.3f}] ({boot_width:.3f}){flag}")

    print("\nIf cell-clustered widths are meaningfully larger than per-window widths,")
    print("that confirms within-cell correlation was inflating precision claims based")
    print("on treating windows as independent trials.")


if __name__ == "__main__":
    main()
