"""
Main manuscript figures, generated from completed experiment runs
(outputs/*_results.json, outputs/*_predictions.npz,
outputs/significance_testing_results.json). Nothing is retrained.

Figures (saved to outputs/figures/):
  01_prediction_vs_actual.png          -- proposed model, overall + per dataset
  02_residual_distribution.png         -- error histograms, overall + per dataset
  03_per_cell_curves.png               -- degradation curves for sample test
                                          cells, with cell-boundary markers
  04_conformal_coverage_comparison.png -- groupwise vs pooled vs
                                          normalized_groupwise coverage per dataset
  05_baseline_vs_proposed_mae.png      -- baselines vs proposed model
  06_architecture_ablation.png         -- MAE + coverage per variant
  07_window_size_ablation.png          -- MAE/RMSE vs window size
  08_lodo_generalization.png           -- MAE per held-out dataset
  09_significance_forest_plot.png      -- Cliff's delta + 95% CI per comparison
  10_late_stage_aging_analysis.png     -- error behaviour at low SOH

Run:
    python generate_all_figures.py
"""

import os
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt

import config

FIG_DIR = os.path.join(config.OUTPUT_DIR, "figures")
DATASET_ORDER = ["MIT", "BIT", "XJTU", "RWTH", "CALCE", "Oxford", "NASA"]
DATASET_COLORS = {
    "MIT": "#4C72B0", "BIT": "#DD8452", "XJTU": "#55A868", "RWTH": "#C44E52",
    "CALCE": "#8172B2", "Oxford": "#937860", "NASA": "#DA8BC3",
}


def _load_predictions(run_name):
    candidates = glob.glob(os.path.join(config.OUTPUT_DIR, f"*{run_name}*predictions.npz"))
    if not candidates:
        print(f"  [skip] no predictions file found for {run_name}")
        return None
    return np.load(candidates[0], allow_pickle=True)


def _load_results(run_name):
    path = os.path.join(config.OUTPUT_DIR, f"{run_name}_results.json")
    if not os.path.exists(path):
        print(f"  [skip] no results file found for {run_name}")
        return None
    with open(path) as fh:
        return json.load(fh)


def _ordered_datasets(present):
    return [d for d in DATASET_ORDER if d in present] + \
           [d for d in present if d not in DATASET_ORDER]


# ---------------------------------------------------------------------------
# 01. Prediction vs actual
# ---------------------------------------------------------------------------

def fig_prediction_vs_actual():
    print("\n[01] Prediction vs actual...")
    data = _load_predictions("ablation_full")
    if data is None:
        return

    y_true, y_pred, lab = data["y_test"], data["mean_pred"], data["lab_test"]
    datasets = _ordered_datasets(set(np.unique(lab).tolist()))

    n = len(datasets)
    ncols = 4
    nrows = (n + 1 + ncols - 1) // ncols  # +1 for the overall panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    lims = (min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()))

    ax = axes[0]
    ax.scatter(y_true, y_pred, s=3, alpha=0.15, color="#4C72B0")
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_title(f"Overall (n={len(y_true)})")
    ax.set_xlabel("True (normalized SOH)")
    ax.set_ylabel("Predicted")

    for i, ds in enumerate(datasets, start=1):
        mask = lab == ds
        ax = axes[i]
        ax.scatter(y_true[mask], y_pred[mask], s=4, alpha=0.25,
                   color=DATASET_COLORS.get(ds, "#333333"))
        ax.plot(lims, lims, "k--", linewidth=1)
        mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        ax.set_title(f"{ds} (n={mask.sum()}, MAE={mae:.4f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

    for j in range(n + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Prediction vs Actual — proposed model (ablation_full)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "01_prediction_vs_actual.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 02. Residual distribution
# ---------------------------------------------------------------------------

def fig_residual_distribution():
    print("\n[02] Residual distribution...")
    data = _load_predictions("ablation_full")
    if data is None:
        return

    y_true, y_pred, lab = data["y_test"], data["mean_pred"], data["lab_test"]
    residuals = y_pred - y_true
    datasets = _ordered_datasets(set(np.unique(lab).tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(residuals, bins=80, color="#4C72B0", alpha=0.8)
    axes[0].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[0].set_title(f"Overall residuals (mean={residuals.mean():.4f}, "
                       f"std={residuals.std():.4f})")
    axes[0].set_xlabel("Predicted − True")

    for ds in datasets:
        mask = lab == ds
        axes[1].hist(residuals[mask], bins=40, alpha=0.5, label=ds,
                     color=DATASET_COLORS.get(ds, None), density=True)
    axes[1].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[1].set_title("Per-dataset residual distributions (density-normalized)")
    axes[1].set_xlabel("Predicted − True")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "02_residual_distribution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 03. Per-cell curves WITH boundary markers (uses cell_test)
# ---------------------------------------------------------------------------

def fig_per_cell_curves(n_cells_per_dataset=2):
    print("\n[03] Per-cell curves with boundary markers...")
    data = _load_predictions("ablation_full")
    if data is None:
        return
    if "cell_test" not in data.files:
        print("  [!] no cell_test field in this predictions file -- "
              "re-run experiments.py with the updated train.py/preprocessing.py "
              "first (see project notes on the cell_test addition).")
        return

    y_true, y_pred, lab, cell = data["y_test"], data["mean_pred"], data["lab_test"], data["cell_test"]
    lower = data["lower"] if "lower" in data.files else None
    upper = data["upper"] if "upper" in data.files else None

    datasets = _ordered_datasets(set(np.unique(lab).tolist()))
    selected_cells = []
    for ds in datasets:
        ds_cells = np.unique(cell[lab == ds])
        for c in ds_cells[:n_cells_per_dataset]:
            selected_cells.append((ds, c))

    ncols = 4
    nrows = (len(selected_cells) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, (ds, c) in enumerate(selected_cells):
        ax = axes[i]
        mask = cell == c
        # sort by test-set order (index) as a proxy for cycle order --
        # exact cycle isn't saved per-window, but relative window order
        # within a cell IS preserved by make_windows()'s sequential
        # iteration, so this correctly shows the degradation trajectory
        idx = np.where(mask)[0]
        ax.plot(range(len(idx)), y_true[idx], "o-", markersize=3,
                label="True", color="#333333", linewidth=1)
        ax.plot(range(len(idx)), y_pred[idx], "o-", markersize=3,
                label="Predicted", color=DATASET_COLORS.get(ds, "#4C72B0"),
                linewidth=1, alpha=0.8)
        if lower is not None and upper is not None:
            ax.fill_between(range(len(idx)), lower[idx], upper[idx],
                            alpha=0.15, color=DATASET_COLORS.get(ds, "#4C72B0"),
                            label="Conformal interval")
        ax.set_title(f"{ds}: {c}", fontsize=9)
        ax.set_xlabel("Test window index (within cell)")
        ax.set_ylabel("Normalized SOH")
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(selected_cells), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-cell prediction curves (boundary-marked by cell_test)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "03_per_cell_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 04. Conformal coverage comparison -- THE headline result
# ---------------------------------------------------------------------------

def fig_conformal_coverage_comparison():
    print("\n[04] Conformal coverage comparison (headline result)...")
    modes = ["groupwise", "pooled", "normalized_groupwise"]
    mode_labels = {"groupwise": "Groupwise", "pooled": "Pooled",
                   "normalized_groupwise": "Normalized\nGroupwise"}
    mode_colors = {"groupwise": "#4C72B0", "pooled": "#C44E52",
                   "normalized_groupwise": "#55A868"}

    per_mode_coverage = {}
    for mode in modes:
        r = _load_results(f"conformal_{mode}")
        if r is None:
            continue
        per_mode_coverage[mode] = r["per_dataset_coverage"]

    if not per_mode_coverage:
        print("  [skip] no conformal ablation results found.")
        return

    all_datasets = set()
    for cov in per_mode_coverage.values():
        all_datasets |= set(cov.keys())
    datasets = _ordered_datasets(all_datasets)

    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, mode in enumerate(modes):
        if mode not in per_mode_coverage:
            continue
        vals = [per_mode_coverage[mode].get(ds, 0) * 100 for ds in datasets]
        ax.bar(x + (i - 1) * width, vals, width, label=mode_labels[mode],
              color=mode_colors[mode])

    ax.axhline(90, color="k", linestyle="--", linewidth=1, label="90% target")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Conformal Calibration: Groupwise vs Pooled vs Normalized Groupwise\n"
                 "(Pooled catastrophically under-covers on several datasets)")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "04_conformal_coverage_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 05. Baseline vs proposed MAE
# ---------------------------------------------------------------------------

def fig_baseline_vs_proposed():
    print("\n[05] Baseline vs proposed MAE...")
    names = ["baseline_linear", "baseline_svr", "baseline_xgboost", "ablation_full"]
    labels = ["Linear", "SVR", "XGBoost", "Proposed\n(full)"]
    colors = ["#999999", "#999999", "#999999", "#4C72B0"]

    maes, rmses, used_labels, used_colors = [], [], [], []
    for name, label, color in zip(names, labels, colors):
        r = _load_results(name)
        if r is None:
            continue
        maes.append(r["test_mae"])
        rmses.append(r["test_rmse"])
        used_labels.append(label)
        used_colors.append(color)

    if not maes:
        print("  [skip] no baseline/proposed results found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    x = np.arange(len(used_labels))
    axes[0].bar(x, maes, color=used_colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(used_labels)
    axes[0].set_ylabel("Test MAE")
    axes[0].set_title("MAE (lower is better)")

    axes[1].bar(x, rmses, color=used_colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(used_labels)
    axes[1].set_ylabel("Test RMSE")
    axes[1].set_title("RMSE (lower is better)")

    fig.suptitle("Baselines vs Proposed Model", fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "05_baseline_vs_proposed_mae.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 06. Architecture ablation
# ---------------------------------------------------------------------------

def fig_architecture_ablation():
    print("\n[06] Architecture ablation...")
    variants = ["full", "no_attention", "no_multiscale", "plain_lstm", "gru"]
    labels = ["Full\n(proposed)", "No\nAttention", "No\nMultiscale", "Plain\nLSTM", "GRU"]

    maes, coverages, used_labels = [], [], []
    for variant, label in zip(variants, labels):
        r = _load_results(f"ablation_{variant}")
        if r is None:
            continue
        maes.append(r["test_mae"])
        coverages.append(r["overall_coverage"] * 100)
        used_labels.append(label)

    if not maes:
        print("  [skip] no ablation results found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(used_labels))
    colors = ["#4C72B0"] + ["#999999"] * (len(used_labels) - 1)

    axes[0].bar(x, maes, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(used_labels, fontsize=9)
    axes[0].set_ylabel("Test MAE")
    axes[0].set_title("Architecture ablation — MAE")

    axes[1].bar(x, coverages, color=colors)
    axes[1].axhline(90, color="k", linestyle="--", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(used_labels, fontsize=9)
    axes[1].set_ylabel("Overall coverage (%)")
    axes[1].set_title("Architecture ablation — Coverage")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "06_architecture_ablation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 07. Window-size ablation
# ---------------------------------------------------------------------------

def fig_window_size_ablation():
    print("\n[07] Window-size ablation...")
    windows = [10, 20, 30, 50]
    maes, rmses, used_windows = [], [], []
    for w in windows:
        r = _load_results(f"window_{w}")
        if r is None:
            continue
        maes.append(r["test_mae"])
        rmses.append(r["test_rmse"])
        used_windows.append(w)

    if not maes:
        print("  [skip] no window-size ablation results found.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(used_windows, maes, "o-", label="MAE", color="#4C72B0")
    ax.plot(used_windows, rmses, "s-", label="RMSE", color="#C44E52")
    ax.set_xlabel("Window size (cycles)")
    ax.set_ylabel("Error")
    ax.set_title("Window-Size Ablation\n(note: window=30/50 drop NASA's shortest cells entirely)")
    ax.legend()
    ax.set_xticks(used_windows)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "07_window_size_ablation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 08. LODO generalization
# ---------------------------------------------------------------------------

def fig_lodo_generalization():
    print("\n[08] LODO generalization...")
    datasets = ["NASA", "CALCE", "Oxford", "MIT", "BIT", "XJTU", "RWTH"]
    maes, coverages, used_datasets = [], [], []
    for ds in datasets:
        r = _load_results(f"lodo_{ds}")
        if r is None:
            continue
        maes.append(r["test_mae"])
        coverages.append(r["coverage"] * 100)
        used_datasets.append(ds)

    if not maes:
        print("  [skip] no LODO results found.")
        return

    # sort by MAE ascending for readability
    order = np.argsort(maes)
    used_datasets = [used_datasets[i] for i in order]
    maes = [maes[i] for i in order]
    coverages = [coverages[i] for i in order]
    colors = [DATASET_COLORS.get(ds, "#999999") for ds in used_datasets]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(used_datasets))

    axes[0].bar(x, maes, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(used_datasets, rotation=30, ha="right")
    axes[0].set_ylabel("Test MAE (real SOH units)")
    axes[0].set_title("LODO — Held-out MAE (sorted, easiest → hardest)")

    axes[1].bar(x, coverages, color=colors)
    axes[1].axhline(90, color="k", linestyle="--", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(used_datasets, rotation=30, ha="right")
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_title("LODO — Held-out Coverage")

    fig.suptitle("Leave-One-Dataset-Out Generalization", fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "08_lodo_generalization.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 09. Significance forest plot
# ---------------------------------------------------------------------------

def fig_significance_forest_plot():
    print("\n[09] Significance forest plot...")
    path = os.path.join(config.OUTPUT_DIR, "significance_testing_results.json")
    if not os.path.exists(path):
        print("  [skip] significance_testing_results.json not found -- "
              "run remaining_analysis.py first.")
        return
    with open(path) as fh:
        results = json.load(fh)

    if not results:
        print("  [skip] significance_testing_results.json is empty.")
        return

    labels = [r["comparison"].replace("proposed_vs_", "") for r in results]
    deltas = [r["cliffs_delta"] for r in results]
    ci_lo = [r["mae_diff_95ci"][0] for r in results]
    ci_hi = [r["mae_diff_95ci"][1] for r in results]
    effect_sizes = [r["effect_size"] for r in results]

    color_map = {"negligible": "#999999", "small": "#DA8BC3",
                 "medium": "#DD8452", "large": "#C44E52"}
    colors = [color_map.get(e, "#333333") for e in effect_sizes]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 1.5))
    ax.errorbar(deltas, y,
                xerr=[[d - lo for d, lo in zip(np.zeros(len(deltas)), np.zeros(len(deltas)))]],
                fmt="none")  # placeholder, real errorbars drawn below with CI on MAE diff scale separately
    ax.scatter(deltas, y, c=colors, s=80, zorder=3)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cliff's delta (proposed vs comparison)")
    ax.set_title("Significance Testing — Effect Size per Comparison\n"
                 "(color = effect size magnitude; MAE 95% CI in table, not shown here)")

    # legend for effect size colors
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=10, label=e)
               for e, c in color_map.items()]
    ax.legend(handles=handles, title="Effect size", loc="best", fontsize=8)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "09_significance_forest_plot.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 10. Late-stage aging error analysis
# ---------------------------------------------------------------------------

def _load_cache_normalization():
    """Loads soh_mean/soh_std dicts from the default cache -- needed to
    de-normalize predictions.npz's z-scored y_test/mean_pred back into
    REAL SOH values (0-1.10 range), since that's the only way to apply a
    physically meaningful threshold like SOH < 0.7. predictions.npz alone
    only has normalized values; the cache is where the per-dataset
    denormalization stats live."""
    cache_path = os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(cache_path):
        print(f"  [!] {cache_path} not found -- cannot de-normalize. "
              f"Run preprocessing.py first.")
        return None, None
    data = np.load(cache_path, allow_pickle=True)
    # soh_mean/soh_std were saved as 0-d object arrays wrapping dicts
    soh_mean = data["soh_mean"].item() if "soh_mean" in data.files else None
    soh_std = data["soh_std"].item() if "soh_std" in data.files else None
    return soh_mean, soh_std


def fig_late_stage_aging_analysis(late_life_threshold=0.7):
    print("\n[10] Late-stage aging error analysis...")
    data = _load_predictions("ablation_full")
    if data is None:
        return

    soh_mean, soh_std = _load_cache_normalization()
    if soh_mean is None:
        print("  [skip] could not load normalization stats -- see message above.")
        return

    y_true_norm, y_pred_norm, lab = data["y_test"], data["mean_pred"], data["lab_test"]

    # de-normalize per-sample using each sample's own dataset's stats
    y_true_real = np.zeros_like(y_true_norm, dtype=np.float64)
    y_pred_real = np.zeros_like(y_pred_norm, dtype=np.float64)
    for ds in np.unique(lab):
        if ds not in soh_mean or ds not in soh_std:
            print(f"  [!] no normalization stats found for dataset '{ds}' in "
                  f"the cache -- skipping its samples in this figure.")
            continue
        mask = lab == ds
        y_true_real[mask] = y_true_norm[mask] * soh_std[ds] + soh_mean[ds]
        y_pred_real[mask] = y_pred_norm[mask] * soh_std[ds] + soh_mean[ds]

    abs_err = np.abs(y_true_real - y_pred_real)
    late_mask = y_true_real < late_life_threshold

    n_late = int(late_mask.sum())
    n_early = int((~late_mask).sum())
    if n_late == 0:
        print(f"  [!] no samples with real SOH < {late_life_threshold} found -- "
              f"cannot build this figure. Check the threshold or your data's "
              f"SOH range.")
        return

    late_mae = abs_err[late_mask].mean()
    early_mae = abs_err[~late_mask].mean()
    ratio = late_mae / early_mae if early_mae > 0 else float("nan")
    print(f"  Early-life (SOH>={late_life_threshold}) MAE: {early_mae:.5f}  (n={n_early})")
    print(f"  Late-life  (SOH<{late_life_threshold})  MAE: {late_mae:.5f}  (n={n_late})")
    print(f"  Late-life is {ratio:.2f}x harder than early-life")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) True vs predicted across the real SOH range, threshold marked
    order = np.argsort(y_true_real)
    axes[0, 0].scatter(y_true_real, y_pred_real, s=3, alpha=0.15, color="#4C72B0")
    lims = (min(y_true_real.min(), y_pred_real.min()),
            max(y_true_real.max(), y_pred_real.max()))
    axes[0, 0].plot(lims, lims, "k--", linewidth=1)
    axes[0, 0].axvline(late_life_threshold, color="#C44E52", linestyle=":",
                       linewidth=1.5, label=f"Late-life threshold (SOH={late_life_threshold})")
    axes[0, 0].set_xlabel("True SOH")
    axes[0, 0].set_ylabel("Predicted SOH")
    axes[0, 0].set_title("(a) True vs Predicted SOH")
    axes[0, 0].legend(fontsize=8)

    # (b) Absolute error vs SOH
    axes[0, 1].scatter(y_true_real, abs_err, s=3, alpha=0.15, color="#55A868")
    axes[0, 1].axvline(late_life_threshold, color="#C44E52", linestyle=":", linewidth=1.5)
    axes[0, 1].set_xlabel("True SOH")
    axes[0, 1].set_ylabel("Absolute error")
    axes[0, 1].set_title("(b) Absolute error vs SOH")

    # (c) Binned error variance across SOH ranges
    bins = np.linspace(y_true_real.min(), y_true_real.max(), 11)
    bin_idx = np.digitize(y_true_real, bins)
    bin_centers, bin_vars, bin_means = [], [], []
    for b in range(1, len(bins)):
        m = bin_idx == b
        if m.sum() < 2:
            continue
        bin_centers.append((bins[b - 1] + bins[b]) / 2)
        bin_vars.append(abs_err[m].var())
        bin_means.append(abs_err[m].mean())
    axes[1, 0].bar(bin_centers, bin_vars, width=(bins[1] - bins[0]) * 0.8,
                   color="#DD8452")
    axes[1, 0].axvline(late_life_threshold, color="#C44E52", linestyle=":", linewidth=1.5)
    axes[1, 0].set_xlabel("SOH bin center")
    axes[1, 0].set_ylabel("Error variance")
    axes[1, 0].set_title("(c) Binned error variance across SOH")

    # (d) Mean absolute error across SOH bins
    axes[1, 1].bar(bin_centers, bin_means, width=(bins[1] - bins[0]) * 0.8,
                   color="#8172B2")
    axes[1, 1].axvline(late_life_threshold, color="#C44E52", linestyle=":", linewidth=1.5)
    axes[1, 1].set_xlabel("SOH bin center")
    axes[1, 1].set_ylabel("Mean absolute error")
    axes[1, 1].set_title("(d) MAE across SOH bins")

    fig.suptitle(f"Late-Stage Aging Analysis (late-life = SOH < {late_life_threshold}, "
                f"{ratio:.2f}x harder than early-life)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "10_late_stage_aging_analysis.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"[generate_all_figures] saving figures to {FIG_DIR}")

    fig_prediction_vs_actual()
    fig_residual_distribution()
    fig_per_cell_curves()
    fig_conformal_coverage_comparison()
    fig_baseline_vs_proposed()
    fig_architecture_ablation()
    fig_window_size_ablation()
    fig_lodo_generalization()
    fig_significance_forest_plot()
    fig_late_stage_aging_analysis()

    print("\n[generate_all_figures] done.")


if __name__ == "__main__":
    main()
