"""
Generates publication-quality figures from the results already saved in
outputs/ by train.py, baselines.py, leave_one_out.py, and experiments.py.

Run this AFTER experiments.py (or at least train.py) has produced result
files. It reads the saved .json and _predictions.npz files — it does NOT
retrain anything, so it's fast.

Figures produced (saved to outputs/figures/):
  1. prediction_vs_actual.png   — predicted vs true SOH scatter, colored by dataset
  2. per_dataset_mae.png         — bar chart of MAE per dataset, proposed vs baselines
  3. conformal_coverage.png      — pooled vs groupwise vs normalized_groupwise coverage per dataset vs 90% target
  4. ablation_mae.png            — MAE across architecture variants
  5. window_size.png             — MAE vs window size
  6. lodo_degradation.png        — LODO MAE per held-out dataset (log scale)
  7. error_distribution.png      — histogram of prediction errors
  8. per_cell_curves.png         — predicted vs true SOH per dataset (ordered test windows)
  9. finetune_comparison.png     — cold vs fine-tuned MAE per held-out dataset

Every figure is built ONLY from saved result files — no numbers are typed in
by hand, so the figures can never silently disagree with your tables.

FIXED (this version): DATASET_ORDER/DATASET_COLORS were hardcoded to the
original 4-dataset project (MIT, NASA, CALCE, Oxford). After the expansion
to 7 datasets (adding BIT, XJTU, RWTH), every figure function that iterates
over DATASET_ORDER silently dropped the 3 new datasets, since these
functions only look up names found in this list rather than whatever's
actually present in the saved data — this is why per_dataset_mae.png,
conformal_coverage.png, prediction_vs_actual.png, and per_cell_curves.png
kept regenerating as 4-dataset figures even after the underlying result
files already had all 7. DATASET_ORDER/DATASET_COLORS now cover all 7, and
plot_conformal_coverage's bar width/offset logic is generalized to N
calibration modes instead of being hardcoded for exactly 2 (pooled vs
groupwise) — it silently misaligned once normalized_groupwise was added as
a third mode.

Run:
    python plots.py
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt

import config

FIG_DIR = os.path.join(config.OUTPUT_DIR, "figures")
# All 7 datasets (order roughly descending cell count: MIT=139, BIT=72,
# XJTU=55, RWTH=48, CALCE=10, Oxford=8, NASA=7). This list was previously
# left over from the original 4-dataset project (MIT, NASA, CALCE, Oxford)
# and silently dropped BIT/XJTU/RWTH from every figure that iterates over
# it — even though the underlying result files already had all 7 datasets
# — since these plotting functions only ever look up names found in this
# list, not whatever's actually present in the data.
DATASET_ORDER = ["MIT", "BIT", "XJTU", "RWTH", "CALCE", "Oxford", "NASA"]
DATASET_COLORS = {
    "MIT": "#2ca02c",
    "BIT": "#1f77b4",
    "XJTU": "#e377c2",
    "RWTH": "#8c564b",
    "CALCE": "#ff7f0e",
    "Oxford": "#9467bd",
    "NASA": "#d62728",
}


def _load_json(name):
    path = os.path.join(config.OUTPUT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _find_current_predictions():
    """Prefers conformal_normalized_groupwise or ablation_full over the
    generic 'best_' file, since 'best_predictions.npz' is only refreshed by
    running train.py directly — if experiments.py has since been re-run on
    a different dataset composition, 'best_' can be silently stale while
    still looking like a valid, current file."""
    npz_candidates = glob.glob(os.path.join(config.OUTPUT_DIR, "*predictions.npz"))
    for preferred in ["conformal_normalized_groupwise", "ablation_full"]:
        match = [p for p in npz_candidates if preferred in os.path.basename(p)]
        if match:
            return match[0]
    best = [p for p in npz_candidates if os.path.basename(p).startswith("best_")]
    if best:
        print("  [!] WARNING: falling back to 'best_predictions.npz' — verify this "
              "isn't stale from before a dataset composition change.")
        return best[0]
    return npz_candidates[0] if npz_candidates else None


def plot_prediction_vs_actual():
    path = _find_current_predictions()
    if path is None:
        print("  [skip] no *predictions.npz found — run train.py first")
        return
    data = np.load(path, allow_pickle=True)
    y_true, y_pred, lab = data["y_test"], data["mean_pred"], data["lab_test"]

    fig, ax = plt.subplots(figsize=(6, 6))
    for ds in DATASET_ORDER:
        mask = lab == ds
        if mask.any():
            ax.scatter(y_true[mask], y_pred[mask], s=4, alpha=0.3,
                       label=ds, color=DATASET_COLORS.get(ds))
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1, label="perfect")
    ax.set_xlabel("True SOH (normalized)")
    ax.set_ylabel("Predicted SOH (normalized)")
    ax.set_title("Predicted vs. true SOH by dataset")
    ax.legend(markerscale=3)
    _save(fig, "prediction_vs_actual.png")


def plot_per_dataset_mae():
    summary = _load_json("experiment_summary.json")
    if summary is None:
        print("  [skip] no experiment_summary.json — run experiments.py first")
        return

    proposed = next((r for r in summary.get("ablations", []) if r["model_variant"] == "full"), None)
    baselines = {r["run_name"]: r for r in summary.get("baselines", [])}
    if proposed is None:
        print("  [skip] no full-model result in summary")
        return

    models = {"Proposed": proposed.get("per_dataset_mae", {})}
    for bname, bres in baselines.items():
        models[bname.replace("baseline_", "").upper()] = bres.get("per_dataset_mae", {})

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DATASET_ORDER))
    width = 0.8 / max(1, len(models))
    for i, (mname, per_ds) in enumerate(models.items()):
        vals = [per_ds.get(ds, np.nan) for ds in DATASET_ORDER]
        ax.bar(x + i * width, vals, width, label=mname)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Test MAE (normalized SOH)")
    ax.set_title("Per-dataset MAE: proposed model vs. baselines")
    ax.legend()
    _save(fig, "per_dataset_mae.png")


def plot_conformal_coverage():
    summary = _load_json("experiment_summary.json")
    if summary is None:
        print("  [skip] no experiment_summary.json")
        return
    conf = summary.get("conformal_ablation", [])
    if not conf:
        print("  [skip] no conformal_ablation in summary")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DATASET_ORDER))
    width = 0.8 / max(1, len(conf))
    for i, r in enumerate(conf):
        cov = r.get("per_dataset_coverage", {})
        vals = [cov.get(ds, np.nan) * 100 for ds in DATASET_ORDER]
        ax.bar(x + i * width, vals, width, label=f"{r['conformal_mode']}")
    target = (1 - config.CONFORMAL_ALPHA) * 100
    ax.axhline(target, color="k", ls="--", lw=1, label=f"{target:.0f}% target")
    ax.set_xticks(x + width * (len(conf) - 1) / 2)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Conformal coverage: pooled vs. groupwise vs. normalized groupwise\n(closer to target line = better calibrated)")
    ax.legend()
    _save(fig, "conformal_coverage.png")


def plot_ablation_mae():
    summary = _load_json("experiment_summary.json")
    if summary is None:
        print("  [skip] no experiment_summary.json")
        return
    abl = summary.get("ablations", [])
    if not abl:
        print("  [skip] no ablations in summary")
        return
    names = [r["model_variant"] for r in abl]
    maes = [r["test_mae"] for r in abl]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, maes, color="#4c72b0")
    ax.set_ylabel("Test MAE (normalized SOH)")
    ax.set_title("Architecture ablation")
    for b, m in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.4f}", ha="center", va="bottom", fontsize=9)
    _save(fig, "ablation_mae.png")


def plot_window_size():
    summary = _load_json("experiment_summary.json")
    if summary is None:
        print("  [skip] no experiment_summary.json")
        return
    ws = summary.get("window_size", [])
    if not ws:
        print("  [skip] no window_size in summary")
        return
    windows = [int(r["run_name"].replace("window_", "")) for r in ws]
    maes = [r["test_mae"] for r in ws]
    order = np.argsort(windows)
    windows = np.array(windows)[order]
    maes = np.array(maes)[order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(windows, maes, "o-", color="#4c72b0")
    ax.set_xlabel("Input window (cycles)")
    ax.set_ylabel("Test MAE (normalized SOH)")
    ax.set_title("Effect of input window size")
    _save(fig, "window_size.png")


def plot_lodo():
    summary = _load_json("experiment_summary.json")
    if summary is None:
        print("  [skip] no experiment_summary.json")
        return
    lodo = summary.get("lodo", [])
    if not lodo:
        print("  [skip] no lodo in summary")
        return
    names = [r["held_out_dataset"] for r in lodo]
    maes = [r["test_mae"] for r in lodo]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, maes, color="#c44e52")
    ax.set_ylabel("LODO Test MAE (normalized SOH, log scale)")
    ax.set_yscale("log")
    ax.set_title("Leave-one-dataset-out generalization\n(higher = worse generalization to unseen dataset)")
    for b, m in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    _save(fig, "lodo_degradation.png")


def plot_error_distribution():
    """Histogram of prediction errors (predicted - true), with the fraction
    within a tolerance band annotated — mirrors the error-distribution figure
    common in SOH papers."""
    path = _find_current_predictions()
    if path is None:
        print("  [skip] no *predictions.npz found — run train.py first")
        return
    data = np.load(path, allow_pickle=True)
    errors = data["mean_pred"] - data["y_test"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(errors, bins=60, color="#4c72b0", alpha=0.8)
    mu, sigma = float(np.mean(errors)), float(np.std(errors))
    tol = 0.05  # normalized-SOH tolerance band for the "within" annotation
    within = float(np.mean(np.abs(errors) <= tol)) * 100
    ax.axvline(mu, color="k", ls="--", lw=1, label=f"mean={mu:.4f}")
    ax.axvline(tol, color="r", ls=":", lw=1)
    ax.axvline(-tol, color="r", ls=":", lw=1, label=f"±{tol} band ({within:.1f}% inside)")
    ax.set_xlabel("Prediction error (predicted − true, normalized SOH)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Error distribution (μ={mu:.4f}, σ={sigma:.4f})")
    ax.legend()
    _save(fig, "error_distribution.png")


def plot_per_cell_curves():
    """Predicted vs true SOH across the test set, plotted per dataset as an
    ordered sequence — a visual sanity check that the model tracks the
    degradation trend, not just minimizes average error."""
    path = _find_current_predictions()
    if path is None:
        print("  [skip] no *predictions.npz found — run train.py first")
        return
    data = np.load(path, allow_pickle=True)
    y_true, y_pred, lab = data["y_test"], data["mean_pred"], data["lab_test"]

    present = [ds for ds in DATASET_ORDER if (lab == ds).any()]
    n = len(present)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.6 * n), squeeze=False)
    for i, ds in enumerate(present):
        ax = axes[i][0]
        mask = lab == ds
        yt, yp = y_true[mask], y_pred[mask]
        idx = np.arange(len(yt))
        ax.plot(idx, yt, color="k", lw=1, label="true")
        ax.plot(idx, yp, color=DATASET_COLORS.get(ds), lw=1, alpha=0.8, label="predicted")
        ax.set_title(f"{ds} (test windows, ordered)")
        ax.set_ylabel("SOH (norm.)")
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1][0].set_xlabel("Test window index")
    fig.tight_layout()
    _save(fig, "per_cell_curves.png")


def plot_finetune():
    """Cold (LODO, no fine-tuning) vs fine-tuned MAE per held-out dataset —
    visualizes whether a few cells of fine-tuning recover the generalization
    failure. Reads finetune_*_results.json; skips silently if not run yet."""
    ft_files = glob.glob(os.path.join(config.OUTPUT_DIR, "finetune_*_results.json"))
    if not ft_files:
        print("  [skip] no finetune_*_results.json — run finetune.py first")
        return
    results = []
    for f in ft_files:
        with open(f) as fh:
            results.append(json.load(fh))
    results.sort(key=lambda r: r["held_out_dataset"])

    names = [r["held_out_dataset"] for r in results]
    cold = [r["cold_mae"] for r in results]
    warm = [r["finetuned_mae"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, cold, width, label="cold (no fine-tuning)", color="#c44e52")
    ax.bar(x + width / 2, warm, width, label="after fine-tuning", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Held-out MAE (normalized SOH, log scale)")
    ax.set_yscale("log")
    ax.set_title("Fine-tuning vs cold generalization on unseen datasets")
    ax.legend()
    _save(fig, "finetune_comparison.png")


def run_all():
    print("[plots] generating figures from saved results...")
    plot_prediction_vs_actual()
    plot_per_dataset_mae()
    plot_conformal_coverage()
    plot_ablation_mae()
    plot_window_size()
    plot_lodo()
    plot_error_distribution()
    plot_per_cell_curves()
    plot_finetune()
    print(f"[plots] done — figures in {FIG_DIR}")


if __name__ == "__main__":
    run_all()