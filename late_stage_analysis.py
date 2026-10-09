"""
Late-stage aging analysis: does prediction error/uncertainty behave worse
in the RUL-critical region (low SOH, near end-of-life)? This is the region
where getting SOH estimation right matters most for real decisions (replace
the battery now vs. keep using it), so error behavior there deserves its
own dedicated analysis rather than being averaged into one overall MAE.

IMPORTANT: your saved predictions are NORMALIZED (per-dataset z-scored) SOH
values, not raw 0-1 SOH — a normalized value of e.g. -0.5 means nothing
comparable across datasets with different means/stds. This script
DENORMALIZES back to raw SOH using the per-dataset mean/std saved in
cache/processed.npz before binning by true SOH, so "SOH < 0.7" is
meaningful and comparable across all 7 datasets.

Produces:
  1. True vs. predicted SOH scatter, colored by dataset, with the 0.7
     late-life threshold marked
  2. |error| vs. true SOH scatter — visual check for whether dispersion
     increases at low SOH
  3. Binned MAE and binned error variance across SOH ranges
  4. Binned CONFORMAL INTERVAL WIDTH across SOH ranges — a genuinely novel
     check specific to this project: does the calibrated interval actually
     widen appropriately near end-of-life (where it should, if the model's
     uncertainty is meaningful), or does it stay flat regardless of how
     degraded the cell is?

Run:
    python late_stage_analysis.py
Requires: cache/processed.npz (for soh_mean/soh_std) and a
conformal_normalized_groupwise (or 'best') predictions.npz file.
"""

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

FIG_DIR = os.path.join(config.OUTPUT_DIR, "figures")
LATE_LIFE_THRESHOLD = 0.7  # raw SOH threshold, matches common EOL convention
SOH_BINS = np.arange(0.5, 1.05, 0.05)  # bin edges for raw SOH


def _find_predictions(name_substring):
    candidates = glob.glob(os.path.join(config.OUTPUT_DIR, f"*{name_substring}*predictions.npz"))
    return candidates[0] if candidates else None


def denormalize(values, labels, soh_mean, soh_std):
    """values, labels are per-sample arrays; soh_mean/soh_std are dicts
    keyed by dataset name (saved in cache/processed.npz)."""
    raw = np.empty_like(values, dtype=float)
    for ds in np.unique(labels):
        mask = labels == ds
        mean = float(soh_mean[str(ds)]) if str(ds) in soh_mean else float(soh_mean[ds])
        std = float(soh_std[str(ds)]) if str(ds) in soh_std else float(soh_std[ds])
        raw[mask] = values[mask] * std + mean
    return raw


def analyze(cache_path=None, run_name_substring="conformal_normalized_groupwise"):
    cache_path = cache_path or os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(cache_path):
        print(f"[late_stage] cache not found: {cache_path} — run preprocessing.py first")
        return None

    pred_path = _find_predictions(run_name_substring)
    if pred_path is None:
        pred_path = _find_predictions("best")
        print(f"[late_stage] '{run_name_substring}' not found, falling back to 'best'")
    if pred_path is None:
        print("[late_stage] no predictions file found — run train.py/experiments.py first")
        return None

    cache = np.load(cache_path, allow_pickle=True)
    preds = np.load(pred_path, allow_pickle=True)

    soh_mean = cache["soh_mean"].item() if hasattr(cache["soh_mean"], "item") else dict(cache["soh_mean"])
    soh_std = cache["soh_std"].item() if hasattr(cache["soh_std"], "item") else dict(cache["soh_std"])

    y_norm, pred_norm = preds["y_test"], preds["mean_pred"]
    lab = preds["lab_test"]
    has_interval = "lower" in preds and "upper" in preds

    y_raw = denormalize(y_norm, lab, soh_mean, soh_std)
    pred_raw = denormalize(pred_norm, lab, soh_mean, soh_std)
    abs_err_raw = np.abs(y_raw - pred_raw)

    if has_interval:
        lower_norm, upper_norm = preds["lower"], preds["upper"]
        # width scales linearly under this affine transform, so denormalize
        # by multiplying by std only (mean shift cancels in the subtraction)
        width_raw = np.empty_like(lower_norm, dtype=float)
        for ds in np.unique(lab):
            mask = lab == ds
            std = float(soh_std[str(ds)]) if str(ds) in soh_std else float(soh_std[ds])
            width_raw[mask] = (upper_norm[mask] - lower_norm[mask]) * std

    print(f"[late_stage] analyzing {os.path.basename(pred_path)}")
    print(f"  raw SOH range in test set: [{y_raw.min():.3f}, {y_raw.max():.3f}]")

    below_thresh = y_raw < LATE_LIFE_THRESHOLD
    print(f"  samples below SOH={LATE_LIFE_THRESHOLD}: {below_thresh.sum()} "
          f"({100*below_thresh.mean():.1f}% of test set)")
    if below_thresh.sum() < 10:
        print(f"  [!] very few samples below threshold — late-life statistics "
              f"below will be noisy; report with appropriate caution.")

    mae_above = float(np.mean(abs_err_raw[~below_thresh])) if (~below_thresh).any() else float("nan")
    mae_below = float(np.mean(abs_err_raw[below_thresh])) if below_thresh.any() else float("nan")
    print(f"  MAE (raw SOH) above threshold: {mae_above:.4f}")
    print(f"  MAE (raw SOH) below threshold: {mae_below:.4f}")
    if not np.isnan(mae_below) and not np.isnan(mae_above):
        ratio = mae_below / mae_above if mae_above > 0 else float("nan")
        print(f"  ratio (below/above): {ratio:.2f}x "
              f"{'— errors ARE worse in late-life region' if ratio > 1.1 else '— no clear late-life degradation in error'}")

    # ---- binned statistics ----
    bin_centers, bin_mae, bin_var, bin_width, bin_n = [], [], [], [], []
    for i in range(len(SOH_BINS) - 1):
        lo, hi = SOH_BINS[i], SOH_BINS[i + 1]
        mask = (y_raw >= lo) & (y_raw < hi)
        n = int(mask.sum())
        bin_n.append(n)
        bin_centers.append((lo + hi) / 2)
        if n >= 5:
            bin_mae.append(float(np.mean(abs_err_raw[mask])))
            bin_var.append(float(np.var(abs_err_raw[mask])))
            bin_width.append(float(np.mean(width_raw[mask])) if has_interval else np.nan)
        else:
            bin_mae.append(np.nan)
            bin_var.append(np.nan)
            bin_width.append(np.nan)

    print("\n  Binned statistics (raw SOH):")
    for c, m, v, w, n in zip(bin_centers, bin_mae, bin_var, bin_width, bin_n):
        print(f"    SOH~{c:.2f}: n={n:4d}  MAE={m:.4f}  var={v:.5f}  "
              f"interval_width={w:.4f}" if not np.isnan(m) else f"    SOH~{c:.2f}: n={n} (too few)")

    # ---- figures ----
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    order = np.argsort(y_raw)
    ax.plot(np.arange(len(y_raw)), y_raw[order], "-", color="k", lw=1, label="true SOH (sorted)")
    ax.scatter(np.arange(len(y_raw)), pred_raw[order], s=2, alpha=0.3, color="#4c72b0", label="predicted")
    ax.axhline(LATE_LIFE_THRESHOLD, color="r", ls="--", lw=1, label=f"late-life threshold ({LATE_LIFE_THRESHOLD})")
    ax.set_xlabel("Test sample (sorted by true SOH)")
    ax.set_ylabel("SOH (raw)")
    ax.set_title("True vs. predicted SOH")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.scatter(y_raw, abs_err_raw, s=3, alpha=0.2, color="#c44e52")
    ax.axvline(LATE_LIFE_THRESHOLD, color="r", ls="--", lw=1)
    ax.set_xlabel("True SOH (raw)")
    ax.set_ylabel("|prediction error| (raw)")
    ax.set_title("Absolute error vs. true SOH")

    ax = axes[1, 0]
    ax.plot(bin_centers, bin_var, "o-", color="#55a868")
    ax.axvline(LATE_LIFE_THRESHOLD, color="r", ls="--", lw=1)
    ax.set_xlabel("SOH bin center")
    ax.set_ylabel("Error variance")
    ax.set_title("Binned error variance across SOH")

    ax = axes[1, 1]
    ax.plot(bin_centers, bin_mae, "o-", color="#4c72b0", label="MAE")
    if has_interval:
        ax2 = ax.twinx()
        ax2.plot(bin_centers, bin_width, "s--", color="#dd8452", label="interval width")
        ax2.set_ylabel("Mean conformal interval width", color="#dd8452")
    ax.axvline(LATE_LIFE_THRESHOLD, color="r", ls="--", lw=1)
    ax.set_xlabel("SOH bin center")
    ax.set_ylabel("MAE", color="#4c72b0")
    ax.set_title("Binned MAE (and interval width) across SOH")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "late_stage_aging_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {os.path.join(FIG_DIR, 'late_stage_aging_analysis.png')}")

    if has_interval:
        # does interval width correlate with proximity to end-of-life?
        corr = float(np.corrcoef(1 - y_raw, width_raw)[0, 1])  # 1-y_raw: higher = more degraded
        print(f"\n  correlation(degradation level, interval width) = {corr:.4f}")
        if corr > 0.15:
            print("  -> intervals meaningfully WIDEN as cells approach end-of-life — "
                  "the calibrated uncertainty tracks degradation risk appropriately.")
        else:
            print("  -> little relationship between degradation level and interval width — "
                  "intervals do NOT widen appropriately near end-of-life; report this honestly "
                  "as a limitation of the current calibration approach.")
    else:
        corr = None

    results = {
        "late_life_threshold": LATE_LIFE_THRESHOLD,
        "n_below_threshold": int(below_thresh.sum()),
        "mae_above_threshold": mae_above,
        "mae_below_threshold": mae_below,
        "degradation_vs_interval_width_correlation": corr,
        "binned": {
            "soh_bin_centers": bin_centers,
            "mae": bin_mae,
            "variance": bin_var,
            "interval_width": bin_width,
            "n_samples": bin_n,
        },
    }
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "late_stage_analysis.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[late_stage] saved to {config.OUTPUT_DIR}/late_stage_analysis.json")
    return results


if __name__ == "__main__":
    analyze()
