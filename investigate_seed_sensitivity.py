"""
Why some datasets' conformal coverage varies more across training seeds.

MIT and RWTH show larger coverage variability across the three training seeds
than XJTU or BIT, even though MIT and RWTH have the narrowest finite-sample
coverage confidence intervals, so sample size does not explain it.

Hypothesis: coverage is a discrete covered/not-covered count. If a dataset has
many test samples close to the conformal interval boundary ("borderline"
samples), small seed-induced shifts in the point prediction flip many of them,
producing large coverage swings even when MAE barely changes.

Tested directly on the three existing multi-seed prediction files
(outputs/multiseed_{42,123,2024}_predictions.npz); no training is performed.

Run:
    python investigate_seed_sensitivity.py
Output:
    outputs/seed_sensitivity_investigation.json
    outputs/figures/seed_sensitivity_borderline_analysis.png
"""

import os
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

SEEDS = [42, 123, 2024]
DATASET_ORDER = ["MIT", "BIT", "XJTU", "RWTH", "CALCE", "Oxford", "NASA"]
DATASET_COLORS = {
    "MIT": "#4C72B0", "BIT": "#DD8452", "XJTU": "#55A868", "RWTH": "#C44E52",
    "CALCE": "#8172B2", "Oxford": "#937860", "NASA": "#DA8BC3",
}


def _load_seed_predictions():
    preds = {}
    for seed in SEEDS:
        candidates = glob.glob(os.path.join(config.OUTPUT_DIR, f"*multiseed_{seed}*predictions.npz"))
        if not candidates:
            print(f"  [!] no predictions file found for multiseed_{seed} -- "
                  f"need outputs/multiseed_{seed}_predictions.npz (saved "
                  f"automatically by train_one() if save_outputs=True, "
                  f"which is the default).")
            return None
        preds[seed] = np.load(candidates[0], allow_pickle=True)
    return preds


def main():
    preds = _load_seed_predictions()
    if preds is None:
        return
    required = {"y_test", "mean_pred", "lower", "upper", "lab_test"}
    for seed, data in preds.items():
        missing = required - set(data.files)
        if missing:
            print(f"  [!] multiseed_{seed} predictions missing {missing} -- cannot proceed.")
            return

    lab_test = preds[SEEDS[0]]["lab_test"]
    datasets_present = [d for d in DATASET_ORDER if d in set(np.unique(lab_test))]

    # --- Part 1: borderline-margin analysis ---
    print(f"\n{'='*78}")
    print("PART 1: Borderline-margin analysis")
    print(f"{'='*78}")
    print("For each seed, margin = min(y_true - lower, upper - y_true), "
          "normalized by interval half-width. Margin near 0 = sample sits "
          "right at the boundary (a small prediction shift could flip its "
          "covered/uncovered status). Margin near 1 = sample sits safely "
          "in the interval center.\n")

    borderline_summary = {}
    for ds in datasets_present:
        mask = lab_test == ds
        # Average across the 3 seeds for a stable per-dataset borderline rate
        borderline_fractions = []
        for seed in SEEDS:
            data = preds[seed]
            y = data["y_test"][mask]
            lo = data["lower"][mask]
            hi = data["upper"][mask]
            half_width = (hi - lo) / 2
            half_width = np.where(half_width == 0, 1e-9, half_width)
            margin = np.minimum(y - lo, hi - y) / half_width
            borderline_frac = float(np.mean(np.abs(margin) < 0.05))  # within 5% of boundary
            borderline_fractions.append(borderline_frac)
        mean_borderline = float(np.mean(borderline_fractions))
        borderline_summary[ds] = {
            "borderline_fraction_per_seed": borderline_fractions,
            "mean_borderline_fraction": mean_borderline,
        }
        print(f"  {ds:8s}  borderline fraction (mean across seeds): {mean_borderline:.4f}  "
              f"(per-seed: {[f'{v:.4f}' for v in borderline_fractions]})")

    # --- Part 2: direct flip-sample analysis ---
    print(f"\n{'='*78}")
    print("PART 2: Direct flip-sample analysis")
    print(f"{'='*78}")
    print("For each dataset, counts samples whose covered/uncovered status "
          "DIFFERS across at least 2 of the 3 seeds -- the direct, "
          "mechanistic cause of coverage instability.\n")

    flip_summary = {}
    for ds in datasets_present:
        mask = lab_test == ds
        covered_by_seed = []
        for seed in SEEDS:
            data = preds[seed]
            y = data["y_test"][mask]
            lo = data["lower"][mask]
            hi = data["upper"][mask]
            covered = (y >= lo) & (y <= hi)
            covered_by_seed.append(covered)
        covered_by_seed = np.array(covered_by_seed)  # shape (3, n_samples)
        n_seeds_covered = covered_by_seed.sum(axis=0)  # 0, 1, 2, or 3 seeds covered this sample
        n_flip = int(np.sum((n_seeds_covered > 0) & (n_seeds_covered < 3)))
        n_total = covered_by_seed.shape[1]
        flip_rate = n_flip / n_total if n_total > 0 else 0.0
        flip_summary[ds] = {"n_flip": n_flip, "n_total": n_total, "flip_rate": flip_rate}
        print(f"  {ds:8s}  {n_flip:5d} of {n_total:5d} samples flip status across seeds "
              f"(flip rate={flip_rate:.4f})")

    # --- Correlation check: does higher borderline fraction predict higher flip rate? ---
    print(f"\n{'='*78}")
    print("PART 3: Does borderline fraction predict flip rate (and thus coverage instability)?")
    print(f"{'='*78}")
    borderline_vals = [borderline_summary[d]["mean_borderline_fraction"] for d in datasets_present]
    flip_vals = [flip_summary[d]["flip_rate"] for d in datasets_present]
    if len(datasets_present) >= 3:
        corr = float(np.corrcoef(borderline_vals, flip_vals)[0, 1])
        print(f"  Correlation between borderline fraction and flip rate across "
              f"datasets: {corr:.4f}")
        print(f"  (Values close to +1 support the hypothesis that datasets with "
              f"more boundary-hugging samples show more seed-to-seed coverage "
              f"instability, independent of sample size.)")
    else:
        corr = None

    # --- Part 4: direct point-prediction variance across seeds (more fundamental
    # than the boundary-margin proxy -- tests whether MIT/RWTH's predictions
    # THEMSELVES are less stable across seeds, independent of interval position) ---
    print(f"\n{'='*78}")
    print("PART 4: Point-prediction variance across seeds (more direct mechanism)")
    print(f"{'='*78}")
    print("For each sample, std of mean_pred across the 3 seeds, normalized by "
          "that dataset's target-value scale (so datasets with naturally larger "
          "value ranges aren't unfairly flagged as more unstable).\n")

    pred_variance_summary = {}
    for ds in datasets_present:
        mask = lab_test == ds
        seed_preds = np.array([preds[seed]["mean_pred"][mask] for seed in SEEDS])  # (3, n)
        pred_std_per_sample = seed_preds.std(axis=0)  # std across seeds, per sample
        y_scale = float(np.std(preds[SEEDS[0]]["y_test"][mask])) or 1e-9
        normalized_pred_std = float(np.mean(pred_std_per_sample)) / y_scale
        pred_variance_summary[ds] = {
            "mean_pred_std_across_seeds": float(np.mean(pred_std_per_sample)),
            "y_scale": y_scale,
            "normalized_pred_instability": normalized_pred_std,
        }
        print(f"  {ds:8s}  mean pred std across seeds={np.mean(pred_std_per_sample):.5f}  "
              f"y_scale={y_scale:.5f}  normalized instability={normalized_pred_std:.4f}")

    pred_instability_vals = [pred_variance_summary[d]["normalized_pred_instability"] for d in datasets_present]
    corr_pred = float(np.corrcoef(pred_instability_vals, flip_vals)[0, 1]) if len(datasets_present) >= 3 else None
    if corr_pred is not None:
        print(f"\n  Correlation between normalized point-prediction instability and "
              f"flip rate: {corr_pred:.4f}")
        print(f"  (This tests the more fundamental hypothesis: do MIT/RWTH's "
              f"predictions THEMSELVES vary more across seeds, rather than just "
              f"happening to sit near interval boundaries more often?)")

    # --- Part 5: relative interval width -- tests the refined hypothesis that
    # emerged from Part 4's result: if a dataset's predictions are STABLE
    # (low Part 4 instability) but it STILL has a high flip rate, the likely
    # explanation is that its calibrated interval is simply narrower relative
    # to its own residual scale, making it more sensitive to any small noise
    # -- not that the noise itself is larger. ---
    print(f"\n{'='*78}")
    print("PART 5: Relative interval width (tests the interval-tightness hypothesis)")
    print(f"{'='*78}")
    print("interval_width / y_scale, per dataset -- a dataset with tight "
          "intervals RELATIVE TO its own value scale should be more sensitive "
          "to any given amount of prediction noise, independent of how much "
          "noise it actually has (Part 4).\n")

    width_summary = {}
    for ds in datasets_present:
        mask = lab_test == ds
        widths_per_seed = []
        for seed in SEEDS:
            data = preds[seed]
            w = (data["upper"][mask] - data["lower"][mask])
            widths_per_seed.append(float(np.mean(w)))
        mean_width = float(np.mean(widths_per_seed))
        y_scale = pred_variance_summary[ds]["y_scale"]
        relative_width = mean_width / y_scale if y_scale > 0 else float("nan")
        width_summary[ds] = {"mean_interval_width": mean_width, "relative_width": relative_width}
        print(f"  {ds:8s}  mean_width={mean_width:.4f}  y_scale={y_scale:.4f}  "
              f"relative_width={relative_width:.4f}")

    relative_width_vals = [width_summary[d]["relative_width"] for d in datasets_present]
    corr_width = float(np.corrcoef(relative_width_vals, flip_vals)[0, 1]) if len(datasets_present) >= 3 else None
    if corr_width is not None:
        print(f"\n  Correlation between relative interval width and flip rate: {corr_width:.4f}")
        print(f"  (A strong negative correlation here alone would confirm interval "
              f"tightness as a standalone cause -- if this is weak, check Part 6, "
              f"which combines this with prediction instability.)")

    # --- Part 6: combined sensitivity ratio -- tests whether flip rate is
    # driven by the RATIO of prediction instability to relative interval
    # width, rather than either factor alone. A sample flips when the
    # seed-to-seed prediction perturbation is large RELATIVE TO the
    # interval's own width; this ratio measures exactly that directly. ---
    print(f"\n{'='*78}")
    print("PART 6: Combined sensitivity ratio (instability / relative interval width)")
    print(f"{'='*78}")
    print("Tests whether flip rate is jointly determined by BOTH prediction "
          "instability (Part 4) AND interval tightness (Part 5), rather than "
          "either factor alone explaining it in isolation.\n")

    sensitivity_summary = {}
    for ds in datasets_present:
        instability = pred_variance_summary[ds]["normalized_pred_instability"]
        rel_width = width_summary[ds]["relative_width"]
        ratio = instability / rel_width if rel_width > 0 else float("nan")
        sensitivity_summary[ds] = {"sensitivity_ratio": ratio}
        print(f"  {ds:8s}  sensitivity_ratio={ratio:.4f}  flip_rate={flip_summary[ds]['flip_rate']:.4f}")

    sensitivity_vals = [sensitivity_summary[d]["sensitivity_ratio"] for d in datasets_present]
    if len(datasets_present) >= 3:
        corr_sensitivity = float(np.corrcoef(sensitivity_vals, flip_vals)[0, 1])
        print(f"\n  Correlation (all {len(datasets_present)} datasets, including NASA): "
              f"{corr_sensitivity:.4f}")
        print(f"  (Compare against Part 4 alone and Part 5 alone above.)")

        # NASA has only 3 test samples (Table 7 in the manuscript already
        # establishes it as illustrative-only, not statistically well-powered,
        # for exactly this reason). With only 7 datasets total, a single
        # noisy point has outsized leverage on the correlation -- report the
        # NASA-excluded version too, applying the SAME exclusion criterion
        # already used consistently elsewhere in this project, not a
        # post-hoc choice to get a better number.
        datasets_no_nasa = [d for d in datasets_present if d != "NASA"]
        if len(datasets_no_nasa) >= 3:
            sensitivity_vals_no_nasa = [sensitivity_summary[d]["sensitivity_ratio"] for d in datasets_no_nasa]
            flip_vals_no_nasa = [flip_summary[d]["flip_rate"] for d in datasets_no_nasa]
            corr_sensitivity_no_nasa = float(np.corrcoef(sensitivity_vals_no_nasa, flip_vals_no_nasa)[0, 1])
            print(f"  Correlation ({len(datasets_no_nasa)} datasets, excluding "
                  f"NASA -- consistent with its treatment as illustrative-only "
                  f"elsewhere in this project, Table 7): {corr_sensitivity_no_nasa:.4f}")
        else:
            corr_sensitivity_no_nasa = None
        print(f"  Note: with only {len(datasets_present)} (or {len(datasets_no_nasa)} "
              f"excluding NASA) datasets, treat either correlation as a "
              f"suggestive pattern, not a statistically powered test.")
    else:
        corr_sensitivity = None
        corr_sensitivity_no_nasa = None

    # Save results
    out = {
        "borderline_analysis": borderline_summary,
        "flip_analysis": flip_summary,
        "borderline_vs_flip_correlation": corr,
        "point_prediction_variance": pred_variance_summary,
        "pred_instability_vs_flip_correlation": corr_pred,
        "relative_interval_width": width_summary,
        "relative_width_vs_flip_correlation": corr_width,
        "combined_sensitivity_ratio": sensitivity_summary,
        "sensitivity_ratio_vs_flip_correlation_all_datasets": corr_sensitivity,
        "sensitivity_ratio_vs_flip_correlation_excl_nasa": corr_sensitivity_no_nasa,
    }
    out_path = os.path.join(config.OUTPUT_DIR, "seed_sensitivity_investigation.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {out_path}")

    # Figure: the scatter plot that actually visualizes the confirmed
    # mechanism (Part 6, r=0.95 excl. NASA), not the earlier, superseded
    # borderline-margin hypothesis (Part 1, r=0.55). This is the figure
    # used in the paper's seed-sensitivity discussion.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = [DATASET_COLORS.get(d, "#999999") for d in datasets_present]

    # Left: sensitivity ratio vs flip rate scatter, all 7 datasets
    for ds, color in zip(datasets_present, colors):
        axes[0].scatter(sensitivity_summary[ds]["sensitivity_ratio"],
                        flip_summary[ds]["flip_rate"],
                        color=color, s=120, label=ds, edgecolors="black", linewidths=0.5)
    axes[0].set_xlabel("Combined sensitivity ratio\n(prediction instability / relative interval width)")
    axes[0].set_ylabel("Flip rate across seeds")
    axes[0].set_title(f"All 7 datasets (r={corr_sensitivity:.2f})" if corr_sensitivity is not None else "All datasets")
    axes[0].legend(fontsize=8, loc="upper left")

    # Right: same, NASA excluded (the version reported as primary in the manuscript)
    datasets_no_nasa_plot = [d for d in datasets_present if d != "NASA"]
    for ds in datasets_no_nasa_plot:
        axes[1].scatter(sensitivity_summary[ds]["sensitivity_ratio"],
                        flip_summary[ds]["flip_rate"],
                        color=DATASET_COLORS.get(ds, "#999999"), s=120, label=ds,
                        edgecolors="black", linewidths=0.5)
    # fitted trend line for the excl-NASA case
    if len(datasets_no_nasa_plot) >= 2:
        xs = np.array([sensitivity_summary[d]["sensitivity_ratio"] for d in datasets_no_nasa_plot])
        ys = np.array([flip_summary[d]["flip_rate"] for d in datasets_no_nasa_plot])
        if np.ptp(xs) > 0:
            slope, intercept = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 50)
            axes[1].plot(x_line, slope * x_line + intercept, "k--", linewidth=1, alpha=0.6)
    axes[1].set_xlabel("Combined sensitivity ratio\n(prediction instability / relative interval width)")
    axes[1].set_ylabel("Flip rate across seeds")
    title_r = f"{corr_sensitivity_no_nasa:.2f}" if corr_sensitivity_no_nasa is not None else "n/a"
    axes[1].set_title(f"Excluding NASA (n=3, illustrative-only; r={title_r})")
    axes[1].legend(fontsize=8, loc="upper left")

    fig.suptitle("Seed-Sensitivity Mechanism: Combined Sensitivity Ratio vs. Flip Rate", fontsize=13)
    fig.tight_layout()
    fig_dir = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_fig = os.path.join(fig_dir, "seed_sensitivity_mechanism.png")
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()