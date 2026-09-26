"""
Why CALCE is the hardest leave-one-dataset-out target.

Compares CALCE with the pooled other six datasets on three measurable axes,
each of which could independently explain poor transfer:

  1. Degradation rate (capacity fade per cycle, per cell).
  2. Curve shape / nonlinearity (knee-point behaviour, i.e. late-life
     acceleration of fade).
  3. SOH range covered (whether CALCE reaches deeper end-of-life SOH than the
     other datasets).

Also runs a two-sample Kolmogorov-Smirnov test between CALCE's SOH
distribution and that of the pooled other datasets.

Run:
    python diagnose_calce_generalization_gap.py
Output:
    outputs/calce_generalization_gap_report.txt
    outputs/figures/calce_generalization_gap.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

import config
import preprocessing

TARGET = "CALCE"
OTHERS = ["NASA", "Oxford", "MIT", "BIT", "XJTU", "RWTH"]
DATASET_COLORS = {
    "MIT": "#4C72B0", "BIT": "#DD8452", "XJTU": "#55A868", "RWTH": "#C44E52",
    "CALCE": "#000000", "Oxford": "#937860", "NASA": "#DA8BC3",
}


def per_cell_stats(df, dataset_name):
    """Returns a list of dicts, one per cell: degradation rate (linear
    slope of SOH vs cycle), a nonlinearity/knee score (R^2 improvement
    of a quadratic fit over a linear fit -- higher means more curved/
    knee-like), min SOH reached, and cycle count."""
    stats = []
    sub = df[df["dataset"] == dataset_name]
    for cell_id, group in sub.groupby("cell_id"):
        g = group.sort_values("cycle")
        cycles = g["cycle"].to_numpy(dtype=float)
        soh = g["soh"].to_numpy(dtype=float)
        if len(cycles) < 5:
            continue

        # linear fit
        lin_coef = np.polyfit(cycles, soh, 1)
        lin_pred = np.polyval(lin_coef, cycles)
        ss_res_lin = np.sum((soh - lin_pred) ** 2)

        # quadratic fit
        quad_coef = np.polyfit(cycles, soh, 2)
        quad_pred = np.polyval(quad_coef, cycles)
        ss_res_quad = np.sum((soh - quad_pred) ** 2)

        ss_tot = np.sum((soh - soh.mean()) ** 2)
        r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0
        r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0
        knee_score = max(0.0, r2_quad - r2_lin)  # how much curvature helps

        stats.append({
            "cell_id": cell_id,
            "degradation_rate": -lin_coef[0],  # SOH loss per cycle, positive = fading
            "knee_score": knee_score,
            "min_soh": soh.min(),
            "max_soh": soh.max(),
            "n_cycles": len(cycles),
        })
    return stats


def main():
    print("Loading raw data for all datasets (this may take a while, "
          "same as preprocessing.py)...")
    raw = preprocessing.load_all_raw()
    df = preprocessing.add_soh(raw)

    calce_stats = per_cell_stats(df, TARGET)
    if not calce_stats:
        print(f"[!] No cells found for {TARGET} -- check the dataset name.")
        return

    others_stats = []
    for ds in OTHERS:
        others_stats += per_cell_stats(df, ds)

    def _vals(stats, key):
        return np.array([s[key] for s in stats])

    report_lines = [f"CALCE vs. pooled other 6 datasets -- generalization gap diagnosis", "=" * 70]

    print(f"\n{'Metric':30s} {'CALCE (n=' + str(len(calce_stats)) + ')':>20s} "
          f"{'Others (n=' + str(len(others_stats)) + ')':>20s}")
    report_lines.append(f"\n{'Metric':30s} {'CALCE':>20s} {'Others (pooled)':>20s}")

    for key, label in [
        ("degradation_rate", "Degradation rate (SOH/cycle)"),
        ("knee_score", "Nonlinearity/knee score"),
        ("min_soh", "Min SOH reached"),
        ("n_cycles", "Cycle count per cell"),
    ]:
        c_vals = _vals(calce_stats, key)
        o_vals = _vals(others_stats, key)
        line = (f"{label:30s} {c_vals.mean():>12.5f} ± {c_vals.std():<6.5f} "
                f"{o_vals.mean():>12.5f} ± {o_vals.std():<6.5f}")
        print(line)
        report_lines.append(line)

        # simple effect-size flag: how many std devs apart are the means?
        pooled_std = np.sqrt((c_vals.std()**2 + o_vals.std()**2) / 2) or 1e-9
        gap = abs(c_vals.mean() - o_vals.mean()) / pooled_std
        flag = ""
        if gap > 1.0:
            flag = f"  <-- LARGE gap ({gap:.2f} pooled std devs apart)"
        elif gap > 0.5:
            flag = f"  <-- moderate gap ({gap:.2f} pooled std devs apart)"
        if flag:
            print(f"  {flag.strip()}")
            report_lines.append(f"  {flag.strip()}")

    # KS test on the raw SOH value distributions (not per-cell aggregates --
    # the full pooled sample of SOH readings)
    calce_soh = df[df["dataset"] == TARGET]["soh"].to_numpy()
    others_soh = df[df["dataset"].isin(OTHERS)]["soh"].to_numpy()
    ks_stat, ks_p = ks_2samp(calce_soh, others_soh)
    print(f"\nKolmogorov-Smirnov test (CALCE SOH distribution vs pooled others'): "
          f"D={ks_stat:.4f}, p={ks_p:.2e}")
    report_lines.append(f"\nKS test: D={ks_stat:.4f}, p={ks_p:.2e}")
    if ks_p < 0.001:
        msg = ("  ==> CALCE's SOH value distribution is statistically "
               "significantly different from the pooled other 6 datasets' "
               "distribution (this alone doesn't say WHY, but confirms a "
               "real distribution shift exists, consistent with LODO's "
               "poor CALCE generalization).")
        print(msg)
        report_lines.append(msg)

    # Figure: overlay degradation curves (a sample) + knee-score comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left: knee score distribution per dataset (boxplot-style via scatter+mean)
    all_datasets = [TARGET] + OTHERS
    knee_by_ds = {TARGET: _vals(calce_stats, "knee_score")}
    for ds in OTHERS:
        knee_by_ds[ds] = _vals(per_cell_stats(df, ds), "knee_score")

    positions = range(len(all_datasets))
    for i, ds in enumerate(all_datasets):
        vals = knee_by_ds[ds]
        axes[0].scatter([i] * len(vals), vals, alpha=0.5,
                        color=DATASET_COLORS.get(ds, "#333333"), s=20)
        axes[0].scatter([i], [vals.mean()], color="red", marker="_", s=300, zorder=5)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(all_datasets, rotation=30, ha="right")
    axes[0].set_ylabel("Knee/nonlinearity score (higher = more curved)")
    axes[0].set_title("Degradation curve nonlinearity by dataset\n(red bar = mean)")

    # right: degradation rate distribution per dataset
    rate_by_ds = {TARGET: _vals(calce_stats, "degradation_rate")}
    for ds in OTHERS:
        rate_by_ds[ds] = _vals(per_cell_stats(df, ds), "degradation_rate")
    for i, ds in enumerate(all_datasets):
        vals = rate_by_ds[ds]
        axes[1].scatter([i] * len(vals), vals, alpha=0.5,
                        color=DATASET_COLORS.get(ds, "#333333"), s=20)
        axes[1].scatter([i], [vals.mean()], color="red", marker="_", s=300, zorder=5)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(all_datasets, rotation=30, ha="right")
    axes[1].set_ylabel("Degradation rate (SOH loss per cycle)")
    axes[1].set_title("Degradation rate by dataset\n(red bar = mean)")

    fig.suptitle("CALCE vs. Other Datasets: Candidate Explanations for Poor LODO Generalization",
                fontsize=12)
    fig.tight_layout()

    fig_dir = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_png = os.path.join(fig_dir, "calce_generalization_gap.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nsaved {out_png}")

    out_txt = os.path.join(config.OUTPUT_DIR, "calce_generalization_gap_report.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(report_lines))
    print(f"saved {out_txt}")

    print("\n" + "=" * 70)
    print("HOW TO USE THIS: whichever metric above shows the LARGEST gap "
          "(flagged '<-- LARGE gap') is your evidence-based hypothesis. "
          "E.g. if knee_score is much higher for CALCE, write something "
          "like: 'CALCE cells exhibit pronounced nonlinear (knee-point) "
          "degradation not well-represented in the other six datasets' "
          "training signal, which is consistent with its poor LODO "
          "transfer (MAE 0.268 vs. 0.082 for the next-hardest dataset).' "
          "Adjust the specific claim to whichever metric actually shows "
          "the gap in YOUR data -- don't assume it's the knee score "
          "without checking the printed numbers above.")


if __name__ == "__main__":
    main()
