"""
Results figures for the manuscript, built from outputs/*.json.

  coverage_comparison           pooled vs. groupwise coverage per dataset (Section 5.1)
  calibration_size_sensitivity  coverage vs. calibration-cell budget (Section 5.3)
  cqr_comparison                CQR vs. groupwise calibration: coverage and width (Section 5.4)
  repeated_split_variability    coverage across cell-level splits (Section 5.6)
  mwhc_sensitivity              MWHC temperature sweep (Appendix B)

Run from the repository root:  python figures/make_results_figures.py [--out DIR]
"""
import numpy as np
import matplotlib.pyplot as plt

from style import (DARK, DATASETS, INK, LIGHT, MID, TEXT_WIDTH, output, output_dir, save, setup,
                   target_line)


def bar_labels(ax, xs, vals, fmt="{:.0f}", dy=1.2, size=6.0):
    for x, v in zip(xs, vals):
        ax.text(x, v + dy, fmt.format(v), ha="center", va="bottom", fontsize=size)


def coverage_comparison(out):
    pooled = output("conformal_pooled_results")["per_dataset_coverage"]
    groupwise = output("conformal_groupwise_results")["per_dataset_coverage"]
    x = np.arange(len(DATASETS))
    w = 0.38
    pv = [pooled[d] * 100 for d in DATASETS]
    gv = [groupwise[d] * 100 for d in DATASETS]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.5))
    ax.bar(x - w / 2, pv, w, facecolor=LIGHT, edgecolor=INK, linewidth=0.6, label="Pooled")
    ax.bar(x + w / 2, gv, w, facecolor=INK, edgecolor=INK, linewidth=0.6, label="Groupwise")
    bar_labels(ax, x - w / 2, pv)
    bar_labels(ax, x + w / 2, gv)
    target_line(ax)
    ax.set_xticks(x, DATASETS)
    ax.set_ylabel("Test coverage (%)")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3)
    save(fig, out, "coverage_comparison")


def calibration_size_sensitivity(out):
    cs = output("calibration_size_sensitivity")["per_dataset"]
    sweep = [d for d in DATASETS if len(cs[d]["per_budget"]) > 1]
    fig, axes = plt.subplots(1, len(sweep), figsize=(TEXT_WIDTH, 1.9), sharey=True)
    for ax, d in zip(axes, sweep):
        pb = cs[d]["per_budget"]
        b = sorted(int(k) for k in pb)
        mean = [pb[str(k)]["coverage_mean"] * 100 for k in b]
        lo = [pb[str(k)]["coverage_p05"] * 100 for k in b]
        hi = [pb[str(k)]["coverage_p95"] * 100 for k in b]
        ax.fill_between(b, lo, hi, color=LIGHT, linewidth=0, zorder=1)
        ax.plot(b, mean, "o-", color=INK, zorder=3)
        target_line(ax, label=False)
        ax.set_xscale("log")
        ax.set_title(d, pad=3)
        ax.set_xlabel("Cells")
        ax.set_xticks([1, 3, 10], ["1", "3", "10"])
        ax.minorticks_off()
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_ylim(40, 102)
    fig.subplots_adjust(wspace=0.12)
    save(fig, out, "calibration_size_sensitivity")


def cqr_comparison(out):
    cqr_gw = output("cqr_groupwise_results")
    cqr_pooled = output("cqr_pooled_results")
    main_gw = output("conformal_groupwise_results")
    pred = np.load("outputs/conformal_groupwise_predictions.npz", allow_pickle=True)
    main_mpiw = float(np.mean(pred["upper"] - pred["lower"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(DATASETS))
    w = 0.38
    ax1.bar(x - w / 2, [main_gw["per_dataset_coverage"][d] * 100 for d in DATASETS], w,
            facecolor=INK, edgecolor=INK, linewidth=0.6, label="Groupwise")
    ax1.bar(x + w / 2, [cqr_gw["per_dataset_coverage"][d] * 100 for d in DATASETS], w,
            facecolor=LIGHT, edgecolor=INK, linewidth=0.6, label="CQR (groupwise)")
    target_line(ax1)
    ax1.set_xticks(x, DATASETS, rotation=30)
    ax1.set_ylabel("Test coverage (%)")
    ax1.set_ylim(0, 112)
    ax1.set_title("(a) Coverage per dataset", loc="left", pad=16)
    ax1.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0), borderaxespad=0.2,
               handlelength=1.6, columnspacing=1.0)

    names = ["Group-\nwise", "CQR\ngroup-\nwise", "CQR\npooled"]
    widths = [main_mpiw, cqr_gw["mpiw"], cqr_pooled["mpiw"]]
    ax2.bar(names, widths, color=[INK, LIGHT, "white"], edgecolor=INK, linewidth=0.6)
    bar_labels(ax2, range(3), widths, fmt="{:.3f}", dy=0.004)
    ax2.set_ylabel("Mean interval width (z-units)")
    ax2.set_ylim(0, max(widths) * 1.22)
    ax2.set_title("(b) Interval width", loc="left", pad=16)
    fig.subplots_adjust(wspace=0.42)
    save(fig, out, "cqr_comparison")


def repeated_split_variability(out):
    summary = output("repeated_split_summary")
    var = summary["groupwise_coverage_variability_across_splits"]
    order = sorted(DATASETS, key=lambda d: var[d]["range"])
    mean = np.array([var[d]["mean"] * 100 for d in order])
    lo = np.array([var[d]["min"] * 100 for d in order])
    hi = np.array([var[d]["max"] * 100 for d in order])

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.3))
    x = np.arange(len(order))
    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="o", color=INK, ecolor=INK,
                elinewidth=0.9, capsize=3, label=f"mean and range over {len(summary['split_seeds'])} splits")
    for xi, d, h in zip(x, order, hi):
        ax.text(xi, h + 1.2, f"{var[d]['range'] * 100:.1f}", ha="center", va="bottom", fontsize=6.0)
    target_line(ax)
    ax.set_xticks(x, order)
    ax.set_ylabel("Groupwise coverage (%)")
    ax.set_ylim(60, 106)
    ax.legend(loc="lower left")
    save(fig, out, "repeated_split_variability")


def mwhc_sensitivity(out):
    mwhc = output("mwhc_results")
    s = mwhc["summary"]
    lam = mwhc["lam_grid"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.1))
    ax1.axhline(s["mean_abs_dev_flat"] * 100, color=MID, linestyle=":", linewidth=1.0,
                label="hierarchical (no weights)")
    ax1.plot(lam, [s["mean_abs_dev_by_lam"][str(v)] * 100 for v in lam], "o-", color=INK,
             label="MWHC")
    ax1.plot(lam, [s["mean_abs_dev_by_lam_floored"][str(v)] * 100 for v in lam], "s--", color=DARK,
             markerfacecolor="white", label="MWHC, floored")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Temperature $\lambda$")
    ax1.set_ylabel(r"Mean $|$coverage$-90|$ (points)")
    ax1.set_title("(a) Coverage deviation", loc="left")
    ax1.legend(loc="upper left")

    ax2.plot(lam, [s["mean_ess_by_lam"][str(v)] for v in lam], "o-", color=INK)
    ax2.axhline(1, color=INK, linestyle=(0, (4, 2)), linewidth=0.8)
    ax2.text(lam[0], 1.08, "one dataset", fontsize=6.0, va="bottom")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks(lam, [f"{v:g}" for v in lam])
        ax.minorticks_off()
    ax2.set_xlabel(r"Temperature $\lambda$")
    ax2.set_ylabel("Effective sample size")
    ax2.set_title("(b) Weight concentration", loc="left")
    fig.subplots_adjust(wspace=0.35)
    save(fig, out, "mwhc_sensitivity")


if __name__ == "__main__":
    setup()
    out_dir = output_dir()
    coverage_comparison(out_dir)
    calibration_size_sensitivity(out_dir)
    cqr_comparison(out_dir)
    repeated_split_variability(out_dir)
    mwhc_sensitivity(out_dir)
