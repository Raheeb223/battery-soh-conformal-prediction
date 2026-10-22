"""
Figures built from the saved per-window predictions rather than from summary results.

  prediction_examples   true and predicted SOH with pooled and groupwise 90% intervals, on one
                        test cell per dataset
  calibration_errors    each dataset's validation errors against the pooled and groupwise
                        quantiles, and where its test errors fall
  strategy_comparison   coverage of every calibration strategy on the five datasets where all
                        are defined

Inputs: outputs/conformal_{groupwise,pooled}_predictions.npz (test intervals, as evaluated in the
paper), outputs/conformal_groupwise_predictions_val.npz (validation errors),
outputs/figure_inputs.npz (SOH normalization and cycle numbers; written by
make_results_registry.py), and results_registry.json.

Run from the repository root:  python figures/make_prediction_figures.py [--out DIR]
"""
import numpy as np
import matplotlib.pyplot as plt

from style import (DARK, DATASETS, INK, LIGHT, MID, TEXT_WIDTH, number, output_dir,
                   registry_values, save, setup)

# NASA has a single test cell with 3 test windows, too few to show a trajectory.
EXAMPLE_DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford"]
FIVE = ["MIT", "RWTH", "XJTU", "BIT", "CALCE"]


def load_predictions():
    gw = np.load("outputs/conformal_groupwise_predictions.npz", allow_pickle=True)
    pooled = np.load("outputs/conformal_pooled_predictions.npz", allow_pickle=True)
    assert (gw["cell_test"] == pooled["cell_test"]).all() and np.allclose(gw["y_test"], pooled["y_test"])
    fi = np.load("outputs/figure_inputs.npz", allow_pickle=True)
    assert len(fi["test_end_cycle"]) == len(gw["y_test"])
    norm = {d: (m, s) for d, m, s in zip(fi["datasets"], fi["soh_mean"], fi["soh_std"])}
    return gw, pooled, fi["test_end_cycle"], norm


def example_cell(gw, dataset):
    """The dataset's test cell with the most test windows (ties broken by name)."""
    cells, counts = np.unique(gw["cell_test"][gw["lab_test"] == dataset], return_counts=True)
    return sorted(zip(-counts, cells))[0][1]


def prediction_examples(out):
    gw, pooled, cycle, norm = load_predictions()
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.9))
    for ax, d in zip(axes.flat, EXAMPLE_DATASETS):
        cell = example_cell(gw, d)
        m = gw["cell_test"] == cell
        order = np.argsort(cycle[m], kind="stable")
        mean, std = norm[d]
        real = lambda z: z[m][order] * std + mean  # noqa: E731  (z-units -> SOH)
        x, y = cycle[m][order], real(gw["y_test"])
        ax.fill_between(x, real(gw["lower"]), real(gw["upper"]), color=LIGHT, linewidth=0,
                        label="Groupwise 90% interval")
        ax.plot(x, real(pooled["lower"]), color=DARK, linestyle=(0, (3, 1.5)), linewidth=0.6,
                label="Pooled 90% interval")
        ax.plot(x, real(pooled["upper"]), color=DARK, linestyle=(0, (3, 1.5)), linewidth=0.6)
        ax.plot(x, y, "o", color=MID, markersize=1.3, markeredgewidth=0, label="True SOH")
        ax.plot(x, real(gw["mean_pred"]), color=INK, linewidth=0.5, label="Prediction")
        cov = lambda p: 100 * np.mean((gw["y_test"][m] >= p["lower"][m]) & (gw["y_test"][m] <= p["upper"][m]))  # noqa: E731
        ax.set_title(f"{d}: {cell}\ncoverage pooled {cov(pooled):.0f}%, groupwise {cov(gw):.0f}%",
                     loc="left", fontsize=6.0, pad=2, linespacing=1.2)
        ax.tick_params(labelsize=6)
    for ax in axes[1]:
        ax.set_xlabel("Cycle")
    for ax in axes[:, 0]:
        ax.set_ylabel("SOH")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(top=0.86, hspace=0.55, wspace=0.32)
    save(fig, out, "prediction_examples")


def calibration_errors(out):
    gw, pooled, _, _ = load_predictions()
    val = np.load("outputs/conformal_groupwise_predictions_val.npz", allow_pickle=True)
    q_pooled = float(np.mean(pooled["upper"] - pooled["lower"])) / 2

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.6))
    x = np.arange(len(DATASETS))
    for xi, d in zip(x, DATASETS):
        err = np.abs(val["y_val"] - val["mean_pred"])[val["lab_val"] == d]
        p5, p25, p50, p75, p95 = np.percentile(err, [5, 25, 50, 75, 95])
        ax.vlines(xi, p5, p25, color=INK, linewidth=0.7)
        ax.vlines(xi, p75, p95, color=INK, linewidth=0.7)
        ax.add_patch(plt.Rectangle((xi - 0.2, p25), 0.4, p75 - p25, facecolor=LIGHT, edgecolor=INK,
                                   linewidth=0.6, zorder=2))
        ax.hlines(p50, xi - 0.2, xi + 0.2, color=INK, linewidth=0.9, zorder=3)
        mt = gw["lab_test"] == d
        q_d = float(np.mean(gw["upper"][mt] - gw["lower"][mt])) / 2
        test_q90 = float(np.quantile(np.abs(gw["y_test"][mt] - gw["mean_pred"][mt]), 0.9))
        ax.plot(xi + 0.32, q_d, marker="D", color=INK, markersize=4, linestyle="none", zorder=4,
                label="Own (groupwise) 90% quantile" if xi == 0 else None)
        ax.plot(xi + 0.32, test_q90, marker="o", markerfacecolor="white", markeredgecolor=INK,
                markersize=4.5, linestyle="none", zorder=4,
                label="90th percentile of test errors" if xi == 0 else None)
    ax.axhline(q_pooled, color=INK, linestyle=(0, (4, 2)), linewidth=0.8, zorder=1,
               label="Pooled 90% quantile")
    ax.set_yscale("log")
    ax.set_xticks(x, DATASETS)
    ax.set_xlim(-0.6, len(DATASETS) - 0.3)
    ax.set_ylabel("Absolute prediction error (z-units)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
    save(fig, out, "calibration_errors")


def strategy_comparison(out):
    v = registry_values()
    rows = [  # label, registry name pattern, marker, filled
        ("Groupwise (own data)", "Hier{}CovOwn", "o", True),
        ("Hierarchical (own data, shrunk)", "Hier{}CovEBShrunk", "s", True),
        ("Hierarchical prior (no own data)", "Hier{}CovPriorOnly", "^", False),
        ("Other datasets pooled (no own data)", "Hier{}CovCrossPooled", "s", False),
        ("HGC-CP (no own data)", "HgcCp{}Coverage", "x", False),
    ]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.9))
    y = np.arange(len(FIVE))[::-1]
    offsets = np.linspace(-0.28, 0.28, len(rows))
    for (label, pattern, marker, filled), dy in zip(rows, offsets):
        cov = [number(v, pattern.format(d)) for d in FIVE]
        kw = dict(color=INK) if marker == "x" else dict(
            markerfacecolor=INK if filled else "white", markeredgecolor=INK)
        ax.plot(cov, y + dy, marker=marker, linestyle="none", markersize=4, label=label, **kw)
    ax.axvline(90, color=INK, linestyle=(0, (4, 2)), linewidth=0.8, zorder=0)
    for yi in y[:-1]:
        ax.axhline(yi - 0.5, color=LIGHT, linewidth=0.5, zorder=0)
    ax.set_yticks(y, FIVE)
    ax.set_xlabel("Test coverage (%)")
    ax.set_xlim(20, 103)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.3,
              columnspacing=1.5)
    save(fig, out, "strategy_comparison")


if __name__ == "__main__":
    setup()
    out_dir = output_dir()
    prediction_examples(out_dir)
    calibration_errors(out_dir)
    strategy_comparison(out_dir)
