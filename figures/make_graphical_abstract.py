"""
Graphical abstract (submitted as a separate file, in colour; not part of the manuscript body).

A one-line message and three panels, all values from results_registry.json: coverage of a pooled
vs. a groupwise quantile per dataset; borrowing from other datasets vs. own-data calibration; and
calibration of new data (a dataset left out of training, and field vehicles) on other data vs. on
its own data. One colour key serves all panels: red is calibration on pooled or other data, blue on
the dataset's own data. Landscape at 5:2, as Elsevier asks for graphical abstracts (at least
1328 x 531 px).

Run from the repository root:  python figures/make_graphical_abstract.py [--out DIR]
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from style import DATASETS, number, output_dir, registry_values, save, setup

OUT = output_dir()
V = registry_values()
setup()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
})

OTHER = "#C44E52"
OWN = "#4C72B0"
NEUTRAL = "#9a9a9a"
INK = "#222222"
GRID = "#e6e6e6"
W = 0.38

fig = plt.figure(figsize=(13.3, 5.85))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], left=0.045, right=0.99, bottom=0.12,
                      top=0.72, wspace=0.42)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

fig.text(0.045, 0.955, "Calibrated uncertainty for battery state of health transfers poorly "
         "between datasets", fontsize=17, fontweight="bold", color=INK, va="top")
fig.text(0.045, 0.885, f"Conformal 90% intervals on seven public laboratory datasets "
         f"({V['TotalCellsLoaded']} cells) and an electric-vehicle fleet "
         f"({V['FieldVehiclesTotal']} vehicles)", fontsize=12.5, color="#555555", va="top")
fig.legend(handles=[Patch(color=OTHER, label="Calibrated on pooled or other data"),
                    Patch(color=OWN, label="Calibrated on the dataset's own data"),
                    Line2D([], [], color=INK, linestyle=(0, (4, 2)), linewidth=1.1,
                           label="90% target")],
           loc="upper right", bbox_to_anchor=(0.99, 0.975), ncol=1, fontsize=10.5,
           frameon=False, handlelength=1.6)


def style(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)


def title(ax, letter, text):
    ax.set_title(f"{letter}  {text}", fontsize=12.5, fontweight="bold", loc="left", pad=10,
                 color=INK)


def coverage_bars(ax, labels, left, right):
    x = np.arange(len(labels))
    ax.bar(x - W / 2, left, W, color=OTHER, zorder=2)
    ax.bar(x + W / 2, right, W, color=OWN, zorder=2)
    for xs, vals in ((x - W / 2, left), (x + W / 2, right)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK,
                    zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    ax.axhline(90, color=INK, linestyle=(0, (4, 2)), linewidth=1.1, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Test coverage (%)")
    style(ax)


# (a) One pooled quantile vs. each dataset on its own cells
ax = axes[0]
coverage_bars(ax, DATASETS, [number(V, f"PooledCoverage{d}") for d in DATASETS],
              [number(V, f"Coverage{d}") for d in DATASETS])
title(ax, "a", "One pooled quantile fails on single datasets")

# (b) Borrowing vs. own data, averaged over the cell-level splits
ax = axes[1]
rows = [("Own cells only", "BvOwnDevMean", OWN),
        ("Bayesian borrowing,\nlog scale", "BvBayesLogDevMean", NEUTRAL),
        ("Shrinkage borrowing,\nlinear scale", "BvHierLinDevMean", OTHER),
        ("Best method\nwithout own cells", "BvPriorNoiseDevMean", OTHER)]
y = np.arange(len(rows))[::-1]
vals = [number(V, k) for _, k, _ in rows]
ax.barh(y, vals, color=[c for *_, c in rows], height=0.62, zorder=2)
for yi, v in zip(y, vals):
    ax.text(v + 0.25, yi, f"{v:.1f}", va="center", fontsize=10, color=INK)
ax.set_yticks(y, [r[0] for r in rows], fontsize=10)
ax.set_xlim(0, max(vals) * 1.22)
ax.set_xlabel(f"Points from the 90% target (mean of {V['RepeatedSplitNSplits']} splits)")
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.spines["left"].set_visible(False)
title(ax, "b", "Borrowing does not beat own data")

# (c) New data: calibrated on other data vs. on its own data
ax = axes[2]
cases = [("BIT\nunseen", "LodoTBITOthers", "LodoTBITOwnVal"),
         ("XJTU\nunseen", "LodoTXJTUOthers", "LodoTXJTUOwnVal"),
         ("EV\nmaker 1", "FieldLabPooledEvOne", "FieldLabOwnEvOne"),
         ("EV\nmaker 2", "FieldLabPooledEvTwo", "FieldLabOwnEvTwo")]
coverage_bars(ax, [c[0] for c in cases], [number(V, a) for _, a, _ in cases],
              [number(V, b) for *_, b in cases])
title(ax, "c", "New data needs its own calibration")

save(fig, OUT, "graphical_abstract")
