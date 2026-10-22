"""
Graphical abstract (submitted as a separate file, in colour; not part of the manuscript body).

Three panels, all values from results_registry.json: coverage of a pooled vs. a groupwise quantile
per dataset; borrowing from other datasets vs. own-data calibration; and calibration of new data
(a dataset left out of training, and field vehicles) on other data vs. on its own data.
Landscape, as Elsevier asks for graphical abstracts (at least 1328 x 531 px).

Run from the repository root:  python figures/make_graphical_abstract.py [--out DIR]
"""
import numpy as np
import matplotlib.pyplot as plt

from style import DATASETS, number, output_dir, registry_values, save, setup

OUT = output_dir()
V = registry_values()
setup()
plt.rcParams["font.size"] = 9

BAD = "#C44E52"
GOOD = "#4C72B0"
NEUTRAL = "#8c8c8c"

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.35, 1, 1]})
fig.suptitle(f"How far does calibrated uncertainty for battery SOH transfer? "
             f"Seven public datasets ({V['TotalCellsLoaded']} cells) and "
             f"{V['FieldVehiclesTotal']} electric vehicles", fontsize=11.5, fontweight="bold", y=0.99)


def target(ax):
    ax.axhline(90, color="black", linestyle=(0, (4, 2)), linewidth=0.9, zorder=1, label="90% target")


def labels(ax, xs, vals, fmt="{:.0f}"):
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.5, fmt.format(v), ha="center", va="bottom", fontsize=7.5)


# (a) Pooled vs. groupwise coverage per dataset
ax = axes[0]
x = np.arange(len(DATASETS))
w = 0.38
pooled = [number(V, f"PooledCoverage{d}") for d in DATASETS]
group = [number(V, f"Coverage{d}") for d in DATASETS]
ax.bar(x - w / 2, pooled, w, color=BAD, label="One pooled quantile")
ax.bar(x + w / 2, group, w, color=GOOD, label="Each dataset on its own cells")
labels(ax, x - w / 2, pooled)
labels(ax, x + w / 2, group)
ax.set_xticks(x, DATASETS)
ax.set_ylim(0, 118)
ax.set_xlim(-0.6, len(DATASETS) - 0.4)
ax.set_ylabel("Test coverage (%)")
target(ax)
ax.legend(loc="upper left", ncol=3, fontsize=8)
ax.set_title("(a) A pooled quantile looks right overall\nbut fails on individual datasets",
             fontsize=10, loc="left")

# (b) Borrowing vs. own data, averaged over five cell-level splits
ax = axes[1]
rows = [("Own cells only", "BvOwnDevMean", GOOD),
        ("Shrinkage, linear scale", "BvHierLinDevMean", BAD),
        ("Bayesian, log scale", "BvBayesLogDevMean", NEUTRAL),
        ("Best method without\nown cells", "BvPriorNoiseDevMean", BAD)]
y = np.arange(len(rows))[::-1]
vals = [number(V, k) for _, k, _ in rows]
ax.barh(y, vals, color=[c for *_, c in rows], height=0.6)
for yi, v in zip(y, vals):
    ax.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=8)
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlabel("Distance of coverage from 90% (points),\nmean over 5 splits")
ax.set_xlim(0, max(vals) * 1.25)
ax.set_title("(b) Borrowing from other datasets\ndoes not beat own data", fontsize=10, loc="left")

# (c) New data: calibrated on other data vs. on its own data
ax = axes[2]
cases = [("BIT\n(left out)", "LodoTBITOthers", "LodoTBITOwnVal"),
         ("XJTU\n(left out)", "LodoTXJTUOthers", "LodoTXJTUOwnVal"),
         ("EV fleet 1", "FieldLabPooledEvOne", "FieldLabOwnEvOne"),
         ("EV fleet 2", "FieldLabPooledEvTwo", "FieldLabOwnEvTwo")]
x = np.arange(len(cases))
other = [number(V, a) for _, a, _ in cases]
own = [number(V, b) for *_, b in cases]
ax.bar(x - w / 2, other, w, color=BAD, label="Calibrated on other data")
ax.bar(x + w / 2, own, w, color=GOOD, label="Calibrated on its own data")
labels(ax, x - w / 2, other)
labels(ax, x + w / 2, own)
ax.set_xticks(x, [c[0] for c in cases])
ax.set_ylim(0, 118)
ax.set_xlim(-0.6, len(cases) - 0.4)
ax.set_ylabel("Test coverage (%)")
target(ax)
ax.legend(loc="upper left", ncol=2, fontsize=8)
ax.set_title("(c) New datasets and field data need\ncalibration data of their own", fontsize=10,
             loc="left")

fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=2.5)
save(fig, OUT, "graphical_abstract")
