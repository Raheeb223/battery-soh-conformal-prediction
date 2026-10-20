"""
Graphical abstract (submitted as a separate file, in colour; not part of the manuscript body).

Six panels: seven real sources; data-quality corrections; the predictor; pooled vs. groupwise
coverage (values from results_registry.json); leave-one-dataset-out; registry-based reporting.

Run from the repository root:  python figures/make_graphical_abstract.py [--out DIR]
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

from style import number, output_dir, registry_values, save

OUT = output_dir()
V = registry_values()

plt.rcParams["font.size"] = 8

fig = plt.figure(figsize=(13, 8.5))

COLORS = {
    "nasa": "#4C72B0", "calce": "#DD8452", "oxford": "#55A868", "mit": "#C44E52",
    "bit": "#8172B2", "xjtu": "#937860", "rwth": "#CCB974",
    "good": "#55A868", "bad": "#C44E52",
}


def panel_frame(ax, number, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 10, 10, fill=False, edgecolor="black", linewidth=1.3, zorder=1))
    ax.add_patch(Circle((0.5, 9.3), 0.36, facecolor="black", edgecolor="none", zorder=5))
    ax.text(0.5, 9.3, str(number), ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold", zorder=6)
    ax.text(1.15, 9.3, title, ha="left", va="center", fontsize=8.8, fontweight="bold", zorder=5)


# ------------------------------------------------------------------
# Panel 1: seven heterogeneous real-world sources
# ------------------------------------------------------------------
ax1 = fig.add_subplot(2, 3, 1)
panel_frame(ax1, 1, "Seven real sources,\none shared problem")
labels = ["NASA", "CALCE", "Oxford", "MIT", "BIT", "XJTU", "RWTH"]
colors = [COLORS[k] for k in ["nasa", "calce", "oxford", "mit", "bit", "xjtu", "rwth"]]
positions = [(1.1, 7.0), (2.9, 7.7), (4.7, 7.0), (6.5, 7.7), (8.3, 7.0), (2.0, 5.3), (7.4, 5.3)]
for (x, y), c, lab in zip(positions, colors, labels):
    ax1.add_patch(FancyBboxPatch((x - 0.68, y - 0.42), 1.36, 0.84, boxstyle="round,pad=0.06",
                                  facecolor=c, edgecolor="black", linewidth=0.8))
    ax1.text(x, y, lab, ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
    ax1.add_patch(FancyArrowPatch((x, y - 0.45), (5.0, 2.3), arrowstyle="-|>",
                                   color=c, linewidth=1.0, alpha=0.75,
                                   connectionstyle="arc3,rad=0.08", zorder=2))
ax1.add_patch(FancyBboxPatch((3.6, 1.4), 2.8, 1.1, boxstyle="round,pad=0.1",
                              facecolor="white", edgecolor="black", linewidth=1.3, zorder=4))
ax1.text(5.0, 1.95, f"{V['TotalCellsLoaded']} cells\n7 institutions", ha="center", va="center",
         fontsize=7, zorder=5)

# ------------------------------------------------------------------
# Panel 2: hidden data-quality defects, found and fixed
# ------------------------------------------------------------------
ax2 = fig.add_subplot(2, 3, 2)
panel_frame(ax2, 2, "Hidden defects,\nfound and fixed")
cyc = np.linspace(0, 10, 60)
true_curve = 6.8 - 0.35 * cyc
broken = true_curve.copy()
broken[20:30] += np.linspace(0, 2.3, 10)
ax2.plot(cyc, broken, color=COLORS["bad"], linewidth=1.6, label="raw (uncorrected)", zorder=3)
ax2.plot(cyc, true_curve, color=COLORS["good"], linewidth=1.8, linestyle="--", label="corrected", zorder=3)
peak_idx = 25
ax2.add_patch(Circle((cyc[peak_idx], broken[peak_idx]), 0.75, fill=False, edgecolor="black",
                      linewidth=1.2, linestyle=":", zorder=4))
ax2.annotate("cumulative-counter\nartifact (CALCE)", xy=(cyc[peak_idx], broken[peak_idx]),
             xytext=(6.8, 6.0), fontsize=6.6, ha="left",
             arrowprops=dict(arrowstyle="-", color="black", linewidth=0.7), zorder=4)
ax2.legend(loc="lower left", fontsize=6.3, frameon=False, bbox_to_anchor=(0.0, 0.12))
ax2.text(5.0, 0.6, "Re-verified against raw files\nfor 3 of 7 datasets", fontsize=6.6, ha="center")

# ------------------------------------------------------------------
# Panel 3: model + uncertainty
# ------------------------------------------------------------------
ax3 = fig.add_subplot(2, 3, 3)
panel_frame(ax3, 3, "Multi-scale BiLSTM +\nattention + MC-Dropout")
layer_x = [1.4, 3.0, 4.6, 6.2]
layer_labels = ["input\nwindow\n(20 cyc.)", "BiLSTM\n(fine)", "BiLSTM\n(coarse)", "attention\n+ MLP"]
for x, lab in zip(layer_x, layer_labels):
    ax3.add_patch(FancyBboxPatch((x - 0.58, 4.3), 1.16, 2.1, boxstyle="round,pad=0.06",
                                  facecolor="#DCE6F1", edgecolor="black", linewidth=0.9, zorder=3))
    ax3.text(x, 5.35, lab, ha="center", va="center", fontsize=6.2)
for i in range(len(layer_x) - 1):
    ax3.add_patch(FancyArrowPatch((layer_x[i] + 0.62, 5.35), (layer_x[i + 1] - 0.62, 5.35),
                                   arrowstyle="-|>", color="black", linewidth=0.9, zorder=2))
ax3.add_patch(FancyArrowPatch((6.8, 5.35), (7.7, 5.35), arrowstyle="-|>", color="black", linewidth=0.9))
x_un = np.linspace(7.9, 9.3, 20)
y_mean = np.full_like(x_un, 2.6)
y_band = 0.5 + 0.2 * np.sin(np.linspace(0, 3, 20))
ax3.fill_between(x_un, y_mean - y_band, y_mean + y_band, color=COLORS["mit"], alpha=0.25, zorder=2)
ax3.plot(x_un, y_mean, color=COLORS["mit"], linewidth=1.3, zorder=3)
ax3.text(8.6, 1.5, "$\\hat{y}$ +/- MC-Dropout\nuncertainty", ha="center", fontsize=6.4)

# ------------------------------------------------------------------
# Panel 4: calibration across heterogeneity
# ------------------------------------------------------------------
ax4 = fig.add_subplot(2, 3, 4)
panel_frame(ax4, 4, "One interval width does\nnot fit all datasets")
ds = ["MIT", "RWTH", "XJTU", "BIT", "CALCE"]
pooled_cov = [number(V, f"PooledCoverage{d}") for d in ds]
group_cov = [number(V, f"Coverage{d}") for d in ds]
bar_w = 0.62
x0 = 1.4
gap = 1.75
scale = 5.6
for i, (d, pc, gc) in enumerate(zip(ds, pooled_cov, group_cov)):
    xc = x0 + i * gap
    ax4.bar(xc - bar_w / 2, pc / 100 * scale, width=bar_w, bottom=1.2, color=COLORS["bad"], zorder=3)
    ax4.bar(xc + bar_w / 2, gc / 100 * scale, width=bar_w, bottom=1.2, color=COLORS["good"], zorder=3)
    ax4.text(xc, 0.75, d, ha="center", fontsize=6.6)
target_y = 1.2 + 0.9 * scale
ax4.plot([0.5, 9.5], [target_y, target_y], color="black", linewidth=1.0, linestyle="--", zorder=4)
ax4.text(9.5, target_y + 0.2, "90% target", fontsize=6.2, ha="right", va="bottom")
ax4.add_patch(Rectangle((1.2, 7.75), 0.45, 0.3, facecolor=COLORS["bad"]))
ax4.text(1.8, 7.9, "pooled", fontsize=6.3, va="center")
ax4.add_patch(Rectangle((3.4, 7.75), 0.45, 0.3, facecolor=COLORS["good"]))
ax4.text(4.0, 7.9, "groupwise", fontsize=6.3, va="center")

# ------------------------------------------------------------------
# Panel 5: the real test -- leave-one-dataset-out
# ------------------------------------------------------------------
ax5 = fig.add_subplot(2, 3, 5)
panel_frame(ax5, 5, "The real test: a source\nnever seen in training")
row1 = [(1.3, 7.3), (3.1, 7.3), (4.9, 7.3), (6.7, 7.3), (8.5, 7.3)]
for (x, y), c, lab in zip(row1, colors[:5], labels[:5]):
    ax5.add_patch(FancyBboxPatch((x - 0.65, y - 0.4), 1.3, 0.8, boxstyle="round,pad=0.05",
                                  facecolor=c, edgecolor="black", linewidth=0.7, zorder=3))
    ax5.text(x, y, lab, ha="center", va="center", fontsize=6.2, color="white")
ax5.add_patch(FancyBboxPatch((1.3 - 0.65, 5.9 - 0.4), 1.3, 0.8, boxstyle="round,pad=0.05",
                              facecolor=colors[5], edgecolor="black", linewidth=0.7, zorder=3))
ax5.text(1.3, 5.9, labels[5], ha="center", va="center", fontsize=6.2, color="white")
hx, hy = 3.4, 5.9
ax5.add_patch(FancyBboxPatch((hx - 0.7, hy - 0.42), 1.4, 0.84, boxstyle="round,pad=0.06",
                              facecolor="white", edgecolor=COLORS["rwth"], linewidth=2.2,
                              linestyle="--", zorder=3))
ax5.text(hx, hy, "RWTH\n(held out)", ha="center", va="center", fontsize=6.3, fontweight="bold")
ax5.add_patch(FancyArrowPatch((hx, hy - 0.5), (5.0, 2.6), arrowstyle="-|>",
                               color=COLORS["rwth"], linewidth=1.3, linestyle="--", zorder=2))
ax5.add_patch(FancyBboxPatch((3.3, 1.6), 3.4, 1.15, boxstyle="round,pad=0.1",
                              facecolor="white", edgecolor="black", linewidth=1.2, zorder=4))
ax5.text(5.0, 2.18, "trained on the other 6 --\nnever sees RWTH's\ndistribution at all", ha="center",
         va="center", fontsize=6.4, zorder=5)

# ------------------------------------------------------------------
# Panel 6: honest reporting
# ------------------------------------------------------------------
ax6 = fig.add_subplot(2, 3, 6)
panel_frame(ax6, 6, "Report every number\nfrom the same registry")
ax6.text(5, 7.5, "results_registry.json", ha="center", fontsize=8, fontweight="bold",
         family="monospace")
ax6.add_patch(FancyArrowPatch((5, 7.1), (5, 6.2), arrowstyle="-|>", color="black", linewidth=1.1))
ax6.text(5, 5.6, "numbers.tex", ha="center", fontsize=7.6, family="monospace")
ax6.add_patch(FancyArrowPatch((5, 5.2), (5, 4.3), arrowstyle="-|>", color="black", linewidth=1.1))
ax6.add_patch(FancyBboxPatch((1.0, 1.1), 8.0, 3.0, boxstyle="round,pad=0.12",
                              facecolor="#FBEEEE", edgecolor="black", linewidth=1.0))
ax6.text(5.0, 2.6,
         "Where the model falls short -- does not\nbeat persistence on MIT/RWTH, HGC-CP\ncannot "
         "generalize with one setting --\nis reported, not hidden.",
         ha="center", va="center", fontsize=6.8)

plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=1.5)
save(fig, OUT, "graphical_abstract")
