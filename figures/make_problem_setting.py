"""
Problem-setting figure (Figure 1 of the manuscript), black and white.

Four panels: seven real sources -> one pooled calibration quantile -> per-dataset coverage under
that pooled quantile -> per-dataset coverage when each dataset is calibrated on its own cells.
The bar heights in panels 3 and 4 are the test coverages from results_registry.json.

Run from the repository root:  python figures/make_problem_setting.py [--out DIR]
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from style import (DATASETS, FS_SMALL, INK, LIGHT, MID, TEXT_WIDTH, number, output_dir,
                   registry_values, save, setup)

setup()
OUT = output_dir()
V = registry_values()

SHORT = {"MIT": "MIT", "RWTH": "RW", "XJTU": "XJ", "BIT": "BIT", "CALCE": "CA", "Oxford": "OX",
         "NASA": "NA"}
UNDER = 80.0   # bars below this are hatched as clearly under-covered

fig = plt.figure(figsize=(TEXT_WIDTH, 2.35))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)
ax.axis("off")

PW, GAP, PY, PH = 21.5, 4.0, 1.0, 42.0
XS = [0.6 + i * (PW + GAP) for i in range(4)]


def panel(i, title, caption):
    x = XS[i]
    ax.add_patch(Rectangle((x, PY), PW, PH, facecolor="white", edgecolor=INK, linewidth=0.7))
    ax.text(x + 1.2, PY + PH - 3.0, f"({i + 1})", fontsize=7.5, fontweight="bold", va="center")
    ax.text(x + PW / 2 + 1.5, PY + PH - 3.0, title, fontsize=7.5, fontweight="bold",
            ha="center", va="center")
    ax.text(x + PW / 2, PY + 6.0, caption, fontsize=FS_SMALL, ha="center", va="center",
            linespacing=1.25)
    if i < 3:
        ax.add_patch(FancyArrowPatch((x + PW + 0.4, PY + PH / 2), (x + PW + GAP - 0.4, PY + PH / 2),
                                     arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=INK))
    return x


def coverage_bars(x0, values, base=13.5, height=20.0):
    n = len(DATASETS)
    slot = (PW - 3.0) / n
    ax.plot([x0 + 1.5, x0 + PW - 1.5], [base, base], color=INK, linewidth=0.6)
    ty = base + height * 0.90
    ax.plot([x0 + 1.5, x0 + PW - 1.5], [ty, ty], color=INK, linewidth=0.7, linestyle=(0, (3, 2)))
    for k, (d, v) in enumerate(zip(DATASETS, values)):
        bx = x0 + 1.5 + k * slot + slot * 0.15
        bad = v < UNDER
        ax.add_patch(Rectangle((bx, base), slot * 0.7, height * v / 100,
                               facecolor="white" if bad else LIGHT, edgecolor=INK, linewidth=0.5,
                               hatch="//////" if bad else None))
        ax.text(bx + slot * 0.35, base - 1.2, SHORT[d], fontsize=5.6, ha="center", va="top",
                rotation=90)


# (1) seven sources
x = panel(0, "Seven sources",
          f"{V['TotalCellsLoaded']} cells; different\nlabs, cells, protocols")
for k, d in enumerate(DATASETS):
    cx = x + 2.4 + k * (PW - 4.8) / 6
    ax.add_patch(Rectangle((cx - 0.9, 21.0), 1.8, 7.0, facecolor=LIGHT, edgecolor=INK, linewidth=0.6))
    ax.add_patch(Rectangle((cx - 0.45, 28.0), 0.9, 0.8, facecolor=INK, edgecolor=INK, linewidth=0.4))
    ax.text(cx, 19.4, SHORT[d], fontsize=5.6, ha="center", va="top", rotation=90)

# (2) pool and calibrate once
x = panel(1, "Pool, calibrate once", "one quantile $q$ for\nevery source")
cx = x + PW / 2
ax.add_patch(Polygon([(cx - 8.0, 31.0), (cx + 8.0, 31.0), (cx + 1.6, 22.0), (cx - 1.6, 22.0)],
                     closed=True, facecolor=LIGHT, edgecolor=INK, linewidth=0.6))
for k in range(len(DATASETS)):
    sx = cx - 7.0 + k * 14.0 / 6
    ax.plot([sx, sx], [33.5, 31.0], color=MID, linewidth=0.6)
ax.add_patch(FancyArrowPatch((cx, 22.0), (cx, 15.5), arrowstyle="-|>", mutation_scale=7,
                             linewidth=0.8, color=INK))
ax.text(cx, 13.4, r"$\hat{y}\pm q$", fontsize=8, ha="center", va="center")

# (3) per-dataset coverage under the pooled quantile
pooled = [number(V, f"PooledCoverage{d}") for d in DATASETS]
x = panel(2, "Pooled coverage",
          f"from {V['PooledCoverageMin']}% to {V['PooledCoverageMax']}%;\nhatched: below {UNDER:.0f}%")
coverage_bars(x, pooled)

# (4) per-dataset coverage, each dataset calibrated on its own cells
own = [number(V, f"Coverage{d}") for d in DATASETS]
x = panel(3, "Per-dataset", "own cells per dataset:\nmost near the 90% line")
coverage_bars(x, own)

save(fig, OUT, "problem_setting")
