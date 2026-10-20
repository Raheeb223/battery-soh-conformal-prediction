"""
Pipeline block diagram (Figure 2 of the manuscript), black and white.

Drawn at the manuscript text width and included at width=\textwidth, so the font sizes below are
the printed sizes. Every number is read from results_registry.json, so the figure cannot disagree
with the text.

Run from the repository root:  python figures/make_block_diagram.py [--out DIR]
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from style import TEXT_WIDTH, output_dir, registry_values, save, setup

setup()
OUT = output_dir()
N = registry_values()

BLACK = "#000000"
SHADE = "#e6e6e6"
LW_BOX = 0.8
LW_ARROW = 0.8
FS_TITLE = 8.5
FS_BODY = 7.4
FS_SMALL = 7.0


def count(name):
    return f"{int(N[name].replace(',', '')):,}"


fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 6.0))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 100)
ax.set_ylim(0, 128)
ax.axis("off")

LEFT, RIGHT = 9.0, 99.5          # content column; 0-9 holds the stage labels
WIDTH = RIGHT - LEFT
CX = (LEFT + RIGHT) / 2


def rect(x, y, w, h, fill="white", ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=BLACK, linewidth=LW_BOX,
                           linestyle=ls, zorder=2))


def text(x, y, s, size=FS_BODY, weight="normal", style="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style, ha=ha, va=va,
            color=BLACK, zorder=3, linespacing=1.3)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=LW_ARROW, color=BLACK, shrinkA=0, shrinkB=0, zorder=1))


def down_arrow(x, y_top, y_bot, label=None):
    arrow(x, y_top, x, y_bot)
    if label:
        text(x + 1.5, (y_top + y_bot) / 2, label, size=FS_SMALL, style="italic", ha="left")


def stage_label(y_top, y_bot, letter, name):
    xb = LEFT - 1.8
    ax.plot([xb, xb], [y_bot, y_top], color=BLACK, linewidth=0.5, zorder=1)
    ax.plot([xb, xb + 0.9], [y_top, y_top], color=BLACK, linewidth=0.5, zorder=1)
    ax.plot([xb, xb + 0.9], [y_bot, y_bot], color=BLACK, linewidth=0.5, zorder=1)
    ax.text(xb - 2.6, (y_top + y_bot) / 2, f"({letter}) {name}", rotation=90, fontsize=FS_BODY,
            fontweight="bold", ha="center", va="center", color=BLACK)


# ---------------------------------------------------------------- (a) Data
top = 126.5
h = 10.0
y = top - h
rect(LEFT, y, WIDTH, h, fill=SHADE)
text(CX, y + 6.6, f"Seven public datasets: {count('TotalCellsLoaded')} cells, seven institutions",
     size=FS_TITLE, weight="bold")
text(CX, y + 2.8, "MIT  ·  RWTH  ·  XJTU  ·  BIT  ·  CALCE  ·  Oxford  ·  NASA")
stage_label(top, y, "a", "Data")
down_arrow(CX, y, y - 5.5, "raw files re-verified; CALCE, XJTU, RWTH corrected")

# ---------------------------------------------------------- (b) Preprocessing
top = y - 5.5
head_h, gap_v, split_h = 8.5, 3.0, 9.5
rect(LEFT, top - head_h, WIDTH, head_h)
text(CX, top - 2.9, "Preprocessing", size=FS_TITLE, weight="bold")
text(CX, top - 6.2, r"SOH $=Q_k/\mathrm{median}(Q_1,Q_2,Q_3)$;  cell-level split;  20-cycle windows")
split_top = top - head_h - gap_v
split_gap = 2.5
split_w = (WIDTH - 2 * split_gap) / 3
splits = [("Training cells", count("TrainWindows"), "fit the predictor"),
          ("Validation cells", count("ValWindows"), "calibrate the interval"),
          ("Test cells", count("TotalTestWindows"), "evaluate only")]
split_cx = []
for i, (title, n_win, role) in enumerate(splits):
    x = LEFT + i * (split_w + split_gap)
    split_cx.append(x + split_w / 2)
    rect(x, split_top - split_h, split_w, split_h)
    text(x + split_w / 2, split_top - 2.6, title, weight="bold")
    text(x + split_w / 2, split_top - 5.2, f"{n_win} windows", size=FS_SMALL)
    text(x + split_w / 2, split_top - 7.7, role, size=FS_SMALL, style="italic")
    arrow(x + split_w / 2, top - head_h, x + split_w / 2, split_top)
y = split_top - split_h
stage_label(top, y, "b", "Preprocessing")
down_arrow(split_cx[0], y, y - 5.5, "training windows")

# --------------------------------------------------------------- (c) Predictor
top = y - 5.5
h = 13.0
y = top - h
pred_w = WIDTH * 0.64
rect(LEFT, y, pred_w, h)
pcx = LEFT + pred_w / 2
text(pcx, top - 3.0, "Point predictor", size=FS_TITLE, weight="bold")
text(pcx, top - 6.3, "multi-scale bidirectional LSTM, self-attention")
text(pcx, top - 9.6, fr"centre $\hat{{y}}$: mean of {N['DeployMcDropoutSamples']} MC-Dropout passes,"
                     "\nor one deterministic pass", size=FS_SMALL, style="italic")
base_x = LEFT + pred_w + 2.5
base_w = RIGHT - base_x
rect(base_x, y, base_w, h, ls=(0, (3, 2)))
text(base_x + base_w / 2, top - 3.0, "Reference predictors", weight="bold")
text(base_x + base_w / 2, top - 8.0, "linear, SVR, XGBoost,\nfour ablations,\npersistence",
     size=FS_SMALL)
stage_label(top, y, "c", "Prediction")
down_arrow(CX, y, y - 5.5, r"validation residuals $|y-\hat{y}|$, per dataset")

# ------------------------------------------------------------- (d) Calibration
top = y - 5.5
h = 27.0
y = top - h
rect(LEFT, y, WIDTH, h)
text(CX, top - 3.0, "Split-conformal calibration on validation cells (90% target)",
     size=FS_TITLE, weight="bold")
groups = [
    ("Own data only", ["Groupwise (Mondrian)", "Normalized groupwise"], [0, 1]),
    ("Shared across datasets", ["Pooled", "Hierarchical (emp. Bayes)", "MWHC (ours)",
                                "HGC-CP (other datasets)"], [0, 1, 2, 3]),
    ("Learned interval", ["CQR: quantile-", "trained backbone,", "then conformal"], []),
]
fracs = [0.31, 0.42, 0.27]
gap = 2.0
avail = WIDTH - 3.0 - gap * (len(groups) - 1)
x = LEFT + 1.5
inner_top, inner_h = top - 6.0, 19.5
for (title, lines, bullets), frac in zip(groups, fracs):
    w = avail * frac
    rect(x, inner_top - inner_h, w, inner_h, fill=SHADE if title.startswith("Shared") else "white")
    text(x + w / 2, inner_top - 2.6, title, weight="bold", style="italic")
    ax.plot([x + 1.2, x + w - 1.2], [inner_top - 4.6, inner_top - 4.6], color=BLACK,
            linewidth=0.4, zorder=3)
    for k, line in enumerate(lines):
        prefix = "•  " if k in bullets else ("    " if bullets else "")
        text(x + 1.5, inner_top - 7.6 - k * 3.2, prefix + line, size=FS_SMALL, ha="left")
    x += w + gap
stage_label(top, y, "d", "Calibration")
down_arrow(CX, y, y - 5.5, r"intervals $\hat{y}\pm q$ on test cells")

# -------------------------------------------------------------- (e) Evaluation
top = y - 5.5
h = 17.5
y = top - h
rect(LEFT, y, WIDTH, h)
text(CX, top - 3.0, "Evaluation on test cells", size=FS_TITLE, weight="bold")
cols = [
    ["per-dataset coverage and width",
     "calibration-cell budget sweep",
     "leave-one-dataset-out (model level)"],
    [f"{N['RepeatedSplitNSplits']} cell-level splits; cell bootstrap",
     "point accuracy vs. persistence",
     "inference cost"],
]
for c, items in enumerate(cols):
    for k, item in enumerate(items):
        text(LEFT + 2.0 + c * WIDTH / 2, top - 7.2 - k * 3.4, "•  " + item, size=FS_SMALL, ha="left")
stage_label(top, y, "e", "Evaluation")
gap_h = 5.0
cw = (WIDTH - 2 * gap_h) / 3
down_arrow(LEFT + cw / 2, y, y - 5.5, "every script saves its results")

# --------------------------------------------------------------- (f) Reporting
top = y - 5.5
h = 10.0
y = top - h
chain = [("outputs/*.json", "one file per script"),
         ("results_registry.json", "single registry"),
         ("numbers.tex", "macros used by the text")]
for i, (title, sub) in enumerate(chain):
    x = LEFT + i * (cw + gap_h)
    rect(x, y, cw, h, fill=SHADE)
    text(x + cw / 2, y + 6.4, title, weight="bold")
    text(x + cw / 2, y + 3.0, sub, size=FS_SMALL, style="italic")
    if i < len(chain) - 1:
        arrow(x + cw, y + h / 2, x + cw + gap_h, y + h / 2)
stage_label(top, y, "f", "Reporting")
assert y > 0, f"layout overflows the canvas by {-y:.1f} units"

save(fig, OUT, "block_diagram")
