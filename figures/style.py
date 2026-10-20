"""
Shared style and data access for the manuscript figures.

Every figure is drawn at its printed size (the manuscript text width is 390 pt = 5.40 in) and
included in LaTeX at width=\\textwidth, so the font sizes set here are the sizes that appear in
print. Figures are black and white so they reproduce correctly in grayscale.

Numbers come from two places only:
  - results_registry.json["values"]: every number printed in the manuscript, formatted exactly as
    printed (written by make_results_registry.py);
  - outputs/*.json: the per-experiment result files the registry itself is built from.
No figure contains a hand-typed result.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

TEXT_WIDTH = 390 / 72.27            # inches
DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]

INK = "#000000"
DARK = "#4d4d4d"
MID = "#8c8c8c"
LIGHT = "#d4d4d4"

FS = 8.0          # body text
FS_SMALL = 7.0    # annotations, tick labels


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS,
        "xtick.labelsize": FS_SMALL,
        "ytick.labelsize": FS_SMALL,
        "legend.fontsize": FS_SMALL,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })


def output_dir(default="figures/out"):
    """--out DIR (default figures/out). Run from the repository root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", default=default, help="directory for the .pdf/.png files")
    out = parser.parse_args().out
    os.makedirs(out, exist_ok=True)
    return out


def registry_values(path="results_registry.json"):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["values"]


def number(values, name):
    """A registry value as a float (thousands separators removed)."""
    return float(values[name].replace(",", ""))


def output(name):
    with open(os.path.join("outputs", f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {out}/{name}.pdf and .png")


def target_line(ax, label=True):
    ax.axhline(90, color=INK, linestyle=(0, (4, 2)), linewidth=0.8, zorder=1,
               label="90% target" if label else None)
