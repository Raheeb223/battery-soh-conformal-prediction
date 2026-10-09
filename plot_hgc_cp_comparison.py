"""
Generates hgc_cp_comparison.png -- the HGC-CP vs. zero-target-data-baseline
coverage comparison figure used in the manuscript (Figure 12, Section 5.3.7).

IMPORTANT LABELING NOTE, since this was corrected after an earlier version:
the two baselines here are "Cross-source pooled" and "Nearest-source" --
NOT "Pooled" and "Groupwise". Both are zero-target-data baselines computed
from the six OTHER datasets only (the target's own calibration data is
never used, simulating a genuinely unseen deployment source), matching
HGC-CP's own evaluation condition exactly. Do NOT relabel either of these
as "Groupwise" -- that name is reserved elsewhere in the manuscript
(Table 9) for a DIFFERENT quantity: calibration using a dataset's OWN
calibration data, which is not what either baseline here computes. Using
"Groupwise" for the nearest-source baseline was an earlier, confusing
mislabeling that has since been corrected throughout the manuscript --
keep this script consistent with that fix, not the earlier version.

The numbers below are copied directly from the actual evaluation results
reported in Table 17 -- this script does not recompute anything, it only
re-plots already-reported results.

USAGE: python plot_hgc_cp_comparison.py
Produces hgc_cp_comparison.png in the current directory -- upload that file
directly to Overleaf (same filename the manuscript references).
"""

import matplotlib.pyplot as plt
import numpy as np

datasets = ['MIT', 'RWTH', 'XJTU', 'BIT', 'CALCE', 'Oxford', 'NASA']

# Coverage at calib_size=25 (small budget) and calib_size='all' (max data),
# lam=3.0, beta=0.0 -- copied from Table 17's actual reported values.
hgc_coverage_n25 = [0.999886, 0.999033, 0.984261, 0.870927, 0.397959, 1.0, 1.0]
hgc_coverage_all = [0.999886, 0.999033, 0.980422, 0.817043, 0.214286, 1.0, 1.0]

# Cross-source pooled: a single quantile pooled across the six OTHER
# datasets only, excluding the target's own data entirely.
pooled_coverage_n25 = [0.999886, 0.997918, 0.996545, 0.746867, 0.153061, 1.0, 1.0]
pooled_coverage_all = [0.999600, 0.555258, 0.942035, 0.300752, 0.071429, 1.0, 0.333333]

# Nearest-source: the single closest OTHER dataset's own quantile, used as
# a naive stand-in for the target's -- NOT the target's own groupwise
# calibration (see module docstring).
nearest_coverage_n25 = [0.999829, 0.999851, 0.834165, 0.644110, 0.908163, 1.0, 1.0]
nearest_coverage_all = [0.999600, 0.999331, 0.535509, 0.414787, 0.612245, 1.0, 0.333333]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(datasets))
    width = 0.25

    # Panel (a): small calibration budget (n=25)
    ax = axes[0]
    ax.bar(x - width, hgc_coverage_n25, width, label='HGC-CP', color='#c0392b')
    ax.bar(x, pooled_coverage_n25, width, label='Cross-source pooled', color='#7f8c8d')
    ax.bar(x + width, nearest_coverage_n25, width, label='Nearest-source', color='#2980b9')
    ax.axhline(0.9, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('(a) Coverage at small calibration budget (n=25)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Panel (b): maximum available calibration data
    ax = axes[1]
    ax.bar(x - width, hgc_coverage_all, width, label='HGC-CP', color='#c0392b')
    ax.bar(x, pooled_coverage_all, width, label='Cross-source pooled', color='#7f8c8d')
    ax.bar(x + width, nearest_coverage_all, width, label='Nearest-source', color='#2980b9')
    ax.axhline(0.9, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('(b) Coverage at maximum available calibration data')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('hgc_cp_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved hgc_cp_comparison.png")


if __name__ == "__main__":
    main()
