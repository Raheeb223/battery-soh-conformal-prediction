"""
Generates few_shot_coverage.png -- the HGC-CP coverage vs. target-owned
calibration budget figure used in the manuscript (Figure 12, Section 5.3.8).

The numbers below are copied directly from the actual few-shot evaluation
run (few_shot_calibration.py, lam=3.0, beta=0.0) reported in that section --
this script does not recompute anything, it only re-plots already-reported
results so you have a locally reproducible copy of the exact figure.

USAGE: python plot_few_shot_coverage.py
Produces few_shot_coverage.png in the current directory -- upload that file
directly to Overleaf (same filename the manuscript references).
"""

import matplotlib.pyplot as plt

# (cell budgets, coverage) per dataset, copied from the actual reported run.
# CALCE and NASA stop early because those datasets have only 2 and 1 unique
# validation cells respectively -- there is no "5 cells" or "10 cells"
# condition to report for them.
results = {
    'MIT':   ([0, 1, 3, 5, 10, 21], [1.000, 1.000, 1.000, 1.000, 1.000, 1.000]),
    'RWTH':  ([0, 1, 3, 5, 7],      [0.999, 0.998, 0.998, 0.998, 0.998]),
    'XJTU':  ([0, 1, 3, 5, 8],      [0.978, 0.948, 0.959, 0.970, 0.972]),
    'BIT':   ([0, 1, 3, 5, 10, 11], [0.836, 0.834, 0.845, 0.847, 0.852, 0.847]),
    'CALCE': ([0, 1, 2],            [0.235, 0.598, 0.592]),
}

colors = {'MIT': '#2980b9', 'RWTH': '#27ae60', 'XJTU': '#8e44ad', 'BIT': '#e67e22', 'CALCE': '#c0392b'}
markers = {'MIT': 'o', 'RWTH': 's', 'XJTU': '^', 'BIT': 'D', 'CALCE': '*'}


def main():
    fig, ax = plt.subplots(figsize=(7, 5))

    for dataset, (x, y) in results.items():
        is_calce = dataset == 'CALCE'
        ax.plot(x, y, marker=markers[dataset], color=colors[dataset], label=dataset,
                markersize=14 if is_calce else 7, linewidth=2.5 if is_calce else 1.5)

    ax.axhline(0.9, color='black', linestyle='--', linewidth=1, label='90% target')
    ax.set_xlabel('Target-owned calibration cells available')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Coverage vs. target-owned calibration budget (HGC-CP)')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0.15, 1.05)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig('few_shot_coverage.png', dpi=300, bbox_inches='tight')
    print("Saved few_shot_coverage.png")


if __name__ == "__main__":
    main()
