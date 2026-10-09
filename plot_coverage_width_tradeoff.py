"""
Generates coverage_width_tradeoff.png -- the coverage-vs-width trade-off
figure used in the manuscript (Figure 13, Section 5.3.2).

The numbers below are copied directly from coverage_width_tradeoff.py's
normalized-groupwise table (part (b) of that script's output) -- this
script does not recompute anything, it only re-plots already-reported
results so you have a locally reproducible copy of the exact figure.

USAGE: python plot_coverage_width_tradeoff.py
Produces coverage_width_tradeoff.png in the current directory -- upload
that file directly to Overleaf (same filename the manuscript references).
"""

import matplotlib.pyplot as plt

# (mean widths at 80/90/95% targets, coverage at 80/90/95% targets) per
# dataset, copied from coverage_width_tradeoff.py's normalized-groupwise
# output. Oxford and NASA are omitted from the plot (as in the manuscript
# figure) since both are near-saturated at every level and would compress
# the informative datasets' curves on a shared axis.
results = {
    'MIT':   ([0.0028, 0.0041, 0.0059], [0.820, 0.927, 0.972]),
    'RWTH':  ([0.0146, 0.0191, 0.0235], [0.791, 0.891, 0.944]),
    'XJTU':  ([0.0122, 0.0213, 0.0336], [0.828, 0.925, 0.966]),
    'BIT':   ([0.0753, 0.1741, 0.3222], [0.757, 0.870, 0.912]),
    'CALCE': ([0.1490, 0.2224, 0.4067], [0.551, 0.735, 0.898]),
}
NOMINAL_TARGETS = [0.80, 0.90, 0.95]
colors = {'MIT': '#2980b9', 'RWTH': '#27ae60', 'XJTU': '#8e44ad', 'BIT': '#e67e22', 'CALCE': '#c0392b'}


def main():
    fig, ax = plt.subplots(figsize=(7, 5))

    for dataset, (widths, covs) in results.items():
        ax.plot(widths, covs, marker='o', color=colors[dataset], label=dataset,
                linewidth=2, markersize=7)
        for w, c, target in zip(widths, covs, NOMINAL_TARGETS):
            ax.annotate(f'{target:.0%}', (w, c), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))

    ax.axhline(0.9, color='gray', linestyle=':', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Mean interval width (log scale, real SOH units)')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Coverage vs. width trade-off (normalized groupwise), by nominal target')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig('coverage_width_tradeoff.png', dpi=300, bbox_inches='tight')
    print("Saved coverage_width_tradeoff.png")


if __name__ == "__main__":
    main()
