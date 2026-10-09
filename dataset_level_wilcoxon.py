"""
Cross-architecture pooled-vs-groupwise coverage test, with the correct unit
of analysis.

The manuscript's 24 paired observations are 4 architectures x 6 datasets.
The four architectures for a given dataset are evaluated on the SAME test
cells, so the 24 pairs are not independent; the independent units are the six
datasets (and, more fundamentally, the cells within them). This script
reproduces the 24-pair statistic from the manuscript's own table values and
reports the analysis at the dataset level, where the exact one-sided
Wilcoxon signed-rank p-value with n=6 cannot go below 1/64 = 0.0156.

Input values are copied from the manuscript's Table (BiLSTM: pooled and
groupwise coverage, %) and the baseline conformal table (SVR, XGBoost, plain
LSTM). They come from stored run outputs; this script does not retrain.

Run:  python dataset_level_wilcoxon.py
"""

import numpy as np
from scipy.stats import wilcoxon

DATASETS = ["MIT", "BIT", "XJTU", "RWTH", "CALCE", "Oxford"]
# (pooled coverage %, groupwise coverage %) per dataset, in the order above
COVERAGE = {
    "BiLSTM":  [(91.5, 90.8), (38.7, 87.1), (62.7, 96.0), (87.6, 85.1), (10.2, 71.4), (64.3, 100.0)],
    "SVR":     [(97.0, 92.9), (28.3, 86.6), (46.6, 95.2), (94.6, 89.3), (5.1, 83.7), (80.4, 100.0)],
    "XGBoost": [(94.9, 92.5), (22.3, 85.2), (46.4, 93.4), (97.2, 89.7), (4.1, 75.5), (58.9, 100.0)],
    "LSTM":    [(95.1, 92.2), (42.2, 86.6), (62.8, 95.4), (94.7, 90.2), (11.2, 84.7), (62.5, 100.0)],
}
NOMINAL = 90.0


def main():
    pooled = np.array([[abs(p - NOMINAL) for p, g in v] for v in COVERAGE.values()])   # (4, 6)
    group = np.array([[abs(g - NOMINAL) for p, g in v] for v in COVERAGE.values()])

    r24 = wilcoxon(pooled.ravel(), group.ravel(), alternative="greater", method="exact")
    print("24 architecture-x-dataset pairs (NOT independent):")
    print(f"  groupwise closer to 90% in {(pooled > group).sum()}/24; mean |dev| pooled={pooled.mean():.1f} pp, groupwise={group.mean():.1f} pp")
    print(f"  Wilcoxon W+={r24.statistic:.1f}, exact one-sided p={r24.pvalue:.2e}")

    p6, g6 = pooled.mean(axis=0), group.mean(axis=0)
    r6 = wilcoxon(p6, g6, alternative="greater", method="exact")
    print("\nCollapsed to the 6 datasets (architectures averaged within dataset):")
    for d, a, b in zip(DATASETS, p6, g6):
        print(f"  {d:7s} |dev| pooled={a:5.1f}  groupwise={b:5.1f}  {'groupwise closer' if a > b else 'pooled closer'}")
    print(f"  groupwise closer in {(p6 > g6).sum()}/6; Wilcoxon W+={r6.statistic:.1f}, exact one-sided p={r6.pvalue:.4f} (minimum attainable with n=6: {1/64:.4f})")

    print("\nPer dataset: number of the 4 architectures in which groupwise is closer to 90% than pooled:")
    for j, d in enumerate(DATASETS):
        n_better = int((group[:, j] < pooled[:, j]).sum())
        print(f"  {d:7s} groupwise closer in {n_better}/4 architectures")


if __name__ == "__main__":
    main()
