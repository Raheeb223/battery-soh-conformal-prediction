"""
Cross-architecture pooled-vs-groupwise coverage test, with the correct unit
of analysis.

The pair-level statistic uses 6 architectures x 6 datasets.
The architectures for a given dataset are evaluated on the SAME test
cells, so these pairs are not independent; the independent units are the six
datasets (and, more fundamentally, the cells within them). This script
reports the pair-level statistic and the analysis at the dataset level, where the exact one-sided Wilcoxon signed-rank p-value with n=6
cannot go below 1/64 = 0.0156.

Coverage values (see ERROR_REGISTER.md, task K2) are now read directly from the
machine-generated result files below, never hand-copied. BiLSTM comes from
the main conformal pipeline (outputs/conformal_{groupwise,pooled}_results.json);
linear regression, SVR, XGBoost, plain LSTM, and GRU come from outputs/baseline_conformal_results.json
(baseline_conformal_coverage.py). NASA is excluded from this
table, consistent with its treatment as illustrative-only elsewhere in this
project (n=3 test windows, 1 test cell).

Run:  python dataset_level_wilcoxon.py
"""

import json
import os

import numpy as np
from scipy.stats import wilcoxon

import config

DATASETS = ["MIT", "BIT", "XJTU", "RWTH", "CALCE", "Oxford"]
NOMINAL = 90.0


def _load(path):
    full = os.path.join(config.OUTPUT_DIR, path)
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"{full} not found -- run the scripts that produce it before this one "
            f"(train.py / experiments.py for the BiLSTM conformal results, "
            f"baseline_conformal_coverage.py for the baseline architectures)."
        )
    with open(full) as fh:
        return json.load(fh)


def load_coverage_table():
    """Returns {architecture: [(pooled_pct, groupwise_pct), ...]} in DATASETS
    order, read from the actual result files -- never hand-typed."""
    bilstm_group = _load("conformal_groupwise_results.json")["per_dataset_coverage"]
    bilstm_pool = _load("conformal_pooled_results.json")["per_dataset_coverage"]
    baselines = _load("baseline_conformal_results.json")

    table = {
        "BiLSTM": [(bilstm_pool[d] * 100, bilstm_group[d] * 100) for d in DATASETS],
    }
    name_map = {"Linear": "linear", "SVR": "svr", "XGBoost": "xgboost", "LSTM": "plain_lstm", "GRU": "gru"}
    for label, key in name_map.items():
        if key not in baselines:
            raise KeyError(f"'{key}' missing from baseline_conformal_results.json -- "
                            f"rerun baseline_conformal_coverage.py.")
        pooled = baselines[key]["pooled_per_dataset_coverage"]
        group = baselines[key]["groupwise_per_dataset_coverage"]
        table[label] = [(pooled[d] * 100, group[d] * 100) for d in DATASETS]
    return table


COVERAGE = load_coverage_table()


def main():
    pooled = np.array([[abs(p - NOMINAL) for p, g in v] for v in COVERAGE.values()])   # (predictors, 6)
    group = np.array([[abs(g - NOMINAL) for p, g in v] for v in COVERAGE.values()])
    n_arch = len(COVERAGE)

    r24 = wilcoxon(pooled.ravel(), group.ravel(), alternative="greater", method="exact")
    print(f"{pooled.size} architecture-x-dataset pairs (NOT independent):")
    print(f"  groupwise closer to 90% in {(pooled > group).sum()}/{pooled.size}; mean |dev| pooled={pooled.mean():.1f} pp, groupwise={group.mean():.1f} pp")
    print(f"  Wilcoxon W+={r24.statistic:.1f}, exact one-sided p={r24.pvalue:.2e}")

    p6, g6 = pooled.mean(axis=0), group.mean(axis=0)
    r6 = wilcoxon(p6, g6, alternative="greater", method="exact")
    print("\nCollapsed to the 6 datasets (architectures averaged within dataset):")
    for d, a, b in zip(DATASETS, p6, g6):
        print(f"  {d:7s} |dev| pooled={a:5.1f}  groupwise={b:5.1f}  {'groupwise closer' if a > b else 'pooled closer'}")
    print(f"  groupwise closer in {(p6 > g6).sum()}/6; Wilcoxon W+={r6.statistic:.1f}, exact one-sided p={r6.pvalue:.4f} (minimum attainable with n=6: {1/64:.4f})")

    print(f"\nPer dataset: number of the {n_arch} architectures in which groupwise is closer to 90% than pooled:")
    for j, d in enumerate(DATASETS):
        n_better = int((group[:, j] < pooled[:, j]).sum())
        print(f"  {d:7s} groupwise closer in {n_better}/{n_arch} architectures")

    out = {
        "datasets": DATASETS,
        "architectures": list(COVERAGE.keys()),
        "dataset_level": {"n": len(DATASETS), "groupwise_closer": int((p6 > g6).sum()),
                          "statistic": float(r6.statistic), "p_one_sided_exact": float(r6.pvalue),
                          "p_min_attainable": 1 / 2 ** len(DATASETS)},
        "pair_level_not_independent": {"n": int(pooled.size), "groupwise_closer": int((pooled > group).sum()),
                                       "p_one_sided_exact": float(r24.pvalue)},
    }
    path = os.path.join(config.OUTPUT_DIR, "dataset_level_wilcoxon.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
