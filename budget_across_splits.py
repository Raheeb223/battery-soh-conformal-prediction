"""
The calibration-cell budget of calibration_size_sensitivity.py, repeated on every cell-level split.

The budget rule (the smallest number of calibration cells at which the worst 5% of random draws
reach 85% coverage) and the split between "imprecise" and "biased" datasets were derived on the
main split. A different split puts different cells in each dataset's validation and test sets, so
this script reruns the unchanged sweep (calibration_size_sensitivity.budget_sweep) on all five
splits and records, per split and dataset:
  - the cells needed under the rule (None if not reached below the full set);
  - mean coverage with one cell and with all cells;
  - whether the full-set coverage falls short of 90% by more than the rule's 5-point margin
    (a candidate "bias" that more cells from the same split cannot fix).

Requires: the per-split caches and MC-Dropout predictions used by split_replication.py.
Output:   outputs/budget_across_splits.json
Run:      python budget_across_splits.py
"""
import json
import os

import numpy as np

import config
from borrowing_variants import paths
from calibration_size_sensitivity import budget_sweep

SPLITS = [42, 43, 44, 45, 46]
DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE"]
RULE_FLOOR = 0.85


def summarize(sweep):
    out = {}
    for d, v in sweep["per_dataset"].items():
        pb, n = v["per_budget"], v["n_cells_available"]
        ok = [int(b) for b in sorted(pb, key=int) if int(b) < n and pb[b]["coverage_p05"] >= RULE_FLOOR]
        out[d] = {"n_cells_available": n, "needed": ok[0] if ok else None,
                  "one_cell_mean": pb["1"]["coverage_mean"] if "1" in pb else None,
                  "one_cell_p05": pb["1"]["coverage_p05"] if "1" in pb else None,
                  "full_mean": pb[str(n)]["coverage_mean"],
                  "full_below_rule": pb[str(n)]["coverage_mean"] < RULE_FLOOR}
    return out


def main():
    results = {"rule_floor": RULE_FLOOR, "per_split": {}}
    for s in SPLITS:
        print(f"\n=== split {s}")
        sweep = budget_sweep(*paths(s), datasets=DATASETS)
        results["per_split"][str(s)] = summarize(sweep)

    print(f"\n{'dataset':8s}" + "".join(f"{s:>16d}" for s in SPLITS))
    across = {}
    for d in DATASETS:
        row = [results["per_split"][str(s)][d] for s in SPLITS]
        print(f"{d:8s}" + "".join(
            f"{(str(r['needed']) if r['needed'] else 'n/r') + '/' + str(r['n_cells_available']):>8s}"
            f"{100 * r['full_mean']:7.1f}%" for r in row))
        needed = [r["needed"] for r in row if r["needed"] is not None]
        across[d] = {"needed_min": min(needed) if needed else None, "needed_max": max(needed) if needed else None,
                     "not_reached_splits": sum(r["needed"] is None for r in row),
                     "available_min": min(r["n_cells_available"] for r in row),
                     "available_max": max(r["n_cells_available"] for r in row),
                     "full_mean_min": min(r["full_mean"] for r in row),
                     "full_mean_max": max(r["full_mean"] for r in row),
                     "full_below_rule_splits": sum(r["full_below_rule"] for r in row),
                     "one_cell_mean_min": min(r["one_cell_mean"] for r in row if r["one_cell_mean"] is not None),
                     "one_cell_mean_max": max(r["one_cell_mean"] for r in row if r["one_cell_mean"] is not None)}
    results["across_splits"] = across
    with open(os.path.join(config.OUTPUT_DIR, "budget_across_splits.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("saved outputs/budget_across_splits.json")


if __name__ == "__main__":
    main()
