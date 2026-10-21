"""
How sure are we that groupwise calibration beats pooled calibration on each dataset?

dataset_level_wilcoxon.py tests the direction of the effect over 6 datasets, which can at best give
p = 0.0156. This script measures its size on each dataset, with uncertainty that respects the
dependence between windows of one cell. For each dataset and split, the gain is

    gain = |pooled coverage - 90| - |groupwise coverage - 90|     (coverage points)

so a positive gain means groupwise calibration is closer to the target. Its 95% interval comes from
a paired bootstrap over test cells: each replicate resamples the dataset's test cells with
replacement and recomputes both coverages from the same cells. The mean gain over the 5 cell-level
splits gets its interval by resampling every split independently in each replicate (the splits
share about a fifth of their test cells, so this interval is somewhat too narrow, not too wide).

A dataset with a single test cell in some split has no cell-level interval; it is reported with
its point values only.

Requires: outputs/conformal_{pooled,groupwise}_predictions.npz and
          outputs/split{43..46}_{pooled,groupwise}_predictions.npz.
Output:   outputs/pooled_vs_groupwise_bootstrap.json

Run:  python pooled_vs_groupwise_bootstrap.py
"""
import json
import os

import numpy as np

import config

SPLITS = {"42": "conformal", "43": "split43", "44": "split44", "45": "split45", "46": "split46"}
SIX = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford"]
N_BOOT = 5000
SEED = 0
OUTPUT = os.path.join(config.OUTPUT_DIR, "pooled_vs_groupwise_bootstrap.json")


def per_cell_hits(prefix, mode, d):
    z = np.load(os.path.join(config.OUTPUT_DIR, f"{prefix}_{mode}_predictions.npz"), allow_pickle=True)
    m = z["lab_test"] == d
    hit = (z["y_test"][m] >= z["lower"][m]) & (z["y_test"][m] <= z["upper"][m])
    cells, inv = np.unique(z["cell_test"][m], return_inverse=True)
    return cells, np.bincount(inv, weights=hit), np.bincount(inv)


def gain(hp, hg, n):
    cp, cg = hp.sum(-1) / n.sum(-1), hg.sum(-1) / n.sum(-1)
    return (np.abs(cp - 0.9) - np.abs(cg - 0.9)) * 100


def main():
    rng = np.random.default_rng(SEED)
    out = {"n_boot": N_BOOT, "per_dataset": {}}
    for d in SIX:
        per_split, boot = {}, []
        for s, prefix in SPLITS.items():
            cp, hp, n = per_cell_hits(prefix, "pooled", d)
            cg, hg, ng = per_cell_hits(prefix, "groupwise", d)
            assert np.array_equal(cp, cg) and np.array_equal(n, ng), f"{d} split {s}: test cells differ"
            point = float(gain(hp, hg, n))
            entry = {"n_test_cells": int(len(cp)), "gain": point,
                     "pooled": float(hp.sum() / n.sum() * 100), "groupwise": float(hg.sum() / n.sum() * 100)}
            if len(cp) > 1:
                idx = rng.integers(0, len(cp), size=(N_BOOT, len(cp)))
                g = gain(hp[idx], hg[idx], n[idx])
                entry["ci"] = [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]
                boot.append(g)
            else:
                boot.append(None)
            per_split[s] = entry
        mean_gain = float(np.mean([e["gain"] for e in per_split.values()]))
        res = {"per_split": per_split, "mean_gain": mean_gain}
        if all(b is not None for b in boot):
            mg = np.mean(boot, axis=0)
            res["mean_gain_ci"] = [float(np.percentile(mg, 2.5)), float(np.percentile(mg, 97.5))]
        out["per_dataset"][d] = res
        ci = res.get("mean_gain_ci")
        print(f"{d:7s} mean gain over splits {mean_gain:6.1f} points  95% CI "
              f"{'[%.1f, %.1f]' % tuple(ci) if ci else 'n/a (a split has one test cell)'}  "
              f"main split {per_split['42']['gain']:5.1f} {per_split['42'].get('ci', '')}")
    with open(OUTPUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
