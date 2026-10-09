"""
K4: rerun HGC-CP's nested hyperparameter search on the current pipeline state,
for all 7 datasets as the real target in turn, and persist the result.

CONTEXT (must be stated wherever HGC-CP numbers are reported): the design of
HGC-CP itself -- the MMD-based weighting, the heterogeneity inflation term,
and the specific grid this script sweeps -- was arrived at over earlier
iterations by looking at how it performed on the real test sets, before this
nested-search script existed. That means HGC-CP's hyperparameters (lam=3.0,
beta=0.0, the brief's anchor) were originally selected on the pre-fix
pipeline using test-set feedback, not via this held-out protocol. This
script is the honest held-out re-selection asked for by K4, but it cannot
retroactively make HGC-CP's earlier design choices held-out -- only this
specific (lam, beta) pick, going forward, is. HGC-CP results must be
reported as exploratory, not held-out validation.

Depends only on existing outputs (cache/processed.npz,
outputs/conformal_groupwise_predictions.npz,
outputs/conformal_groupwise_predictions_val.npz) -- no raw data, no GPU, so
this runs safely alongside other GPU-bound background jobs.

Run:  python run_nested_hp_search_all.py
Out:  outputs/nested_hp_search_results.json
"""
import json
import os

import config
from nested_hyperparameter_search import nested_search, DATASETS


def main():
    print(f"K4: nested hyperparameter search for all {len(DATASETS)} datasets as real target "
          f"({DATASETS})\n")
    results = {}
    for target in DATASETS:
        print(f"{'='*70}\nreal_target = {target}\n{'='*70}")
        results[target] = nested_search(target, seed=0)
        print()

    out_path = os.path.join(config.OUTPUT_DIR, "nested_hp_search_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'='*70}\nSummary: selected (lam, beta) per real target\n{'='*70}")
    for target, r in results.items():
        print(f"  {target:8s}: lam={r['selected_lam']:<6} beta={r['selected_beta']:<6} "
              f"-> test coverage={r['real_target_test_coverage']:.3f}")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
