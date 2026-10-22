"""
HGC-CP's nested held-out hyperparameter search, with each of the seven datasets as the real target
in turn.

HGC-CP's design (MMD weighting, the inflation term, and the grid swept here) was developed while
looking at results on the main split's test sets. This search selects (lambda, beta) for each target
without using that target's test data, but it cannot make the earlier design choices held-out, so
HGC-CP results are exploratory. hgc_cp_validation_only.py chooses the whole design on validation data
only.

Depends only on cache/processed.npz, outputs/conformal_groupwise_predictions.npz, and
outputs/conformal_groupwise_predictions_val.npz (no raw data, no GPU).

Run:  python run_nested_hp_search_all.py
Out:  outputs/nested_hp_search_results.json
"""
import json
import os

import config
from nested_hyperparameter_search import nested_search, DATASETS


def main():
    print(f"Nested hyperparameter search for all {len(DATASETS)} datasets as real target "
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
