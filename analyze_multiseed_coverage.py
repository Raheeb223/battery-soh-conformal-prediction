"""
Per-dataset conformal coverage and MAE across the three training seeds.

Aggregates the per_dataset_coverage and per_dataset_mae dictionaries saved in
outputs/multiseed_{42,123,2024}_results.json (full model, groupwise conformal
calibration) into mean +- SD per dataset. No training is performed.

Run:
    python analyze_multiseed_coverage.py
Output:
    outputs/multiseed_per_dataset_coverage.json
    Console tables of coverage and MAE (mean +- SD across 3 seeds) per dataset.
"""

import os
import json
import numpy as np
import config

SEEDS = [42, 123, 2024]
DATASET_ORDER = ["MIT", "BIT", "XJTU", "RWTH", "CALCE", "Oxford", "NASA"]


def load_seed_results():
    results = []
    for seed in SEEDS:
        path = os.path.join(config.OUTPUT_DIR, f"multiseed_{seed}_results.json")
        if not os.path.exists(path):
            print(f"  [!] {path} not found -- run experiments.py's multiseed "
                  f"section first (Section 5 in experiments.py).")
            return None
        with open(path) as fh:
            results.append(json.load(fh))
    return results


def main():
    results = load_seed_results()
    if results is None:
        return

    conformal_modes = {r["conformal_mode"] for r in results}
    print(f"Loaded {len(results)} seed runs. Conformal mode used: {conformal_modes} "
          f"(should be {{'groupwise'}} -- these runs used train_one()'s default).")
    if conformal_modes != {"groupwise"}:
        print("  [!] WARNING: not all seeds used groupwise calibration -- "
              "verify this is expected before using the results below.")

    all_datasets = set()
    for r in results:
        all_datasets |= set(r["per_dataset_coverage"].keys())
    datasets = [d for d in DATASET_ORDER if d in all_datasets] + \
               [d for d in all_datasets if d not in DATASET_ORDER]

    print("\n" + "=" * 78)
    print("Per-dataset GROUPWISE COVERAGE across 3 seeds (mean +- SD)")
    print("=" * 78)
    print(f"{'Dataset':10s} {'seed 42':>10s} {'seed 123':>10s} {'seed 2024':>10s} "
          f"{'mean':>8s} {'SD':>8s} {'95% CI (approx, t-dist, n=3)':>30s}")

    coverage_summary = {}
    for ds in datasets:
        vals = []
        for r in results:
            v = r["per_dataset_coverage"].get(ds)
            vals.append(v * 100 if v is not None else float("nan"))
        arr = np.array(vals)
        mean = np.nanmean(arr)
        sd = np.nanstd(arr, ddof=1) if np.sum(~np.isnan(arr)) > 1 else float("nan")
        # approximate 95% CI using t-distribution with n=3 (df=2), t=4.303
        if not np.isnan(sd):
            margin = 4.303 * sd / np.sqrt(len(arr))
            ci_lo, ci_hi = mean - margin, mean + margin
        else:
            ci_lo, ci_hi = float("nan"), float("nan")
        coverage_summary[ds] = {
            "seed_values": vals, "mean": float(mean), "sd": float(sd),
            "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        }
        print(f"{ds:10s} {vals[0]:9.1f}% {vals[1]:9.1f}% {vals[2]:9.1f}% "
              f"{mean:7.1f}% {sd:7.1f}% [{ci_lo:6.1f}%, {ci_hi:6.1f}%]")

    print("\n" + "=" * 78)
    print("Per-dataset MAE across 3 seeds (mean +- SD, real SOH units)")
    print("=" * 78)
    print(f"{'Dataset':10s} {'seed 42':>10s} {'seed 123':>10s} {'seed 2024':>10s} "
          f"{'mean':>10s} {'SD':>10s}")

    mae_summary = {}
    for ds in datasets:
        vals = []
        for r in results:
            v = r["per_dataset_mae"].get(ds)
            vals.append(v if v is not None else float("nan"))
        arr = np.array(vals)
        mean = np.nanmean(arr)
        sd = np.nanstd(arr, ddof=1) if np.sum(~np.isnan(arr)) > 1 else float("nan")
        mae_summary[ds] = {"seed_values": vals, "mean": float(mean), "sd": float(sd)}
        print(f"{ds:10s} {vals[0]:9.4f}  {vals[1]:9.4f}  {vals[2]:9.4f}  "
              f"{mean:9.4f}  {sd:9.4f}")

    out_path = os.path.join(config.OUTPUT_DIR, "multiseed_per_dataset_coverage.json")
    with open(out_path, "w") as fh:
        json.dump({"coverage": coverage_summary, "mae": mae_summary}, fh, indent=2)
    print(f"\nsaved {out_path}")

    print("\n" + "=" * 78)
    print("NOTE ON SCOPE")
    print("=" * 78)
    print("This uses the EXISTING multiseed_{42,123,2024} runs, which all used\n"
          "groupwise calibration (train_one()'s default). This directly\n"
          "strengthens the groupwise-coverage claim -- your paper's central\n"
          "result -- with real seed-to-seed variability, no new training\n"
          "required. It does NOT give seed-level variability for pooled or\n"
          "normalized-groupwise calibration; if you want that too, a further\n"
          "experiment training 3 seeds under each of the other two conformal\n"
          "modes would be needed (9 additional short runs total) -- ask if you\n"
          "want that script written as well.")


if __name__ == "__main__":
    main()
