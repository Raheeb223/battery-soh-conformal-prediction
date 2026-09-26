"""
Runs the full experiment suite and writes one summary table.

Sections, in order:
  1. Baselines: linear regression, SVR, XGBoost (if installed)
  2. Architecture ablations: full, no_attention, no_multiscale, plain_lstm,
     plus a GRU baseline
  3. Conformal calibration ablation: groupwise vs pooled vs normalized_groupwise
  4. Window-size ablation: 10, 20, 30, 50 cycles
  5. Multi-seed runs of the full model (seeds 42, 123, 2024)
  6. Leave-one-dataset-out generalisation (each of the 7 datasets held out)

Each run's results are saved to outputs/<run_name>_results.json. If such a
file already exists and was computed on the same set of datasets, it is
reused instead of retraining (run clear_everything.py to force a full
retrain). Individual sections can be run by calling the run_*_section()
functions directly.

This trains roughly 25 models and takes several hours on CPU.

Run:
    python experiments.py
Output:
    outputs/experiment_summary.json  -- every run's results
    outputs/experiment_summary.md    -- the same, as markdown tables
"""

import os
import json
import numpy as np

import config
import preprocessing
from train import train_one
from baselines import run_baseline
from leave_one_out import run_lodo
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

SEEDS = [42, 123, 2024]


def _current_dataset_signature(cache_path=None):
    """The set of datasets present in the CURRENT cache — compared against
    a cached result's own saved signature to detect staleness."""
    path = cache_path or os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return sorted(np.unique(data["lab_test"]).tolist())


def _load_existing_result(run_name: str, expected_signature=None):
    """If a *_results.json for this run already exists (e.g. from a prior
    crashed attempt), load and reuse it instead of retraining from scratch —
    UNLESS its saved dataset_signature doesn't match the current dataset
    composition, in which case it's stale (e.g. computed before BIT/XJTU/RWTH
    were added) and must be retrained rather than silently reused."""
    path = os.path.join(config.OUTPUT_DIR, f"{run_name}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        result = json.load(fh)

    if expected_signature is not None:
        saved_sig = result.get("dataset_signature")
        if saved_sig is not None and sorted(saved_sig) != sorted(expected_signature):
            print(f"[resume] {run_name}: existing result is STALE "
                  f"(computed on datasets {saved_sig}, current composition is "
                  f"{expected_signature}) — retraining instead of reusing it.")
            return None
        if saved_sig is None:
            print(f"[resume] {run_name}: existing result has no dataset_signature "
                  f"(from an older code version) — cannot verify it's current. "
                  f"Retraining to be safe rather than risk reusing a stale result.")
            return None

    print(f"[resume] {run_name}: found existing result, signature matches, skipping retraining")
    return result


def _train_one_resumable(**kwargs):
    run_name = kwargs.get("run_name")
    cache_path = kwargs.get("cache_path")
    expected_sig = _current_dataset_signature(cache_path)
    existing = _load_existing_result(run_name, expected_sig) if run_name else None
    if existing is not None:
        return existing
    return train_one(**kwargs)


def run_baselines_section():
    print("\n" + "=" * 60)
    print("SECTION 1: Baselines (linear regression, SVR, XGBoost)")
    print("=" * 60)
    results = []
    expected_sig = _current_dataset_signature()

    existing = _load_existing_result("baseline_linear", expected_sig)
    results.append(existing if existing else run_baseline("linear", LinearRegression()))

    existing = _load_existing_result("baseline_svr", expected_sig)
    if existing:
        results.append(existing)
    else:
        results.append(run_baseline("svr", SVR(kernel="rbf", C=1.0, epsilon=0.01), max_train_samples=20000))

    if _HAS_XGBOOST:
        existing = _load_existing_result("baseline_xgboost", expected_sig)
        if existing:
            results.append(existing)
        else:
            xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=config.RANDOM_SEED, n_jobs=-1)
            results.append(run_baseline("xgboost", xgb_model))
    else:
        print("[baseline:xgboost] skipped — install with: pip install xgboost")
    return results


def run_ablations_section():
    print("\n" + "=" * 60)
    print("SECTION 2: Architecture ablations + GRU baseline")
    print("=" * 60)
    results = []
    for variant in ["full", "no_attention", "no_multiscale", "plain_lstm", "gru"]:
        r = _train_one_resumable(model_variant=variant, run_name=f"ablation_{variant}")
        results.append(r)
    return results


def run_conformal_ablation_section():
    print("\n" + "=" * 60)
    print("SECTION 3: Conformal calibration ablation (groupwise vs pooled vs normalized)")
    print("=" * 60)
    results = []
    for mode in ["groupwise", "pooled", "normalized_groupwise"]:
        r = _train_one_resumable(model_variant="full", conformal_mode=mode,
                                 run_name=f"conformal_{mode}")
        results.append(r)
    return results


def run_window_size_section():
    print("\n" + "=" * 60)
    print("SECTION 4: Window-size ablation")
    print("=" * 60)
    results = []
    expected_sig = _current_dataset_signature()  # dataset composition doesn't depend on window size
    for window in [10, 20, 30, 50]:
        run_name = f"window_{window}"
        existing = _load_existing_result(run_name, expected_sig)
        if existing is not None:
            results.append(existing)
            continue
        print(f"\n[window={window}] preprocessing...")
        cache_path = preprocessing.run(window=window)
        r = train_one(cache_path=cache_path, model_variant="full", run_name=run_name)
        results.append(r)
    return results


def run_multiseed_section():
    print("\n" + "=" * 60)
    print("SECTION 5: Multi-seed runs (for mean ± std reporting)")
    print("=" * 60)
    results = []
    for seed in SEEDS:
        r = _train_one_resumable(model_variant="full", seed=seed, run_name=f"multiseed_{seed}")
        results.append(r)

    maes = [r["test_mae"] for r in results]
    rmses = [r["test_rmse"] for r in results]
    coverages = [r["overall_coverage"] for r in results]
    print(f"\n[multiseed summary] MAE = {np.mean(maes):.5f} ± {np.std(maes):.5f}")
    print(f"[multiseed summary] RMSE = {np.mean(rmses):.5f} ± {np.std(rmses):.5f}")
    print(f"[multiseed summary] Overall coverage = {np.mean(coverages):.2%} ± {np.std(coverages):.2%}")
    print(f"[multiseed summary] NOTE: coverage typically varies more across seeds than MAE/RMSE does — "
          f"report this spread explicitly rather than a single run's coverage number.")
    return results


def run_lodo_section():
    print("\n" + "=" * 60)
    print("SECTION 6: Leave-one-dataset-out generalization test")
    print("=" * 60)
    datasets = ("NASA", "CALCE", "Oxford", "MIT", "BIT", "XJTU", "RWTH")
    results = []
    for d in datasets:
        run_name = f"lodo_{d}"
        path = os.path.join(config.OUTPUT_DIR, f"{run_name}_results.json")
        existing = None
        if os.path.exists(path):
            with open(path) as fh:
                cached = json.load(fh)
            expected_training_sig = sorted(x for x in datasets if x != d)
            saved_sig = cached.get("training_dataset_signature")
            if saved_sig is not None and sorted(saved_sig) == expected_training_sig:
                print(f"[resume] {run_name}: found existing result, signature matches, skipping retraining")
                existing = cached
            else:
                print(f"[resume] {run_name}: existing result is STALE "
                      f"(trained on {saved_sig}, expected {expected_training_sig}) — retraining.")
        results.append(existing if existing is not None else run_lodo(d))
    return results


def write_summary(all_results: dict):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(config.OUTPUT_DIR, "experiment_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n[experiments] full results saved to {summary_path}")

    md_path = os.path.join(config.OUTPUT_DIR, "experiment_summary.md")
    with open(md_path, "w") as fh:
        fh.write("# Experiment Summary\n\n")

        fh.write("## Baselines vs proposed model\n\n")
        fh.write("| Model | Test MAE | Test RMSE |\n|---|---|---|\n")
        for r in all_results.get("baselines", []):
            fh.write(f"| {r['run_name']} | {r['test_mae']:.5f} | {r['test_rmse']:.5f} |\n")
        for r in all_results.get("ablations", []):
            if r["model_variant"] == "full":
                fh.write(f"| **{r['run_name']} (proposed)** | **{r['test_mae']:.5f}** | **{r['test_rmse']:.5f}** |\n")

        fh.write("\n## Architecture ablations\n\n")
        fh.write("| Variant | Test MAE | Test RMSE | Overall coverage |\n|---|---|---|---|\n")
        for r in all_results.get("ablations", []):
            fh.write(f"| {r['model_variant']} | {r['test_mae']:.5f} | {r['test_rmse']:.5f} | {r['overall_coverage']:.2%} |\n")

        fh.write("\n## Conformal calibration ablation\n\n")
        fh.write("| Mode | Overall coverage | Per-dataset coverage |\n|---|---|---|\n")
        for r in all_results.get("conformal_ablation", []):
            per_ds = ", ".join(f"{k}={v:.1%}" for k, v in r["per_dataset_coverage"].items())
            fh.write(f"| {r['conformal_mode']} | {r['overall_coverage']:.2%} | {per_ds} |\n")

        fh.write("\n## Window-size ablation\n\n")
        fh.write("| Window (cycles) | Test MAE | Test RMSE |\n|---|---|---|\n")
        for r in all_results.get("window_size", []):
            fh.write(f"| {r['run_name'].replace('window_', '')} | {r['test_mae']:.5f} | {r['test_rmse']:.5f} |\n")

        fh.write("\n## Multi-seed results (mean ± std)\n\n")
        ms = all_results.get("multiseed", [])
        if ms:
            maes = [r["test_mae"] for r in ms]
            rmses = [r["test_rmse"] for r in ms]
            coverages = [r["overall_coverage"] for r in ms]
            fh.write(f"MAE = {np.mean(maes):.5f} ± {np.std(maes):.5f} (n={len(ms)} seeds)\n\n")
            fh.write(f"RMSE = {np.mean(rmses):.5f} ± {np.std(rmses):.5f} (n={len(ms)} seeds)\n\n")
            fh.write(f"Overall coverage = {np.mean(coverages):.2%} ± {np.std(coverages):.2%} (n={len(ms)} seeds)\n\n")
            fh.write("Note: coverage varies more across seeds than MAE/RMSE — report the spread, "
                     "not a single run's number, when discussing conformal calibration reliability.\n\n")

        fh.write("\n## Leave-one-dataset-out generalization\n\n")
        fh.write("| Held-out dataset | Test MAE | Test RMSE | Coverage | n test samples |\n|---|---|---|---|---|\n")
        for r in all_results.get("lodo", []):
            fh.write(f"| {r['held_out_dataset']} | {r['test_mae']:.5f} | {r['test_rmse']:.5f} | "
                     f"{r['coverage']:.2%} | {r['n_test_samples']} |\n")

    print(f"[experiments] markdown table saved to {md_path}")


def run_all():
    all_results = {}
    all_results["baselines"] = run_baselines_section()
    all_results["ablations"] = run_ablations_section()
    all_results["conformal_ablation"] = run_conformal_ablation_section()
    all_results["window_size"] = run_window_size_section()
    all_results["multiseed"] = run_multiseed_section()
    all_results["lodo"] = run_lodo_section()
    write_summary(all_results)
    return all_results


if __name__ == "__main__":
    run_all()