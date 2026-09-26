"""
XJTU data-correction sensitivity analysis: the full pipeline with and without
the XJTU sentinel-value correction.

Steps:
  1. Back up the current outputs/ and cache/ (the results with the correction).
  2. Disable the correction via the XJTU_APPLY_SENTINEL_FIX environment
     variable, for this script's subprocesses only.
  3. Clear all cached results (clear_everything.py --yes) and rerun
     preprocessing.py and experiments.py.
  4. Save the results without the correction to outputs_WITHOUT_fix/ and
     cache_WITHOUT_fix/.
  5. Restore the original outputs/ and cache/ from the backup.
  6. Print and save a before/after comparison of both experiment_summary.json
     files.

Run from the repository root:
    python run_xjtu_sensitivity_experiment.py

This takes as long as a full experiments.py run (every section retrains).
"""

import os
import sys
import json
import shutil
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")
CACHE = os.path.join(PROJECT_ROOT, "cache")
OUTPUTS_WITH = os.path.join(PROJECT_ROOT, "outputs_WITH_fix")
CACHE_WITH = os.path.join(PROJECT_ROOT, "cache_WITH_fix")
OUTPUTS_WITHOUT = os.path.join(PROJECT_ROOT, "outputs_WITHOUT_fix")
CACHE_WITHOUT = os.path.join(PROJECT_ROOT, "cache_WITHOUT_fix")


def run_step(description, cmd, env=None):
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        print(f"\n[!] '{' '.join(cmd)}' exited with code {result.returncode}. "
              f"Stopping here rather than continuing on top of a failed step.")
        sys.exit(1)


def main():
    # --- Safety checks before touching anything ---
    if not os.path.isdir(OUTPUTS) or not os.path.isdir(CACHE):
        print(f"[!] Expected outputs/ and cache/ in {PROJECT_ROOT} -- "
              f"run this from your project root (same folder as config.py).")
        sys.exit(1)
    for existing in (OUTPUTS_WITH, CACHE_WITH, OUTPUTS_WITHOUT, CACHE_WITHOUT):
        if os.path.exists(existing):
            print(f"[!] {existing} already exists -- refusing to overwrite a "
                  f"prior run's backup/output. Delete or rename it manually "
                  f"first if you want to re-run this from scratch.")
            sys.exit(1)

    # --- Step 1: back up current ("with correction") results ---
    print(f"\n{'='*70}")
    print("STEP 1: Backing up current (with-correction) outputs/ and cache/")
    print(f"{'='*70}")
    shutil.copytree(OUTPUTS, OUTPUTS_WITH)
    shutil.copytree(CACHE, CACHE_WITH)
    print(f"  Backed up to {OUTPUTS_WITH} and {CACHE_WITH}")

    # --- Step 2-3: wipe, disable fix, rerun full pipeline ---
    # Only this subprocess environment gets the override -- the parent
    # shell's environment variables are never touched.
    env_without_fix = os.environ.copy()
    env_without_fix["XJTU_APPLY_SENTINEL_FIX"] = "0"

    run_step("Clearing all cached results", [sys.executable, "clear_everything.py", "--yes"])
    run_step("Running preprocessing.py WITHOUT the XJTU fix",
              [sys.executable, "preprocessing.py"], env=env_without_fix)
    run_step("Running experiments.py WITHOUT the XJTU fix (this is the long step)",
              [sys.executable, "experiments.py"], env=env_without_fix)

    # --- Step 4: save the without-fix results under their own name ---
    print(f"\n{'='*70}")
    print("STEP 4: Saving 'without correction' results")
    print(f"{'='*70}")
    os.rename(OUTPUTS, OUTPUTS_WITHOUT)
    os.rename(CACHE, CACHE_WITHOUT)
    print(f"  Saved to {OUTPUTS_WITHOUT} and {CACHE_WITHOUT}")

    # --- Step 5: restore the original results ---
    print(f"\n{'='*70}")
    print("STEP 5: Restoring your original (with-correction) outputs/ and cache/")
    print(f"{'='*70}")
    os.rename(OUTPUTS_WITH, OUTPUTS)
    os.rename(CACHE_WITH, CACHE)
    print(f"  Restored -- outputs/ and cache/ are back to your final, "
          f"manuscript-matching state.")

    # --- Step 6: compute and print the before/after comparison directly ---
    print(f"\n{'='*70}")
    print("STEP 6: Before/after comparison")
    print(f"{'='*70}")
    compare_summaries()


def _load_summary(path):
    if not os.path.exists(path):
        print(f"  [!] {path} not found -- experiments.py may not have "
              f"completed successfully.")
        return None
    with open(path) as fh:
        return json.load(fh)


def compare_summaries():
    with_path = os.path.join(OUTPUTS, "experiment_summary.json")
    without_path = os.path.join(OUTPUTS_WITHOUT, "experiment_summary.json")
    with_summary = _load_summary(with_path)
    without_summary = _load_summary(without_path)
    if with_summary is None or without_summary is None:
        print("  Cannot build automatic comparison -- inspect the raw "
              f"files in {OUTPUTS_WITHOUT} manually instead.")
        return

    report_lines = []
    report_lines.append("XJTU SENTINEL-VALUE FIX: BEFORE/AFTER SENSITIVITY COMPARISON")
    report_lines.append("=" * 70)

    def _get(summary, section, run_name, key, default=None):
        for r in summary.get(section, []):
            if r.get("run_name") == run_name:
                return r.get(key, default)
        return default

    checks = [
        ("baselines", "baseline_linear", "test_mae", "Linear MAE"),
        ("baselines", "baseline_svr", "test_mae", "SVR MAE"),
        ("baselines", "baseline_xgboost", "test_mae", "XGBoost MAE"),
        ("ablations", "ablation_full", "test_mae", "Proposed model MAE"),
        ("ablations", "ablation_full", "test_rmse", "Proposed model RMSE"),
        ("ablations", "ablation_full", "overall_coverage", "Proposed model overall coverage"),
    ]
    report_lines.append(f"\n{'Metric':40s} {'WITH fix':>12s} {'WITHOUT fix':>12s} {'Delta':>12s}")
    for section, run_name, key, label in checks:
        w = _get(with_summary, section, run_name, key)
        wo = _get(without_summary, section, run_name, key)
        if w is None or wo is None:
            report_lines.append(f"{label:40s} {'n/a':>12s} {'n/a':>12s}")
            continue
        delta = wo - w
        report_lines.append(f"{label:40s} {w:12.5f} {wo:12.5f} {delta:+12.5f}")

    # Per-dataset XJTU-specific comparison, groupwise conformal mode
    report_lines.append("\nPer-dataset XJTU-specific comparison (conformal_groupwise):")
    for summary_name, summary in [("WITH fix", with_summary), ("WITHOUT fix", without_summary)]:
        xjtu_mae = _get(summary, "conformal_ablation", "conformal_groupwise", "per_dataset_mae", {})
        xjtu_cov = _get(summary, "conformal_ablation", "conformal_groupwise", "per_dataset_coverage", {})
        xjtu_mae_val = xjtu_mae.get("XJTU") if isinstance(xjtu_mae, dict) else None
        xjtu_cov_val = xjtu_cov.get("XJTU") if isinstance(xjtu_cov, dict) else None
        report_lines.append(f"  {summary_name:12s}  XJTU MAE={xjtu_mae_val}  XJTU coverage={xjtu_cov_val}")

    # LODO XJTU comparison
    report_lines.append("\nLODO (XJTU held out) comparison:")
    for summary_name, summary in [("WITH fix", with_summary), ("WITHOUT fix", without_summary)]:
        lodo_xjtu = None
        for r in summary.get("lodo", []):
            if r.get("held_out_dataset") == "XJTU":
                lodo_xjtu = r
        if lodo_xjtu:
            report_lines.append(f"  {summary_name:12s}  MAE={lodo_xjtu.get('test_mae')}  "
                               f"RMSE={lodo_xjtu.get('test_rmse')}  "
                               f"coverage={lodo_xjtu.get('coverage')}")
        else:
            report_lines.append(f"  {summary_name:12s}  [not found]")

    output = "\n".join(str(l) for l in report_lines)
    print(output)

    out_path = os.path.join(OUTPUTS, "xjtu_sensitivity_comparison.txt")
    with open(out_path, "w") as fh:
        fh.write(output)
    print(f"\nsaved {out_path}")
    print(f"\nThis comparison, together with {without_path}, constitutes the "
          "full before/after evidence for the XJTU sentinel-value correction's "
          "effect on downstream results.")


if __name__ == "__main__":
    main()
