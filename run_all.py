"""
End-to-end pipeline driver: the run order in REPRODUCIBILITY.md as executable code.

SAFE BY DEFAULT: running this with no arguments only PRINTS the plan (what
would run, and whether its output already exists) -- it does not retrain or
overwrite anything. To avoid silently overwriting a verified result or retraining without a
reason, every step is skipped by default if its primary output file already
exists. Use --execute
to actually run, and --force to rerun steps whose output already exists
(use deliberately, not as a default habit -- a forced rerun of a GPU
training step takes real time and will overwrite a result someone may be
relying on being stable).

Run:
    python run_all.py                 # print the plan only, run nothing
    python run_all.py --execute       # run every step whose output is missing
    python run_all.py --execute --only preprocessing train   # run specific steps
    python run_all.py --execute --force --only leave_one_out # force a rerun
"""
import argparse
import os
import subprocess
import sys

import config

STEPS = [
    # (name, command, primary_output_to_check_for_skip)
    ("preprocessing", [sys.executable, "preprocessing.py"],
     os.path.join(config.CACHE_DIR, "processed.npz")),
    ("train", [sys.executable, "train.py"],
     os.path.join(config.OUTPUT_DIR, "best_results.json")),
    ("experiments", [sys.executable, "experiments.py"],
     os.path.join(config.OUTPUT_DIR, "experiment_summary.json")),
    ("baseline_conformal_coverage", [sys.executable, "baseline_conformal_coverage.py"],
     os.path.join(config.OUTPUT_DIR, "baseline_conformal_results.json")),
    ("persistence_baselines", [sys.executable, "persistence_baselines.py"],
     os.path.join(config.OUTPUT_DIR, "persistence_baselines.json")),
    ("leave_one_out", [sys.executable, "leave_one_out.py"],
     os.path.join(config.OUTPUT_DIR, "lodo_RWTH_results.json")),
    ("hierarchical_calibration", [sys.executable, "hierarchical_calibration_v2.py"],
     os.path.join(config.OUTPUT_DIR, "hierarchical_calibration_v2.json")),
    ("dataset_level_wilcoxon", [sys.executable, "dataset_level_wilcoxon.py"],
     os.path.join(config.OUTPUT_DIR, "dataset_level_wilcoxon.json")),
    ("nested_hp_search", [sys.executable, "run_nested_hp_search_all.py"],
     os.path.join(config.OUTPUT_DIR, "nested_hp_search_results.json")),
    ("mwhc", [sys.executable, "mwhc_calibration.py"],
     os.path.join(config.OUTPUT_DIR, "mwhc_results.json")),
    ("calibration_size_sensitivity", [sys.executable, "calibration_size_sensitivity.py"],
     os.path.join(config.OUTPUT_DIR, "calibration_size_sensitivity.json")),
    ("train_cqr", [sys.executable, "train_cqr.py"],
     os.path.join(config.OUTPUT_DIR, "cqr_groupwise_results.json")),
    ("predictor_consistent_recalibration", [sys.executable, "predictor_consistent_recalibration.py"],
     os.path.join(config.OUTPUT_DIR, "predictor_consistent_recalibration.json")),
    ("xjtu_bridging_isolation", [sys.executable, "xjtu_bridging_isolation.py"],
     os.path.join(config.OUTPUT_DIR, "xjtu_bridging_isolation.json")),
    ("xjtu_bridging_sensitivity", [sys.executable, "xjtu_bridging_sensitivity.py"],
     os.path.join(config.OUTPUT_DIR, "xjtu_bridging_sensitivity.json")),
    ("verify_rwth", [sys.executable, "verify_rwth_v2.py"],
     os.path.join(config.OUTPUT_DIR, "rwth_verification_v2.json")),
    # The registry parses this log's summary table, so the step writes it explicitly.
    ("verify_calce", [sys.executable, "verify_calce_v2.py"],
     os.path.join("logs", "verify_calce_v2.log"), os.path.join("logs", "verify_calce_v2.log")),
    ("repeated_split", [sys.executable, "repeated_split_experiment.py"],
     os.path.join(config.OUTPUT_DIR, "repeated_split_summary.json")),
    ("deterministic_predictor", [sys.executable, "deterministic_predictor_comparison.py"],
     os.path.join(config.OUTPUT_DIR, "deterministic_predictor_results.json")),
    ("deployability", [sys.executable, "deployability_benchmark.py"],
     os.path.join(config.OUTPUT_DIR, "deployability_results.json")),
    ("results_registry", [sys.executable, "make_results_registry.py"],
     None),  # always safe/cheap to rerun -- just aggregates existing JSON
]

# Steps that train a model on the GPU and take non-trivial wall-clock time --
# flagged so --execute without --only prints a warning before launching all
# of them back to back.
GPU_HEAVY = {"train", "experiments", "leave_one_out", "repeated_split", "train_cqr"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="actually run steps (default: print plan only)")
    parser.add_argument("--force", action="store_true", help="rerun a step even if its output already exists")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these step names")
    args = parser.parse_args()

    steps = [s for s in STEPS if args.only is None or s[0] in args.only]
    if args.only:
        unknown = set(args.only) - {s[0] for s in STEPS}
        if unknown:
            print(f"[!] unknown step name(s): {unknown}. Valid: {[s[0] for s in STEPS]}")
            return

    gpu_heavy_in_plan = [s[0] for s in steps if s[0] in GPU_HEAVY]
    if args.execute and gpu_heavy_in_plan and not args.only:
        print(f"[!] about to run {len(gpu_heavy_in_plan)} GPU-heavy step(s) back to back: "
              f"{gpu_heavy_in_plan}. This can take many hours. Re-run with --only to run one at "
              f"a time if that's not what you want.")

    for name, cmd, output_check, *stdout_log in steps:
        exists = output_check is not None and os.path.exists(output_check)
        skip = exists and not args.force
        status = "SKIP (output exists)" if skip else ("RUN" if args.execute else "WOULD RUN")
        print(f"[{status}] {name:40s} -> {' '.join(cmd)}" + (f" > {stdout_log[0]}" if stdout_log else ""))
        if args.execute and not skip:
            if stdout_log:
                with open(stdout_log[0], "w", encoding="utf-8") as fh:
                    result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
            else:
                result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[!] {name} exited with code {result.returncode} -- stopping (fix before continuing).")
                return

    if not args.execute:
        print("\n(dry run only -- pass --execute to actually run these steps)")


if __name__ == "__main__":
    main()
