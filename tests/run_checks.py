"""
Run the verification checks and print a one-line result for each.

    python tests/run_checks.py          # fast checks (synthetic data, about a minute) + the
                                        # checkpoint check if model outputs are present
    python tests/run_checks.py --full   # also the data-layer check, which needs the raw datasets
                                        # and reruns preprocessing twice (tens of minutes)

Each check is a standalone script that exits with status 1 on failure; this runner exits with
status 1 if any check fails. Run from the repository root.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CHECKS = [
    # (script, what it verifies, files it needs, only with --full)
    ("check_conformal.py", "split-conformal coverage and calibration quantile", [], False),
    ("check_hierarchical_calibration.py", "DerSimonian-Laird estimator vs. hand calculation", [], False),
    ("check_hgc_cp.py", "HGC-CP weighting properties", [], False),
    ("check_hgc_cp_split_equivalence.py", "fast HGC-CP path is bit-identical", [], False),
    ("check_evbattery_loader.py", "field-data loader refuses unsafe pickles", [], False),
    ("check_checkpoint_reproduction.py", "saved model reproduces the reported MAE",
     ["cache/processed.npz", "outputs/conformal_groupwise_model.pt",
      "outputs/conformal_groupwise_results.json"], False),
    ("check_data_layer.py", "cells, splits, windows; pipeline determinism", [], True),
]


def main():
    parser = argparse.ArgumentParser(description="Run the verification checks.")
    parser.add_argument("--full", action="store_true", help="also run the slow data-layer check")
    args = parser.parse_args()

    rows, failed = [], False
    for script, what, needs, full_only in CHECKS:
        missing = [n for n in needs if not os.path.exists(os.path.join(ROOT, n))]
        if full_only and not args.full:
            rows.append((script, what, "skipped (use --full)", ""))
            continue
        if missing:
            rows.append((script, what, "skipped (missing " + ", ".join(missing) + ")", ""))
            continue
        start = time.time()
        result = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=ROOT,
                                capture_output=True, text=True)
        summary = next((line.strip() for line in reversed(result.stdout.splitlines())
                        if "passed" in line), "")
        status = "PASS" if result.returncode == 0 else "FAIL"
        failed |= result.returncode != 0
        rows.append((script, what, status, f"{summary}  ({time.time() - start:.0f} s)"))
        if result.returncode != 0:
            print(result.stdout[-3000:], result.stderr[-3000:], sep="\n")

    width = max(len(r[0]) for r in rows)
    for script, what, status, detail in rows:
        print(f"{status:<6} {script:<{width}}  {what}  {detail}" if status in ("PASS", "FAIL")
              else f"{'-':<6} {script:<{width}}  {what}  {status}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
