"""
clear_stale_predictions.py

Targeted fix for a specific problem: diagnose_xjtu_anomaly() in
remaining_analysis.py just showed the EXACT same 79-sample anomaly, same
test-sample count (2387), same index range (2229-2385) as the very first
pre-fix diagnostic run -- meaning ablation_full's and/or
conformal_normalized_groupwise's *_predictions.npz files were never
regenerated after the xjtu_loader.py calibration-block fix, even though
experiment_summary.md's aggregate MAE/RMSE numbers looked fine.

This is narrower than clear_stale_xjtu_cache.py -- it does NOT touch the
preprocessing cache or the other 5 ablations/3 baselines/other 2 conformal
modes/4 window sizes/3 multiseed runs/7 LODO results, since nothing in
this session's evidence suggests those are stale. It only targets the two
runs implicated by the anomaly check: ablation_full and
conformal_normalized_groupwise.

What it does:
  1. Prints the modification time of every *ablation_full*predictions.npz
     and *conformal_normalized_groupwise*predictions.npz file found,
     compared against xjtu_loader.py's modification time, so you can SEE
     the staleness directly before anything is deleted.
  2. Deletes those predictions files AND their matching *_results.json
     (deleting only the .npz would leave a stale-but-still-valid-looking
     results.json that _load_existing_result() would happily resume from,
     defeating the point).
  3. Leaves every other file in outputs/ and the preprocessing cache
     completely untouched.

Usage:
    python clear_stale_predictions.py --dry-run   # shows what would be
                                                    # deleted and the
                                                    # timestamp comparison,
                                                    # deletes nothing
    python clear_stale_predictions.py              # asks for confirmation
    python clear_stale_predictions.py --yes        # no confirmation prompt
"""

import os
import sys
import glob
import time

import config  # your existing project config.py

TARGET_RUN_NAMES = ["ablation_full", "conformal_normalized_groupwise"]


def _mtime_str(path):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))


def _find_targets():
    output_dir = config.OUTPUT_DIR
    if not os.path.isdir(output_dir):
        print(f"  [!] config.OUTPUT_DIR ({output_dir}) not found -- nothing to do.")
        return [], None

    loader_path = None
    try:
        import xjtu_loader
        loader_path = os.path.abspath(xjtu_loader.__file__)
    except ImportError:
        print("  [!] could not import xjtu_loader.py to compare timestamps -- "
              "continuing without the staleness comparison, deletion targets "
              "are unaffected by this.")

    targets = []
    for run_name in TARGET_RUN_NAMES:
        npz_matches = sorted(glob.glob(os.path.join(output_dir, f"*{run_name}*predictions.npz")))
        json_path = os.path.join(output_dir, f"{run_name}_results.json")
        json_matches = [json_path] if os.path.exists(json_path) else []

        for p in npz_matches + json_matches:
            targets.append(p)

    return targets, loader_path


def main():
    dry_run = "--dry-run" in sys.argv
    skip_confirm = "--yes" in sys.argv or dry_run

    targets, loader_path = _find_targets()

    if not targets:
        print("\nNo prediction/results files found for ablation_full or "
              "conformal_normalized_groupwise. Either they were already "
              "cleared, or config.OUTPUT_DIR doesn't match what's expected "
              "-- check that path in config.py if this is unexpected.")
        return

    print(f"\nFound {len(targets)} file(s) tied to ablation_full / "
          f"conformal_normalized_groupwise:\n")

    loader_mtime = os.path.getmtime(loader_path) if loader_path and os.path.exists(loader_path) else None
    if loader_mtime is not None:
        print(f"  xjtu_loader.py last modified: {_mtime_str(loader_path)}  "
              f"(your fix should be at or after this time)\n")

    for t in targets:
        line = f"  {t}\n    modified: {_mtime_str(t)}"
        if loader_mtime is not None:
            if os.path.getmtime(t) < loader_mtime:
                line += "  <-- STALE: predates the xjtu_loader.py fix"
            else:
                line += "  (newer than xjtu_loader.py -- may already be fine; " \
                        "deleting anyway to force a clean retrain, since a " \
                        "predictions file being newer doesn't guarantee it " \
                        "was actually retrained rather than resumed from an " \
                        "older results.json)"
        print(line)

    if dry_run:
        print("\n[dry run] nothing was actually deleted. Re-run without "
              "--dry-run to actually clear these.")
        return

    if not skip_confirm:
        resp = input(f"\nDelete these {len(targets)} file(s)? This forces "
                      f"ablation_full and conformal_normalized_groupwise to "
                      f"retrain on your next experiments.py run. [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted -- nothing deleted.")
            return

    n_deleted = 0
    n_failed = 0
    for t in targets:
        try:
            os.remove(t)
            n_deleted += 1
        except OSError as e:
            print(f"  [!] failed to delete {t}: {e}")
            n_failed += 1

    print(f"\nDeleted {n_deleted} file(s)"
          + (f", {n_failed} failed" if n_failed else "") + ".")
    print("Now run: python experiments.py")
    print("Expect [resume] lines for every run EXCEPT ablation_full and "
          "conformal_normalized_groupwise, which should show real training "
          "progress this time.")


if __name__ == "__main__":
    main()
