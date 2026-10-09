"""
clear_everything.py

Every targeted cleanup so far (clear_stale_predictions.py,
clear_default_cache_and_predictions.py) assumed only a FEW specific runs
needed clearing. That assumption is now confirmed wrong: experiment_summary.md
shows conformal_groupwise, conformal_pooled, window_10, and window_20 are
all byte-identical to runs from BEFORE the sentinel-value fix, while
ablation_full, conformal_normalized_groupwise, window_30, and window_50
changed -- proving the dataset_signature staleness check cannot detect
this class of fix at all, since it only compares the SET of dataset
NAMES present (unchanged: XJTU is still 'in' the data, just with 93%
fewer satellite-cell rows), never row-level content.

This affects EVERY run that depends on cache/processed.npz or a
window-specific cache, and EVERY LODO run (build_lodo_cache() has the
same name-only signature check). There is no reliable way to know which
specific runs are stale without just clearing everything.

This script deletes, with no exceptions:
  1. The entire cache directory (config.CACHE_DIR) -- all window
     variants, not just the default.
  2. Every *_results.json and *_predictions.npz in config.OUTPUT_DIR --
     baselines, all ablations, all conformal modes, all window sizes,
     all multiseed runs, all LODO runs.
  3. Every *_model.pt checkpoint in config.OUTPUT_DIR (train_one() saves
     these; stale checkpoints aren't used by anything downstream we've
     seen, but there's no reason to keep them either).

It does NOT touch: raw dataset folders, xjtu_loader.py or any other
loader/model/training code, config.py, or experiment_summary.md/json
themselves (those get overwritten naturally by the next experiments.py
run).

After this, `python preprocessing.py` followed by `python experiments.py`
will regenerate EVERYTHING from scratch -- expect the full ~15-run
training suite (per experiments.py's own docstring) to take a while.
There is no partial/fast path this time; that's the point.

Usage:
    python clear_everything.py --dry-run
    python clear_everything.py
    python clear_everything.py --yes
"""

import os
import sys
import glob
import shutil

import config  # your existing project config.py


def _find_targets():
    targets = []

    cache_dir = config.CACHE_DIR
    if os.path.isdir(cache_dir):
        targets.append(("dir", cache_dir))
    else:
        print(f"  [!] config.CACHE_DIR ({cache_dir}) doesn't exist -- nothing to clear there.")

    output_dir = config.OUTPUT_DIR
    if os.path.isdir(output_dir):
        for pattern in ("*_results.json", "*_predictions.npz", "*_model.pt"):
            targets += [("file", p) for p in sorted(glob.glob(os.path.join(output_dir, pattern)))]
    else:
        print(f"  [!] config.OUTPUT_DIR ({output_dir}) doesn't exist -- nothing to clear there.")

    return targets


def main():
    dry_run = "--dry-run" in sys.argv
    skip_confirm = "--yes" in sys.argv or dry_run

    targets = _find_targets()

    if not targets:
        print("Nothing found to delete.")
        return

    n_files = sum(1 for kind, _ in targets if kind == "file")
    n_dirs = sum(1 for kind, _ in targets if kind == "dir")
    print(f"\n{'Would delete' if dry_run else 'Will delete'}: "
          f"{n_dirs} directory (and everything in it) + {n_files} file(s)\n")
    for kind, path in targets:
        tag = "[DIR] " if kind == "dir" else "[file]"
        print(f"  {tag} {path}")

    if dry_run:
        print("\n[dry run] nothing deleted. Re-run without --dry-run to actually clear these.")
        return

    if not skip_confirm:
        resp = input(f"\nDelete all of the above? This forces EVERY run in "
                      f"experiments.py to retrain from scratch -- the full "
                      f"~15-run suite, no resuming anything. [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted -- nothing deleted.")
            return

    n_deleted = 0
    n_failed = 0
    for kind, path in targets:
        try:
            if kind == "dir":
                shutil.rmtree(path)
            else:
                os.remove(path)
            n_deleted += 1
        except OSError as e:
            print(f"  [!] failed to delete {path}: {e}")
            n_failed += 1

    print(f"\nDeleted {n_deleted} item(s)" + (f", {n_failed} failed" if n_failed else "") + ".")
    print("\nNext steps:")
    print("  1. python preprocessing.py")
    print("     -- confirm the '[XJTU] ... removed N sentinel/calibration rows' "
          "lines appear for all 8 satellite cells.")
    print("  2. python experiments.py")
    print("     -- expect EVERY section to show real training, not [resume] lines.")
    print("  3. python diagnose_cache_directly.py")
    print("     -- final confirmation the anomaly is gone from the cache itself.")


if __name__ == "__main__":
    main()
