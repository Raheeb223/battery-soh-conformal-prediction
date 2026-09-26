"""
Delete all derived artefacts so the next run retrains everything from scratch.

experiments.py resumes from existing outputs/*_results.json files. Its
staleness check compares only the SET of dataset names present, so it cannot
detect row-level changes to the data (for example toggling the XJTU
correction). Use this script whenever the data pipeline changes.

Deletes:
  1. the entire cache directory (config.CACHE_DIR), all window variants;
  2. every *_results.json and *_predictions.npz in config.OUTPUT_DIR;
  3. every *_model.pt checkpoint in config.OUTPUT_DIR.

Does not touch raw data, source code, config.py, or
experiment_summary.md/json (these are overwritten by the next experiments.py
run).

Afterwards run `python preprocessing.py` then `python experiments.py`.

Usage:
    python clear_everything.py --dry-run   # list what would be deleted
    python clear_everything.py             # asks for confirmation
    python clear_everything.py --yes       # no confirmation prompt
"""

import os
import sys
import glob
import shutil

import config  # project config.py


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
