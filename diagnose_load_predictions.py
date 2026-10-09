"""
diagnose_load_predictions.py

Standalone diagnostic -- does NOT modify remaining_analysis.py. Run this
from your project root (same folder as config.py) to see EXACTLY what
_load_predictions()'s glob pattern resolves to for ablation_full and
conformal_normalized_groupwise, with modification timestamps, so we can
tell definitively whether item 4's persistent 2387/-7.224 result is:

  (a) reading a stale file (a leftover match the glob picked up that
      isn't the one experiments.py just wrote), or
  (b) genuinely reading the freshly-retrained file, meaning the anomaly
      has a different, real cause we haven't found yet.

This mirrors remaining_analysis.py's own _load_predictions() glob
pattern exactly, so if this script finds one file, so did
remaining_analysis.py -- and if it finds MORE than one, that's very
likely the bug (candidates[0] may not be the newest one).

Run:
    python diagnose_load_predictions.py
"""

import os
import glob
import time

import numpy as np
import config

RUN_NAMES = ["ablation_full", "conformal_normalized_groupwise"]


def _mtime_str(path):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))


def main():
    output_dir = config.OUTPUT_DIR
    print(f"config.OUTPUT_DIR = {output_dir}\n")

    for run_name in RUN_NAMES:
        pattern = os.path.join(output_dir, f"*{run_name}*predictions.npz")
        candidates = glob.glob(pattern)
        print(f"=== {run_name} ===")
        print(f"glob pattern: {pattern}")
        print(f"matches found: {len(candidates)}")
        for c in candidates:
            print(f"  {c}")
            print(f"    modified: {_mtime_str(c)}  size: {os.path.getsize(c)} bytes")

        if not candidates:
            print("  [!] no matches -- _load_predictions() would return None here.")
            print()
            continue

        chosen = candidates[0]
        print(f"\n  _load_predictions() would pick candidates[0] = {chosen}")
        if len(candidates) > 1:
            print(f"  [!!!] MULTIPLE MATCHES FOUND -- glob does not guarantee "
                  f"newest-first order. candidates[0] may NOT be the file "
                  f"experiments.py just wrote. This is very likely the bug.")

        # actually load it and report what's inside, same fields
        # diagnose_xjtu_anomaly() checks, so we can compare directly
        # against remaining_analysis.py's printed output
        data = np.load(chosen, allow_pickle=True)
        required = {"y_test", "mean_pred", "lab_test"}
        missing = required - set(data.files)
        if missing:
            print(f"  [!] missing expected keys: {missing} -- available: {data.files}")
            print()
            continue

        lab_test = data["lab_test"]
        y_test = data["y_test"]
        mask = lab_test == "XJTU"
        n_xjtu = int(mask.sum())
        print(f"  Total test samples (all datasets): {len(y_test)}")
        print(f"  XJTU test samples in this file: {n_xjtu}")
        if n_xjtu > 0:
            yt = y_test[mask]
            print(f"  XJTU true-value range: [{yt.min():.3f}, {yt.max():.3f}]")
        print()


if __name__ == "__main__":
    main()
