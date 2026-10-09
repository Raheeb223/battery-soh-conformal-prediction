"""
locate_groupwise_predictions.py

windowing_bridge_sensitivity.py needs the main (full BiLSTM) model's
groupwise-calibrated lower/upper interval bounds on the TEST set, to
recompute Table 6's coverage on the non-bridging window subset. It
guessed 'cache/processed.npz' and then 'outputs/ablation_full_predictions.npz'
-- both wrong, or at least not confirmed. Rather than guess a third time,
this just lists what you actually have, so the real fix is a one-line
change instead of another guess.

This is READ-ONLY -- it does not modify, train, or compute anything.

Run from your project root:
    python locate_groupwise_predictions.py
"""

import os
import numpy as np

import config


def inspect_npz(path):
    try:
        data = np.load(path, allow_pickle=True)
        fields = list(data.files)
        return fields
    except Exception as e:
        return [f"[could not load: {e}]"]


def main():
    print(f"{'='*78}")
    print(f"Scanning {config.OUTPUT_DIR} and {config.CACHE_DIR} for .npz files")
    print(f"{'='*78}\n")

    candidates = []
    for base_dir in (config.OUTPUT_DIR, config.CACHE_DIR):
        if not os.path.isdir(base_dir):
            continue
        for fname in sorted(os.listdir(base_dir)):
            if fname.endswith(".npz"):
                candidates.append(os.path.join(base_dir, fname))

    if not candidates:
        print("  No .npz files found in either directory.")
        return

    for path in candidates:
        fields = inspect_npz(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"--- {path}  ({size_mb:.1f} MB) ---")
        print(f"    fields: {fields}")

        lower_ish = [f for f in fields if "lower" in f.lower()]
        upper_ish = [f for f in fields if "upper" in f.lower()]
        lab_ish = [f for f in fields if "lab" in f.lower()]
        if lower_ish or upper_ish:
            print(f"    ^ HAS interval-bound-looking fields: lower={lower_ish}, upper={upper_ish}, lab={lab_ish}")
        print()

    print(f"{'='*78}")
    print("Send this full output back. Whichever file has 'lower'/'upper'-type "
          "fields (or something with a similar name -- e.g. 'lower_groupwise', "
          "'ci_lower', 'interval_lower') alongside a 'lab_test' field is the "
          "one windowing_bridge_sensitivity.py should actually load.")


if __name__ == "__main__":
    main()
