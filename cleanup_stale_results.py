"""
One-time cleanup: deletes result files left over from the ORIGINAL
4-dataset run (before BIT/XJTU/RWTH were added) that the resume mechanism
in experiments.py doesn't know are stale, since it only checks "does a
result file exist", not "was it computed on the current dataset
composition".

Deletes: conformal_groupwise, conformal_pooled, window_10/20/30/50,
multiseed_42/123/2024 — the specific sections confirmed stale. Leaves
everything else (baselines, architecture ablations, normalized_groupwise,
LODO) untouched since those are already correct for the 7-dataset run.

Run once:
    python cleanup_stale_results.py
Then re-run experiments.py — it will correctly redo only these sections.
"""

import os
import config

STALE_RUN_NAMES = [
    "conformal_groupwise", "conformal_pooled",
    "window_10", "window_20", "window_30", "window_50",
    "multiseed_42", "multiseed_123", "multiseed_2024",
]

EXTENSIONS = ["_results.json", "_predictions.npz", "_model.pt"]

deleted = []
for run_name in STALE_RUN_NAMES:
    for ext in EXTENSIONS:
        path = os.path.join(config.OUTPUT_DIR, f"{run_name}{ext}")
        if os.path.exists(path):
            os.remove(path)
            deleted.append(path)

print(f"Deleted {len(deleted)} stale files:")
for p in deleted:
    print(f"  {p}")
if not deleted:
    print("  (none found — already clean, or already deleted)")
print("\nNow re-run: python experiments.py")
