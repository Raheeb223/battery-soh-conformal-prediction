"""
Fast pre-check: list CALCE file names that look like non-cycling sessions
(confirmed example: "..._self discharge test.xlsx"), across ALL configured
cells -- not just files that happened to produce a flagged outlier row.
Opens no Excel files, so it is safe and cheap to run first.

Run:  python scan_calce_filenames.py
"""

import os

import config
from calce_loader_v2 import NON_CYCLING_KEYWORDS, _is_non_cycling


def main():
    total, hits = 0, []
    for chem, dirs in config.CALCE_DIRS.items():
        for d in dirs:
            cid = os.path.basename(d.rstrip("/\\"))
            if not os.path.isdir(d):
                continue
            for dirpath, _, filenames in os.walk(d):
                for f in filenames:
                    if not f.lower().endswith((".xlsx", ".xls")):
                        continue
                    total += 1
                    if _is_non_cycling(f):
                        hits.append((cid, f))
    print(f"scanned {total} Excel files across {sum(len(v) for v in config.CALCE_DIRS.values())} configured cells")
    print(f"keywords checked: {NON_CYCLING_KEYWORDS}\n")
    if not hits:
        print("no matches -- either there are none, or the keyword list needs another term. "
              "If you know of other non-cycling session types (rest tests, HPPC, EIS, etc.) in "
              "these folders, check their exact file names and add matching terms to "
              "NON_CYCLING_KEYWORDS in calce_loader_v2.py.")
        return
    print(f"{len(hits)} match(es):")
    for cid, f in hits:
        print(f"  {cid}: {f}")


if __name__ == "__main__":
    main()
