"""
RWTH dataset has 2216+ CSVs with structured filenames (fields separated by
'='). Before writing any parser, we need to know: how many distinct
physical cells are actually behind these files, which files are capacity-
cycling data vs. EIS (impedance) tests that should be excluded, and whether
a README/documentation file exists anywhere in the archive.

This ONLY reads filenames (fast) — no file contents are opened.
"""

import os
import re
from collections import Counter

ROOT = r"C:\Users\User\PycharmProjects\RWTH-2021-04545_818642\Rawdata\Rohdaten"

# ---- 1. look for any documentation files anywhere in the archive ----
print("=== Searching for README/documentation files ===")
doc_root = r"C:\Users\User\PycharmProjects\RWTH-2021-04545_818642"
found_docs = []
for dirpath, _, filenames in os.walk(doc_root):
    for f in filenames:
        if any(kw in f.lower() for kw in ["readme", "documentation", ".pdf", ".txt", ".md"]) \
           and "Rohdaten" not in dirpath:
            found_docs.append(os.path.join(dirpath, f))
if found_docs:
    for d in found_docs:
        print(f"  FOUND: {d}")
else:
    print("  none found outside the Rohdaten data folder")

# ---- 2. parse all filenames (CSV only, skip the redundant .zip duplicates) ----
print("\n=== Parsing all CSV filenames ===")
all_files = [f for f in os.listdir(ROOT) if f.lower().endswith(".csv")]
print(f"Total CSV files: {len(all_files)}")

test_phase_counter = Counter()
cell_id_counter = Counter()
sample_by_phase = {}

for fname in all_files:
    parts = fname[:-4].split("=")  # strip .csv, split on '='
    # heuristically: look for tokens indicating test phase / cell ID
    phase_tokens = [p for p in parts if any(kw in p for kw in
                    ["BOL", "EOL", "EIS", "Massenalterung", "Kalender", "Zyklen"])]
    phase_key = " | ".join(phase_tokens) if phase_tokens else "UNRECOGNIZED"
    test_phase_counter[phase_key] += 1
    if phase_key not in sample_by_phase:
        sample_by_phase[phase_key] = fname

    # attempt to find a cell/channel identifier token (e.g. "TS010368" or "Kreis 5-002")
    id_tokens = [p for p in parts if re.match(r"^(TS\d+|Kreis\s*\d+.*|EIS\d+.*)", p.strip())]
    for tok in id_tokens:
        cell_id_counter[tok.strip()] += 1

print(f"\n--- Distinct 'phase' patterns found (and file counts) ---")
for phase, count in test_phase_counter.most_common(20):
    print(f"  {count:5d}  {phase}")
    print(f"          e.g. {sample_by_phase[phase]}")

print(f"\n--- Distinct ID-like tokens found (top 30 by frequency) ---")
for tok, count in cell_id_counter.most_common(30):
    print(f"  {count:5d}  {tok}")

print(f"\nTotal distinct ID-like tokens: {len(cell_id_counter)}")
