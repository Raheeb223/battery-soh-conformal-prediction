"""
Second-pass RWTH inspection: extract the TRUE cell count (grouped by
'Kreis N-NNN' channel identifier, excluding EIS/impedance test files),
and check how large a typical file is (row count) to estimate whether
parsing ~1600+ files is practically fast.
"""

import os
import re
from collections import defaultdict

ROOT = r"C:\Users\User\PycharmProjects\RWTH-2021-04545_818642\Rawdata\Rohdaten"

all_files = [f for f in os.listdir(ROOT) if f.lower().endswith(".csv")]

kreis_files = defaultdict(list)
eis_excluded_count = 0

for fname in all_files:
    if "EIS" in fname:
        eis_excluded_count += 1
        continue
    m = re.search(r"Kreis\s*(\d+-\d+)", fname)
    if m:
        kreis_id = f"Kreis_{m.group(1)}"
        kreis_files[kreis_id].append(fname)

print(f"Total CSV files: {len(all_files)}")
print(f"Excluded as EIS (impedance) tests: {eis_excluded_count}")
print(f"Remaining (capacity-cycling) files: {len(all_files) - eis_excluded_count}")
print(f"\nDistinct physical cells found (by 'Kreis N-NNN'): {len(kreis_files)}")

file_counts = sorted(len(v) for v in kreis_files.values())
print(f"Files per cell: min={file_counts[0]}, max={file_counts[-1]}, "
      f"median={file_counts[len(file_counts)//2]}")

# show a few example cells and their file lists
print("\n--- Example: files for first 2 cells ---")
for kreis_id in list(kreis_files.keys())[:2]:
    print(f"\n{kreis_id}: {len(kreis_files[kreis_id])} files")
    for f in sorted(kreis_files[kreis_id])[:5]:
        print(f"    {f}")

# check row count / file size of one example file
example_file = None
for v in kreis_files.values():
    if v:
        example_file = os.path.join(ROOT, v[0])
        break

if example_file:
    size_mb = os.path.getsize(example_file) / (1024 * 1024)
    with open(example_file, "r", errors="replace") as fh:
        row_count = sum(1 for _ in fh)
    print(f"\n--- Example file size/row check ---")
    print(f"File: {example_file}")
    print(f"Size: {size_mb:.2f} MB, rows: {row_count}")
