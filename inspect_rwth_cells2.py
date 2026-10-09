"""
Corrected: group by the TRUE cell identifier 'ep sanyo NNN', not 'Kreis
N-NNN' (which is the reusable test channel, not the physical cell).
"""

import os
import re
from collections import defaultdict

ROOT = r"C:\Users\User\PycharmProjects\RWTH-2021-04545_818642\Rawdata\Rohdaten"

all_files = [f for f in os.listdir(ROOT) if f.lower().endswith(".csv")]

cell_files = defaultdict(list)
eis_excluded = 0
no_match = 0

for fname in all_files:
    if "EIS" in fname:
        eis_excluded += 1
        continue
    m = re.search(r"ep sanyo\s*(\d+)", fname, re.IGNORECASE)
    if m:
        cell_id = f"ep_sanyo_{m.group(1)}"
        cell_files[cell_id].append(fname)
    else:
        no_match += 1

print(f"Total CSV files: {len(all_files)}")
print(f"Excluded as EIS: {eis_excluded}")
print(f"No 'ep sanyo N' match: {no_match}")
print(f"Distinct TRUE physical cells (by 'ep sanyo N'): {len(cell_files)}")

file_counts = sorted(len(v) for v in cell_files.values())
if file_counts:
    print(f"Files per cell: min={file_counts[0]}, max={file_counts[-1]}, "
          f"median={file_counts[len(file_counts)//2]}, total={sum(file_counts)}")

print("\n--- All distinct cell IDs and their file counts ---")
for cell_id in sorted(cell_files.keys(), key=lambda x: int(x.split("_")[-1])):
    print(f"  {cell_id}: {len(cell_files[cell_id])} files")

print("\n--- Example: chronological file list for one cell ---")
example_cell = list(cell_files.keys())[0]
for f in sorted(cell_files[example_cell]):
    print(f"    {f}")
