"""
One-off diagnostic — run this once, paste the full output back.
It prints the exact field names inside a single Oxford cycle's sub-structs,
which is what oxford_loader.py needs to match exactly.
"""

import re
import scipy.io as sio
import config

mat = sio.loadmat(config.OXFORD_MAT, squeeze_me=True, struct_as_record=False)
top_keys = [k for k in mat.keys() if not k.startswith("__")]
root = mat[top_keys[0]]

cell1 = mat["Cell1"]
cycle_fields = sorted(f for f in dir(cell1) if re.match(r"cyc\d+", f, re.IGNORECASE))
print(f"Cycle field names on Cell1: {cycle_fields[:5]} ... ({len(cycle_fields)} total)")

first_cycle_name = cycle_fields[0]
cyc = getattr(cell1, first_cycle_name)
subtests = [f for f in dir(cyc) if not f.startswith("_")]
print(f"\nSub-test fields inside '{first_cycle_name}': {subtests}")

for sub_name in subtests:
    sub = getattr(cyc, sub_name)
    inner_fields = [f for f in dir(sub) if not f.startswith("_")]
    print(f"\n  Fields inside '{first_cycle_name}.{sub_name}': {inner_fields}")
    # print the shape/type of each inner field so we know which one is the capacity/charge trace
    for inner in inner_fields:
        val = getattr(sub, inner)
        try:
            shape = getattr(val, "shape", None)
            print(f"    {inner}: type={type(val).__name__}, shape={shape}")
        except Exception as e:
            print(f"    {inner}: (could not inspect: {e})")