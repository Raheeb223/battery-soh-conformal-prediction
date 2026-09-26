"""
Oxford Battery Degradation Dataset 1 loader.

The .mat file has one top-level struct with fields Cell1..Cell8, each
containing per-cycle sub-structs (named cycNNNN) with fields for
charge/discharge sub-tests (e.g. C1ch, C1dc) holding arrays such as 'v', 'q',
't'. The discharge capacity of each characterisation cycle is taken as
max(abs(q)) of the "C1dc" (C/1 reference discharge) sub-test, the standard
way this dataset is used for capacity-fade curves.

The nesting of this file depends on how it was saved; if loading fails, run
inspect_data.py and compare its printed field names with _get_cycle_fields().
"""

import re
import numpy as np
import scipy.io as sio
import pandas as pd
import config


def _get_cycle_fields(cell_struct):
    """Return sorted list of (cycle_number, field_name) for all cycle sub-structs."""
    fields = [f for f in dir(cell_struct) if not f.startswith("_")]
    cycle_fields = []
    for f in fields:
        m = re.match(r"cyc(\d+)", f, re.IGNORECASE)
        if m:
            cycle_fields.append((int(m.group(1)), f))
    return sorted(cycle_fields)


def load_oxford_cell(cell_struct, cell_id: str) -> pd.DataFrame:
    rows = []
    for cycle_num, field_name in _get_cycle_fields(cell_struct):
        cyc = getattr(cell_struct, field_name)
        # look for the reference discharge sub-test, commonly "C1dc"
        subtests = [f for f in dir(cyc) if not f.startswith("_")]
        dc_field = next((f for f in subtests if "dc" in f.lower() and "1" in f), None)
        if dc_field is None:
            continue
        dc = getattr(cyc, dc_field)
        try:
            q = np.asarray(dc.q, dtype=float)  # charge/capacity trace, typically in Ah
            capacity = float(np.max(np.abs(q)))
        except Exception:
            continue
        rows.append({"cycle": cycle_num, "capacity": capacity})

    df = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    df["cell_id"] = cell_id
    df["dataset"] = "Oxford"
    return df


def load_all() -> pd.DataFrame:
    mat = sio.loadmat(config.OXFORD_MAT, squeeze_me=True, struct_as_record=False)
    # each Cell1..Cell8 is its OWN top-level variable in this .mat file —
    # there's no single wrapper struct containing all of them as sub-fields.
    cell_names = sorted(
        k for k in mat.keys() if not k.startswith("__") and re.match(r"[Cc]ell\d+", k)
    )

    frames = []
    for name in cell_names:
        try:
            cell_struct = mat[name]
            df = load_oxford_cell(cell_struct, name)
            if len(df) > 5:
                frames.append(df)
        except Exception as e:
            print(f"  [Oxford] skipped {name}: {e}")

    if not frames:
        raise RuntimeError(
            "Oxford: no cells parsed — the nested struct layout differs from what "
            "this loader assumes. Run inspect_data.py, print dir(cell_struct) and "
            "dir(one_cycle_substruct) manually, then fix _get_cycle_fields()/load_oxford_cell()."
        )
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_all()
    print(df.groupby("cell_id").size())
    print(df.head())