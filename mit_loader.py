"""
MIT / Stanford / Toyota (Severson et al.) fast-charging dataset loader.

Each FastCharge_XXXXX_CHYY_structure.json file is a BEEP-style structure with
a top-level "summary" dict containing a "QDischarge" array (discharge
capacity per cycle, in Ah) and a "cycle_index" array of matching length.
"""

import os
import json
import pandas as pd
import config


def load_mit_cell(json_path: str) -> pd.DataFrame:
    with open(json_path) as fh:
        d = json.load(fh)

    summary = d.get("summary", {})
    q = summary.get("discharge_capacity")
    cyc = summary.get("cycle_index")
    if q is None or cyc is None:
        raise KeyError("summary missing discharge_capacity/cycle_index — check inspect_data.py output")

    df = pd.DataFrame({"cycle": cyc, "capacity": q})
    cell_id = os.path.basename(json_path).replace("_structure.json", "")
    df["cell_id"] = cell_id
    df["dataset"] = "MIT"
    return df


def load_all() -> pd.DataFrame:
    if not os.path.isdir(config.MIT_DIR):
        raise FileNotFoundError(f"MIT_DIR not found: {config.MIT_DIR}")

    # the JSON files sometimes sit loose directly in ROOT_DIR rather than a
    # dedicated subfolder, so search recursively rather than assuming a flat layout.
    json_paths = []
    for dirpath, _, filenames in os.walk(config.MIT_DIR):
        for f in filenames:
            if f.endswith("_structure.json"):
                json_paths.append(os.path.join(dirpath, f))

    frames = []
    for path in sorted(json_paths):
        f = os.path.basename(path)
        if os.path.getsize(path) == 0:
            print(f"  [MIT] skipping empty/corrupt file: {f}")
            continue
        try:
            df = load_mit_cell(path)
            if len(df) > 5:
                frames.append(df)
        except Exception as e:
            print(f"  [MIT] skipped {f}: {e}")
    if not frames:
        raise RuntimeError("MIT: no cells parsed — run inspect_data.py and fix mit_loader.py")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_all()
    print(df.groupby("cell_id").size())
    print(df.head())
