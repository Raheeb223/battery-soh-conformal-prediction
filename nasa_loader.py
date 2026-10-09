"""
NASA PCoE battery dataset loader.

Each .mat file (B0005.mat, B0006.mat, ...) contains a struct with a 'cycle'
array. Discharge cycles carry a 'Capacity' scalar (in Ah). We pull one
capacity value per discharge cycle -> that's the capacity-fade curve.

If inspect_data.py printed a different field layout than assumed below,
adjust the three lines marked "ADJUST HERE".
"""

import os
import scipy.io as sio
import pandas as pd
import config


def load_nasa_cell(mat_path: str) -> pd.DataFrame:
    key = os.path.basename(mat_path).replace(".mat", "")
    mat = sio.loadmat(mat_path)
    if key not in mat:
        # filename doesn't always match the internal struct name exactly —
        # fall back to whatever the one real (non-dunder) top-level key is.
        real_keys = [k for k in mat.keys() if not k.startswith("__")]
        key = real_keys[0]
    cycles = mat[key][0, 0]['cycle'][0]                      # ADJUST HERE if structure differs

    rows = []
    cycle_idx = 0
    for c in cycles:
        ctype = c['type'][0]
        if ctype != 'discharge':                              # ADJUST HERE
            continue
        cycle_idx += 1
        try:
            capacity = float(c['data'][0, 0]['Capacity'][0, 0])  # ADJUST HERE
        except Exception:
            continue
        rows.append({"cycle": cycle_idx, "capacity": capacity})

    df = pd.DataFrame(rows)
    df["cell_id"] = key
    df["dataset"] = "NASA"
    return df


def load_all() -> pd.DataFrame:
    if not os.path.isdir(config.NASA_DIR):
        raise FileNotFoundError(f"NASA_DIR not found: {config.NASA_DIR}")

    # NASA's zip extracts .mat files into nested subfolders (one per batch),
    # not directly inside NASA_DIR, so we search recursively.
    mat_paths = []
    for dirpath, _, filenames in os.walk(config.NASA_DIR):
        for f in filenames:
            if f.endswith(".mat"):
                mat_paths.append(os.path.join(dirpath, f))

    frames = []
    for path in sorted(mat_paths):
        f = os.path.basename(path)
        try:
            df = load_nasa_cell(path)
            if len(df) > 5:  # skip cells with almost no discharge cycles (parsing failure signal)
                frames.append(df)
        except Exception as e:
            print(f"  [NASA] skipped {f}: {e}")
    if not frames:
        raise RuntimeError("NASA: no cells parsed successfully — run inspect_data.py and fix nasa_loader.py")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_all()
    print(df.groupby("cell_id").size())
    print(df.head())
