"""
diagnose_cache_directly.py

Every check so far has gone through predictions.npz, which depends on a
chain of trust: preprocessing.py wrote the cache correctly -> train_one()
read the RIGHT cache file -> train_one() saved a fresh predictions file.
Any break in that chain reproduces the same symptom. This script skips
the chain entirely and inspects cache/processed.npz directly -- the one
file preprocessing.py itself just confirmed (in its own console output)
was rebuilt with the min_run=2 fix active (198/258/187/198/275/220/198/220
rows removed from the 8 satellite cells).

If the cache itself is clean (no 79-sample cluster in its test split),
but predictions.npz still shows the cluster, the problem is downstream
in train.py/experiments.py (wrong cache path, resumed instead of
retrained, etc.) -- NOT in xjtu_loader.py or preprocessing.py.

If the cache itself is STILL dirty, the problem is upstream of this --
either preprocessing.py's windowing step reintroduces a repeated value
even from clean per-cell data (the windowing-discontinuity theory), or
the console output we saw doesn't actually correspond to this cache file
(e.g. it wrote somewhere else, or a second preprocessing.py run happened
after and silently regenerated a worse cache).

Run:
    python diagnose_cache_directly.py
"""

import os
import time
import numpy as np
import config


def _mtime_str(path):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))


def main():
    cache_path = os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(cache_path):
        print(f"[!] {cache_path} does not exist -- run preprocessing.py first.")
        return

    print(f"Inspecting: {cache_path}")
    print(f"  modified: {_mtime_str(cache_path)}  size: {os.path.getsize(cache_path)} bytes\n")

    data = np.load(cache_path, allow_pickle=True)
    print(f"Keys in this cache: {data.files}\n")

    if "cell_test" not in data.files:
        print("[!] No cell_test field in this cache -- it predates the "
              "updated preprocessing.py. This IS the bug: something is "
              "still pointing at an old cache file. Check config.CACHE_DIR "
              "and make sure preprocessing.py actually wrote to this exact "
              "path.")
        return

    # We don't have y_test in normalized form easily comparable without
    # the model, but we DON'T need the model at all for this check --
    # we can directly re-derive whether the WINDOWED data itself (before
    # any training) still contains a repeated-value cluster, by checking
    # the raw X_test windows' LAST timestep (which is closely related to
    # the y target for horizon=1) for the 3 implicated cells.
    lab_test = data["lab_test"]
    cell_test = data["cell_test"]
    y_test = data["y_test"]

    mask = lab_test == "XJTU"
    yt = y_test[mask]
    ct = cell_test[mask]
    n = len(yt)
    print(f"XJTU test windows in this cache: {n}")

    if n == 0:
        print("No XJTU test windows at all in this cache.")
        return

    min_val = yt.min()
    exact_match = np.isclose(yt, min_val, atol=1e-4)
    n_exact = int(exact_match.sum())
    print(f"Minimum value: {min_val:.6f}")
    print(f"Samples within 1e-4 of minimum: {n_exact} ({100*n_exact/n:.2f}%)")

    if n_exact > 0:
        cells_at_min = ct[exact_match]
        uniq_cells, counts = np.unique(cells_at_min, return_counts=True)
        print("Cell(s) responsible:")
        for cid, cnt in sorted(zip(uniq_cells, counts), key=lambda t: -t[1]):
            print(f"    {cid}: {cnt} samples")
        print("\n==> This cluster exists in the CACHE ITSELF, before any "
              "training. This means the problem is in preprocessing.py's "
              "windowing (or upstream in xjtu_loader.py), NOT in "
              "train.py/experiments.py's handling of the cache. The "
              "windowing-discontinuity theory (or a residual raw-data "
              "issue) is the right thing to chase next, and the "
              "predictions.npz staleness checks can be set aside.")
    else:
        print("\n==> No cluster in the cache itself. If predictions.npz "
              "still shows one, the problem is downstream -- in "
              "train.py/experiments.py picking up a different/stale cache, "
              "not in preprocessing.py or xjtu_loader.py.")


if __name__ == "__main__":
    main()
