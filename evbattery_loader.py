"""
Loader for the EVBattery field dataset (He et al.; figshare DOI 10.6084/m9.figshare.23301881, CC BY 4.0):
charging records of 464 electric vehicles from three manufacturers ("battery_dataset1..3").

Each charging snippet is a .pkl file written with torch.save's legacy format: three short header
pickles followed by the object, a (128, 8) array of charging measurements and an OrderedDict of
metadata (vehicle, mileage, capacity label, ...). The capacity label is the charge (Ah) the
manufacturers' engineers computed for the battery being charged from 3.77 V to 4.05 V at 35 A; only a
subset of snippets carries one.

SAFETY. Unpickling can run arbitrary code, so the files are never loaded with pickle.load or
torch.load. RestrictedUnpickler allows only the globals these files use (numpy array and
scalar reconstruction, numpy dtype, collections.OrderedDict, _codecs.encode for numpy's byte
payload) and refuses anything else. The archives are read member by member straight from the
.tar.gz files, so nothing is extracted to disk.

Output: one row per labelled snippet: (manufacturer, vehicle, charge_segment, mileage, capacity_ah).
Several snippets can come from the same charging event (charge_segment); field_preprocessing.py
merges them into one point per event.
"""
import io
import os
import pickle
import tarfile

import numpy as np
import pandas as pd

import config

EVB_DIR = os.path.join(config.ROOT_DIR, "EVBattery")
ARCHIVES = {f"EV{i}": os.path.join(EVB_DIR, f"battery_dataset{i}.tar.gz") for i in (1, 2, 3)}
CACHE = os.path.join(config.CACHE_DIR, "evbattery_capacity_rows.csv")

_ALLOWED = {
    # numpy 2 names the same module numpy._core; both spellings refer to the same two functions.
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("collections", "OrderedDict"),
    ("_codecs", "encode"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def load_snippet(raw: bytes):
    """Returns the object after torch.save's three legacy header pickles."""
    buf = io.BytesIO(raw)
    objects = []
    while buf.tell() < len(raw):
        objects.append(RestrictedUnpickler(buf).load())
    if len(objects) < 4:
        raise ValueError(f"expected header pickles plus an object, got {len(objects)} pickles")
    return objects[3]


def iter_snippets(archive):
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".pkl"):
                yield m.name, load_snippet(tar.extractfile(m).read())


def _metadata(obj):
    """(array, metadata dict) regardless of the order the two are stored in."""
    if isinstance(obj, (tuple, list)):
        arr = next((o for o in obj if isinstance(o, np.ndarray)), None)
        meta = next((o for o in obj if isinstance(o, dict)), None)
        return arr, meta
    raise TypeError(f"unexpected snippet object of type {type(obj).__name__}")


def build_rows(force=False):
    """All snippets with a capacity label, one row each. Cached to cache/evbattery_capacity_rows.csv."""
    if os.path.exists(CACHE) and not force:
        return pd.read_csv(CACHE)
    rows = []
    for maker, archive in ARCHIVES.items():
        n = n_labelled = 0
        for name, obj in iter_snippets(archive):
            n += 1
            _, meta = _metadata(obj)
            cap = float(meta.get("capacity") or 0.0)
            if not np.isfinite(cap) or cap <= 0:   # unlabelled snippets carry capacity 0
                continue
            n_labelled += 1
            rows.append({"manufacturer": maker, "vehicle": f"{maker}_{int(meta['car'])}",
                         "charge_segment": int(meta["charge_segment"]),
                         "mileage": float(meta["mileage"]), "capacity_ah": cap,
                         "snippet": os.path.basename(name)})
        print(f"{maker}: {n} snippets, {n_labelled} with a capacity label")
    df = pd.DataFrame(rows)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


if __name__ == "__main__":
    print(build_rows(force=True).groupby("manufacturer").agg(
        vehicles=("vehicle", "nunique"), labels=("capacity_ah", "size")))
