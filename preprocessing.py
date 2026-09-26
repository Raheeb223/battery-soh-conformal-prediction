"""
Turns raw per-dataset capacity-fade curves into windowed train/val/test arrays.

Design:
  - SOH = capacity(cycle) / reference capacity, where the reference is the
    median capacity of the cell's first 3 cycles (more robust than a single
    first-cycle reading). SOH is unitless and comparable across datasets with
    different rated capacities. Rows with SOH outside (0, 1.10) are dropped.
  - Duplicate (dataset, cell_id, cycle) rows raise an error, since they would
    bias the reference capacity.
  - Cell-independent split: every cycle of a cell goes to exactly one of
    train / val / test (no leakage between splits). The split is stratified by
    dataset with at least one validation and one test cell for every dataset
    with 3 or more cells, and is seeded by config.RANDOM_SEED, so it is
    identical for every window size.
  - Per-dataset normalisation: each dataset's SOH is z-scored using mean/std
    computed from that dataset's TRAIN cells only.
  - Windows: config.INPUT_WINDOW past cycles -> SOH config.FORECAST_HORIZON
    cycles ahead. For every window the dataset label, cell ID and the raw
    cycle indices it spans are saved as well.

Limitation: windows are formed positionally over each cell's cycle-sorted SOH
sequence. Where rows were removed during data cleaning (XJTU, RWTH) or where
the raw data is sparse, a window can span a gap in the cycle numbering.
windowing_bridge_sensitivity.py and xjtu_bridging_isolation.py quantify how
much this affects coverage.

Run:
    python preprocessing.py
Caches the arrays to config.CACHE_DIR/processed.npz (processed_w<N>.npz for
other window sizes), so raw files need not be re-parsed for every experiment.
"""

import os
import numpy as np
import pandas as pd

import config
import nasa_loader, calce_loader, oxford_loader, mit_loader
import bit_loader, xjtu_loader, rwth_loader


def load_all_raw() -> pd.DataFrame:
    """Load every dataset that's available; skip ones that fail with a warning
    rather than crashing the whole pipeline."""
    loaders = {
        "NASA": nasa_loader.load_all,
        "CALCE": calce_loader.load_all,
        "Oxford": oxford_loader.load_all,
        "MIT": mit_loader.load_all,
        "BIT": bit_loader.load_all,
        "XJTU": xjtu_loader.load_all,
        "RWTH": rwth_loader.load_all,
    }
    frames = []
    for name, fn in loaders.items():
        try:
            df = fn()
            print(f"[preprocessing] {name}: {df['cell_id'].nunique()} cells, {len(df)} rows")
            frames.append(df)
        except Exception as e:
            print(f"[preprocessing] SKIPPING {name} — {e}")
    if not frames:
        raise RuntimeError("No datasets loaded successfully. Fix at least one loader first.")
    return pd.concat(frames, ignore_index=True)


def add_soh(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["dataset", "cell_id", "cycle"]).copy()

    # Guard against duplicate (dataset, cell_id, cycle) rows -- these would
    # silently bias the reference-capacity median below and every downstream
    # SOH value for that cell, with no visible symptom until much later.
    dup_mask = df.duplicated(subset=["dataset", "cell_id", "cycle"], keep=False)
    if dup_mask.any():
        dup_examples = (
            df.loc[dup_mask, ["dataset", "cell_id", "cycle"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"add_soh: found {dup_mask.sum()} duplicate (dataset, cell_id, cycle) "
            f"rows -- this would silently corrupt the reference-capacity median "
            f"and every downstream SOH value for the affected cell(s). First few "
            f"duplicates: {dup_examples}. Fix the responsible loader before "
            f"proceeding rather than let this pass silently."
        )

    def _soh(group):
        # use the median of the first 3 cycles as the reference capacity —
        # more robust than a single first-cycle reading, which is sometimes noisy.
        ref = group["capacity"].iloc[: min(3, len(group))].median()
        group["soh"] = group["capacity"] / ref
        return group

    df = df.groupby(["dataset", "cell_id"], group_keys=False).apply(_soh)
    # drop obviously broken points (e.g. SOH > 1.05 sustained, or negative)
    df = df[(df["soh"] > 0) & (df["soh"] < 1.10)]
    return df


def cell_independent_split(df: pd.DataFrame):
    """Split whole cells (not individual cycles) into train/val/test,
    stratified by dataset so every split has representation from every source.

    Guarantees at least 1 cell in val and 1 in test for any group with 3+
    cells. sklearn's train_test_split (used previously) applies the val/test
    fraction to whatever's left AFTER carving out train, which can round a
    small remaining pool down to 0 in one of the two splits — that's exactly
    what happened to NASA's 7 cells, silently leaving it with 0 validation
    cells and breaking conformal calibration for that group. This version
    computes explicit integer counts up front and never lets a required
    split fall to 0."""
    cell_table = df[["dataset", "cell_id"]].drop_duplicates()
    rng = np.random.RandomState(config.RANDOM_SEED)

    train_cells, val_cells, test_cells = [], [], []
    for dataset_name, group in cell_table.groupby("dataset"):
        ids = group["cell_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)

        if n < 3:
            print(f"[split] WARNING: {dataset_name} has only {n} cells; "
                  f"all going to TRAIN. Consider merging with another split manually.")
            train_cells.extend(ids)
            continue

        n_val = max(1, round(n * config.VAL_FRAC))
        n_test = max(1, round(n * config.TEST_FRAC))
        n_train = n - n_val - n_test
        if n_train < 1:
            # shrink val/test (never below 1 each) until train has at least 1 cell
            deficit = 1 - n_train
            while deficit > 0 and n_test > 1:
                n_test -= 1
                deficit -= 1
            while deficit > 0 and n_val > 1:
                n_val -= 1
                deficit -= 1
            n_train = n - n_val - n_test

        train_cells.extend(ids[:n_train])
        val_cells.extend(ids[n_train: n_train + n_val])
        test_cells.extend(ids[n_train + n_val:])
        print(f"[split] {dataset_name}: {n} cells -> train={n_train}, val={n_val}, test={n_test}")

    return set(train_cells), set(val_cells), set(test_cells)


def make_windows(df: pd.DataFrame, cell_ids: set, soh_mean: dict, soh_std: dict,
                  window: int = None, horizon: int = None):
    """Slide a fixed-length window over each cell's SOH curve.
    Returns X (N, window, 1), y (N,), per-sample dataset labels, per-sample
    cell-ID labels, AND per-sample raw cycle-index arrays (new in this
    version -- needed for windowing_bridge_sensitivity.py to detect
    windows that bridge across rows removed during data-quality cleaning.
    NOTE: this only detects a bridge where a genuine gap survives in the
    'cycle' column. It will NOT catch XJTU's Sim_satellite_* cells, since
    xjtu_loader.py's strip_calibration_blocks() re-numbers 'cycle'
    sequentially (1..N) after removing rows, erasing the gap before it
    ever reaches this function (see this file's module-level docstring,
    'KNOWN LIMITATION' section). Catching that specific case requires a
    xjtu_loader.py change to preserve original cycle numbers or emit an
    explicit boundary flag -- a loader fix, not something addressable
    here."""
    w = window if window is not None else config.INPUT_WINDOW
    h = horizon if horizon is not None else config.FORECAST_HORIZON

    X, y, labels, cell_labels, cycle_windows = [], [], [], [], []
    for (dataset_name, cell_id), group in df.groupby(["dataset", "cell_id"]):
        if cell_id not in cell_ids:
            continue
        sorted_group = group.sort_values("cycle")
        soh = sorted_group["soh"].to_numpy()
        cycle = sorted_group["cycle"].to_numpy()
        soh_norm = (soh - soh_mean[dataset_name]) / soh_std[dataset_name]

        for i in range(len(soh_norm) - w - h + 1):
            X.append(soh_norm[i: i + w])
            y.append(soh_norm[i + w + h - 1])
            labels.append(dataset_name)
            cell_labels.append(cell_id)
            cycle_windows.append(cycle[i: i + w])

    X = np.array(X, dtype=np.float32)[..., None]   # (N, window, 1 feature)
    y = np.array(y, dtype=np.float32)
    cycle_windows = np.array(cycle_windows, dtype=np.float64) if cycle_windows else np.zeros((0, w))
    return X, y, np.array(labels), np.array(cell_labels), cycle_windows


def cache_path_for(window: int) -> str:
    """Each window size gets its own cache file so ablation runs never
    clobber each other or the main experiment's cache."""
    suffix = "" if window == config.INPUT_WINDOW else f"_w{window}"
    return os.path.join(config.CACHE_DIR, f"processed{suffix}.npz")


def _print_window_summary(name: str, labels: np.ndarray):
    """Per-dataset row counts after windowing, printed immediately so a
    silently-empty dataset (e.g. NASA at window=30/50) is visible in THIS
    script's own output rather than only discoverable later via a
    separate diagnostic run."""
    if len(labels) == 0:
        print(f"  [windows:{name}] EMPTY -- no windows produced at all.")
        return
    uniq, counts = np.unique(labels, return_counts=True)
    parts = ", ".join(f"{u}={c}" for u, c in zip(uniq, counts))
    print(f"  [windows:{name}] {parts}")


def run(window: int = None):
    """window: override config.INPUT_WINDOW for this run (used by the window-size
    ablation in experiments.py). The cell split itself is NOT affected by window
    size — it's computed once from config.RANDOM_SEED so every window-size variant
    trains/tests on the exact same cells, making the comparison fair."""
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    w = window if window is not None else config.INPUT_WINDOW

    raw = load_all_raw()
    df = add_soh(raw)

    train_ids, val_ids, test_ids = cell_independent_split(df)
    print(f"[split] train cells: {len(train_ids)}, val cells: {len(val_ids)}, "
          f"test cells: {len(test_ids)}")

    # compute per-dataset normalization stats using ONLY train cells
    train_df = df[df["cell_id"].isin(train_ids)]
    soh_mean = train_df.groupby("dataset")["soh"].mean().to_dict()
    soh_std = train_df.groupby("dataset")["soh"].std().to_dict()
    for k in soh_std:
        if soh_std[k] == 0 or np.isnan(soh_std[k]):
            soh_std[k] = 1.0

    X_train, y_train, lab_train, cell_train, cyc_train = make_windows(df, train_ids, soh_mean, soh_std, window=w)
    X_val, y_val, lab_val, cell_val, cyc_val = make_windows(df, val_ids, soh_mean, soh_std, window=w)
    X_test, y_test, lab_test, cell_test, cyc_test = make_windows(df, test_ids, soh_mean, soh_std, window=w)

    print(f"[windows] window={w}  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    _print_window_summary("train", lab_train)
    _print_window_summary("val", lab_val)
    _print_window_summary("test", lab_test)

    out_path = cache_path_for(w)
    np.savez(out_path,
             X_train=X_train, y_train=y_train, lab_train=lab_train, cell_train=cell_train,
             window_cycle_indices_train=cyc_train,
             X_val=X_val, y_val=y_val, lab_val=lab_val, cell_val=cell_val,
             window_cycle_indices_val=cyc_val,
             X_test=X_test, y_test=y_test, lab_test=lab_test, cell_test=cell_test,
             window_cycle_indices_test=cyc_test,
             soh_mean=soh_mean, soh_std=soh_std)
    print(f"[preprocessing] cached to {out_path}")
    return out_path


if __name__ == "__main__":
    run()