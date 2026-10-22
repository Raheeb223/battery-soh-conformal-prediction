"""
Turns per-dataset capacity curves into windowed train/validation/test arrays for the model.

  - SOH = capacity(cycle) / median(capacity of that cell's first three cycles), so datasets with
    different rated capacities are comparable.
  - The split is by cell: all cycles of a cell go to exactly one of train, validation, and test.
  - Each dataset's SOH is standardized (z-scored) with the mean and standard deviation of that
    dataset's training cells only.
  - CALCE is loaded through calce_loader_v2, which differences the cumulative discharge-capacity
    counter found in most CALCE workbooks (see that module's docstring).
  - load_all_raw() keeps only the columns dataset, cell_id, cycle, capacity, so every loader
    returns the same schema; a loader that fails is skipped with a message.
  - make_windows() also returns the cell of every window (cell_train/cell_val/cell_test), which
    the per-cell figures, the XJTU bridging analysis, and the cell-clustered bootstrap need.
  - run() prints the number of windows per dataset and split, so a dataset that yields no windows
    for a given window length is visible immediately.
  - add_soh() asserts that no (dataset, cell_id, cycle) row is duplicated, since a duplicate would
    bias the reference capacity.

Windows are taken over consecutive rows of each cell. xjtu_loader.py removes sentinel readings from
the eight Sim_satellite_* cells without renumbering cycles, so every test window from those cells
(376 of XJTU's 2,605) spans a removed block. xjtu_bridging_isolation.py measures this and
xjtu_bridging_sensitivity.py reports its effect on error and coverage.

Run:
    python preprocessing.py
The processed arrays are cached in config.CACHE_DIR as .npz files.
"""

import os
import numpy as np
import pandas as pd

import config
import nasa_loader, calce_loader_v2 as calce_loader, oxford_loader, mit_loader
import bit_loader, xjtu_loader, rwth_loader

CANONICAL_COLUMNS = ["dataset", "cell_id", "cycle", "capacity"]


def load_all_raw() -> pd.DataFrame:
    """Load every dataset that's available; skip ones that fail with a warning
    rather than crashing the whole pipeline. Every loaded dataframe is trimmed
    to CANONICAL_COLUMNS before concatenation (see module docstring),
    so a loader that returns extra diagnostic columns (CALCE's, currently)
    never produces a heterogeneous, NaN-scattered combined dataframe."""
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
            missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"loader returned no column(s) {missing} required by preprocessing")
            df = df[CANONICAL_COLUMNS].copy()
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


def cell_independent_split(df: pd.DataFrame, split_seed: int = None):
    """Split whole cells (not individual cycles) into train/val/test,
    stratified by dataset so every split has representation from every source.

    Guarantees at least 1 cell in val and 1 in test for any group with 3+
    cells. Applying the validation/test fraction to what is left after the
    training cells are removed can round a small pool down to 0 (NASA has 7
    cells), so explicit integer counts are computed up front and a required
    split never falls to 0.

    split_seed: overrides config.SPLIT_SEED for this call (used by the
    repeated-split experiment to vary the split independently of the
    model-init seed, config.RANDOM_SEED)."""
    cell_table = df[["dataset", "cell_id"]].drop_duplicates()
    seed = split_seed if split_seed is not None else getattr(config, "SPLIT_SEED", config.RANDOM_SEED)
    rng = np.random.RandomState(seed)

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
    cell-ID labels, AND per-sample raw cycle-index arrays (needed for
    xjtu_bridging_isolation.py / xjtu_bridging_sensitivity.py to detect
    windows that bridge across rows removed during data-quality cleaning.
    This catches XJTU's Sim_satellite_* cells: xjtu_loader.py's detect_and_strip_sentinel_values() does not
    re-number 'cycle' after removing rows, so the gap survives and every
    Sim_satellite_* test window bridges it (see this file's module-level
    docstring for the numbers)."""
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


def cache_path_for(window: int, split_seed: int = None) -> str:
    """Each window size gets its own cache file so ablation runs never
    clobber each other or the main experiment's cache. A non-default
    split_seed also gets its own suffix, so
    those runs never overwrite the verified seed-42 cache."""
    default_seed = getattr(config, "SPLIT_SEED", config.RANDOM_SEED)
    seed = split_seed if split_seed is not None else default_seed
    parts = []
    if window != config.INPUT_WINDOW:
        parts.append(f"w{window}")
    if seed != default_seed:
        parts.append(f"split{seed}")
    suffix = ("_" + "_".join(parts)) if parts else ""
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


def run(window: int = None, split_seed: int = None, raw_df=None, out_path: str = None,
        horizon: int = None):
    """window: override config.INPUT_WINDOW for this run (used by the window-size
    ablation in experiments.py). The cell split itself is NOT affected by window
    size for a GIVEN split_seed — it's computed once per seed so every window-size
    variant at that seed trains/tests on the exact same cells, making the
    comparison fair.

    split_seed: override config.SPLIT_SEED (repeated-split experiment). Left
    None, this reproduces the original seed-42 cache exactly.

    raw_df: pre-loaded, pre-SOH'd dataframe (add_soh(load_all_raw())), reused
    across multiple split_seed / window variants by a caller that needs several
    caches built in one run (e.g. the repeated-split experiment) so raw data
    is parsed once rather than once per variant. Pass None to load fresh.

    out_path: where to write the cache (default: cache_path_for(window, split_seed)). The field
    check (field_preprocessing.py) passes its own path so it never touches the laboratory caches.

    horizon: override config.FORECAST_HORIZON (horizon_experiment.py). The cell split does not
    depend on it; pass out_path so a longer horizon never overwrites the one-step cache."""
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    w = window if window is not None else config.INPUT_WINDOW

    if raw_df is None:
        raw = load_all_raw()
        df = add_soh(raw)
    else:
        df = raw_df

    train_ids, val_ids, test_ids = cell_independent_split(df, split_seed=split_seed)
    print(f"[split] train cells: {len(train_ids)}, val cells: {len(val_ids)}, "
          f"test cells: {len(test_ids)}")

    # compute per-dataset normalization stats using ONLY train cells
    train_df = df[df["cell_id"].isin(train_ids)]
    soh_mean = train_df.groupby("dataset")["soh"].mean().to_dict()
    soh_std = train_df.groupby("dataset")["soh"].std().to_dict()
    for k in soh_std:
        if soh_std[k] == 0 or np.isnan(soh_std[k]):
            soh_std[k] = 1.0

    X_train, y_train, lab_train, cell_train, cyc_train = make_windows(df, train_ids, soh_mean, soh_std, window=w, horizon=horizon)
    X_val, y_val, lab_val, cell_val, cyc_val = make_windows(df, val_ids, soh_mean, soh_std, window=w, horizon=horizon)
    X_test, y_test, lab_test, cell_test, cyc_test = make_windows(df, test_ids, soh_mean, soh_std, window=w, horizon=horizon)

    print(f"[windows] window={w}  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    _print_window_summary("train", lab_train)
    _print_window_summary("val", lab_val)
    _print_window_summary("test", lab_test)

    out_path = out_path or cache_path_for(w, split_seed=split_seed)
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