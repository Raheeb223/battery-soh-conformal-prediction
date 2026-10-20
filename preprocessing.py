"""
Turns raw per-dataset capacity-fade curves into windowed train/val/test arrays
ready for the model.

Key design choices (match what your project already established):
  - SOH = capacity(cycle) / median(capacity of that cell's first 3 cycles)
    -> unitless, comparable across
    datasets with different rated capacities/chemistries.
  - Cell-independent split: an entire cell's cycle history goes to exactly
    one of train/val/test. Never split a single cell's cycles across sets —
    that's the leakage bug you already fixed once.
  - Per-dataset normalization: SOH is already roughly [0,1] and comparable,
    but we still z-score each dataset's SOH values *using statistics computed
    only from that dataset's TRAIN cells*, to correct for any residual scale
    differences between chemistries.

--- CHANGES IN THIS VERSION ---
0. CALCE now loads through calce_loader_v2 instead of the original
   calce_loader. The original loader misread a cumulative discharge-capacity
   counter as a per-cycle reading, which corrupted the CALCE series (see
   calce_loader_v2.py's docstring and the manuscript's CALCE section for the
   full mechanism, fix, and validation). calce_loader_v2 returns the same
   ["dataset", "cell_id", "cycle", "capacity"] schema plus extra per-row
   diagnostic columns (cycle_raw, source_file, complete_discharge,
   straddles_boundary, truncated_final_cycle, terminal_voltage). Those extras
   are informational only — inspect them directly via calce_loader_v2's own
   diagnostic scripts (verify_calce_v2.py, analyze_low_side_outliers.py, etc.)
   if needed — and are trimmed off in load_all_raw() below (see change 1)
   rather than left to flow silently through the rest of the pipeline.
1. load_all_raw() now restricts every loaded dataframe to the canonical
   ["dataset", "cell_id", "cycle", "capacity"] columns before concatenating.
   Previously, concatenating dataframes with different column sets (e.g.
   CALCE's diagnostic extras against every other loader's plain four columns)
   silently NaN-fills the missing columns for every non-CALCE row — harmless
   for correctness (nothing downstream reads those columns), but untidy and
   a plausible source of confusion later (e.g. a stray df.isnull().sum()
   check). This makes the loaded schema homogeneous and explicit instead.
   A loader missing a required column is skipped with a clear message,
   same as any other loader failure, rather than crashing the whole run.
2. make_windows() now also returns a cell_id array (cell_train/cell_val/
   cell_test), saved into the cache alongside lab_train/lab_val/lab_test.
   Downstream analyses (per-cell figures, the XJTU bridging analysis, the
   cell-clustered bootstrap) need to know which cell each window came from. Old code reading only X/y/lab still works
   unchanged — this is a pure addition, no existing keys removed or renamed.
3. run() now prints a per-dataset, per-split row-count summary AFTER
   windowing (not just cell counts), so a silent zero-window dataset
   (like the NASA window=30/50 case you already found) is visible
   immediately in this script's own output, not just discovered later
   via a separate diagnostic script.
4. add_soh() now asserts there are no duplicate (dataset, cell_id, cycle)
   rows before computing SOH -- a duplicate row would silently bias the
   reference-capacity median and every downstream SOH value for that
   cell, and would fail quietly otherwise.

--- KNOWN LIMITATION, NOT FIXED HERE (documented rather than silently
    papered over) ---
make_windows() slides windows positionally over each cell's SOH array,
in cycle-sorted order. For XJTU's Sim_satellite_* cells, xjtu_loader.py's
detect_and_strip_sentinel_values() removes sentinel rows but does NOT
re-number the 'cycle' column afterward -- only the pandas row index is
reset; surviving rows keep their original (now-gapped) cycle numbers.
Verified directly (E9/xjtu_bridging_isolation.py, run against
cache/processed.npz's window_cycle_indices_test): of XJTU's 2,605 test
windows, the 376 drawn from the 8 Sim_satellite_* cells bridge a removed
block on EVERY SINGLE WINDOW (bridging fraction 1.0000); the other 2,229
windows, from cells never touched by this correction, show ZERO bridging.
So the gap is not erased and not undetectable -- every Sim_satellite test
window joins the capacity reading just before a removed sentinel block to
the one just after it as if they were 20 consecutive real cycles, when
they are not. This is the same class of issue as the already-documented
CALCE/RWTH concatenation-boundary limitation (no cell-ID field to mark
internal boundaries), but unlike CALCE/NASA/Oxford's diffuse native
sparsity (see diagnose_bridging_anomaly.py), it is 100% concentrated in
these 8 cells and should be reported as such, with an explicit
with/without-these-376-windows sensitivity check
(xjtu_bridging_sensitivity.py) rather than dismissed as negligible.

Run:
    python preprocessing.py
This caches processed arrays into config.CACHE_DIR as .npz files so you don't
need to re-parse raw files every time you tweak the model.
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
    to CANONICAL_COLUMNS before concatenation (see module docstring, change 1),
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
    cells. sklearn's train_test_split (used previously) applies the val/test
    fraction to whatever's left AFTER carving out train, which can round a
    small remaining pool down to 0 in one of the two splits — that's exactly
    what happened to NASA's 7 cells, silently leaving it with 0 validation
    cells and breaking conformal calibration for that group. This version
    computes explicit integer counts up front and never lets a required
    split fall to 0.

    split_seed: overrides config.SPLIT_SEED for this call (used by the
    repeated-split experiment, K6, to vary the split independently of the
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
    This DOES catch XJTU's Sim_satellite_* cells -- confirmed directly
    (E9): xjtu_loader.py's detect_and_strip_sentinel_values() does not
    re-number 'cycle' after removing rows, so the gap survives and every
    Sim_satellite_* test window bridges it (see this file's module-level
    docstring, 'KNOWN LIMITATION' section, for the verified numbers)."""
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
    split_seed also gets its own suffix (K6 repeated-split experiment), so
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


def run(window: int = None, split_seed: int = None, raw_df=None):
    """window: override config.INPUT_WINDOW for this run (used by the window-size
    ablation in experiments.py). The cell split itself is NOT affected by window
    size for a GIVEN split_seed — it's computed once per seed so every window-size
    variant at that seed trains/tests on the exact same cells, making the
    comparison fair.

    split_seed: override config.SPLIT_SEED (K6 repeated-split experiment). Left
    None, this reproduces the original seed-42 cache exactly.

    raw_df: pre-loaded, pre-SOH'd dataframe (add_soh(load_all_raw())), reused
    across multiple split_seed / window variants by a caller that needs several
    caches built in one run (e.g. the K6 repeated-split experiment) so raw data
    is parsed once rather than once per variant. Pass None to load fresh."""
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

    X_train, y_train, lab_train, cell_train, cyc_train = make_windows(df, train_ids, soh_mean, soh_std, window=w)
    X_val, y_val, lab_val, cell_val, cyc_val = make_windows(df, val_ids, soh_mean, soh_std, window=w)
    X_test, y_test, lab_test, cell_test, cyc_test = make_windows(df, test_ids, soh_mean, soh_std, window=w)

    print(f"[windows] window={w}  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    _print_window_summary("train", lab_train)
    _print_window_summary("val", lab_val)
    _print_window_summary("test", lab_test)

    out_path = cache_path_for(w, split_seed=split_seed)
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