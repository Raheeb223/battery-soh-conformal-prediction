"""
Significance testing and data-quality checks on the completed runs.

Nothing is retrained; the script reads saved *_predictions.npz files, result
JSONs, and the raw dataset folders.

  - run_significance_testing(): paired Wilcoxon signed-rank tests on per-sample
    absolute errors of the proposed model vs each baseline/ablation, with
    Cliff's delta and bootstrap CIs of the MAE difference
    (-> outputs/significance_testing_results.json, used by
    generate_all_figures.py).
  - diagnose_window_nasa_dropout(): why NASA contributes no windows at window
    sizes 30 and 50.
  - diagnose_bit_cell_count(): BIT cells used (72) vs cells released (77).
  - diagnose_xjtu_anomaly(), scan_for_duplicate_runs(),
    locate_xjtu_padding_bug(): checks for repeated/sentinel capacity values in
    XJTU.

Run from the repository root (the directory containing config.py):
    python remaining_analysis.py
"""

import os
import sys
import json
import glob
import numpy as np

import config  # project config.py

try:
    from scipy.stats import wilcoxon
except ImportError:
    print("[!] scipy not found -- pip install scipy --break-system-packages")
    sys.exit(1)

OUTPUT_DIR = config.OUTPUT_DIR
DATASET_ORDER = ["MIT", "BIT", "XJTU", "RWTH", "CALCE", "Oxford", "NASA"]

# ---------------------------------------------------------------------------
# Helper: load a run's per-sample predictions
# ---------------------------------------------------------------------------

def _load_predictions(run_name):
    candidates = glob.glob(os.path.join(OUTPUT_DIR, f"*{run_name}*predictions.npz"))
    if not candidates:
        return None
    data = np.load(candidates[0], allow_pickle=True)
    required = {"y_test", "mean_pred", "lab_test"}
    if not required.issubset(set(data.files)):
        print(f"  [!] {candidates[0]} is missing expected keys {required - set(data.files)}")
        return None
    return data


def _cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n_x, n_y = len(x), len(y)
    gt = 0
    lt = 0
    chunk = 2000
    for i in range(0, n_x, chunk):
        xi = x[i:i + chunk][:, None]
        diff = xi - y[None, :]
        gt += np.sum(diff > 0)
        lt += np.sum(diff < 0)
    return (gt - lt) / (n_x * n_y)


def _bootstrap_ci_mae_diff(err_a, err_b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(err_a)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = np.mean(err_a[idx]) - np.mean(err_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def _effect_size_label(delta):
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# 1. Significance testing
# ---------------------------------------------------------------------------

def run_significance_testing():
    print("\n=== 1. Significance testing (fills the [pending] table in Sec. 4.3) ===")

    proposed = _load_predictions("ablation_full")
    if proposed is None:
        print("  [skip] could not find ablation_full predictions in outputs/ -- "
              "check that _load_predictions()'s glob pattern matches the files present.")
        return

    prop_err = np.abs(proposed["mean_pred"] - proposed["y_test"])

    comparisons = [
        "baseline_linear", "baseline_svr", "baseline_xgboost",
        "ablation_no_attention", "ablation_no_multiscale",
        "ablation_plain_lstm", "ablation_gru",
    ]

    results = []
    for name in comparisons:
        other = _load_predictions(name)
        if other is None:
            print(f"  [skip] no predictions file found for {name}")
            continue

        if len(other["y_test"]) != len(proposed["y_test"]):
            print(f"  [!] WARNING: {name} has {len(other['y_test'])} test "
                  f"samples vs. proposed's {len(proposed['y_test'])} -- "
                  f"these are NOT the same test set. Skipping to avoid an "
                  f"invalid paired comparison.")
            continue

        other_err = np.abs(other["mean_pred"] - other["y_test"])

        stat, p_value = wilcoxon(prop_err, other_err)
        delta = _cliffs_delta(prop_err, other_err)
        ci_lo, ci_hi = _bootstrap_ci_mae_diff(prop_err, other_err)

        results.append({
            "comparison": f"proposed_vs_{name}",
            "n_samples": len(prop_err),
            "proposed_mae": float(np.mean(prop_err)),
            "other_mae": float(np.mean(other_err)),
            "wilcoxon_p_value": float(p_value),
            "cliffs_delta": float(delta),
            "effect_size": _effect_size_label(delta),
            "mae_diff_95ci": [ci_lo, ci_hi],
        })
        print(f"  {name:28s}  p={p_value:.2e}  delta={delta:+.3f} "
              f"({_effect_size_label(delta)})  MAE diff 95% CI=[{ci_lo:.5f}, {ci_hi:.5f}]")

    out_path = os.path.join(OUTPUT_DIR, "significance_testing_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Shared helper: group ANY loader's return value (dict, list, or a single
# merged/flat DataFrame) into proper {cell_id: per-cell-object} pairs.
# This is what item 5 (and item 2, for symmetry) were missing before --
# without it, enumerate()'ing a merged DataFrame iterates over COLUMN
# NAMES, not rows, which silently produced the bogus "cell_0: 5 cycles"
# result last run.
# ---------------------------------------------------------------------------

def _cell_groups(obj):
    """Returns a list of (cell_id, sub_object) pairs regardless of whether
    the loader returns a dict, a list/tuple of per-cell objects, or a
    single merged/flat DataFrame (the actual case for xjtu_loader.load_all()
    and bit_loader.load_all(), confirmed by section 3's "returns a
    DataFrame" log line). For the DataFrame case, groups by whichever
    cell-identifier column is actually present -- trying common names --
    rather than treating each row or column as a "cell"."""
    if isinstance(obj, dict):
        return list(obj.items())

    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            id_col = None
            for col in ("cell_id", "cell", "cell_name", "id", "battery_id", "cell_no"):
                if col in obj.columns:
                    id_col = col
                    break
            if id_col is None:
                print(f"  [!] WARNING: DataFrame has no recognizable cell-ID "
                      f"column (checked cell_id/cell/cell_name/id/battery_id/"
                      f"cell_no; actual columns: {list(obj.columns)}). "
                      f"Cannot group into real cells -- add the correct "
                      f"column name to this function's search list.")
                return []
            groups = list(obj.groupby(id_col))
            return [(str(cid), sub) for cid, sub in groups]
    except ImportError:
        pass

    if isinstance(obj, (list, tuple)):
        return [(f"cell_{i}", seq) for i, seq in enumerate(obj)]

    print(f"  [!] WARNING: don't know how to group object of type "
          f"{type(obj).__name__} into cells.")
    return []


def _sequence_length(sub):
    """Best-effort row/cycle count for one cell's sub-object (DataFrame,
    Series, dict, or plain array)."""
    try:
        import pandas as pd
        if isinstance(sub, (pd.DataFrame, pd.Series)):
            return len(sub)
    except ImportError:
        pass
    if isinstance(sub, dict):
        for key in ("soh", "SOH", "capacity", "Capacity"):
            if key in sub:
                return len(sub[key])
        return len(next(iter(sub.values())))
    return len(sub)


def _count_cells(obj):
    if isinstance(obj, dict):
        return len(obj)
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            for col in ("cell_id", "cell", "cell_name", "id"):
                if col in obj.columns:
                    return obj[col].nunique()
            return len(obj)
    except ImportError:
        pass
    if isinstance(obj, (list, tuple)):
        return len(obj)
    return len(obj)


def _find_loader_function(module_name, name_hints):
    import importlib
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        print(f"  [skip] could not import {module_name}: {e}")
        return None, None

    for hint in name_hints:
        if hasattr(mod, hint):
            return mod, getattr(mod, hint)

    public_callables = [
        n for n in dir(mod)
        if not n.startswith("_") and callable(getattr(mod, n))
    ]
    print(f"  [!] none of {name_hints} found in {module_name}. "
          f"Public callables actually defined there: {public_callables}")
    print(f"      Edit this script's name_hints list for {module_name} to "
          f"include the correct one, then re-run.")
    return mod, None


def _call_loader(fn, dir_attr):
    try:
        return fn()
    except TypeError:
        data_dir = getattr(config, dir_attr, None)
        if data_dir is None:
            raise
        return fn(data_dir)


# ---------------------------------------------------------------------------
# 2. NASA window=30/50 dropout -- unchanged, already correct (this one
#    was never affected by the DataFrame-enumeration bug since NASA's
#    loader apparently returns a dict of per-cell objects, confirmed by
#    last run's correct-looking cell_3/n_cycles=7 output).
# ---------------------------------------------------------------------------

def diagnose_window_nasa_dropout():
    print("\n=== 2. NASA window=30/50 dropout diagnosis ===")

    mod, load_nasa = _find_loader_function(
        "nasa_loader",
        ["load_all", "load_nasa", "load", "load_cells", "get_cells",
         "load_dataset", "load_nasa_cells", "nasa_loader", "load_data"],
    )
    if load_nasa is None:
        return

    cells = _call_loader(load_nasa, "NASA_DIR")
    groups = _cell_groups(cells)
    cell_dict = {cid: _sequence_length(sub) for cid, sub in groups}

    report_lines = []
    for w in (10, 20, 30, 50):
        report_lines.append(f"\n--- window={w} ---")
        total_valid_windows = 0
        for cell_id, n_cycles in cell_dict.items():
            n_windows = max(0, n_cycles - w)
            total_valid_windows += n_windows
            flag = "  <-- TOO SHORT FOR THIS WINDOW" if n_windows <= 0 else ""
            report_lines.append(f"  {cell_id:20s} n_cycles={n_cycles:5d}  "
                                 f"valid_windows={n_windows:5d}{flag}")
        report_lines.append(f"  TOTAL valid NASA windows at w={w}: {total_valid_windows}")
        if total_valid_windows == 0:
            report_lines.append(f"  ==> This confirms window={w} produces ZERO "
                                 f"usable NASA windows -- a real data constraint.")

    out_path = os.path.join(OUTPUT_DIR, "window_nasa_dropout_report.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(report_lines))
    print(f"  saved {out_path}")
    print("  ".join(report_lines[-3:]))


# ---------------------------------------------------------------------------
# 3. BIT cell count -- unchanged, already worked correctly last run.
# ---------------------------------------------------------------------------

def diagnose_bit_cell_count():
    print("\n=== 3. BIT cell-count discrepancy (72 used vs. 77 released) ===")

    bit_dir = getattr(config, "BIT_DIR", None)
    if bit_dir is None or not os.path.isdir(bit_dir):
        print(f"  [skip] config.BIT_DIR not set or not found ({bit_dir}).")
        return

    raw_files = sorted(
        glob.glob(os.path.join(bit_dir, "**", "*.csv"), recursive=True) +
        glob.glob(os.path.join(bit_dir, "**", "*.xlsx"), recursive=True) +
        glob.glob(os.path.join(bit_dir, "**", "*.mat"), recursive=True)
    )
    print(f"  Found {len(raw_files)} raw cell files under {bit_dir}")

    mod, load_bit = _find_loader_function(
        "bit_loader",
        ["load_all", "load_bit", "load", "load_cells", "get_cells",
         "load_dataset", "load_bit_cells", "bit_loader", "load_data"],
    )
    if load_bit is not None:
        used_cells = _call_loader(load_bit, "BIT_DIR")
        n_used = _count_cells(used_cells)
        print(f"  Loader (bit_loader.{load_bit.__name__}) returns a "
              f"{type(used_cells).__name__} -- inferred cell count: {n_used}")
    else:
        n_used = None

    report_lines = [f"Raw BIT files found: {len(raw_files)}"]
    report_lines += [f"  {f}" for f in raw_files]
    if n_used is not None:
        report_lines.append(f"\nCells returned by bit_loader.load_bit(): {n_used}")

    out_path = os.path.join(OUTPUT_DIR, "bit_cell_count_report.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(report_lines))
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# 4. XJTU anomaly -- unchanged, already correctly found the padding artifact.
# ---------------------------------------------------------------------------

def diagnose_xjtu_anomaly():
    print("\n=== 4. XJTU anomaly cell identification ===")

    proposed = _load_predictions("conformal_normalized_groupwise")
    if proposed is None:
        proposed = _load_predictions("ablation_full")
    if proposed is None:
        print("  [skip] no predictions file found.")
        return

    y_true = proposed["y_test"]
    y_pred = proposed["mean_pred"]
    lab = proposed["lab_test"]
    mask = lab == "XJTU"
    if not mask.any():
        print("  [skip] no XJTU samples found in this predictions file.")
        return

    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    print(f"  XJTU test samples: {n}")

    median = np.median(yt)
    mad = np.median(np.abs(yt - median)) or 1e-9
    robust_z = (yt - median) / (1.4826 * mad)

    print(f"  True-value range: [{yt.min():.3f}, {yt.max():.3f}], median={median:.3f}")

    mag_threshold = -6.0
    mag_outlier_idx = np.where(robust_z < mag_threshold)[0]
    print(f"  Samples with robust z-score < {mag_threshold}: "
          f"{len(mag_outlier_idx)} ({100*len(mag_outlier_idx)/n:.2f}%)")

    padding_artifact_suspected = False
    if len(mag_outlier_idx) > 0:
        outlier_vals = yt[mag_outlier_idx]
        val_min, val_max = outlier_vals.min(), outlier_vals.max()
        print(f"  Magnitude-outlier index range: "
              f"{mag_outlier_idx.min()}-{mag_outlier_idx.max()}, "
              f"true-value range there: [{val_min:.3f}, {val_max:.3f}]")
        if (val_max - val_min) < 1e-4:
            padding_artifact_suspected = True
            print(f"  [!!!] CRITICAL: these {len(mag_outlier_idx)} 'outlier' "
                  f"values are IDENTICAL to 4 decimal places.")

    lo, hi = min(2060, n - 1), min(2400, n)
    print(f"  Direct inspection of index range [{lo}:{hi}]: true values "
          f"[{yt[lo:hi].min():.3f}, {yt[lo:hi].max():.3f}], "
          f"predicted [{yp[lo:hi].min():.3f}, {yp[lo:hi].max():.3f}]")

    cell_id = None
    for key in ("cell_id_test", "cell_test", "cell_ids_test"):
        if key in proposed.files:
            cell_id = proposed[key][mask]
            break

    report_lines = [
        f"XJTU test samples: {n}",
        f"True-value range: [{yt.min():.3f}, {yt.max():.3f}], median={median:.3f}",
        f"\nMagnitude-outlier detection (robust z-score < {mag_threshold}):",
        f"  count: {len(mag_outlier_idx)}",
        f"  index range: {mag_outlier_idx.min() if len(mag_outlier_idx) else 'n/a'}"
        f"-{mag_outlier_idx.max() if len(mag_outlier_idx) else 'n/a'}",
    ]
    if padding_artifact_suspected:
        report_lines.append(
            "\n[!!!] CRITICAL: padding/duplication artifact confirmed -- "
            "these samples should be excluded and XJTU stats recomputed."
        )
    if cell_id is not None:
        outlier_idx = mag_outlier_idx
        outlier_cells = sorted(set(cell_id[outlier_idx])) if len(outlier_idx) else []
        report_lines.append(f"\nSpecific XJTU cell IDs responsible: {outlier_cells}")
        print(f"  Specific cell(s) responsible: {outlier_cells}")
    else:
        report_lines.append(
            "\nNo per-window cell-identifier field found in this predictions "
            "file -- cannot map the anomaly to a specific cell ID directly."
        )
        print("  [!] No cell-ID field available.")

    out_path = os.path.join(OUTPUT_DIR, "xjtu_anomaly_report.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(report_lines))
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# 5. Locate the actual XJTU padding bug -- FIXED to use _cell_groups()
#    (DataFrame-aware, not enumerate()-over-columns) and to add a direct
#    duplicate-run scan, which is the actual unambiguous signature of
#    padding regardless of cell-count comparisons.
# ---------------------------------------------------------------------------

def scan_for_duplicate_runs(min_run=5):
    """Directly scans every real XJTU cell's raw sequence for a run of
    >= min_run identical consecutive values in any numeric column -- the
    unambiguous signature of padding/duplication, independent of which
    cell or column is involved. This replaces relying on (buggy) cycle-
    count comparisons."""
    print(f"\n  Scanning for duplicate-value runs (>= {min_run} identical "
          f"consecutive values) in every XJTU cell's raw sequence:")

    mod, load_xjtu = _find_loader_function(
        "xjtu_loader",
        ["load_all", "load_xjtu", "load", "load_cells", "get_cells",
         "load_dataset", "load_xjtu_cells", "load_data"],
    )
    if load_xjtu is None:
        return []

    try:
        cells = _call_loader(load_xjtu, "XJTU_DIR")
    except Exception as e:
        print(f"  [!] load_xjtu() raised an error: {e}")
        return []

    groups = _cell_groups(cells)
    if not groups:
        print("  [skip] could not group XJTU data into real cells.")
        return []

    import pandas as pd
    findings = []
    for cell_id, sub in groups:
        if not isinstance(sub, pd.DataFrame):
            continue
        numeric_cols = sub.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            vals = sub[col].to_numpy()
            if len(vals) < min_run:
                continue
            # find runs of identical consecutive values
            run_start = 0
            for i in range(1, len(vals) + 1):
                same = i < len(vals) and vals[i] == vals[i - 1]
                if not same:
                    run_len = i - run_start
                    if run_len >= min_run:
                        findings.append({
                            "cell_id": str(cell_id),
                            "column": col,
                            "run_start_row": run_start,
                            "run_len": run_len,
                            "value": float(vals[run_start]),
                        })
                        print(f"    [FOUND] cell={cell_id}  col={col}  "
                              f"rows[{run_start}:{run_start+run_len}]  "
                              f"run_len={run_len}  value={vals[run_start]}")
                    run_start = i

    if not findings:
        print("  No duplicate-value runs >= min_run found in any XJTU cell's "
              "raw data -- the padding may be introduced downstream, in the "
              "windowing/batching step rather than the raw per-cell data "
              "itself (check preprocessing.py's window-slicing/batching code "
              "next, not just the loader).")
    return findings


def locate_xjtu_padding_bug():
    print("\n=== 5. Locating the XJTU padding/duplication bug at its source ===")

    report_lines = []

    mod, load_xjtu = _find_loader_function(
        "xjtu_loader",
        ["load_all", "load_xjtu", "load", "load_cells", "get_cells",
         "load_dataset", "load_xjtu_cells", "load_data"],
    )
    if load_xjtu is not None:
        try:
            cells = _call_loader(load_xjtu, "XJTU_DIR")
        except Exception as e:
            print(f"  [!] load_xjtu() raised an error: {e}")
            cells = None

        if cells is not None:
            groups = _cell_groups(cells)
            if not groups:
                report_lines.append(
                    "Could not group XJTU loader output into real cells -- "
                    "see the WARNING above about the missing cell-ID column."
                )
            else:
                lengths = [(cid, _sequence_length(sub)) for cid, sub in groups]
                lengths.sort(key=lambda t: (t[1] is None, t[1]))

                print(f"  XJTU cells by cycle count (shortest first):")
                report_lines.append("XJTU cells by row/cycle count (shortest first):")
                for cell_id, n in lengths[:10]:
                    line = f"    {cell_id}: {n} rows"
                    print(line)
                    report_lines.append(line)
    else:
        report_lines.append(
            "Could not auto-detect xjtu_loader's load function."
        )

    # The real, unambiguous check -- run this regardless of whether the
    # cycle-count listing above found a clean single short-cell suspect.
    findings = scan_for_duplicate_runs(min_run=5)
    if findings:
        report_lines.append("\nDuplicate-value runs found (the actual padding signature):")
        for f in findings:
            report_lines.append(
                f"  cell={f['cell_id']}  col={f['column']}  "
                f"rows[{f['run_start_row']}:{f['run_start_row']+f['run_len']}]  "
                f"run_len={f['run_len']}  value={f['value']}"
            )
        report_lines.append(
            "\n==> These are the exact cell(s)/column(s)/rows responsible for "
            "the padding artifact. Fix the loader/windowing code so these "
            "rows are not duplicated (or exclude them), then re-run "
            "experiments.py for XJTU before trusting its MAE/RMSE/coverage."
        )
    else:
        report_lines.append(
            "\nNo duplicate-value runs found in raw per-cell data -- padding "
            "is likely introduced later, in preprocessing.py's window-slicing "
            "or batch-padding step. Check there next (search for pad/tile/"
            "np.full/resize calls applied to test windows after cell-level "
            "loading)."
        )

    print("\n  Static-searching project source files for padding/repeat "
          "patterns (widened -- no longer requires the word 'xjtu' nearby, "
          "since xjtu_loader.py/preprocessing.py are already XJTU-specific "
          "by nature):")
    report_lines.append("\nStatic search of source files for padding/repeat patterns:")

    suspect_keywords = [
        "pad", "repeat", "tile", "ffill", "fillna", "bfill",
        "np.full", "broadcast_to", "resize", "iloc[-1]", "min_cycles",
    ]
    candidate_files = [
        "xjtu_loader.py", "preprocessing.py", "config.py", "train.py",
        "experiments.py", "model.py",
    ]
    project_root = os.path.dirname(os.path.abspath(config.__file__))

    for fname in candidate_files:
        fpath = os.path.join(project_root, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", errors="ignore") as fh:
            lines = fh.readlines()

        is_xjtu_specific_file = "xjtu" in fname.lower()

        hits = []
        for i, line in enumerate(lines, start=1):
            lowered = line.lower()
            if not any(kw in lowered for kw in suspect_keywords):
                continue
            # Widened: a file whose whole purpose is XJTU (xjtu_loader.py,
            # or generically preprocessing.py which handles XJTU among
            # others) doesn't need "xjtu" literally nearby every line --
            # only require the proximity check for genuinely shared/
            # generic files where we need to disambiguate dataset scope.
            if is_xjtu_specific_file:
                hits.append((i, line.rstrip()))
            elif "xjtu" in "".join(lines[max(0, i - 6):i + 5]).lower():
                hits.append((i, line.rstrip()))

        if hits:
            print(f"  {fname}: {len(hits)} candidate line(s)")
            report_lines.append(f"\n{fname}: {len(hits)} candidate line(s)")
            for lineno, text in hits:
                # flag the known false-lead pattern explicitly so it isn't
                # mistaken for the real cause again
                note = ""
                if "repeat_interleave" in text and fname == "model.py":
                    note = ("  <-- FALSE LEAD: this is the multi-scale "
                            "branch's PREDICTED-value upsampling in the "
                            "forward pass; it cannot touch y_test (ground "
                            "truth), which is where the duplication actually "
                            "lives. Already checked and ruled out.")
                print(f"    L{lineno}: {text.strip()}{note}")
                report_lines.append(f"    L{lineno}: {text.strip()}{note}")

    out_path = os.path.join(OUTPUT_DIR, "xjtu_padding_bug_location.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(report_lines))
    print(f"  saved {out_path}")


def run_all():
    print("[remaining_analysis] extracting remaining manuscript gaps from "
          "already-completed runs...")
    run_significance_testing()
    diagnose_window_nasa_dropout()
    diagnose_bit_cell_count()
    diagnose_xjtu_anomaly()
    locate_xjtu_padding_bug()
    print("\n[remaining_analysis] done.")


if __name__ == "__main__":
    run_all()