"""
CALCE CS2 / CX2 loader, version 2.

DEFECTS IN THE ORIGINAL LOADER (calce_loader.py, since removed) THAT THIS VERSION ADDRESSES
  1. Session order. v1 sorted workbooks by file NAME and assumed that was
     chronological. CALCE names use unpadded month_day_year (for example
     CS2_35_10_4_10.xlsx sorts before CS2_35_8_17_10.xlsx), so sessions could
     be concatenated out of time order, scrambling the capacity trajectory and
     the "first three cycles" used as the SOH reference. v2 orders by the date
     parsed from the file name, falls back to name order with a printed warning
     when a name has no parseable date, and never reorders silently.
  2. Silent renumbering. v1 replaced Cycle_Index by 1..N over the concatenated
     (file, cycle) groups, so any removed or split group left no trace. v2 keeps
     the original Cycle_Index (`cycle_raw`) and source file, and defines
     `cycle` as the index over ALL groups in chronological order BEFORE any
     dropping, so a dropped group leaves a visible gap instead of being hidden.
  3. Cumulative capacity counter (dominant defect, confirmed on the raw files). v1 took the max of
     Discharge_Capacity(Ah) per (file, Cycle_Index). If that column is a running
     counter within a file, the "capacity" of the k-th cycle is k times the true
     per-cycle capacity. Downstream, SOH = capacity / median(first three rows)
     and rows with SOH >= 1.10 are dropped, which would keep only the first
     ~two rows of each file: exactly the pairs (SOH ~0.5 and ~1.0, ratio 2.0) and
     the ~94% row loss observed in the cached CALCE data, where every other
     dataset keeps >=96% of its rows. v2 detects a cumulative counter per file
     (CAPACITY_MODE="auto") and differences it: per-cycle capacity = max at this
     cycle minus max at the previous cycle in the same file, restarting at each
     file (a decrease within a file is treated as a counter reset).
  4. Partial cycles. A session can begin or end mid-cycle, giving a group whose
     max discharge capacity is only part of a full discharge. Confirmed on the
     raw files as the dominant remaining pattern (analyze_low_side_outliers.py):
     of 141 residual low-capacity outlier rows, 84 are the LAST group of their
     file and 90 are within the file's final 5%, with terminal voltage (median
     3.31 V) consistent with a discharge cut off partway through, not a
     complete discharge (~2.7 V) or a rest/no-discharge step (~4.0 V) -- the
     session most likely ended (equipment stopped, data downloaded) before that
     final logged cycle finished. v2 flags
       - straddles_boundary : the group's Cycle_Index continues across a file
                              boundary (last group of a file == first of the next).
                              Does NOT catch the pattern above, since a truncated
                              final cycle typically does not share its Cycle_Index
                              with the next file's first cycle.
       - complete_discharge : terminal voltage at the max-capacity row is within
                              CUTOFF_TOL_V of the discharge cutoff voltage
                              (NaN if no voltage column is present). Uncalibrated
                              for CALCE: the empirical terminal voltage of ordinary
                              (non-outlier) rows sits well above DISCHARGE_CUTOFF_V
                              + CUTOFF_TOL_V, so this flag is near-universally
                              True and is not currently a useful filter on its own.
       - truncated_final_cycle : the file's LAST group's capacity, AFTER any
                              cumulative-counter differencing (defect 3), is far
                              below that SAME file's own median (a per-file,
                              relative test needing no assumed absolute voltage
                              cutoff). Computed post-differencing deliberately:
                              on the raw values a cumulative file's last group is
                              always its largest, so the rule could never fire
                              there even for a genuinely truncated final cycle.
                              An earlier version computed it on the raw values
                              and never fired on real CALCE data as a result.
  5. Non-cycling session files. A small number of CALCE workbooks are not
     ordinary charge/discharge cycling sessions at all (confirmed example:
     "..._self discharge test.xlsx", which records a rest/self-discharge
     measurement with zero discharge capacity, not a missing or partial
     cycle). Folding such a file into the capacity series produces a
     spurious near-zero reading that a partial-cycle flag would not explain.
     v2 excludes files whose name matches NON_CYCLING_KEYWORDS and always
     prints which ones it drops. This is a narrow, name-based rule for an
     unambiguous case, not a numeric judgment call like INCOMPLETE_POLICY;
     confirm with scan_calce_filenames.py that the keyword list covers what
     is actually present before relying on it.

STATUS. The cumulative-counter mechanism (3) was confirmed on the raw CALCE
workbooks. A first pass classified 331 files as cumulative counters (>=90% of
consecutive groups strictly increasing), recovering 98.6% of all rows to within
tolerance of a smooth series; per-cell diagnostics on the remainder found 219
residual outlier rows (1.5%), concentrated in ~127 short files (5-10 cycles)
that were themselves fully cumulative but missed the 90% threshold because
their last row duplicates the previous cumulative total verbatim (confirmed on
12 such files with inspect_calce_outliers.py: each has zero true decreases and
exactly one trailing tie). The classifier now scores ties as consistent with a
cumulative counter and additionally requires last/first capacity >= 1.5x, which
resolves this without weakening the check that excludes genuinely flat,
non-cumulative files. Partial cycles (4) are
only FLAGGED, not assumed: the flags are diagnostic and INCOMPLETE_POLICY
defaults to "flag" (all groups kept, chronological order fixed). "drop" removes
groups with straddles_boundary == True or truncated_final_cycle == True -- NOT
complete_discharge (see above: uncalibrated, wiring it into "drop" previously
removed 82-100% of some cells' rows). Adopt "drop" only if verify_calce_v2.py's
outliers(drop)/rows(drop) columns confirm it helps on your actual data.
File names without a parseable date fall back to NAME order (printed as a
warning); check those cells with verify_calce_v2.py.

Output columns: cycle, capacity, cell_id, dataset, chemistry (as v1, so
preprocessing.py works unchanged) plus cycle_raw, source_file,
complete_discharge, straddles_boundary, truncated_final_cycle, terminal_voltage.
"""

import os
import re

import numpy as np
import pandas as pd

import config

CANDIDATE_CYCLE_COLS = ["Cycle_Index", "Cycle_Index(#)", "Cycle Index"]
CANDIDATE_CAP_COLS = ["Discharge_Capacity(Ah)", "Discharge_Capacity", "Discharge Capacity (Ah)"]
CANDIDATE_VOLT_COLS = ["Voltage(V)", "Voltage", "Voltage (V)"]

DISCHARGE_CUTOFF_V = 2.7     # verify against each cell's test protocol
CUTOFF_TOL_V = 0.15
INCOMPLETE_POLICY = "flag"   # "flag" (default) or "drop"; see module docstring
NON_CYCLING_KEYWORDS = ["self discharge", "self_discharge", "calendar aging", "shelf test", "storage test"]
EXCLUDE_NON_CYCLING_FILES = True   # drop files matching NON_CYCLING_KEYWORDS; always printed when it fires
TRUNCATED_FINAL_CYCLE_MIN_GROUPS = 4     # need this many groups for "this file's own median" to be meaningful
TRUNCATED_FINAL_CYCLE_RATIO = 0.5        # flag the last group if its capacity < this fraction of the file's own median (other groups)
CAPACITY_MODE = "auto"       # "max" (v1 behaviour), "difference", or "auto" (detect a cumulative counter per file)
CUMULATIVE_MIN_INCREASING = 0.90   # auto: file treated as cumulative if >=90% of consecutive groups increase
CUMULATIVE_MIN_GROUPS = 4          # ...and it has at least this many groups


def _find_col(columns, candidates):
    return next((c for c in candidates if c in columns), None)


def _parse_date(fname):
    m = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{2})\.xls", fname.lower())
    return None if not m else (int(m.group(3)), int(m.group(1)), int(m.group(2)))


def _is_non_cycling(fname):
    low = fname.lower()
    return any(k in low for k in NON_CYCLING_KEYWORDS)


def _ordered_files(cell_dir, cell_id):
    paths = []
    for dirpath, _, filenames in os.walk(cell_dir):
        paths += [os.path.join(dirpath, f) for f in filenames if f.lower().endswith((".xlsx", ".xls"))]
    if not paths:
        raise FileNotFoundError(f"No excel files found under {cell_dir} (searched recursively)")
    if EXCLUDE_NON_CYCLING_FILES:
        keep, dropped = [], []
        for p in paths:
            (dropped if _is_non_cycling(os.path.basename(p)) else keep).append(p)
        if dropped:
            print(f"  [CALCE] {cell_id}: excluding {len(dropped)} non-cycling file(s): "
                  f"{[os.path.basename(p) for p in dropped]}")
        paths = keep
        if not paths:
            raise FileNotFoundError(f"All files under {cell_dir} matched NON_CYCLING_KEYWORDS")
    dates = {p: _parse_date(os.path.basename(p)) for p in paths}
    if all(v is not None for v in dates.values()):
        return sorted(paths, key=lambda p: (dates[p], os.path.basename(p)))
    print(f"  [CALCE] {cell_id}: cannot parse a date from every file name; falling back to NAME order "
          f"(may not be chronological)")
    return sorted(paths, key=os.path.basename)


def _file_groups(path):
    xls = pd.ExcelFile(path)
    out = []
    for sheet in xls.sheet_names:
        if "channel" not in sheet.lower():
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        c = _find_col(df.columns, CANDIDATE_CYCLE_COLS)
        q = _find_col(df.columns, CANDIDATE_CAP_COLS)
        v = _find_col(df.columns, CANDIDATE_VOLT_COLS)
        if c is None or q is None:
            continue
        sub = df[[c, q] + ([v] if v else [])].dropna(subset=[c, q]).reset_index(drop=True)
        for cyc, g in sub.groupby(c, sort=False):
            i = g[q].idxmax()
            out.append({"cycle_raw": cyc, "capacity": float(g[q].max()),
                        "v_at_max": float(sub.loc[i, v]) if v else np.nan})
    return out


MIN_GROWTH_RATIO = 1.5   # auto (>=4 groups): also require last/first >= this, to reject a genuinely
                         # flat file that happens to have no decreasing step by chance


def _flag_truncated_final_cycle(groups):
    """Per-file, relative test for a session that ended mid-discharge on its
    last logged cycle (see defect 4): the last group's capacity far below the
    median of the file's OTHER groups. Needs no assumed absolute voltage."""
    n = len(groups)
    flags = [False] * n
    if n >= TRUNCATED_FINAL_CYCLE_MIN_GROUPS:
        others = [g["capacity"] for g in groups[:-1]]
        med = float(np.median(others)) if others else 0.0
        if med > 0 and groups[-1]["capacity"] < TRUNCATED_FINAL_CYCLE_RATIO * med:
            flags[-1] = True
    return flags


def _apply_capacity_mode(groups, fname):
    """Convert per-(file, cycle) max capacity into per-cycle capacity when the
    column is a cumulative counter within the file."""
    caps = [g["capacity"] for g in groups]
    mode = CAPACITY_MODE
    if mode == "auto":
        cumulative = False
        if len(caps) >= CUMULATIVE_MIN_GROUPS:
            # A tie (equal consecutive readings) is scored as "not decreasing", not as evidence
            # against a cumulative counter: several short CALCE sessions end with the last row
            # repeating the previous cumulative total verbatim (no further discharge recorded in
            # that file), which is a boundary artifact, not a sign of a real per-cycle reading. A
            # true decrease still counts against cumulative, since a within-file reset is possible.
            # Verified against 12 real CALCE files, each ending in exactly one such tie and
            # otherwise strictly increasing (inspect_calce_outliers.py).
            not_decreasing = np.mean([caps[k + 1] >= caps[k] for k in range(len(caps) - 1)])
            growth = (caps[-1] / caps[0]) if caps[0] > 0 else 0.0
            cumulative = bool(not_decreasing >= CUMULATIVE_MIN_INCREASING and growth >= MIN_GROWTH_RATIO)
        elif len(caps) >= 2 and caps[0] > 0:
            # Short file: consecutive genuine cycles differ by well under 1%, whereas a running
            # counter gives ratios of about 2 (second/first) and 3 (third/first).
            ok = 1.85 <= caps[1] / caps[0] <= 2.15
            if len(caps) == 3:
                ok = ok and 2.7 <= caps[2] / caps[0] <= 3.3
            cumulative = bool(ok)
        mode = "difference" if cumulative else "max"
        if cumulative:
            print(f"  [CALCE] {fname}: Discharge_Capacity behaves as a cumulative counter within the file; differencing")
    if mode == "max":
        return groups
    out = []
    for k, g in enumerate(groups):
        c = caps[k] if k == 0 or caps[k] < caps[k - 1] else caps[k] - caps[k - 1]
        out.append({**g, "capacity": float(c)})
    return out


def load_calce_cell(cell_dir: str, cell_id: str, chemistry: str,
                    policy: str = None) -> pd.DataFrame:
    policy = policy or INCOMPLETE_POLICY
    if policy not in ("flag", "drop"):
        raise ValueError("policy must be 'flag' or 'drop'")

    files = _ordered_files(cell_dir, cell_id)
    rows = []
    for path in files:
        fname = os.path.basename(path)
        try:
            groups = _file_groups(path)
        except Exception as e:
            print(f"  [CALCE] could not read {path}: {e}")
            continue
        # Truncation must be tested on the FINAL per-cycle capacity, i.e. AFTER differencing, not
        # before: in a file classified as a cumulative counter (defect 3), the RAW running total is
        # monotonically increasing, so its last group is always the LARGEST value in the file, never
        # the smallest -- the rule could never fire there even when the true final per-cycle capacity
        # (after differencing) is genuinely truncated. Confirmed with a synthetic cumulative file
        # whose last group differences to 0.1 Ah against ~1.0 Ah elsewhere: flagged correctly only
        # when computed on the post-differencing values, never on the raw ones.
        groups = _apply_capacity_mode(groups, fname)
        trunc_flags = _flag_truncated_final_cycle(groups)
        for gi, g in enumerate(groups):
            rows.append({**g, "source_file": fname, "_first": gi == 0, "_last": gi == len(groups) - 1,
                         "_n": len(groups), "truncated_final_cycle": trunc_flags[gi]})
    if not rows:
        raise RuntimeError(f"Parsed nothing for {cell_dir} -- check column names")

    df = pd.DataFrame(rows)
    df["cycle"] = np.arange(1, len(df) + 1)          # index over ALL groups, before any dropping

    straddle = np.zeros(len(df), dtype=bool)
    for i in range(len(df) - 1):
        # Cycle_Index restarts in every file, so two one-cycle files trivially share cycle_raw == 1;
        # require both files to hold at least two groups before calling it a split cycle.
        if df.loc[i, "_n"] >= 2 and df.loc[i + 1, "_n"] >= 2 and \
           df.loc[i, "_last"] and df.loc[i + 1, "_first"] and \
           df.loc[i, "source_file"] != df.loc[i + 1, "source_file"] and \
           df.loc[i, "cycle_raw"] == df.loc[i + 1, "cycle_raw"]:
            straddle[i] = straddle[i + 1] = True
    df["straddles_boundary"] = straddle

    if df["v_at_max"].notna().any():
        df["complete_discharge"] = df["v_at_max"] <= DISCHARGE_CUTOFF_V + CUTOFF_TOL_V
    else:
        df["complete_discharge"] = np.nan

    if policy == "drop":
        # complete_discharge is deliberately excluded: it is uncalibrated for CALCE (see module
        # docstring) and wiring it in previously caused policy="drop" to remove 82-100% of some
        # cells' rows. Report it, don't act on it, until it has a validated cutoff.
        bad = df["straddles_boundary"] | df["truncated_final_cycle"]
        print(f"  [CALCE] {cell_id}: policy=drop removes {int(bad.sum())} of {len(df)} groups "
              f"(straddles_boundary or truncated_final_cycle; complete_discharge not used, see docstring)")
        df = df[~bad]

    df["terminal_voltage"] = df["v_at_max"]
    df["cell_id"] = cell_id
    df["dataset"] = "CALCE"          # grouped as one dataset, as in v1
    df["chemistry"] = chemistry
    return df[["cycle", "capacity", "cell_id", "dataset", "chemistry",
               "cycle_raw", "source_file", "complete_discharge", "straddles_boundary",
               "truncated_final_cycle", "terminal_voltage"]].reset_index(drop=True)


def load_all() -> pd.DataFrame:
    frames = []
    for chemistry, dirs in config.CALCE_DIRS.items():
        for d in dirs:
            cell_id = os.path.basename(d.rstrip("/\\"))
            if not os.path.isdir(d):
                print(f"  [CALCE] folder not found, skipping: {d}")
                continue
            try:
                frames.append(load_calce_cell(d, cell_id, chemistry))
            except Exception as e:
                print(f"  [CALCE] skipped {cell_id}: {e}")
    if not frames:
        raise RuntimeError("CALCE: no cells parsed")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_all()
    print(df.groupby("cell_id").size())
    print(df.head())