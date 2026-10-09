"""
RWTH Aachen battery aging dataset loader.

Raw structure: a single flat folder of 2216 CSVs (Basytec cycler export,
German column names), one file per test SESSION, not one file per cell's
full life. Each physical cell ('ep sanyo NNN', confirmed via inspection —
NOT the 'Kreis N-NNN' test-channel token, which is a REUSED equipment slot
shared across different physical cells over time) has 24-41 session files
spanning its full cycling history, which must be concatenated chronologically.

EIS (impedance spectroscopy) files are excluded — they measure impedance,
not cycling capacity, and would corrupt a capacity-fade curve if included.

Column mapping (German Basytec export):
  Zyklus  = cycle number (within this session file)
  AhEla   = Ah "Entladen" = CUMULATIVE discharged Ah since test start —
            NOT a per-cycle value. Per-cycle capacity is derived as
            max(AhEla) - min(AhEla) within each cycle's rows (confirmed
            necessary after an early run showed capacity values increasing
            ~1.08/cycle instead of fading — a clear sign of an unremoved
            cumulative baseline).
  Zeit    = timestamp

Each CSV's row 0 is the header, row 1 is a units row (skipped), data starts
row 2. Cycle numbers RESET per session file, so after taking each session's
per-cycle capacity, sessions are concatenated in chronological order (by
the timestamp embedded in the filename) and cycles are renumbered globally
per cell — same approach used for CALCE's multi-session files.

Chemistry: cells are Sanyo-manufactured (per 'ep sanyo' naming) — verify
exact chemistry (the dataset is commonly cited as NMC) against RWTH's own
publication describing this dataset before stating it precisely in your
methods section.

--- SESSION-BOUNDARY PARTIAL-CYCLE FIX ---
Confirmed root cause of a systematic artifact affecting ALL 48 cells:
AhEla resets to (near) zero at the START of every new session file's
logging window, not at the true start of whatever cycle happens to be
in progress at that moment. If a session begins mid-cycle (routine --
cyclers get paused/restarted for many operational reasons), the FIRST
Zyklus group's max(AhEla)-min(AhEla) only spans the PARTIAL remainder
of that cycle actually captured in this session, not the true full-cycle
discharge -- producing an artificially low capacity reading. Diagnosed
via diagnose_rwth_local_outliers.py: this reading is remarkably
consistent at ~26-27% of true baseline capacity across many independent
cells (e.g. 0.288/1.076=26.8%, 0.281/1.059=26.5%, 0.283/1.064=26.6%),
which is inconsistent with random sensor noise and consistent with a
structural, mechanism-level truncation.

Fix: drop the first Zyklus group of every session (see
_parse_session_file()'s new `drop_first=True` behavior) -- this is a
STRUCTURAL fix (removes the row that is unreliable BY CONSTRUCTION),
not a statistical outlier guess. strip_local_outliers() is still
applied afterward in load_rwth_cell() as a safety net for a separate,
less common failure mode (e.g. RWTH_ep_sanyo_019's multi-row
near-zero stretch, which looks like a genuinely corrupted session's
AhEla trace rather than a boundary-truncation artifact, and needs
statistical detection since there's no single structural rule for it).
"""

import os
import re
import pandas as pd
import config
from rwth_local_outlier_fix import strip_local_outliers

RWTH_ROOT = getattr(config, "RWTH_DIR", None)


def _extract_timestamp(fname: str):
    """Filenames embed a timestamp like '2013-01-04 144456' — used to sort
    a cell's session files into chronological order."""
    m = re.search(r"(\d{4}-\d{2}-\d{2} \d{6})", fname)
    return m.group(1) if m else fname  # fallback: alphabetical if no match


def _parse_session_file(path: str, drop_first: bool = True) -> pd.DataFrame:
    """Returns (cycle_within_session, capacity) pairs for one session file.

    drop_first=True (default): drops the first Zyklus group in this
    session, since AhEla resets at the session's logging start rather
    than at the true start of whatever cycle was in progress -- making
    the first group's max(AhEla)-min(AhEla) a partial, unreliable
    capacity reading by construction (confirmed empirically: this
    consistently reads ~26-27% of true baseline across many cells,
    diagnose_rwth_local_outliers.py). Only skips dropping if a session
    has just one Zyklus group total (dropping it would discard the
    whole session for no benefit -- keep it and let the statistical
    safety net in strip_local_outliers() catch it if it's genuinely bad).
    """
    df = pd.read_csv(path, skiprows=[1])  # row 0 = header, row 1 = units (skip)
    if "Zyklus" not in df.columns or "AhEla" not in df.columns:
        raise KeyError(f"Expected columns not found; got: {list(df.columns)}")

    df = df[df["Zyklus"] > 0]  # exclude pre-test/conditioning rows at cycle 0
    if df.empty:
        return pd.DataFrame(columns=["cycle_in_session", "capacity"])

    # IMPORTANT: AhEla is a CUMULATIVE counter (total Ah discharged since the
    # test began), not a per-cycle value. Taking max(AhEla) per cycle would
    # just capture the ever-growing running total (confirmed empirically —
    # early debugging showed capacity increasing ~1.08/cycle instead of
    # fading). The actual Ah discharged DURING a given cycle is the
    # within-cycle range: max(AhEla) - min(AhEla) for that Zyklus group.
    per_cycle = (
        df.groupby("Zyklus")["AhEla"]
        .agg(lambda x: x.max() - x.min())
        .reset_index()
        .rename(columns={"Zyklus": "cycle_in_session", "AhEla": "capacity"})
        .sort_values("cycle_in_session")
        .reset_index(drop=True)
    )

    if drop_first and len(per_cycle) > 1:
        per_cycle = per_cycle.iloc[1:].reset_index(drop=True)

    return per_cycle


def load_rwth_cell(session_files: list, cell_id: str, drop_first: bool = True,
                    apply_local_outlier_fix: bool = True) -> pd.DataFrame:
    """drop_first / apply_local_outlier_fix: both default True (the
    production path). Exposed as toggles ONLY so a K7-style with/without
    sensitivity check (verify_rwth_v2.py) can quantify each correction's
    effect independently, same convention as CALCE's INCOMPLETE_POLICY and
    XJTU's XJTU_APPLY_SENTINEL_FIX -- not meant to be changed for a real run."""
    # sort sessions chronologically using the embedded filename timestamp
    session_files = sorted(session_files, key=_extract_timestamp)

    all_rows = []
    for path in session_files:
        try:
            per_cycle = _parse_session_file(path, drop_first=drop_first)
            per_cycle["source_file"] = os.path.basename(path)
            all_rows.append(per_cycle)
        except Exception as e:
            print(f"    [RWTH] skipped session file for {cell_id}: {e} ({os.path.basename(path)})")

    if not all_rows:
        raise RuntimeError(f"No usable sessions for {cell_id}")

    combined = pd.concat(all_rows, ignore_index=True)
    combined["cycle"] = range(1, len(combined) + 1)  # global renumbering, session order preserved
    combined["cell_id"] = cell_id
    combined["dataset"] = "RWTH"
    combined = combined[["cycle", "capacity", "cell_id", "dataset"]]

    # Statistical safety net for anomalies NOT explained by the
    # session-boundary truncation mechanism above (e.g. a genuinely
    # corrupted session's AhEla trace producing a multi-row near-zero
    # stretch, as seen in RWTH_ep_sanyo_019).
    if apply_local_outlier_fix:
        combined = strip_local_outliers(combined, verbose=True)

    return combined


def load_all(drop_first: bool = True, apply_local_outlier_fix: bool = True) -> pd.DataFrame:
    if not RWTH_ROOT or not os.path.isdir(RWTH_ROOT):
        raise FileNotFoundError(f"RWTH_DIR not set or not found: {RWTH_ROOT}")

    all_files = [f for f in os.listdir(RWTH_ROOT) if f.lower().endswith(".csv")]

    cell_files = {}
    for fname in all_files:
        if "EIS" in fname:
            continue  # exclude impedance tests
        m = re.search(r"ep sanyo\s*(\d+)", fname, re.IGNORECASE)
        if not m:
            continue
        cell_id = f"RWTH_ep_sanyo_{m.group(1)}"
        cell_files.setdefault(cell_id, []).append(os.path.join(RWTH_ROOT, fname))

    print(f"  [RWTH] found {len(cell_files)} physical cells "
          f"({sum(len(v) for v in cell_files.values())} session files total)")

    frames = []
    for i, (cell_id, files) in enumerate(sorted(cell_files.items()), 1):
        print(f"  [RWTH] ({i}/{len(cell_files)}) parsing {cell_id} ({len(files)} sessions)...", flush=True)
        try:
            df = load_rwth_cell(files, cell_id, drop_first=drop_first,
                                 apply_local_outlier_fix=apply_local_outlier_fix)
            if len(df) > 5:
                frames.append(df)
        except Exception as e:
            print(f"  [RWTH] skipped {cell_id}: {e}")

    if not frames:
        raise RuntimeError("RWTH: no cells parsed — check RWTH_DIR and file structure")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_all()
    print(df.groupby("cell_id").size())
    print(df.head())