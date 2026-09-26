"""
RWTH Aachen battery ageing dataset loader (RWTH-2021-04545).

Raw structure: one flat folder of Basytec cycler CSV exports (German column
names), one file per test SESSION rather than one per cell. Each physical cell
is identified by the 'ep sanyo NNN' token in the filename (not 'Kreis N-NNN',
which is a test channel reused across cells over time) and has 24-41 session
files, which are concatenated chronologically.

EIS (impedance spectroscopy) files are excluded; they contain no cycling
capacity.

Column mapping:
  Zyklus  = cycle number (within the session file)
  AhEla   = cumulative discharged Ah since the session start (not per cycle);
            per-cycle capacity = max(AhEla) - min(AhEla) within a cycle's rows
  Zeit    = timestamp

Row 0 of each CSV is the header, row 1 a units row (skipped). Cycle numbers
reset in every session file, so sessions are ordered by the timestamp in the
filename and cycles are renumbered globally per cell.

Session-boundary handling: AhEla restarts at the beginning of each session's
logging window, not at the start of the cycle in progress. When a session
starts mid-cycle, the first Zyklus group covers only part of a cycle and gives
an artificially low capacity (consistently about 26-27% of the cell's
baseline). The first Zyklus group of every session is therefore dropped
(_parse_session_file(drop_first=True)). strip_local_outliers()
(rwth_local_outlier_fix.py) is then applied per cell to remove the remaining
isolated invalid readings (e.g. a multi-row near-zero stretch in
RWTH_ep_sanyo_019).
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
    consistently reads ~26-27% of true baseline across many cells). Only skips dropping if a session
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
    # test began), not a per-cycle value. Since it only ever increases within
    # a session, taking max(AhEla) per cycle would report a monotonically
    # growing running total rather than the fading per-cycle discharge
    # capacity. The actual Ah discharged DURING a given cycle is instead the
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


def load_rwth_cell(session_files: list, cell_id: str) -> pd.DataFrame:
    # sort sessions chronologically using the embedded filename timestamp
    session_files = sorted(session_files, key=_extract_timestamp)

    all_rows = []
    for path in session_files:
        try:
            per_cycle = _parse_session_file(path)
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
    combined = strip_local_outliers(combined, verbose=True)

    return combined


def load_all() -> pd.DataFrame:
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
            df = load_rwth_cell(files, cell_id)
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