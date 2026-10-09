"""
CALCE series diagnostic.

PURPOSE. The cached CALCE SOH series is bimodal and alternating: in all 10
cells, 55-65% of rows sit near SOH ~0.5 and the rest near 1.0, with strongly
negative correlation between successive step changes. This script identifies
WHICH mechanism in the loading/SOH pipeline is responsible, using the same
file discovery, column names, per-(file, cycle) max-capacity aggregation, and
SOH rule (reference = median of the first three rows; keep 0 < SOH < 1.10) as
calce_loader.py and preprocessing.py, but exposing the intermediate
quantities they discard.

MECHANISMS TESTED (each reported separately, none assumed)
  M1  File ordering. calce_loader sorts sessions by file NAME and assumes that
      is chronological. Names such as CS2_35_10_4_10.xlsx / CS2_35_8_17_10.xlsx
      (month_day_year, not zero-padded) do not sort chronologically. Reported:
      whether name order == date-parsed order, per cell.
  M2  Partial cycles at file boundaries. A test session can start or end
      mid-cycle. Reported: share of low-capacity groups that are the first or
      last group of a file, and whether the same Cycle_Index continues across
      consecutive files.
  M3  Incomplete discharges. A complete discharge ends near the lower cutoff
      voltage. Reported (if a voltage column exists): terminal voltage at the
      max-capacity row of each group, and the share of low-capacity groups whose
      terminal voltage is well above the cutoff.
  M5  Cumulative discharge capacity. If Discharge_Capacity(Ah) is a running
      counter within a file (not reset each cycle), the per-cycle MAX returns
      Q, 2Q, 3Q, ... . Reported: share of consecutive within-file groups whose
      capacity increases, the median ratio second/first group (about 2.0 for a
      cumulative counter), and the implied per-cycle capacity from differencing.
      The cached CALCE series shows this signature indirectly (adjacent surviving
      rows have SOH ratio median 1.995, and only ~6% of raw rows survive the
      SOH filter); this test checks it against the raw files.
  M4  Reference capacity and SOH filter. SOH = capacity / median(first 3
      groups); rows with SOH <= 0 or >= 1.10 are dropped. If the first three
      groups include partial cycles, the reference is wrong and the filter
      then removes most of the remaining rows. Reported: reference vs. the cell's
      median capacity, and the fraction of groups the filter removes -- under
      NAME order (as run) and DATE order.

OUTPUT. A printed per-cell summary and outputs/calce_group_diagnostics.csv with
one row per (cell, file, cycle_raw). Send the summary (and, ideally, the CSV
head for one cell) back before adopting any loader change.

Run from the project root:
    python diagnose_calce_series.py
"""

import os
import re

import numpy as np
import pandas as pd

import config

CYC = ["Cycle_Index", "Cycle_Index(#)", "Cycle Index"]
CAP = ["Discharge_Capacity(Ah)", "Discharge_Capacity", "Discharge Capacity (Ah)"]
VOLT = ["Voltage(V)", "Voltage", "Voltage (V)"]
LOW_FRACTION = 0.75          # a group is "low" if capacity < 0.75 x the cell's median group capacity
CUTOFF_V_GUESS = 2.7         # CALCE discharge cutoff (verify against the cell's protocol)


def _find(cols, cands):
    return next((c for c in cands if c in cols), None)


def parse_date(fname):
    """CALCE-style ..._M_D_YY.xlsx -> (yy, m, d); None if the pattern is absent."""
    m = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{2})\.xls", fname.lower())
    return None if not m else (int(m.group(3)), int(m.group(1)), int(m.group(2)))


def list_files(cell_dir):
    paths = []
    for dp, _, fns in os.walk(cell_dir):
        paths += [os.path.join(dp, f) for f in fns if f.lower().endswith((".xlsx", ".xls"))]
    return paths


def read_groups(path):
    """Per-(cycle_raw) aggregates for one workbook."""
    xls = pd.ExcelFile(path)
    rows = []
    for sheet in xls.sheet_names:
        if "channel" not in sheet.lower():
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        c, q, v = _find(df.columns, CYC), _find(df.columns, CAP), _find(df.columns, VOLT)
        if c is None or q is None:
            continue
        keep = [c, q] + ([v] if v else [])
        sub = df[keep].dropna(subset=[c, q]).reset_index(drop=True)
        for cyc, g in sub.groupby(c, sort=False):
            i = g[q].idxmax()
            rows.append({"cycle_raw": cyc, "capacity": float(g[q].max()), "n_rows": len(g),
                         "v_at_max": float(sub.loc[i, v]) if v else np.nan,
                         "v_min": float(g[v].min()) if v else np.nan})
    return rows


def soh_survivors(caps):
    """Replicates preprocessing.add_soh: ref=median(first 3), keep 0<soh<1.10."""
    caps = np.asarray(caps, float)
    if len(caps) == 0:
        return np.nan, np.nan, np.nan
    ref = np.median(caps[:min(3, len(caps))])
    soh = caps / ref if ref != 0 else np.full_like(caps, np.nan)
    keep = (soh > 0) & (soh < 1.10)
    return ref, float(keep.mean()), soh


def diagnose_cell(cell_dir, cell_id):
    paths = list_files(cell_dir)
    if not paths:
        print(f"  {cell_id}: no excel files found")
        return None
    name_order = sorted(paths, key=os.path.basename)
    dates = {p: parse_date(os.path.basename(p)) for p in paths}
    have_dates = all(dates[p] is not None for p in paths)
    date_order = sorted(paths, key=lambda p: (dates[p] or (99, 99, 99), os.path.basename(p))) if have_dates else None

    per_file = {}
    for p in paths:
        try:
            per_file[p] = read_groups(p)
        except Exception as e:
            print(f"  [{cell_id}] cannot read {os.path.basename(p)}: {e}")
            per_file[p] = []

    def build(order):
        recs = []
        for fr, p in enumerate(order):
            gs = per_file[p]
            for gi, g in enumerate(gs):
                recs.append({"cell": cell_id, "file": os.path.basename(p), "file_rank": fr,
                             "group_in_file": gi, "n_groups_in_file": len(gs), **g})
        return pd.DataFrame(recs)

    dn = build(name_order)
    if dn.empty:
        print(f"  {cell_id}: parsed nothing")
        return None
    dn["global_idx_name_order"] = np.arange(1, len(dn) + 1)
    med = dn["capacity"].median()
    dn["low"] = dn["capacity"] < LOW_FRACTION * med
    dn["first_or_last_in_file"] = (dn["group_in_file"] == 0) | (dn["group_in_file"] == dn["n_groups_in_file"] - 1)

    # M2: does a Cycle_Index continue across consecutive files (session split mid-cycle)?
    def _continuations(order):
        return sum(1 for a, b in zip(order[:-1], order[1:])
                   if per_file[a] and per_file[b] and per_file[a][-1]["cycle_raw"] == per_file[b][0]["cycle_raw"])
    continues = _continuations(name_order)
    continues_date = _continuations(date_order) if date_order is not None else None

    print(f"\n=== {cell_id}: {len(paths)} files, {len(dn)} (file, cycle) groups, median group capacity {med:.3f} Ah ===")
    print(f"M1 ordering: name order == date order? "
          f"{'unknown (dates not parseable)' if not have_dates else ('YES' if name_order == date_order else 'NO  <-- sessions concatenated out of chronological order')}")
    n_low = int(dn["low"].sum())
    print(f"   low-capacity groups (< {LOW_FRACTION:.2f} x median): {n_low} of {len(dn)} ({100*n_low/len(dn):.0f}%)")
    if n_low:
        print(f"M2 boundaries: {100*dn.loc[dn['low'], 'first_or_last_in_file'].mean():.0f}% of low groups are the first/last group of a file "
              f"(all groups: {100*dn['first_or_last_in_file'].mean():.0f}%);  same Cycle_Index continues into the next file in {continues} of {max(len(paths)-1,0)} file pairs (name order)"
              + ("" if continues_date is None else f", {continues_date} of {max(len(paths)-1,0)} (date order)"))
    if dn["v_at_max"].notna().any():
        cut = CUTOFF_V_GUESS
        hi = dn["v_at_max"] > cut + 0.15
        print(f"M3 terminal voltage at max-capacity row: median {dn['v_at_max'].median():.2f} V; "
              f"share of low groups ending > {cut+0.15:.2f} V (incomplete discharge): {100*hi[dn['low']].mean() if n_low else float('nan'):.0f}%  "
              f"(complete groups: {100*(~hi[~dn['low']]).mean():.0f}% end near cutoff)")
    else:
        print("M3 terminal voltage: no voltage column found -- add the correct name to VOLT to test this mechanism")

    # M5: cumulative-counter signature, evaluated within each file (name order irrelevant here)
    inc, ratios, first3 = [], [], None
    for p in name_order:
        c = [g["capacity"] for g in per_file[p]]
        if len(c) >= 2:
            inc += [c[k + 1] > c[k] for k in range(len(c) - 1)]
            ratios.append(c[1] / c[0] if c[0] > 0 else np.nan)
        if first3 is None and len(c) >= 3:
            first3 = [round(x, 3) for x in c[:3]]
    if inc:
        r_med = float(np.nanmedian(ratios))
        diffs = []
        for p in name_order:
            c = [g["capacity"] for g in per_file[p]]
            diffs += [c[0]] + [c[k] - c[k - 1] if c[k] >= c[k - 1] else c[k] for k in range(1, len(c))]
        print(f"M5 cumulative counter: {100*np.mean(inc):.0f}% of consecutive within-file groups increase; median ratio group2/group1 = {r_med:.2f} "
              f"(~2.0 => cumulative);  first 3 group capacities of a file: {first3};  median per-cycle capacity if differenced: {np.median(diffs):.3f} Ah")

    ref, frac_name, _ = soh_survivors(dn["capacity"].to_numpy())
    print(f"M4 reference (median of first 3 groups, NAME order as run): {ref:.3f} Ah = {100*ref/med:.0f}% of cell median; "
          f"SOH filter keeps {100*frac_name:.0f}% of groups")
    if date_order is not None and date_order != name_order:
        dd = build(date_order)
        ref2, frac_date, _ = soh_survivors(dd["capacity"].to_numpy())
        print(f"   same, DATE order: reference {ref2:.3f} Ah = {100*ref2/med:.0f}% of cell median; SOH filter keeps {100*frac_date:.0f}% of groups")

    cap = dn["capacity"].to_numpy()
    if len(cap) > 6:
        d = np.diff(cap)
        print(f"   alternation: corr(successive step changes) = {np.corrcoef(d[:-1], d[1:])[0, 1]:.2f}")
    return dn


def main():
    frames = []
    for chem, dirs in config.CALCE_DIRS.items():
        for d in dirs:
            cid = os.path.basename(d.rstrip("/\\"))
            if not os.path.isdir(d):
                print(f"[skip] {cid}: folder not found ({d})")
                continue
            df = diagnose_cell(d, cid)
            if df is not None:
                frames.append(df)
    if frames:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        out = os.path.join(config.OUTPUT_DIR, "calce_group_diagnostics.csv")
        pd.concat(frames).to_csv(out, index=False)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
