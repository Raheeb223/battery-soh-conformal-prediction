"""
Verify the corrected CALCE series BEFORE re-running preprocessing and training.

Loads every configured CALCE cell with calce_loader_v2 and reports, per cell:
  * rows loaded, and rows surviving the same SOH rule as preprocessing.py
    (reference = median of first three rows; keep 0 < SOH < 1.10)
  * alternation: correlation of successive step changes and median |step| in
    SOH units (the broken series had -0.78 and ~0.42; a healthy one has small steps)
  * capacity outliers: rows whose capacity is > 1.3x or < 0.7x the median of the
    surrounding 21 rows (catches short cumulative files that the automatic
    detection, which needs >= 4 groups, would have left undifferenced)
  * flag rates: share of rows with complete_discharge == False, straddles_boundary,
    and truncated_final_cycle
  * file ordering: file names without a parseable date (these fall back to name order)
  * what policy="drop" would remove: outlier count and row count if the loader's
    flagged rows (straddles_boundary, complete_discharge==False, truncated_final_cycle)
    were actually dropped, not just flagged -- the default "flag" policy keeps every
    row and every capacity value unchanged, so this comparison is the only way to see
    the flags' real effect without changing the project-wide default

Run from the project root:  python verify_calce_v2.py
Output: outputs/calce_v2_verification.csv (per-row outliers) and a printed summary.
"""
import os
import re

import numpy as np
import pandas as pd

import calce_loader_v2 as ld
import config


def _outlier_count(df):
    cap = df["capacity"].to_numpy()
    med = pd.Series(cap).rolling(21, center=True, min_periods=5).median().to_numpy()
    return int(((cap > 1.3 * med) | (cap < 0.7 * med)).sum())


def main():
    outlier_rows, summary = [], []
    for chem, dirs in config.CALCE_DIRS.items():
        for d in dirs:
            cid = os.path.basename(d.rstrip("/\\"))
            if not os.path.isdir(d):
                print(f"[{cid}] folder not found"); continue
            try:
                df = ld.load_calce_cell(d, cid, chem)                       # policy="flag" (default): nothing removed
                df_drop = ld.load_calce_cell(d, cid, chem, policy="drop")   # same flags, but actually applied
            except Exception as e:
                print(f"[{cid}] NOT LOADED: {e}"); continue
            cap = df["capacity"].to_numpy()
            ref = np.median(cap[:3]); soh = cap / ref
            keep = float(np.mean((soh > 0) & (soh < 1.10)))
            step = np.diff(soh)
            alt = float(np.corrcoef(step[:-1], step[1:])[0, 1]) if len(step) > 5 else np.nan
            med = pd.Series(cap).rolling(21, center=True, min_periods=5).median().to_numpy()
            out = (cap > 1.3 * med) | (cap < 0.7 * med)
            for i in np.where(out)[0]:
                outlier_rows.append({"cell": cid, "row": int(i), "cycle": int(df.loc[i, "cycle"]),
                                     "file": df.loc[i, "source_file"], "capacity": float(cap[i]), "local_median": float(med[i])})
            names = {f for f in df["source_file"].unique()}
            undated = sorted(f for f in names if ld._parse_date(f) is None)
            summary.append({"cell": cid, "rows": len(df), "SOH-filter keeps": f"{100*keep:.1f}%",
                            "median |step|": f"{np.median(np.abs(step)):.4f}", "step corr": f"{alt:.2f}",
                            "outliers (flag)": int(out.sum()),
                            "outliers (drop)": _outlier_count(df_drop),
                            "rows (drop)": len(df_drop),
                            "incomplete": f"{100*np.mean(df['complete_discharge'] == False):.0f}%",
                            "straddle": f"{100*df['straddles_boundary'].mean():.0f}%",
                            "truncated_final": f"{100*df['truncated_final_cycle'].mean():.1f}%",
                            "undated files": len(undated)})
            if undated:
                print(f"[{cid}] {len(undated)} file(s) without a parseable date (ordered by NAME); first few: {undated[:6]}")
    print("\n" + pd.DataFrame(summary).to_string(index=False))
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(outlier_rows).to_csv(os.path.join(config.OUTPUT_DIR, "calce_v2_verification.csv"), index=False)
    print(f"\ntotal rows {sum(r['rows'] for r in summary)}; outliers saved to outputs/calce_v2_verification.csv")


if __name__ == "__main__":
    main()