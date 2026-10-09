"""
diagnose_alternating_pattern.py

Confirms and measures the period-2 alternating artifact
(diagnose_window_duplication.py found: last-5-timestep values like
[-7.2235193, -7.2235193, 0.2491684, 0.2491684, -7.2235193] -- an A-B
alternation, invisible to any consecutive-run detector regardless of
min_run, since no two ADJACENT values are ever equal).

This reloads XJTU fresh (post current min_run=2 strip) and scans the 3
implicated cells' surviving capacity sequences for runs where the value
2 rows back repeats (capacity[i] == capacity[i-2]), to measure how many
rows are involved before writing a period-aware fix.

Run:
    python diagnose_alternating_pattern.py
"""

import numpy as np
import xjtu_loader

TARGET_CELLS = [
    "XJTU_Batch-6_Sim_satellite_1",
    "XJTU_Batch-6_Sim_satellite_3",
    "XJTU_Batch-6_Sim_satellite_7",
]


def find_period_runs(values, period, min_cycles=2):
    """Finds runs where values[i] == values[i-period] for at least
    min_cycles consecutive matches (i.e. the pattern repeats with the
    given period for at least min_cycles*period rows)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    runs = []
    i = period
    while i < n:
        if values[i] == values[i - period]:
            start = i - period
            match_count = 1
            j = i
            while j < n and values[j] == values[j - period]:
                match_count += 1
                j += 1
            if match_count >= min_cycles:
                runs.append((start, j, match_count))
            i = j
        else:
            i += 1
    return runs


def main():
    print("Reloading XJTU fresh (current xjtu_loader.py, min_run=2 already applied)...")
    df = xjtu_loader.load_all()

    for cell_id in TARGET_CELLS:
        sub = df[df["cell_id"] == cell_id].sort_values("cycle")
        capacity = sub["capacity"].to_numpy()
        print(f"\n{cell_id}: {len(capacity)} rows after existing strip")

        for period in (1, 2, 3, 4):
            runs = find_period_runs(capacity, period, min_cycles=2)
            if runs:
                total_rows = sum(end - start for start, end, _ in runs)
                print(f"  period={period}: {len(runs)} run(s), {total_rows} rows total")
                for start, end, matches in runs[:5]:
                    print(f"      rows[{start}:{end}]  matches={matches}  "
                          f"sample_values={capacity[start:min(end, start+6)]}")
                if len(runs) > 5:
                    print(f"      ... and {len(runs)-5} more run(s)")


if __name__ == "__main__":
    main()
