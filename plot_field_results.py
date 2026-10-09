"""
Plot the field-data exploration result: (a) the extracted capacity trend over
odometer for the one real EV tested (raw and temperature-corrected estimates,
with a rolling-median smoothed trend), and (b) a direct visual comparison of
the observed noise spread against the plausible true fade signal over the
same odometer range -- the figure this project's field-data investigation
was actually building toward.

Run this on field_capacity_trend.csv (the output of field_experiment.py
--mode single). Produces field_data_signal_vs_noise.png at 300 DPI, matching
this project's other figures.

USAGE: python plot_field_results.py --csv field_capacity_trend.csv
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_figure(csv_path: str, out_path: str = "field_data_signal_vs_noise.png"):
    df = pd.read_csv(csv_path).sort_values("odometer_km").reset_index(drop=True)

    # Rolling-median smoothing across segments (window=15), same as tested
    # earlier in this project's exploration.
    df["capacity_smoothed"] = (
        df["capacity_corrected"].rolling(window=15, center=True, min_periods=8).median()
    )

    initial_cap = df["capacity_corrected"].iloc[:5].mean()
    plausible_fade_low = initial_cap * (1 - 0.06)   # 6% fade -> lower bound
    plausible_fade_high = initial_cap * (1 - 0.02)  # 2% fade -> upper bound

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel (a): raw vs corrected vs smoothed capacity trend ---
    ax = axes[0]
    ax.scatter(df.odometer_km, df.capacity_Ah_est, s=14, alpha=0.35,
               color="#7f8c8d", label="Raw Coulomb-counted estimate")
    ax.scatter(df.odometer_km, df.capacity_corrected, s=14, alpha=0.55,
               color="#2980b9", label="Temperature-corrected")
    ax.plot(df.odometer_km, df.capacity_smoothed, color="#c0392b",
            linewidth=2, label="Rolling median (w=15)")
    ax.set_xlabel("Odometer (km)")
    ax.set_ylabel("Estimated capacity (Ah)")
    ax.set_title("(a) Field-derived capacity trend, one EV")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    # --- Panel (b): observed spread vs plausible true fade band ---
    ax = axes[1]
    ax.axhspan(plausible_fade_low, plausible_fade_high, color="#27ae60",
               alpha=0.25, label="Plausible true fade range\n(2-6% over this distance)")
    ax.axhline(initial_cap, color="#27ae60", linestyle="--", linewidth=1,
               label="Early-life capacity")
    ax.scatter(df.odometer_km, df.capacity_corrected, s=14, alpha=0.55,
               color="#2980b9", label="Temperature-corrected estimate")
    spread = df.capacity_corrected.max() - df.capacity_corrected.min()
    plausible = plausible_fade_high - plausible_fade_low
    ax.set_xlabel("Odometer (km)")
    ax.set_ylabel("Estimated capacity (Ah)")
    ax.set_title(f"(b) Observed spread ({spread:.1f} Ah) vs.\nplausible true fade ({plausible:.1f} Ah)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")

    corr = df.odometer_km.corr(df.capacity_corrected)
    print(f"\nSummary for caption/text:")
    print(f"  n segments = {len(df)}, odometer range = "
          f"{df.odometer_km.min():.0f}-{df.odometer_km.max():.0f} km")
    print(f"  Observed spread (corrected) = {spread:.1f} Ah")
    print(f"  Plausible true fade range = {plausible_fade_low:.1f}-{plausible_fade_high:.1f} Ah "
          f"({plausible:.1f} Ah band)")
    print(f"  Ratio (observed spread / plausible fade band) = {spread/plausible:.1f}x")
    print(f"  Correlation(odometer, corrected capacity) = {corr:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="field_capacity_trend.csv")
    parser.add_argument("--out", default="field_data_signal_vs_noise.png")
    args = parser.parse_args()
    make_figure(args.csv, args.out)
