# Corrections made during development

Corrections to code or data handling that affect reported results. The current values are in
`results_registry.json`; data-quality corrections are described in more detail in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md#data-quality-corrections) and in the loader docstrings.

## Data

- **CALCE** (`calce_loader_v2.py`). Most CALCE workbooks store discharge capacity as a running
  counter within each file; the loader detects this per file and takes differences between cycles.
  Workbooks are ordered by the date in their file name, not by name. A self-discharge test file is
  excluded, and cycles cut off at the end of a file are flagged.
- **RWTH** (`rwth_loader.py`, `rwth_local_outlier_fix.py`). The capacity counter resets at the start
  of each session file, so the first cycle of each session under-reports capacity and is dropped.
  A local-outlier filter removes the remaining isolated artifacts.
- **XJTU** (`xjtu_loader.py`). Repeated sentinel readings in the eight `Sim_satellite_*` cells are
  removed. The windows that span the removed rows are kept, and their effect on error and coverage
  is reported (`xjtu_bridging_isolation.py`, `xjtu_bridging_sensitivity.py`).
- **Cell split** (`preprocessing.py`). Validation and test cell counts are computed explicitly, so
  every dataset with at least three cells gets at least one validation and one test cell.

## Calibration and evaluation

- **Leave-one-dataset-out** (`leave_one_out.py`). Calibration and test residuals now come from the
  same predictor (the MC-Dropout mean); earlier, calibration used a single deterministic pass.
  `make_results_registry.py` accepts only result files written by the current code.
- **Hierarchical calibration** (`hierarchical_calibration_v2.py`). The shrinkage weight is the
  empirical-Bayes weight tau2 / (tau2 + sigma2_g), with sigma2_g the cell-bootstrap variance of the
  dataset's quantile, and all quantities come from the same validation residuals in SOH units.
- **HGC-CP** (`hgc_cp.py`). The weighted quantile includes the test point's own weight, as in
  weighted conformal prediction; an infinite quantile gives an unbounded interval; the MMD bandwidth
  is estimated on a subsample drawn from a seeded generator, so distances are reproducible.
- **Baselines** (`baselines.py`). Validation predictions are saved, so the baselines can be
  calibrated with the same conformal code as the main model.
- **Coverage intervals** (`predictor_consistent_recalibration.py`). Per-dataset coverage intervals
  resample whole cells; window-level intervals, which treat correlated windows as independent, are
  kept only for comparison.
- **Results registry** (`make_results_registry.py`). Two result names that differ only in digits
  would map to the same LaTeX macro, so a repeated macro name raises an error.
