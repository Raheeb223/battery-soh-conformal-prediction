# Verification checks

These scripts check that the statistical methods are implemented correctly and that the data and
saved model are what the paper says they are. They are standalone Python scripts (not a pytest
suite): each prints one `[PASS]`/`[FAIL]` line per check, ends with `N passed, M failed`, and exits
with status 1 if anything fails. Run them from the repository root.

```
python tests/run_checks.py          # fast checks, plus the checkpoint check if outputs/ exists
python tests/run_checks.py --full   # also the data-layer check (needs the raw datasets)
```

| Script | What it verifies | Needs | Time |
|---|---|---|---|
| `check_conformal.py` | the split-conformal calibration behind every interval | nothing (synthetic data) | < 1 s |
| `check_hierarchical_calibration.py` | the DerSimonian–Laird estimator behind hierarchical calibration | nothing | < 1 s |
| `check_hgc_cp.py` | the similarity weighting in HGC-CP | nothing | < 1 s |
| `check_hgc_cp_split_equivalence.py` | the fast HGC-CP path gives identical intervals | nothing | < 1 s |
| `check_evbattery_loader.py` | the field-data loader refuses unsafe pickles and reads valid ones | nothing | < 1 s |
| `check_checkpoint_reproduction.py` | the saved model reproduces the reported accuracy | `cache/`, `outputs/` | ~10 s |
| `check_data_layer.py` | cells, splits, windows, and pipeline determinism | raw datasets | tens of minutes |

## What each check tests, and why it matters

### `check_conformal.py`: split-conformal calibration (`conformal.py`)

Every prediction interval in the paper comes from this module, so it is tested against known
answers on synthetic data.

1. **Coverage guarantee.** Over 300 independent calibration/test draws from N(0, 1), the average
   coverage of 90% intervals must reach the 90% target, as the split-conformal theorem requires.
2. **Calibration quantile.** An independently written order-statistic implementation is compared
   with the module's `np.quantile`-based one over 20 random configurations. The module's quantile
   is never below the textbook value, so it is never less conservative.
3. **Groupwise = separate split conformal per dataset.** Groupwise (Mondrian) calibration must
   give exactly the quantile obtained by calibrating each group on its own.
4. **Normalized calibration reduces correctly.** With a constant per-sample uncertainty,
   normalized calibration must give the same intervals as plain split conformal.
5. **Missing-group fallback.** If a dataset has no calibration data, the code must fall back to a
   pooled quantile and produce no NaNs. The test triggers this path directly.

### `check_hierarchical_calibration.py`: DerSimonian–Laird estimator (`hierarchical_calibration_v2.py`)

Hierarchical calibration shrinks each dataset's quantile toward the others by an amount set by the
between-dataset variance τ². The estimator is checked against arithmetic done by hand:

1. **Hand-calculated cases.** Homogeneous inputs give τ² = 0; a heterogeneous two-group case gives
   the exact hand value (τ² = 11.5, μ = 2.5); a case where the heterogeneity statistic falls below
   its degrees of freedom is clipped to τ² = 0 instead of going negative.
2. **Independent reimplementation.** A from-scratch implementation in the test file matches the
   module exactly over 30 random configurations.
3. **Direction of pooling.** With clearly heterogeneous inputs (hand value τ² = 47.5), the
   random-effects mean must lie strictly between the fixed-effect and the unweighted mean.

### `check_hgc_cp.py`: heterogeneity-guided conformal prediction (`hgc_cp.py`)

HGC-CP weights the other datasets by how similar they are (maximum mean discrepancy).

1. The weights always sum to one (10 random configurations).
2. Raising the temperature λ never moves weight away from the most similar dataset.
3. With a single, homogeneous group there is nothing to adapt to, so HGC-CP must give the ordinary
   split-conformal quantile (observed relative difference 0.05%).
4. A finite weighted quantile always lies within the range of the calibration scores (20 trials).

`hgc_cp.py` also has its own self-test (`python hgc_cp.py`) covering MMD distance ranking,
reproducible bandwidth selection, and infinite width when calibration data are insufficient.

### `check_hgc_cp_split_equivalence.py`: fast HGC-CP path

The quantile is computed once per calibration set and reused for each test point, instead of being
recomputed per point. The two paths must give bit-identical intervals over 250 (test point, random
configuration) combinations, including a configuration that triggers the infinite-width case.

### `check_evbattery_loader.py`: safe loading of the field data

The EVBattery snippets are Python pickle files, and unpickling can run arbitrary code.
`evbattery_loader.py` therefore uses a restricted unpickler that allows only the numpy and
collections types the files contain. The test checks that it (1) refuses a pickle that would call
`os.system`, (2) refuses one that would call `eval`, and (3) still reads a synthetic snippet built
exactly like the real files.

### `check_checkpoint_reproduction.py`: saved model

1. The saved checkpoint loads into the model defined by the current `model.py` with no missing or
   extra weights, so the code describes exactly the model that was trained.
2. The model's MC-Dropout-mean test MAE reproduces the recorded value to within 10%. MC-Dropout is
   random, so exact equality is not expected; the observed difference is a few hundredths of a
   percent.

### `check_data_layer.py`: data, splits, and determinism

Thirteen checks re-derived from the raw files: no missing or duplicated rows; every cell's cycles
sortable; cell counts reconcile (cells loaded = cells assigned to a split, and how many contribute
at least one window); cells counted per dataset; train/validation/test sets disjoint at the cell
level; every dataset with at least three cells has validation and test cells; normalization
statistics reproduce independently; no window has a non-ascending cycle sequence; the CALCE
corrections give the expected row count, filter pass rate, and step size; and **determinism**: the
whole preprocessing pipeline is run twice into a temporary cache and the two outputs must have the
same SHA-256 hash. It never touches the real `cache/`, and writes `data_layer_report.json`, from
which the registry takes the cell and window counts quoted in the paper.

## Other checks in the pipeline

Some verification lives next to the code it checks rather than in this folder:

- `verify_calce_v2.py` and `verify_rwth_v2.py` re-check the CALCE and RWTH data corrections against
  the raw files (their output is in `logs/`).
- `mwhc_calibration.py` asserts, before producing any result, that its similarity-weighted fit
  with equal weights reproduces plain hierarchical calibration exactly.
- `make_results_registry.py` refuses to write two numbers to the same LaTeX macro name, and writes
  `UNVERIFIED` rather than a number when an input file is missing.
