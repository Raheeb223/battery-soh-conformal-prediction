# Reproducibility

This file did not exist on disk at the start of this audit (the mission brief's claim that it
existed was wrong — see `TEST_REPORT.md`, Stage 0). It is written from scratch here, describing
the pipeline as independently verified during this audit (see `TEST_REPORT.md` / `ERROR_REGISTER.md`
for what was checked and what was found).

## 1. Environment

- OS: Windows 11 Pro, PyCharm project layout.
- Python environment: `.venv/` (not committed). Frozen package list: `requirements_frozen.txt`
  (77 packages, captured via `pip freeze` during this audit). Key versions: numpy 1.26.4,
  pandas 2.0.0, scipy 1.10.0, scikit-learn 1.3.0, torch 2.5.1+cu121, xgboost 3.2.0,
  matplotlib 3.7.0, openpyxl 3.1.0.
- GPU: NVIDIA RTX 3060 (12GB), driver 572.83, CUDA available via torch 2.5.1+cu121. Training
  falls back to CPU automatically (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`)
  but will be impractically slow for the full model-training steps below.

To recreate the environment:
```
pip install -r requirements_frozen.txt
```

## 2. Raw datasets (not committed — must be downloaded separately)

All raw dataset folders/files are excluded from version control (`.gitignore`) because of their
size. Place them directly under the project root (`config.ROOT_DIR`) using these exact names, or
edit the corresponding path in `config.py`:

| Dataset | Expected path (relative to project root) | Loader | Source |
|---|---|---|---|
| NASA | `5. Battery Data Set/` (contains `B0005.mat`, `B0006.mat`, ...) | `nasa_loader.py` | [NASA PCoE Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| CALCE (CS2) | `CS2_33/`, `CS2_34/`, `CS2_35/`, `CS2_38/` | `calce_loader_v2.py` | [CALCE Battery Research Group, University of Maryland](https://calce.umd.edu/battery-data) |
| CALCE (CX2) | `CX2_16/`, `CX2_31/`, `CX2_33/`, `CX2_34/`, `CX2_35/`, `CX2_36/`, `CX2_37/` | `calce_loader_v2.py` | [CALCE Battery Research Group, University of Maryland](https://calce.umd.edu/battery-data) |
| Oxford | `Oxford_Battery_Degradation_Dataset_1.mat` | `oxford_loader.py` | [Oxford University Research Archive, DOI: 10.5287/bodleian:KO2kdmYGg](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) |
| MIT/Severson (FastCharge) | loose `*_structure.json` files directly under project root, or a `FastCharge/` subfolder (loader searches recursively) | `mit_loader.py` | [data.matr.io (MIT/Stanford/Toyota Research Institute)](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) — cite Severson et al., *Nature Energy* 4:383–391 (2019) |
| BIT | `kw34hhw7xg-3/` | `bit_loader.py` | [Mendeley Data, DOI: 10.17632/kw34hhw7xg.3](https://data.mendeley.com/datasets/kw34hhw7xg/3) (Beijing Institute of Technology, Advanced Energy Storage and Application Group) |
| XJTU | `Battery Dataset/` | `xjtu_loader.py` | [Zenodo, DOI: 10.5281/zenodo.10963339](https://doi.org/10.5281/zenodo.10963339) (Xi'an Jiaotong University) |
| RWTH | `RWTH-2021-04545_818642/Rawdata/Rohdaten/` | `rwth_loader.py` | [RWTH Aachen Publications, DOI: 10.18154/RWTH-2021-04545](https://publications.rwth-aachen.de/record/818642) |

Each dataset's own license/usage terms govern redistribution; consult the source page above before
redistributing raw files. This project does not commit any raw dataset file to version control
(see `.gitignore`) -- only the processing code.

**Known, independently-confirmed data-quality caveats** (see `ERROR_REGISTER.md` for the full
evidence trail on each):
- CALCE: 4 defects corrected in `calce_loader_v2.py` (cumulative discharge-capacity counters in
  331/366 files, a tie at the last row of short cumulative files, one non-cycling file excluded,
  truncated final cycles flagged). `INCOMPLETE_POLICY="flag"` (default) vs `"drop"` sensitivity:
  dropping removes 0.634% of rows, central conclusions unaffected. CX2_31 contributes 0 rows (only
  `.txt`-format files present; loader only reads `.xlsx`/`.xls`, matching the CS2_8/CS2_21
  exclusion already documented in `config.py`).
- XJTU: `detect_and_strip_sentinel_values()` removes periodic calibration/sentinel rows from the 8
  `Sim_satellite_*` cells but does NOT re-number the `cycle` column afterward (verified directly,
  see E9) — every one of the 376 Sim_satellite test windows (14.4% of XJTU's 2,605-window test
  set) therefore bridges a removed block. Bridging windows show 5.6x higher RMSE than clean
  windows but do not break conformal coverage validity (94.7% vs 92.7%). Report this explicitly;
  do not claim it is undetectable or negligible.
- RWTH: two corrections, independently re-verified against the current raw CSVs
  (`verify_rwth_v2.py`): session-boundary partial-cycle truncation (`drop_first=True`, removes
  734/114,716 rows = 0.640%) and a statistical local-outlier safety net (787/114,716 = 0.686%).
  Combined: 848/114,716 = 0.739% (less than the sum because the two corrections overlap on
  roughly 673 rows).

## 3. Pipeline run order

```
python preprocessing.py          # parses all 7 raw datasets, builds the cell-independent
                                  # train/val/test split (seed=config.SPLIT_SEED, default
                                  # config.RANDOM_SEED=42), windows them, caches to
                                  # cache/processed.npz
python train.py                  # trains the default "full" model once, saves
                                  # outputs/best_model.pt + outputs/best_results.json
python experiments.py            # baselines (linear/SVR/XGBoost), ablations
                                  # (no_attention/no_multiscale/plain_lstm/gru), window-size
                                  # sweep (10/20/30/50), multi-seed runs (42/123/2024),
                                  # conformal-mode comparison (groupwise/pooled/
                                  # normalized_groupwise) -- all via train_one() with different args
python baseline_conformal_coverage.py   # K3: pooled+groupwise conformal coverage for
                                         # SVR/XGBoost/plain_lstm/gru (no retrain for the last two
                                         # -- reuses existing ablation checkpoints)
python leave_one_out.py          # K1-fixed LODO: 7 held-out-dataset configs, predictor-consistent
                                  # (mc_mean for both calibration and test, matching train.py)
python hierarchical_calibration_v2.py   # canonical hierarchical (empirical-Bayes) calibration
python dataset_level_wilcoxon.py # K2-fixed: reads real coverage numbers from the JSON files
                                  # above (NOT hardcoded), reports both the valid dataset-level
                                  # (n=6) and the invalid-but-shown-for-contrast 24-pair comparison
python run_nested_hp_search_all.py      # K4: HGC-CP (lambda,beta) nested held-out search, all
                                         # 7 datasets as real target in turn
python mwhc_calibration.py              # MWHC: MMD-weighted hierarchical calibration (new
                                         # method, see main.tex sec:mwhc/sec:mwhc-results) --
                                         # same calibration-level LODO protocol as
                                         # hierarchical_calibration_v2.py, plus a lambda-
                                         # sensitivity sweep and an effective-sample-size
                                         # diagnostic; no retraining, pure post-hoc statistics
python calibration_size_sensitivity.py  # new experiment (see main.tex sec:calsize/
                                         # sec:calsize-results): groupwise coverage vs. number of
                                         # calibration CELLS actually used, 500 repeated random
                                         # cell subsamples per budget; no retraining
python predictor_consistent_recalibration.py   # per-dataset coverage with BOTH a window-level
                                                # and a cell-clustered-bootstrap CI (E13-fixed)
python xjtu_bridging_isolation.py       # E9: isolates bridging to the 8 Sim_satellite cells
python xjtu_bridging_sensitivity.py     # quantifies the bridging windows' effect on MAE/RMSE/coverage
python verify_rwth_v2.py                # re-verifies RWTH's two corrections against current raw files
python verify_calce_v2.py               # re-verifies CALCE's corrections (K8)
python repeated_split_experiment.py     # K6: 4 additional cell-level split seeds (43-46),
                                         # groupwise/pooled/normalized_groupwise + persistence +
                                         # cell-clustered bootstrap, model-init seed held fixed
python make_results_registry.py         # Stage 11: compiles every verified *.json above into
                                         # results_registry.json + paper/numbers.tex
```

Tests (do not write to `cache/`/`outputs/`, safe to run any time):
```
python tests/stage2_data_layer.py              # data-layer checks + determinism (uses a
                                                # temporary cache dir, never touches the real one)
python tests/stage3_checkpoint_reproduction.py # confirms the saved checkpoint reproduces its
                                                # own reported MAE
python tests/stage4_conformal_unit_tests.py    # conformal.py unit tests, synthetic data only
python tests/stage6_hgc_cp_properties.py       # HGC-CP property tests, synthetic data only
python tests/stage6_hierarchical_dl_verification.py  # DerSimonian-Laird hand-calculated +
                                                      # independent-reimplementation checks
```

## 4. Caches and outputs

- `cache/processed.npz`: the seed-42 (default) windowed/split arrays. `cache/processed_split{N}.npz`
  for N in {43,44,45,46}: the K6 repeated-split caches. Never overwritten by a different seed —
  `preprocessing.cache_path_for()` suffixes the filename by split seed.
- `outputs/`: every script above writes `{run_name}_results.json` (metrics) and, where
  applicable, `{run_name}_predictions.npz` (raw per-window predictions + intervals, needed for any
  downstream reanalysis without retraining) and `{run_name}_model.pt` (checkpoint).
- `backup_YYYYMMDD_HHMM/`: point-in-time copies of `cache/` and `outputs/` taken before this
  audit's fixes were applied, per the brief's rule 3. Not regenerated automatically; see git
  history / `TEST_REPORT.md` Stage 0 for when this was taken.

## 5. What is machine-verified vs. what still needs a human decision

Every number in `paper/numbers.tex` comes from `results_registry.json`, which comes from the
`outputs/*.json` files listed above — never hand-typed. If a file is missing or has not yet been
regenerated after a fix that should have changed it, `make_results_registry.py` writes
`UNVERIFIED` rather than guessing or reusing a stale number (see `ERROR_REGISTER.md` E16 for a
concrete case of this safeguard catching a real staleness bug before it reached any output).

K6 repeated-split experiment completed 2026-10-10 (`outputs/repeated_split_summary.json`); results
folded into `paper/main.tex` Section 5.9 and the Discussion. Dataset source/license citations for
the manuscript's Data Availability section were verified via live search against each dataset's own
published source (see `paper/main.tex`, Data and code availability).

2026-10-17 audit pass: every pipeline script listed in Section 3 above was confirmed to have a
corresponding file in `outputs/`, and every figure `main.tex` references was confirmed to exist in
`paper/figures/`. This surfaced one real gap and corrected it (see `ERROR_REGISTER.md` E2): the K3
cross-architecture conformal coverage result (`outputs/baseline_conformal_results.json`) had been
computed but never wired into `make_results_registry.py` or `main.tex` -- the "Baseline conformal
coverage" subsection was asserting hand-typed numbers with no macro behind them. Fixed by adding a
macro-emission block and a new table (Table~\ref{tab:baseline-conformal}). The same sweep also
found and corrected several other hand-typed/approximate numbers elsewhere in Results and
Discussion (pooled-vs-groupwise per-dataset deviations, the window-level-vs-cell-clustered CI width
ratio for XJTU/CALCE -- which was stated as "roughly 4--5x" but is actually
7.4x/6.4x -- and the XJTU bridging-window MAE shift, restated from "roughly 8%" to the exact 9.0%)
by adding the missing macros rather than leaving the imprecise prose in place.
