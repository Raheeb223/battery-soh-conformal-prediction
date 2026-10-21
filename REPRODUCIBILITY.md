# Reproducing the results

This guide covers the software environment, where the raw data goes, the order in which the
scripts produce every result, and how those results become the numbers and figures in the paper.

## 1. Environment

- Python 3.10, packages pinned in `requirements.txt` (`pip install -r requirements.txt`).
- The results in the paper were produced on Windows 11 with an NVIDIA RTX 3060 (12 GB) and
  PyTorch 2.5.1 built for CUDA 12.1. Training uses the GPU when one is available and falls back to
  the CPU otherwise; the training steps below are impractically slow on a CPU.

## 2. Raw data

The raw datasets are not redistributed; download them from their original sources (each has its own
license). Extract them under the repository folder using the names below, or put them elsewhere and
set the environment variable `SOH_DATA_DIR` to that folder (`config.py` reads it).

| Dataset | Expected location | Loader | Source |
|---|---|---|---|
| NASA | `5. Battery Data Set/` (`B0005.mat`, `B0006.mat`, ...) | `nasa_loader.py` | [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |
| CALCE (CS2) | `CS2_33/`, `CS2_34/`, `CS2_35/`, `CS2_38/` | `calce_loader_v2.py` | [CALCE Battery Research Group](https://calce.umd.edu/battery-data) |
| CALCE (CX2) | `CX2_16/`, `CX2_31/`, `CX2_33/` to `CX2_37/` | `calce_loader_v2.py` | [CALCE Battery Research Group](https://calce.umd.edu/battery-data) |
| Oxford | `Oxford_Battery_Degradation_Dataset_1.mat` | `oxford_loader.py` | [Oxford University Research Archive, DOI 10.5287/bodleian:KO2kdmYGg](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) |
| MIT / Stanford / TRI | `*_structure.json` files in the repository folder or a `FastCharge/` subfolder | `mit_loader.py` | [data.matr.io](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) (Severson et al., *Nature Energy* 2019) |
| BIT | `kw34hhw7xg-3/` | `bit_loader.py` | [Mendeley Data, DOI 10.17632/kw34hhw7xg.3](https://data.mendeley.com/datasets/kw34hhw7xg/3) |
| XJTU | `Battery Dataset/` | `xjtu_loader.py` | [Zenodo, DOI 10.5281/zenodo.10963339](https://doi.org/10.5281/zenodo.10963339) |
| RWTH | `RWTH-2021-04545_818642/Rawdata/Rohdaten/` | `rwth_loader.py` | [RWTH Publications, DOI 10.18154/RWTH-2021-04545](https://publications.rwth-aachen.de/record/818642) |

CS2_8, CS2_21 and CX2_31 contribute no rows: CALCE recorded them on a different instrument
(`.txt` files), and the loader reads the Arbin `.xlsx`/`.xls` format only.

### Data-quality corrections

Three datasets need corrections to raw-file artifacts that would otherwise look like degradation.
Each was re-verified against the raw files; the paper's Appendix A gives the measured effect.

- **CALCE** (`calce_loader_v2.py`, checked by `verify_calce_v2.py`): 348 of the 366 workbooks for the
  cells used store discharge capacity as a cumulative counter and are differenced; a final-cycle
  truncation flag is applied after differencing; one self-discharge test file is excluded.
  Dropping, rather than flagging, incompletely captured cycles removes 0.63% of rows.
- **XJTU** (`xjtu_loader.py`, quantified by `xjtu_bridging_isolation.py` and
  `xjtu_bridging_sensitivity.py`): periodic sentinel readings are removed from the eight
  `Sim_satellite_*` cells. Cycle numbers are not renumbered afterwards, so test windows from these
  cells span the removed blocks; the paper reports their effect on error and coverage.
- **RWTH** (`rwth_loader.py`, `rwth_local_outlier_fix.py`, checked by `verify_rwth_v2.py`): the first
  cycle of each session file under-reports capacity and is dropped, and a local-outlier filter
  removes isolated artifacts.

## 3. Run order

`run_all.py` runs the same steps in this order. It prints the plan and does nothing by default;
`--execute` runs any step whose main output is missing, `--force` reruns steps whose output exists,
and `--only <step names>` restricts it to particular steps.

```
python preprocessing.py                    # SOH, cell-level split, 20-cycle windows -> cache/processed.npz
python train.py                            # main model (multi-scale BiLSTM + attention)
python generate_val_predictions.py         # its MC-Dropout validation predictions, used by every
                                           # analysis below that recalibrates without retraining
python experiments.py                      # baselines (linear, SVR, XGBoost), architecture ablations,
                                           # window-size sweep, model seeds, pooled / groupwise /
                                           # normalized-groupwise calibration
python baseline_conformal_coverage.py      # pooled vs. groupwise coverage for every point predictor
python persistence_baselines.py            # model vs. one-step persistence, real SOH units
python leave_one_out.py                    # leave-one-dataset-out: each dataset withheld from training
python hierarchical_calibration_v2.py      # empirical-Bayes hierarchical calibration
python dataset_level_wilcoxon.py           # pooled vs. groupwise significance test at dataset level
python run_nested_hp_search_all.py         # HGC-CP hyperparameters by nested held-out search
python mwhc_calibration.py                 # MWHC and its temperature sweep (no retraining)
python calibration_size_sensitivity.py     # coverage vs. number of calibration cells (no retraining)
python train_cqr.py                        # conformalized quantile regression (retrains the backbone)
python predictor_consistent_recalibration.py   # window-level and cell-clustered coverage intervals
python xjtu_bridging_isolation.py          # locates XJTU windows that span removed sentinel blocks
python xjtu_bridging_sensitivity.py        # their effect on error and coverage
python verify_rwth_v2.py                   # re-checks the RWTH corrections against the raw files
python verify_calce_v2.py > logs/verify_calce_v2.log   # re-checks the CALCE corrections
python repeated_split_experiment.py        # four further cell-level splits, model seed fixed
python split_replication.py                # strategy comparison (hierarchical, HGC-CP, ...) re-run
                                           # on those splits with every design frozen (no retraining)
python hgc_cp_validation_only.py           # HGC-CP with its design chosen on validation data only,
                                           # all five splits (no retraining)
python deterministic_predictor_comparison.py   # single pass vs. MC-Dropout mean as interval centre
python deployability_benchmark.py          # model size, latency, calibration overhead
python make_results_registry.py            # every result -> results_registry.json (+ the paper's
                                           # numbers.tex)
```

Timings from `deployability_benchmark.py` depend on the hardware; rerun it on your own machine.

## 4. From results to the paper

- **Results.** Every script above writes its results to `outputs/` (`<run>_results.json`, and for
  model runs `<run>_predictions.npz` and `<run>_model.pt`). The JSON files and the two prediction
  arrays that the registry reads are included in this repository, so steps 4–5 below run without
  retraining; checkpoints and caches are not (they are large and regenerable).
- **Registry.** `make_results_registry.py` reads those files and writes `results_registry.json`:
  the raw results by section, plus a `values` table with every number exactly as printed in the
  paper. It writes `UNVERIFIED` instead of a number whenever an input is missing, and refuses to
  give two different numbers the same name. Window and cell counts need `cache/` and the raw CALCE
  files; when those are present the registry saves the counts to `outputs/data_counts.json`, and
  without them it reads that file, so a clone without the data rebuilds every value.
- **Figures.** `python figures/make_results_figures.py` (and the other `figures/make_*.py`) redraw the
  paper's figures from `outputs/` and `results_registry.json` into `figures/out/`.
- **Checks.** `python tests/run_checks.py` runs the verification checks described in
  `tests/README.md`.

## 5. Determinism and seeds

- The cell-level split seed (`config.SPLIT_SEED`) and the model-initialization seed
  (`config.RANDOM_SEED`) are separate, so split-to-split and seed-to-seed variability can be
  measured independently. Caches for other split seeds are written to `cache/processed_split<N>.npz`
  and never overwrite the default.
- `tests/check_data_layer.py` runs the full preprocessing twice and confirms the two outputs are
  byte-identical.
- MC-Dropout inference is stochastic, so reruns reproduce MC-mean metrics closely but not
  bit-exactly (`tests/check_checkpoint_reproduction.py` shows the size of the difference).
