# How Far Does Calibrated Uncertainty Transfer Across Battery Datasets?

Code and results for the paper *How Far Does Calibrated Uncertainty Transfer Across Battery
Datasets? An Empirical Study of Conformal Calibration for State-of-Health Prediction on Seven Public
Datasets* (R. Sadman, submitted to the *Journal of Energy Storage*).

The study trains one state-of-health (SOH) predictor across seven public lithium-ion battery
datasets (339 cells from seven institutions) and compares seven ways of calibrating its prediction
intervals, reporting coverage per dataset rather than only in aggregate:

| Strategy | Calibrates each dataset on | Code |
|---|---|---|
| Pooled | all datasets together | `conformal.py` |
| Groupwise (Mondrian) | its own validation cells | `conformal.py` |
| Normalized groupwise | its own cells, scaled by MC-Dropout uncertainty | `conformal.py` |
| Hierarchical (empirical Bayes) | its own cells, shrunk toward the other datasets | `hierarchical_calibration_v2.py` |
| HGC-CP | the other datasets, weighted by similarity (MMD) | `hgc_cp.py` |
| MWHC | hierarchical, with similarity-weighted shrinkage | `mwhc_calibration.py` |
| CQR | a model trained to output the interval directly | `train_cqr.py` |

## For reviewers: what you can check without the raw data

The repository includes every per-experiment result file (`outputs/*.json`), so the following run
in a few minutes with only `pip install -r requirements.txt` (Python 3.10):

```bash
python tests/run_checks.py                  # verification checks (see tests/README.md)
python make_results_registry.py             # rebuild results_registry.json from outputs/
python figures/make_results_figures.py      # redraw the results figures into figures/out/
python figures/make_block_diagram.py        # the pipeline diagram (Figure 2)
python figures/make_problem_setting.py      # Figure 1
```

`results_registry.json` contains, under `values`, every number printed in the paper, formatted as
printed; the manuscript's numbers are generated from it and never typed by hand. Reproducing the
results themselves (training, calibration, leave-one-dataset-out) needs the raw datasets and a GPU;
see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## What the tests check

`tests/` holds standalone verification scripts; [tests/README.md](tests/README.md) explains each
check and why it matters. In short:

| Script | Verifies |
|---|---|
| `check_conformal.py` | split-conformal coverage on synthetic data; the calibration quantile against an independent implementation; groupwise and normalized calibration reduce correctly; the missing-data fallback |
| `check_hierarchical_calibration.py` | the DerSimonian–Laird estimator against hand calculations and an independent implementation |
| `check_hgc_cp.py` | HGC-CP's similarity weights and its reduction to ordinary split conformal |
| `check_hgc_cp_split_equivalence.py` | the fast HGC-CP code path gives bit-identical intervals |
| `check_checkpoint_reproduction.py` | the saved model reproduces the reported accuracy |
| `check_data_layer.py` | cell counts, cell-level split, windows, and that preprocessing is deterministic (needs the raw data) |

## Repository map

Every script is part of the pipeline in [REPRODUCIBILITY.md](REPRODUCIBILITY.md), in the order
listed there.

**Data loading and preprocessing**

| File | Purpose |
|---|---|
| `config.py` | data paths, split fractions, model and training hyperparameters, 90% target |
| `nasa_loader.py`, `calce_loader_v2.py`, `oxford_loader.py`, `mit_loader.py`, `bit_loader.py`, `xjtu_loader.py`, `rwth_loader.py` | one loader per dataset: raw files to (cell, cycle, capacity) rows, including the data-quality corrections |
| `rwth_local_outlier_fix.py` | local-outlier filter used by the RWTH loader |
| `preprocessing.py` | SOH, cell-level train/validation/test split, 20-cycle windows, cached arrays |
| `verify_calce_v2.py`, `verify_rwth_v2.py` | re-check the CALCE and RWTH corrections against the raw files |
| `xjtu_bridging_isolation.py`, `xjtu_bridging_sensitivity.py` | locate XJTU windows that span removed sentinel readings and measure their effect |

**Model and training**

| File | Purpose |
|---|---|
| `model.py` | multi-scale bidirectional LSTM with self-attention, ablation variants, GRU baseline, MC-Dropout |
| `train.py` | training loop, MC-Dropout prediction, conformal calibration and evaluation |
| `experiments.py` | baselines, architecture ablations, window sizes, model seeds, calibration modes |
| `baselines.py` | linear regression, SVR, XGBoost on the same windows and splits |
| `train_cqr.py` | conformalized quantile regression |

**Calibration and analysis**

| File | Purpose |
|---|---|
| `conformal.py` | pooled, groupwise, normalized-groupwise, and CQR split conformal |
| `hierarchical_calibration_v2.py` | empirical-Bayes hierarchical calibration |
| `hgc_cp.py`, `nested_hyperparameter_search.py`, `run_nested_hp_search_all.py`, `hgc_cp_evaluation_harness.py` | HGC-CP and its held-out hyperparameter search |
| `mwhc_calibration.py` | MWHC and its temperature sweep |
| `calibration_size_sensitivity.py` | coverage vs. number of calibration cells |
| `leave_one_out.py` | each dataset withheld from training |
| `repeated_split_experiment.py` | further cell-level splits with the model seed fixed |
| `cell_clustered_bootstrap.py`, `predictor_consistent_recalibration.py` | window-level and cell-clustered coverage intervals |
| `baseline_conformal_coverage.py` | pooled vs. groupwise coverage for every point predictor |
| `dataset_level_wilcoxon.py` | pooled vs. groupwise significance test at dataset level |
| `persistence_baselines.py` | model vs. one-step persistence forecast |
| `deterministic_predictor_comparison.py` | single pass vs. MC-Dropout mean as the interval centre |
| `deployability_benchmark.py` | model size, latency, and calibration overhead |

**Results, figures, and records**

| Path | Purpose |
|---|---|
| `make_results_registry.py`, `results_registry.json` | all results gathered in one file; the paper's numbers come from here |
| `run_all.py` | runs the pipeline in order (prints the plan by default) |
| `figures/` | scripts that draw the paper's figures (`style.py` holds the shared style) |
| `outputs/` | per-experiment result files |
| `data_layer_report.json` | output of `tests/check_data_layer.py` |
| `logs/` | console logs of the long runs (leave-one-dataset-out, repeated splits, data checks) |
| `ERROR_REGISTER.md` | every defect found while auditing this pipeline, with evidence and fix |

## Data

The raw datasets are not redistributed. Download them from the sources below and place them as
described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#2-raw-data).

| Dataset | Cells used* | Source |
|---|---|---|
| MIT / Stanford / Toyota Research Institute | 139 | [data.matr.io](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204); Severson et al., *Nature Energy* 4:383–391 (2019) |
| RWTH Aachen | 47 | [publications.rwth-aachen.de/record/818642](https://publications.rwth-aachen.de/record/818642), DOI 10.18154/RWTH-2021-04545 |
| XJTU | 55 | [doi.org/10.5281/zenodo.10963339](https://doi.org/10.5281/zenodo.10963339) |
| BIT | 72 | [data.mendeley.com/datasets/kw34hhw7xg/3](https://data.mendeley.com/datasets/kw34hhw7xg/3), DOI 10.17632/kw34hhw7xg.3 |
| CALCE CS2 / CX2 | 10 | [calce.umd.edu/battery-data](https://calce.umd.edu/battery-data) |
| Oxford Battery Degradation Dataset 1 | 8 | [ora.ox.ac.uk](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac), DOI 10.5287/bodleian:KO2kdmYGg |
| NASA PCoE | 6 | [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |

\*Cells that contribute at least one window after preprocessing (337 of the 339 loaded).

## Citation

> Sadman, R. *How Far Does Calibrated Uncertainty Transfer Across Battery Datasets? An Empirical
> Study of Conformal Calibration for State-of-Health Prediction on Seven Public Datasets.*
> Submitted to the *Journal of Energy Storage* (2026).

Please also cite the original publications of the datasets you use.

## License

Code: [MIT License](LICENSE). The datasets are not covered by this license; see each source for its
terms.
