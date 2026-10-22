# How Far Does Calibrated Uncertainty Transfer Across Battery Datasets?

[![checks](https://github.com/Raheeb223/battery-soh-conformal-prediction/actions/workflows/checks.yml/badge.svg)](https://github.com/Raheeb223/battery-soh-conformal-prediction/actions/workflows/checks.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code and results for the paper *How Far Does Calibrated Uncertainty Transfer Across Battery
Datasets? An Empirical Study of Conformal Calibration for State-of-Health Prediction on Seven Public
Datasets* (R. Sadman, submitted to the *Journal of Energy Storage*).

The study trains one state-of-health (SOH) predictor on seven public lithium-ion battery datasets
(339 cells from seven research groups) and asks how well its prediction intervals transfer between
datasets. It reports coverage per dataset, not only in aggregate, for several ways of calibrating
the intervals. It then repeats the comparison on further cell-level splits, other predictors,
longer forecast horizons, a deep ensemble, datasets left out of training, and electric-vehicle
field data.

**Contents:** [Main findings](#main-findings) ·
[Reviewer quick start](#reviewer-quick-start-no-raw-data-no-gpu) ·
[Where each result comes from](#where-each-result-in-the-paper-comes-from) ·
[Calibration methods](#calibration-methods) · [Tests](#what-the-tests-check) ·
[Repository map](#repository-map) · [Data](#data) · [Full reproduction](REPRODUCIBILITY.md)

## Main findings

Every number below is read from [`results_registry.json`](results_registry.json), the same file the
paper's numbers are generated from.

1. **One pooled quantile hides large errors on individual datasets.** Pooled calibration reaches
   90% coverage overall but covers only 44.2% of BIT's test points. Calibrating each dataset on its
   own cells (groupwise) is closer to the 90% target in 26 of 30 dataset–split pairs over 5
   cell-level splits. The same holds for seven predictors, persistence and drift forecasts,
   horizons of 10 and 50 cycles, and a deep ensemble.
2. **Borrowing from other datasets does not beat own data.** Linear empirical-Bayes shrinkage cuts
   BIT's coverage to 41.5%. A Bayesian model on the log scale avoids this, but only by returning
   each dataset's own quantile: 3.76 vs. 3.77 points from the target, averaged over 5 splits.
   Without any of the dataset's own cells, the best borrowing method is still 9.91 points off.
3. **A dataset never seen in training needs its own calibration cells.** Calibrated on the other
   datasets, coverage ranges from 21.2% (BIT) to 97.2% (RWTH). The left-out dataset's own cells
   restore coverage, with intervals up to 4.9 times wider.
4. **How many calibration cells are enough depends on the dataset.** It ranges from 1 cell (RWTH)
   to 10 (BIT). For some datasets, on some splits, own cells under-cover however many are used.
5. **Laboratory calibration does not transfer to the field.** On EVBattery data from 191 vehicles,
   laboratory calibration covers only 15.6–19.6% of test points; calibrating on field data restores
   about 90%.

## Reviewer quick start (no raw data, no GPU)

Every per-experiment result file is in [`outputs/`](outputs/). The commands below rerun the
verification checks, rebuild every number in the paper from those files, and redraw the figures.
They take a few minutes on a laptop, with Python 3.10:

```bash
git clone https://github.com/Raheeb223/battery-soh-conformal-prediction.git
cd battery-soh-conformal-prediction
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python tests/run_checks.py                  # 6 checks PASS; 2 SKIP (need a model checkpoint / raw data)
python make_results_registry.py             # ends with "0 value(s) UNVERIFIED"
git diff --stat results_registry.json       # empty: the rebuilt numbers match the committed ones
python figures/make_results_figures.py      # results figures -> figures/out/
python figures/make_prediction_figures.py   # example predictions and error spreads -> figures/out/
```

**To trace a number in the paper,** search for it in `results_registry.json`. The `values` table
holds every number exactly as printed in the paper, and the `sections` table holds the raw results
behind it. For example, the 44.2% BIT coverage is `values.PooledCoverageBIT`. The build step in
[`make_results_registry.py`](make_results_registry.py) that produces each value names the
`outputs/` file it reads, and the next table names the script that wrote that file.

Rerunning the experiments themselves (training, leave-one-dataset-out, field check) needs the raw
datasets and a GPU; [REPRODUCIBILITY.md](REPRODUCIBILITY.md) gives the environment, data layout and
run order. [CHANGELOG.md](CHANGELOG.md) lists the corrections to code and data handling that
affect the results.

## Where each result in the paper comes from

Sections, tables and figures are named by their title in the paper, since journal typesetting may
renumber them. All output files are in `outputs/`; figure scripts are in `figures/`.

| Paper section | Result | Script | Output read by the registry |
|---|---|---|---|
| Introduction, Methods | Problem-setting figure; pipeline diagram | `figures/make_problem_setting.py`, `figures/make_block_diagram.py` | (drawings) |
| Data | Cell, window and test-cell counts; data-quality corrections | `preprocessing.py`, loaders, `verify_calce_v2.py`, `verify_rwth_v2.py`, `xjtu_bridging_*.py` | `data_counts.json`, `rwth_verification_v2.json`, `xjtu_bridging_*.json`, `data_layer_report.json` |
| One pooled quantile hides large errors | Coverage per dataset, pooled vs. groupwise (figure) | `train.py`, `experiments.py`, `figures/make_results_figures.py` | `conformal_{pooled,groupwise,normalized_groupwise}_results.json` |
| | Example predictions; error spread per dataset (figures) | `figures/make_prediction_figures.py` | `figure_inputs.npz`, `conformal_*_predictions.npz` |
| | Coverage for every point predictor (table) | `baseline_conformal_coverage.py` | `baseline_conformal_results.json` |
| | Robustness over 5 splits, with cell-bootstrap gains (table) | `repeated_split_experiment.py`, `pooled_vs_groupwise_bootstrap.py`, `dataset_level_wilcoxon.py` | `repeated_split_summary.json`, `pooled_vs_groupwise_bootstrap.json`, `dataset_level_wilcoxon.json` |
| Borrowing from other datasets | Coverage of every method on the main split (table) | `hierarchical_calibration_v2.py`, `run_nested_hp_search_all.py`, `mwhc_calibration.py`, `train_cqr.py` | `hierarchical_calibration_v2.json`, `nested_hp_search_results.json`, `mwhc_results.json`, `cqr_*_results.json` |
| | Every method over 5 splits, including log-scale, Bayesian and covariate borrowing (table) | `split_replication.py`, `borrowing_variants.py`, `hgc_cp_validation_only.py` | `split_replication.json`, `borrowing_variants.json`, `hgc_cp_validation_only.json` |
| How much calibration data is enough | Coverage vs. number of calibration cells (figure); cells needed on every split (table) | `calibration_size_sensitivity.py`, `budget_across_splits.py` | `calibration_size_sensitivity.json`, `budget_across_splits.json` |
| A dataset never seen in training | Leave-one-dataset-out, calibrated on other or own cells (table) | `leave_one_out.py`, `lodo_target_calibration.py` | `lodo_*_results.json`, `lodo_target_calibration.json` |
| Field data from electric vehicles | Field coverage (table and figure) | `field_check.py`, `figures/make_results_figures.py` | `field_check.json` |
| Longer horizons and naive forecasts | Error of model, persistence and drift at 1, 10, 50 cycles; their calibration (two tables) | `horizon_experiment.py`, `persistence_baselines.py` | `horizon_experiment.json`, `persistence_baselines.json` |
| Deep ensemble | Ensemble vs. MC-Dropout on every split (table) | `deep_ensemble.py`, `deep_ensemble_splits.py` | `deep_ensemble.json`, `deep_ensemble_splits.json` |
| Learning the interval directly (CQR) | CQR vs. groupwise calibration (figure) | `train_cqr.py`, `figures/make_results_figures.py` | `cqr_*_results.json` |
| How precise each coverage value is | Coverage range over splits (figure); cell-clustered intervals | `repeated_split_experiment.py`, `cell_clustered_bootstrap.py`, `predictor_consistent_recalibration.py` | `repeated_split_summary.json`, `predictor_consistent_recalibration.json` |
| Appendix: HGC-CP and HGC-CP-val | Design chosen per target and its coverage (table) | `run_nested_hp_search_all.py`, `hgc_cp_validation_only.py` | `nested_hp_search_results.json`, `hgc_cp_validation_only.json` |
| Appendix: MWHC | Sensitivity to the temperature (figure) | `mwhc_calibration.py` | `mwhc_results.json` |
| Appendix: point accuracy | Accuracy of every predictor, ablations, window sizes, seeds (table) | `experiments.py`, `baselines.py` | `baseline_*_results.json`, `ablation_*_results.json`, `window_*_results.json`, `multiseed_*_results.json` |
| Appendix: interval centre, deployment cost | Single pass vs. MC-Dropout mean; latency and size (tables) | `deterministic_predictor_comparison.py`, `deployability_benchmark.py` | `deterministic_predictor_results.json`, `deployability_results.json` |
| Appendix: deep ensemble on the main split | Calibration around the ensemble (table) | `deep_ensemble.py` | `deep_ensemble.json` |
| Appendix: independent checks | Verification of the statistical code | `tests/` | (see [tests/README.md](tests/README.md)) |

## Calibration methods

| Method | Calibrates each dataset on | Code |
|---|---|---|
| Pooled | all datasets together | `conformal.py` |
| Groupwise (Mondrian) | its own validation cells | `conformal.py` |
| Normalized groupwise | its own cells, scaled by the model's uncertainty | `conformal.py` |
| Hierarchical (empirical Bayes) | its own cells, shrunk toward the other datasets | `hierarchical_calibration_v2.py` |
| Stronger borrowing (log scale, Bayesian, covariates) | its own cells, or none, with better-specified shrinkage | `borrowing_variants.py` |
| HGC-CP | the other datasets, weighted by similarity (MMD) | `hgc_cp.py` |
| MWHC | hierarchical, with similarity-weighted shrinkage | `mwhc_calibration.py` |
| CQR | a model trained to output the interval directly | `train_cqr.py` |

## What the tests check

`tests/` holds standalone verification scripts; [tests/README.md](tests/README.md) explains each
check and why it matters. In short:

| Script | Verifies |
|---|---|
| `check_conformal.py` | split-conformal coverage on synthetic data; the calibration quantile against an independent implementation; groupwise and normalized calibration reduce correctly; the missing-data fallback |
| `check_hierarchical_calibration.py` | the DerSimonian–Laird estimator against hand calculations and an independent implementation |
| `check_bayesian_borrowing.py` | the Bayesian log-scale borrowing against a brute-force posterior, its limits, and its meta-regression |
| `check_hgc_cp.py` | HGC-CP's similarity weights and its reduction to ordinary split conformal |
| `check_hgc_cp_split_equivalence.py` | the fast HGC-CP code path gives bit-identical intervals |
| `check_evbattery_loader.py` | the field-data loader refuses unsafe pickle files and reads valid ones |
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
| `model.py` | multi-scale bidirectional LSTM with self-attention, ablation variants, GRU and Transformer baselines, MC-Dropout |
| `train.py` | training loop, MC-Dropout prediction, conformal calibration and evaluation |
| `generate_val_predictions.py` | the main model's MC-Dropout predictions on the validation set, which every recalibration analysis reads |
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
| `lodo_target_calibration.py` | the withheld dataset calibrated on its own cells, and how many it needs |
| `deep_ensemble.py` | a five-member deep ensemble as a second source of uncertainty, calibrated with the same methods |
| `deep_ensemble_splits.py` | the same deep ensemble on the four further cell-level splits |
| `borrowing_variants.py` | stronger borrowing baselines: shrinkage on the log scale, a Bayesian hierarchical model, and label-free covariates, on all five splits |
| `budget_across_splits.py` | the calibration-cell budget repeated on all five splits |
| `horizon_experiment.py` | 10- and 50-cycle forecast horizons, with persistence and drift forecasts and their calibration |
| `pooled_vs_groupwise_bootstrap.py` | size of the groupwise gain on each dataset, with paired cell-bootstrap intervals |
| `repeated_split_experiment.py` | further cell-level splits with the model seed fixed |
| `split_replication.py` | the calibration-strategy comparison re-run on those splits with every design frozen; a held-out check of HGC-CP |
| `hgc_cp_validation_only.py` | HGC-CP-val: HGC-CP with its whole design (representation, λ, β) chosen on validation data only, on all five splits |
| `evbattery_loader.py`, `field_preprocessing.py`, `field_check.py` | field check on the EVBattery fleet data: safe loading of its pickle files, the same preprocessing as the laboratory data, and the main analyses repeated on five vehicle-level splits |
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
| `CHANGELOG.md` | corrections to code and data handling that affect the results |

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

The field check uses one more public dataset, analysed separately with its own model:

| Dataset | Vehicles used | Source |
|---|---|---|
| EVBattery (charging records of electric vehicles, three manufacturers) | 191 with capacity labels | [figshare, DOI 10.6084/m9.figshare.23301881](https://doi.org/10.6084/m9.figshare.23301881), CC BY 4.0; He et al., arXiv:2201.12358 |

## Citation

> Sadman, R. *How Far Does Calibrated Uncertainty Transfer Across Battery Datasets? An Empirical
> Study of Conformal Calibration for State-of-Health Prediction on Seven Public Datasets.*
> Submitted to the *Journal of Energy Storage* (2026).

Please also cite the original publications of the datasets you use.

## License

Code: [MIT License](LICENSE). The datasets are not covered by this license; see each source for its
terms.
