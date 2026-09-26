# Multi-Scale BiLSTM with Attention and Groupwise Conformal Calibration for Reliable State-of-Health Prediction Across Heterogeneous Lithium-Ion Battery Datasets

Code, trained models, and reference results for the paper of the same title.

The pipeline forecasts lithium-ion battery State-of-Health (SOH) from capacity-fade
curves pooled across **seven public datasets** (NASA, CALCE, Oxford, MIT/Severson,
BIT, XJTU, RWTH Aachen). A multi-scale bidirectional LSTM with self-attention
produces point forecasts and MC-Dropout uncertainty; split conformal prediction,
calibrated **separately per dataset (groupwise / Mondrian)**, turns those into
prediction intervals with a finite-sample coverage guarantee for each data source.

## Method at a glance

| Stage | What happens | File |
|---|---|---|
| Loading | One parser per dataset turns raw files into `(dataset, cell_id, cycle, capacity)` rows | `*_loader.py` |
| SOH | `SOH = capacity / median(capacity of first 3 cycles)`, per cell; points outside `(0, 1.10)` dropped | `preprocessing.py` |
| Split | **Cell-independent**, stratified by dataset (70 / 15 / 15, seed 42): every cycle of a cell lands in exactly one split | `preprocessing.py` |
| Normalisation | Per-dataset z-score using **train cells only** | `preprocessing.py` |
| Windows | 20 past cycles → SOH one cycle ahead | `preprocessing.py`, `config.py` |
| Model | Two BiLSTM branches (full resolution + 2× average-pooled) → concatenation → additive self-attention → MLP head | `model.py` |
| Uncertainty | MC-Dropout, 30 stochastic forward passes | `model.py`, `train.py` |
| Calibration | Split conformal, α = 0.10, calibrated on the validation cells: `groupwise` (one quantile per dataset), `pooled`, or `normalized_groupwise` (residuals scaled by MC-Dropout σ) | `conformal.py` |

Hyperparameters live in `config.py` (hidden size 64, 2 LSTM layers, dropout 0.2,
attention dim 32, Adam lr 1e-3, batch 64, up to 100 epochs with early-stopping
patience 15).

## Repository layout

```
config.py                      paths + all hyperparameters
nasa_loader.py … rwth_loader.py  one loader per dataset (7)
rwth_local_outlier_fix.py      RWTH single-row outlier removal (used by rwth_loader)
inspect_data.py                sanity-check that every raw dataset is found and parses
preprocessing.py               SOH, cell-independent split, windowing -> cache/processed.npz
model.py                       multi-scale BiLSTM + attention, ablation variants, GRU baseline
conformal.py                   pooled / groupwise / normalized-groupwise split conformal
train.py                       training engine (train_one) + MC-Dropout + calibration
baselines.py                   linear regression, SVR, XGBoost on the same windows
leave_one_out.py               leave-one-dataset-out (LODO) generalisation
experiments.py                 runs the full experiment suite -> outputs/experiment_summary.*

remaining_analysis.py          significance testing (Wilcoxon, Cliff's delta, bootstrap CIs) + data-quality checks
analyze_multiseed_coverage.py  per-dataset coverage across the 3 training seeds
investigate_seed_sensitivity.py  why coverage varies across seeds (borderline-sample analysis)
uq_quality_analysis.py         MC-Dropout reliability / sparsification (AUSE)
ensemble_conformal_calibration.py  seed-ensemble-normalised groupwise conformal
baseline_conformal_calibration.py  pooled vs groupwise calibration on SVR / XGBoost / plain LSTM
hierarchical_calibration.py    shrinkage (hierarchical) groupwise calibration
field_proxy_validation.py      accuracy/coverage on real-world-like protocols (XJTU RW, BIT arbitrary)
diagnose_calce_generalization_gap.py  CALCE vs pooled datasets: fade rate, knee, SOH range
calce_propensity_overlap.py    CALCE input-space overlap with the training pool (propensity AUC)
calce_output_range_extrapolation.py   CALCE target-range extrapolation check
windowing_bridge_sensitivity.py  coverage on windows that do / do not span removed rows
xjtu_bridging_isolation.py     separates correction-related from native gaps in XJTU
run_xjtu_sensitivity_experiment.py  full rerun with the XJTU correction disabled (uses clear_everything.py)

generate_all_figures.py        main manuscript figures
benchmark_inference_efficiency.py  parameter count and inference latency per architecture
visualize_attention.py         attention weights on healthy vs degraded windows
run_all_manuscript_figures.py  runs the four figure scripts above in order

cell_clustered_bootstrap.py    cell-clustered (not per-window) bootstrap coverage CIs
hgc_cp.py                      heterogeneity-guided conformal prediction (HGC-CP): MMD-based
                                per-source weights + a corrected weighted conformal quantile
generate_val_predictions.py    MC-Dropout predictions on the validation set (input to the below)
hgc_cp_evaluation_harness.py   HGC-CP vs pooled vs groupwise, calibration-level LODO
nested_hyperparameter_search.py  selects HGC-CP's (lambda, beta) without tuning on the target
hgc_cp_ablations.py            isolates HGC-CP's weighting, lambda, and inflation components
few_shot_calibration.py        HGC-CP with 0/1/3/5/10 target-owned calibration cells
plot_hgc_cp_comparison.py      regenerates the HGC-CP comparison figure from reported numbers

field_experiment.py            single- and multi-vehicle field-EV capacity extraction (exploratory)
plot_field_results.py          plots the single-vehicle field capacity trend vs plausible fade

results/                       reference outputs from the runs reported in the paper (see below)
```

## Installation

Python 3.10 was used. From the repository root:

```bash
pip install -r requirements.txt
```

A CUDA GPU is used automatically if available; everything also runs on CPU.

## Data

The raw datasets are **not redistributed** here; download them from their original
sources (each has its own licence and citation requirements):

| Dataset | Cells used* | Source |
|---|---|---|
| NASA PCoE Battery Data Set | 6 | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ |
| CALCE CS2 / CX2 | 10 | https://calce.umd.edu/battery-data |
| Oxford Battery Degradation Dataset 1 (Birkl) | 8 | https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac |
| MIT–Stanford–Toyota fast-charging (Severson et al., *Nature Energy*, 2019) | 139 | https://data.matr.io/1/ |
| BIT (Mendeley Data `kw34hhw7xg`) | 72 | https://data.mendeley.com/datasets/kw34hhw7xg |
| XJTU battery dataset (Batch-1 … Batch-6) | 55 | Wang et al., *Nature Communications* 15, 4332 (2024) |
| RWTH Aachen (RWTH-2021-04545) | 47 | https://publications.rwth-aachen.de/record/818642 |

\*Cells that contribute at least one training/validation/test window after
preprocessing (337 in total).

A single-vehicle sample of real field-EV telemetry (Liu et al., *Nature
Communications* 16, 1137, 2025) is used only for the exploratory,
illustrative analysis in `field_experiment.py` / `plot_field_results.py` —
https://github.com/HoraceLiu1010/Multi-modal-SOH-estimation-framework. This
data is not part of the seven training/evaluation datasets above.

Place everything under one folder (default: `./data`, or set the
`BATTERY_DATA_ROOT` environment variable) with this layout, which is what
`config.py` expects:

```
data/
├── 5. Battery Data Set/            NASA .mat files (B0005, B0006, B0007, B0018, B0049–B0052; searched recursively)
├── CS2_33/ CS2_34/ CS2_35/ CS2_38/ CALCE Arbin .xlsx files, one folder per cell
├── CX2_16/ CX2_31/ CX2_33/ CX2_34/ CX2_35/ CX2_36/ CX2_37/
├── Oxford_Battery_Degradation_Dataset_1.mat
├── FastCharge/                     MIT FastCharge_*_structure.json files (searched recursively from data/)
├── kw34hhw7xg-3/                   BIT: "Cycled with Arbitrary Uses Profiles/", "Cycled with Fixed Current Profiles/"
├── Battery Dataset/                XJTU: Batch-1/ … Batch-6/ with per-cell .mat files
└── RWTH-2021-04545_818642/Rawdata/Rohdaten/   RWTH Basytec session CSVs
```

Then check that every dataset is found and parses:

```bash
python inspect_data.py
```

## Reproducing the results

```bash
python preprocessing.py            # parse raw data, split, window -> cache/processed.npz
python experiments.py              # baselines, ablations, conformal modes, window sizes,
                                   # 3 seeds, 7 LODO runs -> outputs/  (several hours on CPU)
python remaining_analysis.py       # significance tests (needed by the figure scripts)
python run_all_manuscript_figures.py   # figures -> outputs/figures/
```

The additional analyses (list above) each read the saved predictions in
`outputs/` and can be run individually afterwards, e.g.
`python analyze_multiseed_coverage.py`. Each script's docstring states its
inputs, method, and output file.

`experiments.py` resumes from existing `outputs/*_results.json`; run
`python clear_everything.py` first to force a complete retrain.

### Data-quality corrections

Two corrections are applied inside the loaders and are part of the reported results:

- **XJTU**: Batch-6 `Sim_satellite_*` cells contain periodic calibration/reference
  blocks of identical values that are not degradation measurements; they are removed
  in `xjtu_loader.py`. Set `XJTU_APPLY_SENTINEL_FIX=0` to disable this, or run
  `run_xjtu_sensitivity_experiment.py` for the full with/without comparison.
- **RWTH**: isolated single-row capacity outliers (mostly at the start of each session
  file) are removed by `rwth_local_outlier_fix.py`.

### Heterogeneity-guided conformal prediction (HGC-CP) and cell-clustered coverage

Two further analyses build on the main pipeline's saved predictions:

- **Cell-clustered bootstrap** (`cell_clustered_bootstrap.py`): Table 7's per-window
  Clopper-Pearson intervals treat every test window as an independent trial; this
  resamples whole test *cells* instead to check how much within-cell correlation
  widens the true coverage uncertainty.
- **HGC-CP** (`hgc_cp.py`): an alternative to groupwise/hierarchical calibration for a
  target with little or no calibration data of its own, weighting each of the other
  six datasets by MMD-based distributional similarity to the target. Run in order:
  `generate_val_predictions.py` (once, to produce validation-set MC-Dropout
  predictions) → `nested_hyperparameter_search.py` (selects λ, β) →
  `hgc_cp_evaluation_harness.py` (main comparison, calibration-level LODO) →
  `hgc_cp_ablations.py` / `few_shot_calibration.py` (component and calibration-budget
  ablations) → `plot_hgc_cp_comparison.py` (figure).

## Reference results

`results/` holds the outputs of the runs reported in the paper, so the numbers can be
checked without retraining. A fresh run writes to `outputs/` instead, so the two can be
compared directly.

- `results/experiment_summary.md` / `.json`: all tables from `experiments.py`
- `results/*_results.json`: one file per training run (per-dataset MAE and coverage)
- `results/*.json`, `results/*.txt`: outputs of the additional analyses, named after their scripts
- `results/models/*_model.pt`: trained weights for every run (load with `model.build_model(variant)`)
- `results/figures/`: figures generated from these runs

The MAE / RMSE reported by `train.py` and `experiments.py` are computed on the
per-dataset standardised SOH targets (the space the model is trained in).
`leave_one_out.py` reports both standardised and de-normalised (raw SOH) errors.

## Citation

If you use this code, please cite the paper (full reference to be added on publication):

> *Multi-Scale BiLSTM with Attention and Groupwise Conformal Calibration for Reliable
> State-of-Health Prediction Across Heterogeneous Lithium-Ion Battery Datasets.*

Please also cite the original dataset publications listed under [Data](#data).

## License

Code is released under the [MIT License](LICENSE). The datasets are not covered by
this license; see each dataset's original source for its terms.
