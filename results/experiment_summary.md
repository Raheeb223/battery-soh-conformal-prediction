# Experiment Summary

## Baselines vs proposed model

| Model | Test MAE | Test RMSE |
|---|---|---|
| baseline_linear | 0.03040 | 0.12356 |
| baseline_svr | 0.01840 | 0.10881 |
| baseline_xgboost | 0.02071 | 0.09095 |
| **ablation_full (proposed)** | **0.02464** | **0.08363** |

## Architecture ablations

| Variant | Test MAE | Test RMSE | Overall coverage |
|---|---|---|---|
| full | 0.02464 | 0.08363 | 88.93% |
| no_attention | 0.02163 | 0.08109 | 76.53% |
| no_multiscale | 0.02818 | 0.08423 | 81.82% |
| plain_lstm | 0.02172 | 0.08131 | 70.23% |
| gru | 0.02584 | 0.08306 | 83.39% |

## Conformal calibration ablation

| Mode | Overall coverage | Per-dataset coverage |
|---|---|---|
| groupwise | 88.93% | BIT=87.1%, CALCE=71.4%, MIT=91.0%, NASA=33.3%, Oxford=100.0%, RWTH=85.1%, XJTU=96.0% |
| pooled | 86.34% | BIT=38.7%, CALCE=10.2%, MIT=91.6%, NASA=0.0%, Oxford=64.3%, RWTH=87.6%, XJTU=62.7% |
| normalized_groupwise | 83.84% | BIT=85.1%, CALCE=81.6%, MIT=89.3%, NASA=100.0%, Oxford=98.2%, RWTH=75.0%, XJTU=92.3% |

## Window-size ablation

| Window (cycles) | Test MAE | Test RMSE |
|---|---|---|
| 10 | 0.02341 | 0.09368 |
| 20 | 0.02464 | 0.08363 |
| 30 | 0.02152 | 0.07420 |
| 50 | 0.02082 | 0.06952 |

## Multi-seed results (mean ± std)

MAE = 0.02399 ± 0.00112 (n=3 seeds)

RMSE = 0.08321 ± 0.00192 (n=3 seeds)

Overall coverage = 82.33% ± 4.67% (n=3 seeds)

Note: coverage varies more across seeds than MAE/RMSE — report the spread, not a single run's number, when discussing conformal calibration reliability.


## Leave-one-dataset-out generalization

| Held-out dataset | Test MAE | Test RMSE | Coverage | n test samples |
|---|---|---|---|---|
| NASA | 0.00677 | 0.01153 | 54.09% | 562 |
| CALCE | 0.23710 | 0.27242 | 0.97% | 617 |
| Oxford | 0.00381 | 0.01267 | 54.32% | 359 |
| MIT | 0.00091 | 0.00186 | 92.43% | 111349 |
| BIT | 0.09913 | 0.15368 | 14.02% | 5543 |
| XJTU | 0.01288 | 0.02144 | 20.19% | 19464 |
| RWTH | 0.00427 | 0.00756 | 88.57% | 110432 |
