# Experiment Summary

## Baselines vs proposed model

| Model | Test MAE | Test RMSE |
|---|---|---|
| baseline_linear | 0.02897 | 0.10668 |
| baseline_svr | 0.02319 | 0.19159 |
| baseline_xgboost | 0.01785 | 0.07312 |
| **ablation_full (proposed)** | **0.02057** | **0.06498** |

## Architecture ablations

| Variant | Test MAE | Test RMSE | Overall coverage |
|---|---|---|---|
| full | 0.02057 | 0.06498 | 68.89% |
| no_attention | 0.02534 | 0.06563 | 83.78% |
| no_multiscale | 0.02103 | 0.06556 | 72.95% |
| plain_lstm | 0.02641 | 0.06863 | 75.49% |
| gru | 0.02801 | 0.07000 | 82.52% |

## Conformal calibration ablation

| Mode | Overall coverage | Per-dataset coverage |
|---|---|---|
| groupwise | 68.89% | BIT=87.5%, CALCE=85.7%, MIT=75.5%, NASA=66.7%, Oxford=100.0%, RWTH=54.5%, XJTU=94.1% |
| pooled | 76.75% | BIT=26.2%, CALCE=4.1%, MIT=82.6%, NASA=0.0%, Oxford=42.9%, RWTH=79.3%, XJTU=40.4% |
| normalized_groupwise | 81.52% | BIT=88.2%, CALCE=78.6%, MIT=86.9%, NASA=100.0%, Oxford=98.2%, RWTH=71.7%, XJTU=95.1% |

## Window-size ablation

| Window (cycles) | Test MAE | Test RMSE |
|---|---|---|
| 10 | 0.02665 | 0.08094 |
| 20 | 0.02057 | 0.06498 |
| 30 | 0.01999 | 0.05830 |
| 50 | 0.02127 | 0.05635 |

## Multi-seed results (mean ± std)

MAE = 0.02207 ± 0.00114 (n=3 seeds)

RMSE = 0.06714 ± 0.00167 (n=3 seeds)

Overall coverage = 72.58% ± 2.79% (n=3 seeds)

Note: coverage varies more across seeds than MAE/RMSE — report the spread, not a single run's number, when discussing conformal calibration reliability.


## Leave-one-dataset-out generalization

| Held-out dataset | Test MAE | Test RMSE | Coverage | n test samples |
|---|---|---|---|---|
| NASA | 0.00700 | 0.01154 | 43.95% | 562 |
| CALCE | 0.30648 | 0.34739 | 0.81% | 617 |
| Oxford | 0.00377 | 0.01243 | 37.33% | 359 |
| MIT | 0.00134 | 0.00303 | 87.56% | 111349 |
| BIT | 0.08132 | 0.12699 | 15.19% | 5543 |
| XJTU | 0.02223 | 0.06914 | 31.01% | 18786 |
| RWTH | 0.00436 | 0.00771 | 91.72% | 110432 |
