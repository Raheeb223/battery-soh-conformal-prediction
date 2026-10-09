# Test Report

| Stage | Test | Result | Evidence |
|---|---|---|---|
| 0 | Backup cache/ + outputs/ before any change | PASS | `backup_20261009_1210/`, 580.5 MB, 116 files |
| 0 | Git repository initialized | PASS | `git init` succeeded; was not a repo before |
| 0 | Python/package versions recorded | PASS | `requirements_frozen.txt`, 77 packages |
| 0 | GPU / CUDA / torch recorded | PASS | RTX 3060 12GB, driver 572.83, torch 2.5.1+cu121, CUDA available |
| 0 | `tests/` directory created | PASS | |
| 0 | Project-map docs exist as claimed | **FAIL** | `AUDIT_FIX_STATUS.md`, `REPRODUCIBILITY.md`, `main.tex` all absent from disk |
| 1 | K1 (leave_one_out.py predictor mismatch) | **FAIL (confirmed defect)** | See ERROR_REGISTER E1 |
| 1 | K3 (baseline conformal coverage) | **FAIL (confirmed defect)** | See ERROR_REGISTER E2 |
| 1 | K9a (SOH docstring) | **FAIL (confirmed defect)** | See ERROR_REGISTER E3 |
| 1 | K9b (MC-Dropout scope) | **FAIL (confirmed, documentation gap)** | See ERROR_REGISTER E4 |
| 1 | Hierarchical calibration: single source of truth | **FAIL (new finding, not in brief)** | See ERROR_REGISTER E5 |
| 1 | Loaders: bare except / silent skip sweep | PARTIAL | 2 bare `except Exception:` found (nasa_loader.py:37, oxford_loader.py:47), context not yet read |
| 1 | Loaders: unseeded randomness sweep | PASS (grep-level) | No unseeded `np.random.*` in loaders/baselines.py |
| 1 | config.py audit | PASS | No `CALIBRATION_PREDICTOR`, no `SPLIT_SEED` present (brief overstates; `train.py` uses a soft `getattr` default) |

| 1 | All "never seen" files read in full (conformal.py, baselines.py, model.py, 7 loaders, hgc_cp*.py, nearest_source_baseline.py, nested_hyperparameter_search.py, 4 plot_*.py, hierarchical_calibration_v2.py + _corrected.py) | **COMPLETE** | |
| 1 | K5 (HGC-CP arm semantics: nearest_source, few-shot "Pooled") | PASS -- verified consistent, as documented | `hgc_cp_evaluation_harness.py:243-260`, `few_shot_calibration.py:114-133` |
| 1 | plot_*.py hardcoded values | PASS (not a defect) -- all 4 transparently labeled as "copied from the actual run", to be superseded by Stage 11's registry-driven generation | |
| 1 | E9: xjtu_loader cycle-renumbering claim vs code | INVESTIGATING -- pending Stage 2's window_cycle_indices output | |

*(Table continues as each stage completes.)*
