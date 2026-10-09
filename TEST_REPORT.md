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
| 2/8 | **K8 full CALCE verification** (`verify_calce_v2.py`) | **PASS, all sub-checks** | 10 cells, 14,188 total rows (matches brief exactly); per-cell SOH-filter pass rate 99.2-99.8% (>=99% required); median \|step\| 0.0012-0.0024 (within 0.001-0.003 required); drop-policy removes exactly 0.634% of rows (matches brief's "~0.63%"); 0 undated files across all 10 cells (chronological order fully reconstructable); CX2_31 confirmed contributing 0 rows (E7, resolved) |
| 2/8 | CALCE step-correlation (-0.43 to -0.51 across all 10 cells) | FLAGGED FOR MANUSCRIPT, not a defect | Remarkably consistent negative autocorrelation of successive differences -- the expected statistical signature of measurement noise superimposed on a smooth trend (first-differencing white noise induces ~-0.5 lag-1 autocorrelation by construction), not necessarily a residual data-quality artifact. Should be stated and justified explicitly in the manuscript rather than asserted as "no alternation" without explanation. |
| 2/8 | CALCE `complete_discharge==False` flag rate (82-100% of rows per cell) | **RESOLVED, not a defect** | `calce_loader_v2.py:284` explicitly documents this flag as "uncalibrated for CALCE" and deliberately excludes it from the drop-policy logic -- matches the brief's own note ("the uncalibrated voltage flag that nearly deleted most data"). High flag rate is expected under an uncalibrated per-cell voltage cutoff, not a data-quality problem; already correctly handled. |
| 7 | **K7: XJTU Sim_satellite bridging isolation** (`xjtu_bridging_isolation.py`, direct run against seed-42 cache) | **FAIL (confirmed defect in docstring, not in results validity)** | `preprocessing.py`'s "renumbering erases the gap" claim is WRONG -- Sim_satellite_* cells show bridging fraction 1.0000 (n=376, ALL windows), other 47 XJTU cells show 0.0000 (n=2229). 376/2605=14.43% exactly matches the previously-reported overall figure: XJTU's bridging is 100% concentrated in Sim_satellite_*, not diffuse native sparsity. See ERROR_REGISTER E9. |
| 7 | **K7: XJTU bridging sensitivity** (`xjtu_bridging_sensitivity.py`, no retrain -- re-slices existing predictions) | **PASS, with disclosed effect size** | Bridging windows (n=376): RMSE(z)=0.4318 vs. clean (n=2229): RMSE(z)=0.0769 (5.6x higher). Pooled XJTU MAE(z): 0.0459 (all) -> 0.0421 (clean only), -8.3% relative. Coverage NOT degraded (94.7% bridging vs 92.7% clean, both >=90% target) -- conformal validity intact; point-accuracy on this 14.4%-of-XJTU subset is measurably worse and must be disclosed in the manuscript's Limitations. |
| 4 | **K4: nested hyperparameter search rerun** (`run_nested_hp_search_all.py`, all 7 targets) | RUNNING (background) | Launched; CPU-only, reuses existing seed-42 cache + predictions, no raw-data reparse or GPU needed. Output: `outputs/nested_hp_search_results.json` (pending). |

*(Table continues as each stage completes.)*
