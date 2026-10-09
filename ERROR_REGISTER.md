# Error Register

Format: ID | Severity | Location | Problem | Evidence | Consequence | Fix | Status

---

### E1 | MAJOR | leave_one_out.py:196,201 | Predictor mismatch (K1)
**Problem:** `run_lodo()` calibrates `SplitConformal` on `predict_in_batches()` (deterministic,
dropout off) validation predictions but evaluates on `mc_dropout_predict()` (MC-mean, dropout on)
test predictions. `train.py` was fixed to use the same predictor for both (default
`getattr(config, "CALIBRATION_PREDICTOR", "mc_mean")`); `leave_one_out.py` was not touched.
**Evidence:** Direct read, `leave_one_out.py:196` vs `:201`.
**Consequence:** LODO coverage numbers are not valid split-conformal coverage (calibration and
test nonconformity scores come from different predictors, breaking exchangeability).
**Fix:** Use `mc_dropout_predict()` for the validation calibration pass too, matching `train.py`.
No saved LODO checkpoints exist, so this requires a full retrain of all 7 held-out
configurations.
**Status:** CONFIRMED. Fix + retrain in progress (background).

### E2 | MAJOR | baselines.py:82-84 | No baseline conformal coverage possible (K3)
**Problem:** `baseline_{name}_predictions.npz` saves only `y_test, mean_pred, lab_test` — no
validation predictions, no cell IDs. Conformal calibration needs validation residuals; none are
saved. `outputs/baseline_conformal_results.json` does not exist. No plain-LSTM/GRU baseline
result files exist either (only linear/svr/xgboost).
**Evidence:** Direct read, `baselines.py:41-84`; `outputs/` listing.
**Consequence:** Section 5.3.5-equivalent cross-architecture conformal comparison cannot be
reproduced from existing files.
**Fix:** Extend `baselines.py` to also predict and save validation-set predictions + `cell_val`/
`cell_test`; add a plain-LSTM/GRU baseline-conformal path; compute pooled+groupwise coverage per
dataset per architecture; write `baseline_conformal_results.json`.
**Status:** CONFIRMED. Fix pending.

### E3 | MODERATE | preprocessing.py:6 | Stale docstring (K9a)
**Problem:** Module docstring states `SOH = capacity(cycle)/capacity(cycle=1)`; actual code
(`add_soh()`, line 150) uses the median of the first 3 cycles.
**Evidence:** Direct read.
**Consequence:** Documentation-only; no effect on results. Misleads a reader of the code/paper
if copied verbatim.
**Fix:** Correct the docstring to match the implementation.
**Status:** CONFIRMED. Trivial fix, pending.

### E4 | MODERATE | model.py:152-165 | MC-Dropout scope (K9b)
**Problem:** `enable_mc_dropout()` only sets `nn.Dropout` submodules to train mode via
`isinstance` check. `nn.LSTM`'s internal recurrent dropout (set via the `dropout=` kwarg) is a
different mechanism tied to the LSTM module's own `.training` flag and is never re-enabled, so
MC-Dropout sampling only perturbs the post-fusion and head dropout layers, not the LSTM layers.
**Evidence:** Direct read, `model.py:53,60,69,73,152-165`.
**Consequence:** MC-Dropout uncertainty estimates reflect a narrower source of stochasticity
than "full MC-Dropout" might imply. Not necessarily wrong, but must be stated precisely in the
methodology section rather than described as dropout being active "throughout the network."
**Fix:** Document the exact scope in the methodology. Optionally (not required) test enabling
LSTM dropout during MC sampling as a sensitivity check.
**Status:** CONFIRMED. Documentation fix + optional sensitivity check pending.

### E5 | CRITICAL | hierarchical_calibration_v2.py vs hierarchical_calibration_corrected.py | Divergent, unreconciled branches
**Problem:** Two separate "final-looking" hierarchical calibration implementations exist, each
with its own dependent sensitivity script, neither referencing the other:
  - `hierarchical_calibration_v2.py` ← imported by `hierarchical_calibration_log_scale_sensitivity.py`
  - `hierarchical_calibration_corrected.py` ← imported by `prior_plus_n_subsample_check.py`
Five further variants are unimported by anything: `hierarchical_calibration (1).py`,
`corrected_hierarchical_calibration.py`, `bootstrap_tau2_sensitivity.py`, `sigma_g2_patch.py`,
`validate_sigma_g_proxy.py`.
**Evidence:** `grep` import chain across all `*.py`; file listing with mtimes.
**Consequence:** Cannot determine which branch produces the paper's reported hierarchical
calibration numbers without reading both in full and checking the weight formula against the
brief's explicit criterion (`tau2/(tau2+sigma2_g)`, not the withdrawn `n/(n+tau2/sigma2)`).
**Fix:** Read both files in full, compare against the correct formula, pick the correct/more
recent one as canonical, note the decision here, archive (not delete) the others.
**Status:** **RESOLVED.** `hierarchical_calibration_v2.py` is canonical — confirmed by direct
read: `hierarchical_calibration_corrected.py:206-210` still uses the withdrawn weight
`w = n_g/(n_g+tau2/sigma2_g)`; `hierarchical_calibration_v2.py:158` uses the correct
`w = tau2/(tau2+sigma2_g)`. `v2`'s own docstring independently confirms it supersedes both
`hierarchical_calibration.py` and `hierarchical_calibration_corrected.py`, citing a prior audit
register (E1/E3/E6/E13) that no longer exists on disk — consistent with `AUDIT_FIX_STATUS.md`
being missing. All downstream work uses `hierarchical_calibration_v2.py` / its JSON output only.

### E6 | MAJOR | prior_plus_n_subsample_check.py | Consumes the stale, buggy hierarchical file
**Problem:** `import hierarchical_calibration_corrected as h` (line 26) — this script depends on
the withdrawn shrinkage-weight formula (see E5).
**Evidence:** Direct read.
**Consequence:** Any number this script currently produces is invalid and must not be used.
**Fix:** Repoint the import to `hierarchical_calibration_v2`, update any API calls that changed
between the two files (the function/variable names differ), rerun.
**Status:** CONFIRMED. Fix pending.

---

*(IDs continue as Stage 1 proceeds. Severity: CRITICAL/MAJOR/MODERATE/MINOR.)*
