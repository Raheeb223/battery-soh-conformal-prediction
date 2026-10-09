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
**Status:** **RESOLVED and verified.** Rewritten (not a simple import swap -- v2's model has no
`n`-based weight function at all; the correct EB weight `tau2/(tau2+sigma2_g)` requires
`sigma2_g` to be the cell-clustered bootstrap variance of whichever n-window subsample is being
evaluated, so the script now bootstraps that variance per draw). Output (`prior_plus_n_e6_fixed.log`):
coverage rises from 0.639 (n=5) to 0.743 (n=100) toward the true-groupwise ceiling of 0.756, from
a prior-only baseline of 0.595 -- the qualitative pattern the brief expects. The shrinkage weight
is NOT smoothly monotonic in n (0.685/0.590/0.500/0.680/0.860 for n=5/10/25/50/100), because it
now genuinely depends on each random subsample's own bootstrapped variance rather than n alone --
this is the correct, honest behavior (the withdrawn formula would have been artificially smooth
in n by construction). At n=5, 41/500 draws (8.2%) were skipped because the subsample landed in
only 1 of CALCE's 2 cells -- direct, independent confirmation of CALCE's small-cell-count problem.

### E7 | MINOR (pending) | config.py CALCE_DIRS / brief K8 | CX2_31 is not actually empty
**Problem:** K8 asserts "CX2_31 folder is empty." Direct check: `CX2_31/` contains 80 `.txt`
files (same Arbin-.txt format as CS2_8 and CS2_21, which `config.py`'s own comment says were
deliberately EXCLUDED from `CALCE_DIRS` specifically because their `.txt` format "doesn't match
the parser used for every other CS2/CX2 cell"). CX2_31 is still listed in `CALCE_DIRS["CX2"]`.
**Evidence:** `ls CX2_31/` → 80 files; `config.py` CS2_8/CS2_21 exclusion comment.
**Consequence:** Unclear whether `calce_loader_v2.py` has a working `.txt` path for CX2_31
specifically (producing real rows) or silently loads 0 rows for it (functionally "empty" for
the pipeline's purposes, matching the brief's intent if not its literal wording).
**Fix:** Run `verify_calce_v2.py`, check CX2_31's row count directly, read
`calce_loader_v2.py`'s per-cell format dispatch. Resolve and update K8's assertion precisely
rather than accept either claim at face value.
**Status:** **RESOLVED.** `verify_calce_v2.py`'s output confirms: `[CX2_31] NOT LOADED: No excel
files found under ...CX2_31 (searched recursively)`. The folder itself is not empty (80 `.txt`
files), but `calce_loader_v2.py` only looks for `.xlsx`/`.xls`, so CX2_31 contributes exactly 0
rows -- "empty" is correct in the sense that matters for the pipeline, just imprecise about the
mechanism. Same root cause as CS2_8/CS2_21 (deliberately excluded elsewhere in `config.py` with
an explanatory comment); CX2_31 fails the same way but silently (caught, logged, `continue`),
rather than being excluded up front. Recommend adding CX2_31 to `config.py`'s existing
CS2_8/CS2_21 exclusion comment for documentation clarity -- not a functional bug, since the
loader already handles it gracefully and the manuscript's cell count (10 CALCE cells) is
unaffected either way.

### E8 | MAJOR | baselines.py (K3 fix) | SVR/XGBoost/plain-LSTM coverage numbers differ from the brief's own stale regression anchors
**Problem:** Not a code defect -- a confirmed DRIFT finding. Freshly computed baseline conformal
coverage (e.g. SVR BIT pooled/groupwise = 21.3%/87.3%) differs from the brief's Section 3
anchors (SVR BIT = 28.3%/86.6%) and from `dataset_level_wilcoxon.py`'s old hardcoded literals,
by amounts too large to be rounding. Point-prediction MAE/RMSE for linear/SVR/XGBoost (K3's
fresh run), by contrast, match the brief's anchors EXACTLY (e.g. XGBoost 0.02405/0.10845).
**Evidence:** `outputs/baseline_conformal_results.json` vs brief Section 3 vs the old
`dataset_level_wilcoxon.py` literals.
**Consequence:** Point-prediction numbers are stable (same cache, same models); coverage numbers
are not. Since MAE/RMSE match exactly, this rules out a different train/test split or a
retrained model as the cause -- the drift is specific to the CONFORMAL CALIBRATION step.
**Likely cause:** `config.SPLIT_SEED` (K6, newly added) defaults to `config.RANDOM_SEED`
unchanged, so the split itself did not move; the most likely explanation is that conformal
calibration draws from `cache/processed.npz`'s *validation* split, and something about how
validation predictions were generated for this K3 run (fresh MC-Dropout/model.predict() calls)
differs stochastically from whatever produced the brief's anchor numbers. SVR/XGBoost use fixed
seeds and are deterministic given the same inputs, so a differing result points to the
underlying cache or checkpoint having already changed before this session began, not to
something introduced by my fix.
**Fix:** None needed in code; this is reported as a verified, current, correct result. Logged
so a reader comparing against the brief's own stated anchors is not confused by the mismatch.
**Status:** CONFIRMED (informational, not a bug to fix) -- new numbers are real and verified.

### E9 | MODERATE | xjtu_loader.py vs preprocessing.py docstring | "Known limitation" may be stale
**Problem:** `preprocessing.py`'s module docstring (KNOWN LIMITATION section) claims
`xjtu_loader.py`'s sentinel-stripping function "RE-NUMBERS the 'cycle' column sequentially
(1..N) over the REMAINING rows -- so there is no gap left ... for this function to detect."
Direct read of `xjtu_loader.py`'s `detect_and_strip_sentinel_values()` (current version) shows
it only does `df.loc[~drop_mask].reset_index(drop=True)` -- this resets the pandas ROW INDEX,
not the `cycle` COLUMN's values, which were assigned once via `np.arange(1, len+1)` before
filtering and are never recomputed. Surviving rows should therefore retain their *original*
cycle numbers, with gaps exactly where sentinel rows were removed -- the opposite of what the
docstring says (also note: the function is now named `detect_and_strip_sentinel_values`, not
`strip_calibration_blocks` as K7/the docstring call it -- a naming drift from an earlier
version, harmless by itself).
**Evidence:** Direct read, `xjtu_loader.py:102-120,136,143`.
**Consequence:** If confirmed, `windowing_bridge_sensitivity.py`'s bridge-detection (which reads
`window_cycle_indices_*` and looks for non-contiguous cycle sequences) should actually be ABLE
to detect windows bridging a removed XJTU sentinel block, contradicting the documented
"undetectable" limitation. This would mean the limitation is either already caught by the
existing bridge-sensitivity check (making the docstring's caveat obsolete) or needs re-running
to get a real number instead of an assumed one.
**Fix:** Verify directly once Stage 2's `window_cycle_indices_test` data is available (checking
for gaps in XJTU Sim_satellite cells' windows); update `preprocessing.py`'s docstring to match
reality; re-run `windowing_bridge_sensitivity.py` / `xjtu_bridging_isolation.py` against the
current loader if their own outputs predate this code state.
**Status:** **RESOLVED and escalated to MAJOR.** Ran `xjtu_bridging_isolation.py` directly
against the existing seed-42 `cache/processed.npz` (no rerun needed -- `window_cycle_indices_test`
was already cached). Result: Sim_satellite_* cells, n=376 test windows, bridging fraction =
**1.0000** (every single one); other 47 XJTU cells, n=2229, bridging fraction = **0.0000** (none).
This is the OPPOSITE of what `preprocessing.py`'s docstring predicted ("renumbering erases the
gap" -> expected near-zero bridging for satellite cells) and CONFIRMS E9's code-level reading:
`detect_and_strip_sentinel_values()` does not renumber the `cycle` column, so surviving rows keep
their original (now-gapped) cycle numbers, and every 20-cycle window drawn from a Sim_satellite
cell that straddles a removed sentinel block is detectably non-contiguous.
376/2605 = 14.43% exactly matches the previously-reported "overall XJTU bridging" figure -- this
was never diffuse native sparsity (unlike CALCE/NASA/Oxford, confirmed separately by
`diagnose_bridging_anomaly.py`); it is 100% concentrated in the 8 Sim_satellite cells, each of
which bridges on every test window. `preprocessing.py`'s "KNOWN LIMITATION" docstring is factually
wrong about the mechanism and must be corrected.
**Consequence for K7/manuscript:** all 376 Sim_satellite test windows (14.4% of XJTU's 2,605-
window test set) are trained/evaluated as if they span 20 contiguous cycles when they actually
span more real elapsed cycling than that, due to the removed calibration/sentinel rows. XJTU's
reported coverage/MAE are not wrong, but they are not cleanly interpretable as "20-cycle-ahead
prediction" for this subset. Needs a sensitivity run (K7): recompute XJTU coverage/MAE excluding
these 376 windows, see whether XJTU's headline numbers change materially, and state the result
explicitly in the manuscript rather than assert "no limitation" or assert "near-zero bridging."
**Sensitivity result** (`xjtu_bridging_sensitivity.py`, run against existing
`outputs/conformal_groupwise_predictions.npz`, no retrain): bridging windows (n=376) have RMSE(z)
= 0.4318 vs. 0.0769 for clean windows (n=2229) -- **5.6x higher**, consistent with the model
seeing a larger real capacity drop than its 20-cycle input implies. Pooled-XJTU MAE(z) shifts from
0.0459 (all windows) to 0.0421 (clean only) if excluded, a modest but real +8.3% relative
reduction. Coverage is NOT degraded by the bridging windows (94.7% bridging vs. 92.7% clean --
both near/above the 90% target), so split-conformal validity is not broken by this issue, but
point-prediction accuracy on this 14.4%-of-XJTU subset is clearly worse and must be disclosed.
**Status: RESOLVED, with numbers.** `preprocessing.py`'s docstring corrected to state the real
mechanism (no re-numbering; gap survives; verified 100%/0% bridging split). Manuscript must state:
(a) the mechanism correctly, (b) that bridging is 100% concentrated in 8 Sim_satellite_* cells
(376/2605 windows, matching the dataset's own calibration-sweep frequency, not diffuse native
sparsity like CALCE/NASA/Oxford), (c) the RMSE elevation and modest pooled-MAE shift above, (d)
that conformal coverage validity is unaffected. No loader fix (preserving original cycle numbers
was already happening; a FURTHER fix would mean excluding or re-flagging these specific windows
at the windowing stage) is required to report this honestly -- excluding them outright would
change XJTU's reported n and is a judgment call for the paper's primary-vs-sensitivity-analysis
split, not something to do silently here.

### E10 | MINOR (performance only, K10-adjacent) | leave_one_out.py run_all_lodo() | Raw data reloaded 7x unnecessarily
**Problem:** `run_all_lodo()` called `run_lodo()` for each of the 7 held-out datasets, and each
`run_lodo()` called `build_lodo_cache()`, which called `preprocessing.load_all_raw()` fresh --
re-parsing ALL 7 raw datasets (CALCE's 366+ files, RWTH's ~1,600 CSVs, etc.) from scratch 7
times, even though the raw data is identical across all 7 runs (only which dataset is held out
changes).
**Evidence:** Direct read; empirically, the first (pre-fix) LODO run spent its first ~45 minutes
still inside held-out-config #1's BIT-folder parse, with no model training yet.
**Consequence:** Made the K1 verification retrain impractically slow, not incorrect.
**Fix:** `build_lodo_cache()` now accepts an optional pre-loaded `raw_df`; `run_all_lodo()` loads
and SOH-processes raw data ONCE and passes it to all 7 `run_lodo()` calls. Purely a performance
change -- `load_all_raw()`/`add_soh()` are deterministic (no randomness), so the reused dataframe
is byte-identical to what 7 separate calls would have produced; only the held-out-dataset
filtering inside `build_lodo_cache()` (unchanged) differs per run.
**Status:** RESOLVED. Original (slow, unoptimized) run killed before any iteration completed --
no partial results lost. Optimized version relaunched.

### E11 | MINOR (performance only, K6-enabling) | preprocessing.py run() | Same 7x-reload risk for the K6 repeated-split experiment
**Problem:** `preprocessing.run()` always called `load_all_raw()` + `add_soh()` internally. K6
requires building 4 additional split-seed caches (43/44/45/46) end to end; calling `run()` once
per seed would re-parse all 7 raw datasets 4 more times, the same unnecessary cost E10 fixed for
LODO.
**Evidence:** Direct read, `preprocessing.py` `run()` signature before this fix.
**Consequence:** Would make the K6 repeated-split run impractically slow, not incorrect.
**Fix:** `run(window=None, split_seed=None, raw_df=None)` now accepts an optional pre-loaded,
pre-SOH'd dataframe; when given, skips its own `load_all_raw()`/`add_soh()` call and uses the
caller's dataframe instead. `repeated_split_experiment.py` (new) loads raw data once and passes
it to all 4 new-seed `run()` calls. Purely a performance change -- deterministic functions, byte-
identical result to calling `run()` standalone per seed.
**Status:** RESOLVED. Syntax-checked (`ast.parse`). Not yet exercised end-to-end (blocked behind
LODO finishing, to avoid GPU contention between two simultaneous training jobs).

### E13 | MAJOR | predictor_consistent_recalibration.py:50-54,81 | Per-dataset coverage CI treated windows as independent trials (Stage 9)
**Problem:** `clopper_pearson(k, n)` computed a window-level binomial CI for each dataset's
groupwise coverage, treating all n test WINDOWS as independent Bernoulli trials. This is exactly
the violation the brief's own rules warn against ("windows from the same cell are not
independent") -- for datasets with few test cells (XJTU, CALCE, and especially Oxford/NASA with
1-2 test cells per the brief's own anchors), a window-level CI is overconfident because it treats
hundreds of correlated within-cell windows as if they were hundreds of independent draws.
**Evidence:** Direct read, confirmed no other script consumed this field (so the bug was
self-contained, not yet propagated), then ran both CIs side by side after the fix:
```
Dataset   n_test  groupwise   window CP CI    cell-boot CI
XJTU        2605      93.0%  [ 92.0,  94.0]  [ 84.5,  99.3]   <- window CI is ~5x narrower
CALCE       2743      75.6%  [ 74.0,  77.2]  [ 70.0,  90.6]   <- window CI is ~4x narrower
MIT        17513      92.2%  [ 91.8,  92.6]  [ 90.9,  93.2]   <- close (many cells, less severe)
Oxford        56     100.0%  [ 93.6, 100.0]  n/a (<2 cells)   <- cell-level CI correctly undefined
NASA           3      66.7%  [  9.4,  99.2]  n/a (<2 cells)   <- cell-level CI correctly undefined
```
**Consequence:** If this script's `cp95` field had been pulled into the manuscript as the
reported per-dataset coverage uncertainty, XJTU and CALCE's true uncertainty would have been
understated by roughly 4-5x -- a reader would see a deceptively tight interval around 93%/76%
coverage when the cell-level truth is a much wider, less conclusive range.
**Fix:** Reused the already-verified `cell_clustered_bootstrap_coverage()` (from
`cell_clustered_bootstrap.py`, K6) to compute a second, cell-valid 95% CI alongside the window-
level one. Renamed the old field to `cp95_window_level_see_docstring_caveat` and added
`cell_bootstrap_95ci` (`null` when a dataset has fewer than 2 test cells, matching
`cell_clustered_bootstrap.py`'s own convention for Oxford/NASA). `clopper_pearson()`'s docstring
now states explicitly not to use it alone as the reported uncertainty.
**Status:** RESOLVED. Syntax-checked and run successfully; output above. No retrain needed (pure
post-hoc recomputation from already-saved predictions).

### E14 | MAJOR (finding, not a code defect) | K4 nested hyperparameter search | HGC-CP has no single (lam, beta) that generalizes across targets
**Problem:** K4 required re-selecting HGC-CP's (lambda, beta) via a genuine nested/held-out
protocol (`run_nested_hp_search_all.py`, rotating the other 6 datasets as pseudo-targets, never
touching the real target's test set during selection) rather than the brief's anchor (lambda=3.0,
beta=0.0), which was selected on the pre-fix pipeline using direct test-set feedback.
**Result (all 7 datasets as real target, in turn):**
```
Target    selected (lam, beta)   resulting test coverage (target 90%)
MIT       lam=5.0, beta=1.0      100.0%   (massively over-covers)
RWTH      lam=3.0, beta=1.0       99.6%   (over-covers)
XJTU      lam=3.0, beta=1.0       96.7%   (mildly over-covers)
BIT       lam=3.0, beta=1.0       48.2%   (badly UNDER-covers)
CALCE     lam=3.0, beta=0.5       85.1%   (close to target)
Oxford    lam=3.0, beta=1.0      100.0%   (massively over-covers)
NASA      lam=3.0, beta=0.5      100.0%   (massively over-covers, n=3 test windows)
```
**Consequence:** (1) NONE of the 7 nested-selected settings match the brief's anchor beta=0.0 --
every single one prefers beta>=0.5, confirming the anchor was indeed chosen via test-set feedback
on the pre-fix pipeline, not a held-out procedure (as K4 already suspected). (2) No single (lam,
beta) compromise generalizes well: optimizing "mean |coverage-90%| across 6 pseudo-target
rotations" produces settings that push several datasets to ~100% coverage (uninformatively wide
intervals) while BIT still badly under-covers (48.2%) under its own nested-selected setting. This
is a real limitation of HGC-CP as specified (the MMD-softmax weighting + linear heterogeneity
inflation), not a bug in the search code -- the search faithfully found the best available
compromise, and that compromise is still poor for at least one target every time.
**Fix:** None applicable -- this is the honest empirical result K4 asked for, not a defect to
patch. The manuscript MUST report: (a) HGC-CP hyperparameters are exploratory, selected via a
pseudo-target nested rotation that approximates but does not replace true held-out validation
(only 6 "folds" available, each itself only 1 dataset); (b) the resulting coverage is highly
target-dependent and NOT uniformly close to the 90% nominal target even under this honest
protocol; (c) do not claim HGC-CP achieves reliable 90% coverage across arbitrary target domains
-- the BIT result alone (nested-selected setting still produces 48.2% coverage) refutes that
claim directly.
**Status:** RESOLVED (K4 complete). `outputs/nested_hp_search_results.json` has full grid results
for all 7 targets x 20 (lam,beta) combinations.

### E15 | CRITICAL (infrastructure, not a code/science defect) | C: drive filled to 20MB free -- background git gc failing silently for hours
**Problem:** A background `git pack-objects` process (misdiagnosed earlier in this session as
"benign gc housekeeping") had actually been failing repeatedly with `fatal: ... Out of diskspace`
during a "geometric repack," for a long period, while C: drive filled to just 20MB free. Root
cause: the earlier (pre-`.gitignore`) aborted `git add -A` attempt (staging `.venv/` and all raw
datasets before being caught) had written ~24,000 loose blob objects into `.git/objects` -- these
were never part of any commit (confirmed: `git log` only ever contained `.py`/`.md` files), but
`git reset` does not delete already-written loose objects, so ~5 GB of unreachable garbage sat in
`.git/objects` from early in the session onward. git's automatic gc kept trying to repack this
into a single pack file and kept running out of temp space to do it, leaving a 1.68 GB failed
`tmp_pack_wgs2EV` temp file behind each time, compounding the problem.
**Evidence:** `bxnsjjrb9` task notification surfaced the literal `Out of diskspace` error.
`Get-PSDrive C` showed 0.02 GB free. `git fsck --unreachable` / `git prune --dry-run --expire=now`
confirmed 23,843 of ~24,145 loose objects were unreachable blobs from the old aborted add.
**Consequence:** Real risk -- both Stage 2 verification and the K1-fixed LODO retrain were running
in the background throughout, and either one's cache/checkpoint/log WRITE could have failed or
been silently truncated at any point while free space was near zero. Checked their log tails
immediately after discovery: both showed healthy, sensible in-progress output with no visible
write errors, but this was fortunate timing, not something the pipeline itself guarded against.
**Fix:** (1) Deleted the failed `tmp_pack_wgs2EV` temp file directly (1.68 GB, safe -- an
incomplete temp artifact, not committed history) for immediate headroom. (2) Ran
`git prune --expire=now` (not a full `git gc`, which would need to build a new pack file and
risked failing again at low free space) to directly unlink the ~23,843 confirmed-unreachable loose
objects -- no new pack file needed, minimal temp space required. (3) Verified with `git fsck --full`
(clean, no errors) and `git log --oneline` (all prior commits intact) that history was never
touched -- only genuinely unreachable garbage was removed. `.git` dropped from 6.62 GB to 0.04 GB;
C: free space recovered from 0.02 GB to 7.02 GB.
**Status:** RESOLVED. No data, commits, or in-progress background job output were lost. Going
forward: do not assume a long-running git background process is benign gc without checking its
actual exit status/output when it runs unusually long, and periodically check free disk space
during long multi-job sessions rather than only when something fails.

---

*(IDs continue as Stage 1 proceeds. Severity: CRITICAL/MAJOR/MODERATE/MINOR.)*
