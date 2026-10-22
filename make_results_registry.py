"""
Single machine-generated source of every number in the manuscript.
Reads the per-experiment result files in outputs/ and compiles them into
results_registry.json, then emits paper/numbers.tex
(one \\newcommand per number, so main.tex NEVER has a hand-typed number --
every value in the manuscript must come from \\num{...} macros defined here).

Tolerant of missing files (some LODO configs may still be training, or a
later-stage script may not have been run yet) -- each section is wrapped so
a missing input produces an explicit "UNVERIFIED" placeholder rather than a
crash or a silently wrong number: when something can't be verified the
registry says so, and never fabricates or quietly omits a value.

Run:  python make_results_registry.py
Out:  results_registry.json, paper/numbers.tex
"""
import glob
import json
import os
import re

import numpy as np

import config

OUT_DIR = config.OUTPUT_DIR
UNVERIFIED = "UNVERIFIED"


def load(name):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def data_counts(calce_cells):
    """Window and cell counts per split, and the number of raw CALCE workbooks.

    These need cache/processed.npz and the raw CALCE folders. When both are present the counts are
    computed and saved to outputs/data_counts.json; otherwise (e.g. a clone without the data) the
    saved file is read, so the registry can still be rebuilt."""
    cache_path = os.path.join(config.CACHE_DIR, "processed.npz")
    cell_dirs = [os.path.join(config.ROOT_DIR, c) for c in calce_cells]
    if not (os.path.exists(cache_path) and cell_dirs and all(os.path.isdir(d) for d in cell_dirs)):
        return load("data_counts.json")
    cache = np.load(cache_path, allow_pickle=True)
    datasets = ("MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA")
    counts = {
        "train_windows": int(cache["X_train"].shape[0]),
        "val_windows": int(cache["X_val"].shape[0]),
        "test_cells": {d: len(set(cache["cell_test"][cache["lab_test"] == d])) for d in datasets},
        "test_windows": {d: int((cache["lab_test"] == d).sum()) for d in datasets},
        "calce_raw_files": sum(f.lower().endswith(".xlsx")
                               for d in cell_dirs for _, _, files in os.walk(d) for f in files),
    }
    with open(os.path.join(OUT_DIR, "data_counts.json"), "w") as fh:
        json.dump(counts, fh, indent=2)
    return counts


def export_figure_inputs():
    """Save the two things the prediction figures need from cache/processed.npz: each dataset's
    SOH normalization (to plot real SOH instead of z-units) and the cycle number at the end of
    every test window's input. Written to outputs/figure_inputs.npz, which is tracked, so the
    figures can be redrawn without the cache. Test windows are in the same order as in
    outputs/conformal_groupwise_predictions.npz."""
    cache_path = os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(cache_path):
        return
    cache = np.load(cache_path, allow_pickle=True)
    soh_mean, soh_std = cache["soh_mean"].item(), cache["soh_std"].item()
    datasets = np.array(sorted(soh_mean))
    np.savez_compressed(os.path.join(OUT_DIR, "figure_inputs.npz"), datasets=datasets,
                        soh_mean=np.array([soh_mean[d] for d in datasets]),
                        soh_std=np.array([soh_std[d] for d in datasets]),
                        test_end_cycle=cache["window_cycle_indices_test"][:, -1].astype(np.float32))


def main():
    registry = {"generated_from": "make_results_registry.py", "sections": {}}
    export_figure_inputs()

    # ---- Main model (pooled groupwise conformal, seed 42) ----
    r = load("conformal_groupwise_results.json")
    registry["sections"]["main_model"] = r if r else UNVERIFIED

    # MPIW (mean prediction interval width) isn't saved by train_one()'s
    # results JSON -- compute it directly from the saved predictions file so
    # it can be compared fairly against CQR's MPIW (same z-units, same test
    # set) without needing a model re-run.
    gw_pred_path = os.path.join(OUT_DIR, "conformal_groupwise_predictions.npz")
    if os.path.exists(gw_pred_path):
        gw_pred = np.load(gw_pred_path, allow_pickle=True)
        registry["sections"]["main_model_mpiw"] = float(np.mean(gw_pred["upper"] - gw_pred["lower"]))
    else:
        registry["sections"]["main_model_mpiw"] = UNVERIFIED

    r_pooled = load("conformal_pooled_results.json")
    r_normgw = load("conformal_normalized_groupwise_results.json")
    registry["sections"]["conformal_mode_comparison"] = {
        "groupwise": r, "pooled": r_pooled, "normalized_groupwise": r_normgw,
    }

    # ---- Baselines ----
    registry["sections"]["baselines_point_prediction"] = {
        name: load(f"baseline_{name}_results.json") or UNVERIFIED
        for name in ("linear", "svr", "xgboost")
    }
    registry["sections"]["baseline_conformal_coverage"] = load("baseline_conformal_results.json") or UNVERIFIED

    # ---- Ablations ----
    registry["sections"]["ablations"] = {
        name: load(f"ablation_{name}_results.json") or UNVERIFIED
        for name in ("full", "no_attention", "no_multiscale", "plain_lstm", "gru", "transformer")
    }

    # ---- Window size ----
    registry["sections"]["window_size"] = {
        w: load(f"window_{w}_results.json") or UNVERIFIED for w in (10, 20, 30, 50)
    }

    # ---- Multi-seed ----
    multiseed = {s: load(f"multiseed_{s}_results.json") for s in (42, 123, 2024)}
    registry["sections"]["multiseed"] = {str(s): (v or UNVERIFIED) for s, v in multiseed.items()}
    maes = [v["test_mae"] for v in multiseed.values() if v]
    covs = [v["overall_coverage"] for v in multiseed.values() if v]
    if len(maes) == 3:
        import statistics
        registry["sections"]["multiseed_summary"] = {
            "mae_mean": statistics.mean(maes), "mae_std": statistics.pstdev(maes),
            "coverage_mean": statistics.mean(covs), "coverage_std": statistics.pstdev(covs),
        }
    else:
        registry["sections"]["multiseed_summary"] = UNVERIFIED

    # ---- Persistence baseline ----
    registry["sections"]["persistence"] = load("persistence_baselines.json") or UNVERIFIED

    # ---- Hierarchical calibration (v2, canonical) ----
    registry["sections"]["hierarchical_calibration"] = load("hierarchical_calibration_v2.json") or UNVERIFIED

    # ---- HGC-CP nested hyperparameter search ----
    registry["sections"]["hgc_cp_nested_search"] = load("nested_hp_search_results.json") or UNVERIFIED

    # ---- MWHC: MMD-weighted hierarchical calibration (new method) ----
    registry["sections"]["mwhc"] = load("mwhc_results.json") or UNVERIFIED

    # ---- Calibration-cell-budget sensitivity (new experiment) ----
    registry["sections"]["calibration_size_sensitivity"] = load("calibration_size_sensitivity.json") or UNVERIFIED

    # ---- CQR: Conformalized Quantile Regression baseline (new comparison) ----
    registry["sections"]["cqr_pooled"] = load("cqr_pooled_results.json") or UNVERIFIED
    registry["sections"]["cqr_groupwise"] = load("cqr_groupwise_results.json") or UNVERIFIED

    # ---- LODO ----
    # Only result files that record "calibration_predictor" were written by the current
    # leave_one_out.py (same predictor for calibration and test); any other file is treated
    # as missing.
    lodo = {}
    for d in ("NASA", "CALCE", "Oxford", "MIT", "BIT", "XJTU", "RWTH"):
        v = load(f"lodo_{d}_results.json")
        if v and "calibration_predictor" in v:
            lodo[d] = v
        else:
            lodo[d] = UNVERIFIED  # missing, or written by older code
    registry["sections"]["lodo"] = lodo
    registry["sections"]["lodo_complete"] = all(v != UNVERIFIED for v in lodo.values())

    # ---- Predictor-consistent recalibration (per-dataset coverage, window- and cell-level CIs) ----
    registry["sections"]["predictor_consistent_coverage"] = load("predictor_consistent_recalibration.json") or UNVERIFIED

    # ---- Cell-clustered bootstrap / XJTU bridging / RWTH verification ----
    xjtu_bridge = load("xjtu_bridging_isolation.json")
    registry["sections"]["xjtu_bridging_isolation"] = xjtu_bridge or UNVERIFIED
    registry["sections"]["xjtu_bridging_sensitivity"] = load("xjtu_bridging_sensitivity.json") or UNVERIFIED
    registry["sections"]["rwth_verification"] = load("rwth_verification_v2.json") or UNVERIFIED

    # ---- Repeated cell-level splits (between-split variability) ----
    registry["sections"]["repeated_splits"] = load("repeated_split_summary.json") or UNVERIFIED

    # ---- Data-layer verification (tests/check_data_layer.py): cell and window counts ----
    data_layer_path = "data_layer_report.json"
    data_layer = None
    if os.path.exists(data_layer_path):
        with open(data_layer_path) as fh:
            data_layer = json.load(fh)
    registry["sections"]["data_layer"] = data_layer or UNVERIFIED

    # ---- Dataset-level Wilcoxon, pooled vs groupwise (dataset_level_wilcoxon.py) ----
    registry["sections"]["wilcoxon"] = load("dataset_level_wilcoxon.json") or UNVERIFIED

    # ---- Data-quality corrections (verify_rwth_v2.py -> JSON; verify_calce_v2.py -> log) ----
    rwth_v = load("rwth_verification_v2.json")
    registry["sections"]["rwth_corrections"] = rwth_v["summary"] | {
        "rows_raw": rwth_v["neither correction (F, F)"]["n_rows"]} if rwth_v else UNVERIFIED
    calce_log = os.path.join("logs", "verify_calce_v2.log")
    if os.path.exists(calce_log):
        with open(calce_log, encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        table = txt[txt.rindex("  cell  rows"):].splitlines()[1:]
        rows = [ln.split() for ln in table if re.match(r"^C[SX]2_\d+\s", ln)]
        # The log loads every cell twice (flag and drop policies), so each cumulative-counter
        # message appears twice per file.
        registry["sections"]["calce_corrections"] = {
            "n_cells": len(rows),
            "total_rows": int(re.search(r"total rows (\d+)", txt).group(1)),
            "rows_after_drop": sum(int(r[7]) for r in rows),
            "soh_keep_pct_min": min(float(r[2].rstrip("%")) for r in rows),
            "soh_keep_pct_max": max(float(r[2].rstrip("%")) for r in rows),
            "median_step_min": min(float(r[3]) for r in rows),
            "median_step_max": max(float(r[3]) for r in rows),
            "cumulative_files": txt.count("behaves as a cumulative counter") // 2,
            "cells": [r[0] for r in rows],
        }
    else:
        registry["sections"]["calce_corrections"] = UNVERIFIED

    # ---- Splits whose NASA validation cell yields no window (repeated_split_experiment.py logs) ----
    # conformal.py prints a warning and falls back to a pooled quantile when a dataset has no
    # validation residuals; count the splits whose groupwise run printed it for NASA.
    split_logs = sorted(glob.glob(os.path.join("logs", "repeated_split_run*.log")))
    if split_logs:
        fallback_splits = set()
        for path in split_logs:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
            runs = re.split(r"run_name=split(\d+)_", txt)
            for seed, chunk in zip(runs[1::2], runs[2::2]):
                if chunk.startswith("groupwise") and "no calibration data for ['NASA']" in chunk:
                    fallback_splits.add(int(seed))
        registry["sections"]["nasa_fallback_splits"] = sorted(fallback_splits)
    else:
        registry["sections"]["nasa_fallback_splits"] = UNVERIFIED

    # ---- Dataset counts that need the cache or the raw files ----
    calce_cells = (registry["sections"]["calce_corrections"]["cells"]
                   if registry["sections"]["calce_corrections"] != UNVERIFIED else [])
    registry["sections"]["data_counts"] = data_counts(calce_cells) or UNVERIFIED

    # ---- Deterministic pass vs MC-Dropout mean as interval centre (deterministic_predictor_comparison.py) ----
    registry["sections"]["deterministic_predictor"] = load("deterministic_predictor_results.json") or UNVERIFIED

    # ---- Deployment cost (deployability_benchmark.py) ----
    registry["sections"]["deployability"] = load("deployability_results.json") or UNVERIFIED

    # ---- Emit paper/numbers.tex (and the same values into results_registry.json) ----
    os.makedirs("paper", exist_ok=True)
    lines = ["% AUTO-GENERATED by make_results_registry.py -- do not hand-edit.",
             "% Every number in main.tex must come from a \\NUM{...} macro defined here.", ""]

    seen_macro_names = set()
    # Formatted exactly as printed in the manuscript; the public figure scripts read these.
    registry["values"] = {}

    def macro(name, value, fmt="{:.4f}"):
        safe_name = re.sub(r"[^A-Za-z]", "", name)  # LaTeX macro names: letters only
        # Two DIFFERENT inputs (e.g. "Window10MAE" / "Window20MAE") can sanitize to the SAME
        # macro name once digits are stripped, silently overwriting one value with another
        # in the .tex output, so a repeated name raises an error.
        if safe_name in seen_macro_names:
            raise ValueError(f"macro name collision: '{name}' sanitizes to '\\{safe_name}', "
                              f"which was already defined by a previous macro() call. Rename "
                              f"one of the inputs so they don't collide after non-letter "
                              f"characters are stripped.")
        seen_macro_names.add(safe_name)
        if value is None or value == UNVERIFIED:
            text = UNVERIFIED
        elif isinstance(value, (int, float)):
            text = fmt.format(value)
        else:
            text = str(value)
        lines.append(f"\\newcommand{{\\{safe_name}}}{{{text}}}")
        registry["values"][safe_name] = text

    if r:
        macro("MainModelMAE", r["test_mae"], "{:.5f}")
        macro("MainModelRMSE", r["test_rmse"], "{:.5f}")
        macro("MainModelCoverage", r["overall_coverage"] * 100, "{:.1f}")
        for d, v in r["per_dataset_coverage"].items():
            macro(f"Coverage{d}", v * 100, "{:.1f}")
    else:
        macro("MainModelMAE", None)
        macro("MainModelRMSE", None)
        macro("MainModelCoverage", None)
    macro("MainModelMPIW", registry["sections"]["main_model_mpiw"], "{:.5f}")

    # Normalized-groupwise vs groupwise: same model and test set, so coverage and width are
    # directly comparable. "Five" = the datasets where every calibration strategy is defined.
    five = ("MIT", "RWTH", "XJTU", "BIT", "CALCE")
    ng_pred_path = os.path.join(OUT_DIR, "conformal_normalized_groupwise_predictions.npz")
    if r and r_normgw and os.path.exists(ng_pred_path) and registry["sections"]["main_model_mpiw"] != UNVERIFIED:
        ng_pred = np.load(ng_pred_path, allow_pickle=True)
        ng_mpiw = float(np.mean(ng_pred["upper"] - ng_pred["lower"]))
        macro("NormGroupwiseCoverage", r_normgw["overall_coverage"] * 100, "{:.1f}")
        for d, v in r_normgw["per_dataset_coverage"].items():
            macro(f"NormGroupwiseCoverage{d}", v * 100, "{:.1f}")
        macro("NormGroupwiseMPIW", ng_mpiw, "{:.5f}")
        macro("NormGroupwiseWidthIncreasePercent",
              100 * (ng_mpiw / registry["sections"]["main_model_mpiw"] - 1), "{:.0f}")
        macro("NormGroupwiseMeanDevFive",
              float(np.mean([abs(r_normgw["per_dataset_coverage"][d] - 0.90) for d in five])) * 100, "{:.2f}")
        macro("GroupwiseMeanDevFive",
              float(np.mean([abs(r["per_dataset_coverage"][d] - 0.90) for d in five])) * 100, "{:.2f}")
    else:
        for name in ("NormGroupwiseCoverage", "NormGroupwiseMPIW", "NormGroupwiseWidthIncreasePercent",
                     "NormGroupwiseMeanDevFive", "GroupwiseMeanDevFive"):
            macro(name, None)

    if r and r_pooled:
        pooled_pd = r_pooled["per_dataset_coverage"]
        groupwise_pd = r["per_dataset_coverage"]
        for d, v in pooled_pd.items():
            macro(f"PooledCoverage{d}", v * 100, "{:.1f}")
        pooled_vals_pd = [v * 100 for v in pooled_pd.values()]
        macro("PooledCoverageMin", min(pooled_vals_pd), "{:.1f}")
        macro("PooledCoverageMax", max(pooled_vals_pd), "{:.1f}")
        # worst-case groupwise deviation from the 90% nominal target, excluding NASA
        # (NASA's n=3 test windows make its coverage estimate too noisy to be meaningful --
        # see the per-dataset coverage discussion -- so it is excluded from this "how far off
        # does groupwise get, at worst" claim rather than silently dominating it)
        devs_excl_nasa = {d: abs(v * 100 - 90) for d, v in groupwise_pd.items() if d != "NASA"}
        worst_d = max(devs_excl_nasa, key=devs_excl_nasa.get)
        macro("GroupwiseMaxDevExclNASA", devs_excl_nasa[worst_d], "{:.1f}")
        print(f"groupwise max deviation from 90% nominal (excl. NASA): {worst_d} "
              f"({devs_excl_nasa[worst_d]:.1f} points)")

    # Width cost of groupwise over pooled calibration: same predictor and test windows, so the
    # mean interval widths are directly comparable.
    pooled_pred_path = os.path.join(OUT_DIR, "conformal_pooled_predictions.npz")
    if os.path.exists(pooled_pred_path) and registry["sections"]["main_model_mpiw"] != UNVERIFIED:
        pooled_pred = np.load(pooled_pred_path, allow_pickle=True)
        pooled_mpiw = float(np.mean(pooled_pred["upper"] - pooled_pred["lower"]))
        macro("PooledMPIW", pooled_mpiw, "{:.5f}")
        macro("GroupwiseWidthOverPooledPercent",
              100 * (registry["sections"]["main_model_mpiw"] / pooled_mpiw - 1), "{:.0f}")
        # Per dataset: the groupwise quantile actually used for the test intervals (half the
        # interval width), relative to the pooled one, and the 90th percentile of the test errors
        # relative to that groupwise quantile (above 1 means the calibration quantile is too small
        # for the test errors, i.e. under-coverage). Same predictions as the coverage figures.
        gw_pred = np.load(os.path.join(OUT_DIR, "conformal_groupwise_predictions.npz"), allow_pickle=True)
        q_pooled = float(np.mean(pooled_pred["upper"] - pooled_pred["lower"])) / 2
        for d in sorted(set(gw_pred["lab_test"])):
            m = gw_pred["lab_test"] == d
            q_d = float(np.mean(gw_pred["upper"][m] - gw_pred["lower"][m])) / 2
            test_q90 = float(np.quantile(np.abs(gw_pred["y_test"][m] - gw_pred["mean_pred"][m]), 0.9))
            macro(f"QuantileOverPooled{d}", q_d / q_pooled, "{:.1f}")
            macro(f"TestErrorQNinetyOverQuantile{d}", test_q90 / q_d, "{:.2f}")

        # CALCE's under-coverage, cell by cell: its two test cells against its validation cells.
        # The lower-coverage test cell is named in the text, so its name is a macro too.
        val_pred = np.load(os.path.join(OUT_DIR, "conformal_groupwise_predictions_val.npz"), allow_pickle=True)
        # Why the pooled quantile tracks MIT and RWTH: their share of the validation windows.
        macro("ValWindowShareMitRwth",
              100 * float(np.isin(val_pred["lab_val"], ["MIT", "RWTH"]).mean()), "{:.0f}")
        vm = val_pred["lab_val"] == "CALCE"
        val_q90 = float(np.quantile(np.abs(val_pred["y_val"][vm] - val_pred["mean_pred"][vm]), 0.9))
        cells = {}
        for c in sorted(set(gw_pred["cell_test"][gw_pred["lab_test"] == "CALCE"])):
            m = gw_pred["cell_test"] == c
            err = np.abs(gw_pred["y_test"][m] - gw_pred["mean_pred"][m])
            hit = (gw_pred["y_test"][m] >= gw_pred["lower"][m]) & (gw_pred["y_test"][m] <= gw_pred["upper"][m])
            cells[c] = (float(hit.mean()), float(np.quantile(err, 0.9)) / val_q90)
        low, high = sorted(cells, key=lambda c: cells[c][0])
        tex = lambda s: s.replace("_", r"\_")  # noqa: E731
        macro("CalceLowCell", tex(low))
        macro("CalceLowCellCoverage", cells[low][0] * 100, "{:.1f}")
        macro("CalceLowCellErrorRatio", cells[low][1], "{:.1f}")
        macro("CalceHighCell", tex(high))
        macro("CalceHighCellCoverage", cells[high][0] * 100, "{:.1f}")
        macro("CalceHighCellErrorRatio", cells[high][1], "{:.1f}")
        macro("CalceValCells", ", ".join(tex(c) for c in sorted(set(val_pred["cell_val"][vm]))))
    else:
        macro("PooledMPIW", None)
        macro("GroupwiseWidthOverPooledPercent", None)

    for name, v in registry["sections"]["baselines_point_prediction"].items():
        if v != UNVERIFIED:
            macro(f"Baseline{name.capitalize()}MAE", v["test_mae"], "{:.5f}")
            macro(f"Baseline{name.capitalize()}RMSE", v["test_rmse"], "{:.5f}")

    for name, v in registry["sections"]["ablations"].items():
        if v != UNVERIFIED:
            tag = "".join(p.capitalize() for p in name.split("_"))
            macro(f"Ablation{tag}MAE", v["test_mae"], "{:.5f}")
            macro(f"Ablation{tag}RMSE", v["test_rmse"], "{:.5f}")
            macro(f"Ablation{tag}Coverage", v["overall_coverage"] * 100, "{:.1f}")

    baseline_conformal = registry["sections"]["baseline_conformal_coverage"]
    if baseline_conformal != UNVERIFIED:
        for name, v in baseline_conformal.items():
            tag = "".join(p.capitalize() for p in name.split("_"))
            macro(f"BaselineConformal{tag}Pooled", v["pooled_overall_coverage"] * 100, "{:.1f}")
            macro(f"BaselineConformal{tag}Groupwise", v["groupwise_overall_coverage"] * 100, "{:.1f}")
            macro(f"BaselineConformal{tag}BitPooled", v["pooled_per_dataset_coverage"]["BIT"] * 100, "{:.1f}")
            macro(f"BaselineConformal{tag}BitGroupwise", v["groupwise_per_dataset_coverage"]["BIT"] * 100, "{:.1f}")
        gw_vals = [v["groupwise_overall_coverage"] * 100 for v in baseline_conformal.values()]
        pooled_vals = [v["pooled_overall_coverage"] * 100 for v in baseline_conformal.values()]
        bit_pooled_vals = [v["pooled_per_dataset_coverage"]["BIT"] * 100 for v in baseline_conformal.values()]
        bit_gw_vals = [v["groupwise_per_dataset_coverage"]["BIT"] * 100 for v in baseline_conformal.values()]
        macro("BaselineConformalGroupwiseMin", min(gw_vals), "{:.1f}")
        macro("BaselineConformalGroupwiseMax", max(gw_vals), "{:.1f}")
        macro("BaselineConformalPooledMin", min(pooled_vals), "{:.1f}")
        macro("BaselineConformalPooledMax", max(pooled_vals), "{:.1f}")
        macro("BaselineConformalBitPooledMin", min(bit_pooled_vals), "{:.1f}")
        macro("BaselineConformalBitPooledMax", max(bit_pooled_vals), "{:.1f}")
        macro("BaselineConformalBitGroupwiseMin", min(bit_gw_vals), "{:.1f}")
        macro("BaselineConformalBitGroupwiseMax", max(bit_gw_vals), "{:.1f}")

    # NOTE: macro() strips all non-letter characters from the name, so a
    # purely-numeric differentiator (e.g. "Window10MAE" vs "Window20MAE")
    # would collapse to the SAME macro name ("WindowMAE") and silently
    # overwrite, so the window size is spelled out in words.
    window_words = {10: "Ten", 20: "Twenty", 30: "Thirty", 50: "Fifty"}
    for w, v in registry["sections"]["window_size"].items():
        if v != UNVERIFIED:
            word = window_words.get(w)
            if word is None:
                raise ValueError(f"window_size key {w!r} has no word mapping -- add one, "
                                  f"don't let it silently fall back to a digit-stripped name")
            macro(f"Window{word}MAE", v["test_mae"], "{:.5f}")
            macro(f"Window{word}RMSE", v["test_rmse"], "{:.5f}")

    persistence = registry["sections"]["persistence"]
    if persistence != UNVERIFIED:
        for d, v in persistence["per_dataset"].items():
            beats = v["model"]["mae_real"] < v["persistence"]["mae_real"]
            macro(f"ModelBeatsPersistence{d}", "yes" if beats else "no")
            macro(f"Persistence{d}MAE", v["persistence"]["mae_real"], "{:.5f}")
            macro(f"Model{d}MAEReal", v["model"]["mae_real"], "{:.5f}")
        pooled = persistence["pooled"]
        macro("PersistencePooledModelMAEReal", pooled["model"]["mae_real"], "{:.5f}")
        macro("PersistencePooledBaselineMAEReal", pooled["persistence"]["mae_real"], "{:.5f}")

    hier = registry["sections"]["hierarchical_calibration"]
    if hier != UNVERIFIED:
        macro("HierMuHat", hier["global"]["mu_hat"], "{:.5f}")
        macro("HierTau2Hat", hier["global"]["tau2_hat"], "{:.2e}")
        for d, v in hier["per_target"].items():
            # A dataset outside the hierarchical fit (too few cells for the cell bootstrap) has no
            # shrunk quantile by design; only its prior-only and cross-pooled values exist.
            if v["cov_eb_shrunk"] is not None:
                macro(f"Hier{d}CovEBShrunk", v["cov_eb_shrunk"] * 100, "{:.1f}")
            macro(f"Hier{d}CovOwn", v["cov_own_quantile"] * 100, "{:.1f}")
            macro(f"Hier{d}CovPriorOnly", v["cov_prior_only"] * 100, "{:.1f}")
            macro(f"Hier{d}CovCrossPooled", v["cov_cross_source_pooled"] * 100, "{:.1f}")
        # Like-for-like comparison on the datasets where an empirical-Bayes value exists.
        fit = [d for d in hier["fit_groups"] if hier["per_target"][d]["cov_eb_shrunk"] is not None]
        macro("HierNumFitDatasets", len(fit), "{:d}")
        macro("HierMeanDevOwn",
              float(np.mean([abs(hier["per_target"][d]["cov_own_quantile"] - 0.90) for d in fit])) * 100, "{:.2f}")
        macro("HierMeanDevEB",
              float(np.mean([abs(hier["per_target"][d]["cov_eb_shrunk"] - 0.90) for d in fit])) * 100, "{:.2f}")
        macro("HierMeanDevOwnExclBIT",
              float(np.mean([abs(hier["per_target"][d]["cov_own_quantile"] - 0.90) for d in fit if d != "BIT"])) * 100, "{:.2f}")
        macro("HierMeanDevEBExclBIT",
              float(np.mean([abs(hier["per_target"][d]["cov_eb_shrunk"] - 0.90) for d in fit if d != "BIT"])) * 100, "{:.2f}")
        macro("HierMeanDevPriorOnly",
              float(np.mean([abs(hier["per_target"][d]["cov_prior_only"] - 0.90) for d in fit])) * 100, "{:.2f}")
        macro("HierMeanDevCrossPooled",
              float(np.mean([abs(hier["per_target"][d]["cov_cross_source_pooled"] - 0.90) for d in fit])) * 100, "{:.2f}")
        hgc_sec = registry["sections"]["hgc_cp_nested_search"]
        if hgc_sec != UNVERIFIED:
            macro("HgcCpMeanDevFit",
                  float(np.mean([abs(hgc_sec[d]["real_target_test_coverage"] - 0.90) for d in fit])) * 100, "{:.2f}")
        macro("HierBITWeight", hier["per_target"]["BIT"]["w_eb"], "{:.3f}")
        others_q = [hier["q_g"][d] for d in fit if d != "BIT"]
        macro("HierBITQuantileRatio", hier["q_g"]["BIT"] / float(np.mean(others_q)), "{:.0f}")

    hgc = registry["sections"]["hgc_cp_nested_search"]
    if hgc != UNVERIFIED:
        for d, v in hgc.items():
            macro(f"HgcCp{d}Lam", v["selected_lam"], "{:.1f}")
            macro(f"HgcCp{d}Beta", v["selected_beta"], "{:.2f}")
            macro(f"HgcCp{d}Coverage", v["real_target_test_coverage"] * 100, "{:.1f}")

    mwhc = registry["sections"]["mwhc"]
    if mwhc != UNVERIFIED:
        lam_words = {1.0: "One", 3.0: "Three", 5.0: "Five", 10.0: "Ten", 20.0: "Twenty"}
        s = mwhc["summary"]
        macro("MwhcFlatDev", s["mean_abs_dev_flat"] * 100, "{:.2f}")
        for lam in mwhc["lam_grid"]:
            word = lam_words.get(lam)
            if word is None:
                raise ValueError(f"mwhc lam {lam!r} has no word mapping -- add one")
            macro(f"MwhcDevLam{word}", s["mean_abs_dev_by_lam"][str(lam)] * 100, "{:.2f}")
            macro(f"MwhcDevFlooredLam{word}", s["mean_abs_dev_by_lam_floored"][str(lam)] * 100, "{:.2f}")
            macro(f"MwhcEssLam{word}", s["mean_ess_by_lam"][str(lam)], "{:.2f}")
        # XJTU is the clearest single-dataset illustration of the weight-concentration
        # failure mode and the floor.
        xjtu_lam20 = mwhc["per_target"]["XJTU"]["per_lam"]["20.0"]
        macro("MwhcXjtuUnflooredLamTwenty", xjtu_lam20["cov_eb_shrunk_sim"] * 100, "{:.1f}")
        macro("MwhcXjtuFlooredLamTwenty", xjtu_lam20["floored"]["cov_eb_shrunk"] * 100, "{:.1f}")
        macro("MwhcXjtuEssLamTwenty", xjtu_lam20["ess"], "{:.2f}")
        macro("MwhcXjtuEssFlooredLamTwenty", xjtu_lam20["floored"]["ess"], "{:.2f}")

    calsize = registry["sections"]["calibration_size_sensitivity"]
    if calsize != UNVERIFIED:
        for d, v in calsize["per_dataset"].items():
            pb = v["per_budget"]
            n_full = v["n_cells_available"]
            macro(f"CalSize{d}FullCells", n_full, "{:d}")
            macro(f"CalSize{d}FullMean", pb[str(n_full)]["coverage_mean"] * 100, "{:.1f}")
            if "1" in pb:
                b1 = pb["1"]
                macro(f"CalSize{d}OneMean", b1["coverage_mean"] * 100, "{:.1f}")
                macro(f"CalSize{d}OnePLow", b1["coverage_p05"] * 100, "{:.1f}")
                macro(f"CalSize{d}OnePHigh", b1["coverage_p95"] * 100, "{:.1f}")
        # Practical budget rule: smallest strict sub-budget whose 5th-percentile coverage is
        # within 5 points of the 90% target. The full budget is excluded because a "subsample"
        # of every cell has zero spread by construction.
        rule_floor = 0.85
        macro("CalSizeRuleFloor", rule_floor * 100, "{:.0f}")
        macro("CalSizeNRepeats", calsize["n_repeats"], "{:d}")
        for d, v in calsize["per_dataset"].items():
            n_full = v["n_cells_available"]
            ok = sorted(int(b) for b, r in v["per_budget"].items()
                        if int(b) < n_full and r["coverage_p05"] >= rule_floor)
            if ok:
                macro(f"CalSize{d}MinBudget", ok[0], "{:d}")
                macro(f"CalSize{d}MinBudgetPLow", v["per_budget"][str(ok[0])]["coverage_p05"] * 100, "{:.1f}")

    cqr_pooled = registry["sections"]["cqr_pooled"]
    cqr_gw = registry["sections"]["cqr_groupwise"]
    if cqr_pooled != UNVERIFIED and cqr_gw != UNVERIFIED:
        macro("CqrMAE", cqr_gw["test_mae"], "{:.5f}")
        macro("CqrRMSE", cqr_gw["test_rmse"], "{:.5f}")
        macro("CqrPooledCoverage", cqr_pooled["overall_coverage"] * 100, "{:.1f}")
        macro("CqrGroupwiseCoverage", cqr_gw["overall_coverage"] * 100, "{:.1f}")
        macro("CqrPooledMPIW", cqr_pooled["mpiw"], "{:.5f}")
        macro("CqrGroupwiseMPIW", cqr_gw["mpiw"], "{:.5f}")
        main_mpiw = registry["sections"]["main_model_mpiw"]
        main_model = registry["sections"]["main_model"]
        if main_mpiw != UNVERIFIED and main_model != UNVERIFIED:
            macro("CqrMPIWRatio", cqr_gw["mpiw"] / main_mpiw, "{:.2f}")
            macro("CqrMAEIncreasePercent", (cqr_gw["test_mae"] / main_model["test_mae"] - 1) * 100, "{:.1f}")
            macro("CqrRMSEIncreasePercent", (cqr_gw["test_rmse"] / main_model["test_rmse"] - 1) * 100, "{:.1f}")
        for d, v in cqr_gw["per_dataset_coverage"].items():
            macro(f"CqrGroupwiseCoverage{d}", v * 100, "{:.1f}")
        macro("CqrMeanDevFive",
              float(np.mean([abs(cqr_gw["per_dataset_coverage"][d] - 0.90) for d in five])) * 100, "{:.2f}")
        for d, v in cqr_pooled["per_dataset_coverage"].items():
            macro(f"CqrPooledCoverage{d}", v * 100, "{:.1f}")

    if registry["sections"]["multiseed_summary"] != UNVERIFIED:
        s = registry["sections"]["multiseed_summary"]
        macro("MultiseedMAEMean", s["mae_mean"], "{:.5f}")
        macro("MultiseedMAEStd", s["mae_std"], "{:.5f}")
        macro("MultiseedCoverageMean", s["coverage_mean"] * 100, "{:.2f}")
        macro("MultiseedCoverageStd", s["coverage_std"] * 100, "{:.2f}")
    else:
        for n in ("MultiseedMAEMean", "MultiseedMAEStd", "MultiseedCoverageMean", "MultiseedCoverageStd"):
            macro(n, None)

    for d, v in lodo.items():
        if v != UNVERIFIED:
            macro(f"Lodo{d}MAE", v["test_mae"], "{:.5f}")
            macro(f"Lodo{d}Coverage", v["coverage"] * 100, "{:.1f}")
        else:
            macro(f"Lodo{d}MAE", None)
            macro(f"Lodo{d}Coverage", None)

    repeated = registry["sections"]["repeated_splits"]
    if repeated != UNVERIFIED:
        var = repeated["groupwise_coverage_variability_across_splits"]
        for d, v in var.items():
            macro(f"RepeatedSplit{d}CovMean", v["mean"] * 100, "{:.1f}")
            macro(f"RepeatedSplit{d}CovMin", v["min"] * 100, "{:.1f}")
            macro(f"RepeatedSplit{d}CovMax", v["max"] * 100, "{:.1f}")
            macro(f"RepeatedSplit{d}CovRange", v["range"] * 100, "{:.1f}")
        macro("RepeatedSplitNSplits", len(repeated["split_seeds"]), "{:d}")
        stable, unstable = ("MIT", "RWTH", "BIT"), ("XJTU", "CALCE", "Oxford", "NASA")
        macro("RepeatedSplitStableRangeMax", max(var[d]["range"] for d in stable) * 100, "{:.1f}")
        macro("RepeatedSplitUnstableRangeMin", min(var[d]["range"] for d in unstable) * 100, "{:.1f}")
        macro("RepeatedSplitUnstableRangeMax", max(var[d]["range"] for d in unstable) * 100, "{:.1f}")

        # Persistence robustness across all splits: the main split (seed 42) from
        # persistence_baselines.json, plus seeds 43-46 from repeated_split_experiment.py.
        if persistence != UNVERIFIED:
            for d in repeated["persistence"]["43"].keys():
                losses = 0
                total = 0
                anchor = persistence["per_dataset"].get(d)
                if anchor is not None:
                    total += 1
                    if not (anchor["model"]["mae_real"] < anchor["persistence"]["mae_real"]):
                        losses += 1
                for seed in repeated["split_seeds"]:
                    if seed == 42:
                        continue
                    entry = repeated["persistence"].get(str(seed), {}).get(d)
                    if entry is not None:
                        total += 1
                        if not entry["model_beats_persistence"]:
                            losses += 1
                macro(f"PersistenceLossCount{d}", losses, "{:d}")
                macro(f"PersistenceSplitCount{d}", total, "{:d}")
    else:
        macro("RepeatedSplitNSplits", None)

    # Pooled vs groupwise over all splits (main + four repeats): does the main comparison hold
    # beyond one split? NASA is left out, as in the Wilcoxon test (3 test windows on the main split).
    if repeated != UNVERIFIED:
        pm = repeated["per_mode"]
        six = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford"]
        dev = lambda mode, s, d: abs(pm[mode][s]["per_dataset_coverage"][d] - 0.9) * 100  # noqa: E731
        pairs = [(s, d) for s in pm["groupwise"] for d in six]
        macro("SplitPairsGroupwiseCloser", sum(dev("groupwise", s, d) < dev("pooled", s, d) for s, d in pairs), "{:d}")
        macro("SplitPairsTotal", len(pairs), "{:d}")
        for d in six:
            macro(f"SplitMeanDevPooled{d}", np.mean([dev("pooled", s, d) for s in pm["pooled"]]), "{:.1f}")
            macro(f"SplitMeanDevGroupwise{d}", np.mean([dev("groupwise", s, d) for s in pm["groupwise"]]), "{:.1f}")
        macro("SplitPooledBITMax", max(pm["pooled"][s]["per_dataset_coverage"]["BIT"] for s in pm["pooled"]) * 100, "{:.1f}")

    # Size of the groupwise gain per dataset, with paired cell-bootstrap intervals
    # (pooled_vs_groupwise_bootstrap.py).
    boot = load("pooled_vs_groupwise_bootstrap.json")
    registry["sections"]["pooled_vs_groupwise_bootstrap"] = boot or UNVERIFIED
    if boot:
        bd = boot["per_dataset"]
        for d, v in bd.items():
            macro(f"BootGain{d}", v["mean_gain"], "{:.1f}")
            if "mean_gain_ci" in v:
                macro(f"BootGain{d}Lo", v["mean_gain_ci"][0], "{:.1f}")
                macro(f"BootGain{d}Hi", v["mean_gain_ci"][1], "{:.1f}")
        with_ci = [v for v in bd.values() if "mean_gain_ci" in v]
        macro("BootDatasetsWithCI", len(with_ci), "{:d}")
        macro("BootDatasetsCIAboveZero", sum(v["mean_gain_ci"][0] > 0 for v in with_ci), "{:d}")
        macro("BootDatasetsTotal", len(bd), "{:d}")
        macro("BootNBoot", boot["n_boot"], "{:,d}")
    else:
        macro("BootDatasetsCIAboveZero", None)

    # Deep ensemble as a second source of uncertainty (deep_ensemble.py).
    ens = load("deep_ensemble.json")
    registry["sections"]["deep_ensemble"] = ens or UNVERIFIED
    if ens:
        macro("EnsNMembers", ens["n_members"], "{:d}")
        for src, tag in (("deep_ensemble", "Ens"), ("mc_dropout", "Mc")):
            r = ens[src]
            macro(f"{tag}MAE", r["test_mae"], "{:.5f}")
            for mode, mtag in (("raw_gaussian", "Raw"), ("pooled", "Pooled"), ("groupwise", "Groupwise"),
                               ("normalized_groupwise", "NormGroupwise")):
                m = r[mode]
                macro(f"{tag}{mtag}Overall", m["overall_coverage"] * 100, "{:.1f}")
                macro(f"{tag}{mtag}Width", m["mean_width"], "{:.4f}")
                macro(f"{tag}{mtag}DevFive", m["mean_abs_dev_five"] * 100, "{:.2f}")
                for d in ("MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford"):
                    macro(f"{tag}{mtag}{d}", m["per_dataset_coverage"][d] * 100, "{:.1f}")
        e, mc = ens["deep_ensemble"], ens["mc_dropout"]
        macro("EnsMaeVsMcPercent", (1 - e["test_mae"] / mc["test_mae"]) * 100, "{:.0f}")
        macro("EnsGroupwiseWidthVsMcPercent",
              (1 - e["groupwise"]["mean_width"] / mc["groupwise"]["mean_width"]) * 100, "{:.0f}")
        macro("EnsMemberMAEMin", min(ens["member_test_mae"]), "{:.5f}")
        macro("EnsMemberMAEMax", max(ens["member_test_mae"]), "{:.5f}")
    else:
        macro("EnsNMembers", None)

    # The deep ensemble on every cell-level split (deep_ensemble_splits.py).
    es = load("deep_ensemble_splits.json")
    registry["sections"]["deep_ensemble_splits"] = es or UNVERIFIED
    if es:
        macro("EsNSplits", len(es["splits"]), "{:d}")
        macro("EsNormBestSplits", es["ens_norm_best_splits"], "{:d}")
        macro("EsNormBeatsMcGroupwiseSplits", es["ens_norm_beats_mc_groupwise_splits"], "{:d}")
        macro("EsNormNarrowerSplits", es["ens_norm_narrower_than_mc_groupwise_splits"], "{:d}")
        macro("EsMoreAccurateSplits", es["ens_more_accurate_splits"], "{:d}")
        macro("EsGroupwiseBeatsPooledSplits", es["ens_groupwise_beats_pooled_splits"], "{:d}")
        macro("EsPooledBITMin", es["ens_pooled_bit_range"][0] * 100, "{:.1f}")
        macro("EsPooledBITMax", es["ens_pooled_bit_range"][1] * 100, "{:.1f}")
        macro("EsGroupwiseBITMin", es["ens_groupwise_bit_range"][0] * 100, "{:.1f}")
        macro("EsGroupwiseBITMax", es["ens_groupwise_bit_range"][1] * 100, "{:.1f}")
        for key, tag in (("deep_ensemble/normalized_groupwise", "EnsNorm"), ("deep_ensemble/groupwise", "EnsGw"),
                         ("deep_ensemble/pooled", "EnsPooled"), ("mc_dropout/groupwise", "McGw"),
                         ("mc_dropout/normalized_groupwise", "McNorm"), ("mc_dropout/pooled", "McPooled")):
            v = es["summary"][key]
            macro(f"Es{tag}DevMean", v["mean_dev_five"] * 100, "{:.2f}")
            macro(f"Es{tag}DevMin", v["min_dev_five"] * 100, "{:.2f}")
            macro(f"Es{tag}DevMax", v["max_dev_five"] * 100, "{:.2f}")
            macro(f"Es{tag}Width", v["mean_width"], "{:.4f}")
    else:
        macro("EsNSplits", None)

    # Stronger borrowing baselines on every split (borrowing_variants.py).
    bv = load("borrowing_variants.json")
    registry["sections"]["borrowing_variants"] = bv or UNVERIFIED
    if bv:
        tags = {"own": "Own", "hier_linear_loo": "HierLin", "eb_linear_joint": "EbLinJoint",
                "eb_log_joint": "EbLogJoint", "bayes_log": "BayesLog", "bayes_log_noise": "BayesNoise",
                "bayes_log_spread": "BayesSpread", "prior_linear_loo": "PriorLin", "prior_log": "PriorLog",
                "prior_log_noise": "PriorNoise", "prior_log_spread": "PriorSpread"}
        sm = bv["summary"]
        macro("BvNSplits", len(bv["splits"]), "{:d}")
        macro("BvPriorScale", bv["prior_scale"], "{:g}")
        for m, tag in tags.items():
            macro(f"Bv{tag}DevMean", sm["mean_abs_dev"][m]["mean"] * 100, "{:.2f}")
            macro(f"Bv{tag}DevMin", sm["mean_abs_dev"][m]["min"] * 100, "{:.2f}")
            macro(f"Bv{tag}DevMax", sm["mean_abs_dev"][m]["max"] * 100, "{:.2f}")
            macro(f"Bv{tag}BeatsOwn", sm["mean_abs_dev"][m]["beats_own_splits"], "{:d}")
            macro(f"Bv{tag}BITMin", sm["bit_coverage"][m]["min"] * 100, "{:.1f}")
            macro(f"Bv{tag}BITMax", sm["bit_coverage"][m]["max"] * 100, "{:.1f}")
            macro(f"Bv{tag}DevMain", bv["splits"][str(config.RANDOM_SEED)]["mean_abs_dev"][m] * 100, "{:.2f}")
            further = [v["mean_abs_dev"][m] * 100 for s, v in bv["splits"].items() if s != str(config.RANDOM_SEED)]
            macro(f"Bv{tag}FurtherMean", np.mean(further), "{:.2f}")
            macro(f"Bv{tag}FurtherMin", min(further), "{:.2f}")
            macro(f"Bv{tag}FurtherMax", max(further), "{:.2f}")
        with_own = ["hier_linear_loo", "eb_linear_joint", "eb_log_joint", "bayes_log", "bayes_log_noise",
                    "bayes_log_spread"]
        best = min(with_own, key=lambda m: sm["mean_abs_dev"][m]["mean"])
        macro("BvBestGainMean", (sm["mean_abs_dev"]["own"]["mean"] - sm["mean_abs_dev"][best]["mean"]) * 100, "{:.2f}")
        gains = [(v["mean_abs_dev"]["own"] - v["mean_abs_dev"][best]) * 100 for v in bv["splits"].values()]
        macro("BvBestGainSplitMax", max(gains), "{:.1f}")
        for m, v in sm["bit_weight_own"].items():
            macro(f"Bv{tags[m]}BITWeightMin", v["min"], "{:.2f}")
            macro(f"Bv{tags[m]}BITWeightMax", v["max"], "{:.2f}")
        for a, name in (("0.5", "Half"), ("2.0", "Two")):
            macro(f"BvBayesLogDevScale{name}", sm["prior_scale_sensitivity"][a]["bayes_log"] * 100, "{:.2f}")
            macro(f"BvPriorLogDevScale{name}", sm["prior_scale_sensitivity"][a]["prior_log"] * 100, "{:.2f}")
        shifts = {d: [100 * (s["per_target"][d]["coverage"]["bayes_log_noise"] - s["per_target"][d]["coverage"]["own"])
                      for s in bv["splits"].values() if d in s["per_target"]]
                  for d in ("MIT", "RWTH", "XJTU", "BIT", "CALCE")}
        macro("BvNoiseXJTUShiftMin", min(shifts["XJTU"]), "{:.1f}")
        macro("BvNoiseXJTUShiftMax", max(shifts["XJTU"]), "{:.1f}")
        macro("BvNoiseOtherShiftMax", max(abs(x) for d in shifts if d != "XJTU" for x in shifts[d]), "{:.1f}")
    else:
        macro("BvNSplits", None)

    # Calibration-cell budget on every split (budget_across_splits.py).
    ba = load("budget_across_splits.json")
    registry["sections"]["budget_across_splits"] = ba or UNVERIFIED
    if ba:
        macro("BaNSplits", len(ba["per_split"]), "{:d}")
        for d, v in ba["across_splits"].items():
            macro(f"Ba{d}NeededMin", v["needed_min"], "{:d}")
            macro(f"Ba{d}NeededMax", v["needed_max"], "{:d}")
            macro(f"Ba{d}NotReached", v["not_reached_splits"], "{:d}")
            macro(f"Ba{d}FullMin", v["full_mean_min"] * 100, "{:.1f}")
            macro(f"Ba{d}FullMax", v["full_mean_max"] * 100, "{:.1f}")
            macro(f"Ba{d}BelowRule", v["full_below_rule_splits"], "{:d}")
            macro(f"Ba{d}OneMin", v["one_cell_mean_min"] * 100, "{:.1f}")
            macro(f"Ba{d}OneMax", v["one_cell_mean_max"] * 100, "{:.1f}")
            macro(f"Ba{d}Available", v["available_max"], "{:d}")
            other = [ba["per_split"][s][d]["full_mean"] * 100 for s in ba["per_split"] if s != str(config.RANDOM_SEED)]
            macro(f"Ba{d}FullOtherMin", min(other), "{:.1f}")
            macro(f"Ba{d}FullOtherMax", max(other), "{:.1f}")
    else:
        macro("BaNSplits", None)

    # Longer forecast horizons and calibrated naive forecasts (horizon_experiment.py).
    hz = load("horizon_experiment.json")
    registry["sections"]["horizon_experiment"] = hz or UNVERIFIED
    if hz:
        names = {"1": "One", "10": "Ten", "50": "Fifty"}
        for h, r in hz.items():
            t = names[h]
            mae = r["mae_real"]
            macro(f"Hz{t}NDatasets", len(mae), "{:d}")
            macro(f"Hz{t}BeatsPersist", sum(v["model_beats_persistence"] for v in mae.values()), "{:d}")
            macro(f"Hz{t}BeatsDrift", sum(v["model_beats_drift"] for v in mae.values()), "{:d}")
            for d, v in mae.items():
                # MAE in SOH percentage points.
                macro(f"Hz{t}{d}Model", v["model"] * 100, "{:.3f}")
                macro(f"Hz{t}{d}Persist", v["persistence"] * 100, "{:.3f}")
                macro(f"Hz{t}{d}Drift", v["drift"] * 100, "{:.3f}")
            for k, tag in (("model", "Model"), ("persistence", "Persist"), ("drift", "Drift")):
                macro(f"Hz{t}Pooled{tag}MAE", r["pooled_mae_real"][k] * 100, "{:.3f}")
            mc = r["model_calibration"]
            for mode, tag in (("pooled", "Pooled"), ("groupwise", "Groupwise"), ("normalized_groupwise", "Norm")):
                macro(f"Hz{t}{tag}Dev", mc[mode]["mean_abs_dev_five"] * 100, "{:.2f}")
                macro(f"Hz{t}{tag}BIT", mc[mode]["per_dataset_coverage"]["BIT"] * 100, "{:.1f}")
            for k, tag in (("persistence", "Persist"), ("drift", "Drift")):
                for mode, mtag in (("pooled", "Pooled"), ("groupwise", "Groupwise")):
                    c = r["naive_calibration"][k][mode]
                    macro(f"Hz{t}{tag}{mtag}Dev", c["mean_abs_dev_five"] * 100, "{:.2f}")
                    macro(f"Hz{t}{tag}{mtag}BIT", c["per_dataset_coverage"]["BIT"] * 100, "{:.1f}")
        naive = {mode: [r["naive_calibration"][k][mode]["mean_abs_dev_five"] * 100
                        for r in hz.values() for k in ("persistence", "drift")]
                 for mode in ("pooled", "groupwise")}
        model = {mode: [r["model_calibration"][mode]["mean_abs_dev_five"] * 100 for r in hz.values()]
                 for mode in ("pooled", "groupwise")}
        for mode, tag in (("pooled", "Pooled"), ("groupwise", "Groupwise")):
            macro(f"HzNaive{tag}DevMin", min(naive[mode]), "{:.1f}")
            macro(f"HzNaive{tag}DevMax", max(naive[mode]), "{:.1f}")
            macro(f"HzModel{tag}DevMin", min(model[mode]), "{:.1f}")
            macro(f"HzModel{tag}DevMax", max(model[mode]), "{:.1f}")
        macro("HzFiftyModelVsPersistPercent",
              (1 - hz["50"]["pooled_mae_real"]["model"] / hz["50"]["pooled_mae_real"]["persistence"]) * 100, "{:.0f}")
    else:
        macro("HzOneNDatasets", None)

    # Leave-one-dataset-out with calibration cells from the left-out dataset (lodo_target_calibration.py).
    lt = load("lodo_target_calibration.json")
    registry["sections"]["lodo_target_calibration"] = lt or UNVERIFIED
    if lt:
        pd_ = lt["per_dataset"]
        for d, v in pd_.items():
            macro(f"LodoT{d}Others", v["others"]["coverage"] * 100, "{:.1f}")
            if v["own_val"] is not None:
                macro(f"LodoT{d}OwnVal", v["own_val"]["coverage"] * 100, "{:.1f}")
            macro(f"LodoT{d}OwnAll", v["own_all"]["coverage"] * 100, "{:.1f}")
            macro(f"LodoT{d}AllWinOthers", v["all_windows_others_coverage"] * 100, "{:.1f}")
            macro(f"LodoT{d}InTraining", v["in_training"]["coverage"] * 100, "{:.1f}")
            macro(f"LodoT{d}WidthRatio", v["own_all"]["half_width_soh"] / v["in_training"]["half_width_soh"], "{:.1f}")
            macro(f"LodoT{d}MaeRatio", v["lodo_test_mae_soh"] / v["in_training"]["test_mae_soh"], "{:.1f}")
            macro(f"LodoT{d}PoolCells", v["n_pool_cells"], "{:d}")
            macro(f"LodoT{d}TestCells", v["n_test_cells"], "{:d}")
            if v.get("budget") is not None:
                pb = v["budget"]["per_budget"]
                if v["cells_needed"] is not None:
                    macro(f"LodoT{d}CellsNeeded", v["cells_needed"], "{:d}")
                macro(f"LodoT{d}OneCellPLow", pb["1"]["coverage_p05"] * 100, "{:.1f}")
                macro(f"LodoT{d}OneCellMean", pb["1"]["coverage_mean"] * 100, "{:.1f}")
                # The largest random subset below the full set of non-test cells.
                sub = max(int(b) for b in pb if int(b) < v["budget"]["n_cells_available"])
                macro(f"LodoT{d}MaxSubCells", sub, "{:d}")
                macro(f"LodoT{d}MaxSubPLow", pb[str(sub)]["coverage_p05"] * 100, "{:.1f}")
                macro(f"LodoT{d}MaxSubMean", pb[str(sub)]["coverage_mean"] * 100, "{:.1f}")
            macro(f"LodoT{d}OwnValWidthRatio",
                  v["own_val"]["half_width_soh"] / v["in_training"]["half_width_soh"] if v["own_val"] else None,
                  "{:.1f}")
        five = [d for d in ("MIT", "RWTH", "XJTU", "BIT", "CALCE") if d in pd_]
        dev = lambda k: float(np.mean([abs(pd_[d][k]["coverage"] - 0.9) for d in five])) * 100  # noqa: E731
        macro("LodoTDevOthers", dev("others"), "{:.2f}")
        macro("LodoTDevOwnAll", dev("own_all"), "{:.2f}")
        macro("LodoTDevOwnVal", dev("own_val"), "{:.2f}")
        macro("LodoTDevInTraining", dev("in_training"), "{:.2f}")
        ratios = [pd_[d]["own_val"]["half_width_soh"] / pd_[d]["in_training"]["half_width_soh"] for d in five]
        macro("LodoTWidthRatioMin", min(ratios), "{:.1f}")
        macro("LodoTWidthRatioMax", max(ratios), "{:.1f}")
        macro("LodoTOwnValBeatsOthers", sum(abs(pd_[d]["own_val"]["coverage"] - 0.9)
                                            < abs(pd_[d]["others"]["coverage"] - 0.9) for d in five), "{:d}")
        macro("LodoTNFive", len(five), "{:d}")
        # How far the retrained LODO models move from leave_one_out.py, both calibrated on the
        # other datasets and scored on all of the left-out dataset's windows.
        shift = {d: abs(pd_[d]["all_windows_others_coverage"] - lodo[d]["coverage"]) * 100
                 for d in pd_ if lodo.get(d, UNVERIFIED) != UNVERIFIED}
        for name, ds in (("Five", five), ("All", list(shift))):
            worst = max(ds, key=shift.get)
            macro(f"LodoTRetrainShift{name}Max", shift[worst], "{:.1f}")
            macro(f"LodoTRetrainShift{name}Dataset", worst, "{}")
    else:
        macro("LodoTDevOwnAll", None)

    # Strategy comparison replicated on splits 43-46 with every design frozen (split_replication.py).
    rep = load("split_replication.json")
    registry["sections"]["split_replication"] = rep or UNVERIFIED
    methods = {"Own": "own", "Hier": "hierarchical", "Prior": "prior_only",
               "OthersPooled": "others_pooled", "HgcCp": "hgc_cp"}
    if rep:
        per = rep["per_split"]
        splits = [str(s) for s in rep["splits"]]
        macro("ReplicationNSplits", len(splits), "{:d}")
        devs = {k: [per[s]["mean_abs_dev_five"][m] * 100 for s in splits] for k, m in methods.items()}
        for k, v in devs.items():
            macro(f"ReplicationMeanDev{k}", np.mean(v), "{:.2f}")
            macro(f"ReplicationMaxDev{k}", max(v), "{:.2f}")
            macro(f"ReplicationMinDev{k}", min(v), "{:.2f}")
        macro("ReplicationOwnBestSplits",
              sum(all(devs["Own"][i] <= devs[k][i] for k in methods) for i in range(len(splits))), "{:d}")
        macro("ReplicationHgcCpBeatsOwnSplits",
              sum(devs["HgcCp"][i] < devs["Own"][i] for i in range(len(splits))), "{:d}")
        for k, m in (("Own", "own"), ("Hier", "hierarchical"), ("HgcCp", "hgc_cp")):
            bit = [per[s]["per_dataset"]["BIT"][m] * 100 for s in splits]
            macro(f"ReplicationBIT{k}Min", min(bit), "{:.1f}")
            macro(f"ReplicationBIT{k}Max", max(bit), "{:.1f}")
        # BIT coverage over all five splits (main split + the four above), for the borrowing table.
        hier_main = registry["sections"]["hierarchical_calibration"]
        hgc_main = registry["sections"]["hgc_cp_nested_search"]
        if hier_main != UNVERIFIED and hgc_main != UNVERIFIED:
            main_bit = {"own": hier_main["per_target"]["BIT"]["cov_own_quantile"],
                        "hierarchical": hier_main["per_target"]["BIT"]["cov_eb_shrunk"],
                        "prior_only": hier_main["per_target"]["BIT"]["cov_prior_only"],
                        "others_pooled": hier_main["per_target"]["BIT"]["cov_cross_source_pooled"],
                        "hgc_cp": hgc_main["BIT"]["real_target_test_coverage"]}
            for k, m in methods.items():
                bit = [main_bit[m] * 100] + [per[s]["per_dataset"]["BIT"][m] * 100 for s in splits]
                macro(f"RepAllBIT{k}Min", min(bit), "{:.1f}")
                macro(f"RepAllBIT{k}Max", max(bit), "{:.1f}")
        w_bit = [per[s]["per_dataset"]["BIT"]["w_eb"] for s in splits]
        macro("ReplicationBITWeightMin", min(w_bit), "{:.3f}")
        macro("ReplicationBITWeightMax", max(w_bit), "{:.3f}")
        overlap = sum(per[s]["test_cells_also_in_main_split_test"] for s in splits)
        total = sum(per[s]["test_cells"] for s in splits)
        macro("ReplicationTestCellOverlapPercent", 100 * overlap / total, "{:.0f}")
    else:
        for k in methods:
            macro(f"ReplicationMeanDev{k}", None)

    # HGC-CP with its design chosen on validation data only, on all five splits
    # (hgc_cp_validation_only.py).
    val_only = load("hgc_cp_validation_only.json")
    registry["sections"]["hgc_cp_validation_only"] = val_only or UNVERIFIED
    hier = registry["sections"]["hierarchical_calibration"]
    hgc = registry["sections"]["hgc_cp_nested_search"]
    if val_only and rep and hier != UNVERIFIED and hgc != UNVERIFIED:
        vs = val_only["per_split"]
        order = ["main"] + [str(s) for s in rep["splits"]]
        dev_val = [vs[s]["mean_abs_dev_five_validation_only_hgc_cp"] * 100 for s in order]
        dev_own = [vs[s]["mean_abs_dev_five_own"] * 100 for s in order]
        five = ["MIT", "RWTH", "XJTU", "BIT", "CALCE"]
        dev_orig = ([float(np.mean([abs(hgc[d]["real_target_test_coverage"] - 0.9) for d in five])) * 100]
                    + [rep["per_split"][s]["mean_abs_dev_five"]["hgc_cp"] * 100 for s in order[1:]])
        macro("HgcCpValNSplits", len(order), "{:d}")
        macro("HgcCpValMeanDevMain", dev_val[0], "{:.2f}")
        macro("HgcCpValMinDev", min(dev_val), "{:.2f}")
        macro("HgcCpValMaxDev", max(dev_val), "{:.2f}")
        macro("HgcCpValOwnMinDev", min(dev_own), "{:.2f}")
        macro("HgcCpValOwnMaxDev", max(dev_own), "{:.2f}")
        macro("HgcCpValBeatsOwnSplits", sum(v < o for v, o in zip(dev_val, dev_own)), "{:d}")
        macro("HgcCpValMeanDevAll", np.mean(dev_val), "{:.2f}")
        macro("HgcCpOrigMeanDevAll", np.mean(dev_orig), "{:.2f}")
        macro("HgcCpValWorseThanOrigSplits", sum(v > o for v, o in zip(dev_val, dev_orig)), "{:d}")
        targets = [t for s in order for t in vs[s]["per_target"].values()]
        macro("HgcCpValTargets", len(targets), "{:d}")
        macro("HgcCpValUnbounded", sum(not np.isfinite(t["half_width"]) for t in targets), "{:d}")
        macro("HgcCpValLamZero", sum(t["lam"] == 0.0 for t in targets), "{:d}")
        bit = [vs[s]["per_target"]["BIT"]["coverage"] * 100 for s in order]
        macro("HgcCpValBITMin", min(bit), "{:.1f}")
        macro("HgcCpValBITMax", max(bit), "{:.1f}")
        mit_ratio = [vs[s]["per_target"]["MIT"]["half_width"] / vs[s]["per_target"]["MIT"]["own_half_width"]
                     for s in order]
        macro("HgcCpValMITWidthMin", min(mit_ratio), "{:.1f}")
        macro("HgcCpValMITWidthMax", max(mit_ratio), "{:.1f}")
        macro("HgcCpValFurtherMeanDev", np.mean(dev_val[1:]), "{:.2f}")
        macro("HgcCpValFurtherMinDev", min(dev_val[1:]), "{:.2f}")
        macro("HgcCpValFurtherMaxDev", max(dev_val[1:]), "{:.2f}")
        rep_names = {"summary5": "5 statistics", "mean_slope": "mean, slope", "window": "raw window"}
        for d, t in vs["main"]["per_target"].items():
            macro(f"HgcCpVal{d}Rep", rep_names[t["representation"]], "{}")
            macro(f"HgcCpVal{d}Lam", t["lam"], "{:.1f}")
            macro(f"HgcCpVal{d}Beta", t["beta"], "{:.2f}")
            macro(f"HgcCpVal{d}Coverage", t["coverage"] * 100, "{:.1f}")

        # The method comparison restricted to the datasets with at least six test cells on the main
        # split (MIT, RWTH, XJTU, BIT), over all five splits: does any conclusion rest on the
        # datasets with one or two test cells?
        four = ["MIT", "RWTH", "XJTU", "BIT"]
        per_split = []
        for s in order:
            if s == "main":
                cov = {d: {"own": hier["per_target"][d]["cov_own_quantile"],
                           "hierarchical": hier["per_target"][d]["cov_eb_shrunk"],
                           "prior_only": hier["per_target"][d]["cov_prior_only"],
                           "others_pooled": hier["per_target"][d]["cov_cross_source_pooled"],
                           "hgc_cp": hgc[d]["real_target_test_coverage"]} for d in four}
            else:
                cov = {d: rep["per_split"][s]["per_dataset"][d] for d in four}
            for d in four:
                cov[d] = dict(cov[d], hgc_cp_val=vs[s]["per_target"][d]["coverage"])
            per_split.append({m: float(np.mean([abs(cov[d][m] - 0.9) for d in four])) * 100
                              for m in ("own", "hierarchical", "prior_only", "others_pooled", "hgc_cp", "hgc_cp_val")})
        for k, m in (("Own", "own"), ("Hier", "hierarchical"), ("HgcCpVal", "hgc_cp_val")):
            macro(f"FourDs{k}MinDev", min(p[m] for p in per_split), "{:.2f}")
            macro(f"FourDs{k}MaxDev", max(p[m] for p in per_split), "{:.2f}")
        macro("FourDsOwnBestSplits", sum(all(p["own"] <= v for v in p.values()) for p in per_split), "{:d}")
    else:
        macro("HgcCpValMeanDevMain", None)

    # Field check on the EVBattery fleet data (field_check.py).
    field = load("field_check.json")
    registry["sections"]["field_check"] = field or UNVERIFIED
    ev_name = {"EV1": "EvOne", "EV2": "EvTwo", "EV3": "EvThree"}
    if field:
        fd = field["data"]
        for d, n in fd["vehicles_with_labels"].items():
            macro(f"Field{ev_name[d]}Vehicles", n, "{:d}")
            macro(f"Field{ev_name[d]}Events", fd["labelled_events"][d], "{:,d}")
        macro("FieldVehiclesTotal", sum(fd["vehicles_with_labels"].values()), "{:d}")
        macro("FieldEventsTotal", sum(fd["labelled_events"].values()), "{:,d}")
        macro("FieldWindowsTotal", sum(fd["windows"].values()), "{:,d}")
        macro("FieldNSplits", len(field["splits"]), "{:d}")
        per = [field["per_split"][str(s)] for s in field["splits"]]
        main = per[0]
        # Pooled vs groupwise (z-units), EV1 and EV2 in every split. EV3 (a single test vehicle when
        # it has one at all) is left out here as everywhere else in the field section.
        pairs = [(p["pooled_vs_groupwise"]["pooled"][d], p["pooled_vs_groupwise"]["groupwise"][d])
                 for p in per for d in ("EV1", "EV2")]
        macro("FieldPairsTotal", len(pairs), "{:d}")
        macro("FieldPairsGroupwiseCloser", sum(abs(g - 0.9) < abs(q - 0.9) for q, g in pairs), "{:d}")
        macro("FieldMeanDevPooled", np.mean([abs(q - 0.9) for q, _ in pairs]) * 100, "{:.1f}")
        macro("FieldMeanDevGroupwise", np.mean([abs(g - 0.9) for _, g in pairs]) * 100, "{:.1f}")
        for d in ("EV1", "EV2"):
            macro(f"FieldPooled{ev_name[d]}", main["pooled_vs_groupwise"]["pooled"][d] * 100, "{:.1f}")
            macro(f"FieldGroupwise{ev_name[d]}", main["pooled_vs_groupwise"]["groupwise"][d] * 100, "{:.1f}")
            pooled_all = [p["pooled_vs_groupwise"]["pooled"][d] * 100 for p in per]
            macro(f"FieldPooled{ev_name[d]}Min", min(pooled_all), "{:.1f}")
            macro(f"FieldPooled{ev_name[d]}Max", max(pooled_all), "{:.1f}")
            macro(f"FieldXpool{ev_name[d]}Main", main["borrowing"]["per_dataset"][d]["others_pooled"] * 100, "{:.1f}")
            # Own data against calibrating on the other manufacturer(s), SOH units, all splits.
            for k, m in (("Own", "own"), ("Xpool", "others_pooled")):
                v = [p["borrowing"]["per_dataset"][d][m] * 100 for p in per]
                macro(f"Field{k}{ev_name[d]}Min", min(v), "{:.1f}")
                macro(f"Field{k}{ev_name[d]}Max", max(v), "{:.1f}")
        two = ["EV1", "EV2"]
        dev = lambda p, m: float(np.mean([abs(p["borrowing"]["per_dataset"][d][m] - 0.9) for d in two])) * 100  # noqa: E731
        own_dev = [dev(p, "own") for p in per]
        xpool_dev = [dev(p, "others_pooled") for p in per]
        macro("FieldOwnDevMin", min(own_dev), "{:.1f}")
        macro("FieldOwnDevMax", max(own_dev), "{:.1f}")
        macro("FieldXpoolDevMin", min(xpool_dev), "{:.1f}")
        macro("FieldXpoolDevMax", max(xpool_dev), "{:.1f}")
        macro("FieldOwnBestSplits", sum(o < x for o, x in zip(own_dev, xpool_dev)), "{:d}")
        losses = [o - x for o, x in zip(own_dev, xpool_dev) if o >= x]
        macro("FieldOwnLossGap", max(losses) if losses else 0.0, "{:.1f}")
        assert all(p["borrowing"]["fit_groups"] == two for p in per), "field hierarchical fit is not EV1+EV2"
        macro("FieldHierXpoolMaxGap", max(abs(p["borrowing"]["per_dataset"][d]["hierarchical"]
                                              - p["borrowing"]["per_dataset"][d]["others_pooled"]) * 100
                                          for p in per for d in two), "{:.1f}")
        hgc_splits = [p for p in per if p["borrowing"]["hgc_cp_val_run"]]
        macro("FieldHgcValSplits", len(hgc_splits), "{:d}")
        if hgc_splits:
            hdev = [dev(p, "hgc_cp_val") for p in hgc_splits]
            odev = [dev(p, "own") for p in hgc_splits]
            macro("FieldHgcValDevMin", min(hdev), "{:.1f}")
            macro("FieldHgcValDevMax", max(hdev), "{:.1f}")
            macro("FieldHgcValBeatsOwn", sum(h < o for h, o in zip(hdev, odev)), "{:d}")
        # The laboratory model, never trained on field data, on the field test vehicles.
        lab = field["lab_model_on_field"]
        for d in ("EV1", "EV2"):
            macro(f"FieldLabPooled{ev_name[d]}", lab["lab_pooled_quantile"][d] * 100, "{:.1f}")
            macro(f"FieldLabOwn{ev_name[d]}", lab["field_own_quantile"][d] * 100, "{:.1f}")
        macro("FieldLabModelMAE", lab["test_mae"], "{:.3f}")
        macro("FieldModelMAE", main["pooled_vs_groupwise"]["test_mae"], "{:.3f}")
        # Calibration budget in vehicles, with the laboratory rule.
        macro("FieldBudgetNRepeats", field["budget"]["n_repeats"], "{:d}")
        for d, v in field["budget"]["per_dataset"].items():
            n_full = v["n_cells_available"]
            macro(f"FieldBudget{ev_name[d]}Available", n_full, "{:d}")
            ok = sorted(int(b) for b, r in v["per_budget"].items() if int(b) < n_full and r["coverage_p05"] >= 0.85)
            macro(f"FieldBudget{ev_name[d]}", ok[0] if ok else None, "{:d}")
            macro(f"FieldBudget{ev_name[d]}OnePLow", v["per_budget"]["1"]["coverage_p05"] * 100, "{:.1f}")
        pers = field["persistence"]
        macro("FieldPersistenceBetter", sum(v["persistence_mae_soh"] < v["model_mae_soh"] for v in pers.values()), "{:d}")
        macro("FieldPersistenceN", len(pers), "{:d}")
    else:
        macro("FieldVehiclesTotal", None)

    fallback = registry["sections"]["nasa_fallback_splits"]
    macro("NasaFallbackSplits", len(fallback) if fallback != UNVERIFIED else None, "{:d}")

    if data_layer:
        recon_detail = data_layer["checks"]["cell_reconciliation_339_vs_337"]["detail"]
        # Match each count by its label: a bare r"=(\d+)" also matches the "1" in ">=1 window".
        cells_loaded = int(re.search(r"cells loaded[^=]*=(\d+)", recon_detail).group(1))
        cells_with_windows = int(re.search(r"cells contributing >=1 window=(\d+)", recon_detail).group(1))
        macro("TotalCellsLoaded", cells_loaded, "{:d}")
        macro("CellsWithWindows", cells_with_windows, "{:d}")
        ascending_detail = data_layer["checks"]["windows_cycle_ascending_within_window"]["detail"]
        total_test_windows = int(re.search(r"of (\d+) test windows", ascending_detail).group(1))
        macro("TotalTestWindows", total_test_windows, "{:,d}")
    else:
        macro("TotalCellsLoaded", None)
        macro("CellsWithWindows", None)
        macro("TotalTestWindows", None)

    dc = registry["sections"]["data_counts"]
    have = dc != UNVERIFIED
    macro("TrainWindows", dc["train_windows"] if have else None, "{:,d}")
    macro("ValWindows", dc["val_windows"] if have else None, "{:,d}")
    for d in ("MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"):
        macro(f"TestCells{d}", dc["test_cells"][d] if have else None, "{:d}")
        macro(f"TestWindows{d}", dc["test_windows"][d] if have else None, "{:,}")

    if xjtu_bridge:
        n_sat, n_other = xjtu_bridge["n_sim_satellite"], xjtu_bridge["n_other_47_cells"]
        n_bridging_total = (n_sat * xjtu_bridge["frac_bridging_sim_satellite"]
                             + n_other * xjtu_bridge["frac_bridging_other_47_cells"])
        macro("XjtuBridgingFraction", 100 * n_bridging_total / (n_sat + n_other), "{:.1f}")
        macro("XjtuBridgingWindowCount", int(round(n_bridging_total)), "{:d}")
    else:
        macro("XjtuBridgingFraction", None)
        macro("XjtuBridgingWindowCount", None)

    pred_ci = registry["sections"]["predictor_consistent_coverage"]
    if pred_ci != UNVERIFIED:
        for d in ("XJTU", "CALCE"):
            v = pred_ci["per_dataset"][d]
            wlo, whi = v["cp95_window_level_see_docstring_caveat"]
            clo, chi = v["cell_bootstrap_95ci"]
            macro(f"PredCI{d}WindowLow", wlo * 100, "{:.1f}")
            macro(f"PredCI{d}WindowHigh", whi * 100, "{:.1f}")
            macro(f"PredCI{d}CellLow", clo * 100, "{:.1f}")
            macro(f"PredCI{d}CellHigh", chi * 100, "{:.1f}")
            macro(f"PredCI{d}WidthRatio", (chi - clo) / (whi - wlo), "{:.1f}")
        # Pooling in real SOH units instead of per-dataset z-units: the alternative meaning of
        # "pooled" (see predictor_consistent_recalibration.py).
        for d, v in pred_ci["per_dataset"].items():
            macro(f"PooledRealCoverage{d}", v["cov_pooled_real"] * 100, "{:.1f}")

    xjtu_sens = registry["sections"]["xjtu_bridging_sensitivity"]
    if xjtu_sens != UNVERIFIED:
        bridge, clean, allx = (xjtu_sens["bridging_windows_only"], xjtu_sens["clean_windows_only"],
                                xjtu_sens["all_xjtu"])
        macro("XjtuBridgingRMSERatio", bridge["rmse_z"] / clean["rmse_z"], "{:.1f}")
        macro("XjtuBridgingMAEShiftPercent", 100 * (allx["mae_z"] - clean["mae_z"]) / clean["mae_z"], "{:.1f}")
        macro("XjtuBridgingCoverage", bridge["coverage"] * 100, "{:.1f}")
        macro("XjtuCleanCoverage", clean["coverage"] * 100, "{:.1f}")

    wx = registry["sections"]["wilcoxon"]
    if wx != UNVERIFIED:
        dl = wx["dataset_level"]
        macro("WilcoxonN", dl["n"], "{:d}")
        macro("WilcoxonGroupwiseCloser", dl["groupwise_closer"], "{:d}")
        macro("WilcoxonP", dl["p_one_sided_exact"], "{:.4f}")
        macro("WilcoxonPMin", dl["p_min_attainable"], "{:.4f}")
        macro("WilcoxonNArch", len(wx["architectures"]), "{:d}")
        macro("WilcoxonPairN", wx["pair_level_not_independent"]["n"], "{:d}")

    rw = registry["sections"]["rwth_corrections"]
    if rw != UNVERIFIED:
        macro("RwthRowsRaw", rw["rows_raw"], "{:,}")
        macro("RwthDropFirstRows", rw["rows_removed_by_drop_first_alone"], "{:,}")
        macro("RwthDropFirstPct", 100 * rw["rows_removed_by_drop_first_alone"] / rw["rows_raw"], "{:.2f}")
        macro("RwthOutlierRows", rw["rows_removed_by_outlier_fix_alone"], "{:,}")
        macro("RwthOutlierPct", 100 * rw["rows_removed_by_outlier_fix_alone"] / rw["rows_raw"], "{:.2f}")
        macro("RwthBothRows", rw["rows_removed_by_both_combined"], "{:,}")
        macro("RwthBothPct", rw["pct_removed_by_both_combined"], "{:.2f}")
    cc = registry["sections"]["calce_corrections"]
    if cc != UNVERIFIED:
        macro("CalceCells", cc["n_cells"], "{:d}")
        macro("CalceRows", cc["total_rows"], "{:,}")
        macro("CalceCumulativeFiles", cc["cumulative_files"], "{:d}")
        macro("CalceRawFiles", dc["calce_raw_files"] if dc != UNVERIFIED else None, "{:d}")
        macro("CalceDropPct", 100 * (cc["total_rows"] - cc["rows_after_drop"]) / cc["total_rows"], "{:.2f}")
        macro("CalceSohKeepMin", cc["soh_keep_pct_min"], "{:.1f}")
        macro("CalceSohKeepMax", cc["soh_keep_pct_max"], "{:.1f}")
        macro("CalceMedianStepMin", cc["median_step_min"], "{:.4f}")
        macro("CalceMedianStepMax", cc["median_step_max"], "{:.4f}")

    det = registry["sections"]["deterministic_predictor"]
    if det != UNVERIFIED:
        main_split = det["per_split"][str(config.RANDOM_SEED)]
        d, m = main_split["deterministic"], main_split["mc_mean"]
        macro("DetMAE", d["mae_z"], "{:.5f}")
        macro("DetRMSE", d["rmse_z"], "{:.5f}")
        macro("DetCoverage", d["coverage"] * 100, "{:.1f}")
        macro("DetMPIW", d["mpiw"], "{:.5f}")
        macro("DetWidthReductionMainPercent", 100 * (1 - d["mpiw"] / m["mpiw"]), "{:.0f}")
        macro("DetMAEReductionMainPercent", 100 * (1 - d["mae_z"] / m["mae_z"]), "{:.0f}")
        s = det["summary"]
        macro("DetNSplits", s["n_splits"], "{:d}")
        macro("DetLowerMAESplits", s["det_lower_mae_count"], "{:d}")
        macro("DetNarrowerSplits", s["det_narrower_count"], "{:d}")
        macro("DetMAEReductionMeanPercent", s["mae_reduction_pct_mean"], "{:.0f}")
        macro("DetMAEReductionMinPercent", s["mae_reduction_pct_min"], "{:.0f}")
        macro("DetWidthReductionMeanPercent", s["width_reduction_pct_mean"], "{:.0f}")
        macro("DetWidthReductionMinPercent", s["width_reduction_pct_min"], "{:.0f}")
        macro("DetCoverageMin", s["det_coverage_min"] * 100, "{:.1f}")
        macro("DetCoverageMax", s["det_coverage_max"] * 100, "{:.1f}")
        macro("McCoverageMin", s["mc_coverage_min"] * 100, "{:.1f}")
        macro("McCoverageMax", s["mc_coverage_max"] * 100, "{:.1f}")
        macro("DetMeanDevFiveAvg", s["det_mean_dev_five_mean"] * 100, "{:.2f}")
        macro("McMeanDevFiveAvg", s["mc_mean_dev_five_mean"] * 100, "{:.2f}")
        macro("DetBeatsPersistenceMITSplits", s["det_beats_persistence_count"]["MIT"], "{:d}")
        macro("DetBeatsPersistenceRWTHSplits", s["det_beats_persistence_count"]["RWTH"], "{:d}")

    deploy = registry["sections"]["deployability"]
    deploy_labels = {"full": "Full", "no_attention": "NoAttention", "no_multiscale": "NoMultiscale",
                     "plain_lstm": "PlainLstm", "gru": "Gru", "transformer": "Transformer"}
    if deploy != UNVERIFIED:
        for variant, label in deploy_labels.items():
            v = deploy["variants"].get(variant)
            if v is None:  # variant not benchmarked yet: say so instead of printing a number
                for field in ("Params", "CheckpointKB", "DetMs", "McMs", "McPNinetyFiveMs",
                              "McBatchedMs", "GpuThroughputK"):
                    macro(f"Deploy{label}{field}", None)
                continue
            macro(f"Deploy{label}Params", v["n_params"], "{:,}")
            macro(f"Deploy{label}CheckpointKB", v["checkpoint_kb"], "{:.0f}")
            macro(f"Deploy{label}DetMs", v["cpu_batch1_deterministic_median_ms"], "{:.2f}")
            macro(f"Deploy{label}McMs", v["cpu_batch1_mc_mean_as_evaluated_median_ms"], "{:.1f}")
            macro(f"Deploy{label}McPNinetyFiveMs", v["cpu_batch1_mc_mean_as_evaluated_p95_ms"], "{:.1f}")
            macro(f"Deploy{label}McBatchedMs", v["cpu_batch1_mc_mean_batched_replicas_median_ms"], "{:.2f}")
            macro(f"Deploy{label}GpuThroughputK", v["gpu_throughput_deterministic_samples_per_s"] / 1e3, "{:.0f}")
        full = deploy["variants"]["full"]
        macro("DeployMcBatchedSpeedup", full["cpu_batch1_mc_mean_as_evaluated_median_ms"]
              / full["cpu_batch1_mc_mean_batched_replicas_median_ms"], "{:.1f}")
        co = deploy["conformal_overhead_full_model"]
        macro("DeployCalibrationMs", co["calibration_time_ms"], "{:.1f}")
        macro("DeployCalibrationSamples", co["n_calibration_samples"], "{:,}")
        macro("DeployIntervalNs", co["interval_time_per_sample_us"] * 1e3, "{:.0f}")
        macro("DeployIntervalOverheadPercent",
              100 * co["interval_time_per_sample_us"] / (full["cpu_batch1_mc_mean_batched_replicas_median_ms"] * 1e3),
              "{:.4f}")
        macro("DeployMcDropoutSamples", deploy["mc_samples"], "{:d}")
        cpu = re.sub(r"\(R\)|\(TM\)|CPU|@.*$", "", deploy["hardware"]["cpu"])
        macro("DeployCpuName", " ".join(cpu.split()))
        macro("DeployGpuName", deploy["hardware"]["gpu"])

    with open(os.path.join("paper", "numbers.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved paper/numbers.tex ({len(lines)-3} macros)")

    with open("results_registry.json", "w") as fh:
        json.dump(registry, fh, indent=2, default=str)
    print(f"saved results_registry.json ({len(registry['sections'])} sections, "
          f"{len(registry['values'])} values)")

    unverified = sorted(k for k, v in registry["values"].items() if v == UNVERIFIED)
    print(f"\n{len(unverified)} value(s) UNVERIFIED (input missing or not defined): "
          f"{', '.join(unverified) if unverified else 'none'}")


if __name__ == "__main__":
    main()
