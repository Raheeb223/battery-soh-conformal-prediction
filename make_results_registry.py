"""
Stage 11: single machine-generated source of numerical truth for the
manuscript. Reads every verified outputs/*.json produced by this audit and
compiles them into results_registry.json, then emits paper/numbers.tex
(one \\newcommand per number, so main.tex NEVER has a hand-typed number --
every value in the manuscript must come from \\num{...} macros defined here).

Tolerant of missing files (some LODO configs may still be training, or a
later-stage script may not have been run yet) -- each section is wrapped so
a missing input produces an explicit "UNVERIFIED" placeholder rather than a
crash or a silently wrong number, per the brief's rule: write UNVERIFIED
when something can't be verified, never fabricate or quietly omit.

Run:  python make_results_registry.py
Out:  results_registry.json, paper/numbers.tex
"""
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


def main():
    registry = {"generated_from": "make_results_registry.py", "sections": {}}

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
        for name in ("full", "no_attention", "no_multiscale", "plain_lstm", "gru")
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

    # ---- HGC-CP nested hyperparameter search (K4) ----
    registry["sections"]["hgc_cp_nested_search"] = load("nested_hp_search_results.json") or UNVERIFIED

    # ---- MWHC: MMD-weighted hierarchical calibration (new method) ----
    registry["sections"]["mwhc"] = load("mwhc_results.json") or UNVERIFIED

    # ---- Calibration-cell-budget sensitivity (new experiment) ----
    registry["sections"]["calibration_size_sensitivity"] = load("calibration_size_sensitivity.json") or UNVERIFIED

    # ---- CQR: Conformalized Quantile Regression baseline (new comparison) ----
    registry["sections"]["cqr_pooled"] = load("cqr_pooled_results.json") or UNVERIFIED
    registry["sections"]["cqr_groupwise"] = load("cqr_groupwise_results.json") or UNVERIFIED

    # ---- LODO (K1-fixed) -- tolerant of partial completion ----
    # IMPORTANT: lodo_*_results.json files from the OLD (pre-K1-fix) run are
    # still on disk for configs run_all_lodo() hasn't reached yet in the
    # current background run -- they lack "calibration_predictor" (only the
    # fixed train_one()-mirroring code path writes that field) and must be
    # treated as UNVERIFIED/stale here, NOT silently accepted as current.
    # Confirmed concretely: lodo_MIT_results.json (10/1/2026, no
    # calibration_predictor field) matches the brief's stale "as-run" anchor
    # (MAE 0.00089/coverage 92.16%) exactly -- this is the pre-fix file.
    lodo = {}
    for d in ("NASA", "CALCE", "Oxford", "MIT", "BIT", "XJTU", "RWTH"):
        v = load(f"lodo_{d}_results.json")
        if v and "calibration_predictor" in v:
            lodo[d] = v
        else:
            lodo[d] = UNVERIFIED  # missing, or a stale pre-K1-fix file
    registry["sections"]["lodo"] = lodo
    registry["sections"]["lodo_complete"] = all(v != UNVERIFIED for v in lodo.values())

    # ---- Predictor-consistent recalibration (per-dataset coverage + both CI types, E13-fixed) ----
    registry["sections"]["predictor_consistent_coverage"] = load("predictor_consistent_recalibration.json") or UNVERIFIED

    # ---- Cell-clustered bootstrap / XJTU bridging / RWTH verification ----
    xjtu_bridge = load("xjtu_bridging_isolation.json")
    registry["sections"]["xjtu_bridging_isolation"] = xjtu_bridge or UNVERIFIED
    registry["sections"]["xjtu_bridging_sensitivity"] = load("xjtu_bridging_sensitivity.json") or UNVERIFIED
    registry["sections"]["rwth_verification"] = load("rwth_verification_v2.json") or UNVERIFIED

    # ---- K6: repeated cell-level splits (between-split variability) ----
    registry["sections"]["repeated_splits"] = load("repeated_split_summary.json") or UNVERIFIED

    # ---- Stage 2 data-layer verification (cell reconciliation, window counts) ----
    stage2_path = "stage2_report.json"
    stage2 = None
    if os.path.exists(stage2_path):
        with open(stage2_path) as fh:
            stage2 = json.load(fh)
    registry["sections"]["stage2_data_layer"] = stage2 or UNVERIFIED

    # ---- Write registry ----
    with open("results_registry.json", "w") as fh:
        json.dump(registry, fh, indent=2, default=str)
    print(f"saved results_registry.json ({len(registry['sections'])} sections)")

    # ---- Emit paper/numbers.tex ----
    os.makedirs("paper", exist_ok=True)
    lines = ["% AUTO-GENERATED by make_results_registry.py -- do not hand-edit.",
             "% Every number in main.tex must come from a \\NUM{...} macro defined here.", ""]

    seen_macro_names = set()

    def macro(name, value, fmt="{:.4f}"):
        safe_name = re.sub(r"[^A-Za-z]", "", name)  # LaTeX macro names: letters only
        # Guards against exactly the bug found and fixed once already in this file: two
        # DIFFERENT inputs (e.g. "Window10MAE" / "Window20MAE") can sanitize to the SAME
        # macro name once digits are stripped, silently overwriting one value with another
        # in the .tex output. Fail loudly here instead of producing a wrong manuscript number.
        if safe_name in seen_macro_names:
            raise ValueError(f"macro name collision: '{name}' sanitizes to '\\{safe_name}', "
                              f"which was already defined by a previous macro() call. Rename "
                              f"one of the inputs so they don't collide after non-letter "
                              f"characters are stripped.")
        seen_macro_names.add(safe_name)
        if value is None or value == UNVERIFIED:
            lines.append(f"\\newcommand{{\\{safe_name}}}{{UNVERIFIED}}")
        elif isinstance(value, (int, float)):
            lines.append(f"\\newcommand{{\\{safe_name}}}{{{fmt.format(value)}}}")
        else:
            lines.append(f"\\newcommand{{\\{safe_name}}}{{{value}}}")

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
    # overwrite -- confirmed and fixed; spell out the window size in words.
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
            macro(f"Hier{d}CovEBShrunk", v["cov_eb_shrunk"] * 100 if v["cov_eb_shrunk"] is not None else None, "{:.1f}")
            macro(f"Hier{d}CovOwn", v["cov_own_quantile"] * 100, "{:.1f}")

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
        # failure mode and the floor fix -- see Section~\ref{sec:mwhc}.
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

        # Persistence robustness across all 5 splits tried so far (seed 42, the
        # original anchor, from persistence_baselines.json, plus seeds 43-46 from K6).
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

    if stage2:
        recon_detail = stage2["checks"]["cell_reconciliation_339_vs_337"]["detail"]
        nums = [int(n) for n in re.findall(r"=(\d+)", recon_detail)]
        # nums = [cells_loaded, cells_assigned_to_split, cells_with_windows], in that order
        macro("TotalCellsLoaded", nums[0], "{:d}")
        macro("CellsWithWindows", nums[2], "{:d}")
        ascending_detail = stage2["checks"]["windows_cycle_ascending_within_window"]["detail"]
        total_test_windows = int(re.search(r"of (\d+) test windows", ascending_detail).group(1))
        macro("TotalTestWindows", total_test_windows, "{:d}")
    else:
        macro("TotalCellsLoaded", None)
        macro("CellsWithWindows", None)
        macro("TotalTestWindows", None)

    cache_path = os.path.join(config.CACHE_DIR, "processed.npz")
    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        macro("TrainWindows", cache["X_train"].shape[0], "{:d}")
        macro("ValWindows", cache["X_val"].shape[0], "{:d}")
    else:
        macro("TrainWindows", None)
        macro("ValWindows", None)

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

    xjtu_sens = registry["sections"]["xjtu_bridging_sensitivity"]
    if xjtu_sens != UNVERIFIED:
        bridge, clean, allx = (xjtu_sens["bridging_windows_only"], xjtu_sens["clean_windows_only"],
                                xjtu_sens["all_xjtu"])
        macro("XjtuBridgingRMSERatio", bridge["rmse_z"] / clean["rmse_z"], "{:.1f}")
        macro("XjtuBridgingMAEShiftPercent", 100 * (allx["mae_z"] - clean["mae_z"]) / clean["mae_z"], "{:.1f}")
        macro("XjtuBridgingCoverage", bridge["coverage"] * 100, "{:.1f}")
        macro("XjtuCleanCoverage", clean["coverage"] * 100, "{:.1f}")

    with open(os.path.join("paper", "numbers.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved paper/numbers.tex ({len(lines)-3} macros)")

    n_unverified = sum(1 for line in lines if "UNVERIFIED" in line)
    print(f"\n{n_unverified} macros still UNVERIFIED (expected while LODO/other background jobs finish).")


if __name__ == "__main__":
    main()
