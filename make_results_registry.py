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
    registry["sections"]["xjtu_bridging_isolation"] = load("xjtu_bridging_isolation.json") or UNVERIFIED
    registry["sections"]["xjtu_bridging_sensitivity"] = load("xjtu_bridging_sensitivity.json") or UNVERIFIED
    registry["sections"]["rwth_verification"] = load("rwth_verification_v2.json") or UNVERIFIED

    # ---- Write registry ----
    with open("results_registry.json", "w") as fh:
        json.dump(registry, fh, indent=2, default=str)
    print(f"saved results_registry.json ({len(registry['sections'])} sections)")

    # ---- Emit paper/numbers.tex ----
    os.makedirs("paper", exist_ok=True)
    lines = ["% AUTO-GENERATED by make_results_registry.py -- do not hand-edit.",
             "% Every number in main.tex must come from a \\NUM{...} macro defined here.", ""]

    def macro(name, value, fmt="{:.4f}"):
        safe_name = re.sub(r"[^A-Za-z]", "", name)  # LaTeX macro names: letters only
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

    with open(os.path.join("paper", "numbers.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved paper/numbers.tex ({len(lines)-3} macros)")

    n_unverified = sum(1 for line in lines if "UNVERIFIED" in line)
    print(f"\n{n_unverified} macros still UNVERIFIED (expected while LODO/other background jobs finish).")


if __name__ == "__main__":
    main()
