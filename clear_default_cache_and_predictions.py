"""
clear_default_cache_and_predictions.py

Final-mile fix. diagnose_load_predictions.py proved ablation_full's and
conformal_normalized_groupwise's predictions.npz files are NOT stale
(single match each, recent timestamps) yet still contain the XJTU
contamination (2387 samples, 79 at exactly -7.224). The reason: Sections
2 and 3 of experiments.py never call preprocessing.run() themselves --
they only read whatever's already sitting at cache/processed.npz via
train_one()'s load_cached(). Only Section 4 (window-size ablation) calls
preprocessing.run() directly, which is why ITS log showed the
"[XJTU] ... removed N calibration-block rows" messages this run and
Section 2/3 did not -- they never touched the loader at all.

This script deletes:
  1. cache/processed.npz -- the DEFAULT (window=20) preprocessing cache
     used by ablation_full, conformal_normalized_groupwise, baselines,
     the other ablations/conformal modes, and multiseed runs. NOT
     touched by clear_stale_xjtu_cache.py or clear_stale_predictions.py
     before now.
  2. ablation_full_results.json / ablation_full_predictions.npz
  3. conformal_normalized_groupwise_results.json /
     conformal_normalized_groupwise_predictions.npz

It does NOT touch: window-size-specific caches (processed_w10/30/50.npz
-- those were already confirmed fresh, window_30/50 retrained this run
due to the NASA signature check), the other 4 ablations, the other 2
conformal modes, baselines, multiseed runs, or LODO results -- nothing
in the evidence so far suggests those are stale.

IMPORTANT: after running this, EVERYTHING that depends on
cache/processed.npz will be forced to retrain the next time you run
experiments.py, not just ablation_full and conformal_normalized_groupwise
-- because the cache file itself is gone, not just those two runs'
results. That includes: baseline_linear, baseline_svr, baseline_xgboost,
ablation_no_attention, ablation_no_multiscale, ablation_plain_lstm,
ablation_gru, conformal_groupwise, conformal_pooled, and multiseed_42/
123/2024. Their existing *_results.json files will still be present,
but _load_existing_result()'s signature check only compares dataset
NAMES, not cache freshness, so it will likely still resume them with
their OLD (pre-fix, if they were ever wrong) numbers -- UNLESS you
also want to force everything to retrain for full certainty. See the
--nuke-everything flag below.

Usage:
    python clear_default_cache_and_predictions.py --dry-run
    python clear_default_cache_and_predictions.py
    python clear_default_cache_and_predictions.py --yes
    python clear_default_cache_and_predictions.py --nuke-everything
        (also deletes every OTHER *_results.json / *_predictions.npz in
        config.OUTPUT_DIR tied to the default window=20 cache -- i.e.
        everything except the window_10/20/30/50 and lodo_* runs, which
        manage their own caches independently. Use this if you want
        maximum certainty rather than narrow targeting.)
"""

import os
import sys
import glob

import config  # your existing project config.py

NARROW_RUN_NAMES = ["ablation_full", "conformal_normalized_groupwise"]

# Every run that reads cache/processed.npz via load_cached() with no
# window override -- i.e. everything EXCEPT window_* (manages its own
# cache) and lodo_* (builds its own cache via build_lodo_cache(), never
# touches cache/processed.npz at all).
WIDE_RUN_NAMES = NARROW_RUN_NAMES + [
    "baseline_linear", "baseline_svr", "baseline_xgboost",
    "ablation_no_attention", "ablation_no_multiscale",
    "ablation_plain_lstm", "ablation_gru",
    "conformal_groupwise", "conformal_pooled",
    "multiseed_42", "multiseed_123", "multiseed_2024",
]


def _default_cache_path():
    return os.path.join(config.CACHE_DIR, "processed.npz")


def _find_targets(run_names):
    output_dir = config.OUTPUT_DIR
    targets = [_default_cache_path()]

    if not os.path.isdir(output_dir):
        print(f"  [!] config.OUTPUT_DIR ({output_dir}) not found.")
    else:
        for run_name in run_names:
            targets += sorted(glob.glob(os.path.join(output_dir, f"*{run_name}*predictions.npz")))
            json_path = os.path.join(output_dir, f"{run_name}_results.json")
            if os.path.exists(json_path):
                targets.append(json_path)

    # keep only files that actually exist, de-duped, order preserved
    seen = set()
    out = []
    for t in targets:
        if os.path.exists(t) and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main():
    dry_run = "--dry-run" in sys.argv
    skip_confirm = "--yes" in sys.argv or dry_run
    nuke = "--nuke-everything" in sys.argv

    run_names = WIDE_RUN_NAMES if nuke else NARROW_RUN_NAMES
    mode_label = "WIDE (--nuke-everything)" if nuke else "narrow (ablation_full + conformal_normalized_groupwise only)"
    targets = _find_targets(run_names)

    if not targets:
        print("Nothing found to delete -- check config.CACHE_DIR / config.OUTPUT_DIR "
              "if this is unexpected.")
        return

    print(f"Mode: {mode_label}")
    print(f"\n{'Would delete' if dry_run else 'Will delete'} {len(targets)} file(s):")
    for t in targets:
        print(f"  {t}")

    if dry_run:
        print("\n[dry run] nothing deleted. Re-run without --dry-run to actually clear these.")
        return

    if not skip_confirm:
        resp = input(f"\nDelete these {len(targets)} file(s)? This forces a full "
                      f"preprocessing + retrain of everything that depends on the "
                      f"default cache. [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted -- nothing deleted.")
            return

    n_deleted = 0
    n_failed = 0
    for t in targets:
        try:
            os.remove(t)
            n_deleted += 1
        except OSError as e:
            print(f"  [!] failed to delete {t}: {e}")
            n_failed += 1

    print(f"\nDeleted {n_deleted} file(s)" + (f", {n_failed} failed" if n_failed else "") + ".")
    print("\nNext steps:")
    print("  1. python preprocessing.py")
    print("     -- watch for '[XJTU] ... removed N calibration-block rows' lines")
    print("        in THIS run's output to directly confirm the fresh cache is clean.")
    print("  2. python experiments.py")
    print("     -- ablation_full and conformal_normalized_groupwise (and, if you used "
          "--nuke-everything, everything else in Sections 1/2/3/5) will retrain on "
          "the confirmed-clean cache.")
    print("  3. python diagnose_load_predictions.py")
    print("     -- confirm the XJTU sample count / value range are no longer "
          "2387 / [-7.224, 1.033].")


if __name__ == "__main__":
    main()
