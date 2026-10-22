"""
The deep ensemble of deep_ensemble.py, repeated on the four further cell-level splits.

deep_ensemble.py reports the ensemble on the main split only. This script asks whether its result
there (normalized-groupwise calibration around the ensemble is the closest to the 90% target)
holds on the other splits. For each further split it builds the same five-member ensemble: the
seed-42 member is that split's existing full model (outputs/split{s}_groupwise_model.pt, trained by
repeated_split_experiment.py with model-init seed 42), and the members with seeds 123, 2024, 7 and
99 are trained here with the same settings if their checkpoints are missing. The MC-Dropout
reference on each split is that same seed-42 model's saved MC-Dropout predictions. Calibration uses
deep_ensemble.calibrate_all, unchanged.

Run one split's training (several can run side by side on one GPU; --seed trains one member):
    python deep_ensemble_splits.py --split 43 [--seed 99]
Then, with no arguments, compute every split's results and the summary (trains anything missing):
    python deep_ensemble_splits.py

Requires: cache/processed_split{43..46}.npz, outputs/split{s}_groupwise_model.pt,
          outputs/split{s}_groupwise_predictions{,_val}.npz, deep_ensemble.json (main split).
Output:   outputs/deep_ensemble_splits.json
"""
import argparse
import json
import os

import numpy as np
import torch

import config
from deep_ensemble import FIVE, calibrate_all
from model import build_model, predict_in_batches
from train import train_one

SPLITS = [43, 44, 45, 46]
EXTRA_SEEDS = [123, 2024, 7, 99]
MODES = ["raw_gaussian", "pooled", "groupwise", "normalized_groupwise"]
OUTPUT = os.path.join(config.OUTPUT_DIR, "deep_ensemble_splits.json")


def cache_for(split):
    return os.path.join(config.CACHE_DIR, f"processed_split{split}.npz")


def member_checkpoint(split, seed):
    name = f"split{split}_groupwise" if seed == config.RANDOM_SEED else f"split{split}_ens_seed{seed}"
    return name, os.path.join(config.OUTPUT_DIR, f"{name}_model.pt")


def train_split(split, seeds=EXTRA_SEEDS):
    for seed in seeds:
        name, path = member_checkpoint(split, seed)
        if os.path.exists(path):
            print(f"[split {split}] member {name} exists")
            continue
        print(f"[split {split}] training member {name}")
        r = train_one(cache_path=cache_for(split), model_variant="full", seed=seed,
                      conformal_mode="groupwise", run_name=name, verbose=False)
        print(f"[split {split}] member {name}: test MAE {r['test_mae']:.5f}, best epoch {r['best_epoch']}")


def evaluate_split(split, device):
    data = np.load(cache_for(split), allow_pickle=True)
    X_val, y_val, lab_val = data["X_val"], data["y_val"], data["lab_val"]
    X_test, y_test, lab_test = data["X_test"], data["y_test"], data["lab_test"]

    members_val, members_test = [], []
    for seed in [config.RANDOM_SEED] + EXTRA_SEEDS:
        name, path = member_checkpoint(split, seed)
        if not os.path.exists(path):
            train_split(split)
        model = build_model("full").to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        members_val.append(predict_in_batches(model, torch.tensor(X_val).to(device)).cpu().numpy())
        members_test.append(predict_in_batches(model, torch.tensor(X_test).to(device)).cpu().numpy())
    members_val, members_test = np.stack(members_val), np.stack(members_test)

    mc_val = np.load(os.path.join(config.OUTPUT_DIR, f"split{split}_groupwise_predictions_val.npz"),
                     allow_pickle=True)
    mc_test = np.load(os.path.join(config.OUTPUT_DIR, f"split{split}_groupwise_predictions.npz"),
                      allow_pickle=True)
    assert np.array_equal(mc_test["y_test"], y_test) and np.array_equal(mc_val["y_val"], y_val)

    return {
        "deep_ensemble": calibrate_all(y_val, members_val.mean(0), members_val.std(0), lab_val,
                                       y_test, members_test.mean(0), members_test.std(0), lab_test),
        "mc_dropout": calibrate_all(y_val, mc_val["mean_pred"], mc_val["mc_std"], lab_val,
                                    y_test, mc_test["mean_pred"], mc_test["mc_std"], lab_test),
        "member_test_mae": [float(np.mean(np.abs(y_test - p))) for p in members_test],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=SPLITS)
    parser.add_argument("--seed", type=int, choices=EXTRA_SEEDS, help="with --split: train this member only")
    args = parser.parse_args()
    if args.split is not None:
        train_split(args.split, [args.seed] if args.seed is not None else EXTRA_SEEDS)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(config.OUTPUT_DIR, "deep_ensemble.json")) as fh:
        main_split = json.load(fh)
    per_split = {str(config.RANDOM_SEED): {k: main_split[k] for k in ("deep_ensemble", "mc_dropout")}}
    for split in SPLITS:
        per_split[str(split)] = evaluate_split(split, device)

    # For every split: which (source, calibration) is closest to the target on the five datasets.
    candidates = [(src, mode) for src in ("deep_ensemble", "mc_dropout") for mode in MODES]
    summary = {}
    for src, mode in candidates:
        devs = [per_split[s][src][mode]["mean_abs_dev_five"] for s in per_split]
        widths = [per_split[s][src][mode]["mean_width"] for s in per_split]
        summary[f"{src}/{mode}"] = {"mean_dev_five": float(np.mean(devs)), "min_dev_five": float(np.min(devs)),
                                    "max_dev_five": float(np.max(devs)), "mean_width": float(np.mean(widths))}
    best = {s: min(candidates, key=lambda c: per_split[s][c[0]][c[1]]["mean_abs_dev_five"]) for s in per_split}
    ens_norm_best = sum(b == ("deep_ensemble", "normalized_groupwise") for b in best.values())
    ens_norm_beats_mc_gw = sum(per_split[s]["deep_ensemble"]["normalized_groupwise"]["mean_abs_dev_five"]
                               < per_split[s]["mc_dropout"]["groupwise"]["mean_abs_dev_five"] for s in per_split)
    ens_norm_narrower_mc_gw = sum(per_split[s]["deep_ensemble"]["normalized_groupwise"]["mean_width"]
                                  < per_split[s]["mc_dropout"]["groupwise"]["mean_width"] for s in per_split)
    ens_more_accurate = sum(per_split[s]["deep_ensemble"]["test_mae"] < per_split[s]["mc_dropout"]["test_mae"]
                            for s in per_split)
    ens_pooled_bit = [per_split[s]["deep_ensemble"]["pooled"]["per_dataset_coverage"]["BIT"] for s in per_split]
    ens_gw_bit = [per_split[s]["deep_ensemble"]["groupwise"]["per_dataset_coverage"]["BIT"] for s in per_split]
    ens_gw_beats_pooled = sum(per_split[s]["deep_ensemble"]["groupwise"]["mean_abs_dev_five"]
                              < per_split[s]["deep_ensemble"]["pooled"]["mean_abs_dev_five"] for s in per_split)

    results = {
        "splits": list(per_split), "five": FIVE, "per_split": per_split, "summary": summary,
        "best_per_split": {s: "/".join(b) for s, b in best.items()},
        "ens_norm_best_splits": ens_norm_best,
        "ens_norm_beats_mc_groupwise_splits": ens_norm_beats_mc_gw,
        "ens_norm_narrower_than_mc_groupwise_splits": ens_norm_narrower_mc_gw,
        "ens_more_accurate_splits": ens_more_accurate,
        "ens_groupwise_beats_pooled_splits": ens_gw_beats_pooled,
        "ens_pooled_bit_range": [float(min(ens_pooled_bit)), float(max(ens_pooled_bit))],
        "ens_groupwise_bit_range": [float(min(ens_gw_bit)), float(max(ens_gw_bit))],
    }
    with open(OUTPUT, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"{'source/calibration':36s}{'mean dev5':>10s}{'min':>8s}{'max':>8s}{'width':>9s}")
    for k, v in summary.items():
        print(f"{k:36s}{v['mean_dev_five'] * 100:10.2f}{v['min_dev_five'] * 100:8.2f}"
              f"{v['max_dev_five'] * 100:8.2f}{v['mean_width']:9.4f}")
    for s, b in results["best_per_split"].items():
        print(f"split {s}: closest = {b}")
    print(f"ensemble + normalized-groupwise closest in {ens_norm_best} of {len(per_split)} splits; "
          f"beats MC groupwise in {ens_norm_beats_mc_gw}; narrower than MC groupwise in {ens_norm_narrower_mc_gw}")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
