"""
Stage 3: model/checkpoint reproduction check.

Does NOT retrain anything. Loads the saved production checkpoint
(outputs/conformal_groupwise_model.pt) and the cache it was trained
against (cache/processed.npz), runs a single DETERMINISTIC forward pass
(predict_in_batches -- same deterministic path train.py's predictor-
consistency fix uses for calibration) over the test set, and checks that
the resulting MAE/RMSE match what outputs/conformal_groupwise_results.json
reports, directly verifying the checkpoint file actually IS the model that
produced the reported numbers (not a stale/mismatched save), and that
build_model()'s architecture reconstruction is reproducible.

NOTE: the officially-reported test_mae/test_rmse in conformal_groupwise_
results.json come from the MC-DROPOUT MEAN predictor (mc_dropout_predict,
stochastic -- see train.py), not the deterministic forward pass used here.
This check therefore does NOT expect bit-exact equality to those reported
numbers; it expects them to be CLOSE (MC-Dropout mean should be near the
deterministic prediction when dropout is light) and, more importantly,
verifies the checkpoint loads and runs with the exact right architecture/
input shapes -- a basic sanity check that would fail loudly (shape
mismatch, dict key mismatch) if the checkpoint were stale or built with
different model config than what's currently in model.py/config.py.

Run:  python tests/stage3_checkpoint_reproduction.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model import build_model, predict_in_batches, mc_dropout_predict
from train import load_cached

RUN_NAME = "conformal_groupwise"


def main():
    cache_path = os.path.join(config.CACHE_DIR, "processed.npz")
    ckpt_path = os.path.join(config.OUTPUT_DIR, f"{RUN_NAME}_model.pt")
    results_path = os.path.join(config.OUTPUT_DIR, f"{RUN_NAME}_results.json")

    for p in (cache_path, ckpt_path, results_path):
        if not os.path.exists(p):
            print(f"  [!] {p} not found -- cannot run this check.")
            return

    with open(results_path) as fh:
        reported = json.load(fh)

    data = load_cached(cache_path)
    X_test, y_test = data["X_test"], data["y_test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(reported.get("model_variant", "full")).to(device)

    state_dict_loaded = False
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        state_dict_loaded = True
    except Exception as e:
        print(f"  [FAIL] checkpoint failed to load against current build_model() architecture: {e}")
        return
    print(f"  [PASS] checkpoint loaded cleanly against current build_model('{reported.get('model_variant','full')}')")

    model.eval()
    x_test_t = torch.tensor(X_test).to(device)
    det_pred = predict_in_batches(model, x_test_t).cpu().numpy()
    det_mae = float(np.mean(np.abs(y_test - det_pred)))
    det_rmse = float(np.sqrt(np.mean((y_test - det_pred) ** 2)))

    mc_mean, _ = mc_dropout_predict(model, x_test_t)
    mc_mean = mc_mean.cpu().numpy()
    mc_mae = float(np.mean(np.abs(y_test - mc_mean)))
    mc_rmse = float(np.sqrt(np.mean((y_test - mc_mean) ** 2)))

    print(f"\n{'':20s}{'MAE':>10s}{'RMSE':>10s}")
    print(f"{'reported (mc_mean)':20s}{reported['test_mae']:10.5f}{reported['test_rmse']:10.5f}")
    print(f"{'reproduced mc_mean':20s}{mc_mae:10.5f}{mc_rmse:10.5f}")
    print(f"{'reproduced det.':20s}{det_mae:10.5f}{det_rmse:10.5f}")

    # MC-Dropout is stochastic (new random forward passes each call), so an
    # exact match to the ORIGINAL training run's mc_mean is not expected --
    # but it should be close (same checkpoint, same calibration_predictor
    # convention, many MC samples averaging out most of the noise).
    mae_rel_diff = abs(mc_mae - reported["test_mae"]) / reported["test_mae"]
    print(f"\nreproduced-vs-reported MC-mean MAE relative difference: {mae_rel_diff:.4f}")

    close_enough = mae_rel_diff < 0.10  # generous -- MC-Dropout is stochastic
    print(f"  [{'PASS' if close_enough else 'FAIL'}] reproduced MC-mean MAE within 10% of reported "
          f"(checkpoint + cache + architecture are mutually consistent)")

    if not close_enough:
        sys.exit(1)


if __name__ == "__main__":
    main()
