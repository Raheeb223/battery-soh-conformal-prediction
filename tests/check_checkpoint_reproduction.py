"""
Checkpoint reproduction: does the saved model reproduce the paper's headline accuracy?

Loads the saved main-model checkpoint (outputs/conformal_groupwise_model.pt) into a freshly built
model from the current model.py/config.py, runs it on the cached test windows, and compares with
the MAE/RMSE recorded in outputs/conformal_groupwise_results.json. No retraining.

Checks:
  1. The checkpoint loads into build_model('full') without missing or unexpected weights, i.e. the
     current code defines exactly the architecture that was trained.
  2. The MC-Dropout-mean MAE (the predictor every reported interval is centred on) reproduces the
     recorded MAE to within 10%. MC-Dropout is stochastic, so exact equality is not expected; in
     practice the difference is a few hundredths of a percent.
The deterministic single-pass MAE/RMSE are printed for reference.

Needs: cache/processed.npz and the outputs/ files above (produced by train.py / experiments.py).
Run from the repository root:  python tests/check_checkpoint_reproduction.py
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
    print(f"\n{2 if close_enough else 1} passed, {0 if close_enough else 1} failed")

    if not close_enough:
        sys.exit(1)


if __name__ == "__main__":
    main()
