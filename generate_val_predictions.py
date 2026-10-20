"""
MC-Dropout predictions on the validation set, saved for the analyses that recalibrate without
retraining.

train.py calibrates its intervals on validation predictions but saves only the test predictions.
This script reloads the main model (outputs/conformal_groupwise_model.pt), runs the same
MC-Dropout inference train.py uses (model.mc_dropout_predict) on every validation window, and
saves the result. hierarchical_calibration_v2.py, mwhc_calibration.py,
calibration_size_sensitivity.py, run_nested_hp_search_all.py and
predictor_consistent_recalibration.py all calibrate on this file.

MC-Dropout is random, so a rerun reproduces the saved file closely but not bit-for-bit; the
quantiles it gives differ from the ones train.py computed in its own pass by Monte Carlo noise
only. Every analysis that reads this file takes both its calibration and its test residuals from
the same pass of its own, so its comparisons are internally consistent.

Requires: cache/processed.npz, outputs/conformal_groupwise_model.pt.
Output:   outputs/conformal_groupwise_predictions_val.npz
          (y_val, mean_pred, mc_std, lab_val, cell_val; values in per-dataset z-units)

Run:  python generate_val_predictions.py
"""
import os

import numpy as np
import torch

import config
from model import build_model, mc_dropout_predict

CHECKPOINT = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_model.pt")
OUTPUT = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    X_val = data["X_val"]

    model = build_model("full").to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    mean_pred, mc_std = mc_dropout_predict(model, torch.tensor(X_val).to(device))

    np.savez(OUTPUT, y_val=data["y_val"], mean_pred=mean_pred.cpu().numpy(), mc_std=mc_std.cpu().numpy(),
             lab_val=data["lab_val"], cell_val=data["cell_val"])
    print(f"saved {OUTPUT} ({len(X_val)} validation windows)")

    # Per-dataset MAE in real SOH units, as a sanity check against the test-set MAEs.
    soh_mean, soh_std = data["soh_mean"].item(), data["soh_std"].item()
    for lab in np.unique(data["lab_val"]):
        m = data["lab_val"] == lab
        mae = np.mean(np.abs(data["y_val"][m] - mean_pred.cpu().numpy()[m])) * soh_std[lab]
        print(f"  {lab:7s} n={m.sum():6d}  validation MAE (SOH units) = {mae:.5f}")


if __name__ == "__main__":
    main()
