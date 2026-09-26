"""
Generate validation-set predictions using this project's own model.py and
mc_dropout_predict() function (the same one train.py uses for the test set),
rather than a generic reimplementation -- this guarantees consistency with
however conformal_groupwise_predictions.npz was originally produced.

Loads the conformal_groupwise_model.pt checkpoint saved by train_one() (run
via experiments.py's conformal-ablation section) and runs MC-Dropout
inference over the cached validation windows, saving the result to
conformal_groupwise_predictions_val.npz. hgc_cp_evaluation_harness.py and
few_shot_calibration.py read this file for calibration residuals.

USAGE: python generate_val_predictions.py
"""

import os

import numpy as np
import torch

import config
from model import build_model, mc_dropout_predict

PROCESSED_NPZ_PATH = os.path.join(config.CACHE_DIR, "processed.npz")
MODEL_CHECKPOINT_PATH = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_model.pt")
OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz")
MODEL_VARIANT = "full"  # matches train_one()'s default, and this project's reported configuration


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {PROCESSED_NPZ_PATH}...")
    data = np.load(PROCESSED_NPZ_PATH, allow_pickle=True)
    X_val = data["X_val"]
    y_val = data["y_val"]
    lab_val = data["lab_val"]
    cell_val = data["cell_val"] if "cell_val" in data.files else None
    print(f"Loaded X_val: {X_val.shape}")

    print(f"Building model (variant='{MODEL_VARIANT}') and loading checkpoint...")
    model = build_model(MODEL_VARIANT).to(device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()

    print(f"Running MC-Dropout inference on {len(X_val)} validation windows...")
    mean_pred, mc_std = mc_dropout_predict(model, torch.tensor(X_val).to(device))
    mean_pred = mean_pred.cpu().numpy()
    mc_std = mc_std.cpu().numpy()

    print(f"Saving to {OUTPUT_PATH}...")
    payload = dict(y_val=y_val, mean_pred=mean_pred, mc_std=mc_std, lab_val=lab_val)
    if cell_val is not None:
        payload["cell_val"] = cell_val
    np.savez(OUTPUT_PATH, **payload)
    print("Done.")

    print("\nPer-dataset sanity check, de-normalized to real SOH units (y_val/mean_pred")
    print("are stored in the model's native per-dataset z-scored space, so raw MAE here")
    print("is not directly comparable to previously-reported real-unit test MAE):")
    soh_mean = data["soh_mean"].item()
    soh_std = data["soh_std"].item()
    for lab in np.unique(lab_val):
        mask = lab_val == lab
        m, s = soh_mean[lab], soh_std[lab]
        y_real = y_val[mask] * s + m
        pred_real = mean_pred[mask] * s + m
        mae_real = np.mean(np.abs(y_real - pred_real))
        print(f"  {lab}: n={mask.sum()}, MAE (real SOH units)={mae_real:.5f}")


if __name__ == "__main__":
    main()
