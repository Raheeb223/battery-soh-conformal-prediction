"""
Generate validation-set predictions using this project's OWN model.py and
mc_dropout_predict() function directly (seen in train.py), rather than a
generic reimplementation -- this guarantees consistency with however
conformal_groupwise_predictions.npz was actually produced, since it uses the
exact same inference function train_one() itself calls for the test set.

CONFIRM BEFORE RUNNING: the checkpoint path below is inferred from train.py's
naming convention ({run_name}_model.pt) and your predictions file being named
conformal_groupwise_predictions.npz, implying run_name="conformal_groupwise".
This has NOT been verified against your actual outputs folder. Run:

    Get-ChildItem -Path C:\\Users\\User\\PycharmProjects\\outputs -Filter *.pt

and confirm conformal_groupwise_model.pt is really there before trusting this
script's output -- if the filename differs, update MODEL_CHECKPOINT_PATH
below to match.

USAGE: python generate_val_predictions.py
"""

import numpy as np
import torch

from model import build_model, mc_dropout_predict

PROCESSED_NPZ_PATH = r"C:\Users\User\PycharmProjects\cache\processed.npz"
MODEL_CHECKPOINT_PATH = r"C:\Users\User\PycharmProjects\outputs\conformal_groupwise_model.pt"  # CONFIRM this exists
OUTPUT_PATH = r"C:\Users\User\PycharmProjects\outputs\conformal_groupwise_predictions_val.npz"
MODEL_VARIANT = "full"  # matches train_one()'s default -- confirm this is what was actually used


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

    print(f"Running MC-Dropout inference on {len(X_val)} validation windows "
          f"(using this project's own mc_dropout_predict, same as train.py uses for test)...")
    mean_pred, mc_std = mc_dropout_predict(model, torch.tensor(X_val).to(device))
    mean_pred = mean_pred.cpu().numpy()
    mc_std = mc_std.cpu().numpy()

    print(f"Saving to {OUTPUT_PATH}...")
    payload = dict(y_val=y_val, mean_pred=mean_pred, mc_std=mc_std, lab_val=lab_val)
    if cell_val is not None:
        payload["cell_val"] = cell_val
    np.savez(OUTPUT_PATH, **payload)
    print("Done. Upload this file back and I'll wire load_calibration_residuals() to it.")

    print("\nPer-dataset sanity check, DE-NORMALIZED to real SOH units (y_val/mean_pred")
    print("are stored in the model's native per-dataset z-scored space -- comparing")
    print("raw MAE directly against previously-reported real-unit test MAE would be")
    print("an apples-to-oranges units mismatch, not a real problem):")
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