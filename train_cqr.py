"""
CQR baseline: Conformalized Quantile Regression (Romano, Patterson & Candes,
2019, "Conformalized Quantile Regression", NeurIPS).

WHY THIS COMPARISON MATTERS. Every uncertainty method compared elsewhere in
this paper (pooled/groupwise/normalized-groupwise/hierarchical/HGC-CP/MWHC)
starts from the SAME point-prediction model and the SAME residual-based
nonconformity score |y - yhat|, varying only how the calibration quantile is
computed across datasets. CQR is architecturally different: the model is
trained to predict two QUANTILES directly (via the pinball/quantile loss,
not MSE), so the interval can already be asymmetric and vary in width per
sample before any conformal correction is applied -- the correction only
fixes up whatever residual miscalibration the quantile regression itself
has. Comparing against it answers a different question than the ablations
in experiments.py: not "does architecture X change coverage" (holding the
residual-based calibration method fixed), but "does training for quantiles
directly, instead of a point + separate calibration step, help at all for
this problem" -- the single most-cited specific method this project's
heterogeneity-aware calibration ideas have not yet been benchmarked against.

MODEL. The exact same multi-scale BiLSTM-with-attention backbone as the
main model (model.MultiScaleBiLSTMAttention), with its final head widened
from 1 output to 2 (q_lo, q_hi) via the n_outputs parameter -- same
architecture, same hyperparameters, same data split, so any difference in
outcome is attributable to the quantile-loss training objective and CQR's
calibration, not to a confounding architecture change.

LOSS. Pinball (quantile) loss at tau_lo = alpha/2 and tau_hi = 1 - alpha/2,
summed:
    L_tau(y, q) = mean( max(tau*(y-q), (tau-1)*(y-q)) )
Trained with early stopping on validation pinball loss, same optimizer and
patience convention as train.py.

CALIBRATION. conformal.CQRSplitConformal / CQRGroupwiseConformal (added
alongside this script -- see conformal.py), both evaluated: nonconformity
score E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)), conformalized the same
way SplitConformal/GroupwiseConformal are, which is what gives CQR its
marginal coverage guarantee (not a heuristic -- the split-conformal
exchangeability argument applies directly to this score).

Requires: cache/processed.npz (same cache train.py uses), config.py.
Run:  python train_cqr.py
Out:  outputs/cqr_pooled_results.json, outputs/cqr_groupwise_results.json,
      outputs/cqr_predictions.npz, outputs/cqr_model.pt
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config
from model import MultiScaleBiLSTMAttention, predict_in_batches
from conformal import CQRSplitConformal, CQRGroupwiseConformal

ALPHA = getattr(config, "CONFORMAL_ALPHA", 0.10)
TAU_LO, TAU_HI = ALPHA / 2, 1 - ALPHA / 2


def pinball_loss(y, q, tau):
    diff = y - q
    return torch.mean(torch.maximum(tau * diff, (tau - 1) * diff))


def to_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def main(seed=config.RANDOM_SEED, run_name="cqr"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{run_name}] using device: {device}, quantile levels: tau_lo={TAU_LO}, tau_hi={TAU_HI}")

    data = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    lab_val = data["lab_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    lab_test = data["lab_test"]

    train_loader = to_loader(X_train, y_train, config.BATCH_SIZE, shuffle=True)
    val_loader = to_loader(X_val, y_val, config.BATCH_SIZE, shuffle=False)

    model = MultiScaleBiLSTMAttention(use_multiscale=True, use_attention=True, n_outputs=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = pinball_loss(yb, pred[:, 0], TAU_LO) + pinball_loss(yb, pred[:, 1], TAU_HI)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                val_losses.append((pinball_loss(yb, pred[:, 0], TAU_LO)
                                    + pinball_loss(yb, pred[:, 1], TAU_HI)).item())

        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        print(f"[{run_name}][epoch {epoch:03d}] train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"[{run_name}] early stopping at epoch {epoch} (best val_loss={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f"{run_name}_model.pt"))

    # ---- Quantile predictions (deterministic -- CQR's uncertainty IS the
    # quantile spread, not an MC-Dropout estimate layered on top) ----
    model.eval()
    val_pred = predict_in_batches(model, torch.tensor(X_val).to(device)).cpu().numpy()
    test_pred = predict_in_batches(model, torch.tensor(X_test).to(device)).cpu().numpy()
    q_lo_val, q_hi_val = val_pred[:, 0], val_pred[:, 1]
    q_lo_test, q_hi_test = test_pred[:, 0], test_pred[:, 1]

    # A quantile crossing (q_lo > q_hi) is possible with independent pinball
    # losses and no monotonicity constraint -- sort per sample so the
    # reported "lo"/"hi" are always ordered, same remedy as the original CQR
    # paper's reference implementation.
    for lo, hi in ((q_lo_val, q_hi_val), (q_lo_test, q_hi_test)):
        swap = lo > hi
        if swap.any():
            lo[swap], hi[swap] = hi[swap].copy(), lo[swap].copy()
    n_crossed_val = int((q_lo_val > q_hi_val).sum())  # should be 0 post-sort; sanity only
    print(f"[{run_name}] quantile crossings fixed by sorting -- val/test now monotonic (sanity check: {n_crossed_val} remaining)")

    point_pred_test = (q_lo_test + q_hi_test) / 2  # for MAE/RMSE comparability with other methods
    mae = float(np.mean(np.abs(y_test - point_pred_test)))
    rmse = float(np.sqrt(np.mean((y_test - point_pred_test) ** 2)))

    for mode, Conf in (("pooled", CQRSplitConformal), ("groupwise", CQRGroupwiseConformal)):
        conf = Conf(alpha=ALPHA)
        if mode == "pooled":
            conf.calibrate(y_val, q_lo_val, q_hi_val)
            lower, upper = conf.predict_interval(q_lo_test, q_hi_test)
            cov_by_group = {}
            for name in np.unique(lab_test):
                m = lab_test == name
                cov_by_group[name] = float(np.mean((y_test[m] >= lower[m]) & (y_test[m] <= upper[m])))
        else:
            conf.calibrate(y_val, q_lo_val, q_hi_val, lab_val)
            lower, upper = conf.predict_interval(q_lo_test, q_hi_test, lab_test)
            cov_by_group = conf.coverage_by_group(y_test, q_lo_test, q_hi_test, lab_test)

        overall_coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
        mpiw = float(np.mean(upper - lower))
        print(f"[{run_name}:{mode}][test] overall coverage={overall_coverage:.2%}  MPIW={mpiw:.5f}  MAE={mae:.5f}")
        for name in cov_by_group:
            print(f"  {name}: coverage={cov_by_group[name]:.2%}")

        results = {
            "run_name": f"{run_name}_{mode}", "model_variant": "cqr", "conformal_mode": mode,
            "seed": seed, "best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "tau_lo": TAU_LO, "tau_hi": TAU_HI, "test_mae": mae, "test_rmse": rmse,
            "overall_coverage": overall_coverage, "mpiw": mpiw, "per_dataset_coverage": cov_by_group,
            "dataset_signature": sorted(np.unique(lab_test).tolist()),
        }
        with open(os.path.join(config.OUTPUT_DIR, f"cqr_{mode}_results.json"), "w") as fh:
            json.dump(results, fh, indent=2)

        if mode == "groupwise":  # save one predictions file (groupwise is this method's headline mode)
            np.savez(os.path.join(config.OUTPUT_DIR, "cqr_predictions.npz"),
                     y_test=y_test, q_lo_test=q_lo_test, q_hi_test=q_hi_test,
                     lower=lower, upper=upper, lab_test=lab_test)

    print(f"\n[{run_name}] saved outputs/cqr_pooled_results.json, outputs/cqr_groupwise_results.json, "
          f"outputs/cqr_predictions.npz, outputs/cqr_model.pt")


if __name__ == "__main__":
    main()
