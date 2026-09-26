"""
Training engine.

train_one() trains one model variant on the cached windows, with early
stopping on validation loss, then:
  - computes MC-Dropout predictions on the test set,
  - calibrates split conformal intervals on the validation cells
    (conformal_mode = 'groupwise', 'pooled' or 'normalized_groupwise'),
  - reports test MAE/RMSE, overall and per-dataset coverage, and per-dataset MAE
    (computed on the per-dataset standardised SOH targets),
  - saves outputs/<run_name>_results.json, _predictions.npz and _model.pt.

Run order:
    1. python inspect_data.py       (check raw file structures)
    2. python preprocessing.py      (parse + cache windowed arrays)
    3. python train.py              (trains the default 'full' model once)

For baselines, ablations, multi-seed runs and leave-one-dataset-out testing use
experiments.py, which calls train_one() with different settings.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config
from model import build_model, mc_dropout_predict, predict_in_batches
from conformal import GroupwiseConformal, SplitConformal, NormalizedGroupwiseConformal


def load_cached(cache_path=None):
    path = cache_path or os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run preprocessing.py first to generate {path}")
    return np.load(path, allow_pickle=True)


def to_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one(cache_path=None, model_variant="full", seed=config.RANDOM_SEED,
              conformal_mode="groupwise", run_name=None, save_outputs=True,
              verbose=True):
    """Trains one model and returns a results dict. This is the single engine
    every experiment (baseline comparison, ablation, multi-seed, LODO) calls —
    only the arguments differ between experiments, not the training logic.

    conformal_mode: 'groupwise' calibrates a separate qhat per dataset (the
    default, fixes the small-group coverage problem). 'pooled' calibrates one
    qhat across all groups together — used for the
    pooled-vs-groupwise ablation.
    'normalized_groupwise' additionally scales each sample's interval width
    by that sample's own MC-Dropout uncertainty (still calibrated per-group
    for validity) — unlike the other two modes, this gives genuinely
    per-sample-varying interval widths instead of one fixed width per
    dataset, which is what makes interval_width_explanation.py's
    feature-attribution analysis possible at all.
    """
    set_seed(seed)
    run_name = run_name or f"{model_variant}_seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[train:{run_name}] using device: {device}")

    data = load_cached(cache_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    lab_val = data["lab_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    lab_test = data["lab_test"]
    # Per-window cell IDs, added in preprocessing.py's make_windows() so
    # per_cell_curves.png and remaining_analysis.py's anomaly cell-lookup
    # can name the specific cell a test window belongs to, instead of only
    # the index range ("No cell-ID field available" in the old
    # xjtu_anomaly_report.txt). Older cache files won't have this key —
    # fall back to None so this still runs against a stale cache rather
    # than crashing, but flag it loudly since it silently loses that
    # traceability.
    if "cell_test" in data.files:
        cell_test = data["cell_test"]
    else:
        cell_test = None
        if verbose:
            print(f"[train:{run_name}] [!] WARNING: this cache has no 'cell_test' "
                  f"field (built with an older preprocessing.py) -- the saved "
                  f"predictions file will have NO per-window cell-ID tracking. "
                  f"Re-run preprocessing.py with the current version to fix this.")

    if len(X_train) == 0:
        raise RuntimeError("Training set is empty — check preprocessing.py output/cell counts.")

    train_loader = to_loader(X_train, y_train, config.BATCH_SIZE, shuffle=True)
    val_loader = to_loader(X_val, y_val, config.BATCH_SIZE, shuffle=False)

    model = build_model(model_variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()

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
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        if verbose:
            print(f"[{run_name}][epoch {epoch:03d}] train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                if verbose:
                    print(f"[{run_name}] early stopping at epoch {epoch} (best val_loss={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)

    if save_outputs:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f"{run_name}_model.pt"))

    # ---- Conformal calibration on the VALIDATION set (never on test) ----
    # For 'groupwise'/'pooled' a deterministic point prediction on val is enough.
    # For 'normalized_groupwise' we additionally need MC-Dropout uncertainty on
    # val (not just test), since the calibration step normalizes residuals by
    # each val sample's own mc_std.
    model.eval()
    val_preds = predict_in_batches(model, torch.tensor(X_val).to(device))
    val_preds = val_preds.cpu().numpy()

    if conformal_mode == "groupwise":
        conformal = GroupwiseConformal(alpha=config.CONFORMAL_ALPHA)
        conformal.calibrate(y_val, val_preds, lab_val)
        predict_interval = lambda pred, std: conformal.predict_interval(pred, lab_test)
        coverage_by_group = lambda pred, std: conformal.coverage_by_group(y_test, pred, lab_test)
    elif conformal_mode == "pooled":
        conformal = SplitConformal(alpha=config.CONFORMAL_ALPHA)
        conformal.calibrate(y_val, val_preds)
        predict_interval = lambda pred, std: conformal.predict_interval(pred)
        def coverage_by_group(pred, std):
            lower, upper = conformal.predict_interval(pred)
            out = {}
            for name in np.unique(lab_test):
                mask = lab_test == name
                out[name] = float(np.mean((y_test[mask] >= lower[mask]) & (y_test[mask] <= upper[mask])))
            return out
    elif conformal_mode == "normalized_groupwise":
        val_mean_mc, val_std_mc = mc_dropout_predict(model, torch.tensor(X_val).to(device))
        val_std_mc = val_std_mc.cpu().numpy()
        conformal = NormalizedGroupwiseConformal(alpha=config.CONFORMAL_ALPHA)
        conformal.calibrate(y_val, val_preds, val_std_mc, lab_val)
        predict_interval = lambda pred, std: conformal.predict_interval(pred, std, lab_test)
        coverage_by_group = lambda pred, std: conformal.coverage_by_group(y_test, pred, std, lab_test)
    else:
        raise ValueError(f"Unknown conformal_mode '{conformal_mode}'")

    # ---- Final test evaluation, with both MC-Dropout and conformal intervals ----
    mean_pred, mc_std = mc_dropout_predict(model, torch.tensor(X_test).to(device))
    mean_pred = mean_pred.cpu().numpy()
    mc_std_np = mc_std.cpu().numpy()
    lower, upper = predict_interval(mean_pred, mc_std_np)
    coverage_by_grp = coverage_by_group(mean_pred, mc_std_np)
    overall_coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))

    mae = float(np.mean(np.abs(y_test - mean_pred)))
    rmse = float(np.sqrt(np.mean((y_test - mean_pred) ** 2)))

    per_dataset_mae = {}
    for name in np.unique(lab_test):
        mask = lab_test == name
        per_dataset_mae[name] = float(np.mean(np.abs(y_test[mask] - mean_pred[mask])))

    if verbose:
        print(f"\n[{run_name}][test] overall MAE={mae:.5f}  RMSE={rmse:.5f}")
        print(f"[{run_name}][test] overall coverage @ {1 - config.CONFORMAL_ALPHA:.0%} target: {overall_coverage:.2%}")
        for name in per_dataset_mae:
            print(f"  {name}: MAE={per_dataset_mae[name]:.5f}  coverage={coverage_by_grp.get(name, float('nan')):.2%}")

    results = {
        "run_name": run_name,
        "model_variant": model_variant,
        "seed": seed,
        "conformal_mode": conformal_mode,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_mae": mae,
        "test_rmse": rmse,
        "overall_coverage": overall_coverage,
        "per_dataset_mae": per_dataset_mae,
        "per_dataset_coverage": coverage_by_grp,
        # datasets present in this run's test set — lets a resume mechanism
        # detect "this result was computed on a DIFFERENT dataset composition
        # than what's currently configured" and force a retrain instead of
        # silently reusing a stale result (this is exactly the bug that
        # produced a summary mixing 4-dataset and 7-dataset results after
        # BIT/XJTU/RWTH were added but old result files weren't invalidated).
        "dataset_signature": sorted(np.unique(lab_test).tolist()),
    }

    if save_outputs:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(config.OUTPUT_DIR, f"{run_name}_results.json"), "w") as fh:
            json.dump(results, fh, indent=2)

        predictions_payload = dict(
            y_test=y_test, mean_pred=mean_pred, mc_std=mc_std.cpu().numpy(),
            lower=lower, upper=upper, lab_test=lab_test,
        )
        # Only add cell_test if this cache actually had it (see the
        # cell_test-loading block above) -- keeps this working against
        # older caches instead of crashing, just without the per-window
        # cell-ID field in that case.
        if cell_test is not None:
            predictions_payload["cell_test"] = cell_test
        np.savez(os.path.join(config.OUTPUT_DIR, f"{run_name}_predictions.npz"),
                 **predictions_payload)

    return results


def train():
    """Backward-compatible entry point: trains the default full model once,
    exactly like before. Run `python train.py` for this."""
    return train_one(run_name="best")


if __name__ == "__main__":
    train()