"""
Leave-one-dataset-out (LODO) generalization test.

This is different from your normal train/val/test split: instead of mixing
cells from all 4 datasets into every split, this holds out ONE ENTIRE
DATASET for testing and trains only on the other 3. It directly answers:
"does this model generalize to a battery chemistry/lab it has never seen
during training?" — which is a much stronger and more meaningful claim than
your normal cell-independent split (where the model has at least seen SOME
cells from every dataset).

Expect LODO MAE to be noticeably worse than your normal test MAE — that's
not a failure, that's the honest, expected result of testing true
generalization instead of interpolation. Report both numbers side by side.

--- UNITS FIX (added) ---
The held-out dataset's SOH values are z-scored using ITS OWN mean/std
(soh_mean[held_out_dataset], soh_std[held_out_dataset]) rather than any
other dataset's stats -- this is correct and necessary to avoid leakage
(see build_lodo_cache()'s comment). But it means MAE/RMSE computed
directly on y_test/mean_pred in that normalized space are in DIFFERENT
UNITS for every held-out dataset: a dataset with a small soh_std gets its
errors inflated relative to a dataset with a large soh_std, purely from
the local scale, not from the model actually predicting worse. This
made CALCE's LODO MAE (1.13, larger than SOH's entire [0,1.10] range)
look like a broken/impossible result when it was actually a units
artifact -- CALCE's held-out soh_std is small relative to the other six
datasets, so the same real-world capacity-fade error gets amplified in
CALCE's local normalized units.

run_lodo() now reports BOTH the raw normalized-space MAE/RMSE (kept for
debugging/comparison against older logged runs) and de-normalized
MAE/RMSE (multiplied back by soh_std[held_out_dataset]) in real SOH
units, which ARE comparable across the seven held-out-dataset rows.
Report the de-normalized numbers in the manuscript's LODO table.
Coverage is unaffected by this issue -- it's a hit-rate, not a
magnitude, so it was already comparable across rows.

Run:
    python leave_one_out.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config
import preprocessing
from model import build_model, mc_dropout_predict, predict_in_batches
from conformal import SplitConformal


def build_lodo_cache(held_out_dataset: str, window=None, raw_df=None):
    """Loads raw data, then splits by DATASET rather than by cell:
    - train/val: all cells from the 3 datasets NOT held out (further split by cell
      into train/val so we still get early stopping)
    - test: ALL cells from the held-out dataset
    Returns the same dict shape as preprocessing's cache so train_one-style
    logic can reuse it directly, PLUS the held-out dataset's own soh_std so
    callers can de-normalize error metrics back into real SOH units.

    raw_df: pre-loaded, pre-SOH'd dataframe (preprocessing.add_soh(load_all_raw())),
    reused across all 7 held-out datasets by run_all_lodo() so raw data (CALCE's
    366+ files, RWTH's 1600+ CSVs, etc.) is parsed ONCE rather than 7 times --
    the raw data is identical across LODO runs; only which dataset is held out
    changes. Pass None to load fresh (used when calling this function standalone)."""
    w = window if window is not None else config.INPUT_WINDOW

    if raw_df is None:
        raw = preprocessing.load_all_raw()
        df = preprocessing.add_soh(raw)
    else:
        df = raw_df

    test_df = df[df["dataset"] == held_out_dataset]
    train_val_df = df[df["dataset"] != held_out_dataset]
    if test_df.empty:
        raise ValueError(f"No rows found for held-out dataset '{held_out_dataset}' — "
                          f"check the name matches exactly what preprocessing.py prints.")

    # split the remaining 3 datasets' cells into train/val by CELL (never split
    # a cell's own cycles across train/val) so early stopping still works
    rng = np.random.RandomState(config.RANDOM_SEED)
    train_cells, val_cells = [], []
    cell_table = train_val_df[["dataset", "cell_id"]].drop_duplicates()
    for dataset_name, group in cell_table.groupby("dataset"):
        ids = group["cell_id"].tolist()
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * 0.15))
        val_cells.extend(ids[:n_val])
        train_cells.extend(ids[n_val:])

    train_df = train_val_df[train_val_df["cell_id"].isin(train_cells)]
    val_df = train_val_df[train_val_df["cell_id"].isin(val_cells)]

    # normalization stats from TRAIN cells of the 3 non-held-out datasets only
    soh_mean = train_df.groupby("dataset")["soh"].mean().to_dict()
    soh_std = train_df.groupby("dataset")["soh"].std().to_dict()
    for k in soh_std:
        if soh_std[k] == 0 or np.isnan(soh_std[k]):
            soh_std[k] = 1.0
    # the held-out dataset needs its OWN normalization stats computed from
    # ITS OWN data, since the model has never seen it — using train-set stats
    # from other datasets would leak information the model doesn't actually have
    # access to in a true generalization setting, or silently break normalization
    # if the held-out dataset's SOH scale differs from the others.
    soh_mean[held_out_dataset] = test_df["soh"].mean()
    soh_std[held_out_dataset] = test_df["soh"].std()
    if soh_std[held_out_dataset] == 0 or np.isnan(soh_std[held_out_dataset]):
        soh_std[held_out_dataset] = 1.0

    # make_windows() returns 5 values: X, y, labels, cell_labels, cycle_windows.
    # (This comment previously said 4 -- that was stale and is what caused a
    # "too many values to unpack" crash here; cycle_windows was added to
    # preprocessing.py for windowing_bridge_sensitivity.py's bridge-detection
    # check well before this comment was last updated.) LODO doesn't currently
    # use per-window cell IDs or cycle indices anywhere downstream, so both are
    # captured here (cell_train/cell_val/cell_test, cyc_train/cyc_val/cyc_test)
    # for parity with the cache build_lodo_cache() returns below, but not
    # otherwise acted on.
    X_train, y_train, lab_train, cell_train, cyc_train = preprocessing.make_windows(train_df, set(train_cells), soh_mean, soh_std, window=w)
    X_val, y_val, lab_val, cell_val, cyc_val = preprocessing.make_windows(val_df, set(val_cells), soh_mean, soh_std, window=w)
    all_test_cells = set(test_df["cell_id"].unique())
    X_test, y_test, lab_test, cell_test, cyc_test = preprocessing.make_windows(test_df, all_test_cells, soh_mean, soh_std, window=w)

    return {
        "X_train": X_train, "y_train": y_train, "lab_train": lab_train, "cell_train": cell_train,
        "window_cycle_indices_train": cyc_train,
        "X_val": X_val, "y_val": y_val, "lab_val": lab_val, "cell_val": cell_val,
        "window_cycle_indices_val": cyc_val,
        "X_test": X_test, "y_test": y_test, "lab_test": lab_test, "cell_test": cell_test,
        "window_cycle_indices_test": cyc_test,
        # held-out dataset's own std, needed to de-normalize error metrics
        # back into real, cross-dataset-comparable SOH units
        "held_out_soh_std": float(soh_std[held_out_dataset]),
    }


def run_lodo(held_out_dataset: str, model_variant="full", save_outputs=True, raw_df=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"lodo_{held_out_dataset}"
    print(f"[{run_name}] holding out '{held_out_dataset}' entirely for testing...")

    data = build_lodo_cache(held_out_dataset, raw_df=raw_df)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    held_out_soh_std = data["held_out_soh_std"]
    print(f"[{run_name}] held-out dataset's own soh_std = {held_out_soh_std:.5f} "
          f"(used to de-normalize error metrics into real SOH units)")

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError(f"[{run_name}] empty train or test set — check dataset name and windowing.")

    model = train_lodo_model(data, run_name, model_variant, device)
    return evaluate_lodo(model, data, run_name, held_out_dataset, model_variant, device, save_outputs)


def train_lodo_model(data, run_name, model_variant="full", device=None):
    """The LODO training loop: Adam on MSE with early stopping on the other datasets' validation
    cells. lodo_target_calibration.py reuses it to retrain and save the LODO models."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = build_model(model_variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                val_losses.append(criterion(pred, yb).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"[{run_name}] early stopping at epoch {epoch} (best val_loss={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)
    return model


def evaluate_lodo(model, data, run_name, held_out_dataset, model_variant, device, save_outputs):
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    held_out_soh_std = data["held_out_soh_std"]

    # simple pooled conformal here — the held-out dataset IS the only test
    # group, so per-group calibration would be identical to pooled anyway
    #
    # PREDICTOR CONSISTENCY (fix for ERROR_REGISTER E1 / brief K1): split-conformal
    # validity requires calibration and test nonconformity scores to come from the
    # SAME predictor. This previously calibrated on predict_in_batches() (dropout
    # off) but evaluated on mc_dropout_predict() (dropout on, MC-mean) -- a
    # predictor mismatch. Now both use the same predictor, matching train.py's
    # default (config.CALIBRATION_PREDICTOR, defaulting to "mc_mean").
    model.eval()
    calibration_predictor = getattr(config, "CALIBRATION_PREDICTOR", "mc_mean")
    if calibration_predictor == "mc_mean":
        val_preds, _ = mc_dropout_predict(model, torch.tensor(X_val).to(device))
        val_preds = val_preds.cpu().numpy()
    elif calibration_predictor == "deterministic":
        val_preds = predict_in_batches(model, torch.tensor(X_val).to(device))
        val_preds = val_preds.cpu().numpy()
    else:
        raise ValueError("config.CALIBRATION_PREDICTOR must be 'mc_mean' or 'deterministic'")
    conformal = SplitConformal(alpha=config.CONFORMAL_ALPHA)
    conformal.calibrate(y_val, val_preds)

    mean_pred, _ = mc_dropout_predict(model, torch.tensor(X_test).to(device))
    mean_pred = mean_pred.cpu().numpy()
    # coverage is a hit-rate (is y_test inside the predicted interval?), not a
    # magnitude, so it's already invariant to the held-out dataset's local
    # normalization scale -- no de-normalization needed here.
    coverage = conformal.coverage(y_test, mean_pred)

    # MAE/RMSE computed directly on y_test/mean_pred are in the held-out
    # dataset's own normalized (z-scored) units -- NOT comparable across
    # different held-out datasets, since each one has a different soh_std.
    mae_normalized = float(np.mean(np.abs(y_test - mean_pred)))
    rmse_normalized = float(np.sqrt(np.mean((y_test - mean_pred) ** 2)))

    # De-normalize: true_soh = normalized*std + mean, so an error in
    # normalized space scales by std when converted back to real SOH units
    # (the mean cancels out in a difference). This makes MAE/RMSE directly
    # comparable across all seven held-out-dataset rows in the LODO table.
    mae = mae_normalized * held_out_soh_std
    rmse = rmse_normalized * held_out_soh_std

    print(f"[{run_name}] held-out test MAE={mae:.5f}  RMSE={rmse:.5f}  coverage={coverage:.2%}  "
          f"(real SOH units; normalized-space MAE={mae_normalized:.5f}, RMSE={rmse_normalized:.5f})")

    results = {
        "run_name": run_name,
        "held_out_dataset": held_out_dataset,
        "model_variant": model_variant,
        "test_mae": mae,
        "test_rmse": rmse,
        # kept for transparency / debugging against older logged runs, which
        # reported these normalized-space numbers as if they were "test_mae"
        "test_mae_normalized_space": mae_normalized,
        "test_rmse_normalized_space": rmse_normalized,
        "held_out_soh_std": held_out_soh_std,
        "calibration_predictor": calibration_predictor,
        # Coverage is a hit-rate (is y_test inside the interval?): applying the
        # SAME affine de-normalization to y_test and to [lower, upper] preserves
        # the inequality, so coverage is numerically identical in z-units and
        # real SOH units. Reported once rather than duplicated as two equal
        # numbers; see ERROR_REGISTER.md note on this under E1.
        "coverage": coverage,
        "n_test_samples": int(len(y_test)),
        # datasets actually used for TRAINING in this LODO run (everything
        # except the held-out one) — lets a resume mechanism detect if the
        # available training datasets changed since this ran (e.g. a new
        # dataset was added later) and force a re-run rather than silently
        # reusing a result trained on a smaller set of datasets.
        "training_dataset_signature": sorted(np.unique(data["lab_train"]).tolist()),
    }
    if save_outputs:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(config.OUTPUT_DIR, f"{run_name}_results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
    return results


def run_all_lodo(datasets=("NASA", "CALCE", "Oxford", "MIT", "BIT", "XJTU", "RWTH")):
    """Loads and SOH-processes raw data ONCE and reuses it for all 7 held-out
    runs (see build_lodo_cache's raw_df docstring) -- the original version
    called preprocessing.load_all_raw() fresh inside each run_lodo() call,
    re-parsing identical raw data (CALCE, RWTH, XJTU, ...) seven times over."""
    print("[run_all_lodo] loading and SOH-processing raw data once for all "
          f"{len(datasets)} held-out configurations...")
    raw = preprocessing.load_all_raw()
    raw_df = preprocessing.add_soh(raw)
    return [run_lodo(d, raw_df=raw_df) for d in datasets]


if __name__ == "__main__":
    run_all_lodo()