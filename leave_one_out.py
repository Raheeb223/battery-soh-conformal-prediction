"""
Leave-one-dataset-out (LODO) generalisation test.

Unlike the main cell-independent split (where every dataset contributes cells
to training), LODO holds out one entire dataset for testing and trains on the
other six. It measures transfer to a battery chemistry / lab / protocol never
seen during training, so LODO errors are expected to be larger than the main
test errors.

Units: the held-out dataset is z-scored with its own mean/std
(soh_mean[held_out], soh_std[held_out]) to avoid leakage from the training
datasets' statistics. Errors computed in that normalised space are therefore
in different units for each held-out dataset. run_lodo() reports both:
  - test_mae / test_rmse: de-normalised (multiplied by soh_std[held_out]),
    i.e. in raw SOH units, comparable across held-out datasets;
  - test_mae_normalized_space / test_rmse_normalized_space: in the held-out
    dataset's normalised units.
Coverage is a hit rate and is unaffected by the choice of units.

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


def build_lodo_cache(held_out_dataset: str, window=None):
    """Loads raw data, then splits by DATASET rather than by cell:
    - train/val: all cells from the 3 datasets NOT held out (further split by cell
      into train/val so we still get early stopping)
    - test: ALL cells from the held-out dataset
    Returns the same dict shape as preprocessing's cache so train_one-style
    logic can reuse it directly, PLUS the held-out dataset's own soh_std so
    callers can de-normalize error metrics back into real SOH units."""
    w = window if window is not None else config.INPUT_WINDOW

    raw = preprocessing.load_all_raw()
    df = preprocessing.add_soh(raw)

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

    # make_windows() now returns 4 values (X, y, labels, cell_labels) --
    # the cell_labels field was added for per_cell_curves.png boundary
    # markers and the XJTU anomaly cell-lookup. LODO doesn't currently
    # use per-window cell IDs anywhere downstream, so they're captured
    # here (cell_train/cell_val/cell_test) for parity with the cache
    # build_lodo_cache() returns below, but not otherwise acted on.
    X_train, y_train, lab_train, cell_train = preprocessing.make_windows(train_df, set(train_cells), soh_mean, soh_std, window=w)
    X_val, y_val, lab_val, cell_val = preprocessing.make_windows(val_df, set(val_cells), soh_mean, soh_std, window=w)
    all_test_cells = set(test_df["cell_id"].unique())
    X_test, y_test, lab_test, cell_test = preprocessing.make_windows(test_df, all_test_cells, soh_mean, soh_std, window=w)

    return {
        "X_train": X_train, "y_train": y_train, "lab_train": lab_train, "cell_train": cell_train,
        "X_val": X_val, "y_val": y_val, "lab_val": lab_val, "cell_val": cell_val,
        "X_test": X_test, "y_test": y_test, "lab_test": lab_test, "cell_test": cell_test,
        # held-out dataset's own std, needed to de-normalize error metrics
        # back into real, cross-dataset-comparable SOH units
        "held_out_soh_std": float(soh_std[held_out_dataset]),
    }


def run_lodo(held_out_dataset: str, model_variant="full", save_outputs=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"lodo_{held_out_dataset}"
    print(f"[{run_name}] holding out '{held_out_dataset}' entirely for testing...")

    data = build_lodo_cache(held_out_dataset)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    held_out_soh_std = data["held_out_soh_std"]
    print(f"[{run_name}] held-out dataset's own soh_std = {held_out_soh_std:.5f} "
          f"(used to de-normalize error metrics into real SOH units)")

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError(f"[{run_name}] empty train or test set — check dataset name and windowing.")

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

    # simple pooled conformal here — the held-out dataset IS the only test
    # group, so per-group calibration would be identical to pooled anyway
    model.eval()
    val_preds = predict_in_batches(model, torch.tensor(X_val).to(device))
    val_preds = val_preds.cpu().numpy()
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
    return [run_lodo(d) for d in datasets]


if __name__ == "__main__":
    run_all_lodo()