"""
Fine-tuning experiment — the honest attempt to fix the leave-one-dataset-out
(LODO) generalization failure.

LODO showed the model fails badly on an entirely unseen dataset (e.g. CALCE
held out -> MAE 1.08). The practical question this experiment answers:

   "If a new battery type arrives with only a FEW labeled cells, can we
    briefly fine-tune a model pre-trained on other datasets and recover
    usable accuracy — instead of needing to train from scratch?"

This mirrors a real deployment workflow: you have a model trained on existing
chemistries, a new chemistry shows up with little data, and you want to adapt
quickly.

Method:
  1. Pre-train on the 3 non-held-out datasets (same as LODO).
  2. Take a SMALL number of cells from the held-out dataset (default: half of
     them, but never more than `max_finetune_cells`) as fine-tuning data.
  3. Fine-tune the pre-trained model briefly on just those cells.
  4. Test on the REMAINING held-out cells (which were never used for
     fine-tuning) — so this is still an honest held-out test.

Reports MAE before fine-tuning (= cold LODO) vs after, so the improvement
(or lack of it) is explicit. IMPORTANT: whatever this shows — big recovery,
small recovery, or none — is the honest result. Fine-tuning is not guaranteed
to help, especially for the smallest datasets; if it doesn't, that itself is
a finding worth reporting.

Run:
    python finetune.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config
import preprocessing
from model import build_model, mc_dropout_predict


def _windows_for_cells(df, cells, soh_mean, soh_std, window):
    return preprocessing.make_windows(df, set(cells), soh_mean, soh_std, window=window)


def finetune_one(held_out_dataset: str, max_finetune_cells=4,
                 finetune_epochs=30, finetune_lr=None, model_variant="full",
                 save_outputs=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"finetune_{held_out_dataset}"
    w = config.INPUT_WINDOW
    ft_lr = finetune_lr if finetune_lr is not None else config.LEARNING_RATE * 0.1

    raw = preprocessing.load_all_raw()
    df = preprocessing.add_soh(raw)

    other_df = df[df["dataset"] != held_out_dataset]
    held_df = df[df["dataset"] == held_out_dataset]
    held_cells = sorted(held_df["cell_id"].unique())

    if len(held_cells) < 2:
        print(f"[{run_name}] held-out dataset has <2 cells — can't split into "
              f"fine-tune + test. Skipping.")
        return None

    # split held-out cells: some for fine-tuning, the REST for testing
    rng = np.random.RandomState(config.RANDOM_SEED)
    rng.shuffle(held_cells)
    n_ft = min(max_finetune_cells, len(held_cells) - 1)  # always leave >=1 for test
    finetune_cells = held_cells[:n_ft]
    test_cells = held_cells[n_ft:]
    print(f"[{run_name}] {len(held_cells)} held-out cells -> "
          f"{len(finetune_cells)} for fine-tuning, {len(test_cells)} for testing")

    # normalization: pre-training datasets from their own train stats;
    # held-out dataset normalized from its FINE-TUNE cells only (what you'd
    # actually have access to in the real workflow — not the test cells)
    soh_mean = other_df.groupby("dataset")["soh"].mean().to_dict()
    soh_std = other_df.groupby("dataset")["soh"].std().to_dict()
    ft_df = held_df[held_df["cell_id"].isin(finetune_cells)]
    soh_mean[held_out_dataset] = ft_df["soh"].mean()
    soh_std[held_out_dataset] = ft_df["soh"].std()
    for k in soh_std:
        if soh_std[k] == 0 or np.isnan(soh_std[k]):
            soh_std[k] = 1.0

    # ---- pre-train on the 3 other datasets ----
    other_cells = other_df["cell_id"].unique()
    X_pre, y_pre, _ = _windows_for_cells(other_df, other_cells, soh_mean, soh_std, w)
    if len(X_pre) == 0:
        raise RuntimeError(f"[{run_name}] no pre-training windows built")

    model = build_model(model_variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    pre_loader = DataLoader(TensorDataset(torch.tensor(X_pre), torch.tensor(y_pre)),
                            batch_size=config.BATCH_SIZE, shuffle=True)

    print(f"[{run_name}] pre-training on {len(X_pre)} windows from 3 datasets...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        for xb, yb in pre_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        # simple fixed-length pre-training (no early stopping here to keep the
        # before/after comparison clean); cap at a reasonable number of epochs
        if epoch >= 40:
            break

    # ---- measure COLD performance on test cells (before fine-tuning) ----
    X_test, y_test, _ = _windows_for_cells(held_df, test_cells, soh_mean, soh_std, w)
    if len(X_test) == 0:
        raise RuntimeError(f"[{run_name}] no test windows built for held-out cells")

    def evaluate():
        mean_pred, _ = mc_dropout_predict(model, torch.tensor(X_test).to(device))
        mp = mean_pred.cpu().numpy()
        return float(np.mean(np.abs(y_test - mp))), float(np.sqrt(np.mean((y_test - mp) ** 2)))

    cold_mae, cold_rmse = evaluate()
    print(f"[{run_name}] COLD (no fine-tuning): MAE={cold_mae:.5f}  RMSE={cold_rmse:.5f}")

    # ---- fine-tune on the small held-out fine-tune set ----
    X_ft, y_ft, _ = _windows_for_cells(held_df, finetune_cells, soh_mean, soh_std, w)
    if len(X_ft) == 0:
        print(f"[{run_name}] no fine-tune windows built — held-out cells too short "
              f"for window={w}. Reporting cold result only.")
        warmed_mae, warmed_rmse = cold_mae, cold_rmse
    else:
        ft_optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr)
        ft_loader = DataLoader(TensorDataset(torch.tensor(X_ft), torch.tensor(y_ft)),
                               batch_size=config.BATCH_SIZE, shuffle=True)
        print(f"[{run_name}] fine-tuning on {len(X_ft)} windows for {finetune_epochs} epochs...")
        model.train()
        for _ in range(finetune_epochs):
            for xb, yb in ft_loader:
                xb, yb = xb.to(device), yb.to(device)
                ft_optimizer.zero_grad()
                pred, _ = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                ft_optimizer.step()
        warmed_mae, warmed_rmse = evaluate()

    print(f"[{run_name}] AFTER fine-tuning: MAE={warmed_mae:.5f}  RMSE={warmed_rmse:.5f}")
    improvement = (cold_mae - warmed_mae) / cold_mae * 100 if cold_mae > 0 else 0.0
    print(f"[{run_name}] MAE change: {improvement:+.1f}% "
          f"({'improved' if warmed_mae < cold_mae else 'no improvement / worse'})")

    results = {
        "run_name": run_name,
        "held_out_dataset": held_out_dataset,
        "n_finetune_cells": len(finetune_cells),
        "n_test_cells": len(test_cells),
        "cold_mae": cold_mae,
        "cold_rmse": cold_rmse,
        "finetuned_mae": warmed_mae,
        "finetuned_rmse": warmed_rmse,
        "mae_improvement_pct": improvement,
    }
    if save_outputs:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(config.OUTPUT_DIR, f"{run_name}_results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
    return results


def run_all_finetune(datasets=("NASA", "CALCE", "Oxford", "MIT")):
    results = []
    for d in datasets:
        r = finetune_one(d)
        if r is not None:
            results.append(r)
    print("\n[finetune] summary (cold -> fine-tuned MAE):")
    for r in results:
        print(f"  {r['held_out_dataset']}: {r['cold_mae']:.4f} -> {r['finetuned_mae']:.4f} "
              f"({r['mae_improvement_pct']:+.1f}%)")
    return results


if __name__ == "__main__":
    run_all_finetune()
