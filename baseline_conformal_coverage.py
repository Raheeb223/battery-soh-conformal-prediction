"""
K3 fix (ERROR_REGISTER E2): pooled and groupwise conformal coverage for the
non-BiLSTM baselines (linear, SVR, XGBoost, plain LSTM), computed under the
same predictor-consistency requirement as train.py (calibration and test
nonconformity scores from the same predictor).

- linear / SVR / XGBoost: trained fresh via baselines.run_baseline(), which
  now also saves validation predictions (ERROR_REGISTER E2 fix). No
  retraining ambiguity here since these are fast, deterministic sklearn/
  XGBoost fits (predict() has no train/eval-mode distinction to mismatch).
- plain LSTM: NOT retrained. Loads the existing outputs/ablation_plain_lstm_model.pt
  checkpoint and computes predictor-consistent (mc_dropout_predict for both
  calibration and test, matching train.py's default) pooled AND groupwise
  coverage directly. Groupwise coverage for this variant already exists in
  outputs/ablation_plain_lstm_results.json (computed by train.py, already
  predictor-consistent) and is reused rather than recomputed, for exactness;
  only the POOLED number is new here, since experiments.py's ablation section
  only ran groupwise mode for architecture variants.

Run:  python baseline_conformal_coverage.py
Out:  outputs/baseline_conformal_results.json
"""
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import config
from baselines import run_baseline, flatten
from conformal import SplitConformal, GroupwiseConformal
from model import build_model, mc_dropout_predict
from train import load_cached

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


def sklearn_conformal_coverage(name, model, max_train_samples=None):
    """Train (or reuse this turn's fit), predict val+test, calibrate pooled
    and groupwise conformal on the validation residuals, evaluate coverage
    on the test set. No MC-Dropout concept applies to these models, so the
    predictor is simply model.predict() for both calibration and test by
    construction -- no mismatch is possible here."""
    result = run_baseline(name, model, max_train_samples=max_train_samples)
    npz = np.load(os.path.join(config.OUTPUT_DIR, f"baseline_{name}_predictions.npz"), allow_pickle=True)
    y_val, val_pred, lab_val = npz["y_val"], npz["val_pred"], npz["lab_val"]
    y_test, test_pred, lab_test = npz["y_test"], npz["mean_pred"], npz["lab_test"]

    pooled = SplitConformal(alpha=config.CONFORMAL_ALPHA)
    pooled.calibrate(y_val, val_pred)
    pooled_cov = pooled.coverage(y_test, test_pred)
    lo, hi = pooled.predict_interval(test_pred)
    pooled_per_ds = {}
    for g in np.unique(lab_test):
        m = lab_test == g
        pooled_per_ds[str(g)] = float(np.mean((y_test[m] >= lo[m]) & (y_test[m] <= hi[m])))

    group = GroupwiseConformal(alpha=config.CONFORMAL_ALPHA)
    group.calibrate(y_val, val_pred, lab_val)
    group_per_ds = group.coverage_by_group(y_test, test_pred, lab_test)
    group_lo, group_hi = group.predict_interval(test_pred, lab_test)
    group_overall = float(np.mean((y_test >= group_lo) & (y_test <= group_hi)))

    return {
        "architecture": name,
        "test_mae": result["test_mae"], "test_rmse": result["test_rmse"],
        "pooled_overall_coverage": pooled_cov,
        "pooled_per_dataset_coverage": {k: float(v) for k, v in pooled_per_ds.items()},
        "groupwise_overall_coverage": group_overall,
        "groupwise_per_dataset_coverage": {str(k): float(v) for k, v in group_per_ds.items()},
    }


def ablation_checkpoint_conformal_coverage(variant_name, model_variant):
    """No retrain: loads an EXISTING ablation checkpoint (saved by train_one()
    via experiments.py), recomputes calibration in pooled mode (predictor-
    consistent, mc_dropout_predict for both val and test -- matching
    train.py's default). Groupwise coverage is read from the existing,
    already predictor-consistent outputs/ablation_{variant}_results.json
    rather than recomputed, to avoid introducing a second, possibly-non-
    identical source of truth for a number that already exists correctly.
    Used for plain_lstm and gru (Stage 5's K3 scope beyond SVR/XGBoost)."""
    ckpt_path = os.path.join(config.OUTPUT_DIR, f"ablation_{variant_name}_model.pt")
    existing_path = os.path.join(config.OUTPUT_DIR, f"ablation_{variant_name}_results.json")
    if not (os.path.exists(ckpt_path) and os.path.exists(existing_path)):
        print(f"[{variant_name}] checkpoint or existing results missing -- skipping. "
              f"({ckpt_path}, {existing_path})")
        return None

    with open(existing_path) as fh:
        existing = json.load(fh)
    if existing.get("calibration_predictor") != "mc_mean":
        print(f"[{variant_name}] WARNING: existing results' calibration_predictor="
              f"{existing.get('calibration_predictor')!r}, expected 'mc_mean' -- "
              f"groupwise number below may not be predictor-consistent with the pooled "
              f"number computed here. Flagged, not silently used.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_variant).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    data = load_cached()
    X_val, y_val, lab_val = data["X_val"], data["y_val"], data["lab_val"]
    X_test, y_test, lab_test = data["X_test"], data["y_test"], data["lab_test"]

    val_pred, _ = mc_dropout_predict(model, torch.tensor(X_val).to(device))
    val_pred = val_pred.cpu().numpy()
    test_pred, _ = mc_dropout_predict(model, torch.tensor(X_test).to(device))
    test_pred = test_pred.cpu().numpy()

    pooled = SplitConformal(alpha=config.CONFORMAL_ALPHA)
    pooled.calibrate(y_val, val_pred)
    pooled_cov = pooled.coverage(y_test, test_pred)
    lo, hi = pooled.predict_interval(test_pred)
    pooled_per_ds = {}
    for g in np.unique(lab_test):
        m = lab_test == g
        pooled_per_ds[str(g)] = float(np.mean((y_test[m] >= lo[m]) & (y_test[m] <= hi[m])))

    mae = float(np.mean(np.abs(y_test - test_pred)))
    rmse = float(np.sqrt(np.mean((y_test - test_pred) ** 2)))

    return {
        "architecture": variant_name,
        "test_mae": mae, "test_rmse": rmse,
        "pooled_overall_coverage": pooled_cov,
        "pooled_per_dataset_coverage": pooled_per_ds,
        "groupwise_overall_coverage": existing["overall_coverage"],
        "groupwise_per_dataset_coverage": existing["per_dataset_coverage"],
        "groupwise_source": f"reused from outputs/ablation_{variant_name}_results.json (not recomputed)",
        "no_retrain": True,
    }


def main():
    out = {}
    out["linear"] = sklearn_conformal_coverage("linear", LinearRegression())
    out["svr"] = sklearn_conformal_coverage("svr", SVR(kernel="rbf", C=1.0, epsilon=0.01), max_train_samples=20000)
    if _HAS_XGBOOST:
        xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=config.RANDOM_SEED, n_jobs=-1)
        out["xgboost"] = sklearn_conformal_coverage("xgboost", xgb_model)
    else:
        print("[xgboost] skipped -- not installed")

    lstm_result = ablation_checkpoint_conformal_coverage("plain_lstm", "plain_lstm")
    if lstm_result is not None:
        out["plain_lstm"] = lstm_result

    gru_result = ablation_checkpoint_conformal_coverage("gru", "gru")
    if gru_result is not None:
        out["gru"] = gru_result

    print("\n=== Summary: pooled vs groupwise coverage ===")
    for arch, r in out.items():
        print(f"{arch}: pooled={r['pooled_overall_coverage']:.2%}  groupwise={r['groupwise_overall_coverage']:.2%}")

    out_path = os.path.join(config.OUTPUT_DIR, "baseline_conformal_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
