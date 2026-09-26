"""
Classical (non-deep) baselines evaluated on exactly the same windowed,
normalised data as the neural models (same cache file, same train/val/test
cells, same per-dataset normalisation).

Baselines:
  - linear   : linear regression on the flattened window
  - svr      : support vector regression (RBF kernel) on the flattened window
  - xgboost  : gradient-boosted trees on the flattened window

Point-prediction MAE/RMSE only; conformal calibration of these baselines is
done separately in baseline_conformal_calibration.py.

Run:
    python baselines.py
"""

import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

import config
from train import load_cached


def flatten(X):
    """(N, window, 1) -> (N, window) for sklearn models that expect 2D input."""
    return X.reshape(X.shape[0], -1)


def run_baseline(name: str, model, cache_path=None, save_outputs=True, max_train_samples=None):
    data = load_cached(cache_path)
    X_train, y_train = flatten(data["X_train"]), data["y_train"]
    X_test, y_test = flatten(data["X_test"]), data["y_test"]
    lab_test = data["lab_test"]

    if max_train_samples is not None and X_train.shape[0] > max_train_samples:
        rng = np.random.RandomState(config.RANDOM_SEED)
        idx = rng.choice(X_train.shape[0], size=max_train_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"[baseline:{name}] subsampled training set to {max_train_samples} "
              f"samples (was {data['X_train'].shape[0]}) for tractability")

    print(f"[baseline:{name}] training on {X_train.shape[0]} samples...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = float(np.mean(np.abs(y_test - pred)))
    rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))

    per_dataset_mae = {}
    for group in np.unique(lab_test):
        mask = lab_test == group
        per_dataset_mae[group] = float(np.mean(np.abs(y_test[mask] - pred[mask])))

    print(f"[baseline:{name}] overall MAE={mae:.5f}  RMSE={rmse:.5f}")
    for group, m in per_dataset_mae.items():
        print(f"  {group}: MAE={m:.5f}")

    results = {
        "run_name": f"baseline_{name}",
        "model_variant": f"baseline_{name}",
        "test_mae": mae,
        "test_rmse": rmse,
        "per_dataset_mae": per_dataset_mae,
        "dataset_signature": sorted(np.unique(lab_test).tolist()),
    }
    if save_outputs:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(config.OUTPUT_DIR, f"baseline_{name}_results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        np.savez(os.path.join(config.OUTPUT_DIR, f"baseline_{name}_predictions.npz"),
                 y_test=y_test, mean_pred=pred, lab_test=lab_test)
    return results


def run_all_baselines(cache_path=None):
    results = []
    results.append(run_baseline("linear", LinearRegression(), cache_path))
    # SVR is much slower on large sample counts (the MIT-dominated training set has
    # ~74k+ training windows) — subsample for tractability. This is a real
    # limitation of the SVR baseline specifically; the neural network and
    # linear baseline both still train on the full training set.
    svr_model = SVR(kernel="rbf", C=1.0, epsilon=0.01)
    results.append(run_baseline("svr", svr_model, cache_path, max_train_samples=20000))

    if _HAS_XGBOOST:
        xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=config.RANDOM_SEED, n_jobs=-1)
        results.append(run_baseline("xgboost", xgb_model, cache_path))
    else:
        print("[baseline:xgboost] skipped — install with: pip install xgboost")
    return results


if __name__ == "__main__":
    run_all_baselines()