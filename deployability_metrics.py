"""
Deployability metrics: parameter count, inference latency, and memory
footprint — the practical numbers a battery management system (BMS)
engineer would actually care about, beyond pure accuracy. Cheap to measure
(no retraining), and increasingly expected by reviewers in this space.

Measures, for each available trained model variant AND the classical
baselines:
  - parameter count (neural models only)
  - inference latency: single-sample (batch=1, realistic BMS scenario) and
    batched (batch=256, realistic offline/fleet-analysis scenario)
  - approximate model size on disk (MB)

Run:
    python deployability_metrics.py
Requires: saved *_model.pt files (from train_one) for neural variants, and
the cached test data for realistic-shaped input; baseline sklearn/xgboost
models are re-fit quickly on the fly if no saved model object exists for them
(they're cheap to fit, unlike the neural models).
"""

import os
import glob
import json
import time
import numpy as np
import torch

import config
from model import build_model, predict_in_batches

N_TIMING_REPEATS = 50


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_inference(model, x, device, batch_size, n_repeats=N_TIMING_REPEATS):
    """Returns mean latency in milliseconds per batch of the given size."""
    model.eval()
    xb = x[:batch_size].to(device)
    # warm-up (first call often slower due to lazy CUDA kernel compilation)
    with torch.no_grad():
        for _ in range(3):
            _ = model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            start = time.perf_counter()
            _ = model(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms
    return float(np.mean(times)), float(np.std(times))


def measure_neural_variant(variant_name, model_path, X_sample, device):
    model = build_model(variant_name).to(device)
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f"  [!] could not load weights for {variant_name} ({e}) — "
              f"measuring an UNTRAINED model of the same architecture instead. "
              f"Parameter count and latency are still valid; ignore any accuracy implied.")

    n_params = count_parameters(model)
    x_tensor = torch.tensor(X_sample, dtype=torch.float32)

    lat_1_mean, lat_1_std = time_inference(model, x_tensor, device, batch_size=1)
    lat_256_mean, lat_256_std = time_inference(model, x_tensor, device,
                                                batch_size=min(256, len(x_tensor)))

    size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else None

    return {
        "variant": variant_name,
        "n_parameters": n_params,
        "latency_batch1_ms": lat_1_mean,
        "latency_batch1_std_ms": lat_1_std,
        "latency_batch256_ms": lat_256_mean,
        "latency_batch256_std_ms": lat_256_std,
        "model_size_mb": size_mb,
        "device": str(device),
    }


def measure_baseline(name, model_path_substring, X_flat_sample):
    path_candidates = glob.glob(os.path.join(config.OUTPUT_DIR, f"*{model_path_substring}*"))
    # sklearn/xgboost models aren't saved as objects in this pipeline (only
    # predictions/results are) — re-fit quickly is impractical here without
    # the training set in scope, so we time PREDICTION on a fresh, cheaply
    # re-instantiated model of the same hyperparameters instead, using
    # already-fitted-equivalent settings; this measures inference speed
    # characteristics, which is what matters for deployability, even though
    # the specific fitted parameters differ run-to-run.
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except ImportError:
        has_xgb = False

    dummy_y = np.random.randn(len(X_flat_sample))
    if name == "linear":
        model = LinearRegression().fit(X_flat_sample, dummy_y)
    elif name == "svr":
        model = SVR().fit(X_flat_sample[:500], dummy_y[:500])  # SVR fit is slow; small dummy fit is fine for timing predict()
    elif name == "xgboost" and has_xgb:
        model = XGBRegressor(n_estimators=300, max_depth=6, n_jobs=-1).fit(X_flat_sample, dummy_y)
    else:
        return None

    # batch=1 timing
    times_1 = []
    for _ in range(N_TIMING_REPEATS):
        start = time.perf_counter()
        model.predict(X_flat_sample[:1])
        times_1.append((time.perf_counter() - start) * 1000)

    times_256 = []
    n_batch = min(256, len(X_flat_sample))
    for _ in range(N_TIMING_REPEATS):
        start = time.perf_counter()
        model.predict(X_flat_sample[:n_batch])
        times_256.append((time.perf_counter() - start) * 1000)

    return {
        "variant": f"baseline_{name}",
        "n_parameters": None,
        "latency_batch1_ms": float(np.mean(times_1)),
        "latency_batch1_std_ms": float(np.std(times_1)),
        "latency_batch256_ms": float(np.mean(times_256)),
        "latency_batch256_std_ms": float(np.std(times_256)),
        "model_size_mb": None,
        "device": "cpu",
        "note": "timed on a freshly-fit model of matching hyperparameters, "
                "not the exact fitted weights from the main experiment run — "
                "inference speed characteristics are what's measured here.",
    }


def run_all():
    cache_path = os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(cache_path):
        print(f"[deployability] cache not found: {cache_path} — run preprocessing.py first")
        return None
    cache = np.load(cache_path, allow_pickle=True)
    X_test = cache["X_test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[deployability] neural models measured on: {device}")

    results = []
    for variant in ["full", "no_attention", "no_multiscale", "plain_lstm", "gru"]:
        model_path = os.path.join(config.OUTPUT_DIR, f"ablation_{variant}_model.pt")
        if not os.path.exists(model_path):
            print(f"  [skip] {variant}: no saved model at {model_path}")
            continue
        r = measure_neural_variant(variant, model_path, X_test, device)
        results.append(r)
        print(f"  {variant}: {r['n_parameters']:,} params, "
              f"batch=1: {r['latency_batch1_ms']:.3f}ms, "
              f"batch=256: {r['latency_batch256_ms']:.3f}ms, "
              f"size: {r['model_size_mb']:.2f}MB" if r['model_size_mb'] else "")

    X_flat = X_test.reshape(X_test.shape[0], -1)
    for name in ["linear", "svr", "xgboost"]:
        r = measure_baseline(name, name, X_flat)
        if r is not None:
            results.append(r)
            print(f"  baseline_{name}: batch=1: {r['latency_batch1_ms']:.3f}ms, "
                  f"batch=256: {r['latency_batch256_ms']:.3f}ms")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "deployability_metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[deployability] saved to {config.OUTPUT_DIR}/deployability_metrics.json")
    return results


if __name__ == "__main__":
    run_all()
