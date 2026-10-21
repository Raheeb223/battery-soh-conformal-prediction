"""
Deployment cost of the point model and of the conformal step on top of it.

Measured on the real cached splits (cache/processed.npz) and the saved
ablation checkpoints, no retraining:
  - parameter count and checkpoint size per architecture variant
  - CPU latency, batch=1, single thread (closest desktop proxy for a BMS
    controller -- not a substitute for on-device measurement), for:
      * one deterministic forward pass
      * the MC-Dropout mean predictor as evaluated in the paper (train.py
        centres every conformal interval on it: config.MC_DROPOUT_SAMPLES
        sequential stochastic passes, via mc_dropout_predict)
      * the same estimator with the passes run as one replicated batch
  - GPU throughput, batched, deterministic pass
  - groupwise conformal overhead: calibration time on the real validation
    set and per-sample interval-construction time on the real test set

Run:
    python deployability_benchmark.py
Output:
    outputs/deployability_results.json
"""

import json
import os
import platform
import subprocess
import time

import numpy as np
import torch

import config
from conformal import GroupwiseConformal
from model import build_model, enable_mc_dropout, mc_dropout_predict, predict_in_batches

VARIANTS = ["full", "no_attention", "no_multiscale", "plain_lstm", "gru", "transformer"]
N_WARMUP = 20
N_CPU_SINGLE = 500
N_CPU_MC = 100
N_GPU_REPEATS = 20
GPU_BATCH = 4096
N_CONFORMAL_REPEATS = 20


def cpu_name():
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor()


def load_variant(variant, device):
    path = os.path.join(config.OUTPUT_DIR, f"ablation_{variant}_model.pt")
    model = build_model(variant)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval(), path


def cpu_single_sample_ms(predict_one, x_test, n_timed):
    torch.set_num_threads(1)
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.integers(0, len(x_test), size=N_WARMUP + n_timed)
    times = []
    with torch.no_grad():
        for k, i in enumerate(idx):
            xb = torch.from_numpy(x_test[i:i + 1])
            t0 = time.perf_counter()
            predict_one(xb)
            dt = time.perf_counter() - t0
            if k >= N_WARMUP:
                times.append(dt * 1e3)
    return float(np.median(times)), float(np.percentile(times, 95))


def mc_mean_batched_replicas(model, xb, n_samples=config.MC_DROPOUT_SAMPLES):
    # Same estimator as mc_dropout_predict (independent dropout mask per replica),
    # but the n_samples passes run as one forward over a replicated batch.
    enable_mc_dropout(model)
    preds = model(xb.repeat(n_samples, 1, 1))[0].view(n_samples, xb.shape[0])
    return preds.mean(dim=0)


def gpu_throughput(model, x_test):
    xb = torch.from_numpy(x_test[:GPU_BATCH]).cuda()
    with torch.no_grad():
        for _ in range(5):
            model(xb)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_GPU_REPEATS):
            model(xb)
        torch.cuda.synchronize()
    per_batch = (time.perf_counter() - t0) / N_GPU_REPEATS
    return float(xb.shape[0] / per_batch)


def conformal_overhead(model, d, device):
    x_val = torch.from_numpy(d["X_val"]).to(device)
    x_test = torch.from_numpy(d["X_test"]).to(device)
    pred_val = predict_in_batches(model, x_val).cpu().numpy()
    pred_test = predict_in_batches(model, x_test).cpu().numpy()
    y_val, lab_val, lab_test = d["y_val"], d["lab_val"], d["lab_test"]

    cal_times, int_times = [], []
    for _ in range(N_CONFORMAL_REPEATS):
        gc = GroupwiseConformal(alpha=0.10)
        t0 = time.perf_counter()
        gc.calibrate(y_val, pred_val, lab_val)
        cal_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        gc.predict_interval(pred_test, lab_test)
        int_times.append(time.perf_counter() - t0)
    return {
        "n_calibration_samples": int(len(y_val)),
        "n_test_samples": int(len(lab_test)),
        "n_groups": int(len(np.unique(lab_val))),
        "calibration_time_ms": float(np.median(cal_times) * 1e3),
        "interval_time_per_sample_us": float(np.median(int_times) / len(lab_test) * 1e6),
    }


def main():
    d = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    x_test = d["X_test"]
    has_gpu = torch.cuda.is_available()

    results = {
        "hardware": {
            "cpu": cpu_name(),
            "gpu": torch.cuda.get_device_name(0) if has_gpu else None,
            "torch": torch.__version__,
            "cpu_threads_for_single_sample": 1,
        },
        "input_window": int(config.INPUT_WINDOW),
        "mc_samples": int(config.MC_DROPOUT_SAMPLES),
        "variants": {},
    }

    for variant in VARIANTS:
        model, path = load_variant(variant, "cpu")
        n_params = int(sum(p.numel() for p in model.parameters()))
        det_med, det_p95 = cpu_single_sample_ms(model, x_test, N_CPU_SINGLE)
        mc_med, mc_p95 = cpu_single_sample_ms(lambda xb: mc_dropout_predict(model, xb), x_test, N_CPU_MC)
        rep_med, rep_p95 = cpu_single_sample_ms(lambda xb: mc_mean_batched_replicas(model, xb), x_test, N_CPU_MC)
        model.eval()
        entry = {
            "n_params": n_params,
            "checkpoint_kb": float(os.path.getsize(path) / 1024),
            "cpu_batch1_deterministic_median_ms": det_med,
            "cpu_batch1_deterministic_p95_ms": det_p95,
            "cpu_batch1_mc_mean_as_evaluated_median_ms": mc_med,
            "cpu_batch1_mc_mean_as_evaluated_p95_ms": mc_p95,
            "cpu_batch1_mc_mean_batched_replicas_median_ms": rep_med,
            "cpu_batch1_mc_mean_batched_replicas_p95_ms": rep_p95,
        }
        if has_gpu:
            entry["gpu_throughput_deterministic_samples_per_s"] = gpu_throughput(model.cuda(), x_test)
        results["variants"][variant] = entry
        print(f"[{variant}] params={n_params:,} det={det_med:.2f}ms mc30={mc_med:.2f}ms "
              f"mc30_batched={rep_med:.2f}ms")

    device = "cuda" if has_gpu else "cpu"
    full, _ = load_variant("full", device)
    results["conformal_overhead_full_model"] = conformal_overhead(full, d, device)
    print(json.dumps(results["conformal_overhead_full_model"], indent=2))

    out = os.path.join(config.OUTPUT_DIR, "deployability_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
