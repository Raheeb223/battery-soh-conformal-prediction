"""
Model size and inference cost per architecture variant.

Loads the *_model.pt checkpoints saved by train_one() (no retraining) and
measures, for each variant (full, no_attention, no_multiscale, plain_lstm,
gru):
  - parameter count
  - mean batched inference latency per sample (ms)
  - throughput (samples/s)

Run:
    python benchmark_inference_efficiency.py
Output:
    outputs/inference_efficiency_results.json
    outputs/figures/24_inference_efficiency.png
"""

import os
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from model import build_model

VARIANTS = ["full", "no_attention", "no_multiscale", "plain_lstm", "gru"]
VARIANT_LABELS = {
    "full": "Full\n(proposed)", "no_attention": "No\nAttention",
    "no_multiscale": "No\nMultiscale", "plain_lstm": "Plain\nLSTM", "gru": "GRU",
}
N_WARMUP = 10
N_TIMED_BATCHES = 50
BATCH_SIZE = 64


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def _make_dummy_input(batch_size, window, device):
    """Builds a dummy (batch, window, 1) input tensor -- shape-only,
    values don't matter for timing purposes."""
    return torch.randn(batch_size, window, 1, device=device)


def benchmark_variant(variant, device, window):
    run_name = f"ablation_{variant}"
    ckpt_path = os.path.join(config.OUTPUT_DIR, f"{run_name}_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [skip] {ckpt_path} not found -- run experiments.py first "
              f"(need ablation_{variant} to have completed and saved a "
              f"checkpoint).")
        return None

    model = build_model(variant).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"  [!] failed to load {ckpt_path} into a fresh '{variant}' "
              f"model: {e}. This likely means build_model('{variant}') "
              f"doesn't reproduce the exact architecture the checkpoint "
              f"was saved from -- check model.py's build_model() for how "
              f"it varies by model_variant.")
        return None
    model.eval()

    n_params = _count_params(model)

    dummy = _make_dummy_input(BATCH_SIZE, window, device)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(N_TIMED_BATCHES):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    total_samples = N_TIMED_BATCHES * BATCH_SIZE
    ms_per_sample = (elapsed / total_samples) * 1000
    samples_per_sec = total_samples / elapsed

    print(f"  {variant:16s}  params={n_params:>10,d}  "
          f"latency={ms_per_sample:.4f} ms/sample  "
          f"throughput={samples_per_sec:,.0f} samples/sec")

    return {
        "variant": variant,
        "n_params": n_params,
        "device": str(device),
        "ms_per_sample": ms_per_sample,
        "samples_per_sec": samples_per_sec,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}")
    print(f"(Note: if your training environment is CPU-only per the project's "
          f"own documentation, CPU numbers are the ones to report in the "
          f"manuscript, even if a GPU is available here.)\n")

    window = getattr(config, "INPUT_WINDOW", 20)

    results = []
    for variant in VARIANTS:
        r = benchmark_variant(variant, device, window)
        if r is not None:
            results.append(r)

    if not results:
        print("\nNo results -- no checkpoints found. Run experiments.py first.")
        return

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_json = os.path.join(config.OUTPUT_DIR, "inference_efficiency_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved {out_json}")

    # Figure
    fig_dir = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    labels = [VARIANT_LABELS.get(r["variant"], r["variant"]) for r in results]
    params = [r["n_params"] for r in results]
    latency = [r["ms_per_sample"] for r in results]
    colors = ["#4C72B0"] + ["#999999"] * (len(results) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(labels))

    axes[0].bar(x, params, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Parameter count")
    axes[0].set_title("Model size")

    axes[1].bar(x, latency, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Latency (ms/sample)")
    axes[1].set_title(f"Inference latency (batch={BATCH_SIZE}, {device})")

    fig.suptitle("Inference Efficiency Across Architectures", fontsize=14)
    fig.tight_layout()
    out_png = os.path.join(fig_dir, "24_inference_efficiency.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
