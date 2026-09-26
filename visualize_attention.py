"""
Attention weights of the proposed model on healthy vs degraded input windows.

Loads outputs/ablation_full_model.pt into build_model("full"), finds the
attention module by name (any module whose name contains "attn" or
"attention"; set ATTENTION_MODULE_NAME_HINT to choose one explicitly) and
captures its per-timestep weights with a forward hook. The script prints the
named modules and the captured tensor shape, which must be (batch, seq_len)
or (batch, seq_len, 1).

Run:
    python visualize_attention.py
Output:
    outputs/figures/21_attention_visualization.png
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from model import build_model

# Edit this if the auto-detected module name is wrong -- see the
# printed module list when you run this script.
ATTENTION_MODULE_NAME_HINT = None  # e.g. "attention" or "temporal_attn"


def find_attention_module(model):
    """Prints every named module and returns the first one whose name
    contains 'attn' or 'attention' (case-insensitive), unless
    ATTENTION_MODULE_NAME_HINT is set, in which case that exact name
    is used instead."""
    print("All named modules in the model:")
    candidates = []
    for name, module in model.named_modules():
        if name == "":
            continue
        print(f"  {name}: {type(module).__name__}")
        if "attn" in name.lower() or "attention" in name.lower():
            candidates.append(name)

    if ATTENTION_MODULE_NAME_HINT is not None:
        if ATTENTION_MODULE_NAME_HINT in dict(model.named_modules()):
            return ATTENTION_MODULE_NAME_HINT
        print(f"\n  [!] ATTENTION_MODULE_NAME_HINT='{ATTENTION_MODULE_NAME_HINT}' "
              f"not found among named modules -- falling back to auto-detection.")

    if not candidates:
        print("\n  [!] No module with 'attn'/'attention' in its name was found. "
              "Set ATTENTION_MODULE_NAME_HINT at the top of this script to the "
              "correct module name from the list printed above, then re-run.")
        return None

    print(f"\n  Auto-detected attention module: '{candidates[0]}' "
          f"(first match; edit ATTENTION_MODULE_NAME_HINT if this is wrong)")
    return candidates[0]


def extract_attention_weights(model, x, module_name, window):
    """Registers a forward hook on the named module, runs one forward
    pass, and returns the per-timestep attention weight tensor.

    IMPORTANT: a SelfAttention-style module typically returns a tuple of
    (context_vector, attention_weights) -- the context vector is usually
    shaped (batch, hidden_dim) and is NOT what we want; the attention
    weights are usually shaped (batch, seq_len) or (batch, seq_len, 1)
    and SHOULD match `window`. Rather than blindly picking tuple index 0
    (which silently grabbed the wrong 256-dim context vector last run),
    this inspects EVERY element of the tuple/output and picks whichever
    one's flattened size actually matches `window` -- printing all
    candidates either way so you can verify the choice."""
    captured = {}

    def hook(module, inp, out):
        captured["output"] = out

    module = dict(model.named_modules())[module_name]
    handle = module.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    handle.remove()

    out = captured.get("output")
    if out is None:
        print(f"  [!] hook on '{module_name}' captured nothing.")
        return None

    candidates = out if isinstance(out, tuple) else (out,)
    print(f"  Module output has {len(candidates)} tensor(s):")
    match = None
    for i, t in enumerate(candidates):
        arr = t.detach().cpu().numpy()
        flat_size = arr.reshape(-1).shape[0]
        is_match = flat_size == window
        print(f"    [{i}] shape={arr.shape}  flattened size={flat_size}"
              f"{'  <-- MATCHES window size, using this one' if is_match else ''}")
        if is_match and match is None:
            match = arr

    if match is None:
        print(f"  [!] NONE of the {len(candidates)} output tensor(s) have a "
              f"flattened size matching window={window}. The attention "
              f"module's output doesn't look like a simple per-timestep "
              f"weight vector -- inspect model.py's SelfAttention.forward() "
              f"manually to find the right tensor/shape, then adjust this "
              f"function accordingly rather than trusting a guess.")
        return None

    return match


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = getattr(config, "INPUT_WINDOW", 20)

    ckpt_path = os.path.join(config.OUTPUT_DIR, "ablation_full_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"[!] {ckpt_path} not found -- run experiments.py first.")
        return

    model = build_model("full").to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    module_name = find_attention_module(model)
    if module_name is None:
        return

    # Illustrative SYNTHETIC inputs, not test-set windows: a healthy curve
    # (linear 5% fade over the window) and a degraded curve (35% fade with a
    # late-life knee). To visualise real inputs, load X_test windows from
    # cache/processed.npz instead.
    t = np.linspace(0, 1, window)
    healthy = 1.0 - 0.05 * t
    degraded = 1.0 - 0.35 * t - 0.1 * (t > 0.7) * (t - 0.7)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for ax, curve, label in zip(axes, [healthy, degraded], ["Healthy cell", "Degraded cell"]):
        x = torch.tensor(curve, dtype=torch.float32).reshape(1, window, 1).to(device)
        weights = extract_attention_weights(model, x, module_name, window)
        if weights is None:
            continue
        weights = weights.reshape(-1)

        ax2 = ax.twinx()
        ax.plot(range(window), curve, "k-", linewidth=1.5, label="Input SOH curve")
        ax2.bar(range(window), weights, alpha=0.4, color="#C44E52", label="Attention weight")
        ax.set_ylabel("SOH")
        ax2.set_ylabel("Attention weight")
        ax.set_title(label)

    axes[-1].set_xlabel("Timestep within input window")
    fig.suptitle("Attention Weights on Input Degradation Curves", fontsize=14)
    fig.tight_layout()

    fig_dir = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, "21_attention_visualization.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
