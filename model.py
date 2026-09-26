"""
Multi-scale bidirectional LSTM with self-attention, for SOH sequence
regression, with MC-Dropout enabled at inference for uncertainty estimates.

"Multi-scale" here = two parallel LSTM branches reading the same window at
different effective resolutions (raw sequence, and a downsampled/pooled
version), concatenated before attention. This is a common way to capture
both short-term noise and longer-term degradation trend in one model.

ABLATION SUPPORT: use_multiscale=False drops the coarse branch entirely
(plain single-branch BiLSTM). use_attention=False replaces the attention
pooling with simple mean-pooling over the sequence. Use these two flags to
run the ablation experiments that isolate what each architectural choice
actually contributes over a plain BiLSTM baseline.
"""

import torch
import torch.nn as nn
import config


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        e = torch.tanh(self.proj(x))         # (batch, seq_len, attn_dim)
        scores = self.score(e).squeeze(-1)    # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context, weights


class MultiScaleBiLSTMAttention(nn.Module):
    def __init__(self,
                 input_size=1,
                 hidden_size=config.HIDDEN_SIZE,
                 num_layers=config.NUM_LSTM_LAYERS,
                 dropout=config.DROPOUT,
                 attn_dim=config.ATTENTION_DIM,
                 use_multiscale=True,
                 use_attention=True):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_attention = use_attention

        # branch 1: full-resolution sequence (always present)
        self.lstm_fine = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                  batch_first=True, bidirectional=True,
                                  dropout=dropout if num_layers > 1 else 0.0)

        if use_multiscale:
            # branch 2: coarse resolution (avg-pool factor 2 before LSTM)
            self.pool = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)
            self.lstm_coarse = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                        batch_first=True, bidirectional=True,
                                        dropout=dropout if num_layers > 1 else 0.0)
            fused_dim = hidden_size * 2 * 2  # both branches, both directions
        else:
            fused_dim = hidden_size * 2      # fine branch only

        if use_attention:
            self.attention = SelfAttention(fused_dim, attn_dim)
        # else: mean-pool over the sequence dimension, no learned pooling

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, 1)
        fine_out, _ = self.lstm_fine(x)                      # (batch, seq_len, 2*hidden)

        if self.use_multiscale:
            x_coarse = self.pool(x.transpose(1, 2)).transpose(1, 2)  # (batch, seq_len/2, 1)
            coarse_out, _ = self.lstm_coarse(x_coarse)               # (batch, seq_len/2, 2*hidden)
            # upsample coarse branch back to fine's seq_len by repeating each step twice
            coarse_out = coarse_out.repeat_interleave(2, dim=1)[:, : fine_out.size(1), :]
            fused = torch.cat([fine_out, coarse_out], dim=-1)   # (batch, seq_len, fused_dim)
        else:
            fused = fine_out

        fused = self.dropout(fused)

        if self.use_attention:
            context, attn_weights = self.attention(fused)
        else:
            context = fused.mean(dim=1)   # simple mean-pool over time
            attn_weights = None

        pred = self.head(context).squeeze(-1)                # (batch,)
        return pred, attn_weights


def build_model(variant: str = "full") -> nn.Module:
    """Factory for the ablation variants used in experiments.py:
    - 'full'          : multi-scale + attention (the proposed model)
    - 'no_attention'  : multi-scale + mean-pooling instead of attention
    - 'no_multiscale' : single branch + attention
    - 'plain_lstm'    : single branch + mean-pooling (closest thing to a vanilla BiLSTM)
    - 'gru'           : a plain GRU baseline (common comparison in SOH papers)
    """
    if variant == "gru":
        return GRUBaseline()
    variants = {
        "full":          dict(use_multiscale=True,  use_attention=True),
        "no_attention":  dict(use_multiscale=True,  use_attention=False),
        "no_multiscale": dict(use_multiscale=False, use_attention=True),
        "plain_lstm":    dict(use_multiscale=False, use_attention=False),
    }
    if variant not in variants:
        raise ValueError(f"Unknown model variant '{variant}'. Choose from {list(variants) + ['gru']}.")
    return MultiScaleBiLSTMAttention(**variants[variant])


class GRUBaseline(nn.Module):
    """A plain unidirectional GRU + linear head — a standard, widely-used
    SOH baseline. Included so the comparison against the proposed model is
    against a real recurrent baseline (not just linear/SVR/XGBoost), and so
    it gets the same conformal treatment through the shared training engine.
    Returns (pred, None) to match the (pred, attn_weights) interface the
    training loop expects."""
    def __init__(self, input_size=1, hidden_size=config.HIDDEN_SIZE,
                 num_layers=config.NUM_LSTM_LAYERS, dropout=config.DROPOUT):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)             # (batch, seq_len, hidden)
        last = out[:, -1, :]             # take the final time step
        last = self.dropout(last)
        pred = self.head(last).squeeze(-1)
        return pred, None


def enable_mc_dropout(model: nn.Module):
    """Call before inference: keeps ONLY dropout layers in train mode
    (so they still sample), while everything else (e.g. batchnorm, if any)
    stays in eval mode. This is what makes MC-Dropout uncertainty work."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_in_batches(model, x, batch_size=8192):
    """Deterministic (non-MC-Dropout) forward pass, batched to avoid OOM on
    large inputs. Use this instead of calling model(x) directly on a full
    test/val tensor."""
    model.eval()
    n = x.shape[0]
    outs = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = x[start:start + batch_size]
            pred, _ = model(xb)
            outs.append(pred)
    return torch.cat(outs)


def mc_dropout_predict(model, x, n_samples=config.MC_DROPOUT_SAMPLES, batch_size=4096):
    """Returns (mean_pred, std_pred) over n_samples stochastic forward passes.

    IMPORTANT: processes x in batches rather than one giant tensor. Without
    this, a large test set (e.g. an entire held-out dataset in
    leave_one_out.py — RWTH's full 114,716 rows all become the test set for
    that one LODO run) tries to allocate the whole thing on the GPU at once
    for EVERY one of the n_samples stochastic passes simultaneously (since
    they're stacked together), which can exceed available VRAM even on a
    12GB GPU. Batching keeps memory bounded regardless of how large x is.
    """
    enable_mc_dropout(model)
    n = x.shape[0]
    all_means, all_stds = [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = x[start:start + batch_size]
            preds = torch.stack([model(xb)[0] for _ in range(n_samples)], dim=0)  # (n_samples, batch)
            all_means.append(preds.mean(dim=0))
            all_stds.append(preds.std(dim=0))
    return torch.cat(all_means), torch.cat(all_stds)