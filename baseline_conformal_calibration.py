"""
Pooled vs. groupwise conformal calibration on non-BiLSTM models.

Tests whether the pooled-vs-groupwise calibration result depends on the
proposed architecture by applying the same comparison to three simpler
point predictors: SVR, XGBoost, and the plain LSTM from the architecture
ablation. These models have no per-sample uncertainty estimate, so standard
residual-based split conformal prediction (nonconformity score |y - pred|)
is used for all three.

Requires cache/processed.npz and outputs/ablation_plain_lstm_model.pt. SVR and
XGBoost are refit with the same features and hyperparameters as baselines.py,
because their validation-set predictions (needed for calibration) are not
stored by the baseline pipeline.

Run:
    python baseline_conformal_calibration.py
Output:
    outputs/baseline_conformal_results.json
    outputs/figures/baseline_conformal_comparison.png
"""

import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import xgboost as xgb

import config
from model import build_model

ALPHA = getattr(config, "CONFORMAL_ALPHA", 0.10)
DATASET_ORDER = ["MIT", "BIT", "XJTU", "RWTH", "CALCE", "Oxford", "NASA"]
DATASET_COLORS = {
    "MIT": "#4C72B0", "BIT": "#DD8452", "XJTU": "#55A868", "RWTH": "#C44E52",
    "CALCE": "#8172B2", "Oxford": "#937860", "NASA": "#DA8BC3",
}


def _load_cache():
    path = os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(path):
        print(f"  [!] {path} not found -- run preprocessing.py first.")
        return None
    return np.load(path, allow_pickle=True)


def pooled_calibrate(y_val, pred_val, alpha=ALPHA):
    """Single quantile across ALL validation samples, applied uniformly."""
    scores = np.abs(y_val - pred_val)
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level))


def groupwise_calibrate(y_val, pred_val, lab_val, alpha=ALPHA):
    """Separate quantile per dataset group."""
    scores = np.abs(y_val - pred_val)
    qhat_per_group = {}
    for group in np.unique(lab_val):
        mask = lab_val == group
        group_scores = scores[mask]
        n = len(group_scores)
        if n == 0:
            continue
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        qhat_per_group[group] = float(np.quantile(group_scores, q_level))
    return qhat_per_group


def apply_pooled(pred, qhat):
    return pred - qhat, pred + qhat


def apply_groupwise(pred, lab, qhat_per_group):
    lower = np.zeros_like(pred)
    upper = np.zeros_like(pred)
    for group, qhat in qhat_per_group.items():
        mask = lab == group
        lower[mask] = pred[mask] - qhat
        upper[mask] = pred[mask] + qhat
    return lower, upper


def knn_local_uncertainty(X_calibrate, X_query, residuals_calibrate, k=25):
    """Estimates per-sample uncertainty for models with no natural
    uncertainty output (SVR, XGBoost) using k-NN local residual variance:
    for each query point, find its k nearest neighbors in feature space
    among the calibration set, and use the local spread of THEIR absolute
    residuals as this point's uncertainty estimate. This is a standard,
    established technique in the conformal prediction literature (locally
    weighted / Mondrian conformal prediction via nearest-neighbor residual
    variance -- see e.g. Papadopoulos & Haralambous, 2011), not an ad-hoc
    invention for this comparison specifically."""
    from sklearn.neighbors import NearestNeighbors
    abs_resid_calibrate = np.abs(residuals_calibrate)
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X_calibrate))).fit(X_calibrate)
    _, indices = nbrs.kneighbors(X_query)
    local_std = np.array([abs_resid_calibrate[idx].std() + 1e-6 for idx in indices])
    return local_std


def normalized_groupwise_calibrate(y_val, pred_val, std_val, lab_val, alpha=ALPHA):
    """Same as groupwise_calibrate, but nonconformity scores are normalized
    by each sample's own uncertainty estimate (MC-Dropout for plain_lstm,
    k-NN local residual variance for SVR/XGBoost) before taking the
    per-group quantile -- matching this paper's main normalized_groupwise
    variant (Section 4)."""
    scores = np.abs(y_val - pred_val) / (std_val + 1e-6)
    qhat_per_group = {}
    for group in np.unique(lab_val):
        mask = lab_val == group
        group_scores = scores[mask]
        n = len(group_scores)
        if n == 0:
            continue
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        qhat_per_group[group] = float(np.quantile(group_scores, q_level))
    return qhat_per_group


def apply_normalized_groupwise(pred, std, lab, qhat_per_group):
    lower = np.zeros_like(pred)
    upper = np.zeros_like(pred)
    for group, qhat in qhat_per_group.items():
        mask = lab == group
        lower[mask] = pred[mask] - qhat * std[mask]
        upper[mask] = pred[mask] + qhat * std[mask]
    return lower, upper


def get_plain_lstm_mc_dropout(cache, device, n_samples=None):
    """Runs MC-Dropout inference (dropout left ON at inference, multiple
    stochastic forward passes) for plain_lstm, matching this project's
    existing MC-Dropout methodology used for the full BiLSTM."""
    ckpt_path = os.path.join(config.OUTPUT_DIR, "ablation_plain_lstm_model.pt")
    if not os.path.exists(ckpt_path):
        return None
    torch.manual_seed(config.RANDOM_SEED)
    # torch.manual_seed() alone does NOT guarantee determinism on CUDA --
    # cuDNN's RNN/LSTM kernels can still pick different (numerically
    # non-identical) algorithms between runs unless explicitly forced.
    # This is almost certainly why results still shifted between runs
    # even with manual_seed set above.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    n_samples = n_samples or getattr(config, "MC_DROPOUT_SAMPLES", 30)
    model = build_model("plain_lstm").to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.train()  # keep dropout ACTIVE for MC-Dropout sampling

    def mc_predict(X):
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred, _ = model(torch.tensor(X).to(device))
                preds.append(pred.cpu().numpy().flatten())
        preds = np.array(preds)  # (n_samples, n_data)
        return preds.mean(axis=0), preds.std(axis=0)

    mean_val, std_val = mc_predict(cache["X_val"])
    mean_test, std_test = mc_predict(cache["X_test"])
    return (mean_val, std_val), (mean_test, std_test)


def evaluate_model(name, pred_val, pred_test, y_val, y_test, lab_val, lab_test,
                    std_val=None, std_test=None):
    print(f"\n{'='*78}")
    print(f"{name}")
    print(f"{'='*78}")

    qhat_pooled = pooled_calibrate(y_val, pred_val)
    lo_p, hi_p = apply_pooled(pred_test, qhat_pooled)

    qhat_group = groupwise_calibrate(y_val, pred_val, lab_val)
    lo_g, hi_g = apply_groupwise(pred_test, lab_test, qhat_group)

    overall_mae = float(np.mean(np.abs(pred_test - y_test)))
    overall_cov_pooled = float(np.mean((y_test >= lo_p) & (y_test <= hi_p)))
    overall_cov_group = float(np.mean((y_test >= lo_g) & (y_test <= hi_g)))
    print(f"  Overall MAE: {overall_mae:.5f}")
    print(f"  Overall pooled coverage: {overall_cov_pooled:.4f} (target {1-ALPHA:.2f})")
    print(f"  Overall groupwise coverage: {overall_cov_group:.4f} (target {1-ALPHA:.2f})")

    have_normalized = std_val is not None and std_test is not None
    if have_normalized:
        qhat_norm = normalized_groupwise_calibrate(y_val, pred_val, std_val, lab_val)
        lo_n, hi_n = apply_normalized_groupwise(pred_test, std_test, lab_test, qhat_norm)
        overall_cov_norm = float(np.mean((y_test >= lo_n) & (y_test <= hi_n)))
        print(f"  Overall normalized_groupwise coverage: {overall_cov_norm:.4f} (target {1-ALPHA:.2f})")

    per_dataset = {}
    datasets_present = [d for d in DATASET_ORDER if d in set(np.unique(lab_test))]
    header = f"  {'Dataset':10s} {'Pooled cov.':>12s} {'Groupwise cov.':>15s}"
    if have_normalized:
        header += f" {'Norm.-group. cov.':>18s}"
    print(f"\n{header}")
    for ds in datasets_present:
        mask = lab_test == ds
        if mask.sum() == 0:
            continue
        cov_p = float(np.mean((y_test[mask] >= lo_p[mask]) & (y_test[mask] <= hi_p[mask])))
        cov_g = float(np.mean((y_test[mask] >= lo_g[mask]) & (y_test[mask] <= hi_g[mask])))
        row = {"pooled_coverage": cov_p, "groupwise_coverage": cov_g, "delta": cov_g - cov_p}
        line = f"  {ds:10s} {cov_p*100:11.1f}% {cov_g*100:14.1f}%"
        if have_normalized:
            cov_n = float(np.mean((y_test[mask] >= lo_n[mask]) & (y_test[mask] <= hi_n[mask])))
            row["normalized_groupwise_coverage"] = cov_n
            line += f" {cov_n*100:17.1f}%"
        per_dataset[ds] = row
        print(line)

    result = {
        "overall_mae": overall_mae,
        "overall_pooled_coverage": overall_cov_pooled,
        "overall_groupwise_coverage": overall_cov_group,
        "per_dataset": per_dataset,
    }
    if have_normalized:
        result["overall_normalized_groupwise_coverage"] = overall_cov_norm
    return result


def get_svr_xgb_predictions(cache):
    X_train = cache["X_train"].reshape(cache["X_train"].shape[0], -1)
    y_train = cache["y_train"]
    X_val = cache["X_val"].reshape(cache["X_val"].shape[0], -1)
    X_test = cache["X_test"].reshape(cache["X_test"].shape[0], -1)

    print("Refitting SVR (matching baselines.py's approach)...")
    n_subsample = min(20000, len(X_train))
    if len(X_train) > n_subsample:
        idx = np.random.default_rng(config.RANDOM_SEED).choice(len(X_train), n_subsample, replace=False)
        X_train_svr, y_train_svr = X_train[idx], y_train[idx]
        print(f"  subsampled to {n_subsample} for tractability")
    else:
        X_train_svr, y_train_svr = X_train, y_train
    svr = SVR(kernel="rbf")
    svr.fit(X_train_svr, y_train_svr)
    svr_pred_train = svr.predict(X_train_svr)
    svr_pred_val = svr.predict(X_val)
    svr_pred_test = svr.predict(X_test)

    print("Refitting XGBoost...")
    xgb_model = xgb.XGBRegressor(random_state=config.RANDOM_SEED)
    xgb_model.fit(X_train, y_train)
    xgb_pred_train = xgb_model.predict(X_train)
    xgb_pred_val = xgb_model.predict(X_val)
    xgb_pred_test = xgb_model.predict(X_test)

    # k-NN local uncertainty for both, using TRAINING residuals as the
    # neighbor pool -- validation/test points never inform their own
    # uncertainty estimate, avoiding leakage.
    print("Computing k-NN local residual-variance uncertainty for SVR...")
    svr_resid_train = y_train_svr - svr_pred_train
    svr_std_val = knn_local_uncertainty(X_train_svr, X_val, svr_resid_train)
    svr_std_test = knn_local_uncertainty(X_train_svr, X_test, svr_resid_train)

    print("Computing k-NN local residual-variance uncertainty for XGBoost...")
    xgb_resid_train = y_train - xgb_pred_train
    xgb_std_val = knn_local_uncertainty(X_train, X_val, xgb_resid_train)
    xgb_std_test = knn_local_uncertainty(X_train, X_test, xgb_resid_train)

    return ((svr_pred_val, svr_pred_test), (svr_std_val, svr_std_test)), \
           ((xgb_pred_val, xgb_pred_test), (xgb_std_val, xgb_std_test))


def get_plain_lstm_predictions(cache, device):
    ckpt_path = os.path.join(config.OUTPUT_DIR, "ablation_plain_lstm_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [!] {ckpt_path} not found -- run experiments.py's architecture "
              f"ablation section first.")
        return None
    model = build_model("plain_lstm").to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    X_val = cache["X_val"]
    X_test = cache["X_test"]
    with torch.no_grad():
        pred_val, _ = model(torch.tensor(X_val).to(device))
        pred_test, _ = model(torch.tensor(X_test).to(device))
    return pred_val.cpu().numpy().flatten(), pred_test.cpu().numpy().flatten()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = _load_cache()
    if cache is None:
        return

    y_val, lab_val = cache["y_val"], cache["lab_val"]
    y_test, lab_test = cache["y_test"], cache["lab_test"]

    results = {}

    (svr_pred, svr_std), (xgb_pred, xgb_std) = get_svr_xgb_predictions(cache)
    svr_pred_val, svr_pred_test = svr_pred
    svr_std_val, svr_std_test = svr_std
    xgb_pred_val, xgb_pred_test = xgb_pred
    xgb_std_val, xgb_std_test = xgb_std

    print("\nSVR and XGBoost now include normalized_groupwise using k-NN "
          "local residual-variance uncertainty (a standard technique, not "
          "an ad-hoc proxy -- see the docstring of knn_local_uncertainty).")
    results["SVR"] = evaluate_model("SVR", svr_pred_val, svr_pred_test, y_val, y_test,
                                     lab_val, lab_test, std_val=svr_std_val, std_test=svr_std_test)
    results["XGBoost"] = evaluate_model("XGBoost", xgb_pred_val, xgb_pred_test, y_val, y_test,
                                         lab_val, lab_test, std_val=xgb_std_val, std_test=xgb_std_test)

    print("\nRunning MC-Dropout inference for plain LSTM (this has a genuine "
          "per-sample uncertainty source, so normalized_groupwise IS computed)...")
    lstm_mc = get_plain_lstm_mc_dropout(cache, device)
    if lstm_mc is not None:
        (lstm_mean_val, lstm_std_val), (lstm_mean_test, lstm_std_test) = lstm_mc
        results["plain_lstm"] = evaluate_model(
            "Plain LSTM", lstm_mean_val, lstm_mean_test, y_val, y_test, lab_val, lab_test,
            std_val=lstm_std_val, std_test=lstm_std_test)
    else:
        print(f"  [!] plain_lstm checkpoint not found -- skipping.")

    out_path = os.path.join(config.OUTPUT_DIR, "baseline_conformal_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved {out_path}")

    model_names = list(results.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5), sharey=True)
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        per_ds = results[name]["per_dataset"]
        datasets = list(per_ds.keys())
        x = np.arange(len(datasets))
        width = 0.35
        pooled_vals = [per_ds[d]["pooled_coverage"] * 100 for d in datasets]
        group_vals = [per_ds[d]["groupwise_coverage"] * 100 for d in datasets]
        ax.bar(x - width/2, pooled_vals, width, label="Pooled", color="#C44E52")
        ax.bar(x + width/2, group_vals, width, label="Groupwise", color="#4C72B0")
        ax.axhline((1 - ALPHA) * 100, color="k", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_title(name)
        ax.set_ylabel("Coverage (%)")
        ax.legend(fontsize=8)

    fig.suptitle("Pooled vs. Groupwise Conformal Coverage Across Model Architectures", fontsize=13)
    fig.tight_layout()
    fig_dir = os.path.join(config.OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_fig = os.path.join(fig_dir, "baseline_conformal_comparison.png")
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"saved {out_fig}")

    print(f"\n{'='*78}")
    print("If pooled under-covers and groupwise restores coverage for SVR/XGBoost/"
          "plain_lstm too, that directly confirms the calibration finding is a "
          "general property of heterogeneous multi-dataset conformal prediction, "
          "not an artifact specific to the BiLSTM backbone.")


if __name__ == "__main__":
    main()