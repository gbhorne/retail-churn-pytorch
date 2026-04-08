# src/evaluate.py
# Evaluation utilities for MLP and TabNet models.
# Reports ROC-AUC, PR-AUC, classification report, and confusion matrix.
# Generates SHAP summary plot for feature importance.

import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)


def evaluate_mlp(model, loader, device="cpu") -> dict:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(y_batch.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    binary = (preds >= 0.5).astype(int)

    metrics = {
        "roc_auc":     round(roc_auc_score(labels, preds), 4),
        "pr_auc":      round(average_precision_score(labels, preds), 4),
        "report":      classification_report(labels, binary),
        "conf_matrix": confusion_matrix(labels, binary),
    }

    print(f"ROC-AUC : {metrics['roc_auc']}")
    print(f"PR-AUC  : {metrics['pr_auc']}")
    print(metrics["report"])

    return metrics


def shap_summary(model, X_sample: np.ndarray,
                 feature_names: list, save_path="shap_summary.png", device="cpu"):

    model.eval()

    def predict_fn(x):
        t = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            return torch.sigmoid(model(t)).cpu().numpy()

    background = shap.sample(X_sample, 100)
    explainer  = shap.KernelExplainer(predict_fn, background)
    shap_vals  = explainer.shap_values(X_sample[:300], nsamples=100)

    shap.summary_plot(
        shap_vals, X_sample[:300],
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary saved to {save_path}")
    return shap_vals