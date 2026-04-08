# run_shap.py
# Loads the trained MLP, runs SHAP KernelExplainer on a sample of test customers,
# and saves a summary plot to docs/charts/shap_summary.png
# Run: python run_shap.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import os

from src.features import load_and_engineer, get_feature_cols, split_and_scale
from src.mlp import ChurnMLP

os.makedirs("docs/charts", exist_ok=True)

DATA_PATH  = "data/features.csv"
MODEL_PATH = "mlp_churn.pth"
device     = "cpu"

# load data and get test split
df           = load_and_engineer(DATA_PATH)
feature_cols = get_feature_cols(df)
X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df, feature_cols)

# load trained model
model = ChurnMLP(input_dim=len(feature_cols))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded.")

# prediction function for SHAP
def predict_fn(x):
    t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(t)).numpy()

# use 100 background samples and explain 300 test samples
print("Running SHAP (this takes a few minutes)...")
background  = shap.sample(X_test, 100)
explainer   = shap.KernelExplainer(predict_fn, background)
shap_values = explainer.shap_values(X_test[:300], nsamples=100)

# summary plot
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_test[:300],
    feature_names=feature_cols,
    show=False,
    plot_type="dot"
)
plt.title("SHAP feature importance — PyTorch MLP churn model",
          fontsize=13, pad=15)
plt.tight_layout()
plt.savefig("docs/charts/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: docs/charts/shap_summary.png")

# bar plot of mean absolute SHAP values
mean_shap = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_shap)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(
    [feature_cols[i] for i in sorted_idx],
    mean_shap[sorted_idx],
    color="#534AB7"
)
plt.xlabel("Mean absolute SHAP value")
plt.title("Feature importance ranking — MLP churn model", fontsize=13)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("docs/charts/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: docs/charts/shap_bar.png")