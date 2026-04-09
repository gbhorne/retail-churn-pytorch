# run_shap.py
# Uses training data as SHAP background -- not test data.
import torch, numpy as np, shap, os, pickle, json
import matplotlib.pyplot as plt
from src.features import set_seeds, load_and_engineer, split_data, prepare_mlp
from src.mlp import ChurnMLP

set_seeds()
os.makedirs('docs/charts', exist_ok=True)
DATA_PATH  = 'data/features.csv'
MODEL_PATH = 'mlp_churn.pth'

df = load_and_engineer(DATA_PATH)
train, val, test = split_data(df)
X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols = prepare_mlp(train, val, test)

model = ChurnMLP(input_dim=len(feature_cols))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print('Model loaded.')

def predict_fn(x):
    t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad(): return torch.sigmoid(model(t)).numpy()

print('Running SHAP using training data as background...')
background  = shap.sample(X_train, 100)
explainer   = shap.KernelExplainer(predict_fn, background)
shap_values = explainer.shap_values(X_test[:300], nsamples=100)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test[:300], feature_names=feature_cols, show=False, plot_type='dot')
plt.title('SHAP feature importance: PyTorch MLP churn model', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig('docs/charts/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: docs/charts/shap_summary.png')

mean_shap  = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_shap)[::-1]
plt.figure(figsize=(10, 6))
plt.barh([feature_cols[i] for i in sorted_idx], mean_shap[sorted_idx], color='#534AB7')
plt.xlabel('Mean absolute SHAP value')
plt.title('Feature importance ranking: MLP churn model', fontsize=13)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('docs/charts/shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: docs/charts/shap_bar.png')