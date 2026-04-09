# train_tabnet.py
import numpy as np, pickle, json, time
import matplotlib.pyplot as plt
from src.features import set_seeds, load_and_engineer, split_data, prepare_tabnet
from src.tabnet import build_and_train_tabnet, evaluate_tabnet

set_seeds()
DATA_PATH = 'data/features.csv'

df = load_and_engineer(DATA_PATH)
train, val, test = split_data(df)
X_train, X_val, X_test, y_train, y_val, y_test, scaler, cat_mappings, feature_cols, cat_idxs, cat_dims = prepare_tabnet(train, val, test)

print(f'Training TabNet on {len(y_train):,} samples')
t0 = time.time()
model = build_and_train_tabnet(X_train, y_train, X_val, y_val, cat_idxs=cat_idxs, cat_dims=cat_dims)
train_seconds = round(time.time() - t0, 1)

print('Test set evaluation:')
metrics = evaluate_tabnet(model, X_test, y_test)

importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_cols)), importance[indices], color='#D85A30')
plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=45, ha='right')
plt.title('TabNet feature importances')
plt.tight_layout()
import os; os.makedirs('docs/charts', exist_ok=True)
plt.savefig('docs/charts/tabnet_feature_importance.png', dpi=150)
plt.close()
print('Saved: docs/charts/tabnet_feature_importance.png')

model.save_model('tabnet_churn')
with open('tabnet_scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('tabnet_metadata.json', 'w') as f:
    json.dump({'feature_cols': feature_cols, 'cat_idxs': cat_idxs, 'cat_dims': cat_dims,
               'cat_mappings': {k: {str(kk): vv for kk, vv in v.items()} for k, v in cat_mappings.items()}}, f)

results = {
    'model': 'TabNet',
    'roc_auc': metrics['roc_auc'],
    'pr_auc':  metrics['pr_auc'],
    'train_seconds': train_seconds,
    'device': 'cpu',
    'n_features': len(feature_cols),
    'cat_idxs': cat_idxs,
    'cat_dims': cat_dims,
    'n_train': len(y_train),
    'n_val':   len(y_val),
    'n_test':  len(y_test),
    'churn_rate_train': round(float(y_train.mean()), 4),
}
with open('tabnet_results.json', 'w') as f: json.dump(results, f, indent=2)
print(f'Saved: tabnet_churn.zip, tabnet_scaler.pkl, tabnet_metadata.json, tabnet_results.json')
print(f'Training time: {train_seconds}s')