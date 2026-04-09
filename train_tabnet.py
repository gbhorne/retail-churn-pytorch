# train_tabnet.py
import numpy as np, pickle, json
import matplotlib.pyplot as plt
from src.features import set_seeds, load_and_engineer, split_data, prepare_tabnet
from src.tabnet import build_and_train_tabnet, evaluate_tabnet

set_seeds()
DATA_PATH = 'data/features.csv'

df = load_and_engineer(DATA_PATH)
train, val, test = split_data(df)
X_train, X_val, X_test, y_train, y_val, y_test, scaler, cat_mappings, feature_cols, cat_idxs, cat_dims = prepare_tabnet(train, val, test)

print(f'Training TabNet on {len(y_train):,} samples')
model = build_and_train_tabnet(X_train, y_train, X_val, y_val, cat_idxs=cat_idxs, cat_dims=cat_dims)

print('Test set evaluation:')
metrics = evaluate_tabnet(model, X_test, y_test)

importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_cols)), importance[indices], color='#D85A30')
plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=45, ha='right')
plt.title('TabNet feature importances')
plt.tight_layout()
plt.savefig('docs/charts/tabnet_feature_importance.png', dpi=150)
plt.close()
print('Saved: docs/charts/tabnet_feature_importance.png')

model.save_model('tabnet_churn')
with open('tabnet_scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('tabnet_metadata.json', 'w') as f:
    json.dump({'feature_cols': feature_cols, 'cat_idxs': cat_idxs, 'cat_dims': cat_dims,
               'cat_mappings': {k: {str(kk): vv for kk, vv in v.items()} for k, v in cat_mappings.items()}}, f)
print('Saved: tabnet_churn.zip, tabnet_scaler.pkl, tabnet_metadata.json')