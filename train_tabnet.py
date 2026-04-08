# train_tabnet.py
# End-to-end TabNet training script.
# Run: python train_tabnet.py

import numpy as np
import matplotlib.pyplot as plt
from src.features import load_and_engineer, get_feature_cols, split_and_scale
from src.tabnet import build_and_train_tabnet, evaluate_tabnet

DATA_PATH = "data/features.csv"

# load and prepare data — TabNet takes raw numpy arrays, no DataLoader needed
df           = load_and_engineer(DATA_PATH)
feature_cols = get_feature_cols(df)
X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df, feature_cols)

# TabNet needs integer labels
y_train = y_train.astype(int)
y_val   = y_val.astype(int)
y_test  = y_test.astype(int)

print(f"Training TabNet on {len(y_train):,} samples with {len(feature_cols)} features\n")

model = build_and_train_tabnet(X_train, y_train, X_val, y_val)

print("\nTest set evaluation:")
metrics = evaluate_tabnet(model, X_test, y_test)

# save feature importance plot
importance = model.feature_importances_
indices    = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_cols)), importance[indices])
plt.xticks(range(len(feature_cols)),
           [feature_cols[i] for i in indices],
           rotation=45, ha="right")
plt.title("TabNet feature importances")
plt.tight_layout()
plt.savefig("tabnet_feature_importance.png", dpi=150)
plt.close()
print("Feature importance plot saved to tabnet_feature_importance.png")

# save model
model.save_model("tabnet_churn")
print("Model saved to tabnet_churn.zip")