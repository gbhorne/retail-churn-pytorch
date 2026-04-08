# train_mlp.py
# End-to-end MLP training script.
# Run: python train_mlp.py

import torch
from src.features import load_and_engineer, get_feature_cols, split_and_scale
from src.dataset import make_loader
from src.mlp import ChurnMLP
from src.train import train_mlp
from src.evaluate import evaluate_mlp

DATA_PATH = "data/features.csv"
device    = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load and prepare data
df           = load_and_engineer(DATA_PATH)
feature_cols = get_feature_cols(df)
X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df, feature_cols)

# build dataloaders
train_loader = make_loader(X_train, y_train, batch_size=512, oversample=True)
val_loader   = make_loader(X_val,   y_val,   batch_size=512)
test_loader  = make_loader(X_test,  y_test,  batch_size=512, shuffle=False)

# build and train model
model = ChurnMLP(input_dim=len(feature_cols))
print(f"\nModel architecture:\n{model}\n")

model, history = train_mlp(
    model, train_loader, val_loader,
    epochs=50, lr=1e-3, patience=7, device=device
)

# evaluate on held-out test set
print("\nTest set evaluation:")
metrics = evaluate_mlp(model, test_loader, device=device)

# save model weights
torch.save(model.state_dict(), "mlp_churn.pth")
print("Model saved to mlp_churn.pth")