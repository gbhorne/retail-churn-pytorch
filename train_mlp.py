# train_mlp.py
import torch, pickle, json
from src.features import set_seeds, load_and_engineer, split_data, prepare_mlp
from src.dataset import make_loader
from src.mlp import ChurnMLP
from src.train import train_mlp
from src.evaluate import evaluate_mlp

set_seeds()
DATA_PATH = 'data/features.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

df = load_and_engineer(DATA_PATH)
train, val, test = split_data(df)
X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols = prepare_mlp(train, val, test)

train_loader = make_loader(X_train, y_train, batch_size=512, oversample=True)
val_loader   = make_loader(X_val,   y_val,   batch_size=512)
test_loader  = make_loader(X_test,  y_test,  batch_size=512, shuffle=False)

model = ChurnMLP(input_dim=len(feature_cols))
print(f'Input features: {len(feature_cols)}')
model, history = train_mlp(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=7, device=device)

print('Test set evaluation:')
metrics = evaluate_mlp(model, test_loader, device=device)

torch.save(model.state_dict(), 'mlp_churn.pth')
with open('mlp_scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('mlp_feature_cols.json', 'w') as f: json.dump(feature_cols, f)
print('Saved: mlp_churn.pth, mlp_scaler.pkl, mlp_feature_cols.json')