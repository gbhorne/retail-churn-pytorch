# src/train.py
# BCEWithLogitsLoss without pos_weight -- WeightedRandomSampler already handles class balance.
import torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

def train_mlp(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=7, device='cpu'):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, patience_ctr, best_state = 0.0, 0, None
    history = {'train_loss': [], 'val_auc': []}
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                probs = torch.sigmoid(model(X_batch.to(device))).cpu().numpy()
                all_preds.extend(probs); all_labels.extend(y_batch.numpy())
        val_auc = roc_auc_score(all_labels, all_preds)
        scheduler.step(val_auc)
        history['train_loss'].append(avg_loss)
        history['val_auc'].append(val_auc)
        print(f'Epoch {epoch+1:02d} | loss {avg_loss:.4f} | val AUC {val_auc:.4f}')
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f'Early stopping at epoch {epoch+1} -- best AUC {best_auc:.4f}')
                break
    model.load_state_dict(best_state)
    print(f'Best val AUC: {best_auc:.4f}')
    return model, history