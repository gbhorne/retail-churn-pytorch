# src/tabnet.py
# TabNet with explicit cat_idxs and cat_dims -- categoricals as integer IDs, not scaled floats.
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import numpy as np

def build_and_train_tabnet(X_train, y_train, X_val, y_val, cat_idxs, cat_dims):
    model = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5, gamma=1.5,
        n_independent=2, n_shared=2, momentum=0.02, epsilon=1e-15,
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=4,
        seed=42, verbose=1,
        optimizer_params={'lr': 2e-3, 'weight_decay': 1e-5},
        scheduler_params={'step_size': 10, 'gamma': 0.9},
        scheduler_fn=None, mask_type='sparsemax'
    )
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)], eval_name=['val'], eval_metric=['auc'],
        max_epochs=100, patience=10, batch_size=1024,
        virtual_batch_size=128, num_workers=0, drop_last=False
    )
    return model

def evaluate_tabnet(model, X_test, y_test):
    probs  = model.predict_proba(X_test)[:, 1]
    binary = (probs >= 0.5).astype(int)
    metrics = {
        'roc_auc': round(roc_auc_score(y_test, probs), 4),
        'pr_auc':  round(average_precision_score(y_test, probs), 4),
        'report':  classification_report(y_test, binary),
        'conf_matrix': confusion_matrix(y_test, binary),
    }
    print(f'ROC-AUC : {metrics["roc_auc"]}')
    print(f'PR-AUC  : {metrics["pr_auc"]}')
    print(metrics['report'])
    return metrics