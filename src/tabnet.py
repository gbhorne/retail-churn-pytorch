# src/tabnet.py
# TabNet wrapper for churn classification.
# TabNet uses sequential attention to select which features matter
# per sample at each decision step — gives us feature masks for free.

from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np


def build_and_train_tabnet(X_train, y_train, X_val, y_val) -> TabNetClassifier:

    model = TabNetClassifier(
        n_d=32,           # width of decision step embedding
        n_a=32,           # width of attention embedding
        n_steps=5,        # number of sequential attention steps
        gamma=1.5,        # coefficient for feature reusage in attention
        n_independent=2,  # independent GLU layers per step
        n_shared=2,       # shared GLU layers across steps
        momentum=0.02,
        epsilon=1e-15,
        seed=42,
        verbose=1,
        optimizer_params={"lr": 2e-3, "weight_decay": 1e-5},
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=None,
        mask_type="sparsemax"  # sparse attention — most weights go to zero
    )

    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    return model


def evaluate_tabnet(model, X_test, y_test) -> dict:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        classification_report, confusion_matrix
    )

    probs  = model.predict_proba(X_test)[:, 1]
    binary = (probs >= 0.5).astype(int)

    metrics = {
        "roc_auc":     round(roc_auc_score(y_test, probs), 4),
        "pr_auc":      round(average_precision_score(y_test, probs), 4),
        "report":      classification_report(y_test, binary),
        "conf_matrix": confusion_matrix(y_test, binary),
    }

    print(f"ROC-AUC : {metrics['roc_auc']}")
    print(f"PR-AUC  : {metrics['pr_auc']}")
    print(metrics["report"])

    return metrics