# src/features.py
# Feature engineering shared by MLP and TabNet models.
# Loads features.csv, engineers new features, encodes categoricals,
# splits into train/val/test, and scales numerics.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

CATEGORICAL_COLS = ["traffic_source", "country", "segment"]
TARGET           = "churned"

NUMERIC_COLS = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "active_months",
    "r_score",
    "f_score",
    "m_score",
    "age",
]


def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # log-scale right-skewed numerics to compress outliers
    for col in ["recency_days", "frequency", "monetary", "avg_order_value"]:
        df[f"{col}_log"] = np.log1p(df[col])

    # composite RFM score
    df["rfm_score"] = (df["r_score"] + df["f_score"] + df["m_score"]) / 3.0

    # recency x frequency interaction — captures "used to buy often but stopped"
    df["recency_x_frequency"] = df["recency_days"] * df["frequency"]

    # encode categoricals as integers
    for col in CATEGORICAL_COLS:
        df[col] = LabelEncoder().fit_transform(df[col].fillna("unknown"))

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    log_cols         = [c for c in df.columns if c.endswith("_log")]
    interaction_cols = ["rfm_score", "recency_x_frequency"]
    return NUMERIC_COLS + log_cols + interaction_cols + CATEGORICAL_COLS


def split_and_scale(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].values
    y = df[TARGET].values

    # stratified split preserves churn ratio in each set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    # fit scaler on train only — never on val or test
    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train)
    X_val_s     = scaler.transform(X_val)
    X_test_s    = scaler.transform(X_test)

    print(f"Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Churn rate — Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")

    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler