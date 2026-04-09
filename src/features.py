# src/features.py
# MLP path: one-hot encode categoricals, StandardScaler on all features.
# TabNet path: scale numerics only, pass categoricals as integer IDs.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random, torch
NUMERIC_COLS = ['recency_days','frequency','monetary','avg_order_value','active_months','r_score','f_score','m_score','age']
CATEGORICAL_COLS = ['traffic_source','country','segment']
TARGET = 'churned'
SEED   = 42

def set_seeds():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_and_engineer(path):
    df = pd.read_csv(path)
    for col in ['recency_days','frequency','monetary','avg_order_value']:
        df[f'{col}_log'] = np.log1p(df[col])
    df['rfm_score'] = (df['r_score'] + df['f_score'] + df['m_score']) / 3.0
    df['recency_x_frequency'] = df['recency_days'] * df['frequency']
    return df

def split_data(df):
    train, test = train_test_split(df, test_size=0.15, stratify=df[TARGET], random_state=SEED)
    train, val  = train_test_split(train, test_size=0.15, stratify=train[TARGET], random_state=SEED)
    print(f'Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}')
    print(f'Churn - Train: {train[TARGET].mean():.3f} | Val: {val[TARGET].mean():.3f} | Test: {test[TARGET].mean():.3f}')
    return train, val, test

def prepare_mlp(train, val, test):
    log_cols = [c for c in train.columns if c.endswith('_log')]
    numeric_features = NUMERIC_COLS + log_cols + ['rfm_score','recency_x_frequency']
    train_enc = pd.get_dummies(train[CATEGORICAL_COLS], drop_first=False)
    val_enc   = pd.get_dummies(val[CATEGORICAL_COLS],   drop_first=False)
    test_enc  = pd.get_dummies(test[CATEGORICAL_COLS],  drop_first=False)
    train_enc, val_enc  = train_enc.align(val_enc,  join='left', axis=1, fill_value=0)
    train_enc, test_enc = train_enc.align(test_enc, join='left', axis=1, fill_value=0)
    cat_cols     = list(train_enc.columns)
    feature_cols = numeric_features + cat_cols
    X_tr = np.hstack([train[numeric_features].values, train_enc.values]).astype(np.float32)
    X_va = np.hstack([val[numeric_features].values,   val_enc.values]).astype(np.float32)
    X_te = np.hstack([test[numeric_features].values,  test_enc.values]).astype(np.float32)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    y_train = train[TARGET].values.astype(np.float32)
    y_val   = val[TARGET].values.astype(np.float32)
    y_test  = test[TARGET].values.astype(np.float32)
    print(f'MLP features: {len(feature_cols)} ({len(numeric_features)} numeric + {len(cat_cols)} one-hot)')
    return X_tr_s, X_va_s, X_te_s, y_train, y_val, y_test, scaler, feature_cols

def prepare_tabnet(train, val, test):
    log_cols = [c for c in train.columns if c.endswith('_log')]
    numeric_features = NUMERIC_COLS + log_cols + ['rfm_score','recency_x_frequency']
    cat_mappings = {}
    for col in CATEGORICAL_COLS:
        cats = sorted(train[col].unique())
        cat_mappings[col] = {v: i for i, v in enumerate(cats)}
    def apply_mapping(df, col): return df[col].map(cat_mappings[col]).fillna(0).astype(int)
    feature_cols = numeric_features + CATEGORICAL_COLS
    cat_idxs = [feature_cols.index(c) for c in CATEGORICAL_COLS]
    cat_dims = [len(cat_mappings[c]) + 1 for c in CATEGORICAL_COLS]
    scaler = StandardScaler()
    num_tr = scaler.fit_transform(train[numeric_features].values)
    num_va = scaler.transform(val[numeric_features].values)
    num_te = scaler.transform(test[numeric_features].values)
    def cat_arr(df): return np.stack([apply_mapping(df, c) for c in CATEGORICAL_COLS], axis=1)
    X_tr = np.hstack([num_tr, cat_arr(train)]).astype(np.float32)
    X_va = np.hstack([num_va, cat_arr(val)]).astype(np.float32)
    X_te = np.hstack([num_te, cat_arr(test)]).astype(np.float32)
    y_train = train[TARGET].values.astype(int)
    y_val   = val[TARGET].values.astype(int)
    y_test  = test[TARGET].values.astype(int)
    print(f'TabNet features: {len(feature_cols)}, cat_idxs: {cat_idxs}, cat_dims: {cat_dims}')
    return X_tr, X_va, X_te, y_train, y_val, y_test, scaler, cat_mappings, feature_cols, cat_idxs, cat_dims