"""
DRW Crypto Market Prediction — Baseline v1
LightGBM + XGBoost ensemble with feature cleaning and time-based CV.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

# === Paths ===
INPUT = Path('/kaggle/input/drw-crypto-market-prediction')
OUTPUT = Path('/kaggle/working')

# === Load Data ===
print("Loading data...")
train = pd.read_parquet(INPUT / 'train.parquet')
test = pd.read_parquet(INPUT / 'test.parquet')
print(f"Train: {train.shape}, Test: {test.shape}")

sample_sub = pd.read_csv(INPUT / 'sample_submission.csv')
print(f"Submission shape: {sample_sub.shape}")
print(f"Submission columns: {sample_sub.columns.tolist()}")

# === Identify target and ID ===
target_col = 'target'
id_col = 'id'

y_train = train[target_col].values
train_ids = train[id_col].values
test_ids = test[id_col].values

feature_cols = [c for c in train.columns if c not in [target_col, id_col]]
print(f"Total features: {len(feature_cols)}")

# === Feature Cleaning ===
print("\n--- Feature Cleaning ---")

# Drop constant features
nunique = train[feature_cols].nunique()
constant_cols = nunique[nunique <= 1].index.tolist()
print(f"Constant features to drop: {len(constant_cols)}")
feature_cols = [c for c in feature_cols if c not in constant_cols]

# Drop perfectly correlated features (keep first of each pair)
print("Checking correlations (sampling 50k rows for speed)...")
sample = train[feature_cols].sample(min(50000, len(train)), random_state=42)
corr_matrix = sample.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.999)]
print(f"Perfectly correlated features to drop: {len(to_drop_corr)}")
feature_cols = [c for c in feature_cols if c not in to_drop_corr]

print(f"Features after cleaning: {len(feature_cols)}")
del sample, corr_matrix, upper_tri
gc.collect()

# === Downcast to float32 ===
print("\nDowncasting to float32...")
train[feature_cols] = train[feature_cols].astype(np.float32)
test[feature_cols] = test[feature_cols].astype(np.float32)
gc.collect()

# === Time-based CV ===
print("\n--- Cross Validation ---")
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

X_train = train[feature_cols].values
X_test = test[feature_cols].values

# === LightGBM ===
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 100,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 1000,
    'verbose': -1,
    'random_state': 42,
}

print("\n--- LightGBM Training ---")
lgb_oof = np.zeros(len(X_train))
lgb_preds = np.zeros(len(X_test))
lgb_scores = []

for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)]
    )

    lgb_oof[va_idx] = model.predict(X_va)
    lgb_preds += model.predict(X_test) / N_SPLITS

    fold_r2 = r2_score(y_va, lgb_oof[va_idx])
    lgb_scores.append(fold_r2)
    print(f"  Fold {fold+1} R²: {fold_r2:.6f}")

    del X_tr, X_va, y_tr, y_va, model
    gc.collect()

# OOF score (only on validation indices)
va_mask = lgb_oof != 0
lgb_oof_r2 = r2_score(y_train[va_mask], lgb_oof[va_mask])
print(f"\nLGB OOF R²: {lgb_oof_r2:.6f}")
print(f"LGB Mean CV R²: {np.mean(lgb_scores):.6f} ± {np.std(lgb_scores):.6f}")

# === XGBoost ===
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 100,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 1000,
    'verbosity': 0,
    'random_state': 42,
}

print("\n--- XGBoost Training ---")
xgb_oof = np.zeros(len(X_train))
xgb_preds = np.zeros(len(X_test))
xgb_scores = []

for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=50,
        verbose=200
    )

    xgb_oof[va_idx] = model.predict(X_va)
    xgb_preds += model.predict(X_test) / N_SPLITS

    fold_r2 = r2_score(y_va, xgb_oof[va_idx])
    xgb_scores.append(fold_r2)
    print(f"  Fold {fold+1} R²: {fold_r2:.6f}")

    del X_tr, X_va, y_tr, y_va, model
    gc.collect()

va_mask_xgb = xgb_oof != 0
xgb_oof_r2 = r2_score(y_train[va_mask_xgb], xgb_oof[va_mask_xgb])
print(f"\nXGB OOF R²: {xgb_oof_r2:.6f}")
print(f"XGB Mean CV R²: {np.mean(xgb_scores):.6f} ± {np.std(xgb_scores):.6f}")

# === Ensemble ===
print("\n--- Ensemble ---")
ensemble_preds = 0.5 * lgb_preds + 0.5 * xgb_preds

# OOF ensemble score
ens_oof = 0.5 * lgb_oof + 0.5 * xgb_oof
ens_oof_r2 = r2_score(y_train[va_mask], ens_oof[va_mask])
print(f"Ensemble OOF R²: {ens_oof_r2:.6f}")

# === Prediction clipping (anti-aliasing) ===
clip_lower = np.percentile(y_train, 1)
clip_upper = np.percentile(y_train, 99)
ensemble_preds = np.clip(ensemble_preds, clip_lower, clip_upper)
print(f"Clipped predictions to [{clip_lower:.4f}, {clip_upper:.4f}]")

# === Generate Submission ===
print("\n--- Generating Submission ---")
submission = pd.DataFrame({
    id_col: test_ids,
    target_col: ensemble_preds
})
submission.to_csv(OUTPUT / 'submission.csv', index=False)
print(f"Submission saved: {submission.shape}")
print(f"Predictions stats: mean={ensemble_preds.mean():.6f}, std={ensemble_preds.std():.6f}")

# === Summary ===
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Features used: {len(feature_cols)}")
print(f"LGB CV R²: {np.mean(lgb_scores):.6f} ± {np.std(lgb_scores):.6f}")
print(f"XGB CV R²: {np.mean(xgb_scores):.6f} ± {np.std(xgb_scores):.6f}")
print(f"Ensemble OOF R²: {ens_oof_r2:.6f}")
print(f"Clip range: [{clip_lower:.4f}, {clip_upper:.4f}]")
print("="*50)
