---
name: model-train
description: >
  Train models for the current competition. Handles model selection, cross-validation,
  hyperparameter tuning via Optuna, and ensemble building. Includes hard-won lessons
  on leakage prevention, test alignment, and CV reliability.
  Trigger: "train", "训练模型", "tune hyperparameters", "调参", "build ensemble",
  "集成学习", "cross validate", "try lightgbm", "baseline".
---

# Model Training Skill

自动化模型训练、调参与集成。

## Hard Rules (从实战 Bug 中总结)

### 1. Target Leakage Prevention
```python
# ALWAYS separate target before ANY feature computation
y = df[target_col].values
feature_df = df.drop(columns=[target_col])
X = build_features(feature_df)  # target NEVER touches build()
```
**Signal**: CV > 0.99 = leakage. Stop immediately, don't submit.

### 2. Test Feature Alignment
```python
# Test MUST go through identical pipeline
test_feat = build(same_feature_names, test_df)
# Then align columns
for col in train_X.columns:
    if col not in test_feat.columns:
        test_feat[col] = 0
test_feat = test_feat[train_X.columns]
```
**Signal**: LB << CV by huge margin (e.g., LB=0.37 vs CV=0.84) = test misalignment.

### 3. GroupKFold When Using Group Features
When FamilySurvRate / TicketSurvRate / any LOO-computed feature is present:
- Same group members MUST be in same fold
- Use `GroupKFold` with ticket/surname as group key
- GroupKFold CV is lower but more honest than StratifiedKFold

### 4. CV Calibration (Don't Trust Absolute Values)
- Small data (n<5000): CV overestimates LB by 3-10x the improvement
- Titanic observed: CV +0.01 ≈ LB +0.002
- Rule: if `cv_delta < 0.005`, treat as noise, don't submit

## Workflow

### 1. 数据加载
- 从 `data/features/` 加载最新版本特征
- 按 `config.yaml` 设置切分验证集

### 2. 模型选择
根据竞赛类型和数据量：

| 条件 | 推荐 |
|------|------|
| n > 50k, tabular | LightGBM → XGBoost → CatBoost |
| n < 5k, tabular | RF + LR + simple LGB (少叶子) |
| crypto/timeseries | LightGBM + lag features |
| text/LLM | DeBERTa fine-tune |

### 3. 交叉验证
- 支持：KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
- **默认 5-fold**；小数据用 10-fold 减 variance
- 输出每折分数 + 均值/标准差 + OOF predictions

### 4. 超参调优 (Optuna)
- 先跑默认参数 baseline
- Optuna 搜索 50-200 trials
- Early stopping: patience=50 on val metric
- 保存 best params to `models/v{N}/params.json`

### 5. Ensemble 策略

| 数据量 | 策略 |
|--------|------|
| 小 (n<5k) | 2-3 模型 weighted blend 就够 |
| 中 (5k-50k) | 5 模型 blend + Nelder-Mead 权重优化 |
| 大 (>50k) | 2-layer stacking (LR/Ridge meta-learner) |

**Diversity check**: `np.corrcoef(oof_preds)` — 相关性 > 0.95 的模型不要同时入选。

### 6. 保存产物
```
models/v{N}/
├── model.pkl          # 模型文件
├── params.json        # 超参
├── cv_scores.json     # CV 分数 + meta info
├── oof_preds.npy      # Out-of-fold 预测 (for ensemble)
├── test_preds.npy     # Test 预测 (for submission)
└── importance.csv     # 特征重要性
```

## Usage

```bash
kar train <name>                          # Baseline
kar train <name> --model xgboost          # Specific model
kar train <name> --trials 100             # With Optuna
kar ensemble <name> --top 5               # Ensemble top 5
```

## NEVER Do

- Train on data that includes target column (leakage)
- Predict test without same feature pipeline (misalignment)
- Use StratifiedKFold with group survival features (inflated CV)
- Submit based on CV alone without considering calibration ratio
- Auto-submit after training (use submit skill with budget check)

## Lesson Log

| Date | Lesson | Impact |
|------|--------|--------|
| 2026-06-06 | Target leakage via interactions feature | CV 0.999→0.84, LB 0.37→0.77 |
| 2026-06-06 | Test not going through feature pipeline | LB 0.37→0.77 |
| 2026-06-06 | Multi-seed averaging: no LB gain on small data | Wasted 1 submission |
| 2026-06-07 | GroupKFold: lower CV but more honest | Better calibration |
