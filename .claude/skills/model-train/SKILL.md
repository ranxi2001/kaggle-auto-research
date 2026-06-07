---
name: model-train
description: >
  Train models for the current competition. Handles model selection, cross-validation,
  hyperparameter tuning via Optuna, and ensemble building.
  Trigger: "train", "训练模型", "tune hyperparameters", "调参", "build ensemble",
  "集成学习", "cross validate", "try lightgbm", "baseline".
---

# Model Training Skill

自动化模型训练、调参与集成。

## Workflow

1. **数据加载**
   - 从 `data/features/` 加载最新版本特征
   - 按 `config.yaml` 设置切分验证集

2. **模型选择**
   - 根据竞赛类型选择候选模型：
     - Tabular: LightGBM, XGBoost, CatBoost, TabNet
     - Crypto: LightGBM, LSTM, Temporal Fusion Transformer
     - LLM: DeBERTa, Llama fine-tune
   - 先跑 baseline 建立基准

3. **交叉验证**
   - 按 `config.yaml` 的 cv_strategy 执行
   - 支持：KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
   - 输出每折分数 + 均值/标准差

4. **超参调优**
   - Optuna 搜索（可配置 n_trials）
   - 内置默认搜索空间，支持自定义
   - Early stopping 避免浪费

5. **集成学习**
   - 多模型 Stacking / Blending / Voting
   - 自动计算最优权重
   - OOF predictions 用于下层训练

6. **保存产物**
   - 模型文件：`models/v{N}/model.pkl`
   - 参数记录：`models/v{N}/params.json`
   - CV 分数：`models/v{N}/cv_scores.json`
   - OOF 预测：`models/v{N}/oof_preds.npy`
   - 特征重要性：`models/v{N}/importance.csv`

## Usage

```bash
kar train <competition-name>
kar train <competition-name> --model lightgbm --trials 100
kar train <competition-name> --ensemble
```

或在 Claude Code 中：
```
> 训练一个 LightGBM baseline
> 用 Optuna 调一下超参，跑 200 trials
> 把 LightGBM 和 XGBoost 做个 ensemble
```

## Output

```
models/v{N}/
├── model.pkl          # 模型文件
├── params.json        # 最优超参
├── cv_scores.json     # CV 分数记录
├── oof_preds.npy      # Out-of-fold 预测
└── importance.csv     # 特征重要性
```
