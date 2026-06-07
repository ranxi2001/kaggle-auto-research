---
name: eda-features
description: >
  Automated EDA and feature engineering. Generates profiling reports, suggests and creates
  features based on data patterns and competition type. Includes leakage prevention guardrails
  and feature quality validation.
  Trigger: "EDA", "explore data", "数据探索", "generate features", "特征工程",
  "feature engineering", "data profiling", "分析数据".
---

# EDA + Feature Engineering Skill

自动化数据探索和特征工程，内置防泄露机制。

## Hard Rules (从实战 Bug 中总结)

### 1. Target 永远不进入特征构建
```python
# In build() pipeline:
assert target_col not in df.columns, f"LEAKAGE: {target_col} found in feature input!"
```
如果发现 target 在 feature 输入中 → 立即报错，不要继续。

### 2. Train/Test 特征一致性
- 所有特征必须能同时在 train 和 test 上计算
- 不能用 train-only 的信息（如 target mean）直接当特征（需要 LOO/smoothing）
- 列名不能和原始列冲突（如生成 "Age" 时改名为 "AgeFilled"）

### 3. LOO 编码的正确实现
```python
# For group survival rate:
for idx in group:
    others = [j for j in group if j != idx]
    feature[idx] = target[others].mean()  # exclude self

# For test: use full train group mean
test_feature = test_key.map(train_group_mean).fillna(global_mean)
```

## Workflow

### Phase 1: EDA

1. **数据 Profiling**
   - dtypes、缺失率、基数、分布、相关性
   - **检测 leakage**: 任何与 target 相关性 > 0.95 的特征标记为可疑
   - **检测 shift**: train/test 分布差异 (KS test)

2. **可视化** → `reports/eda_report.html`

3. **关键发现** → `reports/eda_summary.md`

### Phase 2: Feature Engineering

4. **特征优先级** (按 ROI 排序)

| 优先级 | 类型 | 预期收益 | 风险 |
|--------|------|---------|------|
| 1 | Domain-specific (Title, FamilySize 等) | 高 | 低 |
| 2 | Group survival rate (LOO) | 高 | 中 (需要 GroupKFold) |
| 3 | Target encoding (with smoothing) | 中-高 | 中 |
| 4 | Interactions (A×B, A/B) | 中 | 低 |
| 5 | Binning/discretization | 低-中 | 低 |
| 6 | Auto-generated (featuretools) | 低 | 高 (噪声) |

5. **特征生成**
   - 生成代码到 `src/kaggle_auto/features/`
   - 注册到 registry (BaseFeature + @register)
   - 验证: 无 NaN、列名无冲突、test 可计算

6. **特征验证 Checklist**
   - [ ] Target 不在输入 DataFrame 中
   - [ ] 无 NaN (或已处理)
   - [ ] 列名不与原始列重复
   - [ ] Test 上可以计算 (无 train-only 依赖)
   - [ ] 对 CV 有正向贡献 (importance > 0)

## Competition Type Templates

### Tabular (通用)
```python
features = ["basic_stats", "interactions", "null_indicator", "target_encoding"]
```

### Titanic (domain-specific)
```python
features = ["titanic_custom"]
# Includes: Title, Sex, AgeFilled, FamilySize, TicketGroup, Cabin, Fare, Interactions
# KEY: FamilySurvRate (LOO) — 需要 GroupKFold
```

### Crypto/Time Series
```python
features = ["lag_features", "rolling_stats", "technical_indicators", "volatility"]
```

## Output

```
reports/eda_report.html      — 交互式报告
reports/eda_summary.md       — 关键发现
data/features/v{N}.parquet   — 版本化特征文件
```

## NEVER Do

- 在包含 target 的 DataFrame 上计算 interactions
- 用 "Age" 这样的名字覆盖原始列
- 用 train 的 target mean 直接当 test 特征 (必须 LOO + smoothing)
- 生成特征后不验证 test 能否计算
- 堆积大量低质量特征而不做 selection

## Lesson Log

| Date | Lesson | Impact |
|------|--------|--------|
| 2026-06-06 | interactions feature 用了 target → CV=0.999 | 修复后 CV 降到 0.84 (正常) |
| 2026-06-06 | "Age" 列名冲突 → ValueError duplicate | 改名 "AgeFilled" |
| 2026-06-07 | FamilySurvRate (LOO) 贡献 +0.008 accuracy | 关键特征 |
| 2026-06-07 | GroupKFold 必须配合 group features 使用 | 否则 CV 虚高 |
