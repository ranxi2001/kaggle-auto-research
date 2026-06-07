---
name: eda-features
description: >
  Automated EDA and feature engineering for the current competition workspace.
  Generates profiling reports, suggests and creates features based on data patterns and competition type.
  Trigger: "EDA", "explore data", "数据探索", "generate features", "特征工程",
  "feature engineering", "data profiling", "分析数据".
---

# EDA + Feature Engineering Skill

自动化数据探索和特征工程。

## Workflow

### Phase 1: EDA

1. **数据 Profiling**
   - 加载 `data/raw/` 中的训练/测试数据
   - 统计：dtypes、缺失率、基数、分布、相关性
   - 检测：数据泄露、train/test 分布偏移、异常值

2. **可视化生成**
   - Target 分布
   - 特征与 Target 的关系图
   - 相关性热力图
   - 时序趋势（crypto 类型）

3. **输出 EDA 报告**
   - HTML 交互式报告：`reports/eda_report.html`
   - 关键发现摘要：`reports/eda_summary.md`

### Phase 2: Feature Engineering

4. **特征建议**
   - 基于竞赛类型和数据模式推荐特征方案
   - Tabular: 交叉特征、聚合、Target Encoding
   - Crypto: 技术指标 (MA/RSI/MACD)、滞后特征、波动率
   - LLM: 文本统计、Embedding 特征

5. **特征生成**
   - 生成特征代码到 `src/kaggle_auto/features/`
   - 计算特征并存储为 `data/features/v{N}.parquet`
   - 更新 `config.yaml` 的 features 段

## Usage

```bash
kar eda <competition-name>
kar eda <competition-name> --features-only
```

或在 Claude Code 中：
```
> 对当前竞赛数据做一下 EDA
> 基于 EDA 结果生成特征
> 加一些技术指标特征（MA、RSI）
```

## Output

- `reports/eda_report.html` — 交互式报告
- `reports/eda_summary.md` — 关键发现
- `data/features/v{N}.parquet` — 生成的特征
- `config.yaml` features 段更新
