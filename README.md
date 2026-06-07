# kaggle-auto-research

AI Agent 驱动的 Kaggle 竞赛自动化框架。通过 Claude agent skills 实现从竞赛调研到提交优化的全流程自动化。

## Features

- **竞赛调研** — 自动抓取竞赛信息、分析赛题、研究 Top Solution
- **EDA + 特征工程** — 自动数据探索、特征生成、数据清洗
- **模型训练与调参** — 模型选择、Optuna 超参搜索、集成学习
- **提交与监控** — 自动提交、排名追踪、迭代优化闭环

## Supported Competition Types

| Type | Description | Key Models |
|------|-------------|------------|
| `crypto` | 加密货币/量化交易 | LightGBM, LSTM, Transformer |
| `tabular` | 通用表格数据 | LightGBM, XGBoost, CatBoost |
| `llm` | 大模型竞赛 | Fine-tuned Transformers, Prompt Engineering |

## Quick Start

```bash
# Install
pip install -e .

# Initialize a competition workspace
kar init titanic --type tabular --url https://www.kaggle.com/competitions/titanic

# Run the full pipeline
kar pipeline titanic --full

# Or run individual stages
kar research titanic
kar eda titanic
kar train titanic
kar submit titanic
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Claude Agent Skills (AI Brain)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ Research  │ │EDA+Feat  │ │  Train   │ │ Submit │ │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └───┬────┘ │
└────────┼─────────────┼───────────┼───────────┼──────┘
         │             │           │           │
┌────────▼─────────────▼───────────▼───────────▼──────┐
│              Python Core Library                      │
│  research/ │ eda/ │ features/ │ models/ │ submission/ │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│              Competition Workspace                    │
│  config.yaml │ data/ │ models/ │ submissions/        │
└─────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
RESEARCH → EDA → FEATURES → TRAIN → EVALUATE → SUBMIT
                                ^                    │
                                └─── ITERATE ←───────┘
```

## Project Structure

```
kaggle-auto-research/
├── .claude/skills/          # Agent Skills（AI 决策层）
│   ├── competition-research/  竞赛调研与分析
│   ├── eda-features/          EDA + 特征工程
│   ├── model-train/           模型训练调参
│   ├── submit-monitor/        提交与监控
│   └── iteration-loop/        迭代优化
├── src/kaggle_auto/         # Python 核心库（执行层）
│   ├── research/              竞赛调研
│   ├── eda/                   数据探索
│   ├── features/              特征工程
│   ├── models/                模型训练
│   ├── tuning/                超参搜索
│   ├── submission/            提交管理
│   └── pipeline/              流水线编排
├── templates/               # 竞赛模板
│   ├── crypto/                加密货币/量化
│   ├── tabular/               表格数据
│   └── llm/                   大模型
├── workspaces/              # 竞赛工作区（每竞赛独立）
└── cli/                     # CLI 命令入口
```

## Usage with Claude Agent

在 Claude Code 中直接用自然语言触发 skills：

```
> 帮我调研一下 DRW Crypto Market Prediction 这个比赛
> 对当前竞赛数据做一下 EDA
> 训练一个 LightGBM baseline
> 提交最新模型的预测结果
> 分析一下为什么分数没提升，建议下一步怎么做
```

## Competition Workspace

每个竞赛有独立的工作区：

```
workspaces/<competition>/
├── config.yaml        # 竞赛配置（唯一真相源）
├── data/
│   ├── raw/           # 原始竞赛数据
│   ├── processed/     # 处理后数据
│   └── features/      # 特征存储（版本化 parquet）
├── models/            # 模型产物（版本化）
│   └── v001/
├── submissions/       # 提交历史
├── reports/           # EDA/分析报告
└── .state/            # Pipeline 状态（断点恢复）
```

## Configuration

竞赛配置示例 (`config.yaml`)：

```yaml
competition:
  name: "drw-crypto-market-prediction"
  type: "crypto"
  metric: "weighted_pearson"
  metric_direction: "maximize"

data:
  train: "data/raw/train.parquet"
  test: "data/raw/test.parquet"
  target_column: "target"

model:
  primary: "lightgbm"
  cv_strategy: "time_series_split"
  cv_folds: 5

submission:
  auto_submit: false
  best_threshold: 0.001
  max_daily: 5
```

## Requirements

- Python >= 3.11
- Kaggle API credentials (`~/.kaggle/kaggle.json`)
- Claude Code with agent skills support

## License

MIT
