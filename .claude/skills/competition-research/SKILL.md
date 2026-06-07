---
name: competition-research
description: >
  Research and analyze a Kaggle competition. Scrapes competition page, reads rules/data description,
  analyzes top solutions and discussion threads, produces a structured research report.
  Seeds the IdeaPool with actionable improvement ideas.
  Trigger: "调研", "research competition", "analyze this kaggle", "what's the winning approach",
  "study top solutions", "竞赛分析".
---

# Competition Research Skill

自动调研 Kaggle 竞赛，生成结构化分析报告，播种 IdeaPool。

## Workflow

### 1. 获取竞赛信息
- Kaggle API: 竞赛元数据（规则、时间线、评估指标、数据量）
- 解析数据描述: 字段含义、数据类型、特殊约束

### 2. 分析社区讨论 (Top 10 Discussion)
- 关键洞察提取：
  - 数据泄露提示 / 已知 bug
  - CV 策略建议（GroupKFold? TimeSeriesSplit?）
  - 特征工程思路
  - 模型选择建议
  - Post-processing tricks

### 3. 研究公开方案 (Top Notebooks)
- 技术栈分析: 哪些库/模型出现频率高
- 共性模式: 什么方法是 baseline
- 差异化策略: Top 方案做了什么不同的事

### 4. 生成调研报告
```
reports/research_notes.md
├── Competition Summary (metric, deadline, prize, n_teams)
├── Data Schema & Key Observations
├── Top Discussion Insights (5-10条, 按价值排序)
├── Top Notebook Approaches (3-5个, 带代码片段)
├── Recommended Strategy (baseline → advanced roadmap)
└── IdeaPool Seeds (actionable improvement ideas)
```

### 5. 播种 IdeaPool
```python
from kaggle_auto.pipeline import IdeaPool
pool = IdeaPool(workspace)
pool.seed_from_research(workspace / "reports/research_notes.md")
# Extracts: feature ideas, model ideas, preprocessing ideas
```

## Research Output → Downstream Impact

| Research Finding | Feeds Into | Example |
|-----------------|------------|---------|
| "Top solutions use family survival rate" | EDA-Features skill | Generate LOO feature |
| "GroupKFold required for this metric" | Model-Train skill | Switch CV strategy |
| "Ensemble of LGB+XGB+CatBoost wins" | Iteration skill | Plan model diversity |
| "Post-processing: clip predictions to [0.05, 0.95]" | Submit skill | Apply before submission |

## Hard Rules

### 1. Research 不修改代码
只产出报告和 idea seeds。不自动改 config、不生成特征代码、不训练模型。

### 2. 报告必须 Actionable
每个 insight 必须附带 "so what" — 具体建议下一步该做什么。
Bad: "Top solutions use neural networks"
Good: "Top solutions use TabNet (n_steps=5, n_a=32) as diversity model in ensemble with LGB"

### 3. 识别竞赛 "地雷"
- 数据泄露警告
- 已知的 test set 特殊性
- Evaluation metric 的陷阱 (e.g., micro vs macro F1)
- 提交格式的常见错误

## Usage

```bash
kar research <name>
```

或 Claude Code 中：
```
> 调研一下这个比赛的 top solution
> 分析竞赛讨论区有什么有用信息
```

## Lesson Log

| Date | Lesson | Impact |
|------|--------|--------|
| 2026-06-06 | Research 要输出到 IdeaPool 才有后续价值 | Added seed_from_research() |
| 2026-06-07 | 识别 GroupKFold necessity 应在 research 阶段 | 避免后期发现 CV 不可信 |
