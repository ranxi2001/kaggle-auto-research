---
name: competition-research
description: >
  Research and analyze a Kaggle competition. Scrapes competition page, reads rules/data description,
  analyzes top solutions and discussion threads, produces a structured research report.
  Trigger: "调研", "research competition", "analyze this kaggle", "what's the winning approach",
  "study top solutions", "竞赛分析".
---

# Competition Research Skill

自动调研 Kaggle 竞赛，生成结构化分析报告。

## Workflow

1. **获取竞赛信息**
   - 通过 Kaggle API 获取竞赛元数据（规则、时间线、评估指标）
   - 解析数据描述页面，提取字段含义

2. **分析社区讨论**
   - 抓取高赞 Discussion 帖子（Top 10）
   - 提取关键洞察：数据泄露提示、特征工程思路、模型选择建议

3. **研究公开方案**
   - 获取排名靠前的公开 Notebook 列表
   - 分析其使用的技术栈和方法论
   - 总结共性模式和差异化策略

4. **生成调研报告**
   - 输出到 `workspaces/<name>/reports/research_notes.md`
   - 包含：赛题摘要、评估指标解读、数据 schema、推荐方案

## Usage

```bash
kar research <competition-name>
```

或在 Claude Code 中：
```
> 帮我调研一下 DRW Crypto Market Prediction 这个比赛
> 分析一下这个比赛的 top solution 都用了什么方法
```

## Output

```
reports/research_notes.md
├── Competition Summary
├── Evaluation Metric
├── Data Schema & Key Observations
├── Top Discussion Insights (5-10条)
├── Top Notebook Approaches (3-5个)
└── Recommended Strategy
```

## Dependencies

- `kaggle` CLI configured (`~/.kaggle/kaggle.json`)
- `src/kaggle_auto/research/` module
