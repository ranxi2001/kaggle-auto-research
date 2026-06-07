---
name: iteration-loop
description: >
  Analyze current performance and drive iterative improvement using tree-search.
  Uses IdeaPool, ErrorAnalysis, and CodeGen to systematically explore the solution space.
  NEVER submits during iteration — only generates candidates for review.
  Trigger: "iterate", "improve", "下一步", "next step", "why score bad",
  "分数为什么低", "怎么提升", "error analysis", "误差分析".
---

# Iteration Loop Skill

分析当前状态，驱动 AIDE 风格的迭代优化闭环。**迭代阶段绝不提交** — 只生成候选方案。

## Critical Rule: Iteration vs Submission are SEPARATE

```
┌──────────────────────────────────────────────────────────┐
│  ITERATION PHASE (unlimited, free, no cost)              │
│                                                          │
│  Train → Evaluate CV → Compare → Pick winners            │
│  Repeat 5-20 rounds until convergence                    │
│                                                          │
│  Output: ranked list of candidates + submission files    │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼ (human decision point)
┌──────────────────────────────────────────────────────────┐
│  SUBMISSION PHASE (limited, expensive, max 2/day)        │
│                                                          │
│  Review candidates → Pick best 1-2 → Submit             │
│  Budget-protected, threshold-checked                     │
└──────────────────────────────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Iteration Loop (Agent Decision Layer)              │
│                                                     │
│  1. Analyze → 2. Ideate → 3. Execute → 4. Evaluate │
│       ↑                                       │     │
│       └───────────── Feedback ←───────────────┘     │
└─────────────────────────────────────────────────────┘
         │ Convergence reached
         ▼
   Generate final submission candidates (DO NOT submit)
```

## Workflow

### 1. 状态收集 & 分析

```python
from kaggle_auto.pipeline import IterationAnalyzer, IdeaPool

workspace = Path("workspaces/<name>")
analyzer = IterationAnalyzer(workspace)
pool = IdeaPool(workspace)

analysis = analyzer.analyze_latest()
comparisons = analyzer.compare_models()
recommendations = analyzer.get_recommendations()
```

### 2. Idea Pool Management

```python
pool.seed_from_research(workspace / "reports" / "research_notes.md")
pool.seed_from_analysis(analysis)
next_ideas = pool.get_next(n=3)
pool.mark_tried(idea.id, result="improved", metric_delta=+0.003)
```

### 3. Code Generation (AIDE-style)

```python
from kaggle_auto.pipeline import CodeGenerator

codegen = CodeGenerator(workspace)
script = codegen.generate_feature_script(
    feature_code='...',
    feature_name="ticket_survival",
    train_path=str(workspace / "data/raw/train.csv"),
    target_col="Survived",
    id_col="PassengerId",
)
result = codegen.execute(script)
```

### 4. Tree-Search Iteration (CLI)

```bash
kar pipeline <name> --iterate 10    # Run 10 iterations (NO submission)
kar analyze <name>                  # View results & recommendations
kar submit <name> --status          # Check budget before submitting
kar submit <name>                   # Submit best candidate (manual trigger)
```

### 5. 策略推荐

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 | `cv_stability > 0.1` | Reduce complexity |
| 2 | `zero_features > 5` | Feature selection |
| 3 | `feature_concentration > 0.8` | Diversify features |
| 4 | `stale_rounds >= 3` | Model switch or ensemble |
| 5 | All stable | **Generate submission candidate** (not submit!) |

### 6. 收敛判断 & 候选输出

When iteration converges:
1. Generate submission files for top 2-3 diverse approaches
2. Log their CV scores and method descriptions
3. **Report to user**: "Here are the candidates — which to submit?"
4. WAIT for explicit submit command

## Decision Framework for Agent

When the user asks "下一步" or "improve":

1. **Check submission budget** — report remaining submissions
2. **Run analysis** to understand current state
3. **Iterate locally** (5-20 rounds, no submission)
4. **Present candidates** with CV scores ranked
5. **Wait for user** to choose which to submit

**NEVER do this:**
- Call `submitter.submit()` inside an iteration loop
- Set `auto_submit: true` during experimentation
- Submit more than the budget allows

## Pipeline Integration

The pipeline `submit` stage should be OFF during development:

```yaml
# config.yaml during iteration
submission:
  auto_submit: false    # Always false during iteration!
  max_daily: 2
  best_threshold: 0.005
```

Only enable auto_submit for final production runs with strong confidence.

## State Files

```
.state/
├── pipeline_state.json
├── iteration_history.json
├── journal.json
├── idea_pool.json
├── submission_budget.json
└── scripts/
```

## Auto-Evolution Hook

迭代结束后自动触发 skill-evolution：

```
After each iteration batch completes:
1. If new best found → record what worked to model-train Lesson Log
2. If stale x3 → record that strategy is exhausted to iteration-loop Lesson Log
3. If CV-LB gap data available → update calibration in submit-monitor
4. If bug found → add to NEVER Do in relevant skill
```

## Hard Rules

### 1. 永远不在迭代中提交
即使结果非常好。生成 submission file 可以，但 API call 绝不触发。

### 2. 连续无提升时切换策略
不要在同一方向死磕。Feature → Hyperparam → Model switch → Ensemble，螺旋前进。

### 3. 记录每次实验的完整上下文
不只记分数，还要记：用了什么特征集、什么模型、什么超参。方便回溯最佳组合。

### 4. 候选输出必须有多样性
Top 3 candidates 之间的 OOF correlation < 0.95。否则只保留最好的那个。

## Lesson Log

| Date | Lesson | Impact |
|------|--------|--------|
| 2026-06-06 | Auto-submit in pipeline burned 10 submissions in minutes | Added strict budget |
| 2026-06-06 | Threshold optimization on OOF doesn't transfer to LB | Don't optimize threshold |
| 2026-06-07 | 5-model stacking not better than 2-model blend on small data | Keep it simple |
| 2026-06-07 | Multi-seed averaging: no LB gain for Titanic | Skip for n<5000 |
