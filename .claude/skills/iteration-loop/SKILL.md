---
name: iteration-loop
description: >
  Analyze current performance and drive iterative improvement using tree-search.
  Uses IdeaPool, ErrorAnalysis, and CodeGen to systematically explore the solution space.
  Trigger: "iterate", "improve", "下一步", "next step", "why score bad",
  "分数为什么低", "怎么提升", "error analysis", "误差分析".
---

# Iteration Loop Skill

分析当前状态，驱动 AIDE 风格的迭代优化闭环。

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Iteration Loop (Agent Decision Layer)              │
│                                                     │
│  1. Analyze → 2. Ideate → 3. Execute → 4. Evaluate │
│       ↑                                       │     │
│       └───────────── Feedback ←───────────────┘     │
└─────────────────────────────────────────────────────┘
```

## Workflow

### 1. 状态收集 & 分析

```python
from kaggle_auto.pipeline import IterationAnalyzer, IdeaPool

workspace = Path("workspaces/<name>")
analyzer = IterationAnalyzer(workspace)
pool = IdeaPool(workspace)

# Get current state
analysis = analyzer.analyze_latest()
comparisons = analyzer.compare_models()
recommendations = analyzer.get_recommendations()
```

Key outputs:
- `analysis["cv_mean"]` — current best CV score
- `analysis["top_features"]` — what's driving predictions
- `analysis["zero_importance_features"]` — dead weight features
- `analysis["cv_stability"]` — CV variance (overfitting signal)

### 2. Idea Pool Management

```python
# Seed ideas from research report
pool.seed_from_research(workspace / "reports" / "research_notes.md")

# Seed ideas from analysis
pool.seed_from_analysis(analysis)

# Get top untried ideas
next_ideas = pool.get_next(n=3)

# After execution, mark result
pool.mark_tried(idea.id, result="improved", metric_delta=-0.003)
```

Categories: `feature`, `model`, `preprocessing`, `ensemble`, `postprocess`

### 3. Code Generation (AIDE-style)

For custom experiments that can't be expressed as simple patches:

```python
from kaggle_auto.pipeline import CodeGenerator

codegen = CodeGenerator(workspace)

# Generate a feature experiment script
script = codegen.generate_feature_script(
    feature_code='''
# Custom feature: passenger title extraction
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3}).fillna(4)
df = df.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")
''',
    feature_name="title_extraction",
    train_path=str(workspace / "data/raw/train.csv"),
    target_col="Survived",
    id_col="PassengerId",
)

# Execute and get metric
result = codegen.execute(script)
# result = {"status": "completed", "cv_mean": 0.412, "cv_std": 0.02, ...}
```

### 4. Tree-Search Iteration (CLI)

```bash
# Run N iterations with automatic strategy selection
kar pipeline <name> --iterate 10

# View experiment tree
kar status <name>

# Get detailed analysis and recommendations
kar analyze <name>
```

### 5. 策略推荐（按预期收益排序）

Based on analysis, recommend in this priority order:

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 | `cv_stability > 0.1` | Reduce complexity (fewer leaves, more regularization) |
| 2 | `zero_features > 5` | Feature selection (drop dead features) |
| 3 | `feature_concentration > 0.8` | Diversify (new feature types) |
| 4 | `stale_rounds >= 3` | Model switch or ensemble |
| 5 | All stable | Try postprocessing or submit |

### 6. 收敛判断

- **Early Stop**: 3 consecutive rounds without improvement → change strategy
- **Plateau**: score variance < 1e-6 over 5 rounds → try radical change (ensemble, model switch)
- **Deadline Near**: switch to ensemble and threshold optimization
- **Target Reached**: lock and submit

## Decision Framework for Agent

When the user asks "下一步" or "improve":

1. **Run `kar analyze <name>`** to understand current state
2. **Check idea pool** for untried high-priority ideas
3. **Choose strategy**:
   - If `recommendations` suggest feature work → generate feature code
   - If `recommendations` suggest model changes → use pipeline `--iterate`
   - If `recommendations` suggest ensemble → build ensemble from top models
4. **Execute** the chosen strategy
5. **Report** the result with metric delta

## State Files

```
.state/
├── pipeline_state.json     # Stage completion tracking
├── iteration_history.json  # Iteration log
├── journal.json           # Experiment tree
├── idea_pool.json         # Persistent idea store
└── scripts/               # Generated experiment scripts
    ├── title_extraction.py
    ├── freq_encoding.py
    └── ...
```

## Integration with Other Skills

- **competition-research** seeds the idea pool with patterns from top solutions
- **eda-features** provides feature importance analysis input
- **model-train** executes training variants
- **submit-monitor** provides LB feedback to update idea priorities
