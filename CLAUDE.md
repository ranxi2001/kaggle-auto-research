# kaggle-auto-research

AI-agent-powered Kaggle competition framework.

## Project Structure

- `src/kaggle_auto/` — Core Python library (deterministic computation)
- `.claude/skills/` — Agent skills (AI-driven decisions)
- `workspaces/` — Isolated competition workspaces (one per competition)
- `templates/` — Competition type templates (crypto, tabular, llm)
- `cli/` — CLI entry points (`kar` command)
- `references/agents/` — Agent projects for secondary development
- `tests/` — Test suite

## Conventions

- All competition work happens inside `workspaces/<name>/`
- Never modify `templates/` during competition work
- `config.yaml` is the single source of truth for each workspace
- Features are versioned: `data/features/v{N}.parquet`
- Models are versioned: `models/v{N}/`
- Always validate submission format before submitting
- Stages communicate via filesystem (parquet, json, pkl), not in-memory

## Common Commands

```bash
kar init <name> --type <type> --url <url>   # Initialize workspace
kar research <name>                          # Run competition research
kar eda <name>                               # Run EDA
kar train <name>                             # Train models
kar pipeline <name> --full                   # Run full pipeline
kar pipeline <name> --iterate 10            # Run 10 improvement iterations (NO submit)
kar submit <name> --status                   # Check submission budget
kar submit <name> -f submissions/xxx.csv     # Submit specific file (budget-protected)
kar submit <name> --flush                    # Submit best from reserve queue
kar submit <name> --history                  # View submission history
kar analyze <name>                           # Model comparison & recommendations
kar ensemble <name>                          # Build optimized ensemble
```

## Skills

Skills in `.claude/skills/` are triggered by natural language. Each skill reads/writes the active workspace filesystem.

### Custom Skills (our pipeline)

| Skill | Trigger keywords |
|-------|-----------------|
| competition-research | "调研", "research", "analyze competition", "top solutions" |
| eda-features | "EDA", "explore data", "generate features", "特征工程" |
| model-train | "train", "训练", "tune", "ensemble", "调参" |
| submit-monitor | "submit", "提交", "leaderboard", "排名" |
| iteration-loop | "iterate", "improve", "下一步", "why score bad" |
| skill-evolution | "反思", "升级技能", "evolve", "retrospective", "总结经验" |
| notebook-workflow | "notebook", "在线跑", "push kernel", "远程训练", "GPU训练" |

### Integrated External Skills

| Skill | Source | Purpose |
|-------|--------|---------|
| agentic-kaggle-competition | FrankS-IntelLab | Real competition patterns: score stabilization, kernel workflow, spec-driven dev |
| kaggle | shepsci | Full Kaggle integration: comp reports, dataset download, notebook exec, badges |
| kaggle-agent-exam | Kaggle Official | Standardized agent exam for capability benchmarking |

## Reference Agent Architectures (for secondary development)

| Agent | Key Idea | Stars |
|-------|----------|-------|
| `references/agents/aide-core/` | **Tree search** — each script is a node, LLM generates patches as children, metric-guided pruning. 4x medals vs linear agents | 1.3k |
| `references/agents/qgentic-ai/` | **MainAgent + Researcher** — infinite loop, idea pool, LLM bash judge, live training monitor. Silver medals (Top 1%) | 60 |
| `references/agents/kaggle-agent/` | **Plan-and-Execute** — Planner/Enhancer/CodeGen/Executor cycle, vector DB memory | 84 |

## Pipeline State Machine

```
RESEARCH → EDA → FEATURES → TRAIN → EVALUATE → SUBMIT
                                ^                    │
                                └─── ITERATE ←───────┘
```

## Submission Strategy (CRITICAL)

**提交是稀缺资源，绝不浪费。** 核心规则：

1. **迭代和提交完全分离** — iterate 阶段只做本地 CV 评估，绝不触发提交
2. **每日预算 max_daily=2** — 比 Kaggle 实际限额更严格，留有余地
3. **阈值检查** — 新模型必须比当前 best CV 提升 ≥ threshold 才值得提交
4. **排队机制** — 不满足条件的自动进 reserve 队列，等条件满足再提交
5. **多样性原则** — 每天最多 1 次"探索性"提交（不同方法论验证）
6. **auto_submit 默认 OFF** — 只有最终生产环境且高度置信时才打开

典型工作流：
```
Iterate 10-20 rounds → Pick top 2 candidates → Check budget → Submit 1-2
```

## Key Design Rules

1. Skills decide WHAT to do; Python library does HOW
2. Each workspace is fully isolated — no cross-contamination
3. Pipeline can resume from any checkpoint via `.state/`
4. Configuration constrains the agent's autonomy (metric, CV, submission limits)
5. All artifacts are versioned for rollback and comparison
6. **NEVER auto-submit during iteration loops**
7. **Submission budget is enforced even with force=True**
