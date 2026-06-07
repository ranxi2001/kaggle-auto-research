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
kar submit <name>                            # Submit best model
kar pipeline <name> --full                   # Run full pipeline
kar pipeline <name> --iterate 5             # Run 5 improvement iterations
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

## Key Design Rules

1. Skills decide WHAT to do; Python library does HOW
2. Each workspace is fully isolated — no cross-contamination
3. Pipeline can resume from any checkpoint via `.state/`
4. Configuration constrains the agent's autonomy (metric, CV, submission limits)
5. All artifacts are versioned for rollback and comparison
