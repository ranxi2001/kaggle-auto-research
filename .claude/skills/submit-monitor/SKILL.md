---
name: submit-monitor
description: >
  Smart submission management with strict daily budget. Never waste submission opportunities.
  Queue candidates, submit only the best, track CV-LB gap, analyze score trends.
  Trigger: "submit", "提交", "check leaderboard", "排名", "track score",
  "submission history", "提交历史", "compare submissions", "budget", "提交预算".
---

# Submit & Monitor Skill

智能提交管理 — 严格预算控制，绝不浪费提交次数。

## Core Philosophy

**提交是稀缺资源。** 每天最多 N 次（通常 2-5 次），必须用在刀刃上。

策略优先级：
1. **先训练，后提交** — 在本地充分验证，只提交最有信心的结果
2. **预算保护** — 即使 `force=True` 也不能突破每日预算
3. **排队机制** — 不满足条件的提交进入 reserve 队列，不直接提交
4. **对比提交** — 每天留 1 次给"多样性"尝试（不同方法论的结果）

## Submission Decision Tree

```
┌─ 新预测生成 ─┐
│              │
▼              │
格式验证 ──失败──→ 拒绝 (fix format)
│ 通过
▼
CV 超过 best + threshold?
│ 否 → 加入 reserve 队列 (不提交)
│ 是
▼
今日预算还有余量?
│ 否 → 加入 reserve 队列 (等明天)
│ 是
▼
确认提交 → 记录到 budget tracker
```

## Workflow

### 1. 生成预测（不自动提交）

```python
from kaggle_auto.submission import Submitter
from kaggle_auto.config import load_config

workspace = Path("workspaces/<name>")
config = load_config(workspace)
submitter = Submitter(workspace, config)

# Generate file only — does NOT submit
sub_path = submitter.generate_submission(predictions, model_version="stacking_v2")
```

### 2. 检查预算状态

```python
status = submitter.status()
# {
#   "submitted_today": 1,
#   "remaining_today": 1,
#   "max_daily": 2,
#   "reserved_queue": 3,
#   "reserved": [...]
# }
```

### 3. 提交（有严格保护）

```python
# Normal submit — checks threshold + budget
result = submitter.submit(sub_path, cv_score=0.854, message="5-model stacking")

# Force submit — skips threshold but STILL respects budget
result = submitter.submit(sub_path, cv_score=0.854, force=True)

# If budget exhausted, result will be:
# {"success": False, "queued": True, "errors": ["Daily budget exhausted..."]}
```

### 4. 提交 Reserve 队列（第二天）

```python
# Submit the best N reserved candidates
results = submitter.submit_reserved(n=1)
```

### 5. CV-LB Gap 分析

每次拿到 LB 分数后，记录并分析：

```python
# After getting LB score from Kaggle
tracker = submitter.tracker
history = tracker.get_history()

# Analyze gap pattern
for entry in history:
    gap = entry.get("lb_score", 0) - entry.get("cv_score", 0)
    # Positive gap = CV underestimates (good)
    # Negative gap = overfitting to CV
```

## CLI Commands

```bash
kar submit <name>                 # Submit best candidate (respects budget)
kar submit <name> --dry-run      # Validate only, don't submit
kar submit <name> --status       # Show budget and queue status
kar submit <name> --flush        # Submit from reserve queue
kar submit <name> --force        # Skip threshold (still respects budget)
kar submit <name> --history      # View submission history with scores
```

## Configuration (config.yaml)

```yaml
submission:
  auto_submit: false             # NEVER auto-submit in pipeline! Manual only.
  max_daily: 2                   # Our budget (not Kaggle's limit)
  best_threshold: 0.005          # Minimum CV improvement to justify submission
  reserve_for_diversity: 1       # Reserve 1 slot for diverse approach
```

## Anti-Patterns (NEVER DO THIS)

- `auto_submit: true` in iteration loops — burns all submissions on incremental changes
- `force=True` in automated code — bypasses threshold check thoughtlessly
- Submitting every variant — use CV to decide, submit only winners
- Submitting without recording CV score — makes gap analysis impossible

## Best Practices

1. **Run experiments offline** — iterate 10-20 times before considering submission
2. **Compare diversity** — if top 3 models predict identically, only submit 1
3. **Track CV-LB gap** — if gap > 0.02, you have a validation problem to fix first
4. **Reserve 1 slot** — always save 1 daily submission for an unexpected insight
5. **Group submissions** — if testing 3 approaches, submit the best 1, not all 3

## Output

```
submissions/
├── sub_001_lgbm_v1.csv          # Generated files
├── sub_002_stacking_v2.csv
└── history.json                  # Full tracking log

.state/
└── submission_budget.json        # Budget + reserve queue
```

## CV-LB Calibration (auto-updated by skill-evolution)

| Competition | CV→LB Ratio | Threshold Adjustment |
|-------------|:-----------:|:--------------------:|
| Titanic (n=891) | 5:1 | CV+0.01 ≈ LB+0.002 |
| Tabular (n>50k) | ~1:1 | CV+0.005 ≈ LB+0.004 |

**Decision rule**: `expected_lb_delta = cv_delta / calibration_ratio`
If `expected_lb_delta < 0.003` → don't waste a submission.

## Hard Rules

### 1. Budget 不可绕过
即使 `force=True` 也不能超过 `max_daily`。这是硬限制。

### 2. 先检查再提交
每次提交前必须显示: 今日已用/剩余、CV score、预期 LB 提升。

### 3. 多样性提交
如果今天 2 个 quota，第 1 个给最佳 CV，第 2 个给不同方法论（diversity）。

### 4. 失败提交要复盘
如果 LB 比预期差很多 → 触发 skill-evolution 更新 calibration ratio。

## Lesson Log

| Date | Lesson | Impact |
|------|--------|--------|
| 2026-06-06 | 10 submissions in one session, most redundant | Built budget system |
| 2026-06-06 | LGB+XGB blend (0.770) vs multi-seed (0.768) = wasted submission | Need diversity check |
| 2026-06-06 | Threshold=0.62 hurt LB despite improving OOF | Don't optimize threshold on small data |
| 2026-06-07 | Kaggle 400 error = daily limit reached on their side | Track Kaggle's limit too |
