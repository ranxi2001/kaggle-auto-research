---
name: skill-evolution
description: >
  Meta-skill: automatically upgrade other skills based on new learnings from competition runs.
  Tracks what worked/failed, updates Lesson Logs, adds new Hard Rules, refines strategies.
  Trigger: "升级技能", "update skills", "evolve", "what did we learn", "反思", "retrospective",
  "skill upgrade", "进化", "总结经验".
---

# Skill Evolution (Meta-Skill)

自动升级其他 skills — 基于每次竞赛运行的经验反馈。

## Core Principle

```
竞赛执行 → 结果观察 → 提取教训 → 写入对应 Skill → 下次自动生效
```

不需要人工记住教训，不需要 memory 重新加载，教训直接嵌入工作流。

## Trigger Conditions

自动触发 (agent 应主动执行):
1. 提交后拿到 LB 分数 → 更新 CV-LB calibration
2. 发现 bug 并修复 → 写入对应 skill 的 "NEVER Do" + "Lesson Log"
3. 某个策略连续 3 次无效 → 降低优先级或移除
4. 新的有效策略确认 → 写入 "What Works" 部分

手动触发:
- 用户说 "反思" / "总结经验" / "升级技能"
- 竞赛结束复盘

## Evolution Protocol

### Step 1: Collect Evidence

```python
# From iteration history
from kaggle_auto.pipeline import IterationAnalyzer, IdeaPool
analyzer = IterationAnalyzer(workspace)
comparisons = analyzer.compare_models()

# From submission history
from kaggle_auto.submission import Submitter
submitter = Submitter(workspace, config)
# Compare CV predictions vs LB results
```

### Step 2: Extract Lessons

判断标准：
| Signal | Lesson Type | Target Skill |
|--------|-------------|--------------|
| CV >> LB (gap > 0.05) | CV strategy broken | model-train |
| New feature +0.005 CV | Feature works | eda-features |
| New feature -0.002 CV | Feature is noise | eda-features |
| Model switch no gain | Strategy exhausted | iteration-loop |
| Submit without gain | Wasted submission | submit-monitor |
| Bug found and fixed | Anti-pattern | relevant skill |

### Step 3: Write Into Skills

每个 Skill 文件有两个自动更新区域：

#### Lesson Log (append-only)
```markdown
## Lesson Log

| Date | Lesson | Impact |
|------|--------|--------|
| 2026-06-06 | Target leakage via interactions | CV 0.999→0.84 |
| 2026-06-07 | GroupKFold needed with LOO features | Better calibration |
| <new entry appended here> |
```

#### Hard Rules (accumulate, occasionally prune)
```markdown
## Hard Rules
### N+1. New Rule From Experience
<description>
**Signal**: <how to detect the problem>
```

### Step 4: Validate No Regression

升级后检查：
- [ ] Skill 文件语法正确 (YAML frontmatter + markdown)
- [ ] Hard Rules 不相互矛盾
- [ ] Lesson Log 按时间顺序
- [ ] 没有删除已有的有效规则

## Auto-Upgrade Implementation

当 agent 执行以下动作时，应自动调用 skill-evolution：

```
┌────────────────────────────────────────────────────────┐
│  Event                    │  Auto-Action               │
├────────────────────────────────────────────────────────┤
│  Bug fix applied          │  Add to NEVER Do list      │
│  LB score received        │  Update calibration        │
│  Iteration stale x3       │  Mark strategy as weak     │
│  New best model found     │  Record what worked        │
│  User says "不要这样做"    │  Add Hard Rule             │
│  Competition finished      │  Full retrospective        │
└────────────────────────────────────────────────────────┘
```

## Retrospective Template (竞赛结束时)

```markdown
## Retrospective: <competition_name>

### Score Progression
| Method | CV | LB | Rank |

### What Worked (写入对应 skill)
1. ...

### What Didn't Work (写入 NEVER Do)
1. ...

### System Gaps Found (写入 system_evolution memory)
1. ...

### Strategy for Next Similar Competition
1. ...
```

## Skill Health Metrics

定期检查各 skill 的 "健康度":

| Metric | Healthy | Needs Update |
|--------|---------|--------------|
| Last lesson added | < 7 days | > 30 days |
| Hard Rules count | 3-8 | > 15 (过于复杂) |
| Lesson Log entries | > 3 | 0 (没有学习) |
| Contradictions | 0 | > 0 |

## Cross-Skill Knowledge Transfer

某些教训适用于多个 skills：

```
"不浪费提交" → submit-monitor (primary) + iteration-loop (mention)
"Target leakage" → eda-features (primary) + model-train (validation)
"CV calibration" → model-train (primary) + submit-monitor (decision)
```

写入 primary skill 的 Hard Rules，在 related skills 的 context 中引用。

## Usage

```
> 总结一下这次竞赛的经验
> 升级一下 skills，把今天的教训写进去
> 反思：为什么 CV 和 LB 差这么多
> 进化：哪些策略该保留，哪些该废弃
```

## Evolution History

| Date | Change | Skill Affected |
|------|--------|----------------|
| 2026-06-06 | Added target leakage prevention | eda-features, model-train |
| 2026-06-06 | Added test alignment rule | model-train |
| 2026-06-07 | Added submission budget system | submit-monitor |
| 2026-06-07 | Added GroupKFold guidance | model-train, eda-features |
| 2026-06-07 | Created this meta-skill | skill-evolution |
