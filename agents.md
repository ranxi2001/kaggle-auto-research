# Agent System Architecture

kaggle-auto-research 的 Agent 系统设计。

## Agent 角色分工

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator (Claude Code)                     │
│  解读用户意图 → 调度 Skills → 管理状态 → 汇报结果                 │
└────────┬────────────┬────────────┬────────────┬────────────┬─────┘
         │            │            │            │            │
    ┌────▼───┐  ┌────▼────┐  ┌───▼────┐  ┌───▼─────┐  ┌──▼──────┐
    │Research│  │EDA+Feat │  │ Train  │  │ Submit  │  │Iteration│
    │  Agent │  │  Agent  │  │ Agent  │  │ Agent   │  │  Agent  │
    └────────┘  └─────────┘  └────────┘  └─────────┘  └─────────┘
```

## Agent 行为准则

### 1. Research Agent
- **输入**: 竞赛 URL 或 slug
- **输出**: `reports/research_notes.md`
- **行为**: 只读（fetch API + scrape），不修改代码或数据

### 2. EDA + Feature Agent
- **输入**: `data/raw/` 中的数据文件
- **输出**: `reports/eda_report.html`, `data/features/v{N}.parquet`
- **行为**: 分析数据 → 生成特征代码 → 执行特征计算
- **约束**: 特征计算不能接触 target 列（防止泄露）

### 3. Train Agent
- **输入**: 特征数据 + config.yaml 中的模型配置
- **输出**: `models/v{N}/` (model.pkl, cv_scores.json, oof_preds.npy)
- **行为**: CV 训练 → 记录分数 → 保存模型
- **约束**: 只做本地评估，**绝不触发提交**

### 4. Submit Agent
- **输入**: 模型预测 + 预算状态
- **输出**: `submissions/sub_*.csv`, `.state/submission_budget.json`
- **行为**: 格式验证 → 预算检查 → 阈值检查 → 提交或排队
- **约束**: 
  - 每日最多 `max_daily` 次（通常 2 次）
  - 必须通过 CV 阈值检查才能提交
  - 预算耗尽时自动排队，**不报错不重试**

### 5. Iteration Agent
- **输入**: 当前最佳模型 + idea pool + 分析结果
- **输出**: 新的模型版本 + 更新后的 journal
- **行为**: 选策略 → 生成实验 → 执行 → 评估 → 更新搜索树
- **约束**: 
  - **永远不调用 submit**
  - 收敛后输出候选列表供用户选择
  - 连续 3 轮无提升则切换策略

## 状态机与通信

### Pipeline 状态机
```
RESEARCH → EDA → FEATURES → TRAIN → EVALUATE → [CANDIDATE_READY]
                                ^                        │
                                └─── ITERATE ←───────────┘
                                                         │
                                                    [USER DECISION]
                                                         │
                                                         ▼
                                                      SUBMIT
```

### Agent 间通信方式
- **文件系统** — 所有中间结果通过文件传递 (parquet, json, npy, csv)
- **config.yaml** — 共享配置（metric, CV strategy, submission limits）
- **`.state/`** — 运行状态持久化（checkpoint/resume）
- **journal.json** — 实验树（树搜索状态）
- **idea_pool.json** — 共享想法池

## 决策权限矩阵

| 决策 | Agent | 需要用户确认? |
|------|-------|:---:|
| 选择特征方案 | EDA+Feat | No |
| 选择模型类型 | Train | No |
| 选择超参范围 | Train | No |
| 执行迭代 | Iteration | No |
| 生成提交文件 | Submit | No |
| **实际提交到 Kaggle** | Submit | **Yes (预算内可自动)** |
| 修改 config.yaml | Any | No |
| 删除已有模型/数据 | Any | **Yes** |
| 切换竞赛/workspace | Orchestrator | **Yes** |

## 安全边界

### 不可逆操作（需要确认）
- 提交到 Kaggle（消耗每日配额）
- 删除模型或数据文件
- 修改 `.env` 中的 credentials
- Push to git remote

### 可逆操作（自动执行）
- 训练新模型（版本化，不覆盖）
- 生成新特征（版本化，不覆盖）
- 生成提交文件（不提交）
- 更新 idea pool / journal
- 修改 config.yaml（可 git revert）

## AIDE 树搜索集成

参考 `references/agents/aide-core/` 的设计：

```
              Root (baseline)
              /    |    \
           v001  v002  v003
           /  \    |
        v004 v005 v006
                    |
                   v007 ← best
```

- 每个节点 = 一次实验（特征集 + 模型 + 超参）
- 搜索策略 = UCB (Upper Confidence Bound) 或 贪心
- 剪枝 = 连续 N 轮子节点无提升则标记为 exhausted
- 最终提交 = 选树上 metric 最佳的节点

## 典型工作流示例

### 新竞赛启动
```
用户: "帮我打这个比赛 https://kaggle.com/c/xxx"
→ Research Agent: 调研竞赛背景
→ EDA Agent: 探索数据 + 生成特征
→ Train Agent: baseline 训练
→ Iteration Agent: 10 轮迭代优化
→ Report: "以下是 top 3 候选方案：..."
→ 用户: "提交第一个"
→ Submit Agent: 预算检查 → 提交
```

### 每日优化
```
用户: "继续优化"
→ Iteration Agent: 分析 + 5 轮迭代
→ Report: "新最佳 CV=0.856 (之前 0.851)"
→ 用户: "提交"
→ Submit Agent: 阈值通过 → 预算 1/2 → 提交成功
```

### 预算耗尽
```
用户: "提交这个模型"
→ Submit Agent: 预算 0/2 → "已排队，明天可用 `kar submit --flush` 提交"
→ 用户: (次日) "kar submit titanic --flush"
→ Submit Agent: 预算 2/2 → 从队列取最佳 → 提交
```
