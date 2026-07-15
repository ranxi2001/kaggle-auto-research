# Agent System Architecture

kaggle-auto-research 的 Agent 系统设计与实战约束。

## Loop Engineer 循环契约

根目录 `PROGRESS.md` 是跨会话、跨 workspace 的短期交接记忆。它只记录当前主线、已验证状态、阻塞点、已排除路径、下一步和停止条件，不是日报或实验数据库。

每个新会话或上下文交接必须：

1. 先读取 `AGENTS.md`。
2. 再读取 `PROGRESS.md`，确认 active workspace、当前 gate 和已有用户决策。
3. 执行多步工程或竞赛循环时，在第一个有意义的修改前将 `Current Loop` 更新为实际任务。
4. 结束前只写入已验证事实、证据路径、未解风险和唯一明确的下一步；删除已过期的短期状态，不追加终端滚屏或聊天摘要。

`PROGRESS.md` 只负责指向事实：比赛配置仍以 workspace `config.yaml` 为准，实验和提交结论必须落到 `models/`、`reports/`、`journal.json`、`idea_pool.json` 和 `.state/` 等稳定产物中。不得只依赖聊天历史或 `PROGRESS.md` 中的摘要替代原始证据。

## Agent 角色分工

```text
Orchestrator (Claude Code)
  解读用户意图 -> 调度 Skills/CLI -> 管理状态 -> 汇报结果

  Research Agent -> EDA + Feature Agent -> Train Agent -> Submit Agent
                                      \-> Iteration Agent
```

## 通用原则

- 所有竞赛工作必须发生在 `workspaces/<competition>/` 下。
- `config.yaml` 是 workspace 的单一事实源，但必须以真实数据 schema 校验后修正。
- 数据、特征、模型、提交文件都通过文件系统传递，不依赖内存态。
- 所有大产物必须版本化保存，不覆盖已有模型/特征/提交文件。
- 迭代和提交严格分离：训练、分析、集成可以自动执行，实际 Kaggle 提交必须受预算和用户决策约束。
- CLI 必须优先提供短命令，例如 `kar auth`、`kar data <workspace>`，避免要求用户直接调用 `.venv` 路径。
- 不提交 Kaggle 数据、模型、提交文件、notebook cache、本地凭据或大体积 workspace artifact。

## Agent 行为准则

### 1. Research Agent

- **输入**: 竞赛 URL 或 slug。
- **输出**: `reports/research_notes.md`。
- **行为**: 只读抓取竞赛信息、数据文件、公开 notebooks、讨论和 baseline 思路。
- **约束**:
  - 不修改代码或数据。
  - 如果 Kaggle API / notebook 抓取失败，且已有非空 `research_notes.md`，不得用空模板覆盖旧报告。
  - 抓取结果不完整时，应在报告或返回结果中标明来源缺失，而不是伪造结论。

### 2. EDA + Feature Agent

- **输入**: `data/raw/` 中的数据文件。
- **输出**: `reports/eda_summary.md`, `data/features/v{N}.parquet`。
- **行为**: 分析数据 schema、目标列、提交格式、缺失值、常量列、类型、分布漂移，并生成可训练特征。
- **约束**:
  - 特征计算不能接触 target 列，防止泄露。
  - 面对 GB 级 parquet，默认采样 EDA，例如最多 100k 行，并在报告中写明采样范围和真实总行数。
  - 生成特征前必须确认 `config.yaml` 中的 `target_column`、`id_column`、`sample_submission` 与真实文件一致。
  - 如果发现配置和数据不一致，应先修正配置，再生成新版本特征。

### 3. Train Agent

- **输入**: 特征数据 + `config.yaml` 中的模型配置。
- **输出**: `models/v{N}/`，包含 `model.pkl`, `cv_scores.json`, `oof_preds.npy`, `importance.csv`，有测试集时保存 `test_preds.npy`。
- **行为**: CV 训练、记录分数、保存模型和预测。
- **约束**:
  - 只做本地评估，绝不触发提交。
  - metric 必须严格匹配比赛指标和方向。例如 DRW Crypto 使用 weighted/Pearson-style metric 时，不能默认用 RMSE/minimize。
  - 时间序列、金融或交易类数据默认使用 time-based CV，不使用随机切分。
  - 训练失败时保留错误信息，不覆盖已有成功模型。

### 4. Submit Agent

- **输入**: 模型预测 + sample submission + 预算状态。
- **输出**: `submissions/sub_*.csv`, `.state/submission_budget.json`。
- **行为**: 生成提交文件、格式验证、预算检查、阈值检查、提交或排队。
- **约束**:
  - 默认只生成和 dry-run 验证，不自动提交。
  - 每日最多 `max_daily` 次，通常 2 次。
  - 必须通过 CV 阈值检查才值得提交。
  - 预算耗尽时自动排队，不报错不重试。
  - 实际提交到 Kaggle 是不可逆操作，必须有明确用户确认或显式 auto-submit 配置。

### 5. Iteration Agent

- **输入**: 当前最佳模型 + idea pool + 分析结果。
- **输出**: 新模型版本 + 更新后的 journal / idea pool。
- **行为**: 选策略、生成实验、执行、评估、更新搜索树。
- **约束**:
  - 永远不调用 submit。
  - 收敛后输出候选列表供用户选择。
  - 连续 3 轮无提升则切换策略。
  - 对低信噪比金融数据，优先尝试清洗、特征选择、强正则、去常量列、去重复/高度相关列，再考虑复杂模型。

## 状态机与通信

```text
RESEARCH -> EDA -> FEATURES -> TRAIN -> EVALUATE -> CANDIDATE_READY
                                ^                       |
                                |                       v
                             ITERATE              USER DECISION
                                                        |
                                                        v
                                                     SUBMIT
```

## Agent 间通信方式

- **文件系统**: parquet, json, npy, csv。
- **config.yaml**: metric、CV strategy、submission limits、schema 映射。
- **.state/**: checkpoint、预算、pipeline 状态。
- **journal.json**: 实验树。
- **idea_pool.json**: 共享想法池。

## 开源项目 Agent 产物契约

这个仓库的目标是让不同 Coding Agent 都能接手同一个比赛。因此每个自动化能力都应优先写入稳定产物，而不是只把信息留在终端输出里。

### 实验产物

每次训练或候选实验应写入：

```text
models/vNNN/
  run.json              # command, git commit, params, data paths, runtime, status
  cv_scores.json        # metric name, direction, fold scores, mean/std
  oof_preds.npy         # out-of-fold predictions
  test_preds.npy        # test predictions when available
  feature_list.txt      # exact feature names and order
  importance.csv        # optional but preferred
  model.pkl             # or models.pkl for fold models
```

`run.json` 还应记录 parent run、template/recipe name、random seed、CV splitter、target column、ID column、data fingerprint 和错误信息。失败实验也应保留状态文件，避免下一位 Agent 重复跑同一个失败方向。

### 提交产物

每个 submission CSV 应配套一个同名 JSON：

```text
submissions/sub_xxx.csv
submissions/sub_xxx.json
```

JSON 至少应包含 source model versions、ensemble weights、local CV score、metric、生成命令、生成时间、dry-run 验证结果，以及真实提交后的 Kaggle submission id、LB score、rank。手工指定 `--file` 提交时也必须尽量读取相邻 JSON 元数据。

### CLI 输出

- 面向人的默认输出可以使用表格和颜色。
- 面向 Agent 的命令应逐步提供 `--json`，字段名稳定，失败时也返回可解析错误。
- 长任务应输出阶段进度，并在结束时写入可复查的本地 artifact。
- 不要让成功/失败只存在于日志滚屏中。

## 决策权限矩阵

| 决策 | Agent | 需要用户确认? |
|------|-------|:---:|
| 选择特征方案 | EDA+Feat | No |
| 选择模型类型 | Train | No |
| 选择超参范围 | Train | No |
| 执行迭代 | Iteration | No |
| 生成提交文件 | Submit | No |
| dry-run 验证提交文件 | Submit | No |
| 实际提交到 Kaggle | Submit | Yes |
| 修改 config.yaml | Any | No |
| 删除已有模型/数据 | Any | Yes |
| 切换竞赛/workspace | Orchestrator | Yes |
| Push to git remote | Orchestrator | Yes |

## 安全边界

### 不可逆操作，需要确认

- 提交到 Kaggle。
- 删除模型或数据文件。
- 修改 `.env` 或 credentials。
- Push to git remote。

### 可逆操作，可自动执行

- 训练新模型，版本化保存。
- 生成新特征，版本化保存。
- 生成提交文件但不提交。
- 更新 idea pool / journal。
- 修改 `config.yaml`。
- 下载或解压竞赛数据到 `data/raw/`。

## 实战后工具链规则

DRW Crypto 暴露出一些通用规则，后续 Agent 应默认遵守：

- 训练前先确认真实数据 schema，不要只相信模板配置。
- 训练前确认 metric name、direction、CV splitter 和 competition metric 一致。
- 金融、交易、时间序列数据默认使用 time-based CV；随机 KFold 只能作为对照。
- 对 GB 级 parquet 默认先读取 metadata 或采样，不直接全量 EDA。
- 低信噪比数据优先尝试简单、强正则、可解释的模型和特征选择；复杂模型不是默认第一选择。
- 每次发现有效的一次性命令，应评估是否能沉淀为 recipe/template。
- 真实提交后应同步 LB score 到本地 submission history 和实验日志。
- 如果本地 CV 与 LB 明显背离，应先分析 split/metric/leakage，不要盲目继续堆模型。

## AIDE 树搜索集成

```text
Root baseline
  |- v001
  |- v002
  |    |- v004
  |- v003
       |- v005 best
```

- 每个节点代表一次实验：特征集 + 模型 + 超参。
- 搜索策略可以是 UCB、贪心或基于 stale count 的策略切换。
- 连续 N 轮子节点无提升时标记 exhausted。
- 最终提交候选来自树上本地 metric 最佳节点，而不是最近节点。

## 典型工作流

### 新竞赛启动

```bash
kar init <name> --type <type> --url <url>
kar auth
kar data <name>
kar research <name>
kar eda <name>
kar train <name>
kar pipeline <name> --from evaluate
kar submit <name> --file submissions/sub_xxx.csv --dry-run
```

### 每日优化

```bash
kar pipeline <name> --iterate 5
kar analyze <name>
kar ensemble <name>
kar submit <name> --status
```

### 预算耗尽

```bash
kar submit <name> --file submissions/sub_xxx.csv
# 若预算耗尽，进入 reserve queue
kar submit <name> --flush
```

## DRW Crypto 实战记录

- Kaggle OAuth 应通过 `kar auth` 处理。
- 数据下载和解压应通过 `kar data drw-crypto` 处理。
- DRW 真实 target 列为 `label`。
- sample submission 为 `ID,prediction`。
- 本地 EDA 应采样读取大 parquet。
- baseline LightGBM 首轮 CV R2 为负，说明需要清洗和特征选择，不应直接提交。
- 当前已验证 DRW 使用 Pearson-style local objective；不要再用 RMSE/minimize 做主排序。
- 当前最佳本地候选来自 rank-normalized OOF ensemble，真实 LB 仍需要少量提交校准。
