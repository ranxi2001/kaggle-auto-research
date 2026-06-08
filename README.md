# Kaggle Auto Research

面向 Kaggle 竞赛的 Agent 化自动研究框架。

Kaggle Auto Research 把竞赛调研、数据下载、EDA、特征生成、模型训练、实验迭代、集成、提交验证和排行榜反馈拆成可复用 CLI 与版本化 workspace artifacts。目标不是写一次性的比赛脚本，而是让人类和 Coding Agent 能持续、安全、可复现地推进竞赛结果。

> 当前项目处于实验阶段，适合研究、原型验证和内部竞赛自动化。默认不会自动向 Kaggle 提交，真实提交必须显式执行。

## 项目状态

- 当前版本：`0.1.0`
- Python：`>=3.11`
- 主要场景：Kaggle tabular / time-series / low-signal competition automation
- 默认策略：本地生成、验证和记录；不自动真实提交
- 公开项目入口：[README-EN.md](README-EN.md)、[AGENTS.md](AGENTS.md)、[CONTRIBUTING.md](CONTRIBUTING.md)、[SECURITY.md](SECURITY.md)、[CHANGELOG.md](CHANGELOG.md)

## 核心特性

- **Agent-first 工作流**：Research、EDA、Feature、Train、Iteration、Submit 等阶段通过文件系统通信，便于 Agent 检查、恢复和继续优化。
- **Workspace 隔离**：每个竞赛在 `workspaces/<competition>/` 下独立保存配置、数据、模型、报告和提交。
- **配置即事实源**：`config.yaml` 管理数据路径、目标列、提交格式、metric、CV 策略、模型和提交预算。
- **版本化产物**：特征、模型、OOF、测试预测、提交文件都按版本保存，避免覆盖历史结果。
- **提交安全**：默认只生成和 dry-run 验证提交文件；真实 Kaggle 提交受预算和用户确认约束。
- **公开 notebook 可吸收**：支持拉取公开 notebook，后续会进一步挖掘特征列表、模型参数、CV 和集成公式。
- **Agent 工具链路线图**：见 [docs/agent-tooling-roadmap.md](docs/agent-tooling-roadmap.md)。

## 当前 DRW Crypto 进展

这个仓库正在用 DRW Crypto Market Prediction 做工具链实战验证。

- 初始自动特征 baseline：CV R2 `-0.002859`
- 清洗特征 + LightGBM：最佳单模型 Pearson 约 `0.0724`
- Pearson OOF grid ensemble：Pearson `0.077989`
- 当前最佳本地 submission：`sub_ensemble_v010_v011_v007_v003.csv`
- 公开榜 top 约 `0.11 - 0.14`

结论：工具链已经能跑通并持续改善，但距离真正强结果还有差距。下一步重点是复现公开 notebook 的 25 特征、time-decay slices、XGB/LGBM/Ridge ensemble，并把它们沉淀成可复用 recipe。

## 安装

```bash
git clone https://github.com/ranxi2001/kaggel-auto-research.git
cd kaggel-auto-research

python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Windows 项目根目录提供了短命令包装：

```bash
.\kar auth
.\kar ls
```

Git Bash 中可以使用：

```bash
./kar.cmd auth
```

开发依赖：

```bash
pip install -e ".[dev]"
```

## 快速开始

```bash
# 1. 登录 Kaggle
kar auth

# 2. 创建 workspace
kar init titanic --type tabular --url https://www.kaggle.com/competitions/titanic

# 3. 下载并解压数据
kar data titanic

# 4. 分阶段执行
kar research titanic
kar eda titanic
kar train titanic

# 5. 生成/验证提交文件，不实际提交
kar submit titanic --dry-run -f submissions/sub_001.csv
```

完整 pipeline：

```bash
kar pipeline titanic --full
```

迭代优化：

```bash
kar pipeline titanic --iterate 5
kar analyze titanic
kar ensemble titanic
```

## 常用 CLI

```bash
kar ls
kar auth
kar init <name> --type tabular --url <competition-url>
kar data <name>
kar research <name>
kar eda <name>
kar eda <name> --features-only
kar train <name>
kar train <name> --trials 30
kar pipeline <name> --full
kar pipeline <name> --from train
kar pipeline <name> --iterate 5
kar status <name>
kar analyze <name>
kar ensemble <name> --top 5
kar leaderboard <name>
kar leaderboard <name> --top -n 10
kar submit <name> --status
kar submit <name> --history
kar submit <name> --dry-run -f submissions/sub_001.csv
kar submit <name> --flush
```

DRW Crypto 当前研究命令：

```bash
kar drw-clean drw-crypto --top-k 130 --n-estimators 1200 --learning-rate 0.015
kar drw-public drw-crypto --model lgbm --folds 3
kar drw-ensemble drw-crypto --models v010,v011,v007,v003
```

## Workspace 结构

```text
workspaces/<competition>/
|-- config.yaml
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- features/
|-- models/
|   `-- v001/
|       |-- model.pkl
|       |-- cv_scores.json
|       |-- oof_preds.npy
|       |-- test_preds.npy
|       `-- importance.csv
|-- reports/
|   |-- research_notes.md
|   `-- eda_summary.md
|-- submissions/
|-- journal.json
|-- idea_pool.json
`-- .state/
```

大数据、模型、提交和缓存默认不提交到 git。

## 配置示例

```yaml
competition:
  name: "titanic"
  url: "https://www.kaggle.com/competitions/titanic"
  type: "tabular"
  metric: "accuracy"
  metric_direction: "maximize"

data:
  train: "data/raw/train.csv"
  test: "data/raw/test.csv"
  sample_submission: "data/raw/sample_submission.csv"
  target_column: "Survived"
  id_column: "PassengerId"

model:
  primary: "lightgbm"
  cv_strategy: "stratified_kfold"
  cv_folds: 5
  seed: 42

submission:
  auto_submit: false
  best_threshold: 0.001
  max_daily: 2
```

金融、交易和时间序列类竞赛建议使用 time-based CV：

```yaml
model:
  cv_strategy: "time_series_split"
```

## Agent 架构

```text
User / Agent
    |
    v
kar CLI
    |
    v
Pipeline Runner
    |
    +--> Research Agent  --> reports/research_notes.md
    +--> EDA Agent       --> reports/eda_summary.md
    +--> Feature Agent   --> data/features/v*.parquet
    +--> Train Agent     --> models/v*/
    +--> Iteration Agent --> journal.json, idea_pool.json
    +--> Submit Agent    --> submissions/, .state/submission_budget.json
```

状态机：

```text
RESEARCH -> EDA -> FEATURES -> TRAIN -> EVALUATE -> CANDIDATE_READY
                                ^                       |
                                |                       v
                             ITERATE              USER DECISION
                                                        |
                                                        v
                                                     SUBMIT
```

详细规则见 [AGENTS.md](AGENTS.md)。

## 为什么面向 Agent 设计

Agent 跑竞赛时最容易失败的地方不是“不会写模型”，而是：

- metric 和 CV 搞错；
- config 和真实数据 schema 不一致；
- 大 parquet 直接爆内存；
- 实验产物缺元数据，后续无法比较；
- 公开 notebook 的有效经验没有结构化吸收；
- 提交预算被浪费；
- CLI 输出难以被机器解析。

本项目会优先补齐这些工具链能力，而不是只堆模型脚本。

## Roadmap

近期重点：

- `kar inspect <workspace> --fix-config`：检查并修复 schema/config。
- Experiment registry：统一记录每次实验的 command、params、features、metric、CV、runtime。
- Notebook miner：提取公开 notebook 的 feature list、params、CV 和 ensemble 公式。
- Recipe system：把 `drw-clean` 这类一赛一脚本沉淀为可复用模板。
- OOF ensemble builder：稳定复现 grid/ridge/rank-average ensemble。
- `--json` 输出：让 Agent 能可靠解析 CLI。
- `kar sync-lb`：同步 Kaggle LB 分数和 rank 到本地 history。

完整路线图见 [docs/agent-tooling-roadmap.md](docs/agent-tooling-roadmap.md)。

## 安全原则

- 默认不自动提交 Kaggle。
- 提交到 Kaggle、删除数据/模型、修改 credentials、push remote 都需要明确用户意图。
- 训练、生成特征、生成提交文件、dry-run 验证、更新实验记录可以自动执行。
- Kaggle 数据、模型、提交文件、notebook cache 和本地凭据不应提交到 git。

## 贡献

欢迎围绕 Agent 可靠性和开源工具链补能力，优先方向包括：

- `kar inspect <workspace>` schema/config 检查；
- experiment registry 和 run metadata；
- metric/CV contract；
- public notebook miner；
- reusable recipe system；
- OOF ensemble builder；
- `--json` 输出和稳定 exit codes。

贡献前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。Coding Agent 请先阅读 [AGENTS.md](AGENTS.md)。

## License

MIT，见 [LICENSE](LICENSE)。
