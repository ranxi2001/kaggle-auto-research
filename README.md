# Kaggle Auto Research

<p align="center">
  <strong>AI Agent 驱动的 Kaggle 竞赛自动研究框架</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-experimental-orange">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

<p align="center">
  简体中文 | <a href="./README-EN.md">English</a>
</p>

Kaggle Auto Research 是一个面向 Kaggle 竞赛的 Agent 化自动研究框架。它把竞赛调研、数据下载、EDA、特征生成、模型训练、迭代优化、集成和提交预算管理拆成可复用的 CLI 与工作区产物，让每次实验都能被追踪、复现和继续优化。

> 当前项目仍处于实验阶段，适合研究、原型验证和内部竞赛工作流自动化。实际 Kaggle 提交默认不会自动执行。

## 核心特性

- **工作区优先**：所有竞赛产物都保存在 `workspaces/<competition>/`，避免不同竞赛互相污染。
- **配置即事实源**：每个工作区的 `config.yaml` 管理数据路径、目标列、指标、CV、模型和提交预算。
- **Agent 流水线**：Research、EDA、Feature、Train、Iteration、Submit 阶段通过文件系统传递产物。
- **产物版本化**：特征、模型、OOF、测试集预测和提交文件按版本保存，不覆盖历史结果。
- **提交保护**：默认只生成和 dry-run 验证提交文件；真实提交受预算和用户确认约束。
- **实验树迭代**：内置 `journal.json` 和 `idea_pool.json`，记录实验树、候选方案和下一步优化思路。

## 能做什么

| 阶段 | 命令 | 产物 |
| --- | --- | --- |
| 认证 | `kar auth` | Kaggle 凭证状态 |
| 初始化 | `kar init <name>` | 新竞赛工作区 |
| 数据 | `kar data <name>` | 下载并解压后的原始数据 |
| 调研 | `kar research <name>` | `reports/research_notes.md` |
| EDA + 特征 | `kar eda <name>` | `reports/eda_summary.md`, `data/features/v*.parquet` |
| 训练 | `kar train <name>` | `models/v*/model.pkl`, CV 分数、OOF/测试集预测 |
| 迭代 | `kar pipeline <name> --iterate 5` | 更新实验树和 idea pool |
| 集成 | `kar ensemble <name>` | 基于本地最优模型的集成结果 |
| 提交 | `kar submit <name> --dry-run -f ...` | 验证提交文件，可进入候选队列 |

## 安装

```bash
git clone https://github.com/<your-org>/kaggle-auto-research.git
cd kaggle-auto-research

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS / Linux

pip install -e .
```

开发环境：

```bash
pip install -e ".[dev]"
```

可选深度学习依赖：

```bash
pip install -e ".[deep]"
```

## Kaggle 认证

首次下载竞赛数据前先认证：

```bash
kar auth
```

也可以使用 Kaggle API 标准凭证文件：

```text
~/.kaggle/kaggle.json
```

Windows 常见路径：

```text
C:\Users\<you>\.kaggle\kaggle.json
```

## 快速开始

```bash
# 1. 创建工作区
kar init titanic --type tabular --url https://www.kaggle.com/competitions/titanic

# 2. 下载并解压数据
kar data titanic

# 3. 分阶段执行
kar research titanic
kar eda titanic
kar train titanic

# 4. 查看流水线状态
kar status titanic

# 5. 只验证提交文件，不提交到 Kaggle
kar submit titanic --dry-run -f submissions/sub_001.csv
```

从头运行完整流水线：

```bash
kar pipeline titanic --full
```

运行多轮优化：

```bash
kar pipeline titanic --iterate 5
kar analyze titanic
kar ensemble titanic
```

## CLI 命令

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
kar submit <name> --status
kar submit <name> --history
kar submit <name> --dry-run -f submissions/sub_001.csv
kar submit <name> --flush
```

当前 DRW Crypto 研究线还有一个专用清洗训练命令：

```bash
kar drw-clean drw-crypto --top-k 350 --n-estimators 700
```

## 工作区结构

每个竞赛都有独立工作区：

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

## 配置示例

`config.yaml` 是每个工作区的单一事实源。

```yaml
competition:
  name: "titanic"
  url: "https://www.kaggle.com/competitions/titanic"
  type: "tabular"
  metric: "rmse"
  metric_direction: "minimize"

data:
  train: "data/raw/train.csv"
  test: "data/raw/test.csv"
  sample_submission: "data/raw/sample_submission.csv"
  target_column: "target"
  id_column: "id"

model:
  primary: "lightgbm"
  cv_strategy: "stratified_kfold"
  cv_folds: 5
  seed: 42

submission:
  auto_submit: false
  best_threshold: 0.01
  max_daily: 5
```

时间序列、金融和交易类竞赛建议使用 time-based CV：

```yaml
model:
  cv_strategy: "time_series_split"
```

## 架构

```text
用户 / Agent
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

流水线状态机：

```text
RESEARCH -> EDA -> FEATURES -> TRAIN -> EVALUATE -> CANDIDATE_READY
                                ^                       |
                                |                       v
                             ITERATE              USER DECISION
                                                        |
                                                        v
                                                     SUBMIT
```

## 安全边界

Kaggle Auto Research 默认自动执行可逆工作，并保护不可逆操作。

默认可自动执行：

- 创建工作区。
- 下载并解压竞赛数据。
- 生成报告、特征、模型、预测和提交 CSV。
- 更新 `config.yaml`、`.state/`、`journal.json` 和 `idea_pool.json`。
- dry-run 验证提交文件。

需要明确用户决策：

- 提交到 Kaggle。
- 删除模型、数据或提交文件。
- 修改凭证或 `.env`。
- push 到远程 Git 仓库。

## 支持的竞赛类型

| 类型 | 适用场景 | 默认模型族 |
| --- | --- | --- |
| `tabular` | 结构化 CSV/parquet 竞赛 | LightGBM, XGBoost |
| `crypto` | 金融、加密货币、时间序列预测 | LightGBM + time-based CV |
| `llm` | NLP 和生成式 AI 竞赛 | Transformers / prompt workflows |

## 开发

```bash
make install
make dev
make test
make lint
make format
```

等价命令：

```bash
pytest tests/ -v
ruff check src/ cli/ tests/
ruff format src/ cli/ tests/
```

## 项目结构

```text
kaggle-auto-research/
|-- cli/                    # Typer CLI 入口
|-- src/kaggle_auto/        # Python 核心包
|   |-- eda/
|   |-- features/
|   |-- models/
|   |-- pipeline/
|   |-- research/
|   |-- submission/
|   |-- tuning/
|   `-- utils/
|-- templates/              # 工作区配置模板
|-- tests/
|-- workspaces/             # 本地竞赛工作区
|-- agents.md               # Agent 行为规则
|-- CLAUDE.md               # Claude Code 项目说明
`-- pyproject.toml
```

## 路线图

- 更稳健的数据 schema 校验。
- 更完整的 Kaggle 指标注册表。
- 公开 notebook / discussion 调研导出。
- Public leaderboard 追踪和本地/LB 相关性分析。
- 面向全自动 Agent 的安全提交确认流程。
- Vision、NLP、推荐系统等更多竞赛模板。

## 贡献

欢迎提交 PR。适合优先改进的方向：

- 为流水线阶段和提交校验补测试。
- 改进竞赛模板。
- 增加指标和 CV 策略。
- 改进大 parquet 数据集的 EDA 报告。
- 增加模型适配器，同时保持产物格式稳定。

提交 PR 前建议运行：

```bash
make test
make lint
```

## 许可证

MIT License.

