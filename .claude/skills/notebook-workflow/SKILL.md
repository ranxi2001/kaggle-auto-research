---
name: notebook-workflow
description: >
  在线 Kaggle Notebook 开发工作流。本地编写代码 → push 到 Kaggle 运行 → 拉取结果。
  避免大数据集下载，利用 Kaggle 免费 GPU/TPU。
  触发关键词：notebook、在线跑、push kernel、远程训练、kaggle run、GPU训练。
---

# Notebook Workflow（在线开发技能）

**核心理念**：不下载数据到本地，代码在 Kaggle 上跑，结果拉回本地分析。

```
本地写代码 → push → Kaggle GPU 执行 → poll 状态 → 拉取 output
     ↑                                                    │
     └──── 分析结果，修改代码 ←────────────────────────────┘
```

---

## 为什么用在线方案

| 痛点 | 本地方案 | Notebook 方案 |
|------|----------|---------------|
| 大数据集 (>1GB) | 下载慢/断线 | 已挂载，零延迟 |
| GPU 训练 | 需要本地显卡 | 免费 T4/P100/TPU |
| 提交 | API token 认证复杂 | notebook output 直接提交 |
| 环境 | 本地依赖冲突 | Kaggle 预装全套 ML 库 |

---

## 工作区结构

每个竞赛在 `workspaces/<name>/notebooks/` 下管理 kernel 代码：

```
workspaces/drw-crypto/notebooks/
├── kernel-metadata.json       # Kaggle kernel 配置
├── baseline.py                # 主脚本（或 .ipynb）
├── utils.py                   # 辅助模块（会被一起 push）
└── outputs/                   # 拉取的运行结果
    ├── submission.csv
    └── logs.txt
```

---

## 操作流程

### 1. 初始化 Kernel

```bash
KAGGLE=/Library/Frameworks/Python.framework/Versions/3.14/bin/kaggle
COMP=drw-crypto-market-prediction
USER=ranxi169  # Kaggle username

# 创建 kernel-metadata.json
cd workspaces/<name>/notebooks/
$KAGGLE kernels init -p .
```

手动编辑 `kernel-metadata.json`：

```json
{
  "id": "<username>/<kernel-slug>",
  "title": "DRW Crypto Baseline",
  "code_file": "baseline.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "competition_sources": ["drw-crypto-market-prediction"],
  "dataset_sources": [],
  "kernel_sources": []
}
```

**关键字段**：
- `competition_sources`: 挂载竞赛数据（出现在 `/kaggle/input/<comp-slug>/`）
- `enable_gpu`: true 使用 GPU（每周 30h 额度）
- `enable_internet`: 竞赛提交通常需要 false
- `is_private`: 始终 true，避免泄露方案

### 2. 编写代码（本地）

代码中数据路径使用 Kaggle 标准路径：

```python
import pandas as pd
from pathlib import Path

# Kaggle notebook 中数据路径
INPUT = Path('/kaggle/input/drw-crypto-market-prediction')
OUTPUT = Path('/kaggle/working')

train = pd.read_parquet(INPUT / 'train.parquet')
test = pd.read_parquet(INPUT / 'test.parquet')

# ... 训练逻辑 ...

submission.to_csv(OUTPUT / 'submission.csv', index=False)
```

### 3. Push 并运行

```bash
$KAGGLE kernels push -p workspaces/<name>/notebooks/
```

Push 会自动触发远程执行。返回 kernel URL。

### 4. 监控执行状态

```bash
# 检查状态（running / complete / error）
$KAGGLE kernels status <username>/<kernel-slug>

# 查看实时日志
$KAGGLE kernels logs <username>/<kernel-slug> --lines 50
```

状态轮询策略：
- 前 2 分钟：每 30 秒查一次（排队中）
- 之后：每 2 分钟查一次
- 超过 30 分钟未完成：可能卡住，检查日志

### 5. 拉取结果

```bash
# 拉取 output 文件
$KAGGLE kernels output <username>/<kernel-slug> -p workspaces/<name>/notebooks/outputs/

# 查看生成的文件
ls workspaces/<name>/notebooks/outputs/
```

### 6. 提交（如果 notebook 产出 submission.csv）

两种提交方式：

**方式 A：Notebook 直接提交**（推荐）
- 在 kernel-metadata.json 中设置竞赛关联
- Kaggle 会自动将 notebook output 作为 submission
- 需要 `enable_internet: false`

**方式 B：手动提交 output**
```bash
$KAGGLE competitions submit $COMP \
  -f workspaces/<name>/notebooks/outputs/submission.csv \
  -m "baseline v1 - lgbm"
```

---

## 迭代开发模式

```
Round 1: baseline.py (简单 LightGBM)
  → push → 等结果 → R²=0.012

Round 2: 修改 baseline.py (加特征选择)
  → push → 等结果 → R²=0.015

Round 3: 修改 baseline.py (ensemble)
  → push → 等结果 → R²=0.018
```

**版本管理**：
- 每次 push 前在本地 git commit（保留历史）
- Kaggle 自动保留 kernel 版本历史
- 重要版本的 output 保存到 `outputs/v{N}/`

---

## 多 Kernel 策略

复杂竞赛可用多个 kernel 分工：

```
notebooks/
├── eda/
│   ├── kernel-metadata.json
│   └── eda.py                  # 探索性分析（不提交）
├── train/
│   ├── kernel-metadata.json
│   └── train.py                # 训练 + 保存模型到 output
├── inference/
│   ├── kernel-metadata.json
│   └── inference.py            # 加载模型 + 生成 submission
```

Kernel 间数据传递：
- train kernel 输出模型文件 → 发布为 Kaggle Dataset
- inference kernel 引用该 dataset（`dataset_sources`）

---

## GPU/TPU 额度管理

| 资源 | 每周额度 | 典型用时 |
|------|----------|----------|
| GPU T4 | 30 小时 | LightGBM 训练 ~5-15 min |
| GPU P100 | 30 小时 | 深度学习 ~30-60 min |
| TPU v3 | 20 小时 | 大模型微调 |

**省额度技巧**：
- 先不开 GPU 跑通逻辑（CPU 模式调试）
- 减少数据量测试（`train.head(10000)`）
- 开 GPU 只跑最终版本

---

## 常用命令速查

| 操作 | 命令 |
|------|------|
| 初始化 | `kaggle kernels init -p <dir>` |
| 推送执行 | `kaggle kernels push -p <dir>` |
| 查看状态 | `kaggle kernels status <user>/<slug>` |
| 查看日志 | `kaggle kernels logs <user>/<slug>` |
| 拉取输出 | `kaggle kernels output <user>/<slug> -p <dir>` |
| 列出我的 kernels | `kaggle kernels list --mine` |
| 拉取别人代码 | `kaggle kernels pull <user>/<slug> -p <dir> -m` |

---

## Hard Rules

1. **永远 `is_private: true`** — 泄露方案 = 被抄 = 白干
2. **代码提交前本地 git commit** — 保留完整迭代历史
3. **不在 notebook 里硬编码 API token** — 用 Kaggle Secrets 或环境变量
4. **遵守提交预算** — notebook output 提交也算在 max_daily 内
5. **先 CPU 调试，后 GPU 运行** — 别浪费额度在 debug 上
6. **结果必须拉回本地存档** — Kaggle kernel 可能被清理

---

## 与其他 Skills 的协作

| 阶段 | 本 Skill 职责 | 协作 Skill |
|------|---------------|------------|
| 研究 | 拉取 top kernel 代码分析 | competition-research |
| EDA | push EDA notebook，拉取图表 | eda-features |
| 训练 | push 训练脚本，监控执行 | model-train |
| 提交 | 管理 kernel output 提交 | submit-monitor |
| 迭代 | 版本管理，对比各 version | iteration-loop |

---

## Lesson Log

| 日期 | 教训 |
|------|------|
| 2026-06-07 | 6GB 数据本地下载不可靠（断线、认证），直接用 notebook 更高效 |
