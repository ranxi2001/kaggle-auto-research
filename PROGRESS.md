# PROGRESS.md

这是 Loop Engineer 的跨会话短期记忆，不是日报、roadmap 或实验数据库。每轮开始先读本文件，结束时只保留能让下一轮直接继续的已验证状态。详细证据必须写入对应 workspace 产物。

## Goal

让不同 Coding Agent 能基于文件系统安全、可复现地接续 Kaggle 研究。当前竞赛主线是 ROGII Wellbore：以 v006 public RMSE `10.693` 为新基线，继续验证 latent-surface HMM / forward-backward smoother；没有新的大幅 grouped-CV 改善前不再提交。

## Current Loop

- Updated: `2026-07-15` (`Asia/Shanghai`)
- Baseline commit: `e3d9b69`
- Status: `rogii_v006_submitted_public_rmse_10p693`
- Active workspace: `workspaces/rogii-wellbore-geology-prediction/`
- Generic loop readiness: `pre_alpha_do_not_run_unattended`
- Current gate: v006 已完成真实提交；下一次提交必须来自新的核心序列方法，并在 canonical grouped CV 上再次取得大幅改善。
- This loop: 用户明确授权后提交私有 kernel version 1。submission ref `54693365` 状态 `COMPLETE`，public RMSE `10.693`，rank `2940 / 4909`；本地预算和全部实验产物已同步。

## Last Verified State

| Workstream | State | Gate / next decision |
|---|---|---|
| ROGII | v006 grouped OOF RMSE `12.184563`，public RMSE `10.693`，rank `2940 / 4909`；相比 v004 public `14.683` 改善 `3.990 / 27.17%`，但仍高于 top-10% 边界 `7.120` | latent-surface 方向已验证；下一核心实验做 HMM / forward-backward smoother + windowed GR registration，不做小幅 blend |
| Skill Lift | `v004` 已由认证 Kaggle UI 确认 Writeup submitted；候选 mean lift `0.5`，primary lift `1.0` | 公开 URL 仍返回 `404`；下一步是验证公开可见性并回写 host score / validation / rank，不重复提交 |
| DRW Crypto | 当前最佳 public LB 仍为 `0.08199`；2026-07-10 同步的两个新提交为 `0.07751` 和 `0.07251` | 暂停盲目几何外推；若重启，先同步 LB 并使用全部失败方向重建校准 |
| Generic iteration loop | CLI、patch planner、journal、idea pool 和 iteration state 已有骨架，但没有 workspace 产生过完整的通用 iteration history | `pre-alpha`；安全、metric/split 和 run artifact 契约修复前，ROGII 等真实竞赛继续使用专用、可审计脚本 |

## Blockers And Ruled-Out Paths

- ROGII `v005` 把插值后的缺失 suffix GR 当成真实观测累计 likelihood；其 `data/features/v002.parquet` 是无效且 footer 不完整的 partial artifact，不得复用。
- ROGII `v002` 与 `v003` 的 well-fold 映射不同；v004 pooled OOF 仍有效，但旧报告中的跨版本 fold dispersion 和“固定某折最难”结论不可比。v006 只与 canonical manifest 下重算的 v004 folds 比较。
- ROGII `v006` public `10.693` 比 v004 改善 `27.17%`，但仍比 2026-07-14 17:25 UTC top-10% 边界 `7.120` 高 `3.573`；不能靠小幅调参或 blend 缩小这一差距。
- v006 是 causal particle filter，不是全程 forward-backward smoother；31.7347% 的 suffix GR 缺失行只由运动先验传播，长缺口和重复 GR motif 仍是主要风险。
- ROGII `v003` 使用的 fixed beam 曾经全量 target scan 选择，其 CV 不是完全 nested；新方案必须把配准、特征选择和调参放进外层 well-group split。
- Skill Lift 的 standard `CreateSubmission` 会返回 `400`；该竞赛必须走 Kaggle Writeup + ZIP 流程。
- `workspaces/drw-crypto/reports/next_submit_plan.md` 早于 2026-07-10 的真实提交，不得单独作为当前决策依据；先读 `lb_sync.csv` 和更新的 scan 产物。
- `README.md` 中的 DRW “尚未真实提交”描述已过期；在修正前以 workspace leaderboard 产物为准。
- 不得无人值守运行 `kar improve`：其 baseline 阶段会执行 `runner.run(full=True)`，而 workspace 若包含 `submit` stage 且 `auto_submit: true`，可触发真实提交。
- 通用 CV 尚未实现 competition metric 契约：LightGBM regression 返回 R2，XGBoost regression 返回 RMSE，ensemble 又固定按 minimize 评估；这些分数不能直接用于 DRW Pearson 或其他指标。
- `PipelineRunner._run_variant()` 没有从 config 读取并传入 groups，因此通用 `group_kfold` 不能支持 ROGII；maximize 的 journal/idea 方向也尚未全程对齐。
- 通用 variant 尚不满足产物契约：只保存首折 model、OOF 和简化 `cv_scores.json`，没有 `run.json`、`feature_list.txt`、test predictions 或失败 run 状态；当前也没有 pipeline/loop 单测。

## Next

以 v006 为父节点实现一个新的 fully nested latent-surface HMM / forward-backward smoother：状态继续使用 `U = TVT + Z` 与 dip rate，emission 改为窗口 GR registration，并在 canonical outer folds 内选择平滑/先验参数。预先声明新的候选 gate，保存完整 run/OOF/diagnostics；未达到大幅改善时不生成或提交新候选。

如果当前任务转为仓库工具链，优先级是：先用测试硬隔离 `improve` 和 submit，再实现统一 MetricSpec/direction 与 groups 传递，然后补齐不可变 run artifacts 和 idea/journal 反馈闭环。

## Evidence

- ROGII 当前结论：`workspaces/rogii-wellbore-geology-prediction/reports/leaderboard_summary.md`
- ROGII 实验比较：`workspaces/rogii-wellbore-geology-prediction/reports/experiment_summary.md`
- ROGII v005 失败状态：`workspaces/rogii-wellbore-geology-prediction/models/v005/run.json`
- ROGII v006 核心产物：`workspaces/rogii-wellbore-geology-prediction/models/v006/run.json`、`cv_scores.json` 和 `reports/v006_state_space_summary.md`
- ROGII v006 本地候选：`workspaces/rogii-wellbore-geology-prediction/submissions/sub_v006_particle_state_space.csv` 及同名 JSON
- ROGII v006 notebook：`workspaces/rogii-wellbore-geology-prediction/notebooks/v006/rogii-v006.ipynb` 和 `kernel-metadata.json`
- ROGII v006 远端构建：`workspaces/rogii-wellbore-geology-prediction/reports/v006_remote_kernel.json`
- ROGII 搜索状态：`workspaces/rogii-wellbore-geology-prediction/journal.json` 和 `idea_pool.json`
- Skill Lift 状态：`workspaces/skill-lift/reports/research_notes.md` 和 `workspaces/skill-lift/reports/v004_writeup_submission.json`
- DRW LB 真实记录：`workspaces/drw-crypto/reports/lb_sync.csv`
- 跨竞赛选择和用户决策：`docs/competition-watchlist.md`
- 通用循环入口：`cli/main.py`、`src/kaggle_auto/pipeline/runner.py` 和 `.claude/skills/iteration-loop/SKILL.md`

## Worktree Notes

创建本文件前，工作树已有下列未跟踪的用户产物，不得删除、覆盖或纳入无关修改：

- `workspaces/ai-agent-security-multi-step-tool-attacks/`
- `workspaces/autonomous-agent-prediction-beta/`
- `workspaces/skill-lift/reports/assets/v004-thumbnail.png`

## Stop Conditions

- 实际 Kaggle 提交、切换 active competition/workspace、删除数据或模型、修改 credentials、push remote 前必须获得用户明确确认。
- 数据 schema、metric direction 或 CV splitter 未与真实竞赛契约对齐时，停止训练，先修正 `config.yaml` 和验证产物。
- 迭代内永远不调用 submit；候选未达到预先声明的本地 gate 时，只记录结果并切换策略。

## Verification

- `uv run pytest -q`: `35 passed`
- ROGII targeted tests: `19 passed`
- Targeted Ruff and Ruff format check for all changed Python files: passed
- `jq empty` for updated JSON/notebook artifacts: passed
- `git diff --check`: passed
- Pre-submission `kar submit ... --dry-run`: `Valid: Yes`
- Kaggle private kernel v1: `COMPLETE`; remote visible output matches local candidate
- Kaggle competition submission ref `54693365`: `COMPLETE`, public RMSE `10.693`, rank `2940 / 4909`
- 2026-07-15 local submission budget: `1 / 2` used; no retry or second submission
- `uv run ruff check .`: baseline still fails with the same `69` pre-existing findings; new v006 files add no full-repo Ruff findings.

```bash
uv run pytest -q
uv run ruff check .
uv run ruff check workspaces/rogii-wellbore-geology-prediction/scripts/build_kaggle_notebook.py workspaces/rogii-wellbore-geology-prediction/scripts/rogii_state_space.py tests/test_rogii_notebook.py tests/test_rogii_baseline.py
git diff --check
git status --short
```
