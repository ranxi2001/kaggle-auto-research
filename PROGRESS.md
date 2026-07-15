# PROGRESS.md

这是 Loop Engineer 的跨会话短期记忆，不是日报、roadmap 或实验数据库。每轮开始先读本文件，结束时只保留能让下一轮直接继续的已验证状态。详细证据必须写入对应 workspace 产物。

## Goal

让不同 Coding Agent 能基于文件系统安全、可复现地接续 Kaggle 研究。当前竞赛主线是 ROGII Wellbore：v006 仍是最后一个 test-ready / 已提交候选，v009 是最后一个契约完整的本地 promising 节点；HMM shrink、formation stack 和 progress shaping 已收敛，下一轮必须切换核心方法，未达到新 candidate gate 前不再提交。

## Current Loop

- Updated: `2026-07-15` (`Asia/Shanghai`)
- Baseline commit: `34c4e4a`
- Status: `rogii_v010_exhausted_v009_last_clean_promising`
- Active workspace: `workspaces/rogii-wellbore-geology-prediction/`
- Generic loop readiness: `pre_alpha_do_not_run_unattended`
- Current gate: v009 严格四流 nested OOF 为 `11.633994`，相对 v006 改善 `0.550569 / 4.52%` 且五折全胜，但未达到 `<=10.95` candidate gate。v010 adaptive front20 diagnostic 为 `11.618124`，仅比 v009 改善 `0.015870 / 0.136%`，未达到预声明 `<11.50`，判定 `exhausted`。没有新的 test prediction 或提交。
- This loop: 已安装并审计 NVIDIA Kaggle skill；完成 v007 HMM、v008 fold-safe formation、v009 strict residual simplex、v010 adaptive progress diagnostic 及独立 artifact audits。v007-v010 的小修补方向停止；v010 分数可复核但产物契约不完整，不得作为候选或父节点。用户已授权将源码、测试和小型稳定证据 commit/push；大体积 workspace diagnostics、模型、数据和提交文件保持本地。

## Last Verified State

| Workstream | State | Gate / next decision |
|---|---|---|
| ROGII | v006 public RMSE `10.693` 仍是最后已提交候选；v009 contract-clean nested OOF `11.633994`、5/5 折优于 v006，但未过 `10.95` candidate gate；v010 adaptive diagnostic `11.618124` 已 `exhausted` | 停止 HMM/formation/progress blend；下一实验必须是新的核心方法，且不得复用 v010 为父节点 |
| Skill Lift | `v004` 已由认证 Kaggle UI 确认 Writeup submitted；候选 mean lift `0.5`，primary lift `1.0` | 公开 URL 仍返回 `404`；下一步是验证公开可见性并回写 host score / validation / rank，不重复提交 |
| DRW Crypto | 当前最佳 public LB 仍为 `0.08199`；2026-07-10 同步的两个新提交为 `0.07751` 和 `0.07251` | 暂停盲目几何外推；若重启，先同步 LB 并使用全部失败方向重建校准 |
| Generic iteration loop | CLI、patch planner、journal、idea pool 和 iteration state 已有骨架，但没有 workspace 产生过完整的通用 iteration history | `pre-alpha`；安全、metric/split 和 run artifact 契约修复前，ROGII 等真实竞赛继续使用专用、可审计脚本 |

## Blockers And Ruled-Out Paths

- ROGII `v005` 把插值后的缺失 suffix GR 当成真实观测累计 likelihood；其 `data/features/v002.parquet` 是无效且 footer 不完整的 partial artifact，不得复用。
- ROGII `v002` 与 `v003` 的 well-fold 映射不同；v004 pooled OOF 仍有效，但旧报告中的跨版本 fold dispersion 和“固定某折最难”结论不可比。v006 只与 canonical manifest 下重算的 v004 folds 比较。
- ROGII `v006` public `10.693` 比 v004 改善 `27.17%`，但仍比 2026-07-14 17:25 UTC top-10% 边界 `7.120` 高 `3.573`；不能靠小幅调参或 blend 缩小这一差距。
- v006 是 causal particle filter，不是全程 forward-backward smoother；31.7347% 的 suffix GR 缺失行只由运动先验传播，长缺口和重复 GR motif 仍是主要风险。
- v007 pure HMM 的 moving corridor 没有 parent attraction，重复 GR motif 导致 confident drift；slow/flex 为 `21.596877 / 29.487150`，该 pure smoother 路径已排除。
- v008 fold-safe formation local plane standalone RMSE 接近 40，nested stack 只改善 v006 `0.005891`；formation 只能保留为弱辅助流，不能作为核心。
- v009 证明约 `0.15-0.17` HMM shrink 有稳定互补性，但 `11.633994` 仍离 `10.95` gate 差 `0.683994`；继续调 blend 权重不是大幅改善路径。
- v010 的 front20 形状由 v009 全五折 target residual 启发，只是 adaptive diagnostic。其独立分数审计通过，但 in-run audit 生命周期、feature list 和 run provenance 有四项契约缺口；以 `reports/v010_independent_artifact_audit.json` 为准，禁止将 v010 用作 candidate 或 parent run。
- canonical folds 已被 v007-v010 多轮自适应观察；后续小幅 CV 改善的选择乐观偏差会继续增大，必须改核心假设并预声明一次性 gate。
- ROGII `v003` 使用的 fixed beam 曾经全量 target scan 选择，其 CV 不是完全 nested；新方案必须把配准、特征选择和调参放进外层 well-group split。
- Skill Lift 的 standard `CreateSubmission` 会返回 `400`；该竞赛必须走 Kaggle Writeup + ZIP 流程。
- `workspaces/drw-crypto/reports/next_submit_plan.md` 早于 2026-07-10 的真实提交，不得单独作为当前决策依据；先读 `lb_sync.csv` 和更新的 scan 产物。
- `README.md` 中的 DRW “尚未真实提交”描述已过期；在修正前以 workspace leaderboard 产物为准。
- 不得无人值守运行 `kar improve`：其 baseline 阶段会执行 `runner.run(full=True)`，而 workspace 若包含 `submit` stage 且 `auto_submit: true`，可触发真实提交。
- 通用 CV 尚未实现 competition metric 契约：LightGBM regression 返回 R2，XGBoost regression 返回 RMSE，ensemble 又固定按 minimize 评估；这些分数不能直接用于 DRW Pearson 或其他指标。
- `PipelineRunner._run_variant()` 没有从 config 读取并传入 groups，因此通用 `group_kfold` 不能支持 ROGII；maximize 的 journal/idea 方向也尚未全程对齐。
- 通用 variant 尚不满足产物契约：只保存首折 model、OOF 和简化 `cv_scores.json`，没有 `run.json`、`feature_list.txt`、test predictions 或失败 run 状态；当前也没有 pipeline/loop 单测。

## Next

以 v009（不是 v010）为最后可复用本地证据，先为一个新的 target-free 核心方法写预声明方案，再评分一次。优先实现 public evidence 中的 physical-surface / NCC registration 之一，但参数、特征和停止 gate 必须在读取新分数前固定；不得继续扫描 HMM shrink、formation blend 或 progress shape，未达到 `<=10.95` 且至少四折改善时不生成 test candidate。

## Evidence

- ROGII 当前结论：`workspaces/rogii-wellbore-geology-prediction/reports/leaderboard_summary.md`
- ROGII 实验比较：`workspaces/rogii-wellbore-geology-prediction/reports/experiment_summary.md`
- ROGII v005 失败状态：`workspaces/rogii-wellbore-geology-prediction/models/v005/run.json`
- ROGII v006 核心产物：`workspaces/rogii-wellbore-geology-prediction/models/v006/run.json`、`cv_scores.json` 和 `reports/v006_state_space_summary.md`
- ROGII v006 本地候选：`workspaces/rogii-wellbore-geology-prediction/submissions/sub_v006_particle_state_space.csv` 及同名 JSON
- ROGII v006 notebook：`workspaces/rogii-wellbore-geology-prediction/notebooks/v006/rogii-v006.ipynb` 和 `kernel-metadata.json`
- ROGII v006 远端构建：`workspaces/rogii-wellbore-geology-prediction/reports/v006_remote_kernel.json`
- Kaggle skill 与公开 notebook 审计：`workspaces/rogii-wellbore-geology-prediction/reports/v007_kaggle_skill_audit.md` 和 `v007_public_notebook_evidence.json`
- ROGII v007 结论：`workspaces/rogii-wellbore-geology-prediction/reports/v007_hmm_smoother_summary.md`、`v007_postmortem.md` 和 `v007_artifact_audit.json`
- ROGII v008 结论：`workspaces/rogii-wellbore-geology-prediction/models/v008/run.json` 和 `reports/v008_formation_spatial_summary.md`
- ROGII v009 最后 clean promising 节点：`workspaces/rogii-wellbore-geology-prediction/models/v009/run.json`、`reports/v009_artifact_audit.json` 和 `v009_postmortem.md`
- ROGII v010 exhausted diagnostic：`workspaces/rogii-wellbore-geology-prediction/models/v010/run.json` 和 `reports/v010_independent_artifact_audit.json`；原 `v010_artifact_audit.json` 已标记 superseded
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

- `uv run pytest -q`: `141 passed`
- v009 targeted / requirements targeted: `19 passed / 19 passed`
- v010 targeted / requirements targeted: `17 passed / 17 passed`
- Targeted Ruff and Ruff format check for all changed Python files: passed
- `py_compile` for v007-v010 scripts: passed
- `jq empty` for updated JSON artifacts: passed
- `git diff --check`: passed
- v009 independent recomputation: exact pooled/fold scores, 21 input and 10 output hashes stable, six 15-support fits valid, no test/submission
- v010 independent recomputation: exact pooled/fold scores, six 31-support fits valid, no partial/test/submission; four artifact-contract gaps documented and self-audit superseded
- Installed skill: `/home/ranxi/.codex/skills/nvidia-kaggle-skill`, NVIDIA repository commit `410c70b0b076b0d0ca76f10a855e7e337d9bd09b`
- This loop made no Kaggle API submission, kernel push, credential change, deletion, or remote git push.

```bash
uv run pytest -q
uv run pytest -q tests/test_rogii_residual_simplex.py tests/test_rogii_progress_simplex.py
uv run ruff check workspaces/rogii-wellbore-geology-prediction/scripts/rogii_residual_simplex.py workspaces/rogii-wellbore-geology-prediction/scripts/rogii_progress_simplex.py tests/test_rogii_residual_simplex.py tests/test_rogii_progress_simplex.py
git diff --check
git status --short
```
