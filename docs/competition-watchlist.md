# Competition Watchlist

竞赛研究与执行状态登记表。只有明确标为 submitted 的项目才代表已经产生真实提交。

- 登记日期：2026-07-10
- 信息快照：Kaggle 官方 API 和竞赛页面，初始于 2026-07-10，最新核验于 2026-07-14
- 当前状态：`skill-lift` 已提交；ROGII v006 已完成真实提交，public RMSE 10.693，rank 2940 / 4909，仍不在 top 10%
- 启动约束：选择并切换具体竞赛前需要用户确认；启动后所有产物必须写入 `workspaces/<competition>/`
- 截止时间：均为 UTC；临近启动时必须重新查询截止时间、报名状态、规则和数据可用性

## Recommended For Medals: ROGII - Wellbore Geology Prediction

| Field | Value |
|---|---|
| Slug | `rogii-wellbore-geology-prediction` |
| URL | https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction |
| Track | Grouped regression over horizontal-well trajectories and reference logs |
| Category | Featured |
| Status | `submitted_improved_below_top10` - v006 code submission complete, public RMSE 10.693, rank 2940 / 4909 |
| Entry deadline | 2026-07-29 23:59 UTC |
| Final deadline | 2026-08-05 23:59 UTC |
| Reward | USD 50,000 |
| Awards points | Yes; official API reports `awards_points=true` |
| Teams at 2026-07-14 17:25 UTC snapshot | 4,909 |
| Metric | RMSE, minimize |
| Submission | Kaggle Notebook only; offline runtime at most 9 hours; output `submission.csv` with `id,tvt` |
| Data | About 1.23 GiB across 2,327 files: 773 train wells with CSV/log/image triplets and a hidden test of about 200 wells |
| Why recommended | The only currently open standard prediction task found with both an explicit local metric and competition points/medals enabled |
| Main risks | Crowded leaderboard, short runway, domain-specific well correlation, hidden code test, and potential leakage around partially observed `TVT_input` |
| Local result | v006 pooled well-GroupKFold RMSE 12.184563, canonical 5 / 5 folds better than v004 |
| Real submission | Ref 54693365, latent-surface kernel v1, public RMSE 10.693, rank 2940 / 4909; v004 was 14.683 |
| Next gate | Keep latent `U = TVT + Z`; require a fully nested HMM/smoother with windowed GR registration before another submission |

## Practice Only: Predicting Student Health Risk

| Field | Value |
|---|---|
| Slug | `playground-series-s6e7` |
| URL | https://www.kaggle.com/competitions/playground-series-s6e7 |
| Track | Standard tabular multiclass classification |
| Category | Playground |
| Status | `shortlisted_no_competition_medals` - useful for pipeline development, not for a medal goal |
| Deadline | 2026-07-31 23:59 UTC |
| Reward | Swag |
| Awards points | No; official API reports `awards_points=false` |
| Teams at 2026-07-13 snapshot | 1,605 |
| Metric | Balanced Accuracy, maximize |
| Submission | Standard CSV with `id,health_condition` |
| Data | `train.csv` 62.7 MB, `test.csv` 24.6 MB, `sample_submission.csv` 4.4 MB |
| Why shortlisted | Explicit local metric, manageable data size, standard file submission, and fast iteration cycle |
| Repository gaps | Add balanced accuracy, multiclass LightGBM/XGBoost predictions, label-aware OOF/test artifacts, and multiclass submission generation before training |
| Start gate | User confirms the competition switch; then join/accept rules, initialize the workspace, download data, validate the real schema, and implement the multiclass path |

## Priority 1: MSCapital - Real Financial Market Forecasting

| Field | Value |
|---|---|
| Slug | `ms-capital-real-financial-market-forecasting` |
| URL | https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting |
| Track | Quant / market microstructure |
| Category | Community |
| Awards points | No |
| Deadline | 2026-10-09 16:00 UTC |
| Reward | Kudos; top-10 offline event expense reimbursement is described on the competition page |
| Teams at snapshot | 28 |
| Metric | Cosine similarity between prediction and target |
| Data | About 9.95 GB of Feather files covering market, order, transaction, label, and test data |
| Why shortlisted | Closest match to the existing DRW workflow: large financial data, low-signal prediction, correlation-style objective, time-based validation, and ensemble iteration |
| Main risks | Community competition; large local storage and memory footprint; schema, timestamp ordering, ID mapping, and leakage boundaries must be verified before feature generation |
| Start gate | Confirm this as the active competition, then run `kar init`, read-only research, file metadata inspection, and sampled EDA before any full-data load |

## Priority 2: Autonomous Agent Prediction (Beta)

| Field | Value |
|---|---|
| Slug | `autonomous-agent-prediction-beta` |
| URL | https://www.kaggle.com/competitions/autonomous-agent-prediction-beta |
| Track | Autonomous ML agent |
| Category | Playground |
| Awards points | No |
| Status | `paused_rules_not_accepted` - workspace initialized, Kaggle data download returned 403 |
| Deadline | 2026-08-06 23:59 UTC |
| Reward | Swag |
| Teams at snapshot | 66 |
| Evaluation | Agent predictions are scored by ROC AUC in Kaggle CPU environments |
| Submission | ZIP containing `agent.yaml` plus optional prompts, tools, skills, scripts, and resources |
| Why shortlisted | Direct product-level evaluation of this repository's core goal: an agent must inspect unseen data, train models, use public feedback, and select final predictions autonomously |
| Main risks | Experimental competition format; Google ADK agent configuration and sandbox restrictions do not directly match the current `kar` runtime |
| Start gate | Audit the sample agent and sandbox contract, then map existing research, EDA, train, and iteration capabilities into a minimal submission package |

## Priority 3: XForecast Challenge @ KDD 2026

| Field | Value |
|---|---|
| Slug | `xforecast-challenge-kdd` |
| URL | https://www.kaggle.com/competitions/xforecast-challenge-kdd |
| Track | Quant + LLM / multimodal financial forecasting |
| Category | Community |
| Awards points | No |
| Deadline | 2026-07-27 12:00 UTC |
| Reward | Kudos |
| Teams at snapshot | 4 |
| Metric | Weighted hit rate for four-week stock direction; private weights are undisclosed |
| Data | OHLCV for 100 stocks plus news, SEC-derived context, and precomputed text embeddings |
| Why shortlisted | Strongest intersection of financial time series and LLM-derived textual signals in the current list |
| Main risks | Dataset access is delivered outside Kaggle after a Google Form; the public leaderboard overlaps training data and is explicitly unsuitable for model selection; little time remains |
| Start gate | Confirm that dataset access can be obtained and stored reproducibly, then design a strict chronological validation split that ignores the leakable public leaderboard |

## Priority 4: Agent Skills and Security

This priority contains two related competitions. They remain separate candidates and must not share a workspace.

### Skill Lift

| Field | Value |
|---|---|
| Slug | `skill-lift` |
| URL | https://www.kaggle.com/competitions/skill-lift |
| Track | Agent skills / meta-skills |
| Category | Community |
| Awards points | No |
| Status | `submitted_public_pending` - authenticated Kaggle UI confirmed the v004 Writeup submission; anonymous visibility is not yet verified |
| Deadline | 2026-08-13 06:55 UTC |
| Reward | USD 20,000 total; two USD 10,000 tracks |
| Teams at snapshot | 105 |
| Submission | Skill folders containing `SKILL.md` with optional scripts and references |
| Why shortlisted | Measures whether reusable skills improve held-out agent tasks, closely matching the repository's skill-oriented agent architecture |
| Main risks | External BenchFlow/SkillsBench harness; hidden capability and safety tasks; broad skills may regress performance or trigger safety penalties |
| Next gate | Verify public visibility and sync any host-side score, validation result, or rank into the local submission history |

### AI Agent Security - Multi-Step Tool Attacks

| Field | Value |
|---|---|
| Slug | `ai-agent-security-multi-step-tool-attacks` |
| URL | https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks |
| Track | Tool-using agent security |
| Category | Featured |
| Awards points | Yes |
| Host | OpenAI |
| Status | `deferred_by_user` - workspace initialized, no competition submission |
| Deadline | 2026-09-01 23:59 UTC |
| New entrant deadline at snapshot | 2026-08-25 23:59 UTC |
| Reward | USD 50,000 |
| Teams at snapshot | 1,712 |
| Submission | Kaggle code submission implementing an `attack.py` search algorithm against an offline sandbox |
| Why shortlisted | High-quality benchmark for multi-step attacks, replayable traces, tool safety, and search algorithms; the results can improve the project's own agent boundaries |
| Main risks | Requires a new security-search and sandbox-evaluation capability rather than the current tabular training pipeline; notebook-only evaluation and strict replayability constraints |
| Start gate | Reproduce the SDK baseline, enumerate attack predicates and budget limits, then version every search run and replay trace as experiment artifacts |

## Decision Log

| Date | Decision |
|---|---|
| 2026-07-10 | User shortlisted four priority directions: MSCapital, Autonomous Agent Prediction, XForecast, and the combined Agent Skills/Security priority. No competition was joined or initialized. |
| 2026-07-10 | Agent Security was deferred by the user. Autonomous Agent data was blocked by unaccepted Kaggle rules, so active local work moved to Skill Lift using the public SkillsBench corpus. |
| 2026-07-11 | Skill Lift v001-v003 used 1,748,452 model tokens across valid and exploratory pairs. Valid mean lift was 0.0, the XLSX public verifier was excluded for an internal inconsistency, and no submission candidate was promoted. |
| 2026-07-11 | V004 introduced a narrow deterministic FJSP repair skill, achieved +1.0 primary lift with 3/3 skill-enabled passes, and passed ZIP dry-run. |
| 2026-07-11 | Competition entry was confirmed. The standard API returned 400 because this Hackathon requires a Kaggle Writeup, so the workspace switched to writeup submission mode. |
| 2026-07-11 | The authenticated Kaggle UI reported the v004 Writeup as submitted. Anonymous access still returned 404, so public visibility remains unverified. |
| 2026-07-13 | `playground-series-s6e7` was initially shortlisted for its clear Balanced Accuracy metric, but the official API reports `awards_points=false`; it remains a practice-only option. |
| 2026-07-13 | Official Kaggle API checks identified Featured competition `rogii-wellbore-geology-prediction` as the best open medal-eligible task with a standard metric: RMSE and `awards_points=true`. No workspace was created, no rules were accepted, and no data was downloaded pending user confirmation. |
| 2026-07-13 | The user accepted ROGII rules and confirmed the switch. Full-data well GroupKFold produced v004 RMSE 14.743751 versus 15.909853 for v001; a private offline notebook package passed local contract checks. No notebook was pushed and no real submission was made. |
| 2026-07-14 | User approved upload and submission. Private kernel v1 completed and submission ref 54653094 scored public RMSE 14.683, rank 3628 / 4829; the result validates the pipeline but is outside medal range. |
| 2026-07-15 | User approved v006 kernel build and submission. Ref 54693365 scored public RMSE 10.693, rank 2940 / 4909, a 27.17% improvement over v004; the next branch is a latent-surface HMM/smoother because top-10% is still 7.120. |
