# Competition Watchlist

候选竞赛登记表。这里只记录研究意向，不代表已经接受竞赛规则、加入竞赛、创建 workspace、下载数据或提交结果。

- 登记日期：2026-07-10
- 信息快照：Kaggle 官方 API 和竞赛页面，2026-07-10
- 当前状态：`skill-lift` 已启动本地研究；其余候选暂停或保持 `shortlisted`
- 启动约束：选择并切换具体竞赛前需要用户确认；启动后所有产物必须写入 `workspaces/<competition>/`
- 截止时间：均为 UTC；临近启动时必须重新查询截止时间、报名状态、规则和数据可用性

## Priority 1: MSCapital - Real Financial Market Forecasting

| Field | Value |
|---|---|
| Slug | `ms-capital-real-financial-market-forecasting` |
| URL | https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting |
| Track | Quant / market microstructure |
| Category | Community |
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
