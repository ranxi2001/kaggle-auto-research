# Agent-Oriented Kaggle Tooling Roadmap

This project is becoming more than a collection of Kaggle scripts. The goal is to provide an open-source toolchain that lets coding agents run competitions safely, reproducibly, and with enough telemetry to improve over time.

The DRW Crypto experiment exposed the current gaps clearly.

## What Was Missing

### 1. Experiment Registry

Current state:
- Models are saved under `models/vNNN/`.
- Metadata is inconsistent across model types.
- Submission files can collide when experiments run in parallel.

Needed:
- A first-class experiment registry with immutable run IDs.
- Required metadata for every run: command, git commit, data fingerprint, feature list, params, CV splitter, fold scores, OOF path, test prediction path, runtime, status, and parent run.
- Atomic ID allocation so parallel runs cannot generate colliding submissions.

Agent value:
- Agents can compare experiments without scraping folders.
- Agents can resume, rank, ensemble, and explain runs reliably.

### 2. Metric and CV Contract

Current state:
- The framework previously defaulted regression scoring to RMSE even when the competition metric was R2/Pearson.
- CV strategy is configured, but metric semantics are not enforced everywhere.

Needed:
- A `MetricSpec` object loaded from `config.yaml`.
- Separate optimization metric, display metric, and Kaggle metric aliases.
- Validation that every trainer writes scores using the configured metric and direction.
- Built-in split contracts: time series, grouped, stratified, public-LB-like holdout, and custom split files.

Agent value:
- Prevents agents from optimizing the wrong objective.
- Makes score comparisons trustworthy.

### 3. Data Schema Inspector

Current state:
- DRW config said `target`, but real data used `label`.
- Test data had a placeholder `label` column, which could be mistaken for leakage.
- Sample submission used `ID,prediction`.

Needed:
- `kar inspect <workspace>` to infer:
  - train/test/sample columns
  - target candidates
  - ID and prediction columns
  - placeholder columns
  - row counts and parquet metadata
  - mismatch warnings against `config.yaml`
- Optional `--fix-config` mode to patch obvious schema errors.

Agent value:
- Agents can verify assumptions before training.
- Reduces silent bad runs.

### 4. Public Notebook Importer

Current state:
- Public notebooks can be pulled manually.
- They are not parsed into reusable ideas.
- CLI output can break on emoji/GBK terminals.

Needed:
- `kar notebooks pull <workspace> --top 10`.
- Notebook-to-script extraction.
- Pattern mining for:
  - feature lists
  - model params
  - CV strategy
  - ensemble formulas
  - post-processing
- UTF-8-safe output and local cache ignored by git.

Agent value:
- Agents can ground iterations in public baselines.
- Public ideas become structured candidates, not copy-paste blobs.

### 5. Run Templates for Common Competition Types

Current state:
- Generic pipeline is useful but not strong enough for DRW-style low-signal crypto.
- A one-off `drw-clean` command improved results faster than the generic pipeline.

Needed:
- Versioned run templates:
  - `tabular/lgbm_clean`
  - `tabular/xgb_lgbm_blend`
  - `crypto/time_decay_slices`
  - `crypto/low_signal_feature_select`
  - `llm/prompt_ablation`
- Templates should be parameterized and registered, not hardcoded into the CLI.

Agent value:
- Agents can choose proven workflows by competition type.
- One-off discoveries can graduate into reusable templates.

### 6. Ensemble Builder With OOF Contracts

Current state:
- Manual OOF ensembling improved DRW from `0.003998` to `0.005707`.
- The built-in ensemble path is not yet aligned with maximize/minimize and artifact metadata.

Needed:
- `kar ensemble <workspace> --models v011,v010,v007 --metric pearson`.
- OOF length and split compatibility checks.
- Grid, linear, ridge, rank-average, and simple blend modes.
- Save ensemble metadata and generated submission.

Agent value:
- Agents can combine strong candidates safely.
- Ensemble gains become reproducible.

### 7. Submission and Leaderboard Feedback Loop

Current state:
- Submission budget exists.
- We can query submissions and public leaderboard.
- LB scores are not automatically reconciled into local history.

Needed:
- `kar leaderboard <workspace>` for own submissions and public top rows.
- `kar sync-lb <workspace>` to update local submission history with Kaggle scores and ranks.
- CV-vs-LB gap analysis.
- Warnings when a submission candidate is below threshold or too similar to existing submissions.

Agent value:
- Agents can learn from public/private feedback without wasting quota.
- Keeps submit decisions auditable.

### 8. Robust CLI UX for Agent and Human Use

Current state:
- Root `kar.cmd` hides `.venv`.
- Some commands still produce terminal encoding issues.
- Long-running commands lack structured progress output.

Needed:
- Cross-platform launchers: `kar`, `kar.cmd`, possibly `uvx` support.
- JSON output mode for agents: `--json`.
- Human output mode for terminals.
- Stable exit codes.
- Retry wrappers for flaky Kaggle network calls.

Agent value:
- Agents can parse outputs reliably.
- Humans get short commands.

### 9. Resource-Aware Execution

Current state:
- Full EDA on multi-GB parquet is risky.
- Sampling EDA fixed the first failure.
- Training memory/runtime is not tracked.

Needed:
- Automatic parquet metadata reads.
- Sampling defaults for EDA.
- Memory estimates before loading.
- Runtime and peak memory logging per experiment.
- Optional remote/Kaggle notebook execution backend for heavy GPU/CPU jobs.

Agent value:
- Agents avoid crashing the local machine.
- Heavy experiments can be routed to appropriate compute.

## Proposed CLI Surface

```bash
kar auth
kar data <workspace>
kar inspect <workspace> --fix-config
kar notebooks pull <workspace> --top 10
kar notebooks mine <workspace>
kar run <workspace> --template crypto/low_signal_feature_select --set top_k=130
kar experiments <workspace>
kar compare <workspace>
kar ensemble <workspace> --models v011,v010,v007 --mode grid
kar submit <workspace> --file submissions/sub_xxx.csv --dry-run
kar leaderboard <workspace>
kar sync-lb <workspace>
```

## Agent-Facing Artifact Contracts

Each experiment should write:

```text
models/vNNN/
  run.json
  cv_scores.json
  oof_preds.npy
  test_preds.npy
  feature_list.txt
  importance.csv
  model.pkl or models.pkl
```

Each submission should write:

```text
submissions/sub_NNN_<run_id>.csv
submissions/sub_NNN_<run_id>.json
```

The JSON metadata should include:

- source model versions
- weights for ensembles
- CV score and fold scores
- command used
- generated timestamp
- validation result
- Kaggle submission ID, LB score, and rank when available

## Immediate Next Steps

1. Extract `drw-clean` into a reusable template under `src/kaggle_auto/templates/` or `src/kaggle_auto/recipes/`.
2. Add `kar inspect` and make pipeline refuse to train when config schema does not match data.
3. Add an experiment registry and atomic run/submission ID allocator.
4. Upgrade `kar ensemble` to reproduce the DRW OOF grid blend.
5. Add notebook mining for public feature lists and params.
6. Add JSON output mode for key commands.

## DRW Lessons So Far

- Baseline auto lag/rolling features were worse than a simple cleaned feature selection.
- Public notebook feature lists are much stronger priors than generic generated features.
- Strong regularization and small feature sets help low-signal financial data.
- OOF ensembling gave an immediate lift from `0.003998` to `0.005707`.
- A good toolchain should make those discoveries repeatable, not trapped in one-off scripts.
