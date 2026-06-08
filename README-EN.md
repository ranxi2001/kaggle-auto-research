# Kaggle Auto Research

Agent-oriented automation framework for Kaggle competitions.

Kaggle Auto Research turns competition research, data download, EDA, feature generation, model training, iterative improvement, ensembling, submission validation, and leaderboard feedback into reusable CLI stages and versioned workspace artifacts. The goal is not to write one-off competition scripts, but to give humans and coding agents a safe, reproducible way to keep improving results.

> This project is experimental. It is designed for research, prototyping, and internal competition workflows. Real Kaggle submissions are not executed automatically by default.

## Highlights

- **Agent-first workflow**: Research, EDA, Feature, Train, Iteration, and Submit stages communicate through filesystem artifacts.
- **Workspace isolation**: Every competition lives under `workspaces/<competition>/`.
- **Config as source of truth**: `config.yaml` stores data paths, target columns, submission format, metrics, CV strategy, models, and submission budget.
- **Versioned artifacts**: Features, models, OOF predictions, test predictions, and submissions are saved by version.
- **Submission safety**: The default flow generates and dry-run validates submission files; real submissions require explicit control.
- **Public notebook ingestion**: Public notebooks can be pulled locally; future work will mine feature lists, model params, CV strategy, and ensemble formulas.
- **Agent tooling roadmap**: See [docs/agent-tooling-roadmap.md](docs/agent-tooling-roadmap.md).

## Current DRW Crypto Progress

This repository is using DRW Crypto Market Prediction as the first real tooling testbed.

- Initial auto-feature baseline: CV R2 `-0.002859`
- Clean feature selection + LightGBM: best single-model Pearson around `0.0724`
- Pearson OOF grid ensemble: Pearson `0.077989`
- Current best local submission: `sub_ensemble_v010_v011_v007_v003.csv`
- Public leaderboard top is around `0.11 - 0.14`

Conclusion: the toolchain can run end-to-end and improve, but the result is not yet competitive. The next step is to reproduce public-notebook ideas: 25-feature priors, time-decay slices, XGB/LGBM/Ridge blends, and then turn them into reusable recipes.

## Installation

```bash
git clone https://github.com/ranxi2001/kaggel-auto-research.git
cd kaggel-auto-research

python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

On Windows, the repository includes a short launcher:

```bash
.\kar auth
.\kar ls
```

In Git Bash:

```bash
./kar.cmd auth
```

Development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# 1. Authenticate with Kaggle
kar auth

# 2. Create a workspace
kar init titanic --type tabular --url https://www.kaggle.com/competitions/titanic

# 3. Download and extract data
kar data titanic

# 4. Run stages
kar research titanic
kar eda titanic
kar train titanic

# 5. Validate a submission without sending it to Kaggle
kar submit titanic --dry-run -f submissions/sub_001.csv
```

Full pipeline:

```bash
kar pipeline titanic --full
```

Improvement loop:

```bash
kar pipeline titanic --iterate 5
kar analyze titanic
kar ensemble titanic
```

## Common CLI

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

DRW Crypto research helpers:

```bash
kar drw-clean drw-crypto --top-k 130 --n-estimators 1200 --learning-rate 0.015
kar drw-public drw-crypto --model lgbm --folds 3
kar drw-ensemble drw-crypto --models v010,v011,v007,v003
```

## Workspace Layout

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

Large data, models, submissions, and caches are ignored by git.

## Configuration Example

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

For time-series, finance, and trading competitions, prefer time-based CV:

```yaml
model:
  cv_strategy: "time_series_split"
```

## Architecture

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

State machine:

```text
RESEARCH -> EDA -> FEATURES -> TRAIN -> EVALUATE -> CANDIDATE_READY
                                ^                       |
                                |                       v
                             ITERATE              USER DECISION
                                                        |
                                                        v
                                                     SUBMIT
```

Detailed agent rules are in [agents.md](agents.md).

## Why Agent-Oriented Tooling

Agents usually fail Kaggle workflows because of operational issues, not because they cannot write model code:

- wrong metric or CV setup;
- mismatched config and data schema;
- loading huge parquet files without sampling;
- missing experiment metadata;
- public notebook ideas not converted into structured candidates;
- wasted submission budget;
- CLI output that is hard to parse.

This project prioritizes those tooling problems before adding more model scripts.

## Roadmap

Near-term priorities:

- `kar inspect <workspace> --fix-config`: verify and patch schema/config mismatch.
- Experiment registry: command, params, features, metric, CV, runtime, and status for every run.
- Notebook miner: extract public feature lists, params, CV strategy, and ensemble formulas.
- Recipe system: turn one-off commands like `drw-clean` into reusable templates.
- OOF ensemble builder: grid/ridge/rank-average blends with metadata.
- `--json` output for agent parsing.
- `kar sync-lb`: sync Kaggle leaderboard scores and ranks into local history.

Full roadmap: [docs/agent-tooling-roadmap.md](docs/agent-tooling-roadmap.md).

## Safety

- Kaggle submissions are never automatic by default.
- Real submission, data/model deletion, credential edits, and remote pushes require explicit user intent.
- Training, feature generation, submission file generation, dry-run validation, and experiment logging can run automatically.

## License

MIT
