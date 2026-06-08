# Kaggle Auto Research

<p align="center">
  <strong>AI Agent powered Kaggle competition automation framework</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-experimental-orange">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

<p align="center">
  <a href="./README.md">简体中文</a> | English
</p>

Kaggle Auto Research is an agent-oriented automation framework for Kaggle competitions. It turns competition research, data download, EDA, feature generation, model training, iterative improvement, ensembling, and submission-budget management into reusable CLI stages and versioned workspace artifacts.

> This project is experimental. It is best suited for research, prototyping, and internal competition workflows. Real Kaggle submissions are not executed automatically by default.

## Highlights

- **Workspace-first**: Every competition lives under `workspaces/<competition>/`.
- **Config as source of truth**: `config.yaml` stores data paths, target columns, metrics, CV strategy, models, and submission limits.
- **Agent-ready pipeline**: Research, EDA, Feature, Train, Iteration, and Submit stages communicate through filesystem artifacts.
- **Versioned artifacts**: Features, models, OOF predictions, test predictions, and submissions are saved by version.
- **Submission safety**: The default flow generates and dry-run validates submission files; real submissions require explicit control.
- **Tree-search iteration**: `journal.json` and `idea_pool.json` track experiments, candidates, and next ideas.

## What It Can Do

| Stage | Command | Output |
| --- | --- | --- |
| Auth | `kar auth` | Kaggle credential status |
| Init | `kar init <name>` | New competition workspace |
| Data | `kar data <name>` | Downloaded and extracted raw data |
| Research | `kar research <name>` | `reports/research_notes.md` |
| EDA + features | `kar eda <name>` | `reports/eda_summary.md`, `data/features/v*.parquet` |
| Train | `kar train <name>` | `models/v*/model.pkl`, CV scores, OOF/test predictions |
| Iterate | `kar pipeline <name> --iterate 5` | Updated experiment tree and idea pool |
| Ensemble | `kar ensemble <name>` | Optimized ensemble from top local models |
| Submit | `kar submit <name> --dry-run -f ...` | Validated submission file, optional queue |

## Installation

```bash
git clone https://github.com/<your-org>/kaggle-auto-research.git
cd kaggle-auto-research

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS / Linux

pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

Optional deep learning dependencies:

```bash
pip install -e ".[deep]"
```

## Kaggle Setup

Authenticate once before downloading competition data:

```bash
kar auth
```

You can also use the standard Kaggle API credential file:

```text
~/.kaggle/kaggle.json
```

On Windows this is usually:

```text
C:\Users\<you>\.kaggle\kaggle.json
```

## Quick Start

```bash
# 1. Create a workspace
kar init titanic --type tabular --url https://www.kaggle.com/competitions/titanic

# 2. Download and extract data
kar data titanic

# 3. Run individual stages
kar research titanic
kar eda titanic
kar train titanic

# 4. Check pipeline state
kar status titanic

# 5. Validate a submission without sending it to Kaggle
kar submit titanic --dry-run -f submissions/sub_001.csv
```

Run the pipeline from scratch:

```bash
kar pipeline titanic --full
```

Run improvement iterations:

```bash
kar pipeline titanic --iterate 5
kar analyze titanic
kar ensemble titanic
```

## CLI Reference

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

There is also a DRW Crypto helper for the current research workflow:

```bash
kar drw-clean drw-crypto --top-k 350 --n-estimators 700
```

## Workspace Layout

Every competition lives under `workspaces/<competition>/`:

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

## Configuration

`config.yaml` is the source of truth for each workspace.

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

For time series, finance, and trading competitions, prefer time-based CV:

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

The pipeline state machine:

```text
RESEARCH -> EDA -> FEATURES -> TRAIN -> EVALUATE -> CANDIDATE_READY
                                ^                       |
                                |                       v
                             ITERATE              USER DECISION
                                                        |
                                                        v
                                                     SUBMIT
```

## Safety Rules

Kaggle Auto Research automates reversible work and protects irreversible actions.

Automated by default:

- Create workspaces.
- Download and extract competition data.
- Generate reports, features, models, predictions, and submission CSVs.
- Update `config.yaml`, `.state/`, `journal.json`, and `idea_pool.json`.
- Dry-run validate submission files.

Requires explicit user control:

- Submit to Kaggle.
- Delete models, data, or submissions.
- Modify credentials or `.env`.
- Push to a remote Git repository.

## Supported Competition Types

| Type | Typical use | Default model family |
| --- | --- | --- |
| `tabular` | Structured CSV/parquet competitions | LightGBM, XGBoost |
| `crypto` | Finance, crypto, time series prediction | LightGBM with time-based CV |
| `llm` | NLP and generative AI competitions | Transformers / prompt workflows |

## Development

```bash
make install
make dev
make test
make lint
make format
```

Equivalent commands:

```bash
pytest tests/ -v
ruff check src/ cli/ tests/
ruff format src/ cli/ tests/
```

## Project Structure

```text
kaggle-auto-research/
|-- cli/                    # Typer CLI entrypoint
|-- src/kaggle_auto/        # Core Python package
|   |-- eda/
|   |-- features/
|   |-- models/
|   |-- pipeline/
|   |-- research/
|   |-- submission/
|   |-- tuning/
|   `-- utils/
|-- templates/              # Workspace config templates
|-- tests/
|-- workspaces/             # Local competition workspaces
|-- agents.md               # Agent operating rules
|-- CLAUDE.md               # Claude Code project instructions
`-- pyproject.toml
```

## Roadmap

- More robust schema validation before feature generation.
- Better metric registry for Kaggle-specific objectives.
- Notebook and discussion research exporters.
- Public leaderboard tracking and local/LB correlation analysis.
- Safer submit confirmation flow for fully autonomous agents.
- More template presets for vision, NLP, and recommender competitions.

## Contributing

Contributions are welcome. Good first areas:

- Add tests for pipeline stages and submission validation.
- Improve competition templates.
- Add metrics and CV strategies.
- Improve EDA reports for large parquet datasets.
- Add model adapters while keeping artifact formats stable.

Before opening a pull request:

```bash
make test
make lint
```

## License

MIT License.

