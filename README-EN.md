<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0EA5E9,100:7C3AED&height=180&section=header&text=Kaggle%20Auto%20Research&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=36&desc=Agent-powered%20Kaggle%20competition%20automation&descSize=16&descAlignY=58" alt="Kaggle Auto Research banner" />
</p>

<p align="center">
  <strong>Agent-oriented automation framework for Kaggle competitions</strong>
</p>

<p align="center">
  <a href="./README.md">简体中文</a>
  ·
  <a href="./README-EN.md">English</a>
  ·
  <a href="./AGENTS.md">Agent Rules</a>
  ·
  <a href="./CONTRIBUTING.md">Contributing</a>
  ·
  <a href="./SECURITY.md">Security</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-Automation-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Experimental-F59E0B?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge">
</p>

<p align="center">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
  <img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-02569B?style=flat-square">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-FF6600?style=flat-square">
  <img alt="Optuna" src="https://img.shields.io/badge/Optuna-3.4%2B-334155?style=flat-square">
  <img alt="Typer" src="https://img.shields.io/badge/Typer-CLI-0F766E?style=flat-square">
  <img alt="pytest" src="https://img.shields.io/badge/pytest-ready-0A9EDC?style=flat-square&logo=pytest&logoColor=white">
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-0EA5E9?style=for-the-badge" alt="Quick Start"></a>
  <a href="#common-cli"><img src="https://img.shields.io/badge/Common_CLI-111827?style=for-the-badge" alt="Common CLI"></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/Architecture-7C3AED?style=for-the-badge" alt="Architecture"></a>
  <a href="#contributing"><img src="https://img.shields.io/badge/Contributing-10B981?style=for-the-badge" alt="Contributing"></a>
</p>

Kaggle Auto Research turns competition research, data download, EDA, feature generation, model training, iterative improvement, ensembling, submission validation, and leaderboard feedback into reusable CLI stages and versioned workspace artifacts. The goal is not to write one-off competition scripts, but to give humans and coding agents a safe, reproducible way to keep improving results.

> This project is experimental. It is designed for research, prototyping, and internal competition workflows. Real Kaggle submissions are not executed automatically by default.

## Project Positioning

This project focuses on the engineering problems that make real agent-driven Kaggle work fragile:

- Short commands hide virtualenv and Kaggle CLI details, for example `kar auth` and `kar data drw-crypto`.
- Workspaces isolate each competition's config, data, models, submissions, and experiment logs.
- Versioned artifacts preserve models, OOF predictions, test predictions, and submission files.
- Submission budgets and dry-run validation protect scarce Kaggle daily submissions.
- Public notebooks, feature selection, CV, metrics, ensembling, and leaderboard feedback are meant to become reusable tools instead of one-off scripts.

At this stage, Kaggle Auto Research is a runnable open-source research toolkit, not a polished AutoML product. Contributions should prioritize agent operability, reproducibility, and safety boundaries.

## Project Status

- Current version: `0.1.0`
- Python: `>=3.11`
- Main use cases: Kaggle tabular / time-series / low-signal competition automation
- Default behavior: generate, validate, and log locally; never submit by default
- Open-source entrypoints: [README.md](README.md), [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CHANGELOG.md](CHANGELOG.md)

## Maturity

| Module | Status | Notes |
| --- | --- | --- |
| Workspace / config | Usable | Supports multiple competition directories, templates, and basic config. |
| Kaggle OAuth / data / leaderboard | Usable | `kar auth`, `kar data`, and `kar leaderboard` work directly. |
| Baseline pipeline | Usable but early | Generic tabular flow runs; hard competitions still need recipes. |
| DRW Crypto recipe | In active validation | Strong local OOF exists, but real LB calibration is still needed. |
| Submit safety | Usable | Supports dry-run, budget status, and submission history. |
| Experiment registry | Planned | Needs immutable run IDs, params, commands, data fingerprints, and runtime metadata. |
| Notebook mining | Planned | Public notebook ideas can be absorbed manually; structured mining is not built yet. |
| JSON output | Planned | Agent parsing still needs stable `--json` output. |

## Highlights

| Capability | Description |
| --- | --- |
| <img src="https://img.shields.io/badge/Agent-Workflow-7C3AED?style=flat-square" alt="Agent Workflow"> | Research, EDA, Feature, Train, Iteration, and Submit stages communicate through filesystem artifacts. |
| <img src="https://img.shields.io/badge/Workspace-Isolation-0EA5E9?style=flat-square" alt="Workspace Isolation"> | Every competition keeps config, data, models, reports, and submissions under `workspaces/<competition>/`. |
| <img src="https://img.shields.io/badge/Config-Source_of_Truth-10B981?style=flat-square" alt="Config Source of Truth"> | `config.yaml` stores data paths, target columns, submission format, metrics, CV strategy, models, and submission budget. |
| <img src="https://img.shields.io/badge/Artifacts-Versioned-F59E0B?style=flat-square" alt="Versioned Artifacts"> | Features, models, OOF predictions, test predictions, and submissions are saved by version. |
| <img src="https://img.shields.io/badge/Submit-Safe_by_Default-EF4444?style=flat-square" alt="Safe Submit"> | The default flow generates and dry-run validates submission files; real submissions require explicit control. |
| <img src="https://img.shields.io/badge/Notebook-Ingestion-334155?style=flat-square" alt="Notebook Ingestion"> | Public notebooks can be pulled locally; future work will mine feature lists, model params, CV strategy, and ensemble formulas. |
| <img src="https://img.shields.io/badge/Roadmap-Agent_Tooling-111827?style=flat-square" alt="Agent Tooling Roadmap"> | See [docs/agent-tooling-roadmap.md](docs/agent-tooling-roadmap.md) for the Agent tooling roadmap. |

## Current DRW Crypto Progress

This repository is using DRW Crypto Market Prediction as the first real tooling testbed.

- Initial auto-feature baseline: CV R2 `-0.002859`
- Clean feature selection + LightGBM: best single-model Pearson around `0.0724`
- Public-notebook-style XGB time-slice: Pearson `0.053338`, but useful as an ensemble-diversity signal
- Closed-form Ridge on top-correlation features: TimeSeriesSplit Pearson `0.143261`
- Pearson OOF rank-normalized simplex ensemble: common-valid-OOF Pearson `0.149746`
- Current best Kaggle submission: `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv`, public LB `0.08199`
- Failed recency-proxy submission: `sub_calibrated_tail_cli_full.csv`, public LB `0.07184`
- Next offline candidate: `sub_anchor_blend_conservative.csv`, a small rank blend around the best submitted anchor
- Public leaderboard top is around `0.11 - 0.14`

Conclusion: the toolchain can run end-to-end and improve, but DRW exposed a clear CV/LB gap. The next submission should be anchor-calibrated against the best real LB result instead of blindly maximizing local tail proxies.

## Tooling Gaps Exposed By The DRW Run

The recent DRW Crypto work showed that model code was not the main bottleneck. The missing infrastructure was:

| Gap | Why it matters | Planned tool |
| --- | --- | --- |
| schema/config inspection | DRW's real target column is `label`, so templates cannot be trusted blindly. | `kar inspect <workspace> --fix-config` |
| metric contract | RMSE, R2, and Pearson confusion can mislead low-signal search. | `MetricSpec` + trainer validation |
| experiment registry | Comparing runs currently requires scanning `models/vNNN` and metadata files. | `kar experiments` / immutable run registry |
| submission metadata | Manual file submissions still need CV, model versions, and ensemble weights. | normalized submission `.json` files |
| LB sync | Kaggle scores should be written back into local history after submission. | `kar sync-lb <workspace>` |
| LB-anchor calibration | Once real LB feedback exists, next candidates should stay close to the best submitted anchor and avoid known failed directions. | `kar drw-anchor-blend <workspace>` |
| notebook miner | Public feature lists, CV schemes, and ensembles should become structured ideas. | `kar notebooks mine` |
| recipe system | Commands like `drw-ridge` should graduate from hardcoded scripts into templates. | `kar run --template crypto/low_signal_feature_select` |
| agent-friendly output | Long tables and styled output are not stable machine interfaces. | global `--json` + stable exit codes |

See [docs/agent-tooling-roadmap.md](docs/agent-tooling-roadmap.md) for the full roadmap.

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
kar sync-lb <name>
kar submit <name> --status
kar submit <name> --history
kar submit <name> --dry-run -f submissions/sub_001.csv
kar submit <name> --flush
```

Note: `kar.cmd` is a short Windows launcher at the repository root, so users do not need to type `.\.venv\Scripts\kar.exe`. After editable installation, `kar` also works directly.

DRW Crypto research helpers:

```bash
kar drw-clean drw-crypto --top-k 130 --n-estimators 1200 --learning-rate 0.015
kar drw-ridge drw-crypto --top-k 180 --alphas 300,1000,3000,10000 --cv timeseries
kar drw-public drw-crypto --model lgbm --folds 3
kar drw-ensemble drw-crypto --models v010,v011,v007,v003
kar drw-ensemble drw-crypto --models v005,v010,v012,v015,v017,v018,v019,v020,v021,v022,v023,v024,v025,v026 --method optimize --transform ranknorm
kar drw-tail-ensemble drw-crypto --samples 12000 --seed 47 --output-tag tail_cli
kar drw-anchor-blend drw-crypto --output-tag anchor_blend_safe
kar drw-anchor-blend drw-crypto --groups safe:v016+v017+v031+v032 --alpha-grid 0.15,0.18,0.19,0.20,0.21 --min-spearman 0.994 --max-rank-delta 0.025 --output-tag anchor_blend_conservative
kar drw-anchor-blend drw-crypto --selection-metric utility --failed-threshold 0.935 --risk-penalty 0.60 --output-tag anchor_blend_utility_scan
kar drw-compare-submissions drw-crypto --files sub_anchor_blend_utility_scan.csv,sub_anchor_blend_conservative.csv,sub_anti_failed_rank_beta020.csv
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

Detailed agent rules are in [AGENTS.md](AGENTS.md).

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
- Kaggle data, models, submissions, notebook caches, and local credentials should not be committed to git.

## Contributing

Contributions are welcome, especially around agent reliability and open-source
tooling:

- `kar inspect <workspace>` schema/config checks;
- experiment registry and run metadata;
- metric/CV contracts;
- public notebook mining;
- reusable recipe system;
- OOF ensemble builders;
- `--json` output and stable exit codes.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Coding
agents should read [AGENTS.md](AGENTS.md).

## License

MIT, see [LICENSE](LICENSE).
