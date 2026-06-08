# Changelog

All notable project changes should be documented here.

The format follows the spirit of Keep a Changelog, and this project uses
pre-1.0 versions while the CLI and artifact contracts are still evolving.

## 0.1.0 - 2026-06-09

### Added

- Agent-oriented Kaggle workflow documentation.
- `kar` CLI project entrypoint and Windows launcher.
- Kaggle OAuth helper via `kar auth`.
- Workspace lifecycle commands for init, data download, EDA, training, pipeline
  execution, analysis, ensembling, leaderboard lookup, and submission dry-runs.
- DRW Crypto research helpers for cleaned LightGBM runs, public-notebook-style
  baselines, and OOF ensembling.
- Open-source project docs: README, contributing guide, security policy,
  changelog, license, and agent instructions.

### Changed

- Documented the toolchain roadmap around experiment registry, metric/CV
  contracts, schema inspection, public notebook mining, reusable recipes, and
  agent-parseable CLI output.

### Security

- Documented submission, credential, and data-handling safety boundaries.
