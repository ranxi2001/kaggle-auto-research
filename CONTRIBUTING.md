# Contributing

Thanks for considering a contribution to Kaggle Auto Research. This project is
an experimental, agent-oriented toolchain for reproducible Kaggle workflows.

## Development Setup

```bash
git clone https://github.com/ranxi2001/kaggel-auto-research.git
cd kaggel-auto-research

python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Use the short launcher on Windows:

```bash
.\kar auth
.\kar ls
```

Or, after activating the virtual environment:

```bash
kar --help
```

## What To Contribute

High-value areas:

- schema inspection and config repair;
- metric and CV contracts;
- experiment registry and reproducible run metadata;
- public notebook mining;
- reusable competition recipes;
- OOF ensemble builders;
- JSON output and stable exit codes for agents;
- tests for CLI behavior and artifact contracts.

The roadmap is in [docs/agent-tooling-roadmap.md](docs/agent-tooling-roadmap.md).

## Pull Request Guidelines

- Keep changes focused and explain the workflow they improve.
- Add or update tests when behavior changes.
- Do not commit Kaggle data, credentials, model artifacts, submissions, or
  local notebook caches.
- Preserve existing workspace artifacts unless the change explicitly migrates
  them.
- Prefer reusable templates and CLI surfaces over one-off competition scripts.

## Verification

Before opening a pull request, run:

```bash
ruff check .
pytest
```

If a command cannot be run locally, mention that in the pull request.

## Agent Contributions

Coding agents should follow [AGENTS.md](AGENTS.md).
Real Kaggle submissions, artifact deletion, credential edits, and remote pushes
require explicit user intent.
