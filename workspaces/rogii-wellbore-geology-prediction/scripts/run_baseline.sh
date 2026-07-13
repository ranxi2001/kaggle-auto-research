#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$WORKSPACE/../.." && pwd)"

cd "$PROJECT_ROOT"
uv run python "$WORKSPACE/scripts/rogii_baseline.py" --workspace "$WORKSPACE" inspect
MODEL_DIR="$(uv run python "$WORKSPACE/scripts/rogii_baseline.py" --workspace "$WORKSPACE" train)"
uv run python "$WORKSPACE/scripts/rogii_baseline.py" \
  --workspace "$WORKSPACE" predict --model-dir "$MODEL_DIR"
