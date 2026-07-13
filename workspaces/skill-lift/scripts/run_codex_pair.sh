#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-v002}"
TASK="${TASK:-offer-letter-generator}"
TASK_ROOT="$WORKSPACE/data/raw/skillsbench/tasks/$TASK"
SKILLS_ROOT="$WORKSPACE/models/$VERSION/skills"
MODEL="${MODEL:-gpt-5.5}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$WORKSPACE/models/$VERSION/eval/$RUN_ID"
RUN_MODES="${RUN_MODES:-no-skill with-skill}"
IMAGE="skill-lift-${TASK}:local"
START_EPOCH="$(date +%s)"
PRIOR_RUNTIME_SECONDS=0
if [[ -f "$RUN_ROOT/run.json" ]]; then
  PRIOR_RUNTIME_SECONDS="$(jq -r '.runtime_seconds // 0' "$RUN_ROOT/run.json")"
fi
if [[ -z "${AGENT_TIMEOUT:-}" ]]; then
  AGENT_TIMEOUT=900
  if [[ "$TASK" == "manufacturing-fjsp-optimization" ]]; then
    AGENT_TIMEOUT=1800
  fi
fi
CODEX_HOME_HOST="${CODEX_HOME:-$HOME/.codex}"
CONFIG_FILE="${CODEX_CONFIG_FILE:-$CODEX_HOME_HOST/config.toml}"
AUTH_FILE="${CODEX_AUTH_FILE:-$CODEX_HOME_HOST/auth.json}"
WHEELHOUSE="${PYTEST_WHEELHOUSE:-$WORKSPACE/.state/wheelhouse}"
TEMP_AUTH_DIR=""
ACTIVE_CONTAINER=""

cleanup() {
  if [[ -n "$ACTIVE_CONTAINER" ]]; then
    docker rm --force "$ACTIVE_CONTAINER" > /dev/null 2>&1 || true
  fi
  if [[ -n "$TEMP_AUTH_DIR" ]]; then
    rm -rf "$TEMP_AUTH_DIR"
  fi
}
trap cleanup EXIT

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  TEMP_AUTH_DIR="$(mktemp -d)"
  AUTH_FILE="$TEMP_AUTH_DIR/auth.json"
  jq -n --arg key "$OPENAI_API_KEY" '{OPENAI_API_KEY:$key}' > "$AUTH_FILE"
  chmod 600 "$AUTH_FILE"
fi

CODEX_JS="$(readlink -f "$(command -v codex)")"
CODEX_PACKAGE="$(cd "$(dirname "$CODEX_JS")/.." && pwd)"
CODEX_BIN="$(find "$CODEX_PACKAGE/node_modules/@openai/codex-linux-x64/vendor" -type f -path '*/bin/codex' -print -quit)"

if [[ ! -x "$CODEX_BIN" ]]; then
  echo "Codex native binary not found" >&2
  exit 2
fi
if [[ ! -f "$AUTH_FILE" ]]; then
  echo "Codex API authentication not found" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Codex provider config not found at $CONFIG_FILE" >&2
  exit 2
fi
if [[ ! -d "$WHEELHOUSE" ]]; then
  echo "Offline pytest wheelhouse not found at $WHEELHOUSE" >&2
  exit 2
fi
if [[ ! -d "$TASK_ROOT" || ! -d "$SKILLS_ROOT" ]]; then
  echo "Task or skills directory is missing" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
awk 'BEGIN { blocks=0 } /^---$/ { blocks += 1; next } blocks >= 2 { print }' \
  "$TASK_ROOT/task.md" > "$RUN_ROOT/prompt.txt"

docker build --quiet --tag "$IMAGE" "$TASK_ROOT/environment" > "$RUN_ROOT/image_id.txt"

run_mode() {
  local mode="$1"
  local result_dir="$RUN_ROOT/$mode"
  local container="skill-lift-${RUN_ID,,}-${mode}"
  local codex_rc verifier_rc score
  local -a mounts
  local -a verifier_packages
  local skill_dir skill_name

  mkdir -p "$result_dir"
  mounts=(
    --volume "$CODEX_BIN:/usr/local/bin/codex:ro"
    --volume "$AUTH_FILE:/root/.codex/auth.json:ro"
    --volume "$CONFIG_FILE:/root/.codex/config.toml:ro"
    --volume "$RUN_ROOT:/artifacts:rw"
    --volume "$WHEELHOUSE:/wheelhouse:ro"
  )
  if [[ "$mode" == "with-skill" ]]; then
    for skill_dir in "$SKILLS_ROOT"/*; do
      if [[ -d "$skill_dir" && -f "$skill_dir/SKILL.md" ]]; then
        skill_name="$(basename "$skill_dir")"
        mounts+=(--volume "$skill_dir:/root/.codex/skills/$skill_name:ro")
      fi
    done
  fi

  docker run --detach --name "$container" "${mounts[@]}" "$IMAGE" sleep infinity > /dev/null
  ACTIVE_CONTAINER="$container"

  verifier_packages=(pytest==8.4.1)
  if [[ "$TASK" == "threejs-to-obj" ]]; then
    verifier_packages+=(numpy==2.3.1)
  fi
  docker exec "$container" pip3 install --break-system-packages --quiet \
    --no-index --find-links /wheelhouse "${verifier_packages[@]}"

  set +e
  timeout --foreground "$AGENT_TIMEOUT" docker exec --interactive "$container" env CODEX_HOME=/root/.codex \
    /usr/local/bin/codex exec \
      --model "$MODEL" \
      --dangerously-bypass-approvals-and-sandbox \
      --skip-git-repo-check \
      --ephemeral \
      --color never \
      --json \
      --cd /root \
      - < "$RUN_ROOT/prompt.txt" > "$result_dir/codex.jsonl" 2> "$result_dir/codex.stderr.log"
  codex_rc=$?

  docker exec "$container" mkdir -p /verifier
  docker cp "$TASK_ROOT/verifier/." "$container:/verifier" > /dev/null
  if [[ "$TASK" == "threejs-to-obj" ]]; then
    docker cp "$TASK_ROOT/verifier/gen_ground_truth.mjs" \
      "$container:/root/gen_ground_truth.mjs" > /dev/null
    docker exec --workdir /root "$container" node /root/gen_ground_truth.mjs \
      > "$result_dir/ground_truth.log" 2>&1
  fi
  docker exec "$container" python3 -m pytest -q /verifier/test_outputs.py \
    > "$result_dir/verifier.log" 2>&1
  verifier_rc=$?
  set -e

  case "$TASK" in
    offer-letter-generator)
      if docker exec "$container" test -f /root/offer_letter_filled.docx; then
        docker cp "$container:/root/offer_letter_filled.docx" "$result_dir/" > /dev/null
      fi
      ;;
    xlsx-recover-data)
      if docker exec "$container" test -f /root/nasa_budget_recovered.xlsx; then
        docker cp "$container:/root/nasa_budget_recovered.xlsx" "$result_dir/" > /dev/null
      fi
      ;;
    lake-warming-attribution|threejs-to-obj)
      if docker exec "$container" test -d /root/output; then
        docker cp "$container:/root/output" "$result_dir/output" > /dev/null
      fi
      ;;
    manufacturing-fjsp-optimization)
      if docker exec "$container" test -d /app/output; then
        docker cp "$container:/app/output" "$result_dir/output" > /dev/null
      fi
      ;;
  esac

  score=0
  if [[ "$verifier_rc" -eq 0 ]]; then
    score=1
  fi
  jq -n \
    --arg mode "$mode" \
    --arg model "$MODEL" \
    --argjson codex_exit_code "$codex_rc" \
    --argjson verifier_exit_code "$verifier_rc" \
    --argjson score "$score" \
    '{mode:$mode, model:$model, codex_exit_code:$codex_exit_code, verifier_exit_code:$verifier_exit_code, score:$score}' \
    > "$result_dir/result.json"

  docker rm --force "$container" > /dev/null
  ACTIVE_CONTAINER=""
}

for mode in $RUN_MODES; do
  case "$mode" in
    no-skill|with-skill) run_mode "$mode" ;;
    *) echo "Unknown run mode: $mode" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$RUN_ROOT/no-skill/result.json" || ! -f "$RUN_ROOT/with-skill/result.json" ]]; then
  echo "A complete pair requires both no-skill and with-skill results" >&2
  exit 2
fi

NO_SKILL_SCORE="$(jq '.score' "$RUN_ROOT/no-skill/result.json")"
WITH_SKILL_SCORE="$(jq '.score' "$RUN_ROOT/with-skill/result.json")"
NO_SKILL_CODEX_RC="$(jq '.codex_exit_code' "$RUN_ROOT/no-skill/result.json")"
WITH_SKILL_CODEX_RC="$(jq '.codex_exit_code' "$RUN_ROOT/with-skill/result.json")"
LIFT=$((WITH_SKILL_SCORE - NO_SKILL_SCORE))
SOURCE_COMMIT="$(git -C "$WORKSPACE/data/raw/skillsbench" rev-parse HEAD)"
END_EPOCH="$(date +%s)"
RUNTIME_SECONDS=$((PRIOR_RUNTIME_SECONDS + END_EPOCH - START_EPOCH))
NO_SKILL_USAGE="$(jq -c 'select(.type == "turn.completed") | .usage' \
  "$RUN_ROOT/no-skill/codex.jsonl" | tail -n 1)"
WITH_SKILL_USAGE="$(jq -c 'select(.type == "turn.completed") | .usage' \
  "$RUN_ROOT/with-skill/codex.jsonl" | tail -n 1)"
if [[ -z "$NO_SKILL_USAGE" ]]; then
  NO_SKILL_USAGE="{}"
fi
if [[ -z "$WITH_SKILL_USAGE" ]]; then
  WITH_SKILL_USAGE="{}"
fi
STATUS="completed"
if [[ "$NO_SKILL_CODEX_RC" -ne 0 || "$WITH_SKILL_CODEX_RC" -ne 0 ]]; then
  STATUS="failed"
fi

jq -n \
  --arg run_id "$RUN_ID" \
  --arg status "$STATUS" \
  --arg version "$VERSION" \
  --arg model "$MODEL" \
  --arg task "$TASK" \
  --arg source_commit "$SOURCE_COMMIT" \
  --argjson runtime_seconds "$RUNTIME_SECONDS" \
  --argjson no_skill_score "$NO_SKILL_SCORE" \
  --argjson with_skill_score "$WITH_SKILL_SCORE" \
  --argjson lift "$LIFT" \
  --argjson no_skill_usage "$NO_SKILL_USAGE" \
  --argjson with_skill_usage "$WITH_SKILL_USAGE" \
  '{
    run_id:$run_id,
    status:$status,
    version:$version,
    model:$model,
    task:$task,
    source_commit:$source_commit,
    evaluator:"task verifier via isolated Docker containers",
    runtime_seconds:$runtime_seconds,
    no_skill_score:$no_skill_score,
    with_skill_score:$with_skill_score,
    lift:$lift,
    usage:{no_skill:$no_skill_usage, with_skill:$with_skill_usage}
  }' > "$RUN_ROOT/run.json"

echo "$RUN_ROOT"
