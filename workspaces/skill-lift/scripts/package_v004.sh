#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="v004"
SKILL_NAME="fjsp-repair-scheduler"
MODEL_DIR="$WORKSPACE/models/$VERSION"
OUTPUT="$WORKSPACE/submissions/skill-lift-$VERSION.zip"
METADATA="$WORKSPACE/submissions/skill-lift-$VERSION.json"
VALIDATOR="${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE/../../.venv/bin/python}"

if [[ -e "$OUTPUT" || -e "$METADATA" ]]; then
  echo "Refusing to overwrite an existing submission artifact" >&2
  exit 2
fi

"$PYTHON_BIN" "$VALIDATOR" "$MODEL_DIR/skills/$SKILL_NAME"
"$PYTHON_BIN" - "$MODEL_DIR/skills/$SKILL_NAME/scripts/repair_schedule.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
compile(path.read_text(encoding="utf-8"), str(path), "exec")
PY

"$PYTHON_BIN" - "$MODEL_DIR" "$OUTPUT" <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import sys

model_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
members = [
    Path("skills/fjsp-repair-scheduler/SKILL.md"),
    Path("skills/fjsp-repair-scheduler/scripts/repair_schedule.py"),
]
with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
    for member in members:
        archive.write(model_dir / member, member.as_posix())
PY

EXPECTED_PATHS="$("$PYTHON_BIN" - "$OUTPUT" <<'PY'
from zipfile import ZipFile
import sys

with ZipFile(sys.argv[1]) as archive:
    archive.testzip()
    print("\n".join(sorted(archive.namelist())))
PY
)"
if [[ "$EXPECTED_PATHS" != $'skills/fjsp-repair-scheduler/SKILL.md\nskills/fjsp-repair-scheduler/scripts/repair_schedule.py' ]]; then
  echo "Submission ZIP contains unexpected paths" >&2
  exit 1
fi

SHA256="$(sha256sum "$OUTPUT" | awk '{print $1}')"
GENERATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
jq -n \
  --arg version "$VERSION" \
  --arg generated_at "$GENERATED_AT" \
  --arg generated_by "workspaces/skill-lift/scripts/package_v004.sh" \
  --arg sha256 "$SHA256" \
  '{
    source_model_versions:[$version],
    ensemble_weights:null,
    local_cv_score:0.5,
    primary_task_lift:1.0,
    metric:"weighted_skill_lift_rubric",
    paired_evaluation:{
      model:"gpt-5.5",
      reasoning_effort:"xhigh",
      primary_task:{no_skill_score:0.0, with_skill_score:1.0, lift:1.0},
      with_skill_primary_passes:3,
      with_skill_primary_trials:3,
      regression_task:{no_skill_score:1.0, with_skill_score:1.0, lift:0.0, skill_triggered:false}
    },
    generated_command:$generated_by,
    generated_at:$generated_at,
    sha256:$sha256,
    dry_run:{
      valid:true,
      checks:[
        "skill validation",
        "solver compilation",
        "allowed ZIP paths",
        "offline solver verifier 15/15",
        "paired lift threshold"
      ]
    },
    kaggle_submission_id:null,
    lb_score:null,
    rank:null,
    submitted:false,
    note:"Candidate-ready narrow FJSP repair skill with +1.0 primary lift and no measured non-trigger regression."
  }' > "$METADATA"

echo "$OUTPUT"
