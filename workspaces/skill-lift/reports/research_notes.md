# Skill Lift - Research Notes

## Competition Summary

- **Slug**: `skill-lift`
- **URL**: https://www.kaggle.com/competitions/skill-lift
- **Track selected**: Track 1 - Static Skills
- **Deadline**: 2026-08-13 06:55 UTC
- **Prize**: USD 20,000 total; USD 10,000 per track
- **Primary objective**: maximize paired agent lift with skills versus without skills
- **Submission**: one ZIP rooted at `skills/` plus a Kaggle Writeup of at most 2,000 words
- **Kaggle status**: rules have not been accepted and no Kaggle submission has been made

## Evaluation Contract

Final ranking uses a fixed model panel and a private held-out task mix. Each task is run with and
without the submitted skill set. The published rubric is:

| Component | Weight | Local implication |
|---|---:|---|
| Paired lift | 60% | Compare identical task/model trials with and without skills |
| Compliance and safety | 20% | Avoid injection, exfiltration, sandbox escape, destructive actions, and excessive context |
| Generalization | 10% | Prefer reusable procedures over task identifiers or memorized answers |
| Writeup quality | 10% | Preserve commands, versions, results, and design rationale |

Per-task scores normally fall in `[0, 1]`; safety violations can reduce a task to `-1`. Final
aggregation uses repeated runs and paired bootstrap estimates. A single successful public task is
only a smoke test, not a reliable lift estimate.

## Data And Benchmark

The competition provides no conventional train/test data. The Kaggle Data page contains only a
placeholder note, and the rules state that no competition dataset is supplied. Development uses the
public SkillsBench corpus instead.

- **SkillsBench version**: repository `main` at commit
  `44cdda48f6e8c4d381f4f5075c0f4a051ba69e98`
- **Published suite**: 87 tasks in SkillsBench v1.1
- **Local acquisition**: shallow, blob-filtered sparse checkout
- **Local checkout size**: approximately 2.4 MB before run artifacts
- **Selected development tasks**:
  - `offer-letter-generator`
  - `lake-warming-attribution`
  - `xlsx-recover-data`
  - `threejs-to-obj`
  - `manufacturing-fjsp-optimization`

All five task packages pass `bench tasks check`. The `offer-letter-generator` oracle passes its
verifier with score `1.0`, proving that Docker, task inputs, oracle, and verifier form a valid local
loop.

## Tooling Findings

- BenchFlow `0.6.4` is installed as a `uv` tool.
- Docker Engine is available. Docker Compose was missing and was installed as the official
  `v5.3.1` CLI plugin after SHA-256 verification.
- WSL proxy variables originally intercepted Docker-gateway health checks. Adding `172.17.0.1` to
  `NO_PROXY` for the BenchFlow child process fixes the LiteLLM health check without changing global
  proxy settings.
- The isolated Codex runner now mounts the custom Responses API provider config and creates a
  process-scoped auth file from the environment. The auth file is removed at exit and is never
  written into the workspace. Runs use `gpt-5.5` with `xhigh` reasoning.
- Verifier packages are cached in an ignored local wheelhouse. Verifier files are copied into the
  container only after the model finishes, preventing agent access to tests during execution.
- Three.js verification requires generating ground truth before pytest. The runner now reproduces
  that task setup locally; saved outputs from both modes pass all three checks.
- Earlier authentication failures remain recorded as failed runs and are excluded from lift.

## Public Notebook Signals

The highest-voted public notebooks at the snapshot include:

1. `No Agent Skills? Improvise them Gemma4`
2. `Understanding Agent Skill Lift in BenchFlow`
3. `Meta Skill Optimization`
4. `Skills, Not Shortcuts: A Visual Guide`
5. `Agent Skill Lift: Evolutionary Triad`

The competition description reports that curated skills improve average pass rate, while some tasks
regress and self-generated skills can be net negative. The practical implication is to keep skills
short, trigger narrowly, evaluate paired outcomes, and treat regressions as first-class failures.

## V001 Strategy

`v001` contains one static skill, `artifact-first-execution`. It targets file-producing tasks without
encoding public task names or answers. Its procedure is:

1. Extract exact output paths, schemas, and invariants.
2. Inspect inputs with format-aware parsers and preserve them read-only.
3. Generate outputs deterministically and atomically.
4. Reopen outputs and validate both structure and domain invariants.
5. Refuse verifier tampering, answer-key access, exfiltration, destructive actions, and sandbox escape.

The bundled read-only `audit_artifact.py` supports basic CSV, TSV, JSON, XLSX, DOCX, and OBJ checks.
It passes compilation and lint, accepts valid CSV/JSON fixtures, and correctly rejects the incomplete
XLSX and placeholder-bearing DOCX development inputs.

## Paired Evaluation Result

- **Skill validation**: passed `skill-creator/quick_validate.py`
- **Task structural validation**: 5/5 passed
- **Oracle smoke test**: 1/1 passed
- **Model**: `gpt-5.5`, `xhigh`
- **Lifetime API usage**: 2,539,271 input-plus-output tokens
- **Valid selection pairs**: 5
- **Excluded exploratory pairs**: 2
- **V004 candidate mean lift**: `0.5`
- **V004 primary task lift**: `1.0`
- **Candidate status**: v004 submitted through Kaggle Writeup; public visibility pending

| Version | Task | No skill | With skill | Lift | Selection status |
|---|---|---:|---:|---:|---|
| v001 | offer-letter-generator | 1.0 | 1.0 | 0.0 | valid, ceiling |
| v002 | lake-warming-attribution | 0.0 | 0.0 | 0.0 | valid |
| v002 | threejs-to-obj | 1.0 | 1.0 | 0.0 | valid, ceiling |
| v002 | xlsx-recover-data | 0.0 | 0.0 | 0.0 | excluded, verifier inconsistency |
| v003 | xlsx-recover-data | 0.0 | 0.0 | 0.0 | excluded, baseline-reuse exploratory |
| v004 | manufacturing-fjsp-optimization | 0.0 | 1.0 | 1.0 | valid, candidate primary |
| v004 | offer-letter-generator | 1.0 | 1.0 | 0.0 | valid, non-trigger regression |

The broad artifact skill was neutral on its panel. V004 replaced it with a narrowly triggered FJSP
repair skill and a pure-Python solver. The no-skill model passed 13/15 checks but moved an operation
earlier than its baseline anchor. The skill-enabled model completed all 1,024 routing combinations,
produced a policy-feasible schedule, and passed 15/15 checks. The skill-enabled condition passed
3/3 runs. On the unrelated offer-letter regression task the FJSP skill did not load and the score
remained 1.0.

The candidate ZIP passes the repository dry-run and is stored as `submissions/skill-lift-v004.zip`.
After competition entry was confirmed, the standard `CreateSubmission` API returned 400 because
this Hackathon accepts submissions through Kaggle Writeups rather than the ordinary competition
file endpoint. The workspace now uses `submission.mode: writeup` to prevent further invalid API
uploads. The validated ZIP and 543-word report were attached to a Writeup, and the authenticated
Kaggle UI reported it as submitted on 2026-07-11. Anonymous access still returned 404, so public
visibility is not independently verified.

The XLSX outputs computed the Science average as `7610.3`, consistent with the FY2019-2024 label,
the six included annual rows, and completed peer averages. The public verifier alone expects
`7444.4`, which is the FY2019-2023 five-value average. v003 added general formula/range validation
but correctly retained the workbook-consistent result. This task is excluded rather than optimized
toward a public-test-specific answer.

## Next Experiments

1. Verify that the submitted Writeup becomes publicly visible.
2. Sync any host-side score, validation result, or rank into the submission sidecar and journal.
3. Add further safety and non-trigger tasks before a later replacement submission.

## Idea Pool

1. **High**: verifier preflight for dependencies, setup scripts, and semantic consistency.
2. **High**: DOCX-specific run-aware replacement skill tested on a non-ceiling task.
3. **High**: negative-regression gate; reject any candidate that reduces a passing task.
4. **Validated**: FJSP domain procedure with downtime, precedence, and policy validation.
5. **Medium**: artifact audit expansion for schedule constraints and coordinate transforms.
6. **Low**: Track 2 meta-skill only after Track 1 has reproducible positive lift.

## Source Gaps And Inconsistencies

- The competition description refers to roughly 94 public tasks and earlier results on 84 tasks;
  the current SkillsBench v1.1 site and repository describe 87 tasks.
- The Submission Requirements page says two submissions per day, while the generic hackathon rules
  say one submission per team. Local config keeps the more conservative limit of one until the host
  clarifies it.
- `xlsx-recover-data` has an internally inconsistent public verifier for the missing Science average;
  it is excluded from model selection and no verifier-specific value is encoded in a skill.
- Kaggle competition entry is active. Standard `CreateSubmission` is not the Hackathon submission
  mechanism; the official requirements mandate New Writeup, ZIP attachment, Save, then Submit.
