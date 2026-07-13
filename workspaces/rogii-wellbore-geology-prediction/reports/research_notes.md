# ROGII Wellbore Geology Prediction Research Notes

Snapshot: 2026-07-13 UTC. Sources are the official Kaggle competition API, competition pages,
organizer materials, and the downloaded competition files.

## Competition Contract

- Competition: `rogii-wellbore-geology-prediction`
- Category: Featured; official API reports `awards_points=true`
- Metric: pooled root mean squared error (RMSE), minimize
- Entry deadline: 2026-07-29 23:59 UTC
- Final deadline: 2026-08-05 23:59 UTC
- Submission: Kaggle Notebook only, internet disabled, CPU/GPU runtime at most 9 hours
- Output: `/kaggle/working/submission.csv` with columns `id,tvt`
- Data: about 1.23 GiB across 2,327 files; 773 training wells and a hidden test of about 200 wells

The authenticated account has joined the competition and the data is available locally. Each
training well has a horizontal-well CSV, a typewell CSV, and a PNG. Public test files are format
examples; Kaggle replaces them with hidden wells when executing a submitted notebook.

## Verified Data Contract

The 773 horizontal training files contain 5,092,255 rows. Every well has a finite `TVT_input`
prefix followed by one contiguous missing suffix, producing 3,783,989 scored rows. In the observed
prefix, `TVT_input` exactly equals `TVT`. `sample_submission.id` maps exactly to missing-suffix rows
as `{well_id}_{original_row_index}`.

Only `MD`, `X`, `Y`, `Z`, `GR`, and `TVT_input` from horizontal wells are treated as inference-time
inputs. `TVT` is the target. `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, and `BUDA` are train-only
surfaces and are excluded. Typewell `TVT` and `GR` are available reference logs; typewell
`Geology`, PNGs, public-test target lookups, and prebuilt public submission artifacts are excluded.

The three visible test well IDs overlap training IDs. They are used only to verify file discovery,
ID alignment, and output format. Their training targets do not participate in model selection.

## Validation Design

Rows from the same well are dependent, so all learned candidates use five-fold GroupKFold by
`_well_id`. Features may use the validation well's own observed `TVT_input` prefix and all visible
GR/trajectory values because those inputs are present at hidden-test inference. They never use the
validation suffix's `TVT`. The primary score is pooled OOF RMSE across all 3,783,989 scored rows;
fold mean and per-well scores are diagnostic only.

The v001 baseline extends the last finite `TVT_input` value through the missing suffix. GR beam
alignment then maps the horizontal GR sequence to each well's typewell reference. The learned
model predicts a strongly regularized LightGBM residual over the last-known baseline using only
target-independent trajectory, GR, prefix-anchor, typewell, and beam features.

## Current Candidate

The best local candidate is v004: 17.5% v002 nested-CV GR beam and 82.5% v003 grouped residual
LightGBM. Its pooled OOF RMSE is 14.743751, versus 15.909853 for v001, a 7.33% relative reduction.
The fixed beam feature parameters were first identified from full-data target scans, and the
ensemble weight was selected on the same OOF predictions used for reporting. The v003/v004 scores
therefore have selection optimism and are not fully nested estimates; leaderboard calibration
remains necessary.

A self-contained, private, internet-disabled notebook package exists under `notebooks/v004/`. Kaggle
kernel version 1 completed in about 11.4 minutes, dynamically discovered the mounted data, trained
from competition data, mapped predictions to runtime sample IDs, and wrote only `submission.csv`.
Submission ref 54653094 completed with public RMSE 14.683 and rank 3628 of 4829 teams.

## Residual Risks

- GroupKFold measures generalization to unseen wells but cannot reproduce the hidden well mix.
- The fifth fold is materially harder than the others, so well-distribution shift remains visible.
- Beam-feature and blend choices are not fully nested; v003 is the fallback candidate.
- The first leaderboard calibration is directionally consistent with local CV but far outside the
  current medal range; a materially different sequence-alignment strategy is required.
