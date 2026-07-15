# ROGII Wellbore Geology Prediction Research Notes

Research snapshot: 2026-07-13 UTC. Local experiment state updated: 2026-07-14 UTC. Sources are the
official Kaggle competition API, competition pages, organizer materials, and the downloaded
competition files.

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

Rows from the same well are dependent, so learned candidates use a canonical five-fold GroupKFold
manifest balanced by scored-row count and grouped by well. Features may use the validation well's
own observed `TVT_input` prefix and all visible GR/trajectory values because those inputs are
present at hidden-test inference. They never use the validation suffix's `TVT`. The primary score
is pooled OOF RMSE across all 3,783,989 scored rows; fold mean and per-well scores are diagnostic
only. Parameter comparisons for v006 are selected from the other four outer folds before each
validation fold is scored.

The v001 baseline extends the last finite `TVT_input` value through the missing suffix. GR beam
alignment then maps the horizontal GR sequence to each well's typewell reference. The learned
model predicts a strongly regularized LightGBM residual over the last-known baseline using only
target-independent trajectory, GR, prefix-anchor, typewell, and beam features.

## Current Candidate

The best valid local candidate is v006, a latent-surface particle filter over
`U = TVT + Z` and its dip rate. It uses the visible prefix to initialize the state, the paired
typewell GR as the emission reference, and the real MD/Z trajectory for propagation. Non-finite
suffix GR rows skip the measurement update rather than receiving interpolated evidence. The fixed
candidate `pf_scale_12_hold_0p2` was selected independently by all five nested outer-fold choices.

v006 pooled OOF RMSE is 12.184563, versus 14.743751 for v004 and 15.909853 for v001. It improves all
five canonical v004 folds, has zero fallback wells, and passes the predeclared local candidate gate.
The preceding v005 attempt has no valid CV score: audit stopped it at 230 of 773 wells because it
treated interpolated missing GR as observed evidence. Its partial feature parquet must not be used.

With explicit user authorization, v006 private kernel version 1 was submitted as ref 54693365. It
completed at public RMSE 10.693, rank 2940 of 4909, versus v004 public RMSE 14.683. The 27.17%
public improvement confirms that the latent-surface direction transfers to hidden data, while the
1.491563 advantage over grouped OOF shows that this local estimate was conservative for v006.

## Residual Risks

- GroupKFold measures generalization to unseen wells but cannot reproduce the hidden well mix.
- v006 is a causal particle filter, not a full forward-backward smoother; repeated GR motifs can
  still cause path ambiguity and the 20% last-known hold is only a global shrinkage choice.
- The 1,200,837 missing suffix GR rows receive transition-prior propagation only. Long missing runs
  therefore depend heavily on the prefix dip-rate estimate and motion model.
- The fixed grid is fully nested for the reported outer score, and the first LB calibration is
  directionally strong, but one public result does not establish robustness to the private split.
- V006 public RMSE 10.693 remains 3.573 above the 7.120 top-10% boundary. The next core direction is
  a full latent-surface HMM/forward-backward smoother with windowed GR registration, not another
  small blend or tree-model adjustment.
