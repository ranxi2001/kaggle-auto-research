# ROGII Experiment Summary

Primary metric: pooled OOF RMSE over 3,783,989 natural suffix rows from 773 wells. Lower is better.

| Version | Recipe | Status / decision | Pooled RMSE | Fold mean | Fold std | Relative to v001 |
|---|---|---|---:|---:|---:|---:|
| v001 | Last-known `TVT_input` | completed | 15.909853 | 15.886199 | 0.871379 | baseline |
| v002 | Nested-CV GR beam alignment | completed | 15.691115 | 15.674416 | 0.896424 | -1.37% |
| v003 | Grouped LightGBM residual with beam features | completed | 14.787502 | 14.752702 | 1.016404 | -7.05% |
| v004 | 17.5% v002 + 82.5% v003 OOF ensemble | submitted | 14.743751 | 14.728042 | 0.814993 | -7.33% |
| v005 | Latent-surface particle filter with invalid missing-GR handling | failed at 230/773 | - | - | - | invalid |
| v006 | Corrected latent-surface particle filter with nested grid selection | `candidate_ready` | **12.184563** | **12.163748** | **0.709535** | **-23.41%** |

## Selection Notes

v002 beam OOF predictions use nested well GroupKFold selection. The fixed `very_loose` beam with
0.7 shrinkage used as a v003 feature was first identified in a full-data target scan, so v003's
reported score is not a fully nested estimate. v003 averages five LightGBM fold models. v004's
weight comes from a 41-point OOF blend scan; its 0.043751 RMSE gain over v003 is also subject to
OOF weight-selection optimism.

The old v002 and v003 runs used different well-to-fold mappings. v004's pooled OOF remains valid
because predictions were joined by ID, but its old fold labels inherit the v002 mapping. Therefore,
the former claims that v004 reduced v003 fold dispersion or that a particular old fold was hardest
are not cross-version comparisons and have been retired.

v005 was interrupted after 230 of 773 wells when audit found that interpolated missing suffix GR
values were being treated as independent observed evidence. Its `data/features/v002.parquet` is an
invalid partial artifact and has no CV score. v006 preserves the original missing mask and skips the
particle measurement update on non-finite GR rows. It processed 2,583,152 observed-GR rows and
1,200,837 missing-GR rows with zero training fallback wells.

The v006 scale/hold grid was fixed before the corrected run. The earlier 12-well target-reading
smoke was used only as a runtime gate and did not change that grid. For each canonical validation
fold, the candidate was selected using only the other four folds; all five selections independently
chose `pf_scale_12_hold_0p2`. The final pooled score improves on v004 by 2.559187 RMSE, or 17.36%.

| Canonical fold | v004 RMSE | v006 RMSE | v006 - v004 |
|---:|---:|---:|---:|
| 1 | 14.135772 | 12.048074 | -2.087699 |
| 2 | 14.291809 | 11.384291 | -2.907518 |
| 3 | 13.510538 | 11.547903 | -1.962635 |
| 4 | 15.144821 | 13.352526 | -1.792295 |
| 5 | 16.460358 | 12.485944 | -3.974413 |

The predeclared gate was pooled RMSE at most 12.5, improvement in at least four of five canonical
folds, and no fallback wells. v006 passes at 12.184563 with five of five folds improved. This gate
permits a local candidate and notebook package; it does not authorize a Kaggle submission.

## Leaderboard Calibration

Kaggle kernel version 1 retrained v004 offline and reported pooled CV RMSE 14.903702. Submission ref
54653094 scored public RMSE 14.683, rank 3628 of 4829. V006 submission ref 54693365 subsequently
completed at public RMSE 10.693, rank 2940 of 4909. That is a 3.990 RMSE / 27.17% public improvement
over v004, and 1.491563 better than the strict v006 grouped OOF estimate.

The local v006 CSV passes `kar submit --dry-run`. Its private, internet-disabled notebook package
reproduced the 14,151-row visible-test CSV byte for byte with zero fallback wells both locally and
in completed Kaggle kernel version 1. A 2026-07-14 17:25 UTC snapshot places the top-10% boundary at
RMSE 7.120, so v006 validates the new core direction but remains well outside the competitive range.
