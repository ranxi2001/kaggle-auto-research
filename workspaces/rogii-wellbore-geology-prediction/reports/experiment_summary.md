# ROGII Experiment Summary

Primary metric: pooled OOF RMSE over 3,783,989 natural suffix rows from 773 wells. Lower is better.

| Version | Recipe | Pooled RMSE | Fold mean | Fold std | Relative to v001 |
|---|---|---:|---:|---:|---:|
| v001 | Last-known `TVT_input` | 15.909853 | 15.886199 | 0.871379 | baseline |
| v002 | Nested-CV GR beam alignment | 15.691115 | 15.674416 | 0.896424 | -1.37% |
| v003 | Grouped LightGBM residual with beam features | 14.787502 | 14.752702 | 1.016404 | -7.05% |
| v004 | 17.5% v002 + 82.5% v003 OOF ensemble | **14.743751** | 14.728042 | 0.814993 | **-7.33%** |

## Selection Notes

v002 beam OOF predictions use nested well GroupKFold selection. The fixed `very_loose` beam with
0.7 shrinkage used as a v003 feature was first identified in a full-data target scan, so v003's
reported score is not a fully nested estimate. v003 averages five LightGBM fold models. v004's
weight comes from a 41-point OOF blend scan; its 0.043751 RMSE gain over v003 is also subject to
OOF weight-selection optimism.

The v003 model improves on v001 for 450 of 773 wells and worsens on 323, indicating real but
heterogeneous signal. v004 reduces fold dispersion versus v003. The hardest v004 fold scores
16.311239, so hidden distribution shift remains the dominant risk.

## Leaderboard Calibration

Kaggle kernel version 1 retrained v004 offline and reported pooled CV RMSE 14.903702. The code
submission completed as ref 54653094 with public LB RMSE 14.683, rank 3628 of 4829. Local CV and LB
move in the same direction, but the candidate is far outside the current top-10% reference boundary
at rank 483 and RMSE 7.154.

The submission verifies the end-to-end notebook and validation design, but v004 is no longer a medal
candidate. Further submissions should require a large local improvement from a new geological
sequence-alignment strategy, not another small blend or LightGBM parameter adjustment.
