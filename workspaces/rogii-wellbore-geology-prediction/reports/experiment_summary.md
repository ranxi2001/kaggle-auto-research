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

## Candidate Decision

v004 is the first leaderboard calibration candidate. v003 is the conservative fallback if the
blend underperforms. Both were generated without public leaderboard artifacts or training-target
lookups for the visible test IDs.

The local visible-test CSV and JSON sidecar passed sample-order, uniqueness, row-count, and finite
prediction checks. Because ROGII is notebook-only, these files are format smoke tests rather than
direct API submissions. Actual submission remains pending explicit user approval.
