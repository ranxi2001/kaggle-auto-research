# ROGII v008 Fold-Safe Formation Spatial Stack

Status: `exhausted`.

## Grouped CV

- Nested pooled OOF RMSE: `12.178672471`
- Canonical folds better than v006: `3 / 5`
- Fold RMSE: `11.881213, 11.119481, 11.824426, 13.102638, 12.856869`
- Final all-OOF simplex weights: `parent_v006=0.941892560`, `sp_plane_ancc_k10=0.013938349`, `sp_plane_best6_k10=0.044169090`
- Audited donor contexts: `15`

## Leakage Boundary

Each outer-fold meta fit uses inner streams whose donor universe excludes both the outer fold and the inner query fold. Validation streams exclude their own outer fold. Final test inference excludes a same-ID donor before coordinate scaling.

No submission CSV, notebook, Kaggle API request, or competition submission is created by this experiment.
