# ROGII v009 Cross-Fitted Residual Simplex

Status: `promising_continue`.

## Canonical Nested CV

- Pooled OOF RMSE: `11.633993822`
- Canonical folds better than v006: `5 / 5`
- Fold RMSE: `11.495554, 10.897975, 11.350480, 12.171255, 12.200958`
- Final all-OOF weights: `parent_v006=0.789938300`, `hmm_wr21_slow=0.159370733`, `sp_plane_ancc_k10=0.013518446`, `sp_plane_best6_k10=0.037172521`
- Visible-test status: `not_eligible`

Each outer fit uses canonical parent/HMM streams and formation streams whose donor universe excludes both the outer fold and each meta row's fold. Held-out evaluation uses only canonical formation columns.

No submission file, notebook, kernel, API request, or competition submission is created by this experiment.
