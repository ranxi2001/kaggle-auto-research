# ROGII Leaderboard Summary

Snapshot: 2026-07-13 16:11 UTC. Metric is RMSE; lower is better.

| Field | Value |
|---|---:|
| Kaggle submission ref | 54653094 |
| Kernel | `ranxi169/kar-rogii-v004-grouped-residual-beam-ensemble` |
| Kernel version | 1 |
| Public LB RMSE | 14.683 |
| Current rank | 3,628 / 4,829 |
| Local pooled OOF RMSE | 14.743751 |
| Kaggle-run pooled CV RMSE | 14.903702 |
| Top-10% reference rank | 483 |
| Score at top-10% reference | 7.154 |

The public LB score is 0.060751 better than local OOF, so the split is directionally calibrated at
this score level. However, rank 3,628 is not competitive for a medal. Even the broad top-10%
reference is 7.529 RMSE better, which is too large to close with blend-weight or small tree-model
tuning.

The next viable branch must change the core inference model: stronger GR/typewell registration,
trajectory/dip priors, or a genuinely learned sequence model with fully nested validation. A second
submission should be reserved until one of those directions produces a large grouped-CV gain.
