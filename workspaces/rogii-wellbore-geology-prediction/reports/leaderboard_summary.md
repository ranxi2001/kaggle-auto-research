# ROGII Leaderboard Summary

Snapshot: 2026-07-14 17:25 UTC. Metric is RMSE; lower is better.

| Field | Value |
|---|---:|
| Kaggle submission ref | 54693365 |
| Kernel | `ranxi169/kar-rogii-v006-latent-surface-particle-filter` |
| Kernel version | 1 |
| Submission status | `COMPLETE` |
| Public LB RMSE | **10.693** |
| Current rank | **2,940 / 4,909** |
| Local pooled OOF RMSE | 12.184563 |
| Local minus public | 1.491563 |
| Prior v004 public RMSE | 14.683 |
| Public improvement vs v004 | 3.990 RMSE / 27.17% |
| Prior v004 rank | 3,628 / 4,829 |
| Top-10% reference rank | 491 |
| Score at top-10% reference | 7.120 |

Submission ref `54693365` completed from private kernel version 1 with public RMSE 10.693. This is a
material improvement over v004's 14.683 and confirms that the latent-surface particle-filter
direction transfers to hidden data. Public performance is 1.491563 better than the strict grouped
OOF estimate, so the local gate was conservative for this submission.

The model remains outside the competitive region: rank 2,940 of 4,909 and 3.573 RMSE behind the
top-10% reference. A second small blend or tree-model adjustment is not justified. The next core
experiment should keep the validated `U = TVT + Z` state representation while replacing causal
single-point filtering with a full forward-backward HMM/smoother and windowed GR registration.

The private leaderboard score is unavailable before competition close. No submission after ref
`54693365` has been made.
