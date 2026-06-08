# DRW Submit Status

## Current Best

| Field | Value |
| --- | --- |
| Competition | `drw-crypto-market-prediction` |
| Best file | `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv` |
| Best public LB | `0.08199` |
| Private score shown | `0.08268` |

## Latest Submission

| Field | Value |
| --- | --- |
| File | `sub_anchor_blend_micro_scan.csv` |
| CV / utility | `0.128042` |
| Kaggle ref | `53488725` |
| Public LB | `0.07720` |
| Private score shown | `0.07698` |
| Result | Did not beat current best public LB `0.08199`. |

## Queue Status

- Local reserve queue is empty.
- Do not re-submit `sub_anchor_blend_micro_scan.csv`.
- Treat this result as another failed-direction reference for the next candidate scan.

## Candidate Ranking

| Candidate | Valid | Local score | Score source | Anchor Spearman | Max failed Spearman | Rank delta | Geometry score | Role |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `sub_anchor_blend_micro_scan.csv` | yes | `0.128042` | `scores.utility` | `0.998170` | `0.999508` | `0.013142` | `0.664044` | Submitted; failed to beat best |
| `sub_anti_failed_rank_beta100.csv` | yes |  |  | `0.999154` | `0.991965` | `0.008920` | `0.650248` | Diagnostic public-feedback probe |
| `sub_low_failed_pool_random_best.csv` | yes | `0.132554` | `scores.utility` | `0.929671` | `0.954688` | `0.079369` | `0.646105` | Exploratory model-backed candidate |
| `sub_low_failed_pool_grid.csv` | yes | `0.130378` | `scores.utility` | `0.911003` | `0.939423` | `0.090004` | `0.631737` | Larger geometry exploration |

## Next Work

Run:

```powershell
.\kar.cmd leaderboard drw-crypto
.\kar.cmd sync-lb drw-crypto
```

Then rebuild candidate scoring with `sub_anchor_blend_micro_scan.csv` included in the failed reference set before spending another submission.

## Decision Notes

- Keep the first submission as best until a candidate beats public LB `0.08199`.
- Do not submit exploratory fallbacks without a fresh geometry scan that includes all failed submissions.
- The ignored local `reports/next_submit_plan.json` mirrors this plan for local automation, but the Markdown file is the tracked source for handoff.
