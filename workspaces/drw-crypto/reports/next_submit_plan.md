# DRW Next Submit Plan

## Current Best

| Field | Value |
| --- | --- |
| Competition | `drw-crypto-market-prediction` |
| Best file | `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv` |
| Best public LB | `0.08199` |
| Private score shown | `0.08268` |

## Queued Submission

| Field | Value |
| --- | --- |
| File | `sub_anchor_blend_micro_scan.csv` |
| CV / utility | `0.128042` |
| Reserved date | `2026-06-09` |
| Reason | Conservative model-backed move; highest geometry score among current unsubmitted candidates and already in reserve queue. |

Before submitting, run:

```powershell
.\kar.cmd submit drw-crypto --status
```

Confirm the reserve queue contains only `sub_anchor_blend_micro_scan.csv`.

After the daily budget resets, submit with:

```powershell
.\kar.cmd submit drw-crypto --flush
```

## Candidate Ranking

| Candidate | Valid | Local score | Score source | Anchor Spearman | Max failed Spearman | Rank delta | Geometry score | Role |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `sub_anchor_blend_micro_scan.csv` | yes | `0.128042` | `scores.utility` | `0.998170` | `0.999508` | `0.013142` | `0.664044` | Conservative next submit |
| `sub_anti_failed_rank_beta100.csv` | yes |  |  | `0.999154` | `0.991965` | `0.008920` | `0.650248` | Diagnostic public-feedback probe |
| `sub_low_failed_pool_random_best.csv` | yes | `0.132554` | `scores.utility` | `0.929671` | `0.954688` | `0.079369` | `0.646105` | Exploratory model-backed candidate |
| `sub_low_failed_pool_grid.csv` | yes | `0.130378` | `scores.utility` | `0.911003` | `0.939423` | `0.090004` | `0.631737` | Larger geometry exploration |

## Post-Submit

Run:

```powershell
.\kar.cmd leaderboard drw-crypto
.\kar.cmd sync-lb drw-crypto
```

Then update `reports/experiment_log.md`, `reports/lb_sync.csv`, and the next-submit plan with the new public LB result.

## Decision Notes

- Do not submit exploratory fallbacks before the queued micro candidate unless explicitly choosing exploration.
- If the queued candidate does not beat public LB `0.08199`, keep the first submission as best and add the new result to failed-direction references.
- The ignored local `reports/next_submit_plan.json` mirrors this plan for local automation, but the Markdown file is the tracked source for handoff.
