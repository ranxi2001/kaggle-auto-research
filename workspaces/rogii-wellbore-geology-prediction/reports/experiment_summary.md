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
| v007 | Window-registration latent-surface HMM, fully nested | `exhausted` | 21.596877 | 21.582929 | 0.769249 | +35.75% |
| v008 | Fold-safe formation local planes + nested simplex | `exhausted` | 12.178672 | 12.156925 | 0.727620 | -23.45% |
| v009 | Cross-fitted HMM + formation residual simplex | `promising_continue` | **11.633994** | **11.623245** | **0.500158** | **-26.88%** |
| v010 | Adaptive front20 HMM basis + five-stream simplex | `exhausted` diagnostic | 11.618124 | 11.607395 | 0.499498 | -26.98% |

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

V007 replaced the causal particle path with two pure forward-backward HMMs using non-overlapping
21-row GR registration blocks. A target-free six-well smoke projected 3.72 hours and used 0.293 GiB;
the complete 773-well run finished in 3,112 seconds with zero inference fallbacks. Both candidates
failed decisively: `hmm_wr21_slow` scored 21.596877 and `hmm_wr21_flex` scored 29.487150. Nested
selection chose slow in every fold, zero folds improved over v006, and 186 selected slow wells had
an unresolved boundary. V007 was therefore marked `exhausted`; it produced no test prediction,
notebook, submission CSV, or Kaggle request.

Independent dense-lattice and artifact audits found no forward-backward sign/index error. The
failure mechanism is structural: v006 supplies only the moving corridor coordinates, while the HMM
transition cancels that center motion and has no parent attraction. Repeated GR motifs can therefore
create confidently wrong paths whose drift grows down the suffix. A post-hoc, strictly fold-wise
linear shrink diagnostic (`alpha=0.149..0.178`) scored 11.643301 with all five folds improved, but
the hypothesis was discovered after observing v007 targets and is not reported as a v007 candidate.
The next core experiment instead uses outer-train-only formation spatial surfaces; HMM shrink is
retained only as a later, separately predeclared weak-feature idea.

V008 rebuilt the train/test formation boundary inside every outer fold. Formation donors excluded
both the evaluated outer fold and each meta-training query fold; prospective formation selection
and vertical bias calibration used visible-prefix rows only. The complete run finished in 397
seconds with zero inference, donor-fold, self-donor, or non-finite failures. Its nested simplex
scored 12.178672, only 0.005891 (0.048%) better than v006, and improved three of five folds. The
standalone ANCC and best-six streams scored 41.905122 and 39.898377, while final all-OOF weights
remained 94.19% on v006. V008 therefore failed the fixed `<11.75` promising gate and was marked
`exhausted`; it produced no test predictions or submission artifacts.

V009 is a separately predeclared hypothesis-confirming experiment. It combines the locked v006
parent with v007 slow HMM and both v008 formation streams using exact four-stream simplex weights
fit only on the other four canonical folds. Its protocol and statistical caveat were frozen before
the four-stream target score in `reports/v009_cross_fitted_residual_simplex_plan.md`.

The complete v009 run scored 11.633994, improving v006 by 0.550569 (4.52%) and all five canonical
folds. Outer-fold HMM weights stayed in a narrow `0.147-0.173` range; formation weights remained
small and less stable. The fixed promising gate passed, but the `<=10.95` candidate gate did not,
so v009 produced no test prediction or submission artifact. Two independent artifact checks
recomputed every pooled/fold score and selected simplex face, verified all 15 supports in six fits,
and found zero alignment, provenance, direct-SSE, partial, test, or submission issues.

Post-hoc progress analysis shows that v009 still underweights HMM correction in the first 20% of a
suffix but is near the late-suffix optimum. Boundary neutralization is ruled out: the 186 inherited
boundary wells improved more than non-boundary wells. V010 therefore fixes one linear front20 basis
before scoring; its expected gain is known to be small and the direction stops unless it beats
11.50 and at least three v009 folds.

V010 scored 11.618124 and improved all five v009 folds, but the pooled gain was only 0.015870
(0.136%). It failed the fixed `<11.50` diagnostic gate and is `exhausted`. Because the front20 basis
was selected from all-fold v009 target residuals, v010 was predeclared as adaptive diagnostic
evidence and permanently forbidden from candidate/test inference. No test or submission artifact
was produced.

Independent recomputation reproduced the v010 score and all 31-support fits, but a post-run source
review found four artifact-contract gaps: its in-run audit commit ordering was unsafe, audit status
was not conditioned on integrity counts, `feature_list.txt` omitted ten nested meta columns, and
`run.json` omitted required provenance fields. The original self-audit is marked superseded and
`reports/v010_independent_artifact_audit.json` records the verified score plus these gaps. V010 must
not be reused as a candidate or parent run; v009 remains the last contract-clean promising result.

## Leaderboard Calibration

Kaggle kernel version 1 retrained v004 offline and reported pooled CV RMSE 14.903702. Submission ref
54653094 scored public RMSE 14.683, rank 3628 of 4829. V006 submission ref 54693365 subsequently
completed at public RMSE 10.693, rank 2940 of 4909. That is a 3.990 RMSE / 27.17% public improvement
over v004, and 1.491563 better than the strict v006 grouped OOF estimate.

The local v006 CSV passes `kar submit --dry-run`. Its private, internet-disabled notebook package
reproduced the 14,151-row visible-test CSV byte for byte with zero fallback wells both locally and
in completed Kaggle kernel version 1. A 2026-07-14 17:25 UTC snapshot places the top-10% boundary at
RMSE 7.120, so v006 validates the new core direction but remains well outside the competitive range.
