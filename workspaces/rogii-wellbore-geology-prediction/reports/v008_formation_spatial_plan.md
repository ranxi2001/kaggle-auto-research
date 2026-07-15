# ROGII v008 Fold-Safe Formation Spatial Plan

Snapshot: 2026-07-15. This plan was fixed after v007 was declared exhausted and before any v008
formation prediction was scored against a suffix `TVT` target.

## Why The Strategy Changes

V007's pure slow/flex HMM candidates scored `21.596877 / 29.487150`, with zero of five folds
improving over v006. The implementation and artifacts passed independent audit: the failure comes
from allowing GR registration to replace the v006 path without a center prior, not from a sign,
index, or forward-backward bug. A post-hoc nested shrink diagnostic reaches `11.643301`, so the HMM
contains weak complementary information, but it is not reliable enough to remain the core method.

Every training horizontal file contains `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, and `BUDA`
formation surfaces. All three test horizontal files omit those columns. Public formation notebooks
construct their spatial imputer from all 773 training wells before GroupKFold, allowing validation
formation data into the donor pool; their local CV cannot be reused. V008 instead recreates the
train/test information boundary inside every canonical outer fold.

## Target-Free Coverage Audit

- `773 / 773` training files share the declared formation schema; `765 / 773` wells have complete
  finite values for all six formation medians and may act as donors.
- Depending on the outer fold, `96.77% - 99.35%` of validation-well centroids and
  `95.51% - 97.17%` of sampled suffix trajectory points lie inside the outer-training donor convex
  hull.
- Fold-specific normalized k=10 nearest-donor p95 distance is `0.0525 - 0.0766`.
- A 100k-query vectorized k=10 local-plane benchmark took `0.397 s`; the query-only projection for
  the nested data is `18.3 s`. The formal smoke and full run still include CSV I/O and artifact
  checks.

This audit read coordinates and formation columns but no suffix target values.

## Fixed Spatial Streams

Each outer fold builds one donor table from only its outer-training wells. A donor contributes its
median `X`, median `Y`, and six median formation surfaces. Coordinates are standardized from that
donor table using population standard deviation (`ddof=0`). For each query point, k=10 neighbours
are selected by Euclidean distance in standardized coordinates. Weights are
`1 / max(distance, 1e-6)` and normalized per query. The weighted local-plane design is centered on
the query: `[(X_neighbor-X_query)/scale_X, (Y_neighbor-Y_query)/scale_Y, 1]`. A fixed `1e-8`
ridge is added to the two slope diagonals only; the intercept is unpenalized. The 3x3 system must
solve to finite values, otherwise inference fails closed without pseudoinverse or constant
fallback. A held-out validation well is absent from the donor table. Final test inference removes
the entire same-ID donor before computing the scaler and building the tree; all three visible test
IDs currently overlap training IDs.

The query side may read only `MD`, `X`, `Y`, `Z`, `GR`, and `TVT_input`.

Two fixed spatial streams are generated:

1. `sp_plane_ancc_k10`: impute ANCC; estimate
   `b = median(TVT_input + Z - ANCC_hat)` on the last 512 visible-prefix rows; predict suffix as
   `TVT = -Z + ANCC_hat + b`.
2. `sp_plane_best6_k10`: for each of the six formations, calibrate `b` on prefix rows
   `[-640:-128]` and score the last 128 visible-prefix rows. Select the lowest prospective-prefix
   RMSE with formation-name order as the tie break, then recalibrate that formation on the last 512
   visible rows and predict the suffix. The suffix target never participates in selection.

All training wells have at least 851 visible-prefix rows, so these windows are fixed rather than
data-dependent. Non-finite inputs, fewer than k donors, self-donor use, query/order mismatch, or an
unsolved local plane fail closed and are recorded; they are not silently replaced with target-based
values.

## Nested Stack

The sole v008 primary candidate is `sp_nnls_v006`. Its fixed input streams are:

- v006 `pf_scale_12_hold_0p2` OOF or visible-test prediction,
- `sp_plane_ancc_k10`,
- `sp_plane_best6_k10`.

For outer fold k, stack training uses a second spatial cross-fit inside the outer-training universe.
For every j other than k, the inner-j formation stream is queried on fold j from a donor tree that
excludes both folds k and j. The four inner-heldout streams and their fold-j targets fit the three
simplex weights (`w >= 0`, `sum(w) = 1`, no intercept); fold k's formation stream is separately
queried from donors that exclude k and receives those weights once. This prevents fold k's
train-only formation labels from influencing even the meta-training predictors used to choose its
weights.

No row sampling is used; residual-scale 2x2 sufficient statistics reproduce the row-weighted RMSE
objective exactly. To avoid ill-conditioning from absolute TVT values near `1e4`, the solver is
implemented on residual-scale differences: `p0=v006`, `d1=ANCC-p0`, `d2=best6-p0`, and
`r=target-p0`. It enumerates the feasible interior, all three triangle edges, and all vertices for
`w1>=0`, `w2>=0`, `w1+w2<=1`; each candidate is verified by direct
`sum((r-w1*d1-w2*d2)^2)` and deterministic tie breaking. Unconstrained clipping/renormalization and
an absolute-prediction KKT solve are forbidden. `w0=1-w1-w2`.

Final test weights may use canonical five-fold OOF streams because the unseen test donor universe
is absent from all five fits. Final spatial donors may then use all eligible training wells subject
to same-ID exclusion. The v006 parent test stream is the locked `models/v006/test_preds.npy`, joined
and verified against `data/raw/sample_submission.csv` IDs. Any locally recomputed parent must match
that artifact value-for-value; both files and `models/v006/run.json` are critical hashed inputs.
Training also verifies the parent feature's stored `_target` against the current raw suffix target
for every row before using the parent stream.

V007 HMM is deliberately excluded from this first formation stack. Its shrink diagnostic was
selected after observing v007 targets and would blur whether formation adds a genuinely new signal.

## Gate And Stop Conditions

`candidate_ready` requires pooled nested OOF RMSE `<= 10.95`, improvement over the corresponding
v006 score in at least four of five folds, zero inference/non-finite failures, and zero donor-fold or
self-donor violations. `promising_continue` requires RMSE `< 11.75` and at least three improved
folds. Every other result is `exhausted`.

A fixed target-free smoke uses the six suffix-row-count quantile wells below. Each well is queried
under its canonical donor context plus the four double-exclusion inner contexts used by strict
nested stacking.

| Well | Fold | Suffix rows | Contexts | Queried rows including 640-row prefix |
|---|---:|---:|---:|---:|
| `fba7683c` | 2 | 407 | 5 | 5,235 |
| `fef8af96` | 3 | 3,826 | 5 | 22,330 |
| `8bb9c1e6` | 1 | 4,546 | 5 | 25,930 |
| `722cf0d8` | 3 | 5,143 | 5 | 28,915 |
| `ff8bb73a` | 3 | 5,927 | 5 | 32,835 |
| `ea3a0e38` | 1 | 10,052 | 5 | 53,460 |

The smoke therefore executes `168,705` query rows. The complete nested training workload is fixed
at `5 * (3,783,989 + 640 * 773) = 21,393,545` query rows and 15 unique donor contexts: five
single-fold exclusions and ten fold-pair exclusions. Its conservative projection is
`2 * (tree_build_seconds + query_seconds * 21,393,545 / 168,705) / 3600`; this must be at most
60 minutes and peak RSS at most 4 GiB.

The fixed output is `reports/v008_formation_runtime_smoke.json`. The command preflights the final
and `.partial` paths, writes/fsyncs/verifies the partial JSON, and publishes with `os.replace`
without overwrite. It records exact wells, folds, row counts, contexts, failure list, and start/end
hashes for the implementation, this plan, raw training data, and canonical manifest. Full training
must reject a missing, stale, ineligible, or non-exact smoke artifact.

`models/v008/` must atomically publish a `status=running` `run.json` before any formation computation
and replace it with `completed` or `failed` metadata under `BaseException`, including SIGTERM and
KeyboardInterrupt; partial/staging paths remain identified for audit. A completed run must contain `model.pkl`,
`cv_scores.json`, `oof_preds.npy`, `oof_rows.parquet`, `feature_list.txt`, and `importance.csv`, plus
the versioned formation feature/report artifacts and their SHA-256 values in `run.json`. Only
`candidate_ready` may add `test_preds.npy`; no other decision reads test formation shortcuts or
produces test predictions. V008 never creates a submission CSV, notebook, or Kaggle API request,
and this loop never submits to Kaggle.
