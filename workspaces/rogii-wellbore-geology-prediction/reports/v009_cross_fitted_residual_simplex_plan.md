# ROGII v009 Cross-Fitted Residual Simplex Plan

Snapshot: 2026-07-15. This protocol is fixed after v007 and v008 were declared exhausted and
before the four-stream v009 combination is scored against suffix `TVT` targets. Two independent
pre-score reviews amended the join key and v007 provenance bridge below; no v009 combination score
was read before either correction.

## Purpose And Statistical Status

V007's pure slow HMM scored `21.596877`, but a post-hoc diagnostic found that a small HMM residual
shrink learned on the other four canonical folds scored `11.643301`. V008's independently
predeclared formation stack scored `12.178672` versus v006 `12.184563`, improving only three folds.
The spatial streams were weak alone but received non-zero weights in every all-fold fit.

V009 tests one fixed hypothesis: the HMM registration correction and fold-safe formation surfaces
may contain complementary residual information when both remain strongly anchored to v006. This
is a hypothesis-confirming replay after inspecting v007/v008 results, not an independent estimate
of the full strategy-discovery process. There is no candidate scan, stream selection, parameter
search, gate change, or post-score retry in v009.

## Locked Inputs

The four streams, in this exact order, are:

1. `parent_v006`: the locked canonical v006 OOF prediction.
2. `hmm_wr21_slow`: the fixed target-free v007 slow HMM prediction.
3. `sp_plane_ancc_k10`: the fixed v008 ANCC spatial prediction.
4. `sp_plane_best6_k10`: the fixed v008 prospective-prefix-selected formation prediction.

Training reads `data/features/v004.parquet` for parent/HMM rows and
`data/features/v005.parquet` for parent/formation rows. `_row_index` is only unique within a well
and is forbidden as a standalone join key. Both sides must have globally unique, non-null `_id`
values and identical ID sets. They are joined one-to-one by `_id`, restored to the canonical v005
row order, and then checked for exact equality of `_well_id`, `_row_index`, `_target`, `fold`, and
`parent_v006`. A regression test must include two wells sharing the same `_row_index`.

The script also verifies the canonical manifest and v006 OOF provenance using the existing v008
validators. Because the completed v007 `run.json` did not itself bind `data/features/v004.parquet`,
`reports/v007_artifact_audit.json` is a required provenance bridge: its status must be
`verified_no_blocker`, and its recorded hashes for v004 and the v007 run must match the current
files. Selection of slow rather than flex used targets in v007, so v009 must additionally require
that `models/v007/cv_scores.json` records `hmm_wr21_slow` for all five outer-fold selections and as
the final candidate. The v004 slow column must match `models/v007/oof_preds.npy` value-for-value
after a unique-ID join to `models/v007/oof_rows.parquet`; otherwise the single canonical HMM stream
is invalid and v009 aborts rather than substituting an outer-specific stream.

The v009 and v007/v008 inference sources, this plan, both feature files, the v007 artifact audit,
v007 CV scores/OOF rows/OOF predictions/boundary diagnostics, v006/v007/v008 `run.json` files, v006
OOF/test predictions, `models/v008/model.pkl`, sample submission, and canonical fold manifest are
critical hashed inputs. Start/end hashes must match. The v007 and v008 source hashes must equal
their respective completed run records. The v008 run's recorded feature and model output hashes
must likewise equal the current v005 and `model.pkl` file hashes.

The complete raw visible-test tree, including all three horizontal and typewell files, is recorded
with the repository's content fingerprint at run start and end. A changed file, added/removed file,
or fingerprint mismatch aborts the run even when the local gate would otherwise skip test
inference. This keeps a later candidate-ready branch bound to the same test inputs that were
present when target scoring began.

The input artifacts are immutable. V009 must not rerun or modify v006, v007, or v008 predictors.

## Outer-Fold Information Boundary

For each canonical outer fold `k`, simplex fitting uses target rows only from folds `j != k`.
Parent and HMM are fixed target-free row predictions and therefore use their canonical columns on
both meta-training and evaluation rows. Formation predictions on meta-training rows use exactly the
v008 columns `nested_outer_k__sp_plane_ancc_k10` and
`nested_outer_k__sp_plane_best6_k10`; those values were built from donor universes excluding both
outer fold `k` and each query row's fold `j`. Fold `k` evaluation uses the canonical formation
columns built from donors excluding `k`.

No fold-k target, formation label, donor, residual, well summary, or fitted weight may enter its
meta-training data. The final visible-test weights may be fit once on all canonical OOF streams,
but test inference is forbidden unless the candidate gate passes.

## Exact Four-Stream Simplex

All inputs, sufficient statistics, solves, predictions, and scores use float64. Weights satisfy
`w_i >= 0` and `sum(w_i) = 1`; there is no intercept, row sampling, regularization,
clipping, or weight renormalization. Let `p0=parent_v006`, `d_i=p_i-p0` for the other three streams,
and `r=target-p0`. The optimizer minimizes the exact row-weighted SSE
`sum((r - D a)^2)` subject to `a_i >= 0` and `sum(a_i) <= 1`.

The implementation enumerates every non-empty active subset on both classes of simplex faces:

- parent-active faces solve the unconstrained normal equations for each active delta subset and
  require non-negative coefficients with sum at most one;
- parent-zero faces solve the equality-constrained KKT system for every non-empty delta subset and
  require non-negative coefficients summing to one.

The all-parent vertex is included explicitly, giving 15 audited supports: seven parent-active
delta subsets, seven parent-zero subsets, and the parent vertex. Singular systems use
`numpy.linalg.lstsq(..., rcond=1e-12)` and are retained only when the original normal/KKT residual
infinity norm is at most `1e-10 * max(1, ||A||inf * max(1, ||x||inf), ||b||inf)`. Raw feasibility
uses absolute tolerance `1e-10`; values within that tolerance may only be canonicalized to an
already-enumerated lower-dimensional face, never generally clipped or renormalized. The chosen
weights must be non-negative and sum to one within absolute tolerance `1e-12`.

Every feasible face candidate is ranked using direct full-row float64 SSE, not only the quadratic
formula; its quadratic/direct discrepancy must be at most `1e-8 * max(1, direct_SSE)`. SSE values
within `64 * eps(float64) * max(1, SSE_a, SSE_b)` are ties. Ties prefer larger parent weight, then
smaller HMM weight, then smaller ANCC weight, then smaller best6 weight; each hierarchical weight
comparison treats differences within the fixed `1e-10` feasibility tolerance as equal, so roundoff
cannot outrank the next preference. This clarification was added after a pre-score duplicate-stream
synthetic test exposed a `4e-16` raw-tuple ordering error. The output records every
support as solved, infeasible, singular-invalid, or feasible and records the selected active
support. A synthetic brute-force comparison, exact 15-support coverage, and
degenerate/vertex/near-singular tests are required before the full run. These tolerances cannot be
changed after scoring.

For each outer fold, weights are fit once on all rows in the other four folds and applied once to
fold `k`. Final artifact weights are fit once on all four canonical OOF streams. The reported v009
OOF is exclusively the concatenation of the five held-out predictions; the all-OOF weights never
score those same rows as the primary metric.

## Gate And Artifacts

`candidate_ready` requires all of:

- pooled nested OOF RMSE `<= 10.95`;
- lower RMSE than v006 in at least four of five canonical folds;
- exactly 3,783,989 rows, 773 wells, and canonical folds 1 through 5;
- four finite input streams and feasible float64 weights in every fold;
- a complete 15-support solver audit for all five outer fits and the final fit;
- finite predictions and weights;
- zero row/alignment/fold/donor provenance violations;
- exact direct-SSE verification for all five outer fits and the final fit.

`promising_continue` requires pooled RMSE `< 11.75`, at least three folds improved versus v006, and
all integrity checks. Every other outcome is `exhausted`. These thresholds are inherited from v008
and are fixed before the v009 score. The 186 known v007 slow-OOF unresolved-boundary wells are an
inherited finite auxiliary-stream risk, not a new v009 integrity failure; their exact IDs and count
must be carried into v009 diagnostics and `run.json` rather than silently omitted.

The run atomically publishes `models/v009/run.json` as `status=running` before target scoring and
finishes with `model.pkl`, `cv_scores.json`, `oof_preds.npy`, `oof_rows.parquet`,
`feature_list.txt`, and `importance.csv`. It writes a concise report and candidate/fold diagnostic
files. Failures, SIGTERM, and KeyboardInterrupt publish a failed run record without overwriting a
completed version.

Only a locally passing `candidate_ready` gate may trigger visible-test inference. That branch must:

1. align locked `models/v006/test_preds.npy` to unique sample-submission IDs;
2. run the hash-locked v007 source with both slow and flex configs through their shared 64-to-128
   corridor expansion, using the locked v006 test stream as the parent corridor, then retain slow;
3. rebuild raw ANCC and best6 streams with the hash-locked v008 source and model donor catalog,
   removing a same-ID donor before coordinate scaling and tree construction;
4. require all four streams to cover each sample ID exactly once, then apply only the final
   all-OOF weights.

Any non-finite value, unresolved slow HMM test boundary, same-ID donor survival, source/hash drift,
or ID coverage/order mismatch changes the decision to `candidate_blocked_test_failures` and forbids
`test_preds.npy`. A successful branch atomically saves only the aligned prediction array; it still
does not create a submission file. V009 never creates a submission CSV, notebook, Kaggle kernel, or
Kaggle API request. No result in this loop authorizes a real submission.
