# ROGII v007 Kaggle Skill And Public Solution Audit

Snapshot: 2026-07-15. This report was written before running any v007 target-scored smoke or full
CV. It fixes the research sources, candidate family, validation protocol, and promotion gate for
the next experiment.

## Skill Selection

The selected research skill is NVIDIA's
[`nvidia-kaggle-skill`](https://github.com/NVIDIA/nvidia-kaggle/tree/410c70b0b076b0d0ca76f10a855e7e337d9bd09b/skills/nvidia-kaggle-skill),
audited at commit `410c70b0b076b0d0ca76f10a855e7e337d9bd09b` and installed locally through
the Codex `skill-installer`. Its use in this project is restricted to read-only competition,
kernel, discussion, and writeup research. Its submission, dataset upload, dependency installation,
and fetched-notebook execution workflows are disabled for this loop. Kaggle submission remains
behind the repository's explicit user-confirmation gate.

Two alternatives were rejected:

- [`Kaggle/kaggle-skills`](https://github.com/Kaggle/kaggle-skills/tree/e5b4d148d8758a900c9f8192f294de00010e31ce)
  currently contains hackathon judging, agent exam, and benchmark-authoring skills, not competitor
  research or model optimization.
- [`wenmin-wu/ds-skills`](https://github.com/wenmin-wu/ds-skills/tree/718b09722905726c428fcc4eab4d470df4842545)
  is a useful idea index, but its 535 skills contain no HMM, forward-backward, Viterbi, DTW, or GR
  curve-registration implementation. Several relevant examples lack grouping, tests, or correct
  missing-value behavior, so none were installed.

## Public Notebook Evidence

The selected skill's kernel workflow pulled the public source and metadata for the exact versions
below. Stable version refs, retrieval timestamps, metadata hashes, and both Kaggle source-endpoint
hashes are recorded in
[`v007_public_notebook_evidence.json`](v007_public_notebook_evidence.json). Those version refs and
dual hashes make the research snapshot auditable and retrievable without checking notebook source
into the repository; the two source hashes are kept distinct because Kaggle's current-pull and
historical-version endpoints can serialize notebook JSON differently.

| Source | Role | Pull-source SHA-256 |
|---|---|---|
| [Exact HMM smoother](https://www.kaggle.com/code/amerhu/rogii-wellbore-geology-exact-hmm-smoother?scriptVersionId=332922278) | second-order `TVT x dip-rate` forward-backward reference | `2321997c8bcbca442d7d7abfc5b9b7eeed251bac8c671b3554b54c9355c231dc` |
| [9.251 DWT-based](https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based?scriptVersionId=318875876) | PF, beam, DTW, NCC, formation, and stack reference | `bcac73d029699a5636167bbb0bc01bac7006e8045a155bf8f3ee6036e28083af` |
| [LB 7.776 Ridge SP](https://www.kaggle.com/code/lightningv08/lb-7-776-rogii-ridge-sp?scriptVersionId=325217839) | strong public hybrid and selector reference | `d8237531dea2a1b733d1e2255ba9087bafa3bd55f63771c91bdb9efa5809832e` |
| [Drift Targeting + NCC](https://www.kaggle.com/code/mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore?scriptVersionId=322521377) | multi-scale NCC and grouped residual-model reference | `7db8e0b50b8d1e613954c828ebd0fdd8a76d5df82ffa734ac4c64f35ca8748a4` |
| [Physical Model](https://www.kaggle.com/code/sunnywu27/rogii-wellbore-tvt-physical-model?scriptVersionId=323501089) | PF/beam ensemble reference | `0fde93243068a7ccf766ac0f270c489ecc8d2fd4edb4dbeeccaf51853bdb220f` |
| [Target-Free Alignment](https://www.kaggle.com/code/pilkwang/rogii-eda-target-free-alignment-for-tvt?scriptVersionId=324752534) | leakage policy, self-correlation, and fold-specific imputer reference | `8c6491a59a1409c8ac492b8a90d090785627690acd0f9a0401ef568ba23f69aa` |

NVIDIA's published
[`ROGII strategy brief`](https://github.com/daxiongshu/competition-brief-demo/blob/229da91adfdbcaebd09f250a2fd3deab64a11ef8/codex_rogii-wellbore-geology-prediction_002/brief.md)
was used as the discovery index. Claims that affect v007 were checked against the pulled notebook
source rather than accepted from the brief alone.

## Findings That Survive Audit

1. The validated v006 state `U = TVT + Z` plus dip rate remains the right core representation.
   Full forward-backward inference can use later finite suffix GR to resolve earlier ambiguous
   paths and returns the posterior mean, which is the appropriate point estimate for RMSE.
2. Strong public systems repeatedly use several GR samples jointly through NCC, path alignment, or
   window features. V007 should replace repeated single-row emissions with non-overlapping
   window-registration blocks while retaining the physical motion model.
3. Every real finite suffix GR value must contribute to at most one v007 block. Missing GR is never
   interpolated, copied, or counted as evidence. A block with no finite GR has neutral emission.
4. Prefix-only robust affine calibration can reconcile horizontal and typewell GR scale/offset.
   Suffix `TVT` is never read by the prediction function.
5. Formation/spatial priors are the best later contingency for long GR gaps, but public notebooks
   often construct imputers on all 773 wells before CV. A future formation experiment must fit its
   imputer only on each outer-training fold and calibrate a held-out well from its visible prefix.

## Rejected Public Paths

- The public exact-HMM notebook reports only a small sample comparison, not canonical grouped CV.
  Its implementation fills suffix GR in both directions, uses zero-filled prefix GR in one sigma
  estimate, fixes a `last_tvt +/- 100` band without edge diagnostics, and does not normalize the
  long float32 forward lattice at every step. Copying it would repeat the invalid v005 evidence
  handling.
- Public lateral-prefix self-correlation is a useful future weak emission, but its standalone
  nearest-window match can jump among repeated motifs and has no motion or TVT-support constraint.
- Public DTW aligns whole curves to endpoints, includes interpolated missing GR in some variants,
  and did not improve the audited NCC model's OOF result. It is not the v007 core.
- Formation KNNs built globally before GroupKFold, public-LB-shaped selector thresholds, same-ID
  visible-test shortcuts, hill-climbed public blends, and non-nested post-processing are excluded.
- Public beam implementations do not provide a full globally traced path and omit parts of the
  real MD/Z dynamics; they are not substitutes for forward-backward smoothing.

## Predeclared v007 Experiment

The state is `(U, dip_rate)`. The leakage-safe nested v006 OOF path is used only as the center of a
time-varying offset corridor; it adds no center penalty. Hidden inference recomputes the same v006
parent path from allowed inputs.

Common fixed settings:

- offset grid step: `0.5 ft`
- initial corridor: `+/-64 ft`; rerun at `+/-128 ft` when target-free posterior edge mass fails
- rate states: `33`
- rate span: `max(0.12, abs(prefix_rate) + 0.04)`
- registration block: `21` non-overlapping suffix rows
- emission: Student-t with `df=4`, summed over real finite GR and tempered by `20`;
  an observed point outside typewell TVT support receives loss `12.0`
- prefix-only affine GR calibration: slope clipped to `[0.25, 2.5]`, Huber cutoff `1.5`
  residual MAD, scale clipped to `[8, 60]`, and fallback scale `30`
- position-kernel sigma floor: `0.35 x offset step`; truncate at `4 sigma` with a minimum
  radius of `2` offset cells
- corridor expansion threshold: posterior edge mass `> 0.01`; diagnostics aggregate the outer
  `3` position cells and outer `2` rate cells on each side
- hidden-test parent path: v006 `pf_scale_12_hold_0p2` with `500` particles and `32` seeds; training
  uses the already verified, aligned v006 OOF stream
- per-step normalized float32 forward lattice; streaming backward posterior moments

The fixed core grid contains exactly two candidates:

| Candidate | momentum | rate noise | position noise | start sigma | initial-rate sigma |
|---|---:|---:|---:|---:|---:|
| `hmm_wr21_slow` | 0.998 | 0.002 | 0.175 | 0.75 | 0.01 |
| `hmm_wr21_flex` | 0.998 | 0.004 | 0.35 | 0.75 | 0.02 |

For each canonical outer fold, pooled SSE on only the other four folds selects one candidate; that
candidate is then scored on the held-out fold. The final hidden-test configuration may be selected
using all training wells only after nested OOF evaluation. V007 does not scan hold weights, blend
weights, GBMs, or public-LB selector bins.

## Gate And Runtime Stop

`candidate_ready` requires all of the following:

- nested pooled OOF RMSE `<= 10.95`, at least a 10.1% improvement over v006 `12.184563`
- improvement over the corresponding v006 score in at least four of five canonical folds
- zero training and visible-test fallback wells
- zero unresolved position- or rate-boundary wells after target-free corridor expansion

`promising_continue` requires RMSE `< 11.75` and at least three improved folds. Every other result
is `exhausted`. The target-free smoke uses exactly six wells chosen at fixed quantiles after sorting
all training wells by suffix-row count, runs both candidates with the locked parent settings, and
records a content-based SHA-256 raw-data fingerprint and selected well IDs. It binds both the v007
source and dynamically loaded v006 parent source at the start and end of execution. Full CV starts
only if its two-candidate runtime projection is at most eight hours and peak RSS is at most 8 GiB;
the full run revalidates raw-data and source hashes before completion. Failure to pass the local
gate produces no test candidate, notebook, or submission. This loop never submits to Kaggle.
