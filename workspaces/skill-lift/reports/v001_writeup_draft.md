# Artifact-First Execution: A Safety-Gated Static Skill

## Summary

This Track 1 submission tests a narrow hypothesis: agents fail many structured tasks not because
they lack domain knowledge, but because they lose the output contract between inspection,
implementation, and final validation. `artifact-first-execution` makes that contract explicit while
prohibiting grader manipulation and unsafe shortcuts.

## Skill Design

The skill uses five stages:

1. Extract required paths, schemas, types, ordering rules, and semantic constraints.
2. Inspect source files through structured parsers without modifying inputs.
3. Produce deterministic outputs through established libraries and temporary files.
4. Reopen outputs independently and check task-specific invariants.
5. Finalize only exact requested artifacts and report the checks performed.

The skill includes a read-only audit script for CSV, TSV, JSON, XLSX, DOCX, and OBJ files. The script
is intentionally structural. It detects malformed files, inconsistent rows, unresolved placeholders,
invalid document archives, and broken OBJ indices, while leaving domain verification to the agent.

## Safety And Generalization

The skill explicitly forbids inspecting or modifying verifiers, oracle solutions, answer keys, and
hidden tests. It also forbids external transmission of local data, destructive source edits,
permission escalation, and sandbox escape. These constraints are part of the skill procedure rather
than an afterthought because private safety tasks can make otherwise capable skills harmful.

The skill contains no public task names, fixed answers, or task fingerprints. Its trigger is based on
the presence of an explicit local artifact contract, making it applicable across office documents,
data analysis, optimization, and media conversion tasks.

## Reproducibility Status

The public task packages were pinned to SkillsBench commit
`44cdda48f6e8c4d381f4f5075c0f4a051ba69e98`. Five sparse-checkout tasks passed structural checks,
and the `offer-letter-generator` oracle passed its verifier. The skill and audit script passed local
validation, compilation, lint, and positive/negative fixture tests.

`gpt-5.5` paired runs covered offer-letter generation, lake attribution, and Three.js conversion.
The valid pairs scored `1.0 -> 1.0`, `0.0 -> 0.0`, and `1.0 -> 1.0`, respectively. The resulting
mean lift is `0.0`, with no observed regressions. Two XLSX experiments were excluded because the
public verifier's missing Science average conflicts with its labeled period and completed peer
averages. The skill variants retained the workbook-consistent value instead of encoding a
verifier-specific answer. Total model usage was 1,748,452 input-plus-output tokens.

## Evaluation Decision

No version advances to submission because none demonstrated positive valid lift. The next iteration
should replace the broad skill with narrowly triggered format or domain procedures, preflight each
verifier before paid runs, and reserve enough budget for complete paired trials. This draft is an
experiment record, not a submission-ready writeup.
