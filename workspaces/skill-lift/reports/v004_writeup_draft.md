# FJSP Repair Scheduler: Deterministic Baseline Repair Under Disruptions

## Summary

This Track 1 submission contains one narrowly triggered static skill: `fjsp-repair-scheduler`. It
targets flexible job-shop scheduling tasks where an existing production plan must be repaired after
machine downtime while preserving job precedence, frozen work, machine capacity, policy change
budgets, and a makespan guard.

The design follows a simple principle: use the language model to identify the scheduling contract
and invoke the right tool, then use deterministic code for the combinatorial search and constraint
validation. The skill contains no public task names, fixed operation choices, instance values,
expected scores, or verifier-specific behavior.

## Method

The bundled pure-Python solver accepts four explicit inputs: an FJSP instance, a baseline schedule,
downtime windows, and a policy file. It parses alternative machine-duration routes and enumerates
routing assignments up to a declared cap. For each assignment, it processes operations in a stable
precedence-aware order and scans integer time from the right-shift anchor until it finds the earliest
half-open interval that avoids scheduled work and downtime.

Each candidate is independently checked for:

- valid machine-specific duration;
- job precedence and machine non-overlap;
- downtime avoidance;
- completed frozen operations;
- no starts earlier than baseline;
- machine-change and start-shift L1 budgets;
- the policy makespan ratio;
- exact agreement between JSON and CSV operation rows.

Feasible candidates are selected lexicographically by makespan, machine changes, start-shift L1,
and deterministic row order. The solver reports whether assignment enumeration completed and
refuses to claim a solution when validation fails.

## Local Evaluation

Evaluation used `gpt-5.5` with `xhigh` reasoning in isolated Docker containers. The verifier was not
present during agent execution and was copied in only after the model exited.

On `manufacturing-fjsp-optimization`, the no-skill agent passed 13 of 15 checks but moved one
operation earlier than its baseline anchor. Its binary task score was `0.0`. The skill-enabled agent
loaded the submitted skill, ran the deterministic solver, completed all 1,024 routing combinations,
and passed 15 of 15 checks for score `1.0`, producing paired lift `+1.0`.

The skill-enabled condition was repeated three times and passed all three trials. Two repeats reused
the recorded no-skill baseline to reduce model cost, so only the first run is a fully independent
paired trial. As a trigger regression check, the candidate was mounted for `offer-letter-generator`.
The FJSP skill did not trigger and the task retained its `1.0` score, for lift `0.0`.

Across the primary and regression tasks, mean local lift is `0.5`. The narrow trigger produced no
observed regression. These results are development evidence, not a claim that the same lift will
hold for the private task distribution.

## Safety And Generalization

The skill explicitly forbids reading verifier code, oracle outputs, hidden tests, or grader state.
Its script performs no network calls and only reads paths supplied by the agent. Output is written
atomically to the requested directory. Unsupported inputs, truncated search, and infeasible policy
sets are surfaced rather than hidden.

The candidate focuses on a reusable domain procedure rather than recognizing a benchmark task. Its
main limitation is search scale: complete enumeration is appropriate for small routing spaces, while
large instances require a higher cap or a more advanced optimizer. The script reports this boundary
through `search_complete` so downstream agents cannot mistake a capped search for exhaustive proof.
