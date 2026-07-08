---
name: verification-packet
description: Verify acceptance criteria by running specific commands and mapping each result back to the criterion it proves. Use before closing a task to produce a per-criterion PASS/FAIL/BLOCKED verdict with evidence.
---

# Verification Packet

Verify that acceptance criteria are met by running specific commands and mapping results back to the criteria. Use before closing a task, with the exact criteria and commands.

## Inputs

- **Acceptance criteria** — the criteria you were given.
- **Commands to run** — the exact commands you were given; run each one exactly as written.

## Procedure

For each command:

1. Run it and capture the output.
2. Map the result to the acceptance criterion it proves or disproves.
3. Report: PASS / FAIL / BLOCKED (with reason if blocked or failed).

## Constraints

- Do NOT claim a criterion is satisfied if you did not run the command that proves it.
- Do NOT skip commands or substitute equivalent commands without noting the substitution.
- Summarize the key output lines that are evidence for the verdict; do not paste raw logs.
- If an artifact (screenshot, log file, report) was produced, note its path.

## Output

- A per-criterion pass/fail table with evidence summaries and artifact paths.
- End with a one-line overall verdict: ALL PASS, PARTIAL (list failing criteria), or BLOCKED.

Keep output to: a per-criterion pass/fail table, evidence summaries, artifact paths, and a single overall verdict line.
