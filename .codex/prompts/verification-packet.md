# Verification Prompt Packet

Purpose: verify acceptance criteria by running specific commands and mapping results to criteria.
When to use: before closing a task; hand to a verifier subagent with the exact criteria and commands.

```
Verify that the following acceptance criteria are met.

Acceptance criteria:
{acceptance_criteria}

Commands to run (run each exactly as written):
{commands}

For each command:
1. Run it and capture the output.
2. Map the result to the acceptance criterion it proves or disproves.
3. Report: PASS / FAIL / BLOCKED (with reason if blocked or failed).

Rules:
- Do NOT claim a criterion is satisfied if you did not run the command that proves it.
- Do NOT skip commands or substitute equivalent commands without noting the substitution.
- Summarize the key output lines that are evidence for the verdict; do not paste raw logs.
- If an artifact (screenshot, log file, report) was produced, note its path.
- End with a one-line overall verdict: ALL PASS, PARTIAL (list failing criteria), or BLOCKED.
```

Keep output to: a per-criterion pass/fail table, evidence summaries, artifact paths, and a single overall verdict line.
