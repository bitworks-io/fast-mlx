---
name: context-health-triage
description: Triage a context-health drift report into a prioritized, actionable list. Use after a context-health check (missing baseline files, stale dates, dead absolute paths, oversized instruction files, fallback-model chains) to classify each issue BLOCKING, FAST-FOLLOW, or DEFER before acting on findings.
---

# Context Health Triage

Triage a context-health drift report — covering missing baseline files, a missing Claude bridge, stale handoff/evidence dates, oversized instruction files, dead absolute paths, possible secret patterns, and global fallback-model chains — into an actionable priority list. Use after such a report is produced, before acting on findings. Do NOT apply any fixes.

## Inputs

- **Context health report** — the report you were given.

## Procedure

Classify every issue as exactly one of:

- BLOCKING — must be resolved before the next agent session can proceed safely.
- FAST-FOLLOW — should be resolved soon but does not block immediate progress.
- DEFER — low impact; can be addressed in a future cleanup pass.

For every BLOCKING issue, provide:

- The smallest safe fix (one or two sentences describing what to change and where).
- Why it is blocking (one sentence).

For FAST-FOLLOW and DEFER issues, list them with a one-line description only.

## Constraints

- Do NOT apply any changes yourself.
- Do NOT escalate FAST-FOLLOW issues to BLOCKING without a specific reason.
- If the report contains no issues, state "No issues found" and confirm the report covered the expected checks.
- Produce a triage list only; no prose analysis beyond the per-issue entries.

## Output

Keep output to: a classified issue list with BLOCKING entries expanded (fix + reason) and lower-priority entries one-line each.
