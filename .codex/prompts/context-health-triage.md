# Context Health Triage Prompt Packet

Purpose: triage the output of scripts/context-health.sh into an actionable priority list.
When to use: after running context-health.sh; hand to a triage subagent before acting on findings.

```
Triage the following context-health report. Classify every issue. Do NOT apply any fixes.

Context health report:
{context_health_report}

Classify each issue as exactly one of:
- BLOCKING   — must be resolved before the next agent session can proceed safely.
- FAST-FOLLOW — should be resolved soon but does not block immediate progress.
- DEFER      — low impact; can be addressed in a future cleanup pass.

For every BLOCKING issue, provide:
- The smallest safe fix (one or two sentences describing what to change and where).
- Why it is blocking (one sentence).

For FAST-FOLLOW and DEFER issues, list them with a one-line description only.

Rules:
- Do NOT apply any changes yourself.
- Do NOT escalate FAST-FOLLOW issues to BLOCKING without a specific reason.
- If the report contains no issues, state "No issues found" and confirm the report
  covered the expected checks.
- Produce a triage list only; no prose analysis beyond the per-issue entries.
```

Keep output to: a classified issue list with BLOCKING entries expanded (fix + reason) and lower-priority entries one-line each.
