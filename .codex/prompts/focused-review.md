# Focused Review Prompt Packet

Purpose: focused code review of a bounded change — not the whole codebase.
When to use: after a diff or PR is ready; hand to a reviewer subagent with a precise scope.

```
Review the following change for correctness, security, and logic issues.

Files / diff to review:
{files_or_diff}

Outcome or acceptance criteria this change must satisfy:
{outcome_or_acceptance_criteria}

Instructions:
- Report findings ordered by severity: Critical > High > Medium > Low.
- Each finding must include a file:line reference and a one-sentence reason it matters.
- Do NOT comment on style, formatting, or naming unless it causes a correctness or
  security problem.
- Cap your output to {max_findings} findings total; pick the most impactful ones.
- If you find no issues, state "No issues found" explicitly and give a one-line
  rationale confirming the change satisfies the acceptance criteria.
- Do not suggest unrelated improvements outside the stated scope.
```

Keep output to: a severity-ordered list of findings with file:line references, or a clear "No issues found" statement.
