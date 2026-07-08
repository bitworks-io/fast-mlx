---
name: focused-review
description: Focused code review of a bounded change or diff — not the whole codebase. Use after a diff or PR is ready to surface correctness, security, and logic issues ordered by severity, capped to the most impactful findings.
---

# Focused Review

Focused code review of a bounded change — not the whole codebase. Use after a diff or PR is ready, with a precise scope.

## Inputs

- **Files / diff to review** — the files or diff you were given.
- **Outcome or acceptance criteria this change must satisfy** — the criteria you were given.
- **Max findings** — the cap on the number of findings you were given (if none given, use judgment to keep the list tight).

## Procedure

Review the change for correctness, security, and logic issues.

- Report findings ordered by severity: Critical > High > Medium > Low.
- Each finding must include a file:line reference and a one-sentence reason it matters.
- Cap output to the max findings total; pick the most impactful ones.
- If you find no issues, state "No issues found" explicitly and give a one-line rationale confirming the change satisfies the acceptance criteria.

## Constraints

- Do NOT comment on style, formatting, or naming unless it causes a correctness or security problem.
- Do not suggest unrelated improvements outside the stated scope.

## Output

Keep output to: a severity-ordered list of findings with file:line references, or a clear "No issues found" statement.
