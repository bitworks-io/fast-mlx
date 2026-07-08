---
name: session-handoff
description: Produce a concise session handoff document for the next agent or session. Use at a clean stopping point to capture objective, current state, accepted decisions, changed files, verification results, blockers, and the next safe action.
---

# Session Handoff

Produce a session handoff document for the incoming agent, at a clean stopping point. Fill every section. Keep the total under ~400 words. Link artifacts by path; do not paste raw logs.

## Inputs

- The session's context: what was attempted, what was decided, what changed, what was verified, and what remains open.

## Procedure

Produce the handoff document with the following structure:

- **Objective** — One sentence: what this session was trying to accomplish.
- **Current State** — Where things stand right now: what works, what is partial, what is broken.
- **Accepted Decisions** — Bullet list of design or implementation decisions that are settled and should not be re-litigated without a specific reason. Include brief rationale for each.
- **Changed Files** — List of files modified, created, or deleted. Format: path — one-line description of what changed and why.
- **Verification Run + Result** — Command(s) run, pass/fail/blocked verdict per criterion, path to any artifacts. No raw log output.
- **Blockers / Residual Risk** — What is unresolved or risky. If nothing, write "None identified."
- **Next Safe Action** — The single next step the incoming agent should take, phrased as an action.

## Constraints

- Fill every section; do not leave placeholders.
- Keep the total under ~400 words.
- Link artifacts by path; do not paste raw logs.

## Output

Keep output to: a filled handoff document under ~400 words with artifact links and no raw logs.
