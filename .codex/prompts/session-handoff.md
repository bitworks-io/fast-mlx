# Session Handoff Prompt Packet

Purpose: manual handoff packet for transferring work to another agent or session.
When to use: at a clean stopping point; mirrors the output of scripts/session-handoff.sh.

```
Produce a session handoff document for the incoming agent. Fill every section.
Keep the total under ~400 words. Link artifacts by path; do not paste raw logs.

---

## Objective
{objective}
<!-- One sentence: what this session was trying to accomplish. -->

## Current State
{current_state}
<!-- Where things stand right now: what works, what is partial, what is broken. -->

## Accepted Decisions
{accepted_decisions}
<!-- Bullet list of design or implementation decisions that are settled and should not
     be re-litigated without a specific reason. Include brief rationale for each. -->

## Changed Files
{changed_files}
<!-- List of files modified, created, or deleted. Format: path — one-line description
     of what changed and why. -->

## Verification Run + Result
{verification_run_and_result}
<!-- Command(s) run, pass/fail/blocked verdict per criterion, path to any artifacts.
     No raw log output. -->

## Blockers / Residual Risk
{blockers_and_residual_risk}
<!-- What is unresolved or risky. If nothing, write "None identified." -->

## Next Safe Action
{next_safe_action}
<!-- The single next step the incoming agent should take, phrased as an action. -->

---
```

Keep output to: a filled handoff document under ~400 words with artifact links and no raw logs.
