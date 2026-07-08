# Task Inbox

This folder is a durable capture queue for future work. Use it when a feature idea, bug, cleanup, research question, or operational task should be saved for later planning or implementation.

Captured tasks are planning seeds, not implementation plans. Do not expand, prioritize, estimate, or implement a captured item unless the user explicitly asks for planning or execution.

## Capture Rules

- Create one Markdown file per captured task.
- Preserve the user's request verbatim or near-verbatim in `Raw Capture`.
- Add only light triage: type, priority if stated, obvious open questions, and the next planning step.
- Capture the intended user/operator and Desired outcome when those are obvious from the request, but do not invent a full plan.
- Use `status: captured` for new items.
- Use `planning_ready: false` and `implementation_ready: false` unless the user explicitly provides enough detail to change those values.
- If the task spans projects, capture it in the coordinator project or ask which project owns it.
- If GitHub Issues, Linear, Notion, or another external tracker would add value, suggest it separately; still capture locally first unless the user asks to file it externally.

## Suggested Filename

Use a short date-prefixed slug:

```text
YYYY-MM-DD-short-task-title.md
```

## Task Template

```md
---
status: captured
type: feature
priority: unset
created: YYYY-MM-DD
source: user
planning_ready: false
implementation_ready: false
---

# Short Task Title

## Raw Capture

Paste or closely preserve the user's request here.

## Agent Notes

Captured only. Not planned or implemented yet.

## Planning Seed

User/operator:

Desired outcome:

Acceptance signals:
- What observable behavior would prove the task later?
- What automated, browser, simulator, hosted, log, or artifact check could verify it?

Known failure cases:
- What edge, permission, empty-state, or recovery behavior may need attention?

Open questions:
- What decision or context is needed before planning?
- Which project, component, or workflow owns this?
- What verification would prove the work later?

## Next Step

Use this as input to a future planning session.
```
