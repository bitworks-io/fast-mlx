@AGENTS.md

AGENTS.md holds the shared Codex methodology; this file adds Claude-Code-specific session guidance only.

## Claude Code Session Hygiene

- Use `/clear` between unrelated tasks; keep each session focused on one outcome.
- When context grows large, compact or hand off: preserve only accepted decisions, changed files, verification evidence, and open blockers — see `docs/agent-handoff.md` for the durable-state format.
- Delegate verbose, bounded work (exploration, log reading, doc lookups) to scoped subagents on a cheap model; keep the main thread as the integrating owner.
- Store bulky logs, traces, and generated artifacts in files and link them — do not paste them into the transcript.
- Path-specific guidance lives in `.claude/rules/`; consult it rather than expanding this file.

## Model Routing

Route work by task tier (haiku/sonnet/opus/fable) and prefer pinned sub-agents over ad-hoc `/model` switching — see `.claude/rules/model-routing.md` for the full policy.

Scoped rules live in `.claude/rules/`; keep this file thin.
