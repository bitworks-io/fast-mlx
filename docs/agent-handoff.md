# Agent Handoff

Last reviewed: YYYY-MM-DD

This document is intended for Codex, Claude Code, or another implementation agent taking over the project. Keep it current when architecture, workflows, verification commands, deployment/release steps, operational assumptions, security posture, or known risks change.

## Project Purpose

Describe what this project does, who uses it, and the production or operational context that matters.

## Current Feature Set

- List the major user-facing and operator-facing capabilities that exist today.
- Note important feature flags, integrations, platform support, or intentionally unsupported behavior.

## User Stories And Acceptance Evidence

Use this table for durable scenario coverage. Keep it focused on the workflows a user, operator, customer, or maintainer would expect to keep working.

| Story | Acceptance Criteria | Automated Proof | Manual/Smoke Proof | Last Verified | Gaps |
| --- | --- | --- | --- | --- | --- |
| As a user/operator, I need ... | Observable outcome, including key edge or failure behavior. | Command, test, or CI check. | Browser, simulator, hosted smoke, screenshot, log, or artifact check. | YYYY-MM-DD | Known skipped checks or residual risk. |

## Key Components

- `path/or/module`: describe its responsibility and key contracts.
- `path/or/module`: describe external dependencies, data ownership, or operational assumptions.

## Local And Hosted Test Commands

```sh
# Narrow checks first
```

```sh
# Broader checks, browser/simulator/integration checks, or hosted smoke tests
```

## Review Findings

- Summarize recent implementation reviews, acceptance passes, bug fixes, and unresolved observations.
- Include dates and environments when useful.
- Link to or summarize relevant entries from `docs/verification-evidence.md` when they explain what was actually proven.

## Security Posture

- Record auth, permission, input-validation, secrets, telemetry, dependency, and data-retention considerations.
- Note known threat model boundaries and checks that must be rerun for risky changes.

## Operational Notes

- Document recurring jobs, queues, background workers, observability, deployment assumptions, rollback steps, and local-only escape hatches.
- Call out files that are intentionally local, generated, or excluded from release artifacts.

## Context Handoff Hygiene

- Keep durable context short and decision-focused: current architecture, contracts, operational assumptions, verification commands, known risks, and unresolved blockers.
- Link to detailed logs, screenshots, traces, reports, or release artifacts instead of pasting bulky output.
- Move exploratory notes, abandoned approaches, and raw command output out of this document unless they explain a current risk or decision.
- When a chat grows long, summarize only the accepted decisions, changed files, verification evidence, and remaining work needed for the next agent to continue safely.

## Commit And Release Guidance

- Document version bump rules, generated artifacts, release packaging, deployment smoke tests, and checkin boundaries.
- Note any multi-repo coordination rules or files that should not be committed.
