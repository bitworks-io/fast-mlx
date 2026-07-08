# AGENTS.md

This repository captures the Codex working environment and the practices expected around it.

## Operating Mode

- Begin non-trivial work by reading local context and stating a short plan.
- Drive work end to end: plan, research, design review, implement, test, debug, security review, documentation, coverage, diff review, and checkin.
- Treat configuration as production infrastructure: keep it portable, reviewed, and documented.
- Prefer small, intentional changes that can be inspected and reverted.
- Use high reasoning for risky work, including auth, data migrations, production infrastructure, concurrency, complex refactors, unclear failures, security-sensitive behavior, or changes with broad blast radius.
- Do not commit secrets, tokens, transient logs, absolute runtime cache paths, or machine-local state unless the file is explicitly documented as local.
- Before sharing or publishing artifacts, run a secret scanner (for example gitleaks or trufflehog) when one is available and confirm it is clean.
- Preserve user-authored changes. If the tree is dirty, work with existing changes rather than reverting them.

## User Story And Acceptance Discipline

- For feature, bugfix, workflow, UI, API, integration, or automation work, state the Target user or operator story before editing: who needs the change, what they should be able to do, and why the outcome matters.
- Distill the requested outcome into acceptance criteria that are observable from the product, API, CLI, admin surface, automation, logs, or generated artifact.
- Name at least one representative happy path and the relevant failure, edge, permission, empty-state, or recovery paths. If a path is intentionally out of scope, say so before implementation.
- Choose proof methods before implementation. Prefer automated tests for core behavior and add browser, simulator, CLI, hosted smoke, screenshot, log, or artifact checks where those are the only meaningful proof.
- Verification must map back to acceptance criteria. Do not call work complete by listing commands alone; state which user story or acceptance criterion each command or manual check proves.
- If the expected outcome is ambiguous, ask enough to clarify it or state the assumption being implemented. Do not substitute a technically convenient behavior for the user's intended workflow.

## Multi-Agent Workflow

### Parallel Agent Default

For this project, the user authorizes Codex and Claude Code to use parallel sub-agents by default when work splits into genuinely independent scopes — most naturally read-only ones. Fan-out pays off for separable work; it is a poor fit for tightly-coupled changes, where sub-agents that share files, types, or invariants duplicate effort and multiply token cost for little gain. Parallelize the separable parts and perform coupled edits serially under one owner.

- Parallelize freely for read-only or separable work: research, codebase exploration, documentation lookup, review, testing, security analysis, and release-readiness checks.
- Perform coupled implementation serially under one integrating owner. Delegate code changes in parallel only when the write areas are genuinely disjoint.
- Keep one owner responsible for integration and final judgment.
- Give agents bounded scopes, explicit outputs, and disjoint write areas when code changes are delegated.
- Treat security and architecture reviews as first-class work for production-facing changes, not optional cleanup.
- Prefer read-only review mode for unfamiliar or not-yet-trusted codebases until the workspace has been inspected.

### Context Hygiene

- Keep the main chat as the integrating owner; use sub-agents for bounded sidecar work that can return compact findings or scoped patches.
- Delegate with a narrow context packet: objective, relevant files or paths, constraints, expected output, and verification needs. Do not fork full chat context unless the sub-agent truly needs it.
- Prefer project-scoped sub-agents when available — `.codex/agents/` for Codex and the mirrored `.claude/agents/` for Claude Code: `codebase_explorer`, `docs_researcher`, `implementation_worker`, `reviewer`, and `verifier`. Brief them with the reusable prompt packets in `.codex/prompts/` (codebase-exploration, docs-source-review, focused-review, verification-packet, context-health-triage, session-handoff).
- Ask sub-agents for concise file-referenced output. Put long logs, screenshots, traces, and generated reports in durable files or artifacts instead of the main transcript.
- Close completed sub-agents and summarize only accepted findings, changed files, verification evidence, and remaining risks in the main chat.
- Do not treat chat history as a source of truth; durable state lives in `docs/agent-handoff.md` and `docs/verification-evidence.md`.

## Capability Discovery

- Proactively suggest missing official or third-party plugins, MCP servers, skills, connectors, browser tools, design tools, test runners, security scanners, observability tools, or domain-specific CLIs that would materially improve planning, implementation, testing, review, release, or operations.
- Do not assume the user already knows which capabilities can or should be enabled; call out concrete benefits, setup cost, trust/security implications, and whether the capability is optional or important for the current work.
- Credential or account requirements are not a reason to omit a useful recommendation; they are a reason to clearly label setup, approval, data exposure, and trust implications before enabling the capability.
- Multi-account git identity (pinning each repo to the correct GitHub account via SSH host aliases so pushes never depend on the active `gh` account) follows `docs/git-identity.md`. Machine-specific account rosters, key fingerprints, and repo mappings live in the private overlay (`$CODEX_ENV_PRIVATE_OVERLAY`), never in this public baseline.
- Prefer project-relevant recommendations over broad lists. Tie suggestions to the project type, tech stack, files present in the repo, current task, and observed workflow gaps.
- Do not silently install, enable, authenticate, or connect account-level services. Ask before enabling connectors, cloud services, third-party plugins, tools that require credentials, or tools that send code, telemetry, prompts, repository data, or user data outside the local machine.
- When a capability is useful but unavailable, name the capability, explain why it helps, and give the smallest safe next step to evaluate or install it.
- Claude Code sessions also auto-discover the Agent Skills in `.claude/skills/`, alongside the sub-agent pointers above; no separate invocation step is needed.

## Engineering Standards

- Use existing project patterns before inventing new ones.
- Keep public behavior, config keys, and operational assumptions documented.
- Add comments only where they clarify intent, invariants, risk, or non-obvious tradeoffs.
- Favor boring, reliable implementation choices over cleverness.
- For security-sensitive or production-facing changes, call out threat model, blast radius, rollback, and verification.

## Methodology Skills

- Automatically use relevant installed methodology skills or plugins, such as Superpowers, when they materially improve planning, implementation, debugging, testing, review, or release readiness.
- Use methodology skills without waiting for the user to name them for ambiguous or multi-step feature design, complex implementation plans, bugs, test failures, flaky behavior, build failures, behavior changes needing regression protection, cross-repo/API contract work, data migrations, auth or permissions work, security-sensitive changes, release readiness, and code review preparation or response.
- When Superpowers is available, prefer these mappings:
  - systematic debugging for bugs, failures, and unexpected behavior;
  - verification-before-completion before any completion claim;
  - test-driven-development for behavioral code changes and bug fixes where a meaningful automated test can be written first;
  - brainstorming and writing-plans for ambiguous, high-impact, or multi-component feature work;
  - dispatching-parallel-agents or subagent-driven-development for planned work that can be split into bounded independent scopes;
  - requesting-code-review and receiving-code-review for major changes, risky changes, PR prep, or external review feedback.
- Do not let methodology skills override direct user instructions, project-specific AGENTS.md guidance, trust mode, security rules, connector approval boundaries, or pragmatic execution for low-risk mechanical changes.
- For low-risk config, docs, generated files, formatting, dependency metadata, or one-line mechanical fixes, apply the relevant principles without forcing a full methodology workflow unless risk or ambiguity justifies it.

## Debugging

- Do not patch symptoms before identifying the root cause.
- For bugs, test failures, flaky behavior, build failures, performance problems, integration issues, or unexpected behavior, first reproduce the issue, read the complete error output, inspect recent changes, compare with working examples, and state the root-cause hypothesis before fixing.
- Trace failures across component boundaries with evidence: inputs, outputs, config, environment, and persisted state.
- Test one hypothesis at a time. Avoid bundled speculative fixes.
- If three fix attempts fail, or each fix exposes a different class of problem, stop and reassess the architecture or ask for direction instead of continuing to stack patches.

## Verification Before Completion

- Do not claim work is complete, fixed, passing, ready, deployed, pushed, or released without fresh verification evidence from the current turn.
- Identify the command or manual check that proves the claim, run it, read the output, and report the result.
- If verification is skipped, blocked, partial, stale, or impossible in the current environment, say that explicitly and explain the remaining risk.
- Do not trust sub-agent success reports without reviewing changed files and running or checking the relevant verification yourself.

## Test-First Behavioral Changes

- For bug fixes and behavior changes, prefer writing or updating a failing test before changing production code.
- Verify the test fails for the expected reason, make the smallest implementation change, then verify it passes.
- Test-first development is expected for core logic, contracts, parsers, APIs, data transforms, auth and permissions, migrations, and regressions.
- If a test-first approach is impractical, document why and use the narrowest meaningful alternate verification.
- TDD is optional for docs-only edits, config-only edits, generated files, scaffolding, formatting, exploratory spikes, and visual polish unless risk or ambiguity justifies it.

## Code Review Discipline

- Request or perform focused code review for major changes, risky changes, cross-repo/API work, release readiness, and before merging or pushing production-facing work.
- Treat external or sub-agent review feedback as technical input to verify, not orders to apply blindly.
- Clarify unclear review items before implementing them.
- Fix critical and important review findings before proceeding, or document why a finding is intentionally rejected with codebase-specific evidence.

## Project Documentation

- Maintain a durable project handoff document for future agents, preferably `docs/agent-handoff.md` unless the project already has an equivalent current-state document.
- Keep the handoff document current when architecture, workflows, verification commands, deployment/release steps, operational assumptions, security posture, or known risks change.
- Capture enough context for a new Codex, Claude Code, or human maintainer to resume safely: purpose, current feature set, key components, test commands, review findings, security notes, operational notes, and release/checkin guidance.
- Do not duplicate every code detail. Focus on decisions, contracts, invariants, external dependencies, and verification evidence that would otherwise be rediscovered.
- Keep handoff and evidence docs concise. Link bulky artifacts instead of pasting raw logs or transient output.

## Private Project Guides

- Support optional local private overlays through `CODEX_ENV_PRIVATE_OVERLAY` for project-type-specific implementation guides that should not be committed to this public baseline.
- Keep private overlay content outside this repository, preferably under `~/.codex/private-overlays/codex-env/`.
- Use the gitignored repository-root `project-init.local` file for machine-specific `scripts/project-init.sh` customizations such as `CODEX_ENV_PRIVATE_OVERLAY`; never commit it.
- Treat private guides copied into a target project as project-local implementation guidance. Follow them when relevant, but do not assume the public baseline repository contains or should contain their private content.
- Do not commit private overlay source directories, secrets, credentials, customer-specific host details, or environment-specific implementation playbooks to this public repository.

## Task Inbox

- Use `docs/task-inbox/` as the durable local capture queue for future work that should not be planned or implemented immediately.
- When the user asks to capture, backlog, save, remember, queue, or record a future task, create one dated Markdown file in `docs/task-inbox/` and preserve the raw request verbatim or near-verbatim.
- Treat captured tasks as planning seeds, not implementation plans. Do not expand scope, prioritize, estimate, plan, or implement the task unless the user explicitly asks for planning or execution.
- Add only light triage: status, task type, priority if stated, obvious open questions, owner/project if clear, and the next planning step.
- If a task spans projects, capture it in the coordinator project when one exists; otherwise ask which project owns it.
- If GitHub Issues, Linear, Notion, or another external tracker would add value, suggest it separately. Still capture locally first unless the user asks to file it externally.

## Testing

- Add or update tests for behavioral changes.
- Run the narrowest meaningful test first, then broader suites when shared contracts or infrastructure are touched.
- Report what was run, what was not run, and why.
- For functional work, include at least one acceptance-oriented check that exercises the user or operator outcome, not only isolated implementation details.
- When coverage tooling exists, include the coverage command and measured result for meaningful changes.
- Do not mark work complete while known failing tests remain unexplained.

## Definition Of Done

- The plan has been executed or deliberately revised.
- User stories, acceptance criteria, and proof methods have been stated for functional work.
- Relevant research and architectural tradeoffs are captured in the final explanation or docs.
- Code, tests, docs, security posture, and operational implications have been checked.
- Automated verification and any manual checks are reported.
- Residual risks, skipped checks, and follow-up work are explicit.

## Checkins

- Inspect the diff before committing.
- Keep commits focused on one coherent change.
- Use direct commit messages that explain the behavioral or operational outcome.
- Include generated lockfiles, schemas, snapshots, or docs in the same change when they are required by the source change.
- Avoid formatting-only churn mixed into functional commits unless the formatter is required for that change.
