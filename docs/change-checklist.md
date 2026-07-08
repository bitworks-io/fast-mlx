# Change Checklist

Use this before considering a Codex-assisted change complete.

## Plan

- [ ] Problem statement is clear.
- [ ] Primary user or operator story is stated.
- [ ] Acceptance criteria are listed in observable terms.
- [ ] Scope and non-goals are explicit.
- [ ] Risks, rollback, and affected surfaces are named for production-facing changes.
- [ ] Test strategy is chosen before implementation.
- [ ] Verification maps each acceptance criterion to an automated or manual check.

## Implement

- [ ] Changes are scoped to the requested behavior.
- [ ] Existing patterns and ownership boundaries are respected.
- [ ] Public contracts, config keys, or behavior changes are documented.
- [ ] No secrets, local cache paths, tokens, or transient artifacts are included.

## Verify

- [ ] Verification results map back to the stated acceptance criteria.
- [ ] Focused tests pass.
- [ ] Broader tests pass when shared code or infrastructure changed.
- [ ] Coverage command and result are recorded when coverage tooling exists.
- [ ] At least one representative happy path is exercised.
- [ ] Relevant failure, edge, permission, empty-state, or recovery paths are exercised or explicitly deferred.
- [ ] Manual verification is documented when automated coverage is insufficient.
- [ ] Browser, simulator, hosted smoke, screenshot, log, or artifact evidence is captured when it is the meaningful proof of user-facing behavior.

## Check In

- [ ] Diff reviewed.
- [ ] Commit is focused.
- [ ] Commit message describes the outcome.
- [ ] Generated artifacts required by the change are included.
- [ ] Known gaps or follow-up work are documented.
