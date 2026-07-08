# Codex Project Profile

Project: /Users/braymond/Projects/fast-mlx
Trust mode: full-auto
Codex profile: full-auto-local
Project type: ai-llm-service

## Recommended Plugins

OpenAI Developers, Hugging Face, GitHub, Browser, Spreadsheets, Documents, Superpowers; optional Vercel, Render, Supabase, Neon Postgres, Sentry

## Optional Capability Candidates

Use OpenAI Developers for current API, Agents SDK, and ChatGPT Apps guidance; Hugging Face for model/dataset research; Sentry for runtime failures; Supabase/Neon for eval or app data; Vercel/Render for deployment.

## MCP and CLI Candidates

Context7 for current SDK/library docs, GitHub MCP for repo metadata, promptfoo or OpenAI eval tooling for regression evals, LiteLLM or provider CLIs only when the project intentionally uses them, Semgrep and secret scanners for prompt/tooling surfaces.

## Methodology

This project follows the shared methodology in `AGENTS.md` (auto-loaded as the Codex project doc and imported by `CLAUDE.md`): user-story and acceptance discipline, context hygiene and delegation, capability discovery, methodology skills, project documentation, and the task inbox. Reusable delegation prompt packets live in `.codex/prompts/`.

Two safety reminders that hold even when only this profile is read:
- Do not silently install, enable, authenticate, or connect account-level services; recommend, then ask.
- Verification must map back to acceptance criteria — do not call work complete by listing commands alone.

## Expected Workflow

Prompt/version tracking, evals, API contract tests, cost/latency/quality benchmarks, regression datasets.

## Verification

Unit and contract tests, eval suite, latency/cost report, prompt regression checks, safety and abuse-case review.

## Automation Candidates

- GitHub: PR review, CI failure triage, issue creation, release readiness.
- Slack or Teams: monitor operational channels, summarize incidents, turn threads into tracked tasks.
- Gmail or Outlook Email: monitor approved inboxes, extract action items, draft replies for review.
- Google Calendar or Outlook Calendar: daily briefs, meeting prep, follow-up reminders.
- Linear: convert repeatable findings into tracked engineering work.
- Notion, Google Drive, or SharePoint: maintain runbooks, specs, discovery logs, and decision records.
- Browser: periodic website/app/admin smoke checks and evidence capture.

## Automation Boundary

Account-level connectors such as email, chat, calendar, Drive, SharePoint, Notion, and Linear should be enabled deliberately per workspace. Do not silently connect them from project bootstrap.
