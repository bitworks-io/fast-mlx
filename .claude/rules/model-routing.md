# Model Routing

- Route by task tier: `haiku` for search, recon, doc lookups, and verification; `sonnet` for well-scoped implementation and tests; `opus` for planning, design, and review judgment; `fable` for root-cause debugging and long autonomous work. The project sub-agents in `.claude/agents/` already pin these — prefer them over ad-hoc `/model` switching.
- Re-route up one tier on failure; do not retry the same task at the same tier.
- Pick model and effort at the start of a session. Mid-session model or effort switches and `/compact` bust the prompt cache (cache reads bill ~10x cheaper than writes); prefer `/rewind` to abandon a path. Sub-agent invocation is cache-safe.

## Fallback

- Judgment-tier work should not silently degrade. Avoid configuring a multi-step `fallbackModel` downgrade chain in personal `~/.claude/settings.json` for judgment work — prefer letting an overloaded premium model retry, or fall back at most one tier, never to the bottom tier.
- `fallbackModel` and `model` are user/global-scope settings; they do NOT merge from project scope, so a project config cannot neutralize a global fallback chain. This is guidance the repo can document, not something it can enforce per-project.
- `CLAUDE_CODE_RETRY_WATCHDOG=1` is an option worth knowing about, but it is experimental and officially undocumented — do not depend on it being stable.
