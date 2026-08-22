# An exact prefix cache needs exact provenance

An inference cache can return the right bytes and still report the wrong reason.

fast-mlx's first loaded Qwen3-32B proof exposed that distinction. The runtime correctly chose the
longest reusable state for a multi-turn request. That state was the prior successful
**final context**—prompt plus generated continuation—not merely the shorter source prompt. The
proof format assumed prompt-only reuse, so it rejected a correct runtime decision.

The fix was not to shorten the lookup or weaken validation. fast-mlx evidence schema 3 now records:

- the source proof case;
- whether the snapshot is prompt-only or final-context;
- the exact reused token count;
- the SHA-256 of those exact tokens.

That turns “cache hit” from a vague label into a checkable claim about which state was restored.

## What the fresh proof showed

On the authenticated Apple M5 Max boundary, both retained model snapshots completed the same
11-case sequence: cold controls, cold commit, exact hit, partial tail, A/B/A return, pressure
eviction, eager warmup, and template/token reuse.

| Model | Exact-hit request start | Partial-hit request start | Result |
| --- | ---: | ---: | --- |
| Qwen3-32B-4bit | 76.67% faster than cold | 29.28% faster than control | model-scoped pass |
| Phi-4-mini-instruct-4bit | 36.14% faster than cold | 4.88% faster than control | model-scoped pass |

Every retained cache-on output matched its cache-off temperature-zero token and byte hashes.
Exact hits physically prefetched zero prompt tokens. Partial hits prefetched only their uncached
tails. Both runs stayed inside explicit entry, retained-byte, MLX cache, peak, and sampled-process
limits.

The Qwen partial case is the useful detail: it reused the prior 202-token final context, then
physically evaluated the remaining 209 tokens of a 411-token prompt. The proof binds that decision
to the source case's exact final-context hash.

## The negative result matters too

The source-locked Llama-3.3-70B snapshot did not pass on the same 128-GiB machine. Its diagnostic
rows were byte-identical and faster on warm starts, but the process exceeded the declared proof
memory bound. Full checkpoint revalidation then pushed process footprint beyond host RAM, and
macOS terminated the run before finalization.

That is not a near-pass. It is a model-and-hardware-specific rejection. fast-mlx preserves the
failed boundary and makes no Llama-70B cache claim from its diagnostic rows.

## What is—and is not—ready

The exact-prefix engine path is now model-scoped for the authenticated Qwen3-32B and Phi-4-mini
snapshots. It remains disabled by default and scalar-only.

This result does not authorize:

- a production OpenAI-compatible serving switch;
- continuous-batch slot restore;
- compressed, sliding, recurrent, hybrid, vision, or speculative state;
- SSD persistence;
- a broad “all models” default.

Those are separate ownership, privacy, memory, and failure-recovery problems. The next fast-mlx
gate is the production continuous-batching serving route, where the scalar snapshot contract must
survive slot admission, cancellation, disconnects, stale plans, and hostile compaction without
weakening the exactness proved here.

The durable technical verdict and machine-readable evidence mapping live in
[`2026-07-24-exact-prefix-session-cache.md`](../superpowers/verdicts/2026-07-24-exact-prefix-session-cache.md).
