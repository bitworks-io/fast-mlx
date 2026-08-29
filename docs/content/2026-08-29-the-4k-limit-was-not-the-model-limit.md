# The 4K limit was not the model limit

**Whitepaper themes:** OpenAI-compatible serving; Long-horizon inference; Fail-closed capacity

A completion limit can look like a model fact when it is really only a server default.

fast-mlx used to apply one flat 4,096-token ceiling while decoding every chat-completions request.
That was conservative, but it also meant a model and host that could safely serve a much longer
answer still returned HTTP 400 when a client asked for more. Raising the one global number moved the
failure in the other direction: a short-context model or a memory-capped deployment could accept a
request it could not safely finish.

The fix was not a larger constant. It was one model-aware capability shared by the whole service.

## Two context limits answer different questions

The loaded checkpoint supplies its authenticated native context window. The pre-load capacity
planner supplies the context this host and runtime configuration can actually serve. fast-mlx keeps
both:

- `native_max_context_tokens` describes the checkpoint;
- `effective_max_context_tokens` describes this admitted deployment; and
- `maximum_completion_tokens` is at most the effective context minus one prompt token.

An operator may narrow that maximum, but an option cannot widen it beyond the model or the admitted
host fit. This matters for model families with very different native windows and cache geometries:
the server no longer needs a table of model names or a universal completion ceiling.

## The exact rendered prompt decides the final budget

Raw message text is not the prompt the model sees. Chat templates add role markers, tool schemas,
reasoning controls, and generation markers. Each backend therefore renders with its own loaded
tokenizer first, counts the exact resulting tokens, and applies:

```text
allowed completion = min(operator/model maximum, effective context - rendered prompt)
```

The default requested budget remains 4,096 for compatibility. It is a default, not the maximum. A
client can request the full discovered budget, and an omitted request safely shrinks when a very long
prompt leaves less room.

The default policy rejects an explicit over-budget request with a typed 400. Operators that need
compatibility with clients that over-request can select deterministic clamping instead. Response
headers report the requested, applied, maximum-allowed, clamped, and policy values.

## Long output uses streaming without reducing the model limit

The full model-aware maximum is available to streaming requests. Non-streaming output is collected
in memory, so it has a separate configurable token ceiling, 16,384 by default, plus an independent
byte guard. A larger admitted request fails before generation with `stream_required`; setting
`stream=true` preserves the full completion budget without accumulating the complete answer in RAM.

This separates a transport-memory decision from a model-context decision. An operator can raise the
non-streaming ceiling when a client requires it without lowering the maximum available to long-lived
streaming agents.

Long input has a separate wire-size concern. A fixed one-megabyte HTTP body would still reject some
valid long-context prompts before the tokenizer could measure them. The loaded service therefore
derives its request-body cap from the admitted context, with conservative one- and 64-megabyte
bounds, and lets an operator override the byte value for unusually dense tool schemas or deployment
proxies. The non-streaming response byte cap is independently configurable. Both effective byte
limits are discoverable, tool-call arguments count against the same bounded mailbox and response
collector as text, and the final encoded JSON body is checked before it is written.

## Route changes cannot create tokens

Scalar generation, continuous batching, exact MTP, and the exact-MTP scalar fallback all receive the
same immutable capability object. Validation happens after exact prompt rendering but before queue,
scheduler, reservation, cache, runner, or random-state mutation.

If exact MTP resolves a budget and later falls back, the fallback must re-render and prove the same
resolution before it can run. A batch join, route transition, retry, or fallback may never increase
the applied number. Continuous serving also reserves the same default completion budget at startup,
removing a former mismatch between the advertised default and the initial decode cohort.

## Clients can ask instead of guessing

Authenticated `GET /v1/models` now returns the standard model list shape with `max_model_len` and a
namespaced `fast_mlx_capabilities` object. It reports native and effective context, default and maximum
completion budgets, the non-streaming ceiling, both HTTP byte ceilings, the reject-or-clamp policy,
and the fact that reasoning tokens count toward the completion budget.

This release is a serving-contract and regression-test result. It does not claim that every model can
use its native maximum on every Mac, or that a long request will be fast. The effective value is
deliberately host-fit-specific, and performance remains a separate measured gate. What changed is
that a valid long-horizon request is no longer rejected because 4,096 was mistaken for a property of
the model.
