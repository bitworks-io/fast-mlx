# Continuous batching with decode-first chunked prefill

- **Status:** ACTIVE — Phases 0–1 verified; Phase 2 actor integration next
- **Date:** 2026-07-12
- **Owner:** Codex
- **Evaluation lane:** `EXACT`
- **Queue seed:** [`2026-07-12-continuous-batching-chunked-prefill.md`](../../task-inbox/2026-07-12-continuous-batching-chunked-prefill.md)

## User story

As a Concierge/server operator handling concurrent requests, I can admit new dense-model work
without changing an existing stream's greedy tokens, starving decode behind a long prompt, or
retaining cancelled work, so aggregate Apple-Silicon throughput rises while each request keeps
bounded and observable service quality.

## Acceptance contract

1. **Exact transition.** A solo stream produces the same temperature-zero token bytes whether
   it remains alone or a second request joins mid-generation. Any pending submit-first
   lookahead is drained before the first shared decode forward.
2. **Decode first.** Every scheduler tick services active decode work before at most one
   configured chunk per admitted prefill slot. A long prefill cannot stop existing streams
   from advancing.
3. **Fair bounded admission.** Requests enter active slots FIFO. Short prompts admitted beside
   a long prompt can become decodable independently; queue and active limits are explicit.
4. **Cancellation and recovery.** Cancellation removes queued, prefilling, ready, or decoding
   work idempotently and releases its slot by the next scheduling boundary. The serving path
   must demonstrate disconnect-to-removal within one configured keepalive interval.
5. **Fail-closed scope.** Dense attention is the only initially batchable architecture.
   MoE, hybrid/recurrent, vision, diffusion, and unknown state layouts are refused until their
   cache merge/extract and numerical contracts are proven.
6. **No batched speculation.** PLD/trained speculation is unavailable inside a shared decode
   batch. The serving policy may still choose a faster solo speculative lane when measurements
   show it beats batching for the workload.
7. **Measured service frontier.** Clean-SHA evidence records concurrency 1/2/4/8 aggregate
   throughput, per-request p50/p95 TTFT and TPOT, completion fairness, cancellation latency,
   memory, model/runtime versions, and the selected scheduling policy.
8. **Stability.** After a warmup sample, the 24-hour mixed-workload soak holds RSS drift below
   5%, probes responsiveness separately from process liveness, and includes state-poison A/B/A.

## Happy path and edge paths

- **Happy path:** one dense request uses the existing compiled solo pipeline; more requests
  prefill in bounded chunks; the solo lookahead drains once; ready slots enter one greedy
  batched forward and receive one demultiplexed token each.
- **Mid-join edge:** the drain barrier is a distinct scheduled action. It completes before a
  shared decode action can be planned; no duplicate KV position or token is possible.
- **Short beside long:** both receive at most one chunk per tick. The short request moves to
  ready/decode while the long request continues chunking after decode work.
- **Cancellation:** cancelling in any nonterminal phase removes the slot; a repeated or unknown
  cancellation is a harmless no-op.
- **Invalid input:** duplicate request IDs, empty prompts, non-positive output budgets, invalid
  limits, and unsupported architecture classes return typed failures rather than fallback.
- **Out of scope:** paged KV, prompt/prefix reuse, sampled generation, mixed adapters, vision,
  and speculative decoding inside a shared batch. They require separate contracts.

## Evidence informing the design

### Local Swift state

- `HarnessEngineActor` owns the model and one reusable `CompiledMLXDecoder` per KV kind, but its
  generation methods contain no suspension point and serialize whole requests.
- `CompiledMLXDecoder` is fixed to `[1, prompt]` and `[1, 1]`, owns one cache set and a lazy
  `pendingNext`, and deliberately preserves submit-first overlap. That pending token is the
  state the mid-join barrier must resolve.
- `HarnessCore` is MLX-free and testable off-box, so scheduler policy belongs there. MLX arrays,
  cache merge/extract, and compiled functions remain actor-confined in `SpikeCore`.

### Retired Zig evidence: prior, not inherited result

- Qwen3-8B dense aggregate decode rose from 61 to 174 tok/s from one to eight streams (~2.8x).
- MoE/hybrid batching was intentionally disabled; concurrent requests serialized.
- A real mid-stream join bug duplicated/dropped tokens until `drainPipelineForBatch` landed.
- Resumable prefill stayed byte-identical but measured 46 ms before and after on a 14K prompt;
  it is a fairness/cancellation mechanism, not a claimed speedup.
- A model-size-agnostic 256-token hybrid/MoE chunk boundary regressed prefill 20–23% by
  repeatedly streaming expert weights. Dense defaults cannot be copied to hybrid models.
- Blanket batch-over-spec policy lost on echo/agent traffic: serialized PLD beat no-PLD batch
  throughput. The new scheduler exposes the mode boundary; policy is chosen from measurements.

### Current official upstream references

- Python MLX-LM v0.31.2's
  [`BatchGenerator`](https://github.com/ml-explore/mlx-lm/blob/dcbf6e3/mlx_lm/generate.py#L1486-L1837)
  decodes first, then admits bounded prompt work; it provides prompt/generation stages plus
  cache extraction, filtering, and removal.
- Its
  [`PromptProcessingBatch`](https://github.com/ml-explore/mlx-lm/blob/dcbf6e3/mlx_lm/generate.py#L1004-L1206)
  chunks prefill and moves completed prompts into a separate generation batch.
- Its
  [`BatchKVCache`](https://github.com/ml-explore/mlx-lm/blob/dcbf6e3/mlx_lm/models/cache.py#L880-L1085)
  uses left padding, per-sequence offsets, merge, filter, extend, and extract. The v0.31.3
  release subsequently fixed batch-dimension mismatches in these cache extensions, so the
  Swift port must regression-test those transitions rather than copy an older shape blindly.
- The pinned Swift package already exposes
  [`BatchPositionedKVCache`](https://github.com/ml-explore/mlx-swift-lm/blob/702e5a0eaf990e1f6d3db2b6e7d8872858a44055/Libraries/MLXLMCommon/RoPEApplication.swift#L7-L23)
  and cache `prepare(lengths:)` hooks, but no continuous scheduler or dense merge/extract cache
  implementation. A bounded cache/shape probe is therefore required before engine integration.

## Architecture

```text
Sendable request values
        │
        ▼
HarnessCore ContinuousBatchScheduler (pure reducer)
 queued → prefilling → ready → decoding → terminal
        │ ordered TickPlan
        ▼
Inference actor (only MLX owner)
 solo compiled decoder OR dense batch executor
        │
        ├─ token events / completion / cancellation telemetry
        └─ actor-confined per-slot caches and compiled shape state
```

### Pure scheduler API

`HarnessCore/ContinuousBatching/` owns only value types:

- `ContinuousBatchConfiguration`: active-slot limit, prefill-slot limit, prefill chunk size;
- `BatchRequest`: stable integer ID, prompt token count, output budget, architecture class,
  and whether solo speculation was requested;
- `BatchSlotPhase`: queued, prefilling, ready, or decoding with the solo-lookahead flag;
- `BatchTickPlan`: ordered decode action first (`drainSolo`, `solo`, or `batch`) followed by
  zero or more bounded prefill slices;
- `ContinuousBatchScheduler`: FIFO submit, deterministic plan, successful-plan application,
  terminal completion, cancellation, and observable snapshots.

The first implementation deliberately plans a drain-only decode action when a ready request
would join a solo slot with pending lookahead. The shared batch begins on the next tick. That
costs one scheduling boundary but makes the correctness order explicit and testable: no plan
can contain a shared decode before the drain has committed.

### MLX executor boundary

The executor remains a separate actor-isolated layer. The first probe must prove:

1. `[B, 1]` dense model forward matches B independent `[1, 1]` forwards at temperature zero;
2. per-sequence RoPE offsets and left-padded causal masks match scalar-cache results;
3. cache merge → batched append → filter/extract preserves each slot's next-token result;
4. batch sizes 1/2/4/8 do not retrace every token after warmup;
5. removing the middle slot preserves row-to-request identity;
6. the local Qwen3-32B dense target is batchable while unsupported state layouts fail closed.

No MLX array, cache object, or compiled function crosses the actor boundary. No
`@unchecked Sendable` or `nonisolated(unsafe)` escape hatch is permitted.

### Phase 1 result — 2026-07-12

The pinned Swift API can support exact dense batching, but the cache layout must exploit
fast-mlx's fixed-capacity buffers rather than copy Python MLX-LM's dynamically sized
left-padding design.

- The first left-padded prototype was mechanically correct—merge, mask, append, filter, and
  extract unit tests passed—but the real-model probe exposed numerical drift at close argmax
  boundaries. On Qwen3-32B-4bit at B=8 it diverged from scalar decode by compiled step 1 even
  though cache row identity was intact. This was an avoidable layout-induced loss, not a useful
  speed↔quality dial tier, so Phase 1 stopped and redesigned rather than relabeling the bug.
- The accepted cache keeps each row in the scalar layout: valid K/V at
  `0..<logicalLength`, zero right-padding to fixed capacity, per-row scatter positions, per-row
  RoPE offsets, and per-row prefix masks. Extracting a row is then a direct scalar prefix rather
  than a physical-layout conversion.
- Clean SHA `7b9d7090da29a7babc0c6c73e299e8948dbcef39` passed Qwen3-32B-4bit B=1/2/4/7/8 with both
  fixed and shapeless compilation: exact greedy tokens, `0.000000` initial max logit delta,
  one main trace per stable shape, and one intentional trace after middle-row removal. B=7 is
  the ragged shape produced by removing a middle member from B=8.
- The extended fixed-shape run passed 64 compiled steps at B=4 and B=8 with the same exactness
  and trace counts. Shapeless compilation showed no steady-state advantage in this probe;
  fixed-shape remains selected because membership changes intentionally create a new compiled
  function. Probe step timings are shape diagnostics, not the Phase 3 throughput frontier.
- The probe reads `config.json` before model load and currently admits only dense `model_type =
  "qwen3"`; `qwen3_moe` fails closed. Capacity is proven from authoritative array offsets
  outside compiled replay, and any driver-supplied lengths must equal those offsets.

Verification: 9 focused batch-cache tests and 29 total `SpikeCoreTests` passed through Xcode on
the bench Mac; 134 HarnessCore XCTest tests plus 17 Swift Testing tests passed off-box. A focused
review's capacity, length-validation, transition-proof, and architecture-gate findings were
fixed; re-review found no High or Medium issues.

## TDD and implementation sequence

### Phase 0 — pure scheduler

1. Add failing tests for configuration/input rejection and duplicate IDs.
2. Add FIFO admission and bounded one-chunk-per-tick tests.
3. Add decode-before-prefill and short-beside-long fairness tests.
4. Add the red-on-revert drain-before-join transition test.
5. Add queued/prefill/ready/decode cancellation and slot-reuse tests.
6. Add solo-speculation versus batched-no-speculation plan tests.
7. Implement the minimum pure state machine; run focused then full HarnessCore tests and
   coverage.

### Phase 1 — batch cache and shape probe

1. Add Xcode-only tests for per-sequence offsets, masks, merge, filter, extract, and row order.
2. Implement the smallest dense fp16 batch cache behind an internal flag.
3. Run scalar-versus-batch logits/token parity at B=1/2/4/8 on the bench Mac.
4. Run the requested shapeless/fixed-shape compile probe and record retrace evidence.
5. Stop and redesign if batch state cannot remain actor-confined or parity fails.

### Phase 2 — actor integration

1. Add a streaming request API whose continuation termination enqueues actor cancellation.
2. Keep the existing compiled solo decoder for one active slot.
3. Execute pure tick plans inside the actor; drain before mode transition; demultiplex tokens
   by stable request ID.
4. Fail closed on unsupported architecture or batched speculation.
5. Add structured shutdown that cancels and drains every tracked request task before model
   state is released.

### Phase 3 — measurement and promotion gate

1. Extend evidence schemas for aggregate/per-request rate, TTFT, TPOT, Jain fairness index,
   cancellation latency, active slots, prompt chunks, batch-size distribution, and memory.
2. Verify solo, simultaneous burst, staggered mid-join, short+long prompt, cancellation, and
   A/B/A recovery.
3. Benchmark 1/2/4/8 on the product dense model and compare batch-no-spec with solo PLD policy.
4. Run a short soak as a harness check, then the required 24-hour mixed-workload soak.
5. Publish a dated promote/shelve verdict, compact evidence, handoff update, and content piece.

## Verification mapping

| Acceptance criterion | Primary proof |
| --- | --- |
| Exact mid-join | Pure drain-order test plus on-box base/solo→batch byte comparison |
| Decode-first fairness | Pure ordered-plan and short-beside-long tests; TTFT/TPOT evidence |
| Cancellation | Pure all-phase cancellation tests plus disconnect latency bench |
| Fail-closed scope | Typed pure rejection tests plus Xcode architecture gate tests |
| No batched speculation | Pure plan assertion plus engagement telemetry |
| Throughput frontier | Clean-SHA 1/2/4/8 CSV/JSONL with per-request distributions |
| Stability | Responsiveness probes, A/B/A, memory samples, and 24-hour RSS verdict |

## Rollback and blast radius

All new behavior stays behind a concurrency flag. Concurrency one continues to use the
existing compiled decoder until the transition gate passes. The pure scheduler is additive;
the batch executor is dense/fp16-only initially. Rollback is disabling batching or reverting
the feature branch—no model conversion, persisted schema, or user data migration is involved.
