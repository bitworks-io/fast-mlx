# Continuous batching with decode-first chunked prefill

- **Status:** COMPLETE — exact dense-Qwen3 service building block PROMOTED 2026-07-14
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
4. **Cancellation and runtime recovery.** Cancellation removes queued, prefilling, ready, or
   decoding work idempotently and releases its slot by the next scheduling boundary. This
   engine-building-block cycle measures runtime cancellation inside one configured keepalive
   interval. Real transport disconnect-to-runtime propagation belongs to the separately
   tracked [production serving-route gate](../../task-inbox/2026-07-14-continuous-batching-serving-route.md).
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
2. per-sequence RoPE offsets and scalar-aligned right-padded causal masks match scalar-cache
   results;
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

### Phase 2 result — 2026-07-12

The MLX-free coordinator and actor-confined dense runtime now execute the scheduler contract
end to end. The runtime keeps one staged greedy token outside committed KV state, performs a
blocking drain before solo→batch membership, reuses fixed-shape compiled functions for stable
membership, and extracts/remerges only at membership boundaries. Chunked prefill evaluates
each boundary so decode and cancellation can interleave without retaining an unbounded lazy
graph.

- Admission is bounded three ways: an explicit 256-request queue default, a config-derived
  per-request context ceiling, and an atomic aggregate logical-token reservation. The runtime
  initially reserved only a bounded decode window and grew in chunks. This was the Phase-2
  proxy; Phase 3 below replaces it with dense KV byte admission that includes rounding and a
  conservative five-copy merge/rebuild transition envelope.
- Runtime construction requires a dense Qwen3 `config.json` proof (`model_type`, context cap,
  vocabulary); unsupported architecture and invalid token IDs fail before MLX indexing. The
  initializer is package-scoped so external callers cannot forge the pairing. A stronger
  loader-bound model identity remains a low-risk hardening item.
- Solo speculation is rejected atomically at admission for this continuous runtime. It cannot
  become a terminal executor failure or silently lose PLD. The service-policy comparison
  remains separate: shared batch without speculation versus the existing solo PLD lane.
- Clean SHA `2a5a5f4df1659ecd14adfbbf9fa1a111698c6acb` passes the Qwen3-32B-4bit
  B1→drain→B2→B1 probe with both streams token- and byte-identical to independent compiled
  scalar baselines. The same SHA passes B3→B2 middle cancellation: both survivors are exact,
  and the cancelled stream equals its two-token scalar prefix. A Qwen3-4B-4bit chunk-size-1
  run also passes after 17 single-token first-prompt chunks and 11 interleaved joiner chunks.
  Those three probes were replayed from clean docs/code SHA `3cc5f63` into the compact
  [Phase 2 evidence artifact](../verdicts/continuous-batching-phase2-evidence-2026-07-12.jsonl)
  (SHA-256 `96f403e04abf9cf6c4a80963892a9247f0a65df8e930a7a7c7667599fc6d0c93`).
- 12 cache-history-sensitive runtime/Xcode tests pass inside 41/41 total `SpikeCoreTests`;
  145 HarnessCore XCTest tests plus 17 Swift Testing tests pass off-box. Final focused review
  found no High or Medium issues and no unsafe Sendable escape.

### Phase 3 result — 2026-07-14

The measured gate passes. A corrected Qwen3-32B-4bit frontier pins one workload nonce across
all eight policy/concurrency processes, drops one warmup, and aggregates three measured runs
per cell. Preliminary rows made with per-process random nonces are disqualified. On the common
workload, solo PLD wins C=1 (28.30 versus 26.72 aggregate service tok/s); batch-no-spec wins
C=2/4/8 by 45.8%/58.6%/74.7% and reaches 56.56 tok/s at C=8 with Jain fairness 1.0000.
These are complete-burst service rates, not decode-only rates.

Final clean harness `7a775f6f1db9495d60eecdf030bf63d752f936e0` replays the real-model
B1→drain→B2→B1, B3→B2 middle-cancel, and chunk-size-1 probes byte-identically; dense batching
engages and speculation remains absent. Dense KV admission now charges allocation rounding,
per-row metadata, and a conservative five-copy membership-transition envelope atomically,
then releases reservations on removal.

The required Qwen3-32B resident run measured 86,412.8508 seconds after warmup, completed 3,518
measured cycles, passed all 33 predicates 3,519/3,519 including warmup, held peak RSS drift to
2.2444% (<5%), cancellation to 28.833 microseconds max (<1 second), and responsiveness to
344.469 milliseconds max (<30 seconds). The [dated verdict](../verdicts/2026-07-14-continuous-batching-chunked-prefill.md)
and [compact Phase 3 evidence](../verdicts/continuous-batching-phase3-evidence-2026-07-14.jsonl)
record the full frontier, raw artifact hashes, provenance, cancellation, A/B/A, and soak.

Disposition: **PROMOTE the exact dense-Qwen3 runtime as a measured service-policy building
block.** Isolated requests prefer solo PLD; the tested simultaneous C≥2 bursts prefer
batch-no-spec. No production serving/API route, runtime default, or dynamic PLD↔batch handoff
is wired by this plan.

**Deliberate scope split:** the original acceptance wording joined runtime removal to a real
client disconnect. The former is proven here; the latter cannot be exercised before a serving
route exists and is now an explicit acceptance gate in the linked production-route task. This
plan is complete only at the named engine-building-block boundary, not at product default-on.

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
2. Keep concurrency-one product behavior on the existing compiled solo decoder; inside the
   probe-only continuous runtime, use its actor-confined scalar compiled step until membership
   becomes a shared batch.
3. Execute pure tick plans inside the actor; drain before mode transition; demultiplex tokens
   by stable request ID.
4. Fail closed on unsupported architecture or batched speculation.
5. Add structured shutdown that cancels and drains every tracked request task before model
   state is released.

### Phase 3 — measurement and promotion gate

1. [x] Extend evidence schemas for aggregate/per-request rate, TTFT, TPOT, Jain fairness index,
   cancellation latency, active slots, prompt chunks, batch-size distribution, and memory.
2. [x] Verify solo, simultaneous burst, staggered mid-join, short+long prompt, cancellation, and
   A/B/A recovery.
3. [x] Benchmark 1/2/4/8 on the product dense model and compare batch-no-spec with solo PLD policy.
4. [x] Run a short soak as a harness check, then the required 24-hour mixed-workload soak.
5. [x] Publish a dated promote/shelve verdict, compact evidence, handoff update, and content piece.

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

The continuous coordinator/runtime remains reachable only through explicit probe CLI
subcommands; no production service route or concurrency flag has been wired. Concurrency one
therefore continues to use the existing compiled decoder. The transition gate now passes;
rollback is still simply not invoking the probes or reverting the feature branch. The scheduler
is additive and the executor remains dense-Qwen3-only; model weight quantization is allowed,
but the continuous KV path remains the exact fp16-cache design. No model conversion, persisted
schema, or user data migration is involved.
