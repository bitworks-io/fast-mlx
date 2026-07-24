# Production continuous-batching serving route plan

- **Date:** 2026-07-24
- **Status:** active — review amendments incorporated; Phase 0 contract frozen
- **Owner:** Codex
- **Branch:** `codex/continuous-batching-serving-route`
- **Evaluation lane:** EXACT
- **Queue seed:** [`2026-07-14-continuous-batching-serving-route.md`](../../task-inbox/2026-07-14-continuous-batching-serving-route.md)

## Operator story

A local fast-mlx operator can launch one source-locked text model and point an
OpenAI-compatible client at a bounded HTTP endpoint. A temperature-zero chat completion should
stream promptly, preserve the model's exact greedy bytes, use the already-promoted dense-Qwen
continuous runtime for simultaneous requests, and release disconnected work within one configured
keepalive interval. Models outside the optimized geometry must retain an honest scalar path rather
than disappearing from the product surface.

The first transport owner is HTTP/1.1 with Server-Sent Events for
`POST /v1/chat/completions`. This follows the project's explicit OpenAI-compatible serving
priority and provides a real client-close signal. WebSocket and Concierge-specific transports
remain separate consumers of the same service contract.

## Observable acceptance criteria

1. **OpenAI-compatible text boundary.** `stream:false` returns one chat-completion JSON object;
   `stream:true` returns ordered `chat.completion.chunk` SSE data objects followed by exactly one
   `data: [DONE]`. Model, messages, primary `max_completion_tokens`, deprecated `max_tokens`
   compatibility, `temperature`, and supported stop semantics validate before admission. An
   OpenAI-shaped error always contains `type`, `message`, nullable `param`, and nullable `code`.
2. **Exact model behavior.** For the same rendered chat prompt and output budget, retained
   temperature-zero HTTP output is token- and byte-identical to the in-process control. Streaming
   chunk boundaries may differ, concatenated bytes may not.
3. **Real disconnect propagation.** Closing a real TCP/SSE client cancels the response-consumer
   task, reaches `ContinuousBatchCoordinator.cancel`, and removes the logical slot within one
   configured keepalive interval. Physical KV retained by an old batch may survive until the next
   bounded membership rebuild; that rebuild and final shutdown must release it completely.
   Repeated close/cancel is harmless.
4. **Survivor and reuse safety.** A hostile three-request test disconnects the middle/longest row,
   keeps both survivors byte-identical, admits a replacement into the released slot, and finishes
   with zero live slots and zero reserved KV bytes.
5. **Bounded backpressure.** The transport and coordinator never accumulate an unbounded token or
   body buffer. A bounded, suspending per-request delta mailbox carries explicit capacity from the
   channel to the producer. Buffer-full starts a configured stall deadline; expiry cancels that
   request, while the mailbox may never hold more than its configured byte and delta limits.
6. **Measured policy honesty.** A source-locked Qwen policy may select isolated PLD only after a
   bounded coalescing decision and may select shared batch only for simultaneous admitted work.
   PLD stays absent inside a shared batch. Once solo PLD starts, later arrivals queue; no live
   state silently migrates. Each response exposes the selected policy in internal telemetry.
7. **Model-generic fallback.** A launched source-locked text LM receives a scalar greedy route only
   after text-only preparation and cache-off greedy parity pass for that model. The response never
   labels that fallback as continuous batching or PLD. Broad loader-family support remains
   unclaimed until the representative architecture matrix passes.
8. **Fail-closed scope.** Sampling, multi-choice generation, tools/media, unsupported cache
   layouts, malformed chat templates, invalid token IDs, unknown models, request-body overflow,
   and proof/identity mismatch fail before optimized admission. There is no lossy-KV-plus-PLD
   route.
9. **Operational safety.** Loopback is the default bind. A non-loopback bind requires an explicit
   API key. Keys and prompts do not enter logs or evidence. Shutdown stops admission, cancels or
   drains every request within a bounded grace interval, and releases the listener and model.
10. **Production evidence.** A clean-SHA Release binary binds model/tokenizer/checkpoint,
    source/dependency/config/workload identities, redacted canonical HTTP envelopes, cancellation
    timing, slot/resource telemetry, output hashes, MLX active/cache/peak, process RSS, and
    fresh-output provenance. Evidence stores allowed headers, status, timing, chunk counts, byte
    counts, and SHA-256 values only; it never stores authorization, raw prompts, or raw generated
    text.

## Happy, failure, and recovery paths

- **Happy isolated request:** one request survives the coalescing window, uses the qualified solo
  policy for that exact model/workload, streams deltas, publishes usage, and closes with `[DONE]`.
- **Happy simultaneous requests:** two or more requests admitted in the window enter the existing
  decode-first continuous coordinator; shared batches never engage speculation.
- **Scalar fallback:** an otherwise supported text model that lacks a continuous-batch proof uses
  one actor-owned scalar decoder, resets before every request, and reports that route honestly.
- **Client disconnect:** channel close or response write failure cancels the channel's active
  request lease. The scheduler/runtime removes its logical slot; survivors continue; the next
  bounded membership rebuild compacts physical ownership; replacement admission works.
- **Backpressure stall:** the bounded delta mailbox reaches its deadline, records a typed transport
  cancellation reason, cancels the backend handle, and closes without accepting publication
  beyond the configured capacity.
- **Invalid or unsupported request:** return an OpenAI-shaped 4xx error without loading request
  state into the scheduler. Queue/resource exhaustion returns 429 with a bounded retry signal.
- **Backend failure before headers:** return an OpenAI-shaped 5xx. Failure after SSE headers emits
  one error event when writable, cancels the handle, and closes; it never emits a false `[DONE]`.
- **Shutdown:** stop accepting, reject new work, cancel stalled/disconnected work, wait through the
  grace deadline, then force cancellation and prove listener/slot/reservation cleanup.

## Architecture and support matrix

| Surface or state | Initial disposition |
| --- | --- |
| Text-only chat, temperature zero, one completion | supported |
| HTTP/1.1 non-stream and SSE stream | supported |
| Dense source-locked Qwen3, fp16 KV | continuous batch eligible after exact proof |
| Launched source-locked text LM with preparation/parity proof | scalar greedy fallback; no batch/PLD claim |
| Unproven loader-compatible family | startup rejection for serving; remains a product support candidate |
| Solo Qwen PLD | gated on an incremental, actor-confined implementation and fresh parity/performance proof |
| Exact-prefix/session cache | disabled in this route until slot-ownership integration is separately proven |
| Affine/KVTuner/KVarN continuous KV | explicit-only and out of the first serving promotion |
| Sampling, `n > 1`, logprobs, tools, images/audio, adapters | reject before admission |
| Sliding/recurrent/hybrid/MLA/MoE batched state | scalar fallback where loader-compatible; never silently batch |
| WebSocket, remote unauthenticated bind, SSD session state | out of scope |

The optimized matrix does not narrow fast-mlx's model-generic product boundary. It only controls
which loaded models may use the continuous-batch or PLD fast paths.

## Design

```text
NIO HTTP/1.1 channel
  parse + validate + auth + bounded body
                 │
                 ▼
ServingCore request lease (one channel owns at most one active handle)
  render chat + route policy + bounded decoded-delta stream
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
scalar actor          ContinuousBatchCoordinator
model-generic         dense-Qwen proof only
       │                   │
       └─────────┬─────────┘
                 ▼
     actor-confined model/tokenizer/MLX state
```

### Pure service contract

A new MLX-free `ServingCore` target owns:

- strict OpenAI request/response/error and SSE value types;
- request validation and supported-field rejection;
- bounded body, output, coalescing, queue, backpressure, keepalive, and shutdown policies;
- a `ServingGenerationBackend` protocol returning one explicitly cancellable request handle with a
  bounded, suspending delta mailbox rather than an unbounded `AsyncThrowingStream`;
- a lease state machine that permits exactly one terminal path and one idempotent cancellation;
- policy telemetry (`scalar-greedy`, `solo-pld`, `continuous-batch-no-spec`) without prompt data;
- exact delta assembly and usage accounting contracts.

The transport receives decoded text deltas, not MLX arrays or tokenizer objects. Tokenization,
incremental detokenization, and the model remain in an actor-confined adapter.

### Target boundaries

SwiftPM enforces the dependency direction:

- `ServingCore`: pure request/response/error values, leases, bounded mailbox, and policies;
- `ServingNIO`: `ServingCore` plus NIO only; request parsing, HTTP/1.1, SSE, and channel lifecycle;
- `SpikeServingAdapters`: `ServingCore` plus `SpikeCore`; actor-confined scalar and continuous
  runtime adapters;
- `fastmlx-serve`: thin composition executable depending on `ServingNIO` and
  `SpikeServingAdapters`.

Both `ServingCore` and `ServingNIO` must build without `SpikeCore` or any MLX product.

### HTTP transport

Add `.package(url: "https://github.com/apple/swift-nio.git", exact: "2.101.2")` directly to the
manifest and give `ServingNIO` explicit `NIOCore`, `NIOPosix`, and `NIOHTTP1` product
dependencies. The channel pipeline:

1. rejects oversized/invalid requests before backend admission;
2. creates one request lease only after validation and authentication;
3. writes non-stream JSON or SSE through the bounded mailbox and observes channel writability;
4. cancels the lease on `channelInactive`, input half-close, write failure, timeout, or task
   cancellation;
5. stops admission before structured server shutdown.

No channel handler owns MLX state, and no `@unchecked Sendable` escape is permitted.

HTTP/1.1 permits sequential keep-alive reuse, but a channel may own only one active generation
lease. A pipelined or overlapping second generation request on that channel is rejected before
backend admission. Channel close cancels the one active lease; sequential completion clears it
before the next request is accepted.

### Runtime adapters and policy

The first adapter exposes the proven automatic-drive `ContinuousBatchCoordinator` seam. It loads a
model once, verifies `DenseContinuousBatchModelProof`, renders chat through the checkpoint
template, and converts token IDs to exact incremental text without leaking tokenizer state across
actors. The adapter must not expose the coordinator's current unbounded `AsyncThrowingStream`
directly: coordinator publication becomes capacity-aware before this adapter can ship.

A scalar adapter preserves loader-compatible text models and resets its decoder before each
request. It serializes requests explicitly and remains cancellable between token steps.

The measured dynamic policy needs a separate incremental PLD slice in the same MLX owner. A pure
admission reducer holds the first request for a small configured coalescing window:

- one retained request at expiry may start qualified solo PLD;
- two or more retained requests enter continuous batch without speculation;
- arrivals after solo execution begins queue until that request terminates;
- cancellation during the hold removes the request without model work;
- no policy may change after execution starts.

Until incremental PLD passes exactness, cancellation, and performance gates, the production route
remains explicit `batch-no-spec`/scalar and cannot claim the dynamic default.

## TDD and implementation sequence

### Phase 0 — plan and pure contracts

1. Add failing request-validation tests for supported chat, unknown fields/parameters, invalid
   model/temperature/budgets, body limits, and OpenAI-shaped errors.
2. Add failing lease tests for one terminal path, idempotent disconnect cancellation,
   backpressure timeout, and shutdown.
3. Add failing bounded-mailbox tests proving a producer cannot publish more than the configured
   delta/byte capacity while the consumer is paused and that deadline cancellation is idempotent.
4. Add failing admission-policy tests for one-request hold, simultaneous batch choice,
   cancellation during hold, arrivals after solo start, and no silent PLD downgrade.
5. Implement the smallest MLX-free values/reducers. Run focused then full HarnessCore/ServingCore
   tests off-box.

### Phase 1 — transport with a scripted backend

1. Add NIO embedded-channel tests for parsing, JSON, SSE ordering, `[DONE]`, auth, body bounds,
   writability changes, write failure, `channelInactive`, sequential keep-alive reuse, and
   rejection of a pipelined second request while one lease is active.
2. Add real loopback socket tests for a closed SSE client and a slow reader. Prove backend
   cancellation within the configured keepalive/stall deadline and prove mailbox capacity is
   never exceeded.
3. Implement `fastmlx-serve` transport and structured shutdown against a scripted backend.
4. Build `ServingCore` and `ServingNIO` independently and prove neither links `SpikeCore`/MLX.
5. Run Thread Sanitizer where the package/toolchain supports it; otherwise document the skipped
   check and use repeated hostile cancellation tests.

### Phase 2 — model-generic scalar route

1. Add reset-before-request and cancellation regression tests around `InferenceActor`.
2. Add tokenizer/chat-template and incremental-detokenizer tests.
3. Wire the pinned loader into a scalar serving adapter; require startup text preparation and
   verify non-stream and SSE against the same cache-off in-process greedy control on a small
   source-locked model.
4. Exercise a representative architecture matrix: at least one second positive text family and
   negative recurrent/hybrid/unsupported-cache cases with typed startup/route failures. Keep any
   broad loader-family serving claim closed until that matrix is complete.
5. Fail closed on media/tool/sampling requests and never label fallback as batch/PLD.

### Phase 3 — dense continuous adapter and real disconnect

1. Add adapter tests for atomic admission, stream termination, queue exhaustion, bounded
   coordinator publication, slot/resource telemetry, and shutdown.
2. Bound each synchronous runtime tick below the keepalive SLA or thread an actor-confined
   cancellation token through chunked prefill/decode and poll it at chunk boundaries. Prove a
   disconnect during loaded prefill and during decode, not only while awaiting the next token.
3. Sync with `spike/scripts/sync_llmbench.sh`; run MLX-coupled tests only through
   `xcodebuild -skipPackagePluginValidation` on `llmbench`.
4. Run a fresh real-network Qwen test: two survivors plus a longest middle request, disconnect the
   middle socket, append a replacement, and prove output parity, logical slot reuse, bounded
   membership rebuild, final zero physical reservations, bounded cancellation, and no batched
   speculation.
5. Run non-stream and SSE exactness against the frozen in-process controls.

### Phase 4 — measured solo-PLD policy

1. Add failing pure transition tests before modifying the runtime.
2. Implement the smallest incremental actor-confined solo PLD round that can drain to canonical
   scalar state before batch membership. Preserve byte-exact temperature-zero semantics.
3. Prove cancellation in every PLD state, stale-plan rejection, solo-to-batch drain ordering,
   bounded draft lookback, and no lossy-KV combination.
4. Re-run the identical Qwen C=1/2/4/8 workload through real HTTP. The dynamic router must preserve
   the measured policy frontier; otherwise keep `batch-no-spec` explicit and do not promote a
   default.

### Phase 5 — production evidence and closure

1. Run a fresh Release smoke, hostile disconnect/A/B/A packet, and transport-level resident soak
   with progress/lock/watchdog/RSS/MLX/resource telemetry.
2. Authenticate exact source, binary, dependency, model/tokenizer/checkpoint, config, request,
   workload, redacted-envelope, and evidence hashes. Run a regression fixture and secret scan that
   prove known prompt/key sentinels are absent from evidence.
3. Write the dated verdict, public fast-mlx-only content, verification packet, and handoff.
4. Run focused correctness/security review, diff inspection, relative-link checks, ShellCheck,
   staged gitleaks, coherent commits with the required trailer, fresh proof, and `--no-ff` merge.

## Proof mapping

| Criterion | Primary proof |
| --- | --- |
| OpenAI JSON/SSE contract | pure codecs + NIO embedded channel + loopback client |
| Exact bytes | HTTP transcript hashes paired with in-process token/output hashes |
| Disconnect propagation | real socket close during loaded prefill/decode → logical removal timing |
| Survivor/slot reuse | loaded hostile three-row disconnect, bounded rebuild, replacement, final zero physical KV |
| Backpressure | pure capacity bound + embedded/loopback slow reader + bounded stall cancellation |
| Dynamic policy | pure reducer, engagement trace, same-workload HTTP frontier |
| Generic fallback | second materially different text family scalar HTTP smoke |
| Fail-closed/security | parser/auth/geometry/parameter tests and secret-safe logs |
| Stability | fresh-output resident transport soak and resource-release packet |

## Threat model, blast radius, and rollback

The threats are unauthenticated remote model access, prompt/key leakage, request-body memory
exhaustion, slow-reader token retention, disconnect leaks, cross-request slot confusion, and
shutdown races. Default loopback binding, remote-bind authentication, hard byte/queue/output
budgets, one-channel/one-lease ownership, actor confinement, and stale-plan validation contain the
blast radius.

The route is additive and disabled until explicitly launched. It does not change model files,
cache formats, existing CLI defaults, or current scalar engine behavior. Rollback is stopping
`fastmlx-serve` or reverting the serving commits; cache-off CLI and harness paths remain intact.

## Official compatibility sources

The Phase 0 wire contract was checked on 2026-07-24 against the official
[Chat Completions create reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[Chat Completions streaming events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events),
and the official [OpenAPI specification](https://github.com/openai/openai-openapi). The downloaded
OpenAPI JSON used for this review had SHA-256
`9f65dd3582af1404d00d22f56d32595524a88459a98310afbb3cc488eb3fa270`.
The implementation supports only the text/temperature-zero subset named above and rejects
documented fields outside that subset instead of implying full endpoint coverage.
