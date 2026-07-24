# Production continuous-batching serving route plan

- **Date:** 2026-07-24
- **Status:** active — Phases 0–4 complete; Phase 5 product smoke and bounded transport preflight
  pass, while the 24-hour resident soak awaits a correctly identified 256-GiB host
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
4. Preserve exact greedy semantics across ragged service traffic by assigning every request an
   actor-derived fixed KV-capacity cohort. Only equal-capacity requests may share a model
   forward; incompatible cohorts receive fair solo/shared turns and never silently widen one
   another's attention reduction shape. Requests that can outgrow their initial reserve remain
   isolated until a separately proven dynamic-capacity contract exists.
5. Run a fresh real-network Qwen test: two survivors plus a logically longest same-capacity middle
   request, disconnect the middle socket, append a same-capacity replacement, and prove output
   parity, logical slot reuse, bounded membership rebuild, final zero physical reservations,
   bounded cancellation, and no batched speculation. Add a separate cross-capacity case that
   proves incompatible rows never share a decode action and still make progress.
6. Run non-stream and SSE exactness against the frozen in-process controls.

#### Phase 3 diagnostic amendment — 2026-07-24

The first production-route proof found a real scope gap in the earlier engine promotion. A
long-lived equal-capacity B=2 pair remained byte-identical to independent scalar controls, but
adding a much longer middle row widened the shared fixed-capacity attention buffers; both
survivors later diverged while the replacement remained exact. The hash-bound failure therefore
rules out transport and stable batching and localizes the unsafe behavior to cross-capacity
membership. Masked padding is mechanically unreachable, but changing the reduction width can
still change close logits on the loaded quantized model. The production route must not treat
different physical widths as one exact batch merely because their logical cache layouts are
valid.

Recovery is test-first and fail-closed: pure scheduling tests cover cohort isolation, round-robin
progress, drain-before-same-cohort join, and legacy unrestricted runtimes; MLX runtime tests bind
the capacity classification and reject mixed-capacity decode directly; the loaded Apple proof
then covers same-capacity B3→B2→B3 hostile compaction plus cross-capacity isolation. Historical
failed boundaries remain diagnostic-only and unchanged.

#### Phase 3 closure — 2026-07-24

Phase 3 is complete at clean source
`d9143ce24ac084d295e3903443c96aa07c1fe6e7`. The pure serving/core verification passes 675
XCTest plus 17 Swift Testing tests. The synchronized Apple verification passes 140
`FastMLXHarnessTests`, 176 `SpikeCoreTests`, and 30 `SpikeServingAdaptersTests` with the two loaded
tests intentionally skipped when their model environment is absent; the Release build succeeds.
Xcode test and Release log SHA-256 values are
`eb763b849d7d4d53f4d7e094cbe9091c8c620efa1252d84ce09413bc8b7cf937` and
`463f331dbd79a0d6f27d2738688839dc70338e9590d32a0c0b12e61b10385324`.

The fresh clean loaded proof is terminal `COMPLETE` at
`/Users/llmbench/perf-work/results/continuous-serving-phase3-loaded-capacity-cohort-d9143ce-v6`.
Exactly one selected real-Qwen integration test passed in 48 seconds. It closes a real TCP/SSE
client during loaded prefill, closes the logically longest same-capacity middle row during shared
decode, preserves survivor and replacement bytes, proves B3→B2→B3 slot reuse, proves two
incompatible 2-row capacity cohorts share only internally, keeps speculation absent, and ends
with zero live slots, reserved KV bytes, and TCP connections. The launcher, artifact manifest,
xcresult-file manifest, and test log SHA-256 values are
`62af08aeb247e1c932258f7dd2fc005cf2f8cb180a01dc40f9fc3f66ca87917a`,
`8a540c31cacd79254432ca6e300d5c27b3d3fde1c23f8a8202f3239aefa0ecc9`,
`0b2ab7a3f2c79a96f8fc57cf02766d7a4fbcf08a98a187c41050b3082a53ecdf`, and
`0a242aa14209452c3480c2906b4e5b4895cc7ff9ed3ba8dd65c83f8858998a16`.
The generic XCTest child was identity-tracked at a maximum 18,231,568 KiB RSS; no lock, watchdog,
or orphan remains. The verification packet is
[`continuous-serving-phase3-verification-2026-07-24.json`](../verdicts/continuous-serving-phase3-verification-2026-07-24.json),
SHA-256 `d99094319c899048bf27db32ed6bfc8e1e836393a71747fb2edfeb03d8ae4702`.

This authorizes the explicit `continuous-batch-no-spec` route for the qualified dense-Qwen
boundary only. It does not authorize the dynamic router, solo PLD, a broad loader-family claim, or
a default switch. At the Phase 3 boundary, those remained Phase 4 and later acceptance work.

### Phase 4 — measured solo-PLD policy

1. Add failing pure transition tests before modifying the runtime.
2. Implement the smallest incremental actor-confined solo PLD round that can drain to canonical
   scalar state before batch membership. Preserve byte-exact temperature-zero semantics.
3. Prove cancellation in every PLD state, stale-plan rejection, solo-to-batch drain ordering,
   bounded draft lookback, and no lossy-KV combination.
4. Re-run the identical Qwen C=1/2/4/8 workload through real HTTP. The dynamic router must preserve
   the measured policy frontier; otherwise keep `batch-no-spec` explicit and do not promote a
   default.

#### Phase 4 closure — 2026-07-24

Phase 4 is complete at clean source
`520a106708f4f0d47cf8bc9f08a188078c4915d8`. The actor-confined incremental PLD session preserves
bounded lookback, stale-plan rejection, cancellation, outputless drain-before-join ordering,
shared-batch no-spec execution, exact temperature-zero bytes, cache lifecycle, masks/GQA, hostile
compaction, and lossy-KV-plus-PLD rejection. The shipping CLI does not expose dynamic PLD.

The clean Apple regression passes 140 `FastMLXHarnessTests`, 194 `SpikeCoreTests`, and 50
`SpikeServingAdaptersTests` with four expected loaded-environment skips; Release and
build-for-testing succeed. The fresh loaded exactness boundary is terminal `COMPLETE` at
`/Users/llmbench/perf-work/results/continuous-serving-phase4-loaded-exact-520a106-v1`: exactly one
selected Qwen test passed in 22.459 seconds, all artifact/xcresult hashes reauthenticate, maximum
XCTest RSS was 18,261,232 KiB, and no watchdog, lock, or orphan remains.

The identical real-HTTP diagnostic closes the speed gate negative. Retained ngram-3 improved C=1
from 21.7851 to 22.0880 tok/s (**+1.3902%**); bounded ngram-2 improved 21.8526 to 22.0981 tok/s
(**+1.1236%**). Both preserve exact output SHA-256
`e2bd50d266a2af3f7913eb8ad8b6c5842d4131f62fa3d3bc6dcc98ce93a7270d`, but both miss the required
+5%. Diagnostic evidence is non-promotable and forbidden from speed aggregation.

Verdict: **SHELVE dynamic solo PLD for this cycle; retain the correctness implementation and
continue with explicit `continuous-batch-no-spec`.** Do not rerun or retune the unchanged policy.
Reopen only after a profile identifies one bounded actor-confined change with a credible route
past +5%.

Verdict:
[`2026-07-24-continuous-serving-solo-pld.md`](../verdicts/2026-07-24-continuous-serving-solo-pld.md);
verification packet:
[`continuous-serving-phase4-verification-2026-07-24.json`](../verdicts/continuous-serving-phase4-verification-2026-07-24.json),
SHA-256 `19dc0345b0362af71aea4d503baf804cf0b0ea06ab4c491f6c76b2b14f85edd3`.

### Phase 5 — production evidence and closure

1. **PARTIAL:** fresh Release product smoke v8 and the hostile C=4 transport preflight v2 are
   `COMPLETE` and independently authenticated. The preflight carries one dropped warmup plus
   135.580501 measured seconds, 24 evidence rows, four typed `clientDisconnected` rows, bounded
   recovery, stable memory, and no watchdog/orphan. The transport-level 24-hour resident soak
   remains unlaunched.
2. **PARTIAL:** exact source, binary, dependency, model/tokenizer/checkpoint, workload, redaction,
   and evidence hashes authenticate for smoke v8 and preflight v2. The prepared soak packet uses
   incremental evidence scanning, exact request/response pairing, normalized power snapshots, and
   explicit memory/cache limits; its final host-specific hashes must be reviewed again after the
   target host is admitted.
3. Write the dated verdict, public fast-mlx-only content, verification packet, and handoff.
4. Run focused correctness/security review, diff inspection, relative-link checks, ShellCheck,
   staged gitleaks, coherent commits with the required trailer, fresh proof, and `--no-ff` merge.

Phase-5 checkpoint verification:
[`continuous-serving-phase5-preflight-verification-2026-07-24.json`](../verdicts/continuous-serving-phase5-preflight-verification-2026-07-24.json).
The intended `192.168.1.253` host currently presents Dropbear 2022.83 with only an RSA host key,
not the expected macOS OpenSSH Remote Login identity. No key was trusted and no credentials or
workloads were sent. Correct the network/Remote Login mapping, authenticate the host read-only,
then derive a fresh host-specific soak packet; do not copy the `.252` laptop paths or 140-W
charger contract onto a desktop host.

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
