# Exact Prefix / Session Cache and Request-Start Stack Plan

**Date:** 2026-07-23
**Status:** active design; implementation not yet admitted
**Classification:** EXACT
**Branch:** `codex/exact-prefix-session-cache`

## Operator story

A Concierge or coding-agent operator repeatedly extends a long, source-locked conversation. After
one successful turn, the next request should restore the longest exact actor-owned prefix and
prefill only the new tail. The warm path must reduce time to first token without changing one
temperature-zero output byte, crossing semantic request boundaries, retaining an unbounded unified
memory working set, or silently claiming support for cache geometries whose complete state cannot
yet be restored.

The first implementation is a reusable engine and request-start proof surface. It is not yet the
production OpenAI serving route: continuous-batch ownership remains a separately gated consumer of
the same contracts.

## Observable acceptance criteria

1. **Exact nearest-prefix reuse.** For an admitted request, lookup returns the longest token prefix
   under the exact semantic key, restores the actor-owned state into the same compiled cache
   identities, and reports logical cache-read tokens separately from physically evaluated tail
   tokens.
2. **Byte-identical behavior.** Cache-off and cache-on temperature-zero generation are byte- and
   token-identical for exact-hit, partial-tail, A/B/A poison, and multi-turn extension cases.
3. **Positive commit only.** Only a clean, successful, non-empty text generation may publish a
   snapshot. Errors, cancellation, zero generated tokens, pad-only output, media/image input,
   unsupported architecture, unsupported cache route, or incomplete snapshot evidence publish
   nothing.
4. **Semantic isolation.** The key binds the loaded model instance and revision evidence,
   tokenizer, prompt/template format, tools, exact KV storage/attention route, position/RoPE
   semantics, architecture-state class, drafter state, and an explicit privacy/isolation
   namespace supplied by the request owner. A mismatch is a miss or a typed rejection, never a
   partial restore.
5. **Bounded true retention.** Entry and retained-byte limits are both mandatory. A two-phase
   reservation evicts LRU entries before MLX snapshot allocation, accounts detached array bytes
   plus token/control storage, and releases the reservation on capture or commit failure.
6. **Compiled identity preservation.** Restore mutates existing cache buffers and control arrays in
   place. It neither swaps the cache objects captured by `MLX.compile` nor uses the custom cache
   types' intentionally unsupported generic `state` setters or `copy()` methods.
7. **Honest telemetry.** Evidence separates cold/warm TTFT, template/render time, tokenize time,
   lookup time, restore time, cache-read tokens, tail-prefill tokens, apparent prefill rate,
   physical prefill rate, retained bytes, entry count, evictions, hit/miss/rejection reason, and
   eager-warmup time.
8. **Request-start warmup.** Optional eager model warmup exercises the one- and eight-token prefill
   and decode shapes after load, is excluded from prefix entries and timed request evidence, resets
   state in place, and reports its own duration and memory policy.
9. **Template/tokenize cache.** A separately bounded exact host cache keys rendered/tokenized work
   by tokenizer, template, tool schema, prompt content, and formatting options. It cannot be counted
   as model-prefix reuse and has independent hit/byte/timing telemetry.
10. **Fail-closed support boundary.** Unsupported geometry, sliding/rotating state,
    recurrent/hybrid/SSM state, vision/media embeddings, speculative drafter state, compressed KV,
    continuous-batch slot state, non-finite output, or partial evidence cannot engage or commit.
11. **Memory admission remains explicit.** Any run with a raised wired limit sets
    `Memory.cacheLimit`; the complete configured hot-cache byte budget is reserved in context
    admission rather than assumed empty.
12. **No accidental benchmark hits.** Cold controls use distinct salted prompts and a fresh cache;
    warm cells declare the exact committed prefix and never inherit warm state from an unrelated
    trial.

## Happy path

1. Load a source-authenticated, admitted dense causal model.
2. Optionally run isolated eager warmup, then reset the decoder in place.
3. Render and tokenize request A through the bounded host cache.
4. Miss the model-prefix cache, prefill all of A, generate successfully, reserve bytes, detach the
   already-staged prompt-only checkpoint plus the exact final context snapshot, and commit only
   the snapshots admitted by the entry/byte budgets.
5. Render and tokenize A plus a new user tail.
6. Find A as the longest exact prefix, restore its cache buffers in place, evaluate only the new
   tail, and generate the same tokens as a cache-off control.
7. Commit the new longer successful context and evict older entries as needed.

## Failure and recovery paths

- **Semantic mismatch:** report a miss/rejection and perform a clean full prefill; do not inspect or
  partially restore the incompatible entry.
- **Isolation mismatch:** a different tenant/session namespace is always a miss, even when every
  token and model field matches. Do not expose cross-namespace hit timing or entry existence.
- **Unsupported cache or architecture:** reject cache enablement before model work. The normal
  cache-off path remains available.
- **Insufficient byte budget:** evict LRU entries before allocation. If one snapshot exceeds the
  whole budget, skip the commit and keep generation successful.
- **Snapshot/restore validation failure:** discard the reservation, reset the live decoder in place,
  and fail the cache-enabled request closed. A later request may use the ordinary cold path.
- **Generation error/cancel/zero/pad-only/media:** do not commit; prior entries remain unchanged.
- **A/B/A poisoning attempt:** B may use or commit only under B's exact key. A's later output must
  match the original A control byte for byte.
- **Memory pressure:** evict before capture; never raise wired or MLX cache limits implicitly.
- **Process restart:** phase 1 has no SSD tier, so hot entries disappear. The next request is a cold
  miss, not an error.
- **Continuous-batch request:** phase 1 rejects reuse. Later integration must transfer a complete
  scalar snapshot into a newly owned slot before admission and commit only after coordinator
  success; cancellation and stale-plan paths publish nothing.

## Support and architecture matrix

| Model/cache shape | Phase-1 disposition | Required proof |
|---|---|---|
| Qwen3 dense GQA, full attention, fp16 `CompiledKVCache`, scalar actor | Supported after on-box proof | in-place round trip, partial tail, exact hit, A/B/A, byte accounting |
| Llama dense GQA, full attention, fp16 `CompiledKVCache`, scalar actor | Supported after independent on-box proof | same suite on source-locked Llama; Qwen evidence cannot promote it |
| Phi3 dense GQA with the admitted inert-window/LongRoPE geometry, fp16 scalar actor | Supported after independent on-box proof | exact position/LongRoPE identity plus same suite |
| Affine, frozen KVTuner, KVarN, TurboQuant custom cache state | Rejected in phase 1 | per-format detached snapshot, exact in-place restore, true byte accounting, loss-policy proof |
| Sliding/rotating/local-window attention | Rejected | window cursor, sink/front-trim, mask, and RoPE checkpoint proof |
| Hybrid linear attention, Mamba/SSM, recurrent or `CacheList` state | Rejected | explicit complete recurrent checkpoint and conservative retained-allocation accounting |
| MLA or architecture-specific latent state | Rejected | complete latent-state schema and restore proof |
| Vision/media embeddings or multimodal cached state | Rejected | media identity and full embedding/state checkpoint contract |
| PLD/MTP/other speculative drafter enabled | Rejected | target and drafter cache/state checkpoint with temperature-zero identity |
| Continuous batching / merged slot caches | Rejected in phase 1 | atomic slot ownership transfer, cancel/failure cleanup, hostile compaction, A/B/A |
| Cold SSD snapshots | Phase 2 only | hot-path closure first; full blocks plus exact tail recomputation, protection and disk budget |

This matrix scopes the optional cache feature, not fast-mlx's model support. Unsupported rows retain
their existing cache-off behavior.

## Architecture

### 1. Pure policy plane (`HarnessCore`)

Add an exact, generic hot-cache index that owns only CPU metadata and an opaque actor-confined
payload:

- `ExactPrefixSemanticKey`: validated, path-free identities for privacy/isolation namespace, model
  instance/revision, tokenizer, prompt template, tools, KV route, position semantics, architecture
  state, and drafter.
- `ExactPrefixCachePolicy`: positive entry limit, positive retained-byte limit, minimum reusable
  tokens, and explicit disabled state.
- `ExactPrefixCommitDisposition`: success/failure/cancel/zero/pad/media and a single predicate that
  permits only a valid text success.
- `ExactPrefixCache<State>`: longest-prefix lookup, deterministic LRU, two-phase byte reservation,
  commit/rollback, touch, eviction, and invariant snapshots.
- `RequestStartMetrics`: distinct lookup/restore/template/tokenize/prefill timing and logical versus
  physical token counters.
- `TemplateTokenCache`: a separate exact entry/byte LRU with no MLX state.

The pure cache uses fake payloads in tests first. It must not import MLX or claim actual snapshot
bytes on behalf of the engine.

### 2. Actor-confined MLX snapshot plane (`SpikeCore`)

Add a phase-1 dense snapshot type and explicit methods on `CompiledKVCache`:

- capture only the logical prefix of K and V into detached evaluated arrays;
- record rank, batch, KV heads, token length, head dimension, dtype, and exact `nbytes`;
- restore into existing buffers with `_updateInternal`, zero the unwritten tail, and update the
  existing `offsetArr` identity and the `CompiledKVCache.offset` host mirror to the same logical
  length;
- reject uninitialized, mismatched layer/shape/dtype/capacity, non-finite, and partial snapshots;
- include the decoder's logical token count and next-token pipeline state;
- reserve/grow live capacity before restore, then rebuild the compiled closure only if an actual
  state-shape growth requires it.

`CompiledMLXDecoder` gains explicit cold-prefill, restored-prefix-plus-tail, snapshot, restore, and
reset transitions. Cold prefill stages a detached prompt-only checkpoint before the submit-first
pipeline writes the first generated token; the candidate is invisible until generation succeeds.
After success, the actor may also capture the exact final context. Publishing either snapshot still
requires a valid positive-commit disposition and an independent byte reservation. This makes both
an identical prompt and the usual prior-conversation-plus-new-tail request reusable without
allowing a failed generation to poison the cache. State-machine invariants are named and tested; a
restored decoder may not call the ordinary full-prefill entry as though it were empty.

Compressed caches do not inherit support merely because they share `CompiledCache`.

### 3. Loaded-model request-start plane (`fastmlx-harness`)

At prefix-cache-enabled load:

- derive and validate path-free source identities on both sides of model loading;
- admit only the source-locked phase-1 architecture matrix;
- create the cache inside `HarnessEngineActor`, keeping every MLX payload actor-confined;
- require the caller's hashed privacy/isolation namespace before lookup or commit;
- optionally perform isolated one- and eight-token eager warmup;
- route generation through lookup -> reset/restore -> tail prefill -> decode -> positive commit;
- expose typed scalar telemetry through `RunResult.engagement` and bounded proof evidence.

The tokenizer cache remains host-side because `EngineDriver.generate` already receives token IDs.
Its reusable component is built now; the production request adapter consumes it in the serving
route task.

### 4. Continuous-batch ownership seam

The scalar snapshot value is the future handoff unit. A later coordinator integration must:

1. choose the hit before slot admission;
2. reserve snapshot bytes and batch capacity atomically;
3. build a scalar cache from the detached snapshot, then merge it into the owned batch slot;
4. bind request revision so stale plans cannot restore or commit;
5. extract and commit only after the coordinator declares successful completion;
6. publish nothing on cancellation, disconnect, invalid decode outcome, or hostile compaction.

No continuous-batch flag is exposed in phase 1.

### 5. SSD phase 2

SSD work begins only after the hot path passes. It stores protected full blocks, recomputes the
partial tail exactly, uses independent disk entry/byte budgets, binds the same semantic key and
source identity, and never extends a live request's context window.

## Proof methods

### Pure off-box tests

- semantic-key inequality for every axis;
- identical tokens under different privacy/isolation namespaces never match;
- longest-prefix selection, exact tie/LRU behavior, and no cross-key match;
- two-phase reservation, oversized skip, eviction ordering, rollback, overflow, and disabled mode;
- success-only commit table including error/cancel/zero/pad/media;
- token/control bytes included in retained accounting;
- template/tokenize exact key, independent budget, and timing counters;
- A/B/A with fake state and concurrent logical callers;
- cold/warm metric derivation and impossible/partial evidence rejection.

Run the narrow tests first, then the full `HarnessCoreTests` suite.

### MLX-coupled Xcode tests on `llmbench`

- dense per-layer snapshot detaches from later mutation;
- restore preserves cache and control-array identities;
- restore updates graph and host offset state to one identical logical length;
- prompt-only candidates remain invisible until a successful generation commits them;
- snapshot -> reset -> restore -> append matches uninterrupted dense control logits;
- exact hit and `blockSize-1`, `blockSize+1`, multi-block-plus-tail boundaries;
- final instruction wholly in the tail;
- capacity growth/retrace boundary;
- mismatch/non-finite/partial snapshot rejection;
- temperature-zero cache-off/on byte identity;
- hostile A/B/A and multi-turn replay;
- `Memory.cacheLimit` and retained-array bytes;
- unsupported compressed, sliding, hybrid, vision, speculative, and batch configurations reject.

Sync with `spike/scripts/sync_llmbench.sh`, then use
`xcodebuild ... -skipPackagePluginValidation`; do not use SwiftPM CLI for MLX-linked tests.

### Bounded loaded-model evidence

Use fresh outputs and source-locked Qwen3, Llama, and Phi model identities independently:

- cold salted control;
- same prompt exact hit;
- long prefix plus short new tail;
- interleaved A/B/A;
- memory-pressure eviction;
- eager-warmup off/on;
- template/tokenize miss/hit.

Record exact source/binary/model/tokenizer/workload/policy hashes, prompt token counts, cache-read
and physical-prefill tokens, TTFT/prefill/decode timing, MLX active/cache/peak, RSS, entry/byte
state, evictions, output hashes, and fresh-output provenance. Diagnostic or partial output cannot
promote.

## TDD and implementation sequence

1. Add failing pure semantic-key, positive-commit, nearest-prefix, reservation/LRU, tokenizer-cache,
   and metrics tests.
2. Implement the smallest pure contracts; run focused and full HarnessCore tests.
3. Add failing MLX dense snapshot/detachment/in-place-restore tests.
4. Implement dense scalar snapshot/restore and decoder continuation state; run focused SpikeCore
   tests through Xcode.
5. Add failing actor request-start, A/B/A, exact-tail, unsupported-mode, and warmup tests.
6. Integrate the actor cache and honest telemetry; run focused harness tests, full Xcode suites, and
   Release build.
7. Build a bounded fresh-output loaded-model proof CLI/artifact; independently review before launch.
8. Run Qwen3 first, then source-locked Llama and Phi independently. Preserve negative cells.
9. Write the dated verdict, public fast-mlx-only content, verification packet, and handoff.
10. Focused correctness/security review, diff inspection, link check, secret scan, coherent commits
    with the required trailer, fresh clean-SHA proof, and `--no-ff` merge.
11. Continue the continuous-batching serving-route roadmap item, consuming but not weakening these
    cache contracts.

## Security, privacy, and operational boundary

Hot snapshots contain prompt-derived state. They never leave the process in phase 1, are bounded,
are not logged, and are dropped on model unload. Keys and evidence use hashes and counts rather than
prompt text. SSD snapshots require a separate storage-protection design and explicit phase-2
review. No connector, credential, install, privilege change, or external cooling/control is needed
for phase 1.

## Promotion rule

Phase 1 may close only when the cache-on path is exact, memory-bounded, observably engaged, and
faster on independently authenticated warm-turn evidence for each claimed model family. A correct
but non-beneficial row is retained as negative evidence. No scalar-model proof authorizes
continuous batching, compressed caches, recurrent models, SSD persistence, or a broad default.
