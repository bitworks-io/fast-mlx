# Exact prefix/session cache verdict — Qwen and Phi model-scoped promotion

- **Date:** 2026-07-24
- **Lane:** exact scalar request-start reuse; internal engine surface
- **Classification:** EXACT
- **Engine source:** `ecd2915a0c4cd036169c7865db9dd07df8b68ca8`
- **Decision:** **Promote the bounded exact-prefix path for the authenticated Qwen3-32B and
  Phi-4-mini snapshots. Reject Llama-3.3-70B on the measured 128-GiB host. Keep the feature
  disabled by default and out of the production serving route until that separate gate closes.**

## Operator story and acceptance

A scalar chat or coding-agent operator repeatedly extends one source-locked conversation. A warm
turn should restore the longest exact actor-owned prefix, evaluate only the new tail, and return
the same temperature-zero bytes as a cold request. The optimization must not cross isolation or
semantic boundaries, publish failed work, change compiled cache identities, retain an unbounded
unified-memory working set, or claim support for state it cannot fully restore.

Acceptance required:

- cache-off and cache-on generated token and output hashes match for exact hit, partial tail,
  A/B/A return, pressure eviction, and eager-warmup cases;
- every reused prefix identifies the actual prompt-only or final-context source case, token count,
  and token SHA-256;
- exact hits report zero physical prompt-prefill tokens and partial hits report only the uncached
  tail as physical work;
- entry, retained-byte, MLX cache, MLX peak, and sampled-footprint limits all hold;
- every warm request-start comparison is strictly faster than its frozen control;
- each model family carries independent source, checkpoint, tokenizer, binary, workload, and
  fresh-output authentication;
- any terminal, memory, geometry, route, evidence, or identity failure remains non-promotable.

The happy path is a successful cold commit followed by an exact or nearest-prefix hit under the
same semantic key and isolation namespace. The recovery path for a mismatch is an ordinary cold
request with no publication. Unsupported compressed, sliding/recurrent/hybrid/MLA, vision/media,
speculative, and continuous-batch state still fails closed.

## Clean verification

The schema-v3 selected-prefix fix is clean commit `ecd2915`. It preserves production
longest-prefix selection and changes proof provenance so a hit may bind either the prompt-only
snapshot or the longer successful final-context snapshot that the runtime actually selected.
Schema 1 and 2 remain readable; only schema 3 can promote the new source binding.

The clean verification boundary is
`/Users/llmbench/perf-work/results/exact-prefix-session-cache-ecd2915/clean-verification-v1`.
Its independently authenticated receipt SHA-256 is
`9a5bf6d29247bc68768d4404e80f9dbb4161ece58f3844b1e9a9c2fde75eb814`.

| Proof | Result |
| --- | --- |
| Pure contracts | 584 HarnessCore XCTest + 17 Swift Testing tests, zero failures |
| Harness integration | 140/140 `FastMLXHarnessTests` |
| Dense MLX snapshot/restore | 8/8 `ExactPrefixMLXTests` |
| Apple engine regression | 163/163 `SpikeCoreTests` |
| Release build | succeeded with `-skipPackagePluginValidation` |
| Clean binary | `477ab6b27dc8042bba393c586b069bf70ea4b230aa5d6400447757141e8e2cdf` |

Focused review found no code or launcher issue. ShellCheck passed both fresh launchers, and the
prelaunch guards independently bound source stamp, binary, launcher, model/tokenizer/checkpoint,
High Power Mode, 140 W AC, Foundation low-power/thermal state, competing-process absence, and
fresh output paths.

## Loaded model results

All percentages below compare request-start time, which is the cache's operator-facing latency
surface. They are bounded proof measurements, not general throughput claims.

| Model | Exact hit vs cold | Partial hit vs control | Post-warmup hit vs control | Maximum sampled footprint | Disposition |
| --- | ---: | ---: | ---: | ---: | --- |
| Phi-4-mini-instruct-4bit | 36.14% faster | 4.88% faster | 27.23% faster | 7,566,578,440 B | **promote, model-scoped** |
| Qwen3-32B-4bit | 76.67% faster | 29.28% faster | 77.00% faster | 58,312,761,976 B | **promote, model-scoped** |
| Llama-3.3-70B-Instruct-4bit | diagnostic only | diagnostic only | diagnostic only | 122,953,061,984 B | **reject on this host** |

### Phi-4-mini

The fresh output at
`/Users/llmbench/perf-work/results/exact-prefix-session-cache-ecd2915/phi4-mini-v2`
completed 11/11 schema-v3 cases with `promotable:true`. It proved homogeneous observed
`float16` dense state under a source config that declares `bfloat16`, exact in-place restore,
byte identity, pressure eviction, eager warmup, template/token reuse, and bounded retention. The
maximum retained prefix state was 185,633,120 bytes.

- evidence:
  `23ae9dae913f00d2bfcc3fb3370910c10044c4ca0c319d5be35d4d9aa5458c0d`
- completion:
  `9ea32d070570975890695df7f6b5caaa17edd401fbbede7ba0d903be70d99688`
- finalization:
  `df3c0e678ed71e3c47db648e2b20577dac45f1bf46ada7b749b14656b8b88478`
- launch receipt:
  `5bba1502843759ab886bf077c37a5bf24e920152d3722f9dd04d6da2293d47f5`

### Qwen3-32B

The fresh output at
`/Users/llmbench/perf-work/results/exact-prefix-session-cache-ecd2915/qwen3-32b-v2`
completed 11/11 schema-v3 cases with `promotable:true`. The previously failing partial case now
authenticates the runtime's real selection: `partial-hit` reused the 202-token
`cold-commit-A` **final-context** snapshot, and its reused-prefix SHA-256 equals that source
case's final-context SHA-256. It physically prefetched only the remaining 209 of 411 prompt
tokens and returned the exact cache-off output. The maximum retained prefix state was
434,169,120 bytes.

- evidence:
  `1c19b0b0ded991d4b089c4227da8c550d13f5b552ca25c1365d9ce784942c9be`
- completion:
  `c166f3b1aa1334dbd1de440edf5fbf3a6e53180bbbf19bde7cfe4d3e864fe369`
- finalization:
  `52fe5176cf36b1382acbae50913cdc7dd4235f2ed4e99d39cf00e0b40402bd24`
- launch receipt:
  `d032827e4756f4ef1787e6faf62381a008ed0ae84b58d67130b3abb7cc43a83f`

### Llama-3.3-70B

The fresh boundary at
`/Users/llmbench/perf-work/results/exact-prefix-session-cache-ecd2915/llama3-70b-v2`
is terminal `FAILED` and must remain unchanged. The harness published 11 diagnostic rows that
were byte-identical, engaged, and faster on warm request-starts, but its own derived packet set
`boundedPassed:false` and `promotable:false`. Maximum per-case sampled footprint was
122,953,061,984 bytes against the explicit 103,079,215,104-byte proof limit. Full checkpoint
content revalidation then raised the process footprint to 162,584,577,424 bytes on a
137,438,953,472-byte host, and macOS killed the process with signal 9 before terminal status,
validation, or finalization.

This is an authenticated negative boundary, not a nearly passing promotion:

- launcher failure:
  `e550e12bd9486fbc0701bc7b86e379d4833634212aa66e960f8d05083f656500`
- launcher log:
  `116d185ced4c3cfd5d32517e6a008d2afec04f7c05898612d997519d04475b34`
- diagnostic evidence:
  `a7827f920741612da3b4356169990cc0fd4c26afd7cd556e9b39bec56e68962c`
- diagnostic completion:
  `be9a80f9bbdf759575462bbb811a51acf4b085bb08cdd712af2050cee8683e0e`
- non-terminal harness status:
  `d2eaded830817a6b6b8dbea4760957dd08cc0f4ec4e7873c62d2f0d480c68d55`

No unchanged retry is justified. A future Llama-70B qualification must first make full-content
source revalidation memory-safe and establish a reviewed memory policy that fits this host; it
must use a new clean source, binary, nonce, and output boundary.

## Support and scope matrix

| State/model class | Disposition |
| --- | --- |
| Source-locked Qwen3 dense GQA, native scalar `CompiledKVCache` | promoted for the authenticated model snapshot |
| Source-locked Phi3 LongRoPE/inert-window geometry, native scalar `CompiledKVCache` | promoted for the authenticated model snapshot |
| Source-locked Llama-3.3-70B on the measured 128-GiB host | rejected; cache-off remains available |
| Other dense full-attention models | implementation-compatible, but no model-scoped claim without independent proof |
| Compressed KV, sliding/rotating, recurrent/hybrid/MLA, vision/media, speculative, batched slot state | rejected in phase 1 |
| Production API route, public/default switch, SSD snapshots | separate roadmap gates |

These dispositions scope the optional exact-prefix feature. They do not narrow fast-mlx's normal
cache-off model support.

## Verdict and next action

The exact-prefix/session cache closes **PASS, MODEL-SCOPED** for Qwen3-32B and Phi-4-mini. Both
materially different families prove temperature-zero identity, selected-source provenance,
bounded retention, exact physical-work accounting, and strict warm request-start benefit. Llama
remains an honest memory-bound negative cell.

No broad/default or production-serving claim follows. The next ranked roadmap item is the
continuous-batching OpenAI-compatible serving route, which may consume these scalar snapshot
contracts only after proving atomic slot ownership, disconnect/cancellation cleanup, stale-plan
rejection, and the existing hostile-compaction invariants.

Machine-readable criterion mapping:
[`exact-prefix-session-cache-verification-2026-07-24.json`](exact-prefix-session-cache-verification-2026-07-24.json),
SHA-256 `a76dc39f30cd4c213c56a1346b8bb3671002bd63af4158f3b4c83917f7601a53`.
