# Qwen3-32B EAGLE-3 Phase 0 verdict — RED / SHELVED

- **Date:** 2026-07-12
- **Harness:** `1a70c4d6b3dce20f3254226123dd588a3c80f052` (clean)
- **Draft checkpoint:** `RedHatAI/Qwen3-32B-speculator.eagle3` at
  `dc84fe7ff1db31efa824776f49c141fc8195eb47`
- **Targets:** authenticated Qwen3-32B 4-bit and 8-bit MLX checkpoints
- **Hardware/runtime:** Apple M5 Max 128 GB; MLX 0.32.0; mlx-lm 0.29.1
- **Evaluation lane:** [`EXACT`](README.md)—speculation may change execution cost, not the
  greedy target result
- **Decision:** **SHELVE this production port. Do not start the Swift EAGLE path.**

The EAGLE head port is faithful, engaged, and apparently fast. It is also not exact. Both
target pairings change the greedy byte stream on a 64-token code prompt. fast-mlx treats a
lossy speculative path as a correctness bug, so Phase 0 stops before the multi-workload
throughput bench and no speed number below is promotion evidence.

## How to read this verdict

This is an `EXACT`-lane failure, not a policy against lossy optimization. The fast-mlx dial is
intended to offer real speed and capacity gains with real, quantified quality loss when an
intentional approximation clears the coherence/garbage floor and occupies a useful point on
the measured frontier. Weight/KV quantization, approximate attention, pruning, and selective
cache policies belong in that `LOSSY_FRONTIER` lane.

EAGLE-3 does not. Greedy speculative decoding claims to reproduce the base target while doing
less target work; byte identity at temperature zero is therefore its correctness contract.
The observed mismatch is shape/history-dependent target behavior, not a calibrated loss knob
with a teacher-forced quality estimate. Relabeling it as lossy would also make the apparent
rate invalid because the changed continuation can alter later work and stopping. A separately
designed approximate decoder would be a new technique requiring its own lossy-frontier plan,
quality measurements, and product guardrails.

## Gate results

| Gate | Result | Clean-SHA evidence |
| --- | --- | --- |
| Accounting contract | **PASS** | 119 HarnessCore XCTest + 17 Swift Testing tests pass; proposal acceptance, accepted drafts/round, and inclusive length remain distinct; zero-time economics now fail closed. |
| Checkpoint authenticity | **PASS** | Config SHA-256 `eaeecf…fbd`; full 3,121,274,856-byte weight SHA-256 `e63437…26b8`; 16 exact tensor names, dtypes, shapes, and non-overlapping payload ranges. |
| Target authenticity | **PASS** | 4-bit: 4 shards / 18,429,850,874 bytes / manifest `a14eff…c5a`; 8-bit: 7 shards / 34,810,574,466 bytes / manifest `64b300…7ce`. Every shard was fully hashed before loading. |
| PyTorch/speculators → MLX head parity | **PASS** | Cosine `0.9999815822` (>0.99), argmax match `1.0`, max absolute error `0.0625`. Fixture versions: speculators 0.6.0, Torch 2.12.0, Transformers 5.10.4, safetensors 0.8.0. |
| 4-bit greedy exactness, `k=1` | **FAIL** | First token mismatch at generated index 17; token and decoded-byte hashes differ. Engagement is non-vacuous: 23/39 drafts accepted across 39 rounds. |
| 8-bit greedy exactness, `k=1` | **FAIL** | First token mismatch at generated index 7; token and decoded-byte hashes differ. Engagement is non-vacuous: 25/38 drafts accepted across 38 rounds. |
| Pairing economics / `k=3` / long-output matrix | **NOT RUN** | Exactness is an earlier hard gate. Benchmarking a different byte stream would create a misleading speed claim. |
| BF16 diagnostic | **NOT RUN** | No Qwen3-32B BF16 target is staged. Quantized-target acceptance is healthy; fidelity is proven; the blocking failure is target computation equivalence, not low draft acceptance. |

## What changed the token stream

The harness replays the first mismatch through four cache histories, with identical cache
offsets before the probe.

### 4-bit: rejected-future cached-state drift

At generated index 17, the baseline expects token `12`. From the same 59-token cache offset:

| Replay | Target argmax |
| --- | --- |
| all-sequential history, one-token probe | `12` |
| all-sequential history, `[current,draft]` probe | `12` |
| retained-only batched history | `12` |
| full verify batches followed by rollback | **`44364`** |

The draft token is correctly rejected, but processing rejected future tokens in earlier verify
batches changes the retained target state enough to flip a later greedy selection. This is not an
offset error: every replay reports offset 59.

### 8-bit: immediate batched-probe drift

At generated index 7, the baseline expects token `279`. From the same exact sequential prefix
and cache offset 49, a one-token probe predicts `279`; probing `[current,draft]` predicts
**`264`**. The retained-only history also predicts `279`, while the live full-verify history
predicts `264`.

This directly demonstrates shape-sensitive argmax behavior for this target pair. It does not
prove a general MLX defect. MLX documents equivalence only “up to numerical precision” and
notes that input-shape changes can select new compilation work; upstream EAGLE documentation
defines a batched target verify and longest-prefix accept, but does not promise byte identity
against a separately shaped autoregressive kernel. Those sources make finite-precision
shape/order effects a plausible mechanism; the clean replay is the evidence for this case.

## Why the apparent speed is not a result

The failed verification runs observed 27.59 → 38.59 tok/s on 4-bit and 15.06 → 23.90 tok/s on
8-bit. Those figures are diagnostic only. The candidate generated different tokens, and a
different continuation can change both model work and stopping behavior. Reporting either as
a multiplier would violate the same-target, same-output benchmark contract.

The acceptance counters are still useful diagnostically: 0.590 and 0.658 accepted drafts per
round (inclusive lengths 1.590 and 1.658). They also illustrate why model-card
`acceptance_length` cannot be compared directly with another pair's break-even: the inclusive
metric contains the target correction/bonus token, while the economic yield is accepted draft
tokens per verify round.

## Decision and recovery gate

This checkpoint/target path is **SHELVED**, not promoted and not ported to Swift. Reopen only
with one of these bounded changes:

1. a target verify kernel whose argmax **and retained KV state** are invariant between the
   autoregressive and multi-token shapes used here;
2. an exact cache repair/commit strategy that replays retained tokens and still beats the
   same target baseline after its added target work is charged;
3. a newer MLX/target conversion that passes byte identity for `k=1`, `k=3`, long output, and
   the full workload matrix; or
4. a product-size DSpark/DFlash/native-MTP pairing with its own authenticated checkpoint and
   exact same-target gate.

The next actionable flywheel cycle is continuous batching plus decode-first chunked prefill.
The Qwen3-32B DSpark and DFlash controls remain blocked by missing compatible checkpoints;
cross-model raw tok/s is not a substitute.

**Subsequent status (2026-07-14):** continuous batching completed its exact engine/policy gate;
the live queue in `docs/agent-handoff.md` now advances to KVarN/asymmetric affine.

## Evidence artifacts

| Bench artifact | SHA-256 / content ID |
| --- | --- |
| `final-v2-checkpoint.json` | `a87be26cac76fb36f5b633d8f8ebd0671bdd5945eacb66485057eeaa7dd7e34c` |
| `final-v2-parity.json` | `68c6c925c9ef7e1d911114cca482e7c20ba8924ae35e01fc1795c8336267f1b3` |
| `final-v2-verify-4bit.json` | `e9edd36b3dc1c8574fb671feb6bc70e7b9f62df60c7117969e65709de7bd23d4`; evidence ID `d0f6025ff0c122ab4139e16c1b8cb4b1b9309274cd2a9c2f0e1a2a3c3b3eb942` |
| `final-v2-verify-8bit.json` | `e4191c59a840b60cf18a75ad7ebe0ed16a48a562280a7b8bd9b32fa4467c4686`; evidence ID `0e540b3df5df24d75728ac38a8230da3cdf797c0a739030dfe2dd7dab5e2fff2` |
| Compact committed extract | [`eagle3-evidence-2026-07-12.jsonl`](eagle3-evidence-2026-07-12.jsonl) |

## Primary sources and claim boundaries

- [Speculators EAGLE-3 inference process](https://github.com/vllm-project/speculators/blob/d1a3ff3ed6a48f990584f56efbb06f990e1c7ab2/docs/user_guide/algorithms/eagle3.md#inference-process): batched target verification and longest-prefix acceptance.
- [mlx-lm 0.29.1 `KVCache.trim`](https://github.com/ml-explore/mlx-lm/blob/f3ed856610d3852e41d691b8968021040f9c4a6b/mlx_lm/models/cache.py#L307-L376): standard trim decrements the cache offset; it does not establish cross-shape numerical identity.
- [MLX compilation documentation](https://github.com/ml-explore/mlx/blob/4367c73b60541ddd5a266ce4644fd93d20223b6e/docs/src/usage/compile.rst#L39-L76): compiled/uncompiled equality is stated up to numerical precision and input-shape changes can trigger compilation work.

**Source-review boundary:** upstream “lossless” describes the speculative acceptance
algorithm. It does not confirm byte-identical output across different numerical target
kernels. fast-mlx's byte-identity requirement is a stricter local product invariant, proven or
failed only by the same-target bench.
