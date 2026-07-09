# Harness spine — first end-to-end run (2026-07-09)

Branch `feat/harness-spine`, Tasks 7–8 of [the harness-spine plan](../plans/2026-07-09-harness-spine.md).
Box: `llmbench@192.168.1.252` (M5 Max 128GB), Release build via
`xcodebuild -scheme fastmlx-harness -configuration Release -skipPackagePluginValidation`.
Reference stack: `~/harness-venv` — mlx_lm 0.29.1, mlx 0.32.0 (python) vs mlx-swift 0.31.6,
transformers 4.57.6. Note the python/Swift **mlx versions differ** — cross-implementation
float noise below includes that.

## What each subcommand proved

### `corpus` — hermetic invariants (no model)

PASS, 6/6 entries (plain, think-strip, tool-lift, control-tag guard, hostile bytes,
unclosed-think truncation). Universal invariants (no `<|` leak, tool-args valid JSON) hold.

### `verify` — the triad (equivalence + engagement + acceptance)

| Model | Prompt | Result |
|---|---|---|
| Qwen3-30B-A3B-Instruct-2507-4bit (MoE) | "The capital of France is", n=60 | identical-prefix **60/60**, engagement decode 0→60, triad **PASS** |
| Qwen3-32B-4bit (dense) | same, n=60 | identical-prefix **60/60**, triad **PASS** |

Matches the spike's known-good equivalence result exactly. Acceptance reported n/a (no
spec-decode configured). Prompts cross to the reference as **token ids**, and the Swift-side
eos id is passed down, so tokenizer/stopping mismatches cannot fake a pass.

### `bench` — stream-timed decode (Release-guarded, warmup-dropped, salted)

Qwen3-30B-A3B-2507-4bit, max-tokens 256, 3 runs after warmup:

```
label,workload,mode,model,decode_tok_s,ttft_ms,quant,concurrency
harness-firstrun,decode,none,Qwen3-30B-A3B-Instruct-2507-4bit,156.11,54.3,int4,1
```

Individual runs 156.16 / 156.11 / 156.06 (<0.1% spread) — **156.11 tok/s**, at parity with the
spike's compiled-decode 155.5 and same-session Zig 153.65–155.6. The harness drives the same
compiled-step + no-sync-readback core through `EngineDriver`, so the seam costs nothing
measurable.

### `kl` — KLDivergenceMetric, first measured numbers

`logprobs` contract held on both sides: full-vocab (151936) **raw logits, token-id order**
(index == token id), not top-k, not softmaxed; logits cross the process boundary as raw
little-endian float32 (exact bytes, no text round-trip).

**Ordering spot-check** (same weights both sides, position 0, shared context): argmax token id
identical (12095 both sides); raw logits at sampled ids (0, 1000, mid-vocab, last, argmax, eos)
agree within |diff| ≤ 0.28 on values of magnitude up to ~26 — cross-implementation fp16
reduction-order noise, no ordering skew. On 32B dense the argmax logit matched to the exact
fp16 value (diff 0.0000).

**Run 1 — pipeline proof (honest label: NOT a quantization-loss number).**
Candidate Swift engine and reference mlx-lm on the **same** Qwen3-30B-A3B-2507-4bit weights;
3 prompts x 24 positions:

| stat | value |
|---|---|
| kl_median, all 72 positions (the committed metric) | 4.23e-03 nats |
| kl_median, context-aligned positions only (51) | **2.80e-03 nats** |
| kl_p95, all positions | 2.05e+01 nats (see caveat) |

Same-weights median KL of ~3e-03 nats is pure cross-implementation float-reduction noise —
the pipeline (two engines → full-vocab logits → softmax → KL) demonstrably measures a near-zero
quantity as near-zero. The 32B-dense same-weights control reproduced this:
aligned median **2.14e-03 nats**. That is the instrument's noise floor.

**Run 2 — stretch, a real first dial point (Qwen3-32B: INT4 candidate vs INT8 reference).**
Swift `Qwen3-32B-4bit` vs mlx-lm `Qwen3-32B-8bit`, 3 prompts x 24 positions:

| stat | value |
|---|---|
| kl_median, context-aligned positions only (8) | **4.28e-01 nats** |
| kl_median, all 72 positions | 1.42e+01 nats (mostly diverged-context, see caveat) |

The aligned 4-bit-vs-8-bit KL (0.43 nats) sits **~200x above the same-arch noise floor**
(2.1e-03 nats) — a genuine quantization-loss signal, with honest limits: only 8
context-aligned positions (the two quantizations' greedy paths diverge after 1–2 tokens on
these short, high-entropy raw prompts), the reference is INT8 (a proxy for fp16, itself
quantized), and single short-prompt corpus. Treat it as the first dial point proving the
instrument, not a benchmark of Qwen3-32B INT4 quality.

## Caveats and known limits (read before quoting numbers)

- **Per-position KL beyond the shared greedy prefix compares different contexts.** Each driver
  follows its own greedy path; position k's distributions are only exchangeable while the two
  token streams still agree (positions 0..prefix). The committed `KLDivergenceMetric` includes
  post-divergence positions, which is why its all-positions median/p95 explode whenever paths
  split early (haiku prompt: prefix 2/24 even same-weights — the spike's documented near-tie
  prompt). The CLI therefore prints the **context-aligned median** alongside; that is the
  number to quote. Follow-on: add a teacher-forced `logprobs` mode (score a fixed continuation)
  so every position is context-locked by construction — the right shape for the dial's
  production measurement.
- Near-tie argmax flips between implementations are real and expected (spike backlog); they cap
  the usable aligned-position count on high-entropy prompts.
- Python mlx 0.32.0 vs mlx-swift 0.31.6: version skew is inside the noise floor we measured,
  but pin-parity is worth restoring when convenient.
- The mlx-lm reference venv lives at `~/harness-venv` on llmbench — deliberately OUTSIDE
  `~/fast-mlx-spike`, because the rsync deploy uses `--delete` (the spike-era in-tree venv was
  wiped by exactly that).

## Reproduce

```sh
rsync -az --delete --exclude .build spike/ llmbench@192.168.1.252:~/fast-mlx-spike/
ssh llmbench@192.168.1.252 'cd ~/fast-mlx-spike && DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -scheme fastmlx-harness -destination "platform=macOS" -configuration Release -skipPackagePluginValidation build'
# BIN=~/Library/Developer/Xcode/DerivedData/fast-mlx-spike-*/Build/Products/Release/fastmlx-harness
$BIN corpus
$BIN verify --model ~/perf-work/models/Qwen3-30B-A3B-Instruct-2507-4bit --n 60 --min-prefix 30
$BIN bench  --model ~/perf-work/models/Qwen3-30B-A3B-Instruct-2507-4bit --max-tokens 256 --runs 3
$BIN kl     --model ~/perf-work/models/Qwen3-30B-A3B-Instruct-2507-4bit --positions 24
$BIN kl     --model ~/perf-work/models/Qwen3-32B-4bit --reference-model ~/perf-work/models/Qwen3-32B-8bit --positions 24
```

Pure-core tests: `cd spike && swift test --filter HarnessCoreTests` → 14/14 pass (off-box).
Swift 6 strict concurrency clean; zero `@unchecked Sendable` / `nonisolated(unsafe)` (one
sanctioned `sending` model transfer into the engine actor, as in the spike).
