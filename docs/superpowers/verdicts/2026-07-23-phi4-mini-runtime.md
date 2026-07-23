# Phi-4-mini runtime verdict - no speed/default claim, close third geometry

- **Date:** 2026-07-23
- **Lane:** model-scoped runtime qualification; non-promotable
- **Model:** `mlx-community/Phi-4-mini-instruct-4bit`
- **Model revision:** `ac1c269cb4222a4e136a3d09edad301056c1f36a`
- **Geometry:** Phi3 Q24/KV8/D128, 32 layers, partial RoPE, source-locked inert window
- **Checkpoint:** `c5ccdee8b3d37fd42c7e42e3e22c47c7549cdaa0b82904bdbfb1a52b31af5ec8`
- **Tokenizer:** `7d8143e8d1f217f392488bddfb4febf2a53b783a2a00bf698b29218865a3620a`
- **Decision:** **Accept this exact source-locked Phi3 geometry as loaded-model closure. Reject
  speed, default, broad-family, and generic sliding-window claims.**

## Operator story and acceptance

A long-context Apple-Silicon operator needs the third model geometry to be qualified without
turning fit, a fast decode-only row, or a narrow source-locked exception into a broad product
route. The useful outcome is a model-scoped closure that says what passed, what failed, and which
claims remain forbidden.

Acceptance required:

- bind the model cycle to exact engine source, revision, checkpoint content, tokenizer, and source
  lock receipt;
- prove source, registry/load, runtime, and failure-path support only for the exact source-locked
  inert-window Phi3 geometry;
- retain generic sliding-window rejection outside that source-locked geometry;
- require affine direct to beat both fp16 and materialize controls before any 8K speed tier exists;
- keep 32K and near-128K affine rows capacity-only with `speedAggregation:"forbidden"`;
- reject quality/task promotion when fp16 is not an admissible reference and affine replay is
  refused before model load.

The representative happy path was exact model admission plus loaded affine direct execution with
zero materialization across all 32 layers. The relevant failure paths were the 8K dominated speed
gate, generic sliding-window refusal, non-promotable long-context capacity rows, material affine
KL drift, and task replay refusal caused by an inadmissible fp16 reference.

## Authenticated boundary

| Artifact | SHA-256 |
| --- | --- |
| Source-lock receipt | `f5abbbfd3d6989488160c9e5cc775c52aa928881786a6bc8893c48e087d61c2f` |
| Source-lock completion | `24c9afffd5b5b741e6845dded295eb36974dc6ee4575b73aae93a69796ab9ece` |
| Source-lock authenticator output | `bff714f37f70e75c8a39df3e34fccd24495b1abe6ee635d090f8481ce978ff05` |
| 8K auth receipt | `ced79c0b3d8fc959ef1c83997817b5f02f58da8007584b67fe5b111b79533337` |
| 8K manifest | `c6ebf9eee6a8fa752021405a3259d4d6954509ad8987422bda0f96a48e98b61a` |
| 8K completion | `976051ffaf75b5cfdf19896c892b0bc0b2ff25fcd62b1dade52ff3488a5f2a38` |
| 8K receipt set | `f1f518288526ead47d86c37bd13e380e8275255748edf8532c8543aa34ab425d` |
| 32K capacity auth receipt | `7251d261075fd14a32b26643d717c7f1a6ab3b246cea28c7b67beb47fe556140` |
| Near-128K capacity auth receipt | `eb9ce27f21af30c00bce8cb1aefe8a8a876ecbfde645012ad9592420e5e4a72a` |
| Sealed teacher manifest | `7d3263012f891d2eb84f4063efabd07189c44dcf916dae7b62246422191d773d` |
| fp16 quality status | `bcd57248091f2b6bd1aeb0a7b5ffd889221f2beb960ac7bf8a9a21823372cbb9` |
| fp16 quality evidence | `a55af6f14b65f4b41c74aee8c680b6dc3a2a20893a6b60ef7ebc99196e78b3e8` |
| Affine quality status | `abaacaa096ac1b5b6a57d1606f8c92623f609a908f3fd58ad06f7262d0977b8b` |
| Affine quality evidence | `99b0b19d0ad80030ce81793b357fbf998cce0b7d5049d47d24aa984672158050` |
| Task boundary raw | `fcbd16fa8f24d05de6f50459394ac658d55cf2025a999b139b89ad9765e486c2` |
| Task boundary summary | `b2599cee31fa0f76f3a474c41fa5d5375c4cf20b2e7bfa8964df238b3b2555e0` |
| Task boundary status | `846b51dde0a0c8f5507f4d7e8ef5185bb465938c9374020da007e53162b64820` |
| Independent task auth receipt | `a2e3f58cde07b760841866bfd1e362efe0a3665e3c8a1a4d1573d49d0d6f5ee7` |

Machine-readable criterion mapping:
[`phi4-mini-runtime-verification-2026-07-23.json`](phi4-mini-runtime-verification-2026-07-23.json).

The source sequence for this closure is:

| Phase | Engine source |
| --- | --- |
| Source-locked runtime admission and failure paths | `850997e93e37575f4922bf213d558d63c80f1c40` |
| Loaded speed/capacity evidence contract | `9c542ef1d681ee1ea93d22958885ff6a41072a50` |
| Deep measurement corpus | `19cbb8e7363e0e37ff2cf095406f6876bf28215a` |
| Corrected LongRoPE sealed quality/task reference | `e29ee4cac7894a10cc6dc532a41f8b5d27f7c034` |

## Source and support boundary

The source lock authenticates the pinned public revision, checkpoint content, tokenizer, and Phi3
geometry. The model-support gate passes for the exact source-locked inert-window geometry: Q24
query heads, KV8 grouped heads, D128 head dimension, 32 layers, and Phi3 partial RoPE. That pass is
not a generic sliding-window feature. The generic sliding-window request path still rejects before
model promotion because the accepted configuration is exactly `sliding_window == 262144` above
`max_position_embeddings == 131072`, with no active `use_sliding_window`, for this locked model.

## 8K loaded speed result

| Cell | Prefill tok/s | Decode tok/s | Retained interpretation |
| --- | ---: | ---: | --- |
| fp16 | 3540.95 | 124.80 | same-model runtime baseline |
| Affine K4V2-g64 materialize | 3538.09 | 90.57 | same-storage control |
| Affine K4V2-g64 direct | 2011.61 | 148.54 | direct path; failed speed gate |

Affine direct improved decode by 19.022% versus fp16 and 64.006% versus the materialize control,
but regressed prefill by 43.190% versus fp16 and 43.144% versus materialize. The unchanged speed
gate requires a non-dominated win, so this cell is negative/dominated. It proves the split direct
path engaged; it is not a speed tier or default route.

## 32K and near-128K capacity

| Boundary | Prompt + output tokens | Prefill tok/s | Decode tok/s | Layers | Materialization | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 32K affine direct | 32,628 + 128 | 665.969 | 106.341 | 32 | 0 bytes | non-promotable capacity |
| Near-128K affine direct | 130,944 + 128 = 131,072 | 178.917 | 45.028 | 32 | 0 bytes | non-promotable capacity |

Both rows are retained as loaded capacity/runtime context only. Both forbid speed aggregation.
Neither repairs the 8K speed-gate failure, authorizes a default route, or generalizes beyond this
source-locked Phi3 checkpoint and hardware boundary.

## Quality and task evidence

| Evidence | Median KL | Pooled p95 | PPL delta | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 0.0000431722755329472 | 0.00013062137410403349 | +0.012199% | 0.9969512 |
| Affine K4V2-g64 direct | 0.2148780525232595 | 1.4232502580311548 | +33.1268966% | 0.77439024 |

The deepest scored teacher-forced context was 27,145 tokens across 328 positions. The affine row
clears the frozen non-garbage hard floor: candidate/reference perplexity remains below 2x,
long-context tail p95 is `1.617082` against a 5-nat ceiling, top-1 remains above 50%, and depth
exceeds 24,000 tokens. The material drift is still an explicit warning and cannot become a model
quality tier without the separate paired task gate.

The task boundary scored 8/20 math, 1/20 code, 15/20 structured, 20/20 long, and 15/20 syntax.
The fp16 reference is inadmissible, so the affine task replay correctly refused before model load.
The task evidence therefore closes as authenticated refusal, not as a candidate comparison.
Because the paired task gate is unavailable, this model cycle cannot expose affine as a
user-selectable quality tier even though the teacher-forced row remains above the non-garbage
floor.

## Verdict

- **Source-locked Phi3 geometry closes.** Exact source, registry/load, runtime, and failure-path
  support pass for this inert-window Phi-4-mini snapshot.
- **Generic sliding-window support remains rejected.** The accepted path is not a broad
  sliding-window implementation.
- **No speed tier.** The 8K direct row is decode-positive but prefill-dominated under the frozen
  gate.
- **Capacity context only.** The 32K and near-128K affine rows are useful loaded evidence with zero
  materialization across 32 layers, but both are non-promotable and `speedAggregation:"forbidden"`.
- **Teacher-forced floor passes; no quality/task tier.** Affine remains above the frozen
  non-garbage KL floor with material measured drift, but task replay refuses because the fp16
  reference is inadmissible. Without a paired task comparison, the candidate cannot become a
  user-selectable tier.
- **No default or broad claim.** This is one source-locked Phi-4-mini/Phi3 checkpoint on one
  hardware boundary. It closes the third-geometry model cycle without promoting a speed/default
  route.
