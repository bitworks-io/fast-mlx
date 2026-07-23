# Qwen3-32B 32K KVarN capacity-only verdict - ACCEPT context, forbid speed use

- **Date:** 2026-07-22
- **Engine source:** `8b454754ed9e631b05ce1164b9c30853b4e416f8` (clean)
- **Binary SHA-256:** `5d72ea1b6ab98749421c059984fe1ff69294d2971d35ba6557bb44c7033675a6`
- **Box:** Apple M5 Max, 128 GiB, macOS 26.5.2, 140 W AC, High Power Mode
- **Model:** Qwen3-32B-4bit, Q64/KV8/D128, maximum context 40,960
- **Workload:** 32,628 prompt tokens plus 128 generated tokens
- **Route:** KVarN i8 (`kvarn-k4v2-g128`), `split-kvarn-quantized-mm`
- **Decision:** **ACCEPT the measured 32K capacity/runtime context. Keep `promotable:false` and
  `speedAggregation:"forbidden"`; make no speed-tier or default claim.**

## Authenticated boundary

The immutable output is:

`/Users/llmbench/perf-work/results/fused-compressed-kv-qwen3-32b-loaded-8b45475/qwen-32k-kvarn-capacity-v20`

| Artifact | SHA-256 |
| --- | --- |
| Runner manifest | `f81a5d95dc2554d1ba749965bf49703fbe546ee14700147d4064d7cc9a980da2` |
| Reviewed launcher | `851a99f525be4c2d108d880dc636e3efad199cd81cf89f0d795b9c3b2aeda59c` |
| Capacity evidence row | `99932d17ec7014b80eb8f50a0f1393e565444cad4dc6e7629c4888c467cad59c` |
| Completion | `d81308c2e233679e2693564e59e23d44757a7be5ca35b6edd26d23d117964b88` |
| Launch receipt | `b47aeb543d20b42d12e1737e9d24108fa69e757895bda64487f2693c2b9331bb` |
| Final progress | `830874b2b57d757a06e6a96d0e851a2d47f4301d793f28f8c2a9b1227fbaf850` |
| Final status | `a4bccf5c13ef5ff8311e59b34d492bc7072a55298723601a6a9c127053aee8fc` |
| Runner log | `b948f95e44a679fd3af7f7cb1d964a1cbfbd85457e9b0eae2d28ada17a512337` |
| Time log | `e0cb79933feb988677ae28ba98ef22d519791c9c144765b952a7ac4e37d0ba17` |
| Validator log | `4c1ac0ecc101469994d690fe5c9740773c906d7ade562c704d2fe30a3e6a24c9` |
| Reviewed read-only authenticator | `08221e1629da4e2b5abe010a8a422abf5844b61eabafb743ebe5489d1ee4de1f` |
| Durable verification receipt | `168da54a58ef6b1b0aede8beda2564f4da24c232c9c8dc563933e11a8f6b328f` |

The typed `validate-bench-capacity` command returned `bench capacity evidence: VALID`. The
focused-review-clean read-only authenticator passed twice against the exact immutable row and
sidecars. Final progress and a separate read-only terminal monitor showed `COMPLETE` after 6,359
seconds with one row of one, no watchdog, no failure artifact, no live process, and no held lock.
The two PASS objects and normalized final-monitor output are preserved in
[`qwen3-32b-kvarn-capacity-verification-2026-07-22.json`](qwen3-32b-kvarn-capacity-verification-2026-07-22.json).

## Measured capacity context

| Metric | Retained value |
| --- | ---: |
| Prompt / generated tokens | 32,628 / 128 |
| Prefill / decode | 10.38 / 4.934 tok/s |
| TTFT | 3,144,859 ms |
| Cached / physical-capacity tokens | 32,756 / 33,024 |
| Compressed tokens / completed tiles | 32,512 / 254 |
| Engaged KVarN layers / materialization bytes | 64 / 0 |
| Payload / metadata / control bytes | 1,616,904,192 / 202,113,024 / 256 |
| Attention / total workspace bytes | 8,659,140,608 / 8,661,237,760 |
| MLX active / cache / peak bytes | 20,343,562,498 / 6,870,047,163 / 59,560,463,681 |
| Sampled physical footprint / process max RSS bytes | 64,618,487,912 / 55,875,010,560 |

The manifest bound a 96 GiB MLX memory limit, 8 GiB cache limit, and 115 GiB wired limit. Both the
dropped warmup and retained measurement remained AC, non-low-power, and nominal -> nominal. The
KVarN route engaged compiled split attention at all 64 layers with mixed bf16/fp32 ingress
normalization, bf16 native storage, zero full-cache materialization, finite telemetry, and exact
token/tile accounting.

## Claim boundary and next action

Timing is capacity context only. It does not revise KVarN's shelved Qwen speed role, enter a speed
aggregate, promote a dial tier, generalize beyond this checkpoint/hardware, or authorize lossy KV
with PLD. Qwen's 8K speed result remains negative, its 32K speed evidence remains unavailable under
the frozen unthrottled contract, and its near-128K request remains an authenticated model-limit
refusal.

The next model-family gate is Llama-3.3-70B-Instruct-4bit. The complete local snapshot was
subsequently source-admitted at exact revision
`de2dfaf56839b7d0e834157d2401dee02726874d`; see
[`2026-07-23-llama3-70b-source-lock.md`](2026-07-23-llama3-70b-source-lock.md).
Proceed with loaded 8K/32K/near-128K evidence. The Qwen-specific KVTuner schedule cannot be reused
without independent calibration.
