# Direct KVarN loaded-speed verdict - SHELVE speed, retain capacity

- **Date:** 2026-07-21
- **Engine source:** `d4102e6a3029b161d99ee27aceabbad8d5696fb5` (clean)
- **Binary SHA-256:** `cfe029ad2138013a5904e6afd2475a881081a37bcf89db25bfbb91abf8484397`
- **Box:** Apple M5 Max, 128 GiB, macOS 26.5.2, 140 W AC, High Power Mode
- **Model:** Qwen3-32B-4bit, Q64/KV8/D128, 8K workload
- **Lane:** `LOSSY_FRONTIER`; diagnostic timings cannot promote a cell
- **Decision:** **SHELVE KVarN's speed role for this model cycle. Retain its already-qualified
  capacity-only Max-fit role and correctness implementation.**

## Why this is an engineering decision, not a promotion result

The loaded v11 preflight retained authenticated fp16 and KVarN-materialize rows. Its direct KVarN
row failed the exact retained thermal-equality contract and therefore emitted no promotable
evidence. The complete hash-bound runner log still records that failed row's diagnostic timings.
They can decide whether further engineering has a plausible envelope; they cannot advertise a
speed tier or enter a public benchmark aggregate.

| Cell | Prefill tok/s | Decode tok/s | Evidence class |
| --- | ---: | ---: | --- |
| fp16 | 533.73 | 23.48 | authenticated v11 row |
| KVarN materialize | 236.92 | 0.46 | authenticated v11 row |
| KVarN direct | 63.26 | 7.18 | failed-row diagnostic only |

Direct attention improved decode by about 15.61x over the same KVarN storage with full
materialization. That is a real implementation signal: compressed-domain attention removed a
large reconstruction cost. It did not approach fp16. The frozen gate needs at least 24.65 decode
tok/s and at least 507.04 prefill tok/s. The direct row would need about 3.43x its observed decode
and 8.02x its observed prefill in one valid retained run.

The source path has multiple independent costs: 512-token prefill synchronization, host tile
packing, eight-iteration KVarN normalization, and capacity-wide packed key/value attention at each
of 64 layers. No single actor-confined change has a credible route to both required multipliers.
That closes the one-recovery allowance without speculative kernel work.

## Trace boundary

The follow-up Metal System Trace was launched only after direct readiness revalidation. `pmset`
reported `powermode 2`, Foundation reported Low Power Mode false and thermal nominal, and the host
was on 140 W AC both before and after. The diagnostic is preserved at:

`/Users/llmbench/perf-work/results/fused-compressed-kv-profile-d4102e6/qwen-8k-kvarn-direct-metal-v1`

`xctrace` reached its 240-second time limit, then remained live beyond the bounded save watchdog.
The recorder peaked at 107,315,600 KiB RSS and left a 16,441,155,536-byte Apple Trace File. The raw
payload SHA-256 is
`12073b786fb06d5569269500129bee3f9b1926319f9a34a9de97c5cdf24853ea`; terminal status SHA-256 is
`982b2e8659cd53abb2c403c15e949cd6268fffa94f48c06cb85837141c2958b2`.
Both the bundle and raw payload fail `xctrace export --toc` with
`Document Missing Template Error`. The artifact is preserved as diagnostic failure evidence and
is not repaired, resumed, or used for kernel attribution.

## What remains

KVarN i8 keeps its prior model-scoped capacity-only Max-fit disposition: useful storage reduction,
measured quality/task warnings, no speed claim. The next speed matrix contains only fp16 and the
affine K4V2-g64/frozen KVTuner materialize/direct pairs in a fresh 5x5 cyclic order. It retains the
same thermal, provenance, hard-floor, and 5% gates. KVarN will be measured separately at Qwen 32K
for actual capacity/runtime context, never as a speed candidate.

A future KVarN speed revival needs a materially new kernel or algorithm task with its own
acceptance proof. It cannot be triggered by another thermal retry, a smaller trace window, or a
weaker threshold.
