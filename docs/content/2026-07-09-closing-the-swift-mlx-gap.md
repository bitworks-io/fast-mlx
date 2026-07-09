---
title: The bottleneck wasn't the language — closing an 18% Swift-vs-Zig MLX inference gap
date: 2026-07-09
whitepaper_theme: Building a high-performance MLX inference engine in Swift
status: draft
---

# The bottleneck wasn't the language

We're building an LLM inference engine in Swift on Apple's MLX. Early on, a benchmark put it **18% behind** our existing hand-tuned **Zig** engine — same Mac, same model, same weights: **127 tokens/sec vs 155**. The obvious explanation was the obvious one. It was also wrong. Here's how we found the real bottleneck and closed the gap to parity — and why the answer is good news for anyone choosing a safe, high-level language for high-performance inference.

## Why Swift, and why the gap mattered

The engine is largely AI-authored under limited human review. That single constraint drove the language choice: when a careful human reviewer isn't in the loop, the **compiler has to be the reviewer**. Swift 6's strict-concurrency checking catches data races at compile time, and its memory safety rules out the whole class of use-after-free and buffer bugs that a systems language ships silently. (On day one, the compiler earned its keep: it flagged a real cache-reuse bug that was quietly inflating our own benchmark numbers.)

The one open risk was performance. Could a safe, high-level language keep up with a hand-tuned engine written in a systems language talking to the same GPU? A spike measured it, and the first answer was discouraging: **−18%.**

## The 18%, and the obvious suspect

The test model — Qwen3-30B-A3B, a fine-grained mixture-of-experts with only 3.3B active parameters — is the *fastest* model in our catalog, which makes it the **worst case** for host overhead: each token's GPU work is small, so any fixed per-token CPU cost is a big fraction of the step.

Per-phase timing cleared the decode loop itself: our design overlaps the next forward pass with the current token's GPU→CPU readback, and the readback was only 1.1ms of a 7.8ms step. The cost lived in *building and dispatching* the forward: **6.7ms per step.**

The obvious suspect was **Swift's automatic reference counting (ARC)**. A 48-layer MoE forward allocates hundreds of intermediate tensors per token; each is a Swift object whose retain/release traffic crosses into MLX's C++ core. Plausible, intuitive — and the kind of thing you can spend a day "fixing" by hand before checking whether it's even the problem.

## What the profiler actually said

So we profiled it — Instruments Time Profiler, on the decode thread. The result inverted the hypothesis:

- **`swift_retain` / `swift_release`: ~0.2%.** ARC was a rounding error.
- **Swift-side graph construction: 17.6%.**
- **The other 81.7% was inside C++** — `mlx_async_eval → eval_impl`: traversing the lazy compute graph and synchronously encoding Metal kernels for roughly **2,000 nodes, every single token.**

The host language wasn't the tax. **Rebuilding the entire compute graph from scratch on every token was.** MLX is a lazy framework — you describe the forward pass and it builds a graph of operations — and by default that description is re-created and re-encoded for each of the 256 tokens you generate. That cost is framework-level; it has nothing to do with Swift.

## The fix: compile the step, once

MLX has exactly the right tool: `compile`, which traces a function's graph a single time and replays the cached version on subsequent calls. The lever was clear — compile the decode step so the 2,000-node graph is built once, not 256 times.

The catch is why this isn't a one-liner, and why Python's `mlx-lm` doesn't ship it either: a **KV cache grows every token**. Its buffers change shape, and the stock cache advances its position with a host-side integer — both of which poison a traced graph (the trace captures a fixed shape and a constant offset). So the real work was a **compile-friendly KV cache**: fixed-size buffers sized to a chunk, the position offset advanced by *in-graph* operations instead of a Swift `Int`, and cache updates and attention masking computed *inside* the graph. Growth rounds up to the next 256-token chunk, so the step retraces only once per chunk instead of once per token. Compile the decode step around that cache, and the per-token rebuild disappears.

## The result

| | decode tok/s | per-step CPU |
|---|---|---|
| Swift, baseline | 127 | 6.7ms rebuild, every token |
| **Swift, compiled step** | **155.5** | rebuild paid once; step is now GPU-bound |
| Zig, same session | 153.65 | — |

The readback phase dropped to **0.00ms** (fully hidden), the per-token CPU cost fell *below* the GPU step time, and decode became **GPU-bound** — the engine is now waiting on the GPU, not on itself. Swift landed at **155.5 tok/s, a hair ahead of the hand-tuned Zig engine** on the hardest model. Greedy output stayed byte-for-byte identical to the reference (we check this on every change), and Swift 6 strict concurrency stayed clean with zero unsafe escape hatches.

## The lessons

1. **Measure before you optimize.** The intuitive culprit — Swift's reference counting — was 0.2% of the cost. A day of hand-optimizing allocations would have bought nothing. One profiler run pointed straight at the real bottleneck.
2. **The host language often isn't the bottleneck.** The expensive thing was a framework-level per-token graph rebuild that *any* MLX host would pay. Swift wasn't slow; the default execution pattern was.
3. **On Apple-Silicon MLX, graph compilation is the decode lever** — and it needs a compile-compatible KV cache, which the popular Python stack doesn't provide out of the box. That cache is now our engine's decode core.
4. **A safe, high-level language can match a hand-tuned systems engine** when the real cost lives in the shared framework beneath both of them. We got Swift 6's memory- and concurrency-safety *and* parity with Zig. We didn't have to trade one for the other — which was the whole bet.
