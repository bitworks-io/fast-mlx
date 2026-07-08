# Swift Decode-Loop Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove (or disprove) that a Swift + mlx-swift decode loop, using fast-mlx's ported single-owner-actor design under Swift 6 strict concurrency, hits **within ~10–15% of the Zig engine's decode tok/s** on one MoE model on the M3 Ultra 256GB — the go/no-go gate for building the whole platform in Swift.

**Architecture:** Reuse `mlx-swift-lm` only for model/arch/tokenizer *loading* (banking the 59-arch head-start); **replace its generation loop** (the code with the 7.3× MoE regression) with our own single-owner `InferenceActor` that submits work async and never blocks on a synchronous per-token GPU→CPU readback. Measure decode tok/s + TTFT with a stream-timed, warmup-dropped bench, verify token-equivalence vs Python `mlx-lm` at temp=0, and record a verdict against the gate.

**Tech Stack:** Swift 6 (strict concurrency), SwiftPM, `ml-explore/mlx-swift`, `ml-explore/mlx-swift-lm`, `huggingface/swift-transformers` (tokenizer, transitive), Python `mlx-lm` (reference only), Xcode/Instruments (profiling). Target model: **Qwen3-30B-A3B-Instruct-2507 4-bit** (Tier A primary). Reference perf baseline: the Zig `mlx-serve` engine run on the same box.

**This is a spike, not the engine.** Code here is throwaway-quality by intent — the deliverable is a *measured verdict*, not production code. But it is written test-first where correctness matters (token-equivalence, bench math) because those tests carry forward into the real engine.

**Reference docs:** [platform spec](../specs/2026-07-08-fast-mlx-platform-design.md) §5 (eval loop) + §10 (plan outline); [carry-forward backlog](../../reference/2026-07-08-carry-forward-performance-backlog.md) (bench methodology + the eval-loop design being ported).

---

## Prerequisites (human, before Task 1)

- On the **M3 Ultra 256GB** dev box (this is the reference hardware; a laptop invalidates the comparison).
- Xcode 26+ / Swift 6.x toolchain installed (`swift --version` shows 6.x).
- Python `mlx-lm` installed for the reference (`pip install mlx-lm`), and the Zig `mlx-serve` binary built `-Doptimize=ReleaseFast` and runnable (for the perf baseline).
- Model present locally: `Qwen3-30B-A3B-Instruct-2507-4bit` (MLX format) in `~/.cache/huggingface` or a known path. If absent, `mlx_lm.convert` or download an MLX-community 4-bit build first.

---

## File structure

- `Package.swift` — SwiftPM manifest, Swift 6 mode, mlx-swift + mlx-swift-lm deps.
- `Sources/SpikeCore/InferenceActor.swift` — the single-owner actor: owns model+tokenizer, `submit(prompt) -> AsyncThrowingStream<Int>` (token ids), greedy decode, async-submit no-sync-readback loop.
- `Sources/SpikeCore/DecodeMetrics.swift` — pure struct + math for TTFT / decode-tok-s from timestamped token events (unit-tested, no MLX).
- `Sources/spike-cli/main.swift` — CLI: `run` (one prompt, print tokens), `bench` (warmup-drop, timed), `equiv` (dump first-N token ids for the reference diff).
- `Tests/SpikeCoreTests/DecodeMetricsTests.swift` — pure bench-math tests.
- `Tests/SpikeCoreTests/InferenceActorTests.swift` — actor decode + isolation tests.
- `scripts/reference_tokens.py` — Python `mlx-lm` greedy first-N token-id dump (equivalence reference).
- `scripts/run_spike_comparison.sh` — orchestrates Swift bench + Zig `mlx-serve` bench on the same box, same prompt, temp=0, and prints the delta.
- `docs/api/mlx-swift-api-notes.md` — pinned mlx-swift version + verified API signatures (Task 2 output; every later mlx-swift call references this).
- `docs/superpowers/verdicts/2026-07-08-swift-spike-verdict.md` — the deliverable: measured numbers + go/no-go decision.

---

## Task 1: Scaffold the Swift 6 package and confirm it builds

**Files:**
- Create: `Package.swift`
- Create: `Sources/SpikeCore/Placeholder.swift`
- Create: `Sources/spike-cli/main.swift`

- [ ] **Step 1: Write `Package.swift`** (pin exact dependency revisions in Task 2; use branch `main` for now)

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "fast-mlx-spike",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "SpikeCore", targets: ["SpikeCore"]),
        .executable(name: "spike-cli", targets: ["spike-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", branch: "main"),
    ],
    targets: [
        .target(
            name: "SpikeCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "spike-cli",
            dependencies: ["SpikeCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(name: "SpikeCoreTests", dependencies: ["SpikeCore"],
                    swiftSettings: [.swiftLanguageMode(.v6)]),
    ]
)
```

- [ ] **Step 2: Add a trivial placeholder so the target compiles**

`Sources/SpikeCore/Placeholder.swift`:
```swift
public enum Spike { public static let ok = true }
```

`Sources/spike-cli/main.swift`:
```swift
import SpikeCore
print("spike ok: \(Spike.ok)")
```

- [ ] **Step 3: Build (Metal shaders require xcodebuild, not bare `swift build`)**

Run: `xcodebuild -scheme spike-cli -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **`. (Known gotcha from research: plain `swift build` cannot compile mlx-swift's Metal shaders — `xcodebuild` can. If SwiftPM resolves but the Metal link fails under `swift build`, that's expected; use xcodebuild.)

- [ ] **Step 4: Commit**

```bash
git add Package.swift Sources/
git commit -m "spike: scaffold Swift 6 package with mlx-swift + mlx-swift-lm"
```

---

## Task 2: Pin mlx-swift version and verify the API surface we depend on

mlx-swift moves fast and its C++/Metal layer is thinly documented; **every mlx-swift call in later tasks must reference the signatures verified here.** This is not a placeholder — it is the required grounding step for a fast-moving dependency.

**Files:**
- Create: `docs/api/mlx-swift-api-notes.md`
- Modify: `Package.swift` (pin exact resolved versions)

- [ ] **Step 1: Resolve and record exact versions**

Run: `xcodebuild -scheme spike-cli -showBuildSettings >/dev/null; cat Package.resolved | grep -A3 mlx-swift`
Record the resolved `revision`/`version` for mlx-swift and mlx-swift-lm into `docs/api/mlx-swift-api-notes.md`. Then pin them in `Package.swift` (`from:`→exact, branch→`.revision("…")`).

- [ ] **Step 2: Verify each needed API by reading the resolved source in `.build/checkouts/`**

For each of the following, find the exact current signature (read `.build/checkouts/mlx-swift*/Sources/...`) and paste it into `docs/api/mlx-swift-api-notes.md` under a heading. If an API differs from the assumption noted, record the real one — later tasks use *these*:
  1. **Load model+tokenizer:** `MLXLLM`/`MLXLMCommon` model container / `loadModel` / `ModelContainer` — how you get a callable model + a `Tokenizer` from a local path. (Assumed: `LLMModelFactory.shared.loadContainer(...)` yielding a `ModelContext` with `.model` and `.tokenizer`.)
  2. **Forward pass returning logits:** the model's `callAsFunction`/`prepare`/`step` signature and how to pass a KV cache. (Assumed: `model(inputs, cache: cache)` → `MLXArray` logits `[1, seqLen, vocab]`.)
  3. **KV cache type + construction:** `KVCache` / `model.newCache(...)`. (Assumed: `MLXLMCommon` `KVCache` array, one per layer.)
  4. **Greedy sample on GPU:** `argMax(logits[.., -1, ..], axis: -1)` → `MLXArray`; and `.item(Int.self)` to read one token id to CPU.
  5. **Async eval:** `asyncEval(_:)` and `eval(_:)` — confirm both exist and their signatures (the no-sync-readback design depends on `asyncEval`).
  6. **Tokenizer:** `encode(text:)` / `decode(tokens:)` on the `Tokenizer` from swift-transformers.

- [ ] **Step 3: Write one smoke test that calls the three riskiest APIs and prints shapes**

`Sources/spike-cli/main.swift` (add an `api-check` subcommand): load the model, encode `"Hello"`, run one forward, print `logits.shape`, argMax the last position, decode it. Run:
`xcodebuild -scheme spike-cli build && .build/.../spike-cli api-check --model <PATH>`
Expected: prints a logits shape like `[1, N, 151936]` and one decoded token without crashing. This proves the assumed API path end-to-end before we build on it.

- [ ] **Step 4: Commit**

```bash
git add docs/api/mlx-swift-api-notes.md Package.swift Package.resolved Sources/spike-cli/main.swift
git commit -m "spike: pin mlx-swift versions + verify API surface (load/forward/cache/sample/asyncEval/tokenizer)"
```

---

## Task 3: Pure decode-metrics math (TTFT + tok/s), test-first — no MLX

This logic carries forward into the real harness, so it is written test-first and kept MLX-free.

**Files:**
- Create: `Sources/SpikeCore/DecodeMetrics.swift`
- Test: `Tests/SpikeCoreTests/DecodeMetricsTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
import XCTest
@testable import SpikeCore

final class DecodeMetricsTests: XCTestCase {
    func testTtftAndDecodeRateFromStreamEvents() {
        // token events at t = 0.10 (first token), 0.20, 0.30, 0.40 seconds; prompt submitted at t=0
        let m = DecodeMetrics(submitTime: 0.0, tokenTimes: [0.10, 0.20, 0.30, 0.40])
        XCTAssertEqual(m.ttftSeconds, 0.10, accuracy: 1e-9)
        // decode rate excludes prefill: 3 inter-token gaps over 0.30s => 10 tok/s
        XCTAssertEqual(m.decodeTokensPerSecond, 10.0, accuracy: 1e-9)
        XCTAssertEqual(m.generatedTokenCount, 4)
    }

    func testSingleTokenHasNoDecodeRate() {
        let m = DecodeMetrics(submitTime: 0.0, tokenTimes: [0.10])
        XCTAssertNil(m.decodeTokensPerSecond)
    }
}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `swift test --filter DecodeMetricsTests 2>&1 | tail -20`
Expected: FAIL — `DecodeMetrics` type not found.

- [ ] **Step 3: Implement `DecodeMetrics`**

```swift
import Foundation

/// Pure, MLX-free decode measurement from stream timestamps.
/// Decode rate deliberately EXCLUDES prefill (measured from first token onward),
/// matching the carry-forward bench methodology (rate from the live stream, not usage fields).
public struct DecodeMetrics: Sendable {
    public let ttftSeconds: Double
    public let generatedTokenCount: Int
    public let decodeTokensPerSecond: Double?

    public init(submitTime: Double, tokenTimes: [Double]) {
        precondition(!tokenTimes.isEmpty, "need at least one token")
        self.generatedTokenCount = tokenTimes.count
        self.ttftSeconds = tokenTimes[0] - submitTime
        if tokenTimes.count >= 2 {
            let decodeSpan = tokenTimes.last! - tokenTimes[0]
            let gaps = Double(tokenTimes.count - 1)
            self.decodeTokensPerSecond = decodeSpan > 0 ? gaps / decodeSpan : nil
        } else {
            self.decodeTokensPerSecond = nil
        }
    }
}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `swift test --filter DecodeMetricsTests 2>&1 | tail -5`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add Sources/SpikeCore/DecodeMetrics.swift Tests/SpikeCoreTests/DecodeMetricsTests.swift
git commit -m "spike: pure decode-metrics math (ttft + stream-timed tok/s), test-first"
```

---

## Task 4: The single-owner `InferenceActor` (design port), with an isolation test

Ports spec §5: one actor owns the model + all MLX calls; callers get an `AsyncThrowingStream<Int>` of token ids and never touch MLX. Greedy only (temp=0) for the spike.

**Files:**
- Create: `Sources/SpikeCore/InferenceActor.swift`
- Test: `Tests/SpikeCoreTests/InferenceActorTests.swift`

- [ ] **Step 1: Write the failing test** (uses a fake model so it runs in CI without weights)

```swift
import XCTest
@testable import SpikeCore

final class InferenceActorTests: XCTestCase {
    func testStreamsExpectedGreedyTokensFromFakeModel() async throws {
        // Fake decoder returns a fixed script, proving the actor's loop/streaming
        // is correct independent of MLX. Real model wired in Task 5.
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 7, /*eos*/ 2], eos: 2))
        var got: [Int] = []
        for try await tok in await actor.submit(promptTokens: [1, 2, 3], maxTokens: 10) {
            got.append(tok)
        }
        XCTAssertEqual(got, [5, 6, 7]) // eos consumed, not emitted
    }
}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `swift test --filter InferenceActorTests 2>&1 | tail -20`
Expected: FAIL — `InferenceActor` / `ScriptedDecoder` not found.

- [ ] **Step 3: Implement the actor + a `Decoder` seam** (so tests use a fake, Task 5 plugs in MLX)

```swift
import Foundation

/// Abstraction over "one decode step" so the actor's loop is testable without MLX.
public protocol Decoder: Sendable {
    /// Prefill the prompt and return the first token id.
    mutating func prefill(_ promptTokens: [Int]) -> Int
    /// Given the last token, produce the next. (Greedy; temp=0.)
    mutating func step(last: Int) -> Int
}

/// Test double: replays a fixed script.
public struct ScriptedDecoder: Decoder {
    let script: [Int]; let eos: Int; var i = 0
    public init(script: [Int], eos: Int) { self.script = script; self.eos = eos }
    public mutating func prefill(_ p: [Int]) -> Int { defer { i += 1 }; return script[i] }
    public mutating func step(last: Int) -> Int { defer { i += 1 }; return script[i] }
}

public actor InferenceActor {
    private var decoder: any Decoder
    public init(decoder: any Decoder) { self.decoder = decoder }

    /// Non-blocking: returns a stream immediately; decode runs inside the actor.
    public func submit(promptTokens: [Int], maxTokens: Int, eos: Int = 2) -> AsyncThrowingStream<Int, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { await self.run(promptTokens, maxTokens, eos, continuation) }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(_ prompt: [Int], _ maxTokens: Int, _ eos: Int,
                     _ cont: AsyncThrowingStream<Int, Error>.Continuation) {
        var tok = decoder.prefill(prompt)
        var n = 0
        while n < maxTokens {
            if tok == eos { break }
            cont.yield(tok)
            n += 1
            if Task.isCancelled { break }
            tok = decoder.step(last: tok)
        }
        cont.finish()
    }
}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `swift test --filter InferenceActorTests 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/SpikeCore/InferenceActor.swift Tests/SpikeCoreTests/InferenceActorTests.swift
git commit -m "spike: single-owner InferenceActor with AsyncThrowingStream + testable Decoder seam"
```

---

## Task 5: Wire the real MLX decoder — async-submit, NO sync per-token readback

This is the crux: implement `Decoder` against mlx-swift **without** the synchronous per-token GPU→CPU stall that caused the 7.3× regression. Use the APIs verified in Task 2.

**Files:**
- Create: `Sources/SpikeCore/MLXDecoder.swift`
- Modify: `Sources/spike-cli/main.swift` (add `run` subcommand)

- [ ] **Step 1: Implement `MLXDecoder`** (mlx-swift calls MUST match `docs/api/mlx-swift-api-notes.md`)

```swift
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Real decoder. Greedy (temp=0). KEY CONSTRAINT (spec §5, backlog "lazy pipeline"):
/// keep a one-step lookahead — submit the NEXT forward with asyncEval BEFORE reading the
/// current token to CPU, so GPU compute overlaps the CPU-side .item() readback. Never call
/// a blocking eval()+.item() in the hot path with nothing else in flight.
public struct MLXDecoder: Decoder {
    private let model: any LanguageModel      // exact type per api-notes
    private var cache: [KVCache]
    private var pendingLogits: MLXArray?      // the lookahead logits, not yet argmaxed

    public init(model: any LanguageModel, cache: [KVCache]) {
        self.model = model; self.cache = cache
    }

    public mutating func prefill(_ promptTokens: [Int]) -> Int {
        let ids = MLXArray(promptTokens).reshaped([1, promptTokens.count])
        let logits = model(ids, cache: cache)          // [1, seqLen, vocab]
        let last = logits[0..., -1, 0...]              // [1, vocab]
        let next = argMax(last, axis: -1)              // [1] on GPU
        // submit-first: kick the next forward before we read `next` to CPU
        let nextIds = next.reshaped([1, 1])
        pendingLogits = model(nextIds, cache: cache)
        asyncEval(pendingLogits!)                       // overlap GPU with the readback below
        return next.item(Int.self)                     // readback overlaps the pending forward
    }

    public mutating func step(last: Int) -> Int {
        // pendingLogits already computed for the position after `last`
        let logits = pendingLogits!
        let next = argMax(logits[0..., -1, 0...], axis: -1)
        let nextIds = next.reshaped([1, 1])
        pendingLogits = model(nextIds, cache: cache)   // submit next
        asyncEval(pendingLogits!)
        return next.item(Int.self)
    }
}
```

> NOTE for the executor: the exact `model(_:cache:)` call, `KVCache` construction, and slice syntax come from Task 2's notes — adjust to the pinned API. The *invariant to preserve* is submit-next-before-readback; do not "simplify" it into `eval();  .item(); model(next)` (that reintroduces the stall this spike exists to avoid).

- [ ] **Step 2: Add the `run` subcommand** that loads the model, builds an `InferenceActor(decoder: MLXDecoder(...))`, streams a prompt, and prints decoded text.

```swift
// in main.swift, `run` branch (pseudocode calls resolved via api-notes):
// let ctx = try await load(model: path)                // -> model + tokenizer + newCache()
// let dec = MLXDecoder(model: ctx.model, cache: ctx.newCache())
// let actor = InferenceActor(decoder: dec)
// let promptIds = ctx.tokenizer.encode(prompt)
// for try await id in await actor.submit(promptTokens: promptIds, maxTokens: 256, eos: ctx.eos) {
//     print(ctx.tokenizer.decode([id]), terminator: "")
// }
```

- [ ] **Step 3: Run it on the real model, confirm coherent output**

Run: `xcodebuild -scheme spike-cli build && .build/.../spike-cli run --model <PATH> --prompt "Explain unified memory on Apple Silicon in one paragraph." --max-tokens 128`
Expected: coherent, non-garbage paragraph. (Correctness canary — a misrouted decode produces fluent nonsense; read it.)

- [ ] **Step 4: Confirm it builds clean under Swift 6 strict concurrency**

Run: `xcodebuild -scheme spike-cli build 2>&1 | grep -i "SendingRisks\|data race\|concurrency" || echo "no concurrency errors"`
Expected: `no concurrency errors`. (If strict-concurrency errors appear, resolving them cleanly *is part of the gate* — a design that can't express itself under Swift 6 without `@unchecked`/`nonisolated(unsafe)` escape hatches is a partial red flag; note it in the verdict.)

- [ ] **Step 5: Commit**

```bash
git add Sources/SpikeCore/MLXDecoder.swift Sources/spike-cli/main.swift
git commit -m "spike: real MLX greedy decoder with submit-first async lookahead (no sync per-token readback)"
```

---

## Task 6: Token-equivalence vs Python `mlx-lm` at temp=0

Proves the Swift loop is *correct*, not just fast — the minimal version of the equivalence check that becomes the real harness's triad.

**Files:**
- Create: `scripts/reference_tokens.py`
- Modify: `Sources/spike-cli/main.swift` (add `equiv` subcommand: dump first-N token ids as JSON)

- [ ] **Step 1: Write the reference dumper**

```python
# scripts/reference_tokens.py — greedy first-N token ids from mlx-lm (the reference)
import sys, json
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
model, tok = load(sys.argv[1])
prompt = sys.argv[2]; n = int(sys.argv[3])
ids = tok.encode(prompt)
import mlx.core as mx
cache = None
out = []
from mlx_lm.models.cache import make_prompt_cache
cache = make_prompt_cache(model)
y = mx.array(ids)[None]
for _ in range(n):
    logits = model(y, cache=cache)[:, -1, :]
    t = int(mx.argmax(logits, axis=-1).item())
    out.append(t); y = mx.array([[t]])
print(json.dumps(out))
```

- [ ] **Step 2: Add `equiv` subcommand** to `spike-cli` that prints the first-N greedy token ids as a JSON array (same prompt, same N), using the `InferenceActor`.

- [ ] **Step 3: Diff them** (temp=0, first-N; per backlog, expect byte-identical for the first ~30–80 tokens, then legitimate divergence from float-reduction order)

```bash
P="Write a haiku about unified memory."; N=40; M=<PATH>
python3 scripts/reference_tokens.py "$M" "$P" $N > /tmp/ref.json
.build/.../spike-cli equiv --model "$M" --prompt "$P" --n $N > /tmp/swift.json
python3 - <<'PY'
import json
r=json.load(open('/tmp/ref.json')); s=json.load(open('/tmp/swift.json'))
match=0
for a,b in zip(r,s):
    if a!=b: break
    match+=1
print(f"first-{len(r)} identical-prefix = {match}")
assert match >= 30, f"equivalence prefix too short ({match}); loop is likely wrong, not just numerically divergent"
print("PASS")
PY
```

Expected: `identical-prefix >= 30` then `PASS`. A short prefix means a real bug (wrong sampling/cache/position), not numerics — fix before trusting any perf number.

- [ ] **Step 4: Commit**

```bash
git add scripts/reference_tokens.py Sources/spike-cli/main.swift
git commit -m "spike: temp=0 token-equivalence vs mlx-lm (first-N identical-prefix gate)"
```

---

## Task 7: The bench (warmup-drop, stream-timed) + record CSV

Applies the carry-forward methodology: verify release build, warmup run dropped, rate from the stream, salted prompt.

**Files:**
- Modify: `Sources/spike-cli/main.swift` (add `bench` subcommand)

- [ ] **Step 1: Implement `bench`** — for R+1 runs (default R=3), salt the prompt (`[run-<i>-<nonce>]` prefix), time each token via `DecodeMetrics`, **drop run 0** (warmup), average runs 1..R; require a `-c release` build.

```swift
// bench branch (uses DecodeMetrics from Task 3):
// guard isReleaseBuild() else { fatalError("build with -c release / xcodebuild Release; Debug perf is meaningless") }
// var rates: [Double] = []
// for i in 0...runs {                       // run 0 = warmup, dropped
//     let submit = Date().timeIntervalSinceReferenceDate
//     var times: [Double] = []
//     let salted = "[run-\(i)-\(nonce)] \(prompt)"
//     for try await _ in await actor.submit(promptTokens: enc(salted), maxTokens: maxTokens, eos: eos) {
//         times.append(Date().timeIntervalSinceReferenceDate)
//     }
//     let m = DecodeMetrics(submitTime: submit, tokenTimes: times)
//     if i > 0, let r = m.decodeTokensPerSecond { rates.append(r) }
// }
// let avg = rates.reduce(0,+)/Double(rates.count)
// print CSV row: model,quant,ctx,prompt_kind,decode_tok_s_avg,ttft_ms
```

- [ ] **Step 2: Run the Swift bench** on the target model, decode workload, 256GB M3 Ultra.

Run: `xcodebuild -scheme spike-cli -configuration Release build && .build/.../spike-cli bench --model <PATH> --prompt "<fixed decode prompt>" --max-tokens 256 --runs 3`
Expected: a CSV row with a stable `decode_tok_s_avg` (runs 1..3 within a few %).

- [ ] **Step 3: Save the CSV** to `docs/superpowers/verdicts/swift-bench-<date>.csv` and commit.

```bash
git add Sources/spike-cli/main.swift docs/superpowers/verdicts/swift-bench-*.csv
git commit -m "spike: stream-timed warmup-dropped decode bench (release-guarded)"
```

---

## Task 8: Head-to-head vs the Zig engine, and record the verdict

**Files:**
- Create: `scripts/run_spike_comparison.sh`
- Create: `docs/superpowers/verdicts/2026-07-08-swift-spike-verdict.md`

- [ ] **Step 1: Write the comparison orchestrator** — same box, same model, same fixed prompt, temp=0, `max-tokens=256`. Boot the Zig `mlx-serve` (ReleaseFast) and measure its decode tok/s from its SSE stream the same way (reuse `mlx-serve`'s own `bench.sh --family qwen36` if the model matches, or a matched curl); run the Swift `bench`; print both + the delta %.

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL="$1"; PROMPT="${2:-Explain how continuous batching improves LLM serving throughput.}"
echo "== Zig mlx-serve =="   # ReleaseFast binary; stream-timed decode tok/s
ZIG=$( ~/Projects/mlx-serve/tests/bench.sh --family qwen36 2>/dev/null | rg -o 'decode[^0-9]*([0-9.]+)' | head -1 || echo "MANUAL" )
echo "zig_decode_tok_s=$ZIG"
echo "== Swift spike =="
SWIFT=$( .build/release/spike-cli bench --model "$MODEL" --prompt "$PROMPT" --max-tokens 256 --runs 3 | rg -o '[0-9.]+' | tail -1 )
echo "swift_decode_tok_s=$SWIFT"
python3 - "$ZIG" "$SWIFT" <<'PY'
import sys
try:
    z=float(sys.argv[1]); s=float(sys.argv[2])
    print(f"delta = {100*(s-z)/z:+.1f}%  (swift vs zig)")
except ValueError:
    print("record numbers manually")
PY
```

- [ ] **Step 2: Run it and capture numbers**

Run: `bash scripts/run_spike_comparison.sh <PATH>`
Expected: prints `zig_decode_tok_s`, `swift_decode_tok_s`, and a `delta = …%`.

- [ ] **Step 3: Write the verdict** against the spec §10 gate

`docs/superpowers/verdicts/2026-07-08-swift-spike-verdict.md` — fill in: hardware, model, exact prompt, Zig tok/s, Swift tok/s, delta %, TTFT both, the equivalence identical-prefix count, whether Swift 6 strict concurrency was satisfied without unsafe escape hatches, and the **decision**:
  - **GO** if `delta >= -15%` (Swift within ~10–15% of Zig) AND equivalence prefix ≥30 AND no `@unchecked`/`nonisolated(unsafe)` needed → proceed to author the harness-spine and engine plans in Swift.
  - **RE-ASSESS** if `delta <= -50%` (≥1.5× gap — the mlx-swift-lm failure mode) OR the loop needed unsafe concurrency escape hatches → diagnose (is it the readback stall? MoE routing? sampling on CPU?) using Instruments GPU trace before deciding C++/Python/Zig fallback.
  - **INVESTIGATE** if between −15% and −50% → one Instruments pass to find the gap, then decide.

- [ ] **Step 4: Commit the verdict**

```bash
git add scripts/run_spike_comparison.sh docs/superpowers/verdicts/2026-07-08-swift-spike-verdict.md
git commit -m "spike: head-to-head vs Zig engine + go/no-go verdict"
```

---

## Self-Review (completed inline)

- **Spec coverage (spike scope only):** §10 Task 1 gate — covered end to end (scaffold → API pin → actor → real MLX loop with the no-sync-readback invariant from §5 → equivalence → bench per §6.3 methodology → head-to-head + verdict against the §10 gate thresholds). Downstream subsystems (harness spine §6, batching §5, dial §4, catalog §9, app, Python plane) are **intentionally out of scope** — each is a separate plan authored after this verdict.
- **Placeholder scan:** the mlx-swift API calls in Tasks 5–6 are explicitly grounded in Task 2's verified `api-notes` rather than assumed — this is a required grounding step for a fast-moving dependency, not a vague placeholder. The one invariant that must not be "simplified away" (submit-next-before-readback) is called out with a NOTE. No `TODO`/`TBD`/"handle edge cases" left.
- **Type consistency:** `Decoder` protocol (`prefill`/`step`) is defined in Task 4 and implemented by both `ScriptedDecoder` (Task 4) and `MLXDecoder` (Task 5); `DecodeMetrics(submitTime:tokenTimes:)` defined Task 3, used Task 7; `InferenceActor.submit(promptTokens:maxTokens:eos:)` defined Task 4, used Tasks 5–7. Consistent.

---

## Notes for the executor

- **This validates a bet, it doesn't build the engine.** Optimize for a *trustworthy number*, not clean code. The only carry-forward code is `DecodeMetrics` and the equivalence/bench discipline.
- **The whole point is the no-sync-readback loop.** If you find yourself writing `eval(logits); let t = logits.item()` with nothing else submitted, stop — that is the 7.3× failure mode this spike exists to avoid.
- **If GO:** the next plans are (in order) the harness spine (§6.1 triad + hermetic corpus + bench), then the single-model engine behind it, then continuous batching + the drain-before-join invariant. Ask me to author them.
