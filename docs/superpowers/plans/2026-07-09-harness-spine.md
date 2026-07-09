# Harness Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the engine-agnostic conformance + precision-loss **harness spine** in Swift — the equivalence/engagement/acceptance **triad**, the hermetic **corpus**, the **perf-bench** framework, and a pluggable **`QualityMetric`** interface — so every future optimization and every intake candidate (DFlash, oQ4e, …) is quantified through one trustworthy instrument. This is the product's spine (spec §6): the automated reviewer that stands in for limited human review, and the dial's measurement engine.

**Architecture:** A `HarnessCore` library + `fastmlx-harness` CLI added to the engine package. The harness drives the engine through an **`EngineDriver` protocol** — an in-process impl over the spike's compiled decode core now, an HTTP/OpenAI impl later — so the harness is engine-agnostic by construction without blocking on the HTTP server. All triad/corpus/metric/bench *logic* is pure and MLX-free (unit-testable on any host with a `ScriptedDriver` fake); only the `SwiftEngineDriver` touches MLX and runs on llmbench.

**Tech Stack:** Swift 6 (strict concurrency), the engine package under `spike/` (SpikeCore = compiled decode core: `CompiledMLXDecoder`, `CompiledKVCache`, `InferenceActor`, `DecodeMetrics`). Python `mlx-lm` as the reference driver for equivalence + KL. Build/run on `llmbench@192.168.1.252` via SSH (`xcodebuild ... -skipPackagePluginValidation`); pure logic runs anywhere.

**Reference:** spec §6 (the harness), §4 (the dial); [carry-forward backlog](../../reference/2026-07-08-carry-forward-performance-backlog.md) (bench methodology, the triad's origin); [intake log](../../reference/performance-technique-intake.md) (first customers); [spike verdict](../verdicts/2026-07-08-swift-spike-verdict.md) (the engine surface this drives).

**Non-negotiables (carry from the spike):** Swift 6 strict concurrency clean, zero `@unchecked`/`nonisolated(unsafe)`; honest numbers; `@main ... async` (never `DispatchSemaphore`); build with `xcodebuild -skipPackagePluginValidation`; run Python from `~/fast-mlx-spike`.

---

## File structure

- `spike/Sources/HarnessCore/EngineDriver.swift` — `EngineDriver` protocol, `RunConfig`, `RunResult`, `EngagementCounters`; `ScriptedDriver` fake.
- `spike/Sources/HarnessCore/Triad.swift` — `identicalPrefix`, `EquivalenceCheck`, `EngagementCheck`, `AcceptanceCheck`, and the `TriadVerdict` that composes them.
- `spike/Sources/HarnessCore/Corpus.swift` — `CorpusEntry`, the universal invariants, `runCorpus`.
- `spike/Sources/HarnessCore/QualityMetric.swift` — `QualityMetric` protocol; `KLDivergenceMetric`.
- `spike/Sources/HarnessCore/BenchMatrix.swift` — `Workload`, `Mode`, `Cell`, `BenchRunner`, `BenchRow` (CSV), release-build guard.
- `spike/Sources/fastmlx-harness/SwiftEngineDriver.swift` — in-process `EngineDriver` over `CompiledMLXDecoder` (the only MLX-touching file; lives in the executable target, NOT in pure `HarnessCore`).
- `spike/Sources/fastmlx-harness/Harness.swift` — `@main` CLI: `verify` (triad on a model vs mlx-lm), `bench` (cell matrix), `corpus` (hermetic run), `kl` (metric on a dial point).
- `spike/Tests/HarnessCoreTests/{TriadTests,CorpusTests,KLTests,BenchMatrixTests}.swift` — all pure, `ScriptedDriver`-backed.
- `spike/scripts/harness_reference.py` — mlx-lm reference: greedy first-N tokens + per-position top-k logprobs (for equivalence + KL).

---

## Task 1: Scaffold `HarnessCore` + `fastmlx-harness` targets

**Files:**
- Modify: `spike/Package.swift`
- Create: `spike/Sources/HarnessCore/Placeholder.swift`, `spike/Sources/fastmlx-harness/Harness.swift`

- [ ] **Step 1: Add targets to `Package.swift`** (HarnessCore depends on SpikeCore; both Swift 6 mode)

```swift
// in targets: [...]
// HarnessCore is PURE — NO MLX/SpikeCore dependency — so it (and HarnessCoreTests) build+test off-box with `swift test`.
.target(name: "HarnessCore", dependencies: [], swiftSettings: [.swiftLanguageMode(.v6)]),
// Only the executable pulls in SpikeCore (MLX); SwiftEngineDriver.swift lives HERE, not in HarnessCore.
.executableTarget(name: "fastmlx-harness", dependencies: ["HarnessCore", "SpikeCore"], swiftSettings: [.swiftLanguageMode(.v6)]),
.testTarget(name: "HarnessCoreTests", dependencies: ["HarnessCore"], swiftSettings: [.swiftLanguageMode(.v6)]),
// in products: add .executable(name: "fastmlx-harness", targets: ["fastmlx-harness"])
```

- [ ] **Step 2: Minimal placeholder + `@main`** (async main pattern — file NOT named `main.swift`)

`Sources/HarnessCore/Placeholder.swift`: `public enum HarnessCore { public static let ready = true }`
`Sources/fastmlx-harness/Harness.swift`:
```swift
import HarnessCore
@main struct Harness { static func main() async { print("harness ready: \(HarnessCore.ready)") } }
```

- [ ] **Step 3: Build on llmbench.** `rsync -az --delete spike/ llmbench:~/fast-mlx-spike/ && ssh llmbench 'cd ~/fast-mlx-spike && xcodebuild -scheme fastmlx-harness -destination "platform=macOS" -skipPackagePluginValidation build 2>&1 | tail -3'` → Expected `** BUILD SUCCEEDED **`.
- [ ] **Step 4: Commit.** `git add spike/Package.swift spike/Sources/HarnessCore spike/Sources/fastmlx-harness && git commit -m "harness: scaffold HarnessCore + fastmlx-harness targets"`

---

## Task 2: `EngineDriver` protocol + types + `ScriptedDriver` fake

Defines the seam the whole harness drives; the fake lets every later test run MLX-free.

**Files:** Create `spike/Sources/HarnessCore/EngineDriver.swift`; Test `spike/Tests/HarnessCoreTests/EngineDriverTests.swift`

- [ ] **Step 1: Failing test**
```swift
import XCTest
@testable import HarnessCore
final class EngineDriverTests: XCTestCase {
  func testScriptedDriverReplaysTokensAndEngagement() async throws {
    let d = ScriptedDriver(tokens: [5,6,7], engagement: ["pld": 3], logprobs: [[-0.1,-2.0]])
    let r = try await d.generate(prompt: [1,2], config: .greedy(maxTokens: 8))
    XCTAssertEqual(r.tokens, [5,6,7])
    XCTAssertEqual(r.engagement.counts["pld"], 3)
  }
}
```
- [ ] **Step 2: Run → FAIL** (`swift test --filter EngineDriverTests`; types undefined).
- [ ] **Step 3: Implement**
```swift
import Foundation

public struct RunConfig: Sendable, Hashable {
    public var temperature: Float; public var maxTokens: Int
    public var specDecode: String?   // "pld" | "dspark" | nil
    public var kvQuant: String?      // "fp16" | "8" | "turbo4" | nil
    public init(temperature: Float = 0, maxTokens: Int = 256, specDecode: String? = nil, kvQuant: String? = nil) {
        self.temperature = temperature; self.maxTokens = maxTokens; self.specDecode = specDecode; self.kvQuant = kvQuant
    }
    public static func greedy(maxTokens: Int) -> RunConfig { .init(temperature: 0, maxTokens: maxTokens) }
}

public struct EngagementCounters: Sendable { public var counts: [String: Int]; public init(_ c: [String: Int] = [:]) { counts = c } }

public struct RunResult: Sendable {
    public var tokens: [Int]
    public var engagement: EngagementCounters
    public var acceptanceRate: Double?     // for spec-decode runs; nil otherwise
    public var submitTime: Double
    public var tokenTimes: [Double]
    public init(tokens: [Int], engagement: EngagementCounters = .init(), acceptanceRate: Double? = nil,
                submitTime: Double = 0, tokenTimes: [Double] = []) {
        self.tokens = tokens; self.engagement = engagement; self.acceptanceRate = acceptanceRate
        self.submitTime = submitTime; self.tokenTimes = tokenTimes
    }
}

/// The seam. In-process (SwiftEngineDriver) now; HTTP/OpenAI later. Reference impl (mlx-lm) via ReferenceDriver.
public protocol EngineDriver: Sendable {
    func generate(prompt: [Int], config: RunConfig) async throws -> RunResult
    /// Top-k logprobs per generated position (temp=0), for KL. Empty if unsupported.
    func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]]
}

public struct ScriptedDriver: EngineDriver {
    let tokens: [Int]; let engagement: [String: Int]; let lp: [[Float]]
    public init(tokens: [Int], engagement: [String: Int] = [:], logprobs: [[Float]] = []) {
        self.tokens = tokens; self.engagement = engagement; self.lp = logprobs
    }
    public func generate(prompt: [Int], config: RunConfig) async throws -> RunResult {
        RunResult(tokens: tokens, engagement: .init(engagement))
    }
    public func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] { lp }
}
```
- [ ] **Step 4: Run → PASS.** **Step 5: Commit** `harness: EngineDriver protocol + RunConfig/RunResult + ScriptedDriver`.

---

## Task 3: The triad (equivalence + engagement + acceptance)

The core pattern (backlog §6.1): equivalence alone can't tell "ran and matched" from "silently no-op'd and matched" — so a change ships all three proofs.

**Files:** Create `spike/Sources/HarnessCore/Triad.swift`; Test `spike/Tests/HarnessCoreTests/TriadTests.swift`

- [ ] **Step 1: Failing tests**
```swift
import XCTest
@testable import HarnessCore
final class TriadTests: XCTestCase {
  func testIdenticalPrefix() {
    XCTAssertEqual(identicalPrefix([1,2,3,9], [1,2,3,4]), 3)
    XCTAssertEqual(identicalPrefix([1,2], [1,2]), 2)
  }
  func testEngagementDeltaRequiresStrictIncrease() {
    XCTAssertTrue(EngagementCheck(marker: "pld", floor: 1).passed(before: 4, after: 6))
    XCTAssertFalse(EngagementCheck(marker: "pld", floor: 1).passed(before: 6, after: 6)) // presence != engagement
  }
  func testAcceptanceFloor() {
    XCTAssertTrue(AcceptanceCheck(floor: 0.5).passed(rate: 0.66))
    XCTAssertFalse(AcceptanceCheck(floor: 0.5).passed(rate: 0.11))
  }
}
```
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement**
```swift
/// Longest identical prefix of two token streams.
public func identicalPrefix(_ a: [Int], _ b: [Int]) -> Int {
    var i = 0; while i < a.count && i < b.count && a[i] == b[i] { i += 1 }; return i
}

/// Equivalence vs a reference at temp=0. First-N (not full) — INT4/MoE float-reduction order
/// legitimately flips near-tie argmax past a per-family horizon (backlog). `minPrefix` is the
/// documented, tunable gate; below it means a real bug, not numerics.
public struct EquivalenceCheck {
    public let minPrefix: Int
    public init(minPrefix: Int = 30) { self.minPrefix = minPrefix }
    public func evaluate(candidate: [Int], reference: [Int]) -> (prefix: Int, passed: Bool) {
        let p = identicalPrefix(candidate, reference); return (p, p >= minPrefix)
    }
}

/// Engagement DELTA (not presence): the run's structured counter must strictly increase.
public struct EngagementCheck {
    public let marker: String; public let floor: Int
    public init(marker: String, floor: Int = 1) { self.marker = marker; self.floor = floor }
    public func passed(before: Int, after: Int) -> Bool { (after - before) >= floor }
}

/// Effectiveness floor: a feature can engage every request yet accept ~0% and degenerately fall back.
public struct AcceptanceCheck {
    public let floor: Double
    public init(floor: Double) { self.floor = floor }
    public func passed(rate: Double) -> Bool { rate >= floor }
}

public struct TriadVerdict: Sendable {
    public let equivalencePrefix: Int; public let engaged: Bool; public let acceptanceOK: Bool?
    public var passed: Bool { equivalencePrefix >= 0 /* set by caller */ && engaged && (acceptanceOK ?? true) }
    public init(equivalencePrefix: Int, engaged: Bool, acceptanceOK: Bool?) {
        self.equivalencePrefix = equivalencePrefix; self.engaged = engaged; self.acceptanceOK = acceptanceOK
    }
}
```
- [ ] **Step 4: Run → PASS.** **Step 5: Commit** `harness: equivalence+engagement+acceptance triad (pure, tested)`.

---

## Task 4: Hermetic corpus + universal invariants

Weight-free (input→expected) pairs through pure functions, plus invariants that cover the whole table so growth needs no new invariant code (backlog §6.2).

**Files:** Create `spike/Sources/HarnessCore/Corpus.swift`; Test `spike/Tests/HarnessCoreTests/CorpusTests.swift`

- [ ] **Step 1: Failing test** — one entry + the two universal invariants (no control-tag leak; tool-args valid JSON), plus a deliberately hostile-byte entry.
```swift
import XCTest
@testable import HarnessCore
final class CorpusTests: XCTestCase {
  func testUniversalInvariantsHoldAcrossCorpus() throws {
    for e in HarnessCorpus.entries {
      let out = HarnessCorpus.process(e.raw)         // pure: strip think-tags, parse tool calls
      XCTAssertFalse(out.visibleText.contains("<|"), "control-tag leak in \(e.name)")
      if let args = out.toolArgsJSON { XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(args.utf8))) }
      if let exp = e.expectedVisible { XCTAssertEqual(out.visibleText, exp, e.name) }
    }
  }
}
```
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement** `CorpusEntry` (`name`, `raw`, `expectedVisible?`, `expectedTool?`), a pure `process(_:)` placeholder (real tag-strip/tool-parse wired to the engine's chat layer when it exists — for now a minimal implementation that satisfies the invariants), and a seed `entries` table incl. one hostile-byte case. Header documents the harvest workflow (capture real model output → paste as an entry).
- [ ] **Step 4: Run → PASS.** **Step 5: Commit** `harness: hermetic corpus + universal invariants (no-tag-leak, valid-JSON, hostile-byte)`.

---

## Task 5: `QualityMetric` + `KLDivergenceMetric` (the dial's primitive)

KL-vs-reference is the cheap, most-sensitive precision-loss signal (spec §4). Protocol first so perplexity/task metrics slot in later (§6.7 pluggability).

**Files:** Create `spike/Sources/HarnessCore/QualityMetric.swift`; Test `spike/Tests/HarnessCoreTests/KLTests.swift`

- [ ] **Step 1: Failing test** (pure KL math on fixed distributions)
```swift
import XCTest
@testable import HarnessCore
final class KLTests: XCTestCase {
  func testKLZeroForIdentical() {
    let p: [Float] = [0.5,0.5]; XCTAssertEqual(klDivergence(reference: p, candidate: p), 0, accuracy: 1e-6)
  }
  func testKLPositiveAndKnown() {
    // KL(P=[0.9,0.1] || Q=[0.5,0.5]) ≈ 0.368 nats
    XCTAssertEqual(klDivergence(reference: [0.9,0.1], candidate: [0.5,0.5]), 0.368, accuracy: 1e-3)
  }
}
```
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement**
```swift
import Foundation
/// KL(reference || candidate) in nats, over aligned probability vectors. Clamps for numerical safety.
public func klDivergence(reference p: [Float], candidate q: [Float]) -> Double {
    precondition(p.count == q.count)
    var kl = 0.0
    for i in p.indices where p[i] > 0 {
        let qi = max(Double(q[i]), 1e-12); kl += Double(p[i]) * (log(Double(p[i])) - log(qi))
    }
    return kl
}

public protocol QualityMetric: Sendable { var name: String { get }
    /// Measure candidate-vs-reference over a fixed corpus using the driver's logprobs. Lower = closer to reference.
    func measure(driver: EngineDriver, reference: EngineDriver, prompts: [[Int]], config: RunConfig) async throws -> Double }

/// Median per-position KL of the candidate's next-token distribution vs the fp16 reference's.
public struct KLDivergenceMetric: QualityMetric {
    public let name = "kl_median"; public init() {}
    public func measure(driver: EngineDriver, reference: EngineDriver, prompts: [[Int]], config: RunConfig) async throws -> Double {
        var kls: [Double] = []
        for p in prompts {
            let c = try await driver.logprobs(prompt: p, config: config)
            let r = try await reference.logprobs(prompt: p, config: config)
            for i in 0..<min(c.count, r.count) { kls.append(klDivergence(reference: softmax(r[i]), candidate: softmax(c[i]))) }
        }
        kls.sort(); return kls.isEmpty ? 0 : kls[kls.count/2]
    }
}
func softmax(_ x: [Float]) -> [Float] { let m = x.max() ?? 0; let e = x.map { expf($0 - m) }; let s = e.reduce(0,+); return e.map { $0/s } }
```
- [ ] **Step 4: Run → PASS.** **Step 5: Commit** `harness: QualityMetric protocol + KL-divergence primitive (the dial's cheap signal)`.

---

## Task 6: Perf-bench framework (cell matrix + methodology)

Bakes the backlog's methodology into the harness entry point so no future number is misleading.

**Files:** Create `spike/Sources/HarnessCore/BenchMatrix.swift`; Test `spike/Tests/HarnessCoreTests/BenchMatrixTests.swift`

- [ ] **Step 1: Failing test** (pure: warmup-drop + averaging + the release-build guard predicate)
```swift
import XCTest
@testable import HarnessCore
final class BenchMatrixTests: XCTestCase {
  func testDropsWarmupAndAverages() {
    let agg = aggregateRates([nil, 100.0, 102.0, 104.0]) // index 0 = warmup, dropped; nil = skipped
    XCTAssertEqual(agg.mean, 102.0, accuracy: 1e-9); XCTAssertEqual(agg.runs, 3)
  }
}
```
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement** `Workload {prefill,decode,echo,code}`, `Mode {none,pld,dspark}`, `Cell`, `aggregateRates` (drop run 0, average the rest), `saltPrompt(run:nonce:_:)`, `assertReleaseBuild()` (fail fast if a Debug/unoptimized build — the ReleaseFast lesson), and `BenchRow` → append-only CSV with `label,workload,mode,model,decode_tok_s,ttft_ms,quant,concurrency`. Rate comes from `DecodeMetrics` (stream-timed, not usage fields).
- [ ] **Step 4: Run → PASS.** **Step 5: Commit** `harness: perf-bench matrix + methodology (release-guard, warmup-drop, salted, stream-timed CSV)`.

---

## Task 7: `SwiftEngineDriver` — in-process driver over the compiled decode core

The one MLX-touching file. Wraps `CompiledMLXDecoder`/`InferenceActor` to satisfy `EngineDriver`. Code specifics follow the spike's `docs/api/mlx-swift-api-notes.md`; verify on llmbench.

**Files:** Create `spike/Sources/fastmlx-harness/SwiftEngineDriver.swift` (in the executable target — it imports both `HarnessCore` and `SpikeCore`/MLX); Modify `spike/Sources/fastmlx-harness/Harness.swift`

- [ ] **Step 1: Implement `SwiftEngineDriver`** — holds a loaded model + `InferenceActor` (per spike); `generate` runs greedy decode returning tokens + `submitTime`/`tokenTimes` (reuse the bench path) + engagement counters (start with `{"decode": nTokens}`; spec-decode markers wired when spec-decode lands); `logprobs` runs the forward and returns top-k logits per position (temp=0). Preserve the compiled-step + no-sync-readback path; Swift 6 clean.
- [ ] **Step 2: Reference driver.** `scripts/harness_reference.py` (extend the spike's `reference_tokens.py`): greedy first-N token ids AND per-position top-k logprobs from mlx-lm, JSON out. A `ReferenceDriver: EngineDriver` shells to it (or the harness reads its JSON) so equivalence + KL compare against fp16 mlx-lm.
- [ ] **Step 3: Verify on llmbench** — `fastmlx-harness verify --model <path> --min-prefix 30` loads the model, runs greedy decode, diffs first-N vs the Python reference. Expected: identical-prefix ≥30 on the dense control + the known-good MoE prompt (matches the spike's equivalence result). Build via xcodebuild; run in the quiet window.
- [ ] **Step 4: Commit** `harness: in-process SwiftEngineDriver over compiled decode core + mlx-lm reference driver`.

---

## Task 8: Wire the CLI end-to-end + first real harness run

**Files:** Modify `spike/Sources/fastmlx-harness/Harness.swift`

- [ ] **Step 1: Implement subcommands** (`@main async`, `Flags` parser like the spike):
  - `verify --model` → run the **triad** (equivalence vs reference; engagement delta on the decode counter; acceptance floor when spec-decode is on) → print a `TriadVerdict` + exit non-zero on fail.
  - `bench --model [--mode]` → the cell matrix → CSV row(s).
  - `corpus` → the hermetic corpus (no model) → pass/fail.
  - `kl --model [--kv-quant turbo4]` → `KLDivergenceMetric` candidate(dial point) vs fp16 reference → print median/p95 KL. (This is the dial's first real precision-loss measurement.)
- [ ] **Step 2: Run all four on llmbench** on Qwen3-30B-A3B-2507-4bit; capture outputs. Expected: `verify` PASS, `corpus` PASS, `bench` a CSV row ~155 tok/s (compiled), `kl` a small median KL for `turbo4` vs fp16 (the first measured dial point).
- [ ] **Step 3: Write a short run report** to `docs/superpowers/verdicts/2026-07-09-harness-spine-firstrun.md` (what each subcommand proved; the first KL number). **Step 4: Commit** `harness: CLI (verify/bench/corpus/kl) + first end-to-end run report`.

---

## Self-Review (completed inline)

- **Spec coverage (§6 spine):** triad (§6.1) = Task 3; hermetic corpus (§6.2) = Task 4; perf-bench methodology (§6.3) = Task 6; pluggable metric for the dial (§4/§6.7) = Task 5; the engine-agnostic driver seam (§6.7 extensibility) = Task 2/7. **Deferred to follow-on plans (noted, not gaps):** soak/stability (§6.4), API-conformance matrix (§6.5), the HTTP/OpenAI `EngineDriver` impl, and the full chat/tool-parse layer behind `Corpus.process` (stubbed in Task 4 until the engine has one).
- **Placeholder scan:** engine-coupled specifics in Tasks 7–8 reference the spike's verified `api-notes` rather than assumed signatures (a grounding step, as in the spike plan), and are verified on llmbench — not vague TODOs. `TriadVerdict.passed`'s prefix comparison is finalized by the caller (the CLI passes the `EquivalenceCheck.minPrefix`); Task 8 Step 1 wires it.
- **Type consistency:** `EngineDriver.generate/logprobs`, `RunConfig`, `RunResult`, `EngagementCounters` defined Task 2, used Tasks 5/7/8; `EquivalenceCheck/EngagementCheck/AcceptanceCheck` Task 3, used Task 8; `KLDivergenceMetric.measure(driver:reference:prompts:config:)` Task 5, used Task 8; `aggregateRates`/`BenchRow` Task 6, used Task 8.

---

## Follow-on plans (author after this lands)
1. **Soak + API-conformance matrix** (§6.4/§6.5) once the engine has an HTTP surface.
2. **HTTP/OpenAI `EngineDriver`** — makes the harness drive the Python plane / NVIDIA / proxied runtimes (the engine-agnostic payoff + migration insurance).
3. **First flywheel customers through the harness:** DFlash (gate on the acceptance-denominator check first) and oQ4e importance-calibrated weight-quant (a `kl`/task measurement on the produced checkpoints) — see the [intake log](../../reference/performance-technique-intake.md).

## Notes for the executor
- Pure logic (Tasks 3–6) runs anywhere and is the bulk of the value — get it green first; only Tasks 7–8 need llmbench + a model.
- The harness is the instrument every later number depends on — hold it to its own house rules (TDD, honest numbers). If a metric or check would be convenient-but-unsound, don't ship it.
