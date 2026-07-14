# TurboQuant KV-Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Route engine/MLX tasks (Phase 1B onward) to `deep-reasoner` (fable), not `builder`** — prior MLX-coupled engine work (compiled decode, teacher-forcing) exceeded the builder tier and had to be re-routed up. Pure off-box tasks (Phase 1A) are `builder`-appropriate.

**Goal:** Implement Google TurboQuant (arXiv:2504.19874) as a custom KV-cache quantizer in the fast-mlx Swift engine, and measure its precision-loss/size frontier against the established 4-bit-weights/fp16-KV-vs-bf16 reference row — the flywheel's first genuinely novel technique.

> **Evidence clarification — 2026-07-14:** the completed run never measured an ordinary
> affine-KV quality row. The 1.665-nat / +21.4% row is 4-bit weights with fp16 KV versus bf16.
> Affine appeared only as theoretical packed-byte accounting. The KVarN/asymmetric cycle owns
> the missing same-weights affine-KV comparator; historical gates below should be read with
> that correction.

**Architecture:** TurboQuant_prod = a dense-Haar rotation + a non-uniform Lloyd-Max LUT quantizer (`b` base bits) + a 1-bit QJL sign-residual that makes the inner-product estimator unbiased-in-expectation. MLX's native `quantize` is affine-only and cannot represent this, so it's a fully custom build. v1 uses **materialize-then-attend** (dequant stored K/V to full precision, then the existing fused SDPA) — this keeps the KV-cache *storage* win; the attention-bandwidth win via a fused dequant-in-SDPA kernel is explicit backlog. Correctness is proven in a pure/off-box codebook layer + on-box mathematical-property tests, then integrated into `CompiledKVCache`, then run through the hardened harness.

**Tech Stack:** Swift 6 (strict concurrency), mlx-swift 0.31.6 (`MLX` matmul/random/linalg, `MLXFast.scaledDotProductAttention`), the pure `HarnessCore` (codebook), the `SpikeCore` engine module (MLX quantizer + KV cache), the `fastmlx-harness` CLI + `harness_reference.py` (measurement). Source of truth for the algorithm + confirmed constants + open gaps: [`docs/reference/turboquant-algorithm.md`](../../reference/turboquant-algorithm.md).

---

## Design decisions locked for v1 (correctness-first; alternatives are measured-gated follow-ups)

| Decision | v1 choice | Alternative (deferred, gated on measurement) |
|---|---|---|
| Rotation Π | **Dense Haar** (QR of a Gaussian), the paper's spec — correctness first | Fast randomized-Hadamard + re-fit Lloyd-Max (perf); only if rotation cost shows up in the bench AND quality holds |
| LUT dequant | **Plain MLX `take`/gather** — simplest correct op | A `MLXFast.metalKernel` LUT kernel; only if gather is a measured bottleneck |
| Bit allocation | **Uniform** `b` base bits/channel + 1 QJL bit | Outlier channels (paper says this is borrowed/non-novel); only if 2.5-bit quality misses the gate |
| K vs V precision | **Same** `b` for K and V | Asymmetric; only if measurement shows K/V sensitivity differs |
| Attention read | **Materialize-then-attend** (dequant → existing SDPA) | Fused dequant-in-SDPA Metal kernel — **backlog**, not this plan |

**Honest naming:** v1 is *uniform* `(b base + 1 QJL)` = `b+1` bits/element (a `b=2` tier is ~3-bit/element, `b=3` is ~4-bit/element). The paper's "2.5-bit / 3.5-bit" labels require the deferred outlier-channel mixing, so internal tiers are named **`tqB2`** (2 base + QJL) and **`tqB3`** (3 base + QJL); they occupy the harness's `tq2.5`/`tq3.5` recording slots with a documented note that the sub-integer labels await outlier channels. **Never report a `tqB3` result under a "2.5-bit" label.**

**Scheduled spikes (decision points inside the phases, not hand-waves):**
- **Spike A (Phase 1B, Task 4) — ✅ RESOLVED 2026-07-09.** Verified the codec reproduces the paper's Theorem-2 distortion table (`d·D_prod` 0.175/0.0514 vs paper 0.18/0.047) at d=128, with unbiasedness slope 1.0066 — the scale `√(π/2)/d` is confirmed verbatim. The gate *test* was corrected (the naive `prodErr<mseErr` on independent pairs provably favors `_mse`; `_prod` wins in attention's *correlated* regime). Full derivation in the [Spike A resolution](../../reference/turboquant-algorithm.md#spike-a-resolution-2026-07-09-on-box-d128-property-tests--arxiv-html-v1-re-check); the distortion-bound constant is `√3·π²`, not the earlier `3π²`.
- **Spike B (Phase 3):** if `tqB2` (the aggressive tier) fails the quality gate, the decision is *outlier channels vs shelve* — do not silently ship a failing tier.

---

## File Structure

- `spike/Sources/HarnessCore/TurboQuant/LloydMaxCodebook.swift` — **pure** (Foundation-only) Lloyd-Max codebook generation. Off-box testable. One responsibility: given `bits`, return the N(0,1)-optimal centroids.
- `spike/Sources/SpikeCore/TurboQuant/TurboQuantParams.swift` — the fixed global params (Π, S, the 1/√d-scaled codebook) per `head_dim`, generated once. MLX arrays.
- `spike/Sources/SpikeCore/TurboQuant/TurboQuantCodec.swift` — the MLX quantize/dequantize (`_mse` and `_prod`) operating on batched KV tensors.
- `spike/Sources/SpikeCore/TurboQuant/TurboQuantKVCache.swift` — the KV-cache type: stores `(codes, signs, norms)` per token, dequant-on-read → materialize K/V. Conforms to the cache protocol the decode path consumes.
- `spike/Tests/HarnessCoreTests/LloydMaxCodebookTests.swift` — off-box, vs the paper's confirmed constants.
- `spike/Tests/SpikeCoreTests/TurboQuantCodecTests.swift` — on-box, mathematical-property tests.
- `spike/Tests/SpikeCoreTests/TurboQuantKVCacheTests.swift` — on-box, cache round-trip + equivalence.
- Modify: `spike/Sources/HarnessCore/EngineDriver.swift` / `fastmlx-harness` tier plumbing (record `tqB2`/`tqB3`), `harness_reference.py` (already supports forced scoring; no TurboQuant needed on the reference side — the reference stays bf16).

**Build/test:** Phase 1A off-box on this host (`swift test --filter HarnessCoreTests`). Phase 1B onward on `llmbench@192.168.1.252` (rsync via `scripts/sync_llmbench.sh`; `xcodebuild -skipPackagePluginValidation`; model + bf16 reference already staged from the harness-hardening work). Branch `feat/turboquant` off `main`.

---

## Phase 1A — Lloyd-Max codebook (pure, off-box, fully TDD)

### Task 1: Gaussian Lloyd-Max codebook

**Files:**
- Create: `spike/Sources/HarnessCore/TurboQuant/LloydMaxCodebook.swift`
- Test: `spike/Tests/HarnessCoreTests/LloydMaxCodebookTests.swift`

- [ ] **Step 1: Write the failing test** — assert the codebook matches the paper's confirmed constants (`docs/reference/turboquant-algorithm.md` §"The algorithm"): b=1 → `±√(2/π)`; b=2 → `±0.4528, ±1.510` (the classical Max-1960 Gaussian levels; the paper's `±0.453/√d, ±1.51/√d` are these × the 1/√d the *codec* applies later). Codebook is on N(0,1), sorted ascending.

```swift
import XCTest
@testable import HarnessCore

final class LloydMaxCodebookTests: XCTestCase {
    func testB1MatchesSqrt2OverPi() {
        let c = LloydMaxCodebook.gaussian(bits: 1)
        XCTAssertEqual(c.count, 2)
        let expected = (2.0 / Double.pi).squareRoot() // 0.79788
        XCTAssertEqual(c[0], -expected, accuracy: 1e-4)
        XCTAssertEqual(c[1], expected, accuracy: 1e-4)
    }

    func testB2MatchesMax1960Levels() {
        let c = LloydMaxCodebook.gaussian(bits: 2)
        XCTAssertEqual(c.count, 4)
        let expected = [-1.5104, -0.4528, 0.4528, 1.5104]
        for (a, e) in zip(c, expected) { XCTAssertEqual(a, e, accuracy: 2e-3) }
    }

    func testMonotoneAndSymmetric() {
        let c = LloydMaxCodebook.gaussian(bits: 3)
        XCTAssertEqual(c.count, 8)
        XCTAssertEqual(c, c.sorted(), "centroids must be ascending")
        for i in 0..<4 { XCTAssertEqual(c[i], -c[7 - i], accuracy: 1e-3, "symmetric about 0") }
    }
}
```

- [ ] **Step 2: Run it, verify it fails** — `swift test --filter LloydMaxCodebookTests` → FAIL ("cannot find 'LloydMaxCodebook'").

- [ ] **Step 3: Implement** the Lloyd-Max iteration (boundaries = midpoints; centroids = Gaussian conditional means `E[X | a<X≤b] = (φ(a)−φ(b))/(Φ(b)−Φ(a))`, verified: `φ(0)/(1−Φ(0)) = √(2/π)`).

```swift
import Foundation

/// Lloyd-Max optimal scalar quantizer levels for the standard normal N(0,1) (TurboQuant §"The
/// algorithm": post-rotation coordinates are ≈ N(0, 1/d), so we solve on N(0,1) once and let the
/// codec rescale by 1/√d). Pure — no MLX. Solved numerically (no closed form; the paper caches a LUT).
public enum LloydMaxCodebook {
    private static func phi(_ x: Double) -> Double {          // N(0,1) pdf
        guard x.isFinite else { return 0 }
        return exp(-x * x / 2) / (2 * Double.pi).squareRoot()
    }
    private static func bigPhi(_ x: Double) -> Double {        // N(0,1) cdf
        if x == .infinity { return 1 }; if x == -.infinity { return 0 }
        return 0.5 * (1 + erf(x / 2.0.squareRoot()))
    }

    /// `2^bits` ascending centroids for N(0,1). Deterministic (fixed init + fixed iteration count).
    public static func gaussian(bits: Int, iterations: Int = 200) -> [Double] {
        precondition(bits >= 1, "bits must be ≥ 1")
        let k = 1 << bits
        // Deterministic init: evenly spaced in [-3, 3].
        var c = (0..<k).map { 3.0 * (2.0 * (Double($0) + 0.5) / Double(k) - 1.0) }
        for _ in 0..<iterations {
            var t = [Double](repeating: 0, count: k + 1)      // decision boundaries
            t[0] = -.infinity; t[k] = .infinity
            for i in 1..<k { t[i] = (c[i - 1] + c[i]) / 2 }
            for i in 0..<k {                                  // centroid = conditional mean
                let den = bigPhi(t[i + 1]) - bigPhi(t[i])
                if den > 1e-12 { c[i] = (phi(t[i]) - phi(t[i + 1])) / den }
            }
        }
        return c
    }
}
```

- [ ] **Step 4: Run tests, verify pass** — `swift test --filter LloydMaxCodebookTests` → PASS (3 tests).

- [ ] **Step 5: Commit** — `git commit -am "turboquant: Lloyd-Max Gaussian codebook (pure, matches paper b=1,2 constants)"`.

---

## Phase 1B — the MLX quantizer core (on-box, mathematical-property tests)

> All Phase 1B+ tasks run on `llmbench`. Route to `deep-reasoner`. Each task: property test → verify fail → implement → verify pass → commit.

### Task 2: Global params (Π, S, scaled codebook) per head_dim

**Files:** Create `spike/Sources/SpikeCore/TurboQuant/TurboQuantParams.swift`; Test `spike/Tests/SpikeCoreTests/TurboQuantCodecTests.swift`.

- [ ] **Step 1: Write the failing test** — Π is orthogonal (`Π Πᵀ ≈ I`), shapes are `[d,d]`, the scaled codebook has `2^baseBits` entries. Use `d=128` (real head_dim), a fixed seed.

```swift
import XCTest
import MLX
@testable import SpikeCore

final class TurboQuantCodecTests: XCTestCase {
    func testParamsRotationIsOrthogonal() {
        let p = TurboQuantParams(headDim: 128, baseBits: 3, seed: 0)
        let ident = p.rotation.matmul(p.rotation.transposed())
        let err = (ident - MLXArray.eye(128)).abs().max().item(Float.self)
        XCTAssertLessThan(err, 1e-3, "Π Πᵀ must be ≈ I")
        XCTAssertEqual(p.qjl.shape, [128, 128])
        XCTAssertEqual(p.scaledCentroids.shape, [8])
    }
}
```

- [ ] **Step 2: Verify fail** (`swift test --filter TurboQuantCodecTests` on llmbench → FAIL, type missing).

- [ ] **Step 3: Implement** — Π via QR of a seeded Gaussian; S a seeded Gaussian; centroids = `LloydMaxCodebook.gaussian(bits: baseBits)` scaled by `1/√d`, as an `MLXArray`.

```swift
import Foundation
import MLX
import HarnessCore

/// Fixed, once-generated TurboQuant parameters for a given head_dim (TurboQuant §"Setup"): a dense
/// Haar rotation Π, the QJL Gaussian S, and the 1/√d-scaled Lloyd-Max codebook. Generated once and
/// reused for every K/V vector (the paper's "Global Parameters"); seeded for reproducibility.
public struct TurboQuantParams: Sendable {
    public let headDim: Int
    public let baseBits: Int
    public let rotation: MLXArray        // Π  [d, d], orthogonal
    public let qjl: MLXArray             // S  [d, d], i.i.d. N(0,1)
    public let scaledCentroids: MLXArray // [2^baseBits], ascending, ×(1/√d)

    public init(headDim d: Int, baseBits: Int, seed: UInt64) {
        self.headDim = d; self.baseBits = baseBits
        var key = MLXRandom.key(seed)
        let (k1, k2) = MLXRandom.split(key: key)
        // Π = Q from QR of a Gaussian d×d (Haar-orthogonal).
        let g = MLXRandom.normal([d, d], key: k1)
        let (q, _) = MLXLinalg.qr(g)
        self.rotation = q
        self.qjl = MLXRandom.normal([d, d], key: k2)
        let c = LloydMaxCodebook.gaussian(bits: baseBits).map { Float($0) / Float(Double(d).squareRoot()) }
        self.scaledCentroids = MLXArray(c)
        key = k2
    }
}
```
*(If `MLXLinalg.qr` is unavailable in 0.31.6, fall back to modified Gram-Schmidt on `g` — a documented ~15-line helper; verify the orthogonality test either way.)*

- [ ] **Step 4: Verify pass. Step 5: Commit** — `"turboquant: fixed per-head_dim params (Haar Π, QJL S, scaled codebook)"`.

### Task 3: `_mse` quantize/dequantize (rotation + LUT)

**Files:** Create `spike/Sources/SpikeCore/TurboQuant/TurboQuantCodec.swift`; extend the test file.

- [ ] **Step 1: Failing test** — round-trip reconstruction error is bounded and **decreases with more base bits** (the core sanity that the LUT is doing real work). Input: 64 random unit-norm `[·,128]` vectors, fixed seed.

```swift
func testMSERoundTripErrorDecreasesWithBits() {
    let x = l2normalizeRows(MLXRandom.normal([64, 128], key: MLXRandom.key(7)))
    func recon(_ b: Int) -> Float {
        let p = TurboQuantParams(headDim: 128, baseBits: b, seed: 0)
        let codes = TurboQuantCodec.quantizeMSE(x, params: p)
        let xr = TurboQuantCodec.dequantizeMSE(codes, params: p)
        return (x - xr).square().mean().sqrt().item(Float.self)
    }
    let e2 = recon(2), e3 = recon(3)
    XCTAssertLessThan(e3, e2, "more base bits → smaller reconstruction error")
    XCTAssertLessThan(e3, 0.2, "3-base-bit reconstruction should be well under 0.2 RMSE on unit vectors")
}
```

- [ ] **Step 2: Verify fail. Step 3: Implement** — `y = x·Πᵀ` (rows rotated); `idx = argmin_j |y − c_j|` per element (broadcast against `scaledCentroids`); dequant `ỹ = centroids[idx]`, `x̃ = ỹ·Π`.

```swift
public enum TurboQuantCodec {
    /// Per-element nearest-centroid indices of the rotated vector (TurboQuant Algorithm 1).
    public static func quantizeMSE(_ x: MLXArray, params p: TurboQuantParams) -> MLXArray {
        let y = x.matmul(p.rotation.transposed())                    // [n, d]
        // |y[...,None] - centroids[None,...]| → argmin over the centroid axis.
        let diff = (y.expandedDimensions(axis: -1) - p.scaledCentroids).abs()
        return diff.argMin(axis: -1)                                 // [n, d] indices
    }
    public static func dequantizeMSE(_ idx: MLXArray, params p: TurboQuantParams) -> MLXArray {
        let yq = p.scaledCentroids[idx]                              // gather → [n, d]
        return yq.matmul(p.rotation)                                 // undo rotation (Πᵀ inverse = Π)
    }
}
private func l2normalizeRows(_ x: MLXArray) -> MLXArray {
    x / MLX.sqrt((x * x).sum(axis: -1, keepDims: true))
}
```

- [ ] **Step 4: Verify pass. Step 5: Commit** — `"turboquant: _mse codec (rotation + Lloyd-Max LUT round-trip)"`.

### Task 4: `_prod` QJL residual — the inner-product-preserving variant (**Spike A**)

**Files:** extend `TurboQuantCodec.swift` + the test.

- [ ] **Step 1: Failing test — the paper-faithful gate (⚠️ CORRECTED after Spike A, 2026-07-09).** The intuitive `prodErr < mseErr` on *independent* random query/key pairs is the **WRONG** gate and provably fails for a correct implementation: `_prod`'s unbiased QJL correction carries variance `≈ (π/2)·‖r‖²/d` with no `_mse` shrinkage *bias* to remove when `q ⊥ k`, so it loses on independent pairs (crossover ≈ `⟨q,x⟩ ~ 0.2`). Attention is the **correlated** regime (softmax weights the high scores), which is where `_prod` wins — so the gate asserts the three properties the paper actually guarantees (full derivation + numbers: [Spike A resolution](../../reference/turboquant-algorithm.md#spike-a-resolution-2026-07-09-on-box-d128-property-tests--arxiv-html-v1-re-check)). All three must pass on-box at d=128, float32:
  - (a) **Unbiasedness:** regression slope of `⟨q, dequantProd(quantProd(x))⟩` on `⟨q, x⟩` ≈ 1.0 (±0.05) — pins the `√(π/2)/d` scale (measured 1.0066).
  - (b) **Correlated-regime superiority** (the KV-relevant property): for `⟨q,x⟩ ≥ 0.45`, mean-abs inner-product error of `_prod` < `_mse` by margin ≥ 1.8× (measured 4.1× at `⟨q,x⟩=1`).
  - (c) **Theorem-2 table anchor** (the falsifiable pin that the codec *is* the paper's quantizer): `d·D_prod` within ~20% of the paper's `{…, 0.18, 0.047}` at total bits 3, 4 (measured 0.175 / 0.0514).

  Do NOT re-introduce the naive independent-pair assertion, and do NOT adopt an MMSE shrinkage factor to force a random-pair win — that breaks the unbiasedness Theorem 2 proves (and which composition across attention relies on).

- [ ] **Step 2: Verify fail. Step 3: Implement** the Algorithm-2 residual: `r = x − dequantMSE(idx)`; `signs = sign(r·Sᵀ)`; `γ = ‖r‖₂` per row; dequant adds `(√(π/2)/d)·γ·(signs·S)`.

```swift
public struct TurboQuantCode: Sendable {   // per-vector stored state (no per-group scale/zero)
    public let idx: MLXArray               // [n, d] base-bit indices
    public let signs: MLXArray             // [n, d] ±1 (1 bit/element)
    public let norms: MLXArray             // [n, 1] γ = ‖r‖₂ (fp16, amortized over d)
}
extension TurboQuantCodec {
    public static func quantizeProd(_ x: MLXArray, params p: TurboQuantParams) -> TurboQuantCode {
        let idx = quantizeMSE(x, params: p)
        let r = x - dequantizeMSE(idx, params: p)                    // residual in original space
        let signs = MLX.sign(r.matmul(p.qjl.transposed()))          // sign(S·r) per element
        let norms = MLX.sqrt((r * r).sum(axis: -1, keepDims: true))
        return TurboQuantCode(idx: idx, signs: signs, norms: norms)
    }
    public static func dequantizeProd(_ c: TurboQuantCode, params p: TurboQuantParams) -> MLXArray {
        let base = dequantizeMSE(c.idx, params: p)
        let scale = Float((Double.pi / 2).squareRoot() / Double(p.headDim))
        let qjlTerm = (c.signs.matmul(p.qjl)) * c.norms * scale       // (√(π/2)/d)·γ·(signs·S)
        return base + qjlTerm
    }
}
```

- [ ] **Step 3a (Spike A gate):** if the inner-product test does NOT pass, STOP. Re-derive the `√(π/2)/d` scale and the `sign(S·r)` vs `sign(r·Sᵀ)` convention against the PDF (reference doc flags norm/constant bookkeeping as VERIFY). Do not proceed to the cache on an unproven estimator. Record the resolution in the reference doc.
- [ ] **Step 4: Verify pass. Step 5: Commit** — `"turboquant: _prod QJL residual — inner-product-preserving codec (Spike A passed)"`.

### Task 5: Storage accounting + tier config

**Files:** small additions to `TurboQuantParams.swift` + test.

- [ ] **Step 1: Failing test** — a `TurboQuantTier` reports honest bits/element: `tqB2 → 3.0` (2 base + 1 QJL), `tqB3 → 4.0`, plus the amortized `γ` overhead `16/d` bits/element. Assert `tqB2.bitsPerElement(headDim: 128) == 3 + 16.0/128`.
- [ ] **Step 2–4:** implement a `TurboQuantTier` enum (`tqB2`, `tqB3`) with `baseBits` + `bitsPerElement(headDim:)`, and its mapping to the harness recording slot (`tq2.5`/`tq3.5`) **with the documented "uniform, awaiting outlier channels" note**. Commit — `"turboquant: tier config + honest bits/element accounting"`.

---

## Phase 2 — TurboQuantKVCache + materialize-then-attend (on-box; deep-reasoner)

**Gate to enter Phase 2:** Spike A passed (inner products preserved at d=128).

### Task 6: `TurboQuantKVCache` — store codes, dequant-on-read

**Files:** Create `spike/Sources/SpikeCore/TurboQuant/TurboQuantKVCache.swift`; Test `TurboQuantKVCacheTests.swift`. **Read first:** `spike/Sources/SpikeCore/CompiledKVCache.swift` (the fp16 cache it parallels) + how the decode path calls the cache.

- [ ] **Step 1: Failing test — cache round-trip.** Append `n` K/V vectors quantized, read them back materialized; assert the materialized K/V equals `dequantizeProd(quantizeProd(·))` (the cache adds no error beyond the codec) and the shapes match the fp16 cache's.
- [ ] **Step 2: Verify fail. Step 3: Implement** — a cache storing per-token `TurboQuantCode` for K and V (same `baseBits`), growing in chunks like `CompiledKVCache`; `keys()`/`values()` return **dequantized full-precision** tensors (materialize-then-attend). Mind the compiled-decode retrace concern: keep the stored-code buffers fixed-shape/chunked exactly as `CompiledKVCache` does, and dequantize into a materialized buffer the SDPA reads. Apply the 8 GiB `Memory.cacheLimit` discipline from the long-context work.
- [ ] **Step 4: Verify pass. Step 5: Commit** — `"turboquant: TurboQuantKVCache (store codes, dequant-on-read, materialize-then-attend)"`.

### Task 7: Decode-path integration + equivalence

**Files:** Modify the decode/attention call site to select `TurboQuantKVCache` when the tier is `tqB2`/`tqB3`; wire the tier flag through `RunConfig`.

- [ ] **Step 1: Failing test — the triad (lossy mode).** At temp=0, generating with `tqB3` KV must (a) not crash / not NaN, (b) match the fp16-KV output on a short prefix + pass the coherence canary (the §6.1 lossy triad — TurboQuant is a genuinely-lossy mode), and (c) **engagement**: assert the TurboQuant quant path actually ran (a structured marker, delta-checked), so a silent fp16 fallback can't pass. `tqB3` (near-lossless per the paper) is the right tier for the prefix-match arm; `tqB2` uses the canary-only arm.
- [ ] **Step 2: Verify fail. Step 3: Implement** the cache selection + tier plumbing + the engagement marker. **Spec §5 invariant:** speculative paths stay disabled in any batched arm; TurboQuant is orthogonal but don't cross the streams.
- [ ] **Step 4: Verify pass** (on llmbench, real Qwen3-32B checkpoint — toy dims don't exercise the reduction order). **Step 5: Commit** — `"turboquant: decode-path integration + lossy-triad equivalence (tqB2/tqB3)"`.

---

## Phase 3 — measurement vs the bf16 baseline (the promote/shelve gate)

### Task 8: Quantify through the hardened harness

**Files:** run-only + a verdict doc; tier recording already plumbed (Task 5/7).

- [ ] **Step 1:** on llmbench, run teacher-forced KL + perplexity + **long-context tail-p95** for `tqB3` and `tqB2` KV vs the **bf16** reference on `corpus/measurement-corpus-v2.json` (incl. the 24,151-token entry), same protocol as the baseline:

```bash
# on llmbench, from the harness venv + release build
fastmlx-harness kl   --model <qwen3-32b-4bit> --reference <qwen3-32b-bf16> --kv-quant tqB3 --corpus measurement-corpus-v2
fastmlx-harness kl   --model <qwen3-32b-4bit> --reference <qwen3-32b-bf16> --kv-quant tqB2 --corpus measurement-corpus-v2
fastmlx-harness bench --model <qwen3-32b-4bit> --kv-quant tqB3   # decode tok/s + KV bytes/token
```

- [ ] **Step 2: Compare to the measured reference row** ([content piece](../../content/2026-07-09-the-wall-that-wasnt.md): 4-bit weights with fp16 KV vs bf16 = **tail-p95 1.665 nats @24K, ppl +21.4%**). Record `tqB3`/`tqB2` tail-p95 @24K, ppl-delta, decode tok/s, and **KV bytes/token** (the storage win) in `docs/superpowers/verdicts/2026-07-09-turboquant-firstrun.md`.
- [ ] **Step 3: Promote or shelve (explicit).**
  - **Promote** a tier to a dial `kvQuant` tier iff it establishes a useful measured quality/size frontier against the same-weights fp16-KV row; an ordinary affine-KV claim additionally requires its own teacher-forced row. Update the platform spec §4 dial axes.
  - **Shelve** with a **dated negative result** (the flywheel's discipline) if it doesn't — and if only `tqB2` fails, that triggers **Spike B** (outlier channels vs shelve `tqB2`, keep `tqB3`).
- [ ] **Step 4:** write the content-library piece (standing practice) — the honest arc: "we built the paper's exact quantizer; here's where it beat/didn't beat 4-bit affine, measured." **Step 5: Commit** the verdict + content.

---

## Self-Review

**1. Spec coverage** (vs `docs/reference/turboquant-algorithm.md`): rotation Π ✓ (Task 2), Lloyd-Max LUT ✓ (Tasks 1,3), `_prod` QJL residual ✓ (Task 4), `(idx, signs, γ)` no-per-group-metadata storage ✓ (Task 4 `TurboQuantCode`), materialize-then-attend ✓ (Task 6), 2.5/3.5 tiers ✓ (Task 5, honestly named `tqB2/tqB3` pending outliers), validation vs bf16 baseline ✓ (Task 8), KV-bytes/token storage win ✓ (Task 8). Open gaps from the reference are each dispositioned: fast rotation → deferred (locked table), b≥3 Lloyd-Max → Task 1 (general), granularity → per-head_dim (Task 2), outlier selection → deferred/Spike B, K-vs-V bits → same (locked), `√(π/2)/d` + `3π²` constants → Spike A gate (Task 4).

**2. Placeholder scan:** the only intentionally-non-exact tasks are Phase 2 (Tasks 6–7), which depend on `CompiledKVCache`'s internal shape/retrace details that must be read on-box first — flagged as "read first," not hand-waved, with exact test intents + interfaces. All Phase 1 tasks carry complete code. The QR fallback (Gram-Schmidt) is named, not left open.

**3. Type consistency:** `TurboQuantParams` (`rotation`/`qjl`/`scaledCentroids`/`headDim`/`baseBits`), `TurboQuantCode` (`idx`/`signs`/`norms`), `TurboQuantCodec.{quantizeMSE,dequantizeMSE,quantizeProd,dequantizeProd}`, `TurboQuantTier.{tqB2,tqB3}` — names are consistent across Tasks 2→8. `LloydMaxCodebook.gaussian(bits:)` consistent Task 1→2.

**Risk note:** Spike A (Task 4) is the make-or-break — if the inner-product estimator can't be validated at real head_dim, the whole KV application is unsound, and the plan correctly STOPS there rather than building a cache on a broken estimator.
