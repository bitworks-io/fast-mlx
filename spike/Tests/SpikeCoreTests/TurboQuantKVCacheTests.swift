import MLX
import MLXRandom
import XCTest

@testable import SpikeCore

/// TurboQuantKVCache invariants (Phase 2 Task 6, materialize-then-attend): the cache stores
/// `_prod` codes per token and must add NO error beyond the codec round-trip, while presenting
/// exactly the same shape/offset/mask discipline as `CompiledKVCache` to the decode path.
final class TurboQuantKVCacheTests: XCTestCase {

    /// Realistic non-unit-norm K/V rows: unit directions × per-row norms in [0.5, 5]
    /// (the regime Task 6a's normalized codec exists for).
    private func randomKV(batch b: Int, heads h: Int, n: Int, d: Int, seed: UInt64) -> MLXArray {
        let g = MLXRandom.normal([b, h, n, d], key: MLXRandom.key(seed))
        let unit = g / MLX.sqrt((g * g).sum(axis: -1, keepDims: true))
        let scales = MLXRandom.uniform(
            low: Float(0.5), high: Float(5.0), [b, h, n, 1], key: MLXRandom.key(seed &+ 1000))
        return unit * scales
    }

    /// Codec-only reference: dequant(quant(x)) for every head-vector row of `[b, h, n, d]`.
    private func codecRoundTrip(_ x: MLXArray, params p: TurboQuantParams) -> MLXArray {
        let d = x.dim(3)
        let code = TurboQuantCodec.quantizeProdNormalized(x.reshaped([-1, d]), params: p)
        return TurboQuantCodec.dequantizeProdNormalized(code, params: p).reshaped(x.shape)
    }

    func testMaterializedKVEqualsCodecRoundTripAndMatchesCompiledShapes() {
        let (b, h, d, capacity) = (1, 2, 64, 8)
        let p = TurboQuantParams(headDim: d, baseBits: 3, seed: 0)
        let cache = TurboQuantKVCache(capacity: capacity, params: p)
        let ref = CompiledKVCache(capacity: capacity)

        // prefill 3 tokens, then a single decode token — the decoder's two update patterns
        let k1 = randomKV(batch: b, heads: h, n: 3, d: d, seed: 1)
        let v1 = randomKV(batch: b, heads: h, n: 3, d: d, seed: 2)
        let k2 = randomKV(batch: b, heads: h, n: 1, d: d, seed: 3)
        let v2 = randomKV(batch: b, heads: h, n: 1, d: d, seed: 4)

        _ = cache.update(keys: k1, values: v1)
        _ = ref.update(keys: k1, values: v1)
        let (mk, mv) = cache.update(keys: k2, values: v2)
        let (rk, rv) = ref.update(keys: k2, values: v2)

        // Shape/offset/mask discipline identical to CompiledKVCache for the same appends.
        XCTAssertEqual(mk.shape, rk.shape)
        XCTAssertEqual(mv.shape, rv.shape)
        XCTAssertEqual(cache.offset, ref.offset)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), ref.offsetArr.item(Int32.self))
        guard case .array(let mask) = cache.makeMask(n: 1, windowSize: nil, returnArray: true),
            case .array(let refMask) = ref.makeMask(n: 1, windowSize: nil, returnArray: true)
        else { return XCTFail("expected array masks") }
        XCTAssertTrue((mask .== refMask).all().item(Bool.self), "mask must match CompiledKVCache")

        // Filled rows: the cache reproduces dequant(quant(·)) within float tolerance — the
        // scatter/gather/reshape plumbing is lossless, no error beyond the codec itself.
        let kAll = concatenated([k1, k2], axis: 2)  // [b, h, 4, d]
        let vAll = concatenated([v1, v2], axis: 2)
        let kErr = (mk[0..., 0..., 0 ..< 4, 0...] - codecRoundTrip(kAll, params: p))
            .abs().max().item(Float.self)
        let vErr = (mv[0..., 0..., 0 ..< 4, 0...] - codecRoundTrip(vAll, params: p))
            .abs().max().item(Float.self)
        print("turboquant kvcache: max |cache − codec round-trip| K \(kErr), V \(vErr)")
        XCTAssertLessThan(kErr, 1e-4, "cache K must equal the codec round-trip")
        XCTAssertLessThan(vErr, 1e-4, "cache V must equal the codec round-trip")

        // Self-guard: the materialized K must actually be quantized (lossy vs the raw input),
        // so a passthrough cache can never satisfy this test.
        let lossK = (mk[0..., 0..., 0 ..< 4, 0...] - kAll).abs().max().item(Float.self)
        XCTAssertGreaterThan(lossK, 1e-3, "materialized K should differ from raw fp input")

        // Padded tail materializes to exactly 0 (stored ‖x‖ = 0), like CompiledKVCache's
        // zero padding — masked out of attention either way.
        XCTAssertEqual(mk[0..., 0..., 4..., 0...].abs().max().item(Float.self), 0)
        XCTAssertEqual(mv[0..., 0..., 4..., 0...].abs().max().item(Float.self), 0)
    }

    // MARK: Task 7 — tier selection plumbing

    func testKVCacheKindMappingAndFactory() {
        XCTAssertEqual(KVCacheKind(kvQuant: nil), .fp16)
        XCTAssertEqual(KVCacheKind(kvQuant: "fp16"), .fp16)
        XCTAssertEqual(KVCacheKind(kvQuant: "tq2.5"), .turboQuant(.tqB2))
        XCTAssertEqual(KVCacheKind(kvQuant: "tq3.5"), .turboQuant(.tqB3))
        XCTAssertEqual(KVCacheKind(kvQuant: "tqB2"), .turboQuant(.tqB2))
        XCTAssertEqual(KVCacheKind(kvQuant: "tqB3"), .turboQuant(.tqB3))
        XCTAssertNil(KVCacheKind(kvQuant: "8"), "unknown tiers must NOT silently fall back to fp16")
        XCTAssertTrue(KVCacheKind.fp16.makeCache(capacity: 4) is CompiledKVCache)
        XCTAssertTrue(KVCacheKind.turboQuant(.tqB3).makeCache(capacity: 4) is TurboQuantKVCache)
    }

    /// The decoder can't know head_dim before the model's first K/V arrives, so the tier init
    /// resolves `TurboQuantParams` lazily on first update — and must derive exactly the params
    /// an explicit `TurboQuantParams(headDim:baseBits:seed:)` gives (same seed, same Π/S/codebook).
    func testTierInitResolvesParamsLazilyFromFirstUpdate() {
        let d = 64
        let k = randomKV(batch: 1, heads: 2, n: 2, d: d, seed: 31)
        let v = randomKV(batch: 1, heads: 2, n: 2, d: d, seed: 32)
        let lazyCache = TurboQuantKVCache(capacity: 4, tier: .tqB3, seed: 0)
        let fixedCache = TurboQuantKVCache(
            capacity: 4, params: TurboQuantParams(headDim: d, baseBits: 3, seed: 0))
        let (lk, lv) = lazyCache.update(keys: k, values: v)
        let (fk, fv) = fixedCache.update(keys: k, values: v)
        XCTAssertEqual((lk - fk).abs().max().item(Float.self), 0)
        XCTAssertEqual((lv - fv).abs().max().item(Float.self), 0)
    }

    func testGrowExtendsAndResetInPlacePreservesIdentity() {
        let d = 64
        let p = TurboQuantParams(headDim: d, baseBits: 2, seed: 0)
        let cache = TurboQuantKVCache(capacity: 4, params: p)
        let k = randomKV(batch: 1, heads: 2, n: 2, d: d, seed: 9)
        let v = randomKV(batch: 1, heads: 2, n: 2, d: d, seed: 10)
        _ = cache.update(keys: k, values: v)

        // chunked growth: capacity extends, previously written codes survive
        cache.grow(by: 4)
        XCTAssertEqual(cache.capacity, 8)
        let k2 = randomKV(batch: 1, heads: 2, n: 1, d: d, seed: 11)
        let v2 = randomKV(batch: 1, heads: 2, n: 1, d: d, seed: 12)
        let (mk, _) = cache.update(keys: k2, values: v2)
        XCTAssertEqual(mk.shape, [1, 2, 8, d])
        let survived = (mk[0..., 0..., 0 ..< 2, 0...] - codecRoundTrip(k, params: p))
            .abs().max().item(Float.self)
        XCTAssertLessThan(survived, 1e-4, "pre-growth codes must survive grow(by:)")
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 3)

        // in-place reset: the exact MLXArray objects a compiled step is bound to must survive
        let objs = cache.innerState()
        cache.resetInPlace()
        XCTAssertTrue(
            zip(objs, cache.innerState()).allSatisfy { $0 === $1 },
            "resetInPlace must preserve MLXArray identities")
        XCTAssertEqual(cache.offset, 0)
        XCTAssertEqual(cache.offsetArr.item(Int32.self), 0)
    }
}
