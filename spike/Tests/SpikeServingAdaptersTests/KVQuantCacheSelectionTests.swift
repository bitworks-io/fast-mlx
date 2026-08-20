import HarnessCore
import MLXLMCommon
import ServingCore
import XCTest

@testable import SpikeServingAdapters

final class KVQuantCacheSelectionTests: XCTestCase {
    private let allKinds: [ScalarServingNativeCacheKind] = [
        .denseAttention, .rotatingAttention, .recurrentState, .composite, .unknown,
    ]

    // MARK: fp16 always valid, regardless of native cache kinds.

    func testFP16ReturnsFP16ForEachKindListIncludingEmpty() throws {
        for kind in allKinds {
            let decision = try selectKVCacheQuant(requested: .fp16, nativeKinds: [kind])
            XCTAssertEqual(decision, .fp16, "fp16 must be valid for kind \(kind)")
        }
        let decisionAllKinds = try selectKVCacheQuant(requested: .fp16, nativeKinds: allKinds)
        XCTAssertEqual(decisionAllKinds, .fp16)
        let decisionEmpty = try selectKVCacheQuant(requested: .fp16, nativeKinds: [])
        XCTAssertEqual(decisionEmpty, .fp16)
    }

    // MARK: int8 is inert today (default runtimeWired == [.fp16]).

    func testInt8ThrowsTierNotRuntimeWiredWithDefaultRuntimeWired() {
        XCTAssertThrowsError(
            try selectKVCacheQuant(requested: .int8, nativeKinds: [.denseAttention])
        ) { error in
            XCTAssertEqual(error as? KVQuantSelectionError, .tierNotRuntimeWired(.int8))
        }
    }

    // MARK: int8 activates once runtimeWired includes it, and only for all-dense routes.

    func testInt8ReturnsQuantizedDecisionWithBits8WhenWiredAndAllDense() throws {
        let decision = try selectKVCacheQuant(
            requested: .int8,
            nativeKinds: [.denseAttention, .denseAttention],
            runtimeWired: [.fp16, .int8]
        )
        guard case .int8(let groupSize, let bits) = decision else {
            XCTFail("expected .int8 decision, got \(decision)")
            return
        }
        XCTAssertEqual(groupSize, 32)
        XCTAssertEqual(bits, 8, "must pin bits: 8 explicitly, NOT the toQuantized default of 4")
    }

    func testInt8ThrowsTierIncompatibleWithRouteForEachNonDenseKind() {
        let nonDenseKinds: [ScalarServingNativeCacheKind] = [
            .rotatingAttention, .recurrentState, .composite, .unknown,
        ]
        for kind in nonDenseKinds {
            let kinds: [ScalarServingNativeCacheKind] = [.denseAttention, kind]
            XCTAssertThrowsError(
                try selectKVCacheQuant(requested: .int8, nativeKinds: kinds, runtimeWired: [.fp16, .int8])
            ) { error in
                XCTAssertEqual(
                    error as? KVQuantSelectionError,
                    .tierIncompatibleWithRoute(tier: .int8, kinds: kinds),
                    "expected tierIncompatibleWithRoute for kind \(kind)"
                )
            }
        }
    }

    func testInt8ThrowsEmptyCacheRouteWhenWiredButKindsEmpty() {
        XCTAssertThrowsError(
            try selectKVCacheQuant(requested: .int8, nativeKinds: [], runtimeWired: [.fp16, .int8])
        ) { error in
            XCTAssertEqual(error as? KVQuantSelectionError, .emptyCacheRoute(.int8))
        }
    }

    // MARK: unmeasured/unbuilt tiers never auto-serve, regardless of route shape.

    func testUnmeasuredTiersThrowTierNotRuntimeWired() {
        for tier: KVQuantTier in [.turbo4, .tq2_5, .tq3_5] {
            XCTAssertThrowsError(
                try selectKVCacheQuant(
                    requested: tier,
                    nativeKinds: [.denseAttention],
                    runtimeWired: [.fp16, .int8, .turbo4, .tq2_5, .tq3_5]
                )
            ) { error in
                XCTAssertEqual(error as? KVQuantSelectionError, .tierNotRuntimeWired(tier))
            }
        }
    }

    // MARK: construction seam — a decision builds the caches a route actually serves.

    /// fp16 is the byte-identical default: the construction seam returns the SAME native cache
    /// instances (identity, not copies), so routing a route's cache through `buildRouteKVCaches` under
    /// an fp16 decision cannot change what the runtime serves.
    func testBuildRouteKVCachesFP16ReturnsNativeInstancesUnchanged() {
        let natives: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
        let built = buildRouteKVCaches(decision: .fp16, nativeCaches: natives)
        XCTAssertEqual(built.count, natives.count)
        for (b, n) in zip(built, natives) {
            XCTAssertTrue(b as AnyObject === n as AnyObject, "fp16 must pass native instances through")
        }
    }

    /// int8 builds one vendored QuantizedKVCache per native cache with the pinned int8 geometry
    /// (groupSize 32, bits 8, affine) — NOT the toQuantized bits:4 default.
    func testBuildRouteKVCachesInt8BuildsQuantizedCachePerNativeWithPinnedGeometry() {
        let natives: [any KVCache] = [KVCacheSimple(), KVCacheSimple(), KVCacheSimple()]
        let built = buildRouteKVCaches(decision: .int8(groupSize: 32, bits: 8), nativeCaches: natives)
        XCTAssertEqual(built.count, natives.count)
        for cache in built {
            guard let quantized = cache as? QuantizedKVCache else {
                XCTFail("expected QuantizedKVCache, got \(type(of: cache))")
                continue
            }
            XCTAssertEqual(quantized.groupSize, 32)
            XCTAssertEqual(quantized.bits, 8)
            XCTAssertEqual(quantized.mode, .affine)
        }
    }

    /// End-to-end: an all-dense route with int8 wired composes selection → construction into int8
    /// QuantizedKVCache. Proves the two halves of the seam agree (the M5 activation flips
    /// runtimeWired; this is what it then builds).
    func testSelectionComposesIntoInt8ConstructionForWiredAllDenseRoute() throws {
        let decision = try selectKVCacheQuant(
            requested: .int8,
            nativeKinds: [.denseAttention, .denseAttention],
            runtimeWired: [.fp16, .int8])
        let built = buildRouteKVCaches(
            decision: decision,
            nativeCaches: [KVCacheSimple(), KVCacheSimple()])
        XCTAssertEqual(built.count, 2)
        XCTAssertTrue(built.allSatisfy { $0 is QuantizedKVCache })
    }
}
