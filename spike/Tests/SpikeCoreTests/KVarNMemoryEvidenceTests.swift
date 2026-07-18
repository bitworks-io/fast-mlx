import MLX
import XCTest
@testable import SpikeCore

final class KVarNMemoryEvidenceTests: XCTestCase {
    func testProbeConfigurationAcceptsOnlyDeclaredQualificationCells() throws {
        let encode = try KVarNMemoryProbeConfiguration(
            phase: .encode, heads: 8, headDimension: 128,
            groupSize: 128, iterations: 8, capacity: 256,
            cacheLimitBytes: 0, run: 1)
        XCTAssertEqual(encode.phase, .encode)
        XCTAssertEqual(encode.rows, 8)
        XCTAssertEqual(encode.tileElementCount, 131_072)

        let boundary = try KVarNMemoryProbeConfiguration(
            phase: .cacheBoundary, heads: 8, headDimension: 128,
            groupSize: 128, iterations: 16, capacity: 24_192,
            cacheLimitBytes: 0, run: 3)
        XCTAssertEqual(boundary.completedTileCapacity, 188)

        for invalid in [
            { try KVarNMemoryProbeConfiguration(
                phase: .encode, heads: 0, headDimension: 128,
                groupSize: 128, iterations: 8, capacity: 256,
                cacheLimitBytes: 0, run: 1) },
            { try KVarNMemoryProbeConfiguration(
                phase: .decode, heads: 8, headDimension: 96,
                groupSize: 128, iterations: 8, capacity: 256,
                cacheLimitBytes: 0, run: 1) },
            { try KVarNMemoryProbeConfiguration(
                phase: .cacheBoundary, heads: 8, headDimension: 128,
                groupSize: 128, iterations: 4, capacity: 256,
                cacheLimitBytes: 0, run: 1) },
            { try KVarNMemoryProbeConfiguration(
                phase: .cacheBoundary, heads: 8, headDimension: 128,
                groupSize: 128, iterations: 8, capacity: 255,
                cacheLimitBytes: 0, run: 1) },
            { try KVarNMemoryProbeConfiguration(
                phase: .cacheBoundary, heads: 8, headDimension: 128,
                groupSize: 128, iterations: 8, capacity: 256,
                cacheLimitBytes: 1, run: 1) },
            { try KVarNMemoryProbeConfiguration(
                phase: .cacheBoundary, heads: 8, headDimension: 128,
                groupSize: 128, iterations: 8, capacity: 256,
                cacheLimitBytes: 0, run: 0) },
            { try KVarNMemoryProbeConfiguration(
                phase: .encode, heads: 1, headDimension: 128,
                groupSize: 128, iterations: 8, capacity: 256,
                cacheLimitBytes: 0, run: 1) },
            { try KVarNMemoryProbeConfiguration(
                phase: .encode, heads: 8, headDimension: 256,
                groupSize: 128, iterations: 8, capacity: 256,
                cacheLimitBytes: 0, run: 1) },
            { try KVarNMemoryProbeConfiguration(
                phase: .cacheBoundary, heads: 8, headDimension: 128,
                groupSize: 128, iterations: 8, capacity: 384,
                cacheLimitBytes: 0, run: 1) },
            { try KVarNMemoryProbeConfiguration(
                phase: .decode, heads: 8, headDimension: 128,
                groupSize: 128, iterations: 8, capacity: 256,
                cacheLimitBytes: 0, run: 4) },
            { try KVarNMemoryProbeConfiguration(
                phase: .cacheBoundary, heads: 8, headDimension: 128,
                groupSize: 128, iterations: 8, capacity: Int.max,
                cacheLimitBytes: 0, run: 1) },
        ] {
            XCTAssertThrowsError(try invalid()) {
                XCTAssertEqual(
                    $0 as? KVarNMemoryEvidenceError,
                    .invalidProbeConfiguration)
            }
        }
    }

    func testHighWaterSeparatesStableResidencyFromTransientActiveMemory() throws {
        let highWater = try KVarNMemoryHighWater(
            start: KVarNMemoryCounters(
                activeBytes: 100, cacheBytes: 0, peakActiveBytes: 0),
            end: KVarNMemoryCounters(
                activeBytes: 140, cacheBytes: 0, peakActiveBytes: 300))

        XCTAssertEqual(highWater.observedPeakActiveBytes, 300)
        XCTAssertEqual(highWater.retainedActiveBytes, 140)
        XCTAssertEqual(highWater.transientActiveAboveRetainedBytes, 160)
        XCTAssertEqual(highWater.incrementalPeakActiveBytes, 200)
    }

    func testHighWaterFailsClosedOnAllocatorCacheAndInvalidCounters() {
        XCTAssertThrowsError(try KVarNMemoryHighWater(
            start: KVarNMemoryCounters(
                activeBytes: 100, cacheBytes: 1, peakActiveBytes: 0),
            end: KVarNMemoryCounters(
                activeBytes: 100, cacheBytes: 0, peakActiveBytes: 200)
        )) {
            XCTAssertEqual(
                $0 as? KVarNMemoryEvidenceError,
                .allocatorCacheNotEmpty)
        }
        XCTAssertThrowsError(try KVarNMemoryHighWater(
            start: KVarNMemoryCounters(
                activeBytes: -1, cacheBytes: 0, peakActiveBytes: 0),
            end: KVarNMemoryCounters(
                activeBytes: 100, cacheBytes: 0, peakActiveBytes: 50)
        )) {
            XCTAssertEqual(
                $0 as? KVarNMemoryEvidenceError,
                .invalidMemoryCounters)
        }
        XCTAssertThrowsError(try KVarNMemoryHighWater(
            start: KVarNMemoryCounters(
                activeBytes: 100, cacheBytes: 0, peakActiveBytes: 1),
            end: KVarNMemoryCounters(
                activeBytes: 100, cacheBytes: 0, peakActiveBytes: 200)
        )) {
            XCTAssertEqual(
                $0 as? KVarNMemoryEvidenceError,
                .invalidMemoryCounters)
        }
    }

    func testAllocatorReconciliationRequiresTheExactRoundedArraySum() throws {
        let rounded = try KVarNMemoryReconciliation(
            logicalBytes: 100, expectedAllocatorBytes: 112,
            activeBytes: 115, runtimeBaselineBytes: 4,
            arrayCount: 2, allocatorPageBytes: 16)
        XCTAssertEqual(rounded.activeAboveExpectedAllocatorBytes, 3)
        XCTAssertEqual(rounded.maximumActiveBytes, 116)

        for active in [111, 117] {
            XCTAssertThrowsError(try KVarNMemoryReconciliation(
                logicalBytes: 100, expectedAllocatorBytes: 112,
                activeBytes: active, runtimeBaselineBytes: 4,
                arrayCount: 2, allocatorPageBytes: 16)
            ) {
                XCTAssertEqual(
                    $0 as? KVarNMemoryEvidenceError,
                    .unreconciledAllocatorMemory)
            }
        }
    }

    func testAllocatorByteRuleKeepsSubpageRequestsExactAndRoundsLargerOnes() throws {
        let page = 16_384
        XCTAssertEqual(
            try KVarNMemoryEvidence.allocatorBytes(
                forLogicalBytes: 0, pageBytes: page),
            0)
        XCTAssertEqual(
            try KVarNMemoryEvidence.allocatorBytes(
                forLogicalBytes: page, pageBytes: page),
            page)
        XCTAssertEqual(
            try KVarNMemoryEvidence.allocatorBytes(
                forLogicalBytes: page + 1, pageBytes: page),
            page * 2)
        XCTAssertEqual(
            try KVarNMemoryEvidence.allocatorBytes(
                forLogicalBytes: page * 2, pageBytes: page),
            page * 2)

        for invalid in [
            { try KVarNMemoryEvidence.allocatorBytes(
                forLogicalBytes: -1, pageBytes: page) },
            { try KVarNMemoryEvidence.allocatorBytes(
                forLogicalBytes: 1, pageBytes: 0) },
            { try KVarNMemoryEvidence.allocatorBytes(
                forLogicalBytes: 1, pageBytes: 12_000) },
            { try KVarNMemoryEvidence.allocatorBytes(
                forLogicalBytes: Int.max, pageBytes: page) },
        ] {
            XCTAssertThrowsError(try invalid()) {
                XCTAssertEqual(
                    $0 as? KVarNMemoryEvidenceError,
                    .unreconciledAllocatorMemory)
            }
        }
    }

    func testCacheBoundaryModelSeparatesGuaranteedSingleConcatFromDualReference() throws {
        let configuration = try KVarNMemoryProbeConfiguration(
            phase: .cacheBoundary, heads: 8, headDimension: 128,
            groupSize: 128, iterations: 8, capacity: 4_096,
            cacheLimitBytes: 0, run: 1)
        let structural = try KVarNCacheBoundaryStructuralMemory(
            configuration: configuration,
            allocatorPageBytes: 16_384,
            startActiveBytes: 4_493_316,
            observedPeakActiveBytes: 37_523_464)

        XCTAssertEqual(structural.completedTileCount, 31)
        XCTAssertEqual(structural.materializedOutputArrayBytes, 8_388_608)
        XCTAssertEqual(structural.reconstructedTileArrayBytes, 262_144)
        XCTAssertEqual(structural.minimumConcatIncrementBytes, 16_515_072)
        XCTAssertEqual(structural.minimumStructuralPeakActiveBytes, 21_008_388)
        XCTAssertEqual(structural.dualConcatReferenceIncrementBytes, 33_030_144)
        XCTAssertEqual(structural.dualConcatReferencePeakActiveBytes, 37_523_460)
        XCTAssertEqual(structural.observedPeakAboveMinimumStructuralBytes, 16_515_076)
        XCTAssertEqual(structural.observedPeakDeltaFromDualConcatReferenceBytes, 4)

        let singleBranchOnly = try KVarNCacheBoundaryStructuralMemory(
            configuration: configuration,
            allocatorPageBytes: 16_384,
            startActiveBytes: 4_493_316,
            observedPeakActiveBytes: 21_008_388)
        XCTAssertEqual(
            singleBranchOnly.observedPeakDeltaFromDualConcatReferenceBytes,
            -16_515_072)
        XCTAssertThrowsError(try KVarNCacheBoundaryStructuralMemory(
            configuration: configuration,
            allocatorPageBytes: 16_384,
            startActiveBytes: 4_493_316,
            observedPeakActiveBytes: 21_008_387)
        ) {
            XCTAssertEqual(
                $0 as? KVarNMemoryEvidenceError,
                .invalidStructuralMemoryAccounting)
        }
    }

    func testRetainedAccountingExposesGraphResidencyWithoutCallingItScratch() throws {
        let accounting = try KVarNRetainedMemoryAccounting(
            minimumLogicalBytes: 100, activeBytes: 260, arrayCount: 4)
        XCTAssertEqual(accounting.activeAboveMinimumLogicalBytes, 160)
        XCTAssertThrowsError(try KVarNRetainedMemoryAccounting(
            minimumLogicalBytes: 100, activeBytes: 99, arrayCount: 4)
        ) {
            XCTAssertEqual(
                $0 as? KVarNMemoryEvidenceError,
                .invalidRetainedMemoryAccounting)
        }
    }

    func testCleanHarnessSHAValidationRejectsDirtyAndMalformedStamps() {
        let clean = String(repeating: "a", count: 40)
        XCTAssertTrue(KVarNMemoryEvidence.isCleanHarnessSHA(clean))
        XCTAssertFalse(KVarNMemoryEvidence.isCleanHarnessSHA("\(clean)-dirty"))
        XCTAssertFalse(KVarNMemoryEvidence.isCleanHarnessSHA(String(repeating: "g", count: 40)))
        XCTAssertFalse(KVarNMemoryEvidence.isCleanHarnessSHA("abc"))
    }

    func testCacheStorageDetachmentPreservesEveryStateByteWithNewArrays() throws {
        let cache = KVarNKVCache(
            capacity: 384, tier: .k4v2G128, iterations: 8)
        let keys = MLXArray((0 ..< (383 * 128)).map {
            Float16(sin(Double($0) * 0.013))
        }).reshaped([1, 1, 383, 128])
        let values = keys * Float16(-0.25)
        _ = cache.update(keys: keys, values: values)
        eval(cache.innerState())
        let before = cache.innerState()
        let snapshot = try XCTUnwrap(cache.storageSnapshot())

        try KVarNMemoryEvidence.detachCacheStorage(cache)

        let after = cache.innerState()
        XCTAssertEqual(after.count, before.count)
        XCTAssertTrue(zip(before, after).allSatisfy { $0 !== $1 })
        for (original, detached) in zip(before, after) {
            XCTAssertEqual(detached.shape, original.shape)
            XCTAssertEqual(detached.dtype, original.dtype)
            switch original.dtype {
            case .uint8:
                XCTAssertEqual(
                    detached.asArray(UInt8.self), original.asArray(UInt8.self))
            case .float16:
                XCTAssertEqual(
                    detached.asArray(Float16.self).map(\.bitPattern),
                    original.asArray(Float16.self).map(\.bitPattern))
            case .int32:
                XCTAssertEqual(
                    detached.asArray(Int32.self), original.asArray(Int32.self))
            default:
                XCTFail("unexpected KVarN state dtype \(original.dtype)")
            }
        }
        XCTAssertEqual(cache.storageSnapshot(), snapshot)
        XCTAssertEqual(cache.offset, 383)
    }

    func testProbeFinitenessMeasuresFloatingOutputsAndAcceptsStorageScalars() {
        let finite = MLXArray([Float16(1), Float16(-2)])
        let payload = MLXArray([UInt8(0), UInt8.max])
        let control = MLXArray([Int32(0), Int32(4)])
        XCTAssertTrue(KVarNMemoryEvidence.probeArraysAreFinite([
            finite, payload, control,
        ]))

        let nonFinite = MLXArray([Float16.nan])
        XCTAssertFalse(KVarNMemoryEvidence.probeArraysAreFinite([nonFinite]))
    }
}
