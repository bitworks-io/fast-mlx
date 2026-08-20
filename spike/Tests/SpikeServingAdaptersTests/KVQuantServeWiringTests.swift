import Foundation
import HarnessCore
import MLXLMCommon
import XCTest

@testable import SpikeServingAdapters

/// Wiring guards for `--kv-quant` → `ScalarServingModelLoadConfiguration.kvQuantTier` →
/// `selectKVCacheQuant` at load. The end-to-end fail-closed behavior (a non-runtime-wired tier refuses
/// to START, rather than silently serving fp16) is proven by a live small-model CLI smoke recorded in
/// the cycle evidence; these unit guards pin the invariants that make that wiring correct and durable.
final class KVQuantServeWiringTests: XCTestCase {

    private func makeConfig(
        kvQuantTier: KVQuantTier? = nil
    ) -> ScalarServingModelLoadConfiguration {
        let backend = ScalarServingBackendConfiguration(
            defaultMaximumCompletionTokens: 512,
            maximumQueuedRequests: 2,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 8, maxBytes: 32 * 1_024))
        if let kvQuantTier {
            return ScalarServingModelLoadConfiguration(
                launchedModel: "m", modelDirectory: URL(fileURLWithPath: "/tmp/m"),
                memoryLimitBytes: 1_024, cacheLimitBytes: 512,
                backendConfiguration: backend, kvQuantTier: kvQuantTier)
        }
        return ScalarServingModelLoadConfiguration(
            launchedModel: "m", modelDirectory: URL(fileURLWithPath: "/tmp/m"),
            memoryLimitBytes: 1_024, cacheLimitBytes: 512,
            backendConfiguration: backend)
    }

    /// The load config must default to fp16 when the flag is omitted. This is the no-silent-downgrade
    /// invariant: a serve that never asked for quantized KV must build the runtime's always-valid native
    /// storage, never an unproven tier by accident.
    func testConfigDefaultsToFP16WhenTierOmitted() {
        XCTAssertEqual(makeConfig().kvQuantTier, .fp16)
    }

    /// An explicitly requested tier is preserved on the config, so it actually reaches `selectKVCacheQuant`
    /// at load instead of being dropped on the floor (the pre-wiring bug: parsed, then ignored).
    func testConfigPreservesRequestedTier() {
        XCTAssertEqual(makeConfig(kvQuantTier: .fp16).kvQuantTier, .fp16)
        XCTAssertEqual(makeConfig(kvQuantTier: .int8).kvQuantTier, .int8)
    }

    /// fp16 (the wired tier) resolves to the byte-identical native decision for the dense route the load
    /// path classifies — i.e. the default serve is unchanged by the new selection call.
    func testFP16TierResolvesToNativeDecisionForDenseRoute() throws {
        let decision = try selectKVCacheQuant(
            requested: makeConfig().kvQuantTier, nativeKinds: [.denseAttention])
        XCTAssertEqual(decision, .fp16)
    }

    /// int8 is fail-closed at the load-path selection today (runtimeWiredKVTiers == [.fp16]) — the process
    /// throws before any cache is built, so `--kv-quant int8` refuses to start rather than silently
    /// serving fp16. Pins the safety behavior the wiring exists to deliver.
    func testInt8FailsClosedAtSelectionForDenseRoute() {
        XCTAssertThrowsError(
            try selectKVCacheQuant(
                requested: makeConfig(kvQuantTier: .int8).kvQuantTier, nativeKinds: [.denseAttention])
        ) { error in
            XCTAssertEqual(error as? KVQuantSelectionError, .tierNotRuntimeWired(.int8))
        }
    }

    /// The defensive construction-unavailable load error is Equatable and carries the offending tier —
    /// the guard that stops a future runtimeWired flip from silently reaching fp16 construction.
    func testConstructionUnavailableErrorCarriesTier() {
        XCTAssertEqual(
            ScalarServingModelLoadError.kvQuantTierConstructionUnavailable(.int8),
            ScalarServingModelLoadError.kvQuantTierConstructionUnavailable(.int8))
        XCTAssertNotEqual(
            ScalarServingModelLoadError.kvQuantTierConstructionUnavailable(.int8),
            ScalarServingModelLoadError.kvQuantTierConstructionUnavailable(.fp16))
    }

    // MARK: - Continuous-batch route (the DEFAULT serve.sh path)

    private func makeContinuousConfig(
        kvQuantTier: KVQuantTier? = nil
    ) throws -> ContinuousServingModelLoadConfiguration {
        let coordinator = try ContinuousBatchConfiguration(
            maxActiveSlots: 4, maxPrefillSlots: 2, prefillChunkSize: 512, maxQueuedRequests: 8)
        let backend = ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 512,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 8, maxBytes: 32 * 1_024))
        if let kvQuantTier {
            return ContinuousServingModelLoadConfiguration(
                launchedModel: "m", modelDirectory: URL(fileURLWithPath: "/tmp/m"),
                memoryLimitBytes: 1_024, cacheLimitBytes: 512, maxReservedKVBytes: 256,
                coordinatorConfiguration: coordinator, publicationCapacity: 1,
                backendConfiguration: backend, kvQuantTier: kvQuantTier)
        }
        return ContinuousServingModelLoadConfiguration(
            launchedModel: "m", modelDirectory: URL(fileURLWithPath: "/tmp/m"),
            memoryLimitBytes: 1_024, cacheLimitBytes: 512, maxReservedKVBytes: 256,
            coordinatorConfiguration: coordinator, publicationCapacity: 1,
            backendConfiguration: backend)
    }

    /// The continuous-batch load config — the route the default `serve.sh` (which adds
    /// `--continuous-batch-no-spec`) actually takes — must also default to fp16, so its int8 request is
    /// fail-closed rather than silently served as fp16.
    func testContinuousConfigDefaultsToFP16WhenTierOmitted() throws {
        XCTAssertEqual(try makeContinuousConfig().kvQuantTier, .fp16)
    }

    /// An explicit tier reaches `selectKVCacheQuant` on the continuous route too.
    func testContinuousConfigPreservesRequestedTier() throws {
        XCTAssertEqual(try makeContinuousConfig(kvQuantTier: .int8).kvQuantTier, .int8)
    }

    /// The continuous route's defensive construction-unavailable error carries the offending tier.
    func testContinuousConstructionUnavailableErrorCarriesTier() {
        XCTAssertEqual(
            ContinuousServingModelLoadError.kvQuantTierConstructionUnavailable(.int8),
            ContinuousServingModelLoadError.kvQuantTierConstructionUnavailable(.int8))
        XCTAssertNotEqual(
            ContinuousServingModelLoadError.kvQuantTierConstructionUnavailable(.int8),
            ContinuousServingModelLoadError.kvQuantTierConstructionUnavailable(.fp16))
    }
}
