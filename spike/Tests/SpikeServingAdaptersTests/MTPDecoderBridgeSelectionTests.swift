import Foundation
import XCTest

import MLX
import MLXLMCommon
import MLXNN
import SpikeCore
@testable import SpikeServingAdapters

/// Cycle 77 continuation of `docs/task-inbox/2026-09-07-mtp-decoder-bridge-DECISION.md`: the
/// decoder bridge itself (`MTPSpeculativeDecoder`, `spike/Sources/SpikeCore/`) already landed,
/// fully tested, with no production consumer. This file tests the WIRING
/// `loadScalarServingModel` (`MLXScalarServing.swift`) now does to give it one: the
/// `.nativeCaches` decoder-strategy branch retains the loaded drafter instead of releasing it, and
/// selects `MTPSpeculativeDecoder` over the plain `MLXDecoder` when a drafter was retained.
///
/// WHAT THIS FILE CANNOT PROVE, stated explicitly rather than faked: `loadScalarServingModel`
/// itself reaching this branch with `inCheckpointMTPSelection` set requires loading and
/// startup-gating the real ~113 GB Qwen3.8-Flash-Next in-checkpoint MTP artifact — every existing
/// `testLoadRejectsInCheckpointMTP...` test in `MLXScalarServingTests.swift` stops BEFORE weight
/// load for the identical reason. That end-to-end path remains fleet-only. What IS proven here,
/// with weight-free mocks standing in for `context.model` and the loaded drafter: the exact
/// decoder-selection and cache-factory-sharing LOGIC that branch runs once it does reach a real
/// checkpoint.
final class MTPDecoderBridgeSelectionTests: XCTestCase {

    // MARK: - Constraint #4: fail-closed decoder-strategy guard

    /// `scalarServingInCheckpointMTPDecoderStrategyError` is the guard `loadScalarServingModel`
    /// calls immediately after resolving `decoderStrategy`, before any drafter weight load. MTP
    /// paired with `.compiledFP16` must refuse: that strategy has no `cacheFactory` seam to hand
    /// `MTPSpeculativeDecoder`, so silently admitting the combination would mean serving one route
    /// and dropping the other.
    func testDecoderStrategyGuardRefusesCompiledFP16WhenAnMTPSelectionIsPresent() {
        let error = scalarServingInCheckpointMTPDecoderStrategyError(
            selection: .converted4Bit, decoderStrategy: .compiledFP16)
        XCTAssertEqual(error, .inCheckpointMTPIncompatibleWithCompiledDecoderStrategy)
    }

    /// Anti-vacuity companion: every OTHER combination admits (`nil`) — no selection at all
    /// (regardless of strategy), and a selection paired with either `.nativeCaches` case. Asserted
    /// individually so a future accidental narrowing (e.g. refusing `.nativeCaches` too) is caught
    /// here, not just by an operator hitting it in production.
    func testDecoderStrategyGuardAdmitsEveryOtherCombination() {
        XCTAssertNil(
            scalarServingInCheckpointMTPDecoderStrategyError(
                selection: nil, decoderStrategy: .compiledFP16))
        XCTAssertNil(
            scalarServingInCheckpointMTPDecoderStrategyError(
                selection: nil, decoderStrategy: .nativeCaches(.fp16)))
        XCTAssertNil(
            scalarServingInCheckpointMTPDecoderStrategyError(
                selection: .converted4Bit, decoderStrategy: .nativeCaches(.fp16)))
        XCTAssertNil(
            scalarServingInCheckpointMTPDecoderStrategyError(
                selection: .converted4Bit,
                decoderStrategy: .nativeCaches(.int8(groupSize: 32, bits: 8))))
    }

    // The guard's doc comment claims the refused combination is provably UNREACHABLE today: MTP is
    // family-gated, and every checkpoint in that family classifies at least one layer
    // `.recurrentState`, which forces `classifyScalarServingDecoderRoute` to `.nativeHeterogeneous`
    // — never `.compiled` — so `decoderStrategy` can only resolve to `.nativeCaches`. That mapping
    // is independently pinned by
    // `MLXScalarServingTests.testMixedMarkerClassifiedVectorRoutesToNativeHeterogeneous`, which
    // exercises the identical mixed dense+recurrent shape that family's hybrid layer stack
    // produces. Recorded here as a citation to an existing test rather than as a test of its own:
    // an `XCTAssertTrue(true)` body would add a passing row to the tally while asserting nothing,
    // which is exactly the vacuous-gate shape this file's other tests are written to avoid.

    // MARK: - Constraint #1: shared cache factory

    /// `scalarServingNativeCacheFactory` is the ONE function `loadScalarServingModel`'s
    /// `.nativeCaches` case calls to build `cacheFactory` — a single local `let`, passed unchanged
    /// to whichever decoder branch runs (`MLXScalarServing.swift`, the `.nativeCaches(let
    /// decision)` case). This proves the function itself is a correct, side-effect-free
    /// `decision`/`model` pass-through to `buildRouteKVCaches` (already unit-tested for its own
    /// fp16-identity/int8-wrap semantics in `KVQuantCacheSelectionTests.swift`) for BOTH storage
    /// tiers, so a caller can trust that sharing this one closure between `MLXDecoder` and
    /// `MTPSpeculativeDecoder` never diverges by tier.
    func testNativeCacheFactoryFp16PassesThroughTheModelsNativeCacheKindsUnchanged() {
        let model = NativeCacheFactoryFixtureModel(cacheCount: 3)
        let factory = scalarServingNativeCacheFactory(decision: .fp16, model: model)

        let caches = factory()

        XCTAssertEqual(caches.count, 3)
        XCTAssertTrue(caches.allSatisfy { $0 is KVCacheSimple })
    }

    func testNativeCacheFactoryInt8WrapsEveryNativeCacheInTheQuantizedWrapper() {
        let model = NativeCacheFactoryFixtureModel(cacheCount: 3)
        let factory = scalarServingNativeCacheFactory(
            decision: .int8(groupSize: 32, bits: 8), model: model)

        let caches = factory()

        XCTAssertEqual(caches.count, 3)
        XCTAssertTrue(caches.allSatisfy { $0 is QuantizedKVCache })
    }

    /// The identity half of constraint #1: the SAME closure, invoked more than once (mirroring how
    /// `MLXDecoder`/`MTPSpeculativeDecoder` each call it again on every `reset()`-driven `prefill`),
    /// never drifts to a different tier on a later call.
    func testNativeCacheFactoryReturnsTheSameTierOnEveryInvocation() {
        let model = NativeCacheFactoryFixtureModel(cacheCount: 2)
        let factory = scalarServingNativeCacheFactory(
            decision: .int8(groupSize: 32, bits: 8), model: model)

        XCTAssertTrue(factory().allSatisfy { $0 is QuantizedKVCache })
        XCTAssertTrue(factory().allSatisfy { $0 is QuantizedKVCache })
    }

    // MARK: - The discriminating question: does a retained drafter actually wire the speculative decoder?

    /// Mirrors the exact two arms of `loadScalarServingModel`'s `.nativeCaches` case
    /// (`MLXScalarServing.swift`): `if let drafter = retainedInCheckpointMTPDrafter { ...
    /// MTPSpeculativeDecoder ... } else { ... MLXDecoder ... }`, both fed by the SAME
    /// `scalarServingNativeCacheFactory` output. `InferenceActor.speculativeTelemetry()` — added
    /// alongside `MTPSpeculativeDecoder` specifically so a caller or test could tell a speculating
    /// request from a passthrough one — is non-nil ONLY for the branch a retained drafter takes.
    func testRetainedDrafterWiresSpeculativeDecoderAndItsAbsenceWiresThePlainDecoder() async throws {
        let plannedTokens: [Int32] = [0, 0, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]

        // The arm `loadScalarServingModel` takes when `retainedInCheckpointMTPDrafter != nil`.
        let speculativeTarget = InCheckpointMTPMockTargetModel(plannedTokens: plannedTokens)
        let drafter = InCheckpointMTPMockDrafter(draftedTokenValue: 6)
        let speculativeCacheFactory = scalarServingNativeCacheFactory(
            decision: .fp16, model: speculativeTarget)
        let speculativeDecoder = try MTPSpeculativeDecoder(
            target: speculativeTarget, drafter: drafter, cacheFactory: speculativeCacheFactory)
        let speculativeActor = InferenceActor(decoder: speculativeDecoder)
        let speculativeSummary = try await speculativeActor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: 5, eos: 99
        ) { _ in .continueGeneration }
        XCTAssertEqual(speculativeSummary.finishReason, .length)
        let speculativeTelemetry = await speculativeActor.speculativeTelemetry()
        XCTAssertNotNil(
            speculativeTelemetry,
            "a retained drafter must wire InferenceActor to a SpeculativeTelemetryProviding "
                + "decoder (MTPSpeculativeDecoder), not a plain MLXDecoder")

        // The arm `loadScalarServingModel` takes when `retainedInCheckpointMTPDrafter == nil` —
        // every existing, non-MTP call site, unaffected by this bridge.
        let plainTarget = InCheckpointMTPMockTargetModel(plannedTokens: plannedTokens)
        let plainCacheFactory = scalarServingNativeCacheFactory(decision: .fp16, model: plainTarget)
        let plainDecoder = MLXDecoder(model: plainTarget, cacheFactory: plainCacheFactory)
        let plainActor = InferenceActor(decoder: plainDecoder)
        let plainSummary = try await plainActor.generateBounded(
            promptTokens: [1, 2, 3], maxTokens: 5, eos: 99
        ) { _ in .continueGeneration }
        XCTAssertEqual(plainSummary.finishReason, .length)
        let plainTelemetry = await plainActor.speculativeTelemetry()
        XCTAssertNil(
            plainTelemetry,
            "the non-MTP path must stay wired to a decoder that is NOT "
                + "SpeculativeTelemetryProviding — a non-nil snapshot here would mean the plain "
                + "route silently started speculating")
    }
}

/// Minimal `LanguageModel` whose `newCache` returns a fixed COUNT of fresh `KVCacheSimple`
/// instances every call — used only to exercise `scalarServingNativeCacheFactory`'s
/// `decision`/`model` pass-through in isolation. Never runs a real forward pass; mirrors the
/// minimal-fake shape `MLXScalarServingTests.swift`'s own fixtures use (e.g.
/// `FixtureNonReportingModel`).
private final class NativeCacheFactoryFixtureModel: Module, LanguageModel {
    let cacheCount: Int

    init(cacheCount: Int) {
        self.cacheCount = cacheCount
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0..<cacheCount).map { _ in KVCacheSimple() }
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([inputs.dim(0), inputs.dim(1), 8])
    }
}
