import XCTest

import HarnessCore
import ServingCore
@testable import SpikeServingAdapters

/// Verifies `InCheckpointMTPFitComposition`'s `nAttnLayers + 1` correction against
/// independently-computed byte quantities (not verdict colors — at the deploy operating point both
/// MTP-off and MTP-on fit checks are GREEN, so asserting only `verdict.color` would pass identically
/// with this composition deleted), asserts every other profile field survives unchanged, and pins the
/// negative (flag-absent) case. See the type's own doc comment for the modeled shape and why weights
/// need no adjustment.
final class InCheckpointMTPFitCompositionTests: XCTestCase {

    /// The real "Qwen3.8-Flash-Next" catalog entry's geometry (`ModelArchProfile.swift`), reproduced
    /// here rather than imported so this test is pinned to values, not to the catalog entry's
    /// continued existence at that identifier.
    private func flashNextProfile(weightsBytes: Int = 113_324_747_928) -> ModelArchProfile {
        ModelArchProfile(
            id: "Qwen3.8-Flash-Next",
            modelType: .hybridLinear,
            nLayers: 48,
            nAttnLayers: 12,
            nKVHeads: 2,
            headDim: 256,
            fixedStateBytes: 115_458_048,
            nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: weightsBytes,
            license: "fixture-license",
            auxPerLayerKeyDim: 128)
    }

    // MARK: - (e) Negative case: base profile alone, un-composed

    func testBaseProfileAloneHasTwelveAttnLayersAndTwentySevenThousandSixHundredFortyEightBytesPerToken()
    {
        let base = flashNextProfile()
        XCTAssertEqual(base.nAttnLayers, 12)
        XCTAssertEqual(CapacityModel.kvBytesPerToken(base, kvQuant: .fp16), 27_648)
    }

    // MARK: - (a)+(b) Byte quantity AND decomposition, independently computed

    /// Asserts the TOTAL per-token byte figure the task spec names (29,952, delta 2,304) AND the two
    /// sub-terms (`K+V` and the QSA-indexer `auxPerLayerKeyDim` term) separately. A composition tested
    /// against a profile with `auxPerLayerKeyDim == nil` cannot distinguish "+1 layer of K+V" (2,048
    /// B/tok) from the true "+1 layer of K+V + indexer" (2,304 B/tok) — this fixture carries the real
    /// `auxPerLayerKeyDim: 128`, so both terms are live and separately checkable.
    func testComposedKVBytesPerTokenAndBothSubTermsMatchIndependentComputation() {
        let base = flashNextProfile()
        let composed = InCheckpointMTPFitComposition.make(base: base)

        XCTAssertEqual(composed.nAttnLayers, 13)

        let basePerToken = CapacityModel.kvBytesPerToken(base, kvQuant: .fp16)
        let composedPerToken = CapacityModel.kvBytesPerToken(composed, kvQuant: .fp16)
        XCTAssertEqual(basePerToken, 27_648)
        XCTAssertEqual(composedPerToken, 29_952)
        XCTAssertEqual(composedPerToken - basePerToken, 2_304)

        // Sub-term 1: the standard K+V term (`nAttnLayers * nKVHeads * kvHeadDimSum * bpe`),
        // independently recomputed from the fixture's own construction parameters — not by re-calling
        // `kvBytesPerToken` a second time, which would only prove the total again.
        let kvHeadDimSum = Double(base.headDim + base.headDim)  // symmetric K/V: 256 + 256 = 512
        let baseKVTerm = Double(base.nAttnLayers) * Double(base.nKVHeads) * kvHeadDimSum * 2.0
        let composedKVTerm = Double(composed.nAttnLayers) * Double(composed.nKVHeads) * kvHeadDimSum * 2.0
        XCTAssertEqual(baseKVTerm, 24_576)
        XCTAssertEqual(composedKVTerm, 26_624)
        XCTAssertEqual(composedKVTerm - baseKVTerm, 2_048)

        // Sub-term 2: the QSA-indexer `auxPerLayerKeyDim` term, fixed at 2 bytes/element regardless of
        // KV-quant tier (see `CapacityModel.kvBytesPerToken`'s own comment on the literal `2.0`).
        let baseIndexerTerm = Double(base.nAttnLayers) * Double(base.auxPerLayerKeyDim ?? 0) * 2.0
        let composedIndexerTerm =
            Double(composed.nAttnLayers) * Double(composed.auxPerLayerKeyDim ?? 0) * 2.0
        XCTAssertEqual(baseIndexerTerm, 3_072)
        XCTAssertEqual(composedIndexerTerm, 3_328)
        XCTAssertEqual(composedIndexerTerm - baseIndexerTerm, 256)

        // The two sub-term deltas must sum to the total delta — proves the decomposition is complete,
        // not just individually plausible.
        XCTAssertEqual(
            (composedKVTerm - baseKVTerm) + (composedIndexerTerm - baseIndexerTerm),
            composedPerToken - basePerToken)
    }

    // MARK: - (c) Every other field preserved; composed is strictly more expensive

    func testEveryOtherProfileFieldIsPreservedIndividuallyAndComposedIsStrictlyMoreExpensive() {
        let base = flashNextProfile(weightsBytes: 113_324_747_928)
        let composed = InCheckpointMTPFitComposition.make(base: base)

        XCTAssertEqual(composed.id, "\(base.id)+qwen4exp-mtp-composition")
        XCTAssertEqual(composed.nAttnLayers, base.nAttnLayers + 1)

        // Every OTHER field, asserted individually — a field-by-field rebuild that silently drops one
        // (as a prior composition in this codebase once dropped `auxPerLayerKeyDim`) would make the
        // MTP profile CHEAPER per token while these individual equality checks still pass.
        XCTAssertEqual(composed.modelType, base.modelType)
        XCTAssertEqual(composed.nLayers, base.nLayers)
        XCTAssertEqual(composed.nKVHeads, base.nKVHeads)
        XCTAssertEqual(composed.headDim, base.headDim)
        XCTAssertEqual(composed.slidingWindow, base.slidingWindow)
        XCTAssertEqual(composed.fixedStateBytes, base.fixedStateBytes)
        XCTAssertEqual(composed.nativeMaxContext, base.nativeMaxContext)
        XCTAssertEqual(composed.weightsBytes4bitEstimate, base.weightsBytes4bitEstimate)
        XCTAssertEqual(composed.license, base.license)
        XCTAssertEqual(composed.mlaHeads, base.mlaHeads)
        XCTAssertEqual(composed.mlaRopeDim, base.mlaRopeDim)
        XCTAssertEqual(composed.mlaNopeDim, base.mlaNopeDim)
        XCTAssertEqual(composed.mlaVDim, base.mlaVDim)
        XCTAssertEqual(composed.swaKVHeads, base.swaKVHeads)
        XCTAssertEqual(composed.swaHeadDim, base.swaHeadDim)
        XCTAssertEqual(composed.vHeadDim, base.vHeadDim)
        XCTAssertEqual(composed.swaVHeadDim, base.swaVHeadDim)
        XCTAssertEqual(composed.auxPerLayerKeyDim, base.auxPerLayerKeyDim)

        // Not weaker than the total-figure check above, but a distinct assertion aimed directly at
        // the concern named in the task spec: the composed profile must be strictly MORE expensive
        // per token, never equal or cheaper.
        let basePerToken = CapacityModel.kvBytesPerToken(base, kvQuant: .fp16)
        let composedPerToken = CapacityModel.kvBytesPerToken(composed, kvQuant: .fp16)
        XCTAssertGreaterThan(composedPerToken, basePerToken)
    }

    // MARK: - Field-preservation test with EVERY optional field populated (non-Flash-Next shape)

    /// Same field-preservation assertion as above, but against a fixture with every MLA/SWA field
    /// populated (unlike the real Flash Next entry, which leaves them `nil`) — so a field-drop
    /// specifically among the "always nil for this family" fields cannot hide by coincidence.
    func testEveryFieldIsPreservedWhenEveryOptionalFieldIsNonNil() {
        let base = ModelArchProfile(
            id: "fixture-flash-next",
            modelType: .hybridLinear,
            nLayers: 48,
            nAttnLayers: 12,
            nKVHeads: 2,
            headDim: 256,
            slidingWindow: 4096,
            fixedStateBytes: 115_458_048,
            nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: 113_324_747_928,
            license: "fixture-license",
            mlaHeads: 7,
            mlaRopeDim: 11,
            mlaNopeDim: 13,
            mlaVDim: 17,
            swaKVHeads: 19,
            swaHeadDim: 23,
            vHeadDim: 29,
            swaVHeadDim: 31,
            auxPerLayerKeyDim: 128)
        let composed = InCheckpointMTPFitComposition.make(base: base)

        XCTAssertEqual(composed.modelType, base.modelType)
        XCTAssertEqual(composed.nLayers, base.nLayers)
        XCTAssertEqual(composed.nAttnLayers, base.nAttnLayers + 1)
        XCTAssertEqual(composed.nKVHeads, base.nKVHeads)
        XCTAssertEqual(composed.headDim, base.headDim)
        XCTAssertEqual(composed.slidingWindow, base.slidingWindow)
        XCTAssertEqual(composed.fixedStateBytes, base.fixedStateBytes)
        XCTAssertEqual(composed.nativeMaxContext, base.nativeMaxContext)
        XCTAssertEqual(composed.weightsBytes4bitEstimate, base.weightsBytes4bitEstimate)
        XCTAssertEqual(composed.license, base.license)
        XCTAssertEqual(composed.mlaHeads, base.mlaHeads)
        XCTAssertEqual(composed.mlaRopeDim, base.mlaRopeDim)
        XCTAssertEqual(composed.mlaNopeDim, base.mlaNopeDim)
        XCTAssertEqual(composed.mlaVDim, base.mlaVDim)
        XCTAssertEqual(composed.swaKVHeads, base.swaKVHeads)
        XCTAssertEqual(composed.swaHeadDim, base.swaHeadDim)
        XCTAssertEqual(composed.vHeadDim, base.vHeadDim)
        XCTAssertEqual(composed.swaVHeadDim, base.swaVHeadDim)
        XCTAssertEqual(composed.auxPerLayerKeyDim, base.auxPerLayerKeyDim)
        XCTAssertEqual(composed.id, "\(base.id)+qwen4exp-mtp-composition")
    }

    // MARK: - (f) Sentinel refusal: the fail-open this composition must not reintroduce

    /// A `nAttnLayers: 0` profile is `ModelArchProfile`'s deliberate sentinel for "growing-attention-
    /// layer count unconfirmed — do not multiply blind" (`ModelArchProfile.isKVDerivable`'s doc
    /// comment). `make` must refuse to touch it at all: incrementing 0 -> 1 would convert an honest
    /// "not derivable" refusal into a fabricated, computed fit color. Asserts `nAttnLayers`, `id`
    /// (NOT relabeled — relabeling would claim a composition happened when none did), and
    /// `isKVDerivable` are unchanged by the call.
    func testSentinelProfileWithZeroAttnLayersIsReturnedCompletelyUnchanged() {
        let sentinel = ModelArchProfile(
            id: "sentinel-unconfirmed-attn-layers",
            modelType: .hybridMamba2MoE,
            nLayers: 48,
            nAttnLayers: 0,
            nKVHeads: 2,
            headDim: 256,
            nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: 10_000_000_000,
            license: "fixture-license")

        XCTAssertEqual(sentinel.nAttnLayers, 0)
        XCTAssertFalse(sentinel.isKVDerivable)

        let result = InCheckpointMTPFitComposition.make(base: sentinel)

        XCTAssertEqual(result.nAttnLayers, 0)
        XCTAssertEqual(result.id, sentinel.id)
        XCTAssertFalse(result.isKVDerivable)
    }

    /// The consequence that makes the sentinel guard meaningful rather than cosmetic: a
    /// `CapacityPrediction` built for the sentinel profile must still classify to RED
    /// `.kvNotDerivable` AFTER passing through `make` — proving the guard actually prevents a
    /// fabricated fit color downstream, not merely that a raw field happens to read `0`.
    func testSentinelProfileStillClassifiesAsKVNotDerivableAfterComposition() {
        let sentinel = ModelArchProfile(
            id: "sentinel-unconfirmed-attn-layers",
            modelType: .hybridMamba2MoE,
            nLayers: 48,
            nAttnLayers: 0,
            nKVHeads: 2,
            headDim: 256,
            nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: 10_000_000_000,
            license: "fixture-license")

        let composed = InCheckpointMTPFitComposition.make(base: sentinel)

        let prediction = CapacityModel.predictPeakBytes(
            model: composed, context: 4096, concurrency: 1, kvQuant: .fp16,
            profile: .m5Max128)
        let verdict = CapacityModel.classify(
            prediction, profile: .m5Max128, weightsBytes: Double(composed.weightsBytes4bitEstimate))

        XCTAssertEqual(verdict.color, .red)
        XCTAssertEqual(verdict.bindingConstraint, .kvNotDerivable)
    }

    /// The guard must be DISCRIMINATING, not a blanket "small values are suspicious" check: a
    /// non-sentinel profile with `nAttnLayers == 1` (the smallest legitimate confirmed count) IS
    /// composed to 2, exactly like every other non-sentinel value. This is what proves the guard
    /// fires only on the `== 0` sentinel, not on every small `nAttnLayers`.
    func testProfileWithOneAttnLayerIsComposedToTwoNotTreatedAsSentinel() {
        let base = flashNextProfile()
        let oneLayer = ModelArchProfile(
            id: base.id,
            modelType: base.modelType,
            nLayers: base.nLayers,
            nAttnLayers: 1,
            nKVHeads: base.nKVHeads,
            headDim: base.headDim,
            fixedStateBytes: base.fixedStateBytes,
            nativeMaxContext: base.nativeMaxContext,
            weightsBytes4bitEstimate: base.weightsBytes4bitEstimate,
            license: base.license,
            auxPerLayerKeyDim: base.auxPerLayerKeyDim)

        let composed = InCheckpointMTPFitComposition.make(base: oneLayer)

        XCTAssertEqual(oneLayer.nAttnLayers, 1)
        XCTAssertEqual(composed.nAttnLayers, 2)
        XCTAssertEqual(composed.id, "\(oneLayer.id)+qwen4exp-mtp-composition")
    }

    // MARK: - (d) The wire: argument→fit seam

    /// `resolveServingLimits` (private, `fastmlx-serve`'s `FastMLXServe.swift`) has no test target of
    /// its own — `fastmlx-serve` is an executable target with no corresponding `.testTarget` in
    /// `Package.swift` (mirroring the `ExactQwen35MTPCompositeFitProfile` precedent, whose own tests
    /// likewise stop at the composition function and never reach through `prepareBackend`/
    /// `resolveServingLimits`). This test gets as close to the real wire as that precedent does: it
    /// parses REAL `FastMLXServeArguments` (the identical argv shape
    /// `FastMLXServeArgumentsTests.testInCheckpointMTPWithNgramOffloadPlanParsesAsConverted4Bit`
    /// exercises) and then reproduces the EXACT gate FastMLXServe.swift's `resolveServingLimits` runs
    /// (`if arguments.inCheckpointMTPSelection != nil { … InCheckpointMTPFitComposition.make
    /// … }`), proving that a real parsed `--qwen4exp-mtp` invocation drives the composed profile to
    /// `nAttnLayers == 13`. What this does NOT prove: that `resolveServingLimits` itself contains that
    /// exact conditional at runtime — only a direct unit test against that private function could
    /// prove that, and no test target can reach it. Confirmed instead by direct source inspection
    /// (`FastMLXServe.swift`, the `if arguments.inCheckpointMTPSelection != nil` block immediately
    /// preceding the `--ngram-offload-plan` composition block in `resolveServingLimits`).
    func testRealParsedArgumentsWithInCheckpointMTPDriveTheComposedProfileToThirteenAttnLayers() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
        ])
        XCTAssertEqual(arguments.inCheckpointMTPSelection, .converted4Bit)

        let decoded = flashNextProfile()
        // Reproduces `resolveServingLimits`'s own gate expression verbatim.
        var parsedProfile = decoded
        if arguments.inCheckpointMTPSelection != nil {
            parsedProfile = InCheckpointMTPFitComposition.make(base: decoded)
        }

        XCTAssertEqual(parsedProfile.nAttnLayers, 13)
        XCTAssertEqual(CapacityModel.kvBytesPerToken(parsedProfile, kvQuant: .fp16), 29_952)
    }

    /// (e) restated at the argument-parsing boundary: without `--qwen4exp-mtp`, the same gate leaves
    /// the profile untouched at 12 attention layers / 27,648 B/tok.
    func testRealParsedArgumentsWithoutInCheckpointMTPLeaveProfileAtTwelveAttnLayers() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])
        XCTAssertNil(arguments.inCheckpointMTPSelection)

        let decoded = flashNextProfile()
        var parsedProfile = decoded
        if arguments.inCheckpointMTPSelection != nil {
            parsedProfile = InCheckpointMTPFitComposition.make(base: decoded)
        }

        XCTAssertEqual(parsedProfile.nAttnLayers, 12)
        XCTAssertEqual(CapacityModel.kvBytesPerToken(parsedProfile, kvQuant: .fp16), 27_648)
    }
}
