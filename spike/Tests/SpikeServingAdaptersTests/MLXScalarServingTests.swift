import Foundation
import XCTest

import MLX
import MLXLMCommon
import MLXNN
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class MLXScalarServingTests: XCTestCase {
    func testCodecRendersExactOpenAIRolesAndContentThroughChatTemplate() throws {
        let codec = MLXScalarTextCodec(tokenizer: FixtureTokenizer())

        let tokens = try codec.render(
            messages: [
                OpenAIChatMessage(role: .developer, text: "developer text"),
                OpenAIChatMessage(role: .system, text: "system text"),
                OpenAIChatMessage(role: .user, text: "user text"),
                OpenAIChatMessage(role: .assistant, text: "assistant text"),
            ],
            tools: [],
            enableThinking: nil,
            reasoningEffort: nil)

        XCTAssertEqual(tokens, [41, 42])
    }

    func testCodecDetokenizerPublishesExactIncrementalSuffixes() {
        let codec = MLXScalarTextCodec(tokenizer: FixtureTokenizer())
        var detokenizer = codec.makeDetokenizer()

        detokenizer.append(token: 1)
        XCTAssertEqual(detokenizer.next(), "hel")
        detokenizer.append(token: 2)
        XCTAssertEqual(detokenizer.next(), "lo")
        detokenizer.append(token: 3)
        XCTAssertEqual(detokenizer.next(), "\n")
    }

    func testStopTokenResolutionUnionsConfigurationTokenizerExtrasAndUnknown() throws {
        let configuration = ModelConfiguration(
            directory: URL(fileURLWithPath: "/tmp/fixture-model"),
            extraEOSTokens: ["<turn>"],
            eosTokenIds: [7])

        let ids = try resolveScalarServingStopTokenIDs(
            configuration: configuration,
            tokenizer: FixtureTokenizer())

        XCTAssertEqual(ids, [7, 8, 9, 10])
    }

    func testNativeCacheClassifierSeparatesDenseRotatingRecurrentAndComposite() {
        let kinds = classifyScalarServingNativeCaches([
            KVCacheSimple(),
            RotatingKVCache(maxSize: 128, keep: 4),
            MambaCache(),
            CacheList(KVCacheSimple(), MambaCache()),
        ])

        XCTAssertEqual(
            kinds,
            [
                .denseAttention,
                .rotatingAttention,
                .recurrentState,
                .composite,
            ])
    }

    /// Regression guard: the marker-protocol probe must not change classification of any of the
    /// five concrete types the classifier has always recognized, and each must report source
    /// `.concreteType`.
    ///
    /// Note that `KVCacheSimple` DOES conform to `ServingCacheKindReporting` inside this test
    /// target (see the retroactive conformance at the bottom of this file), and deliberately
    /// reports a kind that disagrees with its concrete-type classification. So this test is not
    /// merely asserting the status quo — for that one case it is also asserting that the
    /// concrete-type switch wins over the marker probe.
    func testNativeCacheClassifierEntryStillMatchesFiveConcreteTypesBySourceConcreteType() {
        let denseAttention = classifyScalarServingNativeCacheEntry(KVCacheSimple())
        let rotatingAttention = classifyScalarServingNativeCacheEntry(
            RotatingKVCache(maxSize: 128, keep: 4))
        let mambaRecurrentState = classifyScalarServingNativeCacheEntry(MambaCache())
        let arraysRecurrentState = classifyScalarServingNativeCacheEntry(ArraysCache(size: 1))
        let composite = classifyScalarServingNativeCacheEntry(
            CacheList(KVCacheSimple(), MambaCache()))

        XCTAssertTrue(denseAttention == (.denseAttention, .concreteType))
        XCTAssertTrue(rotatingAttention == (.rotatingAttention, .concreteType))
        XCTAssertTrue(mambaRecurrentState == (.recurrentState, .concreteType))
        XCTAssertTrue(arraysRecurrentState == (.recurrentState, .concreteType))
        XCTAssertTrue(composite == (.composite, .concreteType))
    }

    /// Every `ServingCacheLayerKind` case must map to the matching `ScalarServingNativeCacheKind`
    /// with source `.markerProtocol`, driven off `.allCases` so a future 5th case fails this test
    /// (rather than silently compiling to `.unknown`) until the classifier's exhaustive switch is
    /// updated to handle it.
    func testNativeCacheClassifierEntryMapsEveryMarkerLayerKindExhaustively() {
        let expectedKindByLayerKind: [ServingCacheLayerKind: ScalarServingNativeCacheKind] = [
            .denseAttention: .denseAttention,
            .rotatingAttention: .rotatingAttention,
            .recurrentState: .recurrentState,
            .composite: .composite,
        ]

        for layerKind in ServingCacheLayerKind.allCases {
            let (kind, source) = classifyScalarServingNativeCacheEntry(
                MarkerReportingFakeCache(servingCacheLayerKind: layerKind))

            XCTAssertEqual(
                kind, expectedKindByLayerKind[layerKind],
                "unexpected kind for marker layer kind \(layerKind)")
            XCTAssertEqual(
                source, .markerProtocol,
                "expected .markerProtocol source for marker layer kind \(layerKind)")
        }
    }

    /// A cache that is neither one of the five concrete types nor `ServingCacheKindReporting`
    /// still yields `(.unknown, .unclassified)` — the classifier's fully-unrecognized fallback.
    func testNativeCacheClassifierEntryReturnsUnknownUnclassifiedForNeitherConcreteNorMarker() {
        let (kind, source) = classifyScalarServingNativeCacheEntry(NonReportingFakeCache())

        XCTAssertEqual(kind, .unknown)
        XCTAssertEqual(source, .unclassified)
    }

    /// Order regression: a cache that is BOTH a concrete type AND `ServingCacheKindReporting`
    /// (with a DIFFERENT reported kind) must still classify via the concrete-type path, source
    /// `.concreteType`. None of the five real concrete types conform to the marker protocol today,
    /// so this synthetic fixture is the only way to actually exercise the "concrete type wins"
    /// ordering guarantee documented on `classifyScalarServingNativeCacheEntry` — without it, an
    /// accidental reordering of the probe and the switch would compile and pass every other test.
    func testConcreteTypeConformingToMarkerProtocolStillClassifiesByConcreteType() {
        let (kind, source) = classifyScalarServingNativeCacheEntry(KVCacheSimple())

        XCTAssertEqual(kind, .denseAttention)
        XCTAssertEqual(source, .concreteType)
    }

    /// A mixed vector mimicking the real 48-layer topology (36 recurrent-reporting + 12
    /// dense-reporting caches, all classified via the marker protocol) must still route through
    /// `classifyScalarServingDecoderRoute` to `.nativeHeterogeneous`, exactly as the equivalent
    /// concrete-typed vector would — the marker protocol changes HOW a cache is classified, not
    /// what decoder route a given classification produces.
    func testMixedMarkerClassifiedVectorRoutesToNativeHeterogeneous() throws {
        let caches: [any KVCache] =
            (0..<36).map { _ in MarkerReportingFakeCache(servingCacheLayerKind: .recurrentState) }
            + (0..<12).map { _ in MarkerReportingFakeCache(servingCacheLayerKind: .denseAttention) }

        let kinds = classifyScalarServingNativeCaches(caches)
        let route = try classifyScalarServingDecoderRoute(kinds)

        XCTAssertEqual(route, .nativeHeterogeneous)
    }

    /// No-regression guarantee for every family serving today: when every classification came
    /// from a concrete-type match, the gate never triggers, regardless of family — including a
    /// nil (unreadable config.json) family.
    func testMarkerFamilyAdmissionAdmitsAllConcreteTypeClassificationsRegardlessOfFamily() {
        let allConcrete: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] = [
            (.denseAttention, .concreteType),
            (.recurrentState, .concreteType),
            (.composite, .concreteType),
        ]

        XCTAssertNil(
            scalarServingMarkerFamilyAdmissionError(
                classifications: allConcrete, family: "qwen3", offloadedNGramPlanResolved: false))
        XCTAssertNil(
            scalarServingMarkerFamilyAdmissionError(
                classifications: allConcrete, family: nil, offloadedNGramPlanResolved: false))
    }

    /// Any marker-classified entry for a family not on the proof allowlist is refused, carrying
    /// that family lowercased.
    func testMarkerFamilyAdmissionRefusesUnprovenFamilyWithMarkerClassification() {
        let classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] = [
            (.recurrentState, .markerProtocol)
        ]

        XCTAssertEqual(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: "SomeNewFamily",
                offloadedNGramPlanResolved: false),
            .unprovenServingFamily("somenewfamily"))
    }

    /// A missing/unreadable family becomes "unknown" before the allowlist check and the refusal.
    func testMarkerFamilyAdmissionRefusesNilFamilyAsUnknown() {
        let classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] = [
            (.denseAttention, .markerProtocol)
        ]

        XCTAssertEqual(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: nil, offloadedNGramPlanResolved: false),
            .unprovenServingFamily("unknown"))
    }

    /// Realistic mixed vector mirroring the real 48-layer topology (36 recurrent + 12 dense, all
    /// marker-classified) is refused for an unproven family.
    func testMarkerFamilyAdmissionRefusesRealistic48LayerMarkerTopology() {
        let classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] =
            (0..<36).map { _ in (kind: ScalarServingNativeCacheKind.recurrentState, source: ScalarServingCacheClassificationSource.markerProtocol) }
            + (0..<12).map { _ in (kind: ScalarServingNativeCacheKind.denseAttention, source: ScalarServingCacheClassificationSource.markerProtocol) }

        XCTAssertEqual(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: "qwen3_5",
                offloadedNGramPlanResolved: false),
            .unprovenServingFamily("qwen3_5"))
    }

    /// An `.unclassified` entry alongside concrete ones is NOT this gate's concern — rejecting
    /// `.unknown` kinds is the route classifier's job, not this admission gate's. No marker
    /// classification is present, so the gate admits.
    func testMarkerFamilyAdmissionIgnoresUnclassifiedEntriesAlongsideConcreteOnes() {
        let classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] = [
            (.denseAttention, .concreteType),
            (.unknown, .unclassified),
        ]

        XCTAssertNil(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: "qwen3", offloadedNGramPlanResolved: false))
    }

    /// The gate triggers on ANY marker-classified entry, not only when every entry is
    /// marker-classified — a mixed-source vector with just one marker entry is still refused.
    func testMarkerFamilyAdmissionRefusesMixedSourceVectorWithAnyMarkerEntry() {
        let classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] = [
            (.denseAttention, .concreteType),
            (.recurrentState, .markerProtocol),
            (.composite, .concreteType),
        ]

        XCTAssertEqual(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: "qwen3", offloadedNGramPlanResolved: false),
            .unprovenServingFamily("qwen3"))
    }

    /// The load-bearing leg for `qwen4_exp` (Qwen3.8-Flash-Next): a marker-classified vector for
    /// the proven family IS admitted when the caller attests the offloaded n-gram plan actually
    /// resolved for this load.
    func testMarkerFamilyAdmissionAdmitsFlashNextOnlyWhenOffloadedNGramPlanResolved() {
        let classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] =
            (0..<36).map { _ in (kind: ScalarServingNativeCacheKind.recurrentState, source: ScalarServingCacheClassificationSource.markerProtocol) }
            + (0..<12).map { _ in (kind: ScalarServingNativeCacheKind.denseAttention, source: ScalarServingCacheClassificationSource.markerProtocol) }

        XCTAssertNil(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: "qwen4_exp", offloadedNGramPlanResolved: true))
    }

    /// The load-bearing leg's inverse: `qwen4_exp` is refused when the offloaded n-gram plan did
    /// NOT resolve — this is what keeps the ~106 GiB fully-resident `loadModel` route from being
    /// silently admitted by a bare family allowlist. This refuses via
    /// `servingFamilyRequiresResolvedOffloadedNGramPlan`, NOT `unprovenServingFamily` — the family's
    /// serving proof is real, only this load's route is wrong, which is a different, actionable
    /// condition from a family having no proof at all.
    func testMarkerFamilyAdmissionRefusesFlashNextWhenOffloadedNGramPlanDidNotResolve() {
        let classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] =
            (0..<36).map { _ in (kind: ScalarServingNativeCacheKind.recurrentState, source: ScalarServingCacheClassificationSource.markerProtocol) }
            + (0..<12).map { _ in (kind: ScalarServingNativeCacheKind.denseAttention, source: ScalarServingCacheClassificationSource.markerProtocol) }

        XCTAssertEqual(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: "qwen4_exp", offloadedNGramPlanResolved: false),
            .servingFamilyRequiresResolvedOffloadedNGramPlan("qwen4_exp"))
    }

    /// A resolved offload plan is not a blanket bypass: an unrelated family is still refused even
    /// when the caller attests the offloaded n-gram plan resolved.
    func testMarkerFamilyAdmissionRefusesUnrelatedFamilyEvenWithOffloadedNGramPlanResolved() {
        let classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] = [
            (.recurrentState, .markerProtocol)
        ]

        XCTAssertEqual(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: "SomeNewFamily",
                offloadedNGramPlanResolved: true),
            .unprovenServingFamily("somenewfamily"))
    }

    /// Case-folding leg: the family comparison is case-insensitive on both the allowlist check and
    /// the offload-plan-resolved admission, so `"QWEN4_EXP"` behaves identically to `"qwen4_exp"`.
    func testMarkerFamilyAdmissionFlashNextIsCaseInsensitive() {
        let classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] = [
            (.recurrentState, .markerProtocol)
        ]

        XCTAssertNil(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: "QWEN4_EXP", offloadedNGramPlanResolved: true))
        XCTAssertEqual(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: "QWEN4_EXP", offloadedNGramPlanResolved: false),
            .servingFamilyRequiresResolvedOffloadedNGramPlan("qwen4_exp"))
    }

    /// An UNLISTED family with a `.markerProtocol` classification still refuses via the generic
    /// `unprovenServingFamily` — only a family on
    /// `markerClassifiedFamiliesProvenOnlyViaResolvedOffloadedNGramPlan` can ever resolve to the
    /// more specific `servingFamilyRequiresResolvedOffloadedNGramPlan` case.
    func testMarkerFamilyAdmissionRefusesUnlistedFamilyAsUnprovenServingFamily() {
        let classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] = [
            (.recurrentState, .markerProtocol)
        ]

        XCTAssertEqual(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: "llama", offloadedNGramPlanResolved: false),
            .unprovenServingFamily("llama"))
        XCTAssertEqual(
            scalarServingMarkerFamilyAdmissionError(
                classifications: classifications, family: "qwen3", offloadedNGramPlanResolved: true),
            .unprovenServingFamily("qwen3"))
    }

    /// Locks the operator-facing announce line for `servingFamilyRequiresResolvedOffloadedNGramPlan`
    /// so it cannot silently regress into something that no longer names the concrete remedy flag.
    /// Mirrors `testFallbackAnnounceLineFormatIsLocked`'s locked-string shape for
    /// `scalarHybridFallbackAnnounceLine`.
    func testServingFamilyRequiresResolvedOffloadedNGramPlanAnnounceLineNamesTheRemedyFlag() {
        let error = ScalarServingModelLoadError.servingFamilyRequiresResolvedOffloadedNGramPlan(
            "qwen4_exp")

        XCTAssertEqual(
            scalarServingModelLoadRefusalAnnounceLine(error),
            "fastmlx-serve configuration=refused reason=serving_family_requires_offload_plan "
                + "model_type=qwen4_exp remedy=--ngram-offload-plan")
    }

    /// A case with no bespoke remedy must render an ACCURATE, non-misleading line naming its own
    /// case rather than borrowing the offload-plan reason/remedy it did not earn — the top-level
    /// catch in `FastMLXServe.main` handles the whole `ScalarServingModelLoadError` type, so a
    /// case with no dedicated branch must never collapse into a reason string implying it does.
    func testUnprovenServingFamilyAnnounceLineIsGenericAndDoesNotNameTheOffloadPlanRemedy() {
        let line = scalarServingModelLoadRefusalAnnounceLine(
            ScalarServingModelLoadError.unprovenServingFamily("qwen3"))

        XCTAssertEqual(
            line,
            "fastmlx-serve configuration=refused reason=scalar_serving_model_load_error "
                + "detail=unprovenServingFamily(\"qwen3\")")
        XCTAssertFalse(line.contains("--ngram-offload-plan"))
        XCTAssertFalse(line.contains("serving_family_requires_offload_plan"))
    }

    /// Anti-vacuity, load-bearing test: a LISTED family (`qwen4_exp`) refused only because the
    /// offloaded n-gram plan did not resolve must NOT collapse into the same error case as an
    /// UNLISTED family that has no serving proof at all. Comparing the two `ScalarServingModelLoadError`
    /// VALUES directly (e.g. `XCTAssertNotEqual`) would be vacuous here — the associated family
    /// string differs ("qwen4_exp" vs "llama") regardless of whether the underlying CASE collapsed,
    /// so that comparison would pass both before and after the fix. Discriminating on the case
    /// itself (ignoring the associated payload) is what actually exercises the bug: today both
    /// inputs produce `.unprovenServingFamily`, so `isUnprovenServingFamily(listedButUnresolved)`
    /// is `true` and the `XCTAssertFalse` below fails. After the fix it resolves to its own
    /// actionable case instead.
    func testMarkerFamilyAdmissionListedUnresolvedAndUnlistedFamilyUseDifferentErrorCases() {
        let classifications: [(kind: ScalarServingNativeCacheKind, source: ScalarServingCacheClassificationSource)] = [
            (.recurrentState, .markerProtocol)
        ]

        let listedButUnresolved = scalarServingMarkerFamilyAdmissionError(
            classifications: classifications, family: "qwen4_exp", offloadedNGramPlanResolved: false)
        let unlistedFamily = scalarServingMarkerFamilyAdmissionError(
            classifications: classifications, family: "llama", offloadedNGramPlanResolved: false)

        func isUnprovenServingFamily(_ error: ScalarServingModelLoadError?) -> Bool {
            if case .unprovenServingFamily = error { return true }
            return false
        }

        XCTAssertTrue(
            isUnprovenServingFamily(unlistedFamily),
            "an unlisted family must still refuse via .unprovenServingFamily")
        XCTAssertFalse(
            isUnprovenServingFamily(listedButUnresolved),
            "a LISTED family with an unresolved offload plan must refuse via its own actionable "
                + "case, not collapse into the generic .unprovenServingFamily used for families "
                + "with no serving proof at all")
    }

    func testResetParityPreflightAcceptsTwoExactOneTokenRuns() async throws {
        let result = try await verifyScalarServingResetParity(
            inference: InferenceActor(
                decoder: ScriptedDecoder(script: [1, 99], eos: 99)),
            promptTokens: [10, 11],
            stopTokenIDs: [99])

        XCTAssertEqual(
            result,
            ScalarServingStartupParity(
                promptTokenCount: 2,
                generatedTokenCount: 1,
                verified: true))
    }

    func testResetParityPreflightRejectsMismatchedRuns() async throws {
        do {
            _ = try await verifyScalarServingResetParity(
                inference: InferenceActor(
                    decoder: ResetSensitiveDecoder()),
                promptTokens: [10],
                stopTokenIDs: [99])
            XCTFail("Expected startup parity rejection")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .startupParityMismatch)
        }
    }

    func testLoaderConfigurationRejectsNonFileURLAndUnsafeMemoryPolicy() {
        let backend = ScalarServingBackendConfiguration(
            defaultMaximumCompletionTokens: 32,
            maximumQueuedRequests: 2,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096))

        XCTAssertThrowsError(
            try validateScalarServingModelLoadConfiguration(
                ScalarServingModelLoadConfiguration(
                    launchedModel: "fixture",
                    modelDirectory: URL(string: "relative-model")!,
                    memoryLimitBytes: 4_096,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: backend))
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .modelDirectoryMustBeAbsolute)
        }

        XCTAssertThrowsError(
            try validateScalarServingModelLoadConfiguration(
                ScalarServingModelLoadConfiguration(
                    launchedModel: "fixture",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 1_024,
                    cacheLimitBytes: 2_048,
                    backendConfiguration: backend))
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .cacheLimitExceedsMemoryLimit)
        }
    }

    func testLoaderConfigurationAcceptsExistingAbsoluteDirectoryAndExplicitLimits() throws {
        let configuration = ScalarServingModelLoadConfiguration(
            launchedModel: "fixture",
            modelDirectory: URL(fileURLWithPath: "/tmp"),
            memoryLimitBytes: 4_096,
            cacheLimitBytes: 1_024,
            backendConfiguration: .init(
                defaultMaximumCompletionTokens: 32,
                maximumQueuedRequests: 2,
                queueRetryAfterSeconds: 1,
                mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096)))

        let validated = try validateScalarServingModelLoadConfiguration(
            configuration)

        XCTAssertEqual(validated.launchedModel, "fixture")
        XCTAssertEqual(validated.modelDirectory.path, "/tmp")
        XCTAssertEqual(validated.memoryLimitBytes, 4_096)
        XCTAssertEqual(validated.cacheLimitBytes, 1_024)
    }

    func testStartupReportMemoryFieldsFragmentRendersSnakeCaseBytes() {
        let report = ScalarServingModelStartupReport(
            launchedModel: "fixture",
            route: .scalarGreedy,
            memoryLimitBytes: 8,
            cacheLimitBytes: 4,
            stopTokenCount: 1,
            stopStringCount: 0,
            nativeCacheKinds: [.denseAttention],
            startupPromptTokenCount: 2,
            startupGeneratedTokenCount: 1,
            resetParityVerified: true,
            mlxActiveBytes: 5_540_000_000,
            mlxCacheBytes: 1_073_741_824,
            mlxPeakBytes: 6_000_000_000)
        XCTAssertEqual(
            report.memoryFieldsFragment,
            "mlx_active_bytes=5540000000 mlx_cache_bytes=1073741824 "
                + "mlx_peak_bytes=6000000000")
    }

    func testStartupReportMemoryFieldsDefaultToZeroWhenUnspecified() {
        // Backward-compatible init: existing construction sites that don't pass MLX
        // memory get a well-defined zero fragment, never a crash or garbage bytes.
        let report = ScalarServingModelStartupReport(
            launchedModel: "fixture",
            route: .scalarGreedy,
            memoryLimitBytes: 8,
            cacheLimitBytes: 4,
            stopTokenCount: 1,
            stopStringCount: 0,
            nativeCacheKinds: [.denseAttention],
            startupPromptTokenCount: 2,
            startupGeneratedTokenCount: 1,
            resetParityVerified: true)
        XCTAssertEqual(
            report.memoryFieldsFragment,
            "mlx_active_bytes=0 mlx_cache_bytes=0 mlx_peak_bytes=0")
    }

    // MARK: - rejectedPromptTokenIDs: mapping a loaded model's own reported unsupported input
    // token IDs (`UnsupportedInputTokenReporting`) into the scalar backend's screen.

    /// A model reporting unsupported input token IDs has them returned, converted to `Int`.
    func testRejectedPromptTokenIDsReturnsAConformingModelsReportedIDsConvertedToInt() {
        let model = FixtureUnsupportedTokenReportingModel(
            unsupportedInputTokenIDs: [100, 200, 300])

        XCTAssertEqual(
            servingRejectedPromptTokenIDs(model: model),
            [100, 200, 300])
    }

    /// A model that does NOT conform to `UnsupportedInputTokenReporting` — every family today —
    /// must map to the empty set, so the screen stays inert (today's unchanged behavior).
    func testRejectedPromptTokenIDsReturnsEmptyForANonConformingModel() {
        let model = FixtureNonReportingModel()

        XCTAssertEqual(servingRejectedPromptTokenIDs(model: model), [])
    }

    /// A conforming model that reports no unsupported input tokens maps to the empty set.
    func testRejectedPromptTokenIDsReturnsEmptyForAConformingModelWithAnEmptyReport() {
        let model = FixtureUnsupportedTokenReportingModel(unsupportedInputTokenIDs: [])

        XCTAssertEqual(servingRejectedPromptTokenIDs(model: model), [])
    }

    /// The mapping is a correct, total identity over the full `Int64` domain, extremes included.
    ///
    /// `servingRejectedPromptTokenIDs` uses `Int(exactly:)` and SKIPS a non-representable ID. On
    /// every platform this package deploys to, `Int` and `Int64` are both 64-bit with identical
    /// range, so no `Int64` value exists that `Int(exactly:)` would reject — the skip branch is
    /// defensive/forward-looking and is **unreachable here**. That is stated rather than faked: a
    /// test claiming to exercise the skip would need a fake whose reported value cannot occur by
    /// construction, which would assert a false claim about tested behavior. This test instead pins
    /// what IS verifiable — that nothing is silently dropped anywhere in the domain, including both
    /// extremes, so a future narrowing of the conversion would fail here.
    func testRejectedPromptTokenIDsPreservesEveryValueAcrossTheFullInt64RangeIncludingItsExtremes() {
        let model = FixtureUnsupportedTokenReportingModel(
            unsupportedInputTokenIDs: [Int64.min, -1, 0, 1, Int64.max])

        XCTAssertEqual(
            servingRejectedPromptTokenIDs(model: model),
            [Int(Int64.min), -1, 0, 1, Int(Int64.max)])
    }

    // MARK: - qwen3_5 scalar-route gated-delta kernel viability guard (Dk%32)

    /// VL-wrapped qwen3_5 config with `linear_key_head_dim = 48` — a valid positive-integer geometry the
    /// gated-delta Metal kernel cannot serve (Dk not divisible by 32). Mirrors the continuous adapter's
    /// incr-4 fixture, exercised here on the DEFAULT scalar route the family falls back to.
    private func qwen35UnalignedDkConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
         "text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144,
           "vocab_size":248320,"num_hidden_layers":48,"full_attention_interval":4,
           "num_key_value_heads":8,"head_dim":128,"torch_dtype":"bfloat16",
           "linear_num_key_heads":16,"linear_num_value_heads":32,
           "linear_key_head_dim":48,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
    }

    /// Same shape with `linear_key_head_dim = 128` (a multiple of 32) — the kernel can serve it, so the
    /// pre-load guard must NOT fire (the probe returns 128 and 128 % 32 == 0).
    private func qwen35AlignedDkConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
         "text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144,
           "vocab_size":248320,"num_hidden_layers":48,"full_attention_interval":4,
           "num_key_value_heads":8,"head_dim":128,"torch_dtype":"bfloat16",
           "linear_num_key_heads":16,"linear_num_value_heads":32,
           "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
    }

    private func writeConfigDirectory(_ json: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("scalar-serving-guard-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data(json.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        return directory
    }

    /// Acceptance: a qwen3_5 checkpoint whose Dk is not a multiple of 32 is refused on the scalar route
    /// BEFORE any weight load — the family's default (un-flagged) path fails closed rather than
    /// truncating/faulting in the gated-delta Metal kernel at decode.
    func testScalarQwen35RejectsUnalignedKeyHeadDimBeforeWeightLoad() async throws {
        let directory = try writeConfigDirectory(qwen35UnalignedDkConfigJSON())
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            _ = try await loadScalarServingModel(
                configuration: ScalarServingModelLoadConfiguration(
                    launchedModel: "qwen3_5-scalar-fallback",
                    modelDirectory: directory,
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: ScalarServingBackendConfiguration(
                        defaultMaximumCompletionTokens: 12,
                        maximumQueuedRequests: 1,
                        queueRetryAfterSeconds: 1,
                        mailboxCapacity: .init(maxDeltas: 4, maxBytes: 16 * 1_024))))
            XCTFail("Dk not divisible by 32 must fail closed before weight load")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .hybridKernelKeyHeadDimUnaligned(48))
        }
    }

    /// The pre-load probe reads the qwen3_5 recurrent Dk from config.json without loading weights: 48
    /// (unaligned) and 128 (aligned) are surfaced exactly; a dense (non-qwen3_5) config and an absent
    /// config both return nil, so the guard is a strict no-op off the hybrid family.
    func testScalarQwen35KeyHeadDimProbeIsFamilyScoped() throws {
        let unaligned = try writeConfigDirectory(qwen35UnalignedDkConfigJSON())
        defer { try? FileManager.default.removeItem(at: unaligned) }
        XCTAssertEqual(
            scalarServingQwen35RecurrentKeyHeadDim(modelDirectory: unaligned), 48)

        let aligned = try writeConfigDirectory(qwen35AlignedDkConfigJSON())
        defer { try? FileManager.default.removeItem(at: aligned) }
        XCTAssertEqual(
            scalarServingQwen35RecurrentKeyHeadDim(modelDirectory: aligned), 128)

        let dense = try writeConfigDirectory(#"{"model_type":"qwen3","num_hidden_layers":4}"#)
        defer { try? FileManager.default.removeItem(at: dense) }
        XCTAssertNil(scalarServingQwen35RecurrentKeyHeadDim(modelDirectory: dense))

        let absent = FileManager.default.temporaryDirectory
            .appendingPathComponent("scalar-serving-guard-absent-\(UUID().uuidString)", isDirectory: true)
        XCTAssertNil(scalarServingQwen35RecurrentKeyHeadDim(modelDirectory: absent))
    }

    // MARK: - ngramOffloadPlanURL: offloaded n-gram serving dispatch seam (qwen4_exp only)

    private func fixtureBackendConfiguration() -> ScalarServingBackendConfiguration {
        ScalarServingBackendConfiguration(
            defaultMaximumCompletionTokens: 32,
            maximumQueuedRequests: 2,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096))
    }

    private func writePlanFile() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("scalar-serving-ngram-plan-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let planURL = directory.appendingPathComponent("plan.json")
        try Data("{}".utf8).write(to: planURL)
        return planURL
    }

    /// Default-nil compatibility: constructing a configuration without the new parameter yields
    /// `ngramOffloadPlanURL == nil`, and validation accepts an otherwise-valid configuration
    /// unchanged — every existing call site (none of which passes this parameter) keeps compiling
    /// and passing validation exactly as before.
    func testNgramOffloadPlanURLDefaultsToNilAndValidationAcceptsUnchangedConfiguration() throws {
        let configuration = ScalarServingModelLoadConfiguration(
            launchedModel: "fixture",
            modelDirectory: URL(fileURLWithPath: "/tmp"),
            memoryLimitBytes: 4_096,
            cacheLimitBytes: 1_024,
            backendConfiguration: fixtureBackendConfiguration())

        XCTAssertNil(configuration.ngramOffloadPlanURL)
        let validated = try validateScalarServingModelLoadConfiguration(configuration)
        XCTAssertNil(validated.ngramOffloadPlanURL)
    }

    /// A relative (non-file) plan URL is refused before any other plan-file check, mirroring the
    /// existing `modelDirectory` absolute-path guard.
    func testNgramOffloadPlanURLMustBeAbsoluteFileURLRejectsRelativeURL() {
        XCTAssertThrowsError(
            try validateScalarServingModelLoadConfiguration(
                ScalarServingModelLoadConfiguration(
                    launchedModel: "fixture",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 4_096,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: fixtureBackendConfiguration(),
                    ngramOffloadPlanURL: URL(string: "relative-plan.json")!))
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .ngramOffloadPlanMustBeAbsolute)
        }
    }

    /// An absolute plan URL pointing at a path that does not exist on disk is refused.
    func testNgramOffloadPlanURLUnavailableForMissingPath() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "scalar-serving-ngram-missing-\(UUID().uuidString).json")

        XCTAssertThrowsError(
            try validateScalarServingModelLoadConfiguration(
                ScalarServingModelLoadConfiguration(
                    launchedModel: "fixture",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 4_096,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: fixtureBackendConfiguration(),
                    ngramOffloadPlanURL: missing))
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .ngramOffloadPlanUnavailable)
        }
    }

    /// An absolute plan URL pointing at a DIRECTORY (not a regular file) is refused with the same
    /// error as a missing path — proving the regular-file check, not merely existence.
    func testNgramOffloadPlanURLUnavailableForDirectoryPath() {
        XCTAssertThrowsError(
            try validateScalarServingModelLoadConfiguration(
                ScalarServingModelLoadConfiguration(
                    launchedModel: "fixture",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 4_096,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: fixtureBackendConfiguration(),
                    ngramOffloadPlanURL: URL(fileURLWithPath: "/tmp", isDirectory: true)))
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .ngramOffloadPlanUnavailable)
        }
    }

    /// Acceptance: a valid plan file supplied against a checkpoint whose `config.json` names a
    /// different family (`qwen3`) is refused BEFORE any weight load or global `Memory` mutation —
    /// the offloaded n-gram path is specific to the qwen4_exp checkpoint layout, so a plan against
    /// any other family is an operator error caught at load, not at decode.
    func testLoadRejectsNgramOffloadPlanAgainstAWrongFamilyCheckpointBeforeWeightLoad() async throws {
        let directory = try writeConfigDirectory(#"{"model_type":"qwen3","num_hidden_layers":4}"#)
        defer { try? FileManager.default.removeItem(at: directory) }
        let planURL = try writePlanFile()
        defer {
            try? FileManager.default.removeItem(
                at: planURL.deletingLastPathComponent())
        }
        let memoryLimitBefore = Memory.memoryLimit
        let cacheLimitBefore = Memory.cacheLimit

        do {
            _ = try await loadScalarServingModel(
                configuration: ScalarServingModelLoadConfiguration(
                    launchedModel: "qwen4-exp-ngram-offload",
                    modelDirectory: directory,
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: fixtureBackendConfiguration(),
                    ngramOffloadPlanURL: planURL))
            XCTFail("A plan file against a non-qwen4_exp family must fail closed before weight load")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .ngramOffloadPlanUnsupportedFamily("qwen3"))
        }

        XCTAssertEqual(Memory.memoryLimit, memoryLimitBefore)
        XCTAssertEqual(Memory.cacheLimit, cacheLimitBefore)
    }

    /// Same acceptance as above for a directory with no readable `config.json`: the observed model
    /// type is `nil`, carried as such rather than defaulted to a placeholder string.
    func testLoadRejectsNgramOffloadPlanAgainstAnUnreadableConfigBeforeWeightLoad() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "scalar-serving-ngram-no-config-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let planURL = try writePlanFile()
        defer {
            try? FileManager.default.removeItem(
                at: planURL.deletingLastPathComponent())
        }
        let memoryLimitBefore = Memory.memoryLimit
        let cacheLimitBefore = Memory.cacheLimit

        do {
            _ = try await loadScalarServingModel(
                configuration: ScalarServingModelLoadConfiguration(
                    launchedModel: "qwen4-exp-ngram-offload",
                    modelDirectory: directory,
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: fixtureBackendConfiguration(),
                    ngramOffloadPlanURL: planURL))
            XCTFail("A plan file against an unreadable config.json must fail closed before weight load")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .ngramOffloadPlanUnsupportedFamily(nil))
        }

        XCTAssertEqual(Memory.memoryLimit, memoryLimitBefore)
        XCTAssertEqual(Memory.cacheLimit, cacheLimitBefore)
    }

    // MARK: - qwen4_exp in-checkpoint MTP drafter (item 4 wiring)

    /// Default-nil compatibility: constructing a configuration without the new parameter yields
    /// `inCheckpointMTPSelection == nil`, and validation accepts an otherwise-valid configuration
    /// unchanged — every existing call site (none of which passes this parameter) keeps compiling
    /// and passing validation exactly as before.
    func testInCheckpointMTPSelectionDefaultsToNilAndValidationAcceptsUnchangedConfiguration() throws {
        let configuration = ScalarServingModelLoadConfiguration(
            launchedModel: "fixture",
            modelDirectory: URL(fileURLWithPath: "/tmp"),
            memoryLimitBytes: 4_096,
            cacheLimitBytes: 1_024,
            backendConfiguration: fixtureBackendConfiguration())

        XCTAssertNil(configuration.inCheckpointMTPSelection)
        let validated = try validateScalarServingModelLoadConfiguration(configuration)
        XCTAssertNil(validated.inCheckpointMTPSelection)
    }

    /// Acceptance: a selection supplied against a non-qwen4_exp checkpoint fails closed BEFORE any
    /// weight load or global `Memory` mutation — same shape and same guard block as
    /// `testLoadRejectsNgramOffloadPlanAgainstAWrongFamilyCheckpointBeforeWeightLoad`.
    func testLoadRejectsInCheckpointMTPSelectionAgainstAWrongFamilyCheckpointBeforeWeightLoad() async throws {
        let directory = try writeConfigDirectory(#"{"model_type":"qwen3","num_hidden_layers":4}"#)
        defer { try? FileManager.default.removeItem(at: directory) }
        let memoryLimitBefore = Memory.memoryLimit
        let cacheLimitBefore = Memory.cacheLimit

        do {
            _ = try await loadScalarServingModel(
                configuration: ScalarServingModelLoadConfiguration(
                    launchedModel: "qwen4-exp-mtp",
                    modelDirectory: directory,
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: fixtureBackendConfiguration(),
                    inCheckpointMTPSelection: .converted4Bit))
            XCTFail("A qwen4_exp MTP selection against a non-qwen4_exp family must fail closed before weight load")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .inCheckpointMTPUnsupportedFamily("qwen3"))
        }

        XCTAssertEqual(Memory.memoryLimit, memoryLimitBefore)
        XCTAssertEqual(Memory.cacheLimit, cacheLimitBefore)
    }

    /// Same acceptance as above for a directory with no readable `config.json`: the observed model
    /// type is `nil`, carried as such rather than defaulted to a placeholder string.
    func testLoadRejectsInCheckpointMTPSelectionAgainstAnUnreadableConfigBeforeWeightLoad() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "scalar-serving-qwen4exp-mtp-no-config-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let memoryLimitBefore = Memory.memoryLimit
        let cacheLimitBefore = Memory.cacheLimit

        do {
            _ = try await loadScalarServingModel(
                configuration: ScalarServingModelLoadConfiguration(
                    launchedModel: "qwen4-exp-mtp",
                    modelDirectory: directory,
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: fixtureBackendConfiguration(),
                    inCheckpointMTPSelection: .converted4Bit))
            XCTFail("A qwen4_exp MTP selection against an unreadable config.json must fail closed before weight load")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .inCheckpointMTPUnsupportedFamily(nil))
        }

        XCTAssertEqual(Memory.memoryLimit, memoryLimitBefore)
        XCTAssertEqual(Memory.cacheLimit, cacheLimitBefore)
    }

    /// Exhaustive mapping regression: `ServingCore`'s MLX-free namespace mirror must map to the
    /// SAME `MLXLLM` runtime case it names, for both known cases. A future third case in either
    /// enum is a compile error at the mapping function's `switch`, not a silent mismap here.
    func testInCheckpointMTPRuntimeNamespaceMapsBothCasesToTheMatchingRuntimeCase() {
        XCTAssertEqual(inCheckpointMTPRuntimeNamespace(.official), .official)
        XCTAssertEqual(inCheckpointMTPRuntimeNamespace(.converted), .converted)
    }

    /// Acceptance (happy path): identical token sequences, no passthrough, and at least one
    /// accepted draft token together yield a passing verdict carrying the observed telemetry, AND
    /// report `tokenSequencesMatched == true` — the "degrade does NOT fire" control for the
    /// divergent-pair coverage below (either alone is inert; see this file's `MARK` for the pair).
    func testStartupReadinessDecisionAdmitsIdenticalSequencesWithGenuineSpeculation() throws {
        let scalar = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: nil)
        let speculative = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 4,
            acceptedDraftTokens: 2,
            passthroughReason: nil)

        let verdict = try inCheckpointMTPStartupReadinessDecision(
            promptTokenCount: 5, scalar: scalar, speculative: speculative)

        XCTAssertEqual(verdict.promptTokenCount, 5)
        XCTAssertEqual(verdict.generatedTokenCount, 3)
        XCTAssertEqual(verdict.proposedDraftTokens, 4)
        XCTAssertEqual(verdict.acceptedDraftTokens, 2)
        XCTAssertTrue(verdict.tokenSequencesMatched)
    }

    /// Clause (i) is DATA, not a throw: a token-count divergence — clause (ii) still fully
    /// evaluated and passing here — SUCCEEDS with `tokenSequencesMatched == false`. This is the
    /// "degrade FIRES" case at the decision-function boundary: a real, previously-measured
    /// architectural divergence between the scalar and speculative arms
    /// (`docs/task-inbox/2026-09-07-mtp-scalar-route-divergence-DECISION.md`) must not refuse a
    /// correct integration.
    func testStartupReadinessDecisionReportsDivergentTokenCountsAsUnmatchedRatherThanThrowing()
        throws
    {
        let scalar = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: nil)
        let speculative = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22],
            proposedDraftTokens: 4,
            acceptedDraftTokens: 2,
            passthroughReason: nil)

        let verdict = try inCheckpointMTPStartupReadinessDecision(
            promptTokenCount: 5, scalar: scalar, speculative: speculative)

        XCTAssertFalse(verdict.tokenSequencesMatched)
        XCTAssertEqual(verdict.proposedDraftTokens, 4)
        XCTAssertEqual(verdict.acceptedDraftTokens, 2)
    }

    /// Clause (i) is DATA: same-length but different-content sequences also report
    /// `tokenSequencesMatched == false` rather than throwing — the SHA-256 fold, not just the
    /// count, discriminates, exactly as it did when this clause threw.
    func testStartupReadinessDecisionReportsSameLengthDivergentTokenContentAsUnmatched() throws {
        let scalar = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: nil)
        let speculative = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 34],
            proposedDraftTokens: 4,
            acceptedDraftTokens: 2,
            passthroughReason: nil)

        let verdict = try inCheckpointMTPStartupReadinessDecision(
            promptTokenCount: 5, scalar: scalar, speculative: speculative)

        XCTAssertFalse(verdict.tokenSequencesMatched)
    }

    /// THE ANTI-VACUITY MUTATION for clause (ii): scalar and speculative token sequences are
    /// IDENTICAL — clause (i) alone would pass this vacuously, exactly the way it would if the
    /// iterator silently degraded to passthrough and simply replayed scalar decode. Only because
    /// clause (ii) separately asserts `passthroughReason == nil` does this run still fail closed.
    /// This is the "stub `supportsSpeculation` to false" mutation from the decision doc, applied
    /// at the decision function's own boundary (its input already reflects that stub's effect —
    /// passthrough engaged despite byte-identical output).
    func testStartupReadinessDecisionRejectsIdenticalSequencesWhenPassthroughEngaged() {
        let scalar = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: nil)
        let speculative = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: "drafter does not support this target model")

        XCTAssertThrowsError(
            try inCheckpointMTPStartupReadinessDecision(
                promptTokenCount: 5, scalar: scalar, speculative: speculative)
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .inCheckpointMTPStartupDidNotSpeculate(
                    reason: "drafter does not support this target model",
                    proposedDraftTokens: 0,
                    acceptedDraftTokens: 0))
        }
    }

    /// Passthrough refuses even when the iterator ran at least one genuine round BEFORE degrading:
    /// `proposedDraftTokens > 0` does not rescue a run whose `passthroughReason != nil` — clause
    /// (ii)'s passthrough guard fires unconditionally, ahead of (and independent from) the
    /// proposed-count guard below it. Sticky passthrough is a statement about the REST of the run,
    /// not just the round that measured `proposedDraftTokens`.
    func testStartupReadinessDecisionRejectsPassthroughEvenAfterAGenuineRound() {
        let scalar = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: nil)
        let speculative = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 4,
            acceptedDraftTokens: 2,
            passthroughReason: "drafter does not support this target model")

        XCTAssertThrowsError(
            try inCheckpointMTPStartupReadinessDecision(
                promptTokenCount: 5, scalar: scalar, speculative: speculative)
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .inCheckpointMTPStartupDidNotSpeculate(
                    reason: "drafter does not support this target model",
                    proposedDraftTokens: 4,
                    acceptedDraftTokens: 2))
        }
    }

    /// REQUIRED TEST 3 (task background): divergence AND passthrough together still THROW
    /// `didNotSpeculate` — proves clause (ii) stays fail-closed and that clause (i)'s demotion to
    /// data did NOT swallow an availability failure. `scalar`/`speculative` here diverge in BOTH
    /// count and content (unlike the passthrough tests above, which reuse identical sequences),
    /// so this is a genuinely different input from either of the two tests above, not a
    /// recombination: clause (i) would report `tokenSequencesMatched == false` if it ever reached
    /// the return statement, and clause (ii) refuses before it gets the chance. This is inherently
    /// a property of the pure decision function's ORDERING (clause (ii) evaluated regardless of
    /// clause (i)'s outcome) — real MLX iterator machinery cannot construct this combination
    /// end to end, because `MTPSpeculativeTokenIterator`'s sticky passthrough is decided entirely
    /// at `init` (`MTPSpeculativeTokenIterator.swift:219-220`) and, once engaged, drives every
    /// subsequent forward through the identical single-token path scalar decode itself uses
    /// (`passthroughStep()`, `MTPSpeculativeTokenIterator.swift:905-927`), so a real passthrough
    /// run against a deterministic target can only ever reproduce clause (i)'s own reference
    /// sequence, never diverge from it.
    func testStartupReadinessDecisionThrowsDidNotSpeculateWhenDivergentAndPassthroughEngaged() {
        let scalar = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: nil)
        let speculative = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 99],
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: "drafter does not support this prompt input")

        XCTAssertThrowsError(
            try inCheckpointMTPStartupReadinessDecision(
                promptTokenCount: 5, scalar: scalar, speculative: speculative)
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .inCheckpointMTPStartupDidNotSpeculate(
                    reason: "drafter does not support this prompt input",
                    proposedDraftTokens: 0,
                    acceptedDraftTokens: 0))
        }
    }

    /// DELIBERATELY INVERTED from this test's prior form (which required REFUSAL here — that WAS
    /// the production-availability bug this change fixes). `MTPSpeculativeTokenIterator`'s greedy
    /// acceptance walk legitimately sets `accepted == 0` whenever the drafter's first proposal for
    /// a round differs from the target's own greedy token: the iterator still emits the target's
    /// greedy token (clause (i) stays byte-exact) and the round completed — `draftBlock` was called,
    /// so `proposedDraftTokens` is real. Gating startup on `acceptedDraftTokens > 0` turned that
    /// legitimate, deterministic outcome into a PERMANENT boot failure for a CORRECT drafter
    /// whenever its first-token prediction for the fixed startup prompt happened to diverge once.
    /// Acceptance is a PERFORMANCE property, not an availability one — it is now reported on the
    /// verdict (`acceptedDraftTokens == 0` is visible to the caller) rather than gated.
    func testStartupReadinessDecisionAdmitsGenuineSpeculationThatAcceptedZeroDraftTokens() throws {
        let scalar = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: nil)
        let speculative = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 2,
            acceptedDraftTokens: 0,
            passthroughReason: nil)

        let verdict = try inCheckpointMTPStartupReadinessDecision(
            promptTokenCount: 5, scalar: scalar, speculative: speculative)

        XCTAssertEqual(verdict.proposedDraftTokens, 2)
        XCTAssertEqual(verdict.acceptedDraftTokens, 0)
    }

    /// THE anti-vacuity test for the fixed clause (ii): identical sequences, `passthroughReason ==
    /// nil` (the iterator claims it stayed speculative end to end), but `proposedDraftTokens == 0`
    /// — `drafter.draftBlock` was never actually called. This is the case the proposed-based
    /// conjunct exists to catch: it is the only remaining way a vacuous scalar-decode replay could
    /// slip past clause (i) alone now that acceptance is no longer gated.
    func testStartupReadinessDecisionRejectsIdenticalSequencesWithZeroProposedDraftTokens() {
        let scalar = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: nil)
        let speculative = InCheckpointMTPGreedyDecodeResult(
            tokens: [11, 22, 33],
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: nil)

        XCTAssertThrowsError(
            try inCheckpointMTPStartupReadinessDecision(
                promptTokenCount: 5, scalar: scalar, speculative: speculative)
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .inCheckpointMTPStartupDidNotSpeculate(
                    reason: nil, proposedDraftTokens: 0, acceptedDraftTokens: 0))
        }
    }

    // MARK: - inCheckpointMTPDrafterRetentionDecision (the retention/reporting desync gap)
    //
    // Mutation-found gap (cycle 82 continuation): `loadScalarServingModel` used to read
    // `readiness.tokenSequencesMatched` TWICE, independently -- once for the reported
    // `drafterServing` field, once for the `if` that actually retains the drafter into
    // `retainedInCheckpointMTPDrafter`. Mutating the retention `if` to unconditional retention left
    // the full `SpikeServingAdaptersTests` suite (257 tests) green, because nothing asserted the
    // two reads agree. `inCheckpointMTPDrafterRetentionDecision` is now the ONE function both
    // reads call, and `loadScalarServingModel` binds its result to a single local `let` shared by
    // both.
    //
    // STATED HONESTLY: `testDrafterRetentionDecisionIsFalseWhenTokenSequencesDiverged` below is
    // the test that actually catches a regression -- re-mutating
    // `inCheckpointMTPDrafterRetentionDecision` to hardcode `true` was verified to fail exactly
    // that one test (260 executed, 1 failure). `testRetainedDrafterPresenceAndReportedDrafterServingAlwaysAgree`
    // below does NOT independently catch that same mutation: it computes both the "retained"
    // stand-in and `verdict.drafterServing` from the identical local, so a wrong-but-*consistent*
    // return value from the helper still leaves them agreeing with each other, just agreeing on
    // the wrong thing. Its purpose is narrower and was verified separately (re-deleting the
    // production `if` guard entirely, so retention and reporting read the helper's result
    // differently, was NOT caught by anything in this file -- see the doc comment on
    // `inCheckpointMTPDrafterRetentionDecision` for that residual, honestly disclosed rather than
    // implied fixed): it pins the INTENDED shape of the call site as an executable mirror, and
    // documents the invariant for a future reader, rather than exercising
    // `loadScalarServingModel`'s own body (fleet-only limitation, same as the rest of this file --
    // see the class-level "WHAT THIS FILE CANNOT PROVE" note in `MTPDecoderBridgeSelectionTests.swift`).

    /// Retention decision mirrors clause (i) exactly: `false` when the readiness gate's scalar and
    /// speculative arms diverged -- the "do not retain a pairing this run gave no evidence for"
    /// case `loadScalarServingModel`'s comment above the `if` describes.
    func testDrafterRetentionDecisionIsFalseWhenTokenSequencesDiverged() {
        let readiness = InCheckpointMTPStartupReadiness(
            promptTokenCount: 5,
            generatedTokenCount: 8,
            proposedDraftTokens: 4,
            acceptedDraftTokens: 2,
            tokenSequencesMatched: false)

        XCTAssertFalse(inCheckpointMTPDrafterRetentionDecision(readiness: readiness))
    }

    /// The other half of the same mapping: `true` when the arms matched. Paired with the test
    /// above so neither a stub `{ true }` nor a stub `{ false }` implementation of
    /// `inCheckpointMTPDrafterRetentionDecision` could pass both.
    func testDrafterRetentionDecisionIsTrueWhenTokenSequencesMatched() {
        let readiness = InCheckpointMTPStartupReadiness(
            promptTokenCount: 5,
            generatedTokenCount: 8,
            proposedDraftTokens: 4,
            acceptedDraftTokens: 2,
            tokenSequencesMatched: true)

        XCTAssertTrue(inCheckpointMTPDrafterRetentionDecision(readiness: readiness))
    }

    /// Executable mirror of `loadScalarServingModel`'s call site: a single
    /// `drafterRetentionDecision` local feeds BOTH the retained-drafter presence (modeled here as
    /// `retainedDrafter: Bool?` standing in for `retainedInCheckpointMTPDrafter`, since this file
    /// cannot construct a real `MTPDrafterModel`) and `ScalarServingInCheckpointMTPStartupVerdict.drafterServing`.
    /// Run for both `tokenSequencesMatched` values so this is not vacuously true for a single
    /// hardcoded outcome. Documents the intended shape and pins it against an accidental future
    /// rewrite of the mirror itself -- see this file's `MARK` comment above for what this test
    /// does NOT independently catch (a wrong-but-internally-consistent helper return value, or a
    /// production edit that deletes the retention `if` guard), verified by mutation rather than
    /// assumed.
    func testRetainedDrafterPresenceAndReportedDrafterServingAlwaysAgree() {
        for tokenSequencesMatched in [true, false] {
            let readiness = InCheckpointMTPStartupReadiness(
                promptTokenCount: 5,
                generatedTokenCount: 8,
                proposedDraftTokens: 4,
                acceptedDraftTokens: 2,
                tokenSequencesMatched: tokenSequencesMatched)

            // Exactly what `loadScalarServingModel` now does: ONE call, ONE local `let`, read by
            // both consumers below.
            let drafterRetentionDecision = inCheckpointMTPDrafterRetentionDecision(
                readiness: readiness)

            var retainedDrafter: Bool?
            if drafterRetentionDecision {
                retainedDrafter = true
            }
            let verdict = ScalarServingInCheckpointMTPStartupVerdict(
                namespace: .converted,
                revision: "rev-desync-check",
                sourceKeyCount: 1,
                promptTokenCount: readiness.promptTokenCount,
                generatedTokenCount: readiness.generatedTokenCount,
                proposedDraftTokens: readiness.proposedDraftTokens,
                acceptedDraftTokens: readiness.acceptedDraftTokens,
                drafterServing: drafterRetentionDecision,
                drafterActiveBytesDelta: 0,
                drafterCacheBytesDelta: 0)

            XCTAssertEqual(
                retainedDrafter != nil, verdict.drafterServing,
                "retained-drafter presence and the reported drafter_serving value must always "
                    + "agree (tokenSequencesMatched=\(tokenSequencesMatched))")
            XCTAssertEqual(retainedDrafter != nil, drafterRetentionDecision)
        }
    }

    // MARK: - Black-box CLI coverage: `FastMLXServe.main()`'s
    // `catch let error as ScalarServingModelLoadError` arm

    /// Same shape as `Qwen38ScorecardProofRunnerCLITests`'s `productsDirectory`/binary-invocation
    /// idiom, applied to the built `fastmlx-serve` binary (a TEST-ONLY dependency of this target —
    /// see `Package.swift`'s `SpikeServingAdaptersTests` comment).
    private static var productsDirectory: URL {
        Bundle(for: MLXScalarServingTests.self).bundleURL.deletingLastPathComponent()
    }

    private static var serveURL: URL {
        productsDirectory.appendingPathComponent("fastmlx-serve")
    }

    private struct ServeCLIResult {
        let exitStatus: Int32
        let stdout: String
        let stderr: String
    }

    private func runServe(arguments: [String]) throws -> ServeCLIResult {
        let binary = Self.serveURL
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: binary.path),
            "fastmlx-serve binary missing at \(binary.path); "
                + "SpikeServingAdaptersTests must depend on the executable target")
        let process = Process()
        process.executableURL = binary
        process.arguments = arguments
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return ServeCLIResult(
            exitStatus: process.terminationStatus,
            stdout: String(decoding: stdoutData, as: UTF8.self),
            stderr: String(decoding: stderrData, as: UTF8.self))
    }

    /// Synthetic qwen3_5 checkpoint directory: a full config.json (every field
    /// `ModelConfigDecoder.decode` requires for the hybrid-linear arch class, so the CLI's pre-load
    /// fit-check decodes and proceeds) with `linear_key_head_dim` set to 33 — not a multiple of 32,
    /// so the scalar route's post-fit-check gated-delta kernel viability guard
    /// (`scalarServingQwen35RecurrentKeyHeadDim`/`ScalarServingModelLoadError
    /// .hybridKernelKeyHeadDimUnaligned`) refuses it — plus a `model.safetensors.index.json`
    /// declaring a small total size so the fit-check sizes the checkpoint honestly (`weightsAreDeclared
    /// = true`) without any real weight shards on disk. Mirrors `qwen35UnalignedDkConfigJSON` above,
    /// which exercises the same guard directly (not through the CLI/full fit-check pipeline).
    private func writeUnalignedQwen35CheckpointDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "scalar-serving-cli-unaligned-dk-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let configJSON = #"""
            {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
             "text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144,
               "vocab_size":248320,"num_hidden_layers":48,"full_attention_interval":4,
               "num_key_value_heads":8,"head_dim":128,"torch_dtype":"bfloat16",
               "linear_num_key_heads":16,"linear_num_value_heads":32,
               "linear_key_head_dim":33,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
            """#
        try Data(configJSON.utf8).write(to: directory.appendingPathComponent("config.json"))
        let indexJSON = #"{"metadata":{"total_size":50000000}}"#
        try Data(indexJSON.utf8).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    /// Acceptance: the CLI's `catch let error as ScalarServingModelLoadError` arm renders the
    /// machine-readable refusal line and exits 2 (rather than a raw Swift top-level fatalError trap,
    /// exit 133) when a real invocation's pre-load kernel viability guard refuses a checkpoint.
    func testServeCLIRefusesUnalignedQwen35KeyHeadDimWithExitTwo() throws {
        let directory = try writeUnalignedQwen35CheckpointDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let result = try runServe(arguments: [
            "--model", "qwen3_5-cli-fixture",
            "--model-path", directory.path,
            "--host", "127.0.0.1",
            "--port", "58732",
        ])

        XCTAssertEqual(result.exitStatus, 2)
        XCTAssertTrue(
            result.stderr.contains("configuration=refused"),
            "unexpected stderr: \(result.stderr)")
        XCTAssertTrue(
            result.stderr.contains("reason=scalar_serving_model_load_error"),
            "unexpected stderr: \(result.stderr)")
        XCTAssertTrue(
            result.stderr.contains("hybridKernelKeyHeadDimUnaligned(33)"),
            "unexpected stderr: \(result.stderr)")
    }

    // MARK: - Black-box CLI coverage: `resolveServingLimits`'s pre-load fit-composition block
    // (`--qwen4exp-mtp` / `--ngram-offload-plan`), driven through the real binary rather than
    // re-implemented locally — see the module doc comments on `InCheckpointMTPFitComposition` and
    // `NGramOffloadFitComposition` for the formulas these tests independently reconstruct.

    /// Builds the 48-layer `layer_types` array a `qwen4_exp`/`qwen4_exp_text` config carries: 12
    /// repetitions of [linear_attention, linear_attention, linear_attention, full_attention] -> 12
    /// growing full-attention layers, 36 linear — the same shape `ModelConfigDecoderTests`'s own
    /// qwen4_exp fixtures use, confirmed there to decode to `nAttnLayers == 12`.
    private func qwen4ExpLayerTypesJSON() -> String {
        var layerTypes: [String] = []
        for _ in 0..<12 {
            layerTypes += ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        }
        return "[" + layerTypes.map { "\"\($0)\"" }.joined(separator: ",") + "]"
    }

    /// A minimal flat (root `model_type`) `qwen4_exp_text` `config.json` that decodes successfully
    /// via `ModelConfigDecoder.decodeModelDirectory` — the field set mirrors
    /// `ModelConfigDecoderTests`'s own equivalent fixture: a 12/36 hybrid-linear split plus the
    /// sparse-indexer aux term (`indexer_head_dim`) the qwen4_exp family requires to decode at all.
    private func qwen4ExpConfigJSON() -> String {
        """
        {
          "model_type": "qwen4_exp_text",
          "num_hidden_layers": 48,
          "num_attention_heads": 16,
          "num_key_value_heads": 2,
          "head_dim": 256,
          "hidden_size": 4096,
          "max_position_embeddings": 262144,
          "full_attention_interval": 4,
          "layer_types": \(qwen4ExpLayerTypesJSON()),
          "linear_num_key_heads": 16,
          "linear_num_value_heads": 48,
          "linear_key_head_dim": 128,
          "linear_value_head_dim": 128,
          "linear_conv_kernel_dim": 4,
          "indexer_head_dim": 128,
          "indexer_kv_heads": 1
        }
        """
    }

    /// Writes a minimal, well-formed safetensors container at `url`: an 8-byte little-endian header
    /// length, the header JSON itself (one entry per `tensors`, sequential non-overlapping
    /// `data_offsets`), then a zero-filled data blob exactly covering the declared offsets — the
    /// same three-part layout `NGramOffloadFitComposition.readSafetensorsTensorByteSizes` parses.
    /// Declared `dtype`/`shape` are cosmetic (the composition only trusts `data_offsets` for byte
    /// sizing, never the blob or the declared shape). Returns the resulting file's exact byte
    /// count, so a caller knows the real "whole-file total" the production composition will
    /// independently re-measure, without reading it back off the binary's own output.
    @discardableResult
    private func writeSafetensorsShard(
        at url: URL, tensors: [(key: String, byteSize: Int)]
    ) throws -> Int {
        var header: [String: Any] = [:]
        var offset = 0
        for tensor in tensors {
            header[tensor.key] = [
                "dtype": "F32",
                "shape": [max(1, tensor.byteSize / 4)],
                "data_offsets": [offset, offset + tensor.byteSize],
            ]
            offset += tensor.byteSize
        }
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var headerLength = UInt64(headerData.count)
        var lengthBytes = [UInt8](repeating: 0, count: 8)
        for index in 0..<8 {
            lengthBytes[index] = UInt8(headerLength & 0xff)
            headerLength >>= 8
        }
        var file = Data(lengthBytes)
        file.append(headerData)
        file.append(Data(count: offset))
        try file.write(to: url)
        return file.count
    }

    /// Writes an offload plan file carrying only the one field `NGramOffloadFitComposition
    /// .planMaxResidentBytes` reads.
    private func writeNGramOffloadPlanFile(at url: URL, maxResidentBytes: Int) throws {
        let json = #"{"limits":{"maxResidentBytes":\#(maxResidentBytes)}}"#
        try Data(json.utf8).write(to: url)
    }

    /// Synthetic `qwen4_exp_text` checkpoint directory carrying exactly one real offloaded n-gram
    /// PLE shard pair (`shard_0`/`shard_1`, dense `weight`-only fields, layer 0) under the bare
    /// `"model."` text-module prefix `NGramOffloadFitComposition.matchOffloadedNGramShardKey`
    /// recognises — satisfying its structural audit (single layer index, contiguous `0..<2` shard
    /// run, identical single-field shard sets). The two tensor byte sizes are caller-chosen and
    /// therefore known exactly to the test, and the returned whole-file total is the REAL on-disk
    /// size of the one shard file written (`ModelConfigDecoder.sumSafetensorsBytes` measures it
    /// directly, and the accompanying index.json declares the identical total so the measured path
    /// is used unambiguously) — so a test can independently reconstruct every number the production
    /// composition is expected to print.
    private func writeOffloadShardCheckpointDirectory(
        shard0Bytes: Int, shard1Bytes: Int
    ) throws -> (directory: URL, wholeFileTotal: Int) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ngram-offload-cli-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(qwen4ExpConfigJSON().utf8).write(to: directory.appendingPathComponent("config.json"))
        let shardURL = directory.appendingPathComponent("model-00001-of-00001.safetensors")
        let wholeFileTotal = try writeSafetensorsShard(
            at: shardURL,
            tensors: [
                ("model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight", shard0Bytes),
                ("model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight", shard1Bytes),
            ])
        let indexJSON = #"{"metadata":{"total_size":\#(wholeFileTotal)}}"#
        try Data(indexJSON.utf8).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        return (directory, wholeFileTotal)
    }

    /// Synthetic `qwen4_exp_text` checkpoint directory with NO `*.safetensors` shard on disk at
    /// all — only a declared `model.safetensors.index.json` total
    /// (`ModelConfigDecoder.decodeModelDirectory` accepts a declared-only checkpoint,
    /// `weightsAreDeclared = true`, when no real shard is present). `NGramOffloadFitComposition`
    /// then scans zero shard files, matches zero offloaded n-gram shard keys, and throws
    /// `.noMatchingOffloadedShardKeys` — a realistic "wrong/missing checkpoint shape" failure
    /// mode, not a contrived one, and one that needs no real tensor content at all: the declared
    /// total alone is what the subsequent fit decision must still size honestly.
    private func writeDeclaredOnlyCheckpointDirectory(declaredTotalBytes: Int) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "ngram-offload-declared-only-cli-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(qwen4ExpConfigJSON().utf8).write(to: directory.appendingPathComponent("config.json"))
        let indexJSON = #"{"metadata":{"total_size":\#(declaredTotalBytes)}}"#
        try Data(indexJSON.utf8).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    /// Extracts the two integers from a `"qwen4exp mtp fit adjustment: nAttnLayers <N> -> <M> ..."`
    /// stderr line (`nil` when no such line is present), so a test asserts on the ACTUAL numbers the
    /// binary printed rather than embedding both sides of the comparison as authored literals.
    private func extractedMTPFitAdjustment(from stderr: String) -> (before: Int, after: Int)? {
        guard let markerRange = stderr.range(of: "nAttnLayers ") else { return nil }
        let scanner = Scanner(string: String(stderr[markerRange.upperBound...]))
        guard let before = scanner.scanInt() else { return nil }
        guard scanner.scanString("->") != nil else { return nil }
        guard let after = scanner.scanInt() else { return nil }
        return (before, after)
    }

    /// Acceptance A (decisive): the `--ngram-offload-plan` fit correction in
    /// `resolveServingLimits` is really applied by the real serve binary, and the four numbers it
    /// prints on stderr are exactly the ones this test independently computed from its own fixture
    /// (never scraped from the binary's own output). Every quantity is asserted separately per
    /// the project's separation-of-quantities discipline; the anti-vacuity check
    /// (`assertLessThan`, not `!=`) proves the correction actually shrinks the resident figure.
    func testServeCLINGramOffloadFitAdjustmentAppliesIndependentlyComputedNumbers() throws {
        let (directory, wholeFileTotal) = try writeOffloadShardCheckpointDirectory(
            shard0Bytes: 4096, shard1Bytes: 6144)
        defer { try? FileManager.default.removeItem(at: directory) }
        let offloadedBytes = 4096 + 6144
        let residencyBudget = 3072
        let adjustedWeights = wholeFileTotal - offloadedBytes + residencyBudget

        let planURL = directory.appendingPathComponent("ngram-offload-plan.json")
        try writeNGramOffloadPlanFile(at: planURL, maxResidentBytes: residencyBudget)

        let result = try runServe(arguments: [
            "--model", "qwen4-exp-cli-fixture-offload",
            "--model-path", directory.path,
            "--ngram-offload-plan", planURL.path,
            "--host", "127.0.0.1",
            "--port", "58733",
        ])

        XCTAssertTrue(
            result.stderr.contains("ngram offload fit adjustment: whole-file total \(wholeFileTotal) B"),
            "unexpected stderr: \(result.stderr)")
        XCTAssertTrue(
            result.stderr.contains("offloaded n-gram bytes \(offloadedBytes) B"),
            "unexpected stderr: \(result.stderr)")
        XCTAssertTrue(
            result.stderr.contains("row-store residency budget \(residencyBudget) B"),
            "unexpected stderr: \(result.stderr)")
        XCTAssertTrue(
            result.stderr.contains("adjusted weights \(adjustedWeights) B"),
            "unexpected stderr: \(result.stderr)")
        XCTAssertLessThan(
            adjustedWeights, wholeFileTotal,
            "an offload correction must always reduce the resident-weights figure, never grow it")
    }

    /// Acceptance B: when the `--ngram-offload-plan` composition FAILS (here: a checkpoint with no
    /// matching offloaded n-gram shard keys on disk at all, a realistic "wrong artifact" shape), the
    /// load-bearing `catch` in `resolveServingLimits` must keep sizing the conservative full-resident
    /// figure rather than silently reducing it. Proven two ways: the `could not be computed` fallback
    /// line names the exact failure this fixture triggers, and the subsequent fit-decision summary's
    /// `weights=` figure equals the UNADJUSTED declared total (computed independently here with the
    /// same `%.2f GiB` formula `ServingFitDecision.summaryLines()` uses), never anything smaller.
    func testServeCLINGramOffloadFitAdjustmentFailureKeepsFullResidentFigure() throws {
        let declaredTotal = 5_368_709_120  // exactly 5 GiB — a round figure under the same
        // formatter the summary line uses, so the expected substring is exact, not approximate.
        let directory = try writeDeclaredOnlyCheckpointDirectory(
            declaredTotalBytes: declaredTotal)
        defer { try? FileManager.default.removeItem(at: directory) }
        let planURL = directory.appendingPathComponent("ngram-offload-plan.json")
        try writeNGramOffloadPlanFile(at: planURL, maxResidentBytes: 1024)

        let result = try runServe(arguments: [
            "--model", "qwen4-exp-cli-fixture-declared-only",
            "--model-path", directory.path,
            "--ngram-offload-plan", planURL.path,
            "--host", "127.0.0.1",
            "--port", "58734",
        ])

        XCTAssertTrue(
            result.stderr.contains(
                "ngram offload fit adjustment could not be computed (noMatchingOffloadedShardKeys); "
                    + "sizing the full-resident checkpoint instead"),
            "unexpected stderr: \(result.stderr)")

        let expectedGiB = String(format: "%.2f GiB", Double(declaredTotal) / 1_073_741_824.0)
        XCTAssertTrue(
            result.stderr.contains("weights=\(expectedGiB) (declared)"),
            "the fit decision must size the UNADJUSTED declared total when the offload composition "
                + "fails, proving the failure did not reduce the resident-weights figure; "
                + "unexpected stderr: \(result.stderr)")
    }

    /// Acceptance C: `--qwen4exp-mtp` and `--ngram-offload-plan` BOTH run in one invocation (the
    /// parser requires the plan whenever the MTP flag is present) rather than either composition
    /// silently winning. The MTP line's `nAttnLayers <N> -> <M>` is parsed from the REAL stderr
    /// output (not both sides of the comparison hardcoded) and asserted `M == N + 1` in addition to
    /// matching this fixture's known 12/13 split; the offload line from the SAME run is asserted
    /// with the identical independently-computed numbers Acceptance A uses.
    ///
    /// Scope, stated honestly: this proves CO-OCCURRENCE, not composition ORDER. A mutation that
    /// swaps the two blocks in `resolveServingLimits` leaves every assertion here passing, and that
    /// is correct rather than a gap in the test — the two compositions touch disjoint profile
    /// fields (the MTP one rewrites only `nAttnLayers`; the offload one reads and rewrites only
    /// `weightsBytes4bitEstimate`), so neither can observe the other's edit and the order is not an
    /// observable property today. If a future composition ever reads a field its sibling writes,
    /// order becomes real and this test must be extended to discriminate it.
    func testServeCLIMTPAndNGramOffloadFitAdjustmentsBothRunInOneInvocation() throws {
        let (directory, wholeFileTotal) = try writeOffloadShardCheckpointDirectory(
            shard0Bytes: 2048, shard1Bytes: 2048)
        defer { try? FileManager.default.removeItem(at: directory) }
        let offloadedBytes = 2048 + 2048
        let residencyBudget = 4096
        let adjustedWeights = wholeFileTotal - offloadedBytes + residencyBudget

        let planURL = directory.appendingPathComponent("ngram-offload-plan.json")
        try writeNGramOffloadPlanFile(at: planURL, maxResidentBytes: residencyBudget)

        let result = try runServe(arguments: [
            "--model", "qwen4-exp-cli-fixture-mtp-stack",
            "--model-path", directory.path,
            "--qwen4exp-mtp",
            "--ngram-offload-plan", planURL.path,
            "--host", "127.0.0.1",
            "--port", "58735",
        ])

        guard let mtpAdjustment = extractedMTPFitAdjustment(from: result.stderr) else {
            XCTFail("no qwen4exp mtp fit adjustment line found in stderr: \(result.stderr)")
            return
        }
        XCTAssertEqual(mtpAdjustment.before, 12, "unexpected stderr: \(result.stderr)")
        XCTAssertEqual(mtpAdjustment.after, 13, "unexpected stderr: \(result.stderr)")
        XCTAssertGreaterThan(
            mtpAdjustment.after, mtpAdjustment.before,
            "the MTP drafter's additional cache layer must strictly grow nAttnLayers")

        XCTAssertTrue(
            result.stderr.contains("ngram offload fit adjustment: whole-file total \(wholeFileTotal) B"),
            "the offload composition must ALSO run in the same invocation, proving neither "
                + "composition silently wins; unexpected stderr: \(result.stderr)")
        XCTAssertTrue(
            result.stderr.contains("offloaded n-gram bytes \(offloadedBytes) B"),
            "unexpected stderr: \(result.stderr)")
        XCTAssertTrue(
            result.stderr.contains("row-store residency budget \(residencyBudget) B"),
            "unexpected stderr: \(result.stderr)")
        XCTAssertTrue(
            result.stderr.contains("adjusted weights \(adjustedWeights) B"),
            "unexpected stderr: \(result.stderr)")
    }
}

private enum FixtureTokenizerError: Error {
    case unexpectedMessages
}

private struct FixtureTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        []
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let pieces = [
            1: "hel",
            2: "lo",
            3: "\n",
        ]
        return tokenIds.compactMap { pieces[$0] }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        [
            "<eos>": 8,
            "<turn>": 9,
            "<unk>": 10,
        ][token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        nil
    }

    var bosToken: String? { "<bos>" }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { "<unk>" }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let expected = [
            ("developer", "developer text"),
            ("system", "system text"),
            ("user", "user text"),
            ("assistant", "assistant text"),
        ]
        guard messages.count == expected.count,
            tools == nil,
            additionalContext == nil
        else {
            throw FixtureTokenizerError.unexpectedMessages
        }
        for (message, expected) in zip(messages, expected) {
            guard message["role"] as? String == expected.0,
                message["content"] as? String == expected.1
            else {
                throw FixtureTokenizerError.unexpectedMessages
            }
        }
        return [41, 42]
    }
}

private struct ResetSensitiveDecoder: Decoder {
    private var resetCount = 0

    mutating func prefill(_ promptTokens: [Int]) -> Int {
        resetCount < 3 ? 1 : 2
    }

    mutating func step(last: Int) -> Int {
        99
    }

    mutating func reset() {
        resetCount += 1
    }
}

/// Minimal `LanguageModel` conforming to `UnsupportedInputTokenReporting`, reporting a fixed set of
/// input token IDs its (fake) forward path would otherwise `preconditionFailure` on. Mirrors the
/// minimal-fake shape of `CacheFactorySpyModel` (SpikeCoreTests/MLXDecoderCacheFactoryTests.swift):
/// only the two required methods are implemented — the rest of `LanguageModel` has default
/// implementations via protocol extension, and nothing here ever runs a real forward pass.
private final class FixtureUnsupportedTokenReportingModel: Module, LanguageModel,
    UnsupportedInputTokenReporting
{
    let unsupportedInputTokenIDs: Set<Int64>

    init(unsupportedInputTokenIDs: Set<Int64>) {
        self.unsupportedInputTokenIDs = unsupportedInputTokenIDs
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([inputs.dim(0), inputs.dim(1), 8])
    }
}

/// A `LanguageModel` that does NOT conform to `UnsupportedInputTokenReporting` — every family
/// today except the ones that opt in — used to prove `servingRejectedPromptTokenIDs` stays a
/// no-op (empty set) for the unchanged-behavior case.
private final class FixtureNonReportingModel: Module, LanguageModel {
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([inputs.dim(0), inputs.dim(1), 8])
    }
}

/// Minimal `KVCache` conforming to `ServingCacheKindReporting`, with an injectable
/// `servingCacheLayerKind`. Never runs a real forward pass; every method is a stub. Mirrors the
/// minimal-fake shape of `FixtureUnsupportedTokenReportingModel` above.
private final class MarkerReportingFakeCache: KVCache, ServingCacheKindReporting {
    var offset: Int = 0
    var maxSize: Int? { nil }
    var state: [MLXArray] = []
    var metaState: [String] = [""]
    var isTrimmable: Bool { false }
    let servingCacheLayerKind: ServingCacheLayerKind

    init(servingCacheLayerKind: ServingCacheLayerKind) {
        self.servingCacheLayerKind = servingCacheLayerKind
    }

    func innerState() -> [MLXArray] { [] }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        MarkerReportingFakeCache(servingCacheLayerKind: servingCacheLayerKind)
    }
}

/// A concrete `KVCacheSimple` instance that ALSO conforms to `ServingCacheKindReporting`,
/// reporting a DIFFERENT kind (`.rotatingAttention`) than its true concrete-type classification
/// (`.denseAttention`). `KVCacheSimple` is `public` but not `open`, so it cannot be subclassed
/// outside its defining module; this retroactive conformance (test-target-scoped, added ONLY to
/// prove the ordering guarantee below) is the only available way to construct a value that is
/// simultaneously a real concrete-type match and a marker-protocol conformer. It is safe for every
/// OTHER test that constructs a bare `KVCacheSimple()`: the classifier always checks the
/// concrete-type switch first (that is precisely the guarantee under test), so this added
/// conformance changes no other test's observed classification.
///
/// Exists only to make the classifier's "concrete type wins" ordering guarantee observable in a
/// test: none of the five real concrete types conform to the marker protocol today, so without
/// this fixture a reordering of the probe and the concrete-type switch would be undetectable.
extension KVCacheSimple: ServingCacheKindReporting {
    public var servingCacheLayerKind: ServingCacheLayerKind { .rotatingAttention }
}

/// Minimal `KVCache` that conforms to neither one of the five concrete types the classifier
/// recognizes NOR `ServingCacheKindReporting` — the classifier's fully-unrecognized fallthrough
/// case.
private final class NonReportingFakeCache: KVCache {
    var offset: Int = 0
    var maxSize: Int? { nil }
    var state: [MLXArray] = []
    var metaState: [String] = [""]
    var isTrimmable: Bool { false }

    func innerState() -> [MLXArray] { [] }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        NonReportingFakeCache()
    }
}
