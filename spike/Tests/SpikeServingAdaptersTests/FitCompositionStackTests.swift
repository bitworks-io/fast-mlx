import XCTest

import HarnessCore
@testable import SpikeServingAdapters

/// Verifies that `InCheckpointMTPFitComposition` (the +1 `nAttnLayers` correction for the
/// in-checkpoint MTP drafter's own `.qsa` layer) and `NGramOffloadFitComposition` (the whole-file-
/// minus-offloaded-plus-residency-budget correction for `--ngram-offload-plan`) STACK correctly when
/// applied together, which is the real deploy shape: `--qwen4exp-mtp` requires
/// `--ngram-offload-plan` at argument-parse time (`FastMLXServeArguments`), so a production serve
/// always composes both, never either alone. Neither composition's own test file exercises the
/// other, so this file closes that gap.
///
/// `resolveServingLimits` (`fastmlx-serve/FastMLXServe.swift`) applies the MTP composition FIRST and
/// the n-gram offload composition SECOND, specifically so the offload composition's field-by-field
/// rebuild (it copies every field of `base.profile` verbatim except the weights figure) carries the
/// MTP-corrected `nAttnLayers` (13) through unchanged rather than silently reading back the
/// uncorrected catalog/decoded value (12). These tests reproduce that same two-call sequence
/// directly against the two composition types — `fastmlx-serve` is an executable target with no test
/// target, so `resolveServingLimits` itself cannot be invoked from a test. Consequently, deleting the
/// serve-side composition-ordering block in `FastMLXServe.swift` would NOT turn any test in this file
/// red: what is pinned here is the CONTRACT between the two composition types (that stacking them
/// preserves both corrections, and that swapping their order does not change the resulting geometry),
/// not the specific call site that currently exercises that contract in production.
final class FitCompositionStackTests: XCTestCase {

    // MARK: - Fixture plumbing (copied from `NGramOffloadFitCompositionTests`; kept minimal/private
    // to this file rather than shared, matching that file's own no-shared-test-helpers convention)

    private func makeTempDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    @discardableResult
    private func writeSafetensorsFixture(
        at fileURL: URL,
        tensors: [(name: String, byteLength: Int)],
        includeMetadata: Bool = false,
        actualDataByteCount: Int? = nil,
        blobFillByte: UInt8 = 0
    ) throws -> Int {
        var offset = 0
        var header: [String: Any] = [:]
        if includeMetadata {
            header["__metadata__"] = ["format": "pt"]
        }
        for (name, length) in tensors {
            header[name] = [
                "dtype": "F32",
                "shape": [length],
                "data_offsets": [offset, offset + length],
            ]
            offset += length
        }
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var lengthBytes = [UInt8](repeating: 0, count: 8)
        var n = UInt64(headerData.count)
        for index in 0..<8 {
            lengthBytes[index] = UInt8(n & 0xFF)
            n >>= 8
        }
        var fileData = Data(lengthBytes)
        fileData.append(headerData)
        fileData.append(Data(repeating: blobFillByte, count: actualDataByteCount ?? offset))
        try fileData.write(to: fileURL)
        return headerData.count
    }

    private func writePlanFile(
        at fileURL: URL, maxResidentBytes: Any? = 12345, includeLimits: Bool = true
    ) throws {
        var json: [String: Any] = [:]
        if includeLimits {
            var limits: [String: Any] = [
                "maxResidentRows": 10,
                "maxRequestRows": 5,
                "maxInFlightBytes": 100,
            ]
            if let maxResidentBytes { limits["maxResidentBytes"] = maxResidentBytes }
            json["limits"] = limits
        }
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: fileURL)
    }

    /// `nAttnLayers: 12`, `nKVHeads: 2`, `headDim: 256`, `auxPerLayerKeyDim: 128` mirror the real
    /// `Qwen3.8-Flash-Next` catalog geometry's shape (`ModelArchProfile.catalog`), but `vHeadDim: 29`
    /// (copied from `NGramOffloadFitCompositionTests`'s own full-fielded fixture, to exercise every
    /// field rather than leaving several `nil`) makes this fixture's K/V ASYMMETRIC, unlike the real
    /// catalog entry (which is symmetric, `vHeadDim: nil`). Consequently `kvHeadDimSum = headDim +
    /// vHeadDim = 256 + 29 = 285` here, NOT the catalog's `2 * headDim = 512` — every `kvBytesPerToken`
    /// expectation below is computed from THIS fixture's own 285, not from the catalog's documented
    /// 27,648 B/tok figure (which assumes the symmetric case and does not apply to this fixture).
    private func fullFieldedProfile(weightsBytes: Int) -> ModelArchProfile {
        ModelArchProfile(
            id: "fixture-flash-next",
            modelType: .hybridLinear,
            nLayers: 48,
            nAttnLayers: 12,
            nKVHeads: 2,
            headDim: 256,
            slidingWindow: 4096,
            fixedStateBytes: 115_458_048,
            nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: weightsBytes,
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
    }

    private func parsed(
        weightsBytes: Int, measured: Bool = true, declared: Bool = false, quantBits: Int? = 4
    ) -> ParsedModelArch {
        ParsedModelArch(
            profile: fullFieldedProfile(weightsBytes: weightsBytes),
            weightsAreMeasured: measured, weightsAreDeclared: declared, quantBits: quantBits)
    }

    /// Rewraps a bare `ModelArchProfile` (as `InCheckpointMTPFitComposition.make` returns) back into
    /// a `ParsedModelArch`, carrying the provenance flags forward — exactly what `resolveServingLimits`
    /// does around its own call to that composition before handing the result to the offload
    /// composition (which requires a `ParsedModelArch`, not a bare profile).
    private func rewrapped(_ profile: ModelArchProfile, from source: ParsedModelArch) -> ParsedModelArch {
        ParsedModelArch(
            profile: profile,
            weightsAreMeasured: source.weightsAreMeasured,
            weightsAreDeclared: source.weightsAreDeclared,
            quantBits: source.quantBits)
    }

    // MARK: - Fixture constants shared by the stacking tests below (documented once, reused by name)

    /// Whole-file resident-weights total BEFORE either composition runs.
    private static let wholeFileTotal = 1_000_000
    /// Bytes the offloaded n-gram shard declares (the safetensors fixture below is built to match
    /// this exactly, so `offloadedNGramTensorBytes` measures precisely this many bytes).
    private static let offloadedBytes = 300_000
    /// The row-store residency budget the plan file declares (`limits.maxResidentBytes`).
    private static let residencyBudget = 50_000
    /// `wholeFileTotal - offloadedBytes + residencyBudget`, computed here from the SAME constants
    /// above rather than by re-invoking `NGramOffloadFitComposition.make` a second time — an
    /// independent expectation, not a tautology.
    private static let expectedAdjustedWeights = wholeFileTotal - offloadedBytes + residencyBudget  // 750,000

    private func writeStandardOffloadFixtures(in directory: URL, planURL: URL) throws {
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"),
            tensors: [
                (
                    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    Self.offloadedBytes
                )
            ])
        try writePlanFile(at: planURL, maxResidentBytes: Self.residencyBudget)
    }

    // MARK: - 1. Stacking preserves both corrections

    /// Reproduces `resolveServingLimits`'s real order (MTP first, offload second) and asserts BOTH
    /// corrections survive in the final profile: the MTP `nAttnLayers` bump is not overwritten by the
    /// offload composition's field-by-field rebuild, and the offload weights adjustment is not
    /// dropped by having gone through the MTP step first.
    func testStackedCompositionsPreserveBothCorrections() throws {
        let directory = try makeTempDirectory()
        let planURL = directory.appendingPathComponent("plan.json")
        try writeStandardOffloadFixtures(in: directory, planURL: planURL)

        let base = parsed(weightsBytes: Self.wholeFileTotal)

        // Step 1 (as in `resolveServingLimits`): MTP composition, pure and non-throwing.
        let mtpProfile = InCheckpointMTPFitComposition.make(base: base.profile)
        let afterMTP = rewrapped(mtpProfile, from: base)

        // Step 2: n-gram offload composition, on top of the MTP-corrected profile.
        //
        // ANTI-VACUITY: `NGramOffloadFitComposition.make` THROWS on a malformed fixture (wrong key
        // prefix, non-contiguous shards, a plan missing `limits`, ...), and the production call site
        // in `FastMLXServe.swift` SWALLOWS that throw and falls back to the unreduced whole-file
        // figure (see its `catch` block's comment: "keep the conservative full-resident `parsed`").
        // That means a test built on a subtly-wrong fixture would not fail loudly here either — it
        // would silently prove nothing about the stacking behavior this test exists to pin. Using
        // `try` (not `try?`) below ensures a thrown error fails THIS test, and the exact-value
        // assertions afterward (rather than merely `XCTAssertNoThrow`) ensure the composition did not
        // just "succeed" on some other, wrong, reduction.
        let final = try NGramOffloadFitComposition.make(
            base: afterMTP, modelDirectory: directory, planFileURL: planURL)

        // The MTP correction survived the offload composition's field-by-field rebuild.
        XCTAssertEqual(final.profile.nAttnLayers, 13)

        // The offload correction was actually applied: strictly less than the base whole-file total,
        // and equal to an independently computed expectation (own fixture constants, not a re-call).
        XCTAssertLessThan(final.profile.weightsBytes4bitEstimate, Self.wholeFileTotal)
        XCTAssertEqual(Self.expectedAdjustedWeights, 750_000)
        XCTAssertEqual(final.profile.weightsBytes4bitEstimate, Self.expectedAdjustedWeights)
        XCTAssertGreaterThan(Self.wholeFileTotal - final.profile.weightsBytes4bitEstimate, 0)

        // `CapacityModel.kvBytesPerToken` at the final (13-layer) geometry, independently computed
        // from the fixture's own nKVHeads(2)/kvHeadDimSum(256+29=285)/auxPerLayerKeyDim(128) rather
        // than by trusting the production call: term1 = 13 * 2 * 285 * 2.0 = 14,820; term2 =
        // 13 * 128 * 2.0 = 3,328; total = 18,148.
        let expectedKVBytesPerToken13Layers = 18_148.0
        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(final.profile, kvQuant: .fp16),
            expectedKVBytesPerToken13Layers)
    }

    // MARK: - 2. Order independence

    /// Applies the two compositions in BOTH orders against IDENTICAL fixtures (the same temp
    /// directory and plan file — reading them is non-destructive, so both orders can safely read the
    /// same files) and asserts the two resulting profiles agree on every field except `id`.
    ///
    /// This holds for a documented structural reason, not by coincidence: `InCheckpointMTPFitComposition
    /// .make` touches ONLY `nAttnLayers` (leaving `weightsBytes4bitEstimate` untouched, per its own doc
    /// comment — the MTP drafter's weights are already inside the measured whole-file total) and
    /// `NGramOffloadFitComposition.make` touches ONLY `weightsBytes4bitEstimate` (copying
    /// `nAttnLayers` straight from `base.profile`, whatever it already is). The two compositions
    /// correct DISJOINT fields, so composing them in either order lands on the same final profile.
    /// If a future edit made either composition start reading from a shared catalog/lookup instead of
    /// `base.profile` (the hazard both compositions' doc comments warn about), or made one touch a
    /// field the other also touches, this test would start failing — see test 3 below for the
    /// narrower guard on exactly that hazard.
    func testCompositionOrderDoesNotChangeTheResultingGeometry() throws {
        let directory = try makeTempDirectory()
        let planURL = directory.appendingPathComponent("plan.json")
        try writeStandardOffloadFixtures(in: directory, planURL: planURL)

        let base = parsed(weightsBytes: Self.wholeFileTotal)

        // Order A: MTP, then offload (the real `resolveServingLimits` order).
        let mtpFirstProfile = InCheckpointMTPFitComposition.make(base: base.profile)
        let mtpFirstWrapped = rewrapped(mtpFirstProfile, from: base)
        let orderA = try NGramOffloadFitComposition.make(
            base: mtpFirstWrapped, modelDirectory: directory, planFileURL: planURL)

        // Order B: offload, then MTP (the reverse).
        let offloadFirst = try NGramOffloadFitComposition.make(
            base: base, modelDirectory: directory, planFileURL: planURL)
        let orderBProfile = InCheckpointMTPFitComposition.make(base: offloadFirst.profile)

        let a = orderA.profile
        let b = orderBProfile

        // Every field except `id` (which both compositions deliberately relabel to name themselves,
        // so the two orders produce different — and expected-to-differ — id suffixes).
        XCTAssertEqual(a.nAttnLayers, b.nAttnLayers)
        XCTAssertEqual(a.nAttnLayers, 13)
        XCTAssertEqual(a.nKVHeads, b.nKVHeads)
        XCTAssertEqual(a.headDim, b.headDim)
        XCTAssertEqual(a.nLayers, b.nLayers)
        XCTAssertEqual(a.modelType, b.modelType)
        XCTAssertEqual(a.slidingWindow, b.slidingWindow)
        XCTAssertEqual(a.fixedStateBytes, b.fixedStateBytes)
        XCTAssertEqual(a.nativeMaxContext, b.nativeMaxContext)
        XCTAssertEqual(a.weightsBytes4bitEstimate, b.weightsBytes4bitEstimate)
        XCTAssertEqual(a.weightsBytes4bitEstimate, Self.expectedAdjustedWeights)
        XCTAssertEqual(a.license, b.license)
        XCTAssertEqual(a.mlaHeads, b.mlaHeads)
        XCTAssertEqual(a.mlaRopeDim, b.mlaRopeDim)
        XCTAssertEqual(a.mlaNopeDim, b.mlaNopeDim)
        XCTAssertEqual(a.mlaVDim, b.mlaVDim)
        XCTAssertEqual(a.swaKVHeads, b.swaKVHeads)
        XCTAssertEqual(a.swaHeadDim, b.swaHeadDim)
        XCTAssertEqual(a.vHeadDim, b.vHeadDim)
        XCTAssertEqual(a.swaVHeadDim, b.swaVHeadDim)
        XCTAssertEqual(a.auxPerLayerKeyDim, b.auxPerLayerKeyDim)

        XCTAssertEqual(
            CapacityModel.kvBytesPerToken(a, kvQuant: .fp16),
            CapacityModel.kvBytesPerToken(b, kvQuant: .fp16))
    }

    // MARK: - 3. The stacking test above is discriminating

    /// Guards that test 1's `nAttnLayers == 13` assertion is not vacuously true — i.e. that dropping
    /// the MTP composition from the stack is DETECTABLE via `kvBytesPerToken`, not merely via the raw
    /// `nAttnLayers` field a future refactor might stop reading. This is exactly the shape of defect
    /// the offload composition's own doc comment warns about: "a field-by-field rebuild that drops
    /// one ... silently under-counts a term a later reader has no way to notice from this call site
    /// alone." If a future edit made `NGramOffloadFitComposition.make` rebuild `nAttnLayers` from a
    /// catalog/lookup instead of copying it from `base.profile`, the MTP correction would be silently
    /// discarded by the offload step — and THIS test (not test 1, which would still show the correct
    /// final number by coincidence only if the catalog also said 13) is the one that pins the
    /// PER-LAYER delta as the reason `nAttnLayers` must be 13, not just that it happens to be 13.
    func testDroppingTheMTPCorrectionIsDetectable() throws {
        let directory = try makeTempDirectory()
        let planURL = directory.appendingPathComponent("plan.json")
        try writeStandardOffloadFixtures(in: directory, planURL: planURL)

        let base = parsed(weightsBytes: Self.wholeFileTotal)

        // The real stack: MTP then offload -> nAttnLayers 13.
        let mtpProfile = InCheckpointMTPFitComposition.make(base: base.profile)
        let withMTP = try NGramOffloadFitComposition.make(
            base: rewrapped(mtpProfile, from: base), modelDirectory: directory, planFileURL: planURL)

        // The same stack WITHOUT the MTP composition -> nAttnLayers stays at the base's 12.
        let withoutMTP = try NGramOffloadFitComposition.make(
            base: base, modelDirectory: directory, planFileURL: planURL)

        let kvWith = CapacityModel.kvBytesPerToken(withMTP.profile, kvQuant: .fp16)
        let kvWithout = CapacityModel.kvBytesPerToken(withoutMTP.profile, kvQuant: .fp16)
        XCTAssertGreaterThan(kvWith, kvWithout)

        // The per-token delta for exactly one extra attention layer: the standard K+V term
        // (nKVHeads * kvHeadDimSum * bpe = 2 * 285 * 2.0 = 1,140) plus the auxPerLayerKeyDim
        // indexer term (auxPerLayerKeyDim * 2.0 = 128 * 2.0 = 256) = 1,396.
        let expectedPerLayerDelta = 1_396.0
        XCTAssertEqual(kvWith - kvWithout, expectedPerLayerDelta)

        // Sanity-pin the absolute figures too (reconciles with test 1's 13-layer total, 18,148 B/tok,
        // for this fixture's asymmetric kvHeadDimSum = 285 — see `fullFieldedProfile`'s doc comment;
        // this is NOT the real catalog's symmetric 27,648 B/tok figure).
        XCTAssertEqual(kvWithout, 16_752.0)
        XCTAssertEqual(kvWith, 18_148.0)
    }

    // MARK: - 4. Concurrency-divergence: the correction only bites under concurrency

    /// At the deploy's DEFAULT concurrency (1), the MTP correction does not move the context ceiling
    /// on ANY of the three hosts below — so an "unchanged ceiling" assertion at concurrency 1 alone
    /// would pass identically with `InCheckpointMTPFitComposition` deleted (see
    /// `testUnchangedCeilingAtConcurrencyOneIsInertByItself` below, which pins exactly that). This
    /// test pins the DIVERGENCE POINT instead: the first concurrency at which the 12-layer
    /// (uncomposed) and 13-layer (MTP-composed) profiles disagree on `CapacityModel.contextCeiling`,
    /// per host. Every value below was computed independently from `CapacityModel`'s own formulas
    /// (`kvBytesPerToken`, `kvBytesForContext`, `transientPrefillPeakBytes`, `classify`'s binary
    /// search in `contextCeiling`) and cross-checked against a real `swift test` run — not derived
    /// from this test's own expectations.
    ///
    /// Uses the REAL Flash Next geometry (symmetric, `vHeadDim: nil`, `kvHeadDimSum = 2*headDim =
    /// 512`) — NOT `fullFieldedProfile`'s asymmetric `vHeadDim: 29` fixture, whose numbers do not
    /// reconcile with the catalog's own documented 27,648 B/tok figure.
    private func realFlashNextGeometry(nAttnLayers: Int, weightsBytes: Int) -> ModelArchProfile {
        ModelArchProfile(
            id: "flash-next-real-geometry",
            modelType: .hybridLinear,
            nLayers: 48,
            nAttnLayers: nAttnLayers,
            nKVHeads: 2,
            headDim: 256,
            fixedStateBytes: 115_458_048,
            nativeMaxContext: 262_144,
            weightsBytes4bitEstimate: weightsBytes,
            license: "fixture-license",
            auxPerLayerKeyDim: 128)
    }

    /// `ModelArchProfile.catalog`'s "Qwen3.8-Flash-Next (n-gram offload)" entry's weights figure: the
    /// measured whole-file total (113,324,747,928 B) minus the 32,000,153,600 B (29.80 GiB) n-gram
    /// embedding table that `--ngram-offload-plan` SSD-streams instead of holding resident. The
    /// deploy this file models always passes `--ngram-offload-plan` (it is required alongside
    /// `--qwen4exp-mtp` at argument-parse time — see this file's own top-of-file doc comment), so
    /// this offload-adjusted figure, not the base 113,324,747,928 B total, is the real operating
    /// weights figure for every row below.
    private static let offloadAdjustedWeightsBytes = 81_324_594_328

    /// `SystemProfile.m5Max128` (128 GiB RAM, 115 GiB measured wired limit) defaults to `.shared`
    /// host use, whose `hostCeilingBeforeOperatorBudget` is `floor(0.75 * RAM)` = 96 GiB, further
    /// bounded by the wired limit only if it is LOWER — 115 GiB is not lower than 96 GiB, so the
    /// wired limit never binds and the shared ceiling is exactly 96 GiB, not 115 GiB.
    private static let sharedHost128 = SystemProfile.m5Max128

    /// The SAME hardware as `sharedHost128`, but with `hostUse` asserted to `.dedicatedServing` —
    /// `SystemProfile` has no `.dedicatedServing` preset of its own, so this is constructed directly
    /// rather than derived from `m5Max128` (which has no "with different hostUse" copy helper; only
    /// `withOperatorMemoryBudget` exists, and that mutates a different field). For dedicated-serving
    /// host use, `hostCeilingBeforeOperatorBudget` is `min(wiredLimitBytes, totalRAMBytes)` — 115 GiB
    /// here, since 115 GiB < 128 GiB — a full 19 GiB more headroom than the shared-host ceiling above.
    private static let dedicatedHost128 = SystemProfile(
        chip: "Apple M5 Max",
        totalRAMBytes: 128 * 1024 * 1024 * 1024,
        wiredLimitBytes: 115 * 1024 * 1024 * 1024,
        wiredLimitIsMeasured: true,
        hostUse: .operatorAssertedDedicatedServing())

    /// `SystemProfile.m3Ultra256`'s preset shared ceiling: `floor(0.75 * 256 GiB)` = 192 GiB (its
    /// `wiredLimitBytes` is ALSO exactly `0.75 * RAM`, so the wired-limit branch never binds tighter
    /// than the shared policy figure either — both land on the identical 192 GiB).
    private static let sharedHost256 = SystemProfile.m3Ultra256

    /// One (host, concurrency) -> (12-layer ceiling, 13-layer ceiling) row, independently computed
    /// from `CapacityModel`'s formulas and pinned as an exact value — never adjusted to make an
    /// assertion pass.
    private struct DivergenceRow {
        let hostName: String
        let host: SystemProfile
        let concurrency: Int
        let ceiling12: Int
        let ceiling13: Int
    }

    private static let divergenceTable: [DivergenceRow] = [
        // 128 GiB shared (96 GiB ceiling): diverges starting at concurrency 2.
        DivergenceRow(hostName: "shared128", host: sharedHost128, concurrency: 1, ceiling12: 262_144, ceiling13: 262_144),
        DivergenceRow(hostName: "shared128", host: sharedHost128, concurrency: 2, ceiling12: 232_969, ceiling13: 214_418),
        DivergenceRow(hostName: "shared128", host: sharedHost128, concurrency: 4, ceiling12: 114_396, ceiling13: 105_282),
        DivergenceRow(hostName: "shared128", host: sharedHost128, concurrency: 8, ceiling12: 55_110, ceiling13: 50_713),
        // 128 GiB dedicated (115 GiB ceiling): diverges starting at concurrency 4.
        DivergenceRow(hostName: "dedicated128", host: dedicatedHost128, concurrency: 1, ceiling12: 262_144, ceiling13: 262_144),
        DivergenceRow(hostName: "dedicated128", host: dedicatedHost128, concurrency: 2, ceiling12: 262_144, ceiling13: 262_144),
        DivergenceRow(hostName: "dedicated128", host: dedicatedHost128, concurrency: 4, ceiling12: 262_144, ceiling13: 258_535),
        DivergenceRow(hostName: "dedicated128", host: dedicatedHost128, concurrency: 8, ceiling12: 138_122, ceiling13: 127_340),
        // 256 GiB shared (192 GiB ceiling): NO divergence at any concurrency <= 8 — the 13-layer
        // correction never binds before the native max ceiling on this much headroom.
        DivergenceRow(hostName: "shared256", host: sharedHost256, concurrency: 1, ceiling12: 262_144, ceiling13: 262_144),
        DivergenceRow(hostName: "shared256", host: sharedHost256, concurrency: 2, ceiling12: 262_144, ceiling13: 262_144),
        DivergenceRow(hostName: "shared256", host: sharedHost256, concurrency: 4, ceiling12: 262_144, ceiling13: 262_144),
        DivergenceRow(hostName: "shared256", host: sharedHost256, concurrency: 8, ceiling12: 262_144, ceiling13: 262_144),
    ]

    /// Every row of the divergence table above, asserted individually against a fresh
    /// `CapacityModel.contextCeiling` call — not against each other, so a defect in the table
    /// construction itself cannot hide behind a self-consistent-but-wrong set of literals.
    func testConcurrencyDivergenceTableMatchesIndependentlyComputedCeilings() {
        for row in Self.divergenceTable {
            let profile12 = realFlashNextGeometry(
                nAttnLayers: 12, weightsBytes: Self.offloadAdjustedWeightsBytes)
            let profile13 = realFlashNextGeometry(
                nAttnLayers: 13, weightsBytes: Self.offloadAdjustedWeightsBytes)

            let ceiling12 = CapacityModel.contextCeiling(
                model: profile12, profile: row.host, kvQuant: .fp16, concurrency: row.concurrency,
                thresholds: .default)
            let ceiling13 = CapacityModel.contextCeiling(
                model: profile13, profile: row.host, kvQuant: .fp16, concurrency: row.concurrency,
                thresholds: .default)

            XCTAssertEqual(
                ceiling12, row.ceiling12,
                "\(row.hostName) @ concurrency \(row.concurrency): 12-layer ceiling")
            XCTAssertEqual(
                ceiling13, row.ceiling13,
                "\(row.hostName) @ concurrency \(row.concurrency): 13-layer ceiling")
        }
    }

    /// Named facts about WHERE the divergence starts on each host, asserted directly rather than
    /// left implicit in the table above: the first diverging concurrency is 2 on shared-128, 4 on
    /// dedicated-128, and there is no divergence at any tested concurrency (<= 8) on the 256 GiB
    /// host — the extra headroom keeps both the 12- and 13-layer ceilings pinned at the model's own
    /// native max (262,144) throughout.
    func testFirstDivergingConcurrencyPerHost() {
        func firstDivergingConcurrency(hostName: String) -> Int? {
            Self.divergenceTable
                .filter { $0.hostName == hostName }
                .first { $0.ceiling12 != $0.ceiling13 }?
                .concurrency
        }

        XCTAssertEqual(firstDivergingConcurrency(hostName: "shared128"), 2)
        XCTAssertEqual(firstDivergingConcurrency(hostName: "dedicated128"), 4)
        XCTAssertNil(firstDivergingConcurrency(hostName: "shared256"))
    }

    /// The inertness this whole section exists to guard against, stated as its own assertion: at
    /// concurrency 1, the 12-layer and 13-layer ceilings are EQUAL on all three hosts. A test that
    /// only checked "ceiling unchanged" at concurrency 1 would pass identically whether or not
    /// `InCheckpointMTPFitComposition`'s `+1 nAttnLayers` correction were deleted entirely — the
    /// correction's cost per token is real (see `InCheckpointMTPFitCompositionTests`'s byte-level
    /// assertions) but too small, at concurrency 1, to cross the binary search's discrete context
    /// boundary before hitting `nativeMaxContext` (262,144) on any of these three hosts. This is
    /// exactly why `testConcurrencyDivergenceTableMatchesIndependentlyComputedCeilings` sweeps
    /// concurrency 1, 2, 4, 8 rather than stopping at 1.
    func testCeilingsAreEqualAtConcurrencyOneOnAllThreeHostsDocumentingWhyThatAloneWouldBeInert() {
        let concurrencyOneRows = Self.divergenceTable.filter { $0.concurrency == 1 }
        XCTAssertEqual(concurrencyOneRows.count, 3)
        for row in concurrencyOneRows {
            XCTAssertEqual(
                row.ceiling12, row.ceiling13,
                "\(row.hostName) @ concurrency 1 must be EQUAL — this is the inert case")
        }
    }
}
