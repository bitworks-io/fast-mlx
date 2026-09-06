import XCTest

import HarnessCore
@testable import SpikeServingAdapters

/// Verifies `NGramOffloadFitComposition`'s formula, provenance preservation, key matching, and
/// fail-closed structural audit against synthetic safetensors headers built at runtime (no
/// committed binary fixtures) — see the type's own doc comment for the modeled shape.
final class NGramOffloadFitCompositionTests: XCTestCase {

    // MARK: - Fixture plumbing

    /// Creates a fresh temp directory and registers cleanup.
    private func makeTempDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    /// Writes a valid safetensors file at `fileURL`: an 8-byte little-endian header length, the
    /// header JSON itself (tensor name → `dtype`/`shape`/`data_offsets`, laid out contiguously in
    /// declaration order starting at 0), and then `actualDataByteCount` bytes of filler data (which
    /// may be far SMALLER than the declared offsets imply — the code under test must never read
    /// this blob, so a short/absent blob must not affect the result).
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

    // MARK: - 1. Exact expected sum, independently computed

    func testOffloadedSumMatchesIndependentlyComputedConstructionTotal() throws {
        let directory = try makeTempDirectory()
        let weightBytes = 100
        let scalesBytes = 20
        let biasesBytes = 20
        let shardCount = 4

        var fileOneTensors: [(String, Int)] = [
            ("model.layers.0.mlp.gate_proj.weight", 999),
        ]
        var fileTwoTensors: [(String, Int)] = [
            ("model.layers.3.self_attn.q_proj.weight", 777),
        ]
        for shard in 0..<2 {
            let prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_\(shard)."
            fileOneTensors.append((prefix + "weight", weightBytes))
            fileOneTensors.append((prefix + "scales", scalesBytes))
            fileOneTensors.append((prefix + "biases", biasesBytes))
        }
        for shard in 2..<4 {
            let prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_\(shard)."
            fileTwoTensors.append((prefix + "weight", weightBytes))
            fileTwoTensors.append((prefix + "scales", scalesBytes))
            fileTwoTensors.append((prefix + "biases", biasesBytes))
        }

        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard-1.safetensors"), tensors: fileOneTensors)
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard-2.safetensors"), tensors: fileTwoTensors)

        let sum = try NGramOffloadFitComposition.offloadedNGramTensorBytes(
            inModelDirectory: directory)

        // Independently computed from the fixture's OWN construction parameters, not by re-summing
        // with the code under test.
        let expected = shardCount * (weightBytes + scalesBytes + biasesBytes)
        XCTAssertEqual(expected, 560)
        XCTAssertEqual(sum, expected)
    }

    // MARK: - 2. Full formula

    func testFormulaAddsResidencyBudgetAndDiffersFromOmittingIt() throws {
        let directory = try makeTempDirectory()
        let planURL = directory.appendingPathComponent("plan.json")
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"),
            tensors: [
                (
                    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    5_600
                )
            ])
        try writePlanFile(at: planURL, maxResidentBytes: 20_000)

        let base = parsed(weightsBytes: 100_000)
        let composed = try NGramOffloadFitComposition.make(
            base: base, modelDirectory: directory, planFileURL: planURL)

        let wholeTotal = 100_000
        let offloaded = 5_600
        let residencyBudget = 20_000
        let expectedAdjusted = wholeTotal - offloaded + residencyBudget
        XCTAssertEqual(composed.profile.weightsBytes4bitEstimate, expectedAdjusted)
        XCTAssertEqual(expectedAdjusted, 114_400)

        let withoutResidencyTerm = wholeTotal - offloaded
        XCTAssertEqual(withoutResidencyTerm, 94_400)
        XCTAssertNotEqual(composed.profile.weightsBytes4bitEstimate, withoutResidencyTerm)
        XCTAssertEqual(
            composed.profile.weightsBytes4bitEstimate - withoutResidencyTerm, residencyBudget)
    }

    // MARK: - 3. Every other profile field preserved

    func testEveryOtherProfileFieldIsPreservedIndividually() throws {
        let directory = try makeTempDirectory()
        let planURL = directory.appendingPathComponent("plan.json")
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"),
            tensors: [
                (
                    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    10
                )
            ])
        try writePlanFile(at: planURL, maxResidentBytes: 5)

        let base = parsed(weightsBytes: 1_000, measured: true, declared: false, quantBits: 4)
        let composed = try NGramOffloadFitComposition.make(
            base: base, modelDirectory: directory, planFileURL: planURL)

        let baseProfile = base.profile
        let composedProfile = composed.profile
        XCTAssertEqual(composedProfile.modelType, baseProfile.modelType)
        XCTAssertEqual(composedProfile.nLayers, baseProfile.nLayers)
        XCTAssertEqual(composedProfile.nAttnLayers, baseProfile.nAttnLayers)
        XCTAssertEqual(composedProfile.nKVHeads, baseProfile.nKVHeads)
        XCTAssertEqual(composedProfile.headDim, baseProfile.headDim)
        XCTAssertEqual(composedProfile.slidingWindow, baseProfile.slidingWindow)
        XCTAssertEqual(composedProfile.fixedStateBytes, baseProfile.fixedStateBytes)
        XCTAssertEqual(composedProfile.nativeMaxContext, baseProfile.nativeMaxContext)
        XCTAssertEqual(composedProfile.license, baseProfile.license)
        XCTAssertEqual(composedProfile.mlaHeads, baseProfile.mlaHeads)
        XCTAssertEqual(composedProfile.mlaRopeDim, baseProfile.mlaRopeDim)
        XCTAssertEqual(composedProfile.mlaNopeDim, baseProfile.mlaNopeDim)
        XCTAssertEqual(composedProfile.mlaVDim, baseProfile.mlaVDim)
        XCTAssertEqual(composedProfile.swaKVHeads, baseProfile.swaKVHeads)
        XCTAssertEqual(composedProfile.swaHeadDim, baseProfile.swaHeadDim)
        XCTAssertEqual(composedProfile.vHeadDim, baseProfile.vHeadDim)
        XCTAssertEqual(composedProfile.swaVHeadDim, baseProfile.swaVHeadDim)
        XCTAssertEqual(composedProfile.auxPerLayerKeyDim, baseProfile.auxPerLayerKeyDim)
        XCTAssertEqual(composedProfile.id, "\(baseProfile.id)+ngram-offload-composition")
        XCTAssertEqual(composed.weightsAreMeasured, base.weightsAreMeasured)
        XCTAssertEqual(composed.weightsAreDeclared, base.weightsAreDeclared)
        XCTAssertEqual(composed.quantBits, base.quantBits)
    }

    // MARK: - 4. All three prefixes match; near-miss does not

    func testAllThreeTextModulePrefixesMatchAndNearMissesDoNot() {
        for prefix in ["language_model.model.", "model.", "model.language_model."] {
            let key = prefix + "layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
            let match = NGramOffloadFitComposition.matchOffloadedNGramShardKey(key)
            XCTAssertNotNil(match, "expected a match for prefix \(prefix)")
            XCTAssertEqual(match?.layerIndex, 1)
            XCTAssertEqual(match?.shardIndex, 0)
            XCTAssertEqual(match?.field, "weight")
        }

        let nearMisses = [
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding_other.shard_0.weight",
            "language_model.model.layers.1.ple.ple_embedding.shard_0.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ]
        for key in nearMisses {
            XCTAssertNil(
                NGramOffloadFitComposition.matchOffloadedNGramShardKey(key),
                "expected no match for near-miss \(key)")
        }
    }

    // MARK: - 5. `__metadata__` skipped

    func testMetadataKeyIsSkipped() throws {
        let directory = try makeTempDirectory()
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"),
            tensors: [
                (
                    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    42
                )
            ],
            includeMetadata: true)

        let sum = try NGramOffloadFitComposition.offloadedNGramTensorBytes(
            inModelDirectory: directory)
        XCTAssertEqual(sum, 42)
    }

    // MARK: - 6. Blob never read (content-invariance), and out-of-bounds offsets are rejected

    /// (a) Proves the byte-size computation depends ONLY on the header's `data_offsets` arithmetic,
    /// never on the blob's own bytes: the declared offsets are fully IN BOUNDS (required since
    /// DEFECT 1's bound check below — a blob shorter than its declared span can no longer be used
    /// to prove this, because that would now be rejected as out-of-bounds), but the on-disk blob is
    /// filled with a distinguishing non-zero sentinel byte the header never mentions. If the code
    /// under test ever derived a tensor's size from inspecting the blob's own content rather than
    /// `end - begin`, a sentinel-filled blob would have no reason to yield the exact declared size.
    /// Bounds enforcement itself is proved separately by (b), immediately below.
    func testOffloadedSumIsComputedFromHeaderOffsetsNotFromReadingTheBlob() throws {
        let directory = try makeTempDirectory()
        let byteLength = 4096
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"),
            tensors: [
                (
                    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    byteLength
                )
            ],
            blobFillByte: 0xAB)

        let sum = try NGramOffloadFitComposition.offloadedNGramTensorBytes(
            inModelDirectory: directory)
        XCTAssertEqual(sum, byteLength)
    }

    /// (b) DEFECT 1: a header can declare `data_offsets` that escape the file's actual blob region
    /// (corruption, an interrupted rewrite, or a hand-edited header) — a corrupted shard could
    /// declare a span at 2x its true size. Because that span becomes the SUBTRAHEND in the
    /// three-term fit formula, an over-declared span makes `adjustedWeights` too SMALL — an
    /// optimistic figure that still clears the `offloadedBytes < wholeFileTotal` guard, passes the
    /// fit check, and then OOMs on serve. This must throw rather than silently accept the declared
    /// span at face value.
    func testOutOfBoundsDeclaredOffsetsThrowsSafetensorsTensorEntryInvalid() throws {
        let directory = try makeTempDirectory()
        let fileURL = directory.appendingPathComponent("shard.safetensors")
        let header: [String: Any] = [
            "some.tensor": ["dtype": "F32", "shape": [1], "data_offsets": [0, 1_000_000]]
        ]
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var lengthBytes = [UInt8](repeating: 0, count: 8)
        var n = UInt64(headerData.count)
        for index in 0..<8 {
            lengthBytes[index] = UInt8(n & 0xFF)
            n >>= 8
        }
        var fileData = Data(lengthBytes)
        fileData.append(headerData)
        // Far short of the declared 1,000,000-byte span — the declared end lies outside the file.
        fileData.append(Data(repeating: 0, count: 4))
        try fileData.write(to: fileURL)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.readSafetensorsTensorByteSizes(at: fileURL)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError,
                .safetensorsTensorEntryInvalid(path: fileURL.path, key: "some.tensor"))
        }
    }

    // MARK: - 7. Audit fails closed

    func testAuditThrowsOnZeroMatches() throws {
        let directory = try makeTempDirectory()
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"),
            tensors: [("model.layers.0.mlp.gate_proj.weight", 10)])

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.offloadedNGramTensorBytes(inModelDirectory: directory)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError, .noMatchingOffloadedShardKeys)
        }
    }

    func testAuditThrowsOnShardIndexGap() throws {
        let directory = try makeTempDirectory()
        var tensors: [(String, Int)] = []
        for shard in [0, 1, 3] {
            let prefix = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_\(shard)."
            tensors.append((prefix + "weight", 10))
        }
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"), tensors: tensors)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.offloadedNGramTensorBytes(inModelDirectory: directory)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError, .nonContiguousShardIndices(shardCount: 3))
        }
    }

    func testAuditThrowsOnTwoDifferentLayerIndices() throws {
        let directory = try makeTempDirectory()
        let tensors: [(String, Int)] = [
            ("model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", 10),
            ("model.layers.2.ple.ple_embedding.ngram_embedding.shard_1.weight", 10),
        ]
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"), tensors: tensors)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.offloadedNGramTensorBytes(inModelDirectory: directory)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError,
                .multipleLayerIndicesInOffloadedShardKeys(indices: [1, 2]))
        }
    }

    func testAuditThrowsWhenAShardIsMissingAField() throws {
        let directory = try makeTempDirectory()
        let tensors: [(String, Int)] = [
            ("model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", 10),
            ("model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.scales", 2),
            ("model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.biases", 2),
            ("model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight", 10),
            ("model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.scales", 2),
            // shard 1 is missing `biases`
        ]
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"), tensors: tensors)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.offloadedNGramTensorBytes(inModelDirectory: directory)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError, .inconsistentShardFieldSets)
        }
    }

    func testAuditThrowsOnInvalidPerShardFieldCount() throws {
        let directory = try makeTempDirectory()
        let tensors: [(String, Int)] = [
            ("model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", 10),
            ("model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.scales", 2),
            ("model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight", 10),
            ("model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.scales", 2),
        ]
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"), tensors: tensors)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.offloadedNGramTensorBytes(inModelDirectory: directory)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError, .invalidPerShardFieldCount(count: 2))
        }
    }

    // MARK: - 8. Guards

    func testMakeThrowsWhenOffloadedBytesAreNotLessThanBaseWeights() throws {
        let directory = try makeTempDirectory()
        let planURL = directory.appendingPathComponent("plan.json")
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"),
            tensors: [
                (
                    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    500
                )
            ])
        try writePlanFile(at: planURL, maxResidentBytes: 10)

        let base = parsed(weightsBytes: 100)
        XCTAssertThrowsError(
            try NGramOffloadFitComposition.make(
                base: base, modelDirectory: directory, planFileURL: planURL)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError,
                .offloadedBytesNotLessThanBaseWeights(offloadedBytes: 500, baseWeightsBytes: 100))
        }
    }

    func testPlanMissingLimitsThrows() throws {
        let directory = try makeTempDirectory()
        let planURL = directory.appendingPathComponent("plan.json")
        try writePlanFile(at: planURL, includeLimits: false)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.planMaxResidentBytes(atPlanFileURL: planURL)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError,
                .planLimitsMissing(path: planURL.path))
        }
    }

    /// DEFECT 4: `intValue` must reject an out-of-`Int`-range JSON number rather than silently
    /// CLAMPING it (`NSNumber.intValue` on `1e300` yields a bogus in-range value on this platform,
    /// verified directly). A clamped `maxResidentBytes` would still be `> 0` and pass validation,
    /// silently substituting a wrong residency figure into the fit formula instead of refusing —
    /// this mirrors `ModelConfigDecoder.intOf`'s `Int(exactly:)` precedent in this codebase.
    func testPlanMaxResidentBytesRejectsOutOfRangeNumberInsteadOfClamping() throws {
        let directory = try makeTempDirectory()
        let planURL = directory.appendingPathComponent("plan.json")
        try writePlanFile(at: planURL, maxResidentBytes: 1e300)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.planMaxResidentBytes(atPlanFileURL: planURL)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError,
                .planMaxResidentBytesInvalid(path: planURL.path))
        }
    }

    func testPlanZeroMaxResidentBytesThrows() throws {
        let directory = try makeTempDirectory()
        let planURL = directory.appendingPathComponent("plan.json")
        try writePlanFile(at: planURL, maxResidentBytes: 0)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.planMaxResidentBytes(atPlanFileURL: planURL)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError,
                .planMaxResidentBytesInvalid(path: planURL.path))
        }
    }

    func testMalformedHeaderTruncatedUnderEightBytesThrows() throws {
        let directory = try makeTempDirectory()
        let fileURL = directory.appendingPathComponent("shard.safetensors")
        try Data([1, 2, 3]).write(to: fileURL)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.readSafetensorsTensorByteSizes(at: fileURL)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError,
                .safetensorsHeaderTruncated(path: fileURL.path))
        }
    }

    func testMalformedHeaderDeclaredLengthExceedsFileSizeThrows() throws {
        let directory = try makeTempDirectory()
        let fileURL = directory.appendingPathComponent("shard.safetensors")
        // Declare a header length far larger than any bytes actually present.
        var lengthBytes = [UInt8](repeating: 0, count: 8)
        var n: UInt64 = 999_999
        for index in 0..<8 {
            lengthBytes[index] = UInt8(n & 0xFF)
            n >>= 8
        }
        try Data(lengthBytes).write(to: fileURL)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.readSafetensorsTensorByteSizes(at: fileURL)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError,
                .safetensorsHeaderLengthInvalid(path: fileURL.path))
        }
    }

    func testMalformedHeaderNonJSONThrows() throws {
        let directory = try makeTempDirectory()
        let fileURL = directory.appendingPathComponent("shard.safetensors")
        let garbage = Data("not json at all".utf8)
        var lengthBytes = [UInt8](repeating: 0, count: 8)
        var n = UInt64(garbage.count)
        for index in 0..<8 {
            lengthBytes[index] = UInt8(n & 0xFF)
            n >>= 8
        }
        var fileData = Data(lengthBytes)
        fileData.append(garbage)
        try fileData.write(to: fileURL)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.readSafetensorsTensorByteSizes(at: fileURL)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError,
                .safetensorsHeaderNotJSON(path: fileURL.path))
        }
    }

    func testMalformedDataOffsetsThrows() throws {
        let directory = try makeTempDirectory()
        let fileURL = directory.appendingPathComponent("shard.safetensors")
        let header: [String: Any] = [
            "some.tensor": ["dtype": "F32", "shape": [1], "data_offsets": [10, 5]]
        ]
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var lengthBytes = [UInt8](repeating: 0, count: 8)
        var n = UInt64(headerData.count)
        for index in 0..<8 {
            lengthBytes[index] = UInt8(n & 0xFF)
            n >>= 8
        }
        var fileData = Data(lengthBytes)
        fileData.append(headerData)
        try fileData.write(to: fileURL)

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.readSafetensorsTensorByteSizes(at: fileURL)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError,
                .safetensorsTensorEntryInvalid(path: fileURL.path, key: "some.tensor"))
        }
    }

    // MARK: - Duplicate keys across shard files

    /// DEFECT 2: the same (layer, shard, field) offloaded n-gram tensor declared by TWO shard
    /// files (a consolidated `model.safetensors` staged beside the sharded set, a leftover copy, a
    /// re-quantized variant left in place) must be a hard error, not a silent double-count. Every
    /// structural-audit check (single layer, contiguous shard run, homogeneous field set, valid
    /// field count) is satisfied identically whether a tensor is counted once or twice, so nothing
    /// else in this file is positioned to catch it — and double-counting makes the sum too LARGE,
    /// which makes `adjustedWeights` too SMALL: the optimistic direction that clears the fit check
    /// and then OOMs on serve.
    func testDuplicateOffloadedShardKeyAcrossTwoShardFilesThrows() throws {
        let directory = try makeTempDirectory()
        let key = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard-a.safetensors"),
            tensors: [(key, 10)])
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard-b.safetensors"),
            tensors: [(key, 10)])

        XCTAssertThrowsError(
            try NGramOffloadFitComposition.offloadedNGramTensorBytes(inModelDirectory: directory)
        ) { error in
            XCTAssertEqual(
                error as? NGramOffloadFitCompositionError,
                .duplicateOffloadedShardKey(key: key))
        }
    }

    // MARK: - 9. Symlinked shard resolves

    func testSymlinkedShardResolvesToRealFile() throws {
        let realDirectory = try makeTempDirectory()
        let modelDirectory = try makeTempDirectory()
        let realFile = realDirectory.appendingPathComponent("real-blob")
        try writeSafetensorsFixture(
            at: realFile,
            tensors: [
                (
                    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    123
                )
            ])
        let symlinkPath = modelDirectory.appendingPathComponent("shard.safetensors")
        try FileManager.default.createSymbolicLink(at: symlinkPath, withDestinationURL: realFile)

        let sum = try NGramOffloadFitComposition.offloadedNGramTensorBytes(
            inModelDirectory: modelDirectory)
        XCTAssertEqual(sum, 123)
    }

    // MARK: - Overflow

    /// DEFECT 3: the final sum over matched tensor byte sizes must be overflow-checked like every
    /// other arithmetic site in this file — a trap here crashes the serving process, which is not
    /// the conservative failure mode this file's fail-closed design promises. Exercised directly
    /// against the internal summation helper (rather than through real fixture files) because
    /// forcing an actual `Int` overflow through the public safetensors-scanning API would require
    /// materializing exabyte-scale files, and a real trap would abort the test process instead of
    /// failing a single test.
    func testSumMatchedByteSizesThrowsOnArithmeticOverflowInsteadOfTrapping() {
        let matches = [
            NGramOffloadFitComposition.MatchedShardKey(
                layerIndex: 1, shardIndex: 0, field: "weight", byteSize: Int.max),
            NGramOffloadFitComposition.MatchedShardKey(
                layerIndex: 1, shardIndex: 1, field: "weight", byteSize: 1),
        ]
        XCTAssertThrowsError(try NGramOffloadFitComposition.sumMatchedByteSizes(matches)) { error in
            XCTAssertEqual(error as? NGramOffloadFitCompositionError, .arithmeticOverflow)
        }
    }

    func testMakeThrowsOnArithmeticOverflow() throws {
        let directory = try makeTempDirectory()
        let planURL = directory.appendingPathComponent("plan.json")
        try writeSafetensorsFixture(
            at: directory.appendingPathComponent("shard.safetensors"),
            tensors: [
                (
                    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    1
                )
            ])
        try writePlanFile(at: planURL, maxResidentBytes: 10)

        let base = parsed(weightsBytes: Int.max)
        XCTAssertThrowsError(
            try NGramOffloadFitComposition.make(
                base: base, modelDirectory: directory, planFileURL: planURL)
        ) { error in
            XCTAssertEqual(error as? NGramOffloadFitCompositionError, .arithmeticOverflow)
        }
    }
}
