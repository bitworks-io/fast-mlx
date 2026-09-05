import XCTest
@testable import HarnessCore

/// Contract test for the sizer-matrix artifact. v2 carries the same auditable host/memory/MLX-budget
/// provenance as serve startup while v1/absent-schema artifacts remain readable.
final class SizerMatrixFixtureContractTests: XCTestCase {

    private func fixtureData(_ name: String) throws -> Data {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: name, withExtension: "json"),
            "fixture \(name).json must be bundled (Package.swift resources)")
        return try Data(contentsOf: url)
    }

    private func v2FixtureObject() throws -> [String: Any] {
        let object = try JSONSerialization.jsonObject(with: fixtureData("sizer-matrix-v2-sample"))
        return try XCTUnwrap(object as? [String: Any])
    }

    private func data(from object: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    private func v2Data(mutatingHost mutation: (inout [String: Any]) -> Void) throws -> Data {
        var object = try v2FixtureObject()
        var host = try XCTUnwrap(object["host"] as? [String: Any])
        mutation(&host)
        object["host"] = host
        return try data(from: object)
    }

    private func v2Data(mutatingFirstRow mutation: (inout [String: Any]) -> Void) throws -> Data {
        var object = try v2FixtureObject()
        var rows = try XCTUnwrap(object["rows"] as? [[String: Any]])
        var first = try XCTUnwrap(rows.first)
        mutation(&first)
        rows[0] = first
        object["rows"] = rows
        return try data(from: object)
    }

    private func assertDecodeValidatedThrows(_ data: Data, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try SizerMatrixArtifact.decodeValidated(from: data), file: file, line: line)
    }

    func testLegacyV1FixtureDecodesValidated() throws {
        let artifact = try SizerMatrixArtifact.decodeValidated(from: fixtureData("sizer-matrix-v1-sample"))
        XCTAssertEqual(artifact.schema, SizerMatrixArtifact.legacySchemaTag)
        XCTAssertEqual(artifact.host.label, "m5Max128")
        XCTAssertNil(artifact.host.observationSource)
        XCTAssertEqual(artifact.host.physicalRAMBytes, 137_438_953_472)
        XCTAssertEqual(artifact.host.wiredLimitProvenance, "measured")
        XCTAssertNil(artifact.host.hostUse)
        XCTAssertNil(artifact.host.hostUseSource)
        XCTAssertNil(artifact.host.hostUsePolicyVersion)
        XCTAssertNil(artifact.host.metalRecommendedWorkingSetBytes)
        XCTAssertNil(artifact.host.metalCurrentAllocatedBytes)
        XCTAssertNil(artifact.host.osServiceReserveBytes)
        XCTAssertNil(artifact.host.mlxMemoryLimitBytes)
        XCTAssertNil(artifact.host.mlxCacheLimitBytes)
        XCTAssertEqual(artifact.kvQuant, "fp16")
        XCTAssertEqual(artifact.concurrency, 1)
        XCTAssertEqual(artifact.rows.count, 4)

        let eightBit = try XCTUnwrap(
            artifact.rows.first { $0.modelID == "Qwen3-8B" && $0.weightBits == 8 })
        XCTAssertEqual(eightBit.classification, "yellow")
        XCTAssertTrue(eightBit.fits)

        let bigRed = try XCTUnwrap(
            artifact.rows.first { $0.modelID == "DeepSeek-V3-671B" && $0.weightBits == 8 })
        XCTAssertFalse(bigRed.fits)
        XCTAssertFalse(bigRed.estimateIsMeasured)
    }

    func testAbsentSchemaDecodesAsLegacyCompatibleArtifact() throws {
        let untagged = """
        {"host":{"label":"legacy","ramBytes":16,"wiredLimitBytes":12,"wiredLimitIsMeasured":false},"kvQuant":"fp16","concurrency":1,"rows":[]}
        """
        let data = try XCTUnwrap(untagged.data(using: .utf8))
        let artifact = try SizerMatrixArtifact.decodeValidated(from: data)
        XCTAssertNil(artifact.schema)
        XCTAssertEqual(artifact.host.physicalRAMBytes, 16)
        XCTAssertEqual(artifact.host.wiredLimitBytes, 12)
        XCTAssertEqual(artifact.host.wiredLimitProvenance, "synthesized")
        XCTAssertNil(artifact.host.mlxMemoryLimitBytes)
    }

    func testLegacyV1AndAbsentArtifactsRequireOriginalHostContract() throws {
        for schemaPrefix in ["\"schema\":\"sizer-matrix/v1\",", ""] {
            for badHost in [
                "\"ramBytes\":16,\"wiredLimitBytes\":12,\"wiredLimitIsMeasured\":true",
                "\"label\":\"legacy\",\"wiredLimitBytes\":12,\"wiredLimitIsMeasured\":true",
                "\"label\":\"legacy\",\"ramBytes\":0,\"wiredLimitBytes\":12,\"wiredLimitIsMeasured\":true",
                "\"label\":\"legacy\",\"ramBytes\":16,\"wiredLimitIsMeasured\":true",
                "\"label\":\"legacy\",\"ramBytes\":16,\"wiredLimitBytes\":0,\"wiredLimitIsMeasured\":true",
                "\"label\":\"legacy\",\"ramBytes\":16,\"wiredLimitBytes\":12",
            ] {
                let json = "{\(schemaPrefix)\"host\":{\(badHost)},\"kvQuant\":\"fp16\",\"concurrency\":1,\"rows\":[]}"
                assertDecodeValidatedThrows(try XCTUnwrap(json.data(using: .utf8)))
            }
        }
    }

    func testV2FixtureDecodesValidatedServingMemoryContract() throws {
        let artifact = try SizerMatrixArtifact.decodeValidated(from: fixtureData("sizer-matrix-v2-sample"))
        XCTAssertEqual(artifact.schema, SizerMatrixArtifact.schemaTag)
        XCTAssertEqual(artifact.host.label, "m5Max128")
        XCTAssertEqual(artifact.host.observationSource, "modeled-preset")
        XCTAssertEqual(artifact.host.hostUse, "shared")
        XCTAssertEqual(artifact.host.hostUseSource, "default")
        XCTAssertEqual(artifact.host.hostUsePolicyVersion, "host-use/v1")
        XCTAssertEqual(artifact.host.physicalRAMBytes, 137_438_953_472)
        XCTAssertEqual(artifact.host.wiredLimitBytes, 123_480_309_760)
        XCTAssertEqual(artifact.host.wiredLimitProvenance, "measured")
        XCTAssertNil(artifact.host.metalRecommendedWorkingSetBytes)
        XCTAssertNil(artifact.host.metalCurrentAllocatedBytes)
        XCTAssertEqual(artifact.host.effectiveMemoryCeilingBytes, 103_079_215_104)
        XCTAssertEqual(artifact.host.effectiveMemoryCeilingSource, "shared-policy")
        XCTAssertEqual(artifact.host.osServiceReserveBytes, CapacityThresholds.default.osReserveBytes)
        XCTAssertEqual(artifact.host.mlxMemoryLimitBytes, 103_079_215_104)
        XCTAssertEqual(artifact.host.mlxCacheLimitBytes, 12_884_901_888)
        XCTAssertLessThan(try XCTUnwrap(artifact.host.mlxCacheLimitBytes), try XCTUnwrap(artifact.host.mlxMemoryLimitBytes))

        // Derived from the catalog rather than hardcoded: the fixture must carry one row per
        // catalog entry per weight-bit tier, which is exactly what `decodeValidated` enforces
        // ("rows must cover the complete catalog matrix", SizerMatrixArtifact.swift:532-536).
        // A literal here goes stale the moment a model is added -- as it did when the
        // Qwen3.8-Flash-Next entry landed -- and fails in a way that looks like a fixture
        // corruption rather than an expected catalog growth.
        XCTAssertEqual(artifact.rows.count, ModelArchProfile.catalog.count * 2)
        let eightBit = try XCTUnwrap(
            artifact.rows.first { $0.modelID == "Qwen3.5-9B" && $0.weightBits == 8 })
        XCTAssertEqual(eightBit.kvBytesAtContext, 1_125_253_120)
        XCTAssertFalse(eightBit.estimateIsMeasured)
    }

    func testV2RejectsUnknownTypedHostProvenance() throws {
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["hostUse"] = "workstation" }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["hostUseSource"] = "heuristic" }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["hostUsePolicyVersion"] = "host-use/v99" }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: {
            $0["hostUse"] = "dedicated-serving"
            $0["hostUseSource"] = "automatic"
        }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["wiredLimitProvenance"] = "estimated" }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["effectiveMemoryCeilingSource"] = "unknown" }))
    }

    func testV2RejectsInconsistentHostBudgets() throws {
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["physicalRAMBytes"] = 0 }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["wiredLimitBytes"] = 0 }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["wiredLimitBytes"] = 200_000_000_000 }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["effectiveMemoryCeilingBytes"] = 0 }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["effectiveMemoryCeilingBytes"] = 200_000_000_000 }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["osServiceReserveBytes"] = 0 }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["mlxMemoryLimitBytes"] = 100_000_000_000 }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: { $0["mlxCacheLimitBytes"] = 4_294_967_296 }))
    }

    func testV2RecomputesHostEnvelopeAndRequiresExactObservations() throws {
        let alternateEffective = 90 * 1_073_741_824
        assertDecodeValidatedThrows(try v2Data(mutatingHost: {
            $0["effectiveMemoryCeilingBytes"] = alternateEffective
            $0["mlxMemoryLimitBytes"] = alternateEffective
            $0["mlxCacheLimitBytes"] = alternateEffective / 8
        }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: {
            $0["effectiveMemoryCeilingSource"] = "wired-limit"
        }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: {
            $0["metalRecommendedWorkingSetBytes"] = 80 * 1_073_741_824
        }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: {
            $0["metalRecommendedWorkingSetBytes"] = -1
        }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: {
            $0["metalCurrentAllocatedBytes"] = -1
        }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: {
            $0["osServiceReserveBytes"] = CapacityThresholds.default.osReserveBytes + 1
        }))
    }

    func testV2RejectsUnknownMatrixPolicyAndModeledRowsClaimingMeasurement() throws {
        var object = try v2FixtureObject()
        object["concurrency"] = 0
        assertDecodeValidatedThrows(try data(from: object))

        object = try v2FixtureObject()
        object["kvQuant"] = "future-quant"
        assertDecodeValidatedThrows(try data(from: object))

        assertDecodeValidatedThrows(try v2Data(mutatingFirstRow: { $0["modelID"] = "" }))
        assertDecodeValidatedThrows(try v2Data(mutatingFirstRow: { $0["weightBits"] = 0 }))
        assertDecodeValidatedThrows(try v2Data(mutatingFirstRow: { $0["classification"] = "blue" }))
        assertDecodeValidatedThrows(try v2Data(mutatingFirstRow: { $0["fits"] = false }))

        let built = SizerMatrixArtifact.build(
            box: .m5Max128, context: nil, kvQuant: .fp16, concurrency: 1,
            observation: .modeledPreset(label: "m5Max128"))
        XCTAssertTrue(built.rows.allSatisfy { !$0.estimateIsMeasured })
        XCTAssertTrue(try SizerMatrixArtifact.decodeValidated(
            from: XCTUnwrap(built.encodedJSON().data(using: .utf8))).rows.allSatisfy {
                !$0.estimateIsMeasured
            })
    }

    func testV2RejectsInconsistentRows() throws {
        for key in [
            "weightBits", "weightsBytes", "kvBytesAtContext", "transientPrefillBytes",
            "totalPeakBytes", "maxContextThatFits", "requestedContext",
        ] {
            assertDecodeValidatedThrows(try v2Data(mutatingFirstRow: { $0[key] = -1 }))
        }
        assertDecodeValidatedThrows(try v2Data(mutatingFirstRow: { $0["totalPeakBytes"] = 6_999_999_999 }))
    }

    func testV2RejectsRowsThatDoNotRecomputeFromCatalogAndHostPolicy() throws {
        assertDecodeValidatedThrows(try v2Data(mutatingFirstRow: {
            $0["kvBytesAtContext"] = (($0["kvBytesAtContext"] as? Int) ?? 0) + 1
            $0["totalPeakBytes"] = (($0["totalPeakBytes"] as? Int) ?? 0) + 1
        }))
        assertDecodeValidatedThrows(try v2Data(mutatingFirstRow: { $0["modelID"] = "unknown-model" }))
        assertDecodeValidatedThrows(try v2Data(mutatingFirstRow: { $0["weightBits"] = 6 }))
        assertDecodeValidatedThrows(try v2Data(mutatingFirstRow: {
            $0["maxContextThatFits"] = (($0["maxContextThatFits"] as? Int) ?? 0) + 1
        }))
        assertDecodeValidatedThrows(try v2Data(mutatingFirstRow: {
            $0["totalPeakBytes"] = (($0["totalPeakBytes"] as? Int) ?? 0) + 1
        }))

        var object = try v2FixtureObject()
        var rows = try XCTUnwrap(object["rows"] as? [[String: Any]])
        rows.removeLast()
        object["rows"] = rows
        assertDecodeValidatedThrows(try data(from: object))
    }

    func testForeignSchemaFailsClosed() throws {
        let foreign = """
        {"schema":"sizer-matrix/v99","host":{},"kvQuant":"fp16","concurrency":1,"rows":[]}
        """
        let data = try XCTUnwrap(foreign.data(using: .utf8))
        XCTAssertThrowsError(try SizerMatrixArtifact.decodeValidated(from: data))
    }

    func testRoundTripEncodeDecode() throws {
        let liveProfile = SystemProfile(
            chip: "live-test", totalRAMBytes: 128 * 1_073_741_824,
            wiredLimitBytes: 115 * 1_073_741_824, wiredLimitIsMeasured: true,
            hostUse: .automaticShared)
        let built = SizerMatrixArtifact.build(
            box: liveProfile, context: nil, kvQuant: .fp16, concurrency: 1,
            observation: .live(label: .auto, currentMetalAllocatedBytes: 123))
        let json = built.encodedJSON()
        let data = try XCTUnwrap(json.data(using: String.Encoding.utf8))
        let decoded = try SizerMatrixArtifact.decodeValidated(from: data)

        XCTAssertEqual(decoded.schema, SizerMatrixArtifact.schemaTag)
        XCTAssertEqual(decoded.host.hostUse, "shared")
        XCTAssertEqual(decoded.host.hostUseSource, "automatic")
        XCTAssertEqual(decoded.host.hostUsePolicyVersion, "host-use/v1")
        XCTAssertEqual(decoded.host.physicalRAMBytes, liveProfile.totalRAMBytes)
        XCTAssertEqual(decoded.host.wiredLimitProvenance, "measured")
        XCTAssertEqual(decoded.host.effectiveMemoryCeilingBytes, liveProfile.effectiveMemoryCeiling.bytes)
        XCTAssertEqual(decoded.host.effectiveMemoryCeilingSource, liveProfile.effectiveMemoryCeiling.source.rawValue)
        XCTAssertEqual(decoded.host.osServiceReserveBytes, CapacityThresholds.default.osReserveBytes)
        XCTAssertEqual(decoded.host.mlxMemoryLimitBytes, liveProfile.effectiveMemoryCeiling.bytes)
        XCTAssertEqual(decoded.host.mlxCacheLimitBytes, CapacityModel.recommendedCacheLimitBytes(
            wiredLimitBytes: liveProfile.effectiveMemoryCeiling.bytes))
        XCTAssertEqual(decoded.host.metalCurrentAllocatedBytes, 123)
        XCTAssertFalse(decoded.rows.isEmpty)
        XCTAssertEqual(decoded.rows.count, built.rows.count)
        XCTAssertEqual(decoded.rows.first?.modelID, built.rows.first?.modelID)
        XCTAssertEqual(decoded.rows.first?.totalPeakBytes, built.rows.first?.totalPeakBytes)
        XCTAssertEqual(decoded.host.label, "auto")
        XCTAssertEqual(decoded.host.observationSource, "live")
    }

    func testBuiltPresetArtifactEncodesUnavailableMetalCurrentAllocationAsNull() throws {
        let built = SizerMatrixArtifact.build(
            box: .m5Max128, context: nil, kvQuant: .fp16, concurrency: 1,
            observation: .modeledPreset(label: "m5Max128"))
        let json = built.encodedJSON()
        XCTAssertTrue(json.contains("\"metalCurrentAllocatedBytes\":null"))

        let data = try XCTUnwrap(json.data(using: String.Encoding.utf8))
        let decoded = try SizerMatrixArtifact.decodeValidated(from: data)
        XCTAssertNil(decoded.host.metalCurrentAllocatedBytes)
    }

    func testModeledPresetObservationDoesNotFabricateCurrentMetalAllocation() throws {
        let built = SizerMatrixArtifact.build(
            box: .m5Max128, context: nil, kvQuant: .fp16, concurrency: 1,
            observation: .modeledPreset(label: "m5Max128"))
        XCTAssertNil(built.host.metalCurrentAllocatedBytes)
    }

    func testV2RejectsModeledPresetWithCurrentMetalAllocation() throws {
        assertDecodeValidatedThrows(try v2Data(mutatingHost: {
            $0["metalCurrentAllocatedBytes"] = 123
        }))
        assertDecodeValidatedThrows(try v2Data(mutatingHost: {
            $0["observationSource"] = "live"
        }))
    }

    func testBuildNormalizesNonpositiveMetalRecommendationToUnavailable() throws {
        let profile = SystemProfile(
            chip: "modeled", totalRAMBytes: 16 * 1_073_741_824,
            wiredLimitBytes: 12 * 1_073_741_824, wiredLimitIsMeasured: false,
            recommendedWorkingSetBytes: 0, hostUse: .defaultShared)
        let built = SizerMatrixArtifact.build(
            box: profile, context: nil, kvQuant: .fp16, concurrency: 1,
            observation: .modeledPreset(label: "modeled"))
        XCTAssertNil(built.host.metalRecommendedWorkingSetBytes)
        _ = try SizerMatrixArtifact.decodeValidated(
            from: XCTUnwrap(built.encodedJSON().data(using: .utf8)))
    }

    /// Regression for `ModelSizer.scaledModel`: it used to rebuild `ModelArchProfile` field by
    /// field, which silently dropped every optional field added after that rebuild was last
    /// written (`swaKVHeads`/`swaHeadDim`/`vHeadDim`/`swaVHeadDim`/`auxPerLayerKeyDim`). This must
    /// NOT be proven by recomputing a row and comparing it to another recomputed row: both
    /// `ModelSizer.report` (the generator) and `SizerMatrixArtifact.validateV2Row` (the fixture
    /// validator) route through the SAME `scaledModel`, so that comparison is self-consistent and
    /// cannot detect a dropped field — that self-consistency is exactly why 16/16 fixture tests
    /// passed while the bug was live. Instead this computes the expected KV bytes on the
    /// UNSCALED catalog profile via `CapacityModel.kvBytesForContext` directly (a path
    /// `scaledModel` never touches) and checks it against `ModelSizer.report`'s row for the same
    /// model/context — a genuine cross-check of the scaling wrapper against the underlying model.
    func testScaledModelPreservesOptionalGeometryFields() throws {
        let context = 32768
        let kvQuant = KVQuantTier.fp16

        // Baichuan-M1-14B (`.dualGeometrySWA`): exercises `swaKVHeads`/`swaHeadDim`, dropped
        // fields that only affect the SWA (window-capped) layers' term. By hand: growing term
        // 20 attn layers x 2 kv x (256+256) x 2 B(fp16) = 40,960 B/tok x 32,768 tok =
        // 1,342,177,280 B; SWA local-cap term 20 local layers x 8 swaKVHeads x (128+128) x 2 B x
        // 8,192 window = 671,088,640 B; + fixedStateBytes 122,880 B (conv state, added once, not
        // scaled by context) = 2,013,388,800 B total — the catalog's own reconciled KV@32K figure.
        try assertKVMatchesCatalogModel(id: "Baichuan-M1-14B", context: context, kvQuant: kvQuant)

        // Qwen3.8-Flash-Next (`.hybridLinear`): exercises `auxPerLayerKeyDim`, the QSA indexer's
        // growing per-layer key cache. By hand: K+V term 12 attn layers x 2 kv x (256+256) x 2 B =
        // 24,576 B/tok; indexer term 12 attn layers x 128 auxPerLayerKeyDim x 2 B(fixed, not
        // kvQuant-scaled) = 3,072 B/tok; total 27,648 B/tok x 32,768 tok = 905,969,664 B, +
        // fixedStateBytes 115,458,048 B (gated-delta-net recurrent+conv state) = 1,021,427,712 B.
        try assertKVMatchesCatalogModel(id: "Qwen3.8-Flash-Next", context: context, kvQuant: kvQuant)
    }

    private func assertKVMatchesCatalogModel(
        id: String, context: Int, kvQuant: KVQuantTier, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let unscaled = try XCTUnwrap(
            ModelArchProfile.catalog.first { $0.id == id }, "catalog must still carry \(id)",
            file: file, line: line)
        let expectedKVBytes = CapacityModel.kvBytesForContext(
            unscaled, context: context, kvQuant: kvQuant, concurrency: 1)

        for weightBits in [4, 8] {
            let rows = ModelSizer.report(
                box: .m5Max128, context: context, weightBitOptions: [weightBits], kvQuant: kvQuant,
                concurrency: 1)
            let row = try XCTUnwrap(
                rows.first { $0.modelID == id && $0.weightBits == weightBits },
                "no \(id)/\(weightBits)-bit row", file: file, line: line)
            XCTAssertEqual(
                Double(row.kvBytesAtContext), expectedKVBytes, accuracy: 1,
                "\(id) weightBits=\(weightBits): scaledModel must preserve every optional "
                    + "geometry field, not just the ones scaledModel's author remembered to list",
                file: file, line: line)
        }
    }
}
