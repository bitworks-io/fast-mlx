import XCTest

@testable import HarnessCore

final class CompressedAttentionProbeEvidenceTests: XCTestCase {
    private let cleanSHA = String(repeating: "a", count: 40)
    private let hashA = String(repeating: "a", count: 64)
    private let hashB = String(repeating: "b", count: 64)
    private let hashC = String(repeating: "c", count: 64)
    private let hashD = String(repeating: "d", count: 64)
    private let hashE = String(repeating: "e", count: 64)
    private let hashF = String(repeating: "f", count: 64)
    private let artifactHash = String(repeating: "0", count: 64)

    func testPromotionEvidencePreservesRawPairedRowsAndRoundTrips() throws {
        let evidence = try makeEvidence()
        let validated = try evidence.validated()
        let decoded = try JSONDecoder().decode(
            CompressedAttentionProbeEvidence.self,
            from: JSONEncoder().encode(validated))

        XCTAssertEqual(decoded, validated)
        XCTAssertEqual(validated.rows.count, 6)
        XCTAssertEqual(validated.plan.contextTokens, 8_192)
        XCTAssertEqual(
            validated.rows.map(\.position.pairedBlockIndex),
            [0, 0, 1, 1, 2, 2])
        XCTAssertEqual(
            validated.rows.filter { $0.role == .candidate }.map(\.position.runPosition),
            [0, 1, 0])
    }

    func testDuplicateOrInvalidRunPositionsFailClosed() throws {
        var rows = makeRows()
        rows[1] = row(role: .fp16Reference, block: 0, position: 0)
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .duplicateRunPosition(block: 0, position: 0))
        }

        rows = makeRows()
        rows[0] = row(role: .candidate, block: 3, position: 0)
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidRunPosition)
        }
    }

    func testPromotionRowsRequireCompleteReceiptsAndValidIdentities() throws {
        var rows = makeRows()
        rows[0] = CompressedAttentionProbeRunRow(
            role: .candidate,
            operation: .swiftLMQuantizedAttention,
            position: CompressedAttentionProbeRunPosition(
                pairedBlockIndex: 0,
                runPosition: 0),
            receipts: nil)
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .missingReceipts)
        }

        let badIdentity = modelIdentity(modelConfigSHA256: "not-a-sha")
        XCTAssertThrowsError(try makeEvidence(model: badIdentity).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidIdentity("modelConfigSHA256"))
        }

        rows = makeRows()
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(
                power: CompressedAttentionProbePowerReceipts(
                    lowPowerModeEnabledBefore: false,
                    lowPowerModeEnabledAfter: false,
                    powerSourceBefore: .unavailable,
                    powerSourceAfter: .acPower)))
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidPowerState)
        }
    }

    func testExploratoryRowsMayRecordUnavailablePowerSource() throws {
        let plan = try CompressedAttentionProbePlan(
            operation: .swiftLMQuantizedAttention,
            contextTokens: 128,
            queryTokens: 1,
            prefillChunkTokens: 32,
            outputTokens: 16,
            stopTokenIDs: [151_643, 151_645],
            batchSize: 1,
            queryHeadCount: 8,
            kvHeadCount: 2,
            headDimension: 128,
            dtype: .float16,
            mask: .none,
            layout: .affine(
                keyBits: 4, valueBits: 4,
                keyGroupSize: 64, valueGroupSize: 64),
            warmupRuns: 1,
            measuredRuns: 1,
            seed: 7,
            workloadNonce: "fused-kv-exploratory",
            harnessGitSHA: cleanSHA,
            promotionEvidence: false)
        let unavailableCandidate = runReceipts(
            role: .candidate,
            power: CompressedAttentionProbePowerReceipts(
                lowPowerModeEnabledBefore: false,
                lowPowerModeEnabledAfter: false,
                powerSourceBefore: .unavailable,
                powerSourceAfter: .unavailable))
        let unavailableReference = runReceipts(
            role: .fp16Reference,
            power: CompressedAttentionProbePowerReceipts(
                lowPowerModeEnabledBefore: false,
                lowPowerModeEnabledAfter: false,
                powerSourceBefore: .unavailable,
                powerSourceAfter: .unavailable))
        let evidence = try makeEvidence(
            plan: plan,
            rows: [
                row(
                    role: .candidate,
                    block: 0,
                    position: 0,
                    receipts: unavailableCandidate),
                row(
                    role: .fp16Reference,
                    block: 0,
                    position: 1,
                    receipts: unavailableReference),
            ])

        XCTAssertNoThrow(try evidence.validated())
    }

    func testByteAndMetricRelationsFailClosed() throws {
        var receipts = runReceipts(
            bytes: CompressedAttentionProbeByteReceipts(
                payloadBytes: 2_048,
                scaleBytes: 1_024,
                biasBytes: 1_024,
                controlBytes: 0,
                alignmentPaddingBytes: 0,
                fp16ResidentBytes: 0,
                persistentKVBytes: 4_096,
                materializationBytes: 0,
                otherWorkspaceBytes: 1_024,
                peakTemporaryBytes: 1_024,
                totalBytes: 4_100))
        var rows = makeRows()
        rows[0] = row(
            role: .candidate, block: 0, position: 0, receipts: receipts)
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidByteAccounting)
        }

        receipts = runReceipts(
            numericControls: CompressedAttentionProbeNumericControls(
                packedMaxAbsoluteError: 0.0021,
                packedMaxRelativeError: 0.001,
                packedTop1TokenID: 42,
                unpackedTop1TokenID: 42,
                fp16MaxAbsoluteError: 0.000001,
                fp16MaxRelativeError: 0.00001,
                fp16Top1TokenID: 42,
                referenceTop1TokenID: 42))
        rows = makeRows()
        rows[0] = row(
            role: .candidate, block: 0, position: 0, receipts: receipts)
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .packedStructuralMismatch)
        }

        receipts = runReceipts(
            role: .fp16Reference,
            numericControls: CompressedAttentionProbeNumericControls(
                packedMaxAbsoluteError: 0.001,
                packedMaxRelativeError: 0.001,
                packedTop1TokenID: 42,
                unpackedTop1TokenID: 42,
                fp16MaxAbsoluteError: 0.000001,
                fp16MaxRelativeError: 0.00011,
                fp16Top1TokenID: 42,
                referenceTop1TokenID: 42))
        rows = makeRows()
        rows[1] = row(
            role: .fp16Reference, block: 0, position: 1, receipts: receipts)
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .fp16ControlMismatch)
        }
    }

    func testPromotionRejectsPowerLowPowerAndThermalTransitions() throws {
        var rows = makeRows()
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(
                power: CompressedAttentionProbePowerReceipts(
                    lowPowerModeEnabledBefore: false,
                    lowPowerModeEnabledAfter: true,
                    powerSourceBefore: .acPower,
                    powerSourceAfter: .acPower)))
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidPowerState)
        }

        rows = makeRows()
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(
                thermal: CompressedAttentionProbeThermalReceipts(
                    before: .nominal,
                    after: .fair)))
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidThermalState)
        }
    }

    func testPairedRowsBindWorkloadMemoryAndEnvironment() throws {
        var rows = makeRows()
        rows[1] = row(
            role: .fp16Reference,
            block: 0,
            position: 1,
            receipts: runReceipts(
                role: .fp16Reference,
                hashes: CompressedAttentionProbeHashes(
                    sourceKVProjectionSHA256: hashC,
                    packedKVProjectionSHA256: hashA,
                    inputTokenIDsSHA256: hashB,
                    outputTokenIDsSHA256: hashE)))
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .pairedBlockIdentityMismatch(0))
        }

        rows = makeRows()
        rows[1] = row(
            role: .fp16Reference,
            block: 0,
            position: 1,
            receipts: runReceipts(
                role: .fp16Reference,
                memorySettings: CompressedAttentionProbeMemorySettings(
                    memoryLimitBytes: 16 << 30,
                    cacheLimitBytes: 4 << 30,
                    wiredLimitBytes: 0,
                    cacheResetPolicy: .preserveAcrossPair)))
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .pairedBlockIdentityMismatch(0))
        }
    }

    func testByteComponentsMustReconcileWithPersistentAndPeakTotals() throws {
        let malformed = CompressedAttentionProbeByteReceipts(
            payloadBytes: 2_048,
            scaleBytes: 1_024,
            biasBytes: 1_024,
            controlBytes: 0,
            alignmentPaddingBytes: 0,
            fp16ResidentBytes: 0,
            persistentKVBytes: 4_095,
            materializationBytes: 0,
            otherWorkspaceBytes: 1_024,
            peakTemporaryBytes: 1_024,
            totalBytes: 5_119)
        var rows = makeRows()
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(bytes: malformed))

        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidByteAccounting)
        }
    }

    func testByteComponentsMustMatchTheAuthenticatedLayout() throws {
        let fp16Only = defaultBytes(role: .fp16Reference)
        var rows = makeRows()
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(bytes: fp16Only))
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidByteAccounting)
        }

        rows = makeRows()
        rows[1] = row(
            role: .fp16Reference,
            block: 0,
            position: 1,
            receipts: runReceipts(
                role: .fp16Reference,
                bytes: defaultBytes(role: .candidate)))
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidByteAccounting)
        }
    }

    func testModelIdentityAcceptsCanonicalRepositoryIDsButRejectsPaths() throws {
        XCTAssertNoThrow(try makeEvidence(
            model: modelIdentity(
                modelID: "mlx-community/Qwen3-32B-4bit"))
            .validated())

        XCTAssertThrowsError(try makeEvidence(
            model: modelIdentity(modelID: "../models/Qwen3-32B"))
            .validated()) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbeEvidenceError,
                    .invalidIdentity("modelID"))
            }
    }

    func testTimingRequiresOrderedMonotonicReceipts() throws {
        var rows = makeRows()
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(
                timing: CompressedAttentionProbeTiming(
                    monotonicStartSeconds: 100.12,
                    monotonicEndSeconds: 100,
                    wallClockSeconds: 0.12,
                    attentionSeconds: 0.08)))

        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidTiming)
        }
    }

    func testCounterbalancedPositionsMustMatchMonotonicOrder() throws {
        var rows = makeRows()
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(
                timing: CompressedAttentionProbeTiming(
                    monotonicStartSeconds: 100.3,
                    monotonicEndSeconds: 100.42,
                    wallClockSeconds: 0.12,
                    attentionSeconds: 0.08)))

        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidTiming)
        }
    }

    func testFP16CandidateCannotClaimMaterializationBytes() throws {
        let plan = try CompressedAttentionProbePlan(
            operation: .fp16SDPA,
            contextTokens: 128,
            queryTokens: 1,
            prefillChunkTokens: 32,
            outputTokens: 16,
            stopTokenIDs: [],
            batchSize: 1,
            queryHeadCount: 8,
            kvHeadCount: 2,
            headDimension: 128,
            dtype: .float16,
            mask: .none,
            layout: .fp16,
            warmupRuns: 1,
            measuredRuns: 1,
            seed: 7,
            workloadNonce: "fused-kv-fp16-fixture",
            harnessGitSHA: cleanSHA,
            promotionEvidence: false)
        let badBytes = CompressedAttentionProbeByteReceipts(
            payloadBytes: 0,
            scaleBytes: 0,
            biasBytes: 0,
            controlBytes: 0,
            alignmentPaddingBytes: 0,
            fp16ResidentBytes: 4_096,
            persistentKVBytes: 4_096,
            materializationBytes: 512,
            otherWorkspaceBytes: 512,
            peakTemporaryBytes: 1_024,
            totalBytes: 5_120)
        let rows = [
            row(
                role: .candidate,
                operation: .fp16SDPA,
                block: 0,
                position: 0,
                receipts: runReceipts(bytes: badBytes)),
            row(
                role: .fp16Reference,
                operation: .fp16SDPA,
                block: 0,
                position: 1),
        ]

        XCTAssertThrowsError(try makeEvidence(
            plan: plan, rows: rows).validated()) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbeEvidenceError,
                    .invalidByteAccounting)
            }
    }

    private func makeEvidence(
        plan: CompressedAttentionProbePlan? = nil,
        model: CompressedAttentionProbeModelIdentity? = nil,
        rows: [CompressedAttentionProbeRunRow]? = nil
    ) throws -> CompressedAttentionProbeEvidence {
        let evidencePlan: CompressedAttentionProbePlan
        if let plan {
            evidencePlan = plan
        } else {
            evidencePlan = try makePlan()
        }
        return CompressedAttentionProbeEvidence(
            schemaVersion: 1,
            artifactID: artifactHash,
            plan: CompressedAttentionProbePlanIdentity(
                plan: evidencePlan),
            model: model ?? modelIdentity(),
            package: CompressedAttentionProbePackageIdentity(
                mlxSwiftVersion: "0.31.6",
                mlxSwiftLMRevision:
                    "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
                swiftVersion: "6.0",
                harnessBuildConfiguration: "Release"),
            rows: rows ?? makeRows())
    }

    private func makePlan() throws -> CompressedAttentionProbePlan {
        try CompressedAttentionProbePlan(
            operation: .swiftLMQuantizedAttention,
            contextTokens: 8_192,
            queryTokens: 1,
            prefillChunkTokens: 512,
            outputTokens: 128,
            stopTokenIDs: [151_643, 151_645],
            batchSize: 1,
            queryHeadCount: 8,
            kvHeadCount: 2,
            headDimension: 128,
            dtype: .float16,
            mask: .none,
            layout: .affine(
                keyBits: 4, valueBits: 4,
                keyGroupSize: 64, valueGroupSize: 64),
            warmupRuns: 1,
            measuredRuns: 3,
            seed: 7,
            workloadNonce: "fused-kv-fixture",
            harnessGitSHA: cleanSHA,
            promotionEvidence: true)
    }

    private func modelIdentity(
        modelID: String = "Qwen3-32B",
        modelConfigSHA256: String? = nil
    ) -> CompressedAttentionProbeModelIdentity {
        CompressedAttentionProbeModelIdentity(
            modelID: modelID,
            modelConfigSHA256: modelConfigSHA256 ?? hashA,
            checkpointManifestSHA256: hashB,
            tokenizerSHA256: hashC,
            tokenizerConfigSHA256: hashD)
    }

    private func makeRows() -> [CompressedAttentionProbeRunRow] {
        [
            row(role: .candidate, block: 0, position: 0),
            row(role: .fp16Reference, block: 0, position: 1),
            row(role: .fp16Reference, block: 1, position: 0),
            row(role: .candidate, block: 1, position: 1),
            row(role: .candidate, block: 2, position: 0),
            row(role: .fp16Reference, block: 2, position: 1),
        ]
    }

    private func row(
        role: CompressedAttentionProbeRunRole,
        operation: CompressedAttentionProbeOperation? = nil,
        block: Int,
        position: Int,
        receipts: CompressedAttentionProbeRunReceipts? = nil
    ) -> CompressedAttentionProbeRunRow {
        let defaultReceiptTiming = CompressedAttentionProbeTiming(
            monotonicStartSeconds: 100,
            monotonicEndSeconds: 100.12,
            wallClockSeconds: 0.12,
            attentionSeconds: 0.08)
        let positionedReceipts: CompressedAttentionProbeRunReceipts
        if let receipts, receipts.timing == defaultReceiptTiming {
            positionedReceipts = replacingTiming(
                in: receipts,
                with: defaultTiming(block: block, position: position))
        } else {
            positionedReceipts = receipts ?? runReceipts(
                role: role,
                timing: defaultTiming(block: block, position: position))
        }
        return CompressedAttentionProbeRunRow(
            role: role,
            operation: operation ?? (role == .candidate
                ? .swiftLMQuantizedAttention : .fp16SDPA),
            position: CompressedAttentionProbeRunPosition(
                pairedBlockIndex: block,
                runPosition: position),
            receipts: positionedReceipts)
    }

    private func replacingTiming(
        in receipts: CompressedAttentionProbeRunReceipts,
        with timing: CompressedAttentionProbeTiming
    ) -> CompressedAttentionProbeRunReceipts {
        CompressedAttentionProbeRunReceipts(
            timing: timing,
            bytes: receipts.bytes,
            mlxMemory: receipts.mlxMemory,
            processRSS: receipts.processRSS,
            memorySettings: receipts.memorySettings,
            power: receipts.power,
            thermal: receipts.thermal,
            numericControls: receipts.numericControls,
            hashes: receipts.hashes)
    }

    private func runReceipts(
        role: CompressedAttentionProbeRunRole = .candidate,
        bytes: CompressedAttentionProbeByteReceipts? = nil,
        timing: CompressedAttentionProbeTiming? = nil,
        power: CompressedAttentionProbePowerReceipts? = nil,
        thermal: CompressedAttentionProbeThermalReceipts? = nil,
        memorySettings: CompressedAttentionProbeMemorySettings? = nil,
        numericControls: CompressedAttentionProbeNumericControls? = nil,
        hashes: CompressedAttentionProbeHashes? = nil
    ) -> CompressedAttentionProbeRunReceipts {
        CompressedAttentionProbeRunReceipts(
            timing: timing ?? CompressedAttentionProbeTiming(
                monotonicStartSeconds: 100,
                monotonicEndSeconds: 100.12,
                wallClockSeconds: 0.12,
                attentionSeconds: 0.08),
            bytes: bytes ?? defaultBytes(role: role),
            mlxMemory: CompressedAttentionProbeMLXMemoryReceipts(
                before: CompressedAttentionProbeMLXMemorySnapshot(
                    activeBytes: 1_000,
                    cacheBytes: 2_000,
                    peakBytes: 3_000),
                after: CompressedAttentionProbeMLXMemorySnapshot(
                    activeBytes: 2_000,
                    cacheBytes: 2_500,
                    peakBytes: 4_000)),
            processRSS: CompressedAttentionProbeProcessRSS(
                residentSizeBeforeBytes: 100_000,
                residentSizeAfterBytes: 110_000,
                physicalFootprintBeforeBytes: 90_000,
                physicalFootprintAfterBytes: 105_000),
            memorySettings: memorySettings ?? CompressedAttentionProbeMemorySettings(
                memoryLimitBytes: 16 << 30,
                cacheLimitBytes: 8 << 30,
                wiredLimitBytes: 0,
                cacheResetPolicy: .preserveAcrossPair),
            power: power ?? CompressedAttentionProbePowerReceipts(
                lowPowerModeEnabledBefore: false,
                lowPowerModeEnabledAfter: false,
                powerSourceBefore: .acPower,
                powerSourceAfter: .acPower),
            thermal: thermal ?? CompressedAttentionProbeThermalReceipts(
                before: .nominal,
                after: .nominal),
            numericControls: numericControls
                ?? CompressedAttentionProbeNumericControls(
                    packedMaxAbsoluteError: 0.001,
                    packedMaxRelativeError: 0.001,
                    packedTop1TokenID: 42,
                    unpackedTop1TokenID: 42,
                    fp16MaxAbsoluteError: 0.000001,
                    fp16MaxRelativeError: 0.00001,
                    fp16Top1TokenID: 42,
                    referenceTop1TokenID: 42),
            hashes: hashes ?? CompressedAttentionProbeHashes(
                sourceKVProjectionSHA256: hashF,
                packedKVProjectionSHA256: hashA,
                inputTokenIDsSHA256: hashB,
                outputTokenIDsSHA256: hashE))
    }

    private func defaultBytes(
        role: CompressedAttentionProbeRunRole
    ) -> CompressedAttentionProbeByteReceipts {
        switch role {
        case .candidate:
            return CompressedAttentionProbeByteReceipts(
                payloadBytes: 2_048,
                scaleBytes: 1_024,
                biasBytes: 1_024,
                controlBytes: 0,
                alignmentPaddingBytes: 0,
                fp16ResidentBytes: 0,
                persistentKVBytes: 4_096,
                materializationBytes: 0,
                otherWorkspaceBytes: 1_024,
                peakTemporaryBytes: 1_024,
                totalBytes: 5_120)
        case .fp16Reference:
            return CompressedAttentionProbeByteReceipts(
                payloadBytes: 0,
                scaleBytes: 0,
                biasBytes: 0,
                controlBytes: 0,
                alignmentPaddingBytes: 0,
                fp16ResidentBytes: 4_096,
                persistentKVBytes: 4_096,
                materializationBytes: 0,
                otherWorkspaceBytes: 1_024,
                peakTemporaryBytes: 1_024,
                totalBytes: 5_120)
        }
    }

    private func defaultTiming(
        block: Int,
        position: Int
    ) -> CompressedAttentionProbeTiming {
        let start = 100 + Double(block) + Double(position) * 0.2
        return CompressedAttentionProbeTiming(
            monotonicStartSeconds: start,
            monotonicEndSeconds: start + 0.12,
            wallClockSeconds: 0.12,
            attentionSeconds: 0.08)
    }
}
