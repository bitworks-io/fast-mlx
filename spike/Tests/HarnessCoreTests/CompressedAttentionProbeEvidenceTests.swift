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

    func testAffineByteGeometryMatchesIndependentClosedFormFixture() throws {
        let plan = try CompressedAttentionProbePlan(
            operation: .materializeThenSDPA,
            contextTokens: 64,
            queryTokens: 1,
            prefillChunkTokens: 32,
            outputTokens: 16,
            stopTokenIDs: [],
            batchSize: 1,
            queryHeadCount: 8,
            kvHeadCount: 2,
            headDimension: 128,
            dtype: .float16,
            mask: .causal,
            layout: .affine(
                keyBits: 4, valueBits: 4,
                keyGroupSize: 64, valueGroupSize: 64),
            warmupRuns: 1,
            measuredRuns: 1,
            seed: 7,
            workloadNonce: "independent-byte-geometry",
            harnessGitSHA: cleanSHA,
            qualificationEvidence: false)

        let candidate = try CompressedAttentionProbeExpectedByteGeometry
            .derive(
                plan: plan,
                role: .candidate,
                operation: .materializeThenSDPA)
        XCTAssertEqual(candidate.payloadBytes, 16_384)
        XCTAssertEqual(candidate.scaleBytes, 1_024)
        XCTAssertEqual(candidate.biasBytes, 1_024)
        XCTAssertEqual(candidate.controlBytes, 0)
        XCTAssertEqual(candidate.alignmentPaddingBytes, 0)
        XCTAssertEqual(candidate.fp16ResidentBytes, 0)
        XCTAssertEqual(candidate.persistentKVBytes, 18_432)
        XCTAssertEqual(candidate.materializationBytes, 65_536)

        let reference = try CompressedAttentionProbeExpectedByteGeometry
            .derive(
                plan: plan,
                role: .fp16Reference,
                operation: .fp16SDPA)
        XCTAssertEqual(reference.payloadBytes, 0)
        XCTAssertEqual(reference.scaleBytes, 0)
        XCTAssertEqual(reference.biasBytes, 0)
        XCTAssertEqual(reference.fp16ResidentBytes, 65_536)
        XCTAssertEqual(reference.persistentKVBytes, 65_536)
        XCTAssertEqual(reference.materializationBytes, 0)
    }

    func testQualificationEvidencePreservesRawPairedRowsAndRoundTrips() throws {
        let evidence = try makeEvidence()
        let validated = try evidence.validated()
        let encoded = try JSONEncoder().encode(validated)
        let decoded = try JSONDecoder().decode(
            CompressedAttentionProbeEvidence.self,
            from: encoded)

        XCTAssertEqual(decoded, validated)
        XCTAssertEqual(
            decoded.evidenceKind,
            .checkpointAuthenticatedSyntheticGeometry)
        XCTAssertEqual(
            decoded.schemaVersion,
            CompressedAttentionProbeEvidence.schemaVersion)
        XCTAssertEqual(validated.rows.count, 6)
        XCTAssertEqual(validated.plan.contextTokens, 8_192)
        XCTAssertEqual(
            validated.rows.map(\.position.pairedBlockIndex),
            [0, 0, 1, 1, 2, 2])
        XCTAssertEqual(
            validated.rows.filter { $0.role == .candidate }.map(\.position.runPosition),
            [0, 1, 0])
        XCTAssertNil(
            String(decoding: encoded, as: UTF8.self)
                .range(of: "promotion", options: .caseInsensitive))
    }

    func testArtifactIdentityAuthenticatesCanonicalPayloadWithoutRecursion() throws {
        let plan = try makePlan()
        let model = modelIdentity()
        let package = CompressedAttentionProbePackageIdentity(
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision:
                "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
            swiftVersion: "6.0",
            harnessBuildConfiguration: "Release")
        let rows = makeRows()

        let first = try CompressedAttentionProbeEvidence.deriveArtifactID(
            plan: CompressedAttentionProbePlanIdentity(plan: plan),
            model: model,
            package: package,
            rows: rows)
        let second = try CompressedAttentionProbeEvidence.deriveArtifactID(
            plan: CompressedAttentionProbePlanIdentity(plan: plan),
            model: model,
            package: package,
            rows: rows)
        XCTAssertEqual(first, second)
        XCTAssertEqual(first.count, 64)

        var changedRows = rows
        changedRows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(
                timing: CompressedAttentionProbeTiming(
                    monotonicStartSeconds: 100,
                    monotonicEndSeconds: 100.13,
                    wallClockSeconds: 0.13,
                    attentionSeconds: 0.08)))
        let changed = try CompressedAttentionProbeEvidence.deriveArtifactID(
            plan: CompressedAttentionProbePlanIdentity(plan: plan),
            model: model,
            package: package,
            rows: changedRows)
        XCTAssertNotEqual(first, changed)
    }

    func testValidatedEvidenceRejectsPayloadTamperingWithRetainedArtifactID() throws {
        let original = try makeEvidence()
        var tamperedRows = original.rows
        tamperedRows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(
                timing: CompressedAttentionProbeTiming(
                    monotonicStartSeconds: 100,
                    monotonicEndSeconds: 100.13,
                    wallClockSeconds: 0.13,
                    attentionSeconds: 0.08)))
        let tampered = CompressedAttentionProbeEvidence(
            schemaVersion: original.schemaVersion,
            evidenceKind: original.evidenceKind,
            artifactID: original.artifactID,
            plan: original.plan,
            model: original.model,
            package: original.package,
            rows: tamperedRows)

        XCTAssertThrowsError(try tampered.validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidArtifactID)
        }
    }

    func testQualificationEvidenceRequiresTheQualifiedPackageAndReleaseBuild() throws {
        let debugPackage = CompressedAttentionProbePackageIdentity(
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision:
                "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
            swiftVersion: "6.0",
            harnessBuildConfiguration: "Debug")
        XCTAssertThrowsError(try makeEvidence(package: debugPackage)) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidIdentity("harnessBuildConfiguration"))
        }

        let unpinnedPackage = CompressedAttentionProbePackageIdentity(
            mlxSwiftVersion: "0.31.5",
            mlxSwiftLMRevision:
                "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
            swiftVersion: "6.0",
            harnessBuildConfiguration: "Release")
        XCTAssertThrowsError(try makeEvidence(package: unpinnedPackage)) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidIdentity("mlxSwiftVersion"))
        }
    }

    func testExploratoryEvidenceMayRecordADifferentValidPackage() throws {
        let plan = try CompressedAttentionProbePlan(
            operation: .swiftLMQuantizedAttention,
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
            layout: .affine(
                keyBits: 4, valueBits: 4,
                keyGroupSize: 64, valueGroupSize: 64),
            warmupRuns: 1,
            measuredRuns: 1,
            seed: 7,
            workloadNonce: "exploratory-package",
            harnessGitSHA: cleanSHA,
            qualificationEvidence: false)
        let debugPackage = CompressedAttentionProbePackageIdentity(
            mlxSwiftVersion: "0.31.5",
            mlxSwiftLMRevision: String(repeating: "b", count: 40),
            swiftVersion: "6.0",
            harnessBuildConfiguration: "Debug")

        XCTAssertNoThrow(try makeEvidence(
            plan: plan,
            package: debugPackage,
            rows: [
                row(
                    role: .candidate,
                    block: 0,
                    position: 0,
                    receipts: runReceipts(
                        role: .candidate,
                        bytes: defaultBytes(
                            role: .candidate, plan: plan))),
                row(
                    role: .fp16Reference,
                    block: 0,
                    position: 1,
                    receipts: runReceipts(
                        role: .fp16Reference,
                        bytes: defaultBytes(
                            role: .fp16Reference, plan: plan))),
            ]).validated())
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

    func testQualificationRowsRequireCompleteReceiptsAndValidIdentities() throws {
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
            qualificationEvidence: false)
        let unavailableCandidate = runReceipts(
            role: .candidate,
            bytes: defaultBytes(role: .candidate, plan: plan),
            power: CompressedAttentionProbePowerReceipts(
                lowPowerModeEnabledBefore: false,
                lowPowerModeEnabledAfter: false,
                powerSourceBefore: .unavailable,
                powerSourceAfter: .unavailable))
        let unavailableReference = runReceipts(
            role: .fp16Reference,
            bytes: defaultBytes(role: .fp16Reference, plan: plan),
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
                candidateMaxAbsoluteError: 0.0021,
                candidateMaxRelativeError: 0.001,
                candidateMaximumToleranceRatio: 1.01,
                candidateTop1Index: 42,
                candidateOracleTop1Index: 42,
                referenceMaxAbsoluteError: 0.000001,
                referenceMaxRelativeError: 0.00001,
                referenceMaximumToleranceRatio: 0.5,
                referenceTop1Index: 42,
                referenceOracleTop1Index: 42))
        rows = makeRows()
        rows[0] = row(
            role: .candidate, block: 0, position: 0, receipts: receipts)
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .candidateStructuralMismatch)
        }

        receipts = runReceipts(
            role: .fp16Reference,
            numericControls: CompressedAttentionProbeNumericControls(
                candidateMaxAbsoluteError: 0.001,
                candidateMaxRelativeError: 0.001,
                candidateMaximumToleranceRatio: 0.5,
                candidateTop1Index: 42,
                candidateOracleTop1Index: 42,
                referenceMaxAbsoluteError: 0.00048828125,
                referenceMaxRelativeError: 0.188,
                referenceMaximumToleranceRatio: 1.01,
                referenceTop1Index: 42,
                referenceOracleTop1Index: 42))
        rows = makeRows()
        rows[1] = row(
            role: .fp16Reference, block: 0, position: 1, receipts: receipts)
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .referenceControlMismatch)
        }
    }

    func testQualificationRejectsPowerLowPowerAndThermalTransitions() throws {
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
                    sourceKVTensorSHA256: hashC,
                    packedKVTensorSHA256: hashA,
                    queryTensorSHA256: hashB,
                    outputTensorSHA256: hashE)))
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
                    cacheResetPolicy: .preserveAcrossRun)))
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

    func testWorkspaceByteReceiptsMustMatchRawMLXPeakTelemetry() throws {
        let geometry = try CompressedAttentionProbeExpectedByteGeometry
            .derive(
                plan: makePlan(),
                role: .candidate,
                operation: .swiftLMQuantizedAttention)
        let tampered = CompressedAttentionProbeByteReceipts(
            payloadBytes: geometry.payloadBytes,
            scaleBytes: geometry.scaleBytes,
            biasBytes: geometry.biasBytes,
            controlBytes: geometry.controlBytes,
            alignmentPaddingBytes: geometry.alignmentPaddingBytes,
            fp16ResidentBytes: geometry.fp16ResidentBytes,
            persistentKVBytes: geometry.persistentKVBytes,
            materializationBytes: geometry.materializationBytes,
            otherWorkspaceBytes: 2_048,
            peakTemporaryBytes: 2_048,
            totalBytes: geometry.persistentKVBytes + 2_048)
        var rows = makeRows()
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(bytes: tampered))

        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidByteAccounting)
        }
    }

    func testResetPeakMayRemainBelowResidentBaselineWithoutAllocations()
        throws
    {
        let geometry = try CompressedAttentionProbeExpectedByteGeometry
            .derive(
                plan: makePlan(),
                role: .candidate,
                operation: .swiftLMQuantizedAttention)
        let noTemporaryBytes = CompressedAttentionProbeByteReceipts(
            payloadBytes: geometry.payloadBytes,
            scaleBytes: geometry.scaleBytes,
            biasBytes: geometry.biasBytes,
            controlBytes: geometry.controlBytes,
            alignmentPaddingBytes: geometry.alignmentPaddingBytes,
            fp16ResidentBytes: geometry.fp16ResidentBytes,
            persistentKVBytes: geometry.persistentKVBytes,
            materializationBytes: 0,
            otherWorkspaceBytes: 0,
            peakTemporaryBytes: 0,
            totalBytes: geometry.persistentKVBytes)
        let rawMemory = CompressedAttentionProbeMLXMemoryReceipts(
            before: CompressedAttentionProbeMLXMemorySnapshot(
                activeBytes: 1_000,
                cacheBytes: 2_000,
                peakBytes: 0),
            after: CompressedAttentionProbeMLXMemorySnapshot(
                activeBytes: 900,
                cacheBytes: 2_500,
                peakBytes: 0))
        var rows = makeRows()
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(
                bytes: noTemporaryBytes,
                mlxMemory: rawMemory))

        XCTAssertNoThrow(try makeEvidence(rows: rows).validated())

        let impossibleGrowth = CompressedAttentionProbeMLXMemoryReceipts(
            before: CompressedAttentionProbeMLXMemorySnapshot(
                activeBytes: 1_000,
                cacheBytes: 2_000,
                peakBytes: 0),
            after: CompressedAttentionProbeMLXMemorySnapshot(
                activeBytes: 2_500,
                cacheBytes: 2_500,
                peakBytes: 500))
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(
                bytes: noTemporaryBytes,
                mlxMemory: impossibleGrowth))
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidMemoryCounters)
        }

        let unstableBaseline = CompressedAttentionProbeMLXMemoryReceipts(
            before: CompressedAttentionProbeMLXMemorySnapshot(
                activeBytes: 1_000,
                cacheBytes: 2_000,
                peakBytes: 0),
            after: CompressedAttentionProbeMLXMemorySnapshot(
                activeBytes: 400,
                cacheBytes: 2_500,
                peakBytes: 500))
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(
                bytes: noTemporaryBytes,
                mlxMemory: unstableBaseline))
        XCTAssertThrowsError(try makeEvidence(rows: rows).validated()) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbeEvidenceError,
                .invalidMemoryCounters)
        }
    }

    func testCachePolicyNamesTheActualRunWidePreservationBoundary() {
        XCTAssertEqual(
            CompressedAttentionProbeCacheResetPolicy.preserveAcrossRun.rawValue,
            "preserve-across-run")
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

    func testSelfConsistentButGeometryImpossibleByteReceiptsFailClosed() throws {
        let impossibleForEightK = CompressedAttentionProbeByteReceipts(
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
        var rows = makeRows()
        rows[0] = row(
            role: .candidate,
            block: 0,
            position: 0,
            receipts: runReceipts(bytes: impossibleForEightK))

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
            qualificationEvidence: false)
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
        package: CompressedAttentionProbePackageIdentity? = nil,
        rows: [CompressedAttentionProbeRunRow]? = nil
    ) throws -> CompressedAttentionProbeEvidence {
        let evidencePlan: CompressedAttentionProbePlan
        if let plan {
            evidencePlan = plan
        } else {
            evidencePlan = try makePlan()
        }
        let planIdentity = CompressedAttentionProbePlanIdentity(
            plan: evidencePlan)
        let modelIdentity = model ?? modelIdentity()
        let packageIdentity = package
            ?? CompressedAttentionProbePackageIdentity(
                mlxSwiftVersion: "0.31.6",
                mlxSwiftLMRevision:
                    "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
                swiftVersion: "6.0",
                harnessBuildConfiguration: "Release")
        let evidenceRows = rows ?? makeRows()
        let artifactID = try CompressedAttentionProbeEvidence
            .deriveArtifactID(
                plan: planIdentity,
                model: modelIdentity,
                package: packageIdentity,
                rows: evidenceRows)
        return CompressedAttentionProbeEvidence(
            schemaVersion: CompressedAttentionProbeEvidence.schemaVersion,
            evidenceKind: .checkpointAuthenticatedSyntheticGeometry,
            artifactID: artifactID,
            plan: planIdentity,
            model: modelIdentity,
            package: packageIdentity,
            rows: evidenceRows)
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
            qualificationEvidence: true)
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
        mlxMemory: CompressedAttentionProbeMLXMemoryReceipts? = nil,
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
            mlxMemory: mlxMemory ?? CompressedAttentionProbeMLXMemoryReceipts(
                before: CompressedAttentionProbeMLXMemorySnapshot(
                    activeBytes: 1_000,
                    cacheBytes: 2_000,
                    peakBytes: 0),
                after: CompressedAttentionProbeMLXMemorySnapshot(
                    activeBytes: 2_000,
                    cacheBytes: 2_500,
                    peakBytes: 2_024)),
            processRSS: CompressedAttentionProbeProcessRSS(
                residentSizeBeforeBytes: 100_000,
                residentSizeAfterBytes: 110_000,
                physicalFootprintBeforeBytes: 90_000,
                physicalFootprintAfterBytes: 105_000),
            memorySettings: memorySettings ?? CompressedAttentionProbeMemorySettings(
                memoryLimitBytes: 16 << 30,
                cacheLimitBytes: 8 << 30,
                wiredLimitBytes: 0,
                cacheResetPolicy: .preserveAcrossRun),
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
                    candidateMaxAbsoluteError: 0.001,
                    candidateMaxRelativeError: 0.001,
                    candidateMaximumToleranceRatio: 0.5,
                    candidateTop1Index: 42,
                    candidateOracleTop1Index: 42,
                    referenceMaxAbsoluteError: 0.000001,
                    referenceMaxRelativeError: 0.00001,
                    referenceMaximumToleranceRatio: 0.5,
                    referenceTop1Index: 42,
                    referenceOracleTop1Index: 42),
            hashes: hashes ?? CompressedAttentionProbeHashes(
                sourceKVTensorSHA256: hashF,
                packedKVTensorSHA256: hashA,
                queryTensorSHA256: hashB,
                outputTensorSHA256: hashE))
    }

    private func defaultBytes(
        role: CompressedAttentionProbeRunRole,
        plan: CompressedAttentionProbePlan? = nil
    ) -> CompressedAttentionProbeByteReceipts {
        let plan = plan ?? (try! makePlan())
        let operation: CompressedAttentionProbeOperation = role == .candidate
            ? plan.operation
            : .fp16SDPA
        let geometry = try! CompressedAttentionProbeExpectedByteGeometry
            .derive(plan: plan, role: role, operation: operation)
        let otherWorkspaceBytes = 1_024
        let peakTemporaryBytes = geometry.materializationBytes
            + otherWorkspaceBytes
        return CompressedAttentionProbeByteReceipts(
            payloadBytes: geometry.payloadBytes,
            scaleBytes: geometry.scaleBytes,
            biasBytes: geometry.biasBytes,
            controlBytes: geometry.controlBytes,
            alignmentPaddingBytes: geometry.alignmentPaddingBytes,
            fp16ResidentBytes: geometry.fp16ResidentBytes,
            persistentKVBytes: geometry.persistentKVBytes,
            materializationBytes: geometry.materializationBytes,
            otherWorkspaceBytes: otherWorkspaceBytes,
            peakTemporaryBytes: peakTemporaryBytes,
            totalBytes: geometry.persistentKVBytes + peakTemporaryBytes)
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
