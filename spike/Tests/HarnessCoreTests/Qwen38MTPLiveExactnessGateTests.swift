import Foundation
import XCTest
@testable import HarnessCore

final class Qwen38MTPLiveExactnessGateTests: XCTestCase {
    private typealias Gate = Qwen38MTPLiveExactnessGate
    private typealias GateError = Qwen38MTPLiveExactnessGateError

    func testCanonicalTwoCaseRecordReturnsGDNOnLaunchBoundProof() throws {
        let proof = try Gate.validateJSONL(try recordData())

        XCTAssertEqual(proof.artifact, Qwen38MTPPerformanceScorecardGate.requiredArtifact)
        XCTAssertEqual(proof.artifactID, Gate.requiredArtifactID)
        XCTAssertEqual(proof.sourceID, Gate.requiredSourceIdentity.sourceID)
        XCTAssertTrue(proof.accepted)
        XCTAssertEqual(proof.gdnMode, .gdnOn)
        XCTAssertEqual(proof.launchBinding, evidence().launchBinding)
        XCTAssertEqual(proof.evidenceID.count, 64)
        XCTAssertTrue(proof.evidenceID.allSatisfy { $0.isHexDigit && !$0.isUppercase })
        XCTAssertNil(Qwen38MTPPerformanceScorecardGate.requiredAcceptedLiveExactnessProof)

        let second = try Gate.validateJSONL(try recordData())
        XCTAssertEqual(second, proof)

        let otherRun = try Gate.validateJSONL(try recordData(
            provenance: provenance(
                date: "2026-08-25T00:00:01Z",
                harnessGitSHA: hex40("f"))))
        XCTAssertNotEqual(
            otherRun.evidenceID,
            proof.evidenceID,
            "the proof identity must bind the complete canonical record, including provenance")
    }

    func testRejectsNonCanonicalMalformedWrongSubcommandAndWrongProvenanceJSONL() throws {
        XCTAssertThrowsError(try Gate.validateJSONL(Data())) { error in
            XCTAssertEqual(error as? GateError, .unterminatedJSONL)
        }
        XCTAssertThrowsError(try Gate.validateJSONL(Data((try evidenceRecord().jsonLine()).utf8))) { error in
            XCTAssertEqual(error as? GateError, .unterminatedJSONL)
        }
        XCTAssertThrowsError(try Gate.validateJSONL(try recordData() + recordData())) { error in
            XCTAssertEqual(error as? GateError, .invalidRecordCardinality(2))
        }

        var wrongSubcommand = evidenceRecord(subcommand: "qwen38-mtp-performance-scorecard")
        XCTAssertThrowsError(try Gate.validateJSONL(Data((try wrongSubcommand.jsonLine() + "\n").utf8))) {
            error in
            XCTAssertEqual(error as? GateError, .wrongSubcommand("qwen38-mtp-performance-scorecard"))
        }

        let nonCanonical = Data(try JSONEncoder().encode(evidenceRecord())) + Data("\n".utf8)
        XCTAssertThrowsError(try Gate.validateJSONL(nonCanonical)) { error in
            XCTAssertEqual(error as? GateError, .nonCanonicalJSONL)
        }

        wrongSubcommand = evidenceRecord(
            provenance: provenance(modelPath: "sensitive-marker/operator/model"))
        XCTAssertThrowsError(try Gate.validateJSONL(Data((try wrongSubcommand.jsonLine() + "\n").utf8))) {
            error in
            XCTAssertEqual(error as? GateError, .invalidProvenance("modelPath"))
        }
    }

    func testRejectsArtifactSourceLaunchProcessBudgetCaseTokenDecodeDraftPassthroughAndCacheDrift()
        throws
    {
        var wrongArtifact = evidence()
        wrongArtifact.artifactID = hex("A")
        XCTAssertThrowsError(try validate(wrongArtifact)) { error in
            XCTAssertEqual(error as? GateError, .invalidArtifactIdentity)
        }

        var wrongSelection = evidence()
        wrongSelection.source.selection = "qwen35_9BDepth1"
        XCTAssertThrowsError(try validate(wrongSelection)) { error in
            XCTAssertEqual(error as? GateError, .invalidSourceIdentity)
        }

        var missingFusion = evidence()
        missingFusion.gdnMode = .gdnOff
        missingFusion.launchBinding = launchBinding(mode: .gdnOff)
        missingFusion.processIsolation.gdnMode = .gdnOff
        missingFusion.processIsolation.observedEnv = .disabled
        XCTAssertThrowsError(try validate(missingFusion)) { error in
            XCTAssertEqual(error as? GateError, .invalidLaunchBinding("gdnMode"))
        }

        var forgedLaunchDigest = evidence()
        forgedLaunchDigest.launchBinding.launchDigest = hex("f")
        XCTAssertThrowsError(try validate(forgedLaunchDigest)) { error in
            XCTAssertEqual(error as? GateError, .invalidLaunchBinding("launchDigest"))
        }

        var callerSuppliedProcessHash = evidence()
        callerSuppliedProcessHash.launchBinding.processIsolationEvidenceID = hex("b")
        callerSuppliedProcessHash.launchBinding.launchDigest = Qwen38MTPPerformanceScorecardGate
            .launchDigest(
                mode: .gdnOn,
                sourceDigest: Gate.requiredSourceIdentity.sourceID,
                observedEnv: .enabled,
                processIsolationEvidenceID: hex("b"))
        XCTAssertThrowsError(try validate(callerSuppliedProcessHash)) { error in
            XCTAssertEqual(error as? GateError, .invalidLaunchBinding("processIsolationEvidenceID"))
        }

        var malformedProcess = evidence()
        malformedProcess.processIsolation.processID = 0
        XCTAssertThrowsError(try validate(malformedProcess)) { error in
            XCTAssertEqual(error as? GateError, .invalidLaunchBinding("processIsolation"))
        }

        var wrongExecutableSource = evidence()
        wrongExecutableSource.processIsolation.executableIdentitySource = .argumentVector
        wrongExecutableSource.launchBinding = launchBinding(
            mode: .gdnOn,
            process: wrongExecutableSource.processIsolation)
        XCTAssertThrowsError(try validate(wrongExecutableSource)) { error in
            XCTAssertEqual(error as? GateError, .invalidLaunchBinding("processIsolation"))
        }

        var zeroExecutableHash = evidence()
        zeroExecutableHash.processIsolation.executableSHA256 = hex("0")
        zeroExecutableHash.launchBinding = launchBinding(
            mode: .gdnOn,
            process: zeroExecutableHash.processIsolation)
        XCTAssertThrowsError(try validate(zeroExecutableHash)) { error in
            XCTAssertEqual(error as? GateError, .invalidLaunchBinding("processIsolation"))
        }

        var missingMemoryLimit = evidence()
        missingMemoryLimit.mlxMemoryBudget.memoryLimitBytes = 0
        XCTAssertThrowsError(try validate(missingMemoryLimit)) { error in
            XCTAssertEqual(error as? GateError, .invalidMemoryBudget("memoryLimitBytes"))
        }

        var cacheOverMemory = evidence()
        cacheOverMemory.mlxMemoryBudget.cacheLimitBytes =
            cacheOverMemory.mlxMemoryBudget.memoryLimitBytes + 1
        XCTAssertThrowsError(try validate(cacheOverMemory)) { error in
            XCTAssertEqual(error as? GateError, .invalidMemoryBudget("cacheLimitBytes"))
        }

        var sharedHost = evidence()
        sharedHost.hostMemoryObservation.hostUse = "shared"
        XCTAssertThrowsError(try validate(sharedHost)) { error in
            XCTAssertEqual(error as? GateError, .invalidHostMemoryObservation("hostUse"))
        }

        var unknownHost = evidence()
        unknownHost.hostMemoryObservation.hostUse = "auto"
        XCTAssertThrowsError(try validate(unknownHost)) { error in
            XCTAssertEqual(error as? GateError, .invalidHostMemoryObservation("hostUse"))
        }

        var synthesizedWired = evidence()
        synthesizedWired.hostMemoryObservation.wiredLimitProvenance = .synthesized
        XCTAssertThrowsError(try validate(synthesizedWired)) { error in
            XCTAssertEqual(error as? GateError, .invalidHostMemoryObservation("wiredLimit"))
        }

        var measuredZeroWired = evidence()
        measuredZeroWired.hostMemoryObservation.wiredLimitMB = 0
        XCTAssertNoThrow(try validate(measuredZeroWired))

        var measuredZeroWiredOverBudget = measuredZeroWired
        measuredZeroWiredOverBudget.hostMemoryObservation.memoryLimitBytes =
            measuredZeroWiredOverBudget.hostMemoryObservation.metalRecommendedMaxWorkingSetSizeBytes
        measuredZeroWiredOverBudget.mlxMemoryBudget.memoryLimitBytes =
            Int(measuredZeroWiredOverBudget.hostMemoryObservation.memoryLimitBytes)
        XCTAssertThrowsError(try validate(measuredZeroWiredOverBudget)) { error in
            XCTAssertEqual(error as? GateError, .invalidHostMemoryObservation("budget"))
        }

        var invalidMetal = evidence()
        invalidMetal.hostMemoryObservation.metalRecommendedMaxWorkingSetSizeBytes = 0
        XCTAssertThrowsError(try validate(invalidMetal)) { error in
            XCTAssertEqual(error as? GateError, .invalidHostMemoryObservation("metal"))
        }

        var reserveOverflow = evidence()
        reserveOverflow.hostMemoryObservation.osServiceReserveBytes =
            reserveOverflow.hostMemoryObservation.physicalRAMBytes
        XCTAssertThrowsError(try validate(reserveOverflow)) { error in
            XCTAssertEqual(error as? GateError, .invalidHostMemoryObservation("budget"))
        }

        let mismatchedProvenance = provenance(harnessGitSHA: String(repeating: "c", count: 40))
        let mismatchedRecord = evidenceRecord(
            provenance: mismatchedProvenance,
            payload: evidence())
        XCTAssertThrowsError(try Gate.validateJSONL(
            Data((try mismatchedRecord.jsonLine() + "\n").utf8))) { error in
            XCTAssertEqual(
                error as? GateError,
                .invalidProvenance("processIsolation.harnessGitSHA"))
        }

        var reordered = evidence()
        reordered.cases.swapAt(0, 1)
        XCTAssertThrowsError(try validate(reordered)) { error in
            XCTAssertEqual(error as? GateError, .invalidCaseOrder)
        }

        var duplicate = evidence()
        duplicate.cases[1].id = "numbers"
        XCTAssertThrowsError(try validate(duplicate)) { error in
            XCTAssertEqual(error as? GateError, .invalidCaseOrder)
        }

        var emptyTokens = evidence()
        emptyTokens.cases[0].mtpTokenIDs = []
        XCTAssertThrowsError(try validate(emptyTokens)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "tokens"))
        }

        var tokenDrift = evidence()
        tokenDrift.cases[0].mtpTokenIDs[0] += 1
        XCTAssertThrowsError(try validate(tokenDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "tokens"))
        }

        var decodedDrift = evidence()
        decodedDrift.cases[0].mtpDecodedUTF8Base64 = Data("different".utf8).base64EncodedString()
        XCTAssertThrowsError(try validate(decodedDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "decoded"))
        }

        var forgedDigest = evidence()
        forgedDigest.cases[0].decodedUTF8SHA256 = hex("B")
        XCTAssertThrowsError(try validate(forgedDigest)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "decoded"))
        }

        var noDraft = evidence()
        noDraft.cases[0].proposedDraftTokens = 0
        XCTAssertThrowsError(try validate(noDraft)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "drafts"))
        }

        var passthrough = evidence()
        passthrough.cases[0].passthroughReason = "private fallback"
        XCTAssertThrowsError(try validate(passthrough)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "passthrough"))
        }

        var cacheDrift = evidence()
        cacheDrift.cases[0].mtpCacheFingerprints[0].stateFingerprints[0].sha256 = hex("C")
        XCTAssertThrowsError(try validate(cacheDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var emptyCache = evidence()
        emptyCache.cases[0].scalarCacheFingerprints = []
        XCTAssertThrowsError(try validate(emptyCache)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var truncatedCacheState = evidence()
        truncatedCacheState.cases[0].scalarCacheFingerprints[0].stateFingerprints.removeLast()
        truncatedCacheState.cases[0].mtpCacheFingerprints[0].stateFingerprints.removeLast()
        XCTAssertThrowsError(try validate(truncatedCacheState)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongHybridTopology = evidence()
        wrongHybridTopology.cases[0].scalarCacheFingerprints[0].cacheType = "dense-attention"
        wrongHybridTopology.cases[0].mtpCacheFingerprints[0].cacheType = "dense-attention"
        XCTAssertThrowsError(try validate(wrongHybridTopology)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongRecurrentOffset = evidence()
        wrongRecurrentOffset.cases[0].scalarCacheFingerprints[0].offset = 1
        wrongRecurrentOffset.cases[0].mtpCacheFingerprints[0].offset = 1
        XCTAssertThrowsError(try validate(wrongRecurrentOffset)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongRecurrentShape = evidence()
        wrongRecurrentShape.cases[0].scalarCacheFingerprints[0]
            .stateFingerprints[0].shape = [1, 3, 10_239]
        wrongRecurrentShape.cases[0].mtpCacheFingerprints[0]
            .stateFingerprints[0].shape = [1, 3, 10_239]
        XCTAssertThrowsError(try validate(wrongRecurrentShape)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongRecurrentDType = evidence()
        wrongRecurrentDType.cases[0].scalarCacheFingerprints[0]
            .stateFingerprints[1].dtype = "bfloat16"
        wrongRecurrentDType.cases[0].mtpCacheFingerprints[0]
            .stateFingerprints[1].dtype = "bfloat16"
        XCTAssertThrowsError(try validate(wrongRecurrentDType)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongDenseGeometry = evidence()
        wrongDenseGeometry.cases[0].scalarCacheFingerprints[3]
            .stateFingerprints[0].shape = [1, 4, 14, 255]
        wrongDenseGeometry.cases[0].mtpCacheFingerprints[3]
            .stateFingerprints[0].shape = [1, 4, 14, 255]
        XCTAssertThrowsError(try validate(wrongDenseGeometry)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var wrongDenseByteCount = evidence()
        wrongDenseByteCount.cases[0].scalarCacheFingerprints[3]
            .stateFingerprints[0].byteCount -= 2
        wrongDenseByteCount.cases[0].mtpCacheFingerprints[3]
            .stateFingerprints[0].byteCount -= 2
        XCTAssertThrowsError(try validate(wrongDenseByteCount)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "cache"))
        }

        var promptDrift = evidence()
        promptDrift.cases[0].promptSHA256 = hex("e")
        XCTAssertThrowsError(try validate(promptDrift)) { error in
            XCTAssertEqual(error as? GateError, .invalidCase("numbers", "workload"))
        }
    }

    func testRejectsIncompleteOrUnboundRunProvenance() throws {
        let invalidRows: [Provenance] = [
            provenance(harnessGitSHA: String(repeating: "0", count: 40)),
            provenance(harnessGitSHA: hex40("e") + "-dirty"),
            provenance(hardwareRAMBytes: 24 * 1024 * 1024 * 1024),
            provenance(hardwareChip: ""),
            provenance(hardwareOS: ""),
            provenance(mlxSwiftVersion: "unknown"),
            provenance(date: "not-a-date"),
        ]
        for row in invalidRows {
            XCTAssertThrowsError(
                try Gate.validateJSONL(try recordData(provenance: row)),
                "invalid live-run provenance must never yield an accepted proof")
        }
    }

    func testLegacySchemaOneUnboundProofCannotAuthorizeFusionOnPerformance() throws {
        var legacy = evidence()
        legacy.schemaVersion = 1
        XCTAssertThrowsError(try validate(legacy)) { error in
            XCTAssertEqual(error as? GateError, .schemaVersionMismatch(1))
        }
    }

    func testCanonicalLiveProofCanAuthorizeFusionOnPerformanceWhenUsedAsTrustedAuthority()
        throws
    {
        let proof = try Gate.validateJSONL(try recordData())

        XCTAssertEqual(proof.gdnMode, .gdnOn)
        XCTAssertEqual(proof.launchBinding, evidence().launchBinding)
        XCTAssertNoThrow(try Qwen38MTPPerformanceScorecardGate.validateAuthority(
            scorecardAuthority(liveProof: proof)))
    }

    private func validate(
        _ evidence: Qwen38MTPLiveExactnessEvidence
    ) throws -> Qwen38MTPPerformanceScorecardLiveExactnessProof {
        try Gate.validateJSONL(Data((try evidenceRecord(payload: evidence).jsonLine() + "\n").utf8))
    }

    private func recordData(provenance: Provenance? = nil) throws -> Data {
        let recordProvenance = provenance ?? self.provenance()
        var payload = evidence()
        payload.processIsolation.harnessGitSHA = recordProvenance.harnessGitSHA
        payload.launchBinding = launchBinding(mode: .gdnOn, process: payload.processIsolation)
        return Data((try evidenceRecord(
            provenance: recordProvenance,
            payload: payload).jsonLine() + "\n").utf8)
    }

    private func evidenceRecord(
        subcommand: String = Gate.subcommand,
        provenance suppliedProvenance: Provenance? = nil,
        payload suppliedPayload: Qwen38MTPLiveExactnessEvidence? = nil
    ) -> ResultRecord<Qwen38MTPLiveExactnessEvidence> {
        ResultRecord(
            subcommand: subcommand,
            provenance: suppliedProvenance ?? provenance(),
            payload: suppliedPayload ?? evidence())
    }

    private func evidence() -> Qwen38MTPLiveExactnessEvidence {
        Qwen38MTPLiveExactnessEvidence(
            schemaVersion: Gate.schemaVersion,
            artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
            artifactID: Gate.requiredArtifactID,
            source: Gate.requiredSourceIdentity,
            gdnMode: .gdnOn,
            launchBinding: launchBinding(mode: .gdnOn),
            processIsolation: processIsolation(),
            mlxMemoryBudget: .init(
                memoryLimitBytes: 220 * 1024 * 1024 * 1024,
                cacheLimitBytes: 48 * 1024 * 1024 * 1024),
            hostMemoryObservation: hostMemoryObservation(),
            cases: [
                caseEvidence(id: "numbers", text: " 13, 17", tokenBase: 100),
                caseEvidence(id: "sentence", text: " the automated one.", tokenBase: 200),
            ])
    }

    private func caseEvidence(
        id: String,
        text: String,
        tokenBase: Int
    ) -> Qwen38MTPLiveExactnessCaseEvidence {
        let bytes = Data(text.utf8)
        let cache = (0 ..< Gate.requiredCacheLayerCount).map {
            cacheFingerprint(layer: $0, digest: hex("1"))
        }
        return Qwen38MTPLiveExactnessCaseEvidence(
            id: id,
            promptSHA256: Gate.requiredCasesByID[id]!.promptSHA256,
            maxTokens: Gate.requiredCasesByID[id]!.maxTokens,
            scalarTokenIDs: [tokenBase, tokenBase + 1],
            mtpTokenIDs: [tokenBase, tokenBase + 1],
            scalarDecodedUTF8Base64: bytes.base64EncodedString(),
            mtpDecodedUTF8Base64: bytes.base64EncodedString(),
            decodedUTF8SHA256: Gate.sha256Hex(bytes),
            proposedDraftTokens: 3,
            acceptedDraftTokens: 2,
            passthroughReason: nil,
            scalarCacheFingerprints: cache,
            mtpCacheFingerprints: cache)
    }

    private func cacheFingerprint(
        layer: Int,
        digest: String
    ) -> Qwen38MTPLiveExactnessCacheFingerprint {
        let isDense = (layer + 1).isMultiple(of: 4)
        return Qwen38MTPLiveExactnessCacheFingerprint(
            layerIndex: layer,
            cacheType: isDense ? "dense-attention" : "recurrent-mamba",
            offset: isDense ? 14 : 0,
            metaStateSHA256: hex("d"),
            stateFingerprints: (0 ..< 2).map { state in
                let shape = isDense
                    ? [1, 4, 14, 256]
                    : (state == 0 ? [1, 3, 10_240] : [1, 48, 128, 128])
                let dtype = isDense || state == 0 ? "bfloat16" : "float32"
                let byteCount = shape.reduce(1, *) * (dtype == "float32" ? 4 : 2)
                return Qwen38MTPLiveExactnessArrayFingerprint(
                    stateIndex: state,
                    shape: shape,
                    dtype: dtype,
                    byteCount: byteCount,
                    sha256: digest)
            })
    }

    private func provenance(
        modelPath: String = Qwen38MTPLiveExactnessGate.modelPathSentinel,
        date: String = "2026-08-25T00:00:00Z",
        hardwareChip: String = "Apple M3 Ultra",
        hardwareRAMBytes: UInt64 = Qwen38MTPPerformanceScorecardGate.requiredRAMBytes,
        hardwareOS: String = "macOS 26.0",
        harnessGitSHA: String = String(repeating: "e", count: 40),
        mlxSwiftVersion: String = "0.31.6"
    ) -> Provenance {
        Provenance(
            date: date,
            hardwareChip: hardwareChip,
            hardwareRAMBytes: hardwareRAMBytes,
            hardwareOS: hardwareOS,
            harnessGitSHA: harnessGitSHA,
            mlxSwiftVersion: mlxSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelPath: modelPath,
            modelConfigHash: Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetConfigSHA256,
            modelCheckpointManifestHash:
                Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusId: nil,
            corpusContentHash: nil,
            nonce: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID)
    }

    private func hex(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    private func hex40(_ character: Character) -> String {
        String(repeating: String(character), count: 40)
    }

    private func processIsolation() -> Qwen38MTPLiveExactnessProcessIsolationEvidence {
        Qwen38MTPLiveExactnessProcessIsolationEvidence(
            processID: 44_001,
            parentProcessID: 44_000,
            processStartUptimeNanoseconds: 123_456_789,
            bootTimeUnixSeconds: 1_777_000_000,
            executableIdentitySource: .procPIDPath,
            executableSHA256: hex("6"),
            harnessGitSHA: String(repeating: "e", count: 40),
            sourceID: Gate.requiredSourceIdentity.sourceID,
            gdnMode: .gdnOn,
            observedEnv: .enabled)
    }

    private func launchBinding(
        mode: Qwen38MTPPerformanceScorecardGDNMode,
        process: Qwen38MTPLiveExactnessProcessIsolationEvidence? = nil
    ) -> Qwen38MTPPerformanceScorecardLaunchBinding {
        let observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv =
            mode == .gdnOn ? .enabled : .disabled
        let processIsolationEvidenceID = Gate.processIsolationEvidenceID(
            for: process ?? processIsolation())
        return Qwen38MTPPerformanceScorecardLaunchBinding(
            mode: mode,
            sourceDigest: Gate.requiredSourceIdentity.sourceID,
            observedEnv: observedEnv,
            processIsolationEvidenceID: processIsolationEvidenceID,
            launchDigest: Qwen38MTPPerformanceScorecardGate.launchDigest(
                mode: mode,
                sourceDigest: Gate.requiredSourceIdentity.sourceID,
                observedEnv: observedEnv,
                processIsolationEvidenceID: processIsolationEvidenceID))
    }

    private func hostMemoryObservation() -> Qwen38MTPLiveExactnessHostMemoryObservation {
        let gib = UInt64(1024 * 1024 * 1024)
        return Qwen38MTPLiveExactnessHostMemoryObservation(
            hostUse: "dedicated-serving",
            hostUseSource: "operator-assertion",
            hostUsePolicyVersion: Gate.requiredHostUsePolicyVersion,
            physicalRAMBytes: 256 * gib,
            wiredLimitMB: 245_760,
            wiredLimitProvenance: .measured,
            metalRecommendedMaxWorkingSetSizeBytes: 245_760 * 1024 * 1024,
            metalCurrentAllocatedSizeBytes: 2 * gib,
            memoryLimitBytes: 220 * gib,
            cacheLimitBytes: 48 * gib,
            reservedKVBytes: 40 * gib,
            reservedIOBytes: 2 * gib,
            reservedPrefetchBytes: 4 * gib,
            osServiceReserveBytes: 8 * gib)
    }

    private func scorecardAuthority(
        liveProof: Qwen38MTPPerformanceScorecardLiveExactnessProof
    ) -> Qwen38MTPPerformanceScorecardAuthorityBundle {
        let sourceID = Gate.requiredSourceIdentity.sourceID
        return Qwen38MTPPerformanceScorecardAuthorityBundle(
            acceptedLiveExactnessProof: liveProof,
            trustedEngineIdentities: .init(
                candidate: .init(
                    label: Qwen38MTPPerformanceScorecardGate.modelArtifactLabel,
                    executionMode: .exactMTP,
                    artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
                    executionDigest: Qwen38MTPPerformanceScorecardGate.promptSHA256(
                        "generic exact mtp execution identity"),
                    sourceDigest: sourceID,
                    gdnMode: .gdnOn,
                    launchBinding: liveProof.launchBinding!),
                reference: .init(
                    label: Qwen38MTPPerformanceScorecardGate.modelArtifactLabel,
                    executionMode: .scalar,
                    artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
                    executionDigest: Qwen38MTPPerformanceScorecardGate.promptSHA256(
                        "generic scalar execution identity"),
                    sourceDigest: sourceID,
                    gdnMode: .gdnOn,
                    launchBinding: scorecardLaunchBinding(
                        mode: .gdnOn,
                        sourceDigest: sourceID,
                        processIsolationEvidenceID: hex("5")))),
            trustedRunIdentity: .init(
                measurementClass: Qwen38MTPPerformanceScorecardGate.measurementClass,
                hardwareChip: "generic-heavy-chip",
                hardwareRAMBytes: Qwen38MTPPerformanceScorecardGate.requiredRAMBytes,
                hardwareOSBuild: "generic-os-build-2026-08-24",
                hostIdentityDigest: Qwen38MTPPerformanceScorecardGate.promptSHA256(
                    "generic dedicated heavy host identity"),
                harnessGitSHA: String(repeating: "1", count: 40),
                candidateMLXSwiftVersion: "generic-mlx-swift-framework-1",
                referenceMLXVersion: nil,
                referenceMLXLMVersion: nil,
                modelLabel: Qwen38MTPPerformanceScorecardGate.modelArtifactLabel,
                modelConfigHash:
                    Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetConfigSHA256,
                modelCheckpointManifestHash:
                    Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetTensorManifestSHA256,
                modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
                corpusID: Qwen38MTPPerformanceScorecardGate.requiredWorkload.id,
                corpusContentHash:
                    Qwen38MTPPerformanceScorecardGate.requiredWorkload.contentSHA256))
    }

    private func scorecardLaunchBinding(
        mode: Qwen38MTPPerformanceScorecardGDNMode,
        sourceDigest: String,
        processIsolationEvidenceID: String
    ) -> Qwen38MTPPerformanceScorecardLaunchBinding {
        let observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv =
            mode == .gdnOn ? .enabled : .disabled
        return Qwen38MTPPerformanceScorecardLaunchBinding(
            mode: mode,
            sourceDigest: sourceDigest,
            observedEnv: observedEnv,
            processIsolationEvidenceID: processIsolationEvidenceID,
            launchDigest: Qwen38MTPPerformanceScorecardGate.launchDigest(
                mode: mode,
                sourceDigest: sourceDigest,
                observedEnv: observedEnv,
                processIsolationEvidenceID: processIsolationEvidenceID))
    }
}
