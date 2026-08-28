import XCTest

@testable import HarnessCore

final class QwenMTPSampledTraceGateTests: XCTestCase {
    func testValidatesAcceptedAndRejectedModelBackedCases() throws {
        XCTAssertNoThrow(try QwenMTPSampledTraceGate.validate(evidence()))
    }

    func testRejectsProposalDrawDrift() throws {
        var payload = evidence()
        payload.cases[0].proposalUniform = 0.2

        XCTAssertThrowsError(try QwenMTPSampledTraceGate.validate(payload))
    }

    func testRejectsAcceptanceDrawPlanDriftEvenWhenBranchStillRejects() throws {
        var payload = evidence()
        payload.cases[1].acceptanceUniform = 0.99

        XCTAssertThrowsError(try QwenMTPSampledTraceGate.validate(payload))
    }

    func testRejectsNonFiniteProposalUniform() throws {
        var payload = evidence()
        payload.cases[0].proposalUniform = .nan

        XCTAssertThrowsError(try QwenMTPSampledTraceGate.validate(payload)) { error in
            XCTAssertEqual(
                error as? QwenMTPSampledTraceGateError,
                .drawPlanMismatch(branch: .accepted))
        }
    }

    func testRejectsCacheMismatch() throws {
        var payload = evidence()
        payload.cases[1].candidateCache[0].states[0].sha256 =
            String(repeating: "c", count: 64)

        XCTAssertThrowsError(try QwenMTPSampledTraceGate.validate(payload)) { error in
            XCTAssertEqual(
                error as? QwenMTPSampledTraceGateError,
                .cacheMismatch(branch: .rejected, layer: 0))
        }
    }

    func testRejectsIncompleteProbabilityVectors() throws {
        var payload = evidence()
        payload.cases[0].targetProbabilities.removeLast()

        XCTAssertThrowsError(try QwenMTPSampledTraceGate.validate(payload)) { error in
            XCTAssertEqual(
                error as? QwenMTPSampledTraceGateError,
                .vocabularyMismatch(branch: .accepted))
        }
    }

    func testRejectsWrongQwenHybridCacheTopology() throws {
        var payload = evidence()
        payload.cases[0].candidateCache[0].cacheType = "KVCacheSimple"
        payload.cases[0].scalarCache[0].cacheType = "KVCacheSimple"

        XCTAssertThrowsError(try QwenMTPSampledTraceGate.validate(payload))
    }

    func testCanonicalJSONLRoundTripsThroughIndependentValidator() throws {
        let payload = evidence()
        let record = record(payload: payload)
        let data = Data((try record.jsonLine() + "\n").utf8)

        XCTAssertEqual(try QwenMTPSampledTraceGate.validateJSONL(data), payload)
    }

    func testJSONLRejectsDirtyHarnessIdentity() throws {
        let record = record(payload: evidence(), harnessGitSHA: String(repeating: "c", count: 40) + "-dirty")
        let data = Data((try record.jsonLine() + "\n").utf8)

        XCTAssertThrowsError(try QwenMTPSampledTraceGate.validateJSONL(data))
    }

    func testJSONLRejectsWrongCheckpointManifest() throws {
        let record = record(payload: evidence(), checkpointManifestHash: "0000000000000000")
        let data = Data((try record.jsonLine() + "\n").utf8)

        XCTAssertThrowsError(try QwenMTPSampledTraceGate.validateJSONL(data))
    }

    private func record(
        payload: QwenMTPSampledTraceEvidence,
        harnessGitSHA: String = String(repeating: "c", count: 40),
        checkpointManifestHash: String = "db2b2480a8525194"
    ) -> ResultRecord<QwenMTPSampledTraceEvidence> {
        ResultRecord(
            subcommand: QwenMTPSampledTraceGate.subcommand,
            provenance: Provenance(
                date: "2026-08-26T00:00:00Z",
                hardwareChip: "test-chip",
                hardwareRAMBytes: 24 * 1_024 * 1_024 * 1_024,
                hardwareOS: "test-os",
                harnessGitSHA: harnessGitSHA,
                mlxSwiftVersion: "0.31.6",
                referenceMLXVersion: nil,
                referenceMLXLMVersion: "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
                modelPath: QwenMTPSampledTraceGate.requiredSource.targetModelID,
                modelConfigHash: "5a99be4477ebdac8",
                modelCheckpointManifestHash: checkpointManifestHash,
                modelQuant: .init(bits: 4, groupSize: 64),
                corpusId: QwenMTPSampledTraceGate.corpusID,
                corpusContentHash: QwenMTPSampledTraceGate.requiredPromptSHA256,
                nonce: "test-nonce"),
            payload: payload)
    }

    private func evidence() -> QwenMTPSampledTraceEvidence {
        let count = QwenMTPSampledTraceGate.requiredVocabularyCount
        var target = Array(repeating: 0.0, count: count)
        var draft = Array(repeating: 0.0, count: count)
        target[0] = 0.5
        target[1] = 0.5
        draft[0] = 0.4
        draft[1] = 0.6
        let promptTokenCount = 7
        let finalTokenCount = promptTokenCount + 2
        let cache = (0 ..< QwenMTPSampledTraceGate.requiredCacheLayerCount).map { layer in
            if (layer + 1).isMultiple(of: 4) {
                return QwenMTPSampledTraceCacheFingerprint(
                    layerIndex: layer,
                    cacheType: "KVCacheSimple",
                    offset: finalTokenCount,
                    metaStateSHA256: String(repeating: "a", count: 64),
                    states: [0, 1].map { stateIndex in
                        .init(
                            stateIndex: stateIndex,
                            shape: [1, 4, finalTokenCount, 256],
                            dtype: "bfloat16",
                            byteCount: 1 * 4 * finalTokenCount * 256 * 2,
                            sha256: String(repeating: "b", count: 64))
                    })
            }
            return QwenMTPSampledTraceCacheFingerprint(
                layerIndex: layer,
                cacheType: "MambaCache",
                offset: 0,
                metaStateSHA256: String(repeating: "a", count: 64),
                states: [
                    .init(
                        stateIndex: 0,
                        shape: [1, 3, 8192],
                        dtype: "bfloat16",
                        byteCount: 49_152,
                        sha256: String(repeating: "b", count: 64)),
                    .init(
                        stateIndex: 1,
                        shape: [1, 32, 128, 128],
                        dtype: "float32",
                        byteCount: 2_097_152,
                        sha256: String(repeating: "b", count: 64)),
                ])
        }
        return QwenMTPSampledTraceEvidence(
            schemaVersion: QwenMTPSampledTraceGate.schemaVersion,
            source: QwenMTPSampledTraceGate.requiredSource,
            sampling: QwenMTPSampledTraceGate.requiredSampling,
            cases: [
                .init(
                    branch: .accepted,
                    promptSHA256: QwenMTPSampledTraceGate.requiredPromptSHA256,
                    promptTokenCount: promptTokenCount,
                    bonusToken: 2,
                    proposedToken: 0,
                    emittedToken: 0,
                    proposalUniform: 0.25,
                    acceptanceUniform: 0,
                    residualUniform: nil,
                    targetProbabilities: target,
                    draftProbabilities: draft,
                    candidateCache: cache,
                    scalarCache: cache),
                .init(
                    branch: .rejected,
                    promptSHA256: QwenMTPSampledTraceGate.requiredPromptSHA256,
                    promptTokenCount: promptTokenCount,
                    bonusToken: 2,
                    proposedToken: 1,
                    emittedToken: 0,
                    proposalUniform: 0.7,
                    acceptanceUniform: Double(1).nextDown,
                    residualUniform: 0.75,
                    targetProbabilities: target,
                    draftProbabilities: draft,
                    candidateCache: cache,
                    scalarCache: cache),
            ])
    }
}
