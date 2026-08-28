import XCTest

@testable import HarnessCore

final class QwenMTPSampledBlockTraceGateTests: XCTestCase {
    func testValidatesThreeTwoProposalBlockOutcomes() throws {
        let payload = evidence()

        XCTAssertNoThrow(try QwenMTPSampledBlockTraceGate.validate(payload))
        XCTAssertEqual(payload.cases.map(\.outcome), [.rejectFirst, .rejectSecond, .acceptAll])
        XCTAssertEqual(payload.cases.map(\.acceptedDraftCount), [0, 1, 2])
        XCTAssertEqual(payload.cases.map(\.outputTokens.count), [1, 2, 3])
    }

    func testRecomputesDecisionThroughBlockAcceptanceAndRejectsOutcomeDrift() throws {
        var payload = evidence()
        payload.cases[1].acceptedDraftCount = 0

        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(
                error as? QwenMTPSampledBlockTraceGateError,
                .decisionMismatch(outcome: .rejectSecond))
        }
    }

    func testRejectsProposalDrawDriftForSecondProposal() throws {
        var payload = evidence()
        payload.cases[2].steps[1].proposalUniform = 0.1

        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(
                error as? QwenMTPSampledBlockTraceGateError,
                .proposalDrawMismatch(outcome: .acceptAll, stepIndex: 1))
        }
    }

    func testRejectsTerminalPurposeDrift() throws {
        var payload = evidence()
        payload.cases[0].terminalDraw = .bonus(0.5)

        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(
                error as? QwenMTPSampledBlockTraceGateError,
                .decisionMismatch(outcome: .rejectFirst))
        }
    }

    func testRejectsCacheMismatchForCommittedVisibleSequence() throws {
        var payload = evidence()
        payload.cases[2].scalarCache[3].states[0].sha256 = String(repeating: "d", count: 64)

        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(
                error as? QwenMTPSampledBlockTraceGateError,
                .cacheMismatch(outcome: .acceptAll, layer: 3))
        }
    }

    func testRejectsWrongFinalCacheLengthForOutcome() throws {
        var payload = evidence()
        payload.cases[1].candidateCache[3].offset += 1
        payload.cases[1].scalarCache[3].offset += 1

        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(
                error as? QwenMTPSampledBlockTraceGateError,
                .malformedCache(outcome: .rejectSecond, layer: 3))
        }
    }

    func testRejectsSchemaSourceSamplingAndOutcomeSetDrift() throws {
        var payload = evidence()
        payload.schemaVersion += 1
        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(error as? QwenMTPSampledBlockTraceGateError, .schemaMismatch)
        }

        payload = evidence()
        payload.source.runtimeBlockSize = 2
        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(error as? QwenMTPSampledBlockTraceGateError, .sourceMismatch)
        }

        payload = evidence()
        payload.sampling.probabilityDType = "float16"
        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(error as? QwenMTPSampledBlockTraceGateError, .samplingMismatch)
        }

        payload = evidence()
        payload.cases.swapAt(0, 1)
        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(error as? QwenMTPSampledBlockTraceGateError, .outcomeSetMismatch)
        }
    }

    func testRejectsIncompleteProbabilityNonFiniteDrawAndWrongCacheTopology() throws {
        var payload = evidence()
        payload.cases[0].steps[0].targetProbabilities.removeLast()
        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(
                error as? QwenMTPSampledBlockTraceGateError,
                .vocabularyMismatch(outcome: .rejectFirst, stepIndex: 0))
        }

        payload = evidence()
        payload.cases[0].steps[0].proposalUniform = .nan
        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(
                error as? QwenMTPSampledBlockTraceGateError,
                .proposalDrawMismatch(outcome: .rejectFirst, stepIndex: 0))
        }

        payload = evidence()
        payload.cases[0].candidateCache[0].cacheType = "KVCacheSimple"
        payload.cases[0].scalarCache[0].cacheType = "KVCacheSimple"
        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(
                error as? QwenMTPSampledBlockTraceGateError,
                .malformedCache(outcome: .rejectFirst, layer: 0))
        }
    }

    func testRejectsSecondStepContextDriftAfterTheSameAcceptedFirstProposal() throws {
        var payload = evidence()
        payload.cases[2].steps[1] = step(
            index: 1,
            target: [0.5, 0.25, 0.25],
            draft: [0.5, 0.25, 0.25],
            proposal: 0,
            uniform: 0.25)
        payload.cases[2].outputTokens = [0, 0, 1]

        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validate(payload)) { error in
            XCTAssertEqual(error as? QwenMTPSampledBlockTraceGateError, .caseContextMismatch)
        }
    }

    func testCanonicalJSONLRoundTripsThroughIndependentValidator() throws {
        let payload = evidence()
        let record = record(payload: payload)
        let data = Data((try record.jsonLine() + "\n").utf8)

        XCTAssertEqual(try QwenMTPSampledBlockTraceGate.validateJSONL(data), payload)
    }

    func testJSONLRejectsOldSampledTraceSubcommand() throws {
        let record = ResultRecord(
            subcommand: QwenMTPSampledTraceGate.subcommand,
            provenance: provenance(),
            payload: evidence())
        let data = Data((try record.jsonLine() + "\n").utf8)

        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validateJSONL(data)) { error in
            XCTAssertEqual(error as? QwenMTPSampledBlockTraceGateError, .wrongSubcommand)
        }
    }

    func testJSONLRejectsDirtyHarnessIdentity() throws {
        let record = record(payload: evidence(), harnessGitSHA: String(repeating: "c", count: 40) + "-dirty")
        let data = Data((try record.jsonLine() + "\n").utf8)

        XCTAssertThrowsError(try QwenMTPSampledBlockTraceGate.validateJSONL(data)) { error in
            XCTAssertEqual(error as? QwenMTPSampledBlockTraceGateError, .invalidProvenance)
        }
    }

    private func record(
        payload: QwenMTPSampledBlockTraceEvidence,
        harnessGitSHA: String = String(repeating: "c", count: 40)
    ) -> ResultRecord<QwenMTPSampledBlockTraceEvidence> {
        ResultRecord(
            subcommand: QwenMTPSampledBlockTraceGate.subcommand,
            provenance: provenance(harnessGitSHA: harnessGitSHA),
            payload: payload)
    }

    private func provenance(
        harnessGitSHA: String = String(repeating: "c", count: 40)
    ) -> Provenance {
        Provenance(
            date: "2026-08-27T00:00:00Z",
            hardwareChip: "test-chip",
            hardwareRAMBytes: 24 * 1_024 * 1_024 * 1_024,
            hardwareOS: "test-os",
            harnessGitSHA: harnessGitSHA,
            mlxSwiftVersion: QwenMTPSampledBlockTraceGate.requiredMLXSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: QwenMTPSampledBlockTraceGate.requiredMLXSwiftLMRevision,
            modelPath: QwenMTPSampledBlockTraceGate.requiredSource.targetModelID,
            modelConfigHash: QwenMTPSampledBlockTraceGate.requiredModelConfigHash,
            modelCheckpointManifestHash: QwenMTPSampledBlockTraceGate.requiredCheckpointManifestHash,
            modelQuant: .init(bits: 4, groupSize: 64),
            corpusId: QwenMTPSampledBlockTraceGate.corpusID,
            corpusContentHash: QwenMTPSampledBlockTraceGate.requiredPromptSHA256,
            nonce: "test-nonce")
    }

    private func evidence() -> QwenMTPSampledBlockTraceEvidence {
        let promptTokenCount = 7
        return QwenMTPSampledBlockTraceEvidence(
            schemaVersion: QwenMTPSampledBlockTraceGate.schemaVersion,
            source: QwenMTPSampledBlockTraceGate.requiredSource,
            sampling: QwenMTPSampledBlockTraceGate.requiredSampling,
            cases: [
                .init(
                    outcome: .rejectFirst,
                    promptSHA256: QwenMTPSampledBlockTraceGate.requiredPromptSHA256,
                    promptTokenCount: promptTokenCount,
                    initialBonusToken: 42,
                    outputTokens: [1],
                    acceptedDraftCount: 0,
                    acceptedDraftEndIndex: nil,
                    acceptanceUniforms: [0.75],
                    terminalDraw: .residual(0.4),
                    steps: [
                        step(index: 0, target: [0.2, 0.5, 0.3], draft: [0.4, 0.4, 0.2], proposal: 0, uniform: 0.2),
                        step(index: 1, target: [0.2, 0.5, 0.3], draft: [0.4, 0.4, 0.2], proposal: 0, uniform: 0.25),
                    ],
                    bonusTargetProbabilities: padded([0.1, 0.2, 0.7]),
                    candidateCache: cache(promptTokenCount: promptTokenCount, outputTokenCount: 1),
                    scalarCache: cache(promptTokenCount: promptTokenCount, outputTokenCount: 1)),
                .init(
                    outcome: .rejectSecond,
                    promptSHA256: QwenMTPSampledBlockTraceGate.requiredPromptSHA256,
                    promptTokenCount: promptTokenCount,
                    initialBonusToken: 42,
                    outputTokens: [0, 2],
                    acceptedDraftCount: 1,
                    acceptedDraftEndIndex: 0,
                    acceptanceUniforms: [0.25, 0.9],
                    terminalDraw: .residual(0.9),
                    steps: [
                        step(index: 0, target: [0.2, 0.5, 0.3], draft: [0.4, 0.4, 0.2], proposal: 0, uniform: 0.25),
                        step(index: 1, target: [0.4, 0.2, 0.4], draft: [0.1, 0.6, 0.3], proposal: 1, uniform: 0.4),
                    ],
                    bonusTargetProbabilities: padded([0.1, 0.2, 0.7]),
                    candidateCache: cache(promptTokenCount: promptTokenCount, outputTokenCount: 2),
                    scalarCache: cache(promptTokenCount: promptTokenCount, outputTokenCount: 2)),
                .init(
                    outcome: .acceptAll,
                    promptSHA256: QwenMTPSampledBlockTraceGate.requiredPromptSHA256,
                    promptTokenCount: promptTokenCount,
                    initialBonusToken: 42,
                    outputTokens: [0, 1, 1],
                    acceptedDraftCount: 2,
                    acceptedDraftEndIndex: 1,
                    acceptanceUniforms: [0, 0],
                    terminalDraw: .bonus(0.25),
                    steps: [
                        step(index: 0, target: [0.2, 0.5, 0.3], draft: [0.4, 0.4, 0.2], proposal: 0, uniform: 0.25),
                        step(index: 1, target: [0.4, 0.2, 0.4], draft: [0.1, 0.6, 0.3], proposal: 1, uniform: 0.25),
                    ],
                    bonusTargetProbabilities: padded([0.1, 0.2, 0.7]),
                    candidateCache: cache(promptTokenCount: promptTokenCount, outputTokenCount: 3),
                    scalarCache: cache(promptTokenCount: promptTokenCount, outputTokenCount: 3)),
            ])
    }

    private func step(
        index: Int,
        target: [Double],
        draft: [Double],
        proposal: Int,
        uniform: Double
    ) -> QwenMTPSampledBlockTraceStepEvidence {
        QwenMTPSampledBlockTraceStepEvidence(
            stepIndex: index,
            proposedToken: proposal,
            proposalUniform: uniform,
            targetProbabilities: padded(target),
            draftProbabilities: padded(draft))
    }

    private func padded(_ prefix: [Double]) -> [Double] {
        prefix + Array(
            repeating: 0.0,
            count: QwenMTPSampledBlockTraceGate.requiredVocabularyCount - prefix.count)
    }

    private func cache(
        promptTokenCount: Int,
        outputTokenCount: Int
    ) -> [QwenMTPSampledTraceCacheFingerprint] {
        let finalTokenCount = promptTokenCount + 1 + outputTokenCount
        return (0 ..< QwenMTPSampledBlockTraceGate.requiredCacheLayerCount).map { layer in
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
    }
}
