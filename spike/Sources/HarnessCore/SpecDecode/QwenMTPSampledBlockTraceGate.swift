import CryptoKit
import Foundation

public enum QwenMTPSampledBlockTraceOutcome: String, Codable, Sendable, Equatable {
    case rejectFirst
    case rejectSecond
    case acceptAll
}

public enum QwenMTPSampledBlockTraceTerminalDraw: Codable, Sendable, Equatable {
    case residual(Double)
    case bonus(Double)

    var blockDraw: SampledMTPBlockTerminalDraw {
        switch self {
        case .residual(let value): .residual(value)
        case .bonus(let value): .bonus(value)
        }
    }
}

public struct QwenMTPSampledBlockTraceStepEvidence: Codable, Sendable, Equatable {
    public var stepIndex: Int
    public var proposedToken: Int
    public var proposalUniform: Double
    public var targetProbabilities: [Double]
    public var draftProbabilities: [Double]

    public init(
        stepIndex: Int,
        proposedToken: Int,
        proposalUniform: Double,
        targetProbabilities: [Double],
        draftProbabilities: [Double]
    ) {
        self.stepIndex = stepIndex
        self.proposedToken = proposedToken
        self.proposalUniform = proposalUniform
        self.targetProbabilities = targetProbabilities
        self.draftProbabilities = draftProbabilities
    }
}

public struct QwenMTPSampledBlockTraceCaseEvidence: Codable, Sendable, Equatable {
    public var outcome: QwenMTPSampledBlockTraceOutcome
    public var promptSHA256: String
    public var promptTokenCount: Int
    public var initialBonusToken: Int
    public var outputTokens: [Int]
    public var acceptedDraftCount: Int
    public var acceptedDraftEndIndex: Int?
    public var acceptanceUniforms: [Double]
    public var terminalDraw: QwenMTPSampledBlockTraceTerminalDraw
    public var steps: [QwenMTPSampledBlockTraceStepEvidence]
    public var bonusTargetProbabilities: [Double]
    public var candidateCache: [QwenMTPSampledTraceCacheFingerprint]
    public var scalarCache: [QwenMTPSampledTraceCacheFingerprint]

    public init(
        outcome: QwenMTPSampledBlockTraceOutcome,
        promptSHA256: String,
        promptTokenCount: Int,
        initialBonusToken: Int,
        outputTokens: [Int],
        acceptedDraftCount: Int,
        acceptedDraftEndIndex: Int?,
        acceptanceUniforms: [Double],
        terminalDraw: QwenMTPSampledBlockTraceTerminalDraw,
        steps: [QwenMTPSampledBlockTraceStepEvidence],
        bonusTargetProbabilities: [Double],
        candidateCache: [QwenMTPSampledTraceCacheFingerprint],
        scalarCache: [QwenMTPSampledTraceCacheFingerprint]
    ) {
        self.outcome = outcome
        self.promptSHA256 = promptSHA256
        self.promptTokenCount = promptTokenCount
        self.initialBonusToken = initialBonusToken
        self.outputTokens = outputTokens
        self.acceptedDraftCount = acceptedDraftCount
        self.acceptedDraftEndIndex = acceptedDraftEndIndex
        self.acceptanceUniforms = acceptanceUniforms
        self.terminalDraw = terminalDraw
        self.steps = steps
        self.bonusTargetProbabilities = bonusTargetProbabilities
        self.candidateCache = candidateCache
        self.scalarCache = scalarCache
    }
}

public struct QwenMTPSampledBlockTraceEvidence: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var source: QwenMTPSampledTraceSource
    public var sampling: QwenMTPSampledTraceSampling
    public var cases: [QwenMTPSampledBlockTraceCaseEvidence]

    public init(
        schemaVersion: Int,
        source: QwenMTPSampledTraceSource,
        sampling: QwenMTPSampledTraceSampling,
        cases: [QwenMTPSampledBlockTraceCaseEvidence]
    ) {
        self.schemaVersion = schemaVersion
        self.source = source
        self.sampling = sampling
        self.cases = cases
    }
}

public enum QwenMTPSampledBlockTraceGateError: Error, Sendable, Equatable {
    case schemaMismatch
    case sourceMismatch
    case samplingMismatch
    case outcomeSetMismatch
    case caseContextMismatch
    case promptMismatch(outcome: QwenMTPSampledBlockTraceOutcome)
    case invalidPromptTokenCount(outcome: QwenMTPSampledBlockTraceOutcome)
    case invalidInitialBonus(outcome: QwenMTPSampledBlockTraceOutcome)
    case stepSetMismatch(outcome: QwenMTPSampledBlockTraceOutcome)
    case vocabularyMismatch(outcome: QwenMTPSampledBlockTraceOutcome, stepIndex: Int)
    case proposalDrawMismatch(outcome: QwenMTPSampledBlockTraceOutcome, stepIndex: Int)
    case invalidBonusDistribution(outcome: QwenMTPSampledBlockTraceOutcome)
    case decisionMismatch(outcome: QwenMTPSampledBlockTraceOutcome)
    case cacheLayerCountMismatch(outcome: QwenMTPSampledBlockTraceOutcome)
    case malformedCache(outcome: QwenMTPSampledBlockTraceOutcome, layer: Int)
    case cacheMismatch(outcome: QwenMTPSampledBlockTraceOutcome, layer: Int)
    case malformedJSONL
    case wrongSubcommand
    case invalidProvenance
    case nonCanonicalJSONL
}

public enum QwenMTPSampledBlockTraceGate {
    public static let schemaVersion = 1
    public static let subcommand = "qwen-mtp-sampled-block-trace"
    public static let corpusID = "qwen35-sampled-mtp-block-v1"
    public static let requiredVocabularyCount = QwenMTPSampledTraceGate.requiredVocabularyCount
    public static let requiredCacheLayerCount = QwenMTPSampledTraceGate.requiredCacheLayerCount
    public static let requiredPrompt = QwenMTPSampledTraceGate.requiredPrompt
    public static let requiredPromptSHA256 = QwenMTPSampledTraceGate.requiredPromptSHA256
    public static let requiredModelConfigHash = QwenMTPSampledTraceGate.requiredModelConfigHash
    public static let requiredCheckpointManifestHash =
        QwenMTPSampledTraceGate.requiredCheckpointManifestHash
    public static let requiredMLXSwiftVersion = QwenMTPSampledTraceGate.requiredMLXSwiftVersion
    public static let requiredMLXSwiftLMRevision =
        QwenMTPSampledTraceGate.requiredMLXSwiftLMRevision
    public static let requiredSource = QwenMTPSampledTraceGate.requiredSource
    public static let requiredSampling = QwenMTPSampledTraceSampling(
        temperature: 1,
        topP: 1,
        topK: 0,
        minP: 0,
        repetitionPenalty: nil,
        presencePenalty: nil,
        frequencyPenalty: nil,
        probabilityDType: "float32",
        probabilityNormalization: "softmax",
        rejectionProposalSelection: "max-positive-q-minus-p-cdf-midpoint",
        proposalContext: "after-committed-initial-bonus",
        drawOrder: ["proposal-per-step", "acceptance-per-visited-step", "residual-or-bonus"])

    @discardableResult
    public static func validateJSONL(_ data: Data) throws -> QwenMTPSampledBlockTraceEvidence {
        guard data.last == 0x0a,
            data.split(separator: 0x0a, omittingEmptySubsequences: false).count == 2,
            let record = try? JSONDecoder().decode(
                ResultRecord<QwenMTPSampledBlockTraceEvidence>.self,
                from: Data(data.dropLast()))
        else {
            throw QwenMTPSampledBlockTraceGateError.malformedJSONL
        }
        guard record.subcommand == subcommand else {
            throw QwenMTPSampledBlockTraceGateError.wrongSubcommand
        }
        guard isValidProvenance(record.provenance),
            record.provenance.modelPath == requiredSource.targetModelID,
            record.provenance.corpusId == corpusID,
            record.provenance.corpusContentHash == requiredPromptSHA256,
            record.provenance.modelConfigHash == requiredModelConfigHash,
            record.provenance.modelCheckpointManifestHash == requiredCheckpointManifestHash,
            record.provenance.modelQuant == ModelQuantInfo(bits: 4, groupSize: 64)
        else {
            throw QwenMTPSampledBlockTraceGateError.invalidProvenance
        }
        _ = try validate(record.payload)
        guard Data((try record.jsonLine() + "\n").utf8) == data else {
            throw QwenMTPSampledBlockTraceGateError.nonCanonicalJSONL
        }
        return record.payload
    }

    @discardableResult
    public static func validate(
        _ evidence: QwenMTPSampledBlockTraceEvidence
    ) throws -> QwenMTPSampledBlockTraceEvidence {
        guard evidence.schemaVersion == schemaVersion else {
            throw QwenMTPSampledBlockTraceGateError.schemaMismatch
        }
        guard evidence.source == requiredSource else {
            throw QwenMTPSampledBlockTraceGateError.sourceMismatch
        }
        guard evidence.sampling == requiredSampling else {
            throw QwenMTPSampledBlockTraceGateError.samplingMismatch
        }
        guard evidence.cases.map(\.outcome) == [.rejectFirst, .rejectSecond, .acceptAll] else {
            throw QwenMTPSampledBlockTraceGateError.outcomeSetMismatch
        }
        for traceCase in evidence.cases {
            try validate(traceCase)
        }
        let first = evidence.cases[0]
        guard evidence.cases.dropFirst().allSatisfy({ traceCase in
            traceCase.promptTokenCount == first.promptTokenCount
                && traceCase.initialBonusToken == first.initialBonusToken
                && traceCase.steps[0].targetProbabilities == first.steps[0].targetProbabilities
                && traceCase.steps[0].draftProbabilities == first.steps[0].draftProbabilities
        }),
            evidence.cases[1].steps[1].targetProbabilities
                == evidence.cases[2].steps[1].targetProbabilities,
            evidence.cases[1].steps[1].draftProbabilities
                == evidence.cases[2].steps[1].draftProbabilities
        else {
            throw QwenMTPSampledBlockTraceGateError.caseContextMismatch
        }
        return evidence
    }

    private static func validate(_ traceCase: QwenMTPSampledBlockTraceCaseEvidence) throws {
        let outcome = traceCase.outcome
        guard traceCase.promptSHA256 == requiredPromptSHA256 else {
            throw QwenMTPSampledBlockTraceGateError.promptMismatch(outcome: outcome)
        }
        guard traceCase.promptTokenCount > 0 else {
            throw QwenMTPSampledBlockTraceGateError.invalidPromptTokenCount(outcome: outcome)
        }
        guard (0 ..< requiredVocabularyCount).contains(traceCase.initialBonusToken) else {
            throw QwenMTPSampledBlockTraceGateError.invalidInitialBonus(outcome: outcome)
        }
        guard traceCase.steps.count == 2,
            traceCase.steps.enumerated().allSatisfy({ $0.offset == $0.element.stepIndex })
        else {
            throw QwenMTPSampledBlockTraceGateError.stepSetMismatch(outcome: outcome)
        }

        var blockSteps = [SampledMTPBlockStep]()
        for step in traceCase.steps {
            guard isDistribution(step.targetProbabilities),
                isDistribution(step.draftProbabilities)
            else {
                throw QwenMTPSampledBlockTraceGateError.vocabularyMismatch(
                    outcome: outcome, stepIndex: step.stepIndex)
            }
            guard step.proposalUniform.isFinite,
                (0 ..< 1).contains(step.proposalUniform),
                categoricalSample(step.draftProbabilities, uniform: step.proposalUniform)
                    == step.proposedToken,
                proposalDrawMatchesPlan(step, outcome: outcome)
            else {
                throw QwenMTPSampledBlockTraceGateError.proposalDrawMismatch(
                    outcome: outcome, stepIndex: step.stepIndex)
            }
            blockSteps.append(.init(
                targetDistribution: step.targetProbabilities,
                draftDistribution: step.draftProbabilities,
                proposedToken: step.proposedToken))
        }
        guard isDistribution(traceCase.bonusTargetProbabilities) else {
            throw QwenMTPSampledBlockTraceGateError.invalidBonusDistribution(outcome: outcome)
        }

        let decision: SampledMTPBlockDecision
        do {
            decision = try SampledMTPBlockAcceptance.decide(
                steps: blockSteps,
                acceptanceUniforms: traceCase.acceptanceUniforms,
                terminalDraws: [traceCase.terminalDraw.blockDraw],
                bonusTargetDistribution: traceCase.bonusTargetProbabilities)
        } catch {
            throw QwenMTPSampledBlockTraceGateError.decisionMismatch(outcome: outcome)
        }
        guard decision.tokens == traceCase.outputTokens,
            decision.acceptedDraftCount == traceCase.acceptedDraftCount,
            decision.acceptedDraftEndIndex == traceCase.acceptedDraftEndIndex,
            decision.tokens.count == decision.acceptedDraftCount + 1,
            outcomeMatches(decision.outcome, expected: outcome)
        else {
            throw QwenMTPSampledBlockTraceGateError.decisionMismatch(outcome: outcome)
        }

        guard traceCase.candidateCache.count == requiredCacheLayerCount,
            traceCase.scalarCache.count == requiredCacheLayerCount
        else {
            throw QwenMTPSampledBlockTraceGateError.cacheLayerCountMismatch(outcome: outcome)
        }
        let finalTokenCount = traceCase.promptTokenCount + 1 + traceCase.outputTokens.count
        for layer in 0 ..< requiredCacheLayerCount {
            let candidate = traceCase.candidateCache[layer]
            let scalar = traceCase.scalarCache[layer]
            try validateCache(
                candidate, outcome: outcome, layer: layer, finalTokenCount: finalTokenCount)
            try validateCache(
                scalar, outcome: outcome, layer: layer, finalTokenCount: finalTokenCount)
            guard candidate == scalar else {
                throw QwenMTPSampledBlockTraceGateError.cacheMismatch(
                    outcome: outcome, layer: layer)
            }
        }
    }

    private static func outcomeMatches(
        _ actual: SampledMTPBlockOutcome,
        expected: QwenMTPSampledBlockTraceOutcome
    ) -> Bool {
        switch (expected, actual) {
        case (.rejectFirst, .rejected(stepIndex: 0, correctionToken: _)),
            (.rejectSecond, .rejected(stepIndex: 1, correctionToken: _)),
            (.acceptAll, .acceptedAll(bonusToken: _)):
            true
        default:
            false
        }
    }

    private static func validateCache(
        _ fingerprint: QwenMTPSampledTraceCacheFingerprint,
        outcome: QwenMTPSampledBlockTraceOutcome,
        layer: Int,
        finalTokenCount: Int
    ) throws {
        let expectedStates: [(shape: [Int], dtype: String, byteCount: Int)]
        let expectedType: String
        let expectedOffset: Int
        if (layer + 1).isMultiple(of: 4) {
            expectedType = "KVCacheSimple"
            expectedOffset = finalTokenCount
            let shape = [1, 4, finalTokenCount, 256]
            expectedStates = [
                (shape, "bfloat16", 1 * 4 * finalTokenCount * 256 * 2),
                (shape, "bfloat16", 1 * 4 * finalTokenCount * 256 * 2),
            ]
        } else {
            expectedType = "MambaCache"
            expectedOffset = 0
            expectedStates = [
                ([1, 3, 8192], "bfloat16", 49_152),
                ([1, 32, 128, 128], "float32", 2_097_152),
            ]
        }
        guard fingerprint.layerIndex == layer,
            fingerprint.cacheType == expectedType,
            fingerprint.offset == expectedOffset,
            isLowercaseHex(fingerprint.metaStateSHA256, count: 64),
            fingerprint.states.count == expectedStates.count
        else {
            throw QwenMTPSampledBlockTraceGateError.malformedCache(outcome: outcome, layer: layer)
        }
        for (index, state) in fingerprint.states.enumerated() {
            let expected = expectedStates[index]
            guard state.stateIndex == index,
                state.shape == expected.shape,
                state.dtype == expected.dtype,
                state.byteCount == expected.byteCount,
                isLowercaseHex(state.sha256, count: 64)
            else {
                throw QwenMTPSampledBlockTraceGateError.malformedCache(
                    outcome: outcome, layer: layer)
            }
        }
    }

    private static func isDistribution(_ values: [Double]) -> Bool {
        guard values.count == requiredVocabularyCount,
            values.allSatisfy({ $0.isFinite && $0 >= 0 })
        else { return false }
        return abs(values.reduce(0, +) - 1) <= 1e-12
    }

    private static func categoricalSample(_ distribution: [Double], uniform: Double) -> Int {
        var cumulative = 0.0
        var lastSupported = 0
        for (index, probability) in distribution.enumerated() {
            if probability > 0 { lastSupported = index }
            cumulative += probability
            if uniform < cumulative { return index }
        }
        return lastSupported
    }

    private static func proposalDrawMatchesPlan(
        _ step: QwenMTPSampledBlockTraceStepEvidence,
        outcome: QwenMTPSampledBlockTraceOutcome
    ) -> Bool {
        let rejectionStep: Int?
        switch outcome {
        case .rejectFirst: rejectionStep = 0
        case .rejectSecond: rejectionStep = 1
        case .acceptAll: rejectionStep = nil
        }
        guard rejectionStep == step.stepIndex else {
            return step.proposalUniform == 0.25
        }
        var selected: Int?
        var greatestDifference = 0.0
        for index in step.draftProbabilities.indices {
            let difference = step.draftProbabilities[index] - step.targetProbabilities[index]
            if difference > greatestDifference {
                greatestDifference = difference
                selected = index
            }
        }
        guard let selected, selected == step.proposedToken else { return false }
        let expectedUniform = step.draftProbabilities[..<selected].reduce(0, +)
            + step.draftProbabilities[selected] / 2
        return step.proposalUniform == expectedUniform
    }

    private static func isValidProvenance(_ provenance: Provenance) -> Bool {
        isLowercaseHex(provenance.harnessGitSHA, count: 40)
            && provenance.mlxSwiftVersion == requiredMLXSwiftVersion
            && provenance.referenceMLXVersion == nil
            && provenance.referenceMLXLMVersion == requiredMLXSwiftLMRevision
            && !provenance.hardwareChip.isEmpty
            && provenance.hardwareChip != "unknown"
            && provenance.hardwareRAMBytes > 0
            && !provenance.hardwareOS.isEmpty
            && ISO8601DateFormatter().date(from: provenance.date) != nil
            && !provenance.nonce.isEmpty
    }

    private static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        value.utf8.count == count && value.utf8.allSatisfy {
            (48 ... 57).contains($0) || (97 ... 102).contains($0)
        }
    }
}
