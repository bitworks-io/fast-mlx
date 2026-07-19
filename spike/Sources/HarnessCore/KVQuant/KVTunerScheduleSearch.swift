import Foundation

public enum KVTunerScheduleSearchError: Error, Equatable, Sendable {
    case invalidAnalysis
    case invalidTargetPairBitTotal
    case invalidCandidateLimit
    case arithmeticOverflow
    case noFeasibleSchedule
    case candidateLimitExceeded(limit: Int)
    case invalidCandidate
    case incompleteEvaluations(expected: Int, actual: Int)
    case duplicateEvaluation(Int)
    case invalidEvaluation(Int)
    case incomparableEvaluations
    case mixedEvaluationEnvironments
    case runtimePolicyMismatch(Int)
    case invalidSearchProtocol(String)
    case sensitivityArtifactMismatch
    case candidateArtifactMismatch
    case calibrationIdentityMismatch
    case searchArtifactMismatch
    case invalidSchedule(KVTunerScheduleError)
}

public enum KVTunerSearchArtifactError: Error, Equatable, Sendable {
    case unsupportedSchema(Int)
    case invalidProtocol(String)
    case invalidIdentity(String)
    case sensitivityArtifactMismatch
    case calibrationManifestMismatch
    case candidateListMismatch
    case evaluationMismatch
    case selectedCandidateMismatch
}

public struct KVTunerScheduleCandidate: Codable, Equatable, Sendable {
    public let ordinal: Int
    public let analysisSHA256: String
    public let totalPairBits: Int
    public let meanAttentionOutputError: Double
    public let layers: [KVLayerPrecision]

    public init(
        ordinal: Int,
        analysisSHA256: String,
        totalPairBits: Int,
        meanAttentionOutputError: Double,
        layers: [KVLayerPrecision]
    ) {
        self.ordinal = ordinal
        self.analysisSHA256 = analysisSHA256
        self.totalPairBits = totalPairBits
        self.meanAttentionOutputError = meanAttentionOutputError
        self.layers = layers
    }
}

public struct KVTunerCandidateEvaluation: Codable, Equatable, Sendable {
    public let candidateOrdinal: Int
    public let candidateSHA256: String
    public let runtimePolicySHA256: String
    public let environmentSHA256: String
    public let correctCount: Int
    public let totalCount: Int
    public let outputSHA256: String

    public init(
        candidateOrdinal: Int,
        candidateSHA256: String,
        runtimePolicySHA256: String,
        environmentSHA256: String,
        correctCount: Int,
        totalCount: Int,
        outputSHA256: String
    ) {
        self.candidateOrdinal = candidateOrdinal
        self.candidateSHA256 = candidateSHA256
        self.runtimePolicySHA256 = runtimePolicySHA256
        self.environmentSHA256 = environmentSHA256
        self.correctCount = correctCount
        self.totalCount = totalCount
        self.outputSHA256 = outputSHA256
    }
}

/// Exact generation and scoring contract used to compare every grouped-search candidate.
/// The prompt bytes and few-shot sample sequence are authenticated by the calibration manifest;
/// these fields pin the remaining lm-eval behavior that can change a candidate's score.
public struct KVTunerSearchEvaluationProtocol:
    Codable, Equatable, Sendable
{
    public static let canonical = KVTunerSearchEvaluationProtocol(
        id: "lm-eval-gsm8k-v3-first200-four-shot-v1",
        promptCount: 200,
        fewShotCount: 4,
        promptMode: "raw-completion-no-chat-template-v1",
        maxGeneratedTokens: 256,
        stopSequences: ["Question:", "</s>", "<|im_end|>"],
        doSample: false,
        temperature: 0,
        scoringFilterID: "exact-match-flexible-extract-v3")

    public var id: String
    public var promptCount: Int
    public var fewShotCount: Int
    public var promptMode: String
    public var maxGeneratedTokens: Int
    public var stopSequences: [String]
    public var doSample: Bool
    public var temperature: Double
    public var scoringFilterID: String

    public init(
        id: String,
        promptCount: Int,
        fewShotCount: Int,
        promptMode: String,
        maxGeneratedTokens: Int,
        stopSequences: [String],
        doSample: Bool,
        temperature: Double,
        scoringFilterID: String
    ) {
        self.id = id
        self.promptCount = promptCount
        self.fewShotCount = fewShotCount
        self.promptMode = promptMode
        self.maxGeneratedTokens = maxGeneratedTokens
        self.stopSequences = stopSequences
        self.doSample = doSample
        self.temperature = temperature
        self.scoringFilterID = scoringFilterID
    }
}

/// Complete durable evidence for grouped exhaustive search. Runtime schedules bind the SHA-256
/// of these exact bytes, closing the provenance gap between measured sensitivity and the selected
/// per-layer policy.
public struct KVTunerSearchArtifact: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var searchMode: String
    public var evaluationProtocol: KVTunerSearchEvaluationProtocol
    public var sourceSensitivityArtifactSHA256: String
    public var promptManifestSHA256: String
    public var modelConfigHash: String
    public var modelConfigSHA256: String
    public var checkpointManifestHash: String
    public var checkpointContentSHA256: String
    public var tokenizerSHA256: String
    public var groupSize: Int
    public var targetPairBitTotal: Int
    public var seed: UInt64
    public var candidateListSHA256: String
    public var candidates: [KVTunerScheduleCandidate]
    public var evaluations: [KVTunerCandidateEvaluation]
    public var selectedCandidateOrdinal: Int

    public init(
        schemaVersion: Int,
        searchMode: String,
        evaluationProtocol: KVTunerSearchEvaluationProtocol,
        sourceSensitivityArtifactSHA256: String,
        promptManifestSHA256: String,
        modelConfigHash: String,
        modelConfigSHA256: String,
        checkpointManifestHash: String,
        checkpointContentSHA256: String,
        tokenizerSHA256: String,
        groupSize: Int,
        targetPairBitTotal: Int,
        seed: UInt64,
        candidateListSHA256: String,
        candidates: [KVTunerScheduleCandidate],
        evaluations: [KVTunerCandidateEvaluation],
        selectedCandidateOrdinal: Int
    ) {
        self.schemaVersion = schemaVersion
        self.searchMode = searchMode
        self.evaluationProtocol = evaluationProtocol
        self.sourceSensitivityArtifactSHA256 =
            sourceSensitivityArtifactSHA256
        self.promptManifestSHA256 = promptManifestSHA256
        self.modelConfigHash = modelConfigHash
        self.modelConfigSHA256 = modelConfigSHA256
        self.checkpointManifestHash = checkpointManifestHash
        self.checkpointContentSHA256 = checkpointContentSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.groupSize = groupSize
        self.targetPairBitTotal = targetPairBitTotal
        self.seed = seed
        self.candidateListSHA256 = candidateListSHA256
        self.candidates = candidates
        self.evaluations = evaluations
        self.selectedCandidateOrdinal = selectedCandidateOrdinal
    }

    @discardableResult
    public func validated(
        sensitivityArtifact: KVTunerSensitivityArtifact,
        exactSensitivityArtifactData: Data,
        calibrationManifest: KVTunerCalibrationManifest,
        exactCalibrationManifestData: Data,
        exactCandidateEvaluationArtifactData: [Data],
        candidateRuntimePolicies: [KVTunerCandidateRuntimePolicy],
        exactModelConfigData: Data,
        eosTokenID: Int,
        decodeTokenIDs: ([Int]) throws -> String
    ) throws -> KVTunerScheduleCandidate {
        guard schemaVersion == 3 else {
            throw KVTunerSearchArtifactError.unsupportedSchema(schemaVersion)
        }
        guard searchMode == "exhaustive-grouped-v1" else {
            throw KVTunerSearchArtifactError.invalidProtocol("searchMode")
        }
        guard evaluationProtocol == .canonical else {
            throw KVTunerSearchArtifactError.invalidProtocol(
                "evaluationProtocol")
        }
        guard seed == KVTunerScheduleSearch.requiredFewShotSeed else {
            throw KVTunerSearchArtifactError.invalidProtocol("seed")
        }
        for (field, digest) in [
            ("sourceSensitivityArtifactSHA256",
             sourceSensitivityArtifactSHA256),
            ("promptManifestSHA256", promptManifestSHA256),
            ("modelConfigSHA256", modelConfigSHA256),
            ("checkpointContentSHA256", checkpointContentSHA256),
            ("tokenizerSHA256", tokenizerSHA256),
            ("candidateListSHA256", candidateListSHA256),
        ] {
            guard Self.isLowercaseHex(digest, length: 64) else {
                throw KVTunerSearchArtifactError.invalidIdentity(field)
            }
        }
        guard Self.isIdentityDigest(modelConfigHash),
            Self.isIdentityDigest(checkpointManifestHash),
            [64, 128].contains(groupSize),
            targetPairBitTotal > 0
        else {
            throw KVTunerSearchArtifactError.invalidIdentity(
                "model-or-search-geometry")
        }

        let decodedSensitivity: KVTunerSensitivityArtifact
        do {
            decodedSensitivity = try JSONDecoder().decode(
                KVTunerSensitivityArtifact.self,
                from: exactSensitivityArtifactData)
            guard decodedSensitivity == sensitivityArtifact else {
                throw KVTunerSearchArtifactError
                    .sensitivityArtifactMismatch
            }
            _ = try decodedSensitivity.validated(
                calibrationManifest: calibrationManifest,
                exactCalibrationManifestData:
                    exactCalibrationManifestData)
        } catch let error as KVTunerSearchArtifactError {
            throw error
        } catch {
            throw KVTunerSearchArtifactError.sensitivityArtifactMismatch
        }
        guard sourceSensitivityArtifactSHA256
                == sha256Hex(exactSensitivityArtifactData),
            promptManifestSHA256
                == sha256Hex(exactCalibrationManifestData),
            modelConfigHash == decodedSensitivity.modelConfigHash,
            modelConfigSHA256 == decodedSensitivity.modelConfigSHA256,
            checkpointManifestHash
                == decodedSensitivity.checkpointManifestHash,
            checkpointContentSHA256
                == decodedSensitivity.checkpointContentSHA256,
            tokenizerSHA256 == decodedSensitivity.tokenizerSHA256,
            groupSize == decodedSensitivity.groupSize
        else {
            throw KVTunerSearchArtifactError.sensitivityArtifactMismatch
        }

        let expectedCandidates: [KVTunerScheduleCandidate]
        do {
            expectedCandidates = try KVTunerScheduleSearch.enumerate(
                analysis: decodedSensitivity.analyzed(),
                targetPairBitTotal: targetPairBitTotal,
                maxCandidates: candidates.count)
        } catch {
            throw KVTunerSearchArtifactError.candidateListMismatch
        }
        guard expectedCandidates == candidates,
            candidateListSHA256
                == (try? KVTunerScheduleSearch.candidateListSHA256(
                    candidates))
        else {
            throw KVTunerSearchArtifactError.candidateListMismatch
        }

        guard candidateRuntimePolicies.count == candidates.count else {
            throw KVTunerSearchArtifactError.evaluationMismatch
        }
        let candidateHashes: [String]
        do {
            candidateHashes = try candidates.map(
                KVTunerScheduleSearch.candidateSHA256)
        } catch {
            throw KVTunerSearchArtifactError.candidateListMismatch
        }
        for (ordinal, policy) in candidateRuntimePolicies.enumerated() {
            let expectedLayers = candidates[ordinal].layers.map {
                KVTunerRuntimeLayerPolicy(
                    layer: $0.layer,
                    keyBits: $0.keyBits,
                    valueBits: $0.valueBits)
            }
            guard policy.candidateOrdinal == ordinal,
                policy.candidateSHA256 == candidateHashes[ordinal],
                policy.candidateCount == candidates.count,
                policy.candidateListSHA256 == candidateListSHA256,
                policy.calibrationManifestSHA256 == promptManifestSHA256,
                policy.sourceSensitivityArtifactSHA256
                    == sourceSensitivityArtifactSHA256,
                policy.matrixID == decodedSensitivity.matrixID,
                policy.modelConfigHash == modelConfigHash,
                policy.modelConfigSHA256 == modelConfigSHA256,
                policy.checkpointManifestHash == checkpointManifestHash,
                policy.checkpointContentSHA256
                    == checkpointContentSHA256,
                policy.tokenizerSHA256 == tokenizerSHA256,
                policy.groupSize == groupSize,
                policy.targetPairBitTotal == targetPairBitTotal,
                policy.layers == expectedLayers
            else {
                throw KVTunerSearchArtifactError.evaluationMismatch
            }
        }

        guard exactCandidateEvaluationArtifactData.count == candidates.count
        else {
            throw KVTunerSearchArtifactError.evaluationMismatch
        }
        let runtimeContract: KVTunerCandidateRuntimeContract
        do {
            guard let firstPolicy = candidateRuntimePolicies.first else {
                throw KVTunerSearchArtifactError.evaluationMismatch
            }
            runtimeContract = try KVTunerCandidateRuntimeContract.load(
                exactModelConfigData: exactModelConfigData,
                runtimePolicy: firstPolicy,
                eosTokenID: eosTokenID)
        } catch {
            throw KVTunerSearchArtifactError.evaluationMismatch
        }
        var authenticatedEvaluations: [KVTunerCandidateEvaluation] = []
        authenticatedEvaluations.reserveCapacity(candidates.count)
        do {
            for (ordinal, data) in
                exactCandidateEvaluationArtifactData.enumerated()
            {
                let artifact = try JSONDecoder().decode(
                    KVTunerCandidateEvaluationArtifact.self, from: data)
                authenticatedEvaluations.append(try artifact.validated(
                    exactArtifactData: data,
                    runtimePolicy: candidateRuntimePolicies[ordinal],
                    runtimeContract: runtimeContract,
                    calibrationManifest: calibrationManifest,
                    exactCalibrationManifestData:
                        exactCalibrationManifestData,
                    decodeTokenIDs: decodeTokenIDs))
            }
        } catch {
            throw KVTunerSearchArtifactError.evaluationMismatch
        }
        guard authenticatedEvaluations == evaluations else {
            throw KVTunerSearchArtifactError.evaluationMismatch
        }

        let selected: KVTunerScheduleCandidate
        do {
            selected = try KVTunerScheduleSearch.select(
                candidates: candidates,
                evaluations: authenticatedEvaluations,
                requiredRuntimePolicySHA256ByCandidate:
                    candidateRuntimePolicies.map(\.runtimePolicySHA256))
        } catch {
            throw KVTunerSearchArtifactError.evaluationMismatch
        }
        guard selected.ordinal == selectedCandidateOrdinal else {
            throw KVTunerSearchArtifactError.selectedCandidateMismatch
        }
        return selected
    }

    private static func isIdentityDigest(_ value: String) -> Bool {
        isLowercaseHex(value, length: 16)
            || isLowercaseHex(value, length: 64)
    }

    private static func isLowercaseHex(
        _ value: String,
        length: Int
    ) -> Bool {
        guard value.count == length else { return false }
        return value.unicodeScalars.allSatisfy {
            CharacterSet(charactersIn: "0123456789abcdef").contains($0)
        }
    }
}

/// Pure deterministic stage between the measured sensitivity artifact and the frozen runtime
/// schedule. Enumeration uses exact integer pair-bit budgets; accuracy evaluation remains a
/// separate, explicitly complete input so a partial run cannot silently select a schedule.
public enum KVTunerScheduleSearch {
    public static let requiredSearchPromptCount = 200
    public static let requiredFewShotSeed: UInt64 = 1234

    /// Authenticates both the legacy compatibility fingerprint and the cryptographic identity of
    /// the exact runtime config bytes before a sensitivity artifact can become a schedule. The
    /// SHA-256 check is load-bearing: FNV alone is not a collision-resistant trust boundary.
    static func validatedModelLayerCount(
        exactModelConfigData: Data,
        expectedModelConfigHash: String,
        expectedModelConfigSHA256: String,
        expectedLayerCount: Int
    ) throws -> Int {
        let actualLayerCount: Int
        do {
            actualLayerCount = try KVTunerModelConfigPreflight.load(
                from: exactModelConfigData)
        } catch {
            throw KVTunerScheduleSearchError.sensitivityArtifactMismatch
        }
        guard fnv1a64(exactModelConfigData) == expectedModelConfigHash,
            sha256Hex(exactModelConfigData) == expectedModelConfigSHA256,
            actualLayerCount == expectedLayerCount
        else {
            throw KVTunerScheduleSearchError.sensitivityArtifactMismatch
        }
        return actualLayerCount
    }

    public static func candidateSHA256(
        _ candidate: KVTunerScheduleCandidate
    ) throws -> String {
        sha256Hex(try candidateTranscript(candidate))
    }

    public static func candidateListSHA256(
        _ candidates: [KVTunerScheduleCandidate]
    ) throws -> String {
        var transcript = TypedTranscript(
            domain: "fast-mlx.kvtuner-candidate-list.v1")
        transcript.appendCount(candidates.count)
        for candidate in candidates {
            try transcript.appendDigest(candidateSHA256(candidate))
        }
        return sha256Hex(transcript.data)
    }

    public static func enumerate(
        analysis: KVTunerSensitivityAnalysis,
        targetPairBitTotal: Int,
        maxCandidates: Int
    ) throws -> [KVTunerScheduleCandidate] {
        try validate(analysis)
        guard targetPairBitTotal > 0 else {
            throw KVTunerScheduleSearchError.invalidTargetPairBitTotal
        }
        guard maxCandidates > 0 else {
            throw KVTunerScheduleSearchError.invalidCandidateLimit
        }
        let analysisSHA256 = try canonicalAnalysisSHA256(analysis)

        let groups = analysis.groups
        var contributions: [[(KVTunerPrecisionPair, Int)]] = []
        contributions.reserveCapacity(groups.count)
        for group in groups {
            var options: [(KVTunerPrecisionPair, Int)] = []
            for pair in group.allowedPairs {
                let result = pair.pairBitCost.multipliedReportingOverflow(
                    by: group.layers.count)
                guard !result.overflow else {
                    throw KVTunerScheduleSearchError.arithmeticOverflow
                }
                options.append((pair, result.partialValue))
            }
            contributions.append(options)
        }

        var suffixReachable = Array(
            repeating: Set<Int>(), count: groups.count + 1)
        suffixReachable[groups.count] = [0]
        if !groups.isEmpty {
            for groupIndex in stride(
                from: groups.count - 1, through: 0, by: -1)
            {
                var reachable = Set<Int>()
                for (_, contribution) in contributions[groupIndex] {
                    for suffix in suffixReachable[groupIndex + 1] {
                        let result = contribution.addingReportingOverflow(suffix)
                        guard !result.overflow else {
                            throw KVTunerScheduleSearchError.arithmeticOverflow
                        }
                        if result.partialValue <= targetPairBitTotal {
                            reachable.insert(result.partialValue)
                        }
                    }
                }
                suffixReachable[groupIndex] = reachable
            }
        }
        guard suffixReachable[0].contains(targetPairBitTotal) else {
            throw KVTunerScheduleSearchError.noFeasibleSchedule
        }

        var assignments = Array<KVTunerPrecisionPair?>(
            repeating: nil, count: groups.count)
        var candidates: [KVTunerScheduleCandidate] = []

        func emitCandidate() throws {
            guard candidates.count < maxCandidates else {
                throw KVTunerScheduleSearchError.candidateLimitExceeded(
                    limit: maxCandidates)
            }
            var policy = analysis.layers.map {
                KVLayerPrecision(layer: $0.layer, keyBits: 0, valueBits: 0)
            }
            for groupIndex in groups.indices {
                guard let pair = assignments[groupIndex] else {
                    throw KVTunerScheduleSearchError.invalidCandidate
                }
                for layer in groups[groupIndex].layers {
                    policy[layer] = KVLayerPrecision(
                        layer: layer,
                        keyBits: pair.keyBits,
                        valueBits: pair.valueBits)
                }
            }

            var meanError = 0.0
            let divisor = Double(policy.count)
            for precision in policy {
                let pair = KVTunerPrecisionPair(
                    keyBits: precision.keyBits,
                    valueBits: precision.valueBits)
                guard let aggregate = analysis.layers[precision.layer]
                    .aggregates.first(where: { $0.pair == pair })
                else {
                    throw KVTunerScheduleSearchError.invalidCandidate
                }
                meanError += aggregate.relativeAttentionOutputError / divisor
            }
            guard meanError.isFinite, meanError >= 0 else {
                throw KVTunerScheduleSearchError.invalidCandidate
            }
            candidates.append(KVTunerScheduleCandidate(
                ordinal: candidates.count,
                analysisSHA256: analysisSHA256,
                totalPairBits: targetPairBitTotal,
                meanAttentionOutputError: meanError,
                layers: policy))
        }

        func visit(groupIndex: Int, accumulated: Int) throws {
            if groupIndex == groups.count {
                if accumulated == targetPairBitTotal {
                    try emitCandidate()
                }
                return
            }
            for (pair, contribution) in contributions[groupIndex] {
                let next = accumulated.addingReportingOverflow(contribution)
                guard !next.overflow else {
                    throw KVTunerScheduleSearchError.arithmeticOverflow
                }
                guard next.partialValue <= targetPairBitTotal else { continue }
                let needed = targetPairBitTotal - next.partialValue
                guard suffixReachable[groupIndex + 1].contains(needed) else {
                    continue
                }
                assignments[groupIndex] = pair
                try visit(
                    groupIndex: groupIndex + 1,
                    accumulated: next.partialValue)
            }
            assignments[groupIndex] = nil
        }

        try visit(groupIndex: 0, accumulated: 0)
        guard !candidates.isEmpty else {
            throw KVTunerScheduleSearchError.noFeasibleSchedule
        }
        return candidates
    }

    static func select(
        candidates: [KVTunerScheduleCandidate],
        evaluations: [KVTunerCandidateEvaluation],
        requiredRuntimePolicySHA256ByCandidate: [String]
    ) throws -> KVTunerScheduleCandidate {
        try validateCandidates(candidates)
        guard evaluations.count == candidates.count else {
            throw KVTunerScheduleSearchError.incompleteEvaluations(
                expected: candidates.count, actual: evaluations.count)
        }
        guard requiredRuntimePolicySHA256ByCandidate.count
                == candidates.count,
            Set(requiredRuntimePolicySHA256ByCandidate).count
                == candidates.count,
            requiredRuntimePolicySHA256ByCandidate.allSatisfy({
                isLowercaseHex($0, length: 64)
            })
        else {
            throw KVTunerScheduleSearchError.incomparableEvaluations
        }

        var byOrdinal: [Int: KVTunerCandidateEvaluation] = [:]
        var commonTotal: Int?
        var commonEnvironmentSHA256: String?
        let candidateHashes = try candidates.map(candidateSHA256)
        for evaluation in evaluations {
            guard byOrdinal[evaluation.candidateOrdinal] == nil else {
                throw KVTunerScheduleSearchError.duplicateEvaluation(
                    evaluation.candidateOrdinal)
            }
            guard candidates.indices.contains(evaluation.candidateOrdinal),
                evaluation.candidateSHA256
                    == candidateHashes[evaluation.candidateOrdinal],
                isLowercaseHex(
                    evaluation.environmentSHA256, length: 64),
                evaluation.totalCount == requiredSearchPromptCount,
                evaluation.correctCount >= 0,
                evaluation.correctCount <= evaluation.totalCount,
                isLowercaseHex(evaluation.outputSHA256, length: 64)
            else {
                throw KVTunerScheduleSearchError.invalidEvaluation(
                    evaluation.candidateOrdinal)
            }
            guard evaluation.runtimePolicySHA256
                    == requiredRuntimePolicySHA256ByCandidate[
                        evaluation.candidateOrdinal]
            else {
                throw KVTunerScheduleSearchError.runtimePolicyMismatch(
                    evaluation.candidateOrdinal)
            }
            if let commonEnvironmentSHA256,
                commonEnvironmentSHA256 != evaluation.environmentSHA256
            {
                throw KVTunerScheduleSearchError
                    .mixedEvaluationEnvironments
            }
            commonEnvironmentSHA256 = evaluation.environmentSHA256
            if let commonTotal, commonTotal != evaluation.totalCount {
                throw KVTunerScheduleSearchError.incomparableEvaluations
            }
            commonTotal = evaluation.totalCount
            byOrdinal[evaluation.candidateOrdinal] = evaluation
        }

        return try candidates.min { lhs, rhs in
            guard let lhsEvaluation = byOrdinal[lhs.ordinal],
                let rhsEvaluation = byOrdinal[rhs.ordinal]
            else { return false }
            if lhsEvaluation.correctCount != rhsEvaluation.correctCount {
                return lhsEvaluation.correctCount
                    > rhsEvaluation.correctCount
            }
            if lhs.meanAttentionOutputError
                != rhs.meanAttentionOutputError
            {
                return lhs.meanAttentionOutputError
                    < rhs.meanAttentionOutputError
            }
            return policyLexicographicallyPrecedes(lhs.layers, rhs.layers)
        } ?? { throw KVTunerScheduleSearchError.invalidCandidate }()
    }

    public static func makeSchedule(
        searchArtifact: KVTunerSearchArtifact,
        exactSearchArtifactData: Data,
        sensitivityArtifact: KVTunerSensitivityArtifact,
        exactSensitivityArtifactData: Data,
        calibrationManifest: KVTunerCalibrationManifest,
        exactCalibrationManifestData: Data,
        exactCandidateEvaluationArtifactData: [Data],
        candidateRuntimePolicies: [KVTunerCandidateRuntimePolicy],
        eosTokenID: Int,
        decodeTokenIDs: ([Int]) throws -> String,
        exactModelConfigData: Data,
        expectedCheckpointManifestHash: String,
        expectedCheckpointContentSHA256: String
    ) throws -> KVTunerSchedule {
        let decodedSearch: KVTunerSearchArtifact
        do {
            decodedSearch = try JSONDecoder().decode(
                KVTunerSearchArtifact.self,
                from: exactSearchArtifactData)
        } catch {
            throw KVTunerScheduleSearchError.searchArtifactMismatch
        }
        guard decodedSearch == searchArtifact else {
            throw KVTunerScheduleSearchError.searchArtifactMismatch
        }
        let selectedCandidate: KVTunerScheduleCandidate
        do {
            selectedCandidate = try decodedSearch.validated(
                sensitivityArtifact: sensitivityArtifact,
                exactSensitivityArtifactData:
                    exactSensitivityArtifactData,
                calibrationManifest: calibrationManifest,
                exactCalibrationManifestData:
                    exactCalibrationManifestData,
                exactCandidateEvaluationArtifactData:
                    exactCandidateEvaluationArtifactData,
                candidateRuntimePolicies: candidateRuntimePolicies,
                exactModelConfigData: exactModelConfigData,
                eosTokenID: eosTokenID,
                decodeTokenIDs: decodeTokenIDs)
        } catch {
            throw KVTunerScheduleSearchError.searchArtifactMismatch
        }

        let actualLayerCount = try validatedModelLayerCount(
            exactModelConfigData: exactModelConfigData,
            expectedModelConfigHash: sensitivityArtifact.modelConfigHash,
            expectedModelConfigSHA256:
                sensitivityArtifact.modelConfigSHA256,
            expectedLayerCount: sensitivityArtifact.layerCount)
        guard expectedCheckpointManifestHash
                == sensitivityArtifact.checkpointManifestHash,
            expectedCheckpointManifestHash
                == calibrationManifest.checkpointManifestHash,
            expectedCheckpointContentSHA256
                == sensitivityArtifact.checkpointContentSHA256,
            expectedCheckpointContentSHA256
                == calibrationManifest.checkpointContentSHA256
        else {
            throw KVTunerScheduleSearchError.sensitivityArtifactMismatch
        }
        var totalPairBits = 0
        for (position, layer) in selectedCandidate.layers.enumerated() {
            let pair = KVTunerPrecisionPair(
                keyBits: layer.keyBits, valueBits: layer.valueBits)
            guard layer.layer == position,
                KVTunerSensitivityArtifact.canonicalPrecisionPairs.contains(pair)
            else {
                throw KVTunerScheduleSearchError.invalidCandidate
            }
            let result = totalPairBits.addingReportingOverflow(
                pair.pairBitCost)
            guard !result.overflow else {
                throw KVTunerScheduleSearchError.arithmeticOverflow
            }
            totalPairBits = result.partialValue
        }
        guard totalPairBits == selectedCandidate.totalPairBits else {
            throw KVTunerScheduleSearchError.invalidCandidate
        }
        let doubledLayerCount = actualLayerCount
            .multipliedReportingOverflow(by: 2)
        guard !doubledLayerCount.overflow else {
            throw KVTunerScheduleSearchError.arithmeticOverflow
        }
        let nominalAverageBits = Double(totalPairBits)
            / Double(doubledLayerCount.partialValue)
        guard nominalAverageBits.isFinite, nominalAverageBits > 0 else {
            throw KVTunerScheduleSearchError.invalidTargetPairBitTotal
        }
        let objective =
            "maximize-gsm8k-accuracy-at-b\(nominalAverageBits)"
        let cellID =
            "kvtuner-g\(sensitivityArtifact.groupSize)-b\(nominalAverageBits)"
        let schedule = KVTunerSchedule(
            schemaVersion: 4,
            matrixID: sensitivityArtifact.matrixID,
            cellID: cellID,
            modelConfigHash: sensitivityArtifact.modelConfigHash,
            modelConfigSHA256: sensitivityArtifact.modelConfigSHA256,
            checkpointManifestHash:
                sensitivityArtifact.checkpointManifestHash,
            checkpointContentSHA256:
                sensitivityArtifact.checkpointContentSHA256,
            tokenizerSHA256: sensitivityArtifact.tokenizerSHA256,
            groupSize: sensitivityArtifact.groupSize,
            calibrationCorpusID: calibrationManifest.corpusID,
            calibrationCorpusHash:
                sha256Hex(exactCalibrationManifestData),
            calibrationEntryHashes:
                calibrationManifest.calibrationEntryDigests,
            calibrationSourceItemDigests:
                calibrationManifest.calibrationSourceItemDigests,
            seed: decodedSearch.seed,
            objective: objective,
            nominalAverageBits: nominalAverageBits,
            sourceSensitivityArtifactSHA256:
                sha256Hex(exactSensitivityArtifactData),
            sourceSearchArtifactSHA256:
                sha256Hex(exactSearchArtifactData),
            layers: selectedCandidate.layers)
        do {
            return try schedule.validated(
                expectedLayerCount: actualLayerCount,
                expectedMatrixID: sensitivityArtifact.matrixID,
                expectedCellID: cellID,
                expectedModelConfigHash:
                    sensitivityArtifact.modelConfigHash,
                expectedModelConfigSHA256:
                    sensitivityArtifact.modelConfigSHA256,
                expectedCheckpointManifestHash:
                    sensitivityArtifact.checkpointManifestHash,
                expectedCheckpointContentSHA256:
                    sensitivityArtifact.checkpointContentSHA256)
        } catch let error as KVTunerScheduleError {
            throw KVTunerScheduleSearchError.invalidSchedule(error)
        }
    }

    private static func validate(
        _ analysis: KVTunerSensitivityAnalysis
    ) throws {
        let canonicalPairs = KVTunerSensitivityArtifact
            .canonicalPrecisionPairs
        guard !analysis.layers.isEmpty,
            analysis.layers.enumerated().allSatisfy({
                $0.offset == $0.element.layer
                    && $0.element.aggregates.map(\.pair) == canonicalPairs
                    && !$0.element.paretoPairs.isEmpty
                    && $0.element.paretoPairs ==
                        KVTunerSensitivityArtifact.paretoSurvivors(
                            $0.element.aggregates)
                    && $0.element.aggregates.allSatisfy { aggregate in
                        let metrics = [
                            aggregate.relativeKeyError,
                            aggregate.relativeValueError,
                            aggregate.attentionScoreError,
                            aggregate.relativeAttentionOutputError,
                        ]
                        return metrics.allSatisfy {
                            $0.isFinite && $0 >= 0
                        }
                    }
            }),
            !analysis.groups.isEmpty
        else {
            throw KVTunerScheduleSearchError.invalidAnalysis
        }

        var coveredLayers: [Int] = []
        for (index, group) in analysis.groups.enumerated() {
            guard group.id == index,
                !group.layers.isEmpty,
                group.layers == group.layers.sorted(),
                Set(group.layers).count == group.layers.count,
                !group.allowedPairs.isEmpty,
                group.allowedPairs
                    == canonicalPairs.filter(group.allowedPairs.contains)
            else {
                throw KVTunerScheduleSearchError.invalidAnalysis
            }
            for layer in group.layers {
                guard analysis.layers.indices.contains(layer),
                    analysis.layers[layer].paretoPairs == group.allowedPairs
                else {
                    throw KVTunerScheduleSearchError.invalidAnalysis
                }
            }
            coveredLayers.append(contentsOf: group.layers)
        }
        guard coveredLayers.sorted() == Array(analysis.layers.indices) else {
            throw KVTunerScheduleSearchError.invalidAnalysis
        }
    }

    private static func validateCandidates(
        _ candidates: [KVTunerScheduleCandidate]
    ) throws {
        guard let first = candidates.first,
            first.totalPairBits > 0,
            isLowercaseHex(first.analysisSHA256, length: 64)
        else {
            throw KVTunerScheduleSearchError.invalidCandidate
        }
        let layerCount = first.layers.count
        guard layerCount > 0 else {
            throw KVTunerScheduleSearchError.invalidCandidate
        }
        var policies = Set<String>()
        for (ordinal, candidate) in candidates.enumerated() {
            guard candidate.ordinal == ordinal,
                candidate.analysisSHA256 == first.analysisSHA256,
                candidate.totalPairBits == first.totalPairBits,
                candidate.meanAttentionOutputError.isFinite,
                candidate.meanAttentionOutputError >= 0,
                candidate.layers.count == layerCount
            else {
                throw KVTunerScheduleSearchError.invalidCandidate
            }
            var policyKey = ""
            var computedTotal = 0
            for (layer, precision) in candidate.layers.enumerated() {
                let pair = KVTunerPrecisionPair(
                    keyBits: precision.keyBits,
                    valueBits: precision.valueBits)
                guard precision.layer == layer,
                    KVTunerSensitivityArtifact.canonicalPrecisionPairs
                        .contains(pair)
                else {
                    throw KVTunerScheduleSearchError.invalidCandidate
                }
                let sum = computedTotal.addingReportingOverflow(
                    pair.pairBitCost)
                guard !sum.overflow else {
                    throw KVTunerScheduleSearchError.arithmeticOverflow
                }
                computedTotal = sum.partialValue
                policyKey += "\(pair.keyBits):\(pair.valueBits);"
            }
            guard computedTotal == candidate.totalPairBits,
                policies.insert(policyKey).inserted
            else {
                throw KVTunerScheduleSearchError.invalidCandidate
            }
        }
    }

    private static func canonicalAnalysisSHA256(
        _ analysis: KVTunerSensitivityAnalysis
    ) throws -> String {
        var transcript = TypedTranscript(
            domain: "fast-mlx.kvtuner-sensitivity-analysis.v1")
        transcript.appendCount(analysis.layers.count)
        for layer in analysis.layers {
            transcript.appendInt(layer.layer)
            transcript.appendCount(layer.aggregates.count)
            for aggregate in layer.aggregates {
                transcript.appendInt(aggregate.pair.keyBits)
                transcript.appendInt(aggregate.pair.valueBits)
                try transcript.appendFiniteDouble(
                    aggregate.relativeKeyError)
                try transcript.appendFiniteDouble(
                    aggregate.relativeValueError)
                try transcript.appendFiniteDouble(
                    aggregate.attentionScoreError)
                try transcript.appendFiniteDouble(
                    aggregate.relativeAttentionOutputError)
            }
            transcript.appendCount(layer.paretoPairs.count)
            for pair in layer.paretoPairs {
                transcript.appendInt(pair.keyBits)
                transcript.appendInt(pair.valueBits)
            }
        }
        transcript.appendCount(analysis.groups.count)
        for group in analysis.groups {
            transcript.appendInt(group.id)
            transcript.appendCount(group.layers.count)
            for layer in group.layers { transcript.appendInt(layer) }
            transcript.appendCount(group.allowedPairs.count)
            for pair in group.allowedPairs {
                transcript.appendInt(pair.keyBits)
                transcript.appendInt(pair.valueBits)
            }
        }
        return sha256Hex(transcript.data)
    }

    /// Versioned language-neutral identity transcript. Counts and signed integers are fixed-width
    /// big-endian lanes; SHA-256 values are raw 32-byte digests; finite doubles use IEEE-754 bits
    /// with both signed zero spellings normalized to +0. This avoids Foundation JSON rendering as
    /// a portability boundary for semantic identities.
    private static func candidateTranscript(
        _ candidate: KVTunerScheduleCandidate
    ) throws -> Data {
        var transcript = TypedTranscript(
            domain: "fast-mlx.kvtuner-candidate.v1")
        transcript.appendInt(candidate.ordinal)
        try transcript.appendDigest(candidate.analysisSHA256)
        transcript.appendInt(candidate.totalPairBits)
        try transcript.appendFiniteDouble(
            candidate.meanAttentionOutputError)
        transcript.appendCount(candidate.layers.count)
        for layer in candidate.layers {
            transcript.appendInt(layer.layer)
            transcript.appendInt(layer.keyBits)
            transcript.appendInt(layer.valueBits)
        }
        return transcript.data
    }

    private struct TypedTranscript {
        var data = Data()

        init(domain: String) {
            data.append(contentsOf: domain.utf8)
            data.append(0)
        }

        mutating func appendCount(_ value: Int) {
            appendUInt64(UInt64(value))
        }

        mutating func appendInt(_ value: Int) {
            var encoded = Int64(value).bigEndian
            withUnsafeBytes(of: &encoded) {
                data.append(contentsOf: $0)
            }
        }

        mutating func appendFiniteDouble(_ value: Double) throws {
            guard value.isFinite else {
                throw KVTunerScheduleSearchError.invalidCandidate
            }
            let normalized = value == 0 ? 0.0 : value
            appendUInt64(normalized.bitPattern)
        }

        mutating func appendDigest(_ value: String) throws {
            guard value.count == 64 else {
                throw KVTunerScheduleSearchError.invalidCandidate
            }
            var index = value.startIndex
            for _ in 0..<32 {
                let next = value.index(index, offsetBy: 2)
                guard let byte = UInt8(value[index..<next], radix: 16) else {
                    throw KVTunerScheduleSearchError.invalidCandidate
                }
                data.append(byte)
                index = next
            }
        }

        private mutating func appendUInt64(_ value: UInt64) {
            var encoded = value.bigEndian
            withUnsafeBytes(of: &encoded) {
                data.append(contentsOf: $0)
            }
        }
    }

    private static func policyLexicographicallyPrecedes(
        _ lhs: [KVLayerPrecision],
        _ rhs: [KVLayerPrecision]
    ) -> Bool {
        for (left, right) in zip(lhs, rhs) {
            if left.keyBits != right.keyBits {
                return left.keyBits < right.keyBits
            }
            if left.valueBits != right.valueBits {
                return left.valueBits < right.valueBits
            }
        }
        return lhs.count < rhs.count
    }

    private static func isLowercaseHex(
        _ value: String,
        length: Int
    ) -> Bool {
        guard value.count == length else { return false }
        return value.unicodeScalars.allSatisfy {
            CharacterSet(charactersIn: "0123456789abcdef").contains($0)
        }
    }
}
