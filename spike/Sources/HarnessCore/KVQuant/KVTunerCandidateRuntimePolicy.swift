import Foundation

public enum KVTunerCandidateRuntimePolicyError:
    Error, Equatable, Sendable
{
    case invalidCalibrationManifest
    case invalidSensitivityArtifact
    case invalidModelConfig(KVTunerModelConfigPreflightError)
    case modelConfigIdentityMismatch
    case layerCountMismatch(expected: Int, actual: Int)
    case checkpointIdentityMismatch
    case tokenizerIdentityMismatch
    case promptTokenizationMismatch(phase: String, position: Int)
    case candidateEnumerationTruncated(limit: Int)
    case candidateEnumerationFailed(KVTunerScheduleSearchError)
    case candidateOrdinalOutOfRange(Int)
    case candidateIdentityFailure
    case unexpectedValidationFailure
}

/// Immutable, preselection-only KVTuner policy derived from exact calibration and sensitivity
/// evidence. This type deliberately has no public initializer: a candidate can become executable
/// only after its complete deterministic candidate set, live model identity, and live tokenizer
/// replay have all passed. It remains separate from `KVTunerRuntimeSelection`, whose stronger
/// qualification-bundle boundary is reserved for the final selected schedule.
public struct KVTunerCandidateRuntimePolicy: Hashable, Sendable {
    public let runtimePolicySHA256: String
    public let calibrationManifestSHA256: String
    public let sourceSensitivityArtifactSHA256: String
    public let candidateListSHA256: String
    public let candidateSHA256: String
    public let matrixID: String
    public let modelConfigHash: String
    /// SHA-256 of the exact runtime config bytes, authenticated against both the calibration
    /// manifest and sensitivity artifact before the policy becomes executable.
    public let modelConfigSHA256: String
    public let checkpointManifestHash: String
    public let tokenizerSHA256: String
    public let groupSize: Int
    public let targetPairBitTotal: Int
    public let candidateCount: Int
    public let candidateOrdinal: Int
    public let layers: [KVTunerRuntimeLayerPolicy]

    private init(
        runtimePolicySHA256: String,
        calibrationManifestSHA256: String,
        sourceSensitivityArtifactSHA256: String,
        candidateListSHA256: String,
        candidateSHA256: String,
        matrixID: String,
        modelConfigHash: String,
        modelConfigSHA256: String,
        checkpointManifestHash: String,
        tokenizerSHA256: String,
        groupSize: Int,
        targetPairBitTotal: Int,
        candidateCount: Int,
        candidateOrdinal: Int,
        layers: [KVTunerRuntimeLayerPolicy]
    ) {
        self.runtimePolicySHA256 = runtimePolicySHA256
        self.calibrationManifestSHA256 = calibrationManifestSHA256
        self.sourceSensitivityArtifactSHA256 =
            sourceSensitivityArtifactSHA256
        self.candidateListSHA256 = candidateListSHA256
        self.candidateSHA256 = candidateSHA256
        self.matrixID = matrixID
        self.modelConfigHash = modelConfigHash
        self.modelConfigSHA256 = modelConfigSHA256
        self.checkpointManifestHash = checkpointManifestHash
        self.tokenizerSHA256 = tokenizerSHA256
        self.groupSize = groupSize
        self.targetPairBitTotal = targetPairBitTotal
        self.candidateCount = candidateCount
        self.candidateOrdinal = candidateOrdinal
        self.layers = layers
    }

    /// Reconstructs the complete candidate list from exact source artifacts and then freezes one
    /// candidate for an evaluation-only runtime path. `maxCandidates` is a fail-closed resource
    /// ceiling, not a truncation request: if the exhaustive set is larger, loading fails instead
    /// of evaluating a prefix that could later masquerade as the complete search.
    public static func load(
        exactCalibrationManifestData: Data,
        exactSensitivityArtifactData: Data,
        exactModelConfigData: Data,
        expectedCheckpointManifestHash: String,
        expectedTokenizerSHA256: String,
        targetPairBitTotal: Int,
        maxCandidates: Int,
        candidateOrdinal: Int,
        tokenizePrompt: (String) throws -> [Int]
    ) throws -> KVTunerCandidateRuntimePolicy {
        let manifest: KVTunerCalibrationManifest
        do {
            manifest = try JSONDecoder().decode(
                KVTunerCalibrationManifest.self,
                from: exactCalibrationManifestData)
            _ = try manifest.validated()
        } catch {
            throw KVTunerCandidateRuntimePolicyError
                .invalidCalibrationManifest
        }

        let sensitivity: KVTunerSensitivityArtifact
        do {
            sensitivity = try JSONDecoder().decode(
                KVTunerSensitivityArtifact.self,
                from: exactSensitivityArtifactData)
            _ = try sensitivity.validated(
                calibrationManifest: manifest,
                exactCalibrationManifestData:
                    exactCalibrationManifestData)
        } catch {
            throw KVTunerCandidateRuntimePolicyError
                .invalidSensitivityArtifact
        }

        let actualLayerCount: Int
        do {
            actualLayerCount = try KVTunerModelConfigPreflight.load(
                from: exactModelConfigData)
        } catch let error as KVTunerModelConfigPreflightError {
            throw KVTunerCandidateRuntimePolicyError.invalidModelConfig(error)
        } catch {
            throw KVTunerCandidateRuntimePolicyError
                .unexpectedValidationFailure
        }
        let exactModelConfigSHA256 = sha256Hex(exactModelConfigData)
        guard fnv1a64(exactModelConfigData) == manifest.modelConfigHash,
            sensitivity.modelConfigHash == manifest.modelConfigHash,
            exactModelConfigSHA256 == manifest.modelConfigSHA256,
            sensitivity.modelConfigSHA256 == manifest.modelConfigSHA256
        else {
            throw KVTunerCandidateRuntimePolicyError
                .modelConfigIdentityMismatch
        }
        guard actualLayerCount == sensitivity.layerCount else {
            throw KVTunerCandidateRuntimePolicyError.layerCountMismatch(
                expected: sensitivity.layerCount,
                actual: actualLayerCount)
        }
        guard expectedCheckpointManifestHash
                == manifest.checkpointManifestHash,
            sensitivity.checkpointManifestHash
                == expectedCheckpointManifestHash
        else {
            throw KVTunerCandidateRuntimePolicyError
                .checkpointIdentityMismatch
        }
        guard expectedTokenizerSHA256 == manifest.tokenizerSHA256,
            sensitivity.tokenizerSHA256 == expectedTokenizerSHA256
        else {
            throw KVTunerCandidateRuntimePolicyError
                .tokenizerIdentityMismatch
        }

        try validateLiveTokenization(
            manifest.sensitivityPrompts,
            phase: "sensitivity",
            tokenizePrompt: tokenizePrompt)
        try validateLiveTokenization(
            manifest.searchPrompts,
            phase: "search",
            tokenizePrompt: tokenizePrompt)

        let analysis: KVTunerSensitivityAnalysis
        do {
            analysis = try sensitivity.analyzed()
        } catch {
            throw KVTunerCandidateRuntimePolicyError
                .invalidSensitivityArtifact
        }
        let candidates: [KVTunerScheduleCandidate]
        do {
            candidates = try KVTunerScheduleSearch.enumerate(
                analysis: analysis,
                targetPairBitTotal: targetPairBitTotal,
                maxCandidates: maxCandidates)
        } catch KVTunerScheduleSearchError.candidateLimitExceeded(let limit) {
            throw KVTunerCandidateRuntimePolicyError
                .candidateEnumerationTruncated(limit: limit)
        } catch let error as KVTunerScheduleSearchError {
            throw KVTunerCandidateRuntimePolicyError
                .candidateEnumerationFailed(error)
        } catch {
            throw KVTunerCandidateRuntimePolicyError
                .unexpectedValidationFailure
        }
        guard candidates.indices.contains(candidateOrdinal) else {
            throw KVTunerCandidateRuntimePolicyError
                .candidateOrdinalOutOfRange(candidateOrdinal)
        }

        let candidate = candidates[candidateOrdinal]
        let candidateListSHA256: String
        let candidateSHA256: String
        do {
            candidateListSHA256 = try KVTunerScheduleSearch
                .candidateListSHA256(candidates)
            candidateSHA256 = try KVTunerScheduleSearch
                .candidateSHA256(candidate)
        } catch {
            throw KVTunerCandidateRuntimePolicyError
                .candidateIdentityFailure
        }
        let calibrationManifestSHA256 = sha256Hex(
            exactCalibrationManifestData)
        let sensitivityArtifactSHA256 = sha256Hex(
            exactSensitivityArtifactData)
        let modelConfigSHA256 = exactModelConfigSHA256
        let layers = candidate.layers.map {
            KVTunerRuntimeLayerPolicy(
                layer: $0.layer,
                keyBits: $0.keyBits,
                valueBits: $0.valueBits)
        }
        let runtimePolicySHA256 = makeRuntimePolicySHA256(
            calibrationManifestSHA256: calibrationManifestSHA256,
            sensitivityArtifactSHA256: sensitivityArtifactSHA256,
            candidateListSHA256: candidateListSHA256,
            candidateSHA256: candidateSHA256,
            matrixID: sensitivity.matrixID,
            modelConfigHash: sensitivity.modelConfigHash,
            modelConfigSHA256: modelConfigSHA256,
            checkpointManifestHash:
                sensitivity.checkpointManifestHash,
            tokenizerSHA256: sensitivity.tokenizerSHA256,
            groupSize: sensitivity.groupSize,
            targetPairBitTotal: targetPairBitTotal,
            candidateCount: candidates.count,
            candidateOrdinal: candidateOrdinal,
            layers: layers)

        return KVTunerCandidateRuntimePolicy(
            runtimePolicySHA256: runtimePolicySHA256,
            calibrationManifestSHA256: calibrationManifestSHA256,
            sourceSensitivityArtifactSHA256:
                sensitivityArtifactSHA256,
            candidateListSHA256: candidateListSHA256,
            candidateSHA256: candidateSHA256,
            matrixID: sensitivity.matrixID,
            modelConfigHash: sensitivity.modelConfigHash,
            modelConfigSHA256: modelConfigSHA256,
            checkpointManifestHash:
                sensitivity.checkpointManifestHash,
            tokenizerSHA256: sensitivity.tokenizerSHA256,
            groupSize: sensitivity.groupSize,
            targetPairBitTotal: targetPairBitTotal,
            candidateCount: candidates.count,
            candidateOrdinal: candidateOrdinal,
            layers: layers)
    }

    /// Test-only construction seam for MLX-coupled runtime tests. Production consumers cannot
    /// call this internal API; they must cross the complete manifest/sensitivity/config/tokenizer
    /// trust boundary in `load`. Even here, reject malformed policies so cache-route tests cannot
    /// accidentally exercise a state that the authenticated loader would never produce.
    static func loadForTesting(
        candidate: KVTunerScheduleCandidate,
        matrixID: String,
        modelConfigHash: String,
        modelConfigSHA256: String,
        checkpointManifestHash: String,
        tokenizerSHA256: String,
        groupSize: Int
    ) throws -> KVTunerCandidateRuntimePolicy {
        guard candidate.ordinal == 0,
            candidate.totalPairBits > 0,
            candidate.meanAttentionOutputError.isFinite,
            candidate.meanAttentionOutputError >= 0,
            !candidate.layers.isEmpty,
            [64, 128].contains(groupSize),
            isLowercaseHex(candidate.analysisSHA256, length: 64),
            isLowercaseHex(modelConfigSHA256, length: 64),
            isLowercaseHex(tokenizerSHA256, length: 64)
        else {
            throw KVTunerCandidateRuntimePolicyError
                .candidateEnumerationFailed(.invalidCandidate)
        }

        var computedPairBits = 0
        for (position, layer) in candidate.layers.enumerated() {
            let pair = KVTunerPrecisionPair(
                keyBits: layer.keyBits, valueBits: layer.valueBits)
            let sum = computedPairBits.addingReportingOverflow(
                pair.pairBitCost)
            guard layer.layer == position,
                KVTunerSensitivityArtifact.canonicalPrecisionPairs
                    .contains(pair),
                !sum.overflow
            else {
                throw KVTunerCandidateRuntimePolicyError
                    .candidateEnumerationFailed(.invalidCandidate)
            }
            computedPairBits = sum.partialValue
        }
        guard computedPairBits == candidate.totalPairBits else {
            throw KVTunerCandidateRuntimePolicyError
                .candidateEnumerationFailed(.invalidCandidate)
        }

        let candidateList = [candidate]
        let candidateListSHA256: String
        let candidateSHA256: String
        do {
            candidateListSHA256 = try KVTunerScheduleSearch
                .candidateListSHA256(candidateList)
            candidateSHA256 = try KVTunerScheduleSearch
                .candidateSHA256(candidate)
        } catch {
            throw KVTunerCandidateRuntimePolicyError
                .candidateIdentityFailure
        }
        let calibrationManifestSHA256 = String(repeating: "d", count: 64)
        let sensitivityArtifactSHA256 = String(repeating: "e", count: 64)
        let layers = candidate.layers.map {
            KVTunerRuntimeLayerPolicy(
                layer: $0.layer,
                keyBits: $0.keyBits,
                valueBits: $0.valueBits)
        }
        let runtimePolicySHA256 = makeRuntimePolicySHA256(
            calibrationManifestSHA256: calibrationManifestSHA256,
            sensitivityArtifactSHA256: sensitivityArtifactSHA256,
            candidateListSHA256: candidateListSHA256,
            candidateSHA256: candidateSHA256,
            matrixID: matrixID,
            modelConfigHash: modelConfigHash,
            modelConfigSHA256: modelConfigSHA256,
            checkpointManifestHash: checkpointManifestHash,
            tokenizerSHA256: tokenizerSHA256,
            groupSize: groupSize,
            targetPairBitTotal: candidate.totalPairBits,
            candidateCount: candidateList.count,
            candidateOrdinal: candidate.ordinal,
            layers: layers)
        return KVTunerCandidateRuntimePolicy(
            runtimePolicySHA256: runtimePolicySHA256,
            calibrationManifestSHA256: calibrationManifestSHA256,
            sourceSensitivityArtifactSHA256:
                sensitivityArtifactSHA256,
            candidateListSHA256: candidateListSHA256,
            candidateSHA256: candidateSHA256,
            matrixID: matrixID,
            modelConfigHash: modelConfigHash,
            modelConfigSHA256: modelConfigSHA256,
            checkpointManifestHash: checkpointManifestHash,
            tokenizerSHA256: tokenizerSHA256,
            groupSize: groupSize,
            targetPairBitTotal: candidate.totalPairBits,
            candidateCount: candidateList.count,
            candidateOrdinal: candidate.ordinal,
            layers: layers)
    }

    private static func validateLiveTokenization(
        _ prompts: [KVTunerCalibrationPromptIdentity],
        phase: String,
        tokenizePrompt: (String) throws -> [Int]
    ) throws {
        for (position, prompt) in prompts.enumerated() {
            guard let text = String(
                data: prompt.promptUTF8,
                encoding: .utf8)
            else {
                throw KVTunerCandidateRuntimePolicyError
                    .promptTokenizationMismatch(
                        phase: phase, position: position)
            }
            let liveTokenIDs: [Int]
            do {
                liveTokenIDs = try tokenizePrompt(text)
            } catch {
                throw KVTunerCandidateRuntimePolicyError
                    .promptTokenizationMismatch(
                        phase: phase, position: position)
            }
            guard liveTokenIDs == prompt.tokenIDs else {
                throw KVTunerCandidateRuntimePolicyError
                    .promptTokenizationMismatch(
                        phase: phase, position: position)
            }
        }
    }

    private static func isLowercaseHex(
        _ value: String, length: Int
    ) -> Bool {
        value.utf8.count == length
            && value.utf8.allSatisfy {
                (48 ... 57).contains($0) || (97 ... 102).contains($0)
            }
    }

    private static func makeRuntimePolicySHA256(
        calibrationManifestSHA256: String,
        sensitivityArtifactSHA256: String,
        candidateListSHA256: String,
        candidateSHA256: String,
        matrixID: String,
        modelConfigHash: String,
        modelConfigSHA256: String,
        checkpointManifestHash: String,
        tokenizerSHA256: String,
        groupSize: Int,
        targetPairBitTotal: Int,
        candidateCount: Int,
        candidateOrdinal: Int,
        layers: [KVTunerRuntimeLayerPolicy]
    ) -> String {
        var transcript = RuntimePolicyTranscript(
            domain: "fast-mlx.kvtuner-candidate-runtime-policy.v1")
        for value in [
            calibrationManifestSHA256,
            sensitivityArtifactSHA256,
            candidateListSHA256,
            candidateSHA256,
            matrixID,
            modelConfigHash,
            modelConfigSHA256,
            checkpointManifestHash,
            tokenizerSHA256,
        ] {
            transcript.appendString(value)
        }
        transcript.appendInt(groupSize)
        transcript.appendInt(targetPairBitTotal)
        transcript.appendInt(candidateCount)
        transcript.appendInt(candidateOrdinal)
        transcript.appendInt(layers.count)
        for layer in layers {
            transcript.appendInt(layer.layer)
            transcript.appendInt(layer.keyBits)
            transcript.appendInt(layer.valueBits)
        }
        return sha256Hex(transcript.data)
    }

    /// Language-neutral policy identity: UTF-8 strings are length-prefixed with unsigned
    /// big-endian 64-bit byte counts and integer fields use signed big-endian 64-bit lanes.
    private struct RuntimePolicyTranscript {
        var data = Data()

        init(domain: String) {
            data.append(contentsOf: domain.utf8)
            data.append(0)
        }

        mutating func appendString(_ value: String) {
            let bytes = Array(value.utf8)
            appendUInt64(UInt64(bytes.count))
            data.append(contentsOf: bytes)
        }

        mutating func appendInt(_ value: Int) {
            var encoded = Int64(value).bigEndian
            withUnsafeBytes(of: &encoded) {
                data.append(contentsOf: $0)
            }
        }

        private mutating func appendUInt64(_ value: UInt64) {
            var encoded = value.bigEndian
            withUnsafeBytes(of: &encoded) {
                data.append(contentsOf: $0)
            }
        }
    }
}
