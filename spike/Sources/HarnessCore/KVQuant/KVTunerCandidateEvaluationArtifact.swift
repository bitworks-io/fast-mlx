import Foundation

public enum KVTunerCandidateEvaluationArtifactError:
    Error, Equatable, Sendable
{
    case malformedArtifact
    case unsupportedSchema(Int)
    case invalidProtocol
    case calibrationManifestMismatch
    case candidateMismatch
    case runtimePolicyMismatch
    case invalidExecutionEnvironment(String)
    case invalidRow(Int)
    case invalidGeneratedTokens(Int)
    case invalidFinishReason(Int)
    case outputDecodingFailed(Int)
    case rawOutputMismatch(Int)
    case stoppedOutputMismatch(Int)
    case runtimeReceiptMismatch(Int)
    case storageReconciliationMismatch(Int)
}

public enum KVTunerCandidateRuntimeContractError:
    Error, Equatable, Sendable
{
    case malformedModelConfig
    case modelConfigIdentityMismatch
    case unsupportedModelType(String)
    case invalidModelGeometry
    case unsupportedScalarDType(String)
    case invalidEOSTokenID
    case invalidCapacity
    case arithmeticOverflow
}

public enum KVTunerGSM8KScorerError: Error, Equatable, Sendable {
    case invalidUTF8
}

/// Pure Swift reproduction of the pinned lm-eval GSM8K `flexible-extract` plus exact-match
/// filters. Keeping this scorer beside the artifact validator makes candidate accuracy a derived
/// fact rather than a self-reported field from the measurement runner.
public enum KVTunerGSM8KScorer: Sendable {
    private static let extraction = try! NSRegularExpression(
        pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)")
    private static let greedyAnswerPrefix = try! NSRegularExpression(
        pattern: ".*#### ", options: [.dotMatchesLineSeparators])
    private static let terminalPeriod = try! NSRegularExpression(
        pattern: "\\.$")

    public static func flexibleExtract(_ outputUTF8: Data) throws -> String {
        guard let output = String(data: outputUTF8, encoding: .utf8) else {
            throw KVTunerGSM8KScorerError.invalidUTF8
        }
        let range = NSRange(output.startIndex..<output.endIndex, in: output)
        guard let match = extraction.matches(
            in: output, range: range
        ).last else {
            return "[invalid]"
        }
        for group in 1...2 {
            let groupRange = match.range(at: group)
            guard groupRange.location != NSNotFound,
                let swiftRange = Range(groupRange, in: output)
            else { continue }
            let value = output[swiftRange].trimmingCharacters(
                in: .whitespacesAndNewlines)
            if !value.isEmpty { return value }
        }
        return "[invalid]"
    }

    public static func exactMatchNormalize(_ value: String) -> String {
        var normalized = value
            .replacingOccurrences(of: ",", with: "")
            .replacingOccurrences(of: "$", with: "")
        let fullRange = NSRange(
            normalized.startIndex..<normalized.endIndex, in: normalized)
        normalized = greedyAnswerPrefix.stringByReplacingMatches(
            in: normalized,
            range: fullRange,
            withTemplate: "")
        let periodRange = NSRange(
            normalized.startIndex..<normalized.endIndex, in: normalized)
        normalized = terminalPeriod.stringByReplacingMatches(
            in: normalized,
            range: periodRange,
            withTemplate: "")
        return normalized.lowercased()
    }

    public static func isCorrect(
        outputUTF8: Data,
        normalizedTarget: String
    ) throws -> Bool {
        exactMatchNormalize(try flexibleExtract(outputUTF8))
            == normalizedTarget
    }
}

/// Path-free identity of the code, packages, hardware, and model inputs that produced every row
/// in one candidate artifact. Exact package pins are part of validation, not descriptive labels.
public struct KVTunerCandidateExecutionEnvironment:
    Codable, Equatable, Sendable
{
    public static let requiredBuildConfiguration = "Release"
    public static let requiredMLXSwiftVersion = "0.31.6"
    public static let requiredMLXSwiftLMRevision =
        "702e5a0eaf990e1f6d3db2b6e7d8872858a44055"
    public static let requiredMemoryCacheLimitBytes: UInt64 = 8 << 30

    public var harnessGitSHA: String
    public var buildConfiguration: String
    public var mlxSwiftVersion: String
    public var mlxSwiftLMRevision: String
    public var hardwareChip: String
    public var hardwareRAMBytes: UInt64
    public var memoryCacheLimitBytes: UInt64
    public var hardwareOS: String
    public var modelConfigHash: String
    public var modelConfigSHA256: String
    public var checkpointManifestHash: String
    public var tokenizerSHA256: String

    public init(
        harnessGitSHA: String,
        buildConfiguration: String,
        mlxSwiftVersion: String,
        mlxSwiftLMRevision: String,
        hardwareChip: String,
        hardwareRAMBytes: UInt64,
        memoryCacheLimitBytes: UInt64,
        hardwareOS: String,
        modelConfigHash: String,
        modelConfigSHA256: String,
        checkpointManifestHash: String,
        tokenizerSHA256: String
    ) {
        self.harnessGitSHA = harnessGitSHA
        self.buildConfiguration = buildConfiguration
        self.mlxSwiftVersion = mlxSwiftVersion
        self.mlxSwiftLMRevision = mlxSwiftLMRevision
        self.hardwareChip = hardwareChip
        self.hardwareRAMBytes = hardwareRAMBytes
        self.memoryCacheLimitBytes = memoryCacheLimitBytes
        self.hardwareOS = hardwareOS
        self.modelConfigHash = modelConfigHash
        self.modelConfigSHA256 = modelConfigSHA256
        self.checkpointManifestHash = checkpointManifestHash
        self.tokenizerSHA256 = tokenizerSHA256
    }

    fileprivate func validatedSHA256(
        runtimePolicy: KVTunerCandidateRuntimePolicy
    ) throws -> String {
        guard Self.isLowercaseHex(harnessGitSHA, length: 40) else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("harnessGitSHA")
        }
        guard buildConfiguration == Self.requiredBuildConfiguration else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("buildConfiguration")
        }
        guard mlxSwiftVersion == Self.requiredMLXSwiftVersion else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("mlxSwiftVersion")
        }
        guard mlxSwiftLMRevision == Self.requiredMLXSwiftLMRevision else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("mlxSwiftLMRevision")
        }
        guard Self.isPathFreeEvidence(hardwareChip) else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("hardwareChip")
        }
        guard hardwareRAMBytes > 0 else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("hardwareRAMBytes")
        }
        guard memoryCacheLimitBytes == Self.requiredMemoryCacheLimitBytes else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("memoryCacheLimitBytes")
        }
        guard Self.isPathFreeEvidence(hardwareOS) else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("hardwareOS")
        }
        guard modelConfigHash == runtimePolicy.modelConfigHash,
            Self.isLowercaseHex(modelConfigHash, length: 16)
        else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("modelConfigHash")
        }
        guard modelConfigSHA256 == runtimePolicy.modelConfigSHA256,
            Self.isLowercaseHex(modelConfigSHA256, length: 64)
        else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("modelConfigSHA256")
        }
        guard checkpointManifestHash
                == runtimePolicy.checkpointManifestHash,
            Self.isIdentityDigest(checkpointManifestHash)
        else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("checkpointManifestHash")
        }
        guard tokenizerSHA256 == runtimePolicy.tokenizerSHA256,
            Self.isLowercaseHex(tokenizerSHA256, length: 64)
        else {
            throw KVTunerCandidateEvaluationArtifactError
                .invalidExecutionEnvironment("tokenizerSHA256")
        }

        var transcript = EnvironmentTranscript(
            domain: "fast-mlx.kvtuner-candidate-environment.v1")
        for value in [
            harnessGitSHA,
            buildConfiguration,
            mlxSwiftVersion,
            mlxSwiftLMRevision,
            hardwareChip,
            hardwareOS,
            modelConfigHash,
            modelConfigSHA256,
            checkpointManifestHash,
            tokenizerSHA256,
        ] {
            transcript.appendString(value)
        }
        transcript.appendUInt64(hardwareRAMBytes)
        transcript.appendUInt64(memoryCacheLimitBytes)
        return sha256Hex(transcript.data)
    }

    private static func isPathFreeEvidence(_ value: String) -> Bool {
        guard !value.isEmpty, value != "unknown", value.count <= 256,
            value == value.trimmingCharacters(in: .whitespacesAndNewlines),
            !value.contains("/"), !value.contains("\\"),
            !value.contains("\n"), !value.contains("\r")
        else { return false }
        return value.unicodeScalars.allSatisfy {
            !CharacterSet.controlCharacters.contains($0)
        }
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

    private struct EnvironmentTranscript {
        var data = Data()

        init(domain: String) {
            data.append(contentsOf: domain.utf8)
            data.append(0)
        }

        mutating func appendString(_ value: String) {
            let bytes = Data(value.utf8)
            appendUInt64(UInt64(bytes.count))
            data.append(bytes)
        }

        mutating func appendUInt64(_ value: UInt64) {
            var encoded = value.bigEndian
            withUnsafeBytes(of: &encoded) {
                data.append(contentsOf: $0)
            }
        }
    }
}

public struct KVTunerCandidateRuntimeGeometry:
    Codable, Equatable, Sendable
{
    public var layerCount: Int
    public var kvHeadCount: Int
    public var headDimension: Int

    public init(
        layerCount: Int,
        kvHeadCount: Int,
        headDimension: Int
    ) {
        self.layerCount = layerCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
    }
}

/// Immutable runtime facts derived independently from the exact authenticated Qwen3 config and
/// the candidate decoder's fixed allocation policy. Candidate receipts are compared against this
/// contract; none of their geometry or byte-accounting inputs are trusted as expected values.
public struct KVTunerCandidateRuntimeContract: Equatable, Sendable {
    public static let cacheReserveTokens = 384
    public static let cacheGrowthChunkTokens = 256

    public let modelConfigSHA256: String
    public let geometry: KVTunerCandidateRuntimeGeometry
    public let sequenceCount: Int
    public let metadataScalarBytes: Int
    /// Production supplies this from the same live tokenizer whose exact files produced
    /// `runtimePolicy.tokenizerSHA256`; contract loading additionally requires it to match the
    /// authenticated Qwen3 config. Pure HarnessCore still cannot attest an arbitrary in-process
    /// tokenizer/decoder closure, so that closure remains a caller trust boundary.
    public let eosTokenID: Int

    private init(
        modelConfigSHA256: String,
        geometry: KVTunerCandidateRuntimeGeometry,
        sequenceCount: Int,
        metadataScalarBytes: Int,
        eosTokenID: Int
    ) {
        self.modelConfigSHA256 = modelConfigSHA256
        self.geometry = geometry
        self.sequenceCount = sequenceCount
        self.metadataScalarBytes = metadataScalarBytes
        self.eosTokenID = eosTokenID
    }

    /// Builds the contract from the same exact config bytes authenticated by the immutable
    /// candidate policy. The Qwen3 geometry fields are required explicitly; a missing field
    /// cannot silently become a model-family default at this evidence boundary.
    public static func load(
        exactModelConfigData: Data,
        runtimePolicy: KVTunerCandidateRuntimePolicy,
        eosTokenID: Int
    ) throws -> KVTunerCandidateRuntimeContract {
        guard eosTokenID >= 0 else {
            throw KVTunerCandidateRuntimeContractError.invalidEOSTokenID
        }
        guard fnv1a64(exactModelConfigData) == runtimePolicy.modelConfigHash,
            sha256Hex(exactModelConfigData) == runtimePolicy.modelConfigSHA256
        else {
            throw KVTunerCandidateRuntimeContractError
                .modelConfigIdentityMismatch
        }
        let config: Qwen3Config
        do {
            config = try JSONDecoder().decode(
                Qwen3Config.self, from: exactModelConfigData)
        } catch {
            throw KVTunerCandidateRuntimeContractError.malformedModelConfig
        }
        guard config.modelType == "qwen3" else {
            throw KVTunerCandidateRuntimeContractError.unsupportedModelType(
                config.modelType)
        }
        guard eosTokenID == config.eosTokenID else {
            throw KVTunerCandidateRuntimeContractError.invalidEOSTokenID
        }
        let queryProjectionWidth = config.numAttentionHeads
            .multipliedReportingOverflow(by: config.headDimension)
        guard !queryProjectionWidth.overflow,
            config.numHiddenLayers == runtimePolicy.layers.count,
            config.numHiddenLayers > 0,
            config.hiddenSize > 0,
            config.numAttentionHeads > 0,
            config.numKeyValueHeads > 0,
            config.numAttentionHeads.isMultiple(
                of: config.numKeyValueHeads),
            config.headDimension > 0,
            config.headDimension.isMultiple(of: runtimePolicy.groupSize)
        else {
            throw KVTunerCandidateRuntimeContractError.invalidModelGeometry
        }
        let metadataScalarBytes: Int
        switch config.torchDType {
        case "bfloat16", "float16":
            metadataScalarBytes = 2
        case "float32":
            metadataScalarBytes = 4
        default:
            throw KVTunerCandidateRuntimeContractError
                .unsupportedScalarDType(config.torchDType)
        }
        return KVTunerCandidateRuntimeContract(
            modelConfigSHA256: runtimePolicy.modelConfigSHA256,
            geometry: KVTunerCandidateRuntimeGeometry(
                layerCount: config.numHiddenLayers,
                kvHeadCount: config.numKeyValueHeads,
                headDimension: config.headDimension),
            sequenceCount: 1,
            metadataScalarBytes: metadataScalarBytes,
            eosTokenID: eosTokenID)
    }

    /// Replays `CompiledMLXDecoder`'s candidate-cache allocation exactly. The first prompt gets
    /// 384 decode-reserve tokens rounded to 256; later prompts reuse that allocation and grow in
    /// 256-token chunks only when the next prefill no longer fits.
    public func capacityTokens(
        promptTokenCount: Int,
        generatedTokenCount: Int,
        previousCapacityTokens: Int?
    ) throws -> Int {
        guard promptTokenCount > 0, generatedTokenCount > 0 else {
            throw KVTunerCandidateRuntimeContractError.invalidCapacity
        }
        let chunk = Self.cacheGrowthChunkTokens
        let startingCapacity: Int
        if let previousCapacityTokens {
            guard previousCapacityTokens > 0,
                previousCapacityTokens.isMultiple(of: chunk)
            else {
                throw KVTunerCandidateRuntimeContractError.invalidCapacity
            }
            startingCapacity = previousCapacityTokens
        } else {
            let reserved = promptTokenCount.addingReportingOverflow(
                Self.cacheReserveTokens)
            guard !reserved.overflow else {
                throw KVTunerCandidateRuntimeContractError
                    .arithmeticOverflow
            }
            startingCapacity = try Self.roundedUp(
                reserved.partialValue, multiple: chunk)
        }
        // The submit-first decoder consumes every emitted token into the cache before returning
        // it, so the final offset is prompt + generated (not prompt + generated - 1).
        let finalOffset = promptTokenCount.addingReportingOverflow(
            generatedTokenCount)
        guard !finalOffset.overflow else {
            throw KVTunerCandidateRuntimeContractError.arithmeticOverflow
        }
        guard finalOffset.partialValue > startingCapacity else {
            return startingCapacity
        }
        return try Self.roundedUp(
            finalOffset.partialValue, multiple: chunk)
    }

    /// Full-precision K plus V materialization for one sequentially consumed layer. The scalar
    /// width comes from the authenticated config, never from receipt metadata.
    public func workspaceBytes(capacityTokens: Int) throws -> Int {
        guard capacityTokens > 0 else {
            throw KVTunerCandidateRuntimeContractError.invalidCapacity
        }
        var result = 1
        for factor in [
            sequenceCount,
            geometry.kvHeadCount,
            capacityTokens,
            geometry.headDimension,
            2,
            metadataScalarBytes,
        ] {
            let product = result.multipliedReportingOverflow(by: factor)
            guard !product.overflow else {
                throw KVTunerCandidateRuntimeContractError.arithmeticOverflow
            }
            result = product.partialValue
        }
        return result
    }

    private static func roundedUp(
        _ value: Int,
        multiple: Int
    ) throws -> Int {
        let adjusted = value.addingReportingOverflow(multiple - 1)
        guard !adjusted.overflow else {
            throw KVTunerCandidateRuntimeContractError.arithmeticOverflow
        }
        let units = adjusted.partialValue / multiple
        let rounded = units.multipliedReportingOverflow(by: multiple)
        guard !rounded.overflow else {
            throw KVTunerCandidateRuntimeContractError.arithmeticOverflow
        }
        return rounded.partialValue
    }

    private struct Qwen3Config: Decodable {
        let modelType: String
        let eosTokenID: Int
        let numHiddenLayers: Int
        let hiddenSize: Int
        let numAttentionHeads: Int
        let numKeyValueHeads: Int
        let headDimension: Int
        let torchDType: String

        private enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case eosTokenID = "eos_token_id"
            case numHiddenLayers = "num_hidden_layers"
            case hiddenSize = "hidden_size"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case headDimension = "head_dim"
            case torchDType = "torch_dtype"
        }
    }
}

/// Typed, scalar-only receipt captured from the exact heterogeneous cache after one generation.
/// It mirrors the actor-confined MLX telemetry without allowing an MLX value to escape the actor.
public struct KVTunerCandidateRuntimeReceipt:
    Codable, Equatable, Sendable
{
    public var runtimePolicySHA256: String
    public var cachedTokens: Int
    public var capacityTokens: Int
    public var layers: [KVTunerRuntimeLayerPolicy]
    public var geometry: KVTunerCandidateRuntimeGeometry
    public var groupSize: Int
    public var sequenceCount: Int
    public var metadataScalarBytes: Int
    public var actualPayloadBytes: Int
    public var actualMetadataBytes: Int
    public var actualControlBytes: Int
    public var actualWorkspaceBytes: Int
    public var actualTotalPersistentBytes: Int
    public var actualTotalBytes: Int

    public init(
        runtimePolicySHA256: String,
        cachedTokens: Int,
        capacityTokens: Int,
        layers: [KVTunerRuntimeLayerPolicy],
        geometry: KVTunerCandidateRuntimeGeometry,
        groupSize: Int,
        sequenceCount: Int,
        metadataScalarBytes: Int,
        actualPayloadBytes: Int,
        actualMetadataBytes: Int,
        actualControlBytes: Int,
        actualWorkspaceBytes: Int,
        actualTotalPersistentBytes: Int,
        actualTotalBytes: Int
    ) {
        self.runtimePolicySHA256 = runtimePolicySHA256
        self.cachedTokens = cachedTokens
        self.capacityTokens = capacityTokens
        self.layers = layers
        self.geometry = geometry
        self.groupSize = groupSize
        self.sequenceCount = sequenceCount
        self.metadataScalarBytes = metadataScalarBytes
        self.actualPayloadBytes = actualPayloadBytes
        self.actualMetadataBytes = actualMetadataBytes
        self.actualControlBytes = actualControlBytes
        self.actualWorkspaceBytes = actualWorkspaceBytes
        self.actualTotalPersistentBytes = actualTotalPersistentBytes
        self.actualTotalBytes = actualTotalBytes
    }
}

/// Why greedy generation stopped. Text stop strings are canonical post-processing in this
/// protocol; the engine itself must either emit the caller-bound live-tokenizer EOS token or
/// consume the full generation budget, preventing an arbitrary early prefix from qualifying as a
/// complete row.
public enum KVTunerCandidateFinishReason: String, Codable, Sendable {
    case endOfSequence = "end-of-sequence"
    case generationBudgetExhausted = "generation-budget-exhausted"
}

public struct KVTunerCandidateOutputRow: Codable, Equatable, Sendable {
    public var ordinal: Int
    public var promptSHA256: String
    public var promptTokenIDsSHA256: String
    public var generatedTokenIDs: [Int]
    public var rawDecodedUTF8: Data
    public var outputUTF8: Data
    public var finishReason: KVTunerCandidateFinishReason
    public var runtimeReceipt: KVTunerCandidateRuntimeReceipt

    public init(
        ordinal: Int,
        promptSHA256: String,
        promptTokenIDsSHA256: String,
        generatedTokenIDs: [Int],
        rawDecodedUTF8: Data,
        outputUTF8: Data,
        finishReason: KVTunerCandidateFinishReason,
        runtimeReceipt: KVTunerCandidateRuntimeReceipt
    ) {
        self.ordinal = ordinal
        self.promptSHA256 = promptSHA256
        self.promptTokenIDsSHA256 = promptTokenIDsSHA256
        self.generatedTokenIDs = generatedTokenIDs
        self.rawDecodedUTF8 = rawDecodedUTF8
        self.outputUTF8 = outputUTF8
        self.finishReason = finishReason
        self.runtimeReceipt = runtimeReceipt
    }
}

/// Exact per-candidate output evidence. Schema 2 authenticates the executable candidate policy,
/// environment, raw token stream, canonical stop handling, and per-row cache/storage telemetry.
public struct KVTunerCandidateEvaluationArtifact:
    Codable, Equatable, Sendable
{
    public var schemaVersion: Int
    public var evaluationProtocol: KVTunerSearchEvaluationProtocol
    public var promptManifestSHA256: String
    public var candidateOrdinal: Int
    public var candidateSHA256: String
    public var runtimePolicySHA256: String
    public var executionEnvironment: KVTunerCandidateExecutionEnvironment
    public var rows: [KVTunerCandidateOutputRow]

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case evaluationProtocol
        case promptManifestSHA256
        case candidateOrdinal
        case candidateSHA256
        case runtimePolicySHA256
        case executionEnvironment
        case rows
    }

    public init(
        schemaVersion: Int,
        evaluationProtocol: KVTunerSearchEvaluationProtocol,
        promptManifestSHA256: String,
        candidateOrdinal: Int,
        candidateSHA256: String,
        runtimePolicySHA256: String,
        executionEnvironment: KVTunerCandidateExecutionEnvironment,
        rows: [KVTunerCandidateOutputRow]
    ) {
        self.schemaVersion = schemaVersion
        self.evaluationProtocol = evaluationProtocol
        self.promptManifestSHA256 = promptManifestSHA256
        self.candidateOrdinal = candidateOrdinal
        self.candidateSHA256 = candidateSHA256
        self.runtimePolicySHA256 = runtimePolicySHA256
        self.executionEnvironment = executionEnvironment
        self.rows = rows
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
        guard schemaVersion == 2 else {
            throw KVTunerCandidateEvaluationArtifactError.unsupportedSchema(
                schemaVersion)
        }
        evaluationProtocol = try container.decode(
            KVTunerSearchEvaluationProtocol.self,
            forKey: .evaluationProtocol)
        promptManifestSHA256 = try container.decode(
            String.self, forKey: .promptManifestSHA256)
        candidateOrdinal = try container.decode(
            Int.self, forKey: .candidateOrdinal)
        candidateSHA256 = try container.decode(
            String.self, forKey: .candidateSHA256)
        runtimePolicySHA256 = try container.decode(
            String.self, forKey: .runtimePolicySHA256)
        executionEnvironment = try container.decode(
            KVTunerCandidateExecutionEnvironment.self,
            forKey: .executionEnvironment)
        rows = try container.decode(
            [KVTunerCandidateOutputRow].self,
            forKey: .rows)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(
            evaluationProtocol, forKey: .evaluationProtocol)
        try container.encode(
            promptManifestSHA256, forKey: .promptManifestSHA256)
        try container.encode(candidateOrdinal, forKey: .candidateOrdinal)
        try container.encode(candidateSHA256, forKey: .candidateSHA256)
        try container.encode(
            runtimePolicySHA256, forKey: .runtimePolicySHA256)
        try container.encode(
            executionEnvironment, forKey: .executionEnvironment)
        try container.encode(rows, forKey: .rows)
    }

    public func validated(
        exactArtifactData: Data,
        runtimePolicy: KVTunerCandidateRuntimePolicy,
        runtimeContract: KVTunerCandidateRuntimeContract,
        calibrationManifest: KVTunerCalibrationManifest,
        exactCalibrationManifestData: Data,
        decodeTokenIDs: ([Int]) throws -> String
    ) throws -> KVTunerCandidateEvaluation {
        let decoded: KVTunerCandidateEvaluationArtifact
        do {
            decoded = try JSONDecoder().decode(
                KVTunerCandidateEvaluationArtifact.self,
                from: exactArtifactData)
        } catch let error as KVTunerCandidateEvaluationArtifactError {
            throw error
        } catch {
            throw KVTunerCandidateEvaluationArtifactError.malformedArtifact
        }
        guard decoded == self else {
            throw KVTunerCandidateEvaluationArtifactError.malformedArtifact
        }
        guard schemaVersion == 2 else {
            throw KVTunerCandidateEvaluationArtifactError.unsupportedSchema(
                schemaVersion)
        }
        guard evaluationProtocol == .canonical else {
            throw KVTunerCandidateEvaluationArtifactError.invalidProtocol
        }

        let decodedManifest: KVTunerCalibrationManifest
        do {
            decodedManifest = try JSONDecoder().decode(
                KVTunerCalibrationManifest.self,
                from: exactCalibrationManifestData)
            guard decodedManifest == calibrationManifest else {
                throw KVTunerCandidateEvaluationArtifactError
                    .calibrationManifestMismatch
            }
            _ = try decodedManifest.validated()
        } catch let error as KVTunerCandidateEvaluationArtifactError {
            throw error
        } catch {
            throw KVTunerCandidateEvaluationArtifactError
                .calibrationManifestMismatch
        }
        guard promptManifestSHA256
                == sha256Hex(exactCalibrationManifestData),
            runtimePolicy.calibrationManifestSHA256
                == promptManifestSHA256
        else {
            throw KVTunerCandidateEvaluationArtifactError
                .calibrationManifestMismatch
        }

        guard candidateOrdinal == runtimePolicy.candidateOrdinal,
            candidateSHA256 == runtimePolicy.candidateSHA256
        else {
            throw KVTunerCandidateEvaluationArtifactError.candidateMismatch
        }
        guard runtimePolicySHA256 == runtimePolicy.runtimePolicySHA256 else {
            throw KVTunerCandidateEvaluationArtifactError
                .runtimePolicyMismatch
        }
        guard runtimeContract.modelConfigSHA256
                == runtimePolicy.modelConfigSHA256,
            runtimeContract.geometry.layerCount == runtimePolicy.layers.count
        else {
            throw KVTunerCandidateEvaluationArtifactError
                .runtimePolicyMismatch
        }
        let environmentSHA256 = try executionEnvironment.validatedSHA256(
            runtimePolicy: runtimePolicy)

        guard rows.count == KVTunerScheduleSearch.requiredSearchPromptCount,
            rows.count == decodedManifest.searchPrompts.count,
            rows.count == decodedManifest.searchNormalizedTargets.count
        else {
            throw KVTunerCandidateEvaluationArtifactError.invalidRow(
                rows.count)
        }
        var correctCount = 0
        var expectedCapacityTokens: Int?
        for (ordinal, row) in rows.enumerated() {
            let prompt = decodedManifest.searchPrompts[ordinal]
            guard row.ordinal == ordinal,
                row.promptSHA256 == prompt.promptSHA256,
                row.promptTokenIDsSHA256 == prompt.tokenIDsSHA256,
                row.promptTokenIDsSHA256
                    == taskTokenIDsSHA256(prompt.tokenIDs)
            else {
                throw KVTunerCandidateEvaluationArtifactError.invalidRow(
                    ordinal)
            }
            guard (1...evaluationProtocol.maxGeneratedTokens).contains(
                row.generatedTokenIDs.count),
                row.generatedTokenIDs.allSatisfy({ $0 >= 0 })
            else {
                throw KVTunerCandidateEvaluationArtifactError
                    .invalidGeneratedTokens(ordinal)
            }
            switch row.finishReason {
            case .endOfSequence:
                guard row.generatedTokenIDs.last
                        == runtimeContract.eosTokenID,
                    !row.generatedTokenIDs.dropLast().contains(
                        runtimeContract.eosTokenID)
                else {
                    throw KVTunerCandidateEvaluationArtifactError
                        .invalidFinishReason(ordinal)
                }
            case .generationBudgetExhausted:
                guard row.generatedTokenIDs.count
                        == evaluationProtocol.maxGeneratedTokens,
                    !row.generatedTokenIDs.contains(
                        runtimeContract.eosTokenID)
                else {
                    throw KVTunerCandidateEvaluationArtifactError
                        .invalidFinishReason(ordinal)
                }
            }

            let rawOutput: String
            do {
                rawOutput = try decodeTokenIDs(row.generatedTokenIDs)
            } catch {
                throw KVTunerCandidateEvaluationArtifactError
                    .outputDecodingFailed(ordinal)
            }
            guard Data(rawOutput.utf8) == row.rawDecodedUTF8 else {
                throw KVTunerCandidateEvaluationArtifactError
                    .rawOutputMismatch(ordinal)
            }
            let stoppedOutput = Self.canonicalStoppedOutput(
                rawOutput,
                stopSequences: evaluationProtocol.stopSequences)
            guard Data(stoppedOutput.utf8) == row.outputUTF8 else {
                throw KVTunerCandidateEvaluationArtifactError
                    .stoppedOutputMismatch(ordinal)
            }

            do {
                expectedCapacityTokens = try runtimeContract.capacityTokens(
                    promptTokenCount: prompt.tokenIDs.count,
                    generatedTokenCount: row.generatedTokenIDs.count,
                    previousCapacityTokens: expectedCapacityTokens)
            } catch {
                throw KVTunerCandidateEvaluationArtifactError
                    .runtimeReceiptMismatch(ordinal)
            }
            try validateRuntimeReceipt(
                row.runtimeReceipt,
                ordinal: ordinal,
                promptTokenCount: prompt.tokenIDs.count,
                generatedTokenCount: row.generatedTokenIDs.count,
                expectedCapacityTokens: expectedCapacityTokens!,
                runtimePolicy: runtimePolicy,
                runtimeContract: runtimeContract)
            do {
                if try KVTunerGSM8KScorer.isCorrect(
                    outputUTF8: row.outputUTF8,
                    normalizedTarget:
                        decodedManifest.searchNormalizedTargets[ordinal])
                {
                    correctCount += 1
                }
            } catch {
                throw KVTunerCandidateEvaluationArtifactError
                    .stoppedOutputMismatch(ordinal)
            }
        }
        return KVTunerCandidateEvaluation(
            candidateOrdinal: candidateOrdinal,
            candidateSHA256: candidateSHA256,
            runtimePolicySHA256: runtimePolicySHA256,
            environmentSHA256: environmentSHA256,
            correctCount: correctCount,
            totalCount: rows.count,
            outputSHA256: sha256Hex(exactArtifactData))
    }

    private func validateRuntimeReceipt(
        _ receipt: KVTunerCandidateRuntimeReceipt,
        ordinal: Int,
        promptTokenCount: Int,
        generatedTokenCount: Int,
        expectedCapacityTokens: Int,
        runtimePolicy: KVTunerCandidateRuntimePolicy,
        runtimeContract: KVTunerCandidateRuntimeContract
    ) throws {
        let expectedCached = promptTokenCount.addingReportingOverflow(
            generatedTokenCount)
        guard !expectedCached.overflow,
            receipt.runtimePolicySHA256
                == runtimePolicy.runtimePolicySHA256,
            receipt.cachedTokens == expectedCached.partialValue,
            receipt.capacityTokens == expectedCapacityTokens,
            receipt.capacityTokens >= receipt.cachedTokens,
            receipt.layers == runtimePolicy.layers,
            receipt.geometry == runtimeContract.geometry,
            receipt.groupSize == runtimePolicy.groupSize,
            receipt.sequenceCount == runtimeContract.sequenceCount,
            receipt.metadataScalarBytes
                == runtimeContract.metadataScalarBytes
        else {
            throw KVTunerCandidateEvaluationArtifactError
                .runtimeReceiptMismatch(ordinal)
        }

        let allocation: KVTunerStorageAllocation
        do {
            let expectedWorkspaceBytes = try runtimeContract.workspaceBytes(
                capacityTokens: expectedCapacityTokens)
            guard receipt.actualWorkspaceBytes == expectedWorkspaceBytes else {
                throw KVTunerCandidateEvaluationArtifactError
                    .runtimeReceiptMismatch(ordinal)
            }
            allocation = try KVStorageFormat.kvtunerAllocation(
                layerPolicy: runtimePolicy.layers.map {
                    KVLayerPrecision(
                        layer: $0.layer,
                        keyBits: $0.keyBits,
                        valueBits: $0.valueBits)
                },
                groupSize: runtimePolicy.groupSize,
                geometry: KVStorageGeometry(
                    layerCount: runtimeContract.geometry.layerCount,
                    kvHeadCount: runtimeContract.geometry.kvHeadCount,
                    headDimension: runtimeContract.geometry.headDimension),
                capacityTokens: expectedCapacityTokens,
                sequences: runtimeContract.sequenceCount,
                metadataScalarBytes:
                    runtimeContract.metadataScalarBytes,
                maximumLayerWorkspaceBytes: expectedWorkspaceBytes)
        } catch let error as KVTunerCandidateEvaluationArtifactError {
            throw error
        } catch {
            throw KVTunerCandidateEvaluationArtifactError
                .storageReconciliationMismatch(ordinal)
        }
        guard receipt.actualPayloadBytes == allocation.payloadBytes,
            receipt.actualMetadataBytes == allocation.metadataBytes,
            receipt.actualControlBytes == allocation.controlBytes,
            receipt.actualWorkspaceBytes == allocation.workspaceBytes,
            receipt.actualTotalPersistentBytes
                == allocation.totalPersistentBytes,
            receipt.actualTotalBytes == allocation.totalBytes
        else {
            throw KVTunerCandidateEvaluationArtifactError
                .storageReconciliationMismatch(ordinal)
        }
    }

    static func canonicalStoppedOutput(
        _ rawOutput: String,
        stopSequences: [String]
    ) -> String {
        var earliest = rawOutput.endIndex
        for stop in stopSequences where !stop.isEmpty {
            if let range = rawOutput.range(of: stop),
                range.lowerBound < earliest
            {
                earliest = range.lowerBound
            }
        }
        return String(rawOutput[..<earliest])
    }
}
