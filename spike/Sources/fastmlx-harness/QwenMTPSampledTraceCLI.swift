import CryptoKit
import Foundation
import HarnessCore
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import SpikeCore
import Tokenizers

struct QwenMTPSampledTraceArguments: Equatable, Sendable {
    let targetPath: String
    let drafterPath: String
    let evidencePath: String
}

enum QwenMTPSampledTraceCLIError: Error, Equatable, CustomStringConvertible, Sendable {
    case missingFlag(String)
    case duplicateFlag(String)
    case unknownFlag
    case missingValue(String)
    case unexpectedPositional
    case releaseBuildRequired
    case evidenceMustBeNew
    case evidenceWriteFailed
    case unexpectedArtifactRequest
    case evidenceReadFailed

    var description: String {
        switch self {
        case .missingFlag(let flag): return "missing required \(flag)"
        case .duplicateFlag(let flag): return "duplicate \(flag)"
        case .unknownFlag: return "unknown flag"
        case .missingValue(let flag): return "\(flag) requires a value"
        case .unexpectedPositional: return "unexpected positional argument"
        case .releaseBuildRequired: return "sampled MTP trace requires a Release build"
        case .evidenceMustBeNew: return "--evidence must name a new file"
        case .evidenceWriteFailed: return "failed to write sampled MTP trace evidence"
        case .unexpectedArtifactRequest: return "unexpected exact artifact request"
        case .evidenceReadFailed: return "failed to read sampled MTP trace evidence"
        }
    }
}

func validateQwenMTPSampledTrace(arguments: [String]) throws {
    guard arguments.count == 2, arguments[0] == "--evidence",
        !arguments[1].hasPrefix("--")
    else {
        throw QwenMTPSampledTraceCLIError.missingFlag("--evidence")
    }
    let data: Data
    do {
        data = try Data(contentsOf: URL(fileURLWithPath: arguments[1]))
    } catch {
        throw QwenMTPSampledTraceCLIError.evidenceReadFailed
    }
    _ = try QwenMTPSampledTraceGate.validateJSONL(data)
    print("validate-qwen-mtp-sampled-trace PASS")
}

func parseQwenMTPSampledTraceArguments(
    _ arguments: [String]
) throws -> QwenMTPSampledTraceArguments {
    let allowed = Set(["--target", "--drafter", "--evidence"])
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw QwenMTPSampledTraceCLIError.unexpectedPositional
        }
        guard allowed.contains(flag) else {
            throw QwenMTPSampledTraceCLIError.unknownFlag
        }
        guard values[flag] == nil else {
            throw QwenMTPSampledTraceCLIError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count,
            !arguments[index + 1].hasPrefix("--")
        else {
            throw QwenMTPSampledTraceCLIError.missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }
    for flag in ["--target", "--drafter", "--evidence"] where values[flag] == nil {
        throw QwenMTPSampledTraceCLIError.missingFlag(flag)
    }
    return QwenMTPSampledTraceArguments(
        targetPath: values["--target"]!,
        drafterPath: values["--drafter"]!,
        evidencePath: values["--evidence"]!)
}

func qwenMTPSampledTraceExternalDiagnostic(_ error: Error) -> String {
    if let error = error as? QwenMTPSampledTraceCLIError {
        return error.description
    }
    return "sampled MTP trace failed"
}

func runQwenMTPSampledTrace(arguments: [String]) async throws {
    #if DEBUG
    throw QwenMTPSampledTraceCLIError.releaseBuildRequired
    #else
    let arguments = try parseQwenMTPSampledTraceArguments(arguments)
    let evidenceURL = URL(fileURLWithPath: arguments.evidencePath)
    guard !FileManager.default.fileExists(atPath: evidenceURL.path) else {
        throw QwenMTPSampledTraceCLIError.evidenceMustBeNew
    }

    let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
        from: QwenMTPSampledTraceDownloader(
            target: URL(fileURLWithPath: arguments.targetPath, isDirectory: true),
            drafter: URL(fileURLWithPath: arguments.drafterPath, isDirectory: true)),
        using: #huggingFaceTokenizerLoader())
    let promptTokens = pair.target.tokenizer.encode(text: QwenMTPSampledTraceGate.requiredPrompt)
    let input = LMInput(tokens: MLXArray(promptTokens))
    let cases = try [
        runSampledMTPDepthOneDiagnostic(
            input: input,
            target: pair.target.model,
            drafter: pair.drafter.model,
            branch: .accepted),
        runSampledMTPDepthOneDiagnostic(
            input: input,
            target: pair.target.model,
            drafter: pair.drafter.model,
            branch: .rejected),
    ].map(makeSampledTraceCase)

    let binding = pair.binding
    let payload = QwenMTPSampledTraceEvidence(
        schemaVersion: QwenMTPSampledTraceGate.schemaVersion,
        source: .init(
            targetModelID: binding.targetModelID,
            targetRevision: binding.targetRevision,
            targetConfigSHA256:
                QwenMTPKnownArtifactLocks.qwen35_9BDepth1.targetIdentity.configSHA256,
            targetTensorManifestSHA256:
                QwenMTPKnownArtifactLocks.qwen35_9BDepth1.targetIdentity.tensorManifestSHA256,
            drafterModelID: binding.drafterModelID,
            drafterRevision: binding.drafterRevision,
            drafterConfigSHA256:
                QwenMTPKnownArtifactLocks.qwen35_9BDepth1.drafterIdentity.configSHA256,
            drafterTensorManifestSHA256:
                QwenMTPKnownArtifactLocks.qwen35_9BDepth1.drafterIdentity.tensorManifestSHA256,
            sourceRevision: binding.sourceRevision,
            runtimeBlockSize: binding.runtimeBlockSize,
            maximumAcceptedDraftTokens: binding.maximumAcceptedDraftTokens),
        sampling: QwenMTPSampledTraceGate.requiredSampling,
        cases: cases)
    _ = try QwenMTPSampledTraceGate.validate(payload)

    let modelConfig = ProvenanceCLI.modelConfig(at: arguments.targetPath)
    let record = ResultRecord(
        subcommand: QwenMTPSampledTraceGate.subcommand,
        provenance: Provenance(
            date: ISO8601DateFormatter().string(from: Date()),
            hardwareChip: ProvenanceCLI.chipBrand(),
            hardwareRAMBytes: ProvenanceCLI.ramBytes(),
            hardwareOS: ProvenanceCLI.osVersion(),
            harnessGitSHA: try ProvenanceCLI.qualificationHarnessGitSHA(),
            mlxSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: ProvenanceCLI.mlxSwiftLMRevision,
            modelPath: binding.targetModelID,
            modelConfigHash: modelConfig.hash,
            modelCheckpointManifestHash:
                try ProvenanceCLI.checkpointManifestHash(at: arguments.targetPath),
            modelQuant: modelConfig.quant,
            corpusId: QwenMTPSampledTraceGate.corpusID,
            corpusContentHash: QwenMTPSampledTraceGate.requiredPromptSHA256,
            nonce: UUID().uuidString),
        payload: payload)
    try writeValidatedFreshSampledTrace(
        Data((try record.jsonLine() + "\n").utf8), to: evidenceURL)
    print("qwen-mtp-sampled-trace PASS cases=2 cacheLayers=32")
    #endif
}

private struct QwenMTPSampledTraceDownloader: Downloader {
    let target: URL
    let drafter: URL

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let source = QwenMTPSampledTraceGate.requiredSource
        guard !useLatest,
            patterns == ["*.safetensors", "*.json", "*.jinja"]
        else {
            throw QwenMTPSampledTraceCLIError.unexpectedArtifactRequest
        }
        progressHandler(Progress(totalUnitCount: 1))
        switch (id, revision) {
        case (source.targetModelID, source.targetRevision): return target
        case (source.drafterModelID, source.drafterRevision): return drafter
        default: throw QwenMTPSampledTraceCLIError.unexpectedArtifactRequest
        }
    }
}

private func makeSampledTraceCase(
    _ result: SampledMTPDepthOneDiagnosticResult
) -> QwenMTPSampledTraceCaseEvidence {
    QwenMTPSampledTraceCaseEvidence(
        branch: result.branch == .accepted ? .accepted : .rejected,
        promptSHA256: QwenMTPSampledTraceGate.requiredPromptSHA256,
        promptTokenCount: result.promptTokenCount,
        bonusToken: result.bonusToken,
        proposedToken: result.proposedToken,
        emittedToken: result.emittedToken,
        proposalUniform: result.proposalUniform,
        acceptanceUniform: result.acceptanceUniform,
        residualUniform: result.residualUniform,
        targetProbabilities: result.targetProbabilities,
        draftProbabilities: result.draftProbabilities,
        candidateCache: sampledTraceCacheFingerprint(result.candidateCache),
        scalarCache: sampledTraceCacheFingerprint(result.scalarCache))
}

private func sampledTraceCacheFingerprint(
    _ cache: [KVCache]
) -> [QwenMTPSampledTraceCacheFingerprint] {
    cache.enumerated().map { layerIndex, entry in
        var metaHasher = SHA256()
        for (index, value) in entry.metaState.enumerated() {
            metaHasher.update(data: Data("\(index)\u{0}\(value)\u{0}".utf8))
        }
        let states = entry.state.enumerated().map { stateIndex, array in
            eval(array)
            let data = array.asData(access: .copy).data
            return QwenMTPSampledTraceArrayFingerprint(
                stateIndex: stateIndex,
                shape: array.shape,
                dtype: String(describing: array.dtype),
                byteCount: data.count,
                sha256: SHA256.hash(data: data).map {
                    String(format: "%02x", $0)
                }.joined())
        }
        return QwenMTPSampledTraceCacheFingerprint(
            layerIndex: layerIndex,
            cacheType: String(describing: type(of: entry)),
            offset: entry.offset,
            metaStateSHA256: metaHasher.finalize().map {
                String(format: "%02x", $0)
            }.joined(),
            states: states)
    }
}

func writeValidatedFreshSampledTrace(_ data: Data, to url: URL) throws {
    _ = try QwenMTPSampledTraceGate.validateJSONL(data)
    let descriptor = url.withUnsafeFileSystemRepresentation { path -> Int32 in
        guard let path else { return -1 }
        return open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
    }
    guard descriptor >= 0 else {
        if errno == EEXIST { throw QwenMTPSampledTraceCLIError.evidenceMustBeNew }
        throw QwenMTPSampledTraceCLIError.evidenceWriteFailed
    }
    defer { _ = close(descriptor) }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
    do {
        try handle.write(contentsOf: data)
        try handle.synchronize()
    } catch {
        throw QwenMTPSampledTraceCLIError.evidenceWriteFailed
    }
}
