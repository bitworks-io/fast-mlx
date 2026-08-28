import CryptoKit
import Foundation
import HarnessCore
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import SpikeCore
import Tokenizers

struct QwenMTPSampledBlockTraceArguments: Equatable, Sendable {
    let targetPath: String
    let drafterPath: String
    let evidencePath: String
}

enum QwenMTPSampledBlockTraceCLIError: Error, Equatable, CustomStringConvertible, Sendable {
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
        case .releaseBuildRequired: return "sampled MTP block trace requires a Release build"
        case .evidenceMustBeNew: return "--evidence must name a new file"
        case .evidenceWriteFailed: return "failed to write sampled MTP block trace evidence"
        case .unexpectedArtifactRequest: return "unexpected exact artifact request"
        case .evidenceReadFailed: return "failed to read sampled MTP block trace evidence"
        }
    }
}

func validateQwenMTPSampledBlockTrace(arguments: [String]) throws {
    guard arguments.count == 2, arguments[0] == "--evidence",
        !arguments[1].hasPrefix("--")
    else {
        throw QwenMTPSampledBlockTraceCLIError.missingFlag("--evidence")
    }
    let data: Data
    do {
        data = try Data(contentsOf: URL(fileURLWithPath: arguments[1]))
    } catch {
        throw QwenMTPSampledBlockTraceCLIError.evidenceReadFailed
    }
    _ = try QwenMTPSampledBlockTraceGate.validateJSONL(data)
    print("validate-qwen-mtp-sampled-block-trace PASS")
}

func parseQwenMTPSampledBlockTraceArguments(
    _ arguments: [String]
) throws -> QwenMTPSampledBlockTraceArguments {
    let allowed = Set(["--target", "--drafter", "--evidence"])
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw QwenMTPSampledBlockTraceCLIError.unexpectedPositional
        }
        guard allowed.contains(flag) else {
            throw QwenMTPSampledBlockTraceCLIError.unknownFlag
        }
        guard values[flag] == nil else {
            throw QwenMTPSampledBlockTraceCLIError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count,
            !arguments[index + 1].hasPrefix("--")
        else {
            throw QwenMTPSampledBlockTraceCLIError.missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }
    for flag in ["--target", "--drafter", "--evidence"] where values[flag] == nil {
        throw QwenMTPSampledBlockTraceCLIError.missingFlag(flag)
    }
    return QwenMTPSampledBlockTraceArguments(
        targetPath: values["--target"]!,
        drafterPath: values["--drafter"]!,
        evidencePath: values["--evidence"]!)
}

func qwenMTPSampledBlockTraceExternalDiagnostic(_ error: Error) -> String {
    if let error = error as? QwenMTPSampledBlockTraceCLIError {
        return error.description
    }
    if let error = error as? SampledMTPBlockDiagnosticError {
        return "sampled MTP block diagnostic failed: \(error)"
    }
    if let error = error as? QwenMTPSampledBlockTraceGateError {
        return "sampled MTP block trace gate failed: \(sampledBlockTraceGateInvariant(error))"
    }
    if let error = error as? SampledMTPBlockAcceptanceError {
        return "sampled MTP block acceptance failed: \(error)"
    }
    if let error = error as? SampledMTPResidualCorrectionError {
        return "sampled MTP residual correction failed: \(error)"
    }
    return "sampled MTP block trace failed"
}

private func sampledBlockTraceGateInvariant(
    _ error: QwenMTPSampledBlockTraceGateError
) -> String {
    switch error {
    case .schemaMismatch: "schemaMismatch"
    case .sourceMismatch: "sourceMismatch"
    case .samplingMismatch: "samplingMismatch"
    case .outcomeSetMismatch: "outcomeSetMismatch"
    case .caseContextMismatch: "caseContextMismatch"
    case .promptMismatch(let outcome): "promptMismatch(\(outcome.rawValue))"
    case .invalidPromptTokenCount(let outcome):
        "invalidPromptTokenCount(\(outcome.rawValue))"
    case .invalidInitialBonus(let outcome): "invalidInitialBonus(\(outcome.rawValue))"
    case .stepSetMismatch(let outcome): "stepSetMismatch(\(outcome.rawValue))"
    case .vocabularyMismatch(let outcome, let stepIndex):
        "vocabularyMismatch(\(outcome.rawValue), step: \(stepIndex))"
    case .proposalDrawMismatch(let outcome, let stepIndex):
        "proposalDrawMismatch(\(outcome.rawValue), step: \(stepIndex))"
    case .invalidBonusDistribution(let outcome):
        "invalidBonusDistribution(\(outcome.rawValue))"
    case .decisionMismatch(let outcome): "decisionMismatch(\(outcome.rawValue))"
    case .cacheLayerCountMismatch(let outcome):
        "cacheLayerCountMismatch(\(outcome.rawValue))"
    case .malformedCache(let outcome, let layer):
        "malformedCache(\(outcome.rawValue), layer: \(layer))"
    case .cacheMismatch(let outcome, let layer):
        "cacheMismatch(\(outcome.rawValue), layer: \(layer))"
    case .malformedJSONL: "malformedJSONL"
    case .wrongSubcommand: "wrongSubcommand"
    case .invalidProvenance: "invalidProvenance"
    case .nonCanonicalJSONL: "nonCanonicalJSONL"
    }
}

func runQwenMTPSampledBlockTrace(arguments: [String]) async throws {
    #if DEBUG
    throw QwenMTPSampledBlockTraceCLIError.releaseBuildRequired
    #else
    let arguments = try parseQwenMTPSampledBlockTraceArguments(arguments)
    let evidenceURL = URL(fileURLWithPath: arguments.evidencePath)
    guard !FileManager.default.fileExists(atPath: evidenceURL.path) else {
        throw QwenMTPSampledBlockTraceCLIError.evidenceMustBeNew
    }

    let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
        from: QwenMTPSampledBlockTraceDownloader(
            target: URL(fileURLWithPath: arguments.targetPath, isDirectory: true),
            drafter: URL(fileURLWithPath: arguments.drafterPath, isDirectory: true)),
        using: #huggingFaceTokenizerLoader())
    let promptTokens = pair.target.tokenizer.encode(text: QwenMTPSampledBlockTraceGate.requiredPrompt)
    let input = LMInput(tokens: MLXArray(promptTokens))
    let outcomes: [SampledMTPBlockDiagnosticOutcome] = [
        .rejectFirst, .rejectSecond, .acceptAll,
    ]
    let cases = try outcomes.map { outcome in
        try runSampledMTPBlockDiagnostic(
            input: input,
            target: pair.target.model,
            drafter: pair.drafter.model,
            outcome: outcome,
            decide: makeSampledBlockDecision).sampledBlockTraceCase
    }

    let binding = pair.binding
    let payload = QwenMTPSampledBlockTraceEvidence(
        schemaVersion: QwenMTPSampledBlockTraceGate.schemaVersion,
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
        sampling: QwenMTPSampledBlockTraceGate.requiredSampling,
        cases: cases)
    _ = try QwenMTPSampledBlockTraceGate.validate(payload)

    let modelConfig = ProvenanceCLI.modelConfig(at: arguments.targetPath)
    let record = ResultRecord(
        subcommand: QwenMTPSampledBlockTraceGate.subcommand,
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
            corpusId: QwenMTPSampledBlockTraceGate.corpusID,
            corpusContentHash: QwenMTPSampledBlockTraceGate.requiredPromptSHA256,
            nonce: UUID().uuidString),
        payload: payload)
    try writeValidatedFreshSampledBlockTrace(
        Data((try record.jsonLine() + "\n").utf8), to: evidenceURL)
    print("qwen-mtp-sampled-block-trace PASS cases=3 cacheLayers=32")
    #endif
}

private struct QwenMTPSampledBlockTraceDownloader: Downloader {
    let target: URL
    let drafter: URL

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let source = QwenMTPSampledBlockTraceGate.requiredSource
        guard !useLatest,
            patterns == ["*.safetensors", "*.json", "*.jinja"]
        else {
            throw QwenMTPSampledBlockTraceCLIError.unexpectedArtifactRequest
        }
        progressHandler(Progress(totalUnitCount: 1))
        switch (id, revision) {
        case (source.targetModelID, source.targetRevision): return target
        case (source.drafterModelID, source.drafterRevision): return drafter
        default: throw QwenMTPSampledBlockTraceCLIError.unexpectedArtifactRequest
        }
    }
}

private func makeSampledBlockDecision(
    _ steps: [SampledMTPBlockDiagnosticStep],
    _ acceptanceUniforms: [Double],
    _ terminalDraw: SampledMTPBlockDiagnosticTerminalDraw,
    _ bonusTargetProbabilities: [Double]
) throws -> SampledMTPBlockDiagnosticDecision {
    let decision = try SampledMTPBlockAcceptance.decide(
        steps: steps.map {
            .init(
                targetDistribution: $0.targetProbabilities,
                draftDistribution: $0.draftProbabilities,
                proposedToken: $0.proposedToken)
        },
        acceptanceUniforms: acceptanceUniforms,
        terminalDraws: [terminalDraw.harnessDraw],
        bonusTargetDistribution: bonusTargetProbabilities)
    return .init(
        outputTokens: decision.tokens,
        acceptedDraftCount: decision.acceptedDraftCount,
        acceptedDraftEndIndex: decision.acceptedDraftEndIndex)
}

private extension SampledMTPBlockDiagnosticTerminalDraw {
    var harnessDraw: SampledMTPBlockTerminalDraw {
        switch self {
        case .residual(let value): .residual(value)
        case .bonus(let value): .bonus(value)
        }
    }

    var evidenceDraw: QwenMTPSampledBlockTraceTerminalDraw {
        switch self {
        case .residual(let value): .residual(value)
        case .bonus(let value): .bonus(value)
        }
    }
}

private extension SampledMTPBlockDiagnosticOutcome {
    var evidenceOutcome: QwenMTPSampledBlockTraceOutcome {
        switch self {
        case .rejectFirst: .rejectFirst
        case .rejectSecond: .rejectSecond
        case .acceptAll: .acceptAll
        }
    }
}

private extension SampledMTPBlockDiagnosticResult {
    var sampledBlockTraceCase: QwenMTPSampledBlockTraceCaseEvidence {
        .init(
            outcome: outcome.evidenceOutcome,
            promptSHA256: QwenMTPSampledBlockTraceGate.requiredPromptSHA256,
            promptTokenCount: promptTokenCount,
            initialBonusToken: initialBonusToken,
            outputTokens: decision.outputTokens,
            acceptedDraftCount: decision.acceptedDraftCount,
            acceptedDraftEndIndex: decision.acceptedDraftEndIndex,
            acceptanceUniforms: acceptanceUniforms,
            terminalDraw: terminalDraw.evidenceDraw,
            steps: steps.map {
                .init(
                    stepIndex: $0.stepIndex,
                    proposedToken: $0.proposedToken,
                    proposalUniform: $0.proposalUniform,
                    targetProbabilities: $0.targetProbabilities,
                    draftProbabilities: $0.draftProbabilities)
            },
            bonusTargetProbabilities: bonusTargetProbabilities,
            candidateCache: sampledBlockTraceCacheFingerprint(candidateCache),
            scalarCache: sampledBlockTraceCacheFingerprint(scalarCache))
    }
}

private func sampledBlockTraceCacheFingerprint(
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
                sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        }
        return QwenMTPSampledTraceCacheFingerprint(
            layerIndex: layerIndex,
            cacheType: String(describing: type(of: entry)),
            offset: entry.offset,
            metaStateSHA256: metaHasher.finalize().map { String(format: "%02x", $0) }.joined(),
            states: states)
    }
}

func writeValidatedFreshSampledBlockTrace(_ data: Data, to url: URL) throws {
    _ = try QwenMTPSampledBlockTraceGate.validateJSONL(data)
    let descriptor = url.withUnsafeFileSystemRepresentation { path -> Int32 in
        guard let path else { return -1 }
        return open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
    }
    guard descriptor >= 0 else {
        if errno == EEXIST { throw QwenMTPSampledBlockTraceCLIError.evidenceMustBeNew }
        throw QwenMTPSampledBlockTraceCLIError.evidenceWriteFailed
    }
    defer { _ = close(descriptor) }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
    do {
        try handle.write(contentsOf: data)
        try handle.synchronize()
    } catch {
        throw QwenMTPSampledBlockTraceCLIError.evidenceWriteFailed
    }
}
