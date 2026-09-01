import CryptoKit
import Foundation
import HarnessCore
import HuggingFace
import MLX
import MLXHuggingFace
@_spi(FastMLXExactMTP) import MLXLLM
import MLXLMCommon
import SpikeCore
import Tokenizers

enum QwenMTPCorpusCLIError: Error, CustomStringConvertible {
    case usage
    case evaluationOrderUsage
    case hiddenFirstRuntimeUsage
    case combinedEvaluationUsage
    case longProfileUsage
    case evidenceMustBeNewOrEmpty(String)
    case unexpectedDownloadRequest(id: String, revision: String?)
    case releaseBuildRequired
    case gdnEnvironmentNotPinned
    case longProfileMeasurementFailure(caseID: String, pairIndex: Int, reason: String)
    case hiddenFirstRuntimeNotQualified
    case cannotCreateEvidence(String)
    case missingEvaluationOrderTelemetry(String)

    var description: String {
        switch self {
        case .usage:
            return "usage: fastmlx-harness qwen-mtp-corpus --target <DIR> --drafter <DIR> --evidence <NEW-OR-EMPTY JSONL> [--profile true|false]"
        case .longProfileUsage:
            return "usage: fastmlx-harness qwen-mtp-long-profile --target <DIR> --drafter <DIR> "
                + "--evidence <NEW-OR-EMPTY JSONL> [--binding qwen35-9b-depth1|"
                + "qwen38-27b-mxfp8-depth1|qwen38-27b-4bit-depth1]"
        case .gdnEnvironmentNotPinned:
            return "qwen-mtp-long-profile requires MLX_QWEN_FOUR_GDN=1 in the environment: the "
                + "pre-check contract pins GDN-on for every arm"
        case .longProfileMeasurementFailure(let caseID, let pairIndex, let reason):
            return "qwen-mtp-long-profile \(caseID) pair \(pairIndex) failed fail-closed "
                + "measurement checks: \(reason)"
        case .evaluationOrderUsage:
            return "usage: fastmlx-harness qwen-mtp-eval-order --target <DIR> --drafter <DIR> --evidence <NEW-OR-EMPTY JSONL>"
        case .hiddenFirstRuntimeUsage:
            return "usage: fastmlx-harness qwen-mtp-hidden-first-runtime --target <DIR> --drafter <DIR> --evidence <NEW-OR-EMPTY JSONL>"
        case .combinedEvaluationUsage:
            return "usage: fastmlx-harness qwen-mtp-eval-combined --target <DIR> --drafter <DIR> --evidence <NEW-OR-EMPTY JSONL>"
        case .evidenceMustBeNewOrEmpty(let path):
            return "--evidence must name a new or empty JSONL file: \(path)"
        case .unexpectedDownloadRequest(let id, let revision):
            return "unexpected exact Qwen MTP artifact request: \(id)@\(revision ?? "nil")"
        case .releaseBuildRequired:
            return "Qwen MTP measurement requires a Release build"
        case .hiddenFirstRuntimeNotQualified:
            return "hidden-first runtime missed the frozen promotion thresholds"
        case .cannotCreateEvidence(let path):
            return "cannot create evidence parent directory for \(path)"
        case .missingEvaluationOrderTelemetry(let order):
            return "missing Qwen MTP prompt-evaluation telemetry for \(order)"
        }
    }
}

func runQwenMTPCorpus(_ flags: Flags) async throws {
    let targetPath = try flags.strictString("target", default: "")
    let drafterPath = try flags.strictString("drafter", default: "")
    let evidencePath = try flags.strictString("evidence", default: "")
    guard !targetPath.isEmpty, !drafterPath.isEmpty, !evidencePath.isEmpty else {
        throw QwenMTPCorpusCLIError.usage
    }
    let profileEnabled = try flags.strictBool("profile", default: true)
    let evidenceURL = URL(fileURLWithPath: evidencePath)
    try requireNewOrEmptyEvidence(evidenceURL)

    let targetURL = URL(fileURLWithPath: targetPath, isDirectory: true)
    let drafterURL = URL(fileURLWithPath: drafterPath, isDirectory: true)
    let downloader = QwenMTPLocalExactRevisionDownloader(target: targetURL, drafter: drafterURL)
    let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
        from: downloader,
        using: #huggingFaceTokenizerLoader())

    var caseResults: [QwenMTPCorpusCaseResult] = []
    caseResults.reserveCapacity(QwenMTPCorpusGate.cases.count)
    for spec in QwenMTPCorpusGate.cases {
        switch spec.kind {
        case .cancellationAcceptedDraft:
            caseResults.append(try runAcceptedDraftCancellationCase(spec, pair: pair))
        default:
            caseResults.append(try runStandardCase(spec, pair: pair))
        }
    }

    var payload = QwenMTPCorpusEvidencePayload(
        schemaVersion: QwenMTPCorpusGate.schemaVersion,
        corpusID: QwenMTPCorpusGate.corpusID,
        corpusContentHash: QwenMTPCorpusGate.corpusContentHash,
        binding: runtimeBinding(pair.binding),
        host: .init(
            chip: ProvenanceCLI.chipBrand(),
            ramBytes: ProvenanceCLI.ramBytes(),
            os: ProvenanceCLI.osVersion()),
        caseResults: caseResults,
        correctness: .pass,
        profile: nil)

    do {
        _ = try QwenMTPCorpusGate.validate(payload)
    } catch {
        let failedPayload = QwenMTPCorpusGate.canonicalCorrectnessFailurePayload(from: payload)
        try RequiredJSONLWriter.append(resultRecord(failedPayload, targetPath: targetPath), to: evidenceURL)
        throw error
    }

    guard profileEnabled else {
        try RequiredJSONLWriter.append(resultRecord(payload, targetPath: targetPath), to: evidenceURL)
        return
    }

    guard qwenMTPCorpusReleaseBuildObserved else {
        try RequiredJSONLWriter.append(resultRecord(payload, targetPath: targetPath), to: evidenceURL)
        throw QwenMTPCorpusCLIError.releaseBuildRequired
    }

    payload.profile = try runQwenMTPProfile(pair: pair)
    let reuseVerdict: QwenMTPPromptHiddenReuseVerdict
    let verificationVerdict: QwenMTPGreedyBatchedVerificationVerdict
    do {
        _ = try QwenMTPCorpusGate.validate(payload)
        reuseVerdict = try QwenMTPCorpusGate.validatePromptHiddenReuse(payload)
        verificationVerdict = try QwenMTPCorpusGate.validateGreedyBatchedVerification(payload)
    } catch {
        let rejectedURL = evidenceURL.appendingPathExtension("rejected")
        try requireNewOrEmptyEvidence(rejectedURL)
        try RequiredJSONLWriter.append(
            resultRecord(
                payload,
                targetPath: targetPath,
                subcommand: "qwen-mtp-corpus-rejected"),
            to: rejectedURL)
        throw error
    }
    try RequiredJSONLWriter.append(resultRecord(payload, targetPath: targetPath), to: evidenceURL)
    print(String(
        format: "qwen-mtp-corpus prompt-hidden-reuse PASS measured=%.4fs reduction=%.4fs required=%.1fs",
        reuseVerdict.measuredDrafterPromptPrimingSeconds,
        reuseVerdict.reductionSeconds,
        reuseVerdict.requiredReductionSeconds))
    print(String(
        format: "qwen-mtp-corpus greedy-batched-verification PASS measured=%.4fs reduction=%.4fs required=%.1fs",
        verificationVerdict.measuredTargetVerificationSeconds,
        verificationVerdict.reductionSeconds,
        verificationVerdict.requiredReductionSeconds))
    if let profileVerdict = try QwenMTPCorpusGate.validate(payload).profile {
        print(String(
            format: "qwen-mtp-corpus prompt-hidden-materialization diagnostic measured=%.4fs prompt_overhead=%.4fs share=%.4f threshold=%.1fs candidate=%@",
            profileVerdict.hiddenMaterializationSecondsTotal,
            profileVerdict.promptOverheadSecondsTotal,
            profileVerdict.hiddenMaterializationShareOfPromptOverhead,
            profileVerdict.hiddenMaterializationCandidateThresholdSeconds,
            profileVerdict.hiddenMaterializationCandidateQualified ? "yes" : "no"))
    }
}

func runQwenMTPEvaluationOrderIsolation(_ flags: Flags) async throws {
    let targetPath = try flags.strictString("target", default: "")
    let drafterPath = try flags.strictString("drafter", default: "")
    let evidencePath = try flags.strictString("evidence", default: "")
    guard !targetPath.isEmpty, !drafterPath.isEmpty, !evidencePath.isEmpty else {
        throw QwenMTPCorpusCLIError.evaluationOrderUsage
    }
    guard qwenMTPCorpusReleaseBuildObserved else {
        throw QwenMTPCorpusCLIError.releaseBuildRequired
    }

    let evidenceURL = URL(fileURLWithPath: evidencePath)
    try requireNewOrEmptyEvidence(evidenceURL)
    let downloader = QwenMTPLocalExactRevisionDownloader(
        target: URL(fileURLWithPath: targetPath, isDirectory: true),
        drafter: URL(fileURLWithPath: drafterPath, isDirectory: true))
    let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
        from: downloader,
        using: #huggingFaceTokenizerLoader())
    guard let spec = QwenMTPCorpusGate.cases.first(where: {
        $0.id == "long-retrieval"
    }) else {
        throw QwenMTPCorpusCLIError.evaluationOrderUsage
    }

    let parameters = GenerateParameters(maxTokens: spec.maxTokens, temperature: 0)
    var pairs = [QwenMTPEvaluationOrderPairEvidence]()
    pairs.reserveCapacity(QwenMTPEvaluationOrderIsolationGate.pairOrders.count)
    for (pairIndex, runOrder) in
        QwenMTPEvaluationOrderIsolationGate.pairOrders.enumerated()
    {
        let scalar = try runScalar(spec, pair: pair, parameters: parameters)
        let cacheFirst: CorpusRun
        let hiddenFirst: CorpusRun
        switch runOrder {
        case .cacheFirstThenHiddenFirst:
            cacheFirst = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters,
                promptPreparationEvaluationOrder: .cacheFirst)
            hiddenFirst = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters,
                promptPreparationEvaluationOrder: .hiddenFirst)
        case .hiddenFirstThenCacheFirst:
            hiddenFirst = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters,
                promptPreparationEvaluationOrder: .hiddenFirst)
            cacheFirst = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters,
                promptPreparationEvaluationOrder: .cacheFirst)
        }
        let pairEvidence = QwenMTPEvaluationOrderPairEvidence(
            pairIndex: pairIndex,
            warmup: pairIndex
                < QwenMTPEvaluationOrderIsolationGate.droppedWarmupPairs,
            runOrder: runOrder,
            cacheFirst: try evaluationOrderRunEvidence(
                .cacheFirst,
                scalar: scalar,
                mtp: cacheFirst),
            hiddenFirst: try evaluationOrderRunEvidence(
                .hiddenFirst,
                scalar: scalar,
                mtp: hiddenFirst))
        try QwenMTPEvaluationOrderIsolationGate.validatePair(
            pairEvidence,
            at: pairIndex)
        pairs.append(pairEvidence)
    }

    let payload = QwenMTPEvaluationOrderIsolationPayload(
        schemaVersion: QwenMTPEvaluationOrderIsolationGate.schemaVersion,
        corpusID: QwenMTPEvaluationOrderIsolationGate.corpusID,
        corpusContentHash: QwenMTPEvaluationOrderIsolationGate.corpusContentHash,
        binding: runtimeBinding(pair.binding),
        host: .init(
            chip: ProvenanceCLI.chipBrand(),
            ramBytes: ProvenanceCLI.ramBytes(),
            os: ProvenanceCLI.osVersion()),
        releaseBuildRequired: true,
        releaseBuildObserved: true,
        pairs: pairs)
    let verdict = try QwenMTPEvaluationOrderIsolationGate.validate(payload)
    try RequiredJSONLWriter.append(
        evaluationOrderResultRecord(payload, targetPath: targetPath),
        to: evidenceURL)
    print(String(
        format: "qwen-mtp-eval-order %@ aggregate=%.4fs/%.1fs median=%.4fs/%.2fs",
        verdict.qualified ? "PROMOTE" : "SHELVE",
        verdict.aggregatePromptImprovementSeconds,
        verdict.requiredAggregatePromptImprovementSeconds,
        verdict.medianPromptImprovementSeconds,
        verdict.requiredMedianPromptImprovementSeconds))
}

func runQwenMTPHiddenFirstRuntimeEquivalence(_ flags: Flags) async throws {
    let targetPath = try flags.strictString("target", default: "")
    let drafterPath = try flags.strictString("drafter", default: "")
    let evidencePath = try flags.strictString("evidence", default: "")
    guard !targetPath.isEmpty, !drafterPath.isEmpty, !evidencePath.isEmpty else {
        throw QwenMTPCorpusCLIError.hiddenFirstRuntimeUsage
    }
    guard qwenMTPCorpusReleaseBuildObserved else {
        throw QwenMTPCorpusCLIError.releaseBuildRequired
    }

    let evidenceURL = URL(fileURLWithPath: evidencePath)
    try requireNewOrEmptyEvidence(evidenceURL)
    let downloader = QwenMTPLocalExactRevisionDownloader(
        target: URL(fileURLWithPath: targetPath, isDirectory: true),
        drafter: URL(fileURLWithPath: drafterPath, isDirectory: true))
    let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
        from: downloader,
        using: #huggingFaceTokenizerLoader())
    guard let spec = QwenMTPCorpusGate.cases.first(where: {
        $0.id == "long-retrieval"
    }) else {
        throw QwenMTPCorpusCLIError.hiddenFirstRuntimeUsage
    }

    let parameters = GenerateParameters(maxTokens: spec.maxTokens, temperature: 0)
    var pairs = [QwenMTPHiddenFirstRuntimePairEvidence]()
    pairs.reserveCapacity(QwenMTPHiddenFirstRuntimeEquivalenceGate.pairOrders.count)
    for (pairIndex, runOrder) in
        QwenMTPHiddenFirstRuntimeEquivalenceGate.pairOrders.enumerated()
    {
        let scalar = try runScalar(spec, pair: pair, parameters: parameters)
        let defaultRuntime: CorpusRun
        let hiddenFirstRuntime: CorpusRun
        switch runOrder {
        case .defaultThenHiddenFirst:
            // Intentionally omit the evaluation-order argument: this run is
            // the ordinary iterator default, not an explicitly relabeled
            // cache-first diagnostic.
            defaultRuntime = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters)
            hiddenFirstRuntime = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters,
                promptPreparationEvaluationOrder: .hiddenFirst)
        case .hiddenFirstThenDefault:
            hiddenFirstRuntime = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters,
                promptPreparationEvaluationOrder: .hiddenFirst)
            defaultRuntime = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters)
        }
        let pairEvidence = QwenMTPHiddenFirstRuntimePairEvidence(
            pairIndex: pairIndex,
            warmup: pairIndex
                < QwenMTPHiddenFirstRuntimeEquivalenceGate.droppedWarmupPairs,
            runOrder: runOrder,
            defaultRuntime: try evaluationOrderRunEvidence(
                .cacheFirst,
                scalar: scalar,
                mtp: defaultRuntime),
            hiddenFirstRuntime: try evaluationOrderRunEvidence(
                .hiddenFirst,
                scalar: scalar,
                mtp: hiddenFirstRuntime))
        try QwenMTPHiddenFirstRuntimeEquivalenceGate.validatePair(
            pairEvidence,
            at: pairIndex)
        pairs.append(pairEvidence)
    }

    let payload = QwenMTPHiddenFirstRuntimeEquivalencePayload(
        schemaVersion: QwenMTPHiddenFirstRuntimeEquivalenceGate.schemaVersion,
        corpusID: QwenMTPHiddenFirstRuntimeEquivalenceGate.corpusID,
        corpusContentHash: QwenMTPHiddenFirstRuntimeEquivalenceGate.corpusContentHash,
        binding: runtimeBinding(pair.binding),
        host: .init(
            chip: ProvenanceCLI.chipBrand(),
            ramBytes: ProvenanceCLI.ramBytes(),
            os: ProvenanceCLI.osVersion()),
        releaseBuildRequired: true,
        releaseBuildObserved: true,
        pairs: pairs)
    let verdict = try QwenMTPHiddenFirstRuntimeEquivalenceGate.validate(payload)
    let subcommand = verdict.qualified
        ? QwenMTPHiddenFirstRuntimeEquivalenceGate.subcommand
        : QwenMTPHiddenFirstRuntimeEquivalenceGate.rejectedSubcommand
    let record = hiddenFirstRuntimeResultRecord(
        payload,
        targetPath: targetPath,
        subcommand: subcommand)
    let recordData = Data((try record.jsonLine() + "\n").utf8)
    if verdict.qualified {
        _ = try QwenMTPHiddenFirstRuntimeEquivalenceGate.validateJSONL(recordData)
    } else {
        _ = try QwenMTPHiddenFirstRuntimeEquivalenceGate
            .validateRejectedJSONL(recordData)
    }
    try RequiredJSONLWriter.append(
        record,
        to: evidenceURL)
    print(String(
        format: "qwen-mtp-hidden-first-runtime %@ aggregate=%.4fs/%.1fs median=%.4fs/%.2fs",
        verdict.qualified ? "PROMOTE" : "SHELVE",
        verdict.aggregatePromptImprovementSeconds,
        verdict.requiredAggregatePromptImprovementSeconds,
        verdict.medianPromptImprovementSeconds,
        verdict.requiredMedianPromptImprovementSeconds))
    guard verdict.qualified else {
        throw QwenMTPCorpusCLIError.hiddenFirstRuntimeNotQualified
    }
}

func runQwenMTPCombinedEvaluationIsolation(_ flags: Flags) async throws {
    let targetPath = try flags.strictString("target", default: "")
    let drafterPath = try flags.strictString("drafter", default: "")
    let evidencePath = try flags.strictString("evidence", default: "")
    guard !targetPath.isEmpty, !drafterPath.isEmpty, !evidencePath.isEmpty else {
        throw QwenMTPCorpusCLIError.combinedEvaluationUsage
    }
    guard qwenMTPCorpusReleaseBuildObserved else {
        throw QwenMTPCorpusCLIError.releaseBuildRequired
    }

    let evidenceURL = URL(fileURLWithPath: evidencePath)
    try requireNewOrEmptyEvidence(evidenceURL)
    let downloader = QwenMTPLocalExactRevisionDownloader(
        target: URL(fileURLWithPath: targetPath, isDirectory: true),
        drafter: URL(fileURLWithPath: drafterPath, isDirectory: true))
    let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
        from: downloader,
        using: #huggingFaceTokenizerLoader())
    guard let spec = QwenMTPCorpusGate.cases.first(where: {
        $0.id == "long-retrieval"
    }) else {
        throw QwenMTPCorpusCLIError.combinedEvaluationUsage
    }

    let parameters = GenerateParameters(maxTokens: spec.maxTokens, temperature: 0)
    var pairs = [QwenMTPCombinedEvaluationPairEvidence]()
    pairs.reserveCapacity(QwenMTPCombinedEvaluationIsolationGate.pairOrders.count)
    for (pairIndex, runOrder) in
        QwenMTPCombinedEvaluationIsolationGate.pairOrders.enumerated()
    {
        let scalar = try runScalar(spec, pair: pair, parameters: parameters)
        let cacheFirst: CorpusRun
        let combined: CorpusRun
        switch runOrder {
        case .cacheFirstThenCombined:
            cacheFirst = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters,
                promptPreparationEvaluationOrder: .cacheFirst)
            combined = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters,
                promptPreparationEvaluationOrder: .combined)
        case .combinedThenCacheFirst:
            combined = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters,
                promptPreparationEvaluationOrder: .combined)
            cacheFirst = try runMTPDrain(
                spec,
                pair: pair,
                parameters: parameters,
                promptPreparationEvaluationOrder: .cacheFirst)
        }
        let pairEvidence = QwenMTPCombinedEvaluationPairEvidence(
            pairIndex: pairIndex,
            warmup: pairIndex
                < QwenMTPCombinedEvaluationIsolationGate.droppedWarmupPairs,
            runOrder: runOrder,
            cacheFirst: try evaluationOrderRunEvidence(
                .cacheFirst,
                scalar: scalar,
                mtp: cacheFirst),
            combined: try evaluationOrderRunEvidence(
                .combined,
                scalar: scalar,
                mtp: combined))
        try QwenMTPCombinedEvaluationIsolationGate.validatePair(
            pairEvidence,
            at: pairIndex)
        pairs.append(pairEvidence)
    }

    let payload = QwenMTPCombinedEvaluationIsolationPayload(
        schemaVersion: QwenMTPCombinedEvaluationIsolationGate.schemaVersion,
        corpusID: QwenMTPCombinedEvaluationIsolationGate.corpusID,
        corpusContentHash: QwenMTPCombinedEvaluationIsolationGate.corpusContentHash,
        binding: runtimeBinding(pair.binding),
        host: .init(
            chip: ProvenanceCLI.chipBrand(),
            ramBytes: ProvenanceCLI.ramBytes(),
            os: ProvenanceCLI.osVersion()),
        releaseBuildRequired: true,
        releaseBuildObserved: true,
        pairs: pairs)
    let verdict = try QwenMTPCombinedEvaluationIsolationGate.validate(payload)
    try RequiredJSONLWriter.append(
        combinedEvaluationResultRecord(payload, targetPath: targetPath),
        to: evidenceURL)
    print(String(
        format: "qwen-mtp-eval-combined %@ aggregate=%.4fs/%.1fs median=%.4fs/%.2fs",
        verdict.qualified ? "PROMOTE" : "SHELVE",
        verdict.aggregatePromptImprovementSeconds,
        verdict.requiredAggregatePromptImprovementSeconds,
        verdict.medianPromptImprovementSeconds,
        verdict.requiredMedianPromptImprovementSeconds))
}

private var qwenMTPCorpusReleaseBuildObserved: Bool {
    #if DEBUG
    false
    #else
    true
    #endif
}

private func requireNewOrEmptyEvidence(_ url: URL) throws {
    let manager = FileManager.default
    let parent = url.deletingLastPathComponent()
    if !manager.fileExists(atPath: parent.path) {
        throw QwenMTPCorpusCLIError.cannotCreateEvidence(url.path)
    }
    guard manager.fileExists(atPath: url.path) else { return }
    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard values.isRegularFile == true, (values.fileSize ?? 0) == 0 else {
        throw QwenMTPCorpusCLIError.evidenceMustBeNewOrEmpty(url.path)
    }
}

private struct QwenMTPLocalExactRevisionDownloader: Downloader {
    let target: URL
    let drafter: URL

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard !useLatest, patterns == ["*.safetensors", "*.json", "*.jinja"] else {
            throw QwenMTPCorpusCLIError.unexpectedDownloadRequest(id: id, revision: revision)
        }
        progressHandler(Progress(totalUnitCount: 1))
        switch (id, revision) {
        case (
            "mlx-community/Qwen3.5-9B-MLX-4bit",
            "938d8919941c6e7efd3c7150eff7fe9d12afa631"
        ):
            return target
        case (
            "mlx-community/Qwen3.5-9B-MTP-5bit",
            "994730d199bff7799aa3ddef33a96723967a3e33"
        ):
            return drafter
        default:
            throw QwenMTPCorpusCLIError.unexpectedDownloadRequest(id: id, revision: revision)
        }
    }
}

private func runStandardCase(
    _ spec: QwenMTPCorpusCaseSpec,
    pair: Qwen35ExactMTPLoadedPair
) throws -> QwenMTPCorpusCaseResult {
    let parameters = parameters(for: spec)
    let scalar = try runScalar(
        spec,
        pair: pair,
        parameters: parameters,
        retainedTokenLimit: QwenMTPCorpusGate.scalarRetainedTokenLimit(forCaseID: spec.id))
    let mtp: CorpusRun
    switch spec.kind {
    case .cancellationRetainedToken:
        mtp = try runMTPCancelAfterExtraGenerated(
            spec,
            pair: pair,
            parameters: parameters,
            retainedTokens: 1)
    default:
        mtp = try runMTPDrain(spec, pair: pair, parameters: parameters)
    }
    return caseResult(spec: spec, scalar: scalar, mtp: mtp)
}

private func runAcceptedDraftCancellationCase(
    _ spec: QwenMTPCorpusCaseSpec,
    pair: Qwen35ExactMTPLoadedPair
) throws -> QwenMTPCorpusCaseResult {
    let parameters = parameters(for: spec)
    let mtp = try runMTPCancelAfterAcceptedDraft(spec, pair: pair, parameters: parameters)
    let scalar = try runScalar(
        spec,
        pair: pair,
        parameters: parameters,
        retainedTokenLimit: mtp.stopOutcome == .cancelled ? mtp.tokens.count : nil)
    return caseResult(spec: spec, scalar: scalar, mtp: mtp)
}

private struct CorpusRun {
    let promptTokenCount: Int
    let tokens: [Int]
    let tokenIDsSHA256: String
    let decodedBytesSHA256: String
    let stopOutcome: QwenMTPCorpusStopOutcome
    let cacheFingerprint: QwenMTPCorpusCacheFingerprint
    let timing: QwenMTPCorpusTiming
    let mtpTelemetry: QwenMTPCorpusMTPTelemetry
    let mtpPhaseAttribution: QwenMTPCorpusMTPPhaseAttribution?
    let promptPreparationEvaluationOrder: MTPPromptPreparationEvaluationOrder?
    let passthroughReason: String?
}

private func runScalar(
    _ spec: QwenMTPCorpusCaseSpec,
    pair: Qwen35ExactMTPLoadedPair,
    parameters: GenerateParameters,
    retainedTokenLimit: Int? = nil
) throws -> CorpusRun {
    let input = input(for: spec, tokenizer: pair.target.tokenizer)
    let promptTokenCount = promptTokenCount(for: spec, tokenizer: pair.target.tokenizer)
    let cache = pair.target.model.newCache(parameters: parameters)
    let wallStart = ProcessInfo.processInfo.systemUptime
    var iterator = try TokenIterator(
        input: input,
        model: pair.target.model,
        cache: cache,
        parameters: parameters)
    let generationStart = ProcessInfo.processInfo.systemUptime
    var tokens: [Int] = []
    let stopTokenIds = buildQwenMTPCorpusStopTokenIds(
        modelConfiguration: pair.target.configuration,
        tokenizer: pair.target.tokenizer)
    var stopOutcome: QwenMTPCorpusStopOutcome = .stop
    while let token = iterator.next() {
        if token == pair.target.tokenizer.unknownTokenId || stopTokenIds.contains(token) {
            iterator.discardGeneratedToken()
            stopOutcome = .stop
            break
        }
        tokens.append(token)
        if let retainedTokenLimit, tokens.count >= retainedTokenLimit {
            stopOutcome = .cancelled
            break
        }
        if tokens.count >= (parameters.maxTokens ?? spec.maxTokens) {
            stopOutcome = .length
        }
    }
    if stopOutcome == .stop, tokens.count >= (parameters.maxTokens ?? spec.maxTokens) {
        stopOutcome = .length
    }
    let end = ProcessInfo.processInfo.systemUptime
    let fingerprint = fingerprintCache(cache)
    return CorpusRun(
        promptTokenCount: promptTokenCount,
        tokens: tokens,
        tokenIDsSHA256: tokenIDsSHA256(tokens),
        decodedBytesSHA256: sha256Hex(Data(pair.target.tokenizer.decode(tokenIds: tokens).utf8)),
        stopOutcome: stopOutcome,
        cacheFingerprint: fingerprint,
        timing: .init(
            promptSeconds: max(iterator.promptPrefillTime, 1e-9),
            generationSeconds: max(end - generationStart, 1e-9),
            wallSeconds: max(end - wallStart, 1e-9),
            e2eSeconds: max(end - wallStart, 1e-9)),
        mtpTelemetry: .init(
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            rejectedDraftTokens: 0,
            roundCount: 0,
            targetModelCallCount: 0,
            draftModelCallCount: 0,
            targetVerifiedTokenCount: 0,
            emittedTokenCount: tokens.count),
        mtpPhaseAttribution: nil,
        promptPreparationEvaluationOrder: nil,
        passthroughReason: nil)
}

private func runMTPDrain(
    _ spec: QwenMTPCorpusCaseSpec,
    pair: Qwen35ExactMTPLoadedPair,
    parameters: GenerateParameters,
    promptPreparationEvaluationOrder: MTPPromptPreparationEvaluationOrder? = nil
) throws -> CorpusRun {
    let input = input(for: spec, tokenizer: pair.target.tokenizer)
    let promptTokenCount = promptTokenCount(for: spec, tokenizer: pair.target.tokenizer)
    let cache = pair.target.model.newCache(parameters: parameters)
    let wallStart = ProcessInfo.processInfo.systemUptime
    let configuredIterator: MTPSpeculativeTokenIterator
    if let promptPreparationEvaluationOrder {
        configuredIterator = try MTPSpeculativeTokenIterator(
            input: input,
            mainModel: pair.target.model,
            drafter: pair.drafter.model,
            mainCache: cache,
            parameters: parameters,
            blockSize: pair.binding.runtimeBlockSize,
            collectPhaseTelemetry: true,
            promptPreparationEvaluationOrder: promptPreparationEvaluationOrder)
    } else {
        configuredIterator = try MTPSpeculativeTokenIterator(
            input: input,
            mainModel: pair.target.model,
            drafter: pair.drafter.model,
            mainCache: cache,
            parameters: parameters,
            blockSize: pair.binding.runtimeBlockSize,
            collectPhaseTelemetry: true)
    }
    var iterator = configuredIterator
    let generationStart = ProcessInfo.processInfo.systemUptime
    var tokens: [Int] = []
    let stopTokenIds = buildQwenMTPCorpusStopTokenIds(
        modelConfiguration: pair.target.configuration,
        tokenizer: pair.target.tokenizer)
    var stopOutcome: QwenMTPCorpusStopOutcome = .stop
    while let token = iterator.next() {
        if token == pair.target.tokenizer.unknownTokenId || stopTokenIds.contains(token) {
            iterator.discardGeneratedToken()
            stopOutcome = .stop
            break
        }
        tokens.append(token)
        if tokens.count >= (parameters.maxTokens ?? spec.maxTokens) {
            stopOutcome = .length
        }
    }
    iterator.finalizeGeneration()
    if stopOutcome == .stop, tokens.count >= (parameters.maxTokens ?? spec.maxTokens) {
        stopOutcome = .length
    }
    let end = ProcessInfo.processInfo.systemUptime
    let fingerprintStart = ProcessInfo.processInfo.systemUptime
    let fingerprint = fingerprintCache(cache)
    let fingerprintSeconds = ProcessInfo.processInfo.systemUptime - fingerprintStart
    let telemetry = mtpTelemetry(from: iterator, emittedTokenCount: tokens.count)
    return CorpusRun(
        promptTokenCount: promptTokenCount,
        tokens: tokens,
        tokenIDsSHA256: tokenIDsSHA256(tokens),
        decodedBytesSHA256: sha256Hex(Data(pair.target.tokenizer.decode(tokenIds: tokens).utf8)),
        stopOutcome: stopOutcome,
        cacheFingerprint: fingerprint,
        timing: .init(
            promptSeconds: max(iterator.promptPrefillTime, 1e-9),
            generationSeconds: max(end - generationStart, 1e-9),
            wallSeconds: max(end - wallStart, 1e-9),
            e2eSeconds: max(end - wallStart, 1e-9)),
        mtpTelemetry: telemetry,
        mtpPhaseAttribution: mtpPhaseAttribution(
            from: iterator,
            cacheFingerprintSeconds: fingerprintSeconds),
        promptPreparationEvaluationOrder:
            iterator.promptPreparationTelemetry?.evaluationOrder,
        passthroughReason: iterator.passthroughReason)
}

private func evaluationOrderRunEvidence(
    _ expectedOrder: MTPPromptPreparationEvaluationOrder,
    scalar: CorpusRun,
    mtp: CorpusRun
) throws -> QwenMTPEvaluationOrderRunEvidence {
    guard mtp.promptPreparationEvaluationOrder == expectedOrder,
        let phaseAttribution = mtp.mtpPhaseAttribution
    else {
        throw QwenMTPCorpusCLIError.missingEvaluationOrderTelemetry(
            expectedOrder.rawValue)
    }
    let evidenceOrder: QwenMTPPromptEvaluationOrderEvidence
    switch expectedOrder {
    case .cacheFirst:
        evidenceOrder = .cacheFirst
    case .hiddenFirst:
        evidenceOrder = .hiddenFirst
    case .combined:
        evidenceOrder = .combined
    }
    return QwenMTPEvaluationOrderRunEvidence(
        evaluationOrder: evidenceOrder,
        timing: mtp.timing,
        exactness: exactnessEvidence(scalar: scalar, mtp: mtp),
        telemetry: mtp.mtpTelemetry,
        phaseAttribution: phaseAttribution,
        passthroughReason: mtp.passthroughReason)
}

private func runMTPCancelAfterExtraGenerated(
    _ spec: QwenMTPCorpusCaseSpec,
    pair: Qwen35ExactMTPLoadedPair,
    parameters: GenerateParameters,
    retainedTokens: Int
) throws -> CorpusRun {
    let input = input(for: spec, tokenizer: pair.target.tokenizer)
    let promptTokenCount = promptTokenCount(for: spec, tokenizer: pair.target.tokenizer)
    let cache = pair.target.model.newCache(parameters: parameters)
    let wallStart = ProcessInfo.processInfo.systemUptime
    var iterator = try MTPSpeculativeTokenIterator(
        input: input,
        mainModel: pair.target.model,
        drafter: pair.drafter.model,
        mainCache: cache,
        parameters: parameters,
        blockSize: pair.binding.runtimeBlockSize,
        collectPhaseTelemetry: true)
    let generationStart = ProcessInfo.processInfo.systemUptime
    var tokens: [Int] = []
    let stopTokenIds = buildQwenMTPCorpusStopTokenIds(
        modelConfiguration: pair.target.configuration,
        tokenizer: pair.target.tokenizer)
    var stopOutcome: QwenMTPCorpusStopOutcome?
    while tokens.count < retainedTokens {
        guard let token = iterator.next() else { break }
        if token == pair.target.tokenizer.unknownTokenId || stopTokenIds.contains(token) {
            iterator.discardGeneratedToken()
            stopOutcome = .stop
            break
        }
        tokens.append(token)
    }
    var discardedExtraGeneratedToken = false
    if stopOutcome == nil, tokens.count == retainedTokens, iterator.next() != nil {
        iterator.discardGeneratedToken()
        discardedExtraGeneratedToken = true
    }
    iterator.finalizeGeneration()
    let end = ProcessInfo.processInfo.systemUptime
    let fingerprintStart = ProcessInfo.processInfo.systemUptime
    let fingerprint = fingerprintCache(cache)
    let fingerprintSeconds = ProcessInfo.processInfo.systemUptime - fingerprintStart
    let telemetry = mtpTelemetry(from: iterator, emittedTokenCount: tokens.count)
    return CorpusRun(
        promptTokenCount: promptTokenCount,
        tokens: tokens,
        tokenIDsSHA256: tokenIDsSHA256(tokens),
        decodedBytesSHA256: sha256Hex(Data(pair.target.tokenizer.decode(tokenIds: tokens).utf8)),
        stopOutcome: stopOutcome
            ?? (tokens.count == retainedTokens && discardedExtraGeneratedToken ? .cancelled : .length),
        cacheFingerprint: fingerprint,
        timing: .init(
            promptSeconds: max(iterator.promptPrefillTime, 1e-9),
            generationSeconds: max(end - generationStart, 1e-9),
            wallSeconds: max(end - wallStart, 1e-9),
            e2eSeconds: max(end - wallStart, 1e-9)),
        mtpTelemetry: telemetry,
        mtpPhaseAttribution: mtpPhaseAttribution(
            from: iterator,
            cacheFingerprintSeconds: fingerprintSeconds),
        promptPreparationEvaluationOrder:
            iterator.promptPreparationTelemetry?.evaluationOrder,
        passthroughReason: iterator.passthroughReason)
}

private func runMTPCancelAfterAcceptedDraft(
    _ spec: QwenMTPCorpusCaseSpec,
    pair: Qwen35ExactMTPLoadedPair,
    parameters: GenerateParameters
) throws -> CorpusRun {
    let input = input(for: spec, tokenizer: pair.target.tokenizer)
    let promptTokenCount = promptTokenCount(for: spec, tokenizer: pair.target.tokenizer)
    let cache = pair.target.model.newCache(parameters: parameters)
    let wallStart = ProcessInfo.processInfo.systemUptime
    var iterator = try MTPSpeculativeTokenIterator(
        input: input,
        mainModel: pair.target.model,
        drafter: pair.drafter.model,
        mainCache: cache,
        parameters: parameters,
        blockSize: pair.binding.runtimeBlockSize,
        collectPhaseTelemetry: true)
    let generationStart = ProcessInfo.processInfo.systemUptime
    var tokens: [Int] = []
    let safetyCap = max(parameters.maxTokens ?? spec.maxTokens, spec.maxTokens) + 8
    let stopTokenIds = buildQwenMTPCorpusStopTokenIds(
        modelConfiguration: pair.target.configuration,
        tokenizer: pair.target.tokenizer)
    var stopped = false
    while tokens.count < safetyCap, iterator.acceptedCount <= 0 {
        guard let token = iterator.next() else { break }
        if token == pair.target.tokenizer.unknownTokenId || stopTokenIds.contains(token) {
            iterator.discardGeneratedToken()
            stopped = true
            break
        }
        tokens.append(token)
    }
    let acceptedDraftObserved = iterator.acceptedCount > 0
    var discardedExtraGeneratedToken = false
    if acceptedDraftObserved, !stopped, iterator.next() != nil {
        iterator.discardGeneratedToken()
        discardedExtraGeneratedToken = true
    }
    iterator.finalizeGeneration()
    let end = ProcessInfo.processInfo.systemUptime
    let fingerprintStart = ProcessInfo.processInfo.systemUptime
    let fingerprint = fingerprintCache(cache)
    let fingerprintSeconds = ProcessInfo.processInfo.systemUptime - fingerprintStart
    let telemetry = mtpTelemetry(from: iterator, emittedTokenCount: tokens.count)
    return CorpusRun(
        promptTokenCount: promptTokenCount,
        tokens: tokens,
        tokenIDsSHA256: tokenIDsSHA256(tokens),
        decodedBytesSHA256: sha256Hex(Data(pair.target.tokenizer.decode(tokenIds: tokens).utf8)),
        stopOutcome: acceptedDraftObserved && discardedExtraGeneratedToken
            ? .cancelled
            : (stopped ? .stop : .length),
        cacheFingerprint: fingerprint,
        timing: .init(
            promptSeconds: max(iterator.promptPrefillTime, 1e-9),
            generationSeconds: max(end - generationStart, 1e-9),
            wallSeconds: max(end - wallStart, 1e-9),
            e2eSeconds: max(end - wallStart, 1e-9)),
        mtpTelemetry: telemetry,
        mtpPhaseAttribution: mtpPhaseAttribution(
            from: iterator,
            cacheFingerprintSeconds: fingerprintSeconds),
        promptPreparationEvaluationOrder:
            iterator.promptPreparationTelemetry?.evaluationOrder,
        passthroughReason: iterator.passthroughReason)
}

private func caseResult(
    spec: QwenMTPCorpusCaseSpec,
    scalar: CorpusRun,
    mtp: CorpusRun
) -> QwenMTPCorpusCaseResult {
    precondition(mtp.mtpPhaseAttribution != nil, "MTP corpus run must collect phase attribution")
    let exactness = MTPStreamExactness.compare(candidate: mtp.tokens, baseline: scalar.tokens)
    let cacheMismatch = firstCacheMismatch(scalar.cacheFingerprint, mtp.cacheFingerprint)
    return QwenMTPCorpusCaseResult(
        caseID: spec.id,
        kind: spec.kind,
        maxTokens: spec.maxTokens,
        promptTokenCount: scalar.promptTokenCount,
        scalarTokenCount: scalar.tokens.count,
        mtpTokenCount: mtp.tokens.count,
        scalarTokenIDsSHA256: scalar.tokenIDsSHA256,
        mtpTokenIDsSHA256: mtp.tokenIDsSHA256,
        tokenExactness: exactness,
        scalarDecodedBytesSHA256: scalar.decodedBytesSHA256,
        mtpDecodedBytesSHA256: mtp.decodedBytesSHA256,
        scalarStopOutcome: scalar.stopOutcome,
        mtpStopOutcome: mtp.stopOutcome,
        scalarCacheFingerprint: scalar.cacheFingerprint,
        mtpCacheFingerprint: mtp.cacheFingerprint,
        firstCacheMismatch: cacheMismatch,
        scalarTiming: scalar.timing,
        mtpTiming: mtp.timing,
        mtpTelemetry: mtp.mtpTelemetry,
        mtpPhaseAttribution: mtp.mtpPhaseAttribution!,
        passthroughReason: mtp.passthroughReason)
}

private func runQwenMTPProfile(pair: Qwen35ExactMTPLoadedPair) throws -> QwenMTPCorpusProfileEvidence {
    var samples: [QwenMTPCorpusProfileSample] = []
    samples.reserveCapacity(QwenMTPCorpusGate.profilePlan.caseIDs.count * QwenMTPCorpusGate.profilePlan.totalPairsPerCase)
    for caseID in QwenMTPCorpusGate.profilePlan.caseIDs {
        let spec = QwenMTPCorpusGate.cases.first { $0.id == caseID }!
        for pairIndex in 0..<QwenMTPCorpusGate.profilePlan.totalPairsPerCase {
            let order = QwenMTPCorpusGate.profilePlan.orders[pairIndex]
            let parameters = GenerateParameters(maxTokens: 128, temperature: 0)
            let scalar: CorpusRun
            let mtp: CorpusRun
            switch order {
            case .scalarThenMTP:
                scalar = try runScalar(spec, pair: pair, parameters: parameters)
                mtp = try runMTPDrain(spec, pair: pair, parameters: parameters)
            case .mtpThenScalar:
                mtp = try runMTPDrain(spec, pair: pair, parameters: parameters)
                scalar = try runScalar(spec, pair: pair, parameters: parameters)
            }
            let scalarTPS = Double(max(scalar.tokens.count, 1)) / scalar.timing.e2eSeconds
            let mtpTPS = Double(max(mtp.tokens.count, 1)) / mtp.timing.e2eSeconds
            let decodeRatio = scalar.timing.generationSeconds > 0
                ? scalar.timing.generationSeconds / mtp.timing.generationSeconds
                : 0
            samples.append(.init(
                caseID: caseID,
                pairIndex: pairIndex,
                warmup: pairIndex < QwenMTPCorpusGate.profilePlan.droppedWarmupPairs,
                order: order,
                exactness: exactnessEvidence(scalar: scalar, mtp: mtp),
                scalarTiming: scalar.timing,
                mtpTiming: mtp.timing,
                scalarTokensPerSecond: scalarTPS,
                mtpTokensPerSecond: mtpTPS,
                decodeOnlyRatio: decodeRatio,
                e2eRatio: scalarTPS > 0 ? mtpTPS / scalarTPS : 0,
                mtpTelemetry: mtp.mtpTelemetry,
                mtpPhaseAttribution: mtp.mtpPhaseAttribution!,
                passthroughReason: mtp.passthroughReason))
        }
    }
    return QwenMTPCorpusProfileEvidence(
        releaseBuildRequired: true,
        releaseBuildObserved: qwenMTPCorpusReleaseBuildObserved,
        samples: samples)
}

private func parameters(for spec: QwenMTPCorpusCaseSpec) -> GenerateParameters {
    switch spec.kind {
    case .forcedFallback:
        GenerateParameters(maxTokens: spec.maxTokens, temperature: 0.7, topP: 0.95, seed: 0x51A7)
    default:
        GenerateParameters(maxTokens: spec.maxTokens, temperature: 0)
    }
}

private func input(
    for spec: QwenMTPCorpusCaseSpec,
    tokenizer: any MLXLMCommon.Tokenizer
) -> LMInput {
    let tokens = tokenizer.encode(text: spec.prompt, addSpecialTokens: true)
        .compactMap(Int32.init(exactly:))
    return LMInput(tokens: MLXArray(tokens))
}

private func promptTokenCount(
    for spec: QwenMTPCorpusCaseSpec,
    tokenizer: any MLXLMCommon.Tokenizer
) -> Int {
    tokenizer.encode(text: spec.prompt, addSpecialTokens: true).count
}

private func exactnessEvidence(
    scalar: CorpusRun,
    mtp: CorpusRun
) -> QwenMTPCorpusExactnessEvidence {
    QwenMTPCorpusExactnessEvidence(
        scalarTokenCount: scalar.tokens.count,
        mtpTokenCount: mtp.tokens.count,
        scalarTokenIDsSHA256: scalar.tokenIDsSHA256,
        mtpTokenIDsSHA256: mtp.tokenIDsSHA256,
        scalarDecodedBytesSHA256: scalar.decodedBytesSHA256,
        mtpDecodedBytesSHA256: mtp.decodedBytesSHA256,
        scalarStopOutcome: scalar.stopOutcome,
        mtpStopOutcome: mtp.stopOutcome,
        scalarCacheFingerprint: scalar.cacheFingerprint,
        mtpCacheFingerprint: mtp.cacheFingerprint,
        firstCacheMismatch: firstCacheMismatch(scalar.cacheFingerprint, mtp.cacheFingerprint))
}

private func tokenIDsSHA256(_ tokens: [Int]) -> String {
    var data = Data()
    data.reserveCapacity(tokens.count * MemoryLayout<Int64>.size)
    for token in tokens {
        var value = Int64(token).littleEndian
        withUnsafeBytes(of: &value) { bytes in
            data.append(contentsOf: bytes)
        }
    }
    return sha256Hex(data)
}

private func mtpTelemetry(
    from iterator: MTPSpeculativeTokenIterator,
    emittedTokenCount: Int
) -> QwenMTPCorpusMTPTelemetry {
    if let telemetry = iterator.speculativeDecodingTelemetry {
        return QwenMTPCorpusMTPTelemetry(
            proposedDraftTokens: telemetry.draftTokenCount,
            acceptedDraftTokens: telemetry.acceptedDraftTokenCount,
            rejectedDraftTokens: telemetry.rejectedDraftTokenCount,
            roundCount: telemetry.roundCount,
            targetModelCallCount: telemetry.targetModelCallCount,
            draftModelCallCount: telemetry.draftModelCallCount,
            targetVerifiedTokenCount: telemetry.targetVerifiedTokenCount,
            emittedTokenCount: telemetry.emittedTokenCount)
    }
    return QwenMTPCorpusMTPTelemetry(
        proposedDraftTokens: iterator.proposedCount,
        acceptedDraftTokens: iterator.acceptedCount,
        rejectedDraftTokens: max(0, iterator.proposedCount - iterator.acceptedCount),
        roundCount: 0,
        targetModelCallCount: 0,
        draftModelCallCount: 0,
        targetVerifiedTokenCount: 0,
        emittedTokenCount: emittedTokenCount)
}

private func mtpPhaseAttribution(
    from iterator: MTPSpeculativeTokenIterator,
    cacheFingerprintSeconds: Double
) -> QwenMTPCorpusMTPPhaseAttribution {
    let phase = iterator.speculativeDecodingPhaseTelemetry
    let targetPromptPreparation = iterator.promptPreparationTelemetry.map { telemetry in
        let chunks = telemetry.chunks.map {
            QwenMTPPromptPreparationChunkAttribution(
                tokenOffset: $0.tokenOffset,
                tokenCount: $0.tokenCount,
                targetForwardSchedulingSeconds:
                    $0.targetForwardSchedulingSeconds)
        }
        let attributedSeconds = chunks.reduce(
            telemetry.cacheEvaluationSeconds
                + telemetry.hiddenEvaluationSeconds
                + telemetry.concatenatedHiddenEvaluationSeconds
                + telemetry.preparedCacheHandoffSeconds
                + iterator.promptPreparationPhaseBoundarySynchronizationSeconds
        ) {
            $0 + $1.targetForwardSchedulingSeconds
        }
        return QwenMTPPromptPreparationAttribution(
            promptTokenCount: telemetry.promptTokenCount,
            hiddenShape: telemetry.hiddenShape,
            hiddenByteCount: telemetry.hiddenByteCount,
            chunks: chunks,
            cacheEvaluationSeconds: telemetry.cacheEvaluationSeconds,
            hiddenEvaluationSeconds: telemetry.hiddenEvaluationSeconds,
            concatenatedHiddenEvaluationSeconds:
                telemetry.concatenatedHiddenEvaluationSeconds,
            preparedCacheHandoffSeconds: telemetry.preparedCacheHandoffSeconds,
            phaseBoundarySynchronizationSeconds:
                iterator.promptPreparationPhaseBoundarySynchronizationSeconds,
            targetPrefillResidualSeconds:
                phase.targetPrefillSeconds - attributedSeconds)
    }
    return QwenMTPCorpusMTPPhaseAttribution(
        targetPrefillSeconds: phase.targetPrefillSeconds,
        drafterPromptPrimingSeconds: phase.drafterPromptPrimingSeconds,
        draftBlockSeconds: phase.draftBlockSeconds,
        targetVerificationSeconds: phase.targetVerificationSeconds,
        targetTailSeconds: phase.targetTailSeconds,
        hybridRewindReplaySeconds: phase.hybridRewindReplaySeconds,
        finalizationSeconds: phase.finalizationSeconds,
        cacheFingerprintSeconds: cacheFingerprintSeconds,
        targetPrefillCount: phase.targetPrefillCount,
        drafterPromptPrimingCount: phase.drafterPromptPrimingCount,
        draftBlockCount: phase.draftBlockCount,
        targetVerificationCount: phase.targetVerificationCount,
        targetTailCount: phase.targetTailCount,
        hybridRewindReplayCount: phase.hybridRewindReplayCount,
        finalizationCount: phase.finalizationCount,
        cacheFingerprintCount: 1,
        targetPromptPreparation: targetPromptPreparation)
}

private func buildQwenMTPCorpusStopTokenIds(
    modelConfiguration: ModelConfiguration,
    tokenizer: any MLXLMCommon.Tokenizer
) -> Set<Int> {
    var stopTokenIds = modelConfiguration.eosTokenIds
    if let tokenizerEOS = tokenizer.eosTokenId {
        stopTokenIds.insert(tokenizerEOS)
    }
    for token in modelConfiguration.extraEOSTokens {
        if let id = tokenizer.convertTokenToId(token) {
            stopTokenIds.insert(id)
        }
    }
    return stopTokenIds
}

private func fingerprintCache(_ cache: [KVCache]) -> QwenMTPCorpusCacheFingerprint {
    var hasher = SHA256()
    var entries: [QwenMTPCorpusCacheLayerFingerprint] = []
    entries.reserveCapacity(cache.count)
    for (layerIndex, entry) in cache.enumerated() {
        let cacheType = String(describing: type(of: entry))
        let state = entry.state
        var states: [QwenMTPCorpusCacheStateFingerprint] = []
        states.reserveCapacity(state.count)
        var metaHasher = SHA256()
        for (metaIndex, value) in entry.metaState.enumerated() {
            update(&metaHasher, "meta=\(metaIndex)")
            update(&metaHasher, value)
        }
        for (stateIndex, array) in state.enumerated() {
            eval(array)
            let shape = array.shape
            let dtype = String(describing: array.dtype)
            let bytes = array.asData(access: .copy).data
            let byteDigest = sha256Hex(bytes)
            states.append(.init(
                stateIndex: stateIndex,
                shape: shape,
                dtype: dtype,
                byteCount: bytes.count,
                sha256: byteDigest))
        }
        let metaDigest = metaHasher.finalize().map { String(format: "%02x", $0) }.joined()
        update(&hasher, "layer=\(layerIndex)")
        update(&hasher, "type=\(cacheType)")
        update(&hasher, "offset=\(entry.offset)")
        update(&hasher, "metaState=\(metaDigest)")
        update(&hasher, "stateCount=\(states.count)")
        for state in states {
            update(&hasher, "state=\(state.stateIndex)")
            update(&hasher, "shape=\(state.shape)")
            update(&hasher, "dtype=\(state.dtype)")
            update(&hasher, "bytes=\(state.byteCount)")
            update(&hasher, "sha256=\(state.sha256)")
        }
        entries.append(.init(
            layerIndex: layerIndex,
            cacheType: cacheType,
            offset: entry.offset,
            metaStateSHA256: metaDigest,
            stateCount: states.count,
            states: states))
    }
    return QwenMTPCorpusCacheFingerprint(
        digest: hasher.finalize().map { String(format: "%02x", $0) }.joined(),
        entries: entries)
}

private func update(_ hasher: inout SHA256, _ value: String) {
    hasher.update(data: Data(value.utf8))
    hasher.update(data: Data([0]))
}

private func firstCacheMismatch(
    _ scalar: QwenMTPCorpusCacheFingerprint,
    _ mtp: QwenMTPCorpusCacheFingerprint
) -> String? {
    guard scalar != mtp else { return nil }
    if scalar.entries.count != mtp.entries.count {
        return "layerCount scalar=\(scalar.entries.count) mtp=\(mtp.entries.count)"
    }
    for (layerIndex, pair) in zip(scalar.entries, mtp.entries).enumerated() {
        if pair.0.layerIndex != pair.1.layerIndex {
            return "layer[\(layerIndex)].index scalar=\(pair.0.layerIndex) mtp=\(pair.1.layerIndex)"
        }
        if pair.0.cacheType != pair.1.cacheType {
            return "layer[\(layerIndex)].type scalar=\(pair.0.cacheType) mtp=\(pair.1.cacheType)"
        }
        if pair.0.offset != pair.1.offset {
            return "layer[\(layerIndex)].offset scalar=\(pair.0.offset) mtp=\(pair.1.offset)"
        }
        if pair.0.metaStateSHA256 != pair.1.metaStateSHA256 {
            return "layer[\(layerIndex)].metaState scalar=\(pair.0.metaStateSHA256) mtp=\(pair.1.metaStateSHA256)"
        }
        if pair.0.stateCount != pair.1.stateCount || pair.0.states.count != pair.1.states.count {
            return "layer[\(layerIndex)].stateCount scalar=\(pair.0.states.count) mtp=\(pair.1.states.count)"
        }
        for (stateIndex, statePair) in zip(pair.0.states, pair.1.states).enumerated() {
            if statePair.0.shape != statePair.1.shape {
                return "layer[\(layerIndex)].state[\(stateIndex)].shape scalar=\(statePair.0.shape) mtp=\(statePair.1.shape)"
            }
            if statePair.0.dtype != statePair.1.dtype {
                return "layer[\(layerIndex)].state[\(stateIndex)].dtype scalar=\(statePair.0.dtype) mtp=\(statePair.1.dtype)"
            }
            if statePair.0.byteCount != statePair.1.byteCount {
                return "layer[\(layerIndex)].state[\(stateIndex)].byteCount scalar=\(statePair.0.byteCount) mtp=\(statePair.1.byteCount)"
            }
            if statePair.0.sha256 != statePair.1.sha256 {
                return "layer[\(layerIndex)].state[\(stateIndex)].sha256 scalar=\(statePair.0.sha256) mtp=\(statePair.1.sha256)"
            }
        }
    }
    if scalar.digest != mtp.digest {
        return "digest scalar=\(scalar.digest) mtp=\(mtp.digest)"
    }
    return "cache fingerprint mismatch"
}

private func runtimeBinding(
    _ binding: Qwen35ExactMTPBinding
) -> QwenMTPCorpusRuntimeBinding {
    QwenMTPCorpusRuntimeBinding(
        targetModelID: binding.targetModelID,
        drafterModelID: binding.drafterModelID,
        targetRevision: binding.targetRevision,
        drafterRevision: binding.drafterRevision,
        sourceRevision: binding.sourceRevision,
        blockSize: binding.runtimeBlockSize,
        maxAcceptedDrafts: binding.maximumAcceptedDraftTokens)
}

private func resultRecord(
    _ payload: QwenMTPCorpusEvidencePayload,
    targetPath: String,
    subcommand: String = "qwen-mtp-corpus"
) -> ResultRecord<QwenMTPCorpusEvidencePayload> {
    let modelConfig = ProvenanceCLI.modelConfig(at: targetPath)
    return ResultRecord(
        subcommand: subcommand,
        provenance: Provenance(
            date: ISO8601DateFormatter().string(from: Date()),
            hardwareChip: ProvenanceCLI.chipBrand(),
            hardwareRAMBytes: ProvenanceCLI.ramBytes(),
            hardwareOS: ProvenanceCLI.osVersion(),
            harnessGitSHA: ProvenanceCLI.harnessGitSHA(),
            mlxSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: ProvenanceCLI.mlxSwiftLMRevision,
            modelPath: payload.binding.targetModelID,
            modelConfigHash: modelConfig.hash,
            modelCheckpointManifestHash: try? ProvenanceCLI.checkpointManifestHash(at: targetPath),
            modelQuant: modelConfig.quant,
            corpusId: QwenMTPCorpusGate.corpusID,
            corpusContentHash: QwenMTPCorpusGate.corpusContentHash,
            nonce: UUID().uuidString),
        payload: payload)
}

private func evaluationOrderResultRecord(
    _ payload: QwenMTPEvaluationOrderIsolationPayload,
    targetPath: String
) -> ResultRecord<QwenMTPEvaluationOrderIsolationPayload> {
    let modelConfig = ProvenanceCLI.modelConfig(at: targetPath)
    return ResultRecord(
        subcommand: "qwen-mtp-eval-order",
        provenance: Provenance(
            date: ISO8601DateFormatter().string(from: Date()),
            hardwareChip: ProvenanceCLI.chipBrand(),
            hardwareRAMBytes: ProvenanceCLI.ramBytes(),
            hardwareOS: ProvenanceCLI.osVersion(),
            harnessGitSHA: ProvenanceCLI.harnessGitSHA(),
            mlxSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: ProvenanceCLI.mlxSwiftLMRevision,
            modelPath: payload.binding.targetModelID,
            modelConfigHash: modelConfig.hash,
            modelCheckpointManifestHash:
                try? ProvenanceCLI.checkpointManifestHash(at: targetPath),
            modelQuant: modelConfig.quant,
            corpusId: payload.corpusID,
            corpusContentHash: payload.corpusContentHash,
            nonce: UUID().uuidString),
        payload: payload)
}

private func combinedEvaluationResultRecord(
    _ payload: QwenMTPCombinedEvaluationIsolationPayload,
    targetPath: String
) -> ResultRecord<QwenMTPCombinedEvaluationIsolationPayload> {
    let modelConfig = ProvenanceCLI.modelConfig(at: targetPath)
    return ResultRecord(
        subcommand: "qwen-mtp-eval-combined",
        provenance: Provenance(
            date: ISO8601DateFormatter().string(from: Date()),
            hardwareChip: ProvenanceCLI.chipBrand(),
            hardwareRAMBytes: ProvenanceCLI.ramBytes(),
            hardwareOS: ProvenanceCLI.osVersion(),
            harnessGitSHA: ProvenanceCLI.harnessGitSHA(),
            mlxSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: ProvenanceCLI.mlxSwiftLMRevision,
            modelPath: payload.binding.targetModelID,
            modelConfigHash: modelConfig.hash,
            modelCheckpointManifestHash:
                try? ProvenanceCLI.checkpointManifestHash(at: targetPath),
            modelQuant: modelConfig.quant,
            corpusId: payload.corpusID,
            corpusContentHash: payload.corpusContentHash,
            nonce: UUID().uuidString),
        payload: payload)
}

private func hiddenFirstRuntimeResultRecord(
    _ payload: QwenMTPHiddenFirstRuntimeEquivalencePayload,
    targetPath: String,
    subcommand: String
) -> ResultRecord<QwenMTPHiddenFirstRuntimeEquivalencePayload> {
    let modelConfig = ProvenanceCLI.modelConfig(at: targetPath)
    return ResultRecord(
        subcommand: subcommand,
        provenance: Provenance(
            date: ISO8601DateFormatter().string(from: Date()),
            hardwareChip: ProvenanceCLI.chipBrand(),
            hardwareRAMBytes: ProvenanceCLI.ramBytes(),
            hardwareOS: ProvenanceCLI.osVersion(),
            harnessGitSHA: ProvenanceCLI.harnessGitSHA(),
            mlxSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: ProvenanceCLI.mlxSwiftLMRevision,
            modelPath: payload.binding.targetModelID,
            modelConfigHash: modelConfig.hash,
            modelCheckpointManifestHash:
                try? ProvenanceCLI.checkpointManifestHash(at: targetPath),
            modelQuant: modelConfig.quant,
            corpusId: payload.corpusID,
            corpusContentHash: payload.corpusContentHash,
            nonce: UUID().uuidString),
        payload: payload)
}

// MARK: - Long-decode (production-completion-shape) non-authoritative profile

func runQwenMTPLongProfile(_ flags: Flags) async throws {
    let targetPath = try flags.strictString("target", default: "")
    let drafterPath = try flags.strictString("drafter", default: "")
    let evidencePath = try flags.strictString("evidence", default: "")
    let bindingRaw = try flags.strictString(
        "binding", default: Qwen35ExactMTPRuntimeSelection.qwen35_9BDepth1.rawValue)
    guard !targetPath.isEmpty, !drafterPath.isEmpty, !evidencePath.isEmpty,
        let selection = Qwen35ExactMTPRuntimeSelection(rawValue: bindingRaw)
    else {
        throw QwenMTPCorpusCLIError.longProfileUsage
    }
    guard qwenMTPCorpusReleaseBuildObserved else {
        throw QwenMTPCorpusCLIError.releaseBuildRequired
    }
    let gdnMode: Qwen38MTPPerformanceScorecardGDNMode = .gdnOn
    let gdnObservedEnv = Qwen38MTPScorecardProcessFacts.observedGDNEnv(mode: gdnMode)
    guard gdnObservedEnv == .enabled else {
        throw QwenMTPCorpusCLIError.gdnEnvironmentNotPinned
    }

    let evidenceURL = URL(fileURLWithPath: evidencePath)
    try requireNewOrEmptyEvidence(evidenceURL)
    let downloader = QwenMTPSelectedExactRevisionDownloader(
        selection: selection,
        target: URL(fileURLWithPath: targetPath, isDirectory: true),
        drafter: URL(fileURLWithPath: drafterPath, isDirectory: true))
    print("qwen-mtp-long-profile loading binding=\(selection.rawValue) gdn=\(gdnMode.rawValue)")
    let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
        selection: selection,
        from: downloader,
        using: #huggingFaceTokenizerLoader())

    let plan = QwenMTPLongProfileGate.profilePlan
    var caseProfiles: [QwenMTPLongProfileCaseProfile] = []
    var samples: [QwenMTPCorpusProfileSample] = []

    func writeRejectedEvidence() throws {
        let payload = longProfilePayload(
            pair: pair,
            gdnMode: gdnMode,
            gdnObservedEnv: gdnObservedEnv,
            caseProfiles: caseProfiles,
            samples: samples)
        let rejectedURL = evidenceURL.appendingPathExtension("rejected")
        try requireNewOrEmptyEvidence(rejectedURL)
        try RequiredJSONLWriter.append(
            longProfileResultRecord(
                payload,
                targetPath: targetPath,
                subcommand: QwenMTPLongProfileGate.rejectedSubcommand),
            to: rejectedURL)
    }

    for caseID in plan.caseIDs {
        guard let spec = QwenMTPLongProfileGate.cases.first(where: { $0.id == caseID }) else {
            throw QwenMTPCorpusCLIError.longProfileUsage
        }
        let promptTokens = promptTokenCount(for: spec, tokenizer: pair.target.tokenizer)
        guard promptTokens >= QwenMTPLongProfileGate.minimumPromptTokenCount else {
            throw QwenMTPCorpusCLIError.longProfileMeasurementFailure(
                caseID: caseID,
                pairIndex: -1,
                reason: "prompt tokenizes to \(promptTokens); below long-context floor "
                    + "\(QwenMTPLongProfileGate.minimumPromptTokenCount) — refusing before any generation")
        }
        print("qwen-mtp-long-profile \(caseID) promptTokens=\(promptTokens)")
        caseProfiles.append(.init(caseID: caseID, promptTokenCount: promptTokens))
    }

    for (caseIndex, caseID) in plan.caseIDs.enumerated() {
        let spec = QwenMTPLongProfileGate.cases[caseIndex]
        for pairIndex in 0..<plan.totalPairsPerCase {
            let order = plan.orders[pairIndex]
            let parameters = GenerateParameters(maxTokens: spec.maxTokens, temperature: 0)
            let scalar: CorpusRun
            let mtp: CorpusRun
            switch order {
            case .scalarThenMTP:
                scalar = try runScalar(spec, pair: pair, parameters: parameters)
                mtp = try runMTPDrain(spec, pair: pair, parameters: parameters)
            case .mtpThenScalar:
                mtp = try runMTPDrain(spec, pair: pair, parameters: parameters)
                scalar = try runScalar(spec, pair: pair, parameters: parameters)
            }

            let exactness = exactnessEvidence(scalar: scalar, mtp: mtp)
            if let failure = longProfilePairFailure(scalar: scalar, mtp: mtp) {
                try writeRejectedEvidence()
                throw QwenMTPCorpusCLIError.longProfileMeasurementFailure(
                    caseID: caseID, pairIndex: pairIndex, reason: failure)
            }

            let scalarTPS = Double(scalar.tokens.count) / scalar.timing.e2eSeconds
            let mtpTPS = Double(mtp.tokens.count) / mtp.timing.e2eSeconds
            samples.append(.init(
                caseID: caseID,
                pairIndex: pairIndex,
                warmup: pairIndex < plan.droppedWarmupPairs,
                order: order,
                exactness: exactness,
                scalarTiming: scalar.timing,
                mtpTiming: mtp.timing,
                scalarTokensPerSecond: scalarTPS,
                mtpTokensPerSecond: mtpTPS,
                decodeOnlyRatio: scalar.timing.generationSeconds / mtp.timing.generationSeconds,
                e2eRatio: mtpTPS / scalarTPS,
                mtpTelemetry: mtp.mtpTelemetry,
                mtpPhaseAttribution: mtp.mtpPhaseAttribution!,
                passthroughReason: mtp.passthroughReason))
            print(String(
                format: "qwen-mtp-long-profile %@ pair %d/%d order=%@ warmup=%@ tokens=%d "
                    + "scalar_decode=%.1fs mtp_decode=%.1fs decode_ratio=%.4f accepted=%d/%d",
                caseID,
                pairIndex + 1,
                plan.totalPairsPerCase,
                order.rawValue,
                pairIndex < plan.droppedWarmupPairs ? "yes" : "no",
                mtp.tokens.count,
                scalar.timing.generationSeconds,
                mtp.timing.generationSeconds,
                scalar.timing.generationSeconds / mtp.timing.generationSeconds,
                mtp.mtpTelemetry.acceptedDraftTokens,
                mtp.mtpTelemetry.proposedDraftTokens))
        }
    }

    let payload = longProfilePayload(
        pair: pair,
        gdnMode: gdnMode,
        gdnObservedEnv: gdnObservedEnv,
        caseProfiles: caseProfiles,
        samples: samples)
    let summary: QwenMTPLongProfileSummary
    do {
        summary = try QwenMTPLongProfileGate.validate(payload)
    } catch {
        try writeRejectedEvidence()
        throw error
    }
    try RequiredJSONLWriter.append(
        longProfileResultRecord(
            payload,
            targetPath: targetPath,
            subcommand: QwenMTPLongProfileGate.subcommand),
        to: evidenceURL)
    for caseSummary in summary.perCase {
        print(String(
            format: "qwen-mtp-long-profile SUMMARY %@ decode_ratio_median=%.4f "
                + "e2e_ratio_median=%.4f scalar_decode_tps=%.2f mtp_decode_tps=%.2f "
                + "acceptance=%.4f",
            caseSummary.caseID,
            caseSummary.medianDecodeOnlyRatio,
            caseSummary.medianE2ERatio,
            caseSummary.medianScalarDecodeTokensPerSecond,
            caseSummary.medianMTPDecodeTokensPerSecond,
            caseSummary.meanDraftAcceptanceRate))
    }
    print(String(
        format: "qwen-mtp-long-profile SUMMARY aggregate decode_ratio_median=%.4f "
            + "e2e_ratio_median=%.4f (%@, non-authoritative)",
        summary.aggregateMedianDecodeOnlyRatio,
        summary.aggregateMedianE2ERatio,
        payload.binding.targetModelID))
}

private func longProfilePairFailure(scalar: CorpusRun, mtp: CorpusRun) -> String? {
    let exact = MTPStreamExactness.compare(candidate: mtp.tokens, baseline: scalar.tokens)
    if !exact.exact || !exact.lengthMatched {
        return "token stream divergence at index \(exact.firstDivergenceIndex.map(String.init) ?? "length")"
    }
    if scalar.decodedBytesSHA256 != mtp.decodedBytesSHA256 {
        return "decoded byte digest divergence"
    }
    if let mismatch = firstCacheMismatch(scalar.cacheFingerprint, mtp.cacheFingerprint) {
        return "cache fingerprint divergence: \(mismatch)"
    }
    if scalar.stopOutcome != mtp.stopOutcome {
        return "stop outcome divergence scalar=\(scalar.stopOutcome.rawValue) mtp=\(mtp.stopOutcome.rawValue)"
    }
    if mtp.tokens.count < QwenMTPLongProfileGate.minimumGeneratedTokens {
        return "generated only \(mtp.tokens.count) tokens; below steady-state floor "
            + "\(QwenMTPLongProfileGate.minimumGeneratedTokens)"
    }
    if let reason = mtp.passthroughReason {
        return "MTP passthrough: \(reason)"
    }
    return nil
}

private func longProfilePayload(
    pair: Qwen35ExactMTPLoadedPair,
    gdnMode: Qwen38MTPPerformanceScorecardGDNMode,
    gdnObservedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv,
    caseProfiles: [QwenMTPLongProfileCaseProfile],
    samples: [QwenMTPCorpusProfileSample]
) -> QwenMTPLongProfileEvidencePayload {
    QwenMTPLongProfileEvidencePayload(
        schemaVersion: QwenMTPLongProfileGate.schemaVersion,
        corpusID: QwenMTPLongProfileGate.corpusID,
        corpusContentHash: QwenMTPLongProfileGate.corpusContentHash,
        measurementClass: QwenMTPLongProfileGate.measurementClass,
        binding: runtimeBinding(pair.binding),
        host: .init(
            chip: ProvenanceCLI.chipBrand(),
            ramBytes: ProvenanceCLI.ramBytes(),
            os: ProvenanceCLI.osVersion()),
        gdnMode: gdnMode,
        gdnObservedEnv: gdnObservedEnv,
        releaseBuildRequired: true,
        releaseBuildObserved: qwenMTPCorpusReleaseBuildObserved,
        caseProfiles: caseProfiles,
        samples: samples)
}

private func longProfileResultRecord(
    _ payload: QwenMTPLongProfileEvidencePayload,
    targetPath: String,
    subcommand: String
) -> ResultRecord<QwenMTPLongProfileEvidencePayload> {
    let modelConfig = ProvenanceCLI.modelConfig(at: targetPath)
    return ResultRecord(
        subcommand: subcommand,
        provenance: Provenance(
            date: ISO8601DateFormatter().string(from: Date()),
            hardwareChip: ProvenanceCLI.chipBrand(),
            hardwareRAMBytes: ProvenanceCLI.ramBytes(),
            hardwareOS: ProvenanceCLI.osVersion(),
            harnessGitSHA: ProvenanceCLI.harnessGitSHA(),
            mlxSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: ProvenanceCLI.mlxSwiftLMRevision,
            modelPath: payload.binding.targetModelID,
            modelConfigHash: modelConfig.hash,
            modelCheckpointManifestHash: try? ProvenanceCLI.checkpointManifestHash(at: targetPath),
            modelQuant: modelConfig.quant,
            corpusId: QwenMTPLongProfileGate.corpusID,
            corpusContentHash: QwenMTPLongProfileGate.corpusContentHash,
            nonce: UUID().uuidString),
        payload: payload)
}

private struct QwenMTPSelectedExactRevisionDownloader: Downloader {
    let selection: Qwen35ExactMTPRuntimeSelection
    let target: URL
    let drafter: URL

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard !useLatest, patterns == ["*.safetensors", "*.json", "*.jinja"] else {
            throw QwenMTPCorpusCLIError.unexpectedDownloadRequest(id: id, revision: revision)
        }
        progressHandler(Progress(totalUnitCount: 1))
        let lock = knownLock
        if id == lock.targetIdentity.modelID, revision == lock.targetIdentity.revision {
            return target
        }
        if id == lock.drafterIdentity.modelID, revision == lock.drafterIdentity.revision {
            return drafter
        }
        throw QwenMTPCorpusCLIError.unexpectedDownloadRequest(id: id, revision: revision)
    }

    private var knownLock: QwenMTPArtifactLock {
        switch selection {
        case .qwen35_9BDepth1:
            QwenMTPKnownArtifactLocks.qwen35_9BDepth1
        case .qwen38_27BMXFP8Depth1:
            QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
        case .qwen38_27B4BitDepth1:
            QwenMTPKnownArtifactLocks.qwen38_27B4BitDepth1
        }
    }
}
