import Foundation
import HarnessCore
import SpikeCore

enum BenchCLIError: Error, Equatable, CustomStringConvertible {
    case missingModel
    case unknownKVQuantTier(String)
    case missingMatrixID
    case missingWorkloadNonce
    case tierCellMismatch(tier: String, cell: String)
    case unauditedKVTunerPrompt
    case invalidPromptRepeat(Int)
    case contextLimitExceeded(
        promptTokens: Int, maxTokens: Int, limit: Int)
    case invalidRuns(Int)
    case invalidQualificationRuns(Int)
    case qualificationFlagsWithoutQualification
    case missingQualificationFlag(String)
    case invalidPostWarmupThermalTarget(String)
    case invalidPostWarmupThermalPolicy
    case invalidMaxTokens(Int)
    case unknownSpec(String)
    case unmeasuredSpecKV(String)
    case unknownAttentionOperation(String)
    case unsupportedAttentionTier(String)
    case invalidCheckpointContentSHA256(String)
    case missingPrefillTiming
    case invalidPrefillTiming
    case missingKVTunerEngagement
    case missingLossyEngagement(String)
    case missingCompressedAttentionEvidence
    case compressedAttentionWorkspaceExceedsMLXPeak(
        workspaceBytes: Int, peakBytes: Int)
    case csvSchemaMismatch
    case outputPathCollision(String)
    case symbolicLinkOutput(String)

    var description: String {
        switch self {
        case .missingModel:
            return "bench requires --model <PATH>"
        case .unknownKVQuantTier(let tier):
            return "unknown --kv-quant tier \(tier) (known: \(knownKVQuantTiers), kvtuner-* with a frozen schedule)"
        case .missingMatrixID:
            return "KVTuner bench requires --matrix-id <ID>"
        case .missingWorkloadNonce:
            return "KVTuner bench requires --workload-nonce <ID> so every KV cell runs identical prompt bytes"
        case .tierCellMismatch(let tier, let cell):
            return "bench requires --kv-quant and --cell-id to identify the same executed tier; got \(tier) and \(cell)"
        case .unauditedKVTunerPrompt:
            return "KVTuner bench accepts only the audited built-in prompt; custom --prompt needs a separately authenticated source identity"
        case .invalidPromptRepeat(let value):
            return "bench --prompt-repeat must be between 1 and 4096; actual=\(value)"
        case .contextLimitExceeded(
            let promptTokens, let maxTokens, let limit):
            return "bench prompt plus output budget exceeds the authenticated model context: prompt=\(promptTokens), max_tokens=\(maxTokens), limit=\(limit)"
        case .invalidRuns(let runs):
            return "bench --runs must be positive and bounded; actual=\(runs)"
        case .invalidQualificationRuns(let runs):
            return "qualification bench requires exactly one post-warmup run per isolated matrix position; actual=\(runs)"
        case .qualificationFlagsWithoutQualification:
            return "qualification-only bench flags require --qualification-evidence true"
        case .missingQualificationFlag(let flag):
            return "qualification bench requires --\(flag) <VALUE>"
        case .invalidPostWarmupThermalTarget(let value):
            return "qualification bench --post-warmup-thermal-target must be nominal; actual=\(value)"
        case .invalidPostWarmupThermalPolicy:
            return "qualification bench post-warmup thermal timeout/poll policy is invalid"
        case .invalidMaxTokens(let value):
            return "bench --max-tokens must be positive; actual=\(value)"
        case .unknownSpec(let spec):
            return "unknown --spec drafter \(spec) (known: pld)"
        case .unmeasuredSpecKV(let tier):
            return "specDecode=pld with kvQuant=\(tier) is an unmeasured combination; use fp16"
        case .unknownAttentionOperation(let operation):
            return "unknown --kv-attention operation \(operation) (known: materialize, split-affine-quantized-mm, split-kvarn-quantized-mm)"
        case .unsupportedAttentionTier(let tier):
            return "--kv-attention is valid only for a matching affine-, KVTuner-, or KVarN-backed KV tier, not \(tier)"
        case .invalidCheckpointContentSHA256(let value):
            return "--checkpoint-content-sha256 must be one lowercase 64-character digest exactly when --kv-attention is requested; actual=\(value)"
        case .missingPrefillTiming:
            return "engine returned no direct prefill timing"
        case .invalidPrefillTiming:
            return "engine returned an invalid direct prefill timing"
        case .missingKVTunerEngagement:
            return "KVTuner schedule did not produce matching runtime engagement"
        case .missingLossyEngagement(let tier):
            return "bench tier \(tier) did not produce non-vacuous runtime engagement"
        case .missingCompressedAttentionEvidence:
            return "bench compressed-attention request did not produce one matching observed-operation receipt"
        case .compressedAttentionWorkspaceExceedsMLXPeak(
            let workspaceBytes, let peakBytes):
            return "bench compressed-attention logical workspace exceeds the raw MLX peak receipt: workspace=\(workspaceBytes), peak=\(peakBytes)"
        case .csvSchemaMismatch:
            return "existing bench CSV header does not match the direct-prefill schema"
        case .outputPathCollision(let path):
            return "bench inputs and outputs must name distinct files: \(path)"
        case .symbolicLinkOutput(let path):
            return "bench output symbolic links are unsupported: \(path)"
        }
    }
}

enum BenchQualificationEvidenceValidationError:
    Error, Equatable, Sendable
{
    case wrongSubcommand(String)
    case missingQualificationEvidence
    case modelIdentityMismatch
    case unexpectedKVTunerSchedule
    case missingKVTunerSchedule
    case missingKVTunerAdmission
    case tokenizerIdentityMismatch
    case kvtunerIdentityMismatch
}

/// Decode qualification evidence through the same durable Swift types that produced it, then
/// re-run the post-decode admission invariants and cross-bind the KVTuner schedule to the exact
/// model admission. The matrix runner performs additional experiment-specific scalar checks;
/// this typed pass prevents a truncated nested object from satisfying those checks by accident.
func validateBenchQualificationEvidenceData(_ data: Data) throws {
    let record = try JSONDecoder().decode(
        ResultRecord<BenchPayload>.self, from: data)
    guard record.subcommand == "bench" else {
        throw BenchQualificationEvidenceValidationError.wrongSubcommand(
            record.subcommand)
    }
    guard let qualification = record.payload.qualification else {
        throw BenchQualificationEvidenceValidationError
            .missingQualificationEvidence
    }

    let runtimeBinding = record.payload.compressedKVAttention
    if let runtimeBinding {
        try runtimeBinding.validated()
        guard runtimeBinding.admission.modelConfigHash
                == record.provenance.modelConfigHash,
            runtimeBinding.admission.checkpointManifestHash
                == record.provenance.modelCheckpointManifestHash
        else {
            throw BenchQualificationEvidenceValidationError
                .modelIdentityMismatch
        }
        guard runtimeBinding.admission.tokenizerSHA256
                == qualification.context.tokenizerSHA256
        else {
            throw BenchQualificationEvidenceValidationError
                .tokenizerIdentityMismatch
        }
    }

    let isKVTuner = isKVTunerTier(record.payload.kvQuantTier)
    guard isKVTuner || record.payload.kvtunerSchedule == nil else {
        throw BenchQualificationEvidenceValidationError
            .unexpectedKVTunerSchedule
    }
    guard !isKVTuner || record.payload.kvtunerSchedule != nil else {
        throw BenchQualificationEvidenceValidationError
            .missingKVTunerSchedule
    }
    if KVarNKVRuntimeCell(rawValue: record.payload.kvQuantTier) != nil,
        let runtimeBinding
    {
        guard let engagementMax = record.payload.engagementMax,
            let expectedStorageDType =
                runtimeBinding.admission.modelNativeDType
        else {
            throw BenchCLIError.missingLossyEngagement(
                record.payload.kvQuantTier)
        }
        try validateBenchRuntimeEngagement(
            tier: record.payload.kvQuantTier,
            engagement: EngagementCounters(engagementMax),
            expectedKVTunerLayerCount: nil,
            requestedCompressedKVAttention: runtimeBinding.request,
            expectedKVarNStorageDType: expectedStorageDType)
    }
    guard let schedule = record.payload.kvtunerSchedule else { return }
    guard let runtimeBinding else {
        throw BenchQualificationEvidenceValidationError
            .missingKVTunerAdmission
    }
    guard schedule.matrixID == record.payload.matrixID,
        schedule.cellID == record.payload.cellID,
        schedule.modelConfigHash == runtimeBinding.admission.modelConfigHash,
        schedule.modelConfigSHA256
            == runtimeBinding.admission.modelConfigSHA256,
        schedule.checkpointManifestHash
            == runtimeBinding.admission.checkpointManifestHash,
        schedule.checkpointContentSHA256
            == runtimeBinding.admission.checkpointContentSHA256,
        schedule.tokenizerSHA256 == runtimeBinding.admission.tokenizerSHA256,
        schedule.layers.count == runtimeBinding.admission.layerCount
    else {
        throw BenchQualificationEvidenceValidationError
            .kvtunerIdentityMismatch
    }
    try runtimeBinding.admission.validateScheduleIdentity(
        modelConfigHash: schedule.modelConfigHash,
        modelConfigSHA256: schedule.modelConfigSHA256,
        checkpointManifestHash: schedule.checkpointManifestHash,
        checkpointContentSHA256: schedule.checkpointContentSHA256,
        tokenizerSHA256: schedule.tokenizerSHA256,
        layerCount: schedule.layers.count,
        groupSize: schedule.groupSize)
}

struct BenchPlan: Sendable {
    let modelPath: String
    let kvQuantTier: String?
    let cellID: String
    let matrixID: String?
    let kvtunerSchedulePath: String
    let workload: BenchWorkloadIdentity
    let promptRepeat: Int
    let maxTokens: Int
    let runs: Int
    let qualificationContext: BenchQualificationContext?
    let label: String
    let spec: String?
    let ngram: Int
    let maxDraft: Int
    let compiledVerify: Bool
    let compressedKVAttention: CompressedKVAttentionRequest?
    let compressedKVAttentionExpectedCheckpointContentSHA256: String?
    let evidencePath: String
    let csvPath: String?
}

func requestedCompressedKVAttention(
    _ flags: Flags,
    tier: String
) throws -> CompressedKVAttentionRequest? {
    let attentionText = try flags.strictString(
        "kv-attention", default: "")
    guard !attentionText.isEmpty else { return nil }
    guard let request = CompressedKVAttentionRequest(
        rawValue: attentionText)
    else {
        throw BenchCLIError.unknownAttentionOperation(attentionText)
    }
    if isKVTunerTier(tier) {
        guard request != .splitKVarNQuantizedMM else {
            throw BenchCLIError.unsupportedAttentionTier(tier)
        }
        return request
    }
    if AffineKVTier(rawValue: tier) != nil {
        guard request != .splitKVarNQuantizedMM else {
            throw BenchCLIError.unsupportedAttentionTier(tier)
        }
        return request
    }
    if KVarNKVRuntimeCell(rawValue: tier) != nil {
        guard request != .splitAffineQuantizedMM else {
            throw BenchCLIError.unsupportedAttentionTier(tier)
        }
        return request
    }
    throw BenchCLIError.unsupportedAttentionTier(tier)
}

func requestedCompressedKVAttentionExpectedCheckpointContentSHA256(
    _ flags: Flags,
    request: CompressedKVAttentionRequest?
) throws -> String? {
    let value = try flags.strictString(
        "checkpoint-content-sha256", default: "")
    guard request != nil else {
        guard value.isEmpty else {
            throw BenchCLIError.invalidCheckpointContentSHA256(value)
        }
        return nil
    }
    guard value.utf8.count == 64,
        value.utf8.allSatisfy({
            (48 ... 57).contains($0) || (97 ... 102).contains($0)
        })
    else {
        throw BenchCLIError.invalidCheckpointContentSHA256(value)
    }
    return value
}

func parseBenchPlan(_ flags: Flags) throws -> BenchPlan {
    guard let modelPath = flags.string("model"), !modelPath.isEmpty else {
        throw BenchCLIError.missingModel
    }
    let kvQuantTier = try requestedKVQuantTier(flags)
    let runtimeTier = kvQuantTier ?? "fp16"
    let kvtuner = isKVTunerTier(runtimeTier)
    guard kvtuner || KVCacheKind(kvQuant: kvQuantTier) != nil else {
        throw BenchCLIError.unknownKVQuantTier(runtimeTier)
    }

    let cellID = try flags.strictString("cell-id", default: runtimeTier)
    let schedulePath = try flags.strictString(
        "kvtuner-schedule", default: "")
    try validateKVTunerScheduleFlag(
        tier: runtimeTier,
        cellID: cellID,
        schedulePath: schedulePath)
    if !kvtuner, cellID != runtimeTier {
        throw BenchCLIError.tierCellMismatch(
            tier: runtimeTier, cell: cellID)
    }

    let matrixText = try flags.strictString("matrix-id", default: "")
    if kvtuner, matrixText.isEmpty {
        throw BenchCLIError.missingMatrixID
    }
    let explicitNonce = try flags.strictString(
        "workload-nonce", default: "")
    if kvtuner, explicitNonce.isEmpty {
        throw BenchCLIError.missingWorkloadNonce
    }

    let evidencePath = try flags.strictString(
        "evidence", default: "harness-evidence.jsonl")
    let csvPath = try flags.strictString("csv", default: "")
    for outputPath in [evidencePath, csvPath] where !outputPath.isEmpty {
        guard !outputPathIsSymbolicLink(outputPath) else {
            throw BenchCLIError.symbolicLinkOutput(outputPath)
        }
    }
    let configuredPaths = [schedulePath, evidencePath, csvPath]
        .filter { !$0.isEmpty }
    for leftIndex in configuredPaths.indices {
        for rightIndex in configuredPaths.indices where rightIndex > leftIndex {
            guard !outputPathsReferToSameFile(
                configuredPaths[leftIndex], configuredPaths[rightIndex])
            else {
                throw BenchCLIError.outputPathCollision(
                    configuredPaths[rightIndex])
            }
        }
    }

    let promptTemplate = try flags.strictString(
        "prompt", default: defaultBenchPrompt)
    if kvtuner, promptTemplate != defaultBenchPrompt {
        throw BenchCLIError.unauditedKVTunerPrompt
    }
    let promptRepeat = try flags.strictInt("prompt-repeat", default: 1)
    guard (1 ... 4_096).contains(promptRepeat) else {
        throw BenchCLIError.invalidPromptRepeat(promptRepeat)
    }
    let prompt = Array(repeating: promptTemplate, count: promptRepeat)
        .joined(separator: "\n")
    let runs = try flags.strictInt("runs", default: 3)
    guard (1 ... 100).contains(runs) else {
        throw BenchCLIError.invalidRuns(runs)
    }
    let qualificationEvidence = try flags.strictBool(
        "qualification-evidence", default: false)
    let runnerManifestSHA256 = try flags.strictString(
        "runner-manifest-sha256", default: "")
    let matrixBlockIndex = try flags.optionalStrictInt(
        "matrix-block-index")
    let matrixRunPosition = try flags.optionalStrictInt(
        "matrix-run-position")
    let matrixCellCount = try flags.optionalStrictInt(
        "matrix-cell-count")
    let memoryLimitBytes = try flags.optionalStrictInt(
        "memory-limit-bytes")
    let cacheLimitBytes = try flags.optionalStrictInt(
        "cache-limit-bytes")
    let wiredLimitBytes = try flags.optionalStrictInt(
        "wired-limit-bytes")
    let modelTokenizerSHA256 = try flags.strictString(
        "model-tokenizer-sha256", default: "")
    let postWarmupThermalTargetText = try flags.strictString(
        "post-warmup-thermal-target", default: "")
    let postWarmupThermalTimeoutSeconds = try flags.optionalStrictInt(
        "post-warmup-thermal-timeout-seconds")
    let postWarmupThermalPollMilliseconds = try flags.optionalStrictInt(
        "post-warmup-thermal-poll-milliseconds")
    let postWarmupThermalStabilitySeconds = try flags.optionalStrictInt(
        "post-warmup-thermal-stability-seconds")
    let qualificationFlagsSupplied = !runnerManifestSHA256.isEmpty
        || matrixBlockIndex != nil || matrixRunPosition != nil
        || matrixCellCount != nil || memoryLimitBytes != nil
        || cacheLimitBytes != nil || wiredLimitBytes != nil
        || !modelTokenizerSHA256.isEmpty
        || !postWarmupThermalTargetText.isEmpty
        || postWarmupThermalTimeoutSeconds != nil
        || postWarmupThermalPollMilliseconds != nil
        || postWarmupThermalStabilitySeconds != nil
    guard qualificationEvidence || !qualificationFlagsSupplied else {
        throw BenchCLIError.qualificationFlagsWithoutQualification
    }
    let qualificationContext: BenchQualificationContext?
    if qualificationEvidence {
        guard !matrixText.isEmpty else {
            throw BenchCLIError.missingMatrixID
        }
        guard !explicitNonce.isEmpty else {
            throw BenchCLIError.missingWorkloadNonce
        }
        guard runs == 1 else {
            throw BenchCLIError.invalidQualificationRuns(runs)
        }
        func required(
            _ value: Int?, flag: String
        ) throws -> Int {
            guard let value else {
                throw BenchCLIError.missingQualificationFlag(flag)
            }
            return value
        }
        guard !runnerManifestSHA256.isEmpty else {
            throw BenchCLIError.missingQualificationFlag(
                "runner-manifest-sha256")
        }
        guard !modelTokenizerSHA256.isEmpty else {
            throw BenchCLIError.missingQualificationFlag(
                "model-tokenizer-sha256")
        }
        guard !postWarmupThermalTargetText.isEmpty else {
            throw BenchCLIError.missingQualificationFlag(
                "post-warmup-thermal-target")
        }
        guard let postWarmupThermalTarget =
            BenchQualificationThermalTarget(
                rawValue: postWarmupThermalTargetText)
        else {
            throw BenchCLIError.invalidPostWarmupThermalTarget(
                postWarmupThermalTargetText)
        }
        let stabilitySeconds = try required(
            postWarmupThermalStabilitySeconds,
            flag: "post-warmup-thermal-stability-seconds")
        guard stabilitySeconds > 0 else {
            throw BenchCLIError.invalidPostWarmupThermalPolicy
        }
        let postWarmupThermalPolicy: BenchQualificationThermalPolicy
        do {
            postWarmupThermalPolicy = try BenchQualificationThermalPolicy(
                target: postWarmupThermalTarget,
                timeoutSeconds: try required(
                    postWarmupThermalTimeoutSeconds,
                    flag: "post-warmup-thermal-timeout-seconds"),
                pollIntervalMilliseconds: try required(
                    postWarmupThermalPollMilliseconds,
                    flag: "post-warmup-thermal-poll-milliseconds"),
                stabilitySeconds: stabilitySeconds)
        } catch let error as BenchCLIError {
            throw error
        } catch {
            throw BenchCLIError.invalidPostWarmupThermalPolicy
        }
        qualificationContext = try BenchQualificationContext(
            runnerManifestSHA256: runnerManifestSHA256,
            matrixBlockIndex: try required(
                matrixBlockIndex, flag: "matrix-block-index"),
            matrixRunPosition: try required(
                matrixRunPosition, flag: "matrix-run-position"),
            matrixCellCount: try required(
                matrixCellCount, flag: "matrix-cell-count"),
            memoryLimitBytes: try required(
                memoryLimitBytes, flag: "memory-limit-bytes"),
            cacheLimitBytes: try required(
                cacheLimitBytes, flag: "cache-limit-bytes"),
            wiredLimitBytes: try required(
                wiredLimitBytes, flag: "wired-limit-bytes"),
            tokenizerSHA256: modelTokenizerSHA256,
            cacheResetPolicy: .inPlaceBeforeEveryGeneration,
            modelResidencyPolicy: .loadOncePerProcess,
            processIsolationPolicy: .freshProcessPerMatrixPosition,
            postWarmupThermalPolicy: postWarmupThermalPolicy)
    } else {
        qualificationContext = nil
    }
    let maxTokens = try flags.strictInt("max-tokens", default: 256)
    guard maxTokens > 0 else {
        throw BenchCLIError.invalidMaxTokens(maxTokens)
    }
    let nonce = explicitNonce.isEmpty
        ? String(Int.random(in: 0..<1_000_000))
        : explicitNonce
    let workload = try BenchWorkloadIdentity(
        basePrompt: prompt,
        nonce: nonce,
        iterations: runs + 1)

    let requestedSpec = try flags.strictString("spec", default: "")
    let spec: String? = requestedSpec.isEmpty ? nil : requestedSpec
    if let spec, spec != "pld" {
        throw BenchCLIError.unknownSpec(spec)
    }
    if spec != nil, kvQuantTier != nil {
        throw BenchCLIError.unmeasuredSpecKV(runtimeTier)
    }

    let compressedKVAttention = try requestedCompressedKVAttention(
        flags, tier: runtimeTier)
    let compressedKVAttentionExpectedCheckpointContentSHA256 = try
        requestedCompressedKVAttentionExpectedCheckpointContentSHA256(
            flags, request: compressedKVAttention)

    return BenchPlan(
        modelPath: modelPath,
        kvQuantTier: kvQuantTier,
        cellID: cellID,
        matrixID: matrixText.isEmpty ? nil : matrixText,
        kvtunerSchedulePath: schedulePath,
        workload: workload,
        promptRepeat: promptRepeat,
        maxTokens: maxTokens,
        runs: runs,
        qualificationContext: qualificationContext,
        label: try flags.strictString("label", default: "harness"),
        spec: spec,
        ngram: try flags.strictInt("ngram", default: 3),
        maxDraft: try flags.strictInt("max-draft", default: 8),
        compiledVerify: try flags.strictBool(
            "compiled-verify", default: false),
        compressedKVAttention: compressedKVAttention,
        compressedKVAttentionExpectedCheckpointContentSHA256:
            compressedKVAttentionExpectedCheckpointContentSHA256,
        evidencePath: evidencePath,
        csvPath: csvPath.isEmpty ? nil : csvPath)
}

/// Validate every exact salted workload before the first model forward. This keeps an
/// over-context long-prompt calibration from allocating or mutating a cache and then leaving a
/// partial evidence file. `addingReportingOverflow` also makes adversarial CLI integer input fail
/// closed instead of wrapping into an apparently admissible request.
func validateBenchContextWindow(
    promptTokenCounts: [Int],
    maxTokens: Int,
    maxPositionEmbeddings: Int
) throws {
    for promptTokens in promptTokenCounts {
        let (requested, overflow) = promptTokens.addingReportingOverflow(
            maxTokens)
        guard promptTokens > 0, maxTokens > 0,
            maxPositionEmbeddings > 0, !overflow,
            requested <= maxPositionEmbeddings
        else {
            throw BenchCLIError.contextLimitExceeded(
                promptTokens: promptTokens,
                maxTokens: maxTokens,
                limit: maxPositionEmbeddings)
        }
    }
}

/// Append one runtime row without treating a failed read as an empty destination. A pre-existing
/// file must have the exact current schema; output symlinks remain unsupported at the write seam
/// as defense in depth against a path changing after plan validation.
func appendBenchCSVRow(_ row: BenchRow, to path: String) throws {
    guard !outputPathIsSymbolicLink(path) else {
        throw BenchCLIError.symbolicLinkOutput(path)
    }
    let url = URL(fileURLWithPath: path)
    let existing: String
    if FileManager.default.fileExists(atPath: url.path) {
        existing = try String(contentsOf: url, encoding: .utf8)
    } else {
        existing = ""
    }
    if !existing.isEmpty,
        existing.split(separator: "\n", omittingEmptySubsequences: false)
            .first.map(String.init) != BenchRow.csvHeader
    {
        throw BenchCLIError.csvSchemaMismatch
    }
    var content = existing
    if content.isEmpty {
        content = BenchRow.csvHeader + "\n"
    } else if !content.hasSuffix("\n") {
        content += "\n"
    }
    content += row.csvLine + "\n"
    try content.write(to: url, atomically: true, encoding: .utf8)
}

/// A lossy runtime row is admissible only when the selected cache actually stored compressed
/// tokens. KVarN additionally must cross a completed tile boundary; otherwise a short request
/// could benchmark only its fp16 sink/tail and falsely attribute that rate to compressed KV.
func validateBenchRuntimeEngagement(
    tier: String,
    engagement: EngagementCounters,
    expectedKVTunerLayerCount: Int?,
    requestedCompressedKVAttention: CompressedKVAttentionRequest? = nil,
    expectedKVarNStorageDType: CompressedKVModelNativeDType? = nil
) throws {
    let counts = engagement.counts
    func hasStorageReceipt(
        _ prefix: String,
        attentionRequest: CompressedKVAttentionRequest? = nil
    ) -> Bool {
        guard let cachedTokens = counts["\(prefix)_tokens"],
            let capacityTokens = counts["\(prefix)_capacity_tokens"]
        else { return false }
        let baseReceipt = cachedTokens > 0
            && capacityTokens >= cachedTokens
            && (counts["\(prefix)_payload_bytes"] ?? 0) > 0
            && (counts["\(prefix)_metadata_bytes"] ?? 0) > 0
            && (counts["\(prefix)_control_bytes"] ?? 0) > 0
        guard baseReceipt else { return false }
        let workspace = counts["\(prefix)_workspace_bytes"] ?? -1
        let materialization =
            counts["\(prefix)_materialization_bytes"] ?? -1
        let normalization =
            counts["\(prefix)_normalization_workspace_bytes"] ?? 0
        let attentionWorkspace =
            counts["\(prefix)_attention_workspace_bytes"] ?? -1
        guard normalization >= 0 else { return false }
        switch attentionRequest {
        case nil:
            // Historical/default rows are materialize-then-attend and retain the old receipt.
            return workspace > 0
        case .materialize:
            return workspace > 0
                && materialization > 0
                && attentionWorkspace == 0
                && workspace == materialization + normalization
                && counts["\(prefix)_attention_materialized"] == 1
                && counts["\(prefix)_attention_split"] == 0
        case .splitAffineQuantizedMM, .splitKVarNQuantizedMM:
            return workspace > 0
                && materialization == 0
                && attentionWorkspace > 0
                && workspace == attentionWorkspace + normalization
                && counts["\(prefix)_attention_split"] == 1
                && counts["\(prefix)_attention_materialized"] == 0
        }
    }
    if tier == "fp16" { return }
    if isKVTunerTier(tier) {
        guard requestedCompressedKVAttention != .splitKVarNQuantizedMM else {
            throw BenchCLIError.missingKVTunerEngagement
        }
        guard let expectedKVTunerLayerCount,
            hasStorageReceipt(
                "kvtuner",
                attentionRequest: requestedCompressedKVAttention),
            counts["kvtuner_layers"] == expectedKVTunerLayerCount
        else { throw BenchCLIError.missingKVTunerEngagement }
        return
    }
    if AffineKVTier(rawValue: tier) != nil {
        guard requestedCompressedKVAttention != .splitKVarNQuantizedMM else {
            throw BenchCLIError.missingLossyEngagement(tier)
        }
        guard hasStorageReceipt(
            "affine",
            attentionRequest: requestedCompressedKVAttention),
            (counts["affine_layers"] ?? 0) > 0
        else {
            throw BenchCLIError.missingLossyEngagement(tier)
        }
        return
    }
    if let cell = KVarNKVRuntimeCell(rawValue: tier) {
        guard requestedCompressedKVAttention != .splitAffineQuantizedMM,
            hasStorageReceipt(
                "kvarn",
                attentionRequest: requestedCompressedKVAttention),
            (counts["kvarn_completed_tiles"] ?? 0) > 0,
            (counts["kvarn_compressed_tokens"] ?? 0) > 0,
            (counts["kvarn_layers"] ?? 0) > 0,
            counts["kvarn_codec_iterations"] == cell.iterations
        else { throw BenchCLIError.missingLossyEngagement(tier) }
        if let expectedKVarNStorageDType {
            func dtypeSet(_ role: String, oneHot: Bool)
                -> Set<CompressedKVModelNativeDType>?
            {
                let dtypes: [CompressedKVModelNativeDType] = [
                    .float16, .bfloat16, .float32,
                ]
                let markers = dtypes.map {
                    counts["kvarn_\(role)_\($0.rawValue)"]
                }
                guard markers.allSatisfy({ $0 == 0 || $0 == 1 }),
                    markers.compactMap({ $0 }).count == dtypes.count
                else { return nil }
                let selected = Set(zip(dtypes, markers).compactMap {
                    dtype, marker in marker == 1 ? dtype : nil
                })
                guard !selected.isEmpty, !oneHot || selected.count == 1 else {
                    return nil
                }
                return selected
            }
            let sourceKey = dtypeSet("source_key", oneHot: false)
            let sourceValue = dtypeSet("source_value", oneHot: false)
            let storageKey = dtypeSet("storage_key", oneHot: true)
            let storageValue = dtypeSet("storage_value", oneHot: true)
            let expectedStorage = Set([expectedKVarNStorageDType])
            let normalized = counts["kvarn_ingress_normalized"]
            let normalizationWorkspace =
                counts["kvarn_normalization_workspace_bytes"]
            guard expectedKVarNStorageDType != .float32,
                let sourceKey,
                sourceKey == sourceValue,
                sourceKey.allSatisfy({
                    $0 == expectedKVarNStorageDType || $0 == .float32
                }),
                storageKey == expectedStorage,
                storageValue == expectedStorage,
                normalized == (sourceKey == expectedStorage ? 0 : 1),
                normalizationWorkspace != nil,
                (normalized == 1
                    ? normalizationWorkspace! > 0
                    : normalizationWorkspace == 0)
            else { throw BenchCLIError.missingLossyEngagement(tier) }
        }
        return
    }
    if TurboQuantTier.allCases.contains(where: {
        $0.harnessSlot == tier || $0.rawValue == tier
    }) {
        guard (counts["turboquant_tokens"] ?? 0) > 0 else {
            throw BenchCLIError.missingLossyEngagement(tier)
        }
    }
}

/// Reconcile the route-specific logical workspace with the independent raw allocator receipt.
/// The raw peak also includes model/persistent allocations, so it is an upper envelope rather
/// than an equality claim; a smaller peak proves that at least one receipt is false or partial.
func validateBenchCompressedAttentionMemoryReceipt(
    tier: String,
    request: CompressedKVAttentionRequest?,
    engagement: EngagementCounters,
    maxMLXPeakBytes: Int
) throws {
    guard request != nil else { return }
    let prefix: String
    if isKVTunerTier(tier) {
        prefix = "kvtuner"
    } else if AffineKVTier(rawValue: tier) != nil {
        prefix = "affine"
    } else if KVarNKVRuntimeCell(rawValue: tier) != nil {
        prefix = "kvarn"
    } else {
        throw BenchCLIError.unsupportedAttentionTier(tier)
    }
    guard let workspaceBytes =
            engagement.counts["\(prefix)_workspace_bytes"],
        workspaceBytes > 0,
        maxMLXPeakBytes >= workspaceBytes
    else {
        throw BenchCLIError.compressedAttentionWorkspaceExceedsMLXPeak(
            workspaceBytes:
                engagement.counts["\(prefix)_workspace_bytes"] ?? -1,
            peakBytes: maxMLXPeakBytes)
    }
}

/// Convert post-run counters into a durable binding. The requested mode is only an expectation;
/// the observed operation comes exclusively from cache telemetry aggregated after execution.
func makeBenchCompressedKVAttentionRuntimeBinding(
    tier: String,
    request: CompressedKVAttentionRequest?,
    admission: CompressedKVAttentionRuntimeAdmission?,
    engagement: EngagementCounters
) throws -> CompressedKVAttentionRuntimeBinding? {
    guard let request else { return nil }
    guard let admission else {
        throw BenchCLIError.missingCompressedAttentionEvidence
    }
    let prefix: String
    if isKVTunerTier(tier) {
        prefix = "kvtuner"
    } else if AffineKVTier(rawValue: tier) != nil {
        prefix = "affine"
    } else if KVarNKVRuntimeCell(rawValue: tier) != nil {
        prefix = "kvarn"
    } else {
        throw BenchCLIError.unsupportedAttentionTier(tier)
    }
    let counts = engagement.counts
    let observed: CompressedKVAttentionObservedOperation
    switch (
        counts["\(prefix)_attention_split"],
        counts["\(prefix)_attention_materialized"]
    ) {
    case (1, 0):
        observed = prefix == "kvarn"
            ? .splitKVarNQuantizedMM
            : .splitQuantizedMM
    case (0, 1):
        observed = .materializedKV
    default:
        throw BenchCLIError.missingCompressedAttentionEvidence
    }
    do {
        return try CompressedKVAttentionRuntimeBinding(
            request: request,
            observedOperation: observed,
            admission: admission)
    } catch {
        throw BenchCLIError.missingCompressedAttentionEvidence
    }
}
