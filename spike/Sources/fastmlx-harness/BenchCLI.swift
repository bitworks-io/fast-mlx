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
    case invalidRuns(Int)
    case invalidMaxTokens(Int)
    case unknownSpec(String)
    case unmeasuredSpecKV(String)
    case missingPrefillTiming
    case invalidPrefillTiming
    case missingKVTunerEngagement
    case missingLossyEngagement(String)
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
        case .invalidRuns(let runs):
            return "bench --runs must be positive and bounded; actual=\(runs)"
        case .invalidMaxTokens(let value):
            return "bench --max-tokens must be positive; actual=\(value)"
        case .unknownSpec(let spec):
            return "unknown --spec drafter \(spec) (known: pld)"
        case .unmeasuredSpecKV(let tier):
            return "specDecode=pld with kvQuant=\(tier) is an unmeasured combination; use fp16"
        case .missingPrefillTiming:
            return "engine returned no direct prefill timing"
        case .invalidPrefillTiming:
            return "engine returned an invalid direct prefill timing"
        case .missingKVTunerEngagement:
            return "KVTuner schedule did not produce matching runtime engagement"
        case .missingLossyEngagement(let tier):
            return "bench tier \(tier) did not produce non-vacuous runtime engagement"
        case .csvSchemaMismatch:
            return "existing bench CSV header does not match the direct-prefill schema"
        case .outputPathCollision(let path):
            return "bench inputs and outputs must name distinct files: \(path)"
        case .symbolicLinkOutput(let path):
            return "bench output symbolic links are unsupported: \(path)"
        }
    }
}

struct BenchPlan: Sendable {
    let modelPath: String
    let kvQuantTier: String?
    let cellID: String
    let matrixID: String?
    let kvtunerSchedulePath: String
    let workload: BenchWorkloadIdentity
    let maxTokens: Int
    let runs: Int
    let label: String
    let spec: String?
    let ngram: Int
    let maxDraft: Int
    let compiledVerify: Bool
    let evidencePath: String
    let csvPath: String?
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

    let prompt = try flags.strictString(
        "prompt", default: defaultBenchPrompt)
    if kvtuner, prompt != defaultBenchPrompt {
        throw BenchCLIError.unauditedKVTunerPrompt
    }
    let runs = try flags.strictInt("runs", default: 3)
    guard (1 ... 100).contains(runs) else {
        throw BenchCLIError.invalidRuns(runs)
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

    return BenchPlan(
        modelPath: modelPath,
        kvQuantTier: kvQuantTier,
        cellID: cellID,
        matrixID: matrixText.isEmpty ? nil : matrixText,
        kvtunerSchedulePath: schedulePath,
        workload: workload,
        maxTokens: maxTokens,
        runs: runs,
        label: try flags.strictString("label", default: "harness"),
        spec: spec,
        ngram: try flags.strictInt("ngram", default: 3),
        maxDraft: try flags.strictInt("max-draft", default: 8),
        compiledVerify: try flags.strictBool(
            "compiled-verify", default: false),
        evidencePath: evidencePath,
        csvPath: csvPath.isEmpty ? nil : csvPath)
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
    expectedKVTunerLayerCount: Int?
) throws {
    let counts = engagement.counts
    func hasStorageReceipt(_ prefix: String) -> Bool {
        guard let cachedTokens = counts["\(prefix)_tokens"],
            let capacityTokens = counts["\(prefix)_capacity_tokens"]
        else { return false }
        return cachedTokens > 0
            && capacityTokens >= cachedTokens
            && (counts["\(prefix)_payload_bytes"] ?? 0) > 0
            && (counts["\(prefix)_metadata_bytes"] ?? 0) > 0
            && (counts["\(prefix)_control_bytes"] ?? 0) > 0
            && (counts["\(prefix)_workspace_bytes"] ?? 0) > 0
    }
    if tier == "fp16" { return }
    if isKVTunerTier(tier) {
        guard let expectedKVTunerLayerCount,
            hasStorageReceipt("kvtuner"),
            counts["kvtuner_layers"] == expectedKVTunerLayerCount
        else { throw BenchCLIError.missingKVTunerEngagement }
        return
    }
    if AffineKVTier(rawValue: tier) != nil {
        guard hasStorageReceipt("affine"),
            (counts["affine_layers"] ?? 0) > 0
        else {
            throw BenchCLIError.missingLossyEngagement(tier)
        }
        return
    }
    if let cell = KVarNKVRuntimeCell(rawValue: tier) {
        guard hasStorageReceipt("kvarn"),
            (counts["kvarn_completed_tiles"] ?? 0) > 0,
            (counts["kvarn_compressed_tokens"] ?? 0) > 0,
            (counts["kvarn_layers"] ?? 0) > 0,
            counts["kvarn_codec_iterations"] == cell.iterations
        else { throw BenchCLIError.missingLossyEngagement(tier) }
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
