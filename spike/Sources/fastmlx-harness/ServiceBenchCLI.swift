import Foundation
import HarnessCore

private enum ServiceBenchCLIError: Error, CustomStringConvertible {
    case invalidArguments(String)
    case unsupportedPolicy(String)
    case unsupportedScenario(String)
    case missingSharedBatch
    case promptAccounting(expectedChunks: Int, actualChunks: Int, expectedTokens: Int, actualTokens: Int)
    case activeSlotHighWatermark(expected: Int, actual: Int)
    case batchedSpeculationEngaged

    var description: String {
        switch self {
        case .invalidArguments(let message): return message
        case .unsupportedPolicy(let policy): return "unsupported service policy \(policy)"
        case .unsupportedScenario(let scenario): return "unsupported service scenario \(scenario)"
        case .missingSharedBatch: return "concurrent run never executed a shared decode batch"
        case .promptAccounting(let expectedChunks, let actualChunks, let expectedTokens, let actualTokens):
            return "prompt accounting mismatch: chunks \(actualChunks)/\(expectedChunks), tokens \(actualTokens)/\(expectedTokens)"
        case .activeSlotHighWatermark(let expected, let actual):
            return "active-slot high-watermark \(actual), expected \(expected)"
        case .batchedSpeculationEngaged:
            return "continuous batch unexpectedly engaged speculation"
        }
    }
}

private struct ServiceBenchRunEvidence: Codable, Sendable {
    let run: Int
    let droppedWarmup: Bool
    let metrics: ServiceRunMetrics
    let operations: ServiceOperationSummary
    let memory: ServiceMemorySummary
}

private struct ServiceBenchPayload: Codable, Sendable {
    let label: String
    let policy: String
    let scenario: String
    let model: String
    let modelQuantization: String
    let checkpointManifestHash: String
    let checkpointIdentityKind: String
    let mlxSwiftLMRevision: String
    let compilePolicy: String
    let concurrency: Int
    let maxActiveSlots: Int
    let maxPrefillSlots: Int
    let prefillChunkSize: Int
    let maxOutputTokens: Int
    let memoryCacheLimitBytes: Int
    let aggregate: ServiceRunAggregate
    let warmup: ServiceBenchRunEvidence
    let measuredRuns: [ServiceBenchRunEvidence]
}

private struct ServiceBenchCSVRow {
    static let header = [
        "label", "policy", "scenario", "model", "quant", "concurrency",
        "aggregate_tok_s", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms",
        "tpot_p95_ms", "jain_mean", "jain_min", "hardware",
    ].joined(separator: ",")

    let fields: [String]
    var line: String { fields.joined(separator: ",") }
}

func runServiceBench(_ flags: Flags) async {
    guard let modelPath = flags.string("model") else {
        print(
            "usage: fastmlx-harness service-bench --model <PATH> --policy batch-no-spec --scenario burst --concurrency <1|2|4|8> [--max-tokens 128] [--runs 3] [--prefill-chunk 16] [--max-prefill N] [--evidence FILE]")
        exit(2)
    }

    do {
        try assertReleaseBuild()
        let policy = flags.string("policy", default: "batch-no-spec")
        guard policy == "batch-no-spec" else {
            throw ServiceBenchCLIError.unsupportedPolicy(policy)
        }
        let scenario = flags.string("scenario", default: "burst")
        guard scenario == "burst" else {
            throw ServiceBenchCLIError.unsupportedScenario(scenario)
        }
        let concurrency = try flags.strictInt("concurrency", default: 1)
        guard [1, 2, 4, 8].contains(concurrency) else {
            throw ServiceBenchCLIError.invalidArguments(
                "--concurrency must be one of 1,2,4,8")
        }
        let runs = try flags.strictInt("runs", default: 3)
        let maxTokens = try flags.strictInt("max-tokens", default: 128)
        let prefillChunk = try flags.strictInt("prefill-chunk", default: 16)
        let maxPrefill = try flags.strictInt("max-prefill", default: concurrency)
        guard runs > 0, maxTokens > 0, prefillChunk > 0,
            maxPrefill > 0, maxPrefill <= concurrency
        else {
            throw ServiceBenchCLIError.invalidArguments(
                "--runs/--max-tokens/--prefill-chunk must be positive and --max-prefill must be in 1...concurrency")
        }
        let maxReserved = try flags.optionalStrictInt("max-reserved-context-tokens")
        let traceLimit = max(1_024, concurrency * (maxTokens + 64) * 2)
        let loaded = try await loadContinuousSwiftServiceDriver(
            modelPath: modelPath,
            configuration: ContinuousServiceLoadConfiguration(
                maxActiveSlots: concurrency,
                maxPrefillSlots: maxPrefill,
                prefillChunkSize: prefillChunk,
                maxReservedContextTokens: maxReserved,
                traceLimit: traceLimit))

        let prompt = flags.string("prompt", default: benchPrompt)
        let label = flags.string("label", default: "continuous-service")
        let nonce = ProvenanceCLI.nonce()
        var evidenceRuns: [ServiceBenchRunEvidence] = []
        evidenceRuns.reserveCapacity(runs + 1)

        for run in 0 ... runs {
            let prompts = (0 ..< concurrency).map { request in
                loaded.tokenizer.encode(
                    text: "\(saltPrompt(run: run, nonce: nonce, prompt)) [request=\(request)]")
            }
            let observation = try await loaded.driver.runBurst(
                prompts: prompts,
                maxOutputTokens: maxTokens)
            let expectedChunks = prompts.reduce(0) {
                $0 + (($1.count + prefillChunk - 1) / prefillChunk)
            }
            let expectedPromptTokens = prompts.map(\.count).reduce(0, +)
            guard observation.operations.promptChunkCount == expectedChunks,
                observation.operations.promptTokensProcessed == expectedPromptTokens
            else {
                throw ServiceBenchCLIError.promptAccounting(
                    expectedChunks: expectedChunks,
                    actualChunks: observation.operations.promptChunkCount,
                    expectedTokens: expectedPromptTokens,
                    actualTokens: observation.operations.promptTokensProcessed)
            }
            guard observation.operations.maxActiveSlots == concurrency else {
                throw ServiceBenchCLIError.activeSlotHighWatermark(
                    expected: concurrency,
                    actual: observation.operations.maxActiveSlots)
            }
            if concurrency > 1,
                !observation.operations.decodeBatchSizeHistogram.keys.contains(where: { $0 > 1 })
            {
                throw ServiceBenchCLIError.missingSharedBatch
            }
            guard !observation.operations.speculationEngaged else {
                throw ServiceBenchCLIError.batchedSpeculationEngaged
            }
            let tag = run == 0 ? "warmup (dropped)" : "run \(run)"
            print(
                "# \(tag): aggregate=\(fmt(observation.metrics.aggregateTokensPerSecond, 2)) tok/s, "
                    + "ttft-p50=\(fmt(observation.metrics.ttft.p50Seconds * 1_000, 1))ms, "
                    + "ttft-p95=\(fmt(observation.metrics.ttft.p95Seconds * 1_000, 1))ms, "
                    + "jain=\(fmt(observation.metrics.jainCompletionRate, 4)), "
                    + "batch-sizes=\(observation.operations.decodeBatchSizeHistogram)")
            evidenceRuns.append(
                ServiceBenchRunEvidence(
                    run: run,
                    droppedWarmup: run == 0,
                    metrics: observation.metrics,
                    operations: observation.operations,
                    memory: observation.memory))
        }

        let measured = Array(evidenceRuns.dropFirst())
        let aggregate = try aggregateServiceRuns(measured.map(\.metrics))
        let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        let config = ProvenanceCLI.modelConfig(at: modelPath)
        let quantization = [
            config.quant.label,
            config.quant.groupSize.map { "group=\($0)" },
        ].compactMap { $0 }.joined(separator: ":")
        let row = ServiceBenchCSVRow(fields: [
            label, policy, scenario, modelName, quantization, String(concurrency),
            fmt(aggregate.meanAggregateTokensPerSecond, 2),
            fmt(aggregate.ttft.p50Seconds * 1_000, 1),
            fmt(aggregate.ttft.p95Seconds * 1_000, 1),
            aggregate.tpot.map { fmt($0.p50Seconds * 1_000, 1) } ?? "",
            aggregate.tpot.map { fmt($0.p95Seconds * 1_000, 1) } ?? "",
            fmt(aggregate.meanJainCompletionRate, 4),
            fmt(aggregate.minJainCompletionRate, 4),
            ProvenanceCLI.chipBrand(),
        ])
        print(ServiceBenchCSVRow.header)
        print(row.line)
        if let csvPath = flags.string("csv") {
            let url = URL(fileURLWithPath: csvPath)
            let existing = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
            var content = existing.isEmpty ? ServiceBenchCSVRow.header + "\n" : existing
            content += row.line + "\n"
            try content.write(to: url, atomically: true, encoding: .utf8)
            print("# appended to \(csvPath)")
        }

        let payload = ServiceBenchPayload(
            label: label,
            policy: policy,
            scenario: scenario,
            model: modelName,
            modelQuantization: quantization,
            checkpointManifestHash: try ProvenanceCLI.checkpointManifestHash(at: modelPath),
            checkpointIdentityKind: "config-index-shard-name-size-manifest",
            mlxSwiftLMRevision: ProvenanceCLI.mlxSwiftLMRevision,
            compilePolicy: "fixed-membership-fresh-burst-retrace",
            concurrency: concurrency,
            maxActiveSlots: concurrency,
            maxPrefillSlots: maxPrefill,
            prefillChunkSize: prefillChunk,
            maxOutputTokens: maxTokens,
            memoryCacheLimitBytes: 8 << 30,
            aggregate: aggregate,
            warmup: evidenceRuns[0],
            measuredRuns: measured)
        let provenance = ProvenanceCLI.build(
            modelPath: modelPath,
            referenceVersions: nil,
            corpus: nil).provenance
        try appendRequiredJSONLRecord(
            ResultRecord(
                subcommand: "service-bench",
                provenance: provenance,
                payload: payload),
            to: evidencePath(flags))
        print("# provenance: appended to \(evidencePath(flags))")
    } catch BenchGuardError.debugBuild {
        print("service-bench FAILED: Debug build — service numbers would be misleading")
        exit(1)
    } catch {
        print("service-bench FAILED: \(error)")
        exit(1)
    }
}
