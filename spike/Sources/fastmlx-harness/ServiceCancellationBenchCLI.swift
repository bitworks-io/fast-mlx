import Foundation
import HarnessCore
import MLXLMCommon

private enum ServiceCancellationBenchError: Error, CustomStringConvertible {
    case invalidArguments(String)
    case structuralGateFailed(Int)
    case resourcesNotReleased(Int, Int)
    case reservationDroppedBeforeRebuild(Int)
    case keepaliveExceeded(Double)

    var description: String {
        switch self {
        case .invalidArguments(let message): return message
        case .structuralGateFailed(let run):
            return "cancellation recovery structural gate failed in run \(run)"
        case .resourcesNotReleased(let run, let bytes):
            return "run \(run) retained \(bytes) KV bytes after recovery"
        case .reservationDroppedBeforeRebuild(let run):
            return "run \(run) dropped a live old-batch reservation before membership rebuild"
        case .keepaliveExceeded(let seconds):
            return "disconnect-to-removal exceeded keepalive; max=\(seconds)s"
        }
    }
}

private struct ServiceCancellationRunEvidence: Codable, Sendable {
    let run: Int
    let droppedWarmup: Bool
    let requestedAt: Double
    let removedAt: Double
    let latencySeconds: Double
    let promptTokenCounts: [Int]
    let previousPhase: String
    let slotRemoved: Bool
    let cancellationEventObserved: Bool
    let repeatedCancellationNotFound: Bool
    let replacementSlotReused: Bool
    let sharedBatchObserved: Bool
    let decodeFirstInterleaveObserved: Bool
    let cancelledPrefixTokens: Int
    let survivorOutputTokens: Int
    let replacementOutputTokens: Int
    let operations: ServiceOperationSummary
    let memory: ServiceMemorySummary
    let resourcesAtAdmission: ContinuousBatchRuntimeResourceSnapshot
    let resourcesAfterCancellation: ContinuousBatchRuntimeResourceSnapshot
    let resourcesAfterReplacement: ContinuousBatchRuntimeResourceSnapshot
    let resourcesAtEnd: ContinuousBatchRuntimeResourceSnapshot
}

private struct ServiceCancellationBenchPayload: Codable, Sendable {
    let label: String
    let scenario: String
    let model: String
    let modelQuantization: String
    let checkpointManifestHash: String
    let checkpointIdentityKind: String
    let mlxSwiftLMRevision: String
    let maxActiveSlots: Int
    let maxPrefillSlots: Int
    let prefillChunkSize: Int
    let maxReservedContextTokens: Int?
    let maxReservedKVBytes: Int
    let maxOutputTokens: Int
    let longRepeat: Int
    let memoryCacheLimitBytes: Int
    let cancellationGate: ServiceCancellationGate
    let warmup: ServiceCancellationRunEvidence
    let measuredRuns: [ServiceCancellationRunEvidence]
}

func runServiceCancellationBench(_ flags: Flags) async {
    guard let modelPath = flags.string("model") else {
        print(
            "usage: fastmlx-harness service-cancel-bench --model <PATH> [--runs 5] [--max-tokens 64] [--prefill-chunk 16] [--keepalive-ms 1000] [--long-repeat 18] [--max-reserved-kv-bytes N] [--evidence FILE]")
        exit(2)
    }

    do {
        try assertReleaseBuild()
        let runs = try flags.strictInt("runs", default: 5)
        let maxTokens = try flags.strictInt("max-tokens", default: 64)
        let prefillChunk = try flags.strictInt("prefill-chunk", default: 16)
        let keepaliveMS = try flags.strictInt("keepalive-ms", default: 1_000)
        // Keep the survivor decisively longer than the repetitive cancellation target so the
        // scenario necessarily observes target decode interleaved with survivor prefill.
        let longRepeat = try flags.strictInt("long-repeat", default: 18)
        guard (1 ... 100).contains(runs),
            (2 ... 4_096).contains(maxTokens),
            (1 ... 4_096).contains(prefillChunk),
            (1 ... 60_000).contains(keepaliveMS),
            (1 ... 128).contains(longRepeat)
        else {
            throw ServiceCancellationBenchError.invalidArguments(
                "--runs must be 1...100, --max-tokens 2...4096, --prefill-chunk 1...4096, --keepalive-ms 1...60000, and --long-repeat 1...128")
        }
        let maxReservedContext = try flags.optionalStrictInt("max-reserved-context-tokens")
        if let maxReservedContext, maxReservedContext <= 0 {
            throw ServiceCancellationBenchError.invalidArguments(
                "--max-reserved-context-tokens must be positive")
        }
        let maxReservedKVBytes = try flags.optionalStrictInt("max-reserved-kv-bytes")
            ?? Int(min(ProvenanceCLI.ramBytes() / 4, UInt64(Int.max)))
        guard maxReservedKVBytes > 0 else {
            throw ServiceCancellationBenchError.invalidArguments(
                "--max-reserved-kv-bytes must be positive")
        }

        let loaded = try await loadContinuousSwiftServiceDriver(
            modelPath: modelPath,
            configuration: ContinuousServiceLoadConfiguration(
                maxActiveSlots: 2,
                maxPrefillSlots: 2,
                prefillChunkSize: prefillChunk,
                maxReservedContextTokens: maxReservedContext,
                maxReservedKVBytes: maxReservedKVBytes,
                traceLimit: max(2_048, (maxTokens + 128) * 6)))
        let tokenizer = loaded.tokenizer
        let driver = loaded.driver
        let nonce = ProvenanceCLI.nonce()
        let label = flags.string("label", default: "continuous-cancellation")
        var evidenceRuns: [ServiceCancellationRunEvidence] = []
        var timelines: [ServiceCancellationTimeline] = []

        for run in 0 ... runs {
            let salt = saltPrompt(
                run: run,
                nonce: nonce,
                "Continuous service cancellation recovery.")
            let longTail = String(
                repeating:
                    " Decode-first scheduling advances ready tokens while bounded prompt chunks continue.",
                count: longRepeat)
            let prompts = [
                tokenizer.encode(text: salt + longTail),
                tokenizer.encode(text: salt + " " + specVerifyPrompt),
                tokenizer.encode(
                    text: salt
                        + " Replacement request: enumerate stable row identity, bounded admission, and exact recovery."),
            ]
            let observation = try await driver.runCancellationRecovery(
                prompts: prompts,
                maxOutputTokens: maxTokens,
                cancellationWaitLimitSeconds: max(
                    1,
                    Double(keepaliveMS) / 500))
            let structuralPassed = observation.previousPhase == "decoding"
                && observation.slotRemoved
                && observation.cancellationEventObserved
                && observation.repeatedCancellationNotFound
                && observation.replacementSlotReused
                && observation.sharedBatchObserved
                && observation.decodeFirstInterleaveObserved
            guard structuralPassed else {
                throw ServiceCancellationBenchError.structuralGateFailed(run)
            }
            guard observation.resourcesAfterCancellation.reservedKVBytes
                == observation.resourcesAtAdmission.reservedKVBytes
            else {
                throw ServiceCancellationBenchError.reservationDroppedBeforeRebuild(run)
            }
            guard observation.resourcesAtEnd.reservedKVBytes == 0 else {
                throw ServiceCancellationBenchError.resourcesNotReleased(
                    run,
                    observation.resourcesAtEnd.reservedKVBytes)
            }

            let evidence = ServiceCancellationRunEvidence(
                run: run,
                droppedWarmup: run == 0,
                requestedAt: observation.timeline.requestedAt,
                removedAt: observation.timeline.removedAt,
                latencySeconds: observation.timeline.removedAt - observation.timeline.requestedAt,
                promptTokenCounts: prompts.map(\.count),
                previousPhase: observation.previousPhase,
                slotRemoved: observation.slotRemoved,
                cancellationEventObserved: observation.cancellationEventObserved,
                repeatedCancellationNotFound: observation.repeatedCancellationNotFound,
                replacementSlotReused: observation.replacementSlotReused,
                sharedBatchObserved: observation.sharedBatchObserved,
                decodeFirstInterleaveObserved: observation.decodeFirstInterleaveObserved,
                cancelledPrefixTokens: observation.cancelledPrefixTokens,
                survivorOutputTokens: observation.survivorOutputTokens,
                replacementOutputTokens: observation.replacementOutputTokens,
                operations: observation.operations,
                memory: observation.memory,
                resourcesAtAdmission: observation.resourcesAtAdmission,
                resourcesAfterCancellation: observation.resourcesAfterCancellation,
                resourcesAfterReplacement: observation.resourcesAfterReplacement,
                resourcesAtEnd: observation.resourcesAtEnd)
            evidenceRuns.append(evidence)
            if run > 0 { timelines.append(observation.timeline) }
            print(
                "# \(run == 0 ? "warmup (dropped)" : "run \(run)"): cancel=\(fmt(evidence.latencySeconds * 1_000, 3))ms, "
                    + "batch-sizes=\(observation.operations.decodeBatchSizeHistogram), "
                    + "kv=\(observation.resourcesAtAdmission.reservedKVBytes)→\(observation.resourcesAfterReplacement.reservedKVBytes)→0")
        }

        let gate = try evaluateCancellationGate(
            timelines,
            keepaliveSeconds: Double(keepaliveMS) / 1_000)
        let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        let config = ProvenanceCLI.modelConfig(at: modelPath)
        let quantization = [
            config.quant.label,
            config.quant.groupSize.map { "group=\($0)" },
        ].compactMap { $0 }.joined(separator: ":")
        let payload = ServiceCancellationBenchPayload(
            label: label,
            scenario: "decode-cancel-reuse-short-long",
            model: modelName,
            modelQuantization: quantization,
            checkpointManifestHash: try ProvenanceCLI.checkpointManifestHash(at: modelPath),
            checkpointIdentityKind: "config-index-shard-name-size-manifest",
            mlxSwiftLMRevision: ProvenanceCLI.mlxSwiftLMRevision,
            maxActiveSlots: 2,
            maxPrefillSlots: 2,
            prefillChunkSize: prefillChunk,
            maxReservedContextTokens: maxReservedContext,
            maxReservedKVBytes: maxReservedKVBytes,
            maxOutputTokens: maxTokens,
            longRepeat: longRepeat,
            memoryCacheLimitBytes: 8 << 30,
            cancellationGate: gate,
            warmup: evidenceRuns[0],
            measuredRuns: Array(evidenceRuns.dropFirst()))
        let provenance = ProvenanceCLI.build(
            modelPath: modelPath,
            referenceVersions: nil,
            corpus: nil).provenance
        try appendRequiredJSONLRecord(
            ResultRecord(
                subcommand: "service-cancel-bench",
                provenance: provenance,
                payload: payload),
            to: evidencePath(flags))
        print(
            "# cancellation p50=\(fmt(gate.summary.p50Seconds * 1_000, 3))ms, "
                + "p95=\(fmt(gate.summary.p95Seconds * 1_000, 3))ms, "
                + "max=\(fmt(gate.summary.maxSeconds * 1_000, 3))ms, "
                + "keepalive=\(keepaliveMS)ms -> \(gate.withinKeepalive ? "PASS" : "FAIL")")
        print("# provenance: appended to \(evidencePath(flags))")
        guard gate.withinKeepalive else {
            throw ServiceCancellationBenchError.keepaliveExceeded(gate.summary.maxSeconds)
        }
    } catch BenchGuardError.debugBuild {
        print("service-cancel-bench FAILED: Debug build — service numbers would be misleading")
        exit(1)
    } catch {
        print("service-cancel-bench FAILED: \(error)")
        exit(1)
    }
}
