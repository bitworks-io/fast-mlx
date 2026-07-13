import Foundation
import HarnessCore
import MLXLMCommon

private enum ServiceStatePoisonBenchError: Error, CustomStringConvertible {
    case invalidArguments(String)
    case gateFailed

    var description: String {
        switch self {
        case .invalidArguments(let message): message
        case .gateFailed: "state-poison A/B/A recovery gate failed"
        }
    }
}

private struct ServiceStatePoisonArmEvidence: Codable, Sendable {
    let metrics: ServiceRunMetrics
    let operations: ServiceOperationSummary
    let memory: ServiceMemorySummary
    let resourcesAtAdmission: ContinuousBatchRuntimeResourceSnapshot
}

private struct ServiceStatePoisonHostileEvidence: Codable, Sendable {
    let cancellationLatencySeconds: Double
    let previousPhase: String
    let slotRemoved: Bool
    let cancellationEventObserved: Bool
    let repeatedCancellationNotFound: Bool
    let initialActiveRequestCount: Int
    let replacementWasQueued: Bool
    let replacementSlotReused: Bool
    let sharedBatchObserved: Bool
    let decodeFirstInterleaveObserved: Bool
    let cancelledPrefixTokens: Int
    let survivorOutputTokens: Int
    let replacementOutputTokens: Int
    let operations: ServiceOperationSummary
    let resourcesAtAdmission: ContinuousBatchRuntimeResourceSnapshot
    let resourcesBeforeCancellation: ContinuousBatchRuntimeResourceSnapshot
    let resourcesAfterCancellation: ContinuousBatchRuntimeResourceSnapshot
    let resourcesAfterReplacement: ContinuousBatchRuntimeResourceSnapshot
    let resourcesAtEnd: ContinuousBatchRuntimeResourceSnapshot
}

private struct ServiceStatePoisonRunEvidence: Codable, Sendable {
    let run: Int
    let droppedWarmup: Bool
    let goodPromptTokenCounts: [Int]
    let hostilePromptTokenCounts: [Int]
    let before: ServiceStatePoisonArmEvidence
    let resourcesAfterBefore: ContinuousBatchRuntimeResourceSnapshot
    let hostile: ServiceStatePoisonHostileEvidence
    let after: ServiceStatePoisonArmEvidence
    let sequenceMemory: ServiceMemorySummary
    let resourcesAtEnd: ContinuousBatchRuntimeResourceSnapshot
    let outputGate: ServiceStatePoisonGate
    let structuralPassed: Bool
    let passed: Bool
}

private struct ServiceStatePoisonBenchPayload: Codable, Sendable {
    let label: String
    let scenario: String
    let model: String
    let modelQuantization: String
    let checkpointManifestHash: String
    let checkpointIdentityKind: String
    let mlxSwiftLMRevision: String
    let workloadNonce: String
    let concurrency: Int
    let maxActiveSlots: Int
    let maxPrefillSlots: Int
    let prefillChunkSize: Int
    let maxReservedContextTokens: Int?
    let maxReservedKVBytes: Int
    let maxOutputTokens: Int
    let hostileLongRepeat: Int
    let memoryCacheLimitBytes: Int
    let cancellationGate: ServiceCancellationGate
    let warmup: ServiceStatePoisonRunEvidence
    let measuredRuns: [ServiceStatePoisonRunEvidence]
    let passed: Bool
}

func runServiceStatePoisonBench(_ flags: Flags) async {
    guard let modelPath = flags.string("model") else {
        print(
            "usage: fastmlx-harness service-state-poison-bench --model <PATH> [--runs 3] [--concurrency 4] [--max-tokens 64] [--prefill-chunk 16] [--keepalive-ms 1000] [--hostile-long-repeat 18] [--evidence FILE]")
        exit(2)
    }

    do {
        try assertReleaseBuild()
        let runs = try flags.strictInt("runs", default: 3)
        let concurrency = try flags.strictInt("concurrency", default: 4)
        let maxTokens = try flags.strictInt("max-tokens", default: 64)
        let prefillChunk = try flags.strictInt("prefill-chunk", default: 16)
        let keepaliveMS = try flags.strictInt("keepalive-ms", default: 1_000)
        let hostileLongRepeat = try flags.strictInt("hostile-long-repeat", default: 18)
        guard (1 ... 100).contains(runs),
            [2, 4, 8].contains(concurrency),
            (2 ... 4_096).contains(maxTokens),
            (1 ... 4_096).contains(prefillChunk),
            (1 ... 60_000).contains(keepaliveMS),
            (1 ... 128).contains(hostileLongRepeat)
        else {
            throw ServiceStatePoisonBenchError.invalidArguments(
                "--runs must be 1...100, --concurrency one of 2,4,8, --max-tokens 2...4096, --prefill-chunk 1...4096, --keepalive-ms 1...60000, and --hostile-long-repeat 1...128")
        }
        let maxReservedContext = try flags.optionalStrictInt("max-reserved-context-tokens")
        let modelContextLimit = try serviceStatePoisonModelContextLimit(modelPath)
        let (aggregateContextLimit, contextOverflow) = modelContextLimit.multipliedReportingOverflow(
            by: concurrency)
        if let maxReservedContext,
            maxReservedContext <= 0 || contextOverflow
                || maxReservedContext > aggregateContextLimit
        {
            throw ServiceStatePoisonBenchError.invalidArguments(
                "--max-reserved-context-tokens must be in 1...\(contextOverflow ? Int.max : aggregateContextLimit) for this model and concurrency")
        }
        let physicalRAMBytes = Int(min(ProvenanceCLI.ramBytes(), UInt64(Int.max)))
        let maxReservedKVBytes = try flags.optionalStrictInt("max-reserved-kv-bytes")
            ?? physicalRAMBytes / 4
        guard maxReservedKVBytes > 0, maxReservedKVBytes <= physicalRAMBytes else {
            throw ServiceStatePoisonBenchError.invalidArguments(
                "--max-reserved-kv-bytes must be in 1...\(physicalRAMBytes)")
        }

        let loaded = try await loadContinuousSwiftServiceDriver(
            modelPath: modelPath,
            configuration: ContinuousServiceLoadConfiguration(
                maxActiveSlots: concurrency,
                maxPrefillSlots: concurrency,
                prefillChunkSize: prefillChunk,
                maxReservedContextTokens: maxReservedContext,
                maxReservedKVBytes: maxReservedKVBytes,
                traceLimit: max(4_096, concurrency * (maxTokens + 128) * 8)))
        let tokenizer = loaded.tokenizer
        let driver = loaded.driver
        let workloadNonce = ProvenanceCLI.nonce()
        var evidenceRuns: [ServiceStatePoisonRunEvidence] = []
        var cancellationTimelines: [ServiceCancellationTimeline] = []

        for run in 0 ... runs {
            let goodPrompts = serviceStatePoisonGoodPrompts(
                tokenizer: tokenizer,
                concurrency: concurrency,
                run: run,
                nonce: workloadNonce)
            let hostilePrompts = serviceStatePoisonHostilePrompts(
                tokenizer: tokenizer,
                concurrency: concurrency,
                run: run,
                nonce: workloadNonce,
                longRepeat: hostileLongRepeat)
            var sequenceSamples = [serviceMemorySample()]
            let before = try await driver.runBurst(
                prompts: goodPrompts,
                maxOutputTokens: maxTokens)
            sequenceSamples.append(serviceMemorySample())
            guard let resourcesAfterBefore = await driver.coordinator.runtimeResourceSnapshot()
            else {
                throw ServiceStatePoisonBenchError.gateFailed
            }
            let hostile = try await driver.runCancellationRecovery(
                prompts: hostilePrompts,
                maxOutputTokens: maxTokens,
                cancellationWaitLimitSeconds: max(1, Double(keepaliveMS) / 500))
            sequenceSamples.append(serviceMemorySample())
            let after = try await driver.runBurst(
                prompts: goodPrompts,
                maxOutputTokens: maxTokens)
            sequenceSamples.append(serviceMemorySample())
            guard let resourcesAtEnd = await driver.coordinator.runtimeResourceSnapshot() else {
                throw ServiceStatePoisonBenchError.gateFailed
            }

            let outputGate = evaluateServiceStatePoisonRecovery(
                before: decodedServiceOutputBytes(before.outputTokens, tokenizer: tokenizer),
                after: decodedServiceOutputBytes(after.outputTokens, tokenizer: tokenizer))
            let beforeStructural = serviceStatePoisonArmPassed(
                before,
                prompts: goodPrompts,
                concurrency: concurrency,
                prefillChunk: prefillChunk)
            let afterStructural = serviceStatePoisonArmPassed(
                after,
                prompts: goodPrompts,
                concurrency: concurrency,
                prefillChunk: prefillChunk)
            let hostileStructural = hostile.previousPhase == "decoding"
                && hostile.slotRemoved
                && hostile.cancellationEventObserved
                && hostile.repeatedCancellationNotFound
                && hostile.initialActiveRequestCount == concurrency
                && hostile.replacementWasQueued
                && hostile.replacementSlotReused
                && hostile.sharedBatchObserved
                && hostile.decodeFirstInterleaveObserved
                && hostile.resourcesAfterCancellation.reservedKVBytes
                    == hostile.resourcesBeforeCancellation.reservedKVBytes
                && hostile.resourcesAtEnd.reservedKVBytes == 0
            let structuralPassed = beforeStructural && hostileStructural && afterStructural
                && resourcesAfterBefore.reservedKVBytes == 0
                && resourcesAtEnd.reservedKVBytes == 0
            let evidence = ServiceStatePoisonRunEvidence(
                run: run,
                droppedWarmup: run == 0,
                goodPromptTokenCounts: goodPrompts.map(\.count),
                hostilePromptTokenCounts: hostilePrompts.map(\.count),
                before: statePoisonArmEvidence(before),
                resourcesAfterBefore: resourcesAfterBefore,
                hostile: ServiceStatePoisonHostileEvidence(
                    cancellationLatencySeconds: hostile.timeline.removedAt
                        - hostile.timeline.requestedAt,
                    previousPhase: hostile.previousPhase,
                    slotRemoved: hostile.slotRemoved,
                    cancellationEventObserved: hostile.cancellationEventObserved,
                    repeatedCancellationNotFound: hostile.repeatedCancellationNotFound,
                    initialActiveRequestCount: hostile.initialActiveRequestCount,
                    replacementWasQueued: hostile.replacementWasQueued,
                    replacementSlotReused: hostile.replacementSlotReused,
                    sharedBatchObserved: hostile.sharedBatchObserved,
                    decodeFirstInterleaveObserved: hostile.decodeFirstInterleaveObserved,
                    cancelledPrefixTokens: hostile.cancelledPrefixTokens,
                    survivorOutputTokens: hostile.survivorOutputTokens,
                    replacementOutputTokens: hostile.replacementOutputTokens,
                    operations: hostile.operations,
                    resourcesAtAdmission: hostile.resourcesAtAdmission,
                    resourcesBeforeCancellation: hostile.resourcesBeforeCancellation,
                    resourcesAfterCancellation: hostile.resourcesAfterCancellation,
                    resourcesAfterReplacement: hostile.resourcesAfterReplacement,
                    resourcesAtEnd: hostile.resourcesAtEnd),
                after: statePoisonArmEvidence(after),
                sequenceMemory: try summarizeServiceMemory(sequenceSamples),
                resourcesAtEnd: resourcesAtEnd,
                outputGate: outputGate,
                structuralPassed: structuralPassed,
                passed: structuralPassed && outputGate.passed)
            evidenceRuns.append(evidence)
            cancellationTimelines.append(hostile.timeline)
            print(
                "# \(run == 0 ? "warmup (dropped)" : "run \(run)"): "
                    + "A=\(fmt(before.metrics.aggregateTokensPerSecond, 2)) tok/s, "
                    + "B-cancel=\(fmt(evidence.hostile.cancellationLatencySeconds * 1_000, 3))ms, "
                    + "A'=\(fmt(after.metrics.aggregateTokensPerSecond, 2)) tok/s, "
                    + "bytes=\(outputGate.byteIdentical ? "MATCH" : "MISMATCH"), "
                    + "kv-end=\(resourcesAtEnd.reservedKVBytes)")
        }

        let cancellationGate = try evaluateCancellationGate(
            cancellationTimelines,
            keepaliveSeconds: Double(keepaliveMS) / 1_000)
        let measured = Array(evidenceRuns.dropFirst())
        let passed = cancellationGate.withinKeepalive && evidenceRuns.allSatisfy(\.passed)
        let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        let config = ProvenanceCLI.modelConfig(at: modelPath)
        let quantization = [
            config.quant.label,
            config.quant.groupSize.map { "group=\($0)" },
        ].compactMap { $0 }.joined(separator: ":")
        let payload = ServiceStatePoisonBenchPayload(
            label: flags.string("label", default: "continuous-state-poison"),
            scenario: "known-good-burst/disconnect-cancel-reuse/known-good-burst",
            model: modelName,
            modelQuantization: quantization,
            checkpointManifestHash: try ProvenanceCLI.checkpointManifestHash(at: modelPath),
            checkpointIdentityKind: "config-index-shard-name-size-manifest",
            mlxSwiftLMRevision: ProvenanceCLI.mlxSwiftLMRevision,
            workloadNonce: workloadNonce,
            concurrency: concurrency,
            maxActiveSlots: concurrency,
            maxPrefillSlots: concurrency,
            prefillChunkSize: prefillChunk,
            maxReservedContextTokens: maxReservedContext,
            maxReservedKVBytes: maxReservedKVBytes,
            maxOutputTokens: maxTokens,
            hostileLongRepeat: hostileLongRepeat,
            memoryCacheLimitBytes: 8 << 30,
            cancellationGate: cancellationGate,
            warmup: evidenceRuns[0],
            measuredRuns: measured,
            passed: passed)
        let provenance = ProvenanceCLI.build(
            modelPath: modelPath,
            referenceVersions: nil,
            corpus: nil).provenance
        try appendRequiredJSONLRecord(
            ResultRecord(
                subcommand: "service-state-poison-bench",
                provenance: provenance,
                payload: payload),
            to: evidencePath(flags))
        print(
            "# A/B/A \(passed ? "PASS" : "FAIL"): measured=\(measured.count), "
                + "cancel-max=\(fmt(cancellationGate.summary.maxSeconds * 1_000, 3))ms")
        print("# provenance: appended to \(evidencePath(flags))")
        guard passed else { throw ServiceStatePoisonBenchError.gateFailed }
    } catch BenchGuardError.debugBuild {
        print("service-state-poison-bench FAILED: Debug build — service numbers would be misleading")
        exit(1)
    } catch {
        print("service-state-poison-bench FAILED: \(error)")
        exit(1)
    }
}

private func serviceStatePoisonGoodPrompts(
    tokenizer: MLXLMCommon.Tokenizer,
    concurrency: Int,
    run: Int,
    nonce: String
) -> [[Int]] {
    let workloads = [
        "Chat: explain why bounded admission matters for an on-device language model.",
        "Agent recall: remember marker ORCHID-17, then state the marker and its number.",
        "Tool request: return one JSON object with keys action and safety_check.",
        "Swift completion: func clamp(_ value: Int, lower: Int, upper: Int) -> Int {",
    ]
    return (0 ..< concurrency).map { request in
        tokenizer.encode(
            text: saltPrompt(
                run: run,
                nonce: nonce,
                "\(workloads[request % workloads.count]) [request=\(request)]"))
    }
}

private func serviceStatePoisonHostilePrompts(
    tokenizer: MLXLMCommon.Tokenizer,
    concurrency: Int,
    run: Int,
    nonce: String,
    longRepeat: Int
) -> [[Int]] {
    let salt = saltPrompt(run: run, nonce: nonce, "Hostile disconnect recovery.")
    let longTail = String(
        repeating:
            " Decode-first scheduling advances ready tokens while bounded prompt chunks continue.",
        count: longRepeat)
    var prompts = [
        tokenizer.encode(text: salt + longTail),
        tokenizer.encode(text: salt + " " + specVerifyPrompt),
    ]
    if concurrency > 2 {
        for request in 2 ..< concurrency {
            prompts.append(
                tokenizer.encode(
                    text: salt + longTail + " Additional occupied slot \(request)."))
        }
    }
    prompts.append(
        tokenizer.encode(
            text: salt
                + " Replacement request: enumerate stable row identity, bounded admission, and exact recovery."))
    return prompts
}

private func serviceStatePoisonModelContextLimit(_ modelPath: String) throws -> Int {
    struct ModelLimits: Decodable {
        let maxPositionEmbeddings: Int

        enum CodingKeys: String, CodingKey {
            case maxPositionEmbeddings = "max_position_embeddings"
        }
    }
    let url = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
    do {
        let limits = try JSONDecoder().decode(ModelLimits.self, from: Data(contentsOf: url))
        guard limits.maxPositionEmbeddings > 0 else {
            throw ServiceStatePoisonBenchError.invalidArguments(
                "model max_position_embeddings must be positive")
        }
        return limits.maxPositionEmbeddings
    } catch let error as ServiceStatePoisonBenchError {
        throw error
    } catch {
        throw ServiceStatePoisonBenchError.invalidArguments(
            "model config does not expose a valid max_position_embeddings")
    }
}

private func serviceStatePoisonArmPassed(
    _ observation: ContinuousServiceRunObservation,
    prompts: [[Int]],
    concurrency: Int,
    prefillChunk: Int
) -> Bool {
    let expectedPromptTokens = prompts.reduce(0) { $0 + $1.count }
    let expectedChunks = prompts.reduce(0) {
        $0 + (($1.count + prefillChunk - 1) / prefillChunk)
    }
    return observation.operations.promptTokensProcessed == expectedPromptTokens
        && observation.operations.promptChunkCount == expectedChunks
        && observation.operations.maxActiveSlots == concurrency
        && observation.operations.decodeBatchSizeHistogram.keys.contains(where: { $0 > 1 })
        && !observation.operations.speculationEngaged
}

private func statePoisonArmEvidence(
    _ observation: ContinuousServiceRunObservation
) -> ServiceStatePoisonArmEvidence {
    ServiceStatePoisonArmEvidence(
        metrics: observation.metrics,
        operations: observation.operations,
        memory: observation.memory,
        resourcesAtAdmission: observation.resources)
}

private func decodedServiceOutputBytes(
    _ outputTokens: [[Int]],
    tokenizer: MLXLMCommon.Tokenizer
) -> [[UInt8]] {
    outputTokens.map {
        Array(tokenizer.decode(tokenIds: $0, skipSpecialTokens: false).utf8)
    }
}
