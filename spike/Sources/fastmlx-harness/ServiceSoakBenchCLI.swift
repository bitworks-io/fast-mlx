import Foundation
import HarnessCore
import MLXLMCommon

private enum ServiceSoakBenchError: Error, CustomStringConvertible {
    case invalidArguments(String)
    case gateFailed
    case missingRuntimeResources
    case emptyResponsivenessOutput

    var description: String {
        switch self {
        case .invalidArguments(let message): message
        case .gateFailed: "continuous service soak gate failed"
        case .missingRuntimeResources: "continuous service soak could not read runtime resources"
        case .emptyResponsivenessOutput: "continuous service responsiveness probe emitted no bytes"
        }
    }
}

private struct ServiceSoakCycleEvidence: Codable, Sendable {
    let cycle: Int
    let droppedWarmup: Bool
    let durationSeconds: Double
    let statePoison: ServiceStatePoisonRunEvidence
    let statePoisonPassed: Bool
    let beforeAggregateTokensPerSecond: Double
    let afterAggregateTokensPerSecond: Double
    let beforeOutputHashes: [String]
    let afterOutputHashes: [String]
    let outputsByteIdentical: Bool
    let cancellationLatencySeconds: Double
    let initialActiveRequestCount: Int
    let replacementWasQueued: Bool
    let replacementSlotReused: Bool
    let hostileOperations: ServiceOperationSummary
    let resourcesBeforeCancellation: ContinuousBatchRuntimeResourceSnapshot
    let resourcesAfterCancellation: ContinuousBatchRuntimeResourceSnapshot
    let hostileResourcesAtEnd: ContinuousBatchRuntimeResourceSnapshot
    let responsivenessSeconds: Double
    let responsivenessOutputHash: String
    let responsivenessOutputMatchesWarmup: Bool
    let resourcesAtEnd: ContinuousBatchRuntimeResourceSnapshot
    let memory: ServiceMemorySample
    let maxSampledFootprintBytes: UInt64
    let passed: Bool
}

private struct ServiceSoakPredicateCount: Codable, Sendable {
    var passed: Int = 0
    var failed: Int = 0

    mutating func record(_ result: Bool) {
        if result {
            passed += 1
        } else {
            failed += 1
        }
    }
}

private struct ServiceSoakPredicateSummary: Codable, Sendable {
    let cycleCount: Int
    let predicateCount: Int
    let counts: [String: ServiceSoakPredicateCount]
    let allPassed: Bool
}

private struct ServiceSoakCancellationGate: Codable, Sendable {
    let sampleCount: Int
    let meanSeconds: Double
    let maxSeconds: Double
    let keepaliveSeconds: Double
    let withinKeepalive: Bool
}

private struct ServiceSoakBenchPayload: Codable, Sendable {
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
    let checkpointIntervalSeconds: Int
    let retainedCheckpointLimit: Int
    let totalCycleCount: Int
    let targetMeasuredDurationSeconds: Int
    let actualMeasuredDurationSeconds: Double
    let memoryCacheLimitBytes: Int
    let livenessMechanism: String
    let progressHeartbeatEnabled: Bool
    let durationSatisfied: Bool
    let predicateSummary: ServiceSoakPredicateSummary
    let cancellationGate: ServiceSoakCancellationGate
    let soakGate: ServiceSoakGate
    let memoryCheckpoints: [ServiceMemorySample]
    let warmup: ServiceSoakCycleEvidence
    let measuredCycles: [ServiceSoakCycleEvidence]
    let peakMeasuredRSSCycle: ServiceSoakCycleEvidence
    let slowestResponsivenessCycle: ServiceSoakCycleEvidence
    let slowestCancellationCycle: ServiceSoakCycleEvidence
    let passed: Bool
}

private struct ServiceSoakProgress: Codable, Sendable {
    let status: String
    let heartbeat: String
    let processID: Int32
    let completedCycle: Int
    let elapsedSeconds: Double
    let physicalFootprintBytes: UInt64
    let lastCyclePassed: Bool?
    let error: String?
}

private enum ServiceResponsivenessRace: Sendable {
    case completed(ContinuousServiceRunObservation)
    case timedOut
    case timerCancelled
}

func runServiceSoakBench(_ flags: Flags) async {
    let progressPath = flags.string("progress")
    var progressOutputValidated = false
    var completedCycle = -1
    var startedAt = ProcessInfo.processInfo.systemUptime
    guard let modelPath = flags.string("model"), let progressPath else {
        print(
            "usage: fastmlx-harness service-soak --model <PATH> --progress <FILE> [--duration-seconds 86400] [--concurrency 4] [--max-tokens 64] [--prefill-chunk 16] [--keepalive-ms 1000] [--responsiveness-ms 30000] [--max-rss-drift-percent 5] [--checkpoint-interval-seconds 300] [--evidence FILE]")
        exit(2)
    }

    do {
        try assertReleaseBuild()
        let evidenceFile = evidencePath(flags)
        guard !outputPathIsSymbolicLink(evidenceFile),
            !outputPathIsSymbolicLink(progressPath),
            !outputPathsReferToSameFile(evidenceFile, progressPath)
        else {
            throw ServiceSoakBenchError.invalidArguments(
                "--progress and --evidence must name different, non-symbolic-link files")
        }
        progressOutputValidated = true
        let durationSeconds = try flags.strictInt("duration-seconds", default: 86_400)
        let concurrency = try flags.strictInt("concurrency", default: 4)
        let maxTokens = try flags.strictInt("max-tokens", default: 64)
        let prefillChunk = try flags.strictInt("prefill-chunk", default: 16)
        let keepaliveMS = try flags.strictInt("keepalive-ms", default: 1_000)
        let responsivenessMS = try flags.strictInt("responsiveness-ms", default: 30_000)
        let maxRSSDriftPercent = try flags.strictInt("max-rss-drift-percent", default: 5)
        let hostileLongRepeat = try flags.strictInt("hostile-long-repeat", default: 18)
        let checkpointIntervalSeconds = try flags.strictInt(
            "checkpoint-interval-seconds", default: 300)
        guard (1 ... 172_800).contains(durationSeconds),
            [4, 8].contains(concurrency),
            (2 ... 256).contains(maxTokens),
            (1 ... 4_096).contains(prefillChunk),
            (1 ... 60_000).contains(keepaliveMS),
            (1 ... 600_000).contains(responsivenessMS),
            (0 ... 100).contains(maxRSSDriftPercent),
            (1 ... 128).contains(hostileLongRepeat),
            (60 ... 3_600).contains(checkpointIntervalSeconds)
        else {
            throw ServiceSoakBenchError.invalidArguments(
                "--duration-seconds must be 1...172800, --concurrency one of 4,8, --max-tokens 2...256, --prefill-chunk 1...4096, --keepalive-ms 1...60000, --responsiveness-ms 1...600000, --max-rss-drift-percent 0...100, --hostile-long-repeat 1...128, and --checkpoint-interval-seconds 60...3600")
        }
        let maxReservedContext = try flags.optionalStrictInt("max-reserved-context-tokens")
        let modelContextLimit = try serviceStatePoisonModelContextLimit(modelPath)
        let (aggregateContextLimit, contextOverflow) = modelContextLimit.multipliedReportingOverflow(
            by: concurrency)
        if let maxReservedContext,
            maxReservedContext <= 0 || contextOverflow
                || maxReservedContext > aggregateContextLimit
        {
            throw ServiceSoakBenchError.invalidArguments(
                "--max-reserved-context-tokens must be in 1...\(contextOverflow ? Int.max : aggregateContextLimit) for this model and concurrency")
        }
        let physicalRAMBytes = Int(min(ProvenanceCLI.ramBytes(), UInt64(Int.max)))
        let maxReservedKVBytes = try flags.optionalStrictInt("max-reserved-kv-bytes")
            ?? physicalRAMBytes / 4
        guard maxReservedKVBytes > 0, maxReservedKVBytes <= physicalRAMBytes else {
            throw ServiceSoakBenchError.invalidArguments(
                "--max-reserved-kv-bytes must be in 1...\(physicalRAMBytes)")
        }

        try writeServiceSoakProgress(
            status: "loading",
            completedCycle: completedCycle,
            elapsedSeconds: 0,
            sample: serviceMemorySample(),
            lastCyclePassed: nil,
            error: nil,
            to: progressPath)
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
        let responsivenessPrompt = tokenizer.encode(
            text: saltPrompt(
                run: 0,
                nonce: workloadNonce,
                "Responsiveness probe: continue with one concise token."))
        let retainedCheckpointLimit = 1_024
        var baselineResponsivenessBytes: [UInt8]?
        var memoryCheckpoints = [serviceMemorySample()]
        var warmupEvidence: ServiceSoakCycleEvidence?
        var retainedMeasuredCycles: [ServiceSoakCycleEvidence] = []
        var lastMeasuredCycle: ServiceSoakCycleEvidence?
        var peakMeasuredRSSCycle: ServiceSoakCycleEvidence?
        var slowestResponsivenessCycle: ServiceSoakCycleEvidence?
        var slowestCancellationCycle: ServiceSoakCycleEvidence?
        var measuredStartedAt: Double?
        var nextCheckpointAt: Double?
        var totalCycleCount = 0
        var allCyclesPassed = true
        var baselineRSSBytes: UInt64 = 0
        var endRSSBytes = memoryCheckpoints[0].physicalFootprintBytes
        var maxRSSBytes: UInt64 = 0
        var maxResponsivenessSeconds = 0.0
        var cancellationSampleCount = 0
        var cancellationSecondsTotal = 0.0
        var cancellationMaxSeconds = 0.0
        var predicateCounts: [String: ServiceSoakPredicateCount] = [:]
        startedAt = ProcessInfo.processInfo.systemUptime

        while true {
            let cycle = totalCycleCount
            let cycleStartedAt = ProcessInfo.processInfo.systemUptime
            let previouslyCompletedCycle = completedCycle
            let statePoison = try await runServiceStatePoisonCycle(
                driver: driver,
                tokenizer: tokenizer,
                concurrency: concurrency,
                run: cycle,
                nonce: workloadNonce,
                maxOutputTokens: maxTokens,
                prefillChunk: prefillChunk,
                keepaliveMS: keepaliveMS,
                hostileLongRepeat: hostileLongRepeat,
                stageHeartbeat: { stage, sample in
                    try writeServiceSoakProgress(
                        status: "running-\(stage)",
                        completedCycle: previouslyCompletedCycle,
                        elapsedSeconds: ProcessInfo.processInfo.systemUptime - startedAt,
                        sample: sample,
                        lastCyclePassed: nil,
                        error: nil,
                        to: progressPath)
                })

            let responsivenessStartedAt = ProcessInfo.processInfo.systemUptime
            let responsiveness = try await runServiceResponsivenessProbe(
                driver: driver,
                prompts: [responsivenessPrompt],
                deadlineMilliseconds: responsivenessMS,
                completedCycle: completedCycle,
                soakStartedAt: startedAt,
                progressPath: progressPath)
            let responsivenessSeconds = ProcessInfo.processInfo.systemUptime
                - responsivenessStartedAt
            let responsivenessBytes = responsiveness.outputTokens.flatMap { tokens in
                Array(tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false).utf8)
            }
            guard !responsivenessBytes.isEmpty else {
                throw ServiceSoakBenchError.emptyResponsivenessOutput
            }
            if baselineResponsivenessBytes == nil {
                baselineResponsivenessBytes = responsivenessBytes
            }
            let responsivenessMatchesWarmup = responsivenessBytes
                == baselineResponsivenessBytes!
            guard let resourcesAtEnd = await driver.coordinator.runtimeResourceSnapshot() else {
                throw ServiceSoakBenchError.missingRuntimeResources
            }
            let memory = serviceMemorySample()
            let cancellationLatency = statePoison.cancellationTimeline.removedAt
                - statePoison.cancellationTimeline.requestedAt
            let predicateResults = serviceSoakPredicateResults(
                statePoison: statePoison.evidence,
                concurrency: concurrency,
                prefillChunk: prefillChunk,
                cancellationLatencySeconds: cancellationLatency,
                keepaliveSeconds: Double(keepaliveMS) / 1_000,
                responsivenessOutputMatchesWarmup: responsivenessMatchesWarmup,
                resourcesAtEnd: resourcesAtEnd)
            for (name, result) in predicateResults {
                var count = predicateCounts[name, default: ServiceSoakPredicateCount()]
                count.record(result)
                predicateCounts[name] = count
            }
            let cyclePassed = predicateResults.values.allSatisfy { $0 }
            let cycleEvidence = ServiceSoakCycleEvidence(
                cycle: cycle,
                droppedWarmup: cycle == 0,
                durationSeconds: ProcessInfo.processInfo.systemUptime - cycleStartedAt,
                statePoison: statePoison.evidence,
                statePoisonPassed: statePoison.evidence.passed,
                beforeAggregateTokensPerSecond:
                    statePoison.evidence.before.metrics.aggregateTokensPerSecond,
                afterAggregateTokensPerSecond:
                    statePoison.evidence.after.metrics.aggregateTokensPerSecond,
                beforeOutputHashes: statePoison.evidence.outputGate.beforeOutputHashes,
                afterOutputHashes: statePoison.evidence.outputGate.afterOutputHashes,
                outputsByteIdentical: statePoison.evidence.outputGate.byteIdentical,
                cancellationLatencySeconds: cancellationLatency,
                initialActiveRequestCount:
                    statePoison.evidence.hostile.initialActiveRequestCount,
                replacementWasQueued: statePoison.evidence.hostile.replacementWasQueued,
                replacementSlotReused: statePoison.evidence.hostile.replacementSlotReused,
                hostileOperations: statePoison.evidence.hostile.operations,
                resourcesBeforeCancellation:
                    statePoison.evidence.hostile.resourcesBeforeCancellation,
                resourcesAfterCancellation:
                    statePoison.evidence.hostile.resourcesAfterCancellation,
                hostileResourcesAtEnd: statePoison.evidence.hostile.resourcesAtEnd,
                responsivenessSeconds: responsivenessSeconds,
                responsivenessOutputHash: fnv1a64(responsivenessBytes),
                responsivenessOutputMatchesWarmup: responsivenessMatchesWarmup,
                resourcesAtEnd: resourcesAtEnd,
                memory: memory,
                maxSampledFootprintBytes: max(
                    max(
                        statePoison.evidence.sequenceMemory.maxSampledFootprintBytes,
                        max(
                            statePoison.evidence.before.memory.maxSampledFootprintBytes,
                            max(
                                statePoison.evidence.hostile.memory.maxSampledFootprintBytes,
                                statePoison.evidence.after.memory.maxSampledFootprintBytes))),
                    max(
                        responsiveness.memory.maxSampledFootprintBytes,
                        memory.physicalFootprintBytes)),
                passed: cyclePassed)
            totalCycleCount += 1
            completedCycle = cycle
            allCyclesPassed = allCyclesPassed && cyclePassed
            endRSSBytes = memory.physicalFootprintBytes
            if slowestResponsivenessCycle == nil
                || responsivenessSeconds
                    > slowestResponsivenessCycle!.responsivenessSeconds
            {
                slowestResponsivenessCycle = cycleEvidence
            }
            if slowestCancellationCycle == nil
                || cancellationLatency
                    > slowestCancellationCycle!.cancellationLatencySeconds
            {
                slowestCancellationCycle = cycleEvidence
            }
            maxResponsivenessSeconds = max(
                maxResponsivenessSeconds, responsivenessSeconds)
            cancellationSampleCount += 1
            cancellationSecondsTotal += cancellationLatency
            cancellationMaxSeconds = max(cancellationMaxSeconds, cancellationLatency)
            let cycleCompletedAt = ProcessInfo.processInfo.systemUptime
            if cycle == 0 {
                warmupEvidence = cycleEvidence
                baselineRSSBytes = memory.physicalFootprintBytes
                maxRSSBytes = baselineRSSBytes
                memoryCheckpoints.append(memory)
                measuredStartedAt = cycleCompletedAt
                nextCheckpointAt = cycleCompletedAt + Double(checkpointIntervalSeconds)
            } else {
                lastMeasuredCycle = cycleEvidence
                if peakMeasuredRSSCycle == nil
                    || cycleEvidence.maxSampledFootprintBytes
                        > peakMeasuredRSSCycle!.maxSampledFootprintBytes
                {
                    peakMeasuredRSSCycle = cycleEvidence
                }
                maxRSSBytes = max(maxRSSBytes, cycleEvidence.maxSampledFootprintBytes)
                if let checkpoint = nextCheckpointAt, cycleCompletedAt >= checkpoint {
                    if retainedMeasuredCycles.count < retainedCheckpointLimit - 1 {
                        retainedMeasuredCycles.append(cycleEvidence)
                        memoryCheckpoints.append(memory)
                    }
                    var advanced = checkpoint
                    while advanced <= cycleCompletedAt {
                        advanced += Double(checkpointIntervalSeconds)
                    }
                    nextCheckpointAt = advanced
                }
            }
            let elapsed = ProcessInfo.processInfo.systemUptime - startedAt
            try writeServiceSoakProgress(
                status: "running",
                completedCycle: completedCycle,
                elapsedSeconds: elapsed,
                sample: memory,
                lastCyclePassed: cyclePassed,
                error: nil,
                to: progressPath)
            print(
                "# \(cycle == 0 ? "warmup" : "cycle \(cycle)"): "
                    + "duration=\(fmt(cycleEvidence.durationSeconds, 2))s, "
                    + "rss=\(memory.physicalFootprintBytes), "
                    + "responsive=\(fmt(responsivenessSeconds * 1_000, 1))ms, "
                    + "A/B/A=\(statePoison.evidence.passed ? "PASS" : "FAIL")")

            guard let measuredStartedAt else { continue }
            let measuredElapsed = ProcessInfo.processInfo.systemUptime - measuredStartedAt
            if totalCycleCount >= 2,
                !allCyclesPassed || measuredElapsed >= Double(durationSeconds)
            {
                break
            }
        }

        let responsivenessLimitSeconds = Double(responsivenessMS) / 1_000
        guard let measuredStartedAt, let warmupEvidence, let lastMeasuredCycle,
            let peakMeasuredRSSCycle, let slowestResponsivenessCycle,
            let slowestCancellationCycle
        else {
            throw ServiceSoakBenchError.gateFailed
        }
        if retainedMeasuredCycles.last?.cycle != lastMeasuredCycle.cycle {
            if retainedMeasuredCycles.count == retainedCheckpointLimit {
                retainedMeasuredCycles.removeLast()
                memoryCheckpoints.removeLast()
            }
            retainedMeasuredCycles.append(lastMeasuredCycle)
            memoryCheckpoints.append(lastMeasuredCycle.memory)
        }
        let predicateSummary = ServiceSoakPredicateSummary(
            cycleCount: totalCycleCount,
            predicateCount: predicateCounts.count,
            counts: predicateCounts,
            allPassed: !predicateCounts.isEmpty
                && predicateCounts.values.allSatisfy {
                    $0.failed == 0 && $0.passed == totalCycleCount
                })
        let soakGate = try evaluateServiceSoakSummary(
            cycleCount: totalCycleCount,
            allCyclesPassed: predicateSummary.allPassed,
            baselineRSSBytes: baselineRSSBytes,
            endRSSBytes: endRSSBytes,
            maxRSSBytes: maxRSSBytes,
            maxResponsivenessSeconds: maxResponsivenessSeconds,
            maxRSSDriftPercent: Double(maxRSSDriftPercent),
            responsivenessLimitSeconds: responsivenessLimitSeconds)
        let keepaliveSeconds = Double(keepaliveMS) / 1_000
        let cancellationGate = ServiceSoakCancellationGate(
            sampleCount: cancellationSampleCount,
            meanSeconds: cancellationSecondsTotal / Double(cancellationSampleCount),
            maxSeconds: cancellationMaxSeconds,
            keepaliveSeconds: keepaliveSeconds,
            withinKeepalive: cancellationMaxSeconds <= keepaliveSeconds)
        let actualMeasuredDuration = ProcessInfo.processInfo.systemUptime - measuredStartedAt
        let durationSatisfied = actualMeasuredDuration >= Double(durationSeconds)
        let passed = predicateSummary.allPassed && soakGate.passed
            && cancellationGate.withinKeepalive && durationSatisfied
        let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        let config = ProvenanceCLI.modelConfig(at: modelPath)
        let quantization = [
            config.quant.label,
            config.quant.groupSize.map { "group=\($0)" },
        ].compactMap { $0 }.joined(separator: ":")
        let payload = ServiceSoakBenchPayload(
            label: flags.string("label", default: "continuous-service-soak"),
            scenario: "mixed-chat-agent-anthropic-tool/state-poison/responsiveness",
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
            checkpointIntervalSeconds: checkpointIntervalSeconds,
            retainedCheckpointLimit: retainedCheckpointLimit,
            totalCycleCount: totalCycleCount,
            targetMeasuredDurationSeconds: durationSeconds,
            actualMeasuredDurationSeconds: actualMeasuredDuration,
            memoryCacheLimitBytes: 8 << 30,
            livenessMechanism: "external PID + progress-heartbeat watchdog",
            progressHeartbeatEnabled: true,
            durationSatisfied: durationSatisfied,
            predicateSummary: predicateSummary,
            cancellationGate: cancellationGate,
            soakGate: soakGate,
            memoryCheckpoints: memoryCheckpoints,
            warmup: warmupEvidence,
            measuredCycles: retainedMeasuredCycles,
            peakMeasuredRSSCycle: peakMeasuredRSSCycle,
            slowestResponsivenessCycle: slowestResponsivenessCycle,
            slowestCancellationCycle: slowestCancellationCycle,
            passed: passed)
        let provenance = ProvenanceCLI.build(
            modelPath: modelPath,
            referenceVersions: nil,
            corpus: nil).provenance
        try appendRequiredJSONLRecord(
            ResultRecord(
                subcommand: "service-soak",
                provenance: provenance,
                payload: payload),
            to: evidenceFile)
        try writeServiceSoakProgress(
            status: passed ? "passed" : "failed",
            completedCycle: completedCycle,
            elapsedSeconds: ProcessInfo.processInfo.systemUptime - startedAt,
            sample: lastMeasuredCycle.memory,
            lastCyclePassed: lastMeasuredCycle.passed,
            error: passed ? nil : ServiceSoakBenchError.gateFailed.description,
            to: progressPath)
        print(
            "# soak \(passed ? "PASS" : "FAIL"): measured=\(soakGate.measuredCycleCount), "
                + "rss-max-drift=\(fmt(soakGate.maxRSSDriftPercent, 3))%, "
                + "responsiveness-max=\(fmt(soakGate.maxResponsivenessSeconds * 1_000, 1))ms")
        print("# provenance: appended to \(evidenceFile)")
        guard passed else { throw ServiceSoakBenchError.gateFailed }
    } catch BenchGuardError.debugBuild {
        print("service-soak FAILED: Debug build — service numbers would be misleading")
        exit(1)
    } catch {
        let sample = serviceMemorySample()
        if progressOutputValidated {
            try? writeServiceSoakProgress(
                status: "failed",
                completedCycle: completedCycle,
                elapsedSeconds: ProcessInfo.processInfo.systemUptime - startedAt,
                sample: sample,
                lastCyclePassed: false,
                error: String(describing: error),
                to: progressPath)
        }
        print("service-soak FAILED: \(error)")
        exit(1)
    }
}

private func runServiceResponsivenessProbe(
    driver: ContinuousSwiftServiceDriver,
    prompts: [[Int]],
    deadlineMilliseconds: Int,
    completedCycle: Int,
    soakStartedAt: Double,
    progressPath: String
) async throws -> ContinuousServiceRunObservation {
    try await withThrowingTaskGroup(of: ServiceResponsivenessRace.self) { group in
        group.addTask {
            .completed(
                try await driver.runBurst(
                    prompts: prompts,
                    maxOutputTokens: 1))
        }
        group.addTask {
            do {
                try await Task.sleep(
                    nanoseconds: UInt64(deadlineMilliseconds) * 1_000_000)
                return .timedOut
            } catch is CancellationError {
                return .timerCancelled
            }
        }

        while let result = try await group.next() {
            switch result {
            case .completed(let observation):
                group.cancelAll()
                return observation
            case .timedOut:
                let sample = serviceMemorySample()
                try? writeServiceSoakProgress(
                    status: "failed",
                    completedCycle: completedCycle,
                    elapsedSeconds: ProcessInfo.processInfo.systemUptime - soakStartedAt,
                    sample: sample,
                    lastCyclePassed: false,
                    error: "responsiveness deadline exceeded: \(deadlineMilliseconds)ms",
                    to: progressPath)
                print(
                    "service-soak FAILED: responsiveness deadline exceeded "
                        + "(\(deadlineMilliseconds)ms)")
                exit(1)
            case .timerCancelled:
                group.cancelAll()
                throw CancellationError()
            }
        }
        throw CancellationError()
    }
}

private func serviceSoakPredicateResults(
    statePoison: ServiceStatePoisonRunEvidence,
    concurrency: Int,
    prefillChunk: Int,
    cancellationLatencySeconds: Double,
    keepaliveSeconds: Double,
    responsivenessOutputMatchesWarmup: Bool,
    resourcesAtEnd: ContinuousBatchRuntimeResourceSnapshot
) -> [String: Bool] {
    let expectedPromptTokens = statePoison.goodPromptTokenCounts.reduce(0, +)
    let expectedPromptChunks = statePoison.goodPromptTokenCounts.reduce(0) {
        $0 + (($1 + prefillChunk - 1) / prefillChunk)
    }
    let before = statePoison.before.operations
    let hostile = statePoison.hostile
    let after = statePoison.after.operations
    let output = statePoison.outputGate
    return [
        "beforePromptTokensExact": before.promptTokensProcessed == expectedPromptTokens,
        "beforePromptChunksExact": before.promptChunkCount == expectedPromptChunks,
        "beforeActiveSlotsReached": before.maxActiveSlots == concurrency,
        "beforeSharedDecodeBatch": before.decodeBatchSizeHistogram.keys.contains { $0 > 1 },
        "beforeSpeculationDisabled": !before.speculationEngaged,
        "hostilePreviousPhaseDecoding": hostile.previousPhase == "decoding",
        "hostileSlotRemoved": hostile.slotRemoved,
        "hostileCancellationEventObserved": hostile.cancellationEventObserved,
        "hostileRepeatedCancellationNotFound": hostile.repeatedCancellationNotFound,
        "hostileInitialActiveCountExact": hostile.initialActiveRequestCount == concurrency,
        "hostileReplacementQueued": hostile.replacementWasQueued,
        "hostileReplacementSlotReused": hostile.replacementSlotReused,
        "hostileSharedBatchObserved": hostile.sharedBatchObserved,
        "hostileDecodeFirstInterleaveObserved": hostile.decodeFirstInterleaveObserved,
        "hostileReservationPreservedAfterCancellation":
            hostile.resourcesAfterCancellation.reservedKVBytes
                == hostile.resourcesBeforeCancellation.reservedKVBytes,
        "hostileKVReleasedAtEnd": hostile.resourcesAtEnd.reservedKVBytes == 0,
        "afterPromptTokensExact": after.promptTokensProcessed == expectedPromptTokens,
        "afterPromptChunksExact": after.promptChunkCount == expectedPromptChunks,
        "afterActiveSlotsReached": after.maxActiveSlots == concurrency,
        "afterSharedDecodeBatch": after.decodeBatchSizeHistogram.keys.contains { $0 > 1 },
        "afterSpeculationDisabled": !after.speculationEngaged,
        "kvReleasedAfterA": statePoison.resourcesAfterBefore.reservedKVBytes == 0,
        "kvReleasedAfterAPrime": statePoison.resourcesAtEnd.reservedKVBytes == 0,
        "outputRequestCountsMatch": output.requestCountsMatch,
        "outputNonEmpty": output.outputsNonEmpty,
        "outputPerRequestByteMatch": output.perRequestByteMatch.allSatisfy { $0 },
        "outputByteIdentical": output.byteIdentical,
        "outputGatePassed": output.passed,
        "statePoisonStructuralAggregatePassed": statePoison.structuralPassed,
        "statePoisonAggregatePassed": statePoison.passed,
        "cancellationWithinKeepalive": cancellationLatencySeconds <= keepaliveSeconds,
        "responsivenessOutputMatchesWarmup": responsivenessOutputMatchesWarmup,
        "responsivenessKVReleased": resourcesAtEnd.reservedKVBytes == 0,
    ]
}

private func writeServiceSoakProgress(
    status: String,
    completedCycle: Int,
    elapsedSeconds: Double,
    sample: ServiceMemorySample,
    lastCyclePassed: Bool?,
    error: String?,
    to path: String
) throws {
    let progress = ServiceSoakProgress(
        status: status,
        heartbeat: ProvenanceCLI.nowISO8601(),
        processID: ProcessInfo.processInfo.processIdentifier,
        completedCycle: completedCycle,
        elapsedSeconds: elapsedSeconds,
        physicalFootprintBytes: sample.physicalFootprintBytes,
        lastCyclePassed: lastCyclePassed,
        error: error)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(progress).write(
        to: URL(fileURLWithPath: path),
        options: .atomic)
}
