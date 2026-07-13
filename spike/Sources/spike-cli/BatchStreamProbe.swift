import Foundation
import HarnessCore
import MLX
import MLXLMCommon
import SpikeCore

private enum BatchStreamProbeError: Error, CustomStringConvertible {
    case invalidArguments
    case insufficientFirstBudget(required: Int, actual: Int)
    case firstRequestFinishedBeforeJoin
    case missingTransition(String)
    case exactnessFailure(String)

    var description: String {
        switch self {
        case .invalidArguments:
            return "token limits and prefill chunk must be positive"
        case .insufficientFirstBudget(let required, let actual):
            return "--max-tokens must exceed \(required) for this prompt/chunk transition; actual=\(actual)"
        case .firstRequestFinishedBeforeJoin:
            return "first request finished before the staggered join boundary"
        case .missingTransition(let transition):
            return "required transition was not observed: \(transition)"
        case .exactnessFailure(let request):
            return "continuous stream diverged from scalar baseline for \(request)"
        }
    }
}

private struct BatchStreamProbeResult: Codable {
    let harnessSHA: String
    let model: String
    let modelConfigHash: String
    let checkpointManifestHash: String
    let modelQuantization: String
    let mlxSwiftVersion: String
    let mlxSwiftLMRevision: String
    let compilePolicy: String
    let prefillChunkSize: Int
    let joinAfterTokens: Int
    let firstPromptTokens: Int
    let secondPromptTokens: Int
    let firstOutputTokens: Int
    let secondOutputTokens: Int
    let firstTokenExact: Bool
    let secondTokenExact: Bool
    let firstByteExact: Bool
    let secondByteExact: Bool
    let drainObserved: Bool
    let sharedBatchObserved: Bool
    let batchToSoloObserved: Bool
    let batchedSpeculationEngaged: Bool
    let operationTrace: [String]
    let status: String
}

private struct BatchMembershipProbeResult: Codable {
    let harnessSHA: String
    let model: String
    let modelConfigHash: String
    let checkpointManifestHash: String
    let modelQuantization: String
    let mlxSwiftVersion: String
    let mlxSwiftLMRevision: String
    let compilePolicy: String
    let prefillChunkSize: Int
    let cancelAfterTokens: Int
    let survivorTokenExact: Bool
    let survivorByteExact: Bool
    let cancelledPrefixTokenExact: Bool
    let cancelledPrefixByteExact: Bool
    let batchThreeObserved: Bool
    let batchTwoObserved: Bool
    let cancellationPhase: String
    let operationTrace: [String]
    let status: String
}

/// Real-model acceptance gate for chunked prefill plus B1 → drain → B2 → B1 transitions.
func batchStreamProbe(
    modelPath: String,
    maxTokens: Int,
    joinAfter: Int,
    prefillChunkSize: Int
) async {
    do {
        guard maxTokens > joinAfter,
            joinAfter > 0,
            prefillChunkSize > 0
        else {
            throw BatchStreamProbeError.invalidArguments
        }

        let modelURL = URL(fileURLWithPath: modelPath)
        let proof = try DenseContinuousBatchModelProof.verifying(modelDirectory: modelURL)
        let modelProvenance = try batchProbeModelProvenance(modelDirectory: modelURL)
        Memory.cacheLimit = 8 << 30
        let loaded = try await loadModelAndTokenizer(modelPath: modelPath)
        let tokenizer = loaded.tokenizer
        let eos = loaded.eos
        let prompts = [
            tokenizer.encode(
                text: "Explain why fixed-capacity KV buffers can make compiled decode stable on Apple Silicon."),
            tokenizer.encode(
                text: "Give two concise reasons decode-first scheduling protects interactive latency."),
        ]
        let outputBudgets = [maxTokens, 4]
        let joinerPrefillTicks =
            (prompts[1].count + prefillChunkSize - 1) / prefillChunkSize
        let requiredBeforeBatchToSolo = joinAfter + joinerPrefillTicks + 1 + outputBudgets[1]
        guard maxTokens > requiredBeforeBatchToSolo else {
            throw BatchStreamProbeError.insufficientFirstBudget(
                required: requiredBeforeBatchToSolo,
                actual: maxTokens)
        }
        let baselines = [
            scalarGreedyBaseline(
                model: loaded.model,
                prompt: prompts[0],
                maxTokens: outputBudgets[0],
                eos: eos),
            scalarGreedyBaseline(
                model: loaded.model,
                prompt: prompts[1],
                maxTokens: outputBudgets[1],
                eos: eos),
        ]

        let coordinator = try makeBatchStreamCoordinator(
            model: loaded.model,
            proof: proof,
            maxActiveSlots: 2,
            maxPrefillSlots: 1,
            prefillChunkSize: prefillChunkSize,
            traceLimit: promptChunkCount(prompts[0].count, prefillChunkSize)
                + promptChunkCount(prompts[1].count, prefillChunkSize)
                + outputBudgets.reduce(0, +)
                + 32)
        let first = try await coordinator.submit(
            ContinuousBatchSubmission(
                promptTokens: prompts[0],
                maxOutputTokens: outputBudgets[0],
                eosToken: eos,
                architecture: .denseAttention))

        while true {
            if case .decoding(let emitted, _) = await coordinator.snapshot(for: first.id)?.phase,
                emitted >= joinAfter
            {
                break
            }
            guard try await coordinator.runOneTick() else {
                throw BatchStreamProbeError.firstRequestFinishedBeforeJoin
            }
        }

        let second = try await coordinator.submit(
            ContinuousBatchSubmission(
                promptTokens: prompts[1],
                maxOutputTokens: outputBudgets[1],
                eosToken: eos,
                architecture: .denseAttention))
        while try await coordinator.runOneTick() {}

        let actual = [
            try await collectProbeTokens(first.tokens),
            try await collectProbeTokens(second.tokens),
        ]
        let events = await coordinator.executionTrace()
        let operations = events.compactMap { event -> BatchSchedulerOperation? in
            if case .operation(let operation) = event { return operation }
            return nil
        }
        let trace = operations.map(probeOperationDescription)
        let drainObserved = operations.contains { operation in
            if case .decode(.drainSoloPipeline(first.id)) = operation { return true }
            return false
        }
        let sharedBatchObserved = operations.contains { operation in
            if case .decode(.batch(let ids, speculationAllowed: false)) = operation {
                return ids == [first.id, second.id]
            }
            return false
        }
        let batchedSpeculationEngaged = operations.contains { operation in
            if case .decode(.batch(_, speculationAllowed: true)) = operation { return true }
            return false
        }
        let batchIndex = operations.firstIndex { operation in
            if case .decode(.batch(_, _)) = operation { return true }
            return false
        }
        let batchToSoloObserved = batchIndex.map { start in
            operations[(start + 1)...].contains { operation in
                if case .decode(.solo(first.id, speculationAllowed: false)) = operation {
                    return true
                }
                return false
            }
        } ?? false

        let tokenExact = [actual[0] == baselines[0], actual[1] == baselines[1]]
        let byteExact = [
            decodedBytes(tokenizer: tokenizer, tokens: actual[0])
                == decodedBytes(tokenizer: tokenizer, tokens: baselines[0]),
            decodedBytes(tokenizer: tokenizer, tokens: actual[1])
                == decodedBytes(tokenizer: tokenizer, tokens: baselines[1]),
        ]
        let passed = tokenExact.allSatisfy { $0 }
            && byteExact.allSatisfy { $0 }
            && drainObserved
            && sharedBatchObserved
            && batchToSoloObserved
            && !batchedSpeculationEngaged
        let result = BatchStreamProbeResult(
            harnessSHA: try batchProbeHarnessSHA(),
            model: modelURL.lastPathComponent,
            modelConfigHash: modelProvenance.configHash,
            checkpointManifestHash: modelProvenance.checkpointManifestHash,
            modelQuantization: modelProvenance.quantization,
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision: "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
            compilePolicy: "fixed-shape-membership",
            prefillChunkSize: prefillChunkSize,
            joinAfterTokens: joinAfter,
            firstPromptTokens: prompts[0].count,
            secondPromptTokens: prompts[1].count,
            firstOutputTokens: actual[0].count,
            secondOutputTokens: actual[1].count,
            firstTokenExact: tokenExact[0],
            secondTokenExact: tokenExact[1],
            firstByteExact: byteExact[0],
            secondByteExact: byteExact[1],
            drainObserved: drainObserved,
            sharedBatchObserved: sharedBatchObserved,
            batchToSoloObserved: batchToSoloObserved,
            batchedSpeculationEngaged: batchedSpeculationEngaged,
            operationTrace: trace,
            status: passed ? "PASS" : "FAIL")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        print(String(data: try encoder.encode(result), encoding: .utf8)!)

        guard tokenExact[0], byteExact[0] else {
            throw BatchStreamProbeError.exactnessFailure("first request")
        }
        guard tokenExact[1], byteExact[1] else {
            throw BatchStreamProbeError.exactnessFailure("second request")
        }
        guard drainObserved else { throw BatchStreamProbeError.missingTransition("drain") }
        guard sharedBatchObserved else {
            throw BatchStreamProbeError.missingTransition("shared batch")
        }
        guard batchToSoloObserved else {
            throw BatchStreamProbeError.missingTransition("batch to solo")
        }
        guard !batchedSpeculationEngaged else {
            throw BatchStreamProbeError.missingTransition("batched speculation stayed disabled")
        }
    } catch {
        print("batch-stream-probe FAILED: \(error)")
        exit(1)
    }
}

/// Real-model middle-row removal gate for B3 → B2 cache extraction/remerge.
func batchMembershipProbe(
    modelPath: String,
    maxTokens: Int,
    cancelAfter: Int,
    prefillChunkSize: Int
) async {
    do {
        guard maxTokens > cancelAfter,
            cancelAfter > 0,
            prefillChunkSize > 0
        else {
            throw BatchStreamProbeError.invalidArguments
        }

        let modelURL = URL(fileURLWithPath: modelPath)
        let proof = try DenseContinuousBatchModelProof.verifying(modelDirectory: modelURL)
        let modelProvenance = try batchProbeModelProvenance(modelDirectory: modelURL)
        Memory.cacheLimit = 8 << 30
        let loaded = try await loadModelAndTokenizer(modelPath: modelPath)
        let tokenizer = loaded.tokenizer
        let eos = loaded.eos
        let prompts = [
            tokenizer.encode(text: "Batch row alpha."),
            tokenizer.encode(text: "Batch row beta differs."),
            tokenizer.encode(text: "Batch row gamma stays active."),
        ]
        let baselines = prompts.map {
            scalarGreedyBaseline(
                model: loaded.model,
                prompt: $0,
                maxTokens: maxTokens,
                eos: eos)
        }
        let coordinator = try makeBatchStreamCoordinator(
            model: loaded.model,
            proof: proof,
            maxActiveSlots: 3,
            maxPrefillSlots: 3,
            prefillChunkSize: prefillChunkSize,
            traceLimit: prompts.map { promptChunkCount($0.count, prefillChunkSize) }
                .reduce(0, +)
                + maxTokens
                + 32)
        let handles = try await coordinator.submitBatch(
            prompts.map {
                ContinuousBatchSubmission(
                    promptTokens: $0,
                    maxOutputTokens: maxTokens,
                    eosToken: eos,
                    architecture: .denseAttention)
            })

        while true {
            if case .decoding(let emitted, _) = await coordinator.snapshot(for: handles[1].id)?.phase,
                emitted >= cancelAfter
            {
                break
            }
            guard try await coordinator.runOneTick() else {
                throw BatchStreamProbeError.firstRequestFinishedBeforeJoin
            }
        }
        let cancellation = await coordinator.cancel(handles[1].id)
        while try await coordinator.runOneTick() {}

        let actual = [
            try await collectProbeTokens(handles[0].tokens),
            try await collectProbeTokens(handles[1].tokens),
            try await collectProbeTokens(handles[2].tokens),
        ]
        let operations = await coordinator.executionTrace().compactMap {
            event -> BatchSchedulerOperation? in
            if case .operation(let operation) = event { return operation }
            return nil
        }
        let survivorTokenExact = actual[0] == baselines[0] && actual[2] == baselines[2]
        let survivorByteExact = decodedBytes(tokenizer: tokenizer, tokens: actual[0])
            == decodedBytes(tokenizer: tokenizer, tokens: baselines[0])
            && decodedBytes(tokenizer: tokenizer, tokens: actual[2])
                == decodedBytes(tokenizer: tokenizer, tokens: baselines[2])
        let cancelledBaseline = Array(baselines[1].prefix(actual[1].count))
        let cancelledPrefixTokenExact = actual[1] == cancelledBaseline
        let cancelledPrefixByteExact = decodedBytes(tokenizer: tokenizer, tokens: actual[1])
            == decodedBytes(tokenizer: tokenizer, tokens: cancelledBaseline)
        let batchThreeObserved = operations.contains { operation in
            if case .decode(.batch(let ids, speculationAllowed: false)) = operation {
                return ids == handles.map(\.id)
            }
            return false
        }
        let survivorIDs = [handles[0].id, handles[2].id]
        let batchTwoObserved = operations.contains { operation in
            if case .decode(.batch(let ids, speculationAllowed: false)) = operation {
                return ids == survivorIDs
            }
            return false
        }
        let cancellationPhase: String
        let cancellationWasDecoding: Bool
        if case .cancelled(_, let phase) = cancellation {
            cancellationPhase = String(describing: phase)
            if case .decoding(let emitted, _) = phase {
                cancellationWasDecoding = emitted == cancelAfter
            } else {
                cancellationWasDecoding = false
            }
        } else {
            cancellationPhase = "not-found"
            cancellationWasDecoding = false
        }
        let passed = survivorTokenExact
            && survivorByteExact
            && cancelledPrefixTokenExact
            && cancelledPrefixByteExact
            && actual[1].count == cancelAfter
            && cancellationWasDecoding
            && batchThreeObserved
            && batchTwoObserved
        let result = BatchMembershipProbeResult(
            harnessSHA: try batchProbeHarnessSHA(),
            model: modelURL.lastPathComponent,
            modelConfigHash: modelProvenance.configHash,
            checkpointManifestHash: modelProvenance.checkpointManifestHash,
            modelQuantization: modelProvenance.quantization,
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision: "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
            compilePolicy: "fixed-shape-membership",
            prefillChunkSize: prefillChunkSize,
            cancelAfterTokens: cancelAfter,
            survivorTokenExact: survivorTokenExact,
            survivorByteExact: survivorByteExact,
            cancelledPrefixTokenExact: cancelledPrefixTokenExact,
            cancelledPrefixByteExact: cancelledPrefixByteExact,
            batchThreeObserved: batchThreeObserved,
            batchTwoObserved: batchTwoObserved,
            cancellationPhase: cancellationPhase,
            operationTrace: operations.map(probeOperationDescription),
            status: passed ? "PASS" : "FAIL")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        print(String(data: try encoder.encode(result), encoding: .utf8)!)

        guard passed else {
            throw BatchStreamProbeError.exactnessFailure("middle-removal scenario")
        }
    } catch {
        print("batch-membership-probe FAILED: \(error)")
        exit(1)
    }
}

private func makeBatchStreamCoordinator(
    model: sending any LanguageModel,
    proof: DenseContinuousBatchModelProof,
    maxActiveSlots: Int,
    maxPrefillSlots: Int,
    prefillChunkSize: Int,
    traceLimit: Int
) throws -> ContinuousBatchCoordinator {
    let configuration = try ContinuousBatchConfiguration(
        maxActiveSlots: maxActiveSlots,
        maxPrefillSlots: maxPrefillSlots,
        prefillChunkSize: prefillChunkSize)
    return ContinuousBatchCoordinator(
        configuration: configuration,
        runtime: try DenseContinuousBatchRuntime(
            model: model,
            verifiedBy: proof),
        automaticDrive: false,
        traceLimit: traceLimit)
}

private func promptChunkCount(_ promptTokens: Int, _ chunkSize: Int) -> Int {
    (promptTokens + chunkSize - 1) / chunkSize
}

private struct BatchProbeModelProvenance {
    let configHash: String
    let checkpointManifestHash: String
    let quantization: String
}

private func batchProbeModelProvenance(
    modelDirectory: URL
) throws -> BatchProbeModelProvenance {
    let fileManager = FileManager.default
    let config = try Data(
        contentsOf: modelDirectory.appendingPathComponent("config.json"))
    var manifestBytes = Array(config)
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    if fileManager.fileExists(atPath: indexURL.path) {
        manifestBytes.append(contentsOf: try Data(contentsOf: indexURL))
    }
    let weightFiles = try fileManager.contentsOfDirectory(
        at: modelDirectory,
        includingPropertiesForKeys: [.fileSizeKey],
        options: [.skipsHiddenFiles])
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    for file in weightFiles {
        let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1
        manifestBytes.append(
            contentsOf: "\(file.lastPathComponent):\(size)\n".utf8)
    }
    let quant = ModelQuantInfoLoader.load(from: config)
    let quantization = [
        quant.label,
        quant.groupSize.map { "group=\($0)" },
    ].compactMap { $0 }.joined(separator: ":")
    return BatchProbeModelProvenance(
        configHash: fnv1a64(config),
        checkpointManifestHash: fnv1a64(manifestBytes),
        quantization: quantization)
}

private func scalarGreedyBaseline(
    model: any LanguageModel,
    prompt: [Int],
    maxTokens: Int,
    eos: Int
) -> [Int] {
    var decoder = CompiledMLXDecoder(model: model, reserve: maxTokens + 16)
    var current = decoder.prefill(prompt)
    var tokens: [Int] = []
    tokens.reserveCapacity(maxTokens)
    while tokens.count < maxTokens, current != eos {
        tokens.append(current)
        if tokens.count < maxTokens {
            current = decoder.step(last: current)
        }
    }
    return tokens
}

private func collectProbeTokens(
    _ stream: AsyncThrowingStream<Int, Error>
) async throws -> [Int] {
    var tokens: [Int] = []
    for try await token in stream { tokens.append(token) }
    return tokens
}

private func decodedBytes(
    tokenizer: MLXLMCommon.Tokenizer,
    tokens: [Int]
) -> Data {
    Data(tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false).utf8)
}

private func probeOperationDescription(_ operation: BatchSchedulerOperation) -> String {
    switch operation {
    case .prefill(let slice):
        return "prefill:\(slice.id.rawValue):\(slice.startToken)+\(slice.count)"
    case .decode(.solo(let id, let speculationAllowed)):
        return "solo:\(id.rawValue):spec=\(speculationAllowed)"
    case .decode(.drainSoloPipeline(let id)):
        return "drain:\(id.rawValue)"
    case .decode(.batch(let ids, let speculationAllowed)):
        let membership = ids.map { String($0.rawValue) }.joined(separator: ":")
        return "batch:\(membership):spec=\(speculationAllowed)"
    }
}
