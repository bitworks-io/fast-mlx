import Foundation
import HarnessCore
import MLX
import MLXLMCommon

/// Unforgeable outside SpikeCore: created only by reading the model's checked-in config.
public struct DenseContinuousBatchModelProof: Sendable, Equatable {
    fileprivate let maxPositionEmbeddings: Int
    fileprivate let vocabularySize: Int

    private struct Configuration: Decodable {
        let modelType: String
        let maxPositionEmbeddings: Int
        let vocabularySize: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case maxPositionEmbeddings = "max_position_embeddings"
            case vocabularySize = "vocab_size"
        }
    }

    private init(maxPositionEmbeddings: Int, vocabularySize: Int) {
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.vocabularySize = vocabularySize
    }

    public static func verifying(modelDirectory: URL) throws -> Self {
        let url = modelDirectory.appendingPathComponent("config.json")
        let configuration: Configuration
        do {
            configuration = try JSONDecoder().decode(
                Configuration.self, from: Data(contentsOf: url))
        } catch {
            throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        guard configuration.modelType == "qwen3" else {
            throw DenseContinuousBatchRuntimeError.unsupportedModelFamily(
                configuration.modelType)
        }
        guard configuration.maxPositionEmbeddings > 0,
            configuration.maxPositionEmbeddings <= Int(Int32.max),
            configuration.vocabularySize > 0,
            configuration.vocabularySize <= Int(Int32.max)
        else {
            throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        return Self(
            maxPositionEmbeddings: configuration.maxPositionEmbeddings,
            vocabularySize: configuration.vocabularySize)
    }

    static func testing(maxPositionEmbeddings: Int, vocabularySize: Int) -> Self {
        Self(maxPositionEmbeddings: maxPositionEmbeddings, vocabularySize: vocabularySize)
    }
}

public enum DenseContinuousBatchRuntimeError: Error, Equatable {
    case unsupportedModelFamily(String)
    case invalidModelConfiguration
    case invalidAllocationChunk(Int)
    case invalidContextLimit(Int)
    case invalidAggregateContextLimit(Int)
    case invalidInitialDecodeReserve(Int)
    case noCacheLayers
    case unknownRequest(BatchRequestID)
    case outOfOrderPrefill(BatchRequestID, expected: Int, actual: Int)
    case invalidPrefillMetadata(BatchRequestID)
    case prefillAfterFinalization(BatchRequestID)
    case decodeBeforeFinalPrefill(BatchRequestID)
    case drainWithoutPendingLookahead(BatchRequestID)
    case pendingLookaheadInBatch(BatchRequestID)
    case invalidBatchMembership([BatchRequestID])
    case speculationUnsupported
    case cacheLayerMismatch(BatchRequestID, expected: Int, actual: Int)
    case cacheLengthMismatch(BatchRequestID, expected: Int, actual: Int)
    case contextLimitExceeded(BatchRequestID, requested: Int, limit: Int)
    case aggregateContextLimitExceeded(requested: Int, limit: Int)
    case invalidTokenID(BatchRequestID, Int)
    case positionOverflow(BatchRequestID)
}

struct DenseContinuousBatchRuntimeDiagnostics: Equatable {
    let batchTraceCount: Int
    let batchMembership: [BatchRequestID]
    let batchCapacity: Int?
}

/// Actor-confined dense MLX executor for `ContinuousBatchCoordinator`.
///
/// This type is intentionally not `Sendable`. The coordinator's `sending` initializer
/// transfers the runtime, model, cache arrays, and compiled closures into one actor isolation
/// region. Only value-typed request work and decode results cross that boundary.
public final class DenseContinuousBatchRuntime: ContinuousBatchRuntime {
    private typealias Step = ([MLXArray]) -> [MLXArray]

    private final class TraceCounter {
        var count = 0
    }

    private struct Slot {
        let totalPromptTokens: Int
        let maxOutputTokens: Int
        var processedTokens: Int
        var cachedTokens: Int
        var prefillComplete: Bool
        var hasPendingSoloLookahead: Bool
        var scalarCaches: [CompiledKVCache]?
        var stagedToken: MLXArray?
        var scalarStep: Step?
        var scalarTraceCounter: TraceCounter?
    }

    private struct MaterializedSlot {
        let caches: [CompiledKVCache]
        let stagedToken: MLXArray
    }

    private struct BatchState {
        let ids: [BatchRequestID]
        let caches: [BatchedCompiledKVCache]
        var stagedTokens: MLXArray
        var step: Step?
        let traceCounter: TraceCounter
    }

    private let model: any LanguageModel
    private let allocationChunk: Int
    private let maxContextTokens: Int
    private let maxReservedContextTokens: Int
    private let initialDecodeReserve: Int
    private let vocabularySize: Int
    private let layerCount: Int
    private var slots: [BatchRequestID: Slot] = [:]
    private var contextReservations: [BatchRequestID: Int] = [:]
    private var batch: BatchState?

    package init(
        model: any LanguageModel,
        verifiedBy proof: DenseContinuousBatchModelProof,
        allocationChunk: Int = 256,
        maxContextTokens: Int? = nil,
        maxReservedContextTokens: Int? = nil,
        initialDecodeReserve: Int = 384
    ) throws {
        guard allocationChunk > 0 else {
            throw DenseContinuousBatchRuntimeError.invalidAllocationChunk(allocationChunk)
        }
        let resolvedContextLimit = maxContextTokens ?? proof.maxPositionEmbeddings
        guard resolvedContextLimit > 0,
            resolvedContextLimit <= proof.maxPositionEmbeddings,
            resolvedContextLimit <= Int(Int32.max)
        else {
            throw DenseContinuousBatchRuntimeError.invalidContextLimit(resolvedContextLimit)
        }
        guard allocationChunk <= resolvedContextLimit else {
            throw DenseContinuousBatchRuntimeError.invalidAllocationChunk(allocationChunk)
        }
        guard initialDecodeReserve > 0 else {
            throw DenseContinuousBatchRuntimeError.invalidInitialDecodeReserve(
                initialDecodeReserve)
        }
        let resolvedReservationLimit = maxReservedContextTokens ?? resolvedContextLimit
        guard resolvedReservationLimit > 0 else {
            throw DenseContinuousBatchRuntimeError.invalidAggregateContextLimit(
                resolvedReservationLimit)
        }
        let layerCount = model.newCache(parameters: nil).count
        guard layerCount > 0 else {
            throw DenseContinuousBatchRuntimeError.noCacheLayers
        }
        self.model = model
        self.allocationChunk = allocationChunk
        self.maxContextTokens = resolvedContextLimit
        self.maxReservedContextTokens = resolvedReservationLimit
        self.initialDecodeReserve = initialDecodeReserve
        self.vocabularySize = proof.vocabularySize
        self.layerCount = layerCount
    }

    convenience init(
        testing model: any LanguageModel,
        allocationChunk: Int = 256,
        maxContextTokens: Int = 32_768,
        maxReservedContextTokens: Int? = nil,
        initialDecodeReserve: Int = 384
    ) throws {
        try self.init(
            model: model,
            verifiedBy: .testing(
                maxPositionEmbeddings: maxContextTokens,
                vocabularySize: 2_048),
            allocationChunk: allocationChunk,
            maxContextTokens: maxContextTokens,
            maxReservedContextTokens: maxReservedContextTokens,
            initialDecodeReserve: initialDecodeReserve)
    }

    public func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        var additions: [(BatchRequestID, Int)] = []
        additions.reserveCapacity(admissions.count)
        for admission in admissions {
            let submission = admission.submission
            guard !submission.requestsSpeculation else {
                throw DenseContinuousBatchRuntimeError.speculationUnsupported
            }
            try validateRequest(
                id: admission.id,
                promptTokens: submission.promptTokens,
                totalPromptTokens: submission.promptTokens.count,
                maxOutputTokens: submission.maxOutputTokens)
            additions.append(
                (admission.id, submission.promptTokens.count + submission.maxOutputTokens))
        }

        var requested = contextReservations.values.reduce(0, +)
        for (_, tokens) in additions {
            let (next, overflow) = requested.addingReportingOverflow(tokens)
            guard !overflow, next <= maxReservedContextTokens else {
                throw DenseContinuousBatchRuntimeError.aggregateContextLimitExceeded(
                    requested: overflow ? Int.max : next,
                    limit: maxReservedContextTokens)
            }
            requested = next
        }
        for (id, tokens) in additions { contextReservations[id] = tokens }
    }

    public func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        guard !work.tokens.isEmpty,
            work.startToken >= 0,
            work.totalPromptTokens > 0,
            work.maxOutputTokens > 0,
            work.startToken <= work.totalPromptTokens,
            work.tokens.count <= work.totalPromptTokens - work.startToken
        else {
            throw DenseContinuousBatchRuntimeError.invalidPrefillMetadata(work.id)
        }
        try validateRequest(
            id: work.id,
            promptTokens: work.tokens,
            totalPromptTokens: work.totalPromptTokens,
            maxOutputTokens: work.maxOutputTokens)
        let endToken = work.startToken + work.tokens.count
        guard work.isFinal == (endToken == work.totalPromptTokens) else {
            throw DenseContinuousBatchRuntimeError.invalidPrefillMetadata(work.id)
        }

        var slot: Slot
        if let existing = slots[work.id] {
            guard !existing.prefillComplete else {
                throw DenseContinuousBatchRuntimeError.prefillAfterFinalization(work.id)
            }
            guard existing.totalPromptTokens == work.totalPromptTokens,
                existing.maxOutputTokens == work.maxOutputTokens
            else {
                throw DenseContinuousBatchRuntimeError.invalidPrefillMetadata(work.id)
            }
            slot = existing
        } else {
            guard work.startToken == 0 else {
                throw DenseContinuousBatchRuntimeError.outOfOrderPrefill(
                    work.id, expected: 0, actual: work.startToken)
            }
            let capacity = try roundedCapacity(
                promptTokens: work.totalPromptTokens,
                outputTokens: min(work.maxOutputTokens, initialDecodeReserve),
                id: work.id)
            slot = Slot(
                totalPromptTokens: work.totalPromptTokens,
                maxOutputTokens: work.maxOutputTokens,
                processedTokens: 0,
                cachedTokens: 0,
                prefillComplete: false,
                hasPendingSoloLookahead: false,
                scalarCaches: (0 ..< layerCount).map { _ in
                    CompiledKVCache(capacity: capacity)
                },
                stagedToken: nil,
                scalarStep: nil,
                scalarTraceCounter: nil)
        }

        guard work.startToken == slot.processedTokens else {
            throw DenseContinuousBatchRuntimeError.outOfOrderPrefill(
                work.id, expected: slot.processedTokens, actual: work.startToken)
        }
        guard let caches = slot.scalarCaches else {
            throw DenseContinuousBatchRuntimeError.prefillAfterFinalization(work.id)
        }
        guard caches.count == layerCount else {
            throw DenseContinuousBatchRuntimeError.cacheLayerMismatch(
                work.id, expected: layerCount, actual: caches.count)
        }

        let ids = MLXArray(work.tokens.map(Int32.init)).reshaped([1, work.tokens.count])
        let modelCaches: [any KVCache] = caches.map { $0 }
        let logits = model(ids, cache: modelCaches)
        let staged = argMax(logits[0..., -1, 0...], axis: -1)
        eval([staged] + caches.flatMap { $0.innerState() })

        for cache in caches {
            let actual = Int(cache.offsetArr.item(Int32.self))
            guard actual == endToken else {
                throw DenseContinuousBatchRuntimeError.cacheLengthMismatch(
                    work.id, expected: endToken, actual: actual)
            }
        }
        slot.processedTokens = endToken
        slot.cachedTokens = endToken
        slot.prefillComplete = work.isFinal
        slot.stagedToken = work.isFinal ? staged : nil
        slots[work.id] = slot
    }

    public func decode(_ action: BatchDecodeAction) throws
        -> [ContinuousBatchRuntimeDecodeResult]
    {
        switch action {
        case .solo(let id, let speculationAllowed):
            guard !speculationAllowed else {
                throw DenseContinuousBatchRuntimeError.speculationUnsupported
            }
            try ensureScalar(id)
            return [try decodeScalar(id, leavesPendingLookahead: true, synchronously: false)]

        case .drainSoloPipeline(let id):
            guard let slot = slots[id] else {
                throw DenseContinuousBatchRuntimeError.unknownRequest(id)
            }
            guard slot.hasPendingSoloLookahead else {
                throw DenseContinuousBatchRuntimeError.drainWithoutPendingLookahead(id)
            }
            try ensureScalar(id)
            return [try decodeScalar(id, leavesPendingLookahead: false, synchronously: true)]

        case .batch(let ids, let speculationAllowed):
            guard !speculationAllowed else {
                throw DenseContinuousBatchRuntimeError.speculationUnsupported
            }
            return try decodeBatch(ids)
        }
    }

    public func remove(_ id: BatchRequestID) {
        contextReservations[id] = nil
        slots[id] = nil
        guard let batch else { return }
        if !batch.ids.contains(where: { slots[$0] != nil }) {
            self.batch = nil
        }
    }

    func diagnostics() -> DenseContinuousBatchRuntimeDiagnostics {
        DenseContinuousBatchRuntimeDiagnostics(
            batchTraceCount: batch?.traceCounter.count ?? 0,
            batchMembership: batch?.ids ?? [],
            batchCapacity: batch?.caches.first?.capacity)
    }

    private func decodeScalar(
        _ id: BatchRequestID,
        leavesPendingLookahead: Bool,
        synchronously: Bool
    ) throws -> ContinuousBatchRuntimeDecodeResult {
        guard var slot = slots[id] else {
            throw DenseContinuousBatchRuntimeError.unknownRequest(id)
        }
        guard slot.prefillComplete,
            slot.scalarCaches != nil,
            let current = slot.stagedToken
        else {
            throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
        }
        try ensureScalarCapacity(for: id, slot: &slot, additionalTokens: 1)

        if slot.scalarStep == nil {
            let traceCounter = TraceCounter()
            slot.scalarTraceCounter = traceCounter
            slot.scalarStep = makeScalarStep(caches: slot.scalarCaches!, counter: traceCounter)
        }
        let following = slot.scalarStep!([current])[0]
        if synchronously {
            eval([following] + slot.scalarCaches!.flatMap { $0.innerState() })
        } else {
            asyncEval(following)
        }
        let emitted = current.item(Int.self)

        slot.cachedTokens += 1
        slot.stagedToken = following
        slot.hasPendingSoloLookahead = leavesPendingLookahead
        slots[id] = slot
        return ContinuousBatchRuntimeDecodeResult(
            id: id,
            tokens: [emitted],
            finished: false,
            hasPendingSoloLookahead: leavesPendingLookahead)
    }

    private func decodeBatch(_ ids: [BatchRequestID]) throws
        -> [ContinuousBatchRuntimeDecodeResult]
    {
        guard ids.count >= 2, Set(ids).count == ids.count else {
            throw DenseContinuousBatchRuntimeError.invalidBatchMembership(ids)
        }
        for id in ids {
            guard let slot = slots[id] else {
                throw DenseContinuousBatchRuntimeError.unknownRequest(id)
            }
            guard slot.prefillComplete else {
                throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
            }
            guard !slot.hasPendingSoloLookahead else {
                throw DenseContinuousBatchRuntimeError.pendingLookaheadInBatch(id)
            }
        }

        try ensureBatch(ids)
        guard var batch else {
            preconditionFailure("ensureBatch returned without a batch")
        }
        try ensureBatchCapacity(&batch, additionalTokens: 1)
        if batch.step == nil {
            batch.step = makeBatchStep(
                caches: batch.caches,
                batchSize: ids.count,
                counter: batch.traceCounter)
        }

        let current = batch.stagedTokens
        let following = batch.step!([current])[0]
        asyncEval(following)
        let emitted = current.asType(.int32).asArray(Int32.self).map(Int.init)
        guard emitted.count == ids.count else {
            throw DenseContinuousBatchRuntimeError.invalidBatchMembership(ids)
        }

        for id in ids {
            guard var slot = slots[id] else {
                throw DenseContinuousBatchRuntimeError.unknownRequest(id)
            }
            slot.cachedTokens += 1
            slot.hasPendingSoloLookahead = false
            slots[id] = slot
        }
        batch.stagedTokens = following
        self.batch = batch
        return zip(ids, emitted).map { id, token in
            ContinuousBatchRuntimeDecodeResult(
                id: id,
                tokens: [token],
                finished: false,
                hasPendingSoloLookahead: false)
        }
    }

    private func ensureScalar(_ id: BatchRequestID) throws {
        guard var slot = slots[id] else {
            throw DenseContinuousBatchRuntimeError.unknownRequest(id)
        }
        guard slot.prefillComplete else {
            throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
        }
        if slot.scalarCaches != nil, slot.stagedToken != nil {
            return
        }
        guard let batch, let row = batch.ids.firstIndex(of: id) else {
            throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
        }
        let otherLiveRows = batch.ids.filter { $0 != id && slots[$0] != nil }
        guard otherLiveRows.isEmpty else {
            throw DenseContinuousBatchRuntimeError.invalidBatchMembership([id] + otherLiveRows)
        }

        let caches = try batch.caches.map { try $0.extract(slot: row) }
        let staged = batch.stagedTokens[row].reshaped([1])
        eval([staged] + caches.flatMap { $0.innerState() })
        try validateCacheLengths(caches, id: id, expected: slot.cachedTokens)

        slot.scalarCaches = caches
        slot.stagedToken = staged
        slot.scalarStep = nil
        slot.scalarTraceCounter = nil
        slots[id] = slot
        self.batch = nil
    }

    private func ensureBatch(_ ids: [BatchRequestID]) throws {
        if batch?.ids == ids { return }

        var materialized: [BatchRequestID: MaterializedSlot] = [:]
        if let existingBatch = batch {
            let liveExisting = existingBatch.ids.filter { slots[$0] != nil }
            guard liveExisting.allSatisfy(ids.contains) else {
                throw DenseContinuousBatchRuntimeError.invalidBatchMembership(ids)
            }
            for id in ids {
                guard let row = existingBatch.ids.firstIndex(of: id),
                    let slot = slots[id]
                else { continue }
                let caches = try existingBatch.caches.map { try $0.extract(slot: row) }
                let staged = existingBatch.stagedTokens[row].reshaped([1])
                eval([staged] + caches.flatMap { $0.innerState() })
                try validateCacheLengths(caches, id: id, expected: slot.cachedTokens)
                materialized[id] = MaterializedSlot(caches: caches, stagedToken: staged)
            }
        }

        for id in ids where materialized[id] == nil {
            guard let slot = slots[id] else {
                throw DenseContinuousBatchRuntimeError.unknownRequest(id)
            }
            guard let caches = slot.scalarCaches, let staged = slot.stagedToken else {
                throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
            }
            try validateCacheLengths(caches, id: id, expected: slot.cachedTokens)
            materialized[id] = MaterializedSlot(caches: caches, stagedToken: staged)
        }

        let rows = try ids.map { id -> MaterializedSlot in
            guard let row = materialized[id] else {
                throw DenseContinuousBatchRuntimeError.unknownRequest(id)
            }
            return row
        }
        let lengths = ids.map { slots[$0]!.cachedTokens }
        var caches: [BatchedCompiledKVCache] = []
        caches.reserveCapacity(layerCount)
        for layer in 0 ..< layerCount {
            caches.append(
                try BatchedCompiledKVCache.merging(
                    rows.map { $0.caches[layer] }, lengths: lengths))
        }
        let staged = concatenated(rows.map(\.stagedToken), axis: 0)
        eval([staged] + caches.flatMap { $0.innerState() })

        let traceCounter = TraceCounter()
        batch = BatchState(
            ids: ids,
            caches: caches,
            stagedTokens: staged,
            step: nil,
            traceCounter: traceCounter)
        for id in ids {
            var slot = slots[id]!
            slot.scalarCaches = nil
            slot.stagedToken = nil
            slot.scalarStep = nil
            slot.scalarTraceCounter = nil
            slot.hasPendingSoloLookahead = false
            slots[id] = slot
        }
    }

    private func makeScalarStep(
        caches: [CompiledKVCache], counter: TraceCounter
    ) -> Step {
        let model = self.model
        let modelCaches: [any KVCache] = caches.map { $0 }
        let state: [any Updatable] = caches.map { $0 }
        let step: Step = { arguments in
            counter.count += 1
            let input = arguments[0].reshaped([1, 1])
            let logits = model(input, cache: modelCaches)
            return [argMax(logits[0..., -1, 0...], axis: -1)]
        }
        return compile(inputs: state, outputs: state, step)
    }

    private func makeBatchStep(
        caches: [BatchedCompiledKVCache], batchSize: Int, counter: TraceCounter
    ) -> Step {
        let model = self.model
        let modelCaches: [any KVCache] = caches.map { $0 }
        let state: [any Updatable] = caches.map { $0 }
        let step: Step = { arguments in
            counter.count += 1
            let input = arguments[0].reshaped([batchSize, 1])
            let logits = model(input, cache: modelCaches)
            return [argMax(logits[0..., -1, 0...], axis: -1)]
        }
        return compile(inputs: state, outputs: state, step)
    }

    private func ensureScalarCapacity(
        for id: BatchRequestID, slot: inout Slot, additionalTokens: Int
    ) throws {
        guard let caches = slot.scalarCaches, let capacity = caches.first?.capacity else {
            throw DenseContinuousBatchRuntimeError.decodeBeforeFinalPrefill(id)
        }
        let required = slot.cachedTokens + additionalTokens
        let requestLimit = min(
            maxContextTokens,
            slot.totalPromptTokens + slot.maxOutputTokens)
        guard required <= requestLimit else {
            throw DenseContinuousBatchRuntimeError.contextLimitExceeded(
                id, requested: required, limit: requestLimit)
        }
        guard required > capacity else { return }

        if let staged = slot.stagedToken {
            eval([staged] + caches.flatMap { $0.innerState() })
        }
        let growth = roundedGrowth(required: required, capacity: capacity)
        for cache in caches { cache.grow(by: growth) }
        slot.scalarStep = nil
        slot.scalarTraceCounter = nil
    }

    private func ensureBatchCapacity(
        _ batch: inout BatchState, additionalTokens: Int
    ) throws {
        for id in batch.ids {
            guard let slot = slots[id] else { continue }
            let required = slot.cachedTokens + additionalTokens
            let requestLimit = min(
                maxContextTokens,
                slot.totalPromptTokens + slot.maxOutputTokens)
            guard required <= requestLimit else {
                throw DenseContinuousBatchRuntimeError.contextLimitExceeded(
                    id, requested: required, limit: requestLimit)
            }
        }
        let required = batch.ids.compactMap { slots[$0]?.cachedTokens }.max() ?? 0
            + additionalTokens
        guard required <= Int(Int32.max) else {
            throw DenseContinuousBatchRuntimeError.positionOverflow(batch.ids[0])
        }
        guard let capacity = batch.caches.first?.capacity, required > capacity else { return }

        eval([batch.stagedTokens] + batch.caches.flatMap { $0.innerState() })
        let growth = roundedGrowth(required: required, capacity: capacity)
        for cache in batch.caches { cache.grow(by: growth) }
        batch.step = nil
    }

    private func validateCacheLengths(
        _ caches: [CompiledKVCache], id: BatchRequestID, expected: Int
    ) throws {
        guard caches.count == layerCount else {
            throw DenseContinuousBatchRuntimeError.cacheLayerMismatch(
                id, expected: layerCount, actual: caches.count)
        }
        for cache in caches {
            let actual = Int(cache.offsetArr.item(Int32.self))
            guard actual == expected else {
                throw DenseContinuousBatchRuntimeError.cacheLengthMismatch(
                    id, expected: expected, actual: actual)
            }
        }
    }

    private func validateRequest(
        id: BatchRequestID,
        promptTokens: [Int],
        totalPromptTokens: Int,
        maxOutputTokens: Int
    ) throws {
        guard totalPromptTokens > 0, maxOutputTokens > 0 else {
            throw DenseContinuousBatchRuntimeError.invalidPrefillMetadata(id)
        }
        let (requested, overflow) = totalPromptTokens.addingReportingOverflow(maxOutputTokens)
        guard !overflow, requested <= maxContextTokens else {
            throw DenseContinuousBatchRuntimeError.contextLimitExceeded(
                id, requested: overflow ? Int.max : requested, limit: maxContextTokens)
        }
        for token in promptTokens where token < 0 || token >= vocabularySize {
            throw DenseContinuousBatchRuntimeError.invalidTokenID(id, token)
        }
    }

    private func roundedCapacity(
        promptTokens: Int, outputTokens: Int, id: BatchRequestID
    ) throws -> Int {
        let (total, overflow) = promptTokens.addingReportingOverflow(outputTokens)
        guard !overflow, total > 0, total <= Int(Int32.max) else {
            throw DenseContinuousBatchRuntimeError.positionOverflow(id)
        }
        let (withRounding, roundingOverflow) = total.addingReportingOverflow(allocationChunk - 1)
        guard !roundingOverflow else {
            throw DenseContinuousBatchRuntimeError.positionOverflow(id)
        }
        let rounded = min(
            maxContextTokens,
            max(allocationChunk, (withRounding / allocationChunk) * allocationChunk))
        guard rounded <= Int(Int32.max) else {
            throw DenseContinuousBatchRuntimeError.positionOverflow(id)
        }
        return rounded
    }

    private func roundedGrowth(required: Int, capacity: Int) -> Int {
        let missing = required - capacity
        let chunked = ((missing + allocationChunk - 1) / allocationChunk) * allocationChunk
        return min(chunked, maxContextTokens - capacity)
    }
}
