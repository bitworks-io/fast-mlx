import Foundation
import HarnessCore
import MLX
import MLXLMCommon

/// Unforgeable outside SpikeCore: created only by reading the model's checked-in config.
public struct DenseContinuousBatchModelProof: Sendable, Equatable {
    fileprivate let maxPositionEmbeddings: Int
    fileprivate let vocabularySize: Int
    fileprivate let layerCount: Int
    fileprivate let keyValueHeadCount: Int
    fileprivate let headDimension: Int
    fileprivate let elementBytes: Int

    private struct Configuration: Decodable {
        let modelType: String
        let maxPositionEmbeddings: Int
        let vocabularySize: Int
        let layerCount: Int?
        let keyValueHeadCount: Int?
        let headDimension: Int?
        let torchDtype: String?
        let dtype: String?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case maxPositionEmbeddings = "max_position_embeddings"
            case vocabularySize = "vocab_size"
            case layerCount = "num_hidden_layers"
            case keyValueHeadCount = "num_key_value_heads"
            case headDimension = "head_dim"
            case torchDtype = "torch_dtype"
            case dtype
        }
    }

    private init(
        maxPositionEmbeddings: Int,
        vocabularySize: Int,
        layerCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        elementBytes: Int
    ) {
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.vocabularySize = vocabularySize
        self.layerCount = layerCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimension = headDimension
        self.elementBytes = elementBytes
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
        guard let layerCount = configuration.layerCount,
            let keyValueHeadCount = configuration.keyValueHeadCount,
            let headDimension = configuration.headDimension
        else {
            throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        let elementBytes: Int
        switch configuration.torchDtype ?? configuration.dtype {
        case "float16", "bfloat16": elementBytes = 2
        case "float32": elementBytes = 4
        default: throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        guard configuration.maxPositionEmbeddings > 0,
            configuration.maxPositionEmbeddings <= Int(Int32.max),
            configuration.vocabularySize > 0,
            configuration.vocabularySize <= Int(Int32.max),
            layerCount > 0,
            keyValueHeadCount > 0,
            headDimension > 0
        else {
            throw DenseContinuousBatchRuntimeError.invalidModelConfiguration
        }
        return Self(
            maxPositionEmbeddings: configuration.maxPositionEmbeddings,
            vocabularySize: configuration.vocabularySize,
            layerCount: layerCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            elementBytes: elementBytes)
    }

    static func testing(
        maxPositionEmbeddings: Int,
        vocabularySize: Int,
        layerCount: Int = 1,
        keyValueHeadCount: Int = 1,
        headDimension: Int = 1,
        elementBytes: Int = 4
    ) -> Self {
        Self(
            maxPositionEmbeddings: maxPositionEmbeddings,
            vocabularySize: vocabularySize,
            layerCount: layerCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            elementBytes: elementBytes)
    }
}

public enum DenseContinuousBatchRuntimeError: Error, Equatable {
    case unsupportedModelFamily(String)
    case invalidModelConfiguration
    case invalidAllocationChunk(Int)
    case invalidContextLimit(Int)
    case invalidAggregateContextLimit(Int)
    case invalidAggregateKVByteLimit(Int)
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
    case aggregateKVByteLimitExceeded(requested: Int, limit: Int)
    case kvByteAccountingOverflow
    case modelLayerCountMismatch(expected: Int, actual: Int)
    case cacheGeometryMismatch(expectedHeads: Int, expectedDimension: Int, actual: [Int])
    case cacheElementSizeMismatch(expected: Int, actual: Int)
    case invalidTokenID(BatchRequestID, Int)
    case positionOverflow(BatchRequestID)
}

struct DenseContinuousBatchRuntimeDiagnostics: Equatable {
    let batchTraceCount: Int
    let batchMembership: [BatchRequestID]
    let batchCapacity: Int?
    let kvBytesPerToken: Int
    let reservedKVBytes: Int
    let maxReservedKVBytes: Int
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
    private let kvBytePlan: DenseKVByteAdmissionPlan
    private let expectedKVHeads: Int
    private let expectedKVHeadDimension: Int
    private let expectedKVElementBytes: Int
    private let maxReservedKVBytes: Int
    private var slots: [BatchRequestID: Slot] = [:]
    private var contextReservations: [BatchRequestID: Int] = [:]
    private var kvCapacityReservations: [BatchRequestID: Int] = [:]
    private var reservedKVBytes = 0
    private var cacheGeometryCalibrated = false
    private var batch: BatchState?

    package init(
        model: any LanguageModel,
        verifiedBy proof: DenseContinuousBatchModelProof,
        allocationChunk: Int = 256,
        maxContextTokens: Int? = nil,
        maxReservedContextTokens: Int? = nil,
        initialDecodeReserve: Int = 384,
        maxReservedKVBytes: Int? = nil
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
        guard layerCount == proof.layerCount else {
            throw DenseContinuousBatchRuntimeError.modelLayerCountMismatch(
                expected: proof.layerCount, actual: layerCount)
        }
        let kvBytePlan: DenseKVByteAdmissionPlan
        do {
            kvBytePlan = try DenseKVByteAdmissionPlan(
                layerCount: proof.layerCount,
                keyValueHeadCount: proof.keyValueHeadCount,
                headDimension: proof.headDimension,
                elementBytes: proof.elementBytes,
                allocationChunk: allocationChunk,
                maxContextTokens: resolvedContextLimit)
        } catch {
            throw DenseContinuousBatchRuntimeError.kvByteAccountingOverflow
        }
        let resolvedKVByteLimit = maxReservedKVBytes ?? Int.max
        guard resolvedKVByteLimit > 0 else {
            throw DenseContinuousBatchRuntimeError.invalidAggregateKVByteLimit(
                resolvedKVByteLimit)
        }
        self.model = model
        self.allocationChunk = allocationChunk
        self.maxContextTokens = resolvedContextLimit
        self.maxReservedContextTokens = resolvedReservationLimit
        self.initialDecodeReserve = initialDecodeReserve
        self.vocabularySize = proof.vocabularySize
        self.layerCount = layerCount
        self.kvBytePlan = kvBytePlan
        self.expectedKVHeads = proof.keyValueHeadCount
        self.expectedKVHeadDimension = proof.headDimension
        self.expectedKVElementBytes = proof.elementBytes
        self.maxReservedKVBytes = resolvedKVByteLimit
    }

    convenience init(
        testing model: any LanguageModel,
        allocationChunk: Int = 256,
        maxContextTokens: Int = 32_768,
        maxReservedContextTokens: Int? = nil,
        initialDecodeReserve: Int = 384,
        maxReservedKVBytes: Int? = nil
    ) throws {
        try self.init(
            model: model,
            verifiedBy: .testing(
                maxPositionEmbeddings: maxContextTokens,
                vocabularySize: 2_048),
            allocationChunk: allocationChunk,
            maxContextTokens: maxContextTokens,
            maxReservedContextTokens: maxReservedContextTokens,
            initialDecodeReserve: initialDecodeReserve,
            maxReservedKVBytes: maxReservedKVBytes)
    }

    public func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        var additions: [(BatchRequestID, tokens: Int, capacity: Int)] = []
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
            let capacity: Int
            do {
                capacity = try kvBytePlan.reservedCapacity(
                    promptTokens: submission.promptTokens.count,
                    outputTokens: submission.maxOutputTokens)
            } catch {
                throw DenseContinuousBatchRuntimeError.kvByteAccountingOverflow
            }
            additions.append(
                (admission.id, submission.promptTokens.count + submission.maxOutputTokens, capacity))
        }

        if !additions.isEmpty {
            try calibrateCacheGeometryIfNeeded()
        }

        var requested = contextReservations.values.reduce(0, +)
        for (_, tokens, _) in additions {
            let (next, overflow) = requested.addingReportingOverflow(tokens)
            guard !overflow, next <= maxReservedContextTokens else {
                throw DenseContinuousBatchRuntimeError.aggregateContextLimitExceeded(
                    requested: overflow ? Int.max : next,
                    limit: maxReservedContextTokens)
            }
            requested = next
        }
        var candidateCapacities = kvCapacityReservations
        for (id, _, capacity) in additions { candidateCapacities[id] = capacity }
        let requestedKVBytes: Int
        do {
            requestedKVBytes = try kvBytePlan.transitionEnvelopeBytes(
                capacities: Array(candidateCapacities.values))
        } catch {
            throw DenseContinuousBatchRuntimeError.kvByteAccountingOverflow
        }
        guard requestedKVBytes <= maxReservedKVBytes else {
            throw DenseContinuousBatchRuntimeError.aggregateKVByteLimitExceeded(
                requested: requestedKVBytes, limit: maxReservedKVBytes)
        }
        for (id, tokens, capacity) in additions {
            contextReservations[id] = tokens
            kvCapacityReservations[id] = capacity
        }
        reservedKVBytes = requestedKVBytes
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

        try validatePhysicalCacheGeometry(caches)

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

        guard let batch else {
            kvCapacityReservations[id] = nil
            recomputeReservedKVBytes()
            return
        }
        let removedBatchedRow = batch.ids.contains(id)
        let hasLiveBatchedRow = batch.ids.contains(where: { slots[$0] != nil })
        if !hasLiveBatchedRow {
            self.batch = nil
            pruneKVReservationsToLiveSlots()
        } else if !removedBatchedRow {
            kvCapacityReservations[id] = nil
            recomputeReservedKVBytes()
        }
        // A removed row of a still-live batch remains physically allocated until the next
        // membership rebuild. Keep its KV byte reservation until ensureBatch/ensureScalar
        // releases the old batch arrays.
    }

    func diagnostics() -> DenseContinuousBatchRuntimeDiagnostics {
        DenseContinuousBatchRuntimeDiagnostics(
            batchTraceCount: batch?.traceCounter.count ?? 0,
            batchMembership: batch?.ids ?? [],
            batchCapacity: batch?.caches.first?.capacity,
            kvBytesPerToken: kvBytePlan.bytesPerToken,
            reservedKVBytes: reservedKVBytes,
            maxReservedKVBytes: maxReservedKVBytes)
    }

    public func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: kvBytePlan.bytesPerToken,
            reservedKVBytes: reservedKVBytes,
            maxReservedKVBytes: maxReservedKVBytes)
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
        recordPhysicalKVCapacity(caches[0].capacity, for: [id])
        pruneKVReservationsToLiveSlots()
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
        recordPhysicalKVCapacity(caches[0].capacity, for: ids)
        pruneKVReservationsToLiveSlots()
    }

    private func calibrateCacheGeometryIfNeeded() throws {
        guard !cacheGeometryCalibrated else { return }

        let caches = (0 ..< layerCount).map { _ in CompiledKVCache(capacity: 1) }
        let modelCaches: [any KVCache] = caches.map { $0 }
        let token = MLXArray([Int32(0)]).reshaped([1, 1])
        let logits = model(token, cache: modelCaches)
        eval([logits] + caches.flatMap { $0.innerState() })
        try validatePhysicalCacheGeometry(caches)
        cacheGeometryCalibrated = true
    }

    private func validatePhysicalCacheGeometry(_ caches: [CompiledKVCache]) throws {
        guard caches.count == layerCount else {
            throw DenseContinuousBatchRuntimeError.modelLayerCountMismatch(
                expected: layerCount, actual: caches.count)
        }
        for cache in caches {
            guard let keys = cache.keysBuf, let values = cache.valuesBuf else {
                throw DenseContinuousBatchRuntimeError.cacheGeometryMismatch(
                    expectedHeads: expectedKVHeads,
                    expectedDimension: expectedKVHeadDimension,
                    actual: [])
            }
            guard keys.shape.count == 4,
                values.shape == keys.shape,
                keys.dim(1) == expectedKVHeads,
                keys.dim(3) == expectedKVHeadDimension
            else {
                throw DenseContinuousBatchRuntimeError.cacheGeometryMismatch(
                    expectedHeads: expectedKVHeads,
                    expectedDimension: expectedKVHeadDimension,
                    actual: keys.shape + [-1] + values.shape)
            }
            guard keys.itemSize == expectedKVElementBytes,
                values.itemSize == expectedKVElementBytes
            else {
                throw DenseContinuousBatchRuntimeError.cacheElementSizeMismatch(
                    expected: expectedKVElementBytes,
                    actual: keys.itemSize == expectedKVElementBytes
                        ? values.itemSize : keys.itemSize)
            }
        }
    }

    private func pruneKVReservationsToLiveSlots() {
        kvCapacityReservations = kvCapacityReservations.filter { slots[$0.key] != nil }
        recomputeReservedKVBytes()
    }

    private func recomputeReservedKVBytes() {
        reservedKVBytes = (try? kvBytePlan.transitionEnvelopeBytes(
            capacities: Array(kvCapacityReservations.values))) ?? maxReservedKVBytes
    }

    private func recordPhysicalKVCapacity(
        _ capacity: Int,
        for ids: [BatchRequestID]
    ) {
        var changed = false
        for id in ids {
            guard let reserved = kvCapacityReservations[id], capacity > reserved else { continue }
            kvCapacityReservations[id] = capacity
            changed = true
        }
        if changed { recomputeReservedKVBytes() }
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
        recordPhysicalKVCapacity(caches[0].capacity, for: [id])
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
        recordPhysicalKVCapacity(batch.caches[0].capacity, for: batch.ids)
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
