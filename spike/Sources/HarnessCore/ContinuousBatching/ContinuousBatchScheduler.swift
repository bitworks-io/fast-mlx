import Foundation

public struct BatchRequestID: RawRepresentable, Hashable, Comparable, Sendable {
    public let rawValue: UInt64

    public init(_ rawValue: UInt64) {
        self.rawValue = rawValue
    }

    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Runtime-derived equivalence class for one exact shared decode forward.
///
/// The scheduler treats the value as opaque policy metadata. A runtime may use a fixed physical
/// KV width or another shape invariant; requests from different cohorts are never submitted in
/// the same model call. `unrestricted` preserves the behavior of runtimes whose result is
/// independent of shared tensor geometry. `isolated` keeps one request on solo decode.
public enum BatchDecodeCohort: Hashable, Sendable {
    case unrestricted
    case fixedKVCapacity(Int)
    case isolated(BatchRequestID)
}

public enum BatchArchitectureClass: String, CaseIterable, Sendable, Equatable {
    case denseAttention
    case mixtureOfExperts
    case hybridStateSpace
    case vision
    case diffusion
    case unknown
}

public struct BatchRequest: Sendable, Equatable {
    public let id: BatchRequestID
    public let promptTokenCount: Int
    public let maxOutputTokens: Int
    public let architecture: BatchArchitectureClass
    public let requestsSpeculation: Bool
    public let decodeCohort: BatchDecodeCohort

    public init(
        id: BatchRequestID,
        promptTokenCount: Int,
        maxOutputTokens: Int,
        architecture: BatchArchitectureClass,
        requestsSpeculation: Bool = false,
        decodeCohort: BatchDecodeCohort = .unrestricted
    ) {
        self.id = id
        self.promptTokenCount = promptTokenCount
        self.maxOutputTokens = maxOutputTokens
        self.architecture = architecture
        self.requestsSpeculation = requestsSpeculation
        self.decodeCohort = decodeCohort
    }
}

public struct ContinuousBatchConfiguration: Sendable, Equatable {
    public let maxActiveSlots: Int
    public let maxPrefillSlots: Int
    public let prefillChunkSize: Int
    public let maxQueuedRequests: Int

    public init(
        maxActiveSlots: Int,
        maxPrefillSlots: Int,
        prefillChunkSize: Int,
        maxQueuedRequests: Int = 256
    ) throws {
        guard maxActiveSlots > 0 else {
            throw ContinuousBatchSchedulerError.invalidMaxActiveSlots(maxActiveSlots)
        }
        guard maxPrefillSlots > 0 else {
            throw ContinuousBatchSchedulerError.invalidMaxPrefillSlots(maxPrefillSlots)
        }
        guard maxPrefillSlots <= maxActiveSlots else {
            throw ContinuousBatchSchedulerError.prefillSlotsExceedActive(
                prefill: maxPrefillSlots,
                active: maxActiveSlots
            )
        }
        guard prefillChunkSize > 0 else {
            throw ContinuousBatchSchedulerError.invalidPrefillChunkSize(prefillChunkSize)
        }
        guard maxQueuedRequests > 0 else {
            throw ContinuousBatchSchedulerError.invalidMaxQueuedRequests(maxQueuedRequests)
        }
        self.maxActiveSlots = maxActiveSlots
        self.maxPrefillSlots = maxPrefillSlots
        self.prefillChunkSize = prefillChunkSize
        self.maxQueuedRequests = maxQueuedRequests
    }
}

public enum ContinuousBatchSchedulerError: Error, Sendable, Equatable {
    case invalidMaxActiveSlots(Int)
    case invalidMaxPrefillSlots(Int)
    case prefillSlotsExceedActive(prefill: Int, active: Int)
    case invalidPrefillChunkSize(Int)
    case invalidMaxQueuedRequests(Int)
    case queueCapacityExceeded(limit: Int)
    case duplicateRequest(BatchRequestID)
    case emptyPrompt(BatchRequestID)
    case invalidOutputBudget(BatchRequestID, Int)
    case unsupportedArchitecture(BatchRequestID, BatchArchitectureClass)
    case staleTick(expected: Int, actual: Int)
    case tickPlanMismatch(sequence: Int)
    case invalidDecodeOutcomeIDs(expected: [BatchRequestID], actual: [BatchRequestID])
    case invalidEmittedTokenCount(BatchRequestID, Int)
    case invalidPendingLookahead(BatchRequestID)
    case outputBudgetExceeded(BatchRequestID, attempted: Int, remaining: Int)
}

/// Cache/pipeline state left by the most recent solo decode action.
///
/// `pipelinedLookahead` means KV contains every emitted token and the runtime has the next
/// greedy token staged. `speculative` means the last emitted PLD bonus token is staged but is
/// not yet committed to KV. Both require a drain before shared-batch membership, but only the
/// speculative drain is allowed to be outputless because its staged token was already emitted.
public enum BatchSoloPipelineState: Sendable, Equatable {
    case canonical
    case pipelinedLookahead
    case speculative

    public var requiresDrain: Bool { self != .canonical }
}

public enum BatchSlotPhase: Sendable, Equatable {
    case queued
    case prefilling(processedTokens: Int, totalTokens: Int)
    case ready
    case decoding(
        emittedTokens: Int,
        soloPipelineState: BatchSoloPipelineState)
}

public struct BatchSlotSnapshot: Sendable, Equatable {
    public let request: BatchRequest
    public let phase: BatchSlotPhase
    public let arrivalOrdinal: UInt64
}

public struct BatchPrefillSlice: Sendable, Equatable {
    public let id: BatchRequestID
    public let startToken: Int
    public let count: Int

    public init(id: BatchRequestID, startToken: Int, count: Int) {
        self.id = id
        self.startToken = startToken
        self.count = count
    }

    public var endToken: Int { startToken + count }
}

public enum BatchDecodeAction: Sendable, Equatable {
    case drainSoloPipeline(BatchRequestID)
    case solo(BatchRequestID, speculationAllowed: Bool)
    case batch([BatchRequestID], speculationAllowed: Bool)

    fileprivate var requestIDs: [BatchRequestID] {
        switch self {
        case .drainSoloPipeline(let id), .solo(let id, _):
            [id]
        case .batch(let ids, _):
            ids
        }
    }
}

public enum BatchSchedulerOperation: Sendable, Equatable {
    case decode(BatchDecodeAction)
    case prefill(BatchPrefillSlice)
}

/// What the actor-confined executor actually produced for one slot's decode action. The
/// scheduler does not guess about speculative round width or lazy lookahead state: those are
/// executor facts required for output-budget accounting and the next transition decision.
public struct BatchDecodeOutcome: Sendable, Equatable {
    public let id: BatchRequestID
    public let emittedTokenCount: Int
    public let finished: Bool
    public let soloPipelineState: BatchSoloPipelineState

    public var hasPendingSoloLookahead: Bool {
        soloPipelineState.requiresDrain
    }

    public init(
        id: BatchRequestID,
        emittedTokenCount: Int,
        finished: Bool,
        soloPipelineState: BatchSoloPipelineState
    ) {
        self.id = id
        self.emittedTokenCount = emittedTokenCount
        self.finished = finished
        self.soloPipelineState = soloPipelineState
    }

    /// Source-compatible bridge for existing non-speculative runtimes and fixtures.
    public init(
        id: BatchRequestID,
        emittedTokenCount: Int,
        finished: Bool,
        hasPendingSoloLookahead: Bool
    ) {
        self.init(
            id: id,
            emittedTokenCount: emittedTokenCount,
            finished: finished,
            soloPipelineState: hasPendingSoloLookahead
                ? .pipelinedLookahead
                : .canonical)
    }
}

public struct BatchTickPlan: Sendable, Equatable {
    public let sequence: Int
    public let admissions: [BatchRequestID]
    public let decode: BatchDecodeAction?
    public let prefills: [BatchPrefillSlice]

    public init(
        sequence: Int,
        admissions: [BatchRequestID],
        decode: BatchDecodeAction?,
        prefills: [BatchPrefillSlice]
    ) {
        self.sequence = sequence
        self.admissions = admissions
        self.decode = decode
        self.prefills = prefills
    }

    /// The executor must preserve this order. Decode is always submitted before bounded
    /// prompt work so a large prefill cannot become head-of-line blocking for active streams.
    public var operations: [BatchSchedulerOperation] {
        var result: [BatchSchedulerOperation] = []
        if let decode {
            result.append(.decode(decode))
        }
        result.append(contentsOf: prefills.map(BatchSchedulerOperation.prefill))
        return result
    }
}

public enum BatchCancellationResult: Sendable, Equatable {
    case cancelled(id: BatchRequestID, previousPhase: BatchSlotPhase)
    case notFound(BatchRequestID)

    public var id: BatchRequestID {
        switch self {
        case .cancelled(let id, _), .notFound(let id):
            id
        }
    }
}

/// Pure scheduling policy for continuous batching. It owns no model or cache values; the
/// inference actor executes a plan's ordered operations and applies the successful plan back
/// to this reducer. Keeping policy here makes transition, fairness, and cancellation rules
/// testable without importing MLX.
public struct ContinuousBatchScheduler: Sendable {
    private struct Slot: Sendable, Equatable {
        let request: BatchRequest
        var phase: BatchSlotPhase
        let arrivalOrdinal: UInt64
    }

    public let configuration: ContinuousBatchConfiguration
    private var slots: [BatchRequestID: Slot] = [:]
    private var queue: [BatchRequestID] = []
    private var nextArrivalOrdinal: UInt64 = 0
    private var lastDecodedCohort: BatchDecodeCohort?
    /// Monotonic state revision, not merely a count of executed ticks. Submit and cancellation
    /// also advance it so stale executor work cannot attach to a same-ID replacement request.
    private var stateRevision = 0

    public init(configuration: ContinuousBatchConfiguration) {
        self.configuration = configuration
    }

    public var isEmpty: Bool { slots.isEmpty }

    public var queuedRequestIDs: [BatchRequestID] { queue }

    public var activeRequestIDs: [BatchRequestID] {
        orderedSlots(where: { $0.phase != .queued }).map(\.request.id)
    }

    public var snapshots: [BatchSlotSnapshot] {
        slots.values
            .sorted { $0.arrivalOrdinal < $1.arrivalOrdinal }
            .map {
                BatchSlotSnapshot(
                    request: $0.request,
                    phase: $0.phase,
                    arrivalOrdinal: $0.arrivalOrdinal
                )
            }
    }

    public func snapshot(for id: BatchRequestID) -> BatchSlotSnapshot? {
        guard let slot = slots[id] else { return nil }
        return BatchSlotSnapshot(
            request: slot.request,
            phase: slot.phase,
            arrivalOrdinal: slot.arrivalOrdinal
        )
    }

    public mutating func submit(_ request: BatchRequest) throws {
        guard slots[request.id] == nil else {
            throw ContinuousBatchSchedulerError.duplicateRequest(request.id)
        }
        guard request.promptTokenCount > 0 else {
            throw ContinuousBatchSchedulerError.emptyPrompt(request.id)
        }
        guard request.maxOutputTokens > 0 else {
            throw ContinuousBatchSchedulerError.invalidOutputBudget(
                request.id,
                request.maxOutputTokens
            )
        }
        guard request.architecture == .denseAttention else {
            throw ContinuousBatchSchedulerError.unsupportedArchitecture(
                request.id,
                request.architecture
            )
        }
        guard queue.count < configuration.maxQueuedRequests else {
            throw ContinuousBatchSchedulerError.queueCapacityExceeded(
                limit: configuration.maxQueuedRequests)
        }

        slots[request.id] = Slot(
            request: request,
            phase: .queued,
            arrivalOrdinal: nextArrivalOrdinal
        )
        nextArrivalOrdinal += 1
        queue.append(request.id)
        stateRevision += 1
    }

    public mutating func cancel(_ id: BatchRequestID) -> BatchCancellationResult {
        guard let slot = slots.removeValue(forKey: id) else {
            return .notFound(id)
        }
        queue.removeAll { $0 == id }
        if slots.isEmpty {
            lastDecodedCohort = nil
        }
        stateRevision += 1
        return .cancelled(id: id, previousPhase: slot.phase)
    }

    /// Produce one deterministic scheduling decision without mutating state. Applying the
    /// exact plan advances successful work and increments the plan sequence.
    public func makeTick() -> BatchTickPlan {
        let activeCount = slots.values.reduce(into: 0) { count, slot in
            if slot.phase != .queued { count += 1 }
        }
        let currentPrefillCount = slots.values.reduce(into: 0) { count, slot in
            if case .prefilling = slot.phase { count += 1 }
        }
        let admissionCount = min(
            max(0, configuration.maxActiveSlots - activeCount),
            max(0, configuration.maxPrefillSlots - currentPrefillCount),
            queue.count
        )
        let admissions = Array(queue.prefix(admissionCount))

        var projected = slots
        for id in admissions {
            guard var slot = projected[id] else { continue }
            slot.phase = .prefilling(
                processedTokens: 0,
                totalTokens: slot.request.promptTokenCount
            )
            projected[id] = slot
        }

        let decodeCandidates = projected.values
            .filter {
                switch $0.phase {
                case .ready, .decoding:
                    true
                case .queued, .prefilling:
                    false
                }
            }
            .sorted { $0.arrivalOrdinal < $1.arrivalOrdinal }

        var orderedCohorts: [BatchDecodeCohort] = []
        for candidate in decodeCandidates
        where !orderedCohorts.contains(candidate.request.decodeCohort) {
            orderedCohorts.append(candidate.request.decodeCohort)
        }
        let selectedCohort: BatchDecodeCohort? = {
            guard !orderedCohorts.isEmpty else { return nil }
            guard let lastDecodedCohort,
                let previous = orderedCohorts.firstIndex(of: lastDecodedCohort),
                orderedCohorts.count > 1
            else {
                return orderedCohorts[0]
            }
            return orderedCohorts[(previous + 1) % orderedCohorts.count]
        }()
        let selectedDecodeCandidates = decodeCandidates.filter {
            $0.request.decodeCohort == selectedCohort
        }

        let decode: BatchDecodeAction?
        if selectedDecodeCandidates.count >= 2 {
            if let pending = selectedDecodeCandidates.first(where: {
                if case .decoding(_, let state) = $0.phase {
                    return state.requiresDrain
                }
                return false
            }) {
                // A drain is a full decode action for the solo slot. The shared batch may begin
                // only after this plan is applied, making the ordering impossible to bypass.
                decode = .drainSoloPipeline(pending.request.id)
            } else {
                decode = .batch(
                    selectedDecodeCandidates.map(\.request.id),
                    speculationAllowed: false
                )
            }
        } else if let only = selectedDecodeCandidates.first {
            decode = .solo(
                only.request.id,
                speculationAllowed: only.request.requestsSpeculation
            )
        } else {
            decode = nil
        }

        let prefills = projected.values
            .compactMap { slot -> (Slot, Int)? in
                guard case .prefilling(let processed, let total) = slot.phase else {
                    return nil
                }
                return (slot, min(configuration.prefillChunkSize, total - processed))
            }
            .filter { $0.1 > 0 }
            .sorted { $0.0.arrivalOrdinal < $1.0.arrivalOrdinal }
            .prefix(configuration.maxPrefillSlots)
            .map { slot, count -> BatchPrefillSlice in
                guard case .prefilling(let processed, _) = slot.phase else {
                    preconditionFailure("projected prefill slot changed phase")
                }
                return BatchPrefillSlice(
                    id: slot.request.id,
                    startToken: processed,
                    count: count
                )
            }

        return BatchTickPlan(
            sequence: stateRevision,
            admissions: admissions,
            decode: decode,
            prefills: Array(prefills)
        )
    }

    /// Apply one successfully executed plan. Outcomes are explicit because a speculative solo
    /// round may emit multiple tokens and may or may not leave a submit-first lookahead. The
    /// reducer validates every outcome before mutation and enforces output budgets independently.
    public mutating func apply(
        _ plan: BatchTickPlan,
        decodeOutcomes: [BatchDecodeOutcome] = []
    ) throws {
        guard plan.sequence == stateRevision else {
            throw ContinuousBatchSchedulerError.staleTick(
                expected: stateRevision,
                actual: plan.sequence
            )
        }
        guard plan == makeTick() else {
            throw ContinuousBatchSchedulerError.tickPlanMismatch(sequence: plan.sequence)
        }

        let expectedOutcomeIDs = plan.decode?.requestIDs ?? []
        let actualOutcomeIDs = decodeOutcomes.map(\.id)
        guard actualOutcomeIDs.count == Set(actualOutcomeIDs).count,
            Set(actualOutcomeIDs) == Set(expectedOutcomeIDs)
        else {
            throw ContinuousBatchSchedulerError.invalidDecodeOutcomeIDs(
                expected: expectedOutcomeIDs,
                actual: actualOutcomeIDs
            )
        }

        let outcomeByID = Dictionary(
            uniqueKeysWithValues: decodeOutcomes.map { ($0.id, $0) })
        try validateDecodeOutcomes(
            plan.decode,
            expectedIDs: expectedOutcomeIDs,
            outcomes: outcomeByID
        )

        var next = self
        for id in plan.admissions {
            guard var slot = next.slots[id] else { continue }
            slot.phase = .prefilling(
                processedTokens: 0,
                totalTokens: slot.request.promptTokenCount
            )
            next.slots[id] = slot
        }
        let admitted = Set(plan.admissions)
        next.queue.removeAll { admitted.contains($0) }

        if let decode = plan.decode {
            if let firstID = decode.requestIDs.first {
                next.lastDecodedCohort = next.slots[firstID]?.request.decodeCohort
            }
            for id in decode.requestIDs {
                next.advanceDecode(outcomeByID[id]!)
            }
        }

        for slice in plan.prefills {
            guard var slot = next.slots[slice.id] else { continue }
            guard case .prefilling(let processed, let total) = slot.phase,
                processed == slice.startToken
            else {
                throw ContinuousBatchSchedulerError.tickPlanMismatch(sequence: plan.sequence)
            }
            let newProcessed = slice.endToken
            slot.phase = newProcessed >= total
                ? .ready
                : .prefilling(processedTokens: newProcessed, totalTokens: total)
            next.slots[slice.id] = slot
        }

        if next.slots.isEmpty {
            next.lastDecodedCohort = nil
        }
        next.stateRevision += 1
        self = next
    }

    private func orderedSlots(where predicate: (Slot) -> Bool) -> [Slot] {
        slots.values
            .filter(predicate)
            .sorted { $0.arrivalOrdinal < $1.arrivalOrdinal }
    }

    private func validateDecodeOutcomes(
        _ action: BatchDecodeAction?,
        expectedIDs: [BatchRequestID],
        outcomes: [BatchRequestID: BatchDecodeOutcome]
    ) throws {
        for id in expectedIDs {
            guard let slot = slots[id], let outcome = outcomes[id] else { continue }
            let previousSoloState: BatchSoloPipelineState = {
                guard case .decoding(_, let state) = slot.phase else {
                    return .canonical
                }
                return state
            }()
            let isOutputlessSpeculativeDrain =
                action == .drainSoloPipeline(id)
                && previousSoloState == .speculative
                && outcome.emittedTokenCount == 0
                && !outcome.finished
                && outcome.soloPipelineState == .canonical
            guard outcome.emittedTokenCount >= 0,
                outcome.emittedTokenCount > 0
                    || outcome.finished
                    || isOutputlessSpeculativeDrain
            else {
                throw ContinuousBatchSchedulerError.invalidEmittedTokenCount(
                    id,
                    outcome.emittedTokenCount
                )
            }

            let alreadyEmitted: Int
            switch slot.phase {
            case .ready:
                alreadyEmitted = 0
            case .decoding(let emitted, _):
                alreadyEmitted = emitted
            case .queued, .prefilling:
                alreadyEmitted = 0
            }
            let remaining = slot.request.maxOutputTokens - alreadyEmitted
            guard outcome.emittedTokenCount <= remaining else {
                throw ContinuousBatchSchedulerError.outputBudgetExceeded(
                    id,
                    attempted: outcome.emittedTokenCount,
                    remaining: remaining
                )
            }

            switch action {
            case .drainSoloPipeline, .batch:
                guard outcome.soloPipelineState == .canonical else {
                    throw ContinuousBatchSchedulerError.invalidPendingLookahead(id)
                }
                guard outcome.emittedTokenCount <= 1 else {
                    throw ContinuousBatchSchedulerError.invalidEmittedTokenCount(
                        id,
                        outcome.emittedTokenCount
                    )
                }
            case .solo(_, let speculationAllowed):
                if !speculationAllowed, outcome.emittedTokenCount > 1 {
                    throw ContinuousBatchSchedulerError.invalidEmittedTokenCount(
                        id,
                        outcome.emittedTokenCount
                    )
                }
                if !speculationAllowed,
                    outcome.soloPipelineState == .speculative
                {
                    throw ContinuousBatchSchedulerError.invalidPendingLookahead(id)
                }
            case nil:
                break
            }
        }
    }

    private mutating func advanceDecode(_ outcome: BatchDecodeOutcome) {
        guard var slot = slots[outcome.id] else { return }
        let previousEmitted: Int
        switch slot.phase {
        case .ready:
            previousEmitted = 0
        case .decoding(let emitted, _):
            previousEmitted = emitted
        case .queued, .prefilling:
            return
        }

        let emitted = previousEmitted + outcome.emittedTokenCount
        if outcome.finished || emitted >= slot.request.maxOutputTokens {
            slots.removeValue(forKey: outcome.id)
            queue.removeAll { $0 == outcome.id }
        } else {
            slot.phase = .decoding(
                emittedTokens: emitted,
                soloPipelineState: outcome.soloPipelineState
            )
            slots[outcome.id] = slot
        }
    }
}
