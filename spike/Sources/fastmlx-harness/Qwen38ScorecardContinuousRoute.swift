import Foundation

import HarnessCore
import ServingCore
import SpikeServingAdapters

public enum Qwen38ScorecardContinuousRouteError:
    Error, Equatable, CustomStringConvertible, Sendable
{
    case invalidConcurrency(Int)
    case missingAdmissionObservation(expected: Int, actual: Int)
    case duplicateCoordinatorRequestIDs
    case missingSharedBatchDecode
    case speculationEnabled
    case noRequestOverlap
    case incompleteRequest(index: Int)
    case unexpectedRoute(index: Int)
    case malformedTokenText(String)
    case incompleteCleanup(
        activeRequests: Int,
        coordinatorSlots: Int,
        reservedKVBytes: Int)
    case occupancyExceeded(limit: Int, observed: Int)
    case inconsistentPeakSummary

    public var description: String {
        switch self {
        case .invalidConcurrency(let value):
            return "invalid concurrency \(value); expected exactly 2 or 4"
        case .missingAdmissionObservation(let expected, let actual):
            return "expected \(expected) admitted coordinator requests, observed \(actual)"
        case .duplicateCoordinatorRequestIDs:
            return "coordinator request IDs were not distinct"
        case .missingSharedBatchDecode:
            return "no shared continuous-batch decode contained the full cohort"
        case .speculationEnabled:
            return "shared multi-row speculation was enabled"
        case .noRequestOverlap:
            return "request lifetimes did not overlap"
        case .incompleteRequest(let index):
            return "request \(index) did not produce a completion"
        case .unexpectedRoute(let index):
            return "request \(index) did not use the continuous-batch route"
        case .malformedTokenText:
            return "fixture token text was malformed"
        case .incompleteCleanup(let active, let slots, let reserved):
            return "cleanup incomplete active=\(active) slots=\(slots) reservedKVBytes=\(reserved)"
        case .occupancyExceeded(let limit, let observed):
            return "observed occupancy \(observed) exceeds bounded limit \(limit)"
        case .inconsistentPeakSummary:
            return "reported peak occupancy does not match the plan observation trace"
        }
    }
}

public struct Qwen38ScorecardContinuousRouteRevision:
    Equatable, Sendable
{
    public let planSequence: Int
    public let stateRevisionAfterApply: Int

    public init(planSequence: Int, stateRevisionAfterApply: Int) {
        self.planSequence = planSequence
        self.stateRevisionAfterApply = stateRevisionAfterApply
    }
}

public enum Qwen38ScorecardContinuousRouteEvidenceKind:
    String, Equatable, Sendable
{
    case syntheticPathProof
}

public enum Qwen38ScorecardContinuousRouteDecodeKind:
    String, Equatable, Sendable
{
    case none
    case drainSoloPipeline
    case solo
    case batch
}

public struct Qwen38ScorecardContinuousRoutePlanObservation:
    Equatable, Sendable
{
    public let planSequence: Int
    public let stateRevisionAfterApply: Int
    public let admissions: [UInt64]
    public let decodeKind: Qwen38ScorecardContinuousRouteDecodeKind
    public let decodeRequestIDs: [UInt64]
    public let speculationAllowed: Bool
    public let prefillRequestIDs: [UInt64]
    public let activeSlotCount: Int
    public let queuedSlotCount: Int

    public init(
        planSequence: Int,
        stateRevisionAfterApply: Int,
        admissions: [UInt64],
        decodeKind: Qwen38ScorecardContinuousRouteDecodeKind,
        decodeRequestIDs: [UInt64],
        speculationAllowed: Bool,
        prefillRequestIDs: [UInt64],
        activeSlotCount: Int,
        queuedSlotCount: Int
    ) {
        self.planSequence = planSequence
        self.stateRevisionAfterApply = stateRevisionAfterApply
        self.admissions = admissions
        self.decodeKind = decodeKind
        self.decodeRequestIDs = decodeRequestIDs
        self.speculationAllowed = speculationAllowed
        self.prefillRequestIDs = prefillRequestIDs
        self.activeSlotCount = activeSlotCount
        self.queuedSlotCount = queuedSlotCount
    }
}

extension Qwen38ScorecardContinuousRoutePlanObservation {
    init(_ observation: ContinuousBatchPlanObservation) {
        let decodeKind: Qwen38ScorecardContinuousRouteDecodeKind
        let decodeRequestIDs: [UInt64]
        let speculationAllowed: Bool
        switch observation.decode {
        case nil:
            decodeKind = .none
            decodeRequestIDs = []
            speculationAllowed = false
        case .drainSoloPipeline(let requestID):
            decodeKind = .drainSoloPipeline
            decodeRequestIDs = [requestID.rawValue]
            speculationAllowed = false
        case .solo(let requestID, let allowed):
            decodeKind = .solo
            decodeRequestIDs = [requestID.rawValue]
            speculationAllowed = allowed
        case .batch(let requestIDs, let allowed):
            decodeKind = .batch
            decodeRequestIDs = requestIDs.map(\.rawValue)
            speculationAllowed = allowed
        }
        self.init(
            planSequence: observation.planSequence,
            stateRevisionAfterApply: observation.stateRevisionAfterApply,
            admissions: observation.admissions.map(\.rawValue),
            decodeKind: decodeKind,
            decodeRequestIDs: decodeRequestIDs,
            speculationAllowed: speculationAllowed,
            prefillRequestIDs: observation.prefillIDs.map(\.rawValue),
            activeSlotCount: observation.activeSlotCount,
            queuedSlotCount: observation.queuedSlotCount)
    }
}

public struct Qwen38ScorecardContinuousRouteRequestResult:
    Equatable, Sendable
{
    public let requestIndex: Int
    public let coordinatorRequestID: UInt64
    public let route: ServingExecutionRoute
    public let outputTokenIDs: [Int]
    public let finishReason: OpenAIChatFinishReason
    public let usage: OpenAIChatUsage
    public let admittedAtUptime: Double
    public let completedAtUptime: Double

    public init(
        requestIndex: Int,
        coordinatorRequestID: UInt64,
        route: ServingExecutionRoute,
        outputTokenIDs: [Int],
        finishReason: OpenAIChatFinishReason,
        usage: OpenAIChatUsage,
        admittedAtUptime: Double,
        completedAtUptime: Double
    ) {
        self.requestIndex = requestIndex
        self.coordinatorRequestID = coordinatorRequestID
        self.route = route
        self.outputTokenIDs = outputTokenIDs
        self.finishReason = finishReason
        self.usage = usage
        self.admittedAtUptime = admittedAtUptime
        self.completedAtUptime = completedAtUptime
    }
}

public struct Qwen38ScorecardContinuousRouteResult:
    Equatable, Sendable
{
    public let evidenceKind: Qwen38ScorecardContinuousRouteEvidenceKind
    public let concurrency: Int
    public let coordinatorRequestIDs: [UInt64]
    public let coordinatorPlanObservations:
        [Qwen38ScorecardContinuousRoutePlanObservation]
    public let planRevisions: [Qwen38ScorecardContinuousRouteRevision]
    public let sharedBatchDecodeRequestIDs: [UInt64]
    public let peakActiveSlots: Int
    public let peakBatchOccupancy: Int
    public let finalActiveRequests: Int
    public let finalCoordinatorSlots: Int
    public let finalReservedKVBytes: Int
    public let requests: [Qwen38ScorecardContinuousRouteRequestResult]

    public init(
        evidenceKind: Qwen38ScorecardContinuousRouteEvidenceKind,
        concurrency: Int,
        coordinatorRequestIDs: [UInt64],
        coordinatorPlanObservations:
            [Qwen38ScorecardContinuousRoutePlanObservation],
        planRevisions: [Qwen38ScorecardContinuousRouteRevision],
        sharedBatchDecodeRequestIDs: [UInt64],
        peakActiveSlots: Int,
        peakBatchOccupancy: Int,
        finalActiveRequests: Int,
        finalCoordinatorSlots: Int,
        finalReservedKVBytes: Int,
        requests: [Qwen38ScorecardContinuousRouteRequestResult]
    ) {
        self.evidenceKind = evidenceKind
        self.concurrency = concurrency
        self.coordinatorRequestIDs = coordinatorRequestIDs
        self.coordinatorPlanObservations = coordinatorPlanObservations
        self.planRevisions = planRevisions
        self.sharedBatchDecodeRequestIDs = sharedBatchDecodeRequestIDs
        self.peakActiveSlots = peakActiveSlots
        self.peakBatchOccupancy = peakBatchOccupancy
        self.finalActiveRequests = finalActiveRequests
        self.finalCoordinatorSlots = finalCoordinatorSlots
        self.finalReservedKVBytes = finalReservedKVBytes
        self.requests = requests
    }

    public func validate() throws {
        guard concurrency == 2 || concurrency == 4 else {
            throw Qwen38ScorecardContinuousRouteError.invalidConcurrency(concurrency)
        }
        guard coordinatorRequestIDs.count == concurrency,
            requests.count == concurrency
        else {
            throw Qwen38ScorecardContinuousRouteError.missingAdmissionObservation(
                expected: concurrency,
                actual: coordinatorRequestIDs.count)
        }
        guard Set(coordinatorRequestIDs).count == coordinatorRequestIDs.count else {
            throw Qwen38ScorecardContinuousRouteError.duplicateCoordinatorRequestIDs
        }
        guard requests.map(\.coordinatorRequestID).sorted()
            == coordinatorRequestIDs.sorted()
        else {
            throw Qwen38ScorecardContinuousRouteError.missingAdmissionObservation(
                expected: concurrency,
                actual: requests.count)
        }
        guard Set(sharedBatchDecodeRequestIDs) == Set(coordinatorRequestIDs),
            sharedBatchDecodeRequestIDs.count == concurrency
        else {
            throw Qwen38ScorecardContinuousRouteError.missingSharedBatchDecode
        }
        guard !coordinatorPlanObservations.isEmpty else {
            throw Qwen38ScorecardContinuousRouteError.missingSharedBatchDecode
        }
        guard coordinatorPlanObservations.allSatisfy({
            !$0.speculationAllowed
        }) else {
            throw Qwen38ScorecardContinuousRouteError.speculationEnabled
        }
        guard coordinatorPlanObservations.contains(where: {
            $0.decodeKind == .batch
                && Set($0.decodeRequestIDs) == Set(coordinatorRequestIDs)
                && $0.decodeRequestIDs.count == concurrency
        }) else {
            throw Qwen38ScorecardContinuousRouteError.missingSharedBatchDecode
        }
        guard peakActiveSlots <= concurrency,
            peakBatchOccupancy <= concurrency
        else {
            throw Qwen38ScorecardContinuousRouteError.occupancyExceeded(
                limit: concurrency,
                observed: max(peakActiveSlots, peakBatchOccupancy))
        }
        let observedPeakActiveSlots = coordinatorPlanObservations
            .map(\.activeSlotCount)
            .max() ?? 0
        let observedPeakBatchOccupancy = coordinatorPlanObservations.compactMap {
            $0.decodeKind == .batch ? $0.decodeRequestIDs.count : nil
        }.max() ?? 0
        guard peakActiveSlots == observedPeakActiveSlots,
            peakBatchOccupancy == observedPeakBatchOccupancy
        else {
            throw Qwen38ScorecardContinuousRouteError.inconsistentPeakSummary
        }
        guard planRevisions.allSatisfy({
            $0.planSequence > 0 && $0.stateRevisionAfterApply > 0
        }) else {
            throw Qwen38ScorecardContinuousRouteError.missingSharedBatchDecode
        }
        guard planRevisions
            == coordinatorPlanObservations.map({
                Qwen38ScorecardContinuousRouteRevision(
                    planSequence: $0.planSequence,
                    stateRevisionAfterApply: $0.stateRevisionAfterApply)
            })
        else {
            throw Qwen38ScorecardContinuousRouteError.missingSharedBatchDecode
        }
        for (previous, next) in zip(planRevisions, planRevisions.dropFirst()) {
            guard next.planSequence > previous.planSequence,
                next.stateRevisionAfterApply > previous.stateRevisionAfterApply
            else {
                throw Qwen38ScorecardContinuousRouteError.missingSharedBatchDecode
            }
        }
        guard requests.allSatisfy({ !$0.outputTokenIDs.isEmpty }) else {
            let index = requests.first(where: { $0.outputTokenIDs.isEmpty })?
                .requestIndex ?? 0
            throw Qwen38ScorecardContinuousRouteError.incompleteRequest(index: index)
        }
        if let incomplete = requests.first(where: {
            $0.finishReason != .stop
                || $0.usage.completionTokens != $0.outputTokenIDs.count
        }) {
            throw Qwen38ScorecardContinuousRouteError.incompleteRequest(
                index: incomplete.requestIndex)
        }
        if let wrongRoute = requests.first(where: {
            $0.route != .continuousBatchNoSpec
        }) {
            throw Qwen38ScorecardContinuousRouteError.unexpectedRoute(
                index: wrongRoute.requestIndex)
        }
        let latestAdmission = requests.map(\.admittedAtUptime).max() ?? 0
        let earliestCompletion = requests.map(\.completedAtUptime).min() ?? 0
        guard latestAdmission < earliestCompletion else {
            throw Qwen38ScorecardContinuousRouteError.noRequestOverlap
        }
        guard finalActiveRequests == 0,
            finalCoordinatorSlots == 0,
            finalReservedKVBytes == 0
        else {
            throw Qwen38ScorecardContinuousRouteError.incompleteCleanup(
                activeRequests: finalActiveRequests,
                coordinatorSlots: finalCoordinatorSlots,
                reservedKVBytes: finalReservedKVBytes)
        }
    }
}

public enum Qwen38ScorecardContinuousRouteRunner {
    public static func runFixture(
        concurrency: Int
    ) async throws -> Qwen38ScorecardContinuousRouteResult {
        guard concurrency == 2 || concurrency == 4 else {
            throw Qwen38ScorecardContinuousRouteError.invalidConcurrency(concurrency)
        }

        let runtime = FixtureContinuousRouteRuntime(concurrency: concurrency)
        let coordinator = ContinuousBatchCoordinator(
            configuration: try ContinuousBatchConfiguration(
                maxActiveSlots: concurrency,
                maxPrefillSlots: concurrency,
                prefillChunkSize: 8,
                maxQueuedRequests: concurrency),
            runtime: runtime,
            automaticDrive: false,
            publicationCapacity: 8,
            traceLimit: 128)
        let backend = ContinuousServingBackend(
            launchedModel: FixtureContinuousRouteRuntime.model,
            coordinator: coordinator,
            codec: FixtureContinuousRouteCodec(concurrency: concurrency),
            stopTokenIDs: [FixtureContinuousRouteRuntime.stopToken],
            modelStopStrings: [],
            configuration: ContinuousServingBackendConfiguration(
                defaultMaximumCompletionTokens: 4,
                queueRetryAfterSeconds: 2,
                mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096)))

        var drainTask: Task<[Qwen38ScorecardContinuousRouteRequestResult], Error>?
        do {
            var handles: [(Int, ServingGenerationHandle, Double)] = []
            handles.reserveCapacity(concurrency)
            for index in 0 ..< concurrency {
                let handle = try await backend.start(request(index: index))
                handles.append((index, handle, ProcessInfo.processInfo.systemUptime))
            }

            let admission = await backend.scorecardContinuousRouteObservation()
            let ids = admission.coordinatorRequestIDs.map(\.rawValue).sorted()
            guard ids.count == concurrency else {
                throw Qwen38ScorecardContinuousRouteError.missingAdmissionObservation(
                    expected: concurrency,
                    actual: ids.count)
            }
            guard Set(ids).count == ids.count else {
                throw Qwen38ScorecardContinuousRouteError.duplicateCoordinatorRequestIDs
            }

            let requestIDByIndex = Dictionary(
                uniqueKeysWithValues: zip(0 ..< concurrency, ids))
            drainTask = Task {
                try await withThrowingTaskGroup(
                    of: Qwen38ScorecardContinuousRouteRequestResult.self
                ) { group in
                    for (index, handle, admittedAt) in handles {
                        let coordinatorID = requestIDByIndex[index]!
                        group.addTask {
                            try await drain(
                                index: index,
                                coordinatorRequestID: coordinatorID,
                                handle: handle,
                                admittedAtUptime: admittedAt)
                        }
                    }

                    var results: [Qwen38ScorecardContinuousRouteRequestResult] = []
                    results.reserveCapacity(concurrency)
                    for try await result in group {
                        results.append(result)
                    }
                    return results.sorted { $0.requestIndex < $1.requestIndex }
                }
            }

            while try await coordinator.runOneTick() {
                await Task.yield()
            }
            let requestResults = try await drainTask!.value

            let final = try await waitForCleanup(backend)
            let result = try buildResult(
                concurrency: concurrency,
                coordinatorRequestIDs: ids,
                observation: final,
                requests: requestResults)
            try result.validate()
            await backend.shutdown()
            return result
        } catch {
            drainTask?.cancel()
            await backend.shutdown()
            if let drainTask {
                _ = try? await drainTask.value
            }
            throw error
        }
    }

    private static func request(index: Int) -> OpenAIChatCompletionRequest {
        OpenAIChatCompletionRequest(
            model: FixtureContinuousRouteRuntime.model,
            messages: [
                OpenAIChatMessage(
                    role: .user,
                    text: FixtureContinuousRouteRuntime.promptText(index: index)),
            ],
            maxCompletionTokens: 4,
            temperature: 0,
            choiceCount: 1,
            stream: true,
            stop: [])
    }

    private static func drain(
        index: Int,
        coordinatorRequestID: UInt64,
        handle: ServingGenerationHandle,
        admittedAtUptime: Double
    ) async throws -> Qwen38ScorecardContinuousRouteRequestResult {
        var outputTokens: [Int] = []
        var completion: ServingGenerationCompletion?
        while let delta = try await handle.mailbox.next() {
            switch delta {
            case .text(let text):
                guard let token = Int(text) else {
                    throw Qwen38ScorecardContinuousRouteError.malformedTokenText(text)
                }
                outputTokens.append(token)
            case .toolCalls:
                throw Qwen38ScorecardContinuousRouteError.incompleteRequest(index: index)
            case .completion(let value):
                completion = value
            }
        }
        guard let completion else {
            throw Qwen38ScorecardContinuousRouteError.incompleteRequest(index: index)
        }
        return Qwen38ScorecardContinuousRouteRequestResult(
            requestIndex: index,
            coordinatorRequestID: coordinatorRequestID,
            route: handle.route,
            outputTokenIDs: outputTokens,
            finishReason: completion.finishReason,
            usage: completion.usage,
            admittedAtUptime: admittedAtUptime,
            completedAtUptime: ProcessInfo.processInfo.systemUptime)
    }

    private static func waitForCleanup(
        _ backend: ContinuousServingBackend
    ) async throws -> ContinuousServingBackendScorecardObservation {
        var last = await backend.scorecardContinuousRouteObservation()
        for _ in 0 ..< 100
        where last.activeRequests != 0
            || !last.coordinatorSlots.isEmpty
            || (last.runtimeResources?.reservedKVBytes ?? 0) != 0
        {
            await Task.yield()
            last = await backend.scorecardContinuousRouteObservation()
        }
        return await backend.takeScorecardContinuousRouteObservation()
    }

    private static func buildResult(
        concurrency: Int,
        coordinatorRequestIDs ids: [UInt64],
        observation: ContinuousServingBackendScorecardObservation,
        requests: [Qwen38ScorecardContinuousRouteRequestResult]
    ) throws -> Qwen38ScorecardContinuousRouteResult {
        let planRevisions = observation.planObservations.map {
            Qwen38ScorecardContinuousRouteRevision(
                planSequence: $0.planSequence,
                stateRevisionAfterApply: $0.stateRevisionAfterApply)
        }
        let coordinatorPlanObservations = observation.planObservations.map {
            Qwen38ScorecardContinuousRoutePlanObservation($0)
        }
        var sharedBatch: [UInt64] = []
        var peakBatchOccupancy = 0
        for plan in observation.planObservations {
            guard let decode = plan.decode else { continue }
            if decode.speculationAllowed {
                throw Qwen38ScorecardContinuousRouteError.speculationEnabled
            }
            let decodeIDs = decode.requestIDs.map(\.rawValue)
            if case .batch = decode {
                peakBatchOccupancy = max(peakBatchOccupancy, decodeIDs.count)
                if Set(decodeIDs) == Set(ids), decodeIDs.count == concurrency {
                    sharedBatch = decodeIDs
                }
            }
        }

        return Qwen38ScorecardContinuousRouteResult(
            evidenceKind: .syntheticPathProof,
            concurrency: concurrency,
            coordinatorRequestIDs: ids,
            coordinatorPlanObservations: coordinatorPlanObservations,
            planRevisions: planRevisions,
            sharedBatchDecodeRequestIDs: sharedBatch,
            peakActiveSlots: observation.planObservations
                .map(\.activeSlotCount).max() ?? 0,
            peakBatchOccupancy: peakBatchOccupancy,
            finalActiveRequests: observation.activeRequests,
            finalCoordinatorSlots: observation.coordinatorSlots.count,
            finalReservedKVBytes: observation.runtimeResources?.reservedKVBytes ?? 0,
            requests: requests)
    }
}

private final class FixtureContinuousRouteRuntime: ContinuousBatchRuntime {
    struct Slot {
        var processedTokens = 0
        var ready = false
        var outputCursor = 0
        let outputTokens: [Int]
    }

    static let model = "qwen38-scorecard-continuous-fixture"
    static let stopToken = 90_000

    private var slots: [BatchRequestID: Slot] = [:]
    private let promptHeadToOutputTokens: [Int: [Int]]

    init(concurrency: Int) {
        var scripts: [Int: [Int]] = [:]
        for index in 0 ..< concurrency {
            scripts[Self.promptHead(index: index)] = [
                Self.outputToken(index: index),
                Self.stopToken,
            ]
        }
        self.promptHeadToOutputTokens = scripts
    }

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        for admission in admissions {
            guard let head = admission.submission.promptTokens.first,
                let outputTokens = promptHeadToOutputTokens[head]
            else {
                throw Qwen38ScorecardContinuousRouteError.incompleteRequest(
                    index: 0)
            }
            slots[admission.id] = Slot(outputTokens: outputTokens)
        }
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 64,
            reservedKVBytes: slots.count * 64,
            maxReservedKVBytes: 4_096)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        guard var slot = slots[work.id],
            work.startToken == slot.processedTokens
        else {
            throw Qwen38ScorecardContinuousRouteError.incompleteRequest(index: 0)
        }
        slot.processedTokens += work.tokens.count
        slot.ready = work.isFinal
        slots[work.id] = slot
    }

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        let ids: [BatchRequestID]
        switch action {
        case .drainSoloPipeline:
            throw Qwen38ScorecardContinuousRouteError.missingSharedBatchDecode
        case .solo:
            throw Qwen38ScorecardContinuousRouteError.missingSharedBatchDecode
        case .batch(let requestIDs, let speculationAllowed):
            guard !speculationAllowed else {
                throw Qwen38ScorecardContinuousRouteError.speculationEnabled
            }
            ids = requestIDs
        }
        return try ids.map { id in
            guard var slot = slots[id], slot.ready else {
                throw Qwen38ScorecardContinuousRouteError.incompleteRequest(index: 0)
            }
            guard slot.outputCursor < slot.outputTokens.count else {
                return ContinuousBatchRuntimeDecodeResult(
                    id: id,
                    tokens: [],
                    finished: true,
                    soloPipelineState: .canonical)
            }
            let token = slot.outputTokens[slot.outputCursor]
            slot.outputCursor += 1
            slots[id] = slot
            return ContinuousBatchRuntimeDecodeResult(
                id: id,
                tokens: [token],
                finished: false,
                soloPipelineState: .canonical)
        }
    }

    func remove(_ id: BatchRequestID) {
        slots[id] = nil
    }

    static func promptText(index: Int) -> String {
        "fixture-request-\(index)"
    }

    static func promptHead(index: Int) -> Int {
        1_000 + index
    }

    static func outputToken(index: Int) -> Int {
        2_000 + index
    }
}

private struct FixtureContinuousRouteCodec: ScalarServingTextCodec {
    let promptByText: [String: [Int]]

    init(concurrency: Int) {
        var prompts: [String: [Int]] = [:]
        for index in 0 ..< concurrency {
            prompts[FixtureContinuousRouteRuntime.promptText(index: index)] = [
                FixtureContinuousRouteRuntime.promptHead(index: index),
            ]
        }
        self.promptByText = prompts
    }

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        guard let text = messages.last?.text,
            let prompt = promptByText[text]
        else {
            throw Qwen38ScorecardContinuousRouteError.incompleteRequest(index: 0)
        }
        return prompt
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        FixtureContinuousRouteDetokenizer()
    }
}

private struct FixtureContinuousRouteDetokenizer: ScalarServingDetokenizer {
    private var pending: Int?

    mutating func append(token: Int) {
        pending = token
    }

    mutating func next() -> String? {
        defer { pending = nil }
        return pending.map(String.init)
    }
}
