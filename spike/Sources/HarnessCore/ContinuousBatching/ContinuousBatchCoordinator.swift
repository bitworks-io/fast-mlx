import Foundation

/// Value-only prompt work passed from the coordinator to its actor-confined runtime.
public struct ContinuousBatchRuntimePrefill: Sendable, Equatable {
    public let id: BatchRequestID
    public let startToken: Int
    public let tokens: [Int]
    public let isFinal: Bool
    public let totalPromptTokens: Int
    public let maxOutputTokens: Int

    public init(
        id: BatchRequestID,
        startToken: Int,
        tokens: [Int],
        isFinal: Bool,
        totalPromptTokens: Int,
        maxOutputTokens: Int
    ) {
        self.id = id
        self.startToken = startToken
        self.tokens = tokens
        self.isFinal = isFinal
        self.totalPromptTokens = totalPromptTokens
        self.maxOutputTokens = maxOutputTokens
    }
}

/// One runtime result before stream EOS/budget trimming and scheduler accounting.
public struct ContinuousBatchRuntimeDecodeResult: Sendable, Equatable {
    public let id: BatchRequestID
    public let tokens: [Int]
    public let finished: Bool
    public let hasPendingSoloLookahead: Bool

    public init(
        id: BatchRequestID,
        tokens: [Int],
        finished: Bool,
        hasPendingSoloLookahead: Bool
    ) {
        self.id = id
        self.tokens = tokens
        self.finished = finished
        self.hasPendingSoloLookahead = hasPendingSoloLookahead
    }
}

public struct ContinuousBatchRuntimeAdmission: Sendable, Equatable {
    public let id: BatchRequestID
    public let submission: ContinuousBatchSubmission

    public init(id: BatchRequestID, submission: ContinuousBatchSubmission) {
        self.id = id
        self.submission = submission
    }
}

/// Value-only resource state exported by an actor-confined runtime for service evidence.
public struct ContinuousBatchRuntimeResourceSnapshot: Sendable, Equatable, Codable {
    public let kvBytesPerToken: Int
    public let reservedKVBytes: Int
    public let maxReservedKVBytes: Int

    public init(
        kvBytesPerToken: Int,
        reservedKVBytes: Int,
        maxReservedKVBytes: Int
    ) {
        self.kvBytesPerToken = kvBytesPerToken
        self.reservedKVBytes = reservedKVBytes
        self.maxReservedKVBytes = maxReservedKVBytes
    }
}

/// Synchronous runtime seam owned by `ContinuousBatchCoordinator`.
///
/// The protocol deliberately is not `Sendable`: a production implementation owns MLX
/// arrays, caches, and compiled functions. A `sending` initializer transfers that whole
/// isolation region into the coordinator actor, and no runtime value crosses back out.
public protocol ContinuousBatchRuntime: AnyObject {
    /// Return the exact shared-forward cohort for a candidate admission without mutating
    /// runtime state. The coordinator asks before committing scheduler admission, then calls
    /// `admit(_:)` atomically for the same value-only request set.
    func decodeCohort(
        for admission: ContinuousBatchRuntimeAdmission
    ) throws -> BatchDecodeCohort
    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws
    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot?
    func prefill(_ work: ContinuousBatchRuntimePrefill) throws
    func decode(_ action: BatchDecodeAction) throws -> [ContinuousBatchRuntimeDecodeResult]
    func remove(_ id: BatchRequestID)
}

extension ContinuousBatchRuntime {
    /// Runtimes with no additional capability or resource gate can accept scheduler-valid work.
    public func decodeCohort(
        for admission: ContinuousBatchRuntimeAdmission
    ) throws -> BatchDecodeCohort {
        .unrestricted
    }
    public func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {}
    public func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? { nil }
}

public struct ContinuousBatchSubmission: Sendable, Equatable {
    public let promptTokens: [Int]
    public let maxOutputTokens: Int
    public let stopTokenIDs: Set<Int>
    public let architecture: BatchArchitectureClass
    public let requestsSpeculation: Bool

    public init(
        promptTokens: [Int],
        maxOutputTokens: Int,
        stopTokenIDs: Set<Int>,
        architecture: BatchArchitectureClass,
        requestsSpeculation: Bool = false
    ) {
        self.promptTokens = promptTokens
        self.maxOutputTokens = maxOutputTokens
        self.stopTokenIDs = stopTokenIDs
        self.architecture = architecture
        self.requestsSpeculation = requestsSpeculation
    }

    public init(
        promptTokens: [Int],
        maxOutputTokens: Int,
        eosToken: Int,
        architecture: BatchArchitectureClass,
        requestsSpeculation: Bool = false
    ) {
        self.init(
            promptTokens: promptTokens,
            maxOutputTokens: maxOutputTokens,
            stopTokenIDs: [eosToken],
            architecture: architecture,
            requestsSpeculation: requestsSpeculation)
    }
}

public struct ContinuousBatchRequestHandle: Sendable {
    public let id: BatchRequestID
    public let tokens: ContinuousBatchTokenStream

    public init(id: BatchRequestID, tokens: ContinuousBatchTokenStream) {
        self.id = id
        self.tokens = tokens
    }
}

public enum ContinuousBatchCoordinatorError: Error, Sendable, Equatable {
    case shuttingDown
    case requestIDExhausted
    case invalidStopTokenIDs
    case unknownRuntimeRequest(BatchRequestID)
}

/// Bounded, value-only observability for tests and later service telemetry.
public enum ContinuousBatchCoordinatorEvent: Sendable, Equatable {
    case operation(BatchSchedulerOperation)
    case cancelled(BatchCancellationResult)
    case finished(BatchRequestID)
    case failed(String)
}

/// Monotonic service-availability timestamps captured immediately before stream publication.
/// Kept separate from the operation trace so measurement events cannot evict exactness events.
public enum ContinuousBatchTimingEvent: Sendable, Equatable {
    case emitted(BatchRequestID, timestamp: Double)
    case finished(BatchRequestID, timestamp: Double)
}

/// MLX-free orchestration actor around the pure scheduler.
///
/// Publication capacity is reserved before a decode tick. After that reservation, one tick's
/// runtime work and scheduler commit form a transaction with no suspension point. Publishing
/// the already-committed result may suspend only after the MLX mutation window has closed.
public actor ContinuousBatchCoordinator {
    private struct RequestState {
        let submission: ContinuousBatchSubmission
        let mailbox: ContinuousBatchTokenMailbox
        var emittedTokens = 0
    }

    private struct ReservedPublication {
        let mailbox: ContinuousBatchTokenMailbox
        let reservation: ContinuousBatchPublicationReservation
    }

    private struct PreparedDecode {
        let id: BatchRequestID
        let visibleTokens: [Int]
        let finished: Bool
        let outcome: BatchDecodeOutcome
    }

    private var scheduler: ContinuousBatchScheduler
    private let runtime: any ContinuousBatchRuntime
    private let automaticDrive: Bool
    private let configuredPublicationCapacity: Int?
    private let traceLimit: Int
    private var trace: [ContinuousBatchCoordinatorEvent] = []
    private var timingTrace: [ContinuousBatchTimingEvent] = []
    private var requests: [BatchRequestID: RequestState] = [:]
    private var nextRequestID: UInt64? = 1
    private var driveTask: Task<Void, Never>?
    private var shuttingDown = false

    public init(
        configuration: ContinuousBatchConfiguration,
        runtime: sending any ContinuousBatchRuntime,
        automaticDrive: Bool = true,
        traceLimit: Int = 0
    ) {
        self.scheduler = ContinuousBatchScheduler(configuration: configuration)
        self.runtime = runtime
        self.automaticDrive = automaticDrive
        self.configuredPublicationCapacity = nil
        self.traceLimit = max(0, traceLimit)
    }

    public init(
        configuration: ContinuousBatchConfiguration,
        runtime: sending any ContinuousBatchRuntime,
        automaticDrive: Bool = true,
        publicationCapacity: Int,
        traceLimit: Int = 0
    ) {
        precondition(publicationCapacity > 0, "publicationCapacity must be positive")
        self.scheduler = ContinuousBatchScheduler(configuration: configuration)
        self.runtime = runtime
        self.automaticDrive = automaticDrive
        self.configuredPublicationCapacity = publicationCapacity
        self.traceLimit = max(0, traceLimit)
    }

    public func submit(_ submission: ContinuousBatchSubmission) throws
        -> ContinuousBatchRequestHandle
    {
        try enqueue([submission])[0]
    }

    /// Atomically enqueue a simultaneous burst. Either every request validates or none enter.
    public func submitBatch(_ submissions: [ContinuousBatchSubmission]) throws
        -> [ContinuousBatchRequestHandle]
    {
        try enqueue(submissions)
    }

    private func enqueue(_ submissions: [ContinuousBatchSubmission]) throws
        -> [ContinuousBatchRequestHandle]
    {
        guard !shuttingDown else { throw ContinuousBatchCoordinatorError.shuttingDown }
        guard !submissions.isEmpty else { return [] }

        var candidateScheduler = scheduler
        var candidateNextID = nextRequestID
        var ids: [BatchRequestID] = []
        var admissions: [ContinuousBatchRuntimeAdmission] = []
        ids.reserveCapacity(submissions.count)
        admissions.reserveCapacity(submissions.count)
        for submission in submissions {
            guard !submission.stopTokenIDs.isEmpty,
                submission.stopTokenIDs.allSatisfy({ $0 >= 0 })
            else {
                throw ContinuousBatchCoordinatorError.invalidStopTokenIDs
            }
            guard let rawID = candidateNextID else {
                throw ContinuousBatchCoordinatorError.requestIDExhausted
            }
            let id = BatchRequestID(rawID)
            let admission = ContinuousBatchRuntimeAdmission(
                id: id,
                submission: submission)
            let decodeCohort = try runtime.decodeCohort(
                for: admission)
            try candidateScheduler.submit(
                BatchRequest(
                    id: id,
                    promptTokenCount: submission.promptTokens.count,
                    maxOutputTokens: submission.maxOutputTokens,
                    architecture: submission.architecture,
                    requestsSpeculation: submission.requestsSpeculation,
                    decodeCohort: decodeCohort))
            ids.append(id)
            admissions.append(admission)
            candidateNextID = rawID == UInt64.max ? nil : rawID + 1
        }
        try runtime.admit(admissions)

        var handles: [ContinuousBatchRequestHandle] = []
        handles.reserveCapacity(submissions.count)
        for (id, submission) in zip(ids, submissions) {
            let consumerCancellation = ContinuousBatchCancellationHook { [weak self] in
                Task { _ = await self?.cancel(id) }
            }
            let mailbox = ContinuousBatchTokenMailbox(
                // Manual proof drivers historically drain before consuming. Their default
                // remains lossless and bounded by the declared output budget; production
                // serving supplies a smaller explicit capacity for client backpressure.
                capacity: configuredPublicationCapacity ?? submission.maxOutputTokens,
                consumerCancellation: consumerCancellation)
            let stream = ContinuousBatchTokenStream(
                mailbox: mailbox,
                cancellationHook: consumerCancellation)
            requests[id] = RequestState(
                submission: submission,
                mailbox: mailbox)
            handles.append(ContinuousBatchRequestHandle(id: id, tokens: stream))
        }

        scheduler = candidateScheduler
        nextRequestID = candidateNextID
        ensureAutomaticDrive()
        return handles
    }

    @discardableResult
    public func cancel(_ id: BatchRequestID) async -> BatchCancellationResult {
        let result = scheduler.cancel(id)
        guard let state = requests.removeValue(forKey: id) else { return result }
        runtime.remove(id)
        if case .cancelled = result { record(.cancelled(result)) }
        await state.mailbox.cancel()
        return result
    }

    /// Deterministic pump seam for tests. Returns whether work remains after this commit.
    @discardableResult
    public func runOneTick() async throws -> Bool {
        guard !shuttingDown else { throw ContinuousBatchCoordinatorError.shuttingDown }
        guard !scheduler.isEmpty else { return false }
        do {
            try Task.checkCancellation()
            try await executeOneTick()
            try Task.checkCancellation()
            return !scheduler.isEmpty
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            await failAll(with: error, terminal: true)
            throw error
        }
    }

    public func waitUntilIdle() async {
        while let task = driveTask {
            await task.value
        }
    }

    public func shutdown() async {
        if shuttingDown {
            if let task = driveTask { await task.value }
            return
        }
        shuttingDown = true
        let task = driveTask
        task?.cancel()
        let ids = Array(requests.keys)
        var mailboxes: [ContinuousBatchTokenMailbox] = []
        for id in ids {
            let result = scheduler.cancel(id)
            runtime.remove(id)
            if let state = requests.removeValue(forKey: id) {
                mailboxes.append(state.mailbox)
            }
            if case .cancelled = result { record(.cancelled(result)) }
        }
        for mailbox in mailboxes { await mailbox.cancel() }
        if let task { await task.value }
    }

    public func snapshot(for id: BatchRequestID) -> BatchSlotSnapshot? {
        scheduler.snapshot(for: id)
    }

    public func snapshots() -> [BatchSlotSnapshot] {
        scheduler.snapshots
    }

    public func runtimeResourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        runtime.resourceSnapshot()
    }

    public func executionTrace() -> [ContinuousBatchCoordinatorEvent] {
        trace
    }

    /// Returns only this measurement interval's committed trace. Clearing is actor-isolated,
    /// so a later run cannot accidentally report operations from an earlier warmup/burst.
    public func takeExecutionTrace() -> [ContinuousBatchCoordinatorEvent] {
        let result = trace
        trace.removeAll(keepingCapacity: true)
        return result
    }

    public func takeTimingTrace() -> [ContinuousBatchTimingEvent] {
        let result = timingTrace
        timingTrace.removeAll(keepingCapacity: true)
        return result
    }

    public func isShutDown() -> Bool { shuttingDown }

    private func ensureAutomaticDrive() {
        guard automaticDrive, driveTask == nil, !scheduler.isEmpty, !shuttingDown else { return }
        driveTask = Task { await self.driveUntilIdle() }
    }

    private func driveUntilIdle() async {
        defer { driveTask = nil }
        while !Task.isCancelled, !scheduler.isEmpty, !shuttingDown {
            do {
                try await executeOneTick()
            } catch is CancellationError {
                return
            } catch {
                await failAll(with: error, terminal: true)
                return
            }
            await Task.yield()
        }
    }

    private func executeOneTick() async throws {
        let plan: BatchTickPlan
        let reservations: [BatchRequestID: ReservedPublication]
        while true {
            try Task.checkCancellation()
            let candidate = scheduler.makeTick()
            guard let candidateReservations = try await reservePublications(
                for: candidate)
            else {
                if scheduler.isEmpty || shuttingDown || Task.isCancelled { return }
                continue
            }
            plan = candidate
            reservations = candidateReservations
            break
        }

        var preparedDecode: [PreparedDecode] = []
        do {
            for operation in plan.operations {
                switch operation {
                case .decode(let action):
                    preparedDecode = try runtime.decode(action).map(prepareDecodeResult)
                case .prefill(let slice):
                    guard let state = requests[slice.id] else {
                        throw ContinuousBatchCoordinatorError.unknownRuntimeRequest(slice.id)
                    }
                    let range = slice.startToken ..< slice.endToken
                    let prompt = state.submission.promptTokens
                    guard range.lowerBound >= 0, range.upperBound <= prompt.count else {
                        throw ContinuousBatchCoordinatorError.unknownRuntimeRequest(slice.id)
                    }
                    try runtime.prefill(
                        ContinuousBatchRuntimePrefill(
                            id: slice.id,
                            startToken: slice.startToken,
                            tokens: Array(prompt[range]),
                            isFinal: slice.endToken == prompt.count,
                            totalPromptTokens: prompt.count,
                            maxOutputTokens: state.submission.maxOutputTokens))
                }
            }

            try scheduler.apply(
                plan,
                decodeOutcomes: preparedDecode.map(\.outcome))
            for result in preparedDecode {
                guard var state = requests[result.id] else {
                    throw ContinuousBatchCoordinatorError.unknownRuntimeRequest(result.id)
                }
                state.emittedTokens += result.visibleTokens.count
                requests[result.id] = state
            }
            for operation in plan.operations { record(.operation(operation)) }
        } catch {
            await release(reservations)
            throw error
        }

        try await publish(preparedDecode, reservations: reservations)
    }

    private func reservePublications(
        for plan: BatchTickPlan
    ) async throws -> [BatchRequestID: ReservedPublication]? {
        try Task.checkCancellation()
        guard let decode = plan.decode else { return [:] }
        let ids: [BatchRequestID]
        switch decode {
        case .drainSoloPipeline(let id), .solo(let id, _):
            ids = [id]
        case .batch(let batchIDs, _):
            ids = batchIDs
        }

        while true {
            try Task.checkCancellation()
            var result: [BatchRequestID: ReservedPublication] = [:]
            result.reserveCapacity(ids.count)
            var blockedMailbox: ContinuousBatchTokenMailbox?
            do {
                for id in ids {
                    guard let state = requests[id] else {
                        throw ContinuousBatchCoordinatorError.unknownRuntimeRequest(id)
                    }
                    guard let reservation = try await state.mailbox.tryReserve() else {
                        blockedMailbox = state.mailbox
                        break
                    }
                    result[id] = ReservedPublication(
                        mailbox: state.mailbox,
                        reservation: reservation)
                }
            } catch {
                await release(result)
                if Task.isCancelled {
                    throw CancellationError()
                }
                if shuttingDown || scheduler.makeTick() != plan {
                    return nil
                }
                throw error
            }

            guard let blockedMailbox else {
                if Task.isCancelled {
                    await release(result)
                    throw CancellationError()
                }
                guard scheduler.makeTick() == plan else {
                    await release(result)
                    return nil
                }
                return result
            }

            // Never hold one client's capacity while awaiting another. Reserve-and-release
            // one slot only as a readiness signal, then retry the whole batch from scratch.
            await release(result)
            do {
                let signal = try await blockedMailbox.reserve()
                await blockedMailbox.release(signal)
            } catch {
                if Task.isCancelled {
                    throw CancellationError()
                }
                if shuttingDown || scheduler.makeTick() != plan {
                    return nil
                }
                throw error
            }
            try Task.checkCancellation()
            guard scheduler.makeTick() == plan else {
                return nil
            }
        }
    }

    private func release(
        _ reservations: [BatchRequestID: ReservedPublication]
    ) async {
        for publication in reservations.values {
            await publication.mailbox.release(publication.reservation)
        }
    }

    private func prepareDecodeResult(_ result: ContinuousBatchRuntimeDecodeResult) throws
        -> PreparedDecode
    {
        guard let state = requests[result.id] else {
            throw ContinuousBatchCoordinatorError.unknownRuntimeRequest(result.id)
        }
        let remaining = max(
            0,
            state.submission.maxOutputTokens - state.emittedTokens)
        var visible: [Int] = []
        visible.reserveCapacity(min(remaining, result.tokens.count))
        var finished = result.finished

        for token in result.tokens {
            if state.submission.stopTokenIDs.contains(token) {
                finished = true
                break
            }
            if visible.count >= remaining {
                finished = true
                break
            }
            visible.append(token)
            if visible.count >= remaining {
                finished = true
                break
            }
        }

        return PreparedDecode(
            id: result.id,
            visibleTokens: visible,
            finished: finished,
            outcome: BatchDecodeOutcome(
                id: result.id,
                emittedTokenCount: visible.count,
                finished: finished,
                hasPendingSoloLookahead: finished
                    ? false : result.hasPendingSoloLookahead))
    }

    private func publish(
        _ prepared: [PreparedDecode],
        reservations: [BatchRequestID: ReservedPublication]
    ) async throws {
        var unconsumed = reservations
        for result in prepared {
            guard let publication = unconsumed.removeValue(forKey: result.id) else {
                await release(unconsumed)
                throw ContinuousBatchCoordinatorError.unknownRuntimeRequest(result.id)
            }

            do {
                if let first = result.visibleTokens.first {
                    let timestamp = ProcessInfo.processInfo.systemUptime
                    try await publication.mailbox.commit(
                        publication.reservation,
                        token: first)
                    recordTiming(.emitted(result.id, timestamp: timestamp))

                    for token in result.visibleTokens.dropFirst() {
                        let reservation = try await publication.mailbox.reserveCommitted()
                        let extraTimestamp = ProcessInfo.processInfo.systemUptime
                        do {
                            try await publication.mailbox.commit(
                                reservation,
                                token: token)
                        } catch {
                            await publication.mailbox.release(reservation)
                            throw error
                        }
                        recordTiming(.emitted(result.id, timestamp: extraTimestamp))
                    }
                } else {
                    await publication.mailbox.release(publication.reservation)
                }
            } catch {
                await publication.mailbox.release(publication.reservation)
                if requests[result.id] == nil || shuttingDown {
                    continue
                }
                await release(unconsumed)
                throw error
            }

            if result.finished, requests[result.id] != nil {
                let timestamp = ProcessInfo.processInfo.systemUptime
                runtime.remove(result.id)
                requests[result.id] = nil
                record(.finished(result.id))
                await publication.mailbox.finish()
                recordTiming(.finished(result.id, timestamp: timestamp))
            }
        }

        if !unconsumed.isEmpty {
            await release(unconsumed)
            throw ContinuousBatchCoordinatorError.unknownRuntimeRequest(
                unconsumed.keys.sorted().first!)
        }
    }

    private func failAll(with error: Error, terminal: Bool) async {
        let message = String(describing: error)
        let ids = Array(requests.keys)
        var mailboxes: [ContinuousBatchTokenMailbox] = []
        for id in ids {
            _ = scheduler.cancel(id)
            runtime.remove(id)
            if let state = requests.removeValue(forKey: id) {
                mailboxes.append(state.mailbox)
            }
        }
        record(.failed(message))
        if terminal { shuttingDown = true }
        for mailbox in mailboxes {
            await mailbox.fail(error)
        }
    }

    private func record(_ event: ContinuousBatchCoordinatorEvent) {
        guard traceLimit > 0 else { return }
        trace.append(event)
        if trace.count > traceLimit {
            trace.removeFirst(trace.count - traceLimit)
        }
    }


    private func recordTiming(_ event: ContinuousBatchTimingEvent) {
        guard traceLimit > 0 else { return }
        timingTrace.append(event)
        if timingTrace.count > traceLimit {
            timingTrace.removeFirst(timingTrace.count - traceLimit)
        }
    }
}
