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

/// Synchronous runtime seam owned by `ContinuousBatchCoordinator`.
///
/// The protocol deliberately is not `Sendable`: a production implementation owns MLX
/// arrays, caches, and compiled functions. A `sending` initializer transfers that whole
/// isolation region into the coordinator actor, and no runtime value crosses back out.
public protocol ContinuousBatchRuntime: AnyObject {
    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws
    func prefill(_ work: ContinuousBatchRuntimePrefill) throws
    func decode(_ action: BatchDecodeAction) throws -> [ContinuousBatchRuntimeDecodeResult]
    func remove(_ id: BatchRequestID)
}

extension ContinuousBatchRuntime {
    /// Runtimes with no additional capability or resource gate can accept scheduler-valid work.
    public func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {}
}

public struct ContinuousBatchSubmission: Sendable, Equatable {
    public let promptTokens: [Int]
    public let maxOutputTokens: Int
    public let eosToken: Int
    public let architecture: BatchArchitectureClass
    public let requestsSpeculation: Bool

    public init(
        promptTokens: [Int],
        maxOutputTokens: Int,
        eosToken: Int,
        architecture: BatchArchitectureClass,
        requestsSpeculation: Bool = false
    ) {
        self.promptTokens = promptTokens
        self.maxOutputTokens = maxOutputTokens
        self.eosToken = eosToken
        self.architecture = architecture
        self.requestsSpeculation = requestsSpeculation
    }
}

public struct ContinuousBatchRequestHandle: Sendable {
    public let id: BatchRequestID
    public let tokens: AsyncThrowingStream<Int, Error>

    public init(id: BatchRequestID, tokens: AsyncThrowingStream<Int, Error>) {
        self.id = id
        self.tokens = tokens
    }
}

public enum ContinuousBatchCoordinatorError: Error, Sendable, Equatable {
    case shuttingDown
    case requestIDExhausted
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
/// One tick is one transaction with no suspension point: execute decode first, execute bounded
/// prompt slices, validate/apply the exact scheduler plan, then publish tokens and terminal
/// events. The automatic pump yields only after that commit, so new submissions and
/// cancellation can interleave between ticks but never attach to partially executed work.
public actor ContinuousBatchCoordinator {
    private struct RequestState {
        let submission: ContinuousBatchSubmission
        let continuation: AsyncThrowingStream<Int, Error>.Continuation
        var emittedTokens = 0
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
        ids.reserveCapacity(submissions.count)
        for submission in submissions {
            guard let rawID = candidateNextID else {
                throw ContinuousBatchCoordinatorError.requestIDExhausted
            }
            let id = BatchRequestID(rawID)
            try candidateScheduler.submit(
                BatchRequest(
                    id: id,
                    promptTokenCount: submission.promptTokens.count,
                    maxOutputTokens: submission.maxOutputTokens,
                    architecture: submission.architecture,
                    requestsSpeculation: submission.requestsSpeculation))
            ids.append(id)
            candidateNextID = rawID == UInt64.max ? nil : rawID + 1
        }
        try runtime.admit(
            zip(ids, submissions).map {
                ContinuousBatchRuntimeAdmission(id: $0.0, submission: $0.1)
            })

        var handles: [ContinuousBatchRequestHandle] = []
        handles.reserveCapacity(submissions.count)
        for (id, submission) in zip(ids, submissions) {
            var captured: AsyncThrowingStream<Int, Error>.Continuation?
            let stream = AsyncThrowingStream<Int, Error> { continuation in
                captured = continuation
            }
            guard let continuation = captured else {
                preconditionFailure("AsyncThrowingStream did not synchronously vend a continuation")
            }
            continuation.onTermination = { [weak self] termination in
                guard case .cancelled = termination else { return }
                Task { _ = await self?.cancel(id) }
            }
            requests[id] = RequestState(
                submission: submission,
                continuation: continuation)
            handles.append(ContinuousBatchRequestHandle(id: id, tokens: stream))
        }

        scheduler = candidateScheduler
        nextRequestID = candidateNextID
        ensureAutomaticDrive()
        return handles
    }

    @discardableResult
    public func cancel(_ id: BatchRequestID) -> BatchCancellationResult {
        let result = scheduler.cancel(id)
        guard case .cancelled = result else { return result }
        runtime.remove(id)
        if let state = requests.removeValue(forKey: id) {
            state.continuation.finish()
        }
        record(.cancelled(result))
        return result
    }

    /// Deterministic pump seam for tests. Returns whether work remains after this commit.
    @discardableResult
    public func runOneTick() throws -> Bool {
        guard !shuttingDown else { throw ContinuousBatchCoordinatorError.shuttingDown }
        guard !scheduler.isEmpty else { return false }
        do {
            try executeOneTick()
            return !scheduler.isEmpty
        } catch {
            failAll(with: error, terminal: true)
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
        for id in ids {
            let result = scheduler.cancel(id)
            runtime.remove(id)
            requests.removeValue(forKey: id)?.continuation.finish(
                throwing: CancellationError())
            if case .cancelled = result { record(.cancelled(result)) }
        }
        if let task { await task.value }
    }

    public func snapshot(for id: BatchRequestID) -> BatchSlotSnapshot? {
        scheduler.snapshot(for: id)
    }

    public func snapshots() -> [BatchSlotSnapshot] {
        scheduler.snapshots
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
                try executeOneTick()
            } catch {
                failAll(with: error, terminal: true)
                return
            }
            await Task.yield()
        }
    }

    private func executeOneTick() throws {
        let plan = scheduler.makeTick()
        var preparedDecode: [PreparedDecode] = []

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
        for operation in plan.operations { record(.operation(operation)) }
        commit(preparedDecode)
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
            if token == state.submission.eosToken {
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

    private func commit(_ prepared: [PreparedDecode]) {
        for result in prepared {
            guard var state = requests[result.id] else { continue }
            for token in result.visibleTokens {
                recordTiming(
                    .emitted(
                        result.id,
                        timestamp: ProcessInfo.processInfo.systemUptime))
                state.continuation.yield(token)
            }
            state.emittedTokens += result.visibleTokens.count
            if result.finished {
                recordTiming(
                    .finished(
                        result.id,
                        timestamp: ProcessInfo.processInfo.systemUptime))
                state.continuation.finish()
                runtime.remove(result.id)
                requests[result.id] = nil
                record(.finished(result.id))
            } else {
                requests[result.id] = state
            }
        }
    }

    private func failAll(with error: Error, terminal: Bool) {
        let ids = Array(requests.keys)
        for id in ids {
            _ = scheduler.cancel(id)
            runtime.remove(id)
            requests.removeValue(forKey: id)?.continuation.finish(throwing: error)
        }
        record(.failed(String(describing: error)))
        if terminal { shuttingDown = true }
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
