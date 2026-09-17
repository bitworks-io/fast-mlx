import Foundation
import ServingCore
import os

final class ServingHTTPEvidenceTracker: Sendable {
    private struct State: Sendable {
        var accepting = true
        var activeRequests = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    func begin() -> Bool {
        state.withLock { state in
            guard state.accepting else {
                return false
            }
            state.activeRequests += 1
            return true
        }
    }

    func end() {
        state.withLock { state in
            precondition(
                state.activeRequests > 0,
                "serving evidence tracker underflow")
            state.activeRequests -= 1
        }
    }

    func stopAccepting() {
        state.withLock { $0.accepting = false }
    }

    func failClosed() {
        stopAccepting()
    }

    func waitUntilIdle(
        clock: ContinuousClock,
        deadline: ContinuousClock.Instant
    ) async -> Bool {
        while clock.now < deadline {
            if state.withLock({ $0.activeRequests == 0 }) {
                return true
            }
            try? await Task.sleep(for: .milliseconds(1))
        }
        return state.withLock { $0.activeRequests == 0 }
    }
}

public struct ServingHTTPEvidenceConfiguration: Sendable {
    public typealias SnapshotProvider =
        @Sendable () async throws -> ServingEvidence.ResourceSnapshot
    public typealias Recorder =
        @Sendable (ServingEvidence) async throws -> Void
    public typealias FailureReporter =
        @Sendable (String) -> Void

    public let snapshot: SnapshotProvider?
    public let record: Recorder
    public let reportFailure: FailureReporter
    let tracker: ServingHTTPEvidenceTracker

    public init(
        snapshot: SnapshotProvider?,
        record: @escaping Recorder,
        reportFailure: @escaping FailureReporter
    ) {
        self.snapshot = snapshot
        self.record = record
        self.reportFailure = reportFailure
        tracker = ServingHTTPEvidenceTracker()
    }
}

public struct ServingHTTPConfiguration: Sendable {
    public let launchedModel: String
    public let requestLimits: OpenAIChatRequestLimits
    public let requiredBearerToken: String?
    public let maximumNonStreamingResponseBytes: Int
    public let backpressureStallTimeout: Duration
    public let evidence: ServingHTTPEvidenceConfiguration?
    public let modelCapabilities: ServingModelCapabilities?
    /// Carries the `/metrics` snapshot provider independently of `evidence`, so routes that supply
    /// no `--evidence` sink (and therefore no `ServingHTTPEvidenceConfiguration` — see its
    /// fail-closed admission tracker) can still serve real metrics. `runMetrics` reads
    /// `evidence?.snapshot ?? metricsSnapshot`. `nil` by default so every existing construction
    /// site keeps compiling unchanged.
    public let metricsSnapshot: ServingHTTPEvidenceConfiguration.SnapshotProvider?
    /// Carries a sink for the cause swallowed by `runGeneration`'s catch-all, independently of
    /// `evidence`, for the same reason `metricsSnapshot` is independent of it: `evidence` is `nil`
    /// on the default serve path (no `--evidence`), and constructing a
    /// `ServingHTTPEvidenceConfiguration` merely to carry a reporter would arm its fail-closed
    /// admission tracker (see `ServingHTTPEvidenceTracker`). This field must not be routed through
    /// `evidence.reportFailure` — it exists precisely so a swallowed generation error is reportable
    /// even when no evidence sink was configured. `nil` by default so every existing construction
    /// site keeps compiling unchanged.
    public let requestFailureReporter: ServingHTTPEvidenceConfiguration.FailureReporter?
    /// Backs `GET /readyz`. Defaults to always-ready: `fastmlx-serve` only binds its listen socket
    /// after the model has finished loading, so "the process is accepting connections" already
    /// implies "ready" on every production call site — no caller needs to pass this. The hook
    /// exists so a future drain/shutdown state can flip readiness without a new route, and so the
    /// 503 branch of `/readyz` is exercisable from tests. Must stay `Sendable`-clean: it can be
    /// invoked from any NIO event loop thread.
    public let readiness: @Sendable () -> Bool
    /// Backs the structured per-request access log (`--request-log json`). `nil` (the default)
    /// preserves today's behavior byte-for-byte: `OpenAIChatCompletionsHTTPHandler` never builds or
    /// writes a request-log line at all, on any route. Non-`nil` opts a sink in: the handler emits
    /// exactly one compact, sorted-keys JSON object per finished request (see
    /// `ServingRequestLogRecord`) by calling this closure with the already-encoded line (no trailing
    /// newline). Injectable so tests can capture lines into an array instead of writing to stderr;
    /// the production default (`servingRequestLogStandardErrorSink()`) appends a trailing newline and
    /// writes through `FileHandle.standardError.write`, which issues the `write(2)` syscall directly
    /// (unbuffered), satisfying "flushed" with no separate fsync step. Must stay `Sendable`-clean:
    /// invoked from both synchronous NIO event-loop callbacks and detached generation `Task`s.
    public let requestLog: (@Sendable (String) -> Void)?

    public init(
        launchedModel: String,
        requestLimits: OpenAIChatRequestLimits = .productionDefault,
        requiredBearerToken: String?,
        maximumNonStreamingResponseBytes: Int,
        backpressureStallTimeout: Duration,
        evidence: ServingHTTPEvidenceConfiguration? = nil,
        modelCapabilities: ServingModelCapabilities? = nil,
        metricsSnapshot: ServingHTTPEvidenceConfiguration.SnapshotProvider? = nil,
        requestFailureReporter: ServingHTTPEvidenceConfiguration.FailureReporter? = nil,
        readiness: @escaping @Sendable () -> Bool = { true },
        requestLog: (@Sendable (String) -> Void)? = nil
    ) {
        precondition(
            !launchedModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "launchedModel must be non-empty")
        precondition(
            maximumNonStreamingResponseBytes > 0,
            "maximumNonStreamingResponseBytes must be positive")
        precondition(
            backpressureStallTimeout > .zero,
            "backpressureStallTimeout must be positive")
        if let requiredBearerToken {
            precondition(!requiredBearerToken.isEmpty, "requiredBearerToken must be non-empty")
        }

        self.launchedModel = launchedModel
        self.requestLimits = requestLimits
        self.requiredBearerToken = requiredBearerToken
        self.maximumNonStreamingResponseBytes = maximumNonStreamingResponseBytes
        self.backpressureStallTimeout = backpressureStallTimeout
        self.evidence = evidence
        self.modelCapabilities = modelCapabilities
        self.metricsSnapshot = metricsSnapshot
        self.requestFailureReporter = requestFailureReporter
        self.readiness = readiness
        self.requestLog = requestLog
        precondition(
            modelCapabilities == nil || modelCapabilities?.model == launchedModel,
            "modelCapabilities must describe the launched model")
        precondition(
            modelCapabilities == nil
                || modelCapabilities?.maximumRequestBodyBytes == requestLimits.maximumBodyBytes,
            "requestLimits must match advertised model capability")
        precondition(
            modelCapabilities == nil
                || modelCapabilities?.maximumNonStreamingResponseBytes
                    == maximumNonStreamingResponseBytes,
            "response limits must match advertised model capability")
    }
}
