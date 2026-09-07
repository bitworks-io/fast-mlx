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

    public init(
        launchedModel: String,
        requestLimits: OpenAIChatRequestLimits = .productionDefault,
        requiredBearerToken: String?,
        maximumNonStreamingResponseBytes: Int,
        backpressureStallTimeout: Duration,
        evidence: ServingHTTPEvidenceConfiguration? = nil,
        modelCapabilities: ServingModelCapabilities? = nil,
        metricsSnapshot: ServingHTTPEvidenceConfiguration.SnapshotProvider? = nil
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
