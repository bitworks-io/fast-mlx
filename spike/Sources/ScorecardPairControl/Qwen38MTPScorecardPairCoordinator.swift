import Foundation
import HarnessCore

package protocol Qwen38MTPScorecardWorkerClient: Sendable {
    var role: Qwen38MTPPerformanceScorecardEngineRole { get }
    func start() async throws -> Qwen38MTPScorecardWorkerHandshake
    func runCandidateExactness() async throws -> ResultRecord<Qwen38MTPLiveExactnessEvidence>
    func assertReadyForDispatch(expected: Qwen38MTPScorecardWorkerHandshake) async throws
    func measure(
        _ request: Qwen38MTPPerformanceScorecardMeasurementRequest
    ) async throws -> Qwen38MTPPerformanceScorecardEngineMeasurement
    func terminate() async
}

package struct Qwen38MTPScorecardLiveRunResult: Sendable {
    package let authority: Qwen38MTPPerformanceScorecardAuthorityBundle
    package let record: ResultRecord<Qwen38MTPPerformanceScorecardEvidence>

    package init(
        authority: Qwen38MTPPerformanceScorecardAuthorityBundle,
        record: ResultRecord<Qwen38MTPPerformanceScorecardEvidence>
    ) {
        self.authority = authority
        self.record = record
    }
}

/// Per-leg fixed expectations for the pair handshake. Parameterizing these
/// (rather than hard-coding literals inside validateHandshake) lets a future
/// caller express a different axis without touching the coordinator's control
/// flow; the only shipped preset today is `.fixedGDNAxis`.
package struct Qwen38MTPScorecardLegExpectation: Sendable {
    package let executionMode: Qwen38MTPPerformanceScorecardExecutionMode
    package let gdnMode: Qwen38MTPPerformanceScorecardGDNMode
    package let observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv
    package let launchMode: Qwen38MTPPerformanceScorecardGDNMode
    package let label: String

    package init(
        executionMode: Qwen38MTPPerformanceScorecardExecutionMode,
        gdnMode: Qwen38MTPPerformanceScorecardGDNMode,
        observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv,
        launchMode: Qwen38MTPPerformanceScorecardGDNMode,
        label: String
    ) {
        self.executionMode = executionMode
        self.gdnMode = gdnMode
        self.observedEnv = observedEnv
        self.launchMode = launchMode
        self.label = label
    }
}

package struct Qwen38MTPScorecardPairExpectations: Sendable {
    package let candidate: Qwen38MTPScorecardLegExpectation
    package let reference: Qwen38MTPScorecardLegExpectation

    package init(
        candidate: Qwen38MTPScorecardLegExpectation,
        reference: Qwen38MTPScorecardLegExpectation
    ) {
        self.candidate = candidate
        self.reference = reference
    }

    // Fixed-GDN executionMode axis: every worker must observe GDN fusion enabled;
    // roles differ only in the exact-MTP vs scalar execution route.
    package static let fixedGDNAxis = Qwen38MTPScorecardPairExpectations(
        candidate: Qwen38MTPScorecardLegExpectation(
            executionMode: .exactMTP,
            gdnMode: .gdnOn,
            observedEnv: .enabled,
            launchMode: .gdnOn,
            label: qwen38MTPScorecardSharedEngineLabel),
        reference: Qwen38MTPScorecardLegExpectation(
            executionMode: .scalar,
            gdnMode: .gdnOn,
            observedEnv: .enabled,
            launchMode: .gdnOn,
            label: qwen38MTPScorecardSharedEngineLabel))
}

package actor Qwen38MTPScorecardLineProtocolClient: Qwen38MTPScorecardWorkerClient {
    package let role: Qwen38MTPPerformanceScorecardEngineRole
    private let transport: Qwen38MTPScorecardLineTransport
    private var nextSequence = 1
    private var seenResponses: Set<Int> = []

    package init(
        role: Qwen38MTPPerformanceScorecardEngineRole,
        transport: Qwen38MTPScorecardLineTransport
    ) {
        self.role = role
        self.transport = transport
    }

    package func start() async throws -> Qwen38MTPScorecardWorkerHandshake {
        let response = try await roundTrip(kind: .handshake)
        guard response.kind == .handshake, let handshake = response.handshake else {
            throw Qwen38MTPScorecardLiveAdapterError.malformedWorkerResponse
        }
        return handshake
    }

    package func runCandidateExactness() async throws -> ResultRecord<Qwen38MTPLiveExactnessEvidence> {
        guard role == .candidate else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidWorkerRole(role)
        }
        let response = try await roundTrip(kind: .exactness)
        guard response.kind == .exactness, let exactnessRecord = response.exactnessRecord else {
            throw Qwen38MTPScorecardLiveAdapterError.malformedWorkerResponse
        }
        return exactnessRecord
    }

    package func assertReadyForDispatch(
        expected: Qwen38MTPScorecardWorkerHandshake
    ) async throws {
        let response = try await roundTrip(kind: .assertReady)
        guard response.kind == .ok else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
    }

    package func measure(
        _ request: Qwen38MTPPerformanceScorecardMeasurementRequest
    ) async throws -> Qwen38MTPPerformanceScorecardEngineMeasurement {
        let response = try await roundTrip(
            kind: .measure,
            measurement: Qwen38MTPScorecardMeasurementCommand(request))
        guard response.kind == .measurement, let measurement = response.measurement else {
            throw Qwen38MTPScorecardLiveAdapterError.malformedWorkerResponse
        }
        return measurement
    }

    package func terminate() async {
        await transport.terminate()
    }

    private func roundTrip(
        kind: Qwen38MTPScorecardWorkerProtocolRequestKind,
        measurement: Qwen38MTPScorecardMeasurementCommand? = nil
    ) async throws -> Qwen38MTPScorecardWorkerProtocolResponse {
        let sequence = nextSequence
        nextSequence += 1
        let request = Qwen38MTPScorecardWorkerProtocolRequest(
            sequence: sequence,
            kind: kind,
            measurement: measurement)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try await transport.sendLine(String(decoding: try encoder.encode(request), as: UTF8.self))
        guard let line = try await transport.receiveLine() else {
            throw Qwen38MTPScorecardLiveAdapterError.workerExited(role)
        }
        guard let data = line.data(using: .utf8),
            let response = try? JSONDecoder().decode(
                Qwen38MTPScorecardWorkerProtocolResponse.self,
                from: data)
        else {
            throw Qwen38MTPScorecardLiveAdapterError.malformedWorkerResponse
        }
        guard !seenResponses.contains(response.sequence) else {
            throw Qwen38MTPScorecardLiveAdapterError.duplicateWorkerResponse(response.sequence)
        }
        seenResponses.insert(response.sequence)
        guard response.sequence == sequence else {
            throw Qwen38MTPScorecardLiveAdapterError.outOfOrderWorkerResponse(
                expected: sequence,
                actual: response.sequence)
        }
        guard response.kind != .error else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
        return response
    }
}

package struct Qwen38MTPScorecardLiveCoordinator<Candidate: Qwen38MTPScorecardWorkerClient, Reference: Qwen38MTPScorecardWorkerClient>: Sendable {
    package let candidate: Candidate
    package let reference: Reference
    package let runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    package let provenance: Provenance
    package let releaseBuildObserved: Bool
    package let expectations: Qwen38MTPScorecardPairExpectations

    package init(
        candidate: Candidate,
        reference: Reference,
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity,
        provenance: Provenance,
        releaseBuildObserved: Bool,
        expectations: Qwen38MTPScorecardPairExpectations = .fixedGDNAxis
    ) {
        self.candidate = candidate
        self.reference = reference
        self.runIdentity = runIdentity
        self.provenance = provenance
        self.releaseBuildObserved = releaseBuildObserved
        self.expectations = expectations
    }

    package func run() async throws -> Qwen38MTPScorecardLiveRunResult {
        do {
            let candidateHandshake = try await candidate.start()
            let referenceHandshake = try await reference.start()
            try validateHandshake(candidateHandshake, expectedRole: .candidate)
            try validateHandshake(referenceHandshake, expectedRole: .reference)

            let exactnessRecord = try await candidate.runCandidateExactness()
            let exactnessData = Data((try exactnessRecord.jsonLine() + "\n").utf8)
            let exactnessProof = try Qwen38MTPLiveExactnessGate.validateJSONL(exactnessData)
            guard exactnessProof.launchBinding == candidateHandshake.launchBinding,
                exactnessProof.launchBinding == candidateHandshake.model.launchBinding
            else {
                throw Qwen38MTPScorecardLiveAdapterError.exactnessLaunchBindingMismatch
            }

            let authority = Qwen38MTPPerformanceScorecardAuthorityBundle(
                acceptedLiveExactnessProof: exactnessProof,
                trustedEngineIdentities: .init(
                    candidate: candidateHandshake.model,
                    reference: referenceHandshake.model),
                trustedRunIdentity: runIdentity)
            try Qwen38MTPPerformanceScorecardGate.validateAuthority(authority)

            let producer = Qwen38MTPPerformanceScorecardProducer { request in
                switch request.role {
                case .candidate:
                    try await candidate.assertReadyForDispatch(expected: candidateHandshake)
                    return try await candidate.measure(request)
                case .reference:
                    try await reference.assertReadyForDispatch(expected: referenceHandshake)
                    return try await reference.measure(request)
                }
            }
            let record = try await producer.makeRecord(
                authority: authority,
                provenance: provenance,
                releaseBuildObserved: releaseBuildObserved)
            await terminateWorkers()
            return Qwen38MTPScorecardLiveRunResult(authority: authority, record: record)
        } catch {
            await terminateWorkers()
            throw error
        }
    }

    private func terminateWorkers() async {
        await candidate.terminate()
        await reference.terminate()
    }

    package func validateHandshake(
        _ handshake: Qwen38MTPScorecardWorkerHandshake,
        expectedRole: Qwen38MTPPerformanceScorecardEngineRole
    ) throws {
        guard handshake.role == expectedRole else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidWorkerRole(handshake.role)
        }
        // Fixed-GDN executionMode axis: every worker must observe GDN fusion enabled;
        // roles differ only in the exact-MTP vs scalar execution route.
        let expectation = expectedRole == .candidate ? expectations.candidate : expectations.reference
        guard handshake.model.gdnMode == expectation.gdnMode,
            handshake.model.executionMode == expectation.executionMode,
            handshake.model.label == expectation.label,
            handshake.processIsolation.gdnMode == expectation.gdnMode,
            handshake.processIsolation.observedEnv == expectation.observedEnv,
            handshake.launchBinding.mode == expectation.launchMode,
            handshake.launchBinding.observedEnv == expectation.observedEnv,
            handshake.model.launchBinding == handshake.launchBinding
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidHandshake(expectedRole)
        }
    }
}
