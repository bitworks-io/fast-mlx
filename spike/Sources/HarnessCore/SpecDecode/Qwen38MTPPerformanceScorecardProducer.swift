import Foundation

public enum Qwen38MTPPerformanceScorecardEngineRole: String, Codable, Equatable, Sendable {
    case candidate
    case reference
}

public struct Qwen38MTPPerformanceScorecardMeasurementRequest: Equatable, Sendable {
    public let role: Qwen38MTPPerformanceScorecardEngineRole
    public let identity: Qwen38MTPPerformanceScorecardModel
    public let schedule: Qwen38MTPPerformanceScorecardPairSchedule
    public let workload: Qwen38MTPPerformanceScorecardWorkload
    public let settings: Qwen38MTPPerformanceScorecardSettings

    public init(
        role: Qwen38MTPPerformanceScorecardEngineRole,
        identity: Qwen38MTPPerformanceScorecardModel,
        schedule: Qwen38MTPPerformanceScorecardPairSchedule,
        workload: Qwen38MTPPerformanceScorecardWorkload,
        settings: Qwen38MTPPerformanceScorecardSettings
    ) {
        self.role = role
        self.identity = identity
        self.schedule = schedule
        self.workload = workload
        self.settings = settings
    }
}

public typealias Qwen38MTPPerformanceScorecardMeasurementClosure =
    @Sendable (Qwen38MTPPerformanceScorecardMeasurementRequest) async throws
        -> Qwen38MTPPerformanceScorecardEngineMeasurement

public struct Qwen38MTPPerformanceScorecardProducer: Sendable {
    private let measure: Qwen38MTPPerformanceScorecardMeasurementClosure

    public init(
        measure: @escaping Qwen38MTPPerformanceScorecardMeasurementClosure
    ) {
        self.measure = measure
    }

    public func makeRecord(
        authority: Qwen38MTPPerformanceScorecardAuthorityBundle,
        provenance: Provenance,
        releaseBuildObserved: Bool
    ) async throws -> ResultRecord<Qwen38MTPPerformanceScorecardEvidence> {
        try Qwen38MTPPerformanceScorecardGate.validatePreflight(
            authority: authority,
            provenance: provenance,
            releaseBuildObserved: releaseBuildObserved)

        var pairs: [Qwen38MTPPerformanceScorecardPair] = []
        pairs.reserveCapacity(Qwen38MTPPerformanceScorecardGate.runPlan.schedules.count)
        for (index, schedule) in Qwen38MTPPerformanceScorecardGate.runPlan.schedules.enumerated() {
            let candidateRequest = measurementRequest(
                role: .candidate,
                identity: authority.trustedEngineIdentities.candidate,
                schedule: schedule)
            let referenceRequest = measurementRequest(
                role: .reference,
                identity: authority.trustedEngineIdentities.reference,
                schedule: schedule)
            let candidate: Qwen38MTPPerformanceScorecardEngineMeasurement
            let reference: Qwen38MTPPerformanceScorecardEngineMeasurement
            switch schedule.order {
            case .candidateThenReference:
                candidate = try await measureValidated(candidateRequest, pairIndex: index)
                reference = try await measureValidated(referenceRequest, pairIndex: index)
            case .referenceThenCandidate:
                reference = try await measureValidated(referenceRequest, pairIndex: index)
                candidate = try await measureValidated(candidateRequest, pairIndex: index)
            }
            pairs.append(Qwen38MTPPerformanceScorecardPair(
                concurrency: schedule.concurrency,
                pairIndex: schedule.pairIndex,
                warmup: schedule.pairIndex
                    < Qwen38MTPPerformanceScorecardGate.runPlan.droppedWarmupPairs,
                order: schedule.order,
                scheduledCaseIDs: schedule.caseIDs,
                scheduledBenchmarkCells: schedule.benchmarkCells,
                candidate: candidate,
                reference: reference))
        }

        var evidence = Qwen38MTPPerformanceScorecardEvidence(
            schemaVersion: Qwen38MTPPerformanceScorecardGate.schemaVersion,
            artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
            candidate: authority.trustedEngineIdentities.candidate,
            reference: authority.trustedEngineIdentities.reference,
            liveExactnessProof: authority.acceptedLiveExactnessProof,
            measurementClass: Qwen38MTPPerformanceScorecardGate.measurementClass,
            hardware: Qwen38MTPPerformanceScorecardHardware(
                className: Qwen38MTPPerformanceScorecardGate.measurementClass,
                chip: authority.trustedRunIdentity.hardwareChip,
                ramBytes: authority.trustedRunIdentity.hardwareRAMBytes,
                osBuild: authority.trustedRunIdentity.hardwareOSBuild,
                hostIdentityDigest: authority.trustedRunIdentity.hostIdentityDigest),
            releaseBuildRequired: true,
            releaseBuildObserved: releaseBuildObserved,
            workload: Qwen38MTPPerformanceScorecardGate.requiredWorkload,
            settings: Qwen38MTPPerformanceScorecardGate.requiredSettings,
            runPlan: Qwen38MTPPerformanceScorecardGate.runPlan,
            pairs: pairs,
            metrics: .empty,
            verdict: .unqualified)
        evidence.metrics = try Qwen38MTPPerformanceScorecardGate.computeMetrics(
            evidence,
            authority: authority)
        evidence.verdict = try Qwen38MTPPerformanceScorecardGate.evaluateCandidate(
            evidence,
            authority: authority)

        return ResultRecord(
            subcommand: evidence.verdict.qualified
                ? Qwen38MTPPerformanceScorecardGate.subcommand
                : Qwen38MTPPerformanceScorecardGate.rejectedSubcommand,
            provenance: provenance,
            payload: evidence)
    }

    private func measurementRequest(
        role: Qwen38MTPPerformanceScorecardEngineRole,
        identity: Qwen38MTPPerformanceScorecardModel,
        schedule: Qwen38MTPPerformanceScorecardPairSchedule
    ) -> Qwen38MTPPerformanceScorecardMeasurementRequest {
        Qwen38MTPPerformanceScorecardMeasurementRequest(
            role: role,
            identity: identity,
            schedule: schedule,
            workload: Qwen38MTPPerformanceScorecardGate.requiredWorkload,
            settings: Qwen38MTPPerformanceScorecardGate.requiredSettings)
    }

    private func measureValidated(
        _ request: Qwen38MTPPerformanceScorecardMeasurementRequest,
        pairIndex: Int
    ) async throws -> Qwen38MTPPerformanceScorecardEngineMeasurement {
        let measurement = try await measure(request)
        guard measurement.identity == request.identity else {
            throw Qwen38MTPPerformanceScorecardGateError.invalidPair(
                index: pairIndex,
                reason: "engine identity")
        }
        return measurement
    }
}
