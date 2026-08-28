import Foundation

public enum DedicatedServingQualificationError: Error, Equatable, Sendable, CustomStringConvertible {
    case invalidHostUse
    case invalidHostUsePolicy
    case invalidHostMemory
    case invalidMetalObservation
    case unmeasuredOriginalWiredLimit
    case missingOSBuild
    case invalidOSBuildSource
    case invalidReserve
    case invalidBudget(String)
    case invalidCandidate(String)
    case invalidArtifact(String)

    public var description: String {
        switch self {
        case .invalidHostUse:
            return "host must be dedicated-serving with operator-assertion source"
        case .invalidHostUsePolicy:
            return "unsupported host-use policy"
        case .invalidHostMemory:
            return "host memory observations must be positive"
        case .invalidMetalObservation:
            return "Metal observations must be present and valid"
        case .unmeasuredOriginalWiredLimit:
            return "original wired limit must be measured"
        case .missingOSBuild:
            return "OS build must be present"
        case .invalidOSBuildSource:
            return "OS build source must be process-info-operating-system-version"
        case .invalidReserve:
            return "OS/service reserve must be positive"
        case .invalidBudget(let message), .invalidCandidate(let message), .invalidArtifact(let message):
            return message
        }
    }
}

public enum DedicatedServingOriginalWiredLimitProvenance: String, Codable, Equatable, Sendable {
    case measured
    case synthesized
}

public struct DedicatedServingQualificationHost: Codable, Equatable, Sendable {
    public static let processInfoOSBuildSource = "process-info-operating-system-version"

    public var hostUse: String
    public var hostUseSource: String
    public var hostUsePolicyVersion: String
    public var physicalRAMBytes: Int
    public var originalWiredLimitBytes: Int
    public var originalWiredLimitProvenance: DedicatedServingOriginalWiredLimitProvenance
    public var metalRecommendedWorkingSetBytes: Int
    public var metalCurrentAllocatedBytes: Int
    public var osBuild: String
    public var osBuildSource: String

    public init(
        hostUse: String,
        hostUseSource: String,
        hostUsePolicyVersion: String,
        physicalRAMBytes: Int,
        originalWiredLimitBytes: Int,
        originalWiredLimitProvenance: DedicatedServingOriginalWiredLimitProvenance,
        metalRecommendedWorkingSetBytes: Int,
        metalCurrentAllocatedBytes: Int,
        osBuild: String,
        osBuildSource: String
    ) {
        self.hostUse = hostUse
        self.hostUseSource = hostUseSource
        self.hostUsePolicyVersion = hostUsePolicyVersion
        self.physicalRAMBytes = physicalRAMBytes
        self.originalWiredLimitBytes = originalWiredLimitBytes
        self.originalWiredLimitProvenance = originalWiredLimitProvenance
        self.metalRecommendedWorkingSetBytes = metalRecommendedWorkingSetBytes
        self.metalCurrentAllocatedBytes = metalCurrentAllocatedBytes
        self.osBuild = osBuild
        self.osBuildSource = osBuildSource
    }
}

public struct DedicatedServingQualificationBudgets: Codable, Equatable, Sendable {
    /// Total MLX allocator cap. Cache and KV are concurrent MLX subsets and must fit inside this cap
    /// together. I/O prefetch is outside MLX and is reconciled against candidate-minus-reserve.
    public var memoryLimitBytes: Int
    public var cacheLimitBytes: Int
    public var kvBudgetBytes: Int
    public var ioPrefetchBudgetBytes: Int

    public init(
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        kvBudgetBytes: Int,
        ioPrefetchBudgetBytes: Int
    ) {
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.kvBudgetBytes = kvBudgetBytes
        self.ioPrefetchBudgetBytes = ioPrefetchBudgetBytes
    }
}

public struct DedicatedServingQualificationInput: Codable, Equatable, Sendable {
    public var host: DedicatedServingQualificationHost
    public var osServiceReserveBytes: Int
    public var candidateCeilingMiB: Int
    public var stagedFollowOnCandidateMiB: [Int]
    public var mlxBudgets: DedicatedServingQualificationBudgets

    public init(
        host: DedicatedServingQualificationHost,
        osServiceReserveBytes: Int,
        candidateCeilingMiB: Int,
        stagedFollowOnCandidateMiB: [Int] = [],
        mlxBudgets: DedicatedServingQualificationBudgets
    ) {
        self.host = host
        self.osServiceReserveBytes = osServiceReserveBytes
        self.candidateCeilingMiB = candidateCeilingMiB
        self.stagedFollowOnCandidateMiB = stagedFollowOnCandidateMiB
        self.mlxBudgets = mlxBudgets
    }
}

public struct DedicatedServingCandidateCeiling: Codable, Equatable, Sendable {
    public let mib: Int
    public let bytes: Int
}

public struct DedicatedServingBudgetReconciliation: Codable, Equatable, Sendable {
    public let candidateBytes: Int
    public let reserveBytes: Int
    public let memoryAvailableAfterReserveBytes: Int
    public let reserveSubtractedExactlyOnce: Bool
    public let mlxAllocatorLimitFitsCandidateMinusReserve: Bool
    public let cacheAndKVBytes: Int
    public let cacheAndKVFitMLXAllocatorLimit: Bool
    public let mlxAllocatorAndIOPrefetchBytes: Int
    public let mlxAllocatorAndIOFitCandidateMinusReserve: Bool
}

public enum DedicatedServingQualificationStageID: String, Codable, Equatable, Sendable {
    case captureOriginal = "capture-original"
    case temporaryApply = "temporary-apply"
    case readbackRecommendedWorkingSetCheck = "readback-recommended-working-set-check"
    case safetyHealthSoakGates = "safety-health-soak-gates"
    case restoreOnAnyFailure = "restore-on-any-failure"
}

public enum DedicatedServingQualificationGate: String, Codable, Equatable, Sendable {
    case originalWiredLimitCaptured = "original-wired-limit-captured"
    case osBuildCaptured = "os-build-captured"
    case physicalRAMCaptured = "physical-ram-captured"
    case metalRecommendedWorkingSetCaptured = "metal-recommended-working-set-captured"
    case metalCurrentAllocationCaptured = "metal-current-allocation-captured"
    case swapPageoutBaselineCaptured = "swap-pageout-baseline-captured"
    case memoryPressureBaselineCaptured = "memory-pressure-baseline-captured"
    case runningServicesBaselineCaptured = "running-services-baseline-captured"
    case candidateTemporarilyApplied = "candidate-temporarily-applied"
    case wiredLimitReadbackMatchesCandidate = "wired-limit-readback-matches-candidate"
    case recommendedWorkingSetObserved = "recommended-working-set-observed"
    case swapAndPageoutsStable = "swap-and-pageouts-stable"
    case memoryPressureNormal = "memory-pressure-normal"
    case noOOMOrSIGKILL = "no-oom-or-sigkill"
    case responsiveHealth = "responsive-health"
    case boundedAllocationLatencyThermal = "bounded-allocation-latency-thermal"
    case originalWiredLimitRestored = "original-wired-limit-restored"
}

public struct DedicatedServingQualificationStage: Codable, Equatable, Sendable {
    public let id: DedicatedServingQualificationStageID
    public let dryRunOnly: Bool
    public let description: String
    public let gates: [DedicatedServingQualificationGate]
}

public struct DedicatedServingQualificationArtifact: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = "dedicated-serving-qualification-plan/v1"

    public let schemaVersion: String
    public let dryRunOnly: Bool
    public let authoritative: Bool
    public let executionAuthorized: Bool
    public let host: DedicatedServingQualificationHost
    public let osServiceReserveBytes: Int
    public let proposedCandidateCeiling: DedicatedServingCandidateCeiling
    public let stagedFollowOnCandidates: [DedicatedServingCandidateCeiling]
    public let mlxBudgets: DedicatedServingQualificationBudgets
    public let budgetReconciliation: DedicatedServingBudgetReconciliation
    public let stages: [DedicatedServingQualificationStage]

    fileprivate init(
        schemaVersion: String = Self.currentSchemaVersion,
        dryRunOnly: Bool = true,
        authoritative: Bool = false,
        executionAuthorized: Bool = false,
        host: DedicatedServingQualificationHost,
        osServiceReserveBytes: Int,
        proposedCandidateCeiling: DedicatedServingCandidateCeiling,
        stagedFollowOnCandidates: [DedicatedServingCandidateCeiling],
        mlxBudgets: DedicatedServingQualificationBudgets,
        budgetReconciliation: DedicatedServingBudgetReconciliation,
        stages: [DedicatedServingQualificationStage]
    ) {
        self.schemaVersion = schemaVersion
        self.dryRunOnly = dryRunOnly
        self.authoritative = authoritative
        self.executionAuthorized = executionAuthorized
        self.host = host
        self.osServiceReserveBytes = osServiceReserveBytes
        self.proposedCandidateCeiling = proposedCandidateCeiling
        self.stagedFollowOnCandidates = stagedFollowOnCandidates
        self.mlxBudgets = mlxBudgets
        self.budgetReconciliation = budgetReconciliation
        self.stages = stages
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let schemaVersion = try container.decode(String.self, forKey: .schemaVersion)
        guard schemaVersion == Self.currentSchemaVersion else {
            throw DedicatedServingQualificationError.invalidArtifact("unsupported schemaVersion")
        }
        let dryRunOnly = try container.decode(Bool.self, forKey: .dryRunOnly)
        let authoritative = try container.decode(Bool.self, forKey: .authoritative)
        let executionAuthorized = try container.decode(Bool.self, forKey: .executionAuthorized)
        guard dryRunOnly, !authoritative, !executionAuthorized else {
            throw DedicatedServingQualificationError.invalidArtifact("artifact must remain dry-run-only and non-authoritative")
        }

        let host = try container.decode(DedicatedServingQualificationHost.self, forKey: .host)
        let osServiceReserveBytes = try container.decode(Int.self, forKey: .osServiceReserveBytes)
        let proposedCandidateCeiling = try container.decode(
            DedicatedServingCandidateCeiling.self,
            forKey: .proposedCandidateCeiling)
        let stagedFollowOnCandidates = try container.decode(
            [DedicatedServingCandidateCeiling].self,
            forKey: .stagedFollowOnCandidates)
        let mlxBudgets = try container.decode(DedicatedServingQualificationBudgets.self, forKey: .mlxBudgets)
        let budgetReconciliation = try container.decode(
            DedicatedServingBudgetReconciliation.self,
            forKey: .budgetReconciliation)
        let stages = try container.decode([DedicatedServingQualificationStage].self, forKey: .stages)

        let artifact = DedicatedServingQualificationArtifact(
            schemaVersion: schemaVersion,
            dryRunOnly: dryRunOnly,
            authoritative: authoritative,
            executionAuthorized: executionAuthorized,
            host: host,
            osServiceReserveBytes: osServiceReserveBytes,
            proposedCandidateCeiling: proposedCandidateCeiling,
            stagedFollowOnCandidates: stagedFollowOnCandidates,
            mlxBudgets: mlxBudgets,
            budgetReconciliation: budgetReconciliation,
            stages: stages)
        try DedicatedServingQualificationPlanner.validateDecodedArtifact(artifact)
        self = artifact
    }

    public func encodedJSON() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }
}

public enum DedicatedServingQualificationPlanner {
    private static let mib = 1024 * 1024
    private static let requiredCandidateLadderMiB = [224 * 1024, 232 * 1024, 240 * 1024]

    public static func plan(_ input: DedicatedServingQualificationInput) throws -> DedicatedServingQualificationArtifact {
        try validateHost(input.host)
        guard input.osServiceReserveBytes > 0 else { throw DedicatedServingQualificationError.invalidReserve }
        try validateCandidateLadder(
            proposedMiB: input.candidateCeilingMiB,
            followOnMiB: input.stagedFollowOnCandidateMiB)
        let candidate = try candidateCeiling(mib: input.candidateCeilingMiB)
        try validateCandidate(candidate, host: input.host)
        let followOns = try input.stagedFollowOnCandidateMiB.map { mib in
            let candidate = try candidateCeiling(mib: mib)
            try validateCandidate(candidate, host: input.host)
            return candidate
        }

        let reserveSubtraction = candidate.bytes.subtractingReportingOverflow(input.osServiceReserveBytes)
        guard !reserveSubtraction.overflow, reserveSubtraction.partialValue > 0 else {
            throw DedicatedServingQualificationError.invalidBudget("reserve must fit below candidate")
        }
        let availableAfterReserve = reserveSubtraction.partialValue
        try validateBudgets(input.mlxBudgets, availableAfterReserve: availableAfterReserve)
        let cacheAndKV = try checkedSum(
            input.mlxBudgets.cacheLimitBytes,
            input.mlxBudgets.kvBudgetBytes,
            message: "cache plus KV budget overflow")
        let memoryAndIO = try checkedSum(
            input.mlxBudgets.memoryLimitBytes,
            input.mlxBudgets.ioPrefetchBudgetBytes,
            message: "memory plus I/O prefetch budget overflow")

        let reconciliation = DedicatedServingBudgetReconciliation(
            candidateBytes: candidate.bytes,
            reserveBytes: input.osServiceReserveBytes,
            memoryAvailableAfterReserveBytes: availableAfterReserve,
            reserveSubtractedExactlyOnce: input.osServiceReserveBytes > 0,
            mlxAllocatorLimitFitsCandidateMinusReserve: input.mlxBudgets.memoryLimitBytes <= availableAfterReserve,
            cacheAndKVBytes: cacheAndKV,
            cacheAndKVFitMLXAllocatorLimit: cacheAndKV <= input.mlxBudgets.memoryLimitBytes,
            mlxAllocatorAndIOPrefetchBytes: memoryAndIO,
            mlxAllocatorAndIOFitCandidateMinusReserve: memoryAndIO <= availableAfterReserve)

        return DedicatedServingQualificationArtifact(
            host: input.host,
            osServiceReserveBytes: input.osServiceReserveBytes,
            proposedCandidateCeiling: candidate,
            stagedFollowOnCandidates: followOns,
            mlxBudgets: input.mlxBudgets,
            budgetReconciliation: reconciliation,
            stages: qualificationStages())
    }

    private static func validateHost(_ host: DedicatedServingQualificationHost) throws {
        guard host.hostUsePolicyVersion == HostUseClassification.currentPolicyVersion else {
            throw DedicatedServingQualificationError.invalidHostUsePolicy
        }
        guard host.hostUse == HostUseClassification.Use.dedicatedServing.rawValue,
            host.hostUseSource == HostUseClassification.Source.operatorAssertion.rawValue
        else {
            throw DedicatedServingQualificationError.invalidHostUse
        }
        guard host.physicalRAMBytes > 0, host.originalWiredLimitBytes > 0 else {
            throw DedicatedServingQualificationError.invalidHostMemory
        }
        guard host.metalRecommendedWorkingSetBytes > 0, host.metalCurrentAllocatedBytes >= 0 else {
            throw DedicatedServingQualificationError.invalidMetalObservation
        }
        guard host.originalWiredLimitProvenance == .measured else {
            throw DedicatedServingQualificationError.unmeasuredOriginalWiredLimit
        }
        guard !host.osBuild.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DedicatedServingQualificationError.missingOSBuild
        }
        guard host.osBuildSource == DedicatedServingQualificationHost.processInfoOSBuildSource else {
            throw DedicatedServingQualificationError.invalidOSBuildSource
        }
    }

    private static func validateCandidateLadder(proposedMiB: Int, followOnMiB: [Int]) throws {
        guard let index = requiredCandidateLadderMiB.firstIndex(of: proposedMiB) else {
            throw DedicatedServingQualificationError.invalidCandidate("candidate must be one of the qualification ladder steps")
        }
        let requiredSuffix = Array(requiredCandidateLadderMiB.suffix(from: index + 1))
        guard followOnMiB == requiredSuffix else {
            throw DedicatedServingQualificationError.invalidCandidate("follow-on candidates must be the exact remaining ladder suffix")
        }
    }

    private static func candidateCeiling(mib: Int) throws -> DedicatedServingCandidateCeiling {
        guard mib > 0 else {
            throw DedicatedServingQualificationError.invalidCandidate("candidate ceiling MiB must be positive")
        }
        let multiplied = mib.multipliedReportingOverflow(by: Self.mib)
        guard !multiplied.overflow else {
            throw DedicatedServingQualificationError.invalidCandidate("candidate ceiling bytes overflow")
        }
        return DedicatedServingCandidateCeiling(mib: mib, bytes: multiplied.partialValue)
    }

    private static func validateCandidate(
        _ candidate: DedicatedServingCandidateCeiling,
        host: DedicatedServingQualificationHost
    ) throws {
        guard candidate.bytes > host.originalWiredLimitBytes else {
            throw DedicatedServingQualificationError.invalidCandidate("candidate must raise the original wired limit")
        }
        guard candidate.bytes < host.physicalRAMBytes else {
            throw DedicatedServingQualificationError.invalidCandidate("candidate must remain below physical RAM")
        }
    }

    private static func validateBudgets(
        _ budgets: DedicatedServingQualificationBudgets,
        availableAfterReserve: Int
    ) throws {
        guard budgets.memoryLimitBytes > 0 else {
            throw DedicatedServingQualificationError.invalidBudget("memory limit must be positive")
        }
        guard budgets.cacheLimitBytes > 0 else {
            throw DedicatedServingQualificationError.invalidBudget("cache limit must be positive")
        }
        guard budgets.kvBudgetBytes > 0 else {
            throw DedicatedServingQualificationError.invalidBudget("KV budget must be positive")
        }
        guard budgets.ioPrefetchBudgetBytes > 0 else {
            throw DedicatedServingQualificationError.invalidBudget("I/O prefetch budget must be positive")
        }
        guard budgets.cacheLimitBytes < budgets.memoryLimitBytes else {
            throw DedicatedServingQualificationError.invalidBudget("cache limit must be below memory limit")
        }
        guard budgets.memoryLimitBytes <= availableAfterReserve else {
            throw DedicatedServingQualificationError.invalidBudget("memory limit overcommits candidate minus reserve")
        }
        let cacheAndKV = try checkedSum(
            budgets.cacheLimitBytes,
            budgets.kvBudgetBytes,
            message: "cache plus KV budget overflow")
        guard cacheAndKV <= budgets.memoryLimitBytes else {
            throw DedicatedServingQualificationError.invalidBudget("cache plus KV budget must fit within MLX memory limit")
        }
        let memoryAndIO = try checkedSum(
            budgets.memoryLimitBytes,
            budgets.ioPrefetchBudgetBytes,
            message: "memory plus I/O prefetch budget overflow")
        guard memoryAndIO <= availableAfterReserve else {
            throw DedicatedServingQualificationError.invalidBudget(
                "MLX memory limit plus I/O prefetch budget overcommits candidate minus reserve")
        }
    }

    private static func checkedSum(_ lhs: Int, _ rhs: Int, message: String) throws -> Int {
        let sum = lhs.addingReportingOverflow(rhs)
        guard !sum.overflow else {
            throw DedicatedServingQualificationError.invalidBudget(message)
        }
        return sum.partialValue
    }

    fileprivate static func validateDecodedArtifact(_ artifact: DedicatedServingQualificationArtifact) throws {
        let recomputed = try plan(DedicatedServingQualificationInput(
            host: artifact.host,
            osServiceReserveBytes: artifact.osServiceReserveBytes,
            candidateCeilingMiB: artifact.proposedCandidateCeiling.mib,
            stagedFollowOnCandidateMiB: artifact.stagedFollowOnCandidates.map(\.mib),
            mlxBudgets: artifact.mlxBudgets))
        guard artifact == recomputed else {
            throw DedicatedServingQualificationError.invalidArtifact("artifact payload does not match planner output")
        }
    }

    private static func qualificationStages() -> [DedicatedServingQualificationStage] {
        [
            DedicatedServingQualificationStage(
                id: .captureOriginal,
                dryRunOnly: true,
                description: "Record host evidence and safety baselines before any privileged action.",
                gates: [
                    .osBuildCaptured,
                    .physicalRAMCaptured,
                    .originalWiredLimitCaptured,
                    .metalRecommendedWorkingSetCaptured,
                    .metalCurrentAllocationCaptured,
                    .swapPageoutBaselineCaptured,
                    .memoryPressureBaselineCaptured,
                    .runningServicesBaselineCaptured,
                ]),
            DedicatedServingQualificationStage(
                id: .temporaryApply,
                dryRunOnly: true,
                description: "Temporarily apply the candidate ceiling only in an authorized execution path.",
                gates: [.candidateTemporarilyApplied]),
            DedicatedServingQualificationStage(
                id: .readbackRecommendedWorkingSetCheck,
                dryRunOnly: true,
                description: "Read back wired ceiling and Metal recommended working-set observations.",
                gates: [.wiredLimitReadbackMatchesCandidate, .recommendedWorkingSetObserved]),
            DedicatedServingQualificationStage(
                id: .safetyHealthSoakGates,
                dryRunOnly: true,
                description: "Qualify safety, health, allocation, latency, and thermal behavior.",
                gates: [
                    .swapAndPageoutsStable,
                    .memoryPressureNormal,
                    .noOOMOrSIGKILL,
                    .responsiveHealth,
                    .boundedAllocationLatencyThermal,
                ]),
            DedicatedServingQualificationStage(
                id: .restoreOnAnyFailure,
                dryRunOnly: true,
                description: "Restore the original wired ceiling on any failed gate.",
                gates: [.originalWiredLimitRestored]),
        ]
    }
}
