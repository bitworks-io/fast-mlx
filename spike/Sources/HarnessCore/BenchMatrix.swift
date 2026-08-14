import Foundation

public enum Workload: String, Sendable, CaseIterable { case prefill, decode, echo, code }
public enum Mode: String, Sendable, CaseIterable { case none, pld, dspark }

/// Audited first-party prompt used by the batch-1 runtime frontier. KVTuner permits only this
/// source because an arbitrary prompt cannot truthfully assert disjoint upstream source rows.
public let defaultBenchPrompt =
    "Explain in at least 250 words how continuous batching improves LLM serving throughput. "
    + "Cover request scheduling, chunked prefill, decode interleaving, fairness, memory pressure, "
    + "and cancellation."

public struct Cell: Sendable, Hashable {
    public let workload: Workload
    public let mode: Mode
    public let model: String
    public let quant: String
    public let concurrency: Int
    public init(workload: Workload, mode: Mode, model: String, quant: String = "fp16", concurrency: Int = 1) {
        self.workload = workload; self.mode = mode; self.model = model; self.quant = quant; self.concurrency = concurrency
    }
}

public struct RateAggregate: Sendable, Equatable { public let mean: Double; public let runs: Int
    public init(mean: Double, runs: Int) { self.mean = mean; self.runs = runs }
}

/// Drops run 0 (warmup) unconditionally, then averages the remaining non-nil (non-skipped) rates.
/// `nil` entries elsewhere in the array represent explicitly skipped runs and are excluded too.
public func aggregateRates(_ rates: [Double?]) -> RateAggregate {
    let postWarmup = rates.dropFirst()
    let valid = postWarmup.compactMap { $0 }
    guard !valid.isEmpty else { return RateAggregate(mean: 0, runs: 0) }
    return RateAggregate(mean: valid.reduce(0, +) / Double(valid.count), runs: valid.count)
}

/// Salts a prompt per run+nonce so repeated bench runs don't hit a KV/prefix cache that would
/// understate decode cost (backlog methodology: never bench a cached prefix as if it were cold).
public func saltPrompt(run: Int, nonce: String, _ basePrompt: String) -> String {
    "\(basePrompt) [run=\(run) nonce=\(nonce)]"
}

public enum BenchWorkloadIdentityError: Error, Sendable, Equatable {
    case invalidNonce
    case invalidIterations
}

/// Exact prompt set shared across separately launched KV-tier measurements. An explicit nonce is
/// part of durable evidence so every cell can replay identical cold, salted prompt bytes.
public struct BenchWorkloadIdentity: Sendable, Equatable {
    public let basePrompt: String
    public let nonce: String
    public let iterations: Int

    public init(basePrompt: String, nonce: String, iterations: Int) throws {
        guard iterations > 0 else {
            throw BenchWorkloadIdentityError.invalidIterations
        }
        do {
            _ = try ServiceWorkloadIdentity(nonce: nonce)
        } catch {
            throw BenchWorkloadIdentityError.invalidNonce
        }
        self.basePrompt = basePrompt
        self.nonce = nonce
        self.iterations = iterations
    }

    public func prompt(run: Int) -> String {
        precondition((0..<iterations).contains(run), "bench run outside workload identity")
        return saltPrompt(run: run, nonce: nonce, basePrompt)
    }

    public var prompts: [String] {
        (0..<iterations).map(prompt(run:))
    }
}

/// Direct prefill rate from the actor-timed `decoder.prefill` span. This never derives prefill
/// from TTFT, which also contains first-token bookkeeping and is a distinct latency metric.
public func prefillTokensPerSecond(
    promptTokens: Int,
    durationSeconds: Double
) -> Double? {
    guard promptTokens > 0, durationSeconds.isFinite, durationSeconds > 0 else {
        return nil
    }
    return Double(promptTokens) / durationSeconds
}

public enum BenchMemoryEvidenceError: Error, Sendable, Equatable {
    case invalidRunSampleCount(Int)
    case emptyRuns
}

/// Raw, recomputable process/MLX memory evidence for one post-warmup batch-1 run. Exactly two
/// samples are retained: immediately before generation (after resetting MLX's peak counter) and
/// immediately after cache telemetry has been captured. Whole-process maximum RSS remains an
/// independent per-process runner artifact because endpoint sampling cannot observe transients.
public struct BenchRunMemoryEvidence: Sendable, Codable, Equatable {
    public let samples: [ServiceMemorySample]
    public let summary: ServiceMemorySummary

    public init(samples: [ServiceMemorySample]) throws {
        guard samples.count == 2 else {
            throw BenchMemoryEvidenceError.invalidRunSampleCount(
                samples.count)
        }
        self.samples = samples
        self.summary = try summarizeServiceMemory(samples)
    }

    private enum CodingKeys: String, CodingKey {
        case samples
        case summary
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let samples = try container.decode(
            [ServiceMemorySample].self, forKey: .samples)
        let claimedSummary = try container.decode(
            ServiceMemorySummary.self, forKey: .summary)
        let derived = try BenchRunMemoryEvidence(samples: samples)
        guard claimedSummary == derived.summary else {
            throw DecodingError.dataCorruptedError(
                forKey: .summary,
                in: container,
                debugDescription:
                    "bench memory summary does not match its raw samples")
        }
        self = derived
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(samples, forKey: .samples)
        try container.encode(summary, forKey: .summary)
    }
}

/// Headline reducer used only for display/adjudication. Durable evidence keeps every run above,
/// so all maxima remain independently recomputable rather than trusting these convenience fields.
public struct BenchMemoryAggregate: Sendable, Equatable {
    public let measuredRuns: Int
    public let maxSampledPhysicalFootprintBytes: UInt64
    public let maxMLXActiveBytes: Int
    public let maxMLXCacheBytes: Int
    public let maxMLXPeakBytes: Int

    public init(runs: [BenchRunMemoryEvidence]) throws {
        guard !runs.isEmpty else {
            throw BenchMemoryEvidenceError.emptyRuns
        }
        measuredRuns = runs.count
        maxSampledPhysicalFootprintBytes = runs.map {
            $0.summary.maxSampledFootprintBytes
        }.max()!
        maxMLXActiveBytes = runs.map {
            $0.summary.maxMLXActiveBytes
        }.max()!
        maxMLXCacheBytes = runs.map {
            $0.summary.maxMLXCacheBytes
        }.max()!
        maxMLXPeakBytes = runs.map {
            $0.summary.maxMLXPeakBytes
        }.max()!
    }
}

public enum BenchQualificationEvidenceError: Error, Sendable, Equatable {
    case invalidRunnerManifestSHA256
    case invalidTokenizerSHA256
    case invalidMatrixPosition
    case invalidMemorySettings
    case invalidMonotonicTiming
    case invalidProcessMemory
    case invalidPowerState
    case invalidThermalState
    case invalidThermalPolicy
    case invalidWarmupEvidence
    case invalidThermalAdmission
    case invalidThermalTargetContract
    case invalidRunCount(Int)
    case invalidSchemaVersion(Int)
    case invalidCapacityEvidence
}

public enum BenchQualificationCacheResetPolicy:
    String, Codable, Sendable, Equatable
{
    /// `CompiledMLXDecoder.reset()` preserves compiled array identity while clearing every
    /// request's logical KV state before the warmup or retained generation begins.
    case inPlaceBeforeEveryGeneration = "in-place-before-every-generation"
}

public enum BenchQualificationModelResidencyPolicy:
    String, Codable, Sendable, Equatable
{
    /// One checkpoint load is retained for the warmup and measured generation in this process.
    case loadOncePerProcess = "load-once-per-process"
}

public enum BenchQualificationProcessIsolationPolicy:
    String, Codable, Sendable, Equatable
{
    /// The matrix runner starts a new harness process for every cell/block position. This keeps
    /// allocator residue from one cache representation out of another cell's retained row.
    case freshProcessPerMatrixPosition = "fresh-process-per-matrix-position"
}

/// Qualification rows target the unthrottled cohort. Fair/serious/critical states remain
/// observable in the warmup receipt, but are never valid admission targets for retained work.
public enum BenchQualificationThermalTarget:
    String, Codable, Sendable, Equatable
{
    case nominal

    fileprivate var hostState: CompressedAttentionProbeThermalState {
        switch self {
        case .nominal: .nominal
        }
    }
}

/// Bounded policy applied after the dropped warmup and before the one retained generation.
/// Milliseconds keep the manifest/CLI representation integral and exactly representable by jq.
public struct BenchQualificationThermalPolicy:
    Codable, Sendable, Equatable
{
    public let target: BenchQualificationThermalTarget
    public let timeoutSeconds: Int
    public let pollIntervalMilliseconds: Int
    public let stabilitySeconds: Int

    public init(
        target: BenchQualificationThermalTarget,
        timeoutSeconds: Int,
        pollIntervalMilliseconds: Int,
        stabilitySeconds: Int = 0
    ) throws {
        self.target = target
        self.timeoutSeconds = timeoutSeconds
        self.pollIntervalMilliseconds = pollIntervalMilliseconds
        self.stabilitySeconds = stabilitySeconds
        try validate()
    }

    private enum CodingKeys: String, CodingKey {
        case target
        case timeoutSeconds
        case pollIntervalMilliseconds
        case stabilitySeconds
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        target = try container.decode(
            BenchQualificationThermalTarget.self, forKey: .target)
        timeoutSeconds = try container.decode(
            Int.self, forKey: .timeoutSeconds)
        pollIntervalMilliseconds = try container.decode(
            Int.self, forKey: .pollIntervalMilliseconds)
        stabilitySeconds = try container.decodeIfPresent(
            Int.self, forKey: .stabilitySeconds) ?? 0
        try validate()
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(target, forKey: .target)
        try container.encode(timeoutSeconds, forKey: .timeoutSeconds)
        try container.encode(
            pollIntervalMilliseconds,
            forKey: .pollIntervalMilliseconds)
        if stabilitySeconds > 0 {
            try container.encode(
                stabilitySeconds, forKey: .stabilitySeconds)
        }
    }

    fileprivate func validate() throws {
        guard (1 ... 3_600).contains(timeoutSeconds),
            (100 ... 60_000).contains(pollIntervalMilliseconds),
            pollIntervalMilliseconds <= timeoutSeconds * 1_000,
            (0 ... timeoutSeconds).contains(stabilitySeconds)
        else {
            throw BenchQualificationEvidenceError.invalidThermalPolicy
        }
    }
}

/// Static identity for one isolated position in a loaded-model qualification matrix. The runner
/// manifest digest binds the declared order outside this process; the remaining fields make every
/// row independently reject partial order or memory-policy evidence.
public struct BenchQualificationContext: Codable, Sendable, Equatable {
    public let runnerManifestSHA256: String
    public let matrixBlockIndex: Int
    public let matrixRunPosition: Int
    public let matrixCellCount: Int
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let wiredLimitBytes: Int
    public let tokenizerSHA256: String
    public let cacheResetPolicy: BenchQualificationCacheResetPolicy
    public let modelResidencyPolicy: BenchQualificationModelResidencyPolicy
    public let processIsolationPolicy: BenchQualificationProcessIsolationPolicy
    public let postWarmupThermalPolicy: BenchQualificationThermalPolicy?

    public init(
        runnerManifestSHA256: String,
        matrixBlockIndex: Int,
        matrixRunPosition: Int,
        matrixCellCount: Int,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        wiredLimitBytes: Int,
        tokenizerSHA256: String,
        cacheResetPolicy: BenchQualificationCacheResetPolicy,
        modelResidencyPolicy: BenchQualificationModelResidencyPolicy,
        processIsolationPolicy: BenchQualificationProcessIsolationPolicy,
        postWarmupThermalPolicy: BenchQualificationThermalPolicy? = nil
    ) throws {
        self.runnerManifestSHA256 = runnerManifestSHA256
        self.matrixBlockIndex = matrixBlockIndex
        self.matrixRunPosition = matrixRunPosition
        self.matrixCellCount = matrixCellCount
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.wiredLimitBytes = wiredLimitBytes
        self.tokenizerSHA256 = tokenizerSHA256
        self.cacheResetPolicy = cacheResetPolicy
        self.modelResidencyPolicy = modelResidencyPolicy
        self.processIsolationPolicy = processIsolationPolicy
        self.postWarmupThermalPolicy = postWarmupThermalPolicy
        try validate()
    }

    fileprivate func validate() throws {
        guard Self.isLowercaseSHA256(runnerManifestSHA256) else {
            throw BenchQualificationEvidenceError
                .invalidRunnerManifestSHA256
        }
        guard Self.isLowercaseSHA256(tokenizerSHA256) else {
            throw BenchQualificationEvidenceError.invalidTokenizerSHA256
        }
        guard matrixBlockIndex >= 0, matrixCellCount > 0,
            (0 ..< matrixCellCount).contains(matrixRunPosition)
        else {
            throw BenchQualificationEvidenceError.invalidMatrixPosition
        }
        guard cacheLimitBytes > 0, memoryLimitBytes >= cacheLimitBytes,
            wiredLimitBytes >= memoryLimitBytes
        else {
            throw BenchQualificationEvidenceError.invalidMemorySettings
        }
        try postWarmupThermalPolicy?.validate()
    }

    private static func isLowercaseSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (48 ... 57).contains($0) || (97 ... 102).contains($0)
        }
    }
}

/// Host receipts captured immediately around one retained generation. MLX active/cache/peak and
/// endpoint footprint remain in `BenchRunMemoryEvidence`; resident size is retained here because
/// the isolated runner also records the process-wide maximum RSS independently.
public struct BenchQualificationHostSnapshot:
    Codable, Sendable, Equatable
{
    public let monotonicTimestampSeconds: Double
    public let residentSizeBytes: Int
    public let physicalFootprintBytes: Int
    public let lowPowerModeEnabled: Bool
    public let powerSource: CompressedAttentionProbePowerSource
    public let thermalState: CompressedAttentionProbeThermalState

    public init(
        monotonicTimestampSeconds: Double,
        residentSizeBytes: Int,
        physicalFootprintBytes: Int,
        lowPowerModeEnabled: Bool,
        powerSource: CompressedAttentionProbePowerSource,
        thermalState: CompressedAttentionProbeThermalState
    ) {
        self.monotonicTimestampSeconds = monotonicTimestampSeconds
        self.residentSizeBytes = residentSizeBytes
        self.physicalFootprintBytes = physicalFootprintBytes
        self.lowPowerModeEnabled = lowPowerModeEnabled
        self.powerSource = powerSource
        self.thermalState = thermalState
    }
}

public struct BenchQualificationRunEnvironment:
    Codable, Sendable, Equatable
{
    public let before: BenchQualificationHostSnapshot
    public let after: BenchQualificationHostSnapshot

    public init(
        before: BenchQualificationHostSnapshot,
        after: BenchQualificationHostSnapshot
    ) throws {
        self.before = before
        self.after = after
        try validate()
    }

    fileprivate func validate() throws {
        guard before.monotonicTimestampSeconds.isFinite,
            after.monotonicTimestampSeconds.isFinite,
            after.monotonicTimestampSeconds
                > before.monotonicTimestampSeconds
        else {
            throw BenchQualificationEvidenceError.invalidMonotonicTiming
        }
        guard before.residentSizeBytes > 0, after.residentSizeBytes > 0,
            before.physicalFootprintBytes > 0,
            after.physicalFootprintBytes > 0
        else {
            throw BenchQualificationEvidenceError.invalidProcessMemory
        }
        guard before.lowPowerModeEnabled == after.lowPowerModeEnabled,
            before.powerSource == after.powerSource,
            before.powerSource != .unavailable
        else {
            throw BenchQualificationEvidenceError.invalidPowerState
        }
        guard before.thermalState == after.thermalState,
            before.thermalState != .unknown
        else {
            throw BenchQualificationEvidenceError.invalidThermalState
        }
    }
}

/// The warmup is intentionally excluded from performance aggregates but retained here as the
/// causal thermal receipt. It may transition between nominal and fair; unsafe/unknown thermal,
/// power-source drift, battery power, and low-power mode fail closed.
public struct BenchQualificationWarmupEnvironment:
    Codable, Sendable, Equatable
{
    public let before: BenchQualificationHostSnapshot
    public let after: BenchQualificationHostSnapshot

    public init(
        before: BenchQualificationHostSnapshot,
        after: BenchQualificationHostSnapshot
    ) throws {
        self.before = before
        self.after = after
        try validate()
    }

    fileprivate func validate() throws {
        guard before.monotonicTimestampSeconds.isFinite,
            after.monotonicTimestampSeconds.isFinite,
            after.monotonicTimestampSeconds
                > before.monotonicTimestampSeconds,
            before.residentSizeBytes > 0,
            after.residentSizeBytes > 0,
            before.physicalFootprintBytes > 0,
            after.physicalFootprintBytes > 0,
            before.powerSource == .acPower,
            after.powerSource == .acPower,
            !before.lowPowerModeEnabled,
            !after.lowPowerModeEnabled,
            Self.isSafeThermalState(before.thermalState),
            Self.isSafeThermalState(after.thermalState)
        else {
            throw BenchQualificationEvidenceError.invalidWarmupEvidence
        }
    }

    private static func isSafeThermalState(
        _ state: CompressedAttentionProbeThermalState
    ) -> Bool {
        state == .nominal || state == .fair
    }
}

/// Sampled terminal nominal interval and exact host snapshot at which the bounded post-warmup
/// wait admitted the retained generation. Historical schema-v3 evidence contains only `snapshot`;
/// current evidence also retains every sample used to prove the manifest-bound stability window.
public struct BenchQualificationThermalAdmission:
    Codable, Sendable, Equatable
{
    public let snapshot: BenchQualificationHostSnapshot
    public let stabilityObservations: [BenchQualificationHostSnapshot]

    public init(snapshot: BenchQualificationHostSnapshot) throws {
        self.snapshot = snapshot
        stabilityObservations = [snapshot]
        try validate()
    }

    public init(
        stabilityObservations: [BenchQualificationHostSnapshot]
    ) throws {
        guard let snapshot = stabilityObservations.last else {
            throw BenchQualificationEvidenceError.invalidThermalAdmission
        }
        self.snapshot = snapshot
        self.stabilityObservations = stabilityObservations
        try validate()
    }

    private enum CodingKeys: String, CodingKey {
        case snapshot
        case stabilityObservations
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        snapshot = try container.decode(
            BenchQualificationHostSnapshot.self, forKey: .snapshot)
        stabilityObservations = try container.decodeIfPresent(
            [BenchQualificationHostSnapshot].self,
            forKey: .stabilityObservations) ?? [snapshot]
        try validate()
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(snapshot, forKey: .snapshot)
        if stabilityObservations.count > 1 {
            try container.encode(
                stabilityObservations,
                forKey: .stabilityObservations)
        }
    }

    fileprivate func validate() throws {
        guard !stabilityObservations.isEmpty,
            stabilityObservations.last == snapshot
        else {
            throw BenchQualificationEvidenceError.invalidThermalAdmission
        }
        var previousTimestamp: Double?
        for observation in stabilityObservations {
            guard observation.monotonicTimestampSeconds.isFinite,
                observation.residentSizeBytes > 0,
                observation.physicalFootprintBytes > 0,
                observation.powerSource == .acPower,
                !observation.lowPowerModeEnabled,
                observation.thermalState == .nominal,
                previousTimestamp.map({
                    observation.monotonicTimestampSeconds > $0
                }) ?? true
            else {
                throw BenchQualificationEvidenceError
                    .invalidThermalAdmission
            }
            previousTimestamp = observation.monotonicTimestampSeconds
        }
    }
}

/// Qualification-only payload nested inside a historical `bench` row. Exactly one retained run
/// is allowed per process: cross-cell counterbalancing and the required three-or-more repetitions
/// happen at the isolated matrix-runner boundary, preventing allocator history from being shared.
public struct BenchQualificationEvidence: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 4
    public static let instantaneousAdmissionSchemaVersion = 3
    public static let legacySchemaVersion = 2

    public let schemaVersion: Int
    public let context: BenchQualificationContext
    public let warmup: BenchQualificationWarmupEnvironment?
    public let postWarmupThermalAdmission:
        BenchQualificationThermalAdmission?
    public let runs: [BenchQualificationRunEnvironment]

    public init(
        context: BenchQualificationContext,
        runs: [BenchQualificationRunEnvironment]
    ) throws {
        schemaVersion = Self.legacySchemaVersion
        self.context = context
        warmup = nil
        postWarmupThermalAdmission = nil
        self.runs = runs
        try validate()
    }

    public init(
        context: BenchQualificationContext,
        warmup: BenchQualificationWarmupEnvironment?,
        postWarmupThermalAdmission: BenchQualificationThermalAdmission?,
        runs: [BenchQualificationRunEnvironment]
    ) throws {
        schemaVersion = (context.postWarmupThermalPolicy?
            .stabilitySeconds ?? 0) > 0
            ? Self.currentSchemaVersion
            : Self.instantaneousAdmissionSchemaVersion
        self.context = context
        self.warmup = warmup
        self.postWarmupThermalAdmission =
            postWarmupThermalAdmission
        self.runs = runs
        try validate()
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case context
        case warmup
        case postWarmupThermalAdmission
        case runs
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let schemaVersion = try container.decode(
            Int.self, forKey: .schemaVersion)
        guard schemaVersion == Self.currentSchemaVersion
            || schemaVersion == Self.instantaneousAdmissionSchemaVersion
            || schemaVersion == Self.legacySchemaVersion
        else {
            throw BenchQualificationEvidenceError
                .invalidSchemaVersion(schemaVersion)
        }
        let context = try container.decode(
            BenchQualificationContext.self, forKey: .context)
        let runs = try container.decode(
            [BenchQualificationRunEnvironment].self, forKey: .runs)
        self.schemaVersion = schemaVersion
        self.context = context
        warmup = try container.decodeIfPresent(
            BenchQualificationWarmupEnvironment.self,
            forKey: .warmup)
        postWarmupThermalAdmission = try container.decodeIfPresent(
            BenchQualificationThermalAdmission.self,
            forKey: .postWarmupThermalAdmission)
        self.runs = runs
        try validate()
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(context, forKey: .context)
        try container.encodeIfPresent(warmup, forKey: .warmup)
        try container.encodeIfPresent(
            postWarmupThermalAdmission,
            forKey: .postWarmupThermalAdmission)
        try container.encode(runs, forKey: .runs)
    }

    private func validate() throws {
        try context.validate()
        guard runs.count == 1 else {
            throw BenchQualificationEvidenceError.invalidRunCount(runs.count)
        }
        for run in runs { try run.validate() }
        switch schemaVersion {
        case Self.legacySchemaVersion:
            guard context.postWarmupThermalPolicy == nil,
                warmup == nil,
                postWarmupThermalAdmission == nil
            else {
                throw BenchQualificationEvidenceError
                    .invalidThermalTargetContract
            }
        case Self.instantaneousAdmissionSchemaVersion,
            Self.currentSchemaVersion:
            guard let policy = context.postWarmupThermalPolicy,
                let warmup,
                let admission = postWarmupThermalAdmission,
                let run = runs.first
            else {
                throw BenchQualificationEvidenceError
                    .invalidThermalTargetContract
            }
            try policy.validate()
            try warmup.validate()
            try admission.validate()
            if schemaVersion == Self.instantaneousAdmissionSchemaVersion {
                guard policy.stabilitySeconds == 0,
                    admission.stabilityObservations.count == 1
                else {
                    throw BenchQualificationEvidenceError
                        .invalidThermalTargetContract
                }
            } else {
                guard policy.stabilitySeconds > 0,
                    let firstObservation =
                        admission.stabilityObservations.first,
                    admission.stabilityObservations.count >= 2,
                    firstObservation.monotonicTimestampSeconds
                        >= warmup.after.monotonicTimestampSeconds,
                    admission.snapshot.monotonicTimestampSeconds
                        - firstObservation.monotonicTimestampSeconds
                        >= Double(policy.stabilitySeconds)
                else {
                    throw BenchQualificationEvidenceError
                        .invalidThermalAdmission
                }
            }
            guard warmup.after.monotonicTimestampSeconds
                    <= admission.snapshot.monotonicTimestampSeconds,
                admission.snapshot.monotonicTimestampSeconds
                    - warmup.after.monotonicTimestampSeconds
                    <= Double(policy.timeoutSeconds),
                admission.snapshot.monotonicTimestampSeconds
                    <= run.before.monotonicTimestampSeconds,
                admission.snapshot.thermalState == policy.target.hostState,
                run.before.thermalState == policy.target.hostState,
                run.after.thermalState == policy.target.hostState,
                run.before.powerSource == .acPower,
                run.after.powerSource == .acPower,
                !run.before.lowPowerModeEnabled,
                !run.after.lowPowerModeEnabled
            else {
                throw BenchQualificationEvidenceError
                    .invalidThermalAdmission
            }
        default:
            throw BenchQualificationEvidenceError
                .invalidSchemaVersion(schemaVersion)
        }
    }
}

/// Capacity receipts are deliberately a separate evidence lane from speed qualification. They
/// authenticate loaded cache/storage and process-memory behavior while permitting safe thermal
/// drift between nominal and fair. The explicit non-promotable marker and distinct payload keep
/// these rows out of qualification reducers.
public enum BenchCapacityEvidencePurpose:
    String, Codable, Sendable, Equatable
{
    case capacityOnly = "capacity-only"
}

public enum BenchDirectCompressedCapacityRouteKind:
    String, Sendable, Equatable
{
    case affine
    case kvarn
}

/// Pure, closed admission contract for capacity-only evidence rows. The harness runtime still
/// owns execution, but parser and typed validators share this route set so capacity receipts do
/// not drift between CLI spelling and post-decode evidence checks.
public struct BenchDirectCompressedCapacityRoute: Sendable, Equatable {
    public let tier: String
    public let request: CompressedKVAttentionRequest
    public let kind: BenchDirectCompressedCapacityRouteKind
    public let counterPrefix: String
    public let groupSize: Int

    public init?(tier: String, request: CompressedKVAttentionRequest?) {
        guard let request else { return nil }
        switch (tier, request) {
        case ("affine-k4v2-g64", .splitAffineQuantizedMM):
            self.tier = tier
            self.request = request
            kind = .affine
            counterPrefix = "affine"
            groupSize = 64
        case ("kvarn-k4v2-g128", .splitKVarNQuantizedMM),
            ("kvarn-k4v2-g128-i16", .splitKVarNQuantizedMM):
            self.tier = tier
            self.request = request
            kind = .kvarn
            counterPrefix = "kvarn"
            groupSize = 128
        default:
            return nil
        }
    }
}

public struct BenchCapacityEvidenceContext:
    Codable, Sendable, Equatable
{
    public let evidenceManifestSHA256: String
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let wiredLimitBytes: Int
    public let expectedPromptTokens: Int
    public let tokenizerSHA256: String
    public let cacheResetPolicy: BenchQualificationCacheResetPolicy
    public let modelResidencyPolicy: BenchQualificationModelResidencyPolicy
    public let processIsolationPolicy:
        BenchQualificationProcessIsolationPolicy

    public init(
        evidenceManifestSHA256: String,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        wiredLimitBytes: Int,
        expectedPromptTokens: Int,
        tokenizerSHA256: String,
        cacheResetPolicy: BenchQualificationCacheResetPolicy,
        modelResidencyPolicy: BenchQualificationModelResidencyPolicy,
        processIsolationPolicy: BenchQualificationProcessIsolationPolicy
    ) throws {
        self.evidenceManifestSHA256 = evidenceManifestSHA256
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.wiredLimitBytes = wiredLimitBytes
        self.expectedPromptTokens = expectedPromptTokens
        self.tokenizerSHA256 = tokenizerSHA256
        self.cacheResetPolicy = cacheResetPolicy
        self.modelResidencyPolicy = modelResidencyPolicy
        self.processIsolationPolicy = processIsolationPolicy
        try validate()
    }

    fileprivate func validate() throws {
        guard Self.isLowercaseSHA256(evidenceManifestSHA256) else {
            throw BenchQualificationEvidenceError
                .invalidRunnerManifestSHA256
        }
        guard Self.isLowercaseSHA256(tokenizerSHA256) else {
            throw BenchQualificationEvidenceError.invalidTokenizerSHA256
        }
        guard cacheLimitBytes > 0,
            memoryLimitBytes >= cacheLimitBytes,
            wiredLimitBytes >= memoryLimitBytes,
            expectedPromptTokens > 0
        else {
            throw BenchQualificationEvidenceError.invalidMemorySettings
        }
    }

    private static func isLowercaseSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (48 ... 57).contains($0) || (97 ... 102).contains($0)
        }
    }
}

public struct BenchCapacityRunEnvironment:
    Codable, Sendable, Equatable
{
    public let before: BenchQualificationHostSnapshot
    public let after: BenchQualificationHostSnapshot

    public init(
        before: BenchQualificationHostSnapshot,
        after: BenchQualificationHostSnapshot
    ) throws {
        self.before = before
        self.after = after
        try validate()
    }

    fileprivate func validate() throws {
        guard before.monotonicTimestampSeconds.isFinite,
            after.monotonicTimestampSeconds.isFinite,
            after.monotonicTimestampSeconds
                > before.monotonicTimestampSeconds
        else {
            throw BenchQualificationEvidenceError.invalidMonotonicTiming
        }
        guard before.residentSizeBytes > 0,
            after.residentSizeBytes > 0,
            before.physicalFootprintBytes > 0,
            after.physicalFootprintBytes > 0
        else {
            throw BenchQualificationEvidenceError.invalidProcessMemory
        }
        guard before.powerSource == .acPower,
            after.powerSource == .acPower,
            !before.lowPowerModeEnabled,
            !after.lowPowerModeEnabled
        else {
            throw BenchQualificationEvidenceError.invalidPowerState
        }
        guard Self.isSafe(before.thermalState),
            Self.isSafe(after.thermalState)
        else {
            throw BenchQualificationEvidenceError.invalidThermalState
        }
    }

    private static func isSafe(
        _ state: CompressedAttentionProbeThermalState
    ) -> Bool {
        state == .nominal || state == .fair
    }
}

public struct BenchCapacityEvidence: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    public let schemaVersion: Int
    public let purpose: BenchCapacityEvidencePurpose
    public let promotable: Bool
    public let context: BenchCapacityEvidenceContext
    public let warmup: BenchCapacityRunEnvironment
    public let runs: [BenchCapacityRunEnvironment]

    public init(
        context: BenchCapacityEvidenceContext,
        warmup: BenchCapacityRunEnvironment,
        runs: [BenchCapacityRunEnvironment]
    ) throws {
        schemaVersion = Self.currentSchemaVersion
        purpose = .capacityOnly
        promotable = false
        self.context = context
        self.warmup = warmup
        self.runs = runs
        try validate()
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case purpose
        case promotable
        case context
        case warmup
        case runs
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(
            Int.self, forKey: .schemaVersion)
        purpose = try container.decode(
            BenchCapacityEvidencePurpose.self, forKey: .purpose)
        promotable = try container.decode(Bool.self, forKey: .promotable)
        context = try container.decode(
            BenchCapacityEvidenceContext.self, forKey: .context)
        warmup = try container.decode(
            BenchCapacityRunEnvironment.self, forKey: .warmup)
        runs = try container.decode(
            [BenchCapacityRunEnvironment].self, forKey: .runs)
        try validate()
    }

    private func validate() throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw BenchQualificationEvidenceError
                .invalidSchemaVersion(schemaVersion)
        }
        guard purpose == .capacityOnly, !promotable else {
            throw BenchQualificationEvidenceError.invalidCapacityEvidence
        }
        try context.validate()
        try warmup.validate()
        guard runs.count == 1 else {
            throw BenchQualificationEvidenceError.invalidRunCount(runs.count)
        }
        for run in runs { try run.validate() }
    }
}

public enum ServiceWorkloadIdentityError: Error, Sendable, Equatable {
    case invalidNonce
}

/// Stable identity shared by every process in a service-policy frontier. Keeping the nonce
/// explicit prevents separate policy/concurrency invocations from benchmarking different
/// salted prompts while claiming a direct comparison.
public struct ServiceWorkloadIdentity: Sendable, Equatable {
    public let nonce: String

    public init(nonce: String) throws {
        let alphanumericByte: (UInt8) -> Bool = {
            (48 ... 57).contains($0) || (65 ... 90).contains($0)
                || (97 ... 122).contains($0)
        }
        let validByte: (UInt8) -> Bool = {
            alphanumericByte($0) || $0 == 45 || $0 == 46 || $0 == 95
        }
        guard let first = nonce.utf8.first, alphanumericByte(first),
            nonce.utf8.count <= 64, nonce.utf8.allSatisfy(validByte)
        else {
            throw ServiceWorkloadIdentityError.invalidNonce
        }
        self.nonce = nonce
    }

    public func prompt(basePrompt: String, run: Int, request: Int) -> String {
        "\(saltPrompt(run: run, nonce: nonce, basePrompt)) [request=\(request)]"
    }
}

public enum BenchGuardError: Error, Sendable { case debugBuild }

/// Fails fast if this is a Debug/unoptimized build — perf numbers from a Debug build are
/// misleading (the ReleaseFast lesson from the spike).
public func assertReleaseBuild() throws {
    #if DEBUG
    throw BenchGuardError.debugBuild
    #endif
}

/// One CSV row of a bench run. Rate is expected to come from stream-timed metrics
/// (e.g. `DecodeMetrics` in the executable target), never usage/summary fields.
/// `hardware` is spec §6.3's mandated durable-evidence dimension ("enough dimensions... to diff
/// programmatically against a prior run"): label/mode/concurrency/model were already present;
/// hardware was the missing one (Task 5).
public struct BenchRow: Sendable {
    public let label: String
    public let workload: Workload
    public let mode: Mode
    public let model: String
    public let decodeTokS: Double
    public let ttftMs: Double
    public let quant: String
    public let concurrency: Int
    public let hardware: String
    public let prefillTokS: Double?
    public let prefillMs: Double?
    public let promptTokensMin: Int?
    public let promptTokensMax: Int?
    public let kvQuantTier: String?
    public let matrixID: String?
    public let cellID: String?
    public let workloadNonce: String?
    public let kvtunerScheduleSHA256: String?
    public let kvtunerBundleSHA256: String?

    public init(
        label: String, workload: Workload, mode: Mode, model: String,
        decodeTokS: Double, ttftMs: Double, quant: String,
        concurrency: Int, hardware: String,
        prefillTokS: Double? = nil, prefillMs: Double? = nil,
        promptTokensMin: Int? = nil, promptTokensMax: Int? = nil,
        kvQuantTier: String? = nil, matrixID: String? = nil,
        cellID: String? = nil, workloadNonce: String? = nil,
        kvtunerScheduleSHA256: String? = nil,
        kvtunerBundleSHA256: String? = nil
    ) {
        self.label = label; self.workload = workload; self.mode = mode; self.model = model
        self.decodeTokS = decodeTokS; self.ttftMs = ttftMs; self.quant = quant; self.concurrency = concurrency
        self.hardware = hardware
        self.prefillTokS = prefillTokS; self.prefillMs = prefillMs
        self.promptTokensMin = promptTokensMin; self.promptTokensMax = promptTokensMax
        self.kvQuantTier = kvQuantTier; self.matrixID = matrixID
        self.cellID = cellID; self.workloadNonce = workloadNonce
        self.kvtunerScheduleSHA256 = kvtunerScheduleSHA256
        self.kvtunerBundleSHA256 = kvtunerBundleSHA256
    }

    public static let csvHeader =
        "label,workload,mode,model,decode_tok_s,ttft_ms,quant,concurrency,hardware"
        + ",prefill_tok_s,prefill_ms,prompt_tokens_min,prompt_tokens_max"
        + ",kv_quant_tier,matrix_id,cell_id,workload_nonce"
        + ",kvtuner_schedule_sha256,kvtuner_bundle_sha256"

    public var csvLine: String {
        let prefillTokSText = prefillTokS.map { String($0) } ?? ""
        let prefillMsText = prefillMs.map { String($0) } ?? ""
        let promptTokensMinText = promptTokensMin.map { String($0) } ?? ""
        let promptTokensMaxText = promptTokensMax.map { String($0) } ?? ""
        let directPrefill = [
            prefillTokSText,
            prefillMsText,
            promptTokensMinText,
            promptTokensMaxText,
        ].joined(separator: ",")
        let runtimeIdentity = [
            kvQuantTier ?? "",
            matrixID ?? "",
            cellID ?? "",
            workloadNonce ?? "",
            kvtunerScheduleSHA256 ?? "",
            kvtunerBundleSHA256 ?? "",
        ].joined(separator: ",")
        return "\(label),\(workload.rawValue),\(mode.rawValue),\(model),\(decodeTokS),\(ttftMs),\(quant),\(concurrency),\(hardware),\(directPrefill),\(runtimeIdentity)"
    }
}

/// Runs a `Cell` for N iterations via a caller-supplied rate function — the MLX-touching timing
/// itself lives in the executable's `SwiftEngineDriver`, not here — applying warmup-drop and
/// prompt-salting per the bench methodology, and aggregating into a rate.
public struct BenchRunner: Sendable {
    public init() {}

    public func run(
        cell: Cell,
        iterations: Int,
        nonce: String,
        basePrompt: String,
        rate: (_ run: Int, _ saltedPrompt: String) async throws -> Double?
    ) async rethrows -> RateAggregate {
        var rates: [Double?] = []
        for i in 0..<iterations {
            let salted = saltPrompt(run: i, nonce: nonce, basePrompt)
            rates.append(try await rate(i, salted))
        }
        return aggregateRates(rates)
    }
}
