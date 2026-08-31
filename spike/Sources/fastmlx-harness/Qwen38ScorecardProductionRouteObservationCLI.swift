import Darwin
import Foundation
import HarnessCore
import MLX
import ServingCore
import SpikeCore
@_spi(ProductionRouteEvidence)
import SpikeServingAdapters

private let qwen38ProductionRouteObservationKind =
    "fast-mlx.production-route-observation.v1"
private let qwen38ProductionRouteCleanupAttemptKind =
    "fast-mlx.production-route-cleanup-attempt.v1"
private let qwen38ProductionRouteCleanupAttemptLimit = 31
/// Reviewed exception (2026-08-31): mlx-swift 0.31.6 compiled-trace teardown
/// can orphan multi-output sibling reference cycles, leaking kilobyte-scale
/// constant-descriptor buffers per load+serve cycle even after model weights
/// are passed as compile state, so a strict zero-residual gate cannot pass
/// in-process. The post-run comparisons therefore admit this documented
/// absolute residual per Metal metric. The bound keeps >=20x margin below the
/// smallest weight-scale leak signal observed (85 MB fused-only at 9B) and
/// still catches KV-row- and logits-scale leaks. Conditions: every cleanup
/// attempt logs raw before/observed facts; the exception expires on any
/// mlx-swift upgrade (a version-pin regression test fails until the strict
/// gate is re-verified). Ratcheted 2026-08-31 from the initial 4 MiB to
/// max(64 KiB, 4x measured floor) after the first passing 27B observation
/// measured an 11,274-byte post-run active residual (C2+C4, GDN fusion on).
/// See docs/task-inbox/2026-08-31-teardown-compiled-state-followups.md.
let qwen38ProductionRouteResidualToleranceBytes: UInt64 = 65_536

struct Qwen38ScorecardProductionRouteObservationArguments:
    Equatable, Sendable
{
    let targetPath: String
    let drafterPath: String
    let outputPath: String
    let hostUse: String
    let hostUseSource: String
    let expectedChip: String
    let memoryBudget: Qwen38MTPScorecardLiveMemoryBudget
    let contextTokens: UInt64
}

enum Qwen38ScorecardProductionRouteObservationCLIError:
    Error, Equatable, CustomStringConvertible, Sendable
{
    case missingFlag(String)
    case duplicateFlag(String)
    case unknownFlag
    case missingValue(String)
    case unexpectedPositional
    case invalidInteger(String)
    case invalidHostAssertion
    case invalidTarget
    case invalidDrafter
    case invalidOutput
    case outputExists
    case unsafeOutput
    case invalidGDNEnvironment
    case invalidHostIdentity
    case invalidMemoryBudget
    case invalidProcessMemoryReadback
    case initialThermalUnsafe
    case postRunHostReadbackFailed
    case postRunWiredLimitDrift
    case postRunCurrentAllocatedMemoryResidual
    case postRunActiveMemoryResidual
    case postRunCachedMemoryResidual
    case postRunProcessMemoryReadbackFailed
    case postRunThermalUnsafe
    case sourcePreflightFailed
    case modelLoadFailed
    case observerFailed
    case invalidStartupReport
    case invalidArtifact
    case outputWriteFailed

    var description: String {
        switch self {
        case .missingFlag(let flag): return "missing required \(flag)"
        case .duplicateFlag(let flag): return "duplicate \(flag)"
        case .unknownFlag: return "unknown flag"
        case .missingValue(let flag): return "\(flag) requires a value"
        case .unexpectedPositional: return "unexpected positional argument"
        case .invalidInteger(let flag): return "\(flag) must be a positive integer"
        case .invalidHostAssertion: return "invalid dedicated host assertion"
        case .invalidTarget: return "invalid target directory"
        case .invalidDrafter: return "invalid drafter directory"
        case .invalidOutput: return "invalid output artifact"
        case .outputExists: return "output destination already exists"
        case .unsafeOutput: return "output destination is unsafe"
        case .invalidGDNEnvironment: return "required GDN environment is absent"
        case .invalidHostIdentity: return "host identity does not match required dedicated host"
        case .invalidMemoryBudget: return "invalid explicit memory budget"
        case .invalidProcessMemoryReadback:
            return "initial process memory readback failed"
        case .initialThermalUnsafe: return "initial thermal state is unsafe"
        case .postRunHostReadbackFailed: return "post-run host readback failed"
        case .postRunWiredLimitDrift: return "post-run wired limit drifted"
        case .postRunCurrentAllocatedMemoryResidual:
            return "post-run current Metal allocation did not settle"
        case .postRunActiveMemoryResidual: return "post-run active Metal memory did not settle"
        case .postRunCachedMemoryResidual: return "post-run cached Metal memory did not settle"
        case .postRunProcessMemoryReadbackFailed:
            return "post-run process memory readback failed"
        case .postRunThermalUnsafe: return "post-run thermal state is unsafe"
        case .sourcePreflightFailed: return "source preflight failed"
        case .modelLoadFailed: return "model load failed"
        case .observerFailed: return "route observation failed"
        case .invalidStartupReport: return "loaded model startup report is invalid"
        case .invalidArtifact: return "invalid route observation artifact"
        case .outputWriteFailed: return "failed to publish route observation artifact"
        }
    }
}

func qwen38ScorecardProductionRouteObservationExternalDiagnostic(
    _ error: Error
) -> String {
    if let error = error as? Qwen38ScorecardProductionRouteObservationCLIError {
        return error.description
    }
    if error is CancellationError {
        return "route observation cancelled"
    }
    return "route observation failed"
}

func parseQwen38ScorecardProductionRouteObservationArguments(
    _ arguments: [String]
) throws -> Qwen38ScorecardProductionRouteObservationArguments {
    let allowed = Set([
        "--target",
        "--drafter",
        "--output",
        "--host-use",
        "--host-use-source",
        "--expected-chip",
        "--memory-limit-bytes",
        "--cache-limit-bytes",
        "--reserved-kv-bytes",
        "--reserved-io-bytes",
        "--reserved-prefetch-bytes",
        "--os-service-reserve-bytes",
        "--context-tokens",
    ])
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .unexpectedPositional
        }
        guard allowed.contains(flag) else {
            throw Qwen38ScorecardProductionRouteObservationCLIError.unknownFlag
        }
        guard values[flag] == nil else {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .duplicateFlag(flag)
        }
        guard index + 1 < arguments.count,
            !arguments[index + 1].hasPrefix("--")
        else {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }

    func require(_ flag: String) throws -> String {
        guard let value = values[flag], !value.isEmpty else {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .missingFlag(flag)
        }
        return value
    }

    func requireUInt64(_ flag: String) throws -> UInt64 {
        let raw = try require(flag)
        guard let value = UInt64(raw), value > 0, value <= UInt64(Int.max)
        else {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .invalidInteger(flag)
        }
        return value
    }

    let hostUse = try require("--host-use")
    let hostUseSource = try require("--host-use-source")
    guard hostUse == "dedicated-serving",
        hostUseSource == "operator-assertion"
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .invalidHostAssertion
    }

    return try Qwen38ScorecardProductionRouteObservationArguments(
        targetPath: require("--target"),
        drafterPath: require("--drafter"),
        outputPath: require("--output"),
        hostUse: hostUse,
        hostUseSource: hostUseSource,
        expectedChip: require("--expected-chip"),
        memoryBudget: Qwen38MTPScorecardLiveMemoryBudget(
            memoryLimitBytes: requireUInt64("--memory-limit-bytes"),
            cacheLimitBytes: requireUInt64("--cache-limit-bytes"),
            reservedKVBytes: requireUInt64("--reserved-kv-bytes"),
            reservedIOBytes: requireUInt64("--reserved-io-bytes"),
            reservedPrefetchBytes: requireUInt64("--reserved-prefetch-bytes"),
            osServiceReserveBytes: requireUInt64("--os-service-reserve-bytes")),
        contextTokens: requireUInt64("--context-tokens"))
}

struct Qwen38ScorecardProductionRouteObservationFreshOutput {
    static func validate(_ path: String) throws {
        let outputURL = URL(fileURLWithPath: path).standardizedFileURL
        let manager = FileManager.default
        guard !path.isEmpty, path.hasPrefix("/") else {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .unsafeOutput
        }
        guard !outputPathIsSymbolicLink(outputURL.path) else {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .unsafeOutput
        }
        if manager.fileExists(atPath: outputURL.path) {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .outputExists
        }
        var isDirectory: ObjCBool = false
        guard manager.fileExists(
            atPath: outputURL.deletingLastPathComponent().path,
            isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .unsafeOutput
        }
    }
}

struct Qwen38ScorecardProductionRouteObservationDependencies: Sendable {
    var observedGDNEnvironment: @Sendable () async -> String?
    var hostSnapshot:
        @Sendable () async throws -> Qwen38MTPScorecardLiveHostMemorySnapshot
    var sourcePreflight:
        @Sendable (URL, URL) async throws
            -> Qwen38ScorecardProductionRouteSourceLockObservation
    var loadModel:
        @Sendable (ContinuousServingModelLoadConfiguration) async throws
            -> LoadedContinuousServingModel
    var observeLoaded:
        @Sendable (
            LoadedContinuousServingModel,
            ContinuousServingOutputTokenTraceConfiguration
        ) async throws -> Qwen38ScorecardProductionRouteObservation
    var shutdownLoaded:
        @Sendable (LoadedContinuousServingModel) async -> Void
    var clearMLXCache: @Sendable () async -> Void
    var cleanupAttemptLimit: Int
    var cleanupSettle: @Sendable () async throws -> Void
    var cleanupMonotonicNanoseconds: @Sendable () async -> UInt64
    var cleanupAttemptTelemetry:
        @Sendable (
            Qwen38ScorecardProductionRouteCleanupAttemptObservation
        ) async -> Void
    var processIsolation:
        @Sendable (Qwen38MTPPerformanceScorecardGDNObservedEnv) async throws
            -> Qwen38MTPLiveExactnessProcessIsolationEvidence
    var metalMemorySnapshot:
        @Sendable () async -> Qwen38ScorecardProductionRouteMetalMemorySnapshot
    var thermalState: @Sendable () async -> String
    var peakRSSBytes: @Sendable () async -> UInt64
    var physicalFootprintBytes: @Sendable () async -> UInt64
    var date: @Sendable () async -> String
    var writeFresh: @Sendable (Data, String) async throws -> Void

    static let production = Self(
        observedGDNEnvironment: {
            ProcessInfo.processInfo.environment["MLX_QWEN_FOUR_GDN"]
        },
        hostSnapshot: { try Qwen38MTPScorecardLiveHostMemorySnapshot.live() },
        sourcePreflight: { target, drafter in
            do {
                let binding = try Qwen35ExactMTPRuntimeFactory
                    .preloadSourceLockedDepth1Pair(
                        selection: .qwen38_27BMXFP8Depth1,
                        targetDirectory: target,
                        drafterDirectory: drafter)
                return Qwen38ScorecardProductionRouteSourceLockObservation(
                    binding: binding,
                    knownArtifact: Qwen38MTPPerformanceScorecardGate
                        .modelArtifactLabel,
                    selection: Qwen35ExactMTPRuntimeSelection
                        .qwen38_27BMXFP8Depth1.rawValue,
                    lock: Qwen38MTPPerformanceScorecardGate
                        .requiredArtifact)
            } catch let error
                as Qwen38ScorecardProductionRouteObservationCLIError
            {
                throw error
            } catch {
                throw Qwen38ScorecardProductionRouteObservationCLIError
                    .sourcePreflightFailed
            }
        },
        loadModel: { configuration in
            do {
                return try await loadContinuousServingModel(
                    configuration: configuration)
            } catch let error
                as Qwen38ScorecardProductionRouteObservationCLIError
            {
                throw error
            } catch {
                throw Qwen38ScorecardProductionRouteObservationCLIError
                    .modelLoadFailed
            }
        },
        observeLoaded: { loaded, tokenTrace in
            do {
                return try await Qwen38ScorecardProductionRouteRunner
                    .observeLoaded(loaded, tokenTrace: tokenTrace)
            } catch let error
                as Qwen38ScorecardProductionRouteObservationCLIError
            {
                throw error
            } catch {
                throw Qwen38ScorecardProductionRouteObservationCLIError
                    .observerFailed
            }
        },
        shutdownLoaded: { loaded in await loaded.backend.shutdown() },
        clearMLXCache: { Memory.clearCache() },
        cleanupAttemptLimit: qwen38ProductionRouteCleanupAttemptLimit,
        cleanupSettle: { try await Task.sleep(for: .seconds(1)) },
        cleanupMonotonicNanoseconds: { DispatchTime.now().uptimeNanoseconds },
        cleanupAttemptTelemetry: { observation in
            qwen38ProductionRouteEmitCleanupAttempt(observation)
        },
        processIsolation: { observedEnv in
            try Qwen38MTPScorecardProcessFacts.processIsolation(
                mode: .gdnOn,
                observedEnv: observedEnv)
        },
        metalMemorySnapshot: {
            let snapshot = Memory.snapshot()
            return Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: UInt64(max(snapshot.activeMemory, 0)),
                cachedMetalBytes: UInt64(max(snapshot.cacheMemory, 0)),
                peakMetalBytes: UInt64(max(snapshot.peakMemory, 0)))
        },
        thermalState: { qwen38ScorecardProductionRouteThermalState() },
        peakRSSBytes: { qwen38ProductionRoutePeakRSSBytes() },
        physicalFootprintBytes: { physFootprintBytes() },
        date: { ISO8601DateFormatter().string(from: Date()) },
        writeFresh: { data, path in
            try writeFreshQwen38ScorecardProductionRouteObservation(data, path)
        })
}

extension Qwen38ScorecardProductionRouteObservationDependencies {
    static func test(
        environment: @escaping @Sendable () async -> String? = { "1" },
        hostSnapshot: @escaping @Sendable () async throws
            -> Qwen38MTPScorecardLiveHostMemorySnapshot = {
                Qwen38MTPScorecardLiveHostMemorySnapshot(
                    physicalRAMBytes: Qwen38MTPPerformanceScorecardGate
                        .requiredRAMBytes,
                    chipName: Qwen38MTPScorecardLiveHostMemorySnapshot
                        .requiredScorecardChip,
                    wiredLimitMB: 262_144,
                    metalRecommendedMaxWorkingSetSizeBytes:
                        220_000_000_000,
                    metalCurrentAllocatedSizeBytes: 1)
            },
        sourcePreflight: @escaping @Sendable (URL, URL) async throws
            -> Qwen38ScorecardProductionRouteSourceLockObservation = { _, _ in
                Qwen38ScorecardProductionRouteSourceLockObservation(
                    binding: Qwen38MTPPerformanceScorecardGate
                        .requiredArtifact)
            },
        loadModel: @escaping @Sendable (
            ContinuousServingModelLoadConfiguration
        ) async throws -> LoadedContinuousServingModel = { _ in
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .modelLoadFailed
        },
        observeLoaded: @escaping @Sendable (
            LoadedContinuousServingModel,
            ContinuousServingOutputTokenTraceConfiguration
        ) async throws -> Qwen38ScorecardProductionRouteObservation = {
            loaded,
            trace in
            try await Qwen38ScorecardProductionRouteRunner.observeLoaded(
                loaded,
                tokenTrace: trace)
        },
        shutdownLoaded: @escaping @Sendable (
            LoadedContinuousServingModel
        ) async -> Void = { loaded in await loaded.backend.shutdown() },
        clearMLXCache: @escaping @Sendable () async -> Void = {},
        cleanupAttemptLimit: Int = 5,
        cleanupSettle: @escaping @Sendable () async throws -> Void = {},
        cleanupMonotonicNanoseconds: @escaping @Sendable () async -> UInt64 = {
            DispatchTime.now().uptimeNanoseconds
        },
        cleanupAttemptTelemetry: @escaping @Sendable (
            Qwen38ScorecardProductionRouteCleanupAttemptObservation
        ) async -> Void = { _ in },
        processIsolation: @escaping @Sendable (
            Qwen38MTPPerformanceScorecardGDNObservedEnv
        ) async throws -> Qwen38MTPLiveExactnessProcessIsolationEvidence = {
            observed in
            Qwen38MTPLiveExactnessProcessIsolationEvidence(
                processID: 1,
                parentProcessID: 0,
                processStartUptimeNanoseconds: 1,
                bootTimeUnixSeconds: 1,
                executableIdentitySource: .procPIDPath,
                executableSHA256: String(repeating: "f", count: 64),
                harnessGitSHA: String(repeating: "e", count: 40),
                sourceID: Qwen38MTPLiveExactnessGate.requiredSourceIdentity
                    .sourceID,
                gdnMode: .gdnOn,
                observedEnv: observed)
        },
        metalMemorySnapshot: @escaping @Sendable () async
            -> Qwen38ScorecardProductionRouteMetalMemorySnapshot = {
                Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                    activeMetalBytes: 1,
                    cachedMetalBytes: 2,
                    peakMetalBytes: 3)
            },
        thermalState: @escaping @Sendable () async -> String = { "nominal" },
        peakRSSBytes: @escaping @Sendable () async -> UInt64 = { 4 },
        physicalFootprintBytes: @escaping @Sendable () async -> UInt64 = { 5 },
        date: @escaping @Sendable () async -> String = {
            "2026-08-31T00:00:00Z"
        },
        writeFresh: @escaping @Sendable (Data, String) async throws -> Void = {
            data,
            path in
            try writeFreshQwen38ScorecardProductionRouteObservation(data, path)
        }
    ) -> Self {
        Self(
            observedGDNEnvironment: environment,
            hostSnapshot: hostSnapshot,
            sourcePreflight: sourcePreflight,
            loadModel: loadModel,
            observeLoaded: observeLoaded,
            shutdownLoaded: shutdownLoaded,
            clearMLXCache: clearMLXCache,
            cleanupAttemptLimit: cleanupAttemptLimit,
            cleanupSettle: cleanupSettle,
            cleanupMonotonicNanoseconds: cleanupMonotonicNanoseconds,
            cleanupAttemptTelemetry: cleanupAttemptTelemetry,
            processIsolation: processIsolation,
            metalMemorySnapshot: metalMemorySnapshot,
            thermalState: thermalState,
            peakRSSBytes: peakRSSBytes,
            physicalFootprintBytes: physicalFootprintBytes,
            date: date,
            writeFresh: writeFresh)
    }
}

func produceQwen38ScorecardProductionRouteObservation(
    arguments: [String],
    dependencies: Qwen38ScorecardProductionRouteObservationDependencies =
        .production
) async throws -> String {
    let parsed = try parseQwen38ScorecardProductionRouteObservationArguments(
        arguments)
    let target = try validateQwen38ProductionRouteDirectory(
        parsed.targetPath,
        error: .invalidTarget)
    let drafter = try validateQwen38ProductionRouteDirectory(
        parsed.drafterPath,
        error: .invalidDrafter)
    guard target.path != drafter.path else {
        throw Qwen38ScorecardProductionRouteObservationCLIError.invalidDrafter
    }
    try Qwen38ScorecardProductionRouteObservationFreshOutput.validate(
        parsed.outputPath)

    let observedEnv = try await validateQwen38ProductionRouteGDNEnvironment(
        dependencies)
    let hostSnapshot = try await validateQwen38ProductionRouteHost(
        dependencies,
        expectedChip: parsed.expectedChip,
        memoryBudget: parsed.memoryBudget)
    let beforeFacts = try await qwen38ProductionRouteHostFacts(
        hostSnapshot: hostSnapshot,
        dependencies: dependencies)
    try validateQwen38ProductionRouteInitialSafety(beforeFacts)

    let sourceLock = try await dependencies.sourcePreflight(target, drafter)
    let configuration = try qwen38ProductionRouteModelLoadConfiguration(
        arguments: parsed,
        target: target)

    var clearedAfterLoadedScope = false
    do {
        let loadedObservation = try await qwen38ProductionRouteObserveLoaded(
            configuration: configuration,
            sourceLock: sourceLock,
            target: target,
            drafter: drafter,
            dependencies: dependencies)
        clearedAfterLoadedScope = true
        let afterFacts = try await awaitQwen38ProductionRouteCleanup(
            before: beforeFacts,
            expectedChip: parsed.expectedChip,
            memoryBudget: parsed.memoryBudget,
            dependencies: dependencies)
        let processIsolation = try await dependencies.processIsolation(
            observedEnv)
        let artifact = try Qwen38ScorecardProductionRouteObservationArtifact
            .build(
                date: await dependencies.date(),
                sourceLock: sourceLock,
                observedEnv: observedEnv,
                processIsolation: processIsolation,
                hostUse: parsed.hostUse,
                hostUseSource: parsed.hostUseSource,
                expectedChip: parsed.expectedChip,
                hostSnapshot: hostSnapshot,
                memoryBudget: parsed.memoryBudget,
                contextTokens: parsed.contextTokens,
                before: beforeFacts,
                after: afterFacts,
                startupReport: loadedObservation.startupReport,
                observation: loadedObservation.observation)
        let data = try qwen38ProductionRouteBoundedCanonicalJSON(artifact)
        try await dependencies.writeFresh(data, parsed.outputPath)
        return "qwen38-production-route-observation: WROTE unsigned"
    } catch {
        if !clearedAfterLoadedScope {
            await dependencies.clearMLXCache()
        }
        throw error
    }
}

private struct Qwen38ProductionRouteLoadedObservation: Sendable {
    let startupReport: ContinuousServingModelStartupReport
    let observation: Qwen38ScorecardProductionRouteObservation
}

private func qwen38ProductionRouteObserveLoaded(
    configuration: ContinuousServingModelLoadConfiguration,
    sourceLock: Qwen38ScorecardProductionRouteSourceLockObservation,
    target: URL,
    drafter: URL,
    dependencies: Qwen38ScorecardProductionRouteObservationDependencies
) async throws -> Qwen38ProductionRouteLoadedObservation {
    var loaded: LoadedContinuousServingModel?
    do {
        loaded = try await dependencies.loadModel(configuration)
        guard let current = loaded else {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .modelLoadFailed
        }
        let startupReport = current.startupReport
        try validateQwen38ProductionRouteStartupReport(
            startupReport,
            configuration: configuration,
            sourceLock: sourceLock)
        try await validateQwen38ProductionRouteStableSourceLock(
            sourceLock,
            target: target,
            drafter: drafter,
            dependencies: dependencies)
        let observation = try await dependencies.observeLoaded(
            current,
            ContinuousServingOutputTokenTraceConfiguration.outputTokenIDs(
                maxCompletedRequests: 8,
                maxTokensPerRequest: 8))
        try await validateQwen38ProductionRouteStableSourceLock(
            sourceLock,
            target: target,
            drafter: drafter,
            dependencies: dependencies)
        await dependencies.shutdownLoaded(current)
        loaded = nil
        return Qwen38ProductionRouteLoadedObservation(
            startupReport: startupReport,
            observation: observation)
    } catch {
        if let current = loaded {
            await dependencies.shutdownLoaded(current)
            loaded = nil
        }
        throw error
    }
}

private func validateQwen38ProductionRouteStableSourceLock(
    _ expected: Qwen38ScorecardProductionRouteSourceLockObservation,
    target: URL,
    drafter: URL,
    dependencies: Qwen38ScorecardProductionRouteObservationDependencies
) async throws {
    let current: Qwen38ScorecardProductionRouteSourceLockObservation
    do {
        current = try await dependencies.sourcePreflight(target, drafter)
    } catch {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .sourcePreflightFailed
    }
    guard current == expected else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .sourcePreflightFailed
    }
}

private func validateQwen38ProductionRouteStartupReport(
    _ report: ContinuousServingModelStartupReport,
    configuration: ContinuousServingModelLoadConfiguration,
    sourceLock: Qwen38ScorecardProductionRouteSourceLockObservation
) throws {
    guard report.launchedModel == configuration.launchedModel,
        report.launchedModel == Qwen38MTPPerformanceScorecardGate
            .modelArtifactLabel,
        report.route == .continuousBatchNoSpec,
        report.memoryLimitBytes == configuration.memoryLimitBytes,
        report.cacheLimitBytes == configuration.cacheLimitBytes,
        report.maxReservedKVBytes == configuration.maxReservedKVBytes,
        configuration.maxContextTokens == report.maxContextTokens,
        configuration.maxReservedContextTokens
            == report.maxReservedContextTokens,
        report.modelFamily == .qwen35,
        report.modelConfigurationSHA256 == sourceLock.targetConfigSHA256,
        report.maxActiveSlots == 4,
        report.maxPrefillSlots == 4,
        report.prefillChunkSize
            == configuration.coordinatorConfiguration.prefillChunkSize,
        report.maxQueuedRequests == 4,
        report.publicationCapacity == 8,
        report.soloPLDPolicy == nil,
        report.modelProofVerified
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .invalidStartupReport
    }
}

private func validateQwen38ProductionRouteDirectory(
    _ path: String,
    error: Qwen38ScorecardProductionRouteObservationCLIError
) throws -> URL {
    let url = URL(fileURLWithPath: path).standardizedFileURL
    var isDirectory: ObjCBool = false
    guard !path.isEmpty,
        path.hasPrefix("/"),
        FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
        isDirectory.boolValue
    else {
        throw error
    }
    return url
}

private func validateQwen38ProductionRouteGDNEnvironment(
    _ dependencies: Qwen38ScorecardProductionRouteObservationDependencies
) async throws -> Qwen38MTPPerformanceScorecardGDNObservedEnv {
    guard await dependencies.observedGDNEnvironment() == "1" else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .invalidGDNEnvironment
    }
    return .enabled
}

private func validateQwen38ProductionRouteHost(
    _ dependencies: Qwen38ScorecardProductionRouteObservationDependencies,
    expectedChip: String,
    memoryBudget: Qwen38MTPScorecardLiveMemoryBudget
) async throws -> Qwen38MTPScorecardLiveHostMemorySnapshot {
    let snapshot = try await dependencies.hostSnapshot()
    guard expectedChip
        == Qwen38MTPScorecardLiveHostMemorySnapshot.requiredScorecardChip,
        snapshot.chipName == expectedChip,
        snapshot.physicalRAMBytes
            == Qwen38MTPPerformanceScorecardGate.requiredRAMBytes
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .invalidHostIdentity
    }
    do {
        try memoryBudget.validateAgainstHost(snapshot)
    } catch let error as Qwen38ScorecardProductionRouteObservationCLIError {
        throw error
    } catch {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .invalidMemoryBudget
    }
    return snapshot
}

private func validateQwen38ProductionRoutePostRunSafety(
    before: Qwen38ScorecardProductionRouteHostFacts,
    after: Qwen38ScorecardProductionRouteHostFacts
) throws {
    guard before.peakRSSBytes > 0,
        before.physicalFootprintBytes > 0,
        after.peakRSSBytes > 0,
        after.physicalFootprintBytes > 0
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .postRunProcessMemoryReadbackFailed
    }
    guard before.wiredLimitMB == after.wiredLimitMB else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .postRunWiredLimitDrift
    }
    let residualTolerance = qwen38ProductionRouteResidualToleranceBytes
    guard after.metal.activeMetalBytes
        <= before.metal.activeMetalBytes + residualTolerance
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .postRunActiveMemoryResidual
    }
    guard after.metalCurrentAllocatedSizeBytes
        <= before.metalCurrentAllocatedSizeBytes + residualTolerance
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .postRunCurrentAllocatedMemoryResidual
    }
    guard after.metal.cachedMetalBytes
        <= before.metal.cachedMetalBytes + residualTolerance
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .postRunCachedMemoryResidual
    }
    guard !["serious", "critical", "unknown"].contains(after.thermalState)
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .postRunThermalUnsafe
    }
}

private func validateQwen38ProductionRouteInitialSafety(
    _ facts: Qwen38ScorecardProductionRouteHostFacts
) throws {
    guard facts.peakRSSBytes > 0, facts.physicalFootprintBytes > 0 else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .invalidProcessMemoryReadback
    }
    guard !["serious", "critical", "unknown"].contains(facts.thermalState)
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .initialThermalUnsafe
    }
}

private func awaitQwen38ProductionRouteCleanup(
    before: Qwen38ScorecardProductionRouteHostFacts,
    expectedChip: String,
    memoryBudget: Qwen38MTPScorecardLiveMemoryBudget,
    dependencies: Qwen38ScorecardProductionRouteObservationDependencies
) async throws -> Qwen38ScorecardProductionRouteHostFacts {
    let attemptLimit = dependencies.cleanupAttemptLimit
    guard attemptLimit > 0, attemptLimit <= 120 else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .postRunHostReadbackFailed
    }
    let startedAt = await dependencies.cleanupMonotonicNanoseconds()
    for attempt in 0 ..< attemptLimit {
        await dependencies.clearMLXCache()
        let snapshot = try await validateQwen38ProductionRoutePostRunHost(
            dependencies,
            expectedChip: expectedChip,
            memoryBudget: memoryBudget)
        let facts = try await qwen38ProductionRouteHostFacts(
            hostSnapshot: snapshot,
            dependencies: dependencies)
        let observedAt = await dependencies.cleanupMonotonicNanoseconds()
        let elapsed = observedAt >= startedAt ? observedAt - startedAt : 0
        let residualAdmissionSafe = (try? memoryBudget.validateAgainstHost(
            snapshot)) != nil
        await dependencies.cleanupAttemptTelemetry(
            Qwen38ScorecardProductionRouteCleanupAttemptObservation(
                kind: qwen38ProductionRouteCleanupAttemptKind,
                attempt: attempt + 1,
                attemptLimit: attemptLimit,
                elapsedNanoseconds: elapsed,
                before: before,
                observed: facts,
                currentAllocatedAtOrBelowBaseline:
                    facts.metalCurrentAllocatedSizeBytes
                        <= before.metalCurrentAllocatedSizeBytes,
                activeAtOrBelowBaseline:
                    facts.metal.activeMetalBytes
                        <= before.metal.activeMetalBytes,
                cachedAtOrBelowBaseline:
                    facts.metal.cachedMetalBytes
                        <= before.metal.cachedMetalBytes,
                processMemoryReadbackValid:
                    before.peakRSSBytes > 0
                        && before.physicalFootprintBytes > 0
                        && facts.peakRSSBytes > 0
                        && facts.physicalFootprintBytes > 0,
                residualAdmissionSafe: residualAdmissionSafe,
                residualToleranceBytes:
                    qwen38ProductionRouteResidualToleranceBytes))
        do {
            try validateQwen38ProductionRoutePostRunSafety(
                before: before,
                after: facts)
            return facts
        } catch let error as Qwen38ScorecardProductionRouteObservationCLIError
            where error == .postRunCurrentAllocatedMemoryResidual
                || error == .postRunActiveMemoryResidual
                || error == .postRunCachedMemoryResidual
        {
            guard attempt + 1 < attemptLimit else {
                throw error
            }
            try await dependencies.cleanupSettle()
        }
    }
    throw Qwen38ScorecardProductionRouteObservationCLIError
        .postRunActiveMemoryResidual
}

private func validateQwen38ProductionRoutePostRunHost(
    _ dependencies: Qwen38ScorecardProductionRouteObservationDependencies,
    expectedChip: String,
    memoryBudget: Qwen38MTPScorecardLiveMemoryBudget
) async throws -> Qwen38MTPScorecardLiveHostMemorySnapshot {
    let snapshot: Qwen38MTPScorecardLiveHostMemorySnapshot
    do {
        snapshot = try await dependencies.hostSnapshot()
    } catch is CancellationError {
        throw CancellationError()
    } catch {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .postRunHostReadbackFailed
    }
    guard expectedChip
        == Qwen38MTPScorecardLiveHostMemorySnapshot.requiredScorecardChip,
        snapshot.chipName == expectedChip,
        snapshot.physicalRAMBytes
            == Qwen38MTPPerformanceScorecardGate.requiredRAMBytes
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .postRunHostReadbackFailed
    }
    do {
        let ceiling = try snapshot.effectiveMetalCeilingBytes
        let required = try memoryBudget.workerResidentPlanBytes
            .addingReportingOverflow(memoryBudget.osServiceReserveBytes)
        guard !required.overflow, required.partialValue <= ceiling else {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .postRunHostReadbackFailed
        }
    } catch let error as Qwen38ScorecardProductionRouteObservationCLIError {
        throw error
    } catch {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .postRunHostReadbackFailed
    }
    return snapshot
}

private func qwen38ProductionRouteModelLoadConfiguration(
    arguments: Qwen38ScorecardProductionRouteObservationArguments,
    target: URL
) throws -> ContinuousServingModelLoadConfiguration {
    let contextTokens = Int(arguments.contextTokens)
    return ContinuousServingModelLoadConfiguration(
        launchedModel: Qwen38MTPPerformanceScorecardGate.modelArtifactLabel,
        modelDirectory: target,
        memoryLimitBytes: Int(arguments.memoryBudget.memoryLimitBytes),
        cacheLimitBytes: Int(arguments.memoryBudget.cacheLimitBytes),
        maxReservedKVBytes: Int(arguments.memoryBudget.reservedKVBytes),
        maxContextTokens: contextTokens,
        maxReservedContextTokens: contextTokens,
        initialDecodeReserve: 8,
        coordinatorConfiguration: try ContinuousBatchConfiguration(
            maxActiveSlots: 4,
            maxPrefillSlots: 4,
            prefillChunkSize: 512,
            maxQueuedRequests: 4),
        publicationCapacity: 8,
        traceLimit: 4_096,
        backendConfiguration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096),
            admission: .dynamic(
                configuration: ServingAdmissionConfiguration(
                    soloPLDQualified: false,
                    maximumBatchRequests: 4,
                    maximumQueuedRequests: 4),
                coalescing: .automatic(.milliseconds(5))),
            disableThinkingWhenToolsActive: false),
        soloPLDPolicy: nil,
        kvQuantTier: .fp16,
        allowHybridQwen35: true)
}

private func qwen38ProductionRouteHostFacts(
    hostSnapshot: Qwen38MTPScorecardLiveHostMemorySnapshot,
    dependencies: Qwen38ScorecardProductionRouteObservationDependencies
) async throws -> Qwen38ScorecardProductionRouteHostFacts {
    Qwen38ScorecardProductionRouteHostFacts(
        metal: await dependencies.metalMemorySnapshot(),
        wiredLimitMB: Int(min(hostSnapshot.wiredLimitMB, UInt64(Int.max))),
        metalRecommendedMaxWorkingSetSizeBytes:
            hostSnapshot.metalRecommendedMaxWorkingSetSizeBytes ?? 0,
        metalCurrentAllocatedSizeBytes:
            hostSnapshot.metalCurrentAllocatedSizeBytes,
        thermalState: await dependencies.thermalState(),
        peakRSSBytes: await dependencies.peakRSSBytes(),
        physicalFootprintBytes: await dependencies.physicalFootprintBytes())
}

private func qwen38ProductionRouteEmitCleanupAttempt(
    _ observation: Qwen38ScorecardProductionRouteCleanupAttemptObservation
) {
    guard var data = try? qwen38MTPScorecardCanonicalJSON(observation),
        !data.isEmpty, data.count <= 65_536
    else { return }
    data.append(0x0A)
    FileHandle.standardError.write(data)
}

private func qwen38ProductionRoutePeakRSSBytes() -> UInt64 {
    var usage = rusage()
    let result = getrusage(RUSAGE_SELF, &usage)
    return qwen38ProductionRoutePeakRSSBytes(
        getrusageResult: result,
        maxResidentSetSize: usage.ru_maxrss)
}

func qwen38ProductionRoutePeakRSSBytes(
    getrusageResult: Int32,
    maxResidentSetSize: Int
) -> UInt64 {
    guard getrusageResult == 0, maxResidentSetSize > 0 else { return 0 }
    return UInt64(maxResidentSetSize)
}

private func qwen38ProductionRouteBoundedCanonicalJSON(
    _ artifact: Qwen38ScorecardProductionRouteObservationArtifact
) throws -> Data {
    let data = try qwen38MTPScorecardCanonicalJSON(artifact)
    guard !data.isEmpty, data.count <= 1_048_576 else {
        throw Qwen38ScorecardProductionRouteObservationCLIError.invalidArtifact
    }
    return data
}

func writeFreshQwen38ScorecardProductionRouteObservation(
    _ data: Data,
    _ path: String
) throws {
    try writeFreshQwen38ScorecardProductionRouteObservation(
        data,
        path,
        directoryFsync: fsync)
}

func writeFreshQwen38ScorecardProductionRouteObservation(
    _ data: Data,
    _ path: String,
    directoryFsync: (Int32) -> Int32
) throws {
    guard !data.isEmpty,
        (try? JSONSerialization.jsonObject(with: data)) != nil
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError.invalidOutput
    }
    try Qwen38ScorecardProductionRouteObservationFreshOutput.validate(path)

    let outputURL = URL(fileURLWithPath: path).standardizedFileURL
    let parent = outputURL.deletingLastPathComponent()
    let temporaryURL = parent.appendingPathComponent(
        ".\(outputURL.lastPathComponent).\(UUID().uuidString).tmp")
    let descriptor = temporaryURL.withUnsafeFileSystemRepresentation {
        pointer -> Int32 in
        guard let pointer else { return -1 }
        return open(
            pointer,
            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
            0o600)
    }
    guard descriptor >= 0 else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .outputWriteFailed
    }
    var descriptorOpen = true
    var destinationLinked = false
    var publishSucceeded = false
    var linkedDestinationIdentity: Qwen38ProductionRouteFileIdentity?
    defer {
        if descriptorOpen { _ = close(descriptor) }
        if destinationLinked && !publishSucceeded {
            if let linkedDestinationIdentity,
                linkedDestinationIdentity
                    == qwen38ProductionRouteFileIdentity(outputURL)
            {
                _ = outputURL.withUnsafeFileSystemRepresentation { pointer in
                    pointer.map(unlink) ?? -1
                }
            }
        }
        _ = temporaryURL.withUnsafeFileSystemRepresentation { pointer in
            pointer.map(unlink) ?? -1
        }
    }

    let wroteAll = data.withUnsafeBytes { rawBuffer -> Bool in
        guard let base = rawBuffer.baseAddress else { return false }
        var offset = 0
        while offset < rawBuffer.count {
            let result = Darwin.write(
                descriptor,
                base.advanced(by: offset),
                rawBuffer.count - offset)
            if result < 0 {
                if errno == EINTR { continue }
                return false
            }
            guard result > 0 else { return false }
            offset += result
        }
        return true
    }
    var temporaryMetadata = stat()
    guard wroteAll,
        fsync(descriptor) == 0,
        fstat(descriptor, &temporaryMetadata) == 0
    else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .outputWriteFailed
    }
    let temporaryIdentity = Qwen38ProductionRouteFileIdentity(
        device: temporaryMetadata.st_dev,
        inode: temporaryMetadata.st_ino)
    guard close(descriptor) == 0 else {
        descriptorOpen = false
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .outputWriteFailed
    }
    descriptorOpen = false

    let linked = temporaryURL.withUnsafeFileSystemRepresentation {
        temporaryPointer -> Int32 in
        outputURL.withUnsafeFileSystemRepresentation { outputPointer -> Int32 in
            guard let temporaryPointer, let outputPointer else { return -1 }
            return Darwin.link(temporaryPointer, outputPointer)
        }
    }
    guard linked == 0 else {
        if errno == EEXIST {
            throw Qwen38ScorecardProductionRouteObservationCLIError
                .outputExists
        }
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .outputWriteFailed
    }
    destinationLinked = true
    linkedDestinationIdentity = temporaryIdentity

    let directoryDescriptor = parent.withUnsafeFileSystemRepresentation {
        pointer -> Int32 in
        guard let pointer else { return -1 }
        return open(pointer, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
    }
    guard directoryDescriptor >= 0 else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .outputWriteFailed
    }
    var directoryDescriptorOpen = true
    defer {
        if directoryDescriptorOpen { _ = close(directoryDescriptor) }
    }
    guard directoryFsync(directoryDescriptor) == 0 else {
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .outputWriteFailed
    }
    guard close(directoryDescriptor) == 0 else {
        directoryDescriptorOpen = false
        throw Qwen38ScorecardProductionRouteObservationCLIError
            .outputWriteFailed
    }
    directoryDescriptorOpen = false
    publishSucceeded = true
}

private struct Qwen38ProductionRouteFileIdentity: Equatable {
    let device: dev_t
    let inode: ino_t
}

private func qwen38ProductionRouteFileIdentity(
    _ url: URL
) -> Qwen38ProductionRouteFileIdentity? {
    var metadata = stat()
    let result = url.withUnsafeFileSystemRepresentation { pointer -> Int32 in
        guard let pointer else { return -1 }
        return lstat(pointer, &metadata)
    }
    guard result == 0 else { return nil }
    return Qwen38ProductionRouteFileIdentity(
        device: metadata.st_dev,
        inode: metadata.st_ino)
}

struct Qwen38ScorecardProductionRouteSourceLockObservation:
    Codable, Equatable, Sendable
{
    private static let metadataOnlyLockAuthenticationScope =
        "config-tokenizer-tensor-header-descriptor-metadata-only"
    private static let metadataOnlyLockExplanation =
        "source lock authenticates config, tokenizer, and safetensor " +
        "tensor header/descriptor metadata only; it does not authenticate " +
        "safetensor payload bytes"

    let knownArtifact: String
    let selection: String
    let lockSourceRevision: String
    let lockAuthenticationScope: String
    let safetensorPayloadBytesAuthenticated: Bool
    let safetensorPayloadBytesAuthenticationExplanation: String
    let targetModelID: String
    let targetRevision: String
    let targetConfigSHA256: String
    let targetTokenizerSHA256: String
    let targetTensorManifestSHA256: String
    let drafterModelID: String
    let drafterRevision: String
    let drafterConfigSHA256: String
    let drafterTokenizerSHA256: String
    let drafterTensorManifestSHA256: String
    let targetQuantizationBits: Int
    let targetQuantizationGroupSize: Int
    let targetQuantizationMode: String
    let drafterQuantizationBits: Int
    let drafterQuantizationGroupSize: Int
    let drafterQuantizationMode: String
    let runtimeBlockSize: Int
    let maximumAcceptedDraftTokens: Int

    init(
        knownArtifact: String,
        selection: String,
        lockSourceRevision: String,
        targetModelID: String,
        targetRevision: String,
        targetConfigSHA256: String,
        targetTokenizerSHA256: String,
        targetTensorManifestSHA256: String,
        drafterModelID: String,
        drafterRevision: String,
        drafterConfigSHA256: String,
        drafterTokenizerSHA256: String,
        drafterTensorManifestSHA256: String,
        targetQuantizationBits: Int,
        targetQuantizationGroupSize: Int,
        targetQuantizationMode: String,
        drafterQuantizationBits: Int,
        drafterQuantizationGroupSize: Int,
        drafterQuantizationMode: String,
        runtimeBlockSize: Int,
        maximumAcceptedDraftTokens: Int
    ) {
        self.knownArtifact = knownArtifact
        self.selection = selection
        self.lockSourceRevision = lockSourceRevision
        self.lockAuthenticationScope =
            Self.metadataOnlyLockAuthenticationScope
        self.safetensorPayloadBytesAuthenticated = false
        self.safetensorPayloadBytesAuthenticationExplanation =
            Self.metadataOnlyLockExplanation
        self.targetModelID = targetModelID
        self.targetRevision = targetRevision
        self.targetConfigSHA256 = targetConfigSHA256
        self.targetTokenizerSHA256 = targetTokenizerSHA256
        self.targetTensorManifestSHA256 = targetTensorManifestSHA256
        self.drafterModelID = drafterModelID
        self.drafterRevision = drafterRevision
        self.drafterConfigSHA256 = drafterConfigSHA256
        self.drafterTokenizerSHA256 = drafterTokenizerSHA256
        self.drafterTensorManifestSHA256 = drafterTensorManifestSHA256
        self.targetQuantizationBits = targetQuantizationBits
        self.targetQuantizationGroupSize = targetQuantizationGroupSize
        self.targetQuantizationMode = targetQuantizationMode
        self.drafterQuantizationBits = drafterQuantizationBits
        self.drafterQuantizationGroupSize = drafterQuantizationGroupSize
        self.drafterQuantizationMode = drafterQuantizationMode
        self.runtimeBlockSize = runtimeBlockSize
        self.maximumAcceptedDraftTokens = maximumAcceptedDraftTokens
    }

    enum CodingKeys: String, CodingKey {
        case knownArtifact
        case selection
        case lockSourceRevision
        case lockAuthenticationScope
        case safetensorPayloadBytesAuthenticated
        case safetensorPayloadBytesAuthenticationExplanation
        case targetModelID
        case targetRevision
        case targetConfigSHA256
        case targetTokenizerSHA256
        case targetTensorManifestSHA256 =
            "targetTensorHeaderDescriptorManifestSHA256"
        case drafterModelID
        case drafterRevision
        case drafterConfigSHA256
        case drafterTokenizerSHA256
        case drafterTensorManifestSHA256 =
            "drafterTensorHeaderDescriptorManifestSHA256"
        case targetQuantizationBits
        case targetQuantizationGroupSize
        case targetQuantizationMode
        case drafterQuantizationBits
        case drafterQuantizationGroupSize
        case drafterQuantizationMode
        case runtimeBlockSize
        case maximumAcceptedDraftTokens
    }

    init(
        binding: QwenMTPArtifactBinding,
        knownArtifact: String,
        selection: String,
        lock: Qwen38MTPPerformanceScorecardArtifact
    ) {
        self.init(
            knownArtifact: knownArtifact,
            selection: selection,
            lockSourceRevision: lock.lockSourceRevision,
            targetModelID: binding.targetModelID,
            targetRevision: binding.targetRevision,
            targetConfigSHA256: lock.targetConfigSHA256,
            targetTokenizerSHA256: lock.tokenizerSHA256,
            targetTensorManifestSHA256: lock.targetTensorManifestSHA256,
            drafterModelID: binding.drafterModelID,
            drafterRevision: binding.drafterRevision,
            drafterConfigSHA256: lock.drafterConfigSHA256,
            drafterTokenizerSHA256: lock.tokenizerSHA256,
            drafterTensorManifestSHA256: lock.drafterTensorManifestSHA256,
            targetQuantizationBits: lock.targetQuantizationBits,
            targetQuantizationGroupSize: lock.targetQuantizationGroupSize,
            targetQuantizationMode: lock.targetQuantizationMode,
            drafterQuantizationBits: lock.drafterQuantizationBits,
            drafterQuantizationGroupSize: lock.drafterQuantizationGroupSize,
            drafterQuantizationMode: lock.drafterQuantizationMode,
            runtimeBlockSize: binding.runtimeBlockSize,
            maximumAcceptedDraftTokens: binding.maximumAcceptedDraftTokens)
    }

    init(binding lock: Qwen38MTPPerformanceScorecardArtifact) {
        self.init(
            knownArtifact: Qwen38MTPPerformanceScorecardGate.modelArtifactLabel,
            selection: Qwen35ExactMTPRuntimeSelection
                .qwen38_27BMXFP8Depth1.rawValue,
            lockSourceRevision: lock.lockSourceRevision,
            targetModelID: QwenMTPKnownArtifactLocks
                .qwen38_27BMXFP8Depth1.targetIdentity.modelID,
            targetRevision: lock.targetRevision,
            targetConfigSHA256: lock.targetConfigSHA256,
            targetTokenizerSHA256: lock.tokenizerSHA256,
            targetTensorManifestSHA256: lock.targetTensorManifestSHA256,
            drafterModelID: QwenMTPKnownArtifactLocks
                .qwen38_27BMXFP8Depth1.drafterIdentity.modelID,
            drafterRevision: lock.drafterRevision,
            drafterConfigSHA256: lock.drafterConfigSHA256,
            drafterTokenizerSHA256: lock.tokenizerSHA256,
            drafterTensorManifestSHA256: lock.drafterTensorManifestSHA256,
            targetQuantizationBits: lock.targetQuantizationBits,
            targetQuantizationGroupSize: lock.targetQuantizationGroupSize,
            targetQuantizationMode: lock.targetQuantizationMode,
            drafterQuantizationBits: lock.drafterQuantizationBits,
            drafterQuantizationGroupSize: lock.drafterQuantizationGroupSize,
            drafterQuantizationMode: lock.drafterQuantizationMode,
            runtimeBlockSize: lock.blockSize,
            maximumAcceptedDraftTokens: lock.maxAcceptedDrafts)
    }
}

struct Qwen38ScorecardProductionRouteEnvironmentObservation:
    Codable, Equatable, Sendable
{
    let gdnMode: String
    let observedGDNEnv: String
}

struct Qwen38ScorecardProductionRouteExecutableObservation:
    Codable, Equatable, Sendable
{
    let executableIdentitySource: String
    let executableSHA256: String
    let harnessGitSHA: String
    let processID: Int
    let parentProcessID: Int
    let processStartUptimeNanoseconds: UInt64
    let bootTimeUnixSeconds: Int64
    let sourceID: String
}

struct Qwen38ScorecardProductionRouteHostObservation:
    Codable, Equatable, Sendable
{
    let hostUse: String
    let hostUseSource: String
    let hostUsePolicyVersion: String
    let expectedChip: String
    let chipName: String
    let physicalRAMBytes: UInt64
    let hardwareOS: String
    let wiredLimitMB: Int
    let wiredLimitProvenance: String
    let metalRecommendedMaxWorkingSetSizeBytes: UInt64
    let metalCurrentAllocatedSizeBytes: UInt64
}

struct Qwen38ScorecardProductionRouteMemoryBudgetObservation:
    Codable, Equatable, Sendable
{
    let memoryLimitBytes: UInt64
    let cacheLimitBytes: UInt64
    let reservedKVBytes: UInt64
    let reservedIOBytes: UInt64
    let reservedPrefetchBytes: UInt64
    let osServiceReserveBytes: UInt64
    let contextTokens: UInt64
}

struct Qwen38ScorecardProductionRouteMetalMemorySnapshot:
    Codable, Equatable, Sendable
{
    let activeMetalBytes: UInt64
    let cachedMetalBytes: UInt64
    let peakMetalBytes: UInt64
}

struct Qwen38ScorecardProductionRouteHostFacts:
    Codable, Equatable, Sendable
{
    let metal: Qwen38ScorecardProductionRouteMetalMemorySnapshot
    let wiredLimitMB: Int
    let metalRecommendedMaxWorkingSetSizeBytes: UInt64
    let metalCurrentAllocatedSizeBytes: UInt64
    let thermalState: String
    let peakRSSBytes: UInt64
    let physicalFootprintBytes: UInt64
}

struct Qwen38ScorecardProductionRouteCleanupAttemptObservation:
    Codable, Equatable, Sendable
{
    let kind: String
    let attempt: Int
    let attemptLimit: Int
    let elapsedNanoseconds: UInt64
    let before: Qwen38ScorecardProductionRouteHostFacts
    let observed: Qwen38ScorecardProductionRouteHostFacts
    let currentAllocatedAtOrBelowBaseline: Bool
    let activeAtOrBelowBaseline: Bool
    let cachedAtOrBelowBaseline: Bool
    let processMemoryReadbackValid: Bool
    let residualAdmissionSafe: Bool
    /// The documented absolute residual tolerance in force for this run; the
    /// `AtOrBelowBaseline` flags above stay strict raw truth, so a record can
    /// honestly show a flag false while the gated comparison passed within
    /// this bound. Raw before/observed facts make the residual derivable.
    let residualToleranceBytes: UInt64
}

struct Qwen38ScorecardProductionRouteStartupReportObservation:
    Codable, Equatable, Sendable
{
    let launchedModel: String
    let route: String
    let memoryLimitBytes: Int
    let cacheLimitBytes: Int
    let maxReservedKVBytes: Int
    let maxContextTokens: Int
    let maxReservedContextTokens: Int
    let modelFamily: String
    let modelConfigurationSHA256: String
    let layerCount: Int
    let keyValueHeadCount: Int
    let headDimension: Int
    let stopTokenCount: Int
    let stopStringCount: Int
    let nativeCacheKinds: [String]
    let startupInputTokenCount: Int
    let startupGeneratedTokenCount: Int
    let maxActiveSlots: Int
    let maxPrefillSlots: Int
    let prefillChunkSize: Int
    let maxQueuedRequests: Int
    let publicationCapacity: Int
    let soloPLDPolicy: String?
    let modelProofVerified: Bool
    let defaultMaximumCompletionTokens: Int

    init(
        _ report: ContinuousServingModelStartupReport,
        defaultMaximumCompletionTokens: Int
    ) {
        self.launchedModel = report.launchedModel
        self.route = report.route.rawValue
        self.memoryLimitBytes = report.memoryLimitBytes
        self.cacheLimitBytes = report.cacheLimitBytes
        self.maxReservedKVBytes = report.maxReservedKVBytes
        self.maxContextTokens = report.maxContextTokens
        self.maxReservedContextTokens = report.maxReservedContextTokens
        self.modelFamily = report.modelFamily.rawValue
        self.modelConfigurationSHA256 = report.modelConfigurationSHA256
        self.layerCount = report.layerCount
        self.keyValueHeadCount = report.keyValueHeadCount
        self.headDimension = report.headDimension
        self.stopTokenCount = report.stopTokenCount
        self.stopStringCount = report.stopStringCount
        self.nativeCacheKinds = report.nativeCacheKinds.map(\.rawValue)
        self.startupInputTokenCount = report.startupPromptTokenCount
        self.startupGeneratedTokenCount = report.startupGeneratedTokenCount
        self.maxActiveSlots = report.maxActiveSlots
        self.maxPrefillSlots = report.maxPrefillSlots
        self.prefillChunkSize = report.prefillChunkSize
        self.maxQueuedRequests = report.maxQueuedRequests
        self.publicationCapacity = report.publicationCapacity
        self.soloPLDPolicy = report.soloPLDPolicy.map { "\($0)" }
        self.modelProofVerified = report.modelProofVerified
        self.defaultMaximumCompletionTokens = defaultMaximumCompletionTokens
    }

    enum CodingKeys: String, CodingKey {
        case launchedModel
        case route
        case memoryLimitBytes
        case cacheLimitBytes
        case maxReservedKVBytes
        case maxContextTokens
        case maxReservedContextTokens
        case modelFamily
        case modelConfigurationSHA256
        case layerCount
        case keyValueHeadCount
        case headDimension
        case stopTokenCount
        case stopStringCount
        case nativeCacheKinds
        case startupInputTokenCount
        case startupGeneratedTokenCount
        case maxActiveSlots
        case maxPrefillSlots
        case prefillChunkSize
        case maxQueuedRequests
        case publicationCapacity
        case soloPLDPolicy
        case modelProofVerified = "metadataProofVerified"
        case defaultMaximumCompletionTokens
    }
}

struct Qwen38ScorecardProductionRouteResultObservation:
    Codable, Equatable, Sendable
{
    let evidenceKind: String
    let concurrency: Int
    let coordinatorRequestIDs: [UInt64]
    let coordinatorPlanObservations:
        [Qwen38ScorecardProductionRoutePlanObservationDTO]
    let planRevisions: [Qwen38ScorecardProductionRouteRevisionObservation]
    let sharedBatchDecodeRequestIDs: [UInt64]
    let peakActiveSlots: Int
    let peakBatchOccupancy: Int
    let finalActiveRequests: Int
    let finalCoordinatorSlots: Int
    let finalReservedKVBytes: Int
    let requests: [Qwen38ScorecardProductionRouteRequestObservation]

    init(_ result: Qwen38ScorecardContinuousRouteResult) {
        evidenceKind = result.evidenceKind.rawValue
        concurrency = result.concurrency
        coordinatorRequestIDs = result.coordinatorRequestIDs
        coordinatorPlanObservations = result.coordinatorPlanObservations.map {
            Qwen38ScorecardProductionRoutePlanObservationDTO($0)
        }
        planRevisions = result.planRevisions.map {
            Qwen38ScorecardProductionRouteRevisionObservation($0)
        }
        sharedBatchDecodeRequestIDs = result.sharedBatchDecodeRequestIDs
        peakActiveSlots = result.peakActiveSlots
        peakBatchOccupancy = result.peakBatchOccupancy
        finalActiveRequests = result.finalActiveRequests
        finalCoordinatorSlots = result.finalCoordinatorSlots
        finalReservedKVBytes = result.finalReservedKVBytes
        requests = result.requests.map {
            Qwen38ScorecardProductionRouteRequestObservation($0)
        }
    }
}

struct Qwen38ScorecardProductionRoutePlanObservationDTO:
    Codable, Equatable, Sendable
{
    let planSequence: Int
    let stateRevisionAfterApply: Int
    let admissions: [UInt64]
    let decodeKind: String
    let decodeRequestIDs: [UInt64]
    let speculationAllowed: Bool
    let prefillRequestIDs: [UInt64]
    let activeSlotCount: Int
    let queuedSlotCount: Int

    init(_ plan: Qwen38ScorecardContinuousRoutePlanObservation) {
        planSequence = plan.planSequence
        stateRevisionAfterApply = plan.stateRevisionAfterApply
        admissions = plan.admissions
        decodeKind = plan.decodeKind.rawValue
        decodeRequestIDs = plan.decodeRequestIDs
        speculationAllowed = plan.speculationAllowed
        prefillRequestIDs = plan.prefillRequestIDs
        activeSlotCount = plan.activeSlotCount
        queuedSlotCount = plan.queuedSlotCount
    }
}

struct Qwen38ScorecardProductionRouteRevisionObservation:
    Codable, Equatable, Sendable
{
    let planSequence: Int
    let stateRevisionAfterApply: Int

    init(_ revision: Qwen38ScorecardContinuousRouteRevision) {
        planSequence = revision.planSequence
        stateRevisionAfterApply = revision.stateRevisionAfterApply
    }
}

struct Qwen38ScorecardProductionRouteRequestObservation:
    Codable, Equatable, Sendable
{
    let requestIndex: Int
    let coordinatorRequestID: UInt64
    let route: String
    let outputTokenIDs: [Int]
    let finishReason: String
    let usagePromptTokens: Int
    let usageCompletionTokens: Int
    let usageTotalTokens: Int
    let admittedAtUptime: Double
    let completedAtUptime: Double

    init(_ request: Qwen38ScorecardContinuousRouteRequestResult) {
        requestIndex = request.requestIndex
        coordinatorRequestID = request.coordinatorRequestID
        route = request.route.rawValue
        outputTokenIDs = request.outputTokenIDs
        finishReason = request.finishReason.rawValue
        usagePromptTokens = request.usage.promptTokens
        usageCompletionTokens = request.usage.completionTokens
        usageTotalTokens = request.usage.totalTokens
        admittedAtUptime = request.admittedAtUptime
        completedAtUptime = request.completedAtUptime
    }
}

struct Qwen38ScorecardProductionRouteObservationArtifact:
    Codable, Equatable, Sendable
{
    let kind: String
    let producedAt: String
    let attestationStatus: String
    let isProductionRouteReceipt: Bool
    let controllerSignatureVerified: Bool
    let promotionAuthorized: Bool
    let runtimeAuthorityGranted: Bool
    let runnerObservationDigestExplanation: String
    let sourceLock: Qwen38ScorecardProductionRouteSourceLockObservation
    let environment: Qwen38ScorecardProductionRouteEnvironmentObservation
    let executable: Qwen38ScorecardProductionRouteExecutableObservation
    let host: Qwen38ScorecardProductionRouteHostObservation
    let memoryBudget: Qwen38ScorecardProductionRouteMemoryBudgetObservation
    let memoryBefore: Qwen38ScorecardProductionRouteHostFacts
    let memoryAfter: Qwen38ScorecardProductionRouteHostFacts
    let startupReport: Qwen38ScorecardProductionRouteStartupReportObservation
    let c2: Qwen38ScorecardProductionRouteResultObservation
    let c4: Qwen38ScorecardProductionRouteResultObservation
    let runnerObservationDigest: String
    let artifactDigest: String

    static func build(
        date: String,
        sourceLock: Qwen38ScorecardProductionRouteSourceLockObservation,
        observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv,
        processIsolation: Qwen38MTPLiveExactnessProcessIsolationEvidence,
        hostUse: String,
        hostUseSource: String,
        expectedChip: String,
        hostSnapshot: Qwen38MTPScorecardLiveHostMemorySnapshot,
        memoryBudget: Qwen38MTPScorecardLiveMemoryBudget,
        contextTokens: UInt64,
        before: Qwen38ScorecardProductionRouteHostFacts,
        after: Qwen38ScorecardProductionRouteHostFacts,
        startupReport: ContinuousServingModelStartupReport,
        observation: Qwen38ScorecardProductionRouteObservation
    ) throws -> Self {
        let unsigned = Self(
            kind: qwen38ProductionRouteObservationKind,
            producedAt: date,
            attestationStatus: "unsigned",
            isProductionRouteReceipt: false,
            controllerSignatureVerified: false,
            promotionAuthorized: false,
            runtimeAuthorityGranted: false,
            runnerObservationDigestExplanation:
                "runner route-observation digest; not attribution receipt digest",
            sourceLock: sourceLock,
            environment: .init(
                gdnMode: Qwen38MTPPerformanceScorecardGDNMode.gdnOn.rawValue,
                observedGDNEnv: observedEnv.rawValue),
            executable: .init(
                executableIdentitySource:
                    processIsolation.executableIdentitySource.rawValue,
                executableSHA256: processIsolation.executableSHA256,
                harnessGitSHA: processIsolation.harnessGitSHA,
                processID: processIsolation.processID,
                parentProcessID: processIsolation.parentProcessID,
                processStartUptimeNanoseconds:
                    processIsolation.processStartUptimeNanoseconds,
                bootTimeUnixSeconds: processIsolation.bootTimeUnixSeconds,
                sourceID: processIsolation.sourceID),
            host: .init(
                hostUse: hostUse,
                hostUseSource: hostUseSource,
                hostUsePolicyVersion: Qwen38MTPLiveExactnessGate
                    .requiredHostUsePolicyVersion,
                expectedChip: expectedChip,
                chipName: hostSnapshot.chipName,
                physicalRAMBytes: hostSnapshot.physicalRAMBytes,
                hardwareOS: ProvenanceCLI.osVersion(),
                wiredLimitMB: Int(min(
                    hostSnapshot.wiredLimitMB,
                    UInt64(Int.max))),
                wiredLimitProvenance: "measured",
                metalRecommendedMaxWorkingSetSizeBytes:
                    hostSnapshot.metalRecommendedMaxWorkingSetSizeBytes ?? 0,
                metalCurrentAllocatedSizeBytes:
                    hostSnapshot.metalCurrentAllocatedSizeBytes),
            memoryBudget: .init(
                memoryLimitBytes: memoryBudget.memoryLimitBytes,
                cacheLimitBytes: memoryBudget.cacheLimitBytes,
                reservedKVBytes: memoryBudget.reservedKVBytes,
                reservedIOBytes: memoryBudget.reservedIOBytes,
                reservedPrefetchBytes: memoryBudget.reservedPrefetchBytes,
                osServiceReserveBytes: memoryBudget.osServiceReserveBytes,
                contextTokens: contextTokens),
            memoryBefore: before,
            memoryAfter: after,
            startupReport: .init(
                startupReport,
                defaultMaximumCompletionTokens: 8),
            c2: .init(observation.c2),
            c4: .init(observation.c4),
            runnerObservationDigest: observation.observationDigest,
            artifactDigest: "")
        let digest = try qwen38ScorecardProductionRouteObservationArtifactDigest(
            unsigned)
        return Self(
            kind: unsigned.kind,
            producedAt: unsigned.producedAt,
            attestationStatus: unsigned.attestationStatus,
            isProductionRouteReceipt: unsigned.isProductionRouteReceipt,
            controllerSignatureVerified:
                unsigned.controllerSignatureVerified,
            promotionAuthorized: unsigned.promotionAuthorized,
            runtimeAuthorityGranted: unsigned.runtimeAuthorityGranted,
            runnerObservationDigestExplanation:
                unsigned.runnerObservationDigestExplanation,
            sourceLock: unsigned.sourceLock,
            environment: unsigned.environment,
            executable: unsigned.executable,
            host: unsigned.host,
            memoryBudget: unsigned.memoryBudget,
            memoryBefore: unsigned.memoryBefore,
            memoryAfter: unsigned.memoryAfter,
            startupReport: unsigned.startupReport,
            c2: unsigned.c2,
            c4: unsigned.c4,
            runnerObservationDigest: unsigned.runnerObservationDigest,
            artifactDigest: digest)
    }
}

func qwen38ScorecardProductionRouteObservationArtifactDigest(
    _ artifact: Qwen38ScorecardProductionRouteObservationArtifact
) throws -> String {
    try qwen38MTPScorecardSHA256Hex(qwen38MTPScorecardCanonicalJSON(
        Qwen38ScorecardProductionRouteObservationArtifactDigestBasis(
            kind: artifact.kind,
            producedAt: artifact.producedAt,
            attestationStatus: artifact.attestationStatus,
            isProductionRouteReceipt: artifact.isProductionRouteReceipt,
            controllerSignatureVerified:
                artifact.controllerSignatureVerified,
            promotionAuthorized: artifact.promotionAuthorized,
            runtimeAuthorityGranted: artifact.runtimeAuthorityGranted,
            runnerObservationDigestExplanation:
                artifact.runnerObservationDigestExplanation,
            sourceLock: artifact.sourceLock,
            environment: artifact.environment,
            executable: artifact.executable,
            host: artifact.host,
            memoryBudget: artifact.memoryBudget,
            memoryBefore: artifact.memoryBefore,
            memoryAfter: artifact.memoryAfter,
            startupReport: artifact.startupReport,
            c2: artifact.c2,
            c4: artifact.c4,
            runnerObservationDigest: artifact.runnerObservationDigest)))
}

private struct Qwen38ScorecardProductionRouteObservationArtifactDigestBasis:
    Codable, Equatable, Sendable
{
    let kind: String
    let producedAt: String
    let attestationStatus: String
    let isProductionRouteReceipt: Bool
    let controllerSignatureVerified: Bool
    let promotionAuthorized: Bool
    let runtimeAuthorityGranted: Bool
    let runnerObservationDigestExplanation: String
    let sourceLock: Qwen38ScorecardProductionRouteSourceLockObservation
    let environment: Qwen38ScorecardProductionRouteEnvironmentObservation
    let executable: Qwen38ScorecardProductionRouteExecutableObservation
    let host: Qwen38ScorecardProductionRouteHostObservation
    let memoryBudget: Qwen38ScorecardProductionRouteMemoryBudgetObservation
    let memoryBefore: Qwen38ScorecardProductionRouteHostFacts
    let memoryAfter: Qwen38ScorecardProductionRouteHostFacts
    let startupReport: Qwen38ScorecardProductionRouteStartupReportObservation
    let c2: Qwen38ScorecardProductionRouteResultObservation
    let c4: Qwen38ScorecardProductionRouteResultObservation
    let runnerObservationDigest: String
}

private func qwen38ScorecardProductionRouteThermalState() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}
