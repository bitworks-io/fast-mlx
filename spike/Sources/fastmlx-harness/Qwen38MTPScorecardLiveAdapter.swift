import CryptoKit
import Darwin
import Foundation
import HarnessCore
import Metal

enum Qwen38MTPScorecardLiveAdapterError: Error, Equatable, CustomStringConvertible, Sendable {
    case missingFlag(String)
    case duplicateFlag(String)
    case unknownFlag
    case missingValue(String)
    case unexpectedPositional
    case invalidInteger(String)
    case invalidMemoryBudget
    case unsafeOutput
    case outputExists
    case outputWriteFailed
    case invalidWorkerRole(Qwen38MTPPerformanceScorecardEngineRole)
    case invalidHandshake(Qwen38MTPPerformanceScorecardEngineRole)
    case exactnessLaunchBindingMismatch
    case workerEnvDrift(Qwen38MTPPerformanceScorecardEngineRole)
    case workerRestarted(Qwen38MTPPerformanceScorecardEngineRole)
    case workerExited(Qwen38MTPPerformanceScorecardEngineRole)
    case malformedWorkerResponse
    case duplicateWorkerResponse(Int)
    case outOfOrderWorkerResponse(expected: Int, actual: Int)
    case workerError

    var description: String {
        switch self {
        case .missingFlag(let flag): return "missing required \(flag)"
        case .duplicateFlag(let flag): return "duplicate \(flag)"
        case .unknownFlag: return "unknown flag"
        case .missingValue(let flag): return "\(flag) requires a value"
        case .unexpectedPositional: return "unexpected positional argument"
        case .invalidInteger(let flag): return "\(flag) must be a positive integer"
        case .invalidMemoryBudget: return "invalid explicit memory budget"
        case .unsafeOutput: return "output destination is unsafe"
        case .outputExists: return "output destination already exists"
        case .outputWriteFailed: return "failed to publish scorecard outputs"
        case .invalidWorkerRole: return "worker role mismatch"
        case .invalidHandshake: return "worker handshake failed validation"
        case .exactnessLaunchBindingMismatch:
            return "candidate exactness binding does not match candidate measurement binding"
        case .workerEnvDrift: return "worker environment drifted"
        case .workerRestarted: return "worker process identity changed"
        case .workerExited: return "worker exited"
        case .malformedWorkerResponse: return "worker returned malformed response"
        case .duplicateWorkerResponse: return "worker returned duplicate response"
        case .outOfOrderWorkerResponse: return "worker returned out-of-order response"
        case .workerError: return "worker command failed"
        }
    }
}

struct Qwen38MTPScorecardLiveHostMemorySnapshot: Equatable, Sendable {
    static let requiredScorecardChip = "Apple M3 Ultra"

    let physicalRAMBytes: UInt64
    let chipName: String
    let wiredLimitMB: UInt64
    let metalRecommendedMaxWorkingSetSizeBytes: UInt64?
    let metalCurrentAllocatedSizeBytes: UInt64

    static func live() throws -> Qwen38MTPScorecardLiveHostMemorySnapshot {
        guard let device = MTLCreateSystemDefaultDevice(),
            device.recommendedMaxWorkingSetSize > 0
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        return Qwen38MTPScorecardLiveHostMemorySnapshot(
            physicalRAMBytes: ProvenanceCLI.ramBytes(),
            chipName: device.name,
            wiredLimitMB: try qwen38MTPScorecardMeasuredWiredLimitMB(),
            metalRecommendedMaxWorkingSetSizeBytes: device.recommendedMaxWorkingSetSize,
            metalCurrentAllocatedSizeBytes: UInt64(max(device.currentAllocatedSize, 0)))
    }

    var effectiveWiredLimitBytes: UInt64 {
        get throws {
            guard physicalRAMBytes > 0 else {
                throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
            }
            guard wiredLimitMB > 0 else { return physicalRAMBytes }
            let multiplied = wiredLimitMB.multipliedReportingOverflow(by: 1_048_576)
            guard !multiplied.overflow, multiplied.partialValue > 0 else {
                throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
            }
            return min(physicalRAMBytes, multiplied.partialValue)
        }
    }

    var effectiveMetalCeilingBytes: UInt64 {
        get throws {
            guard let recommended = metalRecommendedMaxWorkingSetSizeBytes,
                recommended > 0
            else {
                throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
            }
            return min(try effectiveWiredLimitBytes, recommended)
        }
    }
}

struct Qwen38MTPScorecardLiveMemoryBudget: Equatable, Codable, Sendable {
    let memoryLimitBytes: UInt64
    let cacheLimitBytes: UInt64
    let reservedKVBytes: UInt64
    let reservedIOBytes: UInt64
    let reservedPrefetchBytes: UInt64
    let osServiceReserveBytes: UInt64

    init(
        memoryLimitBytes: UInt64,
        cacheLimitBytes: UInt64,
        reservedKVBytes: UInt64,
        reservedIOBytes: UInt64,
        reservedPrefetchBytes: UInt64,
        osServiceReserveBytes: UInt64
    ) throws {
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.reservedKVBytes = reservedKVBytes
        self.reservedIOBytes = reservedIOBytes
        self.reservedPrefetchBytes = reservedPrefetchBytes
        self.osServiceReserveBytes = osServiceReserveBytes
        try validate()
    }

    var liveExactnessBudget: Qwen38MTPLiveExactnessMLXMemoryBudget {
        Qwen38MTPLiveExactnessMLXMemoryBudget(
            memoryLimitBytes: Int(memoryLimitBytes),
            cacheLimitBytes: Int(cacheLimitBytes))
    }

    func validate() throws {
        guard memoryLimitBytes > 0,
            cacheLimitBytes > 0,
            cacheLimitBytes <= memoryLimitBytes,
            reservedKVBytes > 0,
            reservedIOBytes > 0,
            reservedPrefetchBytes > 0,
            osServiceReserveBytes > 0,
            cacheLimitBytes + reservedKVBytes <= memoryLimitBytes
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
    }

    var workerResidentPlanBytes: UInt64 {
        get throws {
            try qwen38MTPScorecardCheckedSum([
                memoryLimitBytes,
                reservedIOBytes,
                reservedPrefetchBytes,
            ])
        }
    }

    func validateAgainstHost(_ snapshot: Qwen38MTPScorecardLiveHostMemorySnapshot) throws {
        let ceiling = try snapshot.effectiveMetalCeilingBytes
        guard ceiling >= memoryLimitBytes else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        let required = try qwen38MTPScorecardCheckedSum([
            workerResidentPlanBytes,
            osServiceReserveBytes,
            snapshot.metalCurrentAllocatedSizeBytes,
        ])
        guard required <= ceiling else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
    }

    func validateTwoWorkerAdmission(
        _ snapshot: Qwen38MTPScorecardLiveHostMemorySnapshot,
        expectedChip: String
    ) throws {
        guard !expectedChip.isEmpty,
            expectedChip == Qwen38MTPScorecardLiveHostMemorySnapshot.requiredScorecardChip,
            snapshot.chipName == expectedChip,
            snapshot.physicalRAMBytes == Qwen38MTPPerformanceScorecardGate.requiredRAMBytes
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        try validateAgainstHost(snapshot)
        let ceiling = try snapshot.effectiveMetalCeilingBytes
        let workers = try workerResidentPlanBytes.multipliedReportingOverflow(by: 2)
        guard !workers.overflow else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        let required = try qwen38MTPScorecardCheckedSum([
            workers.partialValue,
            osServiceReserveBytes,
            snapshot.metalCurrentAllocatedSizeBytes,
        ])
        guard required <= ceiling else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
    }
}

struct Qwen38MTPScorecardLiveRunArguments: Equatable, Sendable {
    let targetPath: String
    let drafterPath: String
    let outputPath: String
    let authorityOutputPath: String
    let hostUse: String
    let hostUseSource: String
    let expectedChip: String
    let memoryBudget: Qwen38MTPScorecardLiveMemoryBudget
}

func parseQwen38MTPScorecardLiveRunArguments(
    _ arguments: [String]
) throws -> Qwen38MTPScorecardLiveRunArguments {
    let allowed = Set([
        "--target",
        "--drafter",
        "--output",
        "--authority-output",
        "--host-use",
        "--host-use-source",
        "--expected-chip",
        "--memory-limit-bytes",
        "--cache-limit-bytes",
        "--reserved-kv-bytes",
        "--reserved-io-bytes",
        "--reserved-prefetch-bytes",
        "--os-service-reserve-bytes",
    ])
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw Qwen38MTPScorecardLiveAdapterError.unexpectedPositional
        }
        guard allowed.contains(flag) else {
            throw Qwen38MTPScorecardLiveAdapterError.unknownFlag
        }
        guard values[flag] == nil else {
            throw Qwen38MTPScorecardLiveAdapterError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count,
            !arguments[index + 1].hasPrefix("--")
        else {
            throw Qwen38MTPScorecardLiveAdapterError.missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }

    func require(_ flag: String) throws -> String {
        guard let value = values[flag], !value.isEmpty else {
            throw Qwen38MTPScorecardLiveAdapterError.missingFlag(flag)
        }
        return value
    }
    func requireUInt64(_ flag: String) throws -> UInt64 {
        let raw = try require(flag)
        guard let value = UInt64(raw), value > 0, value <= UInt64(Int.max) else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidInteger(flag)
        }
        return value
    }

    let hostUse = try require("--host-use")
    let hostUseSource = try require("--host-use-source")
    let expectedChip = try require("--expected-chip")
    guard hostUse == "dedicated-serving",
        hostUseSource == "operator-assertion"
    else {
        throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
    }

    return try Qwen38MTPScorecardLiveRunArguments(
        targetPath: require("--target"),
        drafterPath: require("--drafter"),
        outputPath: require("--output"),
        authorityOutputPath: require("--authority-output"),
        hostUse: hostUse,
        hostUseSource: hostUseSource,
        expectedChip: expectedChip,
        memoryBudget: Qwen38MTPScorecardLiveMemoryBudget(
            memoryLimitBytes: requireUInt64("--memory-limit-bytes"),
            cacheLimitBytes: requireUInt64("--cache-limit-bytes"),
            reservedKVBytes: requireUInt64("--reserved-kv-bytes"),
            reservedIOBytes: requireUInt64("--reserved-io-bytes"),
            reservedPrefetchBytes: requireUInt64("--reserved-prefetch-bytes"),
            osServiceReserveBytes: requireUInt64("--os-service-reserve-bytes")))
}

struct Qwen38MTPScorecardFreshOutputSet: Equatable, Sendable {
    let scorecardPath: String
    let authorityPath: String

    init(scorecardPath: String, authorityPath: String) throws {
        guard scorecardPath != authorityPath else {
            throw Qwen38MTPScorecardLiveAdapterError.unsafeOutput
        }
        try Self.validateFreshPath(scorecardPath)
        try Self.validateFreshPath(authorityPath)
        self.scorecardPath = scorecardPath
        self.authorityPath = authorityPath
    }

    static func validateFreshPath(_ path: String) throws {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let manager = FileManager.default
        guard !path.isEmpty, !outputPathIsSymbolicLink(url.path) else {
            throw Qwen38MTPScorecardLiveAdapterError.unsafeOutput
        }
        if manager.fileExists(atPath: url.path) {
            throw Qwen38MTPScorecardLiveAdapterError.outputExists
        }
        var isDirectory: ObjCBool = false
        guard manager.fileExists(atPath: url.deletingLastPathComponent().path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw Qwen38MTPScorecardLiveAdapterError.unsafeOutput
        }
    }
}

struct Qwen38MTPScorecardWorkerHandshake:
    Codable, Equatable, Sendable
{
    let role: Qwen38MTPPerformanceScorecardEngineRole
    let model: Qwen38MTPPerformanceScorecardModel
    let processIsolation: Qwen38MTPLiveExactnessProcessIsolationEvidence
    let launchBinding: Qwen38MTPPerformanceScorecardLaunchBinding
}

protocol Qwen38MTPScorecardWorkerClient: Sendable {
    var role: Qwen38MTPPerformanceScorecardEngineRole { get }
    func start() async throws -> Qwen38MTPScorecardWorkerHandshake
    func runCandidateExactness() async throws -> ResultRecord<Qwen38MTPLiveExactnessEvidence>
    func assertReadyForDispatch(expected: Qwen38MTPScorecardWorkerHandshake) async throws
    func measure(
        _ request: Qwen38MTPPerformanceScorecardMeasurementRequest
    ) async throws -> Qwen38MTPPerformanceScorecardEngineMeasurement
    func terminate() async
}

struct Qwen38MTPScorecardLiveRunResult: Sendable {
    let authority: Qwen38MTPPerformanceScorecardAuthorityBundle
    let record: ResultRecord<Qwen38MTPPerformanceScorecardEvidence>
}

struct Qwen38MTPScorecardLiveCoordinator<Candidate: Qwen38MTPScorecardWorkerClient, Reference: Qwen38MTPScorecardWorkerClient>: Sendable {
    let candidate: Candidate
    let reference: Reference
    let runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
    let provenance: Provenance
    let releaseBuildObserved: Bool

    func run() async throws -> Qwen38MTPScorecardLiveRunResult {
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

    private func validateHandshake(
        _ handshake: Qwen38MTPScorecardWorkerHandshake,
        expectedRole: Qwen38MTPPerformanceScorecardEngineRole
    ) throws {
        guard handshake.role == expectedRole else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidWorkerRole(handshake.role)
        }
        let expectedMode: Qwen38MTPPerformanceScorecardGDNMode =
            expectedRole == .candidate ? .gdnOn : .gdnOff
        let expectedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv =
            expectedRole == .candidate ? .enabled : .disabled
        guard handshake.model.gdnMode == expectedMode,
            handshake.processIsolation.gdnMode == expectedMode,
            handshake.processIsolation.observedEnv == expectedEnv,
            handshake.launchBinding.mode == expectedMode,
            handshake.launchBinding.observedEnv == expectedEnv,
            handshake.model.launchBinding == handshake.launchBinding
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidHandshake(expectedRole)
        }
    }
}

enum Qwen38MTPScorecardWorkerProtocolResponseKind: String, Codable, Sendable {
    case handshake
    case exactness
    case measurement
    case ok
    case error
}

struct Qwen38MTPScorecardWorkerProtocolResponse: Codable, Sendable {
    let sequence: Int
    let kind: Qwen38MTPScorecardWorkerProtocolResponseKind
    var handshake: Qwen38MTPScorecardWorkerHandshake?
    var exactnessRecord: ResultRecord<Qwen38MTPLiveExactnessEvidence>?
    var measurement: Qwen38MTPPerformanceScorecardEngineMeasurement?
    var error: String?

    init(
        sequence: Int,
        kind: Qwen38MTPScorecardWorkerProtocolResponseKind,
        handshake: Qwen38MTPScorecardWorkerHandshake? = nil,
        exactnessRecord: ResultRecord<Qwen38MTPLiveExactnessEvidence>? = nil,
        measurement: Qwen38MTPPerformanceScorecardEngineMeasurement? = nil,
        error: String? = nil
    ) {
        self.sequence = sequence
        self.kind = kind
        self.handshake = handshake
        self.exactnessRecord = exactnessRecord
        self.measurement = measurement
        self.error = error
    }

    func jsonLine() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }
}

enum Qwen38MTPScorecardWorkerProtocolRequestKind: String, Codable, Sendable {
    case handshake
    case exactness
    case measure
    case assertReady
    case shutdown
}

struct Qwen38MTPScorecardWorkerProtocolRequest: Codable, Sendable {
    let sequence: Int
    let kind: Qwen38MTPScorecardWorkerProtocolRequestKind
    let measurement: Qwen38MTPScorecardMeasurementCommand?
}

struct Qwen38MTPScorecardMeasurementCommand: Codable, Sendable {
    let role: Qwen38MTPPerformanceScorecardEngineRole
    let identity: Qwen38MTPPerformanceScorecardModel
    let schedule: Qwen38MTPPerformanceScorecardPairSchedule
    let workload: Qwen38MTPPerformanceScorecardWorkload
    let settings: Qwen38MTPPerformanceScorecardSettings

    init(_ request: Qwen38MTPPerformanceScorecardMeasurementRequest) {
        role = request.role
        identity = request.identity
        schedule = request.schedule
        workload = request.workload
        settings = request.settings
    }

    var request: Qwen38MTPPerformanceScorecardMeasurementRequest {
        Qwen38MTPPerformanceScorecardMeasurementRequest(
            role: role,
            identity: identity,
            schedule: schedule,
            workload: workload,
            settings: settings)
    }
}

protocol Qwen38MTPScorecardLineTransport: Sendable {
    func sendLine(_ line: String) async throws
    func receiveLine() async throws -> String?
    func terminate() async
}

actor Qwen38MTPScorecardLineProtocolClient: Qwen38MTPScorecardWorkerClient {
    let role: Qwen38MTPPerformanceScorecardEngineRole
    private let transport: Qwen38MTPScorecardLineTransport
    private var nextSequence = 1
    private var seenResponses: Set<Int> = []

    init(
        role: Qwen38MTPPerformanceScorecardEngineRole,
        transport: Qwen38MTPScorecardLineTransport
    ) {
        self.role = role
        self.transport = transport
    }

    func start() async throws -> Qwen38MTPScorecardWorkerHandshake {
        let response = try await roundTrip(kind: .handshake)
        guard response.kind == .handshake, let handshake = response.handshake else {
            throw Qwen38MTPScorecardLiveAdapterError.malformedWorkerResponse
        }
        return handshake
    }

    func runCandidateExactness() async throws -> ResultRecord<Qwen38MTPLiveExactnessEvidence> {
        guard role == .candidate else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidWorkerRole(role)
        }
        let response = try await roundTrip(kind: .exactness)
        guard response.kind == .exactness, let exactnessRecord = response.exactnessRecord else {
            throw Qwen38MTPScorecardLiveAdapterError.malformedWorkerResponse
        }
        return exactnessRecord
    }

    func assertReadyForDispatch(
        expected: Qwen38MTPScorecardWorkerHandshake
    ) async throws {
        let response = try await roundTrip(kind: .assertReady)
        guard response.kind == .ok else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
    }

    func measure(
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

    func terminate() async {
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

func writeFreshQwen38MTPScorecardAuthority(_ data: Data, _ path: String) throws {
    guard !data.isEmpty else {
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
    let outputURL = URL(fileURLWithPath: path).standardizedFileURL
    try Qwen38MTPScorecardFreshOutputSet.validateFreshPath(outputURL.path)
    let manager = FileManager.default
    let parent = outputURL.deletingLastPathComponent()
    let temporaryURL = parent.appendingPathComponent(
        ".\(outputURL.lastPathComponent).\(UUID().uuidString).tmp")
    try data.write(to: temporaryURL, options: .withoutOverwriting)
    do {
        try manager.linkItem(at: temporaryURL, to: outputURL)
        try manager.removeItem(at: temporaryURL)
    } catch CocoaError.fileWriteFileExists {
        try? manager.removeItem(at: temporaryURL)
        throw Qwen38MTPScorecardLiveAdapterError.outputExists
    } catch {
        try? manager.removeItem(at: temporaryURL)
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
}

func writeFreshQwen38MTPScorecardOutputSet(
    scorecardData: Data,
    authorityData: Data,
    outputs: Qwen38MTPScorecardFreshOutputSet,
    fsyncParents: ([URL]) throws -> Void = fsyncQwen38MTPScorecardParentDirectories
) throws {
    guard !scorecardData.isEmpty, scorecardData.last == 0x0a,
        scorecardData.dropLast().firstIndex(of: 0x0a) == nil,
        !authorityData.isEmpty
    else {
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
    try Qwen38MTPScorecardFreshOutputSet.validateFreshPath(outputs.scorecardPath)
    try Qwen38MTPScorecardFreshOutputSet.validateFreshPath(outputs.authorityPath)

    let scorecardURL = URL(fileURLWithPath: outputs.scorecardPath).standardizedFileURL
    let authorityURL = URL(fileURLWithPath: outputs.authorityPath).standardizedFileURL
    let scorecardTemporaryURL = try writeQwen38MTPScorecardTemporaryFile(
        scorecardData,
        near: scorecardURL)
    let authorityTemporaryURL = try writeQwen38MTPScorecardTemporaryFile(
        authorityData,
        near: authorityURL)
    var scorecardLinked = false
    var authorityLinked = false
    defer {
        _ = scorecardTemporaryURL.withUnsafeFileSystemRepresentation { pointer in
            pointer.map(unlink) ?? -1
        }
        _ = authorityTemporaryURL.withUnsafeFileSystemRepresentation { pointer in
            pointer.map(unlink) ?? -1
        }
    }

    do {
        try linkQwen38MTPScorecardTemporaryFile(scorecardTemporaryURL, to: scorecardURL)
        scorecardLinked = true
        try linkQwen38MTPScorecardTemporaryFile(authorityTemporaryURL, to: authorityURL)
        authorityLinked = true
        try fsyncParents(
            [scorecardURL.deletingLastPathComponent(), authorityURL.deletingLastPathComponent()])
    } catch {
        if authorityLinked {
            _ = authorityURL.withUnsafeFileSystemRepresentation { pointer in
                pointer.map(unlink) ?? -1
            }
        }
        if scorecardLinked {
            _ = scorecardURL.withUnsafeFileSystemRepresentation { pointer in
                pointer.map(unlink) ?? -1
            }
        }
        throw error
    }
}

private func writeQwen38MTPScorecardTemporaryFile(
    _ data: Data,
    near outputURL: URL
) throws -> URL {
    let temporaryURL = outputURL.deletingLastPathComponent().appendingPathComponent(
        ".\(outputURL.lastPathComponent).\(UUID().uuidString).tmp")
    let descriptor = temporaryURL.withUnsafeFileSystemRepresentation { pointer -> Int32 in
        guard let pointer else { return -1 }
        return open(pointer, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
    }
    guard descriptor >= 0 else {
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
    var descriptorOpen = true
    var removeTemporaryOnFailure = true
    defer {
        if descriptorOpen { _ = close(descriptor) }
        if removeTemporaryOnFailure {
            _ = temporaryURL.withUnsafeFileSystemRepresentation { pointer in
                pointer.map(unlink) ?? -1
            }
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
    guard wroteAll, fsync(descriptor) == 0 else {
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
    guard close(descriptor) == 0 else {
        descriptorOpen = false
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
    descriptorOpen = false
    removeTemporaryOnFailure = false
    return temporaryURL
}

private func linkQwen38MTPScorecardTemporaryFile(_ temporaryURL: URL, to outputURL: URL) throws {
    let linked = temporaryURL.withUnsafeFileSystemRepresentation { temporaryPointer -> Int32 in
        outputURL.withUnsafeFileSystemRepresentation { outputPointer -> Int32 in
            guard let temporaryPointer, let outputPointer else { return -1 }
            return Darwin.link(temporaryPointer, outputPointer)
        }
    }
    guard linked == 0 else {
        if errno == EEXIST {
            throw Qwen38MTPScorecardLiveAdapterError.outputExists
        }
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
}

private func fsyncQwen38MTPScorecardParentDirectories(_ directories: [URL]) throws {
    var seen: Set<String> = []
    for directory in directories where seen.insert(directory.path).inserted {
        let descriptor = directory.withUnsafeFileSystemRepresentation { pointer -> Int32 in
            guard let pointer else { return -1 }
            return open(pointer, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        }
        guard descriptor >= 0 else {
            throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
        }
        var descriptorOpen = true
        defer {
            if descriptorOpen { _ = close(descriptor) }
        }
        guard fsync(descriptor) == 0 else {
            throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
        }
        guard close(descriptor) == 0 else {
            descriptorOpen = false
            throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
        }
        descriptorOpen = false
    }
}

func qwen38MTPScorecardLiveAdapterExternalDiagnostic(_ error: Error) -> String {
    if let error = error as? Qwen38MTPScorecardLiveAdapterError {
        return error.description
    }
    if error is CancellationError {
        return "scorecard production cancelled"
    }
    return "scorecard production failed"
}

func qwen38MTPScorecardCanonicalJSON<T: Encodable>(_ value: T) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return try encoder.encode(value)
}

func qwen38MTPScorecardSHA256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func qwen38MTPScorecardMeasuredWiredLimitMB() throws -> UInt64 {
    var size = 0
    guard sysctlbyname("iogpu.wired_limit_mb", nil, &size, nil, 0) == 0,
        size > 0
    else {
        throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
    }
    if size == MemoryLayout<Int32>.size {
        var value: Int32 = 0
        guard sysctlbyname("iogpu.wired_limit_mb", &value, &size, nil, 0) == 0,
            value >= 0
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        return UInt64(value)
    }
    if size == MemoryLayout<Int64>.size {
        var value: Int64 = 0
        guard sysctlbyname("iogpu.wired_limit_mb", &value, &size, nil, 0) == 0,
            value >= 0
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        return UInt64(value)
    }
    throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
}

private func qwen38MTPScorecardCheckedSum(_ values: [UInt64]) throws -> UInt64 {
    var total: UInt64 = 0
    for value in values {
        let added = total.addingReportingOverflow(value)
        guard !added.overflow else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        total = added.partialValue
    }
    return total
}
