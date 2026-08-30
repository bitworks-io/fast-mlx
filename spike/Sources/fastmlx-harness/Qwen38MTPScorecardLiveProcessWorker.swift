import CryptoKit
import Darwin
import Foundation
import HarnessCore
import HuggingFace
import MLX
import MLXHuggingFace
@_spi(FastMLXExactMTP) import MLXLLM
import MLXLMCommon
import SpikeCore
import Tokenizers

private let qwen38MTPScorecardGDNEnvironmentKey = "MLX_QWEN_FOUR_GDN"

protocol Qwen38MTPScorecardChildProcess: AnyObject {
    var isRunning: Bool { get }
    func terminate()
    func interrupt()
    func forceKill()
}

extension Process: Qwen38MTPScorecardChildProcess {
    func forceKill() {
        guard isRunning else { return }
        _ = Darwin.kill(processIdentifier, SIGKILL)
    }
}

func qwen38MTPScorecardTerminateChild(
    _ child: Qwen38MTPScorecardChildProcess,
    deadlineChecks: Int,
    pollIntervalMicroseconds: useconds_t = 100_000
) {
    guard child.isRunning else { return }
    child.terminate()
    for _ in 0 ..< max(0, deadlineChecks) {
        guard child.isRunning else { return }
        if pollIntervalMicroseconds > 0 {
            usleep(pollIntervalMicroseconds)
        }
    }
    guard child.isRunning else { return }
    child.interrupt()
    for _ in 0 ..< max(0, deadlineChecks) {
        guard child.isRunning else { return }
        if pollIntervalMicroseconds > 0 {
            usleep(pollIntervalMicroseconds)
        }
    }
    guard child.isRunning else { return }
    child.forceKill()
}

actor Qwen38MTPScorecardProcessLineTransport: Qwen38MTPScorecardLineTransport {
    private let process: Process
    private let role: Qwen38MTPPerformanceScorecardEngineRole
    private let stdin: FileHandle
    private let stdout: FileHandle
    private let stderr: FileHandle
    private let stderrDrain: Task<Void, Never>
    private let stderrSink: Qwen38MTPScorecardBoundedByteSink
    private var buffer = Data()

    init(
        role: Qwen38MTPPerformanceScorecardEngineRole,
        arguments: Qwen38MTPScorecardLiveRunArguments
    ) throws {
        guard let executableURL = Bundle.main.executableURL else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
        let input = Pipe()
        let output = Pipe()
        let error = Pipe()
        let process = Process()
        process.executableURL = executableURL
        process.arguments = [
            "qwen38-mtp-scorecard-worker",
            "--role", role.rawValue,
            "--target", arguments.targetPath,
            "--drafter", arguments.drafterPath,
            "--host-use", arguments.hostUse,
            "--host-use-source", arguments.hostUseSource,
            "--memory-limit-bytes", "\(arguments.memoryBudget.memoryLimitBytes)",
            "--cache-limit-bytes", "\(arguments.memoryBudget.cacheLimitBytes)",
            "--reserved-kv-bytes", "\(arguments.memoryBudget.reservedKVBytes)",
            "--reserved-io-bytes", "\(arguments.memoryBudget.reservedIOBytes)",
            "--reserved-prefetch-bytes", "\(arguments.memoryBudget.reservedPrefetchBytes)",
            "--os-service-reserve-bytes", "\(arguments.memoryBudget.osServiceReserveBytes)",
        ]
        var environment = ProcessInfo.processInfo.environment
        switch role {
        case .candidate:
            environment[qwen38MTPScorecardGDNEnvironmentKey] = "1"
        case .reference:
            environment.removeValue(forKey: qwen38MTPScorecardGDNEnvironmentKey)
        }
        process.environment = environment
        process.standardInput = input
        process.standardOutput = output
        process.standardError = error
        try process.run()
        self.process = process
        self.role = role
        self.stdin = input.fileHandleForWriting
        self.stdout = output.fileHandleForReading
        self.stderr = error.fileHandleForReading
        let sink = Qwen38MTPScorecardBoundedByteSink(limit: 64 * 1024)
        self.stderrSink = sink
        let stderr = error.fileHandleForReading
        self.stderrDrain = Task.detached {
            await qwen38MTPScorecardDrainStderr(from: stderr, into: sink)
        }
    }

    func sendLine(_ line: String) async throws {
        guard process.isRunning else {
            throw Qwen38MTPScorecardLiveAdapterError.workerExited(role)
        }
        var data = Data(line.utf8)
        data.append(0x0a)
        try stdin.write(contentsOf: data)
    }

    func receiveLine() async throws -> String? {
        while true {
            if let newline = buffer.firstIndex(of: 0x0a) {
                let line = buffer[..<newline]
                buffer.removeSubrange(...newline)
                return String(data: Data(line), encoding: .utf8)
            }
            let chunk = stdout.availableData
            guard !chunk.isEmpty else { return nil }
            buffer.append(chunk)
        }
    }

    func terminate() async {
        try? stdin.close()
        qwen38MTPScorecardTerminateChild(process, deadlineChecks: 20)
        try? stdout.close()
        try? stderr.close()
        stderrDrain.cancel()
    }
}

actor Qwen38MTPScorecardBoundedByteSink {
    private let limit: Int
    private var retained = Data()
    private var droppedByteCount = 0

    init(limit: Int) {
        self.limit = max(0, limit)
    }

    func append(_ data: Data) {
        guard !data.isEmpty else { return }
        let remaining = max(0, limit - retained.count)
        if remaining > 0 {
            retained.append(data.prefix(remaining))
        }
        droppedByteCount += max(0, data.count - remaining)
    }

    func snapshot() -> (retained: Data, droppedByteCount: Int) {
        (retained, droppedByteCount)
    }
}

func qwen38MTPScorecardDrainStderr(
    from handle: FileHandle,
    into sink: Qwen38MTPScorecardBoundedByteSink
) async {
    while !Task.isCancelled {
        let data = handle.availableData
        guard !data.isEmpty else { break }
        await sink.append(data)
    }
}

func qwen38MTPScorecardMakeTransportPair<Transport: Qwen38MTPScorecardLineTransport>(
    makeCandidate: () throws -> Transport,
    makeReference: () throws -> Transport
) async throws -> (candidate: Transport, reference: Transport) {
    let candidate = try makeCandidate()
    do {
        return (candidate, try makeReference())
    } catch {
        await candidate.terminate()
        throw error
    }
}

func runQwen38MTPScorecardLiveAdapter(arguments: [String]) async throws -> String {
    #if DEBUG
    throw Qwen38MTPPerformanceScorecardProducerCLIError.producerUnavailable
    #else
    let parsed = try parseQwen38MTPScorecardLiveRunArguments(arguments)
    let outputs = try Qwen38MTPScorecardFreshOutputSet(
        scorecardPath: parsed.outputPath,
        authorityPath: parsed.authorityOutputPath)
    try parsed.memoryBudget.validateTwoWorkerAdmission(
        .live(),
        expectedChip: parsed.expectedChip)
    let transports = try await qwen38MTPScorecardMakeTransportPair(
        makeCandidate: {
            try Qwen38MTPScorecardProcessLineTransport(
                role: .candidate,
                arguments: parsed)
        },
        makeReference: {
            try Qwen38MTPScorecardProcessLineTransport(
                role: .reference,
                arguments: parsed)
        })
    let coordinator = Qwen38MTPScorecardLiveCoordinator(
        candidate: Qwen38MTPScorecardLineProtocolClient(
            role: .candidate,
            transport: transports.candidate),
        reference: Qwen38MTPScorecardLineProtocolClient(
            role: .reference,
            transport: transports.reference),
        runIdentity: try Qwen38MTPScorecardProcessFacts.trustedRunIdentity(),
        provenance: try Qwen38MTPScorecardProcessFacts.scorecardProvenance(),
        releaseBuildObserved: true)
    let result = try await coordinator.run()
    let scorecardData = Data((try result.record.jsonLine() + "\n").utf8)
    let authorityData = try qwen38MTPScorecardCanonicalJSON(result.authority)
    try writeFreshQwen38MTPScorecardOutputSet(
        scorecardData: scorecardData,
        authorityData: authorityData,
        outputs: outputs)
    return "qwen38-mtp-scorecard-live-adapter: WROTE qualified="
        + String(result.record.payload.verdict.qualified)
    #endif
}

func runQwen38MTPScorecardWorker(arguments: [String]) async throws {
    let parsed = try parseQwen38MTPScorecardWorkerArguments(arguments)
    let service = try await Qwen38MTPScorecardLiveWorkerService(arguments: parsed)
    let input = FileHandle.standardInput
    let output = FileHandle.standardOutput
    var buffer = Data()
    while true {
        while buffer.firstIndex(of: 0x0a) == nil {
            let chunk = input.availableData
            guard !chunk.isEmpty else { return }
            buffer.append(chunk)
        }
        guard let newline = buffer.firstIndex(of: 0x0a) else { continue }
        let line = Data(buffer[..<newline])
        buffer.removeSubrange(...newline)
        let response = await service.handle(line)
        var out = Data((try response.jsonLine()).utf8)
        out.append(0x0a)
        try output.write(contentsOf: out)
    }
}

private struct Qwen38MTPScorecardWorkerArguments: Sendable {
    let role: Qwen38MTPPerformanceScorecardEngineRole
    let targetPath: String
    let drafterPath: String
    let hostUse: String
    let hostUseSource: String
    let memoryBudget: Qwen38MTPScorecardLiveMemoryBudget
}

private func parseQwen38MTPScorecardWorkerArguments(
    _ arguments: [String]
) throws -> Qwen38MTPScorecardWorkerArguments {
    let allowed = Set([
        "--role",
        "--target",
        "--drafter",
        "--host-use",
        "--host-use-source",
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
    guard let role = try Qwen38MTPPerformanceScorecardEngineRole(
        rawValue: require("--role"))
    else {
        throw Qwen38MTPScorecardLiveAdapterError.invalidWorkerRole(.candidate)
    }
    let hostUse = try require("--host-use")
    let hostUseSource = try require("--host-use-source")
    guard hostUse == "dedicated-serving",
        hostUseSource == "operator-assertion"
    else {
        throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
    }
    return Qwen38MTPScorecardWorkerArguments(
        role: role,
        targetPath: try require("--target"),
        drafterPath: try require("--drafter"),
        hostUse: hostUse,
        hostUseSource: hostUseSource,
        memoryBudget: try Qwen38MTPScorecardLiveMemoryBudget(
            memoryLimitBytes: requireUInt64("--memory-limit-bytes"),
            cacheLimitBytes: requireUInt64("--cache-limit-bytes"),
            reservedKVBytes: requireUInt64("--reserved-kv-bytes"),
            reservedIOBytes: requireUInt64("--reserved-io-bytes"),
            reservedPrefetchBytes: requireUInt64("--reserved-prefetch-bytes"),
            osServiceReserveBytes: requireUInt64("--os-service-reserve-bytes")))
}

private final class Qwen38MTPScorecardLiveWorkerService: @unchecked Sendable {
    private let arguments: Qwen38MTPScorecardWorkerArguments
    private let pair: Qwen35ExactMTPLoadedPair
    private let handshake: Qwen38MTPScorecardWorkerHandshake
    private let memoryBudget: Qwen38MTPScorecardLiveMemoryBudget
    private let chatTemplateSHA256: String
    private let targetModelDirectory: URL

    init(arguments: Qwen38MTPScorecardWorkerArguments) async throws {
        #if DEBUG
        throw Qwen38MTPPerformanceScorecardProducerCLIError.producerUnavailable
        #else
        self.arguments = arguments
        self.memoryBudget = arguments.memoryBudget
        try arguments.memoryBudget.validateAgainstHost(.live())
        Memory.memoryLimit = Int(arguments.memoryBudget.memoryLimitBytes)
        Memory.cacheLimit = Int(arguments.memoryBudget.cacheLimitBytes)
        guard Memory.memoryLimit == Int(arguments.memoryBudget.memoryLimitBytes),
            Memory.cacheLimit == Int(arguments.memoryBudget.cacheLimitBytes)
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        let mode: Qwen38MTPPerformanceScorecardGDNMode =
            arguments.role == .candidate ? .gdnOn : .gdnOff
        let observedEnv = Qwen38MTPScorecardProcessFacts.observedGDNEnv(mode: mode)
        guard (arguments.role == .candidate && observedEnv == .enabled)
            || (arguments.role == .reference && observedEnv == .disabled)
        else {
            throw Qwen38MTPScorecardLiveAdapterError.workerEnvDrift(arguments.role)
        }
        let processIsolation = try Qwen38MTPScorecardProcessFacts.processIsolation(
            mode: mode,
            observedEnv: observedEnv)
        let launchBinding = Qwen38MTPScorecardProcessFacts.launchBinding(
            mode: mode,
            observedEnv: observedEnv,
            processIsolation: processIsolation)
        let targetURL = URL(fileURLWithPath: arguments.targetPath, isDirectory: true)
        let templateSHA256 = try qwen38MTPScorecardChatTemplateSHA256(
            modelDirectory: targetURL)
        guard templateSHA256 == Qwen38MTPPerformanceScorecardGate.requiredWorkload.chatTemplateSHA256 else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
        self.chatTemplateSHA256 = templateSHA256
        self.targetModelDirectory = targetURL
        let downloader = Qwen38MTPScorecardLocalDownloader(
            target: targetURL,
            drafter: URL(fileURLWithPath: arguments.drafterPath, isDirectory: true))
        let loaded = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
            selection: .qwen38_27BMXFP8Depth1,
            from: downloader,
            using: #huggingFaceTokenizerLoader())
        guard try qwen38MTPScorecardChatTemplateSHA256(modelDirectory: targetURL) == templateSHA256 else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
        self.pair = loaded
        let label = arguments.role.rawValue
        self.handshake = Qwen38MTPScorecardWorkerHandshake(
            role: arguments.role,
            model: Qwen38MTPPerformanceScorecardModel(
                label: label,
                artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
                executionDigest: Qwen38MTPScorecardProcessFacts.executionDigest(
                    role: arguments.role,
                    processIsolation: processIsolation),
                sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
                gdnMode: mode,
                launchBinding: launchBinding),
            processIsolation: processIsolation,
            launchBinding: launchBinding)
        #endif
    }

    func handle(_ line: Data) async -> Qwen38MTPScorecardWorkerProtocolResponse {
        let request: Qwen38MTPScorecardWorkerProtocolRequest
        do {
            request = try JSONDecoder().decode(
                Qwen38MTPScorecardWorkerProtocolRequest.self,
                from: line)
            switch request.kind {
            case .handshake:
                return .init(sequence: request.sequence, kind: .handshake, handshake: handshake)
            case .assertReady:
                try assertStillSameProcess()
                return .init(sequence: request.sequence, kind: .ok)
            case .exactness:
                try assertStillSameProcess()
                guard arguments.role == .candidate else {
                    throw Qwen38MTPScorecardLiveAdapterError.invalidWorkerRole(arguments.role)
                }
                return .init(
                    sequence: request.sequence,
                    kind: .exactness,
                    exactnessRecord: try await makeLiveExactnessRecord())
            case .measure:
                try assertStillSameProcess()
                guard let command = request.measurement else {
                    throw Qwen38MTPScorecardLiveAdapterError.malformedWorkerResponse
                }
                return .init(
                    sequence: request.sequence,
                    kind: .measurement,
                    measurement: try await measure(command.request))
            case .shutdown:
                return .init(sequence: request.sequence, kind: .ok)
            }
        } catch {
            return Qwen38MTPScorecardWorkerProtocolResponse(
                sequence: (try? JSONDecoder().decode(
                    Qwen38MTPScorecardWorkerProtocolRequest.self,
                    from: line).sequence) ?? -1,
                kind: .error,
                error: "worker command failed")
        }
    }

    private func assertStillSameProcess() throws {
        let current = try Qwen38MTPScorecardProcessFacts.processIsolation(
            mode: handshake.processIsolation.gdnMode,
            observedEnv: Qwen38MTPScorecardProcessFacts.observedGDNEnv(
                mode: handshake.processIsolation.gdnMode))
        guard current.observedEnv == handshake.processIsolation.observedEnv else {
            throw Qwen38MTPScorecardLiveAdapterError.workerEnvDrift(arguments.role)
        }
        guard current.processID == handshake.processIsolation.processID,
            current.parentProcessID == handshake.processIsolation.parentProcessID,
            current.processStartUptimeNanoseconds
                == handshake.processIsolation.processStartUptimeNanoseconds,
            current.bootTimeUnixSeconds == handshake.processIsolation.bootTimeUnixSeconds,
            current.executableSHA256 == handshake.processIsolation.executableSHA256,
            current.harnessGitSHA == handshake.processIsolation.harnessGitSHA
        else {
            throw Qwen38MTPScorecardLiveAdapterError.workerRestarted(arguments.role)
        }
        guard try qwen38MTPScorecardChatTemplateSHA256(
            modelDirectory: targetModelDirectory) == chatTemplateSHA256
        else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
    }

    private func makeLiveExactnessRecord() async throws
        -> ResultRecord<Qwen38MTPLiveExactnessEvidence>
    {
        let cases = try qwen38MTPScorecardLiveExactnessCases().map { exactCase in
            let scalar = try runScalar(prompt: exactCase.prompt, maxTokens: exactCase.maxTokens)
            let mtp = try runMTP(prompt: exactCase.prompt, maxTokens: exactCase.maxTokens)
            guard scalar.tokens == mtp.tokens,
                mtp.proposedDraftTokens > 0,
                mtp.acceptedDraftTokens > 0,
                mtp.passthroughReason == nil
            else {
                throw Qwen38MTPScorecardLiveAdapterError.workerError
            }
            let decodedBytes = Data(pair.target.tokenizer.decode(tokenIds: scalar.tokens).utf8)
            return Qwen38MTPLiveExactnessCaseEvidence(
                id: exactCase.id,
                promptSHA256: exactCase.promptSHA256,
                maxTokens: exactCase.maxTokens,
                scalarTokenIDs: scalar.tokens,
                mtpTokenIDs: mtp.tokens,
                scalarDecodedUTF8Base64: decodedBytes.base64EncodedString(),
                mtpDecodedUTF8Base64: decodedBytes.base64EncodedString(),
                decodedUTF8SHA256: qwen38MTPScorecardSHA256Hex(decodedBytes),
                proposedDraftTokens: mtp.proposedDraftTokens,
                acceptedDraftTokens: mtp.acceptedDraftTokens,
                passthroughReason: mtp.passthroughReason,
                scalarCacheFingerprints: scalar.liveCacheFingerprints,
                mtpCacheFingerprints: mtp.liveCacheFingerprints)
        }
        let evidence = Qwen38MTPLiveExactnessEvidence(
            schemaVersion: Qwen38MTPLiveExactnessGate.schemaVersion,
            artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
            artifactID: Qwen38MTPLiveExactnessGate.requiredArtifactID,
            source: Qwen38MTPLiveExactnessGate.requiredSourceIdentity,
            gdnMode: .gdnOn,
            launchBinding: handshake.launchBinding,
            processIsolation: handshake.processIsolation,
            mlxMemoryBudget: memoryBudget.liveExactnessBudget,
            hostMemoryObservation: try Qwen38MTPScorecardProcessFacts.hostMemoryObservation(
                memoryBudget: memoryBudget,
                hostUse: arguments.hostUse,
                hostUseSource: arguments.hostUseSource,
                hostSnapshot: .live()),
            cases: cases)
        let provenance = try Qwen38MTPScorecardProcessFacts.liveExactnessProvenance()
        let record = ResultRecord(
            subcommand: Qwen38MTPLiveExactnessGate.subcommand,
            provenance: provenance,
            payload: evidence)
        _ = try Qwen38MTPLiveExactnessGate.validateJSONL(
            Data((try record.jsonLine() + "\n").utf8))
        return record
    }

    private func measure(
        _ request: Qwen38MTPPerformanceScorecardMeasurementRequest
    ) async throws -> Qwen38MTPPerformanceScorecardEngineMeasurement {
        guard request.role == arguments.role else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidWorkerRole(arguments.role)
        }
        let before = qwen38MTPScorecardThermalState()
        let wallStart = ProcessInfo.processInfo.systemUptime
        let results = try await withThrowingTaskGroup(
            of: Qwen38MTPScorecardGenerationResult.self
        ) { group in
            for (requestIndex, caseID) in request.schedule.caseIDs.enumerated() {
                guard let workloadCase = request.workload.cases.first(where: { $0.id == caseID })
                else {
                    throw Qwen38MTPScorecardLiveAdapterError.workerError
                }
                group.addTask {
                    switch request.role {
                    case .candidate:
                        return try self.runMTP(
                            prompt: workloadCase.prompt,
                            maxTokens: workloadCase.maxCompletionTokens,
                            requestIndex: requestIndex,
                            caseID: caseID)
                    case .reference:
                        return try self.runScalar(
                            prompt: workloadCase.prompt,
                            maxTokens: workloadCase.maxCompletionTokens,
                            requestIndex: requestIndex,
                            caseID: caseID)
                    }
                }
            }
            var values: [Qwen38MTPScorecardGenerationResult] = []
            for try await value in group {
                values.append(value)
            }
            return values.sorted { $0.requestIndex < $1.requestIndex }
        }
        let wallSeconds = max(ProcessInfo.processInfo.systemUptime - wallStart, 1e-9)
        let after = qwen38MTPScorecardThermalState()
        let proposals = results.reduce(0) { $0 + $1.proposedDraftTokens }
        let accepted = results.reduce(0) { $0 + $1.acceptedDraftTokens }
        guard request.role == .reference ? proposals == 0 && accepted == 0 : proposals > 0 && accepted > 0
        else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
        return Qwen38MTPPerformanceScorecardEngineMeasurement(
            identity: request.identity,
            requests: results.map(\.requestMeasurement),
            wallSeconds: wallSeconds,
            peakRSSBytes: qwen38MTPScorecardPeakRSSBytes(),
            peakMetalBytes: UInt64(max(Memory.snapshot().peakMemory, 1)),
            thermalBefore: before,
            thermalAfter: after,
            proposalCount: proposals,
            acceptedCount: accepted,
            fallbackUsed: false,
            passthroughUsed: results.contains { $0.passthroughReason != nil })
    }

    private func runScalar(
        prompt: String,
        maxTokens: Int,
        requestIndex: Int = 0,
        caseID: String = ""
    ) throws -> Qwen38MTPScorecardGenerationResult {
        let promptStart = ProcessInfo.processInfo.systemUptime
        let promptTokens = try qwen38MTPScorecardPromptTokenIDs(
            prompt: prompt,
            tokenizer: pair.target.tokenizer)
        let promptEnd = ProcessInfo.processInfo.systemUptime
        let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        let cache = pair.target.model.newCache(parameters: parameters)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(promptTokens.compactMap(Int32.init(exactly:)))),
            model: pair.target.model,
            cache: cache,
            parameters: parameters)
        var tokens: [Int] = []
        var firstTokenTime: Double?
        let decodeStart = ProcessInfo.processInfo.systemUptime
        while let token = iterator.next(), tokens.count < maxTokens {
            firstTokenTime = firstTokenTime ?? ProcessInfo.processInfo.systemUptime
            tokens.append(token)
        }
        let end = ProcessInfo.processInfo.systemUptime
        let decodeTiming = qwen38MTPScorecardDecodeTiming(
            firstTokenTime: firstTokenTime,
            decodeStart: decodeStart,
            end: end,
            generatedTokenCount: tokens.count)
        return generationResult(
            caseID: caseID,
            requestIndex: requestIndex,
            tokens: tokens,
            cache: cache,
            promptSeconds: max(promptEnd - promptStart, 1e-9),
            prefillSeconds: max(iterator.promptPrefillTime, 1e-9),
            ttftSeconds: max((firstTokenTime ?? end) - promptStart, 1e-9),
            decodeTokenCount: decodeTiming.decodeTokenCount,
            decodeSeconds: decodeTiming.decodeSeconds,
            e2eSeconds: max(end - promptStart, 1e-9),
            proposedDraftTokens: 0,
            acceptedDraftTokens: 0,
            passthroughReason: nil)
    }

    private func runMTP(
        prompt: String,
        maxTokens: Int,
        requestIndex: Int = 0,
        caseID: String = ""
    ) throws -> Qwen38MTPScorecardGenerationResult {
        let promptStart = ProcessInfo.processInfo.systemUptime
        let promptTokens = try qwen38MTPScorecardPromptTokenIDs(
            prompt: prompt,
            tokenizer: pair.target.tokenizer)
        let promptEnd = ProcessInfo.processInfo.systemUptime
        let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        let cache = pair.target.model.newCache(parameters: parameters)
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(promptTokens.compactMap(Int32.init(exactly:)))),
            mainModel: pair.target.model,
            drafter: pair.drafter.model,
            mainCache: cache,
            parameters: parameters,
            blockSize: pair.binding.runtimeBlockSize,
            collectPhaseTelemetry: true)
        var tokens: [Int] = []
        var firstTokenTime: Double?
        let decodeStart = ProcessInfo.processInfo.systemUptime
        while let token = iterator.next(), tokens.count < maxTokens {
            firstTokenTime = firstTokenTime ?? ProcessInfo.processInfo.systemUptime
            tokens.append(token)
        }
        iterator.finalizeGeneration()
        let end = ProcessInfo.processInfo.systemUptime
        let telemetry = iterator.speculativeDecodingTelemetry
        let decodeTiming = qwen38MTPScorecardDecodeTiming(
            firstTokenTime: firstTokenTime,
            decodeStart: decodeStart,
            end: end,
            generatedTokenCount: tokens.count)
        return generationResult(
            caseID: caseID,
            requestIndex: requestIndex,
            tokens: tokens,
            cache: cache,
            promptSeconds: max(promptEnd - promptStart, 1e-9),
            prefillSeconds: max(iterator.promptPrefillTime, 1e-9),
            ttftSeconds: max((firstTokenTime ?? end) - promptStart, 1e-9),
            decodeTokenCount: decodeTiming.decodeTokenCount,
            decodeSeconds: decodeTiming.decodeSeconds,
            e2eSeconds: max(end - promptStart, 1e-9),
            proposedDraftTokens: telemetry?.draftTokenCount ?? iterator.proposedCount,
            acceptedDraftTokens: telemetry?.acceptedDraftTokenCount ?? iterator.acceptedCount,
            passthroughReason: iterator.passthroughReason)
    }

    private func generationResult(
        caseID: String,
        requestIndex: Int,
        tokens: [Int],
        cache: [KVCache],
        promptSeconds: Double,
        prefillSeconds: Double,
        ttftSeconds: Double,
        decodeTokenCount: Int,
        decodeSeconds: Double,
        e2eSeconds: Double,
        proposedDraftTokens: Int,
        acceptedDraftTokens: Int,
        passthroughReason: String?
    ) -> Qwen38MTPScorecardGenerationResult {
        let cacheFingerprints = qwen38MTPScorecardLiveCacheFingerprints(cache)
        let tokenDigest = qwen38MTPScorecardTokenDigest(tokens)
        let cacheDigest = qwen38MTPScorecardCacheDigest(cacheFingerprints)
        return Qwen38MTPScorecardGenerationResult(
            requestIndex: requestIndex,
            tokens: tokens,
            liveCacheFingerprints: cacheFingerprints,
            proposedDraftTokens: proposedDraftTokens,
            acceptedDraftTokens: acceptedDraftTokens,
            passthroughReason: passthroughReason,
            requestMeasurement: Qwen38MTPPerformanceScorecardRequestMeasurement(
                caseID: caseID,
                requestIndex: requestIndex,
                promptSeconds: promptSeconds,
                prefillSeconds: prefillSeconds,
                ttftSeconds: ttftSeconds,
                decodeTokenCount: decodeTokenCount,
                decodeSeconds: decodeSeconds,
                e2eSeconds: e2eSeconds,
                outputDigest: tokenDigest,
                cacheDigest: cacheDigest,
                outputProvenanceID: qwen38MTPScorecardSHA256Hex(
                    Data("output:\(caseID):\(requestIndex):\(tokenDigest)".utf8)),
                cacheProvenanceID: qwen38MTPScorecardSHA256Hex(
                    Data("cache:\(caseID):\(requestIndex):\(cacheDigest)".utf8))))
    }
}

private struct Qwen38MTPScorecardGenerationResult: Sendable {
    let requestIndex: Int
    let tokens: [Int]
    let liveCacheFingerprints: [Qwen38MTPLiveExactnessCacheFingerprint]
    let proposedDraftTokens: Int
    let acceptedDraftTokens: Int
    let passthroughReason: String?
    let requestMeasurement: Qwen38MTPPerformanceScorecardRequestMeasurement
}

private struct Qwen38MTPScorecardExactnessCase {
    let id: String
    let prompt: String
    let promptSHA256: String
    let maxTokens: Int
}

private func qwen38MTPScorecardLiveExactnessCases() -> [Qwen38MTPScorecardExactnessCase] {
    [
        ("numbers", "Continue the exact sequence with one concise answer: 2, 3, 5, 7, 11,", 12),
        ("sentence", "Complete this sentence in a few words: The fastest reliable test is", 12),
    ].map { id, prompt, maxTokens in
        Qwen38MTPScorecardExactnessCase(
            id: id,
            prompt: prompt,
            promptSHA256: Qwen38MTPLiveExactnessGate.requiredCasesByID[id]!.promptSHA256,
            maxTokens: maxTokens)
    }
}

private struct Qwen38MTPScorecardLocalDownloader: Downloader {
    let target: URL
    let drafter: URL

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let source = Qwen38MTPLiveExactnessGate.requiredSourceIdentity
        guard !useLatest, patterns == ["*.safetensors", "*.json", "*.jinja"] else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
        progressHandler(Progress(totalUnitCount: 1))
        switch (id, revision) {
        case (source.targetRepositoryID, source.targetRevision):
            return target
        case (source.drafterRepositoryID, source.drafterRevision):
            return drafter
        default:
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
    }
}

enum Qwen38MTPScorecardProcessFacts {
    static func observedGDNEnv(
        mode: Qwen38MTPPerformanceScorecardGDNMode,
        environmentValue: String? = ProcessInfo.processInfo.environment[
            qwen38MTPScorecardGDNEnvironmentKey]
    ) -> Qwen38MTPPerformanceScorecardGDNObservedEnv {
        switch mode {
        case .gdnOn:
            return environmentValue == "1" ? .enabled : .disabled
        case .gdnOff:
            return environmentValue == nil ? .disabled : .enabled
        }
    }

    static func processIsolation(
        mode: Qwen38MTPPerformanceScorecardGDNMode,
        observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv
    ) throws -> Qwen38MTPLiveExactnessProcessIsolationEvidence {
        let pid = getpid()
        return Qwen38MTPLiveExactnessProcessIsolationEvidence(
            processID: Int(pid),
            parentProcessID: Int(getppid()),
            processStartUptimeNanoseconds: try processStartUptimeNanoseconds(pid: pid),
            bootTimeUnixSeconds: try Int64(bootTimeUnixSeconds()),
            executableIdentitySource: .procPIDPath,
            executableSHA256: try executableSHA256(pid: pid),
            harnessGitSHA: try ProvenanceCLI.qualificationHarnessGitSHA(),
            sourceID: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
            gdnMode: mode,
            observedEnv: observedEnv)
    }

    static func launchBinding(
        mode: Qwen38MTPPerformanceScorecardGDNMode,
        observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv,
        processIsolation: Qwen38MTPLiveExactnessProcessIsolationEvidence
    ) -> Qwen38MTPPerformanceScorecardLaunchBinding {
        let processID = Qwen38MTPLiveExactnessGate.processIsolationEvidenceID(
            for: processIsolation)
        return Qwen38MTPPerformanceScorecardLaunchBinding(
            mode: mode,
            sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
            observedEnv: observedEnv,
            processIsolationEvidenceID: processID,
            launchDigest: Qwen38MTPPerformanceScorecardGate.launchDigest(
                mode: mode,
                sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
                observedEnv: observedEnv,
                processIsolationEvidenceID: processID))
    }

    static func executionDigest(
        role: Qwen38MTPPerformanceScorecardEngineRole,
        processIsolation: Qwen38MTPLiveExactnessProcessIsolationEvidence
    ) -> String {
        qwen38MTPScorecardSHA256Hex(
            Data("\(role.rawValue):\(processIsolation.processID):\(processIsolation.executableSHA256)".utf8))
    }

    static func trustedRunIdentity() throws -> Qwen38MTPPerformanceScorecardTrustedRunIdentity {
        Qwen38MTPPerformanceScorecardTrustedRunIdentity(
            measurementClass: Qwen38MTPPerformanceScorecardGate.measurementClass,
            hardwareChip: ProvenanceCLI.chipBrand(),
            hardwareRAMBytes: ProvenanceCLI.ramBytes(),
            hardwareOSBuild: ProvenanceCLI.osVersion(),
            hostIdentityDigest: qwen38MTPScorecardSHA256Hex(
                Data("\(ProvenanceCLI.chipBrand()):\(ProvenanceCLI.ramBytes()):\(ProvenanceCLI.osVersion())".utf8)),
            harnessGitSHA: try ProvenanceCLI.qualificationHarnessGitSHA(),
            candidateMLXSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelLabel: Qwen38MTPPerformanceScorecardGate.modelArtifactLabel,
            modelConfigHash:
                Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetConfigSHA256,
            modelCheckpointManifestHash:
                Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusID: Qwen38MTPPerformanceScorecardGate.requiredWorkload.id,
            corpusContentHash:
                Qwen38MTPPerformanceScorecardGate.requiredWorkload.contentSHA256)
    }

    static func scorecardProvenance() throws -> Provenance {
        let identity = try trustedRunIdentity()
        return Provenance(
            date: ISO8601DateFormatter().string(from: Date()),
            hardwareChip: identity.hardwareChip,
            hardwareRAMBytes: identity.hardwareRAMBytes,
            hardwareOS: identity.hardwareOSBuild,
            harnessGitSHA: identity.harnessGitSHA,
            mlxSwiftVersion: identity.candidateMLXSwiftVersion,
            referenceMLXVersion: identity.referenceMLXVersion,
            referenceMLXLMVersion: identity.referenceMLXLMVersion,
            modelPath: identity.modelLabel,
            modelConfigHash: identity.modelConfigHash,
            modelCheckpointManifestHash: identity.modelCheckpointManifestHash,
            modelQuant: identity.modelQuant,
            corpusId: identity.corpusID,
            corpusContentHash: identity.corpusContentHash,
            nonce: UUID().uuidString)
    }

    static func liveExactnessProvenance() throws -> Provenance {
        Provenance(
            date: ISO8601DateFormatter().string(from: Date()),
            hardwareChip: ProvenanceCLI.chipBrand(),
            hardwareRAMBytes: ProvenanceCLI.ramBytes(),
            hardwareOS: ProvenanceCLI.osVersion(),
            harnessGitSHA: try ProvenanceCLI.qualificationHarnessGitSHA(),
            mlxSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelPath: Qwen38MTPLiveExactnessGate.modelPathSentinel,
            modelConfigHash: Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetConfigSHA256,
            modelCheckpointManifestHash:
                Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusId: nil,
            corpusContentHash: nil,
            nonce: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID)
    }

    static func hostMemoryObservation(
        memoryBudget: Qwen38MTPScorecardLiveMemoryBudget,
        hostUse: String,
        hostUseSource: String,
        hostSnapshot: Qwen38MTPScorecardLiveHostMemorySnapshot
    ) throws -> Qwen38MTPLiveExactnessHostMemoryObservation {
        try memoryBudget.validateAgainstHost(hostSnapshot)
        guard let recommended = hostSnapshot.metalRecommendedMaxWorkingSetSizeBytes else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        guard hostUse == "dedicated-serving",
            hostUseSource == "operator-assertion"
        else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
        }
        return Qwen38MTPLiveExactnessHostMemoryObservation(
            hostUse: hostUse,
            hostUseSource: hostUseSource,
            hostUsePolicyVersion: Qwen38MTPLiveExactnessGate.requiredHostUsePolicyVersion,
            physicalRAMBytes: hostSnapshot.physicalRAMBytes,
            wiredLimitMB: Int(min(hostSnapshot.wiredLimitMB, UInt64(Int.max))),
            wiredLimitProvenance: .measured,
            metalRecommendedMaxWorkingSetSizeBytes: recommended,
            metalCurrentAllocatedSizeBytes: hostSnapshot.metalCurrentAllocatedSizeBytes,
            memoryLimitBytes: memoryBudget.memoryLimitBytes,
            cacheLimitBytes: memoryBudget.cacheLimitBytes,
            reservedKVBytes: memoryBudget.reservedKVBytes,
            reservedIOBytes: memoryBudget.reservedIOBytes,
            reservedPrefetchBytes: memoryBudget.reservedPrefetchBytes,
            osServiceReserveBytes: memoryBudget.osServiceReserveBytes)
    }

    private static func bootTimeUnixSeconds() throws -> Int {
        var value = timeval()
        var size = MemoryLayout<timeval>.stride
        var mib: [Int32] = [CTL_KERN, KERN_BOOTTIME]
        let result = mib.withUnsafeMutableBufferPointer {
            sysctl($0.baseAddress, u_int($0.count), &value, &size, nil, 0)
        }
        guard result == 0 else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
        return Int(value.tv_sec)
    }

    private static func processStartUptimeNanoseconds(pid: pid_t) throws -> UInt64 {
        var info = proc_bsdinfo()
        let result = proc_pidinfo(
            pid,
            PROC_PIDTBSDINFO,
            0,
            &info,
            Int32(MemoryLayout<proc_bsdinfo>.stride))
        guard result == Int32(MemoryLayout<proc_bsdinfo>.stride) else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
        let boot = try bootTimeUnixSeconds()
        let seconds = max(1, Int(info.pbi_start_tvsec) - boot)
        return UInt64(seconds) * 1_000_000_000 + UInt64(info.pbi_start_tvusec) * 1_000
    }

    private static func executableSHA256(pid: pid_t) throws -> String {
        var buffer = [CChar](repeating: 0, count: 16_384)
        let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
        guard length > 0 else {
            throw Qwen38MTPScorecardLiveAdapterError.workerError
        }
        let pathBytes = buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }
        let path = String(decoding: pathBytes, as: UTF8.self)
        return try qwen38MTPScorecardSHA256Hex(Data(contentsOf: URL(fileURLWithPath: path)))
    }
}

private func qwen38MTPScorecardLiveCacheFingerprints(
    _ cache: [KVCache]
) -> [Qwen38MTPLiveExactnessCacheFingerprint] {
    cache.enumerated().map { layerIndex, entry in
        var metaHasher = SHA256()
        for (index, value) in entry.metaState.enumerated() {
            qwen38MTPScorecardUpdate(&metaHasher, "meta=\(index)")
            qwen38MTPScorecardUpdate(&metaHasher, value)
        }
        return Qwen38MTPLiveExactnessCacheFingerprint(
            layerIndex: layerIndex,
            cacheType: (layerIndex + 1).isMultiple(of: 4)
                ? "dense-attention"
                : "recurrent-mamba",
            offset: entry.offset,
            metaStateSHA256: metaHasher.finalize().map { String(format: "%02x", $0) }.joined(),
            stateFingerprints: entry.state.enumerated().map { stateIndex, array in
                eval(array)
                let bytes = array.asData(access: .copy).data
                return Qwen38MTPLiveExactnessArrayFingerprint(
                    stateIndex: stateIndex,
                    shape: array.shape,
                    dtype: String(describing: array.dtype),
                    byteCount: bytes.count,
                    sha256: qwen38MTPScorecardSHA256Hex(bytes))
            })
    }
}

private func qwen38MTPScorecardCacheDigest(
    _ cache: [Qwen38MTPLiveExactnessCacheFingerprint]
) -> String {
    qwen38MTPScorecardSHA256Hex((try? qwen38MTPScorecardCanonicalJSON(cache)) ?? Data())
}

private func qwen38MTPScorecardTokenDigest(_ tokens: [Int]) -> String {
    var data = Data()
    for token in tokens {
        var value = Int64(token).littleEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    return qwen38MTPScorecardSHA256Hex(data)
}

func qwen38MTPScorecardPromptTokenIDs(
    prompt: String,
    tokenizer: any MLXLMCommon.Tokenizer
) throws -> [Int] {
    let tokens = try tokenizer.applyChatTemplate(
        messages: [["role": "user", "content": prompt]],
        tools: nil,
        additionalContext: ["enable_thinking": false])
    guard !tokens.isEmpty else {
        throw Qwen38MTPScorecardLiveAdapterError.workerError
    }
    return tokens
}

func qwen38MTPScorecardChatTemplateSHA256(
    modelDirectory: URL,
    readFile: (URL) throws -> Data = { try Data(contentsOf: $0) }
) throws -> String {
    let jinjaURL = modelDirectory.appendingPathComponent("chat_template.jinja")
    guard FileManager.default.fileExists(atPath: jinjaURL.path) else {
        throw Qwen38MTPScorecardLiveAdapterError.workerError
    }
    return try qwen38MTPScorecardSHA256Hex(readFile(jinjaURL))
}

struct Qwen38MTPScorecardDecodeTiming: Equatable, Sendable {
    let decodeTokenCount: Int
    let decodeSeconds: Double
}

func qwen38MTPScorecardDecodeTiming(
    firstTokenTime: Double?,
    decodeStart: Double,
    end: Double,
    generatedTokenCount: Int
) -> Qwen38MTPScorecardDecodeTiming {
    let start = firstTokenTime ?? decodeStart
    return Qwen38MTPScorecardDecodeTiming(
        decodeTokenCount: max(0, generatedTokenCount - (firstTokenTime == nil ? 0 : 1)),
        decodeSeconds: max(end - start, 1e-9))
}

private func qwen38MTPScorecardUpdate<T>(_ hasher: inout SHA256, _ value: T) {
    hasher.update(data: Data("\(value)".utf8))
    hasher.update(data: Data([0]))
}

func qwen38MTPScorecardPeakRSSBytes() -> UInt64 {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return 1 }
    return qwen38MTPScorecardPeakRSSBytes(
        maxResidentSetSize: usage.ru_maxrss)
}

func qwen38MTPScorecardPeakRSSBytes(maxResidentSetSize: Int) -> UInt64 {
    UInt64(max(maxResidentSetSize, 1))
}

private func qwen38MTPScorecardThermalState() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}
