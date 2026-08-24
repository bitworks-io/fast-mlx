import CryptoKit
import Foundation
import XCTest
import os

import HarnessCore
import MLX
import MLXHuggingFace
@_spi(FastMLXExactMTP) import MLXLLM
import MLXLMCommon
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters
import Tokenizers

final class LoadedExactQwen35MTPServingBackendIntegrationTests: XCTestCase {
    func testServingLatencyCheckpointHashMatchesCanonicalV2Contract() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }

        let config = Data("{\"model_type\":\"fixture\"}".utf8)
        let index = Data("{\"weight_map\":{\"layer\":\"model.safetensors\"}}".utf8)
        let shardName = "model.safetensors"
        let shard = Data([0x00, 0x01, 0x7f, 0xff])
        try config.write(to: directory.appendingPathComponent("config.json"))
        try index.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        try shard.write(to: directory.appendingPathComponent(shardName))

        var canonical = Data("fastmlx-checkpoint-content-manifest-v2\n".utf8)
        appendCanonicalLengthField(config, to: &canonical)
        appendCanonicalLengthField(index, to: &canonical)
        appendCanonicalLengthField(Data(shardName.utf8), to: &canonical)
        appendCanonicalUInt64(UInt64(shard.count), to: &canonical)
        canonical.append(shard)
        XCTAssertEqual(
            try checkpointContentSHA256(modelDirectory: directory),
            sha256Hex(canonical))

        try Data([0x02]).write(
            to: directory.appendingPathComponent("unindexed.safetensors"))
        XCTAssertThrowsError(
            try checkpointContentSHA256(modelDirectory: directory))
    }

    func testLoadedExactQwen35MTPBackendMatchesScalarControlsWhenConfigured() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FAST_MLX_QWEN35_RUN_LIVE_BACKEND"] == "1" else {
            throw XCTSkip("live exact Qwen3.5 backend proof requires FAST_MLX_QWEN35_RUN_LIVE_BACKEND=1")
        }
        guard
            environment["FAST_MLX_QWEN35_TARGET_SNAPSHOT"] != nil,
            environment["FAST_MLX_QWEN35_MTP_SNAPSHOT"] != nil
        else {
            throw XCTSkip("exact Qwen3.5 target/drafter snapshot paths are not configured")
        }

        try await runLoadedExactQwen35MTPBackendProof(environment: environment)
    }

    func testLoadedExactQwen35MTPBackendServingLatencyProfileWhenConfigured() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FAST_MLX_QWEN35_RUN_LIVE_BACKEND_PROFILE"] == "1" else {
            throw XCTSkip(
                "live exact Qwen3.5 serving latency profile requires FAST_MLX_QWEN35_RUN_LIVE_BACKEND_PROFILE=1")
        }
        guard isReleaseBuild() else {
            throw LoadedExactQwen35MTPBackendProofError.releaseBuildRequired
        }
        _ = try XCTUnwrap(environment["FAST_MLX_QWEN35_TARGET_SNAPSHOT"])
        _ = try XCTUnwrap(environment["FAST_MLX_QWEN35_MTP_SNAPSHOT"])
        let evidencePath = try XCTUnwrap(
            environment["FAST_MLX_QWEN35_SERVING_LATENCY_EVIDENCE"])
        let hostChip = try XCTUnwrap(environment["FAST_MLX_QWEN35_PROFILE_HOST_CHIP"])
        let assertedHarnessGitSHA = try XCTUnwrap(
            environment["FAST_MLX_QWEN35_PROFILE_HARNESS_GIT_SHA"])
        let harnessGitSHA = try verifiedHarnessGitSHA(
            asserted: assertedHarnessGitSHA)
        let acceptedCorpusPath = try XCTUnwrap(
            environment["FAST_MLX_QWEN35_ACCEPTED_CORPUS_EVIDENCE"])
        let lowerLevelProof = try validateAcceptedLowerLevelCorpus(
            at: acceptedCorpusPath)
        let evidenceURL = try prepareNewOrEmptyEvidenceURL(evidencePath)

        try await runLoadedExactQwen35MTPBackendServingLatencyProfile(
            environment: environment,
            evidenceURL: evidenceURL,
            hostChip: hostChip,
            harnessGitSHA: harnessGitSHA,
            lowerLevelProof: lowerLevelProof)
    }
}

private func runLoadedExactQwen35MTPBackendProof(
    environment: [String: String]
) async throws {
    let targetPath = try XCTUnwrap(environment["FAST_MLX_QWEN35_TARGET_SNAPSHOT"])
    let drafterPath = try XCTUnwrap(environment["FAST_MLX_QWEN35_MTP_SNAPSHOT"])
    let launchedModel = "loaded-exact-qwen35-mtp"
    let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
        from: BackendExactSnapshotDownloader(
            target: URL(fileURLWithPath: targetPath, isDirectory: true),
            drafter: URL(fileURLWithPath: drafterPath, isDirectory: true)),
        using: #huggingFaceTokenizerLoader())
    XCTAssertEqual(pair.binding.runtimeBlockSize, 3)
    XCTAssertEqual(pair.binding.maximumAcceptedDraftTokens, 2)

    let runner = try LoadedExactQwen35MTPForwardingRunner(pair: pair)
    let scalarFallback = SentinelScalarFallback()
    let codec = TokenizerBackedScalarServingTextCodec(tokenizer: pair.target.tokenizer)
    let backend = try ExactQwen35MTPServingBackend(
        launchedModel: launchedModel,
        enabled: true,
        runner: runner,
        scalarFallback: scalarFallback,
        scalarFallbackIsolation: .strictlySeparateRawTarget,
        codec: codec,
        configuration: .init(
            defaultMaximumCompletionTokens: 16,
            mailboxCapacity: .init(maxDeltas: 8, maxBytes: 64 * 1_024)))

    let messages = [
        OpenAIChatMessage(role: .system, text: "Continue sequences with only the next values."),
        OpenAIChatMessage(role: .user, text: "Continue exactly: 2, 3, 5, 7, 11,"),
    ]
    let promptTokens = try codec.render(
        messages: messages,
        tools: [],
        enableThinking: nil,
        reasoningEffort: nil)

    let normalControl = try await runScalarControl(
        pair: pair,
        promptTokens: promptTokens,
        maxTokens: 16,
        stop: [])
    let normal = try await collectBackend(
        try await backend.start(request(
            model: launchedModel,
            messages: messages,
            maxTokens: 16)))
    try await assertCompletedMTPRun(
        normal,
        runner: runner,
        scalarControl: normalControl,
        expectedFinishReason: .length)
    XCTAssertEqual(runner.snapshot().startCount, 1)
    await assertReservationReleased(backend)

    let normalTokens = try XCTUnwrap(normalControl.tokens)
    let stopText = try XCTUnwrap(
        normalTokens.dropFirst().lazy.compactMap { token -> String? in
            let decoded = pair.target.tokenizer.decode(
                tokenIds: [token], skipSpecialTokens: false)
            let candidate = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
            return candidate.isEmpty ? nil : candidate
        }.first)
    let stopControl = try await runScalarControl(
        pair: pair,
        promptTokens: promptTokens,
        maxTokens: 16,
        stop: [stopText])
    let stopped = try await collectBackend(
        try await backend.start(request(
            model: launchedModel,
            messages: messages,
            maxTokens: 16,
            stop: [stopText])))
    try await assertCompletedMTPRun(
        stopped,
        runner: runner,
        scalarControl: stopControl,
        expectedFinishReason: .stop)
    XCTAssertEqual(runner.snapshot().startCount, 2)
    await assertReservationReleased(backend)

    let cancelling = try await backend.start(request(
        model: launchedModel,
        messages: messages,
        maxTokens: 512))
    XCTAssertEqual(cancelling.route, .exactQwen35MTP)
    var cancelledText = ""
    while cancelledText.isEmpty {
        guard let delta = try await cancelling.mailbox.next() else {
            XCTFail("MTP backend completed before cancellation point")
            break
        }
        switch delta {
        case .text(let text):
            cancelledText += text
        case .completion:
            XCTFail("MTP backend completed before cancellation point")
        case .toolCalls:
            XCTFail("exact MTP backend should not publish tool calls in this proof")
        }
    }
    XCTAssertFalse(cancelledText.isEmpty)
    let cancellationAccepted = await cancelling.lease.cancel(.clientDisconnected)
    XCTAssertTrue(cancellationAccepted)
    await assertMailboxCancelled(cancelling.mailbox, reason: .clientDisconnected)
    await waitUntil {
        let snapshot = runner.snapshot()
        let backendSnapshot = await backend.snapshot()
        return snapshot.cancelledTaskCount >= 1
            && snapshot.finalizedTaskCount >= 3
            && backendSnapshot.activeMTPReservations == 0
    }
    let cancelSnapshot = runner.snapshot()
    XCTAssertEqual(cancelSnapshot.startCount, 3)
    let cancelInfo = try XCTUnwrap(cancelSnapshot.lastInfo)
    XCTAssertGreaterThan(cancelInfo.acceptedDraftTokens ?? 0, 0)
    XCTAssertEqual(cancelInfo.stopReason, .cancelled)
    XCTAssertGreaterThan(cancelInfo.generationTokenCount, 0)
    let cancelScalarControl = try await runScalarControl(
        pair: pair,
        promptTokens: promptTokens,
        maxTokens: cancelInfo.generationTokenCount,
        stop: [])
    XCTAssertTrue(cancelScalarControl.text.hasPrefix(cancelledText))
    XCTAssertEqual(cancelSnapshot.lastCache, cancelScalarControl.cache)
    XCTAssertEqual(scalarFallback.snapshot().startCount, 0)
    await backend.shutdown()
}

private func runLoadedExactQwen35MTPBackendServingLatencyProfile(
    environment: [String: String],
    evidenceURL: URL,
    hostChip: String,
    harnessGitSHA: String,
    lowerLevelProof: QwenMTPServingLatencyLowerLevelProof
) async throws {
    let targetPath = try XCTUnwrap(environment["FAST_MLX_QWEN35_TARGET_SNAPSHOT"])
    let drafterPath = try XCTUnwrap(environment["FAST_MLX_QWEN35_MTP_SNAPSHOT"])
    let checkpointIdentity = try servingLatencyCheckpointIdentity(
        targetPath: targetPath,
        drafterPath: drafterPath)
    guard checkpointIdentity == QwenMTPServingLatencyGate.requiredCheckpointIdentity else {
        throw LoadedExactQwen35MTPBackendProofError.checkpointIdentityMismatch
    }
    let launchedModel = "loaded-exact-qwen35-mtp"
    let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
        from: BackendExactSnapshotDownloader(
            target: URL(fileURLWithPath: targetPath, isDirectory: true),
            drafter: URL(fileURLWithPath: drafterPath, isDirectory: true)),
        using: #huggingFaceTokenizerLoader())
    XCTAssertEqual(pair.binding.runtimeBlockSize, 3)
    XCTAssertEqual(pair.binding.maximumAcceptedDraftTokens, 2)

    let runner = try LoadedExactQwen35MTPForwardingRunner(pair: pair)
    let scalarFallback = SentinelScalarFallback()
    let codec = TokenizerBackedScalarServingTextCodec(tokenizer: pair.target.tokenizer)
    let backend = try ExactQwen35MTPServingBackend(
        launchedModel: launchedModel,
        enabled: true,
        runner: runner,
        scalarFallback: scalarFallback,
        scalarFallbackIsolation: .strictlySeparateRawTarget,
        codec: codec,
        configuration: .init(
            defaultMaximumCompletionTokens: 128,
            mailboxCapacity: .init(maxDeltas: 128, maxBytes: 512 * 1_024)))

    let host = QwenMTPCorpusHostEvidence(
        chip: hostChip,
        ramBytes: ProcessInfo.processInfo.physicalMemory,
        os: ProcessInfo.processInfo.operatingSystemVersionString)
    let provenance = try servingLatencyProvenance(
        host: host,
        harnessGitSHA: harnessGitSHA,
        targetPath: targetPath)

    var samples: [QwenMTPServingLatencySample] = []
    do {
        for caseID in QwenMTPCorpusGate.profilePlan.caseIDs {
            let spec = try XCTUnwrap(
                QwenMTPCorpusGate.cases.first { $0.id == caseID })
            let messages = [OpenAIChatMessage(role: .user, text: spec.prompt)]
            for pairIndex in 0..<QwenMTPCorpusGate.profilePlan.totalPairsPerCase {
                let order = QwenMTPCorpusGate.profilePlan.orders[pairIndex]
                let scalar: TimedControlRun
                let mtp: TimedBackendResult
                switch order {
                case .scalarThenMTP:
                    scalar = try await runTimedScalarControl(
                        pair: pair,
                        codec: codec,
                        messages: messages,
                        maxTokens: spec.maxTokens)
                    mtp = try await runTimedBackend(
                        backend: backend,
                        model: launchedModel,
                        messages: messages,
                        maxTokens: spec.maxTokens)
                case .mtpThenScalar:
                    mtp = try await runTimedBackend(
                        backend: backend,
                        model: launchedModel,
                        messages: messages,
                        maxTokens: spec.maxTokens)
                    scalar = try await runTimedScalarControl(
                        pair: pair,
                        codec: codec,
                        messages: messages,
                        maxTokens: spec.maxTokens)
                }

                let runnerSnapshot = runner.snapshot()
                let info = try XCTUnwrap(runnerSnapshot.lastInfo)
                let mtpCache = try XCTUnwrap(runnerSnapshot.lastCache)
                XCTAssertEqual(mtp.completion.usage.promptTokens, scalar.promptTokens)
                XCTAssertEqual(mtp.completion.usage.completionTokens, scalar.tokens.count)
                XCTAssertEqual(info.generationTokenCount, scalar.tokens.count)
                XCTAssertEqual(mtp.text, scalar.text)
                XCTAssertEqual(mtpCache, scalar.cache)
                XCTAssertEqual(scalarFallback.snapshot().startCount, 0)
                await assertReservationReleased(backend)
                let mtpTokens = pair.target.tokenizer.encode(
                    text: mtp.text,
                    addSpecialTokens: false)
                XCTAssertEqual(mtpTokens, scalar.tokens)

                let exactness = QwenMTPServingLatencyExactnessEvidence(
                    tokenObservationMode: .decodedRoundTrip,
                    scalarDirectTokenCount: scalar.tokens.count,
                    mtpUsageCompletionTokenCount: mtp.completion.usage.completionTokens,
                    scalarDirectTokenIDsSHA256: sha256Hex(scalar.tokens),
                    mtpDecodedRoundTripTokenIDsSHA256: sha256Hex(mtpTokens),
                    scalarDecodedBytesSHA256: sha256Hex(Data(scalar.text.utf8)),
                    mtpDecodedBytesSHA256: sha256Hex(Data(mtp.text.utf8)),
                    scalarStopOutcome: stopOutcome(scalar.info.stopReason),
                    mtpStopOutcome: stopOutcome(info.stopReason),
                    scalarCacheFingerprint: scalar.cache.qwenFingerprint(),
                    mtpCacheFingerprint: mtpCache.qwenFingerprint(),
                    firstCacheMismatch: nil)
                let scalarUsage = QwenMTPServingLatencyUsageEvidence(
                    promptTokens: scalar.promptTokens,
                    completionTokens: scalar.tokens.count,
                    totalTokens: scalar.promptTokens + scalar.tokens.count)
                let mtpUsage = QwenMTPServingLatencyUsageEvidence(
                    promptTokens: mtp.completion.usage.promptTokens,
                    completionTokens: mtp.completion.usage.completionTokens,
                    totalTokens: mtp.completion.usage.totalTokens)
                let telemetry = info.speculativeDecodingTelemetry
                let scalarTPS = Double(scalar.tokens.count) / scalar.e2eSeconds
                let mtpTPS = Double(mtpTokens.count) / mtp.e2eSeconds
                samples.append(.init(
                    caseID: caseID,
                    pairIndex: pairIndex,
                    warmup: pairIndex < QwenMTPCorpusGate.profilePlan.droppedWarmupPairs,
                    order: order,
                    route: .init(
                        kind: mtp.route == .exactQwen35MTP
                            ? .exactQwen35MTP
                            : .scalarFallback),
                    fallback: .init(
                        scalarFallbackStartCount: scalarFallback.snapshot().startCount),
                    exactness: exactness,
                    scalarUsage: scalarUsage,
                    mtpUsage: mtpUsage,
                    scalarE2ESeconds: scalar.e2eSeconds,
                    mtpE2ESeconds: mtp.e2eSeconds,
                    scalarTokensPerSecond: scalarTPS,
                    mtpTokensPerSecond: mtpTPS,
                    e2eRatio: scalar.e2eSeconds / mtp.e2eSeconds,
                    mtpTelemetry: .init(
                        proposedDraftTokens: telemetry?.draftTokenCount
                            ?? info.proposedDraftTokens ?? 0,
                        acceptedDraftTokens: telemetry?.acceptedDraftTokenCount
                            ?? info.acceptedDraftTokens ?? 0,
                        rejectedDraftTokens: telemetry?.rejectedDraftTokenCount
                            ?? max(
                                0,
                                (info.proposedDraftTokens ?? 0)
                                    - (info.acceptedDraftTokens ?? 0)),
                        roundCount: telemetry?.roundCount ?? 0,
                        targetModelCallCount: telemetry?.targetModelCallCount ?? 0,
                        draftModelCallCount: telemetry?.draftModelCallCount ?? 0,
                        targetVerifiedTokenCount: telemetry?.targetVerifiedTokenCount ?? 0,
                        emittedTokenCount: telemetry?.emittedTokenCount
                            ?? info.generationTokenCount),
                    passthroughReason: info.passthroughReason))
            }
        }

        let candidate = QwenMTPServingLatencyEvidence(
            schemaVersion: QwenMTPServingLatencyGate.schemaVersion,
            corpusID: QwenMTPCorpusGate.corpusID,
            corpusContentHash: QwenMTPCorpusGate.corpusContentHash,
            binding: QwenMTPCorpusGate.requiredBinding,
            lowerLevelProof: lowerLevelProof,
            checkpointIdentity: checkpointIdentity,
            measurementClass: QwenMTPServingLatencyGate.measurementClass,
            host: host,
            profilePlan: QwenMTPCorpusGate.profilePlan,
            releaseBuildRequired: true,
            releaseBuildObserved: isReleaseBuild(),
            request: .init(
                temperature: 0,
                topP: nil,
                topK: nil,
                minP: nil,
                seed: nil,
                toolsEmpty: true,
                penaltiesDisabled: true),
            samples: samples,
            verdict: nil)
        let verdict: QwenMTPServingLatencyVerdict
        do {
            verdict = try QwenMTPServingLatencyGate.evaluateCandidate(candidate)
        } catch {
            let rejectedURL = try prepareNewOrEmptyEvidenceURL(
                evidenceURL.appendingPathExtension("rejected").path)
            try RequiredJSONLWriter.append(
                ResultRecord(
                    subcommand: QwenMTPServingLatencyGate.rejectedSubcommand,
                    provenance: provenance,
                    payload: candidate),
                to: rejectedURL)
            throw error
        }
        let accepted = QwenMTPServingLatencyEvidence(
            schemaVersion: QwenMTPServingLatencyGate.schemaVersion,
            corpusID: QwenMTPCorpusGate.corpusID,
            corpusContentHash: QwenMTPCorpusGate.corpusContentHash,
            binding: QwenMTPCorpusGate.requiredBinding,
            lowerLevelProof: lowerLevelProof,
            checkpointIdentity: checkpointIdentity,
            measurementClass: QwenMTPServingLatencyGate.measurementClass,
            host: host,
            profilePlan: QwenMTPCorpusGate.profilePlan,
            releaseBuildRequired: true,
            releaseBuildObserved: isReleaseBuild(),
            request: .init(
                temperature: 0,
                topP: nil,
                topK: nil,
                minP: nil,
                seed: nil,
                toolsEmpty: true,
                penaltiesDisabled: true),
            samples: samples,
            verdict: verdict)
        do {
            _ = try QwenMTPServingLatencyGate.validate(accepted)
        } catch {
            let rejectedURL = try prepareNewOrEmptyEvidenceURL(
                evidenceURL.appendingPathExtension("rejected").path)
            try RequiredJSONLWriter.append(
                ResultRecord(
                    subcommand: QwenMTPServingLatencyGate.rejectedSubcommand,
                    provenance: provenance,
                    payload: accepted),
                to: rejectedURL)
            throw error
        }
        let record: ResultRecord<QwenMTPServingLatencyEvidence> = ResultRecord(
            subcommand: QwenMTPServingLatencyGate.subcommand,
            provenance: provenance,
            payload: accepted)
        let serializedRecord = Data((try record.jsonLine() + "\n").utf8)
        _ = try QwenMTPServingLatencyGate.validateJSONL(serializedRecord)
        try RequiredJSONLWriter.append(record, to: evidenceURL)
    } catch {
        await backend.shutdown()
        throw error
    }
    await backend.shutdown()
}

private func request(
    model: String,
    messages: [OpenAIChatMessage],
    maxTokens: Int,
    stop: [String] = []
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: model,
        messages: messages,
        maxCompletionTokens: maxTokens,
        temperature: 0,
        choiceCount: 1,
        stream: true,
        stop: stop)
}

private struct BackendResult {
    let text: String
    let completion: ServingGenerationCompletion
}

private func collectBackend(
    _ handle: ServingGenerationHandle
) async throws -> BackendResult {
    XCTAssertEqual(handle.route, .exactQwen35MTP)
    var text = ""
    var completion: ServingGenerationCompletion?
    while let delta = try await handle.mailbox.next() {
        switch delta {
        case .text(let value):
            text += value
        case .toolCalls:
            XCTFail("exact MTP backend should not publish tool calls in this proof")
        case .completion(let value):
            completion = value
        }
    }
    return BackendResult(text: text, completion: try XCTUnwrap(completion))
}

private func assertCompletedMTPRun(
    _ result: BackendResult,
    runner: LoadedExactQwen35MTPForwardingRunner,
    scalarControl: ControlRun,
    expectedFinishReason: OpenAIChatFinishReason,
    file: StaticString = #filePath,
    line: UInt = #line
) async throws {
    XCTAssertEqual(result.text, scalarControl.text, file: file, line: line)
    XCTAssertEqual(Data(result.text.utf8), Data(scalarControl.text.utf8), file: file, line: line)
    XCTAssertEqual(result.completion.finishReason, expectedFinishReason, file: file, line: line)
    XCTAssertEqual(result.completion.usage.promptTokens, scalarControl.promptTokens, file: file, line: line)
    XCTAssertEqual(
        result.completion.usage.completionTokens,
        scalarControl.completionTokenCount,
        file: file,
        line: line)
    let snapshot = runner.snapshot()
    XCTAssertEqual(
        snapshot.lastInfo?.generationTokenCount,
        scalarControl.completionTokenCount,
        file: file,
        line: line)
    XCTAssertEqual(snapshot.lastInfo?.stopReason, expectedFinishReason == .length ? .length : .stop, file: file, line: line)
    XCTAssertGreaterThan(snapshot.lastInfo?.proposedDraftTokens ?? 0, 0, file: file, line: line)
    XCTAssertGreaterThan(snapshot.lastInfo?.acceptedDraftTokens ?? 0, 0, file: file, line: line)
    XCTAssertNil(snapshot.lastInfo?.passthroughReason, file: file, line: line)
    XCTAssertEqual(snapshot.lastCache, scalarControl.cache, file: file, line: line)
}

private struct ControlRun: Sendable {
    let promptTokens: Int
    let tokens: [Int]?
    let completionTokenCount: Int
    let text: String
    let info: GenerateCompletionInfo
    let cache: CacheFingerprint
}

private struct TimedControlRun: Sendable {
    let promptTokens: Int
    let tokens: [Int]
    let text: String
    let info: GenerateCompletionInfo
    let cache: CacheFingerprint
    let e2eSeconds: Double
}

private struct TimedBackendResult {
    let text: String
    let completion: ServingGenerationCompletion
    let route: ServingExecutionRoute
    let e2eSeconds: Double
}

private func runTimedScalarControl(
    pair: Qwen35ExactMTPLoadedPair,
    codec: TokenizerBackedScalarServingTextCodec,
    messages: [OpenAIChatMessage],
    maxTokens: Int
) async throws -> TimedControlRun {
    let context = pair.target
    let parameters = greedyParameters(maxTokens: maxTokens)
    let clock = ContinuousClock()
    let start = clock.now
    let promptTokens = try codec.render(
        messages: messages,
        tools: [],
        enableThinking: nil,
        reasoningEffort: nil)
    let cache = context.model.newCache(parameters: parameters)
    let (stream, task) = try generateTokensTask(
        input: LMInput(tokens: MLXArray(promptTokens)),
        cache: cache,
        parameters: parameters,
        context: context)
    var tokens: [Int] = []
    var info: GenerateCompletionInfo?
    for await event in stream {
        if let token = event.token {
            tokens.append(token)
        }
        if let completion = event.info {
            info = completion
        }
    }
    await task.value
    let unwrappedInfo = try XCTUnwrap(info)
    let text = context.tokenizer.decode(
        tokenIds: tokens,
        skipSpecialTokens: false)
    let end = clock.now
    return TimedControlRun(
        promptTokens: promptTokens.count,
        tokens: tokens,
        text: text,
        info: unwrappedInfo,
        cache: CacheFingerprint(cache),
        e2eSeconds: seconds(from: start.duration(to: end)))
}

private func runTimedBackend(
    backend: ExactQwen35MTPServingBackend,
    model: String,
    messages: [OpenAIChatMessage],
    maxTokens: Int
) async throws -> TimedBackendResult {
    let clock = ContinuousClock()
    let start = clock.now
    let handle = try await backend.start(request(
        model: model,
        messages: messages,
        maxTokens: maxTokens))
    XCTAssertEqual(handle.route, .exactQwen35MTP)
    var text = ""
    var completion: ServingGenerationCompletion?
    var completionEnd: ContinuousClock.Instant?
    while let delta = try await handle.mailbox.next() {
        switch delta {
        case .text(let value):
            text += value
        case .toolCalls:
            XCTFail("exact MTP backend should not publish tool calls in this proof")
        case .completion(let value):
            completion = value
            completionEnd = clock.now
        }
    }
    return TimedBackendResult(
        text: text,
        completion: try XCTUnwrap(completion),
        route: handle.route,
        e2eSeconds: seconds(from: start.duration(to: try XCTUnwrap(completionEnd))))
}

private func runScalarControl(
    pair: Qwen35ExactMTPLoadedPair,
    promptTokens: [Int],
    maxTokens: Int,
    stop: Set<String>
) async throws -> ControlRun {
    var context = pair.target
    context.configuration.stopStrings = context.configuration.effectiveStopStrings.union(stop)
    let parameters = greedyParameters(maxTokens: maxTokens)
    let cache = context.model.newCache(parameters: parameters)
    if !stop.isEmpty {
        let iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(promptTokens)),
            model: context.model,
            cache: cache,
            parameters: parameters)
        let (stream, task) = generateTask(
            promptTokenCount: promptTokens.count,
            modelConfiguration: context.configuration,
            tokenizer: context.tokenizer,
            iterator: iterator)
        var text = ""
        var info: GenerateCompletionInfo?
        for await event in stream {
            if let chunk = event.chunk {
                text += chunk
            }
            if let completion = event.info {
                info = completion
            }
        }
        await task.value
        let unwrappedInfo = try XCTUnwrap(info)
        return ControlRun(
            promptTokens: promptTokens.count,
            tokens: nil,
            completionTokenCount: unwrappedInfo.generationTokenCount,
            text: text,
            info: unwrappedInfo,
            cache: CacheFingerprint(cache))
    }

    let (stream, task) = try generateTokensTask(
        input: LMInput(tokens: MLXArray(promptTokens)),
        cache: cache,
        parameters: parameters,
        context: context)
    var tokens: [Int] = []
    var info: GenerateCompletionInfo?
    for await event in stream {
        if let token = event.token {
            tokens.append(token)
        }
        if let completion = event.info {
            info = completion
        }
    }
    await task.value
    let unwrappedInfo = try XCTUnwrap(info)
    return ControlRun(
        promptTokens: promptTokens.count,
        tokens: tokens,
        completionTokenCount: tokens.count,
        text: context.tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false),
        info: unwrappedInfo,
        cache: CacheFingerprint(cache))
}

private func greedyParameters(maxTokens: Int) -> GenerateParameters {
    GenerateParameters(
        maxTokens: maxTokens,
        temperature: 0,
        topP: 1,
        topK: 0,
        minP: 0)
}

private final class LoadedExactQwen35MTPForwardingRunner: ExactQwen35MTPServingRunner, @unchecked Sendable {
    nonisolated let binding: QwenMTPArtifactBinding?

    private let context: ModelContext
    private let drafter: any MTPDrafterModel
    private let state = OSAllocatedUnfairLock(initialState: RunnerState())

    init(pair: Qwen35ExactMTPLoadedPair) throws {
        self.binding = try Qwen35ExactMTPRuntimeFactory.servingBinding(for: pair)
        self.context = pair.target
        self.drafter = pair.drafter.model
    }

    func start(
        _ request: ExactQwen35MTPServingRunnerRequest
    ) async throws -> ExactQwen35MTPServingRunnerHandle {
        state.withLock {
            $0.startCount += 1
            $0.lastInfo = nil
            $0.lastCache = nil
        }
        var context = context
        if !request.stopStrings.isEmpty {
            context.configuration.stopStrings =
                context.configuration.effectiveStopStrings.union(request.stopStrings)
        }
        let parameters = greedyParameters(maxTokens: request.maximumCompletionTokens)
        let cache = context.model.newCache(parameters: parameters)
        let cacheBox = CacheBox(cache)
        let (source, sourceTask) = try generateTask(
            input: LMInput(tokens: MLXArray(request.promptTokens)),
            cache: cache,
            parameters: parameters,
            context: context,
            mtpDrafter: drafter,
            blockSize: request.descriptor.runtimeBlockSize,
            parseToolCalls: false)
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let forwardingTask = Task {
            for await event in source {
                if case .info(let info) = event {
                    self.record(info: info)
                }
                continuation.yield(event)
            }
        }
        let task = Task {
            await withTaskCancellationHandler {
                await forwardingTask.value
                await sourceTask.value
                self.record(cache: CacheFingerprint(cacheBox.cache), cancelled: Task.isCancelled)
                continuation.finish()
            } onCancel: {
                sourceTask.cancel()
            }
        }
        return ExactQwen35MTPServingRunnerHandle(stream: stream, task: task)
    }

    func snapshot() -> RunnerSnapshot {
        state.withLock {
            RunnerSnapshot(
                startCount: $0.startCount,
                cancelledTaskCount: $0.cancelledTaskCount,
                finalizedTaskCount: $0.finalizedTaskCount,
                lastInfo: $0.lastInfo,
                lastCache: $0.lastCache)
        }
    }

    private func record(info: GenerateCompletionInfo) {
        state.withLock { $0.lastInfo = info }
    }

    private func record(cache: CacheFingerprint, cancelled: Bool) {
        state.withLock {
            $0.lastCache = cache
            $0.finalizedTaskCount += 1
            if cancelled {
                $0.cancelledTaskCount += 1
            }
        }
    }
}

private struct RunnerState {
    var startCount = 0
    var cancelledTaskCount = 0
    var finalizedTaskCount = 0
    var lastInfo: GenerateCompletionInfo?
    var lastCache: CacheFingerprint?
}

private struct RunnerSnapshot: Sendable {
    let startCount: Int
    let cancelledTaskCount: Int
    let finalizedTaskCount: Int
    let lastInfo: GenerateCompletionInfo?
    let lastCache: CacheFingerprint?
}

private final class CacheBox: @unchecked Sendable {
    let cache: [KVCache]

    init(_ cache: [KVCache]) {
        self.cache = cache
    }
}

private struct TokenizerBackedScalarServingTextCodec: ScalarServingTextCodec {
    let tokenizer: any MLXLMCommon.Tokenizer

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        var additionalContext: [String: any Sendable] = [:]
        if let enableThinking {
            additionalContext["enable_thinking"] = enableThinking
        }
        if let reasoningEffort {
            additionalContext["reasoning_effort"] = reasoningEffort
        }
        return try tokenizer.applyChatTemplate(
            messages: messages.map {
                ["role": $0.role.rawValue, "content": $0.text]
            },
            tools: nil,
            additionalContext: additionalContext.isEmpty ? nil : additionalContext)
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        TokenizerBackedScalarServingDetokenizer(
            detokenizer: NaiveStreamingDetokenizer(tokenizer: tokenizer))
    }
}

private struct TokenizerBackedScalarServingDetokenizer: ScalarServingDetokenizer {
    var detokenizer: NaiveStreamingDetokenizer

    mutating func append(token: Int) {
        detokenizer.append(token: token)
    }

    mutating func next() -> String? {
        detokenizer.next()
    }
}

private final class SentinelScalarFallback: ServingGenerationBackend, Sendable {
    struct Snapshot: Sendable {
        let startCount: Int
    }

    private let startCount = OSAllocatedUnfairLock(initialState: 0)

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        startCount.withLock { $0 += 1 }
        throw LoadedExactQwen35MTPBackendProofError.scalarFallbackUsed
    }

    func shutdown() async {}

    func snapshot() -> Snapshot {
        Snapshot(startCount: startCount.withLock { $0 })
    }
}

private struct CacheFingerprint: Equatable, Sendable {
    let layers: [CacheLayerFingerprint]

    init(_ cache: [KVCache]) {
        layers = cache.enumerated().map { index, entry in
            CacheLayerFingerprint(index: index, cache: entry)
        }
    }
}

extension CacheFingerprint {
    func qwenFingerprint() -> QwenMTPCorpusCacheFingerprint {
        let entries = layers.map { layer in
            QwenMTPCorpusCacheLayerFingerprint(
                layerIndex: layer.index,
                cacheType: layer.type,
                offset: layer.offset,
                metaStateSHA256: sha256Hex(layer.metaState.joined(separator: "\u{1f}")),
                stateCount: layer.states.count,
                states: layer.states.map { state in
                    QwenMTPCorpusCacheStateFingerprint(
                        stateIndex: state.index,
                        shape: state.shape,
                        dtype: state.dtype,
                        byteCount: state.byteCount,
                        sha256: state.byteSHA256)
                })
        }
        let digestInput = entries.flatMap { layer -> [String] in
            [
                "\(layer.layerIndex)",
                layer.cacheType,
                "\(layer.offset)",
                layer.metaStateSHA256,
                "\(layer.stateCount)",
            ] + layer.states.flatMap {
                [
                    "\($0.stateIndex)",
                    $0.shape.map(String.init).joined(separator: ","),
                    $0.dtype,
                    "\($0.byteCount)",
                    $0.sha256,
                ]
            }
        }.joined(separator: "\u{1e}")
        return QwenMTPCorpusCacheFingerprint(
            digest: sha256Hex(digestInput),
            entries: entries)
    }
}

private struct CacheLayerFingerprint: Equatable, Sendable {
    let index: Int
    let type: String
    let offset: Int
    let metaState: [String]
    let states: [CacheStateFingerprint]

    init(index: Int, cache: KVCache) {
        self.index = index
        self.type = String(reflecting: Swift.type(of: cache))
        self.offset = cache.offset
        self.metaState = cache.metaState
        self.states = cache.state.enumerated().map {
            CacheStateFingerprint(index: $0.offset, array: $0.element)
        }
    }
}

private struct CacheStateFingerprint: Equatable, Sendable {
    let index: Int
    let shape: [Int]
    let dtype: String
    let byteCount: Int
    let byteSHA256: String

    init(index: Int, array: MLXArray) {
        eval(array)
        let bytes = array.asData(access: .copy).data
        self.index = index
        self.shape = array.shape
        self.dtype = String(describing: array.dtype)
        self.byteCount = bytes.count
        self.byteSHA256 = SHA256.hash(data: bytes).map {
            String(format: "%02x", $0)
        }.joined()
    }
}

private struct BackendExactSnapshotDownloader: Downloader {
    let target: URL
    let drafter: URL

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard !useLatest, patterns == ["*.safetensors", "*.json", "*.jinja"] else {
            throw LoadedExactQwen35MTPBackendProofError.unexpectedDownloadRequest
        }
        progressHandler(Progress(totalUnitCount: 1))
        switch (id, revision) {
        case (
            "mlx-community/Qwen3.5-9B-MLX-4bit",
            "938d8919941c6e7efd3c7150eff7fe9d12afa631"
        ):
            return target
        case (
            "mlx-community/Qwen3.5-9B-MTP-5bit",
            "994730d199bff7799aa3ddef33a96723967a3e33"
        ):
            return drafter
        default:
            throw LoadedExactQwen35MTPBackendProofError.unexpectedDownloadRequest
        }
    }
}

private enum LoadedExactQwen35MTPBackendProofError: Error {
    case scalarFallbackUsed
    case unexpectedDownloadRequest
    case releaseBuildRequired
    case evidencePathNotExplicit
    case evidenceParentMissing
    case evidencePathIsDirectory
    case evidencePathAlreadyContainsData
    case missingCheckpointWeights
    case invalidAcceptedCorpusCardinality
    case acceptedCorpusIdentityMismatch
    case checkpointIdentityMismatch
    case harnessIdentityMismatch
}

private func isReleaseBuild() -> Bool {
#if DEBUG
    false
#else
    true
#endif
}

private func prepareNewOrEmptyEvidenceURL(_ path: String) throws -> URL {
    guard !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw LoadedExactQwen35MTPBackendProofError.evidencePathNotExplicit
    }
    let url = URL(fileURLWithPath: path)
    let parent = url.deletingLastPathComponent()
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: parent.path, isDirectory: &isDirectory),
        isDirectory.boolValue
    else {
        throw LoadedExactQwen35MTPBackendProofError.evidenceParentMissing
    }
    var targetIsDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &targetIsDirectory) else {
        return url
    }
    guard !targetIsDirectory.boolValue else {
        throw LoadedExactQwen35MTPBackendProofError.evidencePathIsDirectory
    }
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    let size = attributes[.size] as? NSNumber
    guard size?.uint64Value == 0 else {
        throw LoadedExactQwen35MTPBackendProofError.evidencePathAlreadyContainsData
    }
    return url
}

private func validateAcceptedLowerLevelCorpus(
    at path: String
) throws -> QwenMTPServingLatencyLowerLevelProof {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let decisions = try QwenMTPCorpusGate.validateJSONL(data)
    let rows = data.split(
        separator: 0x0a,
        omittingEmptySubsequences: false).dropLast()
    guard decisions.count == 1, rows.count == 1 else {
        throw LoadedExactQwen35MTPBackendProofError.invalidAcceptedCorpusCardinality
    }
    let row = rows[rows.startIndex]
    let record = try JSONDecoder().decode(
        ResultRecord<QwenMTPCorpusEvidencePayload>.self,
        from: Data(row))
    _ = try QwenMTPCorpusGate.validateGreedyBatchedVerification(record.payload)
    let required = QwenMTPServingLatencyGate.requiredLowerLevelProof
    guard record.subcommand == required.acceptedCorpusSubcommand,
        record.payload.schemaVersion == required.acceptedCorpusSchemaVersion,
        record.payload.corpusID == required.acceptedCorpusID,
        record.payload.corpusContentHash == required.acceptedCorpusContentHash,
        record.payload.binding == required.acceptedCorpusBinding,
        sha256Hex(data) == required.acceptedCorpusJSONLSHA256,
        record.provenance.harnessGitSHA == required.acceptedCorpusHarnessGitSHA
    else {
        throw LoadedExactQwen35MTPBackendProofError.acceptedCorpusIdentityMismatch
    }
    return required
}

private func servingLatencyProvenance(
    host: QwenMTPCorpusHostEvidence,
    harnessGitSHA: String,
    targetPath: String
) throws -> Provenance {
    let modelDirectory = URL(fileURLWithPath: targetPath, isDirectory: true)
    let configData = try Data(
        contentsOf: modelDirectory.appendingPathComponent("config.json"))
    return Provenance(
        date: ISO8601DateFormatter().string(from: Date()),
        hardwareChip: host.chip,
        hardwareRAMBytes: host.ramBytes,
        hardwareOS: host.os,
        harnessGitSHA: harnessGitSHA,
        mlxSwiftVersion: "0.31.6",
        referenceMLXVersion: nil,
        referenceMLXLMVersion: "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
        modelPath: QwenMTPCorpusGate.requiredBinding.targetModelID,
        modelConfigHash: fnv1a64(configData),
        modelCheckpointManifestHash: try checkpointManifestHash(
            modelDirectory: modelDirectory,
            configData: configData),
        modelQuant: ModelQuantInfoLoader.load(from: configData),
        corpusId: QwenMTPCorpusGate.corpusID,
        corpusContentHash: QwenMTPCorpusGate.corpusContentHash,
        nonce: UUID().uuidString)
}

private func verifiedHarnessGitSHA(asserted: String) throws -> String {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let deployed = try String(
        contentsOf: packageRoot.appendingPathComponent(".harness-sha"),
        encoding: .utf8)
        .trimmingCharacters(in: .whitespacesAndNewlines)
    guard deployed == asserted,
        deployed.count == 40,
        deployed.utf8.allSatisfy({
            ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
        })
    else {
        throw LoadedExactQwen35MTPBackendProofError.harnessIdentityMismatch
    }
    return deployed
}

private func servingLatencyCheckpointIdentity(
    targetPath: String,
    drafterPath: String
) throws -> QwenMTPServingLatencyCheckpointIdentity {
    QwenMTPServingLatencyCheckpointIdentity(
        targetCheckpointContentSHA256: try checkpointContentSHA256(
            modelDirectory: URL(fileURLWithPath: targetPath, isDirectory: true)),
        drafterCheckpointContentSHA256: try checkpointContentSHA256(
            modelDirectory: URL(fileURLWithPath: drafterPath, isDirectory: true)))
}

private func checkpointContentSHA256(modelDirectory: URL) throws -> String {
    let manager = FileManager.default
    let config = modelDirectory.appendingPathComponent("config.json")
    let index = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    let configData = try Data(contentsOf: config)
    let indexData = try Data(contentsOf: index)
    guard let indexRoot = try JSONSerialization.jsonObject(with: indexData)
        as? [String: Any],
        let weightMap = indexRoot["weight_map"] as? [String: String],
        !weightMap.isEmpty
    else {
        throw LoadedExactQwen35MTPBackendProofError.checkpointIdentityMismatch
    }
    let shardNames = Set(weightMap.values).sorted()
    guard !shardNames.isEmpty, shardNames.allSatisfy({ !$0.contains("/") }) else {
        throw LoadedExactQwen35MTPBackendProofError.checkpointIdentityMismatch
    }
    let actualShardNames = Set(try manager.contentsOfDirectory(
        at: modelDirectory,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles])
        .filter { $0.pathExtension == "safetensors" }
        .map(\.lastPathComponent))
    guard actualShardNames == Set(shardNames) else {
        throw LoadedExactQwen35MTPBackendProofError.missingCheckpointWeights
    }

    var hasher = SHA256()
    hasher.update(data: Data("fastmlx-checkpoint-content-manifest-v2\n".utf8))
    updateLengthField(configData, hasher: &hasher)
    updateLengthField(indexData, hasher: &hasher)
    for shardName in shardNames {
        let file = modelDirectory.appendingPathComponent(shardName)
        let values = try file.resourceValues(forKeys: [
            .fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey,
        ])
        guard values.isRegularFile == true,
            values.isSymbolicLink != true,
            let size = values.fileSize,
            size >= 0
        else {
            throw LoadedExactQwen35MTPBackendProofError.checkpointIdentityMismatch
        }
        updateLengthField(Data(shardName.utf8), hasher: &hasher)
        updateUInt64(UInt64(size), hasher: &hasher)
        let handle = try FileHandle(forReadingFrom: file)
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        try handle.close()
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

private func updateLengthField(_ data: Data, hasher: inout SHA256) {
    updateUInt64(UInt64(data.count), hasher: &hasher)
    hasher.update(data: data)
}

private func updateUInt64(_ value: UInt64, hasher: inout SHA256) {
    var bigEndian = value.bigEndian
    withUnsafeBytes(of: &bigEndian) { bytes in
        hasher.update(data: Data(bytes))
    }
}

private func appendCanonicalLengthField(_ data: Data, to output: inout Data) {
    appendCanonicalUInt64(UInt64(data.count), to: &output)
    output.append(data)
}

private func appendCanonicalUInt64(_ value: UInt64, to output: inout Data) {
    var bigEndian = value.bigEndian
    withUnsafeBytes(of: &bigEndian) { bytes in
        output.append(contentsOf: bytes)
    }
}

private func checkpointManifestHash(
    modelDirectory: URL,
    configData: Data
) throws -> String {
    let manager = FileManager.default
    var bytes = Array(configData)
    let index = modelDirectory.appendingPathComponent(
        "model.safetensors.index.json")
    if manager.fileExists(atPath: index.path) {
        bytes.append(contentsOf: try Data(contentsOf: index))
    }
    let weights = try manager.contentsOfDirectory(
        at: modelDirectory,
        includingPropertiesForKeys: [.fileSizeKey],
        options: [.skipsHiddenFiles])
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !weights.isEmpty else {
        throw LoadedExactQwen35MTPBackendProofError.missingCheckpointWeights
    }
    for weight in weights {
        let size = try weight.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1
        bytes.append(contentsOf: "\(weight.lastPathComponent):\(size)\n".utf8)
    }
    return fnv1a64(bytes)
}

private func stopOutcome(_ reason: GenerateStopReason) -> QwenMTPCorpusStopOutcome {
    switch reason {
    case .stop:
        return .stop
    case .length:
        return .length
    case .cancelled:
        return .cancelled
    }
}

private func seconds(from duration: Duration) -> Double {
    let components = duration.components
    return Double(components.seconds) + Double(components.attoseconds) / 1e18
}

private func sha256Hex(_ tokens: [Int]) -> String {
    var data = Data()
    data.reserveCapacity(tokens.count * MemoryLayout<Int64>.size)
    for token in tokens {
        var value = Int64(token).bigEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    return sha256Hex(data)
}

private func sha256Hex(_ text: String) -> String {
    sha256Hex(Data(text.utf8))
}

private func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map {
        String(format: "%02x", $0)
    }.joined()
}

private func assertMailboxCancelled(
    _ mailbox: BoundedDeltaMailbox,
    reason: ServingCancellationReason,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await mailbox.next()
        XCTFail("Expected mailbox cancellation", file: file, line: line)
    } catch let error as ServingMailboxError {
        XCTAssertEqual(error, .cancelled(reason), file: file, line: line)
    } catch {
        XCTFail("Unexpected mailbox error: \(error)", file: file, line: line)
    }
}

private func assertReservationReleased(
    _ backend: ExactQwen35MTPServingBackend,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    await waitUntil {
        await backend.snapshot().activeMTPReservations == 0
    }
    let snapshot = await backend.snapshot()
    XCTAssertEqual(snapshot.activeMTPReservations, 0, file: file, line: line)
}

private func waitUntil(
    timeout: Duration = .seconds(30),
    pollInterval: Duration = .milliseconds(10),
    _ predicate: () async -> Bool
) async {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    while clock.now < deadline {
        if await predicate() {
            return
        }
        try? await clock.sleep(for: pollInterval)
    }
    XCTFail("Condition was not reached")
}
