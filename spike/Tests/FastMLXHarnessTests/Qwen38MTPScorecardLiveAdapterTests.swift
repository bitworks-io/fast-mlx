import Foundation
import HarnessCore
import MLXLMCommon
import XCTest

@testable import fastmlx_harness

private enum ScorecardPromptTokenizerError: Error {
    case rawEncodeForbidden
    case invalidTemplateRequest
}

private struct ScorecardPromptTokenizer: MLXLMCommon.Tokenizer {
    let bosToken: String? = nil
    let eosToken: String? = "<eos>"
    let unknownToken: String? = nil
    private let log: PromptRenderLog

    init(log: PromptRenderLog) {
        self.log = log
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        fatalError("scorecard prompts must not use raw encode")
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        guard tools == nil,
            additionalContext?["enable_thinking"] as? Bool == false,
            messages.count == 1,
            messages[0]["role"] as? String == "user",
            let text = messages[0]["content"] as? String,
            !text.isEmpty
        else {
            throw ScorecardPromptTokenizerError.invalidTemplateRequest
        }
        log.record(messages: messages, additionalContext: additionalContext)
        return [77] + text.utf8.map(Int.init) + [88]
    }
}

private final class PromptRenderLog: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var calls: Int = 0
    private(set) var lastMessages: [[String: any Sendable]] = []
    private(set) var lastAdditionalContext: [String: any Sendable]?

    func record(
        messages: [[String: any Sendable]],
        additionalContext: [String: any Sendable]?
    ) {
        lock.lock()
        calls += 1
        lastMessages = messages
        lastAdditionalContext = additionalContext
        lock.unlock()
    }
}

final class Qwen38MTPScorecardLiveAdapterTests: XCTestCase {
    private typealias AdapterError = Qwen38MTPScorecardLiveAdapterError
    private typealias Gate = Qwen38MTPPerformanceScorecardGate

    func testCoordinatorBuildsAuthorityAfterTwoHandshakesAndCandidateExactnessInSameProcess() async throws {
        let fixture = ScorecardFixture()
        let candidate = FakeScorecardWorker(
            role: .candidate,
            handshake: fixture.handshake(role: .candidate, isolationSeed: "4"),
            exactnessRecord: fixture.liveExactnessRecord(isolationSeed: "4"),
            measurements: fixture.measurements(candidate: true))
        let reference = FakeScorecardWorker(
            role: .reference,
            handshake: fixture.handshake(role: .reference, isolationSeed: "5"),
            measurements: fixture.measurements(candidate: false))
        let coordinator = Qwen38MTPScorecardLiveCoordinator(
            candidate: candidate,
            reference: reference,
            runIdentity: fixture.runIdentity,
            provenance: fixture.provenance,
            releaseBuildObserved: true)

        let result = try await coordinator.run()

        XCTAssertEqual(
            result.authority.acceptedLiveExactnessProof.launchBinding,
            result.authority.trustedEngineIdentities.candidate.launchBinding)
        XCTAssertNotEqual(
            result.authority.trustedEngineIdentities.candidate.launchBinding,
            result.authority.trustedEngineIdentities.reference.launchBinding)
        XCTAssertEqual(result.record.payload.pairs.count, Gate.runPlan.schedules.count)
        let candidateSnapshot = await candidate.snapshot()
        let referenceSnapshot = await reference.snapshot()
        XCTAssertEqual(candidateSnapshot.assertedBeforeDispatchCount, Gate.runPlan.schedules.count)
        XCTAssertEqual(referenceSnapshot.assertedBeforeDispatchCount, Gate.runPlan.schedules.count)
        XCTAssertTrue(candidateSnapshot.terminated)
        XCTAssertTrue(referenceSnapshot.terminated)
    }

    func testCoordinatorRejectsCandidateExactnessFromDifferentBinding() async throws {
        let fixture = ScorecardFixture()
        let candidate = FakeScorecardWorker(
            role: .candidate,
            handshake: fixture.handshake(role: .candidate, isolationSeed: "4"),
            exactnessRecord: fixture.liveExactnessRecord(isolationSeed: "9"),
            measurements: fixture.measurements(candidate: true))
        let reference = FakeScorecardWorker(
            role: .reference,
            handshake: fixture.handshake(role: .reference, isolationSeed: "5"),
            measurements: fixture.measurements(candidate: false))

        await XCTAssertThrowsErrorAsync(
            try await Qwen38MTPScorecardLiveCoordinator(
                candidate: candidate,
                reference: reference,
                runIdentity: fixture.runIdentity,
                provenance: fixture.provenance,
                releaseBuildObserved: true).run()
        ) { error in
            XCTAssertEqual(error as? AdapterError, .exactnessLaunchBindingMismatch)
        }
    }

    func testCoordinatorRejectsEnvDriftRestartAndChildDeathBeforeDispatch() async throws {
        let fixture = ScorecardFixture()
        for failure in [
            AdapterError.workerEnvDrift(.candidate),
            AdapterError.workerRestarted(.candidate),
            AdapterError.workerExited(.candidate),
        ] {
            let candidate = FakeScorecardWorker(
                role: .candidate,
                handshake: fixture.handshake(role: .candidate, isolationSeed: "4"),
                exactnessRecord: fixture.liveExactnessRecord(isolationSeed: "4"),
                measurements: fixture.measurements(candidate: true),
                assertionFailure: failure)
            let reference = FakeScorecardWorker(
                role: .reference,
                handshake: fixture.handshake(role: .reference, isolationSeed: "5"),
                measurements: fixture.measurements(candidate: false))

            await XCTAssertThrowsErrorAsync(
                try await Qwen38MTPScorecardLiveCoordinator(
                    candidate: candidate,
                    reference: reference,
                    runIdentity: fixture.runIdentity,
                    provenance: fixture.provenance,
                    releaseBuildObserved: true).run()
            ) { error in
                XCTAssertEqual(error as? AdapterError, failure)
            }
        }
    }

    func testCoordinatorRejectsReferenceHandshakeReportingGDNOffLaunch() async throws {
        let fixture = ScorecardFixture()
        let candidate = FakeScorecardWorker(
            role: .candidate,
            handshake: fixture.handshake(role: .candidate, isolationSeed: "4"),
            exactnessRecord: fixture.liveExactnessRecord(isolationSeed: "4"),
            measurements: fixture.measurements(candidate: true))
        let reference = FakeScorecardWorker(
            role: .reference,
            handshake: fixture.handshake(
                role: .reference,
                isolationSeed: "5",
                gdnMode: .gdnOff,
                observedEnv: .disabled),
            measurements: fixture.measurements(candidate: false))

        await XCTAssertThrowsErrorAsync(
            try await Qwen38MTPScorecardLiveCoordinator(
                candidate: candidate,
                reference: reference,
                runIdentity: fixture.runIdentity,
                provenance: fixture.provenance,
                releaseBuildObserved: true).run()
        ) { error in
            XCTAssertEqual(error as? AdapterError, .invalidHandshake(.reference))
        }
    }

    func testCoordinatorRejectsHandshakeWhoseExecutionModeMismatchesItsRole() async throws {
        let fixture = ScorecardFixture()
        let candidate = FakeScorecardWorker(
            role: .candidate,
            handshake: fixture.handshake(
                role: .candidate,
                isolationSeed: "4",
                executionMode: .scalar),
            exactnessRecord: fixture.liveExactnessRecord(isolationSeed: "4"),
            measurements: fixture.measurements(candidate: true))
        let reference = FakeScorecardWorker(
            role: .reference,
            handshake: fixture.handshake(role: .reference, isolationSeed: "5"),
            measurements: fixture.measurements(candidate: false))

        await XCTAssertThrowsErrorAsync(
            try await Qwen38MTPScorecardLiveCoordinator(
                candidate: candidate,
                reference: reference,
                runIdentity: fixture.runIdentity,
                provenance: fixture.provenance,
                releaseBuildObserved: true).run()
        ) { error in
            XCTAssertEqual(error as? AdapterError, .invalidHandshake(.candidate))
        }
    }

    func testCoordinatorDrainsAndTerminatesBothWorkersOnCancellation() async throws {
        let fixture = ScorecardFixture()
        let candidate = FakeScorecardWorker(
            role: .candidate,
            handshake: fixture.handshake(role: .candidate, isolationSeed: "4"),
            exactnessRecord: fixture.liveExactnessRecord(isolationSeed: "4"),
            measurements: fixture.measurements(candidate: true),
            measurementFailure: CancellationError())
        let reference = FakeScorecardWorker(
            role: .reference,
            handshake: fixture.handshake(role: .reference, isolationSeed: "5"),
            measurements: fixture.measurements(candidate: false))

        await XCTAssertThrowsErrorAsync(
            try await Qwen38MTPScorecardLiveCoordinator(
                candidate: candidate,
                reference: reference,
                runIdentity: fixture.runIdentity,
                provenance: fixture.provenance,
                releaseBuildObserved: true).run()
        ) { error in
            XCTAssertTrue(error is CancellationError)
        }
        let candidateSnapshot = await candidate.snapshot()
        let referenceSnapshot = await reference.snapshot()
        XCTAssertTrue(candidateSnapshot.terminated)
        XCTAssertTrue(referenceSnapshot.terminated)
    }

    func testLineProtocolRejectsMalformedDuplicateAndOutOfOrderResponses() async throws {
        let fixture = ScorecardFixture()
        let handshake = fixture.handshake(role: .candidate, isolationSeed: "4")
        let valid = try Qwen38MTPScorecardWorkerProtocolResponse(
            sequence: 1,
            kind: .handshake,
            handshake: handshake).jsonLine()

        for lines in [
            ["{"],
            [try Qwen38MTPScorecardWorkerProtocolResponse(
                sequence: 2,
                kind: .handshake,
                handshake: handshake).jsonLine()],
            [valid, valid],
        ] {
            let client = Qwen38MTPScorecardLineProtocolClient(
                role: .candidate,
                transport: FakeLineTransport(lines: lines))
            do {
                _ = try await client.start()
                _ = try await client.start()
                XCTFail("Expected protocol failure")
            } catch {
                XCTAssertTrue(error is Qwen38MTPScorecardLiveAdapterError)
            }
        }
    }

    func testLiveArgumentsRequireExplicitSafeAbsentOutputsAndMemoryBudget() throws {
        let parsed = try parseQwen38MTPScorecardLiveRunArguments([
            "--target", "/models/target",
            "--drafter", "/models/drafter",
            "--output", "/tmp/scorecard.jsonl",
            "--authority-output", "/tmp/authority.json",
            "--host-use", "dedicated-serving",
            "--host-use-source", "operator-assertion",
            "--expected-chip", "Apple M3 Ultra",
            "--memory-limit-bytes", "236223201280",
            "--cache-limit-bytes", "51539607552",
            "--reserved-kv-bytes", "42949672960",
            "--reserved-io-bytes", "2147483648",
            "--reserved-prefetch-bytes", "4294967296",
            "--os-service-reserve-bytes", "8589934592",
        ])

        XCTAssertEqual(parsed.targetPath, "/models/target")
        XCTAssertEqual(parsed.drafterPath, "/models/drafter")
        XCTAssertEqual(parsed.hostUse, "dedicated-serving")
        XCTAssertEqual(parsed.hostUseSource, "operator-assertion")
        XCTAssertEqual(parsed.expectedChip, "Apple M3 Ultra")
        XCTAssertThrowsError(
            try parseQwen38MTPScorecardLiveRunArguments([
                "--target", "/models/target",
                "--drafter", "/models/drafter",
                "--output", "/tmp/scorecard.jsonl",
                "--authority-output", "/tmp/authority.json",
                "--host-use-source", "operator-assertion",
                "--expected-chip", "Apple M3 Ultra",
                "--memory-limit-bytes", "236223201280",
                "--cache-limit-bytes", "51539607552",
                "--reserved-kv-bytes", "42949672960",
                "--reserved-io-bytes", "2147483648",
                "--reserved-prefetch-bytes", "4294967296",
                "--os-service-reserve-bytes", "8589934592",
            ])) { error in
            XCTAssertEqual(error as? AdapterError, .missingFlag("--host-use"))
        }
        XCTAssertThrowsError(
            try parseQwen38MTPScorecardLiveRunArguments([
                "--target", "/models/target",
                "--drafter", "/models/drafter",
                "--output", "/tmp/scorecard.jsonl",
                "--authority-output", "/tmp/authority.json",
                "--host-use", "dedicated-serving",
                "--expected-chip", "Apple M3 Ultra",
                "--memory-limit-bytes", "236223201280",
                "--cache-limit-bytes", "51539607552",
                "--reserved-kv-bytes", "42949672960",
                "--reserved-io-bytes", "2147483648",
                "--reserved-prefetch-bytes", "4294967296",
                "--os-service-reserve-bytes", "8589934592",
            ])) { error in
            XCTAssertEqual(error as? AdapterError, .missingFlag("--host-use-source"))
        }
        XCTAssertThrowsError(
            try parseQwen38MTPScorecardLiveRunArguments([
                "--target", "/models/target",
                "--drafter", "/models/drafter",
                "--output", "/tmp/scorecard.jsonl",
                "--authority-output", "/tmp/authority.json",
                "--host-use", "dedicated-serving",
                "--host-use-source", "operator-assertion",
                "--memory-limit-bytes", "236223201280",
                "--cache-limit-bytes", "51539607552",
                "--reserved-kv-bytes", "42949672960",
                "--reserved-io-bytes", "2147483648",
                "--reserved-prefetch-bytes", "4294967296",
                "--os-service-reserve-bytes", "8589934592",
            ])) { error in
            XCTAssertEqual(error as? AdapterError, .missingFlag("--expected-chip"))
        }
        XCTAssertThrowsError(
            try parseQwen38MTPScorecardLiveRunArguments([
                "--target", "/models/target",
                "--drafter", "/models/drafter",
                "--output", "/tmp/scorecard.jsonl",
                "--authority-output", "/tmp/authority.json",
                "--host-use", "shared",
                "--host-use-source", "operator-assertion",
                "--expected-chip", "Apple M3 Ultra",
                "--memory-limit-bytes", "236223201280",
                "--cache-limit-bytes", "51539607552",
                "--reserved-kv-bytes", "42949672960",
                "--reserved-io-bytes", "2147483648",
                "--reserved-prefetch-bytes", "4294967296",
                "--os-service-reserve-bytes", "8589934592",
            ])) { error in
            XCTAssertEqual(error as? AdapterError, .invalidMemoryBudget)
        }
        XCTAssertThrowsError(
            try parseQwen38MTPScorecardLiveRunArguments([
                "--target", "/models/target",
                "--drafter", "/models/drafter",
                "--output", "/tmp/scorecard.jsonl",
                "--authority-output", "/tmp/authority.json",
                "--host-use", "dedicated-serving",
                "--host-use-source", "synthesized",
                "--expected-chip", "Apple M3 Ultra",
                "--memory-limit-bytes", "236223201280",
                "--cache-limit-bytes", "51539607552",
                "--reserved-kv-bytes", "42949672960",
                "--reserved-io-bytes", "2147483648",
                "--reserved-prefetch-bytes", "4294967296",
                "--os-service-reserve-bytes", "8589934592",
            ])) { error in
            XCTAssertEqual(error as? AdapterError, .invalidMemoryBudget)
        }
        XCTAssertThrowsError(
            try parseQwen38MTPScorecardLiveRunArguments([
                "--target", "/models/target",
                "--drafter", "/models/drafter",
                "--output", "/tmp/scorecard.jsonl",
                "--authority-output", "/tmp/authority.json",
                "--host-use", "dedicated-serving",
                "--host-use-source", "operator-assertion",
                "--expected-chip", "Apple M3 Ultra",
                "--memory-limit-bytes", "100",
                "--cache-limit-bytes", "101",
                "--reserved-kv-bytes", "1",
                "--reserved-io-bytes", "1",
                "--reserved-prefetch-bytes", "1",
                "--os-service-reserve-bytes", "1",
            ])) { error in
            XCTAssertEqual(error as? AdapterError, .invalidMemoryBudget)
        }
        XCTAssertThrowsError(
            try Qwen38MTPScorecardFreshOutputSet(
                scorecardPath: parsed.outputPath,
                authorityPath: parsed.outputPath)) { error in
            XCTAssertEqual(error as? AdapterError, .unsafeOutput)
        }
    }

    func testAggregateTwoWorkerBudgetRejectsIndividuallyValidOvercommitBeforeSpawn() throws {
        let gib = UInt64(1024 * 1024 * 1024)
        let budget = try Qwen38MTPScorecardLiveMemoryBudget(
            memoryLimitBytes: 120 * gib,
            cacheLimitBytes: 48 * gib,
            reservedKVBytes: 40 * gib,
            reservedIOBytes: 2 * gib,
            reservedPrefetchBytes: 4 * gib,
            osServiceReserveBytes: 8 * gib)
        let snapshot = Qwen38MTPScorecardLiveHostMemorySnapshot(
            physicalRAMBytes: 256 * gib,
            chipName: "Apple M3 Ultra",
            wiredLimitMB: 0,
            metalRecommendedMaxWorkingSetSizeBytes: 239 * gib,
            metalCurrentAllocatedSizeBytes: 1 * gib)

        XCTAssertNoThrow(try budget.validateAgainstHost(snapshot))
        XCTAssertThrowsError(
            try budget.validateTwoWorkerAdmission(
                snapshot,
                expectedChip: "Apple M3 Ultra")) { error in
            XCTAssertEqual(error as? AdapterError, .invalidMemoryBudget)
        }

        let wrongRAM = Qwen38MTPScorecardLiveHostMemorySnapshot(
            physicalRAMBytes: 128 * gib,
            chipName: "Apple M3 Ultra",
            wiredLimitMB: 0,
            metalRecommendedMaxWorkingSetSizeBytes: 239 * gib,
            metalCurrentAllocatedSizeBytes: 1 * gib)
        XCTAssertThrowsError(
            try budget.validateTwoWorkerAdmission(
                wrongRAM,
                expectedChip: "Apple M3 Ultra")) { error in
            XCTAssertEqual(error as? AdapterError, .invalidMemoryBudget)
        }
    }

    func testAggregateAdmissionUsesLowerMeasuredWiredLimitAndCurrentMetalAllocation() throws {
        let gib = UInt64(1024 * 1024 * 1024)
        let budget = try Qwen38MTPScorecardLiveMemoryBudget(
            memoryLimitBytes: 72 * gib,
            cacheLimitBytes: 32 * gib,
            reservedKVBytes: 24 * gib,
            reservedIOBytes: 2 * gib,
            reservedPrefetchBytes: 2 * gib,
            osServiceReserveBytes: 8 * gib)
        let snapshot = Qwen38MTPScorecardLiveHostMemorySnapshot(
            physicalRAMBytes: 256 * gib,
            chipName: "Apple M3 Ultra",
            wiredLimitMB: 168_000,
            metalRecommendedMaxWorkingSetSizeBytes: 220 * gib,
            metalCurrentAllocatedSizeBytes: 1 * gib)
        XCTAssertNoThrow(try budget.validateTwoWorkerAdmission(
            snapshot,
            expectedChip: "Apple M3 Ultra"))
        XCTAssertThrowsError(
            try budget.validateTwoWorkerAdmission(
                snapshot,
                expectedChip: "Apple M3 Max")) { error in
            XCTAssertEqual(error as? AdapterError, .invalidMemoryBudget)
        }

        let drifted = Qwen38MTPScorecardLiveHostMemorySnapshot(
            physicalRAMBytes: 256 * gib,
            chipName: "Apple M3 Ultra",
            wiredLimitMB: 168_000,
            metalRecommendedMaxWorkingSetSizeBytes: 220 * gib,
            metalCurrentAllocatedSizeBytes: 6 * gib)
        XCTAssertThrowsError(
            try budget.validateTwoWorkerAdmission(
                drifted,
                expectedChip: "Apple M3 Ultra")) { error in
            XCTAssertEqual(error as? AdapterError, .invalidMemoryBudget)
        }
    }

    func testAdmissionPinsFrozenM3UltraEvenWhenOperatorEchoesDifferentChip() throws {
        let gib = UInt64(1024 * 1024 * 1024)
        let budget = try Qwen38MTPScorecardLiveMemoryBudget(
            memoryLimitBytes: 72 * gib,
            cacheLimitBytes: 24 * gib,
            reservedKVBytes: 24 * gib,
            reservedIOBytes: 2 * gib,
            reservedPrefetchBytes: 2 * gib,
            osServiceReserveBytes: 8 * gib)
        let wrongChip = Qwen38MTPScorecardLiveHostMemorySnapshot(
            physicalRAMBytes: 256 * gib,
            chipName: "Apple M3 Max",
            wiredLimitMB: 0,
            metalRecommendedMaxWorkingSetSizeBytes: 220 * gib,
            metalCurrentAllocatedSizeBytes: 1 * gib)

        XCTAssertThrowsError(
            try budget.validateTwoWorkerAdmission(
                wrongChip,
                expectedChip: "Apple M3 Max")) { error in
            XCTAssertEqual(error as? AdapterError, .invalidMemoryBudget)
        }
    }

    func testHostObservationPreservesMeasuredMetalRecommendationAndRefusesAbsentOrLowPlan() throws {
        let gib = UInt64(1024 * 1024 * 1024)
        let budget = try Qwen38MTPScorecardLiveMemoryBudget(
            memoryLimitBytes: 120 * gib,
            cacheLimitBytes: 48 * gib,
            reservedKVBytes: 40 * gib,
            reservedIOBytes: 2 * gib,
            reservedPrefetchBytes: 4 * gib,
            osServiceReserveBytes: 8 * gib)
        let snapshot = Qwen38MTPScorecardLiveHostMemorySnapshot(
            physicalRAMBytes: 256 * gib,
            chipName: "Apple M3 Ultra",
            wiredLimitMB: 0,
            metalRecommendedMaxWorkingSetSizeBytes: 119 * gib,
            metalCurrentAllocatedSizeBytes: 2 * gib)

        XCTAssertThrowsError(
            try Qwen38MTPScorecardProcessFacts.hostMemoryObservation(
                memoryBudget: budget,
                hostUse: "dedicated-serving",
                hostUseSource: "operator-assertion",
                hostSnapshot: snapshot)) { error in
            XCTAssertEqual(error as? AdapterError, .invalidMemoryBudget)
        }
        let absent = Qwen38MTPScorecardLiveHostMemorySnapshot(
            physicalRAMBytes: 256 * gib,
            chipName: "Apple M3 Ultra",
            wiredLimitMB: 0,
            metalRecommendedMaxWorkingSetSizeBytes: nil,
            metalCurrentAllocatedSizeBytes: 2 * gib)
        XCTAssertThrowsError(
            try Qwen38MTPScorecardProcessFacts.hostMemoryObservation(
                memoryBudget: budget,
                hostUse: "dedicated-serving",
                hostUseSource: "operator-assertion",
                hostSnapshot: absent)) { error in
            XCTAssertEqual(error as? AdapterError, .invalidMemoryBudget)
        }

        let admitted = Qwen38MTPScorecardLiveHostMemorySnapshot(
            physicalRAMBytes: 256 * gib,
            chipName: "Apple M3 Ultra",
            wiredLimitMB: 0,
            metalRecommendedMaxWorkingSetSizeBytes: 140 * gib,
            metalCurrentAllocatedSizeBytes: 2 * gib)
        let observation = try Qwen38MTPScorecardProcessFacts.hostMemoryObservation(
            memoryBudget: budget,
            hostUse: "dedicated-serving",
            hostUseSource: "operator-assertion",
            hostSnapshot: admitted)
        XCTAssertEqual(observation.metalRecommendedMaxWorkingSetSizeBytes, 140 * gib)
        XCTAssertEqual(observation.wiredLimitMB, 0)
        XCTAssertEqual(observation.hostUse, "dedicated-serving")
        XCTAssertEqual(observation.hostUseSource, "operator-assertion")
    }

    func testScorecardPromptRenderingUsesQwenChatTemplateWithThinkingDisabled() throws {
        let log = PromptRenderLog()
        let tokens = try qwen38MTPScorecardPromptTokenIDs(
            prompt: "Explain a bounded scorecard.",
            tokenizer: ScorecardPromptTokenizer(log: log))

        XCTAssertEqual(tokens.first, 77)
        XCTAssertEqual(tokens.last, 88)
        XCTAssertEqual(log.calls, 1)
        XCTAssertEqual(log.lastMessages[0]["role"] as? String, "user")
        XCTAssertEqual(log.lastMessages[0]["content"] as? String, "Explain a bounded scorecard.")
        XCTAssertEqual(log.lastAdditionalContext?["enable_thinking"] as? Bool, false)
    }

    func testChatTemplateHashBindsStandaloneTemplateBytes() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen38-template-\(UUID().uuidString)",
            isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let template = Data("{{ messages }}".utf8)
        try template.write(
            to: directory.appendingPathComponent("chat_template.jinja"),
            options: .withoutOverwriting)

        XCTAssertEqual(
            try qwen38MTPScorecardChatTemplateSHA256(modelDirectory: directory),
            qwen38MTPScorecardSHA256Hex(template))
    }

    func testChatTemplateHashRequiresStandaloneJinjaTemplate() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen38-template-fallback-\(UUID().uuidString)",
            isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{\"chat_template\":\"{{ messages }}\"}".utf8).write(
            to: directory.appendingPathComponent("tokenizer_config.json"),
            options: .withoutOverwriting)

        XCTAssertThrowsError(
            try qwen38MTPScorecardChatTemplateSHA256(modelDirectory: directory)) { error in
            XCTAssertEqual(error as? AdapterError, .workerError)
        }
    }

    func testBoundedTerminationEscalatesAfterDeadline() {
        let script = ScriptedChildProcess(isRunningSequence: Array(repeating: true, count: 8))
        qwen38MTPScorecardTerminateChild(
            script,
            deadlineChecks: 1,
            pollIntervalMicroseconds: 0)

        XCTAssertEqual(script.actions, [.terminate, .interrupt, .kill])
    }

    func testLineProtocolTerminationNeverStartsBlockingShutdownRoundTrip() async {
        let transport = RecordingLineTransport()
        let client = Qwen38MTPScorecardLineProtocolClient(
            role: .candidate,
            transport: transport)

        await client.terminate()

        let snapshot = await transport.snapshot()
        XCTAssertEqual(snapshot.sentLineCount, 0)
        XCTAssertTrue(snapshot.terminated)
    }

    func testSecondTransportSpawnFailureTerminatesFirstTransport() async {
        let candidate = RecordingLineTransport()

        await XCTAssertThrowsErrorAsync(
            try await qwen38MTPScorecardMakeTransportPair(
                makeCandidate: { candidate },
                makeReference: { () throws -> RecordingLineTransport in
                    throw AdapterError.workerError
                })) { error in
            XCTAssertEqual(error as? AdapterError, .workerError)
        }

        let snapshot = await candidate.snapshot()
        XCTAssertTrue(snapshot.terminated)
    }

    func testGDNOffObservationRejectsAnyInheritedOverride() {
        XCTAssertEqual(
            Qwen38MTPScorecardProcessFacts.observedGDNEnv(
                mode: .gdnOff,
                environmentValue: nil),
            .disabled)
        XCTAssertEqual(
            Qwen38MTPScorecardProcessFacts.observedGDNEnv(
                mode: .gdnOff,
                environmentValue: "1"),
            .enabled)
        XCTAssertEqual(
            Qwen38MTPScorecardProcessFacts.observedGDNEnv(
                mode: .gdnOff,
                environmentValue: "unexpected"),
            .enabled)
    }

    func testDecodeTimingCountsOnlyTokensAfterFirstTokenWhenClockStartsAtFirstToken() {
        let timing = qwen38MTPScorecardDecodeTiming(
            firstTokenTime: 12.0,
            decodeStart: 10.0,
            end: 15.5,
            generatedTokenCount: 4)

        XCTAssertEqual(timing.decodeTokenCount, 3)
        XCTAssertEqual(timing.decodeSeconds, 3.5, accuracy: 1e-12)

        let empty = qwen38MTPScorecardDecodeTiming(
            firstTokenTime: nil,
            decodeStart: 10.0,
            end: 11.0,
            generatedTokenCount: 0)
        XCTAssertEqual(empty.decodeTokenCount, 0)
        XCTAssertEqual(empty.decodeSeconds, 1.0, accuracy: 1e-12)
    }

    func testBoundedStderrSinkCannotGrowUnbounded() async {
        let sink = Qwen38MTPScorecardBoundedByteSink(limit: 5)

        await sink.append(Data("abcdef".utf8))
        await sink.append(Data("gh".utf8))

        let snapshot = await sink.snapshot()
        XCTAssertEqual(snapshot.retained, Data("abcde".utf8))
        XCTAssertEqual(snapshot.droppedByteCount, 3)
    }

    func testTwoFilePublicationRollsBackScorecardWhenAuthorityLinkFails() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen38-live-output-\(UUID().uuidString)",
            isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let scorecard = directory.appendingPathComponent("scorecard.jsonl")
        let authority = directory.appendingPathComponent("authority.json")
        let outputs = try Qwen38MTPScorecardFreshOutputSet(
            scorecardPath: scorecard.path,
            authorityPath: authority.path)
        try Data("existing".utf8).write(to: authority, options: .withoutOverwriting)

        XCTAssertThrowsError(
            try writeFreshQwen38MTPScorecardOutputSet(
                scorecardData: Data("{\"ok\":true}\n".utf8),
                authorityData: Data("{\"authority\":true}".utf8),
                outputs: outputs)) { error in
            XCTAssertEqual(error as? AdapterError, .outputExists)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: scorecard.path))
        XCTAssertEqual(try Data(contentsOf: authority), Data("existing".utf8))
    }

    func testTwoFilePublicationRollsBackBothOutputsWhenFsyncFailsAfterBothLinks() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen38-live-output-fsync-\(UUID().uuidString)",
            isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let scorecard = directory.appendingPathComponent("scorecard.jsonl")
        let authority = directory.appendingPathComponent("authority.json")
        let outputs = try Qwen38MTPScorecardFreshOutputSet(
            scorecardPath: scorecard.path,
            authorityPath: authority.path)

        XCTAssertThrowsError(
            try writeFreshQwen38MTPScorecardOutputSet(
                scorecardData: Data("{\"ok\":true}\n".utf8),
                authorityData: Data("{\"authority\":true}".utf8),
                outputs: outputs,
                fsyncParents: { _ in throw AdapterError.outputWriteFailed })) { error in
            XCTAssertEqual(error as? AdapterError, .outputWriteFailed)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: scorecard.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: authority.path))
    }

    func testPeakRSSUsesMaxResidentSetSizeBytes() {
        XCTAssertEqual(qwen38MTPScorecardPeakRSSBytes(maxResidentSetSize: 4096), 4096)
        XCTAssertEqual(qwen38MTPScorecardPeakRSSBytes(maxResidentSetSize: 0), 1)
    }
}

private final class ScriptedChildProcess: Qwen38MTPScorecardChildProcess {
    enum Action: Equatable {
        case terminate
        case interrupt
        case kill
    }

    private let isRunningSequence: [Bool]
    private var isRunningIndex = 0
    private(set) var actions: [Action] = []

    init(isRunningSequence: [Bool]) {
        self.isRunningSequence = isRunningSequence
    }

    var isRunning: Bool {
        guard !isRunningSequence.isEmpty else { return false }
        defer { isRunningIndex += 1 }
        return isRunningSequence[min(isRunningIndex, isRunningSequence.count - 1)]
    }

    func terminate() {
        actions.append(.terminate)
    }

    func interrupt() {
        actions.append(.interrupt)
    }

    func forceKill() {
        actions.append(.kill)
    }
}

private actor RecordingLineTransport: Qwen38MTPScorecardLineTransport {
    private var sentLineCount = 0
    private var terminated = false

    func sendLine(_ line: String) async throws {
        sentLineCount += 1
    }

    func receiveLine() async throws -> String? {
        XCTFail("termination must not wait for a cooperative worker response")
        return nil
    }

    func terminate() async {
        terminated = true
    }

    func snapshot() -> (sentLineCount: Int, terminated: Bool) {
        (sentLineCount, terminated)
    }
}

private actor FakeScorecardWorker: Qwen38MTPScorecardWorkerClient {
    let role: Qwen38MTPPerformanceScorecardEngineRole
    private let handshakeValue: Qwen38MTPScorecardWorkerHandshake
    private let exactnessRecord:
        ResultRecord<Qwen38MTPLiveExactnessEvidence>?
    private let measurementValues: [Qwen38MTPPerformanceScorecardEngineMeasurement]
    private let assertionFailure: Error?
    private let measurementFailure: Error?
    private var measurementIndex = 0
    private(set) var assertedBeforeDispatchCount = 0
    private(set) var terminated = false

    init(
        role: Qwen38MTPPerformanceScorecardEngineRole,
        handshake: Qwen38MTPScorecardWorkerHandshake,
        exactnessRecord: ResultRecord<Qwen38MTPLiveExactnessEvidence>? = nil,
        measurements: [Qwen38MTPPerformanceScorecardEngineMeasurement],
        assertionFailure: Error? = nil,
        measurementFailure: Error? = nil
    ) {
        self.role = role
        self.handshakeValue = handshake
        self.exactnessRecord = exactnessRecord
        self.measurementValues = measurements
        self.assertionFailure = assertionFailure
        self.measurementFailure = measurementFailure
    }

    func start() async throws -> Qwen38MTPScorecardWorkerHandshake {
        handshakeValue
    }

    func runCandidateExactness() async throws -> ResultRecord<Qwen38MTPLiveExactnessEvidence> {
        try exactnessRecord ?? {
            throw Qwen38MTPScorecardLiveAdapterError.invalidWorkerRole(role)
        }()
    }

    func assertReadyForDispatch(
        expected: Qwen38MTPScorecardWorkerHandshake
    ) async throws {
        assertedBeforeDispatchCount += 1
        if let assertionFailure {
            throw assertionFailure
        }
        guard expected == handshakeValue else {
            throw Qwen38MTPScorecardLiveAdapterError.workerRestarted(role)
        }
    }

    func measure(
        _ request: Qwen38MTPPerformanceScorecardMeasurementRequest
    ) async throws -> Qwen38MTPPerformanceScorecardEngineMeasurement {
        if let measurementFailure {
            throw measurementFailure
        }
        defer { measurementIndex += 1 }
        return measurementValues[measurementIndex]
    }

    func terminate() async {
        terminated = true
    }

    func snapshot() -> (assertedBeforeDispatchCount: Int, terminated: Bool) {
        (assertedBeforeDispatchCount, terminated)
    }
}

private struct FakeLineTransport: Qwen38MTPScorecardLineTransport {
    let lines: [String]
    private let index = LockedIndex()

    func sendLine(_ line: String) async throws {}

    func receiveLine() async throws -> String? {
        await index.next(in: lines)
    }

    func terminate() async {}
}

private actor LockedIndex {
    private var index = 0

    func next(in lines: [String]) -> String? {
        guard index < lines.count else { return nil }
        defer { index += 1 }
        return lines[index]
    }
}

private struct ScorecardFixture {
    private typealias Gate = Qwen38MTPPerformanceScorecardGate

    let runIdentity = Qwen38MTPPerformanceScorecardTrustedRunIdentity(
        measurementClass: Qwen38MTPPerformanceScorecardGate.measurementClass,
        hardwareChip: "Apple M3 Ultra",
        hardwareRAMBytes: Qwen38MTPPerformanceScorecardGate.requiredRAMBytes,
        hardwareOSBuild: "macOS 26.0",
        hostIdentityDigest: hex("a"),
        harnessGitSHA: String(repeating: "e", count: 40),
        candidateMLXSwiftVersion: "0.31.6",
        referenceMLXVersion: nil,
        referenceMLXLMVersion: nil,
        modelLabel: Qwen38MTPPerformanceScorecardGate.modelArtifactLabel,
        modelConfigHash:
            Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetConfigSHA256,
        modelCheckpointManifestHash:
            Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetTensorManifestSHA256,
        modelQuant: .init(bits: 8, groupSize: 32),
        corpusID: Qwen38MTPPerformanceScorecardGate.requiredWorkload.id,
        corpusContentHash:
            Qwen38MTPPerformanceScorecardGate.requiredWorkload.contentSHA256)

    var provenance: Provenance {
        Provenance(
            date: "2026-08-25T00:00:00Z",
            hardwareChip: runIdentity.hardwareChip,
            hardwareRAMBytes: runIdentity.hardwareRAMBytes,
            hardwareOS: runIdentity.hardwareOSBuild,
            harnessGitSHA: runIdentity.harnessGitSHA,
            mlxSwiftVersion: runIdentity.candidateMLXSwiftVersion,
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelPath: runIdentity.modelLabel,
            modelConfigHash: runIdentity.modelConfigHash,
            modelCheckpointManifestHash: runIdentity.modelCheckpointManifestHash,
            modelQuant: runIdentity.modelQuant,
            corpusId: runIdentity.corpusID,
            corpusContentHash: runIdentity.corpusContentHash,
            nonce: "scorecard-live-adapter-test")
    }

    func handshake(
        role: Qwen38MTPPerformanceScorecardEngineRole,
        isolationSeed: Character,
        gdnMode mode: Qwen38MTPPerformanceScorecardGDNMode = .gdnOn,
        observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv = .enabled,
        executionMode: Qwen38MTPPerformanceScorecardExecutionMode? = nil
    ) -> Qwen38MTPScorecardWorkerHandshake {
        let isolation = processIsolation(seed: isolationSeed, mode: mode, observedEnv: observedEnv)
        let processID = Qwen38MTPLiveExactnessGate.processIsolationEvidenceID(for: isolation)
        let launchBinding = Qwen38MTPPerformanceScorecardLaunchBinding(
            mode: mode,
            sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
            observedEnv: observedEnv,
            processIsolationEvidenceID: processID,
            launchDigest: Gate.launchDigest(
                mode: mode,
                sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
                observedEnv: observedEnv,
                processIsolationEvidenceID: processID))
        return Qwen38MTPScorecardWorkerHandshake(
            role: role,
            model: Qwen38MTPPerformanceScorecardModel(
                label: "engine",
                executionMode: executionMode ?? (role == .candidate ? .exactMTP : .scalar),
                artifact: Gate.requiredArtifact,
                executionDigest: hex(role == .candidate ? "c" : "d"),
                sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
                gdnMode: mode,
                launchBinding: launchBinding),
            processIsolation: isolation,
            launchBinding: launchBinding)
    }

    func liveExactnessRecord(
        isolationSeed: Character
    ) -> ResultRecord<Qwen38MTPLiveExactnessEvidence> {
        let handshake = handshake(role: .candidate, isolationSeed: isolationSeed)
        let evidence = Qwen38MTPLiveExactnessEvidence(
            schemaVersion: Qwen38MTPLiveExactnessGate.schemaVersion,
            artifact: Gate.requiredArtifact,
            artifactID: Qwen38MTPLiveExactnessGate.requiredArtifactID,
            source: Qwen38MTPLiveExactnessGate.requiredSourceIdentity,
            gdnMode: .gdnOn,
            launchBinding: handshake.launchBinding,
            processIsolation: handshake.processIsolation,
            mlxMemoryBudget: .init(
                memoryLimitBytes: 220 * 1024 * 1024 * 1024,
                cacheLimitBytes: 48 * 1024 * 1024 * 1024),
            hostMemoryObservation: hostMemoryObservation(),
            cases: [
                caseEvidence(id: "numbers", text: " 13, 17", seed: "1"),
                caseEvidence(id: "sentence", text: " reliable.", seed: "2"),
            ])
        return ResultRecord(
            subcommand: Qwen38MTPLiveExactnessGate.subcommand,
            provenance: liveExactnessProvenance,
            payload: evidence)
    }

    func measurements(
        candidate: Bool
    ) -> [Qwen38MTPPerformanceScorecardEngineMeasurement] {
        Gate.runPlan.schedules.enumerated().map { offset, schedule in
            makeEngine(
                identity: handshake(
                    role: candidate ? .candidate : .reference,
                    isolationSeed: candidate ? "4" : "5").model,
                schedule: schedule,
                offset: offset,
                candidate: candidate)
        }
    }

    private var liveExactnessProvenance: Provenance {
        Provenance(
            date: "2026-08-25T00:00:00Z",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: Gate.requiredRAMBytes,
            hardwareOS: "macOS 26.0",
            harnessGitSHA: String(repeating: "e", count: 40),
            mlxSwiftVersion: "0.31.6",
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelPath: Qwen38MTPLiveExactnessGate.modelPathSentinel,
            modelConfigHash: Gate.requiredArtifact.targetConfigSHA256,
            modelCheckpointManifestHash: Gate.requiredArtifact.targetTensorManifestSHA256,
            modelQuant: .init(bits: 8, groupSize: 32),
            corpusId: nil,
            corpusContentHash: nil,
            nonce: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID)
    }

    private func makeEngine(
        identity: Qwen38MTPPerformanceScorecardModel,
        schedule: Qwen38MTPPerformanceScorecardPairSchedule,
        offset: Int,
        candidate: Bool
    ) -> Qwen38MTPPerformanceScorecardEngineMeasurement {
        let e2e = candidate ? 9.0 : 10.0
        let decodeSeconds = candidate ? 0.9 : 1.0
        return Qwen38MTPPerformanceScorecardEngineMeasurement(
            identity: identity,
            requests: schedule.caseIDs.enumerated().map { requestIndex, caseID in
                Qwen38MTPPerformanceScorecardRequestMeasurement(
                    caseID: caseID,
                    benchmarkCell: schedule.benchmarkCells[requestIndex],
                    requestIndex: requestIndex,
                    promptSeconds: 0.25,
                    prefillSeconds: 0.8,
                    ttftSeconds: 1.0,
                    decodeTokenCount: 100,
                    decodeSeconds: decodeSeconds,
                    e2eSeconds: e2e,
                    outputDigest: digest("\(offset)-\(requestIndex)-output"),
                    cacheDigest: digest("\(offset)-\(requestIndex)-cache"),
                    outputProvenanceID: digest("\(offset)-\(requestIndex)-output-proof"),
                    cacheProvenanceID: digest("\(offset)-\(requestIndex)-cache-proof"))
            },
            wallSeconds: e2e,
            peakRSSBytes: 210_000_000_000,
            peakMetalBytes: 190_000_000_000,
            thermalBefore: "nominal",
            thermalAfter: "fair",
            proposalCount: candidate ? 12 * schedule.concurrency : 0,
            acceptedCount: candidate ? 8 * schedule.concurrency : 0,
            fallbackUsed: false,
            passthroughUsed: false)
    }

    private func processIsolation(
        seed: Character,
        mode: Qwen38MTPPerformanceScorecardGDNMode,
        observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv
    ) -> Qwen38MTPLiveExactnessProcessIsolationEvidence {
        Qwen38MTPLiveExactnessProcessIsolationEvidence(
            processID: seed == "4" ? 44_004 : 44_005,
            parentProcessID: 44_000,
            processStartUptimeNanoseconds: seed == "4" ? 123_456_004 : 123_456_005,
            bootTimeUnixSeconds: 1_777_000_000,
            executableIdentitySource: .procPIDPath,
            executableSHA256: hex("6"),
            harnessGitSHA: String(repeating: "e", count: 40),
            sourceID: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
            gdnMode: mode,
            observedEnv: observedEnv)
    }

    private func caseEvidence(
        id: String,
        text: String,
        seed: Character
    ) -> Qwen38MTPLiveExactnessCaseEvidence {
        let bytes = Data(text.utf8)
        return Qwen38MTPLiveExactnessCaseEvidence(
            id: id,
            promptSHA256: Qwen38MTPLiveExactnessGate.requiredCasesByID[id]!.promptSHA256,
            maxTokens: Qwen38MTPLiveExactnessGate.requiredCasesByID[id]!.maxTokens,
            scalarTokenIDs: [1, 2],
            mtpTokenIDs: [1, 2],
            scalarDecodedUTF8Base64: bytes.base64EncodedString(),
            mtpDecodedUTF8Base64: bytes.base64EncodedString(),
            decodedUTF8SHA256: Qwen38MTPLiveExactnessGate.sha256Hex(bytes),
            proposedDraftTokens: 4,
            acceptedDraftTokens: 2,
            passthroughReason: nil,
            scalarCacheFingerprints: cache(seed: seed),
            mtpCacheFingerprints: cache(seed: seed))
    }

    private func cache(seed: Character) -> [Qwen38MTPLiveExactnessCacheFingerprint] {
        (0 ..< Qwen38MTPLiveExactnessGate.requiredCacheLayerCount).map { layer in
            let isDense = (layer + 1).isMultiple(of: 4)
            return Qwen38MTPLiveExactnessCacheFingerprint(
                layerIndex: layer,
                cacheType: isDense ? "dense-attention" : "recurrent-mamba",
                offset: isDense ? 14 : 0,
                metaStateSHA256: hex(seed),
                stateFingerprints: (0 ..< 2).map { state in
                    let shape = isDense
                        ? [1, 4, 14, 256]
                        : (state == 0 ? [1, 3, 10_240] : [1, 48, 128, 128])
                    let dtype = isDense || state == 0 ? "bfloat16" : "float32"
                    return Qwen38MTPLiveExactnessArrayFingerprint(
                        stateIndex: state,
                        shape: shape,
                        dtype: dtype,
                        byteCount: shape.reduce(1, *) * (dtype == "float32" ? 4 : 2),
                        sha256: hex(seed))
                })
        }
    }

    private func hostMemoryObservation() -> Qwen38MTPLiveExactnessHostMemoryObservation {
        let gib = UInt64(1024 * 1024 * 1024)
        return Qwen38MTPLiveExactnessHostMemoryObservation(
            hostUse: "dedicated-serving",
            hostUseSource: "operator-assertion",
            hostUsePolicyVersion: Qwen38MTPLiveExactnessGate.requiredHostUsePolicyVersion,
            physicalRAMBytes: 256 * gib,
            wiredLimitMB: 0,
            wiredLimitProvenance: .measured,
            metalRecommendedMaxWorkingSetSizeBytes: 240 * gib,
            metalCurrentAllocatedSizeBytes: 2 * gib,
            memoryLimitBytes: 220 * gib,
            cacheLimitBytes: 48 * gib,
            reservedKVBytes: 40 * gib,
            reservedIOBytes: 2 * gib,
            reservedPrefetchBytes: 4 * gib,
            osServiceReserveBytes: 8 * gib)
    }

    private func digest(_ seed: String) -> String {
        Qwen38MTPPerformanceScorecardGate.promptSHA256(seed)
    }
}

private func hex(_ character: Character) -> String {
    String(repeating: String(character), count: 64)
}

private func XCTAssertThrowsErrorAsync(
    _ expression: @autoclosure () async throws -> some Any,
    _ verify: (Error) -> Void,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await expression()
        XCTFail("Expected error", file: file, line: line)
    } catch {
        verify(error)
    }
}
