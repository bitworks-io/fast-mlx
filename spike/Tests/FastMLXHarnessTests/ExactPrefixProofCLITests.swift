import CryptoKit
import Foundation
import HarnessCore
import XCTest

@testable import fastmlx_harness

final class ExactPrefixProofCLITests: XCTestCase {
    private let harnessSHA = String(repeating: "a", count: 40)
    private let binarySHA = String(repeating: "b", count: 64)
    private let checkpointSHA = String(repeating: "c", count: 64)
    private let tokenizerSHA = String(repeating: "d", count: 64)

    func testStrictCommandParserBuildsAuthenticatedPlan() throws {
        let raw = try parseExactPrefixProofRawCommand(
            arguments: validArguments())
        XCTAssertEqual(raw.modelPath, "/models/qwen")
        XCTAssertEqual(raw.modelID, "qwen3-32b")
        XCTAssertEqual(raw.expectedHarnessSHA, harnessSHA)
        XCTAssertEqual(raw.expectedExecutableSHA256, binarySHA)

        let command = try parseExactPrefixProofCommand(
            arguments: validArguments(),
            actualHarnessSHA: harnessSHA,
            actualExecutableSHA256: binarySHA,
            admission: try admission())

        XCTAssertEqual(command.modelPath, "/models/qwen")
        XCTAssertEqual(command.plan.modelID, "qwen3-32b")
        XCTAssertEqual(command.plan.maxTokens, 16)
        XCTAssertEqual(command.plan.promptRepeat, 8)
        XCTAssertEqual(
            command.plan.exactPrefixCachePolicy.maxEntries,
            8)
        XCTAssertEqual(
            command.plan.templateTokenCachePolicy.maxEntries,
            16)
        XCTAssertEqual(command.plan.outputPath, "/proofs/qwen-v1")
    }

    func testProofRunConfigUsesTheNativeDenseHalfRoute() throws {
        let command = try parseExactPrefixProofCommand(
            arguments: validArguments(),
            actualHarnessSHA: harnessSHA,
            actualExecutableSHA256: binarySHA,
            admission: try admission())
        let context = try proofRequestContext()

        let config = exactPrefixProofRunConfig(
            plan: command.plan,
            requestContext: context)

        XCTAssertNil(config.kvQuant)
        XCTAssertNil(config.compressedKVAttention)
        XCTAssertNil(
            config
                .compressedKVAttentionExpectedCheckpointContentSHA256)
        XCTAssertEqual(config.exactPrefixRequest, context)
        XCTAssertEqual(
            try resolveSwiftEngineCacheSelection(
                config: config,
                compressedKVAttentionAdmission: command.plan.admission)
                .kind,
            .fp16)
    }

    func testProofPromptEndsEveryRepeatedBlockWithAnExplicitResponseCue() {
        let text = exactPrefixProofPromptText(
            workloadNonce: "nonce",
            label: "A",
            repeatCount: 2)

        XCTAssertEqual(
            text,
            """
            exact-prefix-proof nonce A: deterministic ledger row for cache admission and byte identity.
            Continue the ledger by replying with the single lowercase word ready.
            Response:
            exact-prefix-proof nonce A: deterministic ledger row for cache admission and byte identity.
            Continue the ledger by replying with the single lowercase word ready.
            Response:
            """)
        XCTAssertEqual(
            text.components(separatedBy: "Response:").count - 1,
            2)
        XCTAssertTrue(text.hasSuffix("Response:"))
        XCTAssertEqual(
            exactPrefixProofFormattingOptionsSHA256(),
            digest(
                """
                fast-mlx-exact-prefix-proof-format-v2
                explicit-response-cue

                """))
        XCTAssertEqual(
            exactPrefixProofPromptTemplateSHA256(),
            digest(
                """
                fast-mlx-exact-prefix-proof-template-v2
                fixed-ledger-response

                """))
        XCTAssertEqual(
            exactPrefixProofPromptContentSHA256(
                workloadNonce: "nonce",
                group: "A"),
            digest(
                """
                fast-mlx-exact-prefix-proof-content-v2
                explicit-response-cue
                nonce
                A

                """))
    }

    func testStrictCommandParserRejectsUnknownDuplicateAndIdentityDrift()
        throws
    {
        XCTAssertThrowsError(try parse(arguments: validArguments() + [
            "--unknown", "value",
        ])) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCLIError,
                .unknownFlag("--unknown"))
        }
        XCTAssertThrowsError(try parse(arguments: validArguments() + [
            "--model-id", "duplicate",
        ])) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCLIError,
                .duplicateFlag("--model-id"))
        }
        XCTAssertThrowsError(try parse(
            arguments: validArguments(),
            actualHarnessSHA: String(repeating: "e", count: 40)))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCLIError,
                .harnessIdentityMismatch)
        }
        XCTAssertThrowsError(try parse(
            arguments: validArguments(),
            actualExecutableSHA256: String(repeating: "e", count: 64)))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCLIError,
                .executableIdentityMismatch)
        }
        var relativeOutput = validArguments()
        relativeOutput[
            relativeOutput.firstIndex(of: "--output")! + 1
        ] = "relative/proof"
        XCTAssertThrowsError(
            try parseExactPrefixProofRawCommand(
                arguments: relativeOutput)
        ) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCLIError,
                .invalidValue("--output"))
        }
        var wrongSourceRevision = validArguments()
        wrongSourceRevision[
            wrongSourceRevision.firstIndex(
                of: "--source-revision")! + 1
        ] = digest("different-checkpoint")
        XCTAssertThrowsError(
            try parse(arguments: wrongSourceRevision)
        ) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .sourceRevisionIdentityMismatch)
        }
        var unboundedEntries = validArguments()
        unboundedEntries[
            unboundedEntries.firstIndex(
                of: "--prefix-max-entries")! + 1
        ] = String(Int.max)
        XCTAssertThrowsError(
            try parseExactPrefixProofRawCommand(
                arguments: unboundedEntries)
        ) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCLIError,
                .invalidValue("--prefix-max-entries"))
        }
        var oversizedRetention = validArguments()
        oversizedRetention[
            oversizedRetention.firstIndex(
                of: "--prefix-max-retained-bytes")! + 1
        ] = "103079215104"
        XCTAssertThrowsError(
            try parseExactPrefixProofRawCommand(
                arguments: oversizedRetention)
        ) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCLIError,
                .invalidValue("--prefix/template-memory-budget"))
        }
    }

    func testExecutableIdentityAuthenticatesAndDetectsMutation() throws {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: parent) }
        let executable = parent.appendingPathComponent("proof-harness")
        let initial = Data("proof-binary-v1".utf8)
        try initial.write(to: executable)
        let expected = SHA256.hash(data: initial).map {
            String(format: "%02x", $0)
        }.joined()

        let identity = try authenticateExecutable(
            at: executable,
            expectedSHA256: expected)
        XCTAssertEqual(identity.sha256, expected)
        XCTAssertEqual(identity.size, Int64(initial.count))
        XCTAssertNoThrow(try validateExecutableUnchanged(identity))

        try Data("proof-binary-v2".utf8).write(to: executable)
        XCTAssertThrowsError(
            try validateExecutableUnchanged(identity)
        ) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCLIError,
                .executableIdentityMismatch)
        }
    }

    func testFreshOutputBoundaryIsExclusiveAndPreservesTerminalFiles()
        throws
    {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: parent) }
        let output = parent.appendingPathComponent(
            "proof", isDirectory: true)

        let boundary = try ExactPrefixProofOutputBoundary.claim(
            directoryPath: output.path)
        let status = ExactPrefixProofOutputStatus(
            state: .running,
            processID: 123,
            completedCases: 2,
            totalCases: ExactPrefixProofCaseID.requiredOrder.count,
            elapsedSeconds: 1.5,
            harnessGitSHA: harnessSHA,
            executableSHA256: binarySHA,
            modelID: "qwen3-32b",
            sourceRevision: harnessSHA,
            workloadNonce: "proof-nonce",
            error: nil)
        try boundary.writeStatus(status)
        try boundary.writeImmutable(
            Data("{\"evidence\":true}\n".utf8),
            filename: "evidence.json")
        try boundary.writeImmutable(
            Data("{\"complete\":true}\n".utf8),
            filename: "completion.json")

        XCTAssertThrowsError(try ExactPrefixProofOutputBoundary.claim(
            directoryPath: output.path))
        XCTAssertThrowsError(try boundary.writeImmutable(
            Data("replacement".utf8),
            filename: "evidence.json"))
        XCTAssertEqual(
            try JSONDecoder().decode(
                ExactPrefixProofOutputStatus.self,
                from: Data(contentsOf: output.appendingPathComponent(
                    "status.json"))),
            status)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: output.appendingPathComponent(".lock").path))
    }

    func testClaimedBoundaryWritesStayBoundToClaimedDirectoryInode()
        throws
    {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: parent) }
        let output = parent.appendingPathComponent(
            "proof", isDirectory: true)
        let moved = parent.appendingPathComponent(
            "claimed-proof", isDirectory: true)
        let boundary = try ExactPrefixProofOutputBoundary.claim(
            directoryPath: output.path)

        try FileManager.default.moveItem(at: output, to: moved)
        try FileManager.default.createDirectory(
            at: output, withIntermediateDirectories: false)
        try boundary.writeStatus(ExactPrefixProofOutputStatus(
            state: .running,
            processID: 123,
            completedCases: 0,
            totalCases: ExactPrefixProofCaseID.requiredOrder.count,
            elapsedSeconds: 1,
            harnessGitSHA: harnessSHA,
            executableSHA256: binarySHA,
            modelID: "qwen3-32b",
            sourceRevision: harnessSHA,
            workloadNonce: "proof-nonce",
            error: nil))
        try boundary.writeImmutable(
            Data("{\"evidence\":true}\n".utf8),
            filename: "evidence.json")

        XCTAssertTrue(FileManager.default.fileExists(
            atPath: moved.appendingPathComponent("status.json").path))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: moved.appendingPathComponent("evidence.json").path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: output.appendingPathComponent("status.json").path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: output.appendingPathComponent("evidence.json").path))
    }

    func testCompletedArtifactAuthenticatesAndTamperingFailsClosed()
        throws
    {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: parent) }
        let output = parent.appendingPathComponent(
            "proof", isDirectory: true)
        let boundary = try ExactPrefixProofOutputBoundary.claim(
            directoryPath: output.path)
        let executableURL = parent.appendingPathComponent(
            "proof-harness")
        let executableData = Data("proof-harness-v1".utf8)
        try executableData.write(to: executableURL)
        let executableSHA = SHA256.hash(data: executableData).map {
            String(format: "%02x", $0)
        }.joined()
        let executableIdentity = try authenticateExecutable(
            at: executableURL,
            expectedSHA256: executableSHA)
        let evidence = try proofEvidence(
            expectedExecutableSHA256: executableSHA)
        let runtime = try ExactPrefixProofRuntimeMetadata(
            capturedAtUTC: "2026-07-23T12:34:56Z",
            hardwareChip: "Apple Test",
            hardwareRAMBytes: 128 << 30,
            hardwareOS: "macOS test",
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision:
                "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
            hostEnvironment: BenchQualificationWarmupEnvironment(
                before: hostSnapshot(timestamp: 1),
                after: hostSnapshot(timestamp: 2)))
        let artifact = try ExactPrefixProofArtifact(
            runtime: runtime,
            evidence: evidence)
        _ = try publishExactPrefixProofArtifact(
            artifact,
            boundary: boundary,
            authenticatedExecutableIdentity: executableIdentity)
        try boundary.writeStatus(ExactPrefixProofOutputStatus(
            state: .complete,
            processID: 123,
            completedCases:
                ExactPrefixProofCaseID.requiredOrder.count,
            totalCases:
                ExactPrefixProofCaseID.requiredOrder.count,
            elapsedSeconds: 4,
            harnessGitSHA: harnessSHA,
            executableSHA256:
                evidence.expectedExecutableSHA256,
            modelID: evidence.modelID,
            sourceRevision: evidence.sourceRevision,
            workloadNonce: evidence.workloadNonce,
            error: nil))

        XCTAssertEqual(
            try validateExactPrefixProofOutputDirectory(at: output),
            artifact)

        let evidenceURL = output.appendingPathComponent(
            "evidence.json")
        try FileManager.default.removeItem(at: evidenceURL)
        try Data("{\"tampered\":true}\n".utf8).write(
            to: evidenceURL)
        XCTAssertThrowsError(
            try validateExactPrefixProofOutputDirectory(at: output))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCLIError,
                .evidenceHashMismatch)
        }
    }

    func testRuntimeMetadataAndValidatorArgumentsFailClosed() throws {
        let runtime = try ExactPrefixProofRuntimeMetadata(
            capturedAtUTC: "2026-07-23T12:34:56Z",
            hardwareChip: "Apple Test",
            hardwareRAMBytes: 128 << 30,
            hardwareOS: "macOS test",
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision:
                "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
            hostEnvironment: BenchQualificationWarmupEnvironment(
                before: hostSnapshot(timestamp: 1),
                after: hostSnapshot(timestamp: 2)))
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(runtime))
                as? [String: Any])
        var environment = try XCTUnwrap(
            object["hostEnvironment"] as? [String: Any])
        var before = try XCTUnwrap(
            environment["before"] as? [String: Any])
        before["powerSource"] = "battery"
        environment["before"] = before
        object["hostEnvironment"] = environment
        let tampered = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys])

        XCTAssertThrowsError(try JSONDecoder().decode(
            ExactPrefixProofRuntimeMetadata.self,
            from: tampered))
        XCTAssertThrowsError(try runExactPrefixProofValidation([
            "--output", "relative/proof",
        ]))
        XCTAssertThrowsError(try runExactPrefixProofValidation([
            "--output", "/absolute/proof", "--extra", "value",
        ]))
    }

    private func parse(
        arguments: [String],
        actualHarnessSHA: String? = nil,
        actualExecutableSHA256: String? = nil
    ) throws -> ExactPrefixProofCommand {
        try parseExactPrefixProofCommand(
            arguments: arguments,
            actualHarnessSHA: actualHarnessSHA ?? harnessSHA,
            actualExecutableSHA256:
                actualExecutableSHA256 ?? binarySHA,
            admission: admission())
    }

    private func validArguments() -> [String] {
        [
            "--model", "/models/qwen",
            "--model-id", "qwen3-32b",
            "--source-revision", checkpointSHA,
            "--expected-harness-git-sha", harnessSHA,
            "--expected-binary-sha256", binarySHA,
            "--checkpoint-content-sha256", checkpointSHA,
            "--tokenizer-sha256", tokenizerSHA,
            "--workload-nonce", "proof-nonce",
            "--output", "/proofs/qwen-v1",
            "--max-tokens", "16",
            "--prompt-repeat", "8",
            "--prefix-max-entries", "8",
            "--prefix-max-retained-bytes", "4294967296",
            "--prefix-minimum-reusable-tokens", "16",
            "--template-max-entries", "16",
            "--template-max-retained-bytes", "1048576",
            "--memory-limit-bytes", "103079215104",
            "--cache-limit-bytes", "8589934592",
        ]
    }

    private func admission()
        throws -> CompressedKVAttentionRuntimeAdmission
    {
        let object: [String: Any] = [
            "family": "qwen3",
            "modelType": "qwen3",
            "architecture": "Qwen3ForCausalLM",
            "modelConfigHash": "0123456789abcdef",
            "modelConfigSHA256": String(repeating: "e", count: 64),
            "checkpointManifestHash": "fedcba9876543210",
            "checkpointContentSHA256": checkpointSHA,
            "tokenizerSHA256": tokenizerSHA,
            "layerCount": 64,
            "queryHeadCount": 64,
            "kvHeadCount": 8,
            "headDimension": 128,
            "maxPositionEmbeddings": 40_960,
            "modelNativeDType": "bfloat16",
        ]
        return try JSONDecoder().decode(
            CompressedKVAttentionRuntimeAdmission.self,
            from: JSONSerialization.data(
                withJSONObject: object,
                options: [.sortedKeys]))
            .validatedForEvidence()
    }

    private func proofEvidence(
        expectedExecutableSHA256: String? = nil
    ) throws -> ExactPrefixProofEvidence {
        let exactPolicy = try ExactPrefixCachePolicy(
            maxEntries: 8,
            maxRetainedBytes: 4 << 30,
            minimumReusableTokens: 16)
        let templatePolicy = try TemplateTokenCachePolicy(
            maxEntries: 16,
            maxRetainedBytes: 1 << 20)
        return try ExactPrefixProofEvidence(
            modelID: "qwen3-32b",
            sourceRevision: checkpointSHA,
            admission: admission(),
            expectedHarnessSHA: harnessSHA,
            expectedExecutableSHA256:
                expectedExecutableSHA256 ?? binarySHA,
            workloadNonce: "proof-nonce",
            maxTokens: 16,
            promptRepeat: 8,
            exactPrefixCachePolicy: exactPolicy,
            templateTokenCachePolicy: templatePolicy,
            memoryLimitBytes: 8 << 30,
            cacheLimitBytes: 1 << 20,
            requestContext: try proofRequestContext(),
            warmup: ExactPrefixProofWarmupEvidence(
                durationSeconds: 0.25,
                before: try prefixSnapshot(),
                after: try prefixSnapshot()),
            cases: try ExactPrefixProofCaseID.requiredOrder.map(
                proofCase),
            terminalNonPartial: true)
    }

    private func proofCase(
        _ id: ExactPrefixProofCaseID
    ) throws -> ExactPrefixProofCaseEvidence {
        let group: String
        let outcome: PrefixCacheRequestOutcome?
        let templateReceipt: ExactPrefixProofCacheReceipt
        let requestStartSeconds: Double
        let promptTokenCount: Int
        switch id {
        case .coldControlA:
            group = "a"; outcome = nil
            templateReceipt = .miss; requestStartSeconds = 0.050
            promptTokenCount = 64
        case .coldCommitA:
            group = "a"; outcome = .miss
            templateReceipt = .miss; requestStartSeconds = 0.051
            promptTokenCount = 64
        case .exactHitA:
            group = "a"; outcome = .exactHit
            templateReceipt = .hit; requestStartSeconds = 0.010
            promptTokenCount = 64
        case .partialControl:
            group = "partial"; outcome = nil
            templateReceipt = .miss; requestStartSeconds = 0.055
            promptTokenCount = 128
        case .partialHit:
            group = "partial"; outcome = .partialHit
            templateReceipt = .hit; requestStartSeconds = 0.025
            promptTokenCount = 128
        case .coldCommitB:
            group = "b"; outcome = .miss
            templateReceipt = .miss; requestStartSeconds = 0.052
            promptTokenCount = 96
        case .returnHitA:
            group = "a"; outcome = .exactHit
            templateReceipt = .hit; requestStartSeconds = 0.011
            promptTokenCount = 64
        case .pressureEvictedA:
            group = "a"; outcome = .miss
            templateReceipt = .hit; requestStartSeconds = 0.053
            promptTokenCount = 64
        case .postWarmupControl:
            group = "warm"; outcome = nil
            templateReceipt = .miss; requestStartSeconds = 0.054
            promptTokenCount = 128
        case .postWarmupMiss:
            group = "warm"; outcome = .miss
            templateReceipt = .miss; requestStartSeconds = 0.050
            promptTokenCount = 128
        case .postWarmupHit:
            group = "warm"; outcome = .exactHit
            templateReceipt = .hit; requestStartSeconds = 0.012
            promptTokenCount = 128
        }
        let metrics: RequestStartMetrics?
        if let outcome {
            let readTokens: Int
            switch outcome {
            case .exactHit: readTokens = promptTokenCount
            case .partialHit: readTokens = 64
            case .disabled, .miss, .rejected: readTokens = 0
            }
            metrics = try RequestStartMetrics(
                promptTokenCount: promptTokenCount,
                cacheReadTokenCount: readTokens,
                physicalPrefillTokenCount:
                    promptTokenCount - readTokens,
                prefixCacheOutcome: outcome,
                templateTokenCacheHit:
                    templateReceipt == .hit,
                templateSeconds: 0.001,
                tokenizeSeconds: 0.001,
                lookupSeconds: 0.001,
                restoreSeconds:
                    outcome == .exactHit ? 0.001 : 0,
                prefillSeconds:
                    outcome == .exactHit ? 0.001 : 0.01,
                retainedBytes: 512,
                entryCount: 1,
                evictionCount:
                    id == .pressureEvictedA ? 1 : 0,
                runtimeIdentity:
                    try ExactPrefixDenseRuntimeIdentityEvidence(
                        observedDenseHalfDType: .float16),
                eagerWarmupSeconds:
                    id == .postWarmupMiss
                        || id == .postWarmupHit
                        ? 0.25 : nil)
        } else {
            metrics = nil
        }
        let reusedPrefix: (sha256: String?, count: Int)
        switch id {
        case .exactHitA, .partialHit, .returnHitA:
            reusedPrefix = (digest("prompt-a"), 64)
        case .postWarmupHit:
            reusedPrefix = (digest("prompt-warm"), 128)
        case .coldControlA, .coldCommitA, .partialControl,
            .coldCommitB, .pressureEvictedA, .postWarmupControl,
            .postWarmupMiss:
            reusedPrefix = (nil, 0)
        }
        return try ExactPrefixProofCaseEvidence(
            caseID: id,
            promptTokenIDsSHA256: digest("prompt-\(group)"),
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: digest("tokens-\(group)"),
            outputSHA256: digest("output-\(group)"),
            referenceGeneratedTokenIDsSHA256:
                digest("tokens-\(group)"),
            referenceOutputSHA256: digest("output-\(group)"),
            generatedTokenCount: 8,
            timing: ExactPrefixProofCaseTiming(
                requestStartSeconds: requestStartSeconds,
                ttftSeconds: requestStartSeconds + 0.01,
                decodeSeconds: 0.02,
                totalSeconds: requestStartSeconds + 0.03),
            requestStartMetrics: metrics,
            memoryEvidence: try BenchRunMemoryEvidence(samples: [
                ServiceMemorySample(
                    timestamp: 1,
                    physicalFootprintBytes: 1_000,
                    mlxActiveBytes: 100,
                    mlxCacheBytes: 100,
                    mlxPeakBytes: 200),
                ServiceMemorySample(
                    timestamp: 2,
                    physicalFootprintBytes: 1_100,
                    mlxActiveBytes: 110,
                    mlxCacheBytes: 110,
                    mlxPeakBytes: 220),
            ]),
            templateTokenCacheReceipt: templateReceipt,
            reusedPrefixTokenIDsSHA256: reusedPrefix.sha256,
            reusedPrefixTokenCount: reusedPrefix.count,
            requestContext:
                outcome == nil ? nil : try proofRequestContext())
    }

    private func proofRequestContext()
        throws -> ExactPrefixRequestContext
    {
        try ExactPrefixRequestContext(
            isolationNamespaceSHA256: digest("namespace"),
            promptTemplateSHA256: digest("template"),
            toolsSHA256: digest("tools"))
    }

    private func hostSnapshot(
        timestamp: Double
    ) -> BenchQualificationHostSnapshot {
        BenchQualificationHostSnapshot(
            monotonicTimestampSeconds: timestamp,
            residentSizeBytes: 1_000,
            physicalFootprintBytes: 900,
            lowPowerModeEnabled: false,
            powerSource: .acPower,
            thermalState: .nominal)
    }

    private func prefixSnapshot() throws -> ExactPrefixCacheSnapshot {
        try JSONDecoder().decode(
            ExactPrefixCacheSnapshot.self,
            from: Data("""
                {"entryCount":0,"reservationCount":0,"retainedBytes":0,\
                "reservedBytes":0,"hitCount":0,"missCount":0,\
                "evictionCount":0}
                """.utf8))
    }

    private func digest(_ value: String) -> String {
        sha256Hex(Data(value.utf8))
    }
}
