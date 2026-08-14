import CryptoKit
import Foundation
import HarnessCore
import MLX
import XCTest

@testable import fastmlx_harness

final class CompressedAttentionProbeCLITests: XCTestCase {
    private let cleanSHA = String(repeating: "a", count: 40)

    func testTinyProbePublishesValidatedEvidenceAndCompletionProgress()
        async throws
    {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = directory.appendingPathComponent(
            "model", isDirectory: true)
        try FileManager.default.createDirectory(
            at: model, withIntermediateDirectories: false)
        try writeModelFixture(to: model)
        let evidenceURL = directory.appendingPathComponent("probe.json")
        let progressURL = directory.appendingPathComponent(
            "probe.progress.json")
        let previousMemoryLimit = Memory.memoryLimit
        let previousCacheLimit = Memory.cacheLimit
        defer {
            Memory.memoryLimit = previousMemoryLimit
            Memory.cacheLimit = previousCacheLimit
        }
        let command = try CompressedAttentionProbeCommand(
            arguments: [
                "--model", model.path,
                "--model-id", "mlx-community/Qwen3-32B-4bit",
                "--operation", "fp16-sdpa",
                "--layout", "fp16",
                "--context-tokens", "64",
                "--query-tokens", "1",
                "--prefill-chunk-tokens", "32",
                "--output-tokens", "16",
                "--stop-token-ids", "",
                "--batch-size", "1",
                "--query-heads", "8",
                "--kv-heads", "2",
                "--head-dimension", "128",
                "--dtype", "float16",
                "--mask", "causal",
                "--warmup-runs", "1",
                "--measured-runs", "1",
                "--seed", "7",
                "--workload-nonce", "probe-cli-e2e",
                "--qualification-evidence", "false",
                "--evidence", evidenceURL.path,
                "--progress", progressURL.path,
                "--memory-limit-bytes", String(1 << 30),
                "--cache-limit-bytes", String(64 << 20),
                "--wired-limit-bytes", "0",
            ],
            harnessGitSHA: cleanSHA)
        let package = CompressedAttentionProbePackageIdentity(
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision:
                "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
            swiftVersion: "Swift 6 test",
            harnessBuildConfiguration: "Debug")

        let control = try await CompressedAttentionProbeRunner()
            .runFixture(plan: command.plan)
        XCTAssertTrue(
            control.structuralEquivalent,
            "max=\(control.maxAbsoluteError) ratio=\(control.maximumToleranceRatio)")
        XCTAssertTrue(
            control.top1Matches,
            "output=\(control.outputTop1Index) oracle=\(control.oracleTop1Index)")

        let evidence = try await executeCompressedAttentionProbe(
            command: command,
            packageIdentity: package)
        XCTAssertNoThrow(try evidence.validated())
        XCTAssertEqual(
            evidence.evidenceKind,
            .checkpointAuthenticatedSyntheticGeometry)
        XCTAssertEqual(evidence.rows.count, 2)
        XCTAssertEqual(
            evidence.rows.map(\.role),
            [.candidate, .fp16Reference])
        for row in evidence.rows {
            let receipts = try XCTUnwrap(row.receipts)
            XCTAssertEqual(receipts.mlxMemory.before.peakBytes, 0)
            let derivedWorkspace = try CompressedAttentionProbeWorkspaceBytes
                .derive(
                    persistentKVBytes: receipts.bytes.persistentKVBytes,
                    materializationBytes: receipts.bytes.materializationBytes,
                    mlxMemory: receipts.mlxMemory)
            XCTAssertEqual(
                receipts.bytes.otherWorkspaceBytes,
                derivedWorkspace.otherWorkspaceBytes)
            XCTAssertEqual(
                receipts.bytes.peakTemporaryBytes,
                derivedWorkspace.peakTemporaryBytes)
            XCTAssertEqual(receipts.bytes.totalBytes, derivedWorkspace.totalBytes)
            XCTAssertGreaterThan(receipts.bytes.otherWorkspaceBytes, 0)
        }
        let persisted = try JSONDecoder().decode(
            CompressedAttentionProbeEvidence.self,
            from: Data(contentsOf: evidenceURL))
        XCTAssertEqual(persisted, evidence)
        let progress = try JSONDecoder().decode(
            CompressedAttentionProbeProgress.self,
            from: Data(contentsOf: progressURL))
        XCTAssertEqual(progress.status, .complete)
        XCTAssertEqual(progress.completedWarmupRuns, 2)
        XCTAssertEqual(progress.completedMeasuredRows, 2)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: evidenceURL.deletingLastPathComponent()
                .appendingPathComponent(
                    ".probe.json.fastmlx-compressed-attention.lock").path))
    }

    func testOutputLeaseSerializesWritersAndPreservesFreshEvidence() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let evidence = directory.appendingPathComponent("probe.json")
        let progress = directory.appendingPathComponent("probe.progress.json")

        let lease = try CompressedAttentionProbeOutputLease.acquire(
            evidencePath: evidence.path,
            progressPath: progress.path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: lease.lockURL.path))
        XCTAssertEqual(lease.lockURLs.count, 2)
        XCTAssertTrue(lease.lockURLs.allSatisfy {
            FileManager.default.fileExists(atPath: $0.path)
        })
        XCTAssertEqual(
            try String(contentsOf: lease.lockURL, encoding: .utf8),
            "\(ProcessInfo.processInfo.processIdentifier)\n")
        XCTAssertThrowsError(
            try CompressedAttentionProbeOutputLease.acquire(
                evidencePath: evidence.path,
                progressPath: progress.path)
        ) { error in
            XCTAssertEqual(
                error as? CompressedAttentionProbeCLIError,
                .outputNotFresh(evidence.path))
        }

        let running = CompressedAttentionProbeProgress(
            schemaVersion: 1,
            status: .running,
            completedWarmupRuns: 1,
            completedMeasuredRows: 0,
            totalMeasuredRows: 6,
            activeBlockIndex: 0,
            activeRole: .candidate,
            elapsedSeconds: 1.25,
            processResidentBytes: 123_456,
            processPhysicalFootprintBytes: 120_000,
            harnessGitSHA: String(repeating: "a", count: 40),
            workloadNonce: "probe-cli-test")
        try lease.writeProgress(running)
        let decoded = try JSONDecoder().decode(
            CompressedAttentionProbeProgress.self,
            from: Data(contentsOf: progress))
        XCTAssertEqual(decoded, running)

        let payload = Data("{\"complete\":true}\n".utf8)
        try lease.writeEvidence(payload)
        XCTAssertEqual(try Data(contentsOf: evidence), payload)
        XCTAssertThrowsError(try lease.writeEvidence(payload)) { error in
            XCTAssertEqual(
                error as? CompressedAttentionProbeCLIError,
                .evidenceAlreadyWritten(evidence.path))
        }

        lease.release()
        XCTAssertTrue(lease.lockURLs.allSatisfy {
            !FileManager.default.fileExists(atPath: $0.path)
        })
        XCTAssertThrowsError(
            try CompressedAttentionProbeOutputLease.acquire(
                evidencePath: evidence.path,
                progressPath: progress.path)
        ) { error in
            XCTAssertEqual(
                error as? CompressedAttentionProbeCLIError,
                .outputNotFresh(evidence.path))
        }
    }

    func testOutputLeaseSerializesDifferentEvidenceWritersSharingProgress()
        throws
    {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let firstEvidence = directory.appendingPathComponent("first.json")
        let secondEvidence = directory.appendingPathComponent("second.json")
        let sharedProgress = directory.appendingPathComponent("progress.json")

        let lease = try CompressedAttentionProbeOutputLease.acquire(
            evidencePath: firstEvidence.path,
            progressPath: sharedProgress.path)
        defer { lease.release() }

        XCTAssertThrowsError(
            try CompressedAttentionProbeOutputLease.acquire(
                evidencePath: secondEvidence.path,
                progressPath: sharedProgress.path)
        ) { error in
            XCTAssertEqual(
                error as? CompressedAttentionProbeCLIError,
                .outputNotFresh(secondEvidence.path))
        }
    }

    func testOutputLeaseRejectsSymlinkAndPathAliasBeforeWork() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let evidence = directory.appendingPathComponent("probe.json")
        let progress = directory.appendingPathComponent("probe.progress.json")
        let target = directory.appendingPathComponent("target.json")
        XCTAssertTrue(FileManager.default.createFile(
            atPath: target.path, contents: Data()))
        try FileManager.default.createSymbolicLink(
            at: evidence, withDestinationURL: target)

        XCTAssertThrowsError(
            try CompressedAttentionProbeOutputLease.acquire(
                evidencePath: evidence.path,
                progressPath: progress.path)
        ) { error in
            XCTAssertEqual(
                error as? CompressedAttentionProbeCLIError,
                .outputNotFresh(evidence.path))
        }
        XCTAssertThrowsError(
            try CompressedAttentionProbeOutputLease.acquire(
                evidencePath: progress.path,
                progressPath: progress.path)
        ) { error in
            XCTAssertEqual(
                error as? CompressedAttentionProbeCLIError,
                .outputPathCollision(progress.path))
        }
    }

    func testModelIdentityUsesCryptographicConfigTokenizerAndManifestHashes()
        throws
    {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = Data("{\"model_type\":\"qwen3\"}".utf8)
        let tokenizer = Data("{\"version\":\"1\"}".utf8)
        let tokenizerConfig = Data("{\"chat_template\":\"v1\"}".utf8)
        try config.write(to: directory.appendingPathComponent("config.json"))
        try tokenizer.write(
            to: directory.appendingPathComponent("tokenizer.json"))
        try tokenizerConfig.write(
            to: directory.appendingPathComponent("tokenizer_config.json"))
        try Data([0, 1, 2]).write(
            to: directory.appendingPathComponent("model-00001-of-00001.safetensors"))
        let originalWeights = [
            (
                name: "model-00001-of-00001.safetensors",
                contents: Data([0, 1, 2])
            ),
        ]

        let first = try compressedAttentionProbeModelIdentity(
            modelID: "mlx-community/Qwen3-32B-4bit",
            modelPath: directory.path)
        XCTAssertEqual(first.modelConfigSHA256, sha256Hex(config))
        XCTAssertEqual(first.tokenizerConfigSHA256, sha256Hex(tokenizerConfig))
        XCTAssertEqual(
            first.checkpointManifestSHA256,
            phase0CheckpointManifestSHA256(
                config: config,
                index: nil,
                weights: originalWeights))
        XCTAssertEqual(
            first.checkpointManifestSHA256,
            try ProvenanceCLI.fullContentCheckpointManifestSHA256(
                at: directory.path,
                exactConfigData: config))
        assertSHA256(first.tokenizerSHA256)

        try Data([3, 1, 2]).write(
            to: directory.appendingPathComponent("model-00001-of-00001.safetensors"))
        let replacedWeights = [
            (
                name: "model-00001-of-00001.safetensors",
                contents: Data([3, 1, 2])
            ),
        ]
        let changed = try compressedAttentionProbeModelIdentity(
            modelID: "mlx-community/Qwen3-32B-4bit",
            modelPath: directory.path)
        XCTAssertNotEqual(
            first.checkpointManifestSHA256,
            changed.checkpointManifestSHA256)
        XCTAssertEqual(
            changed.checkpointManifestSHA256,
            phase0CheckpointManifestSHA256(
                config: config,
                index: nil,
                weights: replacedWeights))
        XCTAssertEqual(first.tokenizerSHA256, changed.tokenizerSHA256)
    }

    func testTokenizerManifestAuthenticatesStandaloneChatTemplates()
        throws
    {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{\"version\":\"1\"}".utf8).write(
            to: directory.appendingPathComponent("tokenizer.json"))

        let tokenizerOnly = try ProvenanceCLI.tokenizerManifestSHA256(
            at: directory.path)
        try Data("not tokenizer input".utf8).write(
            to: directory.appendingPathComponent("README.md"))
        XCTAssertEqual(
            tokenizerOnly,
            try ProvenanceCLI.tokenizerManifestSHA256(at: directory.path))

        let jinjaURL = directory.appendingPathComponent("chat_template.jinja")
        try Data("{{ messages | length }}".utf8).write(to: jinjaURL)
        let withJinja = try ProvenanceCLI.tokenizerManifestSHA256(
            at: directory.path)
        XCTAssertNotEqual(tokenizerOnly, withJinja)

        try Data("{{ messages[0].content }}".utf8).write(to: jinjaURL)
        let changedJinja = try ProvenanceCLI.tokenizerManifestSHA256(
            at: directory.path)
        XCTAssertNotEqual(withJinja, changedJinja)

        let jsonURL = directory.appendingPathComponent("chat_template.json")
        try Data("{\"chat_template\":\"{{ messages }}\"}".utf8).write(
            to: jsonURL)
        XCTAssertNotEqual(
            changedJinja,
            try ProvenanceCLI.tokenizerManifestSHA256(at: directory.path))
    }

    func testFullContentCheckpointManifestAuthenticatesSafeShardAliases()
        throws
    {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let target = directory.appendingPathComponent("target.bin")
        let regularModel = directory.appendingPathComponent(
            "regular", isDirectory: true)
        let aliasModel = directory.appendingPathComponent(
            "alias", isDirectory: true)
        let renamedAliasModel = directory.appendingPathComponent(
            "renamed-alias", isDirectory: true)
        for model in [regularModel, aliasModel, renamedAliasModel] {
            try FileManager.default.createDirectory(
                at: model,
                withIntermediateDirectories: false)
            try Data("{\"model_type\":\"qwen3\"}".utf8).write(
                to: model.appendingPathComponent("config.json"))
        }
        try Data([9, 8, 7, 6]).write(to: target)
        try Data([9, 8, 7, 6]).write(
            to: regularModel.appendingPathComponent(
                "model-00001-of-00001.safetensors"))
        try FileManager.default.createSymbolicLink(
            at: aliasModel.appendingPathComponent(
                "model-00001-of-00001.safetensors"),
            withDestinationURL: target)
        try FileManager.default.createSymbolicLink(
            at: renamedAliasModel.appendingPathComponent(
                "model-00002-of-00002.safetensors"),
            withDestinationURL: target)

        let regular = try ProvenanceCLI.fullContentCheckpointManifestSHA256(
            at: regularModel.path)
        let alias = try ProvenanceCLI.fullContentCheckpointManifestSHA256(
            at: aliasModel.path)
        let renamedAlias = try ProvenanceCLI
            .fullContentCheckpointManifestSHA256(at: renamedAliasModel.path)

        XCTAssertEqual(regular, alias)
        XCTAssertNotEqual(alias, renamedAlias)
    }

    func testFullContentCheckpointManifestRejectsNonRegularShardAlias()
        throws
    {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = directory.appendingPathComponent("model", isDirectory: true)
        let target = directory.appendingPathComponent(
            "target-directory", isDirectory: true)
        try FileManager.default.createDirectory(
            at: model,
            withIntermediateDirectories: false)
        try FileManager.default.createDirectory(
            at: target,
            withIntermediateDirectories: false)
        try Data("{\"model_type\":\"qwen3\"}".utf8).write(
            to: model.appendingPathComponent("config.json"))
        let logicalShard = model.appendingPathComponent(
            "model-00001-of-00001.safetensors")
        try FileManager.default.createSymbolicLink(
            at: logicalShard,
            withDestinationURL: target)

        XCTAssertThrowsError(
            try ProvenanceCLI.fullContentCheckpointManifestSHA256(
                at: model.path)
        ) { error in
            guard case let ProvenanceCLI.EvidenceIdentityError
                .invalidCheckpointWeight(path) = error,
                path == logicalShard.path
            else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testModelIdentityFailsClosedWhenRequiredFilesAreMissing() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{}".utf8).write(
            to: directory.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(
            to: directory.appendingPathComponent("tokenizer.json"))
        try Data("{}".utf8).write(
            to: directory.appendingPathComponent("tokenizer_config.json"))

        XCTAssertThrowsError(try compressedAttentionProbeModelIdentity(
            modelID: "Qwen3-32B",
            modelPath: directory.path)) { error in
            XCTAssertEqual(
                error as? CompressedAttentionProbeCLIError,
                .missingCheckpointWeights(directory.path))
        }
    }

    func testModelGeometryMustMatchTheAuthenticatedProbePlan() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("""
        {
          "hidden_size": 1024,
          "max_position_embeddings": 8192,
          "num_attention_heads": 8,
          "num_key_value_heads": 2
        }
        """.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        let matching = try CompressedAttentionProbePlan(
            operation: .fp16SDPA,
            contextTokens: 64,
            queryTokens: 1,
            prefillChunkTokens: 32,
            outputTokens: 16,
            stopTokenIDs: [],
            batchSize: 1,
            queryHeadCount: 8,
            kvHeadCount: 2,
            headDimension: 128,
            dtype: .float16,
            mask: .causal,
            layout: .fp16,
            warmupRuns: 1,
            measuredRuns: 1,
            seed: 7,
            workloadNonce: "geometry-match",
            harnessGitSHA: cleanSHA,
            qualificationEvidence: false)
        XCTAssertNoThrow(try validateCompressedAttentionProbeModelGeometry(
            plan: matching,
            modelPath: directory.path))

        let mismatched = try CompressedAttentionProbePlan(
            operation: .fp16SDPA,
            contextTokens: 64,
            queryTokens: 1,
            prefillChunkTokens: 32,
            outputTokens: 16,
            stopTokenIDs: [],
            batchSize: 1,
            queryHeadCount: 16,
            kvHeadCount: 2,
            headDimension: 128,
            dtype: .float16,
            mask: .causal,
            layout: .fp16,
            warmupRuns: 1,
            measuredRuns: 1,
            seed: 7,
            workloadNonce: "geometry-mismatch",
            harnessGitSHA: cleanSHA,
            qualificationEvidence: false)
        XCTAssertThrowsError(
            try validateCompressedAttentionProbeModelGeometry(
                plan: mismatched,
                modelPath: directory.path)
        ) { error in
            XCTAssertEqual(
                error as? CompressedAttentionProbeCLIError,
                .modelGeometryMismatch("queryHeadCount"))
        }
    }

    func testModelGeometryRejectsContextPlusOutputPastAuthenticatedWindow()
        throws
    {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("""
        {
          "hidden_size": 1024,
          "max_position_embeddings": 8192,
          "num_attention_heads": 8,
          "num_key_value_heads": 2
        }
        """.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        let plan = try CompressedAttentionProbePlan(
            operation: .fp16SDPA,
            contextTokens: 8_192,
            queryTokens: 1,
            prefillChunkTokens: 32,
            outputTokens: 16,
            stopTokenIDs: [],
            batchSize: 1,
            queryHeadCount: 8,
            kvHeadCount: 2,
            headDimension: 128,
            dtype: .float16,
            mask: .causal,
            layout: .fp16,
            warmupRuns: 1,
            measuredRuns: 1,
            seed: 7,
            workloadNonce: "geometry-window-overrun",
            harnessGitSHA: cleanSHA,
            qualificationEvidence: false)

        XCTAssertThrowsError(
            try validateCompressedAttentionProbeModelGeometry(
                plan: plan,
                modelPath: directory.path)
        ) { error in
            XCTAssertEqual(
                error as? CompressedAttentionProbeCLIError,
                .modelGeometryMismatch("contextWindowTokens"))
        }
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "compressed-attention-cli-\(UUID().uuidString)",
            isDirectory: true)
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: false)
        return url
    }

    private func writeModelFixture(to directory: URL) throws {
        try Data("""
        {
          "hidden_size": 1024,
          "max_position_embeddings": 8192,
          "model_type": "qwen3",
          "num_attention_heads": 8,
          "num_key_value_heads": 2
        }
        """.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        try Data("{\"version\":\"1\"}".utf8).write(
            to: directory.appendingPathComponent("tokenizer.json"))
        try Data("{\"chat_template\":\"v1\"}".utf8).write(
            to: directory.appendingPathComponent("tokenizer_config.json"))
        try Data([0, 1, 2]).write(
            to: directory.appendingPathComponent(
                "model-00001-of-00001.safetensors"))
    }

    private func assertSHA256(
        _ value: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(value.count, 64, file: file, line: line)
        XCTAssertTrue(
            value.utf8.allSatisfy {
                (48 ... 57).contains($0) || (97 ... 102).contains($0)
            },
            file: file,
            line: line)
    }

    private func phase0CheckpointManifestSHA256(
        config: Data,
        index: Data?,
        weights: [(name: String, contents: Data)]
    ) -> String {
        var hasher = SHA256()
        hasher.update(data: Data(
            "fastmlx-checkpoint-content-manifest-v2\n".utf8))
        updatePhase0ManifestField(config, hasher: &hasher)
        updatePhase0ManifestField(index ?? Data(), hasher: &hasher)
        for weight in weights.sorted(by: { $0.name < $1.name }) {
            updatePhase0ManifestField(
                Data(weight.name.utf8),
                hasher: &hasher)
            var size = UInt64(weight.contents.count).bigEndian
            withUnsafeBytes(of: &size) {
                hasher.update(data: Data($0))
            }
            hasher.update(data: weight.contents)
        }
        return hasher.finalize().map {
            String(format: "%02x", $0)
        }.joined()
    }

    private func updatePhase0ManifestField(
        _ field: Data,
        hasher: inout SHA256
    ) {
        var count = UInt64(field.count).bigEndian
        withUnsafeBytes(of: &count) {
            hasher.update(data: Data($0))
        }
        hasher.update(data: field)
    }
}
