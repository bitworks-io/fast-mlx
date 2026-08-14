import Foundation
import HarnessCore
import XCTest

@testable import fastmlx_harness

final class SealedKLReferenceCLITests: XCTestCase {
    private let hex64 = String(repeating: "a", count: 64)

    func testCapturePlanStrictlyParsesRequiredDefaultsAndRejectsUnknownFlags() throws {
        let plan = try parseSealedKLReferenceCapturePlan(arguments: [
            "--model", "/models/Llama-3.3-70B-Instruct-4bit",
            "--output", "/evidence/sealed-ref",
            "--workload-nonce", "llama-quality-v1",
        ])

        XCTAssertEqual(plan.modelPath, "/models/Llama-3.3-70B-Instruct-4bit")
        XCTAssertEqual(plan.outputPath, "/evidence/sealed-ref")
        XCTAssertEqual(plan.workloadNonce, "llama-quality-v1")
        XCTAssertEqual(plan.corpusPath, "corpus/measurement-corpus-v2.json")
        XCTAssertEqual(plan.positions, 24)
        XCTAssertEqual(plan.longContextSamplePositions, 128)
        XCTAssertEqual(plan.pythonPath, "~/harness-venv/bin/python")
        XCTAssertEqual(plan.scriptPath, "scripts/harness_reference.py")

        XCTAssertThrowsError(try parseSealedKLReferenceCapturePlan(arguments: [
            "--model", "/m",
            "--output", "/o",
        ])) {
            XCTAssertEqual($0 as? SealedKLReferenceCLIError, .missingRequiredFlag("workload-nonce"))
        }
        XCTAssertThrowsError(try parseSealedKLReferenceCapturePlan(arguments: [
            "--model", "/m",
            "--output", "/o",
            "--workload-nonce", "n",
            "--surprise", "x",
        ])) {
            XCTAssertEqual($0 as? SealedKLReferenceCLIError, .unsupportedFlag("surprise"))
        }
        XCTAssertThrowsError(try parseSealedKLReferenceCapturePlan(arguments: [
            "--model", "/m",
            "--output", "/o",
            "--workload-nonce", "n",
            "--positions", "0",
        ])) {
            XCTAssertEqual($0 as? SealedKLReferenceCLIError, .invalidPositiveInteger("positions", "0"))
        }
    }

    func testReplayPlanRequiresManifestDigestAndRejectsBadDigest() throws {
        let plan = try parseSealedKLReferenceReplayPlan(arguments: [
            "--model", "/models/llama",
            "--sealed-reference", "/evidence/sealed-ref",
            "--sealed-reference-sha256", hex64,
            "--workload-nonce", "llama-quality-v1",
        ])

        XCTAssertEqual(plan.modelPath, "/models/llama")
        XCTAssertEqual(plan.directoryPath, "/evidence/sealed-ref")
        XCTAssertEqual(plan.expectedManifestSHA256, hex64)
        XCTAssertEqual(plan.workloadNonce, "llama-quality-v1")

        XCTAssertThrowsError(try parseSealedKLReferenceReplayPlan(arguments: [
            "--model", "/models/llama",
            "--sealed-reference", "/evidence/sealed-ref",
            "--sealed-reference-sha256", String(repeating: "g", count: 64),
            "--workload-nonce", "llama-quality-v1",
        ])) {
            XCTAssertEqual($0 as? SealedKLReferenceCLIError, .invalidSHA256("sealed-reference-sha256"))
        }
    }

    func testKLReferenceRequestSelectsLiveOrExactSealedReplayFailClosed() throws {
        XCTAssertEqual(
            try requestedKLReference(
                CLIFlags(["--model", "/models/llama"]),
                modelPath: "/models/llama",
                sameResolvedModel: true),
            .livePython)

        let sealed = try requestedKLReference(
            CLIFlags([
                "--model", "/models/llama",
                "--sealed-reference", "/evidence/sealed-ref",
                "--sealed-reference-sha256", hex64,
                "--workload-nonce", "llama-quality-v1",
            ]),
            modelPath: "/models/llama",
            sameResolvedModel: true)
        XCTAssertEqual(
            sealed,
            .sealedReplay(SealedKLReferenceReplayPlan(
                modelPath: "/models/llama",
                directoryPath: "/evidence/sealed-ref",
                expectedManifestSHA256: hex64,
                workloadNonce: "llama-quality-v1",
                corpusPath: "corpus/measurement-corpus-v2.json",
                scriptPath: "scripts/harness_reference.py")))

        XCTAssertThrowsError(try requestedKLReference(
            CLIFlags([
                "--model", "/models/llama",
                "--sealed-reference", "/evidence/sealed-ref",
                "--sealed-reference-sha256", hex64,
                "--workload-nonce", "llama-quality-v1",
            ]),
            modelPath: "/models/llama",
            sameResolvedModel: false)) {
            XCTAssertEqual(
                $0 as? SealedKLReferenceCLIError,
                .sealedReferenceModelMismatch)
        }
        XCTAssertThrowsError(try requestedKLReference(
            CLIFlags([
                "--model", "/models/llama",
                "--sealed-reference", "/evidence/sealed-ref",
                "--workload-nonce", "llama-quality-v1",
            ]),
            modelPath: "/models/llama",
            sameResolvedModel: true))
    }

    func testLittleEndianF32EncodingRejectsNonFiniteRows() throws {
        let blob = try sealedKLReferenceLogitsBlob([[1.0, -2.5], [0.25, 4.0]])
        XCTAssertEqual(blob.count, 16)
        XCTAssertEqual(Array(blob.prefix(4)), [0x00, 0x00, 0x80, 0x3f])

        XCTAssertThrowsError(try sealedKLReferenceLogitsBlob([[.nan]])) {
            XCTAssertEqual($0 as? SealedKLReferenceCLIError, .nonFiniteLogit)
        }
    }

    func testBindingUsesStableLabelsAndExactHashes() throws {
        let corpusData = Data(#"{"corpusId":"corpus-a","entries":[{"id":"p","tag":"prose","text":"hello"}]}"#.utf8)
        let corpus = try MeasurementCorpusLoader.load(from: corpusData)
        let binding = try sealedKLReferenceBinding(
            modelPath: "/models/Llama-3.3-70B-Instruct-4bit",
            sourceSnapshot: sourceSnapshot(),
            corpus: corpus,
            corpusRawData: corpusData,
            referenceScriptData: Data("print('reference')\n".utf8),
            referenceRuntimeVersions: SealedKLReferenceRuntimeVersions(
                mlx: "mlx-0.31.6",
                mlxLM: "mlx-lm-0.27.1"),
            workloadNonce: "llama-quality-v1",
            harnessGitSHA: String(repeating: "b", count: 40))

        XCTAssertEqual(binding.model, "Llama-3.3-70B-Instruct-4bit")
        XCTAssertEqual(binding.harnessGitSHA, String(repeating: "b", count: 40))
        XCTAssertEqual(binding.modelConfigSHA256, SealedKLReferenceBundle.sha256Hex(Data("config".utf8)))
        XCTAssertEqual(binding.corpusContentHash, corpus.contentHash)
        XCTAssertEqual(binding.corpusRawFileSHA256, SealedKLReferenceBundle.sha256Hex(corpusData))
        XCTAssertEqual(binding.tokenizer, "hf-tokenizer-\(String(repeating: "4", count: 12))")
        XCTAssertEqual(binding.harness, sealedKLReferenceHarnessIdentity)
    }

    func testReplayPreparationFailsOnManifestDigestAndCorpusQueryMismatch() throws {
        let root = try temporaryDirectory()
        let directory = root.appendingPathComponent("sealed-ref", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let rows = [[Float(0.0), Float(1.0)]]
        let blob = try sealedKLReferenceLogitsBlob(rows)
        let entry = SealedKLReferenceEntry(
            id: "prose-1",
            tag: "prose",
            promptTokenIDs: [1, 2],
            continuationTokenIDs: [3],
            samplePositions: nil,
            logitsFile: "prose-1.f32",
            logitsSHA256: SealedKLReferenceBundle.sha256Hex(blob),
            rowCount: 1,
            vocabSize: 2,
            byteCount: blob.count)
        let manifest = SealedKLReferenceManifest(
            schema: "sealed-kl-reference",
            version: 1,
            identity: expectedBinding(),
            corpus: "corpus-a",
            referenceVersion: "reference-v1",
            maxTokens: 24,
            sampleSize: 128,
            entries: [entry])
        try writeSealedKLReferenceBundleAtomically(
            manifest: manifest,
            blobs: ["prose-1.f32": blob],
            to: directory)

        let corpus = MeasurementCorpus(
            corpusId: "corpus-a",
            entries: [MeasurementCorpusEntry(id: "prose-1", tag: .prose, text: "hello")],
            contentHash: expectedBinding().corpusContentHash)
        let manifestData = try Data(contentsOf: directory.appendingPathComponent("manifest.json"))
        let manifestSHA = SealedKLReferenceBundle.sha256Hex(manifestData)

        XCTAssertThrowsError(try loadSealedKLReferenceReplay(
            directory: directory,
            expectedManifestSHA256: String(repeating: "f", count: 64),
            expectedBinding: expectedBinding(),
            sourceSnapshot: try sourceSnapshot(),
            expectedCorpus: "corpus-a",
            corpus: corpus,
            tokenize: { _ in [1, 2] }))

        XCTAssertThrowsError(try loadSealedKLReferenceReplay(
            directory: directory,
            expectedManifestSHA256: manifestSHA,
            expectedBinding: expectedBinding(),
            sourceSnapshot: try sourceSnapshot(),
            expectedCorpus: "corpus-a",
            corpus: corpus,
            tokenize: { _ in [9, 9] })) {
            XCTAssertEqual($0 as? SealedKLReferenceCLIError, .corpusTokenizationMismatch("prose-1"))
        }

        try Data(repeating: 0, count: 1_048_576).write(
            to: directory.appendingPathComponent("undeclared-large.bin"))
        XCTAssertThrowsError(try loadSealedKLReferenceReplay(
            directory: directory,
            expectedManifestSHA256: manifestSHA,
            expectedBinding: expectedBinding(),
            sourceSnapshot: try sourceSnapshot(),
            expectedCorpus: "corpus-a",
            corpus: corpus,
            tokenize: { _ in [1, 2] })) {
            XCTAssertEqual($0 as? SealedKLReferenceCLIError, .entrySetMismatch)
        }
    }

    func testAtomicPublishLeavesNoFinalOutputOnInjectedFailure() throws {
        let root = try temporaryDirectory()
        let final = root.appendingPathComponent("sealed-ref", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let blob = try sealedKLReferenceLogitsBlob([[1.0]])
        let manifest = SealedKLReferenceManifest(
            schema: "sealed-kl-reference",
            version: 1,
            identity: expectedBinding(),
            corpus: "corpus-a",
            referenceVersion: "reference-v1",
            maxTokens: 24,
            sampleSize: 128,
            entries: [
                SealedKLReferenceEntry(
                    id: "entry-1",
                    tag: "prose",
                    promptTokenIDs: [1],
                    continuationTokenIDs: [2],
                    samplePositions: nil,
                    logitsFile: "entry-1.f32",
                    logitsSHA256: SealedKLReferenceBundle.sha256Hex(blob),
                    rowCount: 1,
                    vocabSize: 1,
                    byteCount: blob.count),
            ])

        XCTAssertThrowsError(try writeSealedKLReferenceBundleAtomically(
            manifest: manifest,
            blobs: ["entry-1.f32": blob],
            to: final,
            afterStagingWrite: {
                throw SealedKLReferenceCLIError.injectedFailure("boom")
            }))
        XCTAssertFalse(FileManager.default.fileExists(atPath: final.path))
    }

    func testReplaySourceMustRemainExactAcrossCandidateLoad() throws {
        let before = try sourceSnapshot()
        XCTAssertNoThrow(try validateSealedKLReferenceSourceUnchanged(
            before: before,
            after: before))

        let changed = try CompressedKVAttentionRuntimeSourceSnapshot.load(
            exactModelConfigData: Data("changed-config".utf8),
            checkpointManifestHash: String(repeating: "3", count: 64),
            checkpointContentSHA256: String(repeating: "2", count: 64),
            tokenizerSHA256: String(repeating: "4", count: 64))
        XCTAssertThrowsError(try validateSealedKLReferenceSourceUnchanged(
            before: before,
            after: changed)) {
            XCTAssertEqual($0 as? SealedKLReferenceCLIError, .modelSourceDrift)
        }
    }

    private func sourceSnapshot() throws -> CompressedKVAttentionRuntimeSourceSnapshot {
        try CompressedKVAttentionRuntimeSourceSnapshot.load(
            exactModelConfigData: Data("config".utf8),
            checkpointManifestHash: String(repeating: "3", count: 64),
            checkpointContentSHA256: String(repeating: "2", count: 64),
            tokenizerSHA256: String(repeating: "4", count: 64))
    }

    private func expectedBinding() -> SealedKLReferenceBinding {
        SealedKLReferenceBinding(
            model: "model-a",
            harnessGitSHA: String(repeating: "b", count: 40),
            modelConfigSHA256: String(repeating: "1", count: 64),
            checkpointManifestSHA256: String(repeating: "2", count: 64),
            checkpointContentSHA256: String(repeating: "3", count: 64),
            tokenizer: "tokenizer-a",
            tokenizerManifestSHA256: String(repeating: "4", count: 64),
            corpusContentHash: String(repeating: "5", count: 16),
            corpusRawFileSHA256: String(repeating: "6", count: 64),
            harness: sealedKLReferenceHarnessIdentity,
            referenceScriptSHA256: String(repeating: "7", count: 64),
            referenceRuntimeVersions: SealedKLReferenceRuntimeVersions(
                mlx: "mlx-0.31.6",
                mlxLM: "mlx-lm-0.27.1"),
            workloadNonce: "llama-quality-v1")
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        return directory
    }
}
