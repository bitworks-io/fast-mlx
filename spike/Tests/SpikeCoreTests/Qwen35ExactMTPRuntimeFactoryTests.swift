import CryptoKit
import Darwin
import Foundation
import HarnessCore
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

@testable import SpikeCore

final class Qwen35ExactMTPRuntimeFactoryTests: XCTestCase {
    func testQwen38HarnessRevisionUsesCleanLiveCheckoutAndNeverEnvironmentOverride() throws {
        let liveSHA = String(repeating: "a", count: 40)
        let stampSHA = String(repeating: "b", count: 40)
        XCTAssertEqual(
            try resolveQwen38LiveExactnessHarnessGitSHA(
                liveGitOutput: liveSHA,
                liveRepositoryPresent: true,
                shaFile: stampSHA),
            liveSHA)
        XCTAssertThrowsError(try resolveQwen38LiveExactnessHarnessGitSHA(
            liveGitOutput: "\(liveSHA)-dirty",
            liveRepositoryPresent: true,
            shaFile: stampSHA)) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .invalidHarnessGitSHA)
        }
        XCTAssertThrowsError(try resolveQwen38LiveExactnessHarnessGitSHA(
            liveGitOutput: nil,
            liveRepositoryPresent: true,
            shaFile: stampSHA)) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .harnessGitSHAUnavailable)
        }
    }

    func testQwen38HarnessRevisionAllowsOnlyCleanDeployStampWithoutGitCheckout() throws {
        let stampSHA = String(repeating: "c", count: 40)
        XCTAssertEqual(
            try resolveQwen38LiveExactnessHarnessGitSHA(
                liveGitOutput: nil,
                liveRepositoryPresent: false,
                shaFile: " \(stampSHA)\n"),
            stampSHA)
        for invalid in [
            String(repeating: "0", count: 40),
            "\(stampSHA)-dirty",
            String(repeating: "C", count: 40),
        ] {
            XCTAssertThrowsError(try resolveQwen38LiveExactnessHarnessGitSHA(
                liveGitOutput: nil,
                liveRepositoryPresent: false,
                shaFile: invalid))
        }
    }

    func testQwen38ValidatedEvidenceWriterPublishesCompleteRecordAndPreservesExistingFile() throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("jsonl")
        defer { try? FileManager.default.removeItem(at: output) }
        let data = try validQwen38LiveExactnessRecordData()

        try writeValidatedQwen38LiveExactnessEvidenceFile(data, to: output)
        XCTAssertEqual(try Data(contentsOf: output), data)
        XCTAssertNoThrow(try Qwen38MTPLiveExactnessGate.validateJSONL(
            Data(contentsOf: output)))

        XCTAssertThrowsError(try writeValidatedQwen38LiveExactnessEvidenceFile(data, to: output))
        XCTAssertEqual(try Data(contentsOf: output), data)
    }

    func testQwen38ValidatedEvidenceWriterLeavesNoDestinationForInvalidEvidence() throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("jsonl")
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))

        XCTAssertThrowsError(try writeValidatedQwen38LiveExactnessEvidenceFile(
            Data("{\"invalid\":true}\n".utf8),
            to: output))
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: output.path),
            "failed validation must not create even a partial destination")
    }

    func testKnownVendoredLockExactlyMatchesHarnessPreflightLock() throws {
        XCTAssertNoThrow(try Qwen35ExactMTPRuntimeFactory.validateKnownLockParity())
    }

    private func validQwen38LiveExactnessRecordData() throws -> Data {
        let cache = (0 ..< Qwen38MTPLiveExactnessGate.requiredCacheLayerCount).map { layerIndex in
            let isDense = (layerIndex + 1).isMultiple(of: 4)
            return Qwen38MTPLiveExactnessCacheFingerprint(
                layerIndex: layerIndex,
                cacheType: isDense ? "dense-attention" : "recurrent-mamba",
                offset: isDense ? 12 : 0,
                metaStateSHA256: String(repeating: "a", count: 64),
                stateFingerprints: (0 ..< 2).map { stateIndex in
                    let shape = isDense
                        ? [1, 4, 12, 256]
                        : (stateIndex == 0 ? [1, 3, 10_240] : [1, 48, 128, 128])
                    let dtype = isDense || stateIndex == 0 ? "bfloat16" : "float32"
                    let byteCount = shape.reduce(1, *) * (dtype == "float32" ? 4 : 2)
                    return Qwen38MTPLiveExactnessArrayFingerprint(
                        stateIndex: stateIndex,
                        shape: shape,
                        dtype: dtype,
                        byteCount: byteCount,
                        sha256: String(repeating: "b", count: 64))
                })
        }
        let cases = Qwen38MTPLiveExactnessGate.requiredCases.enumerated().map {
            index, specification in
            let bytes = Data("case-\(index)".utf8)
            let tokens = [100 + index, 200 + index]
            return Qwen38MTPLiveExactnessCaseEvidence(
                id: specification.id,
                promptSHA256: specification.promptSHA256,
                maxTokens: specification.maxTokens,
                scalarTokenIDs: tokens,
                mtpTokenIDs: tokens,
                scalarDecodedUTF8Base64: bytes.base64EncodedString(),
                mtpDecodedUTF8Base64: bytes.base64EncodedString(),
                decodedUTF8SHA256: sha256Hex(bytes),
                proposedDraftTokens: 3,
                acceptedDraftTokens: 2,
                passthroughReason: nil,
                scalarCacheFingerprints: cache,
                mtpCacheFingerprints: cache)
        }
        let evidence = Qwen38MTPLiveExactnessEvidence(
            schemaVersion: Qwen38MTPLiveExactnessGate.schemaVersion,
            artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
            artifactID: Qwen38MTPLiveExactnessGate.requiredArtifactID,
            source: Qwen38MTPLiveExactnessGate.requiredSourceIdentity,
            cases: cases)
        let provenance = Provenance(
            date: "2026-08-25T00:00:00Z",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: Qwen38MTPPerformanceScorecardGate.requiredRAMBytes,
            hardwareOS: "macOS 26.0",
            harnessGitSHA: String(repeating: "e", count: 40),
            mlxSwiftVersion: "0.31.6",
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelPath: Qwen38MTPLiveExactnessGate.modelPathSentinel,
            modelConfigHash:
                Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetConfigSHA256,
            modelCheckpointManifestHash:
                Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusId: nil,
            corpusContentHash: nil,
            nonce: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID)
        let record = ResultRecord(
            subcommand: Qwen38MTPLiveExactnessGate.subcommand,
            provenance: provenance,
            payload: evidence)
        return Data((try record.jsonLine() + "\n").utf8)
    }

    func testKnownQwen38VendoredLockExactlyMatchesHarnessPreflightLock() throws {
        XCTAssertNoThrow(try Qwen35ExactMTPRuntimeFactory.validateKnownLockParity(
            selection: .qwen38_27BMXFP8Depth1))
    }

    func testSelectedRuntimeLockRejectsCrossRowEvidenceBeforeConstruction() throws {
        XCTAssertThrowsError(try Qwen35ExactMTPRuntimeFactory.validateSelectedVendoredLock(
            Qwen35ExactMTPKnownArtifactLocks.qwen35_9BDepth1,
            selection: .qwen38_27BMXFP8Depth1
        )) { error in
            XCTAssertEqual(
                error as? Qwen35ExactMTPRuntimeAdmissionError,
                .vendoredLockDrift)
        }
    }

    func testExactSnapshotDownloaderAllowsOnlySelectedQwen38Rows() async throws {
        let target = URL(fileURLWithPath: "/tmp/qwen38-target", isDirectory: true)
        let drafter = URL(fileURLWithPath: "/tmp/qwen38-drafter", isDirectory: true)
        let downloader = ExactSnapshotDownloader(
            target: target,
            drafter: drafter,
            selection: .qwen38_27BMXFP8Depth1)

        let targetURL = try await downloader.download(
            id: "mlx-community/Qwen3.8-27B-mxfp8",
            revision: "d48d163bcdf24acaf656474854ab88ea17d65bd1",
            matching: exactSnapshotPatterns,
            useLatest: false,
            progressHandler: { _ in })
        let drafterURL = try await downloader.download(
            id: "mlx-community/Qwen3.8-27B-MTP-mxfp8",
            revision: "a50634460045613f166b09b13519466e801c6568",
            matching: exactSnapshotPatterns,
            useLatest: false,
            progressHandler: { _ in })

        XCTAssertEqual(targetURL, target)
        XCTAssertEqual(drafterURL, drafter)
    }

    func testExactSnapshotDownloaderRejectsCrossRowAndLatestRequests() async throws {
        let downloader = ExactSnapshotDownloader(
            target: URL(fileURLWithPath: "/tmp/qwen38-target", isDirectory: true),
            drafter: URL(fileURLWithPath: "/tmp/qwen38-drafter", isDirectory: true),
            selection: .qwen38_27BMXFP8Depth1)

        await XCTAssertThrowsErrorAsync(try await downloader.download(
            id: "mlx-community/Qwen3.5-9B-MLX-4bit",
            revision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
            matching: exactSnapshotPatterns,
            useLatest: false,
            progressHandler: { _ in })) { error in
            XCTAssertEqual(error as? ExactSnapshotDownloaderError, .unexpectedDownloadRequest)
        }
        await XCTAssertThrowsErrorAsync(try await downloader.download(
            id: "mlx-community/Qwen3.8-27B-mxfp8",
            revision: "d48d163bcdf24acaf656474854ab88ea17d65bd1",
            matching: exactSnapshotPatterns,
            useLatest: true,
            progressHandler: { _ in })) { error in
            XCTAssertEqual(error as? ExactSnapshotDownloaderError, .unexpectedDownloadRequest)
        }
    }

    func testExactArtifactPreflightAgainstLocalSnapshotsWhenConfigured() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let targetPath = environment["FAST_MLX_QWEN35_TARGET_SNAPSHOT"],
            let drafterPath = environment["FAST_MLX_QWEN35_MTP_SNAPSHOT"]
        else {
            throw XCTSkip("exact Qwen3.5 target/drafter snapshot paths are not configured")
        }

        let downloader = ExactSnapshotDownloader(
            target: URL(fileURLWithPath: targetPath, isDirectory: true),
            drafter: URL(fileURLWithPath: drafterPath, isDirectory: true))

        do {
            _ = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
                from: downloader,
                using: PreflightOnlyTokenizerLoader())
            XCTFail("the preflight-only tokenizer must stop the load")
        } catch ExactSnapshotDownloaderError.tokenizerLoadReached {
            // Fixed resolution, vendored admission, and the mandatory HarnessCore bridge all ran
            // before the wrapper advanced into the model/tokenizer loader.
        }
    }

    func testExactDepthOneArtifactTwoDraftGreedyParityWhenConfigured() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FAST_MLX_QWEN35_RUN_LIVE_EXACTNESS"] == "1" else {
            throw XCTSkip("live exactness requires FAST_MLX_QWEN35_RUN_LIVE_EXACTNESS=1")
        }
        guard let targetPath = environment["FAST_MLX_QWEN35_TARGET_SNAPSHOT"],
            let drafterPath = environment["FAST_MLX_QWEN35_MTP_SNAPSHOT"]
        else {
            throw XCTSkip("exact Qwen3.5 target/drafter snapshot paths are not configured")
        }

        let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
            from: ExactSnapshotDownloader(
                target: URL(fileURLWithPath: targetPath, isDirectory: true),
                drafter: URL(fileURLWithPath: drafterPath, isDirectory: true)),
            using: #huggingFaceTokenizerLoader())
        XCTAssertEqual(pair.binding.runtimeBlockSize, 3)
        XCTAssertEqual(pair.binding.maximumAcceptedDraftTokens, 2)
        let promptTokens = pair.target.tokenizer.encode(
            text: "Continue the exact sequence with one concise answer: 2, 3, 5, 7, 11,")
        let input = LMInput(tokens: MLXArray(promptTokens))
        let parameters = GenerateParameters(maxTokens: 16, temperature: 0)

        let scalarCache = pair.target.model.newCache(parameters: parameters)
        let scalar = try await collectTokens(
            generateTokens(
                input: input,
                cache: scalarCache,
                parameters: parameters,
                context: pair.target))

        let mtpCache = pair.target.model.newCache(parameters: parameters)
        let mtp = try await collectTokens(
            generateTokens(
                input: input,
                cache: mtpCache,
                parameters: parameters,
                context: pair.target,
                mtpDrafter: pair.drafter.model,
                blockSize: pair.binding.runtimeBlockSize))

        XCTAssertFalse(scalar.tokens.isEmpty)
        XCTAssertEqual(mtp.tokens, scalar.tokens)
        XCTAssertEqual(
            Data(pair.target.tokenizer.decode(
                tokenIds: mtp.tokens, skipSpecialTokens: false).utf8),
            Data(pair.target.tokenizer.decode(
                tokenIds: scalar.tokens, skipSpecialTokens: false).utf8))
        XCTAssertEqual(mtp.info?.proposedDraftTokens, 10)
        XCTAssertEqual(mtp.info?.acceptedDraftTokens, 9)
        XCTAssertNil(mtp.info?.passthroughReason)
        try assertCacheEquivalent(scalarCache, mtpCache)

        print(
            "QWEN35_MTP_LIVE_EXACTNESS tokens=\(mtp.tokens.count) "
                + "proposed=\(mtp.info?.proposedDraftTokens ?? -1) "
                + "accepted=\(mtp.info?.acceptedDraftTokens ?? -1) "
                + "scalar_tps=\(scalar.info?.tokensPerSecond ?? 0) "
                + "mtp_tps=\(mtp.info?.tokensPerSecond ?? 0)")
    }

    func testQwen38ExactDepthOneArtifactGreedyParityWhenConfigured() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FAST_MLX_QWEN38_RUN_LIVE_EXACTNESS"] == "1" else {
            throw XCTSkip("live exactness requires FAST_MLX_QWEN38_RUN_LIVE_EXACTNESS=1")
        }
        guard let targetPath = environment["FAST_MLX_QWEN38_TARGET_SNAPSHOT"],
            let drafterPath = environment["FAST_MLX_QWEN38_MTP_SNAPSHOT"]
        else {
            throw XCTSkip("exact Qwen3.8 target/drafter snapshot paths are not configured")
        }
        let evidenceOutputPath = environment["FAST_MLX_QWEN38_LIVE_EXACTNESS_JSONL"]
        if let evidenceOutputPath,
            FileManager.default.fileExists(atPath: evidenceOutputPath)
        {
            throw CocoaError(.fileWriteFileExists)
        }

        let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
            selection: .qwen38_27BMXFP8Depth1,
            from: ExactSnapshotDownloader(
                target: URL(fileURLWithPath: targetPath, isDirectory: true),
                drafter: URL(fileURLWithPath: drafterPath, isDirectory: true),
                selection: .qwen38_27BMXFP8Depth1),
            using: #huggingFaceTokenizerLoader())
        guard pair.binding.runtimeBlockSize == 3,
            pair.binding.maximumAcceptedDraftTokens == 2
        else {
            throw Qwen38LiveExactnessProducerError.runtimeBindingDrift
        }

        var evidenceCases: [Qwen38MTPLiveExactnessCaseEvidence] = []
        for testCase in qwen38GreedyParityCases {
            let promptTokens = pair.target.tokenizer.encode(text: testCase.prompt)
            let input = LMInput(tokens: MLXArray(promptTokens))
            let parameters = GenerateParameters(maxTokens: testCase.maxTokens, temperature: 0)

            let scalarCache = pair.target.model.newCache(parameters: parameters)
            let scalar = try await collectTokens(
                generateTokens(
                    input: input,
                    cache: scalarCache,
                    parameters: parameters,
                    context: pair.target))

            let mtpCache = pair.target.model.newCache(parameters: parameters)
            let mtp = try await collectTokens(
                generateTokens(
                    input: input,
                    cache: mtpCache,
                    parameters: parameters,
                    context: pair.target,
                    mtpDrafter: pair.drafter.model,
                    blockSize: pair.binding.runtimeBlockSize))

            XCTAssertFalse(mtp.tokens.isEmpty, "MTP generated no tokens for \(testCase.id)")
            XCTAssertEqual(mtp.tokens, scalar.tokens, "token mismatch for \(testCase.id)")
            XCTAssertEqual(
                Data(pair.target.tokenizer.decode(
                    tokenIds: mtp.tokens, skipSpecialTokens: false).utf8),
                Data(pair.target.tokenizer.decode(
                    tokenIds: scalar.tokens, skipSpecialTokens: false).utf8),
                "decoded UTF-8 bytes mismatch for \(testCase.id)")
            XCTAssertGreaterThan(
                mtp.info?.proposedDraftTokens ?? 0,
                0,
                "expected draft proposals for \(testCase.id)")
            XCTAssertGreaterThan(
                mtp.info?.acceptedDraftTokens ?? 0,
                0,
                "expected accepted drafts for \(testCase.id)")
            XCTAssertNil(mtp.info?.passthroughReason, "unexpected passthrough for \(testCase.id)")
            try assertCacheEquivalent(scalarCache, mtpCache)
            evidenceCases.append(makeQwen38LiveExactnessCaseEvidence(
                id: testCase.id,
                prompt: testCase.prompt,
                maxTokens: testCase.maxTokens,
                scalar: scalar,
                mtp: mtp,
                scalarCache: scalarCache,
                mtpCache: mtpCache,
                tokenizer: pair.target.tokenizer))

            print(
                "QWEN38_MTP_LIVE_EXACTNESS identity=qwen38_27BMXFP8Depth1 "
                    + "case=\(testCase.id) "
                    + "tokens=\(mtp.tokens.count) "
                    + "proposed=\(mtp.info?.proposedDraftTokens ?? -1) "
                    + "accepted=\(mtp.info?.acceptedDraftTokens ?? -1) "
                    + "tps=scalar:\(scalar.info?.tokensPerSecond ?? 0)"
                    + "/mtp:\(mtp.info?.tokensPerSecond ?? 0)")
        }

        if let evidenceOutputPath {
            let evidence = Qwen38MTPLiveExactnessEvidence(
                schemaVersion: Qwen38MTPLiveExactnessGate.schemaVersion,
                artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
                artifactID: Qwen38MTPLiveExactnessGate.requiredArtifactID,
                source: Qwen38MTPLiveExactnessGate.requiredSourceIdentity,
                cases: evidenceCases)
            let record = ResultRecord(
                subcommand: Qwen38MTPLiveExactnessGate.subcommand,
                provenance: try qwen38LiveExactnessProvenance(),
                payload: evidence)
            let data = Data((try record.jsonLine() + "\n").utf8)
            try writeValidatedQwen38LiveExactnessEvidenceFile(
                data,
                to: URL(fileURLWithPath: evidenceOutputPath))
        }
    }
}

private struct CollectedGeneration {
    let tokens: [Int]
    let info: GenerateCompletionInfo?
}

private struct GreedyParityCase {
    let id: String
    let prompt: String
    let maxTokens: Int
}

private let qwen38GreedyParityCases: [GreedyParityCase] = [
    .init(
        id: "numbers",
        prompt: "Continue the exact sequence with one concise answer: 2, 3, 5, 7, 11,",
        maxTokens: 12),
    .init(
        id: "sentence",
        prompt: "Complete this sentence in a few words: The fastest reliable test is",
        maxTokens: 12),
]

private let exactSnapshotPatterns = ["*.safetensors", "*.json", "*.jinja"]

private func collectTokens(
    _ stream: AsyncStream<TokenGeneration>
) async throws -> CollectedGeneration {
    var tokens: [Int] = []
    var info: GenerateCompletionInfo?
    for await generation in stream {
        if let token = generation.token {
            tokens.append(token)
        }
        if let completion = generation.info {
            info = completion
        }
    }
    return CollectedGeneration(tokens: tokens, info: info)
}

private func assertCacheEquivalent(_ scalar: [KVCache], _ mtp: [KVCache]) throws {
    XCTAssertEqual(mtp.count, scalar.count)
    for (index, pair) in zip(scalar, mtp).enumerated() {
        let scalarCache = pair.0
        let mtpCache = pair.1
        XCTAssertEqual(
            String(reflecting: type(of: mtpCache)),
            String(reflecting: type(of: scalarCache)),
            "cache type mismatch at layer \(index)")
        XCTAssertEqual(mtpCache.offset, scalarCache.offset, "cache offset mismatch at layer \(index)")
        XCTAssertEqual(
            mtpCache.metaState, scalarCache.metaState,
            "cache metadata mismatch at layer \(index)")
        let scalarState = scalarCache.state
        let mtpState = mtpCache.state
        XCTAssertEqual(mtpState.count, scalarState.count, "cache state count mismatch at layer \(index)")
        for (stateIndex, arrays) in zip(scalarState, mtpState).enumerated() {
            eval(arrays.0, arrays.1)
            XCTAssertEqual(
                arrays.1.shape, arrays.0.shape,
                "cache shape mismatch at layer \(index) state \(stateIndex)")
            XCTAssertEqual(
                arrays.1.dtype, arrays.0.dtype,
                "cache dtype mismatch at layer \(index) state \(stateIndex)")
            XCTAssertEqual(
                arrays.1.asData(access: .copy).data,
                arrays.0.asData(access: .copy).data,
                "cache bytes mismatch at layer \(index) state \(stateIndex)")
        }
    }
}

private func makeQwen38LiveExactnessCaseEvidence(
    id: String,
    prompt: String,
    maxTokens: Int,
    scalar: CollectedGeneration,
    mtp: CollectedGeneration,
    scalarCache: [KVCache],
    mtpCache: [KVCache],
    tokenizer: any MLXLMCommon.Tokenizer
) -> Qwen38MTPLiveExactnessCaseEvidence {
    let scalarBytes = Data(tokenizer.decode(
        tokenIds: scalar.tokens,
        skipSpecialTokens: false).utf8)
    let mtpBytes = Data(tokenizer.decode(
        tokenIds: mtp.tokens,
        skipSpecialTokens: false).utf8)
    return Qwen38MTPLiveExactnessCaseEvidence(
        id: id,
        promptSHA256: sha256Hex(Data(prompt.utf8)),
        maxTokens: maxTokens,
        scalarTokenIDs: scalar.tokens,
        mtpTokenIDs: mtp.tokens,
        scalarDecodedUTF8Base64: scalarBytes.base64EncodedString(),
        mtpDecodedUTF8Base64: mtpBytes.base64EncodedString(),
        decodedUTF8SHA256: sha256Hex(scalarBytes),
        proposedDraftTokens: mtp.info?.proposedDraftTokens ?? 0,
        acceptedDraftTokens: mtp.info?.acceptedDraftTokens ?? 0,
        passthroughReason: mtp.info?.passthroughReason,
        scalarCacheFingerprints: qwen38LiveExactnessCacheFingerprints(scalarCache),
        mtpCacheFingerprints: qwen38LiveExactnessCacheFingerprints(mtpCache))
}

private func qwen38LiveExactnessCacheFingerprints(
    _ caches: [KVCache]
) -> [Qwen38MTPLiveExactnessCacheFingerprint] {
    caches.enumerated().map { layerIndex, cache in
        let states = cache.state
        let fingerprints = states.enumerated().map { stateIndex, array in
            eval(array)
            let bytes = array.asData(access: .copy).data
            return Qwen38MTPLiveExactnessArrayFingerprint(
                stateIndex: stateIndex,
                shape: array.shape,
                dtype: String(describing: array.dtype),
                byteCount: bytes.count,
                sha256: sha256Hex(bytes))
        }
        return Qwen38MTPLiveExactnessCacheFingerprint(
            layerIndex: layerIndex,
            cacheType: qwen38LiveExactnessCacheType(cache),
            offset: cache.offset,
            metaStateSHA256: sha256Hex(Data(String(reflecting: cache.metaState).utf8)),
            stateFingerprints: fingerprints)
    }
}

private func qwen38LiveExactnessCacheType(_ cache: KVCache) -> String {
    switch cache {
    case is MambaCache:
        return "recurrent-mamba"
    case is KVCacheSimple:
        return "dense-attention"
    default:
        return "unsupported:\(String(reflecting: type(of: cache)))"
    }
}

private func qwen38LiveExactnessProvenance() throws -> Provenance {
    guard let chip = qwen38SysctlString("machdep.cpu.brand_string")
        ?? qwen38SysctlString("hw.model"),
        !chip.isEmpty
    else {
        throw Qwen38LiveExactnessProducerError.hardwareIdentityUnavailable
    }
    return Provenance(
        date: ISO8601DateFormatter().string(from: Date()),
        hardwareChip: chip,
        hardwareRAMBytes: ProcessInfo.processInfo.physicalMemory,
        hardwareOS: ProcessInfo.processInfo.operatingSystemVersionString,
        harnessGitSHA: try qwen38HarnessGitSHA(),
        mlxSwiftVersion: "0.31.6",
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

private func qwen38HarnessGitSHA() throws -> String {
    let currentDirectory = URL(
        fileURLWithPath: FileManager.default.currentDirectoryPath,
        isDirectory: true)
    let repositoryRoot = qwen38LiveExactnessRepositoryRoot(startingAt: currentDirectory)
    return try resolveQwen38LiveExactnessHarnessGitSHA(
        liveGitOutput: repositoryRoot.flatMap(qwen38LiveGitSHA),
        liveRepositoryPresent: repositoryRoot != nil,
        shaFile: try? String(
            contentsOf: currentDirectory.appendingPathComponent(".harness-sha"),
            encoding: .utf8))
}

private func resolveQwen38LiveExactnessHarnessGitSHA(
    liveGitOutput: String?,
    liveRepositoryPresent: Bool,
    shaFile: String?
) throws -> String {
    let source: String
    if let liveGitOutput {
        source = liveGitOutput
    } else if liveRepositoryPresent {
        throw Qwen38LiveExactnessProducerError.harnessGitSHAUnavailable
    } else if let shaFile {
        source = shaFile
    } else {
        throw Qwen38LiveExactnessProducerError.harnessGitSHAUnavailable
    }
    let candidate = source.trimmingCharacters(in: .whitespacesAndNewlines)
    guard candidate.count == 40,
        candidate != String(repeating: "0", count: 40),
        candidate.utf8.allSatisfy({
            ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
        })
    else {
        throw Qwen38LiveExactnessProducerError.invalidHarnessGitSHA
    }
    return candidate
}

private func qwen38LiveExactnessRepositoryRoot(startingAt start: URL) -> URL? {
    let components = start.standardizedFileURL.pathComponents
    let manager = FileManager.default
    for count in stride(from: components.count, through: 1, by: -1) {
        let directory = URL(
            fileURLWithPath: NSString.path(withComponents: Array(components.prefix(count))),
            isDirectory: true)
        guard manager.fileExists(atPath: directory.appendingPathComponent(".git").path),
            manager.fileExists(atPath: directory.appendingPathComponent("AGENTS.md").path),
            manager.fileExists(atPath: directory.appendingPathComponent("spike/Package.swift").path)
        else { continue }
        return directory
    }
    return nil
}

private func qwen38LiveGitSHA(repositoryRoot: URL) -> String? {
    let expectedRoot = repositoryRoot.standardizedFileURL
    guard let reportedRoot = qwen38RunGit([
        "-C", expectedRoot.path, "rev-parse", "--show-toplevel",
    ])?.trimmingCharacters(in: .whitespacesAndNewlines),
        URL(fileURLWithPath: reportedRoot).standardizedFileURL == expectedRoot,
        let sha = qwen38RunGit([
            "-C", expectedRoot.path, "rev-parse", "HEAD",
        ])?.trimmingCharacters(in: .whitespacesAndNewlines),
        let status = qwen38RunGit([
            "-C", expectedRoot.path, "status", "--porcelain",
            "--untracked-files=normal", "--", "spike", "experiments",
        ])
    else { return nil }
    return status.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        ? sha : "\(sha)-dirty"
}

private func qwen38RunGit(_ arguments: [String]) -> String? {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["git"] + arguments
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = Pipe()
    do {
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        return String(data: data, encoding: .utf8)
    } catch {
        return nil
    }
}

private func qwen38SysctlString(_ name: String) -> String? {
    var size = 0
    let sizeResult = name.withCString { sysctlbyname($0, nil, &size, nil, 0) }
    guard sizeResult == 0, size > 1 else { return nil }
    var bytes = [CChar](repeating: 0, count: size)
    let readResult = name.withCString { namePointer in
        bytes.withUnsafeMutableBytes { buffer in
            sysctlbyname(namePointer, buffer.baseAddress, &size, nil, 0)
        }
    }
    guard readResult == 0 else { return nil }
    if let terminator = bytes.firstIndex(of: 0) {
        bytes.removeSubrange(terminator...)
    }
    return String(decoding: bytes.map(UInt8.init(bitPattern:)), as: UTF8.self)
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

private func writeValidatedQwen38LiveExactnessEvidenceFile(
    _ data: Data,
    to url: URL
) throws {
    _ = try Qwen38MTPLiveExactnessGate.validateJSONL(data)
    guard url.isFileURL else { throw CocoaError(.fileWriteUnsupportedScheme) }
    let directory = url.deletingLastPathComponent()
    let temporaryURL = directory.appendingPathComponent(
        ".\(url.lastPathComponent).\(UUID().uuidString).tmp")
    var descriptor = temporaryURL.withUnsafeFileSystemRepresentation { path -> Int32 in
        guard let path else { return -1 }
        return open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
    }
    guard descriptor >= 0 else {
        throw CocoaError(.fileWriteUnknown)
    }
    defer {
        if descriptor >= 0 { _ = close(descriptor) }
        temporaryURL.withUnsafeFileSystemRepresentation { path in
            if let path { _ = unlink(path) }
        }
    }

    do {
        try data.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            var written = 0
            while written < rawBuffer.count {
                let count = write(
                    descriptor,
                    baseAddress.advanced(by: written),
                    rawBuffer.count - written)
                guard count > 0 else { throw CocoaError(.fileWriteUnknown) }
                written += count
            }
        }
        guard fsync(descriptor) == 0 else {
            throw CocoaError(.fileWriteUnknown)
        }
        let closeResult = close(descriptor)
        descriptor = -1
        guard closeResult == 0 else { throw CocoaError(.fileWriteUnknown) }
    } catch {
        throw error
    }

    let linkResult = temporaryURL.withUnsafeFileSystemRepresentation { temporaryPath in
        url.withUnsafeFileSystemRepresentation { destinationPath in
            guard let temporaryPath, let destinationPath else { return Int32(-1) }
            return link(temporaryPath, destinationPath)
        }
    }
    guard linkResult == 0 else {
        if errno == EEXIST { throw CocoaError(.fileWriteFileExists) }
        throw CocoaError(.fileWriteUnknown)
    }

    let directoryDescriptor = directory.withUnsafeFileSystemRepresentation { path -> Int32 in
        guard let path else { return -1 }
        return open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
    }
    guard directoryDescriptor >= 0 else {
        unlinkQwen38EvidenceDestination(url)
        throw CocoaError(.fileWriteUnknown)
    }
    let directorySyncResult = fsync(directoryDescriptor)
    let directoryCloseResult = close(directoryDescriptor)
    guard directorySyncResult == 0, directoryCloseResult == 0 else {
        unlinkQwen38EvidenceDestination(url)
        throw CocoaError(.fileWriteUnknown)
    }
}

private func unlinkQwen38EvidenceDestination(_ url: URL) {
    url.withUnsafeFileSystemRepresentation { path in
        if let path { _ = unlink(path) }
    }
}

private enum Qwen38LiveExactnessProducerError: Error, Equatable {
    case runtimeBindingDrift
    case hardwareIdentityUnavailable
    case harnessGitSHAUnavailable
    case invalidHarnessGitSHA
}

private func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private enum ExactSnapshotDownloaderError: Error {
    case unexpectedDownloadRequest
    case tokenizerLoadReached
}

private struct ExactSnapshotDownloader: Downloader {
    let target: URL
    let drafter: URL
    var selection: Qwen35ExactMTPRuntimeSelection = .qwen35_9BDepth1

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard !useLatest, patterns == exactSnapshotPatterns else {
            throw ExactSnapshotDownloaderError.unexpectedDownloadRequest
        }
        progressHandler(Progress(totalUnitCount: 1))
        let row = selectedSnapshotRow(selection)
        switch (id, revision) {
        case (row.targetID, row.targetRevision):
            return target
        case (row.drafterID, row.drafterRevision):
            return drafter
        default:
            throw ExactSnapshotDownloaderError.unexpectedDownloadRequest
        }
    }
}

private struct ExactSnapshotRow {
    let targetID: String
    let targetRevision: String
    let drafterID: String
    let drafterRevision: String
}

private func selectedSnapshotRow(
    _ selection: Qwen35ExactMTPRuntimeSelection
) -> ExactSnapshotRow {
    switch selection {
    case .qwen35_9BDepth1:
        ExactSnapshotRow(
            targetID: "mlx-community/Qwen3.5-9B-MLX-4bit",
            targetRevision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
            drafterID: "mlx-community/Qwen3.5-9B-MTP-5bit",
            drafterRevision: "994730d199bff7799aa3ddef33a96723967a3e33")
    case .qwen38_27BMXFP8Depth1:
        ExactSnapshotRow(
            targetID: "mlx-community/Qwen3.8-27B-mxfp8",
            targetRevision: "d48d163bcdf24acaf656474854ab88ea17d65bd1",
            drafterID: "mlx-community/Qwen3.8-27B-MTP-mxfp8",
            drafterRevision: "a50634460045613f166b09b13519466e801c6568")
    }
}

private struct PreflightOnlyTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        throw ExactSnapshotDownloaderError.tokenizerLoadReached
    }
}

private func XCTAssertThrowsErrorAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line,
    _ errorHandler: (_ error: any Error) -> Void = { _ in }
) async {
    do {
        _ = try await expression()
        XCTFail(message(), file: file, line: line)
    } catch {
        errorHandler(error)
    }
}
