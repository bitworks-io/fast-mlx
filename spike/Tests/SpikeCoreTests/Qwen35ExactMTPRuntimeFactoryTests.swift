import HarnessCore
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

@testable import SpikeCore

final class Qwen35ExactMTPRuntimeFactoryTests: XCTestCase {
    func testKnownVendoredLockExactlyMatchesHarnessPreflightLock() throws {
        XCTAssertNoThrow(try Qwen35ExactMTPRuntimeFactory.validateKnownLockParity())
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

    func testExactDepthOneGreedyTokenAndFinalizedCacheParityWhenConfigured() async throws {
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
        XCTAssertEqual(mtp.info?.proposedDraftTokens, 7)
        XCTAssertEqual(mtp.info?.acceptedDraftTokens, mtp.info?.proposedDraftTokens)
        XCTAssertNil(mtp.info?.passthroughReason)
        try assertCacheEquivalent(scalarCache, mtpCache)

        print(
            "QWEN35_MTP_LIVE_EXACTNESS tokens=\(mtp.tokens.count) "
                + "proposed=\(mtp.info?.proposedDraftTokens ?? -1) "
                + "accepted=\(mtp.info?.acceptedDraftTokens ?? -1) "
                + "scalar_tps=\(scalar.info?.tokensPerSecond ?? 0) "
                + "mtp_tps=\(mtp.info?.tokensPerSecond ?? 0)")
    }
}

private struct CollectedGeneration {
    let tokens: [Int]
    let info: GenerateCompletionInfo?
}

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

private enum ExactSnapshotDownloaderError: Error {
    case unexpectedDownloadRequest
    case tokenizerLoadReached
}

private struct ExactSnapshotDownloader: Downloader {
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
            throw ExactSnapshotDownloaderError.unexpectedDownloadRequest
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
            throw ExactSnapshotDownloaderError.unexpectedDownloadRequest
        }
    }
}

private struct PreflightOnlyTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        throw ExactSnapshotDownloaderError.tokenizerLoadReached
    }
}
