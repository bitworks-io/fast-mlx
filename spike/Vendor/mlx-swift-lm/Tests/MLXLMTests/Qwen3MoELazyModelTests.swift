import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest

final class Qwen3MoELazyModelTests: XCTestCase {
    private let immutableRevision = String(repeating: "a", count: 40)
    private let liveModelID = "mlx-community/Qwen3-30B-A3B-3bit"
    private let liveRevision = "6add5f3b9ef4219603698b4b996e20a6da6d8793"

    func testPinnedResolutionLoadsWithoutExpertParametersAndMatchesEagerForward() async throws {
        let fixture = try Qwen3MoELazyArtifactFixture()
        defer { fixture.remove() }
        let downloader = RecordingQwen3MoEDownloader(directory: fixture.directory)

        let runtime = try await loadQwen3MoELazyModel(
            from: downloader,
            modelID: "fixture/Qwen3MoE",
            revision: immutableRevision,
            capacityFraction: 0.375
        )

        let calls = await downloader.calls
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].id, "fixture/Qwen3MoE")
        XCTAssertEqual(calls[0].revision, immutableRevision)
        XCTAssertFalse(calls[0].useLatest)
        XCTAssertEqual(runtime.identity.modelID, "fixture/Qwen3MoE")
        XCTAssertEqual(runtime.identity.resolvedRevision, immutableRevision)
        XCTAssertEqual(runtime.capacityPerLayer, 3)
        XCTAssertTrue(
            runtime.loadedParameterNames.contains("model.layers.0.mlp.gate.weight")
        )
        XCTAssertFalse(
            runtime.loadedParameterNames.contains {
                $0.contains("model.layers.0.mlp.switch_mlp")
            }
        )

        let configuration = try fixture.configuration()
        let eager = Qwen3MoEModel(configuration)
        try loadWeights(modelDirectory: fixture.directory, model: eager)
        let input = MLXArray([1, 3, 5, 7], [1, 4])
        let eagerOutput = eager(input, cache: nil)
        let lazyOutput = try runtime(input, cache: nil)
        eval(eagerOutput, lazyOutput)

        let maxDifference = MLX.max(MLX.abs(lazyOutput - eagerOutput)).item(Float.self)
        XCTAssertTrue(
            allClose(lazyOutput, eagerOutput, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            "max difference \(maxDifference)"
        )
        XCTAssertTrue(
            (lazyOutput.argMax(axis: -1) .== eagerOutput.argMax(axis: -1))
                .all().item(Bool.self)
        )
        let first = runtime.snapshot()
        XCTAssertGreaterThan(first.counters.transactions, 1)
        XCTAssertGreaterThan(first.counters.misses, 0)
        XCTAssertGreaterThan(first.counters.bytesRead, 0)
        XCTAssertGreaterThan(first.counters.readCount, 0)
        XCTAssertGreaterThan(first.counters.readNanoseconds, 0)
        XCTAssertLessThanOrEqual(first.residentExpertsByLayer[0]?.count ?? 0, 3)

        _ = try runtime(input, cache: nil)
        let second = runtime.snapshot()
        XCTAssertEqual(second.counters.transactions, first.counters.transactions * 2)
        XCTAssertGreaterThan(second.counters.hits, first.counters.hits)
    }

    func testFloatingRevisionFailsBeforeDownloaderOrModelLoad() async throws {
        let fixture = try Qwen3MoELazyArtifactFixture()
        defer { fixture.remove() }
        let downloader = RecordingQwen3MoEDownloader(directory: fixture.directory)

        do {
            _ = try await loadQwen3MoELazyModel(
                from: downloader,
                modelID: "fixture/Qwen3MoE",
                revision: "main",
                capacityFraction: 0.375
            )
            XCTFail("expected immutable-revision refusal")
        } catch {
            XCTAssertEqual(
                error as? Qwen3MoEExpertResidencyError,
                .invalidConfiguration("revision is not an immutable SHA")
            )
        }
        let callCount = await downloader.calls.count
        XCTAssertEqual(callCount, 0)
    }

    func testFileIdentityDriftLeavesRuntimeStateUnchangedAndRetryRecovers() throws {
        let fixture = try Qwen3MoELazyArtifactFixture()
        defer { fixture.remove() }
        let runtime = try loadQwen3MoELazyModel(
            modelDirectory: fixture.directory,
            modelID: "fixture/Qwen3MoE",
            resolvedRevision: immutableRevision,
            capacityFraction: 0.375
        )
        let input = MLXArray([2, 4], [1, 2])
        let cache = runtime.newCache()
        let before = runtime.snapshot()
        let original = try Data(contentsOf: fixture.weightsURL)
        var drifted = original
        drifted.append(0)
        try drifted.write(to: fixture.weightsURL, options: .atomic)

        XCTAssertThrowsError(try runtime(input, cache: cache)) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .fileIdentityChanged(fixture.weightsURL.lastPathComponent)
            )
        }
        XCTAssertEqual(runtime.snapshot(), before)
        XCTAssertTrue(cache.allSatisfy { $0.offset == 0 })

        try original.write(to: fixture.weightsURL, options: .atomic)
        let recovered = try runtime(input, cache: cache)
        eval(recovered)
        XCTAssertEqual(runtime.snapshot().counters.transactions, 1)
        XCTAssertTrue(cache.allSatisfy { $0.offset == input.dim(1) })
    }

    func testSymlinkedShardLoadsResolvedTargetAndRunsInference() throws {
        let fixture = try Qwen3MoELazyArtifactFixture(symlinkWeights: true)
        defer { fixture.remove() }
        let runtime = try loadQwen3MoELazyModel(
            modelDirectory: fixture.directory,
            modelID: "fixture/Qwen3MoE",
            resolvedRevision: immutableRevision,
            capacityFraction: 0.375
        )

        let output = try runtime(MLXArray([1, 2], [1, 2]), cache: nil)
        eval(output)
        XCTAssertEqual(output.shape, [1, 2, 32])
        XCTAssertGreaterThan(runtime.snapshot().counters.bytesRead, 0)
    }

    func testLivePinnedQwen3MoEGreedyProof() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FAST_MLX_QWEN3MOE_RUN_LIVE"] == "1" else {
            throw XCTSkip("live proof requires FAST_MLX_QWEN3MOE_RUN_LIVE=1")
        }
        guard let path = environment["FAST_MLX_QWEN3MOE_MODEL_PATH"] else {
            XCTFail("FAST_MLX_QWEN3MOE_MODEL_PATH is required")
            return
        }
        let mode = environment["FAST_MLX_QWEN3MOE_MODE"] ?? "lazy"
        guard mode == "eager" || mode == "lazy" else {
            XCTFail("FAST_MLX_QWEN3MOE_MODE must be eager or lazy")
            return
        }
        guard let fraction = Double(
            environment["FAST_MLX_QWEN3MOE_CAPACITY_FRACTION"] ?? "0.375"
        ), fraction > 0, fraction <= 1 else {
            XCTFail("FAST_MLX_QWEN3MOE_CAPACITY_FRACTION must be in (0, 1]")
            return
        }
        guard let generationCount = Int(
            environment["FAST_MLX_QWEN3MOE_GENERATION_TOKENS"] ?? "8"
        ), generationCount > 0 else {
            XCTFail("FAST_MLX_QWEN3MOE_GENERATION_TOKENS must be positive")
            return
        }
        let directory = URL(filePath: path, directoryHint: .isDirectory)
        let prompt = [151644, 8948, 198, 2610, 525, 498, 30]

        Memory.clearCache()
        Memory.peakMemory = 0
        let loadStart = Date()
        let result: Qwen3MoELiveResult
        var counters: Qwen3MoEExpertResidencyCounters?
        var capacity: Int?
        var loadedParameterCount: Int?
        switch mode {
        case "eager":
            let configData = try Data(contentsOf: directory.appending(path: "config.json"))
            let configuration = try JSONDecoder.json5().decode(
                Qwen3MoEConfiguration.self,
                from: configData
            )
            let baseConfiguration = try JSONDecoder.json5().decode(
                BaseConfiguration.self,
                from: configData
            )
            let model = Qwen3MoEModel(configuration)
            try loadWeights(
                modelDirectory: directory,
                model: model,
                perLayerQuantization: baseConfiguration.perLayerQuantization
            )
            let loadSeconds = Date().timeIntervalSince(loadStart)
            result = try qwen3MoEGreedyResult(
                prompt: prompt,
                count: generationCount,
                loadSeconds: loadSeconds,
                cache: model.newCache(parameters: nil),
                predict: { model($0, cache: $1) }
            )
            loadedParameterCount = model.parameters().flattened().count
        default:
            let runtime = try loadQwen3MoELazyModel(
                modelDirectory: directory,
                modelID: liveModelID,
                resolvedRevision: liveRevision,
                capacityFraction: fraction
            )
            let loadSeconds = Date().timeIntervalSince(loadStart)
            result = try qwen3MoEGreedyResult(
                prompt: prompt,
                count: generationCount,
                loadSeconds: loadSeconds,
                cache: runtime.newCache(),
                predict: { try runtime($0, cache: $1) }
            )
            counters = runtime.snapshot().counters
            capacity = runtime.capacityPerLayer
            loadedParameterCount = runtime.loadedParameterNames.count
        }

        guard let loadedParameterCount else {
            XCTFail("the selected load mode did not report a parameter count")
            return
        }
        let memory = Memory.snapshot()
        var evidence: [String: Any] = [
            "modelID": liveModelID,
            "revision": liveRevision,
            "mode": mode,
            "capacityFraction": fraction,
            "promptTokens": prompt,
            "generatedTokens": result.tokens,
            "loadSeconds": result.loadSeconds,
            "firstTokenSeconds": result.firstTokenSeconds,
            "generationSeconds": result.generationSeconds,
            "generationTokensPerSecond": Double(result.tokens.count) / result.generationSeconds,
            "mlxActiveBytes": memory.activeMemory,
            "mlxCacheBytes": memory.cacheMemory,
            "mlxPeakBytes": memory.peakMemory,
            "loadedParameterCount": loadedParameterCount,
        ]
        if let capacity { evidence["capacityPerLayer"] = capacity }
        if let counters {
            evidence["transactions"] = counters.transactions
            evidence["hits"] = counters.hits
            evidence["misses"] = counters.misses
            evidence["bytesRead"] = counters.bytesRead
            evidence["readCount"] = counters.readCount
            evidence["readNanoseconds"] = counters.readNanoseconds
            evidence["evictions"] = counters.evictions
        }
        let data = try JSONSerialization.data(withJSONObject: evidence, options: [.sortedKeys])
        print("QWEN3MOE_LIVE_PROOF \(String(decoding: data, as: UTF8.self))")
        XCTAssertEqual(generationCount, 8, "the sealed live proof uses exactly eight tokens")
        XCTAssertEqual(result.tokens, [220, 16, 16, 16, 16, 16, 16, 16])
    }
}

private struct Qwen3MoELiveResult {
    let tokens: [Int]
    let loadSeconds: TimeInterval
    let firstTokenSeconds: TimeInterval
    let generationSeconds: TimeInterval
}

private func qwen3MoEGreedyResult(
    prompt: [Int],
    count: Int,
    loadSeconds: TimeInterval,
    cache: [KVCache],
    predict: (MLXArray, [KVCache]) throws -> MLXArray
) throws -> Qwen3MoELiveResult {
    guard count > 0 else {
        throw Qwen3MoEExpertResidencyError.invalidConfiguration(
            "greedy proof requires a positive token count"
        )
    }
    var input = MLXArray(prompt, [1, prompt.count])
    var tokens: [Int] = []
    let generationStart = Date()
    var firstTokenSeconds: TimeInterval?
    for _ in 0 ..< count {
        let logits = try predict(input, cache)
        let token = logits[0..., -1, 0...].argMax(axis: -1).item(Int.self)
        tokens.append(token)
        if firstTokenSeconds == nil {
            firstTokenSeconds = Date().timeIntervalSince(generationStart)
        }
        input = MLXArray([token], [1, 1])
    }
    guard let firstTokenSeconds else {
        throw Qwen3MoEExpertResidencyError.invalidConfiguration(
            "greedy proof produced no tokens"
        )
    }
    return Qwen3MoELiveResult(
        tokens: tokens,
        loadSeconds: loadSeconds,
        firstTokenSeconds: firstTokenSeconds,
        generationSeconds: Date().timeIntervalSince(generationStart)
    )
}

private actor RecordingQwen3MoEDownloader: Downloader {
    struct Call: Sendable {
        let id: String
        let revision: String?
        let useLatest: Bool
    }

    private(set) var calls: [Call] = []
    let directory: URL

    init(directory: URL) {
        self.directory = directory
    }

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        calls.append(Call(id: id, revision: revision, useLatest: useLatest))
        return directory
    }
}

private final class Qwen3MoELazyArtifactFixture {
    let directory: URL
    let weightsURL: URL
    private let configData: Data
    private let symlinkTargetURL: URL?

    init(symlinkWeights: Bool = false) throws {
        directory = FileManager.default.temporaryDirectory.appending(
            path: "qwen3moe-lazy-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        weightsURL = directory.appending(path: "model.safetensors")
        configData = try JSONSerialization.data(
            withJSONObject: [
                "model_type": "qwen3_moe",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 12,
                "num_attention_heads": 2,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "moe_intermediate_size": 8,
                "rms_norm_eps": 0.00001,
                "vocab_size": 32,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "tie_word_embeddings": false,
                "max_position_embeddings": 64,
                "norm_topk_prob": true,
            ],
            options: [.sortedKeys]
        )
        try configData.write(to: directory.appending(path: "config.json"))
        symlinkTargetURL = symlinkWeights
            ? FileManager.default.temporaryDirectory.appending(
                path: "qwen3moe-blob-\(UUID().uuidString).safetensors"
            )
            : nil

        MLXRandom.seed(92)
        let source = Qwen3MoEModel(try configuration())
        let arrays = Dictionary(
            uniqueKeysWithValues: source.parameters().flattened().map {
                ($0.0, $0.1.asType(.float16))
            }
        )
        if symlinkWeights {
            guard let target = symlinkTargetURL else {
                throw Qwen3MoEExpertResidencyError.invalidConfiguration(
                    "symlink fixture target was not created"
                )
            }
            try save(arrays: arrays, url: target)
            try FileManager.default.createSymbolicLink(
                at: weightsURL,
                withDestinationURL: target
            )
        } else {
            try save(arrays: arrays, url: weightsURL)
        }
        let weightMap = Dictionary(uniqueKeysWithValues: arrays.keys.map { ($0, "model.safetensors") })
        let index = try JSONSerialization.data(
            withJSONObject: ["weight_map": weightMap],
            options: [.sortedKeys]
        )
        try index.write(to: directory.appending(path: "model.safetensors.index.json"))
    }

    func configuration() throws -> Qwen3MoEConfiguration {
        try JSONDecoder.json5().decode(Qwen3MoEConfiguration.self, from: configData)
    }

    func remove() {
        try? FileManager.default.removeItem(at: directory)
        if let symlinkTargetURL {
            try? FileManager.default.removeItem(at: symlinkTargetURL)
        }
    }
}
