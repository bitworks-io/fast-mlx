import HarnessCore
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore
@testable import fastmlx_harness

private final class TinyExactPrefixHarnessModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads: [Int]
    private let cacheDType: DType
    private let headDimension: Int
    private let vocabularySize = 4_096

    init(
        cacheDType: DType = .float16,
        kvHeadCount: Int = 2,
        headDimension: Int = 1
    ) {
        kvHeads = [kvHeadCount]
        self.cacheDType = cacheDType
        self.headDimension = headDimension
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        guard let layerCache = cache?.first, cache?.count == 1 else {
            preconditionFailure(
                "tiny exact-prefix model requires one cache")
        }
        let scalar = inputs.asType(cacheDType).reshaped([
            inputs.dim(0), 1, inputs.dim(1), 1,
        ])
        let values = broadcast(
            scalar,
            to: [
                inputs.dim(0), kvHeads[0],
                inputs.dim(1), headDimension,
            ])
        let (keys, _) = layerCache.update(
            keys: values, values: values)
        let nextToken = keys.sum(axes: [1, 2, 3])
            .asType(.int32) + 1
        let target = broadcast(
            nextToken.reshaped([inputs.dim(0), 1, 1]),
            to: [inputs.dim(0), inputs.dim(1), 1])
        let vocabulary = MLXArray(Int32(0) ..< Int32(vocabularySize))
            .reshaped([1, 1, vocabularySize])
        return (target .== vocabulary).asType(.float32) * 100
    }
}

final class ExactPrefixHarnessActorTests: XCTestCase {
    private func digest(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    private func request(
        namespace: Character = "a",
        template: Character = "b",
        tools: Character = "c"
    ) throws -> ExactPrefixRequestContext {
        try ExactPrefixRequestContext(
            isolationNamespaceSHA256: digest(namespace),
            promptTemplateSHA256: digest(template),
            toolsSHA256: digest(tools))
    }

    private func admission(
        nativeDType: String = "float16"
    ) throws
        -> CompressedKVAttentionRuntimeAdmission
    {
        let config = Data(
            """
            {"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_size":2,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":1,"max_position_embeddings":128,"use_sliding_window":false,"torch_dtype":"\(nativeDType)"}
            """.utf8)
        let source =
            try CompressedKVAttentionRuntimeSourceSnapshot.load(
                exactModelConfigData: config,
                checkpointManifestHash: "0123456789abcdef",
                checkpointContentSHA256: digest("d"),
                tokenizerSHA256: digest("e"))
        return try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: source)
    }

    private func cacheConfiguration(
        entries: Int = 8,
        bytes: Int = 4 * 1_024 * 1_024,
        minimumReusableTokens: Int = 2
    ) throws -> ExactPrefixCacheConfiguration {
        try ExactPrefixCacheConfiguration(
            policy: ExactPrefixCachePolicy(
                maxEntries: entries,
                maxRetainedBytes: bytes,
                minimumReusableTokens: minimumReusableTokens))
    }

    private func makeDriver(
        configuration: ExactPrefixCacheConfiguration,
        modelDType: DType = .float16,
        admissionDType: String = "float16",
        modelKVHeadCount: Int = 2,
        modelHeadDimension: Int = 1,
        eos: Int = 4_095
    ) throws -> SwiftEngineDriver {
        let admission = try admission(nativeDType: admissionDType)
        let identity = try ExactPrefixRuntimeIdentity(
            admission: admission,
            modelInstanceID: "tiny-qwen-instance")
        return SwiftEngineDriver(
            engine: HarnessEngineActor(
                model: TinyExactPrefixHarnessModel(
                    cacheDType: modelDType,
                    kvHeadCount: modelKVHeadCount,
                    headDimension: modelHeadDimension),
                exactPrefixCacheConfiguration: configuration,
                exactPrefixRuntimeIdentity:
                    configuration.policy.isEnabled
                    ? identity : nil),
            eos: eos,
            compressedKVAttentionAdmission: admission)
    }

    private func config(
        _ request: ExactPrefixRequestContext,
        maxTokens: Int = 3,
        kvQuant: String? = nil,
        specDecode: String? = nil
    ) -> RunConfig {
        RunConfig(
            maxTokens: maxTokens,
            specDecode: specDecode,
            kvQuant: kvQuant,
            exactPrefixRequest: request)
    }

    func testExactAndPartialHitsMatchColdControlAndReportPhysicalWork()
        async throws
    {
        let enabled = try makeDriver(
            configuration: cacheConfiguration())
        let disabled = try makeDriver(configuration: .disabled)
        let context = try request()
        let prompt = [2, 3, 4]

        let cold = try await enabled.generate(
            prompt: prompt, config: config(context))
        let exact = try await enabled.generate(
            prompt: prompt, config: config(context))
        let exactControl = try await disabled.generate(
            prompt: prompt, config: config(context))

        XCTAssertEqual(cold.tokens, exact.tokens)
        XCTAssertEqual(exact.tokens, exactControl.tokens)
        XCTAssertEqual(
            cold.requestStartMetrics?.prefixCacheOutcome, .miss)
        XCTAssertEqual(cold.requestStartMetrics?.cacheReadTokenCount, 0)
        XCTAssertEqual(
            cold.requestStartMetrics?.physicalPrefillTokenCount,
            prompt.count)
        XCTAssertEqual(
            exact.requestStartMetrics?.prefixCacheOutcome, .exactHit)
        XCTAssertEqual(
            exact.requestStartMetrics?.cacheReadTokenCount,
            prompt.count)
        XCTAssertEqual(
            exact.requestStartMetrics?.physicalPrefillTokenCount, 0)
        XCTAssertEqual(
            exactControl.requestStartMetrics?.prefixCacheOutcome,
            .disabled)

        let extended = prompt + [7]
        let partial = try await enabled.generate(
            prompt: extended, config: config(context))
        let partialControl = try await disabled.generate(
            prompt: extended, config: config(context))
        XCTAssertEqual(partial.tokens, partialControl.tokens)
        XCTAssertEqual(
            partial.requestStartMetrics?.prefixCacheOutcome,
            .partialHit)
        XCTAssertEqual(
            partial.requestStartMetrics?.cacheReadTokenCount,
            prompt.count)
        XCTAssertEqual(
            partial.requestStartMetrics?.physicalPrefillTokenCount, 1)
    }

    func testBFloat16RuntimeCapturesAndRestoresExactHit()
        async throws
    {
        let enabled = try makeDriver(
            configuration: cacheConfiguration(),
            modelDType: .bfloat16,
            admissionDType: "bfloat16")
        let disabled = try makeDriver(
            configuration: .disabled,
            modelDType: .bfloat16,
            admissionDType: "bfloat16")
        let context = try request()
        let prompt = [2, 3, 4]

        let cold = try await enabled.generate(
            prompt: prompt, config: config(context))
        let exact = try await enabled.generate(
            prompt: prompt, config: config(context))
        let control = try await disabled.generate(
            prompt: prompt, config: config(context))

        XCTAssertEqual(cold.tokens, control.tokens)
        XCTAssertEqual(exact.tokens, control.tokens)
        XCTAssertEqual(
            cold.requestStartMetrics?.prefixCacheOutcome, .miss)
        XCTAssertEqual(
            exact.requestStartMetrics?.prefixCacheOutcome, .exactHit)
    }

    func testConfiguredAndObservedDenseDTypeMismatchRejectsCache()
        async throws
    {
        let driver = try makeDriver(
            configuration: cacheConfiguration(),
            modelDType: .float16,
            admissionDType: "bfloat16")
        let disabled = try makeDriver(
            configuration: .disabled,
            modelDType: .float16,
            admissionDType: "float16")
        let context = try request()
        let prompt = [2, 3, 4]

        let result = try await driver.generate(
            prompt: prompt, config: config(context))
        let control = try await disabled.generate(
            prompt: prompt, config: config(context))

        XCTAssertEqual(result.tokens, control.tokens)
        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheOutcome,
            .rejected)
        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheRejectionReason,
            .snapshotEvidenceMismatch)
        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheRejectionDetail,
            "exact prefix snapshot dtype key=float16 value=float16 != expected bfloat16")
        let cache = await driver.engine.exactPrefixCacheSnapshot()
        XCTAssertEqual(cache.entryCount, 0)
    }

    func testEqualByteProductButWrongDenseGeometryRejectsCache()
        async throws
    {
        let driver = try makeDriver(
            configuration: cacheConfiguration(),
            modelKVHeadCount: 1,
            modelHeadDimension: 2)
        let disabled = try makeDriver(
            configuration: .disabled,
            modelKVHeadCount: 1,
            modelHeadDimension: 2)
        let context = try request()
        let prompt = [2, 3, 4]

        let result = try await driver.generate(
            prompt: prompt, config: config(context))
        let control = try await disabled.generate(
            prompt: prompt, config: config(context))

        XCTAssertEqual(result.tokens, control.tokens)
        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheOutcome,
            .rejected)
        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheRejectionReason,
            .snapshotEvidenceMismatch)
        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheRejectionDetail,
            "exact prefix snapshot geometry rank=4,batch=1,layers=1,kvHeads=1,headDim=2 != expected rank=4,batch=1,layers=1,kvHeads=2,headDim=1")
        let cache = await driver.engine.exactPrefixCacheSnapshot()
        XCTAssertEqual(cache.entryCount, 0)
    }

    func testIsolationAndSemanticKeysPreventCrossRequestPoisoning()
        async throws
    {
        let driver = try makeDriver(
            configuration: cacheConfiguration())
        let a = try request(namespace: "a")
        let b = try request(namespace: "f")
        let prompt = [3, 4, 5]

        let firstA = try await driver.generate(
            prompt: prompt, config: config(a))
        let onlyB = try await driver.generate(
            prompt: prompt, config: config(b))
        let secondA = try await driver.generate(
            prompt: prompt, config: config(a))

        XCTAssertEqual(firstA.tokens, onlyB.tokens)
        XCTAssertEqual(onlyB.tokens, secondA.tokens)
        XCTAssertEqual(
            firstA.requestStartMetrics?.prefixCacheOutcome, .miss)
        XCTAssertEqual(
            onlyB.requestStartMetrics?.prefixCacheOutcome, .miss)
        XCTAssertEqual(
            secondA.requestStartMetrics?.prefixCacheOutcome, .exactHit)
    }

    func testAdmissionRejectionFallsBackColdWithTypedTelemetry()
        async throws
    {
        let rejected = try makeDriver(
            configuration: cacheConfiguration(bytes: 1))
        let disabled = try makeDriver(configuration: .disabled)
        let context = try request()
        let prompt = [8, 9, 10]

        let result = try await rejected.generate(
            prompt: prompt, config: config(context))
        let control = try await disabled.generate(
            prompt: prompt, config: config(context))

        XCTAssertEqual(result.tokens, control.tokens)
        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheOutcome,
            .rejected)
        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheRejectionReason,
            .snapshotExceedsBudget)
        XCTAssertEqual(
            result.engagement.counts[
                "prefix_cache_rejection_snapshot_exceeds_budget"],
            1)
        let cache = await rejected.engine
            .exactPrefixCacheSnapshot()
        XCTAssertEqual(cache.entryCount, 0)
        XCTAssertEqual(cache.reservationCount, 0)
    }

    func testRejectedPromptCannotPublishLongerFinalContext()
        async throws
    {
        let driver = try makeDriver(
            configuration: cacheConfiguration(
                minimumReusableTokens: 5))
        let context = try request()
        let prompt = [8, 9, 10]

        let result = try await driver.generate(
            prompt: prompt, config: config(context))

        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheOutcome,
            .rejected)
        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheRejectionReason,
            .prefixTooShort)
        let cache = await driver.engine
            .exactPrefixCacheSnapshot()
        XCTAssertEqual(cache.entryCount, 0)
        XCTAssertEqual(cache.reservationCount, 0)
    }

    func testPadOnlyCommitReportsTypedRejectionAndPublishesNothing()
        async throws
    {
        let driver = try makeDriver(
            configuration: cacheConfiguration(),
            eos: 19)
        let context = try request()
        let prompt = [2, 3, 4]

        let result = try await driver.generate(
            prompt: prompt, config: config(context))

        XCTAssertEqual(result.tokens, [19])
        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheOutcome,
            .rejected)
        XCTAssertEqual(
            result.requestStartMetrics?.prefixCacheRejectionReason,
            .padOnlyOutput)
        let cache = await driver.engine
            .exactPrefixCacheSnapshot()
        XCTAssertEqual(cache.entryCount, 0)
        XCTAssertEqual(cache.reservationCount, 0)
    }

    func testBestEffortFinalSnapshotCannotEvictPrimaryPromptEntry()
        async throws
    {
        let driver = try makeDriver(
            configuration: cacheConfiguration(entries: 1))
        let context = try request()
        let prompt = [11, 12, 13]

        let cold = try await driver.generate(
            prompt: prompt, config: config(context))
        let repeatRequest = try await driver.generate(
            prompt: prompt, config: config(context))

        XCTAssertEqual(cold.tokens, repeatRequest.tokens)
        XCTAssertEqual(
            repeatRequest.requestStartMetrics?.prefixCacheOutcome,
            .exactHit)
        let cache = await driver.engine.exactPrefixCacheSnapshot()
        XCTAssertEqual(cache.entryCount, 1)
    }

    func testZeroOutputDoesNotPublishAndWarmupDoesNotCreateEntries()
        async throws
    {
        let driver = try makeDriver(
            configuration: cacheConfiguration())
        let context = try request()
        let prompt = [5, 6, 7]

        let warmupSeconds =
            try await driver.engine.performExactPrefixWarmup()
        XCTAssertGreaterThanOrEqual(warmupSeconds, 0)
        let postWarmup =
            await driver.engine.exactPrefixCacheSnapshot()
        XCTAssertEqual(postWarmup.entryCount, 0)

        let empty = try await driver.generate(
            prompt: prompt,
            config: config(context, maxTokens: 0))
        XCTAssertTrue(empty.tokens.isEmpty)
        XCTAssertNil(empty.requestStartMetrics)
        let postEmpty =
            await driver.engine.exactPrefixCacheSnapshot()
        XCTAssertEqual(postEmpty.entryCount, 0)

        let firstReal = try await driver.generate(
            prompt: prompt, config: config(context))
        XCTAssertEqual(
            firstReal.requestStartMetrics?.prefixCacheOutcome, .miss)
    }

    func testPrefixRequestRejectsSpeculativeAndNonFP16Routes()
        async throws
    {
        let driver = try makeDriver(
            configuration: cacheConfiguration())
        let context = try request()

        for rejected in [
            config(
                context,
                kvQuant: "affine-k4v2-g64"),
            config(context, specDecode: "pld"),
        ] {
            await XCTAssertThrowsErrorAsync {
                _ = try await driver.generate(
                    prompt: [2, 3, 4], config: rejected)
            }
        }
        let postRejection =
            await driver.engine.exactPrefixCacheSnapshot()
        XCTAssertEqual(postRejection.entryCount, 0)
    }
}

private func XCTAssertThrowsErrorAsync<T>(
    _ expression: () async throws -> T,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await expression()
        XCTFail("expected error", file: file, line: line)
    } catch {}
}
