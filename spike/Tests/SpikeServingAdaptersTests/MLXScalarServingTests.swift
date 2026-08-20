import Foundation
import XCTest

import MLXLMCommon
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class MLXScalarServingTests: XCTestCase {
    func testCodecRendersExactOpenAIRolesAndContentThroughChatTemplate() throws {
        let codec = MLXScalarTextCodec(tokenizer: FixtureTokenizer())

        let tokens = try codec.render(
            messages: [
                OpenAIChatMessage(role: .developer, text: "developer text"),
                OpenAIChatMessage(role: .system, text: "system text"),
                OpenAIChatMessage(role: .user, text: "user text"),
                OpenAIChatMessage(role: .assistant, text: "assistant text"),
            ],
            tools: [],
            enableThinking: nil,
            reasoningEffort: nil)

        XCTAssertEqual(tokens, [41, 42])
    }

    func testCodecDetokenizerPublishesExactIncrementalSuffixes() {
        let codec = MLXScalarTextCodec(tokenizer: FixtureTokenizer())
        var detokenizer = codec.makeDetokenizer()

        detokenizer.append(token: 1)
        XCTAssertEqual(detokenizer.next(), "hel")
        detokenizer.append(token: 2)
        XCTAssertEqual(detokenizer.next(), "lo")
        detokenizer.append(token: 3)
        XCTAssertEqual(detokenizer.next(), "\n")
    }

    func testStopTokenResolutionUnionsConfigurationTokenizerExtrasAndUnknown() throws {
        let configuration = ModelConfiguration(
            directory: URL(fileURLWithPath: "/tmp/fixture-model"),
            extraEOSTokens: ["<turn>"],
            eosTokenIds: [7])

        let ids = try resolveScalarServingStopTokenIDs(
            configuration: configuration,
            tokenizer: FixtureTokenizer())

        XCTAssertEqual(ids, [7, 8, 9, 10])
    }

    func testNativeCacheClassifierSeparatesDenseRotatingRecurrentAndComposite() {
        let kinds = classifyScalarServingNativeCaches([
            KVCacheSimple(),
            RotatingKVCache(maxSize: 128, keep: 4),
            MambaCache(),
            CacheList(KVCacheSimple(), MambaCache()),
        ])

        XCTAssertEqual(
            kinds,
            [
                .denseAttention,
                .rotatingAttention,
                .recurrentState,
                .composite,
            ])
    }

    func testResetParityPreflightAcceptsTwoExactOneTokenRuns() async throws {
        let result = try await verifyScalarServingResetParity(
            inference: InferenceActor(
                decoder: ScriptedDecoder(script: [1, 99], eos: 99)),
            promptTokens: [10, 11],
            stopTokenIDs: [99])

        XCTAssertEqual(
            result,
            ScalarServingStartupParity(
                promptTokenCount: 2,
                generatedTokenCount: 1,
                verified: true))
    }

    func testResetParityPreflightRejectsMismatchedRuns() async throws {
        do {
            _ = try await verifyScalarServingResetParity(
                inference: InferenceActor(
                    decoder: ResetSensitiveDecoder()),
                promptTokens: [10],
                stopTokenIDs: [99])
            XCTFail("Expected startup parity rejection")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .startupParityMismatch)
        }
    }

    func testLoaderConfigurationRejectsNonFileURLAndUnsafeMemoryPolicy() {
        let backend = ScalarServingBackendConfiguration(
            defaultMaximumCompletionTokens: 32,
            maximumQueuedRequests: 2,
            queueRetryAfterSeconds: 1,
            mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096))

        XCTAssertThrowsError(
            try validateScalarServingModelLoadConfiguration(
                ScalarServingModelLoadConfiguration(
                    launchedModel: "fixture",
                    modelDirectory: URL(string: "relative-model")!,
                    memoryLimitBytes: 4_096,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: backend))
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .modelDirectoryMustBeAbsolute)
        }

        XCTAssertThrowsError(
            try validateScalarServingModelLoadConfiguration(
                ScalarServingModelLoadConfiguration(
                    launchedModel: "fixture",
                    modelDirectory: URL(fileURLWithPath: "/tmp"),
                    memoryLimitBytes: 1_024,
                    cacheLimitBytes: 2_048,
                    backendConfiguration: backend))
        ) { error in
            XCTAssertEqual(
                error as? ScalarServingModelLoadError,
                .cacheLimitExceedsMemoryLimit)
        }
    }

    func testLoaderConfigurationAcceptsExistingAbsoluteDirectoryAndExplicitLimits() throws {
        let configuration = ScalarServingModelLoadConfiguration(
            launchedModel: "fixture",
            modelDirectory: URL(fileURLWithPath: "/tmp"),
            memoryLimitBytes: 4_096,
            cacheLimitBytes: 1_024,
            backendConfiguration: .init(
                defaultMaximumCompletionTokens: 32,
                maximumQueuedRequests: 2,
                queueRetryAfterSeconds: 1,
                mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096)))

        let validated = try validateScalarServingModelLoadConfiguration(
            configuration)

        XCTAssertEqual(validated.launchedModel, "fixture")
        XCTAssertEqual(validated.modelDirectory.path, "/tmp")
        XCTAssertEqual(validated.memoryLimitBytes, 4_096)
        XCTAssertEqual(validated.cacheLimitBytes, 1_024)
    }

    func testStartupReportMemoryFieldsFragmentRendersSnakeCaseBytes() {
        let report = ScalarServingModelStartupReport(
            launchedModel: "fixture",
            route: .scalarGreedy,
            memoryLimitBytes: 8,
            cacheLimitBytes: 4,
            stopTokenCount: 1,
            stopStringCount: 0,
            nativeCacheKinds: [.denseAttention],
            startupPromptTokenCount: 2,
            startupGeneratedTokenCount: 1,
            resetParityVerified: true,
            mlxActiveBytes: 5_540_000_000,
            mlxCacheBytes: 1_073_741_824,
            mlxPeakBytes: 6_000_000_000)
        XCTAssertEqual(
            report.memoryFieldsFragment,
            "mlx_active_bytes=5540000000 mlx_cache_bytes=1073741824 "
                + "mlx_peak_bytes=6000000000")
    }

    func testStartupReportMemoryFieldsDefaultToZeroWhenUnspecified() {
        // Backward-compatible init: existing construction sites that don't pass MLX
        // memory get a well-defined zero fragment, never a crash or garbage bytes.
        let report = ScalarServingModelStartupReport(
            launchedModel: "fixture",
            route: .scalarGreedy,
            memoryLimitBytes: 8,
            cacheLimitBytes: 4,
            stopTokenCount: 1,
            stopStringCount: 0,
            nativeCacheKinds: [.denseAttention],
            startupPromptTokenCount: 2,
            startupGeneratedTokenCount: 1,
            resetParityVerified: true)
        XCTAssertEqual(
            report.memoryFieldsFragment,
            "mlx_active_bytes=0 mlx_cache_bytes=0 mlx_peak_bytes=0")
    }

    // MARK: - qwen3_5 scalar-route gated-delta kernel viability guard (Dk%32)

    /// VL-wrapped qwen3_5 config with `linear_key_head_dim = 48` — a valid positive-integer geometry the
    /// gated-delta Metal kernel cannot serve (Dk not divisible by 32). Mirrors the continuous adapter's
    /// incr-4 fixture, exercised here on the DEFAULT scalar route the family falls back to.
    private func qwen35UnalignedDkConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
         "text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144,
           "vocab_size":248320,"num_hidden_layers":48,"full_attention_interval":4,
           "num_key_value_heads":8,"head_dim":128,"torch_dtype":"bfloat16",
           "linear_num_key_heads":16,"linear_num_value_heads":32,
           "linear_key_head_dim":48,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
    }

    /// Same shape with `linear_key_head_dim = 128` (a multiple of 32) — the kernel can serve it, so the
    /// pre-load guard must NOT fire (the probe returns 128 and 128 % 32 == 0).
    private func qwen35AlignedDkConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
         "text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144,
           "vocab_size":248320,"num_hidden_layers":48,"full_attention_interval":4,
           "num_key_value_heads":8,"head_dim":128,"torch_dtype":"bfloat16",
           "linear_num_key_heads":16,"linear_num_value_heads":32,
           "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
    }

    private func writeConfigDirectory(_ json: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("scalar-serving-guard-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data(json.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        return directory
    }

    /// Acceptance: a qwen3_5 checkpoint whose Dk is not a multiple of 32 is refused on the scalar route
    /// BEFORE any weight load — the family's default (un-flagged) path fails closed rather than
    /// truncating/faulting in the gated-delta Metal kernel at decode.
    func testScalarQwen35RejectsUnalignedKeyHeadDimBeforeWeightLoad() async throws {
        let directory = try writeConfigDirectory(qwen35UnalignedDkConfigJSON())
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            _ = try await loadScalarServingModel(
                configuration: ScalarServingModelLoadConfiguration(
                    launchedModel: "qwen3_5-scalar-fallback",
                    modelDirectory: directory,
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: ScalarServingBackendConfiguration(
                        defaultMaximumCompletionTokens: 12,
                        maximumQueuedRequests: 1,
                        queueRetryAfterSeconds: 1,
                        mailboxCapacity: .init(maxDeltas: 4, maxBytes: 16 * 1_024))))
            XCTFail("Dk not divisible by 32 must fail closed before weight load")
        } catch let error as ScalarServingModelLoadError {
            XCTAssertEqual(error, .hybridKernelKeyHeadDimUnaligned(48))
        }
    }

    /// The pre-load probe reads the qwen3_5 recurrent Dk from config.json without loading weights: 48
    /// (unaligned) and 128 (aligned) are surfaced exactly; a dense (non-qwen3_5) config and an absent
    /// config both return nil, so the guard is a strict no-op off the hybrid family.
    func testScalarQwen35KeyHeadDimProbeIsFamilyScoped() throws {
        let unaligned = try writeConfigDirectory(qwen35UnalignedDkConfigJSON())
        defer { try? FileManager.default.removeItem(at: unaligned) }
        XCTAssertEqual(
            scalarServingQwen35RecurrentKeyHeadDim(modelDirectory: unaligned), 48)

        let aligned = try writeConfigDirectory(qwen35AlignedDkConfigJSON())
        defer { try? FileManager.default.removeItem(at: aligned) }
        XCTAssertEqual(
            scalarServingQwen35RecurrentKeyHeadDim(modelDirectory: aligned), 128)

        let dense = try writeConfigDirectory(#"{"model_type":"qwen3","num_hidden_layers":4}"#)
        defer { try? FileManager.default.removeItem(at: dense) }
        XCTAssertNil(scalarServingQwen35RecurrentKeyHeadDim(modelDirectory: dense))

        let absent = FileManager.default.temporaryDirectory
            .appendingPathComponent("scalar-serving-guard-absent-\(UUID().uuidString)", isDirectory: true)
        XCTAssertNil(scalarServingQwen35RecurrentKeyHeadDim(modelDirectory: absent))
    }
}

private enum FixtureTokenizerError: Error {
    case unexpectedMessages
}

private struct FixtureTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        []
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let pieces = [
            1: "hel",
            2: "lo",
            3: "\n",
        ]
        return tokenIds.compactMap { pieces[$0] }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        [
            "<eos>": 8,
            "<turn>": 9,
            "<unk>": 10,
        ][token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        nil
    }

    var bosToken: String? { "<bos>" }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { "<unk>" }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let expected = [
            ("developer", "developer text"),
            ("system", "system text"),
            ("user", "user text"),
            ("assistant", "assistant text"),
        ]
        guard messages.count == expected.count,
            tools == nil,
            additionalContext == nil
        else {
            throw FixtureTokenizerError.unexpectedMessages
        }
        for (message, expected) in zip(messages, expected) {
            guard message["role"] as? String == expected.0,
                message["content"] as? String == expected.1
            else {
                throw FixtureTokenizerError.unexpectedMessages
            }
        }
        return [41, 42]
    }
}

private struct ResetSensitiveDecoder: Decoder {
    private var resetCount = 0

    mutating func prefill(_ promptTokens: [Int]) -> Int {
        resetCount < 3 ? 1 : 2
    }

    mutating func step(last: Int) -> Int {
        99
    }

    mutating func reset() {
        resetCount += 1
    }
}
