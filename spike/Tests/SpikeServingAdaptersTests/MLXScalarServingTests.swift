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
            ])

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
