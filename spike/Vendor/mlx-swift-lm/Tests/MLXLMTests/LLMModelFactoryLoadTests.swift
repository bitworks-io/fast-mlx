// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXLLM
import XCTest

/// Coverage for the `LLMModelFactory._load` extraction seam: `resolveGenerationSettings`
/// (config.json + generation_config.json -> eosTokenIds/stopStrings/toolCallFormat) and
/// `assembleModelContext` (messageGenerator selection + tokenizerSource rule + ModelContext
/// assembly). This is the regression guard for a second model-loading path that must reuse the
/// same tail without silently dropping a step (e.g. `ToolCallFormat.infer`, or overwriting an
/// explicitly supplied `toolCallFormat`).
final class LLMModelFactoryLoadTests: XCTestCase {

    // MARK: - resolveGenerationSettings

    func testGenerationConfigEosTokenIdsOverridesConfigJsonEosTokenIds() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(
            json: #"{"eos_token_id": [42]}"#, to: directory, filename: "generation_config.json")

        let configData = Self.configData(modelType: "qwen2", eosTokenId: [1, 2])
        let baseConfig = try Self.baseConfig(from: configData)
        let configuration = Self.configuration(directory: directory, eosTokenIds: [1, 2])

        let resolved = resolveGenerationSettings(
            into: configuration,
            baseConfig: baseConfig,
            configData: configData,
            modelDirectory: directory)

        XCTAssertEqual(resolved.eosTokenIds, [42])
    }

    func testAbsentGenerationConfigLeavesConfigJsonEosTokenIdsIntact() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // Deliberately do not write generation_config.json.

        let configData = Self.configData(modelType: "qwen2", eosTokenId: [7, 8])
        let baseConfig = try Self.baseConfig(from: configData)
        let configuration = Self.configuration(directory: directory, eosTokenIds: [])

        let resolved = resolveGenerationSettings(
            into: configuration,
            baseConfig: baseConfig,
            configData: configData,
            modelDirectory: directory)

        XCTAssertEqual(resolved.eosTokenIds, [7, 8])
    }

    func testUnreadableGenerationConfigLeavesConfigJsonEosTokenIdsIntact() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // A directory at the expected file path makes `Data(contentsOf:)` fail to read it.
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("generation_config.json"),
            withIntermediateDirectories: true)

        let configData = Self.configData(modelType: "qwen2", eosTokenId: [7, 8])
        let baseConfig = try Self.baseConfig(from: configData)
        let configuration = Self.configuration(directory: directory, eosTokenIds: [])

        let resolved = resolveGenerationSettings(
            into: configuration,
            baseConfig: baseConfig,
            configData: configData,
            modelDirectory: directory)

        XCTAssertEqual(resolved.eosTokenIds, [7, 8])
    }

    func testMalformedGenerationConfigLeavesConfigJsonEosTokenIdsIntactAndDoesNotThrow() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(json: "{ this is not valid json", to: directory, filename: "generation_config.json")

        let configData = Self.configData(modelType: "qwen2", eosTokenId: [7, 8])
        let baseConfig = try Self.baseConfig(from: configData)
        let configuration = Self.configuration(directory: directory, eosTokenIds: [])

        let resolved = resolveGenerationSettings(
            into: configuration,
            baseConfig: baseConfig,
            configData: configData,
            modelDirectory: directory)

        XCTAssertEqual(resolved.eosTokenIds, [7, 8])
    }

    func testGenerationConfigStopStringsAreUnionedIntoExistingStopStrings() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.write(
            json: #"{"stop_strings": ["<gen_stop>"]}"#, to: directory,
            filename: "generation_config.json")

        let configData = Self.configData(modelType: "qwen2", eosTokenId: [])
        let baseConfig = try Self.baseConfig(from: configData)
        let configuration = Self.configuration(
            directory: directory, eosTokenIds: [], stopStrings: ["<existing_stop>"])

        let resolved = resolveGenerationSettings(
            into: configuration,
            baseConfig: baseConfig,
            configData: configData,
            modelDirectory: directory)

        XCTAssertEqual(resolved.stopStrings, ["<existing_stop>", "<gen_stop>"])
    }

    func testToolCallFormatIsInferredFromModelTypeWhenIncomingFormatIsNil() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let configData = Self.configData(modelType: "glm4", eosTokenId: [])
        let baseConfig = try Self.baseConfig(from: configData)
        let configuration = Self.configuration(
            directory: directory, eosTokenIds: [], toolCallFormat: nil)

        let resolved = resolveGenerationSettings(
            into: configuration,
            baseConfig: baseConfig,
            configData: configData,
            modelDirectory: directory)

        // Sanity: confirm ToolCallFormat.infer really does map "glm4" to .glm4, so this test
        // is exercising the inference fallback and not a coincidence.
        XCTAssertEqual(ToolCallFormat.infer(from: "glm4", configData: configData), .glm4)
        XCTAssertEqual(resolved.toolCallFormat, .glm4)
    }

    func testExplicitToolCallFormatIsPreservedNotOverwrittenByInference() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // "glm4" would infer .glm4 (see the sibling test above) -- an explicitly supplied
        // .json format must survive untouched. This is the regression this extraction exists
        // to protect: a duplicated tail that re-runs inference unconditionally would flip this.
        let configData = Self.configData(modelType: "glm4", eosTokenId: [])
        let baseConfig = try Self.baseConfig(from: configData)
        let configuration = Self.configuration(
            directory: directory, eosTokenIds: [], toolCallFormat: .json)

        let resolved = resolveGenerationSettings(
            into: configuration,
            baseConfig: baseConfig,
            configData: configData,
            modelDirectory: directory)

        XCTAssertEqual(resolved.toolCallFormat, .json)
    }

    // MARK: - assembleModelContext

    func testAssembleModelContextUsesDirectoryTokenizerSourceWhenTokenizerDirectoryDiffersFromModelDirectory() {
        let modelDirectory = URL(fileURLWithPath: "/tmp/LLMModelFactoryLoadTests-model-\(UUID().uuidString)")
        let tokenizerDirectory = URL(fileURLWithPath: "/tmp/LLMModelFactoryLoadTests-tok-\(UUID().uuidString)")
        let configuration = ResolvedModelConfiguration(
            modelDirectory: modelDirectory,
            tokenizerDirectory: tokenizerDirectory,
            name: "test/model",
            defaultPrompt: "hi",
            extraEOSTokens: [],
            eosTokenIds: [],
            toolCallFormat: nil)

        let context = assembleModelContext(
            model: FakeLanguageModel(), tokenizer: TestTokenizer(), configuration: configuration)

        XCTAssertEqual(context.configuration.tokenizerSource, .directory(tokenizerDirectory))
    }

    func testAssembleModelContextUsesNilTokenizerSourceWhenTokenizerDirectoryMatchesModelDirectory() {
        let directory = URL(fileURLWithPath: "/tmp/LLMModelFactoryLoadTests-same-\(UUID().uuidString)")
        let configuration = ResolvedModelConfiguration(
            modelDirectory: directory,
            tokenizerDirectory: directory,
            name: "test/model",
            defaultPrompt: "hi",
            extraEOSTokens: [],
            eosTokenIds: [],
            toolCallFormat: nil)

        let context = assembleModelContext(
            model: FakeLanguageModel(), tokenizer: TestTokenizer(), configuration: configuration)

        XCTAssertNil(context.configuration.tokenizerSource)
    }

    func testAssembleModelContextCarriesResolvedSettingsIntoModelConfiguration() {
        let directory = URL(fileURLWithPath: "/tmp/LLMModelFactoryLoadTests-carry-\(UUID().uuidString)")
        let configuration = ResolvedModelConfiguration(
            modelDirectory: directory,
            tokenizerDirectory: directory,
            name: "test/model",
            defaultPrompt: "a default prompt",
            extraEOSTokens: ["<extra_eos>"],
            stopStrings: ["<stop_a>", "<stop_b>"],
            eosTokenIds: [11, 22],
            toolCallFormat: .lfm2)

        let context = assembleModelContext(
            model: FakeLanguageModel(), tokenizer: TestTokenizer(), configuration: configuration)

        XCTAssertEqual(context.configuration.defaultPrompt, "a default prompt")
        XCTAssertEqual(context.configuration.extraEOSTokens, ["<extra_eos>"])
        XCTAssertEqual(context.configuration.stopStrings, ["<stop_a>", "<stop_b>"])
        XCTAssertEqual(context.configuration.eosTokenIds, [11, 22])
        XCTAssertEqual(context.configuration.toolCallFormat, .lfm2)
        if case .directory(let resultDirectory) = context.configuration.id {
            XCTAssertEqual(resultDirectory, directory)
        } else {
            XCTFail("expected a directory-backed ModelConfiguration identifier")
        }
    }

    func testAssembleModelContextUsesModelSuppliedMessageGeneratorForLLMModel() async throws {
        let directory = URL(fileURLWithPath: "/tmp/LLMModelFactoryLoadTests-llm-\(UUID().uuidString)")
        let configuration = ResolvedModelConfiguration(
            modelDirectory: directory,
            tokenizerDirectory: directory,
            name: "test/model",
            defaultPrompt: "",
            extraEOSTokens: [],
            eosTokenIds: [],
            toolCallFormat: nil)
        let tokenizer = RecordingTokenizer()

        let context = assembleModelContext(
            model: FakeLLMModel(generator: NoSystemMessageGenerator()),
            tokenizer: tokenizer,
            configuration: configuration)

        let input = UserInput(chat: [.system("system prompt"), .user("hello")])
        _ = try await context.processor.prepare(input: input)

        // NoSystemMessageGenerator drops the system message -- only the LLMModel's own
        // generator (not DefaultMessageGenerator, which would keep both) can produce this.
        XCTAssertEqual(tokenizer.lastMessages?.count, 1)
        XCTAssertEqual(tokenizer.lastMessages?.first?["role"] as? String, "user")
    }

    func testAssembleModelContextUsesDefaultMessageGeneratorForNonLLMModel() async throws {
        let directory = URL(fileURLWithPath: "/tmp/LLMModelFactoryLoadTests-default-\(UUID().uuidString)")
        let configuration = ResolvedModelConfiguration(
            modelDirectory: directory,
            tokenizerDirectory: directory,
            name: "test/model",
            defaultPrompt: "",
            extraEOSTokens: [],
            eosTokenIds: [],
            toolCallFormat: nil)
        let tokenizer = RecordingTokenizer()

        let context = assembleModelContext(
            model: FakeLanguageModel(),
            tokenizer: tokenizer,
            configuration: configuration)

        let input = UserInput(chat: [.system("system prompt"), .user("hello")])
        _ = try await context.processor.prepare(input: input)

        // A plain LanguageModel (not an LLMModel) falls back to DefaultMessageGenerator, which
        // keeps every message, including the system one.
        XCTAssertEqual(tokenizer.lastMessages?.count, 2)
    }

    // MARK: - Fixtures

    private static func makeTempDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("LLMModelFactoryLoadTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func write(json: String, to directory: URL, filename: String) throws {
        try json.data(using: .utf8)!.write(to: directory.appendingPathComponent(filename))
    }

    private static func configData(modelType: String, eosTokenId: [Int]) -> Data {
        let payload: [String: Any] = ["model_type": modelType, "eos_token_id": eosTokenId]
        return try! JSONSerialization.data(withJSONObject: payload)
    }

    private static func baseConfig(from configData: Data) throws -> BaseConfiguration {
        try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
    }

    private static func configuration(
        directory: URL, eosTokenIds: Set<Int>, stopStrings: Set<String>? = nil,
        toolCallFormat: ToolCallFormat? = nil
    ) -> ResolvedModelConfiguration {
        ResolvedModelConfiguration(
            modelDirectory: directory,
            tokenizerDirectory: directory,
            name: "test/model",
            defaultPrompt: "",
            extraEOSTokens: [],
            stopStrings: stopStrings,
            eosTokenIds: eosTokenIds,
            toolCallFormat: toolCallFormat)
    }
}

// MARK: - Test doubles

/// A minimal `LanguageModel` that is *not* an `LLMModel`, to exercise the
/// `DefaultMessageGenerator` fallback branch of `assembleModelContext`.
private final class FakeLanguageModel: Module, LanguageModel {
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("not exercised by these tests")
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }
}

/// A minimal `LLMModel` with an injectable `MessageGenerator`, to exercise the
/// model-supplied-generator branch of `assembleModelContext`.
private final class FakeLLMModel: Module, LLMModel {
    var loraLayers: [Module] { [] }
    private let generator: MessageGenerator

    init(generator: MessageGenerator) {
        self.generator = generator
        super.init()
    }

    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        generator
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("not exercised by these tests")
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }
}

/// A tokenizer that records the messages passed to `applyChatTemplate`, so tests can observe
/// which `MessageGenerator` `assembleModelContext` wired up without depending on
/// `LLMUserInputProcessor`'s file-private concrete type. Mutation happens synchronously within
/// a single test body, never concurrently, so `@unchecked Sendable` is safe here.
private final class RecordingTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private(set) var lastMessages: [[String: any Sendable]]?

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? = nil
    var eosToken: String? = nil
    var unknownToken: String? = nil

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        lastMessages = messages
        return []
    }
}
