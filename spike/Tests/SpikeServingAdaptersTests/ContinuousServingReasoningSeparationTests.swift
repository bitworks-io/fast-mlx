import Foundation
import os
import XCTest

import HarnessCore
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

/// The default continuous-batch route must separate `<think>` reasoning exactly like the scalar route:
/// `ContinuousServingBackend.start` resolves `enable_thinking` ONCE, renders the prompt from it, and gates
/// `ServingGenerationHandle.separatesReasoning` from the SAME value folded with the family's
/// `thinksByDefault` (computed at load by `continuousServingThinksByDefault`). Before this fix the handle
/// was built without `separatesReasoning:`, so a streamed thinking-on reply arrived as raw
/// `<think>…</think>answer` in `delta.content` and a truncated reply never became `reasoning_content`.
///
/// The handle-flag -> wire behavior (streaming splitter, truncated `<think>…` with `finish_reason=length`
/// becoming `reasoning_content` with empty content, and the non-separating literal-`<think>` answer-loss
/// guard) is already pinned in `ServingNIOTests/OpenAIChatCompletionsHTTPHandlerTests.swift`
/// (`testNonStreamingTruncatedThinkingReturnsReasoningContentAndEmptyContent`,
/// `testNonStreamingNonSeparatingKeepsLiteralThinkAsContent`, and the streaming siblings), so these tests
/// pin only the half that was missing: that this backend sets the flag.
final class ContinuousServingReasoningSeparationTests: XCTestCase {
    // MARK: - Per-request handle flag

    func testThinkingOnChatSeparatesWhenFamilyThinksByDefault() async throws {
        let backend = makeSeparationBackend(thinksByDefault: true)
        let handle = try await backend.start(separationRequest())
        XCTAssertTrue(handle.separatesReasoning)
        await finish(handle, backend)
    }

    func testExplicitThinkingOffDoesNotSeparate() async throws {
        let backend = makeSeparationBackend(thinksByDefault: true)
        let handle = try await backend.start(separationRequest(enableThinking: false))
        XCTAssertFalse(handle.separatesReasoning)
        await finish(handle, backend)
    }

    /// The legacy tool workaround forces the resolved flag to false when tools are attached and the
    /// client did not set it. The gate must consume that SAME resolved value (not the raw request flag),
    /// or it would split a stream whose prompt was rendered thinking-off.
    func testToolsRequestWithWorkaroundDoesNotSeparate() async throws {
        let backend = makeSeparationBackend(
            thinksByDefault: true, disableThinkingWhenToolsActive: true)
        let handle = try await backend.start(separationRequest(tools: [separationWeatherTool]))
        XCTAssertFalse(handle.separatesReasoning)
        await finish(handle, backend)
    }

    func testToolsRequestWithExplicitThinkingOnSeparates() async throws {
        let backend = makeSeparationBackend(
            thinksByDefault: true, disableThinkingWhenToolsActive: true)
        let handle = try await backend.start(
            separationRequest(tools: [separationWeatherTool], enableThinking: true))
        XCTAssertTrue(handle.separatesReasoning)
        await finish(handle, backend)
    }

    /// A family that does not think by default never separates, with the flag omitted or explicitly on:
    /// a plain answer that begins with a literal `<think>` must stay `content` (the answer-loss guard).
    func testNonThinkingFamilyNeverSeparates() async throws {
        let backend = makeSeparationBackend(thinksByDefault: false)
        let omitted = try await backend.start(separationRequest())
        XCTAssertFalse(omitted.separatesReasoning)
        _ = await omitted.lease.cancel(.clientDisconnected)

        let explicitOn = try await backend.start(separationRequest(enableThinking: true))
        XCTAssertFalse(explicitOn.separatesReasoning)
        await finish(explicitOn, backend)
    }

    func testConfigurationDefaultsThinksByDefaultToFalse() {
        let configuration = ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 2, maxBytes: 4_096))
        XCTAssertFalse(configuration.thinksByDefault)
    }

    // MARK: - Loader helper

    private let thinkTemplate = "{% if x %}<think>{% endif %}reasoning</think>answer"

    private func modelDirectory(
        modelType: String?,
        template: String?
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("continuous-thinks-by-default-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let modelType {
            try Data(#"{"model_type": "\#(modelType)"}"#.utf8).write(
                to: directory.appendingPathComponent("config.json"))
        }
        if let template {
            try Data(template.utf8).write(to: directory.appendingPathComponent("chat_template.jinja"))
        }
        return directory
    }

    func testLoaderThinksByDefaultForDenseQwen3WithAttestingTemplate() throws {
        let directory = try modelDirectory(modelType: "qwen3", template: thinkTemplate)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertTrue(
            continuousServingThinksByDefault(modelDirectory: directory, hybridAdmitted: false))
    }

    /// `qwen3` is in the `.compiled` allowlist only; the hybrid-admitted route is `.nativeHeterogeneous`,
    /// where `qwen3` is not an attested family.
    func testLoaderDoesNotThinkByDefaultForQwen3OnHybridAdmittedRoute() throws {
        let directory = try modelDirectory(modelType: "qwen3", template: thinkTemplate)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertFalse(
            continuousServingThinksByDefault(modelDirectory: directory, hybridAdmitted: true))
    }

    func testLoaderThinksByDefaultForQwen35OnHybridAdmittedRoute() throws {
        let directory = try modelDirectory(modelType: "qwen3_5", template: thinkTemplate)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertTrue(
            continuousServingThinksByDefault(modelDirectory: directory, hybridAdmitted: true))
    }

    func testLoaderDoesNotThinkByDefaultForUnattestedFamily() throws {
        let directory = try modelDirectory(modelType: "llama", template: thinkTemplate)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertFalse(
            continuousServingThinksByDefault(modelDirectory: directory, hybridAdmitted: false))
    }

    func testLoaderDoesNotThinkByDefaultWhenTemplateLacksMarkers() throws {
        let directory = try modelDirectory(
            modelType: "qwen3", template: "{% if x %}plain{% endif %}answer only")
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertFalse(
            continuousServingThinksByDefault(modelDirectory: directory, hybridAdmitted: false))
    }

    func testLoaderDoesNotThinkByDefaultWhenConfigIsMissing() throws {
        let directory = try modelDirectory(modelType: nil, template: thinkTemplate)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertFalse(
            continuousServingThinksByDefault(modelDirectory: directory, hybridAdmitted: false))
    }
}

// MARK: - Fixtures

private func finish(
    _ handle: ServingGenerationHandle,
    _ backend: ContinuousServingBackend
) async {
    _ = await handle.lease.cancel(.clientDisconnected)
    await backend.shutdown()
}

private func makeSeparationBackend(
    thinksByDefault: Bool,
    disableThinkingWhenToolsActive: Bool = false
) -> ContinuousServingBackend {
    let coordinator = ContinuousBatchCoordinator(
        configuration: try! ContinuousBatchConfiguration(
            maxActiveSlots: 2, maxPrefillSlots: 2, prefillChunkSize: 8, maxQueuedRequests: 4),
        runtime: SeparationFixtureRuntime(),
        automaticDrive: false,
        publicationCapacity: 1,
        traceLimit: 32)
    return ContinuousServingBackend(
        launchedModel: "fixture",
        coordinator: coordinator,
        codec: SeparationFixtureCodec(),
        stopTokenIDs: [99],
        modelStopStrings: [],
        configuration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 2, maxBytes: 4_096),
            disableThinkingWhenToolsActive: disableThinkingWhenToolsActive,
            thinksByDefault: thinksByDefault))
}

private func separationRequest(
    tools: [OpenAIToolSpec] = [],
    enableThinking: Bool? = nil
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: "fixture",
        messages: [OpenAIChatMessage(role: .user, text: "prompt")],
        maxCompletionTokens: 2,
        temperature: 0,
        choiceCount: 1,
        stream: true,
        stop: [],
        tools: tools,
        toolChoice: tools.isEmpty ? .none : .auto,
        enableThinking: enableThinking)
}

private let separationWeatherTool = OpenAIToolSpec(
    name: "get_weather",
    description: "Look up the weather for a city",
    parameters: .object(["type": .string("object")]),
    raw: .object([
        "type": .string("function"),
        "function": .object([
            "name": .string("get_weather"),
            "description": .string("Look up the weather for a city"),
            "parameters": .object(["type": .string("object")]),
        ]),
    ]))

private struct SeparationFixtureCodec: ScalarServingTextCodec {
    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?,
        addGenerationPrompt: Bool?
    ) throws -> [Int] {
        [10]
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        SeparationFixtureDetokenizer()
    }
}

private struct SeparationFixtureDetokenizer: ScalarServingDetokenizer {
    mutating func append(token: Int) {}
    mutating func next() -> String? { nil }
}

/// Admission-only runtime: these tests inspect the handle returned by `start` and never drive decode.
private final class SeparationFixtureRuntime: ContinuousBatchRuntime, @unchecked Sendable {
    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {}

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 64,
            reservedKVBytes: 0,
            maxReservedKVBytes: 4_096)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {}

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        []
    }

    func remove(_ id: BatchRequestID) {}
}
