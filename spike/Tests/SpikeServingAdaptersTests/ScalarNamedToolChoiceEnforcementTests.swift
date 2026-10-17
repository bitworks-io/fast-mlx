import Foundation
import XCTest
import os

import MLX
import MLXLMCommon
import MLXNN
import NIOCore
import NIOEmbedded
import NIOHTTP1
import ServingCore
import ServingNIO
import SpikeCore
@testable import SpikeServingAdapters

// A named-function `tool_choice` is ENFORCED on the plain scalar route (cycle 257).
//
// Everything here runs the REAL `ScalarServingBackend` -> `InferenceActor` -> `MLXDecoder` with a
// scripted fake `LanguageModel`, and (for the wire-shape tests) through the REAL
// `OpenAIChatCompletionsHTTPHandler` on an embedded channel, so decode, admission, the mask, the
// tool-call assembly and both response encodings are all exercised together.
//
// The fake model's UNCONSTRAINED greedy output is prose ("Sure it!"); the JSON it would write if the
// grammar forced it is a lower-ranked preference at every position. So an unmasked run returns the
// prose, and only a really-installed mask can produce the arguments.

// MARK: - Fixtures

/// Byte vocabulary: ids 0...255 are raw bytes, 256 is EOS, 257 is the (banned-for-JSON) `</think>`.
private let eosID = 256
private let thinkEndID = 257
private let vocabularyWidth = 258

private func ids(_ text: String) -> [Int] { text.utf8.map { Int($0) } }

private func classifications() -> [TokenByteClassification] {
    var all: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }
    all.append(.eos)  // 256
    all.append(.banned)  // 257, </think>
    return all
}

private func schemaSupport(thinkEndTokenID: Int?) -> ScalarServingJSONSchemaConstraintSupport {
    let object = ScalarServingJSONObjectConstraintSupport(
        classifications: classifications(), thinkEndTokenID: thinkEndTokenID)
    return loadScalarServingJSONSchemaConstraintSupport(sharing: object)!
}

/// Position-scripted fake model. After consuming the prompt token (id 10) it is at position 0; every
/// other consumed token advances the position. At position `p` the logits are: `prose[p]` = 5,
/// `json[p]` = 4 (when in range), EOS = 5 once `p >= prose.count`, everything else 0. So greedy
/// without a mask follows `prose` and then stops; a mask that forbids `prose[p]` falls through to
/// `json[p]`. `forwardCalls` counts model invocations (a refused request must leave it at 0).
private final class PositionScriptedModel: Module, LanguageModel, KVCacheDimensionProvider,
    @unchecked Sendable
{
    let kvHeads: [Int] = [1]
    private let prose: [Int]
    private let json: [Int]
    private let counters = OSAllocatedUnfairLock(initialState: (position: 0, calls: 0))

    init(prose: [Int], json: [Int]) {
        self.prose = prose
        self.json = json
    }

    var forwardCalls: Int { counters.withLock { $0.calls } }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func evaluateThrowing(
        _ input: LMInput.Text, cache: [KVCache]?, state _: LMOutput.State?
    ) throws -> LMOutput {
        let tokens = input.tokens.asType(.int32).asArray(Int32.self)
        var rows: [Float] = []
        counters.withLock { $0.calls += 1 }
        for token in tokens {
            let position = counters.withLock { state -> Int in
                state.position = (token == 10) ? 0 : state.position + 1
                return state.position
            }
            var row = [Float](repeating: 0, count: vocabularyWidth)
            if position < json.count { row[json[position]] = 4 }
            if position < prose.count { row[prose[position]] = 5 }
            if position >= prose.count { row[eosID] = 5 }
            rows.append(contentsOf: row)
        }
        return LMOutput(logits: MLXArray(rows).reshaped([1, tokens.count, vocabularyWidth]))
    }
}

private struct NamedChoiceCodec: ScalarServingTextCodec {
    func render(
        messages: [OpenAIChatMessage], tools: [OpenAIToolSpec], enableThinking: Bool?,
        reasoningEffort: String?, addGenerationPrompt: Bool?
    ) throws -> [Int] {
        [10]
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer { NamedChoiceDetokenizer() }
}

private struct NamedChoiceDetokenizer: ScalarServingDetokenizer {
    private var pending: String?
    mutating func append(token: Int) {
        switch token {
        case thinkEndID: pending = "</think>"
        case 0..<128: pending = String(UnicodeScalar(UInt8(token)))
        default: pending = nil
        }
    }
    mutating func next() -> String? {
        defer { pending = nil }
        return pending
    }
}

/// Everything a test needs to drive one scripted backend.
private struct Rig {
    let backend: ScalarServingBackend
    let model: PositionScriptedModel
}

private func makeRig(
    prose: String = "Sure it!",
    json: String = #"{"city":"Paris"}"#,
    proseIDs: [Int]? = nil,
    jsonIDs: [Int]? = nil,
    schemaSupportEnabled: Bool = true,
    thinkEndTokenID: Int? = nil,
    thinksByDefault: Bool = false,
    isNonSpeculativeScalarRoute: Bool = true
) -> Rig {
    let model = PositionScriptedModel(
        prose: proseIDs ?? ids(prose), json: jsonIDs ?? ids(json))
    var configuration = ScalarServingBackendConfiguration(
        defaultMaximumCompletionTokens: 64,
        maximumQueuedRequests: 2,
        queueRetryAfterSeconds: 2,
        mailboxCapacity: .init(maxDeltas: 256, maxBytes: 65_536),
        thinksByDefault: thinksByDefault,
        jsonSchemaConstraintSupport: schemaSupportEnabled
            ? schemaSupport(thinkEndTokenID: thinkEndTokenID) : nil,
        isNonSpeculativeScalarRoute: isNonSpeculativeScalarRoute,
        decoderSupportsResponseFormatConstraint: true)
    configuration.logitBiasVocabularySize = vocabularyWidth
    let backend = ScalarServingBackend(
        launchedModel: "fixture-model",
        inference: InferenceActor(decoder: MLXDecoder(model: model, cache: [KVCacheSimple()])),
        codec: NamedChoiceCodec(),
        stopTokenIDs: [eosID],
        modelStopStrings: [],
        configuration: configuration)
    return Rig(backend: backend, model: model)
}

// MARK: - HTTP driving

private struct HTTPResult {
    let head: HTTPResponseHead
    let body: String

    var json: [String: Any] {
        (try? JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any]) ?? [:]
    }

    var message: [String: Any] {
        ((json["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any]) ?? [:]
    }

    var finishReason: String? {
        (json["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String
    }

    var toolCalls: [[String: Any]] { message["tool_calls"] as? [[String: Any]] ?? [] }

    var errorParam: String? { (json["error"] as? [String: Any])?["param"] as? String }

    /// SSE `data:` JSON objects, in order.
    var events: [[String: Any]] {
        body.components(separatedBy: "\n\n").compactMap { raw in
            guard raw.hasPrefix("data: "), raw != "data: [DONE]" else { return nil }
            return try? JSONSerialization.jsonObject(with: Data(raw.dropFirst(6).utf8)) as? [String: Any]
        }
    }

    /// Per-event `choices[0].delta` objects.
    var deltas: [[String: Any]] {
        events.compactMap { ($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any] }
    }

    var streamFinishReason: String? {
        events.compactMap { ($0["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String }.last
    }
}

private func send(
    _ backend: ScalarServingBackend, payload: String
) async throws -> HTTPResult {
    let configuration = ServingHTTPConfiguration(
        launchedModel: "fixture-model",
        requestLimits: .productionDefault,
        requiredBearerToken: nil,
        maximumNonStreamingResponseBytes: 1_048_576,
        backpressureStallTimeout: .seconds(5))
    let channel = try await NIOAsyncTestingChannel { channel in
        try channel.pipeline.syncOperations.addHandler(
            OpenAIChatCompletionsHTTPHandler(configuration: configuration, backend: backend))
    }
    let head = HTTPRequestHead(
        version: .http1_1, method: .POST, uri: "/v1/chat/completions",
        headers: [
            "host": "localhost", "content-type": "application/json",
            "content-length": "\(payload.utf8.count)",
        ])
    _ = try await channel.writeInbound(HTTPServerRequestPart.head(head))
    _ = try await channel.writeInbound(HTTPServerRequestPart.body(ByteBuffer(string: payload)))
    _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
    var responseHead: HTTPResponseHead?
    var body = ""
    collect: while true {
        let part: HTTPServerResponsePart = try await channel.waitForOutboundWrite()
        switch part {
        case .head(let value): responseHead = value
        case .body(.byteBuffer(var buffer)): body += buffer.readString(length: buffer.readableBytes) ?? ""
        case .body(.fileRegion): XCTFail("no file regions")
        case .end: break collect
        }
    }
    _ = try await channel.finish()
    return HTTPResult(head: try XCTUnwrap(responseHead), body: body)
}

private let weatherTool =
    #"{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}"#
private let timeTool =
    #"{"type":"function","function":{"name":"lookup_time","description":"t","parameters":{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"],"additionalProperties":false}}}"#

private func named(_ name: String) -> String {
    #"{"type":"function","function":{"name":"\#(name)"}}"#
}

private func payload(
    tools: [String] = [weatherTool],
    toolChoice: String? = named("get_weather"),
    stream: Bool = false,
    maxTokens: Int = 64,
    extra: String = ""
) -> String {
    let choice = toolChoice.map { #","tool_choice":\#($0)"# } ?? ""
    return
        #"{"model":"fixture-model","messages":[{"role":"user","content":"weather in Paris?"}]"#
        + #","tools":[\#(tools.joined(separator: ","))]"# + choice
        + #","max_completion_tokens":\#(maxTokens),"temperature":0,"stream":\#(stream)"# + extra + "}"
}

private func isJSONNull(_ value: Any?) -> Bool { value is NSNull }

// MARK: - Tests

final class ScalarNamedToolChoiceEnforcementTests: XCTestCase {

    // MARK: AC1 -- prose-by-default model, named choice => one constrained tool call

    func testNamedToolChoiceNonStreamingReturnsOneConstrainedToolCall() async throws {
        let rig = makeRig()
        let result = try await send(rig.backend, payload: payload())
        XCTAssertEqual(result.head.status, .ok, result.body)

        XCTAssertEqual(result.toolCalls.count, 1, result.body)
        let call = try XCTUnwrap(result.toolCalls.first)
        XCTAssertEqual(call["type"] as? String, "function")
        XCTAssertFalse((call["id"] as? String ?? "").isEmpty)
        let function = try XCTUnwrap(call["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, "get_weather")
        // Byte for byte the generated text: the model preferred prose, only the mask makes this JSON.
        XCTAssertEqual(function["arguments"] as? String, #"{"city":"Paris"}"#)
        let parsed = try JSONSerialization.jsonObject(
            with: Data((function["arguments"] as? String ?? "").utf8)) as? [String: Any]
        XCTAssertEqual(parsed?["city"] as? String, "Paris")

        XCTAssertTrue(isJSONNull(result.message["content"]), "content must be null: \(result.body)")
        XCTAssertEqual(result.finishReason, "stop")
    }

    func testNamedToolChoiceStreamingEmitsToolCallDeltasAndNoContent() async throws {
        let rig = makeRig()
        let result = try await send(rig.backend, payload: payload(stream: true))
        XCTAssertEqual(result.head.status, .ok, result.body)

        let deltas = result.deltas
        for delta in deltas {
            XCTAssertNil(delta["content"] as? String, "no content delta may be emitted: \(result.body)")
        }
        let callDeltas = deltas.compactMap { ($0["tool_calls"] as? [[String: Any]])?.first }
        let head = try XCTUnwrap(callDeltas.first, result.body)
        XCTAssertEqual(head["index"] as? Int, 0)
        XCTAssertEqual(head["type"] as? String, "function")
        XCTAssertFalse((head["id"] as? String ?? "").isEmpty)
        XCTAssertEqual((head["function"] as? [String: Any])?["name"] as? String, "get_weather")
        // Exactly one call: every later delta continues index 0 and carries no new id.
        for later in callDeltas.dropFirst() {
            XCTAssertEqual(later["index"] as? Int, 0)
            XCTAssertNil(later["id"])
            XCTAssertNil((later["function"] as? [String: Any])?["name"])
        }
        let arguments = callDeltas.compactMap { ($0["function"] as? [String: Any])?["arguments"] as? String }
            .joined()
        XCTAssertEqual(arguments, #"{"city":"Paris"}"#)
        XCTAssertEqual(result.streamFinishReason, "stop")
    }

    // MARK: AC6 -- the named tool (not the first declared one) is used

    func testSecondDeclaredToolIsTheOneNamedAndConstrained() async throws {
        let rig = makeRig(json: #"{"zone":"UTC"}"#)
        // Name the FIRST tool: arguments follow its schema (zone).
        let first = try await send(
            rig.backend,
            payload: payload(tools: [timeTool, weatherTool], toolChoice: named("lookup_time")))
        XCTAssertEqual(
            (first.toolCalls.first?["function"] as? [String: Any])?["name"] as? String, "lookup_time")

        // Name the SECOND tool: the name is the second tool's, and the mask is its schema (city), so
        // the script's `zone` JSON cannot be produced -- the grammar admits only `{"city":...}`.
        let cityRig = makeRig(json: #"{"city":"Paris"}"#)
        let second = try await send(
            cityRig.backend,
            payload: payload(tools: [timeTool, weatherTool], toolChoice: named("get_weather")))
        XCTAssertEqual(second.toolCalls.count, 1, second.body)
        let function = try XCTUnwrap(second.toolCalls.first?["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, "get_weather")
        XCTAssertEqual(function["arguments"] as? String, #"{"city":"Paris"}"#)
        XCTAssertEqual(second.finishReason, "stop")
    }

    // MARK: AC2 -- thinking

    private func thinkingRig(thinkEndTokenID: Int?) -> Rig {
        // Reasoning "ok</think>" is unconstrained; after it the prose preference ("Sure it!") is
        // masked away and the JSON preference wins. json[0...2] are inert fillers (the mask is off).
        makeRig(
            proseIDs: ids("ok") + [thinkEndID] + ids("Sure it!"),
            jsonIDs: [0, 0, 0] + ids(#"{"city":"Paris"}"#),
            thinkEndTokenID: thinkEndTokenID,
            thinksByDefault: true)
    }

    func testThinkingRequestKeepsReasoningUnmaskedAndConstrainsArgumentsAfterThinkEnd() async throws {
        let rig = thinkingRig(thinkEndTokenID: thinkEndID)
        let result = try await send(rig.backend, payload: payload())
        XCTAssertEqual(result.head.status, .ok, result.body)
        XCTAssertEqual(
            result.message["reasoning_content"] as? String, "ok",
            "reasoning is separated and was NOT masked: \(result.body)")
        XCTAssertTrue(isJSONNull(result.message["content"]), result.body)
        let function = try XCTUnwrap(result.toolCalls.first?["function"] as? [String: Any], result.body)
        XCTAssertEqual(function["name"] as? String, "get_weather")
        XCTAssertEqual(function["arguments"] as? String, #"{"city":"Paris"}"#)
        XCTAssertEqual(result.finishReason, "stop")
    }

    func testThinkingRequestStreamsReasoningThenToolCallWithoutContent() async throws {
        let rig = thinkingRig(thinkEndTokenID: thinkEndID)
        let result = try await send(rig.backend, payload: payload(stream: true))
        XCTAssertEqual(result.head.status, .ok, result.body)
        let reasoning = result.deltas.compactMap { $0["reasoning_content"] as? String }.joined()
        XCTAssertEqual(reasoning, "ok", result.body)
        XCTAssertTrue(result.deltas.allSatisfy { ($0["content"] as? String) == nil }, result.body)
        let arguments = result.deltas
            .compactMap { ($0["tool_calls"] as? [[String: Any]])?.first }
            .compactMap { ($0["function"] as? [String: Any])?["arguments"] as? String }
            .joined()
        XCTAssertEqual(arguments, #"{"city":"Paris"}"#)
        XCTAssertEqual(result.streamFinishReason, "stop")
    }

    func testThinkingRequestWithoutResolvableThinkEndIs400BeforeAnyGeneration() async throws {
        let rig = thinkingRig(thinkEndTokenID: nil)
        for stream in [false, true] {
            let result = try await send(rig.backend, payload: payload(stream: stream))
            XCTAssertEqual(result.head.status, .badRequest, result.body)
            XCTAssertEqual(result.errorParam, "tool_choice", result.body)
        }
        XCTAssertEqual(rig.model.forwardCalls, 0, "a refused request must not start generation")
    }

    // MARK: AC3 -- unsupported schema

    func testUnsupportedParametersSchemaIs400WithZeroBackendStarts() async throws {
        let rig = makeRig()
        let patterned =
            #"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string","pattern":"^[A-Z]"}}}}}"#
        for stream in [false, true] {
            let result = try await send(
                rig.backend, payload: payload(tools: [patterned], stream: stream))
            XCTAssertEqual(result.head.status, .badRequest, result.body)
            XCTAssertEqual(result.errorParam, "tool_choice", result.body)
        }
        XCTAssertEqual(rig.model.forwardCalls, 0)
    }

    func testUnsupportedSchemaIsNeverServedUnconstrainedWhileAutoStillServes() async throws {
        let rig = makeRig()
        let patterned =
            #"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string","pattern":"^[A-Z]"}}}}}"#
        // Control: the same declared tool under `auto` is not constrained, so it is simply served.
        let auto = try await send(
            rig.backend, payload: payload(tools: [patterned], toolChoice: #""auto""#))
        XCTAssertEqual(auto.head.status, .ok, auto.body)
    }

    // MARK: AC4 -- response_format conflict

    func testResponseFormatTogetherWithNamedToolChoiceIs400OnToolChoice() async throws {
        let rig = makeRig()
        for format in [
            #"{"type":"json_object"}"#,
            #"{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}"#,
        ] {
            let result = try await send(
                rig.backend, payload: payload(extra: #","response_format":\#(format)"#))
            XCTAssertEqual(result.head.status, .badRequest, result.body)
            XCTAssertEqual(result.errorParam, "tool_choice", result.body)
        }
        XCTAssertEqual(rig.model.forwardCalls, 0)
    }

    /// The backend refuses the combination itself too (a request built without the HTTP decode), so
    /// the conflict cannot be bypassed by a caller that skips `decodeStrict`.
    func testBackendRefusesResponseFormatTogetherWithNamedToolChoice() async throws {
        let rig = makeRig()
        let raw = try JSONSchemaSubsetCompiler.parseOrdered(Data(#"{"type":"object"}"#.utf8))
        let format = try JSONSchemaSubsetCompiler.compile(name: "x", schema: raw, strict: false)
        let request = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(payload().utf8))
        var conflicting = request
        conflicting.responseFormat = .jsonSchema(format)
        do {
            _ = try await rig.backend.start(conflicting)
            XCTFail("expected a refusal")
        } catch OpenAIServingError.invalidRequest(_, let param) {
            XCTAssertEqual(param, "tool_choice")
        }
        XCTAssertEqual(rig.model.forwardCalls, 0)
    }

    // MARK: AC5 -- budget exhausted mid-arguments

    func testSmallTokenBudgetYieldsLengthWithPartialArguments() async throws {
        let rig = makeRig()
        let result = try await send(rig.backend, payload: payload(maxTokens: 5))
        XCTAssertEqual(result.head.status, .ok, result.body)
        XCTAssertEqual(result.finishReason, "length", result.body)
        let function = try XCTUnwrap(result.toolCalls.first?["function"] as? [String: Any], result.body)
        XCTAssertEqual(function["name"] as? String, "get_weather")
        XCTAssertEqual(function["arguments"] as? String, #"{"cit"#)
        XCTAssertTrue(isJSONNull(result.message["content"]))

        let streamed = try await send(rig.backend, payload: payload(stream: true, maxTokens: 5))
        XCTAssertEqual(streamed.streamFinishReason, "length", streamed.body)
        let arguments = streamed.deltas
            .compactMap { ($0["tool_calls"] as? [[String: Any]])?.first }
            .compactMap { ($0["function"] as? [String: Any])?["arguments"] as? String }
            .joined()
        XCTAssertEqual(arguments, #"{"cit"#)
    }

    // MARK: A tool without parameters is the empty object schema

    func testToolWithoutParametersCompilesAsEmptyObject() async throws {
        let rig = makeRig(json: "{}")
        let bare = #"{"type":"function","function":{"name":"ping"}}"#
        let result = try await send(
            rig.backend, payload: payload(tools: [bare], toolChoice: named("ping")))
        XCTAssertEqual(result.head.status, .ok, result.body)
        let function = try XCTUnwrap(result.toolCalls.first?["function"] as? [String: Any], result.body)
        XCTAssertEqual(function["name"] as? String, "ping")
        XCTAssertEqual(function["arguments"] as? String, "{}")
    }

    // MARK: Controls -- everything else is unchanged

    /// (`"required"` is no longer in this list: cycle 259 enforces it, see
    /// `ScalarRequiredToolChoiceEnforcementTests`.)
    func testAutoAndNoneAndAbsentStayUnconstrainedProse() async throws {
        for choice: String? in [#""auto""#, #""none""#, nil] {
            let rig = makeRig()
            let result = try await send(rig.backend, payload: payload(toolChoice: choice))
            XCTAssertEqual(result.head.status, .ok, result.body)
            XCTAssertEqual(result.message["content"] as? String, "Sure it!", "\(choice ?? "absent")")
            XCTAssertTrue(result.toolCalls.isEmpty, "\(choice ?? "absent"): \(result.body)")
            XCTAssertEqual(result.finishReason, "stop")
        }
    }

    func testNamedChoiceWithoutSchemaSupportKeepsValidatedOnlyBehavior() async throws {
        let rig = makeRig(schemaSupportEnabled: false)
        let result = try await send(rig.backend, payload: payload())
        XCTAssertEqual(result.head.status, .ok, result.body)
        XCTAssertEqual(result.message["content"] as? String, "Sure it!")
        XCTAssertTrue(result.toolCalls.isEmpty)
    }

    func testNamedChoiceOnSpeculativeRouteKeepsValidatedOnlyBehavior() async throws {
        let rig = makeRig(isNonSpeculativeScalarRoute: false)
        let result = try await send(rig.backend, payload: payload())
        XCTAssertEqual(result.head.status, .ok, result.body)
        XCTAssertEqual(result.message["content"] as? String, "Sure it!")
        XCTAssertTrue(result.toolCalls.isEmpty)
    }

    // MARK: Order -- penalties -> logit_bias -> mask

    /// A +100 bias on tokens the schema mask forbids everywhere (the control byte 0x01, and EOS before
    /// the object is complete) must not change the constrained output: bias runs first, the mask last.
    func testLogitBiasCannotResurrectMaskedTokensInNamedMode() async throws {
        let rig = makeRig()
        let biased = try await send(
            rig.backend,
            payload: payload(extra: #","logit_bias":{"1":100,"256":100}"#))
        XCTAssertEqual(biased.head.status, .ok, biased.body)
        let function = try XCTUnwrap(biased.toolCalls.first?["function"] as? [String: Any], biased.body)
        XCTAssertEqual(function["arguments"] as? String, #"{"city":"Paris"}"#)
        XCTAssertEqual(biased.finishReason, "stop")
    }

    /// The constraint slot of a named request holds the bias composite WRAPPING the schema mask: one
    /// processor application both adds the bias to an allowed id and leaves a forbidden id at -inf.
    func testNamedModeInstallsBiasThenSchemaMaskInTheConstraintSlot() async throws {
        let recorder = CapturingRecorder()
        let backend = ScalarServingBackend(
            launchedModel: "fixture-model",
            inference: InferenceActor(decoder: CapturingDecoder(recorder: recorder)),
            codec: NamedChoiceCodec(),
            stopTokenIDs: [eosID],
            modelStopStrings: [],
            configuration: {
                var configuration = ScalarServingBackendConfiguration(
                    defaultMaximumCompletionTokens: 8,
                    maximumQueuedRequests: 2,
                    queueRetryAfterSeconds: 2,
                    mailboxCapacity: .init(maxDeltas: 16, maxBytes: 4_096),
                    jsonSchemaConstraintSupport: schemaSupport(thinkEndTokenID: nil),
                    isNonSpeculativeScalarRoute: true,
                    decoderSupportsResponseFormatConstraint: true)
                configuration.logitBiasVocabularySize = vocabularyWidth
                return configuration
            }())
        var request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(payload().utf8))
        request.logitBias = [Int(UInt8(ascii: "{")): 3, Int(UInt8(ascii: "x")): 100]
        let handle = try await backend.start(request)
        while try await handle.mailbox.next() != nil {}

        let constraint = try XCTUnwrap(recorder.constraint)
        XCTAssertTrue(constraint is LogitBiasLogitProcessor, "bias must be outermost: \(type(of: constraint))")
        let zeros = MLXArray([Float](repeating: 0, count: vocabularyWidth)).reshaped([1, vocabularyWidth])
        let out = constraint.process(logits: zeros).asArray(Float.self)
        XCTAssertEqual(out[Int(UInt8(ascii: "{"))], 3, "the only legal first byte keeps its bias")
        XCTAssertEqual(out[Int(UInt8(ascii: "x"))], -Float.infinity, "mask wins over +100 bias")
    }

    /// Without a bias the slot holds the schema mask itself (nothing wraps it).
    func testNamedModeWithoutBiasInstallsTheSchemaMaskAlone() async throws {
        let recorder = CapturingRecorder()
        let backend = ScalarServingBackend(
            launchedModel: "fixture-model",
            inference: InferenceActor(decoder: CapturingDecoder(recorder: recorder)),
            codec: NamedChoiceCodec(),
            stopTokenIDs: [eosID],
            modelStopStrings: [],
            configuration: .init(
                defaultMaximumCompletionTokens: 8,
                maximumQueuedRequests: 2,
                queueRetryAfterSeconds: 2,
                mailboxCapacity: .init(maxDeltas: 16, maxBytes: 4_096),
                jsonSchemaConstraintSupport: schemaSupport(thinkEndTokenID: nil),
                isNonSpeculativeScalarRoute: true,
                decoderSupportsResponseFormatConstraint: true))
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(payload().utf8))
        let handle = try await backend.start(request)
        while try await handle.mailbox.next() != nil {}
        let installed = try XCTUnwrap(recorder.constraint)
        XCTAssertTrue(installed is JSONSchemaMaskingLogitProcessor, "\(type(of: installed))")
        let zeros = MLXArray([Float](repeating: 0, count: vocabularyWidth)).reshaped([1, vocabularyWidth])
        let out = installed.process(logits: zeros).asArray(Float.self)
        XCTAssertEqual(out[Int(UInt8(ascii: "{"))], 0)
        XCTAssertEqual(out[Int(UInt8(ascii: "x"))], -Float.infinity)
    }

    /// Thinking request: the installed mask is a no-op until `</think>` is sampled.
    func testNamedModeOnAThinkingRequestInstallsAnInactiveMask() async throws {
        let recorder = CapturingRecorder()
        let backend = ScalarServingBackend(
            launchedModel: "fixture-model",
            inference: InferenceActor(decoder: CapturingDecoder(recorder: recorder)),
            codec: NamedChoiceCodec(),
            stopTokenIDs: [eosID],
            modelStopStrings: [],
            configuration: .init(
                defaultMaximumCompletionTokens: 8,
                maximumQueuedRequests: 2,
                queueRetryAfterSeconds: 2,
                mailboxCapacity: .init(maxDeltas: 16, maxBytes: 4_096),
                thinksByDefault: true,
                jsonSchemaConstraintSupport: schemaSupport(thinkEndTokenID: thinkEndID),
                isNonSpeculativeScalarRoute: true,
                decoderSupportsResponseFormatConstraint: true))
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(payload().utf8))
        let handle = try await backend.start(request)
        while try await handle.mailbox.next() != nil {}
        let installed = try XCTUnwrap(recorder.constraint)
        let zeros = MLXArray([Float](repeating: 0, count: vocabularyWidth)).reshaped([1, vocabularyWidth])
        XCTAssertEqual(
            installed.process(logits: zeros).asArray(Float.self),
            [Float](repeating: 0, count: vocabularyWidth),
            "mask must be inactive before </think>")
    }
}

// MARK: - Constraint-capturing decoder

private final class CapturingRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var captured: (any LogitProcessor)?
    var constraint: (any LogitProcessor)? {
        lock.lock()
        defer { lock.unlock() }
        return captured
    }
    func record(_ value: (any LogitProcessor)?) {
        guard let value else { return }
        lock.lock()
        defer { lock.unlock() }
        captured = value
    }
}

private struct CapturingDecoder: Decoder {
    let recorder: CapturingRecorder
    var supportsResponseFormatConstraint: Bool { true }
    mutating func prefill(_ promptTokens: [Int]) throws -> Int { eosID }
    mutating func step(last: Int) throws -> Int { eosID }
    mutating func reset() {}
    mutating func setResponseFormatConstraint(_ constraint: (any LogitProcessor)?) {
        recorder.record(constraint)
    }
}
