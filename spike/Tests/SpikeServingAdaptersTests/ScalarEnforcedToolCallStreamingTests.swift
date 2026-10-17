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

// Incremental argument streaming for ENFORCED tool calls on the plain scalar route (cycle 260).
//
// A named `tool_choice` or `tool_choice: "required"` used to answer a streaming request with ONE
// tool-call delta (id, type, name, full arguments) after generation finished. Now the head chunk
// (id, type, name, empty `arguments`) is sent as soon as the call is known and the arguments follow
// as fragments while the constrained decode advances. These tests run the REAL `ScalarServingBackend`
// -> `InferenceActor` -> `MLXDecoder` with a scripted fake `LanguageModel` through the REAL HTTP
// handler on an embedded channel.
//
// Invariants pinned here: the concatenated fragments equal the non-streaming `arguments` byte for
// byte (finish_reason `stop` and `length`); the id is only on the head; every delta is at index 0;
// "required" never leaks its `{"name": ..., "arguments": ` frame nor the wrapper's closing `}`; no
// head before reasoning ends; no head/call for a "required" document cut before its name completes;
// and the first fragments are on the wire while the model is still decoding (gated model).

// MARK: - Fixtures

private let eosID = 256
private let thinkEndID = 257

/// Multi-byte tokens, ids 258...: they let a single token straddle a framing boundary.
private let extraTokens: [String] = [
    #"{"name": ""#,  // 258
    #"", "arguments": {""#,  // 259: closes the name, finishes the frame, opens the arguments object
    #"city":"Paris"#,  // 260
    #""}}"#,  // 261: closes the string, the arguments object AND the wrapper
    #"get_weather"#,  // 262
    #"{"ci"#,  // 263
    #"ty":"Pa"#,  // 264
    #"ris"}"#,  // 265
]
private func extraID(_ text: String) -> Int { 258 + extraTokens.firstIndex(of: text)! }
private let vocabularyWidth = 258 + extraTokens.count

/// The model's unconstrained preference: raw 0x01 bytes, which no JSON document position (not even
/// inside a string) admits, so every step falls through to the scripted JSON token under the mask.
private let unmaskableProse = [Int](repeating: 1, count: 80)

private func ids(_ text: String) -> [Int] { text.utf8.map { Int($0) } }

private func classifications() -> [TokenByteClassification] {
    var all: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }
    all.append(.eos)  // 256
    all.append(.banned)  // 257, </think>
    for token in extraTokens { all.append(.bytes(Array(token.utf8))) }
    return all
}

private func tokenText(_ token: Int) -> String? {
    switch token {
    case thinkEndID: return "</think>"
    case 0..<128: return String(UnicodeScalar(UInt8(token)))
    case 258..<(258 + extraTokens.count): return extraTokens[token - 258]
    default: return nil
    }
}

private func schemaSupport(thinkEndTokenID: Int?) -> ScalarServingJSONSchemaConstraintSupport {
    let object = ScalarServingJSONObjectConstraintSupport(
        classifications: classifications(), thinkEndTokenID: thinkEndTokenID)
    return loadScalarServingJSONSchemaConstraintSupport(sharing: object)!
}

/// Parks the model's forward pass at one position until the test releases it (or a safety timeout
/// elapses, so a non-incremental implementation FAILS an assertion instead of hanging).
private final class Gate: @unchecked Sendable {
    let position: Int
    private let reached = DispatchSemaphore(value: 0)
    private let release = DispatchSemaphore(value: 0)
    private let state = OSAllocatedUnfairLock(initialState: false)

    init(position: Int) { self.position = position }

    var timedOut: Bool { state.withLock { $0 } }

    func arriveAndWait() {
        reached.signal()
        if release.wait(timeout: .now() + 8) == .timedOut { state.withLock { $0 = true } }
    }

    func open() { release.signal() }
}

/// Position-scripted fake model (same shape as the enforcement suites'): at position `p` the logits
/// prefer `prose[p]` (5), then `json[p]` (4); EOS once past the prose.
private final class PositionScriptedModel: Module, LanguageModel, KVCacheDimensionProvider,
    @unchecked Sendable
{
    let kvHeads: [Int] = [1]
    private let prose: [Int]
    private let json: [Int]
    private let gate: Gate?
    private let counters = OSAllocatedUnfairLock(initialState: (position: 0, calls: 0))

    init(prose: [Int], json: [Int], gate: Gate?) {
        self.prose = prose
        self.json = json
        self.gate = gate
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func evaluateThrowing(
        _ input: LMInput.Text, cache: [KVCache]?, state _: LMOutput.State?
    ) throws -> LMOutput {
        let tokens = input.tokens.asType(.int32).asArray(Int32.self)
        var rows: [Float] = []
        for token in tokens {
            let position = counters.withLock { state -> Int in
                state.position = (token == 10) ? 0 : state.position + 1
                return state.position
            }
            if let gate, position == gate.position { gate.arriveAndWait() }
            var row = [Float](repeating: 0, count: vocabularyWidth)
            if position < json.count { row[json[position]] = 4 }
            if position < prose.count { row[prose[position]] = 5 }
            if position >= prose.count { row[eosID] = 5 }
            rows.append(contentsOf: row)
        }
        return LMOutput(logits: MLXArray(rows).reshaped([1, tokens.count, vocabularyWidth]))
    }
}

private struct StreamCodec: ScalarServingTextCodec {
    func render(
        messages: [OpenAIChatMessage], tools: [OpenAIToolSpec], enableThinking: Bool?,
        reasoningEffort: String?, addGenerationPrompt: Bool?
    ) throws -> [Int] { [10] }

    func makeDetokenizer() -> any ScalarServingDetokenizer { StreamDetokenizer() }
}

private struct StreamDetokenizer: ScalarServingDetokenizer {
    private var pending: String?
    mutating func append(token: Int) { pending = tokenText(token) }
    mutating func next() -> String? {
        defer { pending = nil }
        return pending
    }
}

private func makeBackend(
    jsonIDs: [Int],
    proseIDs: [Int] = unmaskableProse,
    thinkEndTokenID: Int? = nil,
    thinksByDefault: Bool = false,
    gate: Gate? = nil
) -> ScalarServingBackend {
    let model = PositionScriptedModel(prose: proseIDs, json: jsonIDs, gate: gate)
    var configuration = ScalarServingBackendConfiguration(
        defaultMaximumCompletionTokens: 96,
        maximumQueuedRequests: 2,
        queueRetryAfterSeconds: 2,
        mailboxCapacity: .init(maxDeltas: 256, maxBytes: 65_536),
        thinksByDefault: thinksByDefault,
        jsonSchemaConstraintSupport: schemaSupport(thinkEndTokenID: thinkEndTokenID),
        isNonSpeculativeScalarRoute: true,
        decoderSupportsResponseFormatConstraint: true)
    configuration.logitBiasVocabularySize = vocabularyWidth
    return ScalarServingBackend(
        launchedModel: "fixture-model",
        inference: InferenceActor(decoder: MLXDecoder(model: model, cache: [KVCacheSimple()])),
        codec: StreamCodec(),
        stopTokenIDs: [eosID],
        modelStopStrings: [],
        configuration: configuration)
}

// MARK: - HTTP driving

private let weatherTool =
    #"{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}"#
private let timeTool =
    #"{"type":"function","function":{"name":"lookup_time","description":"t","parameters":{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"],"additionalProperties":false}}}"#
private let requiredChoice = #""required""#
private let namedChoice = #"{"type":"function","function":{"name":"get_weather"}}"#

private func payload(
    toolChoice: String, stream: Bool, maxTokens: Int = 96, extra: String = ""
) -> String {
    #"{"model":"fixture-model","messages":[{"role":"user","content":"weather in Paris?"}]"#
        + #","tools":[\#(weatherTool),\#(timeTool)],"tool_choice":\#(toolChoice)"#
        + #","max_completion_tokens":\#(maxTokens),"temperature":0,"stream":\#(stream)"# + extra + "}"
}

private func parseEvents(_ body: String) -> [[String: Any]] {
    body.components(separatedBy: "\n\n").compactMap { raw in
        guard raw.hasPrefix("data: "), raw != "data: [DONE]" else { return nil }
        return try? JSONSerialization.jsonObject(with: Data(raw.dropFirst(6).utf8)) as? [String: Any]
    }
}

/// An embedded-channel HTTP exchange that can be read incrementally.
private final class Exchange {
    let channel: NIOAsyncTestingChannel
    private(set) var body = ""
    private(set) var status: HTTPResponseStatus?
    private(set) var ended = false

    init(backend: ScalarServingBackend, payload: String) async throws {
        let configuration = ServingHTTPConfiguration(
            launchedModel: "fixture-model",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(10))
        channel = try await NIOAsyncTestingChannel { channel in
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
    }

    /// Reads one outbound write; returns false once the response has ended.
    @discardableResult
    func readOne() async throws -> Bool {
        guard !ended else { return false }
        let part: HTTPServerResponsePart = try await channel.waitForOutboundWrite()
        switch part {
        case .head(let value): status = value.status
        case .body(.byteBuffer(var buffer)): body += buffer.readString(length: buffer.readableBytes) ?? ""
        case .body(.fileRegion): XCTFail("no file regions")
        case .end: ended = true
        }
        return !ended
    }

    func readToEnd() async throws {
        while try await readOne() {}
        _ = try await channel.finish()
    }

    /// Wire-level `choices[0].delta.tool_calls[0]` objects received so far, in order.
    var callDeltas: [[String: Any]] {
        parseEvents(body).compactMap {
            (((($0["choices"] as? [[String: Any]])?.first)?["delta"] as? [String: Any])?["tool_calls"]
                as? [[String: Any]])?.first
        }
    }
}

private struct Streamed {
    let events: [[String: Any]]
    let status: HTTPResponseStatus?

    var deltas: [[String: Any]] {
        events.compactMap { ($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any] }
    }
    var callDeltas: [[String: Any]] {
        deltas.compactMap { ($0["tool_calls"] as? [[String: Any]])?.first }
    }
    var head: [String: Any]? { callDeltas.first }
    /// Every NON-EMPTY `function.arguments` string after the head, in order.
    var fragments: [String] {
        callDeltas.dropFirst().compactMap { ($0["function"] as? [String: Any])?["arguments"] as? String }
    }
    var finishReason: String? {
        events.compactMap { ($0["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String }.last
    }
}

private func stream(_ backend: ScalarServingBackend, _ payload: String) async throws -> Streamed {
    let exchange = try await Exchange(backend: backend, payload: payload)
    try await exchange.readToEnd()
    return Streamed(events: parseEvents(exchange.body), status: exchange.status)
}

private struct NonStreamed {
    let status: HTTPResponseStatus?
    let json: [String: Any]
    var choice: [String: Any]? { (json["choices"] as? [[String: Any]])?.first }
    var calls: [[String: Any]] {
        ((choice?["message"] as? [String: Any])?["tool_calls"] as? [[String: Any]]) ?? []
    }
    var arguments: String? { (calls.first?["function"] as? [String: Any])?["arguments"] as? String }
    var name: String? { (calls.first?["function"] as? [String: Any])?["name"] as? String }
    var finishReason: String? { choice?["finish_reason"] as? String }
}

private func nonStream(_ backend: ScalarServingBackend, _ payload: String) async throws -> NonStreamed {
    let exchange = try await Exchange(backend: backend, payload: payload)
    try await exchange.readToEnd()
    let json =
        (try? JSONSerialization.jsonObject(with: Data(exchange.body.utf8)) as? [String: Any]) ?? [:]
    return NonStreamed(status: exchange.status, json: json)
}

private let weatherDocumentIDs = ids(#"{"name": "get_weather", "arguments": {"city":"Paris"}}"#)
private let argumentIDs = ids(#"{"city":"Paris"}"#)

// MARK: - Tests

final class ScalarEnforcedToolCallStreamingTests: XCTestCase {

    // MARK: Named

    func testNamedStreamsHeadThenManyFragmentsEqualToNonStreamingArguments() async throws {
        let streamed = try await stream(
            makeBackend(jsonIDs: argumentIDs), payload(toolChoice: namedChoice, stream: true))
        let head = try XCTUnwrap(streamed.head)
        XCTAssertEqual(head["index"] as? Int, 0)
        XCTAssertEqual(head["type"] as? String, "function")
        XCTAssertFalse((head["id"] as? String ?? "").isEmpty)
        let headFunction = try XCTUnwrap(head["function"] as? [String: Any])
        XCTAssertEqual(headFunction["name"] as? String, "get_weather")
        XCTAssertEqual(headFunction["arguments"] as? String, "", "head carries empty arguments")
        XCTAssertGreaterThanOrEqual(streamed.fragments.count, 2, "arguments must stream incrementally")
        XCTAssertTrue(streamed.fragments.allSatisfy { !$0.isEmpty })
        for later in streamed.callDeltas.dropFirst() {
            XCTAssertEqual(later["index"] as? Int, 0)
            XCTAssertNil(later["id"])
            XCTAssertNil(later["type"])
            XCTAssertNil((later["function"] as? [String: Any])?["name"])
        }
        XCTAssertEqual(streamed.finishReason, "stop")
        XCTAssertTrue(streamed.deltas.allSatisfy { ($0["content"] as? String) == nil })

        let whole = try await nonStream(
            makeBackend(jsonIDs: argumentIDs), payload(toolChoice: namedChoice, stream: false))
        XCTAssertEqual(whole.arguments, #"{"city":"Paris"}"#)
        XCTAssertEqual(whole.calls.count, 1)
        XCTAssertEqual(streamed.fragments.joined(), whole.arguments)
        XCTAssertEqual(whole.finishReason, "stop")
    }

    func testNamedLengthCutMidArgumentsStreamsHeadAndPartialFragmentsThenLength() async throws {
        let cap = 6
        let streamed = try await stream(
            makeBackend(jsonIDs: argumentIDs), payload(toolChoice: namedChoice, stream: true, maxTokens: cap))
        XCTAssertNotNil(streamed.head)
        XCTAssertGreaterThanOrEqual(streamed.fragments.count, 2)
        XCTAssertEqual(streamed.finishReason, "length")
        let whole = try await nonStream(
            makeBackend(jsonIDs: argumentIDs), payload(toolChoice: namedChoice, stream: false, maxTokens: cap))
        XCTAssertEqual(whole.arguments, #"{"city"#)
        XCTAssertEqual(whole.finishReason, "length")
        XCTAssertEqual(streamed.fragments.joined(), whole.arguments)
    }

    func testNamedMultiByteTokensBecomeOneFragmentEach() async throws {
        let tokens = [extraID(#"{"ci"#), extraID(#"ty":"Pa"#), extraID(#"ris"}"#)]
        let streamed = try await stream(
            makeBackend(jsonIDs: tokens), payload(toolChoice: namedChoice, stream: true))
        XCTAssertEqual(streamed.fragments, [#"{"ci"#, #"ty":"Pa"#, #"ris"}"#])
    }

    // MARK: Named: the first fragments reach the wire while the model is still decoding

    func testNamedFragmentsAreOnTheWireBeforeGenerationFinishes() async throws {
        let gate = Gate(position: 12)
        let exchange = try await Exchange(
            backend: makeBackend(jsonIDs: argumentIDs, gate: gate),
            payload: payload(toolChoice: namedChoice, stream: true))
        // The model is parked at position 12 of 16: only incrementally streamed deltas can satisfy this.
        while try await exchange.readOne() {
            if exchange.callDeltas.count >= 4 { break }
        }
        XCTAssertFalse(gate.timedOut, "the gate must not have timed out while we were waiting")
        let before = exchange.callDeltas
        XCTAssertGreaterThanOrEqual(before.count, 4, "head + >=3 fragments before the model finished")
        XCTAssertFalse(exchange.ended)
        gate.open()
        try await exchange.readToEnd()
        let streamed = Streamed(events: parseEvents(exchange.body), status: exchange.status)
        XCTAssertEqual(streamed.fragments.joined(), #"{"city":"Paris"}"#)
    }

    func testRequiredFragmentsAreOnTheWireBeforeGenerationFinishes() async throws {
        let gate = Gate(position: 50)
        let exchange = try await Exchange(
            backend: makeBackend(jsonIDs: weatherDocumentIDs, gate: gate),
            payload: payload(toolChoice: requiredChoice, stream: true))
        while try await exchange.readOne() {
            if exchange.callDeltas.count >= 4 { break }
        }
        XCTAssertFalse(gate.timedOut, "the gate must not have timed out while we were waiting")
        XCTAssertGreaterThanOrEqual(exchange.callDeltas.count, 4, "head + >=3 fragments before the model finished")
        XCTAssertFalse(exchange.ended)
        gate.open()
        try await exchange.readToEnd()
        let streamed = Streamed(events: parseEvents(exchange.body), status: exchange.status)
        XCTAssertEqual(streamed.fragments.joined(), #"{"city":"Paris"}"#)
    }

    // MARK: Named: thinking

    private func thinkingProse() -> [Int] { ids("ok") + [thinkEndID] + unmaskableProse }
    private func thinkingJSON(_ body: [Int]) -> [Int] { [0, 0, 0] + body }

    func testNamedThinkingEmitsNoHeadBeforeReasoningEnds() async throws {
        let backend = makeBackend(
            jsonIDs: thinkingJSON(argumentIDs), proseIDs: thinkingProse(),
            thinkEndTokenID: thinkEndID, thinksByDefault: true)
        let streamed = try await stream(backend, payload(toolChoice: namedChoice, stream: true))
        let reasoning = streamed.deltas.compactMap { $0["reasoning_content"] as? String }
        XCTAssertEqual(reasoning.joined(), "ok")
        let events = streamed.deltas
        let lastReasoning = try XCTUnwrap(events.lastIndex { $0["reasoning_content"] != nil })
        let firstCall = try XCTUnwrap(events.firstIndex { $0["tool_calls"] != nil })
        XCTAssertLessThan(lastReasoning, firstCall, "no head before reasoning ends")
        XCTAssertGreaterThanOrEqual(streamed.fragments.count, 2)
        XCTAssertEqual(streamed.fragments.joined(), #"{"city":"Paris"}"#)
        XCTAssertTrue(streamed.deltas.allSatisfy { ($0["content"] as? String) == nil })
    }

    func testRequiredThinkingEmitsNoHeadBeforeReasoningEnds() async throws {
        let backend = makeBackend(
            jsonIDs: thinkingJSON(weatherDocumentIDs), proseIDs: thinkingProse(),
            thinkEndTokenID: thinkEndID, thinksByDefault: true)
        let streamed = try await stream(backend, payload(toolChoice: requiredChoice, stream: true))
        XCTAssertEqual(streamed.deltas.compactMap { $0["reasoning_content"] as? String }.joined(), "ok")
        let lastReasoning = try XCTUnwrap(streamed.deltas.lastIndex { $0["reasoning_content"] != nil })
        let firstCall = try XCTUnwrap(streamed.deltas.firstIndex { $0["tool_calls"] != nil })
        XCTAssertLessThan(lastReasoning, firstCall, "no head before reasoning ends")
        XCTAssertEqual(streamed.fragments.joined(), #"{"city":"Paris"}"#)
    }

    // MARK: Required

    func testRequiredStreamsNameInHeadAndOnlyArgumentBytesInFragments() async throws {
        let streamed = try await stream(
            makeBackend(jsonIDs: weatherDocumentIDs), payload(toolChoice: requiredChoice, stream: true))
        let head = try XCTUnwrap(streamed.head)
        XCTAssertEqual(head["index"] as? Int, 0)
        XCTAssertFalse((head["id"] as? String ?? "").isEmpty)
        XCTAssertEqual((head["function"] as? [String: Any])?["name"] as? String, "get_weather")
        XCTAssertEqual((head["function"] as? [String: Any])?["arguments"] as? String, "")
        XCTAssertGreaterThanOrEqual(streamed.fragments.count, 2)
        for later in streamed.callDeltas.dropFirst() {
            XCTAssertEqual(later["index"] as? Int, 0)
            XCTAssertNil(later["id"])
        }
        for fragment in streamed.fragments {
            XCTAssertFalse(fragment.contains("arguments"), "frame must not leak: \(fragment)")
            XCTAssertFalse(fragment.contains("get_weather"), "frame must not leak: \(fragment)")
        }
        let joined = streamed.fragments.joined()
        XCTAssertEqual(joined, #"{"city":"Paris"}"#)
        XCTAssertFalse(joined.hasSuffix("}}"), "the wrapper's closing brace must be held back")
        XCTAssertEqual(streamed.finishReason, "stop")

        let whole = try await nonStream(
            makeBackend(jsonIDs: weatherDocumentIDs), payload(toolChoice: requiredChoice, stream: false))
        XCTAssertEqual(whole.name, "get_weather")
        XCTAssertEqual(whole.arguments, joined)
        XCTAssertEqual(whole.calls.count, 1)
    }

    func testRequiredLengthCutMidArgumentsStreamsPartialFragmentsThenLength() async throws {
        // 37 frame bytes, then 5 argument bytes `{"cit`.
        let cap = 42
        let streamed = try await stream(
            makeBackend(jsonIDs: weatherDocumentIDs),
            payload(toolChoice: requiredChoice, stream: true, maxTokens: cap))
        XCTAssertNotNil(streamed.head)
        XCTAssertGreaterThanOrEqual(streamed.fragments.count, 2)
        XCTAssertEqual(streamed.finishReason, "length")
        let whole = try await nonStream(
            makeBackend(jsonIDs: weatherDocumentIDs),
            payload(toolChoice: requiredChoice, stream: false, maxTokens: cap))
        XCTAssertEqual(whole.arguments, #"{"cit"#)
        XCTAssertEqual(whole.finishReason, "length")
        XCTAssertEqual(streamed.fragments.joined(), whole.arguments)
    }

    func testRequiredLengthCutRightAfterTheFrameStreamsHeadWithNoFragments() async throws {
        let cap = 37
        let streamed = try await stream(
            makeBackend(jsonIDs: weatherDocumentIDs),
            payload(toolChoice: requiredChoice, stream: true, maxTokens: cap))
        XCTAssertNotNil(streamed.head)
        XCTAssertEqual(streamed.fragments, [])
        XCTAssertEqual(streamed.finishReason, "length")
        let whole = try await nonStream(
            makeBackend(jsonIDs: weatherDocumentIDs),
            payload(toolChoice: requiredChoice, stream: false, maxTokens: cap))
        XCTAssertEqual(whole.calls.count, 1)
        XCTAssertEqual(whole.arguments, "")
    }

    // A "required" document truncated before its name completes names no tool: no head, no call.
    func testRequiredLengthCutMidNameEmitsNoHeadAndNoToolCall() async throws {
        for cap in [3, 12, 14, 20] {  // inside the prefix literal, and inside `get_weather`
            let streamed = try await stream(
                makeBackend(jsonIDs: weatherDocumentIDs),
                payload(toolChoice: requiredChoice, stream: true, maxTokens: cap))
            XCTAssertEqual(streamed.status, .ok)
            XCTAssertNil(streamed.head, "cap \(cap): no head before the name completes")
            XCTAssertEqual(streamed.fragments, [])
            XCTAssertTrue(streamed.deltas.allSatisfy { $0["tool_calls"] == nil && ($0["content"] as? String) == nil })
            XCTAssertEqual(streamed.finishReason, "length")
            let whole = try await nonStream(
                makeBackend(jsonIDs: weatherDocumentIDs),
                payload(toolChoice: requiredChoice, stream: false, maxTokens: cap))
            XCTAssertEqual(whole.calls.count, 0)
            XCTAssertEqual(whole.finishReason, "length")
        }
    }

    // Tokens that straddle the framing boundaries: `", "arguments": {"` finishes the frame AND opens
    // the arguments; `"}}` closes the string, the arguments object and the wrapper.
    func testRequiredTokensStraddlingTheFrameAndTheWrapperAreSplitByBytes() async throws {
        let tokens = [
            extraID(#"{"name": ""#), extraID("get_weather"), extraID(#"", "arguments": {""#),
            extraID(#"city":"Paris"#), extraID(#""}}"#),
        ]
        let streamed = try await stream(
            makeBackend(jsonIDs: tokens), payload(toolChoice: requiredChoice, stream: true))
        XCTAssertEqual(streamed.fragments, [#"{""#, #"city":"Paris"#, #""}"#])
        XCTAssertEqual(streamed.fragments.joined(), #"{"city":"Paris"}"#)
        XCTAssertEqual((streamed.head?["function"] as? [String: Any])?["name"] as? String, "get_weather")
        XCTAssertFalse(streamed.fragments.contains { $0.hasSuffix("}}") })
        let whole = try await nonStream(
            makeBackend(jsonIDs: tokens), payload(toolChoice: requiredChoice, stream: false))
        XCTAssertEqual(whole.arguments, #"{"city":"Paris"}"#)
        XCTAssertEqual(whole.name, "get_weather")
    }

    // The head is not sent at the name token: it waits for the frame to complete (so a document cut
    // inside `", "arguments": ` still names no tool, exactly as the non-streaming reply).
    func testRequiredCutBetweenNameAndFrameEndEmitsNoHead() async throws {
        let streamed = try await stream(
            makeBackend(jsonIDs: weatherDocumentIDs),
            payload(toolChoice: requiredChoice, stream: true, maxTokens: 26))
        XCTAssertNil(streamed.head)
        let whole = try await nonStream(
            makeBackend(jsonIDs: weatherDocumentIDs),
            payload(toolChoice: requiredChoice, stream: false, maxTokens: 26))
        XCTAssertEqual(whole.calls.count, 0)
    }
}
