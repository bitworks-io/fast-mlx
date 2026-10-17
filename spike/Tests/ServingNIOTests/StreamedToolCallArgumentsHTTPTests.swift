import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import XCTest

@testable import ServingCore
@testable import ServingNIO

/// Incremental tool-call argument streaming at the HTTP layer (cycle 260): a backend that sends a
/// `.toolCallStart` head followed by `.toolCallArguments` fragments produces ONE head chunk (id, type,
/// name, empty `arguments`) then one chunk per fragment, every chunk at `index` 0 and only the head
/// carrying `id`; the non-streaming reply folds the same deltas into one call whose arguments are the
/// fragments' concatenation. Fragments never count as calls (`parallel_tool_calls: false` caps CALLS).
final class StreamedToolCallArgumentsHTTPTests: XCTestCase {
    private let head = OpenAIToolCall(
        id: "call_head", function: .init(name: "get_weather", arguments: ""))
    private let fragments = [#"{"ci"#, #"ty": "Par"#, #"is"}"#]

    private func body(stream: Bool, parallel: String? = nil, logprobs: Bool = false) -> String {
        let extra =
            (parallel.map { #","parallel_tool_calls":\#($0)"# } ?? "")
            + (logprobs ? #","logprobs":true,"top_logprobs":0"# : "")
        return
            #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"tools":[{"type":"function","function":{"name":"get_weather"}}],"max_completion_tokens":8,"temperature":0,"stream":\#(stream)"#
            + extra + "}"
    }

    private func run(
        deltas: [ServingResponseDelta], stream: Bool, parallel: String? = nil,
        logprobs: Bool = false, finishReason: OpenAIChatFinishReason = .stop
    ) async throws -> (head: HTTPResponseHead, body: String) {
        let backend = DeltasBackend(deltas: deltas, finishReason: finishReason)
        let configuration = ServingHTTPConfiguration(
            launchedModel: "qwen3-32b",
            requestLimits: .productionDefault,
            requiredBearerToken: nil,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(1))
        let channel = try await NIOAsyncTestingChannel { channel in
            try channel.pipeline.syncOperations.addHandler(
                OpenAIChatCompletionsHTTPHandler(configuration: configuration, backend: backend))
        }
        let payload = body(stream: stream, parallel: parallel, logprobs: logprobs)
        let requestHead = HTTPRequestHead(
            version: .http1_1, method: .POST, uri: "/v1/chat/completions",
            headers: [
                "host": "localhost", "content-type": "application/json",
                "content-length": "\(payload.utf8.count)",
            ])
        _ = try await channel.writeInbound(HTTPServerRequestPart.head(requestHead))
        _ = try await channel.writeInbound(HTTPServerRequestPart.body(ByteBuffer(string: payload)))
        _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
        var responseHead: HTTPResponseHead?
        var text = ""
        collect: while true {
            let part: HTTPServerResponsePart = try await channel.waitForOutboundWrite()
            switch part {
            case .head(let value): responseHead = value
            case .body(.byteBuffer(var buffer)):
                text += buffer.readString(length: buffer.readableBytes) ?? ""
            case .body(.fileRegion): XCTFail("no file regions")
            case .end: break collect
            }
        }
        _ = try await channel.finish()
        return (try XCTUnwrap(responseHead), text)
    }

    private func events(_ body: String) throws -> [[String: Any]] {
        try body.components(separatedBy: "\n\n").compactMap { raw -> [String: Any]? in
            guard raw.hasPrefix("data: "), raw != "data: [DONE]" else { return nil }
            return try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(raw.dropFirst("data: ".count).utf8))
                    as? [String: Any])
        }
    }

    private func toolDeltas(_ body: String) throws -> [[String: Any]] {
        try events(body).compactMap { event -> [String: Any]? in
            let delta = ((event["choices"] as? [[String: Any]])?.first)?["delta"] as? [String: Any]
            return (delta?["tool_calls"] as? [[String: Any]])?.first
        }
    }

    private var fullScript: [ServingResponseDelta] {
        [.toolCallStart(head)] + fragments.map { .toolCallArguments($0) }
    }

    // One head chunk, then exactly one chunk per fragment; id/type/name only on the head; index 0.
    func testStreamingEmitsOneHeadThenOneChunkPerFragment() async throws {
        let response = try await run(deltas: fullScript, stream: true)
        XCTAssertEqual(response.head.status, .ok)
        let calls = try toolDeltas(response.body)
        XCTAssertEqual(calls.count, 1 + fragments.count, response.body)
        XCTAssertEqual(calls.compactMap { $0["index"] as? Int }, Array(repeating: 0, count: calls.count))
        XCTAssertEqual(calls.compactMap { $0["id"] as? String }, ["call_head"])
        XCTAssertEqual(calls.compactMap { $0["type"] as? String }, ["function"])
        let headCall = try XCTUnwrap(calls.first, response.body)
        XCTAssertEqual(headCall["id"] as? String, "call_head", "id only on the head")
        let headFunction = try XCTUnwrap(headCall["function"] as? [String: Any])
        XCTAssertEqual(headFunction["name"] as? String, "get_weather")
        XCTAssertEqual(headFunction["arguments"] as? String, "", "head carries empty arguments")
        for call in calls.dropFirst() {
            let function = try XCTUnwrap(call["function"] as? [String: Any])
            XCTAssertNil(function["name"], "name only on the head")
            XCTAssertNil(call["id"])
        }
        let streamed = calls.dropFirst().compactMap {
            ($0["function"] as? [String: Any])?["arguments"] as? String
        }
        XCTAssertEqual(Array(streamed), fragments)
        XCTAssertEqual(streamed.joined(), fragments.joined())
        XCTAssertTrue(response.body.contains(#""finish_reason":"stop""#), response.body)
    }

    // Non-streaming folds head + fragments into the single call a lone `.toolCalls` would carry.
    func testNonStreamingAssemblesHeadAndFragmentsIntoOneCall() async throws {
        let response = try await run(deltas: fullScript, stream: false)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any],
            response.body)
        let choice = try XCTUnwrap((object["choices"] as? [[String: Any]])?.first)
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        let calls = try XCTUnwrap(message["tool_calls"] as? [[String: Any]], response.body)
        XCTAssertEqual(calls.count, 1)
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(call["id"] as? String, "call_head")
        let function = try XCTUnwrap(call["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, "get_weather")
        XCTAssertEqual(function["arguments"] as? String, fragments.joined())
        XCTAssertEqual(choice["finish_reason"] as? String, "stop")
    }

    // A head with no fragments (budget ran out right at the arguments boundary) is one call with
    // empty arguments in the non-streaming reply, matching a lone `.toolCalls` of empty arguments.
    func testNonStreamingHeadWithoutFragmentsKeepsEmptyArguments() async throws {
        let response = try await run(deltas: [.toolCallStart(head)], stream: false)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        let message = try XCTUnwrap(
            ((object["choices"] as? [[String: Any]])?.first)?["message"] as? [String: Any])
        let calls = try XCTUnwrap(message["tool_calls"] as? [[String: Any]], response.body)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual((calls.first?["function"] as? [String: Any])?["arguments"] as? String, "")
    }

    // `parallel_tool_calls:false` caps CALLS: fragments of the first call all flow (they are not
    // calls), a second call's head and fragments are dropped entirely.
    func testFragmentsDoNotAdvanceTheCallCounter() async throws {
        let second = OpenAIToolCall(id: "call_second", function: .init(name: "get_weather", arguments: ""))
        let script: [ServingResponseDelta] =
            fullScript + [.toolCallStart(second), .toolCallArguments(#"{"x":1}"#)]
        let response = try await run(deltas: script, stream: true, parallel: "false")
        let calls = try toolDeltas(response.body)
        XCTAssertEqual(calls.count, 1 + fragments.count, response.body)
        XCTAssertFalse(response.body.contains("call_second"), response.body)
        XCTAssertFalse(response.body.contains(#"{\"x\":1}"#), response.body)
        let streamed = calls.dropFirst().compactMap {
            ($0["function"] as? [String: Any])?["arguments"] as? String
        }
        XCTAssertEqual(streamed.joined(), fragments.joined())
    }

    // Logprobs keep the carry-once rule: every token appears exactly once across all chunks, in order,
    // whether it lands on the head, a fragment, or the finish chunk.
    func testLogprobsAreCarriedExactlyOnceAcrossHeadAndFragments() async throws {
        func token(_ text: String) -> ServingTokenLogprob {
            ServingTokenLogprob(tokenText: text, logprob: -0.1)
        }
        let script: [ServingResponseDelta] = [
            .tokenLogprobs([token("T0")]), .toolCallStart(head), .toolCallArguments(fragments[0]),
            .tokenLogprobs([token("T1")]), .toolCallArguments(fragments[1]),
            .tokenLogprobs([token("T2"), token("T3")]), .toolCallArguments(fragments[2]),
            .tokenLogprobs([token("T4")]),
        ]
        let response = try await run(deltas: script, stream: true, logprobs: true)
        var seen: [String] = []
        for event in try events(response.body).dropFirst() {
            guard let choice = (event["choices"] as? [[String: Any]])?.first,
                let logprobs = choice["logprobs"] as? [String: Any],
                let content = logprobs["content"] as? [[String: Any]]
            else { continue }
            seen.append(contentsOf: content.compactMap { $0["token"] as? String })
        }
        XCTAssertEqual(seen, ["T0", "T1", "T2", "T3", "T4"], response.body)
        XCTAssertEqual(try toolDeltas(response.body).count, 1 + fragments.count)
    }
}

/// Emits an exact scripted delta sequence, then completes.
private final class DeltasBackend: ServingGenerationBackend, Sendable {
    private let deltas: [ServingResponseDelta]
    private let finishReason: OpenAIChatFinishReason

    init(deltas: [ServingResponseDelta], finishReason: OpenAIChatFinishReason) {
        self.deltas = deltas
        self.finishReason = finishReason
    }

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 64, maxBytes: 4_096))
        let lease = ServingRequestLease(id: ServingRequestID("streamed-tool-args"), onCancel: {})
        let handle = ServingGenerationHandle(
            responseID: "chatcmpl-streamed-args",
            created: 1_775_000_000,
            model: request.model,
            route: .continuousBatchNoSpec,
            mailbox: mailbox,
            lease: lease)
        let deltas = self.deltas
        let finishReason = self.finishReason
        Task {
            for delta in deltas {
                try? await mailbox.send(delta)
            }
            try? await mailbox.send(
                .completion(
                    ServingGenerationCompletion(
                        finishReason: finishReason,
                        usage: OpenAIChatUsage(promptTokens: 3, completionTokens: 3))))
            await mailbox.finish()
        }
        return handle
    }
}
