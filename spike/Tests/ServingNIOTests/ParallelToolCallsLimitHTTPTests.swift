import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import XCTest

@testable import ServingCore
@testable import ServingNIO

/// `parallel_tool_calls: false` at the HTTP layer: at most ONE tool call leaves the server, in both
/// the non-streaming reply and the streamed deltas (cumulatively across separate `.toolCalls`
/// events); absent and `true` forward every call. `finish_reason` stays `tool_calls`.
final class ParallelToolCallsLimitHTTPTests: XCTestCase {
    private let first = OpenAIToolCall(
        id: "call_0", function: .init(name: "get_weather", arguments: #"{"city":"Paris"}"#))
    private let second = OpenAIToolCall(
        id: "call_1", function: .init(name: "get_weather", arguments: #"{"city":"Rome"}"#))

    private func body(stream: Bool, parallel: String?) -> String {
        let field = parallel.map { #","parallel_tool_calls":\#($0)"# } ?? ""
        return
            #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"tools":[{"type":"function","function":{"name":"get_weather"}}],"max_completion_tokens":8,"temperature":0,"stream":\#(stream)"#
            + field + "}"
    }

    private func run(
        events: [[OpenAIToolCall]], stream: Bool, parallel: String?
    ) async throws -> (head: HTTPResponseHead, body: String) {
        let backend = EventsBackend(events: events)
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
        let payload = body(stream: stream, parallel: parallel)
        let requestHead = HTTPRequestHead(
            version: .http1_1, method: .POST, uri: "/v1/chat/completions",
            headers: [
                "host": "localhost", "content-type": "application/json",
                "content-length": "\(payload.utf8.count)",
            ])
        _ = try await channel.writeInbound(HTTPServerRequestPart.head(requestHead))
        _ = try await channel.writeInbound(HTTPServerRequestPart.body(ByteBuffer(string: payload)))
        _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))

        var head: HTTPResponseHead?
        var text = ""
        collect: while true {
            let part: HTTPServerResponsePart = try await channel.waitForOutboundWrite()
            switch part {
            case .head(let value): head = value
            case .body(.byteBuffer(var buffer)):
                text += buffer.readString(length: buffer.readableBytes) ?? ""
            case .body(.fileRegion): XCTFail("no file regions")
            case .end: break collect
            }
        }
        _ = try await channel.finish()
        return (try XCTUnwrap(head), text)
    }

    /// Non-streaming: the `message.tool_calls` array and the choice's finish_reason.
    private func nonStreamingCalls(_ body: String) throws -> (calls: [[String: Any]], finish: String?) {
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any], body)
        let choice = try XCTUnwrap((object["choices"] as? [[String: Any]])?.first, body)
        let message = try XCTUnwrap(choice["message"] as? [String: Any], body)
        return (message["tool_calls"] as? [[String: Any]] ?? [], choice["finish_reason"] as? String)
    }

    /// Streaming: every tool-call delta in wire order, plus the finish_reason seen on any chunk.
    private func streamedDeltas(_ body: String) throws -> (deltas: [[String: Any]], finish: String?) {
        var deltas: [[String: Any]] = []
        var finish: String?
        for rawEvent in body.components(separatedBy: "\n\n")
        where rawEvent.hasPrefix("data: ") && rawEvent != "data: [DONE]" {
            let payload = String(rawEvent.dropFirst("data: ".count))
            let object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any])
            let choice = (object["choices"] as? [[String: Any]])?.first
            if let delta = choice?["delta"] as? [String: Any],
                let toolCalls = delta["tool_calls"] as? [[String: Any]]
            {
                deltas.append(contentsOf: toolCalls)
            }
            if let reason = choice?["finish_reason"] as? String { finish = reason }
        }
        return (deltas, finish)
    }

    private func assertOnlyFirstCall(_ deltas: [[String: Any]], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertFalse(deltas.isEmpty, file: file, line: line)
        XCTAssertEqual(deltas.compactMap { $0["index"] as? Int }, Array(repeating: 0, count: deltas.count), file: file, line: line)
        XCTAssertEqual(deltas.compactMap { $0["id"] as? String }, ["call_0"], file: file, line: line)
        XCTAssertEqual(
            deltas.compactMap { ($0["function"] as? [String: Any])?["name"] as? String },
            ["get_weather"], file: file, line: line)
        XCTAssertEqual(
            deltas.compactMap { ($0["function"] as? [String: Any])?["arguments"] as? String },
            [#"{"city":"Paris"}"#], file: file, line: line)
    }

    // AC1
    func testNonStreamingFalseReturnsExactlyTheFirstCall() async throws {
        let response = try await run(events: [[first, second]], stream: false, parallel: "false")
        XCTAssertEqual(response.head.status, .ok)
        let parsed = try nonStreamingCalls(response.body)
        XCTAssertEqual(parsed.calls.count, 1, response.body)
        XCTAssertEqual(parsed.calls.first?["id"] as? String, "call_0")
        let function = parsed.calls.first?["function"] as? [String: Any]
        XCTAssertEqual(function?["name"] as? String, "get_weather")
        XCTAssertEqual(function?["arguments"] as? String, #"{"city":"Paris"}"#)
        XCTAssertEqual(parsed.finish, "tool_calls")
    }

    func testNonStreamingFalseIsCumulativeAcrossTwoEvents() async throws {
        let response = try await run(events: [[first], [second]], stream: false, parallel: "false")
        let parsed = try nonStreamingCalls(response.body)
        XCTAssertEqual(parsed.calls.compactMap { $0["id"] as? String }, ["call_0"], response.body)
        XCTAssertEqual(parsed.finish, "tool_calls")
    }

    // AC2
    func testStreamingFalseWithTwoCallsInOneEventEmitsOnlyIndexZero() async throws {
        let response = try await run(events: [[first, second]], stream: true, parallel: "false")
        let parsed = try streamedDeltas(response.body)
        assertOnlyFirstCall(parsed.deltas)
        XCTAssertEqual(parsed.finish, "tool_calls")
        XCTAssertFalse(response.body.contains("call_1"), response.body)
    }

    func testStreamingFalseWithTwoCallsInTwoEventsEmitsOnlyTheFirst() async throws {
        let response = try await run(events: [[first], [second]], stream: true, parallel: "false")
        let parsed = try streamedDeltas(response.body)
        assertOnlyFirstCall(parsed.deltas)
        XCTAssertEqual(parsed.finish, "tool_calls")
        XCTAssertFalse(response.body.contains("call_1"), response.body)
    }

    // AC3
    func testNonStreamingAbsentAndTrueReturnBothCalls() async throws {
        for parallel in [nil, "true", "null"] {
            let response = try await run(events: [[first, second]], stream: false, parallel: parallel)
            let parsed = try nonStreamingCalls(response.body)
            XCTAssertEqual(
                parsed.calls.compactMap { $0["id"] as? String }, ["call_0", "call_1"],
                "parallel=\(parallel ?? "absent"): \(response.body)")
            XCTAssertEqual(parsed.finish, "tool_calls")
        }
    }

    func testStreamingAbsentAndTrueEmitBothCalls() async throws {
        for parallel in [nil, "true", "null"] {
            let response = try await run(events: [[first, second]], stream: true, parallel: parallel)
            let parsed = try streamedDeltas(response.body)
            XCTAssertEqual(
                parsed.deltas.compactMap { $0["id"] as? String }, ["call_0", "call_1"],
                "parallel=\(parallel ?? "absent"): \(response.body)")
            XCTAssertEqual(Set(parsed.deltas.compactMap { $0["index"] as? Int }), [0, 1])
            XCTAssertEqual(parsed.finish, "tool_calls")
        }
    }

    // AC4: one call under false is unchanged.
    func testFalseWithASingleCallIsUnchangedInBothModes() async throws {
        let plain = try await run(events: [[first]], stream: false, parallel: "false")
        XCTAssertEqual(
            try nonStreamingCalls(plain.body).calls.compactMap { $0["id"] as? String }, ["call_0"])
        let streamed = try await run(events: [[first]], stream: true, parallel: "false")
        assertOnlyFirstCall(try streamedDeltas(streamed.body).deltas)
    }
}

/// Fake backend that emits each element of `events` as one `.toolCalls` delta, then completes with
/// finish_reason `tool_calls`.
private final class EventsBackend: ServingGenerationBackend, Sendable {
    private let events: [[OpenAIToolCall]]

    init(events: [[OpenAIToolCall]]) {
        self.events = events
    }

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 64, maxBytes: 4_096))
        let lease = ServingRequestLease(id: ServingRequestID("parallel-tool-calls"), onCancel: {})
        let handle = ServingGenerationHandle(
            responseID: "chatcmpl-parallel",
            created: 1_775_000_000,
            model: request.model,
            route: .continuousBatchNoSpec,
            mailbox: mailbox,
            lease: lease)
        let events = self.events
        Task {
            for calls in events {
                try? await mailbox.send(.toolCalls(calls))
            }
            try? await mailbox.send(
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .toolCalls,
                        usage: OpenAIChatUsage(promptTokens: 3, completionTokens: 3))))
            await mailbox.finish()
        }
        return handle
    }
}
