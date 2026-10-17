import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import os
import XCTest

@testable import ServingCore
@testable import ServingNIO

/// An unsatisfiable `tool_choice` is a 400 naming `tool_choice`, answered BEFORE the backend starts,
/// for both streaming and non-streaming requests. A declared named function still reaches the backend.
final class ToolChoiceValidationHTTPTests: XCTestCase {
    private func payload(stream: Bool, tools: String?, toolChoice: String) -> String {
        let toolsField = tools.map { #","tools":\#($0)"# } ?? ""
        return
            #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}]"#
            + toolsField + #","tool_choice":\#(toolChoice),"max_completion_tokens":8,"temperature":0,"stream":\#(stream)}"#
    }

    private func run(
        payload: String,
        backend: CountingBackend = CountingBackend()
    ) async throws -> (head: HTTPResponseHead, body: String, starts: Int) {
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
        return (try XCTUnwrap(head), text, backend.startCount)
    }

    private let declared = #"[{"type":"function","function":{"name":"get_weather"}}]"#
    private let namedOther = #"{"type":"function","function":{"name":"other"}}"#
    private let namedDeclared = #"{"type":"function","function":{"name":"get_weather"}}"#

    private func assertRefusedBeforeBackend(
        _ result: (head: HTTPResponseHead, body: String, starts: Int),
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(result.head.status, .badRequest, result.body, file: file, line: line)
        XCTAssertTrue(
            result.body.contains(#""param":"tool_choice""#), result.body, file: file, line: line)
        XCTAssertEqual(result.starts, 0, file: file, line: line)
    }

    func testRequiredWithoutToolsIs400BeforeBackend() async throws {
        for stream in [false, true] {
            let result = try await run(
                payload: payload(stream: stream, tools: nil, toolChoice: #""required""#))
            assertRefusedBeforeBackend(result)
        }
    }

    func testUndeclaredNamedFunctionIs400BeforeBackend() async throws {
        for stream in [false, true] {
            let result = try await run(
                payload: payload(stream: stream, tools: declared, toolChoice: namedOther))
            assertRefusedBeforeBackend(result)
        }
    }

    func testDeclaredNamedFunctionReachesBackend() async throws {
        let result = try await run(
            payload: payload(stream: false, tools: declared, toolChoice: namedDeclared))
        XCTAssertEqual(result.head.status, .ok, result.body)
        XCTAssertEqual(result.starts, 1)
    }

    /// A named choice enforced by a backend answers ONE call with `finish_reason` `stop` (or `length`
    /// for a truncated call) and no `content`; the HTTP layer must pass both through untouched
    /// (never rewriting them to `tool_calls`), streaming and not.
    func testEnforcedNamedCallKeepsStopFinishReasonAndNullContent() async throws {
        let backend = CountingBackend(
            toolCall: OpenAIToolCall(
                id: "call_abc", function: .init(name: "get_weather", arguments: #"{"city":"Paris"}"#)),
            finishReason: .stop)
        let plain = try await run(
            payload: payload(stream: false, tools: declared, toolChoice: namedDeclared), backend: backend)
        XCTAssertEqual(plain.head.status, .ok, plain.body)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(plain.body.utf8)) as? [String: Any])
        let choice = try XCTUnwrap((json["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(choice["finish_reason"] as? String, "stop")
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        XCTAssertTrue(message["content"] is NSNull, plain.body)
        let calls = try XCTUnwrap(message["tool_calls"] as? [[String: Any]])
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual((calls[0]["function"] as? [String: Any])?["name"] as? String, "get_weather")
        XCTAssertEqual(
            (calls[0]["function"] as? [String: Any])?["arguments"] as? String, #"{"city":"Paris"}"#)

        let streamed = try await run(
            payload: payload(stream: true, tools: declared, toolChoice: namedDeclared), backend: backend)
        XCTAssertEqual(streamed.head.status, .ok, streamed.body)
        XCTAssertTrue(streamed.body.contains(#""finish_reason":"stop""#), streamed.body)
        XCTAssertFalse(streamed.body.contains(#""finish_reason":"tool_calls""#), streamed.body)
        XCTAssertTrue(streamed.body.contains(#""name":"get_weather""#), streamed.body)
        XCTAssertFalse(streamed.body.contains(#""content":"#), streamed.body)

        let truncated = try await run(
            payload: payload(stream: false, tools: declared, toolChoice: namedDeclared),
            backend: CountingBackend(
                toolCall: OpenAIToolCall(
                    id: "call_abc", function: .init(name: "get_weather", arguments: #"{"ci"#)),
                finishReason: .length))
        XCTAssertTrue(truncated.body.contains(#""finish_reason":"length""#), truncated.body)
        XCTAssertTrue(truncated.body.contains(#"{\"ci"#), truncated.body)
    }
}

/// Counts `start` calls and completes immediately with plain text, or -- when given a `toolCall` --
/// with exactly that one tool call and no text.
private final class CountingBackend: ServingGenerationBackend, Sendable {
    private let starts = OSAllocatedUnfairLock(initialState: 0)
    private let toolCall: OpenAIToolCall?
    private let finishReason: OpenAIChatFinishReason

    init(toolCall: OpenAIToolCall? = nil, finishReason: OpenAIChatFinishReason = .stop) {
        self.toolCall = toolCall
        self.finishReason = finishReason
    }

    var startCount: Int { starts.withLock { $0 } }

    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        starts.withLock { $0 += 1 }
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 64, maxBytes: 4_096))
        let lease = ServingRequestLease(id: ServingRequestID("tool-choice-validation"), onCancel: {})
        let handle = ServingGenerationHandle(
            responseID: "chatcmpl-toolchoice",
            created: 1_775_000_000,
            model: request.model,
            route: .continuousBatchNoSpec,
            mailbox: mailbox,
            lease: lease)
        let toolCall = toolCall
        let finishReason = finishReason
        Task {
            if let toolCall {
                try? await mailbox.send(.toolCalls([toolCall]))
            } else {
                try? await mailbox.send(.text("ok"))
            }
            try? await mailbox.send(
                .completion(
                    ServingGenerationCompletion(
                        finishReason: finishReason,
                        usage: OpenAIChatUsage(promptTokens: 3, completionTokens: 1))))
            await mailbox.finish()
        }
        return handle
    }
}
