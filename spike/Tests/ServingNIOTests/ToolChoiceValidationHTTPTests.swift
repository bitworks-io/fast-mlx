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
        payload: String
    ) async throws -> (head: HTTPResponseHead, body: String, starts: Int) {
        let backend = CountingBackend()
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
}

/// Counts `start` calls and completes immediately with plain text.
private final class CountingBackend: ServingGenerationBackend, Sendable {
    private let starts = OSAllocatedUnfairLock(initialState: 0)

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
        Task {
            try? await mailbox.send(.text("ok"))
            try? await mailbox.send(
                .completion(
                    ServingGenerationCompletion(
                        finishReason: .stop,
                        usage: OpenAIChatUsage(promptTokens: 3, completionTokens: 1))))
            await mailbox.finish()
        }
        return handle
    }
}
