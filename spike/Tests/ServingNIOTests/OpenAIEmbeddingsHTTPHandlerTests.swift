import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import os
import XCTest

@testable import ServingCore
@testable import ServingNIO

final class OpenAIEmbeddingsHTTPHandlerTests: XCTestCase {
    // MARK: - No backend configured

    func testNoEmbeddingsBackendReturns404WithMessage() async throws {
        let channel = try await makeChannel(embeddingsBackend: nil)

        try await writeEmbeddingsRequest(channel, body: embeddingsRequestBody())
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .notFound)
        let payload = try errorPayload(from: response.body)
        XCTAssertEqual(payload.type, .invalidRequest)
        XCTAssertEqual(
            payload.message,
            "This server does not serve embeddings: no embedding model is loaded")

        _ = try await channel.finish()
    }

    // MARK: - Successful embedding

    func testConfiguredBackendReturns200WithVectorsAndUsage() async throws {
        let backend = FakeEmbeddingsBackend(
            script: .success(
                vectors: [[1, 2, 3], [4, 5, 6]],
                promptTokens: 7))
        let channel = try await makeChannel(embeddingsBackend: backend)

        try await writeEmbeddingsRequest(
            channel,
            body: embeddingsRequestBody(input: ["hello", "world"]))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        let json = try jsonObject(from: response.body)
        XCTAssertEqual(json["object"] as? String, "list")
        let data = try XCTUnwrap(json["data"] as? [[String: Any]])
        XCTAssertEqual(data.count, 2)
        XCTAssertEqual(data[0]["index"] as? Int, 0)
        XCTAssertEqual(data[1]["index"] as? Int, 1)
        XCTAssertEqual(data[0]["object"] as? String, "embedding")
        XCTAssertEqual(data[0]["embedding"] as? [Double], [1, 2, 3])
        XCTAssertEqual(data[1]["embedding"] as? [Double], [4, 5, 6])
        let usage = try XCTUnwrap(json["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 7)
        XCTAssertEqual(usage["total_tokens"] as? Int, 7)
        let callCount = await backend.callCount()
        XCTAssertEqual(callCount, 1)

        _ = try await channel.finish()
    }

    func testBase64EncodingFormatRoundTripsToSameFloatValues() async throws {
        let vector: [Float] = [1.5, -2.25, 3.0]
        let backend = FakeEmbeddingsBackend(
            script: .success(vectors: [vector], promptTokens: 3))
        let channel = try await makeChannel(embeddingsBackend: backend)

        try await writeEmbeddingsRequest(
            channel,
            body: embeddingsRequestBody(input: ["hello"], encodingFormat: "base64"))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        let json = try jsonObject(from: response.body)
        let data = try XCTUnwrap(json["data"] as? [[String: Any]])
        let base64 = try XCTUnwrap(data[0]["embedding"] as? String)
        let bytes = try XCTUnwrap(Data(base64Encoded: base64))
        let decoded = bytes.withUnsafeBytes { raw -> [Float] in
            raw.bindMemory(to: UInt32.self).map { Float(bitPattern: UInt32(littleEndian: $0)) }
        }
        XCTAssertEqual(decoded, vector)

        _ = try await channel.finish()
    }

    // MARK: - Invalid request body

    func testInvalidBodyReturns400WithoutCallingBackend() async throws {
        let backend = FakeEmbeddingsBackend(script: .success(vectors: [[1]], promptTokens: 1))
        let channel = try await makeChannel(embeddingsBackend: backend)

        // `input` is required and missing entirely.
        try await writeEmbeddingsRequest(channel, body: #"{"model":"embed-model"}"#)
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .badRequest)
        let payload = try errorPayload(from: response.body)
        XCTAssertEqual(payload.type, .invalidRequest)
        let callCount = await backend.callCount()
        XCTAssertEqual(callCount, 0)

        _ = try await channel.finish()
    }

    // The API key gates embeddings like every non-probe route: a request without the configured
    // bearer token is refused before the body is decoded or the backend is called.
    func testMissingAPIKeyReturns401WithoutCallingBackend() async throws {
        let backend = FakeEmbeddingsBackend(script: .success(vectors: [[1]], promptTokens: 1))
        let channel = try await makeChannel(
            embeddingsBackend: backend,
            configuration: ServingHTTPConfiguration(
                launchedModel: "qwen3-32b",
                requestLimits: .productionDefault,
                requiredBearerToken: "secret",
                maximumNonStreamingResponseBytes: 1_048_576,
                backpressureStallTimeout: .seconds(1)))

        try await writeEmbeddingsRequest(
            channel, body: #"{"model":"embed-model","input":"hello"}"#)
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .unauthorized)
        let callCount = await backend.callCount()
        XCTAssertEqual(callCount, 0)

        _ = try await channel.finish()
    }

    // MARK: - Backend defects

    func testBackendReturningWrongVectorCountReturns500NotMalformed200() async throws {
        let backend = FakeEmbeddingsBackend(
            script: .success(vectors: [[1, 2, 3]], promptTokens: 5))
        let channel = try await makeChannel(embeddingsBackend: backend)

        try await writeEmbeddingsRequest(
            channel,
            body: embeddingsRequestBody(input: ["hello", "world"]))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .internalServerError)
        let payload = try errorPayload(from: response.body)
        XCTAssertEqual(payload.type, .serverError)

        _ = try await channel.finish()
    }

    func testBackendThrowingReturns500() async throws {
        struct BackendFailure: Error {}
        let backend = FakeEmbeddingsBackend(script: .throwing(BackendFailure()))
        let channel = try await makeChannel(embeddingsBackend: backend)

        try await writeEmbeddingsRequest(channel, body: embeddingsRequestBody())
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .internalServerError)
        let payload = try errorPayload(from: response.body)
        XCTAssertEqual(payload.type, .serverError)

        _ = try await channel.finish()
    }

    func testUnsupportedDimensionsReturns400() async throws {
        let backend = FakeEmbeddingsBackend(
            script: .success(vectors: [[1, 2, 3, 4]], promptTokens: 3))
        let channel = try await makeChannel(embeddingsBackend: backend)

        try await writeEmbeddingsRequest(
            channel,
            body: embeddingsRequestBody(input: ["hello"], dimensions: 2))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .badRequest)
        let payload = try errorPayload(from: response.body)
        XCTAssertEqual(payload.type, .invalidRequest)
        XCTAssertEqual(payload.param, "dimensions")

        _ = try await channel.finish()
    }

    // MARK: - Method / transport

    func testGETIsMethodNotAllowed() async throws {
        let channel = try await makeChannel(embeddingsBackend: nil)

        _ = try await channel.writeInbound(
            HTTPServerRequestPart.head(
                HTTPRequestHead(
                    version: .http1_1,
                    method: .GET,
                    uri: "/v1/embeddings",
                    headers: ["host": "localhost"])))
        _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .methodNotAllowed)

        _ = try await channel.finish()
    }

    func testKeepAliveServesTwoSequentialRequestsOnOneChannel() async throws {
        let backend = FakeEmbeddingsBackend(
            script: .success(vectors: [[1, 2]], promptTokens: 1))
        let channel = try await makeChannel(embeddingsBackend: backend)

        try await writeEmbeddingsRequest(channel, body: embeddingsRequestBody())
        let first = try await collectResponse(from: channel)
        XCTAssertEqual(first.head.status, .ok)

        try await writeEmbeddingsRequest(channel, body: embeddingsRequestBody())
        let second = try await collectResponse(from: channel)
        XCTAssertEqual(second.head.status, .ok)

        let callCount = await backend.callCount()
        XCTAssertEqual(callCount, 2)

        _ = try await channel.finish()
    }

    // MARK: - Logging / metrics

    func testRequestLogLineUsesEmbeddingsRoute() async throws {
        let recorder = EmbeddingsRequestLogRecorder()
        let backend = FakeEmbeddingsBackend(
            script: .success(vectors: [[1, 2]], promptTokens: 1))
        let channel = try await makeChannel(
            embeddingsBackend: backend,
            requestLog: recorder.sink())

        try await writeEmbeddingsRequest(channel, body: embeddingsRequestBody())
        _ = try await collectResponse(from: channel)
        await waitUntilEmbeddings { recorder.lines().count == 1 }

        let line = try XCTUnwrap(recorder.lines().first)
        let object = try jsonObject(from: line)
        XCTAssertEqual(object["route"] as? String, "/v1/embeddings")

        _ = try await channel.finish()
    }

    func testHTTPMetricsCounterIncrementsForEmbeddingsRoute() async throws {
        let backend = FakeEmbeddingsBackend(
            script: .success(vectors: [[1, 2]], promptTokens: 1))
        let configuration = embeddingsConfiguration()
        let channel = try await makeChannel(
            embeddingsBackend: backend,
            configuration: configuration)

        try await writeEmbeddingsRequest(channel, body: embeddingsRequestBody())
        _ = try await collectResponse(from: channel)
        await waitUntilEmbeddings {
            configuration.httpMetrics.snapshot().counters.contains {
                $0.route == "/v1/embeddings" && $0.statusClass == "2xx"
            }
        }

        let snapshot = configuration.httpMetrics.snapshot()
        XCTAssertTrue(
            snapshot.counters.contains { $0.route == "/v1/embeddings" && $0.statusClass == "2xx" })

        _ = try await channel.finish()
    }
}

// MARK: - Test doubles

private final class FakeEmbeddingsBackend: ServingEmbeddingsBackend, Sendable {
    enum Script: Sendable {
        case success(vectors: [[Float]], promptTokens: Int)
        case throwing(any Error)
    }

    private struct State: Sendable {
        var script: Script
        var callCount = 0
    }

    private let state: OSAllocatedUnfairLock<State>

    init(script: Script) {
        state = OSAllocatedUnfairLock(initialState: State(script: script))
    }

    func embed(_ request: OpenAIEmbeddingsRequest) async throws -> ServingEmbeddingsResult {
        let script = state.withLock { state -> Script in
            state.callCount += 1
            return state.script
        }
        switch script {
        case .success(let vectors, let promptTokens):
            return ServingEmbeddingsResult(vectors: vectors, promptTokens: promptTokens)
        case .throwing(let error):
            throw error
        }
    }

    func callCount() async -> Int {
        state.withLock { $0.callCount }
    }
}

/// A no-op chat backend: `OpenAIChatCompletionsHTTPHandler` requires a non-optional chat
/// `backend` on every initializer, but none of these embeddings-focused tests ever exercise
/// `/v1/chat/completions` or `/v1/completions`.
private struct UnusedChatBackend: ServingGenerationBackend, Sendable {
    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        throw OpenAIServingError.server("unused in embeddings tests", code: nil)
    }
}

private final class EmbeddingsRequestLogRecorder: Sendable {
    private let state = OSAllocatedUnfairLock<[String]>(initialState: [])

    func sink() -> @Sendable (String) -> Void {
        { [state] line in
            state.withLock { $0.append(line) }
        }
    }

    func lines() -> [String] {
        state.withLock { $0 }
    }
}

// MARK: - Helpers

private struct DecodedErrorPayload {
    let type: OpenAIErrorType
    let message: String
    let param: String?
}

private func errorPayload(from body: String) throws -> DecodedErrorPayload {
    let object = try jsonObject(from: body)
    let error = try XCTUnwrap(object["error"] as? [String: Any])
    let rawType = try XCTUnwrap(error["type"] as? String)
    let type = try XCTUnwrap(OpenAIErrorType(rawValue: rawType))
    let message = try XCTUnwrap(error["message"] as? String)
    return DecodedErrorPayload(type: type, message: message, param: error["param"] as? String)
}

private func jsonObject(from text: String) throws -> [String: Any] {
    try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
}

private func embeddingsRequestBody(
    model: String = "embed-model",
    input: [String] = ["hello"],
    encodingFormat: String? = nil,
    dimensions: Int? = nil
) -> String {
    var fields: [String] = [
        "\"model\":\"\(model)\"",
        "\"input\":[\(input.map { "\"\($0)\"" }.joined(separator: ","))]",
    ]
    if let encodingFormat {
        fields.append("\"encoding_format\":\"\(encodingFormat)\"")
    }
    if let dimensions {
        fields.append("\"dimensions\":\(dimensions)")
    }
    return "{\(fields.joined(separator: ","))}"
}

private func embeddingsHead(contentLength: Int) -> HTTPRequestHead {
    HTTPRequestHead(
        version: .http1_1,
        method: .POST,
        uri: "/v1/embeddings",
        headers: [
            "host": "localhost",
            "content-type": "application/json",
            "content-length": "\(contentLength)",
        ])
}

private func writeEmbeddingsRequest(
    _ channel: NIOAsyncTestingChannel,
    body: String
) async throws {
    let head = embeddingsHead(contentLength: body.utf8.count)
    _ = try await channel.writeInbound(HTTPServerRequestPart.head(head))
    _ = try await channel.writeInbound(
        HTTPServerRequestPart.body(ByteBuffer(string: body)))
    _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
}

private struct EmbeddingsCollectedResponse {
    let head: HTTPResponseHead
    let body: String
}

private func collectResponse(
    from channel: NIOAsyncTestingChannel
) async throws -> EmbeddingsCollectedResponse {
    var head: HTTPResponseHead?
    var body = ""
    while true {
        let part: HTTPServerResponsePart = try await channel.waitForOutboundWrite()
        switch part {
        case .head(let value):
            head = value
        case .body(.byteBuffer(var buffer)):
            body += buffer.readString(length: buffer.readableBytes) ?? ""
        case .body(.fileRegion):
            XCTFail("Serving responses must not emit file regions")
        case .end:
            return EmbeddingsCollectedResponse(head: try XCTUnwrap(head), body: body)
        }
    }
}

private func waitUntilEmbeddings(
    attempts: Int = 10_000,
    _ predicate: () -> Bool
) async {
    for _ in 0..<attempts {
        if predicate() {
            return
        }
        await Task.yield()
    }
    XCTFail("Condition was not reached")
}

private func embeddingsConfiguration() -> ServingHTTPConfiguration {
    ServingHTTPConfiguration(
        launchedModel: "qwen3-32b",
        requestLimits: .productionDefault,
        requiredBearerToken: nil,
        maximumNonStreamingResponseBytes: 1_048_576,
        backpressureStallTimeout: .seconds(1))
}

private func embeddingsConfiguration(
    requestLog: @escaping @Sendable (String) -> Void
) -> ServingHTTPConfiguration {
    ServingHTTPConfiguration(
        launchedModel: "qwen3-32b",
        requestLimits: .productionDefault,
        requiredBearerToken: nil,
        maximumNonStreamingResponseBytes: 1_048_576,
        backpressureStallTimeout: .seconds(1),
        requestLog: requestLog)
}

private func makeChannel(
    embeddingsBackend: (any ServingEmbeddingsBackend)?,
    configuration: ServingHTTPConfiguration? = nil,
    requestLog: (@Sendable (String) -> Void)? = nil
) async throws -> NIOAsyncTestingChannel {
    let resolvedConfiguration: ServingHTTPConfiguration
    if let configuration {
        resolvedConfiguration = configuration
    } else if let requestLog {
        resolvedConfiguration = embeddingsConfiguration(requestLog: requestLog)
    } else {
        resolvedConfiguration = embeddingsConfiguration()
    }
    return try await NIOAsyncTestingChannel { channel in
        try channel.pipeline.syncOperations.addHandler(
            OpenAIChatCompletionsHTTPHandler(
                configuration: resolvedConfiguration,
                backend: UnusedChatBackend(),
                embeddingsBackend: embeddingsBackend))
    }
}
