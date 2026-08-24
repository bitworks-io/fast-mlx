import XCTest

@testable import HarnessCore
import MLXLMCommon
import NIOCore
import NIOEmbedded
import NIOHTTP1
import ServingCore
import ServingNIO
import SpikeCore
@testable import SpikeServingAdapters

final class ExactQwen35MTPServingHTTPHandlerTests: XCTestCase {
    func testActualExactMTPBackendNonStreamingTraversesOpenAIHandler() async throws {
        let backend = try makeBackend(
            text: ["hel", "lo"],
            finishReason: .length,
            promptTokens: 3,
            completionTokens: 2)
        let recorder = ServingEvidenceRecorder()
        let channel = try await makeChannel(
            backend: backend,
            configuration: defaultConfiguration(
                evidence: ServingHTTPEvidenceConfiguration(
                    snapshot: nil,
                    record: { evidence in try await recorder.record(evidence) },
                    reportFailure: { _ in })))

        try await writeRequest(channel, body: requestBody(stream: false))
        let response = try await collectResponse(from: channel)
        await waitUntil { await recorder.evidence.count == 1 }

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "application/json")
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(response.body.utf8)) as? [String: Any])
        let choice = try XCTUnwrap((object["choices"] as? [[String: Any]])?.first)
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])
        XCTAssertEqual(message["content"] as? String, "hello")
        XCTAssertEqual(choice["finish_reason"] as? String, "length")
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 2)
        XCTAssertEqual(usage["total_tokens"] as? Int, 5)
        let recorded = await recorder.snapshot()
        XCTAssertEqual(recorded.first?.route?.kind, .exactQwen35MTP)

        _ = try await channel.finish()
    }

    func testActualExactMTPBackendStreamingTraversesOpenAIHandler() async throws {
        let backend = try makeBackend(
            text: ["hel", "lo"],
            finishReason: .length,
            promptTokens: 3,
            completionTokens: 2)
        let channel = try await makeChannel(backend: backend)

        try await writeRequest(channel, body: requestBody(stream: true))
        let response = try await collectResponse(from: channel)

        XCTAssertEqual(response.head.status, .ok)
        XCTAssertEqual(response.head.headers.first(name: "content-type"), "text/event-stream")
        let roleRange = try XCTUnwrap(response.body.range(of: #""role":"assistant""#))
        let firstRange = try XCTUnwrap(response.body.range(of: #""content":"hel""#))
        let secondRange = try XCTUnwrap(response.body.range(of: #""content":"lo""#))
        let finishRange = try XCTUnwrap(response.body.range(of: #""finish_reason":"length""#))
        let doneRange = try XCTUnwrap(response.body.range(of: "data: [DONE]\n\n"))
        XCTAssertLessThan(roleRange.lowerBound, firstRange.lowerBound)
        XCTAssertLessThan(firstRange.lowerBound, secondRange.lowerBound)
        XCTAssertLessThan(secondRange.lowerBound, finishRange.lowerBound)
        XCTAssertLessThan(finishRange.lowerBound, doneRange.lowerBound)
        XCTAssertEqual(response.body.components(separatedBy: "data: [DONE]\n\n").count - 1, 1)
        let events = try sseJSONEvents(from: response.body)
        let terminal = try XCTUnwrap(
            events.last { object in
                let choices = object["choices"] as? [[String: Any]]
                return choices?.first?["finish_reason"] as? String == "length"
            })
        let usage = try XCTUnwrap(terminal["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 2)
        XCTAssertEqual(usage["total_tokens"] as? Int, 5)

        _ = try await channel.finish()
    }
}

private func makeBackend(
    text: [String],
    finishReason: GenerateStopReason,
    promptTokens: Int,
    completionTokens: Int
) throws -> ExactQwen35MTPServingBackend {
    try ExactQwen35MTPServingBackend(
        launchedModel: "qwen3-32b",
        enabled: true,
        runner: HTTPFixtureRunner(
            text: text,
            finishReason: finishReason,
            promptTokens: promptTokens,
            completionTokens: completionTokens),
        scalarFallback: HTTPFixtureScalarFallback(),
        scalarFallbackIsolation: .strictlySeparateRawTarget,
        codec: HTTPFixtureCodec(promptTokens: Array(0..<promptTokens)),
        configuration: .init(
            defaultMaximumCompletionTokens: 8,
            mailboxCapacity: .init(maxDeltas: 8, maxBytes: 1_024)))
}

private let lock = QwenMTPKnownArtifactLocks.qwen35_9BDepth1

private func exactBinding() -> QwenMTPArtifactBinding {
    QwenMTPArtifactBinding(
        targetModelID: lock.targetIdentity.modelID,
        drafterModelID: lock.drafterIdentity.modelID,
        targetRevision: lock.targetIdentity.revision,
        drafterRevision: lock.drafterIdentity.revision,
        sourceRevision: lock.sourceRevision,
        architecture: lock.architecture,
        runtimeBlockSize: 3,
        maximumAcceptedDraftTokens: 2)
}

private struct HTTPFixtureCodec: ScalarServingTextCodec {
    let promptTokens: [Int]

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        promptTokens
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        HTTPFixtureDetokenizer()
    }
}

private struct HTTPFixtureDetokenizer: ScalarServingDetokenizer {
    mutating func append(token: Int) {}
    mutating func next() -> String? { nil }
}

private final class HTTPFixtureRunner: ExactQwen35MTPServingRunner, Sendable {
    let binding: QwenMTPArtifactBinding? = exactBinding()
    private let text: [String]
    private let finishReason: GenerateStopReason
    private let promptTokens: Int
    private let completionTokens: Int

    init(
        text: [String],
        finishReason: GenerateStopReason,
        promptTokens: Int,
        completionTokens: Int
    ) {
        self.text = text
        self.finishReason = finishReason
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }

    func start(
        _ request: ExactQwen35MTPServingRunnerRequest
    ) async throws -> ExactQwen35MTPServingRunnerHandle {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        let task = Task {
            for chunk in text {
                continuation.yield(.chunk(chunk))
            }
            continuation.yield(
                .info(
                    GenerateCompletionInfo(
                        promptTokenCount: promptTokens,
                        generationTokenCount: completionTokens,
                        promptTime: 0,
                        generationTime: 0,
                        stopReason: finishReason)))
            continuation.finish()
        }
        return ExactQwen35MTPServingRunnerHandle(stream: stream, task: task)
    }
}

private final class HTTPFixtureScalarFallback: ServingGenerationBackend, Sendable {
    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        throw OpenAIServingError.invalidRequest("unexpected scalar fallback", param: nil)
    }
}

private struct CollectedResponse {
    let head: HTTPResponseHead
    let body: String
}

private func makeChannel(
    backend: any ServingGenerationBackend,
    configuration: ServingHTTPConfiguration = defaultConfiguration()
) async throws -> NIOAsyncTestingChannel {
    try await NIOAsyncTestingChannel { channel in
        try channel.pipeline.syncOperations.addHandler(
            OpenAIChatCompletionsHTTPHandler(
                configuration: configuration,
                backend: backend))
    }
}

private func defaultConfiguration(
    evidence: ServingHTTPEvidenceConfiguration? = nil
) -> ServingHTTPConfiguration {
    ServingHTTPConfiguration(
        launchedModel: "qwen3-32b",
        requestLimits: .productionDefault,
        requiredBearerToken: nil,
        maximumNonStreamingResponseBytes: 1_048_576,
        backpressureStallTimeout: .seconds(1),
        evidence: evidence)
}

private func requestBody(stream: Bool) -> String {
    """
    {"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"max_completion_tokens":8,"temperature":0,"stream":\(stream)}
    """
}

private func writeRequest(
    _ channel: NIOAsyncTestingChannel,
    body: String
) async throws {
    let head = HTTPRequestHead(
        version: .http1_1,
        method: .POST,
        uri: "/v1/chat/completions",
        headers: [
            "host": "localhost",
            "content-type": "application/json",
            "content-length": "\(body.utf8.count)",
        ])
    _ = try await channel.writeInbound(HTTPServerRequestPart.head(head))
    _ = try await channel.writeInbound(
        HTTPServerRequestPart.body(ByteBuffer(string: body)))
    _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
}

private func collectResponse(
    from channel: NIOAsyncTestingChannel
) async throws -> CollectedResponse {
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
            return CollectedResponse(head: try XCTUnwrap(head), body: body)
        }
    }
}

private func sseJSONEvents(from body: String) throws -> [[String: Any]] {
    try body
        .components(separatedBy: "\n\n")
        .compactMap { rawEvent -> [String: Any]? in
            guard rawEvent.hasPrefix("data: "), rawEvent != "data: [DONE]" else {
                return nil
            }
            let payload = String(rawEvent.dropFirst("data: ".count))
            return try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any])
        }
}

private actor ServingEvidenceRecorder {
    private(set) var evidence: [ServingEvidence] = []

    func record(_ value: ServingEvidence) throws {
        evidence.append(value)
    }

    func snapshot() -> [ServingEvidence] {
        evidence
    }
}

private func waitUntil(
    attempts: Int = 10_000,
    _ predicate: () async -> Bool
) async {
    for _ in 0..<attempts {
        if await predicate() {
            return
        }
        await Task.yield()
    }
    XCTFail("Condition was not reached")
}
