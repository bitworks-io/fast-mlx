import Foundation
import XCTest

import ServingCore
import ServingNIO
@testable import SpikeServingAdapters

final class LoadedScalarServingIntegrationTests: XCTestCase {
    func testLoadedScalarHTTPMatchesInProcessControl() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard
            let modelPath = environment["FASTMLX_SCALAR_TEST_MODEL_PATH"],
            let launchedModel = environment["FASTMLX_SCALAR_TEST_MODEL"],
            let memoryLimit = environment["FASTMLX_SCALAR_TEST_MEMORY_LIMIT_BYTES"]
                .flatMap(Int.init),
            let cacheLimit = environment["FASTMLX_SCALAR_TEST_CACHE_LIMIT_BYTES"]
                .flatMap(Int.init)
        else {
            throw XCTSkip(
                "Set the FASTMLX_SCALAR_TEST_* variables for the loaded-model proof")
        }

        let loaded = try await loadScalarServingModel(
            configuration: ScalarServingModelLoadConfiguration(
                launchedModel: launchedModel,
                modelDirectory: URL(
                    fileURLWithPath: modelPath,
                    isDirectory: true),
                memoryLimitBytes: memoryLimit,
                cacheLimitBytes: cacheLimit,
                backendConfiguration: ScalarServingBackendConfiguration(
                    defaultMaximumCompletionTokens: 12,
                    maximumQueuedRequests: 1,
                    queueRetryAfterSeconds: 1,
                    mailboxCapacity: .init(
                        maxDeltas: 4,
                        maxBytes: 16 * 1_024))))
        XCTAssertEqual(
            Set(loaded.startupReport.nativeCacheKinds),
            [.denseAttention])

        let messages = [
            OpenAIChatMessage(role: .system, text: "Answer briefly."),
            OpenAIChatMessage(
                role: .user,
                text: "Name one color of the daytime sky."),
        ]
        let controlRequest = OpenAIChatCompletionRequest(
            model: launchedModel,
            messages: messages,
            maxCompletionTokens: 12,
            temperature: 0,
            choiceCount: 1,
            stream: false,
            stop: [])
        let control = try await collectControl(
            try await loaded.backend.start(controlRequest))

        let server = try await ServingHTTPServer.start(
            configuration: ServingHTTPConfiguration(
                launchedModel: launchedModel,
                requestLimits: .productionDefault,
                requiredBearerToken: nil,
                maximumNonStreamingResponseBytes: 1_048_576,
                backpressureStallTimeout: .seconds(5)),
            backend: loaded.backend)
        do {
            guard let port = server.localAddress.port else {
                XCTFail("Expected a TCP port")
                try await server.shutdown(gracePeriod: .seconds(5))
                return
            }

            let nonStreaming = try await performRequest(
                port: port,
                launchedModel: launchedModel,
                stream: false)
            let streaming = try await performRequest(
                port: port,
                launchedModel: launchedModel,
                stream: true)

            XCTAssertEqual(nonStreaming.statusCode, 200)
            XCTAssertEqual(streaming.statusCode, 200)
            let nonStreamingResult = try parseNonStreaming(nonStreaming.data)
            let streamingResult = try parseStreaming(streaming.data)
            XCTAssertEqual(nonStreamingResult.text, control.text)
            XCTAssertEqual(streamingResult.text, control.text)
            XCTAssertEqual(nonStreamingResult.usage, control.completion.usage)
            XCTAssertEqual(streamingResult.usage, control.completion.usage)
            XCTAssertEqual(streamingResult.usage, nonStreamingResult.usage)
            XCTAssertEqual(streamingResult.finishReason, control.completion.finishReason)
            XCTAssertEqual(streamingResult.doneCount, 1)
            XCTAssertGreaterThan(streamingResult.chunkCount, 1)

            try await server.shutdown(gracePeriod: .seconds(5))
            let activeConnectionCount = await server.activeConnectionCount
            XCTAssertEqual(activeConnectionCount, 0)
        } catch {
            try? await server.shutdown(gracePeriod: .seconds(5))
            throw error
        }
    }
}

private struct LoadedScalarControlResult {
    let text: String
    let completion: ServingGenerationCompletion
}

private struct LoadedScalarHTTPResult {
    let statusCode: Int
    let data: Data
}

private struct LoadedScalarStreamingResult {
    let text: String
    let finishReason: OpenAIChatFinishReason
    let usage: OpenAIChatUsage
    let chunkCount: Int
    let doneCount: Int
}

private func collectControl(
    _ handle: ServingGenerationHandle
) async throws -> LoadedScalarControlResult {
    var text = ""
    var completion: ServingGenerationCompletion?
    while let delta = try await handle.mailbox.next() {
        switch delta {
        case .text(let value):
            text += value
        case .toolCalls:
            break
        case .completion(let value):
            completion = value
        }
    }
    guard let completion else {
        throw LoadedScalarIntegrationError.missingCompletion
    }
    return LoadedScalarControlResult(
        text: text,
        completion: completion)
}

private func performRequest(
    port: Int,
    launchedModel: String,
    stream: Bool
) async throws -> LoadedScalarHTTPResult {
    var request = URLRequest(
        url: URL(
            string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(
        withJSONObject: [
            "model": launchedModel,
            "messages": [
                ["role": "system", "content": "Answer briefly."],
                [
                    "role": "user",
                    "content": "Name one color of the daytime sky.",
                ],
            ],
            "max_completion_tokens": 12,
            "temperature": 0,
            "n": 1,
            "stream": stream,
        ],
        options: [.sortedKeys])
    let (data, response) = try await URLSession.shared.data(for: request)
    guard let response = response as? HTTPURLResponse else {
        throw LoadedScalarIntegrationError.nonHTTPResponse
    }
    return LoadedScalarHTTPResult(
        statusCode: response.statusCode,
        data: data)
}

private func parseNonStreaming(
    _ data: Data
) throws -> (text: String, usage: OpenAIChatUsage) {
    guard
        let root = try JSONSerialization.jsonObject(with: data)
            as? [String: Any],
        let choices = root["choices"] as? [[String: Any]],
        let first = choices.first,
        let message = first["message"] as? [String: Any],
        let text = message["content"] as? String,
        let usage = root["usage"] as? [String: Any],
        let promptTokens = usage["prompt_tokens"] as? Int,
        let completionTokens = usage["completion_tokens"] as? Int
    else {
        throw LoadedScalarIntegrationError.invalidNonStreamingResponse
    }
    return (
        text,
        OpenAIChatUsage(
            promptTokens: promptTokens,
            completionTokens: completionTokens)
    )
}

private func parseStreaming(
    _ data: Data
) throws -> LoadedScalarStreamingResult {
    let lines = String(decoding: data, as: UTF8.self)
        .split(separator: "\n", omittingEmptySubsequences: true)
    var text = ""
    var finishReason: OpenAIChatFinishReason?
    var usage: OpenAIChatUsage?
    var chunkCount = 0
    var doneCount = 0

    for line in lines {
        guard line.hasPrefix("data: ") else {
            throw LoadedScalarIntegrationError.invalidStreamingResponse
        }
        let payload = line.dropFirst("data: ".count)
        if payload == "[DONE]" {
            doneCount += 1
            continue
        }
        guard
            let payloadData = payload.data(using: .utf8),
            let root = try JSONSerialization.jsonObject(with: payloadData)
                as? [String: Any],
            let choices = root["choices"] as? [[String: Any]],
            let first = choices.first,
            let delta = first["delta"] as? [String: Any]
        else {
            throw LoadedScalarIntegrationError.invalidStreamingResponse
        }
        chunkCount += 1
        text += delta["content"] as? String ?? ""
        if let rawFinishReason = first["finish_reason"] as? String {
            finishReason = OpenAIChatFinishReason(
                rawValue: rawFinishReason)
        }
        if let rawUsage = root["usage"] as? [String: Any] {
            guard
                usage == nil,
                let promptTokens = rawUsage["prompt_tokens"] as? Int,
                let completionTokens = rawUsage["completion_tokens"] as? Int,
                rawUsage["total_tokens"] as? Int
                    == promptTokens + completionTokens
            else {
                throw LoadedScalarIntegrationError.invalidStreamingResponse
            }
            usage = OpenAIChatUsage(
                promptTokens: promptTokens,
                completionTokens: completionTokens)
        }
    }

    guard let finishReason, let usage else {
        throw LoadedScalarIntegrationError.invalidStreamingResponse
    }
    return LoadedScalarStreamingResult(
        text: text,
        finishReason: finishReason,
        usage: usage,
        chunkCount: chunkCount,
        doneCount: doneCount)
}

private enum LoadedScalarIntegrationError: Error {
    case missingCompletion
    case nonHTTPResponse
    case invalidNonStreamingResponse
    case invalidStreamingResponse
}
