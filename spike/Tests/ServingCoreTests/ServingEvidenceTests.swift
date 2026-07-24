import Foundation
import XCTest

@testable import ServingCore

final class ServingEvidenceTests: XCTestCase {
    func testCanonicalEvidenceRedactsRequestAndResponseContent() throws {
        let promptSentinel = "PROMPT-SENTINEL-raw-user-secret"
        let apiKeySentinel = "sk-API-KEY-SENTINEL"
        let generatedSentinel = "GENERATED-SENTINEL-output"
        let requestBody = Data("""
        {"model":"qwen3-32b","messages":[{"role":"user","content":"\(promptSentinel)"}],"stream":true,"max_completion_tokens":7}
        """.utf8)
        let responseBody = Data("""
        {"choices":[{"message":{"content":"\(generatedSentinel)"}}]}
        """.utf8)

        let evidence = try ServingEvidence(
            request: ServingEvidence.Request(
                method: "POST",
                path: "/v1/chat/completions",
                headers: [
                    .init(name: "Authorization", value: "Bearer \(apiKeySentinel)"),
                    .init(name: "Content-Type", value: "application/json"),
                    .init(name: "X-Raw-Prompt", value: promptSentinel),
                    .init(name: "Accept", value: "text/event-stream"),
                ],
                body: requestBody,
                stream: true,
                messageCount: 1,
                maxCompletionTokens: 7),
            response: ServingEvidence.Response(
                status: 200,
                durationMilliseconds: 12.5,
                chunkCount: 3,
                body: responseBody),
            route: .init(kind: .continuousBatchNoSpec, batchSize: 2),
            cancellation: .init(cancelled: false, reason: nil),
            resources: .init(admission: .accepted, queueDepth: 0))

        let encoded = try evidence.canonicalJSONData()
        let json = try XCTUnwrap(String(data: encoded, encoding: .utf8))

        XCTAssertFalse(json.contains(promptSentinel))
        XCTAssertFalse(json.contains(apiKeySentinel))
        XCTAssertFalse(json.contains(generatedSentinel))
        XCTAssertFalse(json.contains(#""name":"authorization""#))
        XCTAssertFalse(json.localizedCaseInsensitiveContains("x-raw-prompt"))
        XCTAssertFalse(json.contains("Bearer"))
        XCTAssertTrue(json.contains(#""authorization_present":true"#))
        XCTAssertTrue(json.contains(#""headers":[{"name":"accept","value":"present"},{"name":"content-type","value":"application/json"}]"#))
        XCTAssertTrue(json.contains(#""body_bytes":\#(requestBody.count)"#))
        XCTAssertTrue(json.contains(#""body_sha256":"\#(requestBody.sha256HexForTest())""#))
        XCTAssertTrue(json.contains(#""chunk_count":3"#))
        XCTAssertTrue(json.contains(#""body_bytes":\#(responseBody.count)"#))
        XCTAssertTrue(json.contains(#""body_sha256":"\#(responseBody.sha256HexForTest())""#))
        XCTAssertEqual(encoded, try evidence.canonicalJSONData())
    }

    func testUnsupportedHeadersNeverSurviveEvenWhenTheirValuesLookUseful() throws {
        let evidence = try ServingEvidence(
            request: ServingEvidence.Request(
                method: "POST",
                path: "/v1/chat/completions",
                headers: [
                    .init(name: "X-Request-ID", value: "request-id-that-must-not-leak"),
                    .init(name: "Content-Length", value: "1234"),
                    .init(name: "Accept", value: "application/json"),
                ],
                body: Data("{}".utf8),
                stream: false,
                messageCount: 0,
                maxCompletionTokens: nil),
            response: ServingEvidence.Response(
                status: 400,
                durationMilliseconds: 0,
                chunkCount: 0,
                body: Data()))

        let json = try XCTUnwrap(String(data: try evidence.canonicalJSONData(), encoding: .utf8))

        XCTAssertTrue(json.contains(#""headers":[{"name":"accept","value":"present"}]"#))
        XCTAssertFalse(json.contains("request-id-that-must-not-leak"))
        XCTAssertFalse(json.localizedCaseInsensitiveContains("content-length"))
    }

    func testAllowedHeaderNamesDoNotMakeTheirRawValuesSafeToPersist() throws {
        let headerSentinel = "HEADER-SENTINEL-secret"
        let evidence = try ServingEvidence(
            request: ServingEvidence.Request(
                method: "POST",
                path: "/v1/chat/completions",
                headers: [
                    .init(name: "Accept", value: "text/event-stream; \(headerSentinel)"),
                    .init(name: "Content-Type", value: "application/json; \(headerSentinel)"),
                ],
                body: Data("{}".utf8),
                stream: true,
                messageCount: 0,
                maxCompletionTokens: nil),
            response: ServingEvidence.Response(
                status: 200,
                durationMilliseconds: 0,
                chunkCount: 0,
                body: Data()))

        let json = try XCTUnwrap(String(data: try evidence.canonicalJSONData(), encoding: .utf8))

        XCTAssertFalse(json.contains(headerSentinel))
        XCTAssertTrue(json.contains(#""name":"accept","value":"present""#))
        XCTAssertTrue(json.contains(#""name":"content-type","value":"application/json""#))
    }

    func testRequestRejectsNoncanonicalMethodOrPathThatCouldPersistSecrets() {
        let secret = "QUERY-SENTINEL-secret"
        for (method, path) in [
            ("post", "/v1/chat/completions"),
            ("POST ", "/v1/chat/completions"),
            ("POST", "/v1/chat/completions?api_key=\(secret)"),
            ("POST", "/other/\(secret)"),
        ] {
            XCTAssertThrowsError(
                try ServingEvidence.Request(
                    method: method,
                    path: path,
                    headers: [],
                    body: Data("{}".utf8),
                    stream: false,
                    messageCount: 0,
                    maxCompletionTokens: nil))
        }
    }

    func testDecoderRejectsUnknownFieldsInsteadOfCleaningUnsafeArtifacts() {
        let digest = String(repeating: "0", count: 64)
        let unsafe = Data(
            """
            {
              "schema_version": 1,
              "request": {
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [],
                "authorization_present": false,
                "body_bytes": 0,
                "body_sha256": "\(digest)",
                "stream": false,
                "message_count": 0,
                "max_completion_tokens": null,
                "raw_prompt": "PROMPT-SENTINEL"
              },
              "response": {
                "status": 200,
                "duration_milliseconds": 0,
                "chunk_count": 0,
                "body_bytes": 0,
                "body_sha256": "\(digest)"
              },
              "route": null,
              "cancellation": null,
              "resources": null
            }
            """.utf8)

        XCTAssertThrowsError(try JSONDecoder().decode(ServingEvidence.self, from: unsafe))
    }

    func testDecoderRejectsForbiddenOrNoncanonicalHeadersInsteadOfCleaningThem() {
        let digest = String(repeating: "0", count: 64)
        for headers in [
            #"[{"name":"authorization","value":"Bearer sk-RAW-SENTINEL"}]"#,
            #"[{"name":"accept","value":"text/event-stream; RAW-SENTINEL"}]"#,
            #"[{"name":"Content-Type","value":"application/json"}]"#,
        ] {
            let unsafe = Data(
                """
                {
                  "schema_version": 1,
                  "request": {
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "headers": \(headers),
                    "authorization_present": true,
                    "body_bytes": 0,
                    "body_sha256": "\(digest)",
                    "stream": false,
                    "message_count": 0,
                    "max_completion_tokens": null
                  },
                  "response": {
                    "status": 200,
                    "duration_milliseconds": 0,
                    "chunk_count": 0,
                    "body_bytes": 0,
                    "body_sha256": "\(digest)"
                  },
                  "route": null,
                  "cancellation": null,
                  "resources": null
                }
                """.utf8)

            XCTAssertThrowsError(
                try JSONDecoder().decode(ServingEvidence.self, from: unsafe),
                "decoded evidence must already contain only canonical redacted headers")
        }
    }

    func testCanonicalDecoderRejectsAnyNoncanonicalByteRepresentation() throws {
        let evidence = try makeMinimalEvidence()
        let canonical = try evidence.canonicalJSONData()

        XCTAssertEqual(
            try ServingEvidence.decodeCanonicalJSONData(canonical),
            evidence)

        var leadingWhitespace = Data(" ".utf8)
        leadingWhitespace.append(canonical)
        XCTAssertThrowsError(
            try ServingEvidence.decodeCanonicalJSONData(leadingWhitespace))

        let canonicalJSON = try XCTUnwrap(
            String(data: canonical, encoding: .utf8))
        let duplicateKnownKey = Data(
            canonicalJSON.replacingOccurrences(
                of: #""method":"POST""#,
                with: #""method":"POST","method":"POST""#).utf8)
        XCTAssertThrowsError(
            try ServingEvidence.decodeCanonicalJSONData(duplicateKnownKey))
    }

    func testResponseCanBeFinalizedFromIncrementalBodyFactsWithoutRetainingOutput() throws {
        let generatedSentinel = "GENERATED-SENTINEL-streamed-output"
        let body = Data(generatedSentinel.utf8)
        let response = try ServingEvidence.Response(
            status: 200,
            durationMilliseconds: 7.5,
            chunkCount: 4,
            bodyBytes: body.count,
            bodySHA256: body.sha256HexForTest())

        let encoded = try ServingEvidence.canonicalJSONEncoder.encode(response)
        let json = try XCTUnwrap(String(data: encoded, encoding: .utf8))
        XCTAssertFalse(json.contains(generatedSentinel))
        XCTAssertTrue(json.contains(#""body_bytes":\#(body.count)"#))
        XCTAssertTrue(json.contains(#""body_sha256":"\#(body.sha256HexForTest())""#))
    }

    func testDecoderRejectsExplicitNullForCanonicallyOmittedOptionalFields() throws {
        let canonical = try makeMinimalEvidence().canonicalJSONData()
        let json = try XCTUnwrap(String(data: canonical, encoding: .utf8))
        let explicitTopLevelNull = Data(
            json.replacingOccurrences(
                of: #"{"request":"#,
                with: #"{"route":null,"request":"#).utf8)
        let explicitNestedNull = Data(
            json.replacingOccurrences(
                of: #""message_count":0,"#,
                with: #""max_completion_tokens":null,"message_count":0,"#).utf8)

        XCTAssertThrowsError(
            try JSONDecoder().decode(
                ServingEvidence.self,
                from: explicitTopLevelNull))
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                ServingEvidence.self,
                from: explicitNestedNull))
    }

    func testRouteAndCancellationFactsRejectContradictoryTelemetry() {
        XCTAssertThrowsError(
            try ServingEvidence.RouteFacts(
                kind: .continuousBatchNoSpec,
                batchSize: 0))
        XCTAssertThrowsError(
            try ServingEvidence.CancellationFacts(
                cancelled: true,
                reason: nil))
        XCTAssertThrowsError(
            try ServingEvidence.CancellationFacts(
                cancelled: false,
                reason: .clientDisconnected))
    }

    func testEvidenceRejectsNonfiniteAndNegativeNumericFacts() {
        XCTAssertThrowsError(
            try ServingEvidence.Response(
                status: 200,
                durationMilliseconds: .nan,
                chunkCount: 0,
                body: Data()))
        XCTAssertThrowsError(
            try ServingEvidence.Response(
                status: 200,
                durationMilliseconds: .infinity,
                chunkCount: 0,
                body: Data()))
        XCTAssertThrowsError(
            try ServingEvidence.Response(
                status: 200,
                durationMilliseconds: -0.1,
                chunkCount: 0,
                body: Data()))
        XCTAssertThrowsError(
            try ServingEvidence.Response(
                status: 200,
                durationMilliseconds: 0,
                chunkCount: -1,
                body: Data()))
        XCTAssertThrowsError(
            try ServingEvidence.Request(
                method: "POST",
                path: "/v1/chat/completions",
                headers: [],
                body: Data(),
                stream: false,
                messageCount: -1,
                maxCompletionTokens: nil))
        XCTAssertThrowsError(
            try ServingEvidence.Request(
                method: "POST",
                path: "/v1/chat/completions",
                headers: [],
                body: Data(),
                stream: false,
                messageCount: 0,
                maxCompletionTokens: -1))
        XCTAssertThrowsError(try ServingEvidence.RouteFacts(kind: .soloPLD, batchSize: -1))
        XCTAssertThrowsError(try ServingEvidence.ResourceFacts(admission: .accepted, queueDepth: -1))
    }
}

private func makeMinimalEvidence() throws -> ServingEvidence {
    try ServingEvidence(
        request: ServingEvidence.Request(
            method: "POST",
            path: "/v1/chat/completions",
            headers: [],
            body: Data("{}".utf8),
            stream: false,
            messageCount: 0,
            maxCompletionTokens: nil),
        response: ServingEvidence.Response(
            status: 200,
            durationMilliseconds: 0,
            chunkCount: 0,
            body: Data()))
}

private extension Data {
    func sha256HexForTest() -> String {
        ServingEvidence.SHA256.hexDigest(of: self)
    }
}
