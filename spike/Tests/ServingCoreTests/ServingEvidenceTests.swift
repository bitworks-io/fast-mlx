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
            completed: true,
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

    func testIncompleteResponseAndLifecycleSnapshotsRemainCanonicalAndPromptFree() throws {
        let emptyDigest = Data().sha256HexForTest()
        let before = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 16_384,
            mlxActiveBytes: 4_096,
            mlxCacheBytes: 1_024,
            mlxPeakBytes: 8_192)
        let active = try ServingEvidence.ResourceSnapshot(
            activeRequests: 1,
            coordinatorSlots: 1,
            reservedKVBytes: 2_048,
            maxReservedKVBytes: 16_384,
            mlxActiveBytes: 6_144,
            mlxCacheBytes: 1_024,
            mlxPeakBytes: 8_192)
        let terminal = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 16_384,
            mlxActiveBytes: 4_096,
            mlxCacheBytes: 1_024,
            mlxPeakBytes: 8_192)
        let evidence = try ServingEvidence(
            request: ServingEvidence.Request(
                method: "POST",
                path: "/v1/chat/completions",
                headers: [],
                body: Data(#"{"messages":[{"content":"PROMPT-SENTINEL"}]}"#.utf8),
                stream: true,
                messageCount: 1,
                maxCompletionTokens: 8),
            response: ServingEvidence.Response(
                status: nil,
                completed: false,
                durationMilliseconds: 3,
                chunkCount: 0,
                bodyBytes: 0,
                bodySHA256: emptyDigest),
            cancellation: .init(
                cancelled: true,
                reason: .clientDisconnected),
            resources: .init(
                admission: .accepted,
                before: before,
                active: active,
                terminal: terminal))

        let canonical = try evidence.canonicalJSONData()
        let json = try XCTUnwrap(String(data: canonical, encoding: .utf8))

        XCTAssertEqual(evidence.schemaVersion, 2)
        XCTAssertFalse(json.contains("PROMPT-SENTINEL"))
        XCTAssertTrue(json.contains(#""completed":false"#))
        XCTAssertFalse(json.contains(#""status":"#))
        XCTAssertTrue(json.contains(#""active_requests":1"#))
        XCTAssertEqual(
            try ServingEvidence.decodeCanonicalJSONData(canonical),
            evidence)
    }

    func testResourceSnapshotFailuresAreCanonicalAndUnique() throws {
        let resources = try ServingEvidence.ResourceFacts(
            admission: .accepted,
            failedSnapshots: [.before, .active, .terminal])
        let evidence = try ServingEvidence(
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
                durationMilliseconds: 1,
                chunkCount: 1,
                body: Data("{}".utf8)),
            resources: resources)

        let canonical = try evidence.canonicalJSONData()
        XCTAssertEqual(
            try ServingEvidence.decodeCanonicalJSONData(canonical),
            evidence)
        XCTAssertThrowsError(
            try ServingEvidence.ResourceFacts(
                admission: .accepted,
                failedSnapshots: [.active, .before]))
        XCTAssertThrowsError(
            try ServingEvidence.ResourceFacts(
                admission: .accepted,
                failedSnapshots: [.active, .active]))
        let snapshot = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 0,
            mlxActiveBytes: 0,
            mlxCacheBytes: 0,
            mlxPeakBytes: 0)
        XCTAssertThrowsError(
            try ServingEvidence.ResourceFacts(
                admission: .accepted,
                before: snapshot,
                failedSnapshots: [.before]))
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
        XCTAssertThrowsError(
            try ServingEvidence.ResourceSnapshot(
                activeRequests: -1,
                coordinatorSlots: 0,
                reservedKVBytes: 0,
                maxReservedKVBytes: 0,
                mlxActiveBytes: 0,
                mlxCacheBytes: 0,
                mlxPeakBytes: 0))
    }

    // MARK: - measured-vs-modeled drift fields (fit-checked-serve differentiator #2)

    func testResourceSnapshotFitDriftFieldsRoundTripAndAreAbsentWhenNil() throws {
        let withDrift = try ServingEvidence.ResourceSnapshot(
            activeRequests: 1,
            coordinatorSlots: 1,
            reservedKVBytes: 2_048,
            maxReservedKVBytes: 16_384,
            mlxActiveBytes: 6_144,
            mlxCacheBytes: 1_024,
            mlxPeakBytes: 9_000,
            fitModeledPeakBytes: 10_000,
            fitMeasuredPeakBytes: 9_000,
            fitDriftVerdict: "conservative",
            fitDriftFraction: -0.10,
            fitModeledWeightsBytes: 6_000,
            fitModeledKVBytes: 2_000,
            fitModeledTransientBytes: 1_500,
            fitModeledHeadroomBytes: 500)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = try XCTUnwrap(String(data: encoder.encode(withDrift), encoding: .utf8))
        // Machine contract: flat snake_case keys mirroring FitCheckMeasuredReport.machineReadableFields().
        XCTAssertTrue(json.contains(#""fit_modeled_peak_bytes":10000"#), json)
        XCTAssertTrue(json.contains(#""fit_measured_peak_bytes":9000"#), json)
        XCTAssertTrue(json.contains(#""fit_drift":"conservative""#), json)
        XCTAssertTrue(json.contains(#""fit_drift_frac":-0.1"#), json)
        XCTAssertTrue(json.contains(#""fit_modeled_weights_bytes":6000"#), json)
        XCTAssertTrue(json.contains(#""fit_modeled_kv_bytes":2000"#), json)
        XCTAssertTrue(json.contains(#""fit_modeled_transient_bytes":1500"#), json)
        XCTAssertTrue(json.contains(#""fit_modeled_headroom_bytes":500"#), json)
        XCTAssertEqual(try JSONDecoder().decode(ServingEvidence.ResourceSnapshot.self,
                                                from: Data(json.utf8)), withDrift)

        // Old contract stays byte-compatible: a snapshot with no drift omits every fit_* key.
        let noDrift = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 16_384,
            mlxActiveBytes: 4_096,
            mlxCacheBytes: 1_024,
            mlxPeakBytes: 8_192)
        let plainJSON = try XCTUnwrap(String(data: encoder.encode(noDrift), encoding: .utf8))
        XCTAssertFalse(plainJSON.contains("fit_"), plainJSON)
        XCTAssertEqual(try JSONDecoder().decode(ServingEvidence.ResourceSnapshot.self,
                                                from: Data(plainJSON.utf8)), noDrift)
    }

    func testFitDriftFieldsSurviveTheProductionCanonicalEvidencePath() throws {
        // The acceptance for differentiator #2: the drift fields must survive the SERIALIZER production
        // actually uses — ServingEvidence.canonicalJSONData() (the JSONL sink) and decodeCanonicalJSONData
        // — not just a bare JSONEncoder. This locks the nested ResourceSnapshot's synthesized encode
        // against a future hand-rolled canonical serializer that would silently drop the new keys.
        let terminal = try ServingEvidence.ResourceSnapshot(
            activeRequests: 0,
            coordinatorSlots: 0,
            reservedKVBytes: 0,
            maxReservedKVBytes: 16_384,
            mlxActiveBytes: 4_096,
            mlxCacheBytes: 1_024,
            mlxPeakBytes: 9_500,
            fitModeledPeakBytes: 10_000,
            fitMeasuredPeakBytes: 9_500,
            fitDriftVerdict: "conservative",
            fitDriftFraction: -0.05,
            fitModeledWeightsBytes: 6_000,
            fitModeledKVBytes: 2_000,
            fitModeledTransientBytes: 1_500,
            fitModeledHeadroomBytes: 500)
        let evidence = try ServingEvidence(
            request: ServingEvidence.Request(
                method: "POST",
                path: "/v1/chat/completions",
                headers: [],
                body: Data("{}".utf8),
                stream: false,
                messageCount: 1,
                maxCompletionTokens: 8),
            response: ServingEvidence.Response(
                status: 200,
                durationMilliseconds: 5,
                chunkCount: 0,
                body: Data()),
            resources: .init(admission: .accepted, terminal: terminal))

        let canonical = try evidence.canonicalJSONData()
        let json = try XCTUnwrap(String(data: canonical, encoding: .utf8))
        XCTAssertTrue(json.contains(#""fit_modeled_peak_bytes":10000"#), json)
        XCTAssertTrue(json.contains(#""fit_drift":"conservative""#), json)
        XCTAssertTrue(json.contains("fit_modeled_kv_bytes"), json)
        // Full production round-trip preserves the drift-bearing snapshot exactly.
        XCTAssertEqual(try ServingEvidence.decodeCanonicalJSONData(canonical), evidence)
    }

    func testResourceSnapshotFitDriftRejectsExplicitNullAndNegativeBytes() throws {
        // Explicit null for a canonically-omitted drift field is rejected (matches the evidence contract).
        let explicitNull = Data(#"""
        {"active_requests":0,"coordinator_slots":0,"reserved_kv_bytes":0,\#
        "max_reserved_kv_bytes":0,"mlx_active_bytes":0,"mlx_cache_bytes":0,\#
        "mlx_peak_bytes":0,"fit_modeled_peak_bytes":null}
        """#.utf8)
        XCTAssertThrowsError(
            try JSONDecoder().decode(ServingEvidence.ResourceSnapshot.self, from: explicitNull))

        // A drift byte count is a byte total: negative is rejected. (The signed fraction may be negative.)
        XCTAssertThrowsError(
            try ServingEvidence.ResourceSnapshot(
                activeRequests: 0,
                coordinatorSlots: 0,
                reservedKVBytes: 0,
                maxReservedKVBytes: 0,
                mlxActiveBytes: 0,
                mlxCacheBytes: 0,
                mlxPeakBytes: 0,
                fitModeledPeakBytes: -1))
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
