import XCTest

@testable import ServingCore

final class OpenAIChatCompletionsTests: XCTestCase {
    func testAcceptsSupportedTextOnlyRequest() throws {
        let body = """
        {
          "model": "qwen3-32b",
          "messages": [
            {"role": "developer", "content": "Be terse."},
            {"role": "system", "content": "Use plain text."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": [{"type": "text", "text": "Continue"}]}
          ],
          "max_completion_tokens": 32,
          "temperature": 0,
          "n": 1,
          "stream": true,
          "stop": ["</s>", "<|end|>"]
        }
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))

        XCTAssertEqual(request.model, "qwen3-32b")
        XCTAssertEqual(request.messages.map(\.role), [.developer, .system, .user, .assistant, .user])
        XCTAssertEqual(request.messages.map(\.text), ["Be terse.", "Use plain text.", "Hello", "Hi", "Continue"])
        XCTAssertEqual(request.maxCompletionTokens, 32)
        XCTAssertEqual(request.temperature, 0)
        XCTAssertEqual(request.choiceCount, 1)
        XCTAssertTrue(request.stream)
        XCTAssertEqual(request.stop, ["</s>", "<|end|>"])
    }

    func testAcceptsPositiveTemperatureAndDecodesSamplingFields() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"temperature":0.7,"top_p":0.9,"top_k":40,"min_p":0.05,"seed":42}
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))

        XCTAssertEqual(request.temperature, 0.7)
        XCTAssertEqual(request.topP, 0.9)
        XCTAssertEqual(request.topK, 40)
        XCTAssertEqual(request.minP, 0.05)
        XCTAssertEqual(request.seed, 42)
    }

    func testSamplingFieldsAreNilWhenAbsent() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}]}
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))

        XCTAssertNil(request.temperature)
        XCTAssertNil(request.topP)
        XCTAssertNil(request.topK)
        XCTAssertNil(request.minP)
        XCTAssertNil(request.seed)
    }

    func testOutOfRangeTemperatureIsRejectedByPolicyNotDecode() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"temperature":3.0}
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.temperature, 3.0)

        XCTAssertThrowsError(try ServingSamplingPolicy.resolve(from: request)) { error in
            XCTAssertEqual(
                error as? ServingSamplingPolicyError, .temperatureOutOfRange(3.0))
        }
    }

    func testDeprecatedMaxTokensAliasAndConflictBehavior() throws {
        let alias = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_tokens":7}
        """

        XCTAssertEqual(try OpenAIChatCompletionRequest.decodeStrict(from: Data(alias.utf8)).maxCompletionTokens, 7)

        let matching = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_tokens":7,"max_completion_tokens":7}
        """

        XCTAssertEqual(try OpenAIChatCompletionRequest.decodeStrict(from: Data(matching.utf8)).maxCompletionTokens, 7)

        let conflicting = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_tokens":7,"max_completion_tokens":8}
        """

        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(conflicting.utf8)),
            type: .invalidRequest,
            param: "max_completion_tokens")
    }

    func testRequestLimitsRejectOversizedBodyAndCompletionBudget() throws {
        let limits = OpenAIChatRequestLimits(
            maximumBodyBytes: 128,
            maximumCompletionTokens: 8)

        XCTAssertEqual(
            OpenAIChatRequestLimits.productionDefault.maximumCompletionTokens,
            4_096)

        let boundaryBudget = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":8}
        """
        XCTAssertEqual(
            try OpenAIChatCompletionRequest.decodeStrict(
                from: Data(boundaryBudget.utf8),
                limits: limits).maxCompletionTokens,
            8)

        let oversizedBody = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"\(String(repeating: "x", count: 128))"}]}
        """
        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(
                from: Data(oversizedBody.utf8),
                limits: limits),
            type: .invalidRequest,
            param: nil)

        let oversizedBudget = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":9}
        """
        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(
                from: Data(oversizedBudget.utf8),
                limits: limits),
            type: .invalidRequest,
            param: "max_completion_tokens")
    }

    func testRequestLimitsCanDeferCompletionBudgetToModelAwareResolution() throws {
        let limits = OpenAIChatRequestLimits(
            maximumBodyBytes: 1_048_576,
            maximumCompletionTokens: 4_096,
            enforceMaximumCompletionTokensDuringDecoding: false)
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":8192}
        """

        let request = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(body.utf8),
            limits: limits)

        XCTAssertEqual(request.maxCompletionTokens, 8_192)
    }

    func testProductionDefaultStillRejectsCompletionBudgetAboveLegacyStaticLimit() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":4097}
        """

        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest,
            param: "max_completion_tokens")
    }

    func testLaunchedModelIdentityFailsClosedBeforeAdmission() throws {
        let body = """
        {"model":"other-model","messages":[{"role":"user","content":"Hi"}]}
        """
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))

        XCTAssertOpenAIError(
            try request.requireLaunchedModel("qwen3-32b"),
            type: .invalidRequest,
            param: "model")
        XCTAssertNoThrow(try request.requireLaunchedModel("other-model"))
    }

    // An arbitrary unknown TOP-LEVEL key (e.g. "unknown") is no longer rejected here — it is now
    // tolerated (see `testUnknownTopLevelFieldsAreAcceptedAndRecordedAsIgnored` below). Every
    // remaining case here is either a nested-object violation (still strict) or a typed/range
    // violation of a KNOWN field name, neither of which this policy change affects.
    func testRejectsUnknownAndUnsupportedFieldsBeforeAdmission() throws {
        let cases: [(String, String?)] = [
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"temperature":true}"#, "temperature"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"top_p":true}"#, "top_p"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"top_k":0}"#, "top_k"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"top_k":-1}"#, "top_k"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"min_p":-0.1}"#, "min_p"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"min_p":1.5}"#, "min_p"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"seed":true}"#, "seed"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"n":2}"#, "n"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}"#, "messages.content"),
            (#"{"model":"","messages":[{"role":"user","content":"Hi"}]}"#, "model"),
            (#"{"model":"qwen3-32b","messages":[]}"#, "messages"),
            (#"{"model":"qwen3-32b","messages":[{"role":"tool","content":"Hi"}]}"#, "messages.tool_call_id"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi","tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]}]}"#, "messages.tool_calls"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"tool_choice":"bogus"}"#, "tool_choice"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"tools":[{"type":"function"}]}"#, "tools[0].function"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"stop":["a","b","c","d","e"]}"#, "stop"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"stop":""}"#, "stop"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":0}"#, "max_completion_tokens"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":-1}"#, "max_completion_tokens"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":true}"#, "max_completion_tokens"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":"8192"}"#, "max_completion_tokens"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":1.5}"#, "max_completion_tokens"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":9223372036854775808}"#, "max_completion_tokens"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":18446744073709551617}"#, "max_completion_tokens"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":1e100}"#, "max_completion_tokens"),
        ]

        for (body, param) in cases {
            XCTAssertOpenAIError(
                try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)),
                type: .invalidRequest,
                param: param,
                file: #filePath,
                line: #line)
        }
    }

    func testMetadataOnlyFieldsAreAcceptedAndRecordedAsIgnored() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],
         "user":"user-123","metadata":{"a":"b"},"store":true,"service_tier":"auto"}
        """
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(
            request.ignoredFields,
            ["metadata", "service_tier", "store", "user"])
    }

    func testStoreFalseIsAlsoAcceptedAndRecordedAsIgnored() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"store":false}
        """
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.ignoredFields, ["store"])
    }

    func testWrongTypedUserAndMetadataAreRejected() throws {
        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(
                from: Data(
                    #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"user":42}"#
                        .utf8)),
            type: .invalidRequest,
            param: "user")

        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(
                from: Data(
                    #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"metadata":"nope"}"#
                        .utf8)),
            type: .invalidRequest,
            param: "metadata")

        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(
                from: Data(
                    #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"metadata":{"a":1}}"#
                        .utf8)),
            type: .invalidRequest,
            param: "metadata")
    }

    func testNeutralValuedFieldsAreAcceptedAndRecordedAsIgnored() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],
         "logprobs":false,"response_format":{"type":"text"},"logit_bias":{}}
        """
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(
            request.ignoredFields,
            ["logit_bias", "response_format"])
        // `logprobs:false` is honored (not ignored): it decodes to no logprobs request at all.
        XCTAssertNil(request.logprobsRequest)
    }

    func testNonNeutralValuedFieldsAreRejectedWithSpecificParam() throws {
        let cases: [(String, String)] = [
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"top_logprobs":1}"#, "top_logprobs"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"text","extra":1}}"#, "response_format"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"json_object","extra":1}}"#, "response_format"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"x"}}}"#, "response_format"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"bogus"}}"#, "response_format"),
            (#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"logit_bias":{"123":1}}"#, "logit_bias"),
        ]
        for (body, param) in cases {
            XCTAssertOpenAIError(
                try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)),
                type: .invalidRequest,
                param: param,
                file: #filePath,
                line: #line)
        }
    }

    // MARK: - Chat logprobs / top_logprobs (real feature)

    func testChatLogprobsTrueWithNoTopLogprobsDefaultsToZeroAlternatives() throws {
        let body = #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"logprobs":true}"#
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.logprobsRequest, .chat(topLogprobs: 0))
        XCTAssertEqual(request.ignoredFields, [])
    }

    func testChatLogprobsTrueWithTopLogprobsIsHonored() throws {
        let body = #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"logprobs":true,"top_logprobs":5}"#
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.logprobsRequest, .chat(topLogprobs: 5))
    }

    func testChatLogprobsFalseOrAbsentLeavesNoLogprobsRequest() throws {
        for body in [
            #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}]}"#,
            #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"logprobs":false}"#,
        ] {
            let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
            XCTAssertNil(request.logprobsRequest, body)
        }
    }

    // Threshold is exactly `> 0` (OpenAI's own behavior): `top_logprobs:1` without `logprobs:true`
    // is rejected...
    func testChatTopLogprobsWithoutLogprobsTrueIsRejected() throws {
        let cases = [
            #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"top_logprobs":1}"#,
            #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"logprobs":false,"top_logprobs":1}"#,
        ]
        for body in cases {
            XCTAssertOpenAIError(
                try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)),
                type: .invalidRequest,
                param: "top_logprobs",
                file: #filePath,
                line: #line)
        }
    }

    // ...but `top_logprobs:0` alone is neutral: zero alternatives is the same "not requested" state
    // as omitting the field entirely, so it must NOT 400.
    func testChatTopLogprobsZeroWithoutLogprobsTrueIsNeutral() throws {
        let cases = [
            #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"top_logprobs":0}"#,
            #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"logprobs":false,"top_logprobs":0}"#,
        ]
        for body in cases {
            let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
            XCTAssertNil(request.logprobsRequest, body)
        }
    }

    func testChatTopLogprobsOutOfRangeIsRejected() throws {
        for value in [-1, 21] {
            let body = #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"logprobs":true,"top_logprobs":\#(value)}"#
            XCTAssertOpenAIError(
                try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)),
                type: .invalidRequest,
                param: "top_logprobs",
                file: #filePath,
                line: #line)
        }
    }

    func testChatTopLogprobsBoundaryValuesAreAccepted() throws {
        for value in [0, 20] {
            let body = #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"logprobs":true,"top_logprobs":\#(value)}"#
            let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
            XCTAssertEqual(request.logprobsRequest, .chat(topLogprobs: value))
        }
    }

    // Matches the SAME `as? Bool` convention every other boolean field in this decoder uses (e.g.
    // `validateNeutralEcho`), including its NSNumber-bridging quirk (a numeric `0`/`1` casts to
    // `Bool` on this platform) — a non-numeric, non-boolean value like a string still fails closed.
    func testChatLogprobsWrongTypeIsRejected() throws {
        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(
                from: Data(
                    #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"logprobs":"true"}"#
                        .utf8)),
            type: .invalidRequest,
            param: "logprobs")
    }

    func testStreamOptionsIncludeUsageIsParsedBothWays() throws {
        let trueBody = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"stream":true,"stream_options":{"include_usage":true}}
        """
        XCTAssertTrue(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(trueBody.utf8)).includeUsage)

        let falseBody = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"stream":true,"stream_options":{"include_usage":false}}
        """
        XCTAssertFalse(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(falseBody.utf8)).includeUsage)

        // Absent stream_options entirely: includeUsage stays false, and it never appears ignored
        // (it is honored, not ignored).
        let absentBody = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"stream":true}
        """
        let absent = try OpenAIChatCompletionRequest.decodeStrict(from: Data(absentBody.utf8))
        XCTAssertFalse(absent.includeUsage)
        XCTAssertEqual(absent.ignoredFields, [])
    }

    func testStreamOptionsUnknownInnerKeyIsRejectedWithDottedParam() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"stream":true,"stream_options":{"foo":true}}
        """
        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest,
            param: "stream_options.foo")
    }

    func testStreamOptionsWithoutStreamTrueIsRejected() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"stream":false,"stream_options":{"include_usage":true}}
        """
        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest,
            param: "stream_options")

        let defaultStreamBody = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"stream_options":{"include_usage":true}}
        """
        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(defaultStreamBody.utf8)),
            type: .invalidRequest,
            param: "stream_options")
    }

    // MARK: - Lenient unknown top-level keys (real SDK/vendor fields this server does not recognize)

    func testUnknownTopLevelFieldsAreAcceptedAndRecordedAsIgnored() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"unknown":true,"vendor_extension":"x"}
        """
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.ignoredFields, ["unknown:unknown", "unknown:vendor_extension"])
    }

    func testCompletionsUnknownTopLevelFieldIsAcceptedAndRecordedAsIgnored() throws {
        let body = """
        {"model":"qwen3-32b","prompt":"hi","unknown":true}
        """
        let request = try OpenAICompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.ignoredFields, ["unknown:unknown"])
    }

    // `semanticallyUnsupportedTopLevelKeys` change output semantics this server cannot honor, so they
    // stay a hard 400 even though other unknown top-level keys are now tolerated.
    func testDenylistedSemanticTopLevelKeysAreStillRejected() throws {
        let keys = ["audio", "modalities", "prediction", "web_search_options", "functions", "function_call"]
        for key in keys {
            let chatBody = #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"\#(key)":true}"#
            XCTAssertOpenAIError(
                try OpenAIChatCompletionRequest.decodeStrict(from: Data(chatBody.utf8)),
                type: .invalidRequest,
                param: key,
                file: #filePath,
                line: #line)

            let completionsBody = #"{"model":"qwen3-32b","prompt":"hi","\#(key)":true}"#
            XCTAssertOpenAIError(
                try OpenAICompletionRequest.decodeStrict(from: Data(completionsBody.utf8)),
                type: .invalidRequest,
                param: key,
                file: #filePath,
                line: #line)
        }
    }

    // Nested objects (a messages entry here) stay strict: an unknown key inside a message is still a
    // 400, unaffected by the top-level leniency policy.
    func testUnknownKeyInsideMessageIsStillRejected() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi","bogus":true}]}
        """
        XCTAssertOpenAIError(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest,
            param: "messages.bogus")
    }

    // A key shape outside `^[A-Za-z0-9_.-]{1,64}$` never has its literal name echoed into
    // `ignoredFields`: it collapses into a single `unknown:<invalid-key>` sentinel.
    func testInvalidShapedUnknownKeyNameCollapsesToSentinel() throws {
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"bad key!":true,"also,bad":1}
        """
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.ignoredFields, ["unknown:<invalid-key>"])
    }

    // More than 16 distinct unknown top-level keys are capped at 16 recorded entries plus one
    // trailing `unknown:<more>` sentinel.
    func testUnknownTopLevelKeysAreCappedAtSixteenPlusMoreSentinel() throws {
        let extraKeys = (1...20).map { "extra_\($0)" }
        let extraFields = extraKeys.map { #""\#($0)":true"# }.joined(separator: ",")
        let body = """
        {"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],\(extraFields)}
        """
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        let expected = (extraKeys.sorted().prefix(16).map { "unknown:\($0)" } + ["unknown:<more>"]).sorted()
        XCTAssertEqual(request.ignoredFields, expected)
    }

    func testErrorEnvelopeAlwaysCarriesOfficialShape() throws {
        let error = OpenAIServingError.invalidRequest("Unsupported field: tools", param: "tools")
        let data = try JSONEncoder.openAI.encode(OpenAIErrorEnvelope(error: error.openAIError))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let errorObject = try XCTUnwrap(object["error"] as? [String: Any])

        XCTAssertEqual(errorObject["type"] as? String, "invalid_request_error")
        XCTAssertEqual(errorObject["message"] as? String, "Unsupported field: tools")
        XCTAssertEqual(errorObject["param"] as? String, "tools")
        XCTAssertTrue(errorObject.keys.contains("code"))
        XCTAssertTrue(errorObject["code"] is NSNull)
    }

    func testNonStreamResponseObjectShape() throws {
        let response = OpenAIChatCompletionResponse(
            id: "chatcmpl-test",
            created: 1_775_000_000,
            model: "qwen3-32b",
            content: "hello",
            finishReason: .stop,
            usage: OpenAIChatUsage(promptTokens: 3, completionTokens: 1)
        )

        let data = try JSONEncoder.openAI.encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        let message = try XCTUnwrap(choices.first?["message"] as? [String: Any])
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])

        XCTAssertEqual(object["object"] as? String, "chat.completion")
        XCTAssertEqual(message["role"] as? String, "assistant")
        XCTAssertEqual(message["content"] as? String, "hello")
        XCTAssertEqual(choices.first?["finish_reason"] as? String, "stop")
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 1)
        XCTAssertEqual(usage["total_tokens"] as? Int, 4)
    }

    func testSSEChunksAreOrderedAndTerminateWithDone() throws {
        let role = OpenAIChatCompletionChunk(
            id: "chatcmpl-test",
            created: 1_775_000_000,
            model: "qwen3-32b",
            index: 0,
            delta: .init(role: "assistant", content: nil),
            finishReason: nil)
        let first = OpenAIChatCompletionChunk(
            id: "chatcmpl-test",
            created: 1_775_000_000,
            model: "qwen3-32b",
            index: 0,
            delta: .init(role: nil, content: "hel"),
            finishReason: nil)
        let second = OpenAIChatCompletionChunk(
            id: "chatcmpl-test",
            created: 1_775_000_000,
            model: "qwen3-32b",
            index: 0,
            delta: .init(role: nil, content: "lo"),
            finishReason: nil)
        let finish = OpenAIChatCompletionChunk(
            id: "chatcmpl-test",
            created: 1_775_000_000,
            model: "qwen3-32b",
            index: 0,
            delta: .init(role: nil, content: nil),
            finishReason: .stop)
        let events = try [
            role.sseEvent(),
            first.sseEvent(),
            second.sseEvent(),
            finish.sseEvent(),
            OpenAIChatCompletionChunk.doneSSEEvent,
        ]

        XCTAssertEqual(events.count, 5)
        XCTAssertTrue(events[0].contains(#""object":"chat.completion.chunk""#))
        XCTAssertTrue(events[0].contains(#""role":"assistant""#))
        XCTAssertTrue(events[1].contains(#""content":"hel""#))
        XCTAssertTrue(events[2].contains(#""content":"lo""#))
        XCTAssertTrue(events[3].contains(#""finish_reason":"stop""#))
        XCTAssertEqual(events[4], "data: [DONE]\n\n")
    }

    // MARK: - OpenAICompletionRequest (legacy /v1/completions)

    func testCompletionRequestDecodesStringPrompt() throws {
        let body = """
        {"model":"qwen3-32b","prompt":"Once upon a time"}
        """
        let request = try OpenAICompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.model, "qwen3-32b")
        XCTAssertEqual(request.prompt, "Once upon a time")
    }

    func testCompletionRequestDecodesSingleElementArrayPrompt() throws {
        let body = """
        {"model":"qwen3-32b","prompt":["Once upon a time"]}
        """
        let request = try OpenAICompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.prompt, "Once upon a time")
    }

    func testCompletionRequestRejectsUnsupportedPromptAndFieldShapes() throws {
        let cases: [(String, String)] = [
            (#"{"model":"qwen3-32b","prompt":[1,2,3]}"#, "prompt"),
            (#"{"model":"qwen3-32b","prompt":["a","b"]}"#, "prompt"),
            (#"{"model":"qwen3-32b","prompt":""}"#, "prompt"),
            (#"{"model":"qwen3-32b","prompt":"hi","n":2}"#, "n"),
            (#"{"model":"qwen3-32b","prompt":"hi","echo":true}"#, "echo"),
            (#"{"model":"qwen3-32b","prompt":"hi","suffix":"x"}"#, "suffix"),
            (#"{"model":"qwen3-32b","prompt":"hi","best_of":2}"#, "best_of"),
            (#"{"model":"qwen3-32b","prompt":"hi","logprobs":6}"#, "logprobs"),
            (#"{"model":"qwen3-32b","prompt":"hi","logprobs":-1}"#, "logprobs"),
        ]
        for (body, param) in cases {
            XCTAssertOpenAIError(
                try OpenAICompletionRequest.decodeStrict(from: Data(body.utf8)),
                type: .invalidRequest,
                param: param,
                file: #filePath,
                line: #line)
        }
    }

    // The legacy route decodes through OpenAICompletionRequest, not the chat decoder — every
    // chat-only key is an unknown field here. Under the lenient-unknown-top-level-key policy this is
    // no longer a 400: each is silently accepted and recorded as "unknown:<key>" in `ignoredFields`,
    // exactly like any other unrecognized top-level key (none of these names are on the
    // `semanticallyUnsupportedTopLevelKeys` denylist).
    func testCompletionRequestAcceptsChatOnlyFieldsAsIgnoredUnknownKeys() throws {
        let cases: [(String, String)] = [
            (#"{"model":"qwen3-32b","prompt":"hi","messages":[]}"#, "messages"),
            (#"{"model":"qwen3-32b","prompt":"hi","tools":[]}"#, "tools"),
            (#"{"model":"qwen3-32b","prompt":"hi","enable_thinking":true}"#, "enable_thinking"),
            (#"{"model":"qwen3-32b","prompt":"hi","chat_template_kwargs":{}}"#, "chat_template_kwargs"),
            (#"{"model":"qwen3-32b","prompt":"hi","max_completion_tokens":16}"#, "max_completion_tokens"),
        ]
        for (body, key) in cases {
            let request = try OpenAICompletionRequest.decodeStrict(from: Data(body.utf8))
            XCTAssertEqual(request.ignoredFields, ["unknown:\(key)"], body)
        }
    }

    // Integer `logprobs` 0...5 is a real, honored request: `0` means "the sampled token's logprob
    // only, no alternatives" — a MEANINGFUL, distinct value, never "logprobs off". Only a value
    // outside 0...5 fails closed; `null`/absent means not requested at all.
    func testCompletionRequestAcceptsInRangeIntegerLogprobs() throws {
        for value in [0, 1, 5] {
            let body = #"{"model":"qwen3-32b","prompt":"hi","logprobs":\#(value)}"#
            let request = try OpenAICompletionRequest.decodeStrict(from: Data(body.utf8))
            XCTAssertEqual(request.logprobsRequest, .completions(topLogprobs: value))
            XCTAssertEqual(request.ignoredFields, [])
        }

        let nullBody = #"{"model":"qwen3-32b","prompt":"hi","logprobs":null}"#
        let nullRequest = try OpenAICompletionRequest.decodeStrict(from: Data(nullBody.utf8))
        XCTAssertEqual(nullRequest.ignoredFields, [])
        XCTAssertNil(nullRequest.logprobsRequest)
    }

    func testCompletionRequestRejectsOutOfRangeIntegerLogprobs() throws {
        for value in [-1, 6] {
            let body = #"{"model":"qwen3-32b","prompt":"hi","logprobs":\#(value)}"#
            XCTAssertOpenAIError(
                try OpenAICompletionRequest.decodeStrict(from: Data(body.utf8)),
                type: .invalidRequest,
                param: "logprobs",
                file: #filePath,
                line: #line)
        }
    }

    func testCompletionRequestConversionCarriesLogprobsRequest() throws {
        let body = #"{"model":"qwen3-32b","prompt":"hi","logprobs":3}"#
        let completionRequest = try OpenAICompletionRequest.decodeStrict(from: Data(body.utf8))
        let chatRequest = completionRequest.asChatCompletionRequest()
        XCTAssertEqual(chatRequest.logprobsRequest, .completions(topLogprobs: 3))
    }

    func testCompletionRequestConversionCarriesFieldsAndSetsRawTextPromptInput() throws {
        let body = """
        {"model":"qwen3-32b","prompt":"Once upon a time","temperature":0.5,"top_p":0.9,
         "max_tokens":16,"stop":["END"],"seed":7,"stream":true}
        """
        let completionRequest = try OpenAICompletionRequest.decodeStrict(from: Data(body.utf8))
        let chatRequest = completionRequest.asChatCompletionRequest()

        XCTAssertEqual(chatRequest.model, "qwen3-32b")
        XCTAssertEqual(chatRequest.temperature, 0.5)
        XCTAssertEqual(chatRequest.topP, 0.9)
        XCTAssertEqual(chatRequest.maxCompletionTokens, 16)
        XCTAssertEqual(chatRequest.stop, ["END"])
        XCTAssertEqual(chatRequest.seed, 7)
        XCTAssertTrue(chatRequest.stream)
        XCTAssertTrue(chatRequest.messages.isEmpty)
        XCTAssertTrue(chatRequest.tools.isEmpty)
        XCTAssertNil(chatRequest.enableThinking)
        XCTAssertNil(chatRequest.reasoningEffort)
        XCTAssertEqual(chatRequest.promptInput, .rawText("Once upon a time"))
    }

    // The wire decoder for CHAT never RECOGNIZES a `promptInput` or `prompt` key: neither is in the
    // chat decoder's allowed-keys set, so both fall through the lenient-unknown-top-level-key policy
    // — accepted and recorded as ignored, but never read into `promptInput`/messages. This still
    // proves `.rawText` cannot be set by any chat JSON payload; a client sending either key gets
    // `.chat` regardless.
    func testChatDecodeIgnoresPromptInputAndPromptAsUnknownKeysNotRawText() throws {
        let promptInputRequest = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(
                #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"promptInput":"chat"}"#
                    .utf8))
        XCTAssertEqual(promptInputRequest.ignoredFields, ["unknown:promptInput"])
        XCTAssertEqual(promptInputRequest.promptInput, .chat)

        let promptRequest = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(
                #"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}],"prompt":"hi"}"#
                    .utf8))
        XCTAssertEqual(promptRequest.ignoredFields, ["unknown:prompt"])
        XCTAssertEqual(promptRequest.promptInput, .chat)

        // Every chat-decoded request defaults to `.chat`.
        let plain = try OpenAIChatCompletionRequest.decodeStrict(
            from: Data(#"{"model":"qwen3-32b","messages":[{"role":"user","content":"Hi"}]}"#.utf8))
        XCTAssertEqual(plain.promptInput, .chat)
    }

    func testSSETerminalChunkCanCarryExactUsage() throws {
        let finish = OpenAIChatCompletionChunk(
            id: "chatcmpl-test",
            created: 1_775_000_000,
            model: "qwen3-32b",
            index: 0,
            delta: .init(role: nil, content: nil),
            finishReason: .length,
            usage: OpenAIChatUsage(promptTokens: 3, completionTokens: 2))

        let event = try finish.sseEvent()
        let payload = try XCTUnwrap(
            event
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .split(separator: " ", maxSplits: 1)
                .last)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any])
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])

        XCTAssertEqual(choices.first?["finish_reason"] as? String, "length")
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 3)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 2)
        XCTAssertEqual(usage["total_tokens"] as? Int, 5)
    }
}

private func XCTAssertOpenAIError<T>(
    _ expression: @autoclosure () throws -> T,
    type: OpenAIErrorType,
    param: String?,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    do {
        _ = try expression()
        XCTFail("Expected OpenAIServingError", file: file, line: line)
    } catch let error as OpenAIServingError {
        XCTAssertEqual(error.openAIError.type, type, file: file, line: line)
        XCTAssertEqual(error.openAIError.param, param, file: file, line: line)
    } catch {
        XCTFail("Expected OpenAIServingError, got \(error)", file: file, line: line)
    }
}
