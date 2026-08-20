import XCTest

@testable import ServingCore

/// T1 pure-Swift contract tests for OpenAI tool calling (no MLX, runs off-GPU).
/// Each test names the acceptance criterion it proves (design §1 / §9).
final class OpenAIToolCallingTests: XCTestCase {
    // AC1: tools + tool_choice are accepted and parsed.
    func testAcceptsToolsAndToolChoiceVariants() throws {
        let toolsJSON = #"""
        {
          "model":"qwen3-32b",
          "messages":[{"role":"user","content":"weather?"}],
          "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],
          "tool_choice":"auto"
        }
        """#
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(toolsJSON.utf8))
        XCTAssertEqual(request.tools.count, 1)
        XCTAssertEqual(request.tools.first?.name, "get_weather")
        XCTAssertEqual(request.tools.first?.description, "Get weather")
        XCTAssertEqual(request.toolChoice, .auto)

        // Default is .auto when tools present, .none when absent.
        let noChoice = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}]}"#
        XCTAssertEqual(try OpenAIChatCompletionRequest.decodeStrict(from: Data(noChoice.utf8)).toolChoice, .auto)
        let noTools = #"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#
        XCTAssertEqual(try OpenAIChatCompletionRequest.decodeStrict(from: Data(noTools.utf8)).toolChoice, .none)

        // "required" and forced-function object.
        let required = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"required"}"#
        XCTAssertEqual(try OpenAIChatCompletionRequest.decodeStrict(from: Data(required.utf8)).toolChoice, .required)
        let named = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":{"type":"function","function":{"name":"f"}}}"#
        XCTAssertEqual(try OpenAIChatCompletionRequest.decodeStrict(from: Data(named.utf8)).toolChoice, .function("f"))
        // "none" is honored.
        let none = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"none"}"#
        XCTAssertEqual(try OpenAIChatCompletionRequest.decodeStrict(from: Data(none.utf8)).toolChoice, OpenAIToolChoice.none)
    }

    // AC1 + AC3: the exact Concierge multi-turn wire decodes (system, user, assistant tool_calls
    // with null content, tool result with tool_call_id). See the acceptance fixture doc.
    func testDecodesConciergeMultiTurnToolHistory() throws {
        let body = #"""
        {
          "model":"qwen3-32b",
          "messages":[
            {"role":"system","content":"You are a concierge."},
            {"role":"user","content":"do you have the RTX 6000 Ada in stock?"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"call_1_0","type":"function","function":{"name":"get_product","arguments":"{\"query\":\"RTX 6000 Ada\"}"}}]},
            {"role":"tool","tool_call_id":"call_1_0","content":"{\"products\":[]}"}
          ],
          "tools":[{"type":"function","function":{"name":"get_product","description":"Look up a product","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}}],
          "max_tokens":384
        }
        """#
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.messages.map(\.role), [.system, .user, .assistant, .tool])

        let assistant = request.messages[2]
        XCTAssertEqual(assistant.text, "")  // null content becomes empty
        XCTAssertEqual(assistant.toolCalls.count, 1)
        XCTAssertEqual(assistant.toolCalls.first?.id, "call_1_0")
        XCTAssertEqual(assistant.toolCalls.first?.function.name, "get_product")
        XCTAssertEqual(assistant.toolCalls.first?.function.arguments, #"{"query":"RTX 6000 Ada"}"#)

        let toolResult = request.messages[3]
        XCTAssertEqual(toolResult.toolCallId, "call_1_0")  // round-trips the emitted id
        XCTAssertEqual(toolResult.text, #"{"products":[]}"#)
        XCTAssertEqual(request.maxCompletionTokens, 384)  // max_tokens alias still accepted
    }

    // AC5: enable_thinking + parallel_tool_calls parse.
    func testAcceptsThinkingAndParallelFlags() throws {
        let body = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}],"enable_thinking":false,"parallel_tool_calls":true}"#
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.enableThinking, false)
        XCTAssertEqual(request.parallelToolCalls, true)
    }

    // AC2: response with tool_calls encodes to the exact wire the client parses.
    func testToolCallResponseWireShape() throws {
        let response = OpenAIChatCompletionResponse(
            id: "chatcmpl-x",
            created: 1_775_000_000,
            model: "qwen3-32b",
            content: nil,
            finishReason: .toolCalls,
            usage: OpenAIChatUsage(promptTokens: 10, completionTokens: 5),
            toolCalls: [
                OpenAIToolCall(
                    id: "call_x",
                    function: .init(name: "get_product", arguments: #"{"query":"RTX 6000 Ada"}"#))
            ])
        let data = try JSONEncoder.openAI.encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        let choice = try XCTUnwrap(choices.first)
        let message = try XCTUnwrap(choice["message"] as? [String: Any])

        XCTAssertEqual(choice["finish_reason"] as? String, "tool_calls")
        XCTAssertTrue(message["content"] is NSNull, "content must be explicit null on a tool-only turn")
        let toolCalls = try XCTUnwrap(message["tool_calls"] as? [[String: Any]])
        let call = try XCTUnwrap(toolCalls.first)
        XCTAssertEqual(call["id"] as? String, "call_x")
        XCTAssertEqual(call["type"] as? String, "function")
        let function = try XCTUnwrap(call["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, "get_product")
        // arguments MUST be a JSON *string* the client can JSON.parse.
        let argumentsString = try XCTUnwrap(function["arguments"] as? String)
        let parsed = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(argumentsString.utf8)) as? [String: Any])
        XCTAssertEqual(parsed["query"] as? String, "RTX 6000 Ada")
    }

    // AC6: streaming tool-call delta chunk carries index/id/type/function fragments.
    func testStreamingToolCallDeltaWireShape() throws {
        let head = OpenAIChatCompletionChunk(
            id: "chatcmpl-x", created: 1, model: "m", index: 0,
            delta: .init(
                role: nil, content: nil,
                toolCalls: [OpenAIToolCallDelta(index: 0, id: "call_x", type: "function", function: .init(name: "get_product", arguments: nil))]),
            finishReason: nil)
        let frag = OpenAIChatCompletionChunk(
            id: "chatcmpl-x", created: 1, model: "m", index: 0,
            delta: .init(
                role: nil, content: nil,
                toolCalls: [OpenAIToolCallDelta(index: 0, id: nil, type: nil, function: .init(name: nil, arguments: #"{"query":"#))]),
            finishReason: nil)
        let headEvent = try head.sseEvent()
        let fragEvent = try frag.sseEvent()
        XCTAssertTrue(headEvent.contains(#""tool_calls""#))
        XCTAssertTrue(headEvent.contains(#""index":0"#))
        XCTAssertTrue(headEvent.contains(#""name":"get_product""#))
        XCTAssertTrue(fragEvent.contains(#""arguments":"{\"query\":""#))
    }

    // Supporting: ServingJSONValue preserves structure and serializes to a JSON string.
    func testServingJSONValueRoundTripAndStringify() throws {
        let source: [String: Any] = ["a": 1, "b": ["c": "x"], "d": [true, 2.5]]
        let value = ServingJSONValue(foundation: source)
        guard case .object(let object) = value else { return XCTFail("expected object") }
        XCTAssertEqual(object["a"], .int(1))
        let json = ServingJSONValue.object(["query": .string("RTX 6000 Ada")]).jsonString()
        XCTAssertEqual(json, #"{"query":"RTX 6000 Ada"}"#)
    }

    // AC4 / M3a: tool_choice resolution restricts the callable tool set.
    func testActiveToolsHonorsToolChoice() throws {
        let bothTools = #"[{"type":"function","function":{"name":"get_product"}},{"type":"function","function":{"name":"verify_identity"}}]"#

        // Default (auto when tools present): all tools active.
        let auto = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":\#(bothTools)}"#
        XCTAssertEqual(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(auto.utf8)).activeTools.map(\.name),
            ["get_product", "verify_identity"])

        // Named choice: only that tool is callable.
        let named = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":\#(bothTools),"tool_choice":{"type":"function","function":{"name":"verify_identity"}}}"#
        XCTAssertEqual(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(named.utf8)).activeTools.map(\.name),
            ["verify_identity"])

        // none: no tools callable even when declared.
        let none = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":\#(bothTools),"tool_choice":"none"}"#
        XCTAssertTrue(
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(none.utf8)).activeTools.isEmpty)
    }

    // Tool-thinking policy is family-aware: the LEGACY dense-Qwen3 policy forces enable_thinking:false
    // when tools are active (QwenLM/Qwen3 #1817); the MODERN agentic policy (qwen3_5 / Qwen3.5/3.6/3.8)
    // respects the model's template default even with tools. An explicit client value always wins.
    func testResolvedEnableThinkingHonorsFamilyPolicyAndClientOverride() throws {
        func request(_ json: String) throws -> OpenAIChatCompletionRequest {
            try OpenAIChatCompletionRequest.decodeStrict(from: Data(json.utf8))
        }
        let withTools = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}]}"#
        let explicitTrue = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}],"enable_thinking":true}"#
        let noTools = #"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#
        let choiceNone = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"none"}"#

        // LEGACY (dense Qwen3): tools force thinking off.
        XCTAssertEqual(
            try request(withTools).resolvedEnableThinking(disableThinkingWhenToolsActive: true), false)
        // MODERN (qwen3_5 / Qwen3.8): respect the template default (nil = thinking on) even with tools.
        XCTAssertNil(
            try request(withTools).resolvedEnableThinking(disableThinkingWhenToolsActive: false))
        // Explicit client value wins under either policy.
        XCTAssertEqual(
            try request(explicitTrue).resolvedEnableThinking(disableThinkingWhenToolsActive: true), true)
        XCTAssertEqual(
            try request(explicitTrue).resolvedEnableThinking(disableThinkingWhenToolsActive: false), true)
        // No active tools -> template default (nil) under either policy.
        XCTAssertNil(
            try request(noTools).resolvedEnableThinking(disableThinkingWhenToolsActive: true))
        XCTAssertNil(
            try request(choiceNone).resolvedEnableThinking(disableThinkingWhenToolsActive: true))
    }
}
