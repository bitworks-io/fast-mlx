import XCTest

@testable import ServingCore

/// The gate functions that decide whether the SSE path separates streamed reasoning from the visible
/// answer, and whether the legacy tool-thinking workaround applies. `servingThinksByDefault` and
/// `servingDisablesThinkingWhenToolsActive` are the FAMILY classifiers — keyed on BOTH the decoder
/// route AND the model's `model_type`, not the route alone, because `.nativeHeterogeneous` is a
/// cache-shape route that can carry more than one family. `servingSeparatesReasoning` folds
/// `servingThinksByDefault`'s result with the per-request resolved thinking flag into the load-bearing
/// gate the streaming handler consumes.
///
/// The truth table below is the correctness contract that structurally avoids the answer-loss class that
/// reverted the earlier family-blind wiring (commit `3a806f6`): a stream is only ever routed through the
/// splitter when the family is KNOWN to reason AND thinking was not resolved OFF, so content bytes on a
/// non-thinking stream can never enter the splitter no matter what they look like. The family matrix
/// below extends that discipline one level up: admitting an unattested family onto `.nativeHeterogeneous`
/// must not silently inherit qwen3_5's attested shape.
final class StreamingReasoningPolicyTests: XCTestCase {
    func testThinksByDefaultIsTrueOnlyForNativeHeterogeneousQwen35() {
        // The one live-attested combination: qwen3_5 hybrid family (Qwen3.5/3.6/3.8) on the
        // `.nativeHeterogeneous` route. Its live output shape (93e606a) is no-opener reasoning-first.
        XCTAssertTrue(servingThinksByDefault(route: .nativeHeterogeneous, modelType: "qwen3_5"))
    }

    func testThinksByDefaultIsFalseForEveryOtherRouteFamilyCombination() {
        // .compiled (dense) stays conservative false regardless of modelType — its streamed shape has
        // never been live-captured.
        XCTAssertFalse(servingThinksByDefault(route: .compiled, modelType: "qwen3_5"))
        XCTAssertFalse(servingThinksByDefault(route: .compiled, modelType: "another_family"))
        XCTAssertFalse(servingThinksByDefault(route: .compiled, modelType: nil))
        // .nativeHeterogeneous with a DIFFERENT or unknown family must NOT inherit qwen3_5's attested
        // shape — this is the exact answer-loss class a route-only gate would reintroduce.
        XCTAssertFalse(servingThinksByDefault(route: .nativeHeterogeneous, modelType: "another_family"))
        XCTAssertFalse(servingThinksByDefault(route: .nativeHeterogeneous, modelType: nil))
    }

    func testDisablesThinkingWhenToolsActiveIsFalseOnlyForNativeHeterogeneousQwen35() {
        // The one live-attested combination is trained to think AND call tools, so it respects the
        // template default — the legacy workaround must be OFF.
        XCTAssertFalse(
            servingDisablesThinkingWhenToolsActive(route: .nativeHeterogeneous, modelType: "qwen3_5"))
    }

    func testDisablesThinkingWhenToolsActiveIsTrueForEveryOtherRouteFamilyCombination() {
        // .compiled (dense) always keeps the legacy workaround (QwenLM/Qwen3 #1817), regardless of
        // modelType.
        XCTAssertTrue(
            servingDisablesThinkingWhenToolsActive(route: .compiled, modelType: "qwen3_5"))
        XCTAssertTrue(
            servingDisablesThinkingWhenToolsActive(route: .compiled, modelType: "another_family"))
        XCTAssertTrue(
            servingDisablesThinkingWhenToolsActive(route: .compiled, modelType: nil))
        // .nativeHeterogeneous with a DIFFERENT or unknown family conservatively keeps the workaround
        // ON — avoids the thinking-on latency trap of assuming an unattested family tolerates
        // thinking-with-tools.
        XCTAssertTrue(
            servingDisablesThinkingWhenToolsActive(route: .nativeHeterogeneous, modelType: "another_family"))
        XCTAssertTrue(
            servingDisablesThinkingWhenToolsActive(route: .nativeHeterogeneous, modelType: nil))
    }

    func testSeparatesReasoningTruthTable() {
        // Family thinks by default: nil (client omitted flag → template default is thinking) and explicit
        // true both separate; explicit false does not (a closed empty `<think></think>` is injected into
        // the prompt, nothing is generated to split).
        XCTAssertTrue(servingSeparatesReasoning(thinksByDefault: true, resolvedEnableThinking: nil))
        XCTAssertTrue(servingSeparatesReasoning(thinksByDefault: true, resolvedEnableThinking: true))
        XCTAssertFalse(servingSeparatesReasoning(thinksByDefault: true, resolvedEnableThinking: false))

        // Family does NOT think by default: never separate, whatever the request asked. This is what kills
        // the `nil != false` trap — a non-thinking model with an omitted flag stays passthrough.
        XCTAssertFalse(servingSeparatesReasoning(thinksByDefault: false, resolvedEnableThinking: nil))
        XCTAssertFalse(servingSeparatesReasoning(thinksByDefault: false, resolvedEnableThinking: true))
        XCTAssertFalse(servingSeparatesReasoning(thinksByDefault: false, resolvedEnableThinking: false))
    }
}
