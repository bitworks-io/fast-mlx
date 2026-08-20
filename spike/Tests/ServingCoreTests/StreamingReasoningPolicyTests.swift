import XCTest

@testable import ServingCore

/// The two pure gate functions that decide whether the SSE path separates streamed reasoning from the
/// visible answer. `servingThinksByDefault` is the FAMILY (route) classifier — does the model emit its
/// reasoning block from token 0 with no leading `<think>` opener. `servingSeparatesReasoning` folds that
/// with the per-request resolved thinking flag into the load-bearing gate the streaming handler consumes.
///
/// The truth table below is the correctness contract that structurally avoids the answer-loss class that
/// reverted the earlier family-blind wiring (commit `3a806f6`): a stream is only ever routed through the
/// splitter when the family is KNOWN to reason AND thinking was not resolved OFF, so content bytes on a
/// non-thinking stream can never enter the splitter no matter what they look like.
final class StreamingReasoningPolicyTests: XCTestCase {
    func testThinksByDefaultIsTrueOnlyForTheNativeHeterogeneousFamily() {
        // The qwen3_5 hybrid family (Qwen3.5/3.6/3.8) is the `.nativeHeterogeneous` route; its live output
        // shape (93e606a) is no-opener reasoning-first. Dense/compiled stays conservative false until a
        // live capture attests its streamed shape.
        XCTAssertTrue(servingThinksByDefault(route: .nativeHeterogeneous))
        XCTAssertFalse(servingThinksByDefault(route: .compiled))
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
