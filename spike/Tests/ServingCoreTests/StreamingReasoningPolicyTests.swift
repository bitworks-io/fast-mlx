import XCTest

@testable import ServingCore

/// The gate functions that decide whether the SSE path separates streamed reasoning from the visible
/// answer, and whether the legacy tool-thinking workaround applies. `servingThinksByDefault` and
/// `servingDisablesThinkingWhenToolsActive` are the FAMILY classifiers — keyed on the decoder route,
/// the model's `model_type`, AND whether the loaded checkpoint's own chat template attests the
/// `<think>`/`</think>` markers the streaming splitter hardcodes. All three are REQUIRED: the route
/// alone does not identify a family (`.nativeHeterogeneous` is a cache-shape route that can carry more
/// than one family), and the family string alone does not prove a specific checkpoint's template
/// actually emits the attested markers. `servingSeparatesReasoning` folds `servingThinksByDefault`'s
/// result with the per-request resolved thinking flag into the load-bearing gate the streaming handler
/// consumes.
///
/// The truth table below is the correctness contract that structurally avoids the answer-loss class that
/// reverted the earlier family-blind wiring (commit `3a806f6`): a stream is only ever routed through the
/// splitter when the family is KNOWN to reason AND thinking was not resolved OFF, so content bytes on a
/// non-thinking stream can never enter the splitter no matter what they look like. The family matrix
/// below extends that discipline one level up: admitting an unattested family onto `.nativeHeterogeneous`
/// must not silently inherit an already-attested family's shape, and admitting a template-unattested
/// checkpoint of an otherwise-attested family (e.g. qwen3_5, qwen4_exp) must not silently inherit that
/// family's template-derived shape either.
final class StreamingReasoningPolicyTests: XCTestCase {
    func testThinksByDefaultIsTrueOnlyForNativeHeterogeneousQwen35WithAttestingTemplate() {
        // The qwen3_5 hybrid family (Qwen3.5/3.6/3.8) on the `.nativeHeterogeneous` route, with a
        // template that attests the markers the streaming splitter hardcodes. Live output shape
        // (93e606a) is no-opener reasoning-first.
        XCTAssertTrue(
            servingThinksByDefault(
                route: .nativeHeterogeneous, modelType: "qwen3_5", templateAttestsThinkMarkers: true))
    }

    /// Regression lock: an attested family whose loaded template does NOT attest the markers must
    /// fall back to today's passthrough, not separation — the family string alone is necessary but
    /// not sufficient.
    func testThinksByDefaultIsFalseForQwen35WhenTemplateDoesNotAttestMarkers() {
        XCTAssertFalse(
            servingThinksByDefault(
                route: .nativeHeterogeneous, modelType: "qwen3_5", templateAttestsThinkMarkers: false))
    }

    /// Positive: Flash Next (`qwen4_exp`) on `.nativeHeterogeneous` with an attesting template — the
    /// live defect this cycle fixes (captured at harness `c771346e` against `flashnext-oq4-mtp`:
    /// streaming never separated, zero `reasoning_content` deltas; non-streaming split cleanly).
    func testThinksByDefaultIsTrueForNativeHeterogeneousFlashNextWithAttestingTemplate() {
        XCTAssertTrue(
            servingThinksByDefault(
                route: .nativeHeterogeneous, modelType: "qwen4_exp", templateAttestsThinkMarkers: true))
    }

    /// Discriminating negative: `.compiled` (dense) never separates regardless of family or template
    /// attestation — the dense route's streamed shape has never been live-captured.
    func testThinksByDefaultIsFalseForCompiledFlashNextEvenWithAttestingTemplate() {
        XCTAssertFalse(
            servingThinksByDefault(
                route: .compiled, modelType: "qwen4_exp", templateAttestsThinkMarkers: true))
    }

    /// Discriminating negative: `qwen4_exp` on `.nativeHeterogeneous` whose loaded template does NOT
    /// attest the markers must stay passthrough — a family string is not proof for a specific
    /// checkpoint's template.
    func testThinksByDefaultIsFalseForNativeHeterogeneousFlashNextWhenTemplateDoesNotAttestMarkers() {
        XCTAssertFalse(
            servingThinksByDefault(
                route: .nativeHeterogeneous, modelType: "qwen4_exp", templateAttestsThinkMarkers: false))
    }

    func testThinksByDefaultIsFalseForEveryOtherRouteFamilyCombination() {
        // .compiled (dense) stays conservative false regardless of modelType or template attestation
        // — its streamed shape has never been live-captured.
        XCTAssertFalse(
            servingThinksByDefault(route: .compiled, modelType: "qwen3_5", templateAttestsThinkMarkers: true))
        XCTAssertFalse(
            servingThinksByDefault(
                route: .compiled, modelType: "another_family", templateAttestsThinkMarkers: true))
        XCTAssertFalse(
            servingThinksByDefault(route: .compiled, modelType: nil, templateAttestsThinkMarkers: true))
        // .nativeHeterogeneous with a DIFFERENT or unknown family must NOT inherit an attested family's
        // shape — this is the exact answer-loss class a route-only gate would reintroduce.
        XCTAssertFalse(
            servingThinksByDefault(
                route: .nativeHeterogeneous, modelType: "another_family", templateAttestsThinkMarkers: true))
        XCTAssertFalse(
            servingThinksByDefault(
                route: .nativeHeterogeneous, modelType: nil, templateAttestsThinkMarkers: true))
    }

    func testDisablesThinkingWhenToolsActiveIsFalseOnlyForAttestedNativeHeterogeneousFamilies() {
        // Every live-attested combination is trained to think AND call tools, so it respects the
        // template default — the legacy workaround must be OFF.
        XCTAssertFalse(
            servingDisablesThinkingWhenToolsActive(
                route: .nativeHeterogeneous, modelType: "qwen3_5", templateAttestsThinkMarkers: true))
        // qwen4_exp (Flash Next): live-verified this cycle — thinking ON with tools attached still
        // produced a well-formed `get_weather{"city":"Paris"}` tool call, `finish_reason=tool_calls`,
        // in both streaming and non-streaming.
        XCTAssertFalse(
            servingDisablesThinkingWhenToolsActive(
                route: .nativeHeterogeneous, modelType: "qwen4_exp", templateAttestsThinkMarkers: true))
    }

    /// An attested family (`qwen4_exp`) whose loaded template does NOT attest the markers must keep
    /// the legacy tools workaround ON — the coupling with `servingThinksByDefault` must hold exactly.
    func testDisablesThinkingWhenToolsActiveIsTrueForFlashNextWhenTemplateDoesNotAttestMarkers() {
        XCTAssertTrue(
            servingDisablesThinkingWhenToolsActive(
                route: .nativeHeterogeneous, modelType: "qwen4_exp", templateAttestsThinkMarkers: false))
    }

    func testDisablesThinkingWhenToolsActiveIsTrueForEveryOtherRouteFamilyCombination() {
        // .compiled (dense) always keeps the legacy workaround (QwenLM/Qwen3 #1817), regardless of
        // modelType or template attestation.
        XCTAssertTrue(
            servingDisablesThinkingWhenToolsActive(
                route: .compiled, modelType: "qwen3_5", templateAttestsThinkMarkers: true))
        XCTAssertTrue(
            servingDisablesThinkingWhenToolsActive(
                route: .compiled, modelType: "another_family", templateAttestsThinkMarkers: true))
        XCTAssertTrue(
            servingDisablesThinkingWhenToolsActive(
                route: .compiled, modelType: nil, templateAttestsThinkMarkers: true))
        // .nativeHeterogeneous with a DIFFERENT or unknown family conservatively keeps the workaround
        // ON — avoids the thinking-on latency trap of assuming an unattested family tolerates
        // thinking-with-tools.
        XCTAssertTrue(
            servingDisablesThinkingWhenToolsActive(
                route: .nativeHeterogeneous, modelType: "another_family", templateAttestsThinkMarkers: true))
        XCTAssertTrue(
            servingDisablesThinkingWhenToolsActive(
                route: .nativeHeterogeneous, modelType: nil, templateAttestsThinkMarkers: true))
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
