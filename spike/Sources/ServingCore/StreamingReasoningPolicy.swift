import Foundation

/// Family-level classifier: does this decoder route's model emit its reasoning block by DEFAULT, with no
/// leading `<think>` opener (reasoning starts at token 0)?
///
/// The `.nativeHeterogeneous` route is the agentic qwen3_5 hybrid family (Qwen3.5/3.6/3.8). Its
/// LIVE-verified streamed shape — captured in this repo's own history at commit `93e606a` — is
/// no-opener: the model begins emitting reasoning immediately and closes it with `</think>` before the
/// answer (`<reasoning>…</think>…<answer>`), never a leading `<think>` that a generation-side sniffer
/// could key on. That no-opener shape is exactly why the generation-side `StreamingReasoningGate` (which
/// fires only on a leading `<think>`) is inert for this family, and why the honest gate is a family
/// classifier rather than an output sniffer.
///
/// The `.compiled` (dense) route stays conservative `false`: its streamed reasoning shape has not been
/// live-captured, so it keeps today's byte-identical passthrough until one attests it. Flipping it is a
/// recorded handoff item, not a guess. This mirrors the existing family predicate at
/// `MLXScalarServing.swift` (`disableThinkingWhenToolsActive = decoderRoute == .compiled`).
public func servingThinksByDefault(route: ScalarServingDecoderRoute) -> Bool {
    switch route {
    case .nativeHeterogeneous:
        return true
    case .compiled:
        return false
    }
}

/// The per-request, load-bearing gate the streaming SSE handler consumes: separate streamed reasoning
/// from the visible answer ONLY when the family reasons by default AND thinking was not resolved OFF for
/// this request.
///
/// `resolvedEnableThinking` is the SAME resolved value the codec renders the prompt from
/// (`OpenAIChatCompletionRequest.resolvedEnableThinking(disableThinkingWhenToolsActive:)`): `nil` means
/// "template default", which for a thinks-by-default family IS thinking; explicit `false` means a closed
/// empty `<think></think>` was injected into the prompt so nothing is generated to split.
///
/// `thinksByDefault` DOMINATES the conjunction, which is what structurally prevents the answer-loss class
/// (commit `3a806f6`): a `false`-family stream is never routed through the splitter regardless of what its
/// content bytes look like, so a non-thinking answer that happens to begin with the literal text
/// `<think>` and never closes it can never be mislabeled as reasoning. Never ship the
/// `resolvedEnableThinking != false` half standalone — on its own it mislabels non-thinking families
/// (omitted flag → `nil != false` → true).
public func servingSeparatesReasoning(
    thinksByDefault: Bool,
    resolvedEnableThinking: Bool?
) -> Bool {
    thinksByDefault && (resolvedEnableThinking != false)
}
