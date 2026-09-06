import Foundation

/// Model-type strings whose streamed reasoning shape has been LIVE-attested on the
/// `.nativeHeterogeneous` route: no leading `<think>` opener, reasoning emitted from token 0
/// (captured in this repo's own history at commit `93e606a`). The route itself is NOT a reliable
/// family classifier — it is the recurrent/hybrid *cache shape*, and a second family can share it
/// without sharing the attested streamed output shape. Keeping the attested set as a named,
/// documented constant (rather than a bare `route == .nativeHeterogeneous` check) means admitting a
/// new family onto this route is a reviewed decision to add it here, not an implicit inheritance of
/// qwen3_5's behavior.
private let attestedNativeHeterogeneousThinkingFamilies: Set<String> = ["qwen3_5"]

/// Family-level classifier: does this model emit its reasoning block by DEFAULT, with no leading
/// `<think>` opener (reasoning starts at token 0)?
///
/// The gate is keyed on BOTH the decoder route AND the model's `model_type` (from `config.json`),
/// not the route alone. `.nativeHeterogeneous` today only carries the agentic qwen3_5 hybrid family
/// (Qwen3.5/3.6/3.8), whose LIVE-verified streamed shape is no-opener: the model begins emitting
/// reasoning immediately and closes it with `</think>` before the answer
/// (`<reasoning>…</think>…<answer>`), never a leading `<think>` that a generation-side sniffer could
/// key on. That no-opener shape is exactly why the generation-side `StreamingReasoningGate` (which
/// fires only on a leading `<think>`) is inert for this family, and why the honest gate is a family
/// classifier rather than an output sniffer.
///
/// A future family admitted onto `.nativeHeterogeneous` (same cache shape, different model) must NOT
/// silently inherit qwen3_5's attested shape: an untested family that reasons with a leading
/// `<think>` opener, or does not reason by default at all, would have its answer mislabeled as
/// `reasoning_content` under the old route-only check — the exact answer-loss class this file's
/// `servingSeparatesReasoning` doc already warns about. So any `modelType` not in
/// `attestedNativeHeterogeneousThinkingFamilies` (including `nil`, i.e. unknown) returns `false` —
/// today's byte-identical passthrough, zero regression, until that family is live-captured and added
/// to the attested set.
///
/// The `.compiled` (dense) route stays conservative `false` regardless of `modelType`: its streamed
/// reasoning shape has not been live-captured, so it keeps today's byte-identical passthrough until
/// one attests it. Flipping it is a recorded handoff item, not a guess.
public func servingThinksByDefault(route: ScalarServingDecoderRoute, modelType: String?) -> Bool {
    switch route {
    case .nativeHeterogeneous:
        guard let modelType else {
            return false
        }
        return attestedNativeHeterogeneousThinkingFamilies.contains(modelType)
    case .compiled:
        return false
    }
}

/// Legacy tool-thinking workaround gate: should this request force `enable_thinking:false` when
/// tools are attached and the client did not set it explicitly? Set ONLY for very old dense Qwen3
/// (QwenLM/Qwen3 #1817), where thinking-with-tools regressed reliability.
///
/// This is the exact negation of `servingThinksByDefault`: the one live-attested combination
/// (`.nativeHeterogeneous` + `qwen3_5`) is trained to think AND call tools, so it is the only case
/// that respects the template default (workaround off, `false`). Every other combination —
/// `.compiled` (dense) AND any not-yet-attested `.nativeHeterogeneous` family — keeps the legacy
/// workaround (`true`). Applying it conservatively to an unattested family avoids the thinking-on
/// latency trap of assuming a new model tolerates thinking-with-tools before that is proven live.
public func servingDisablesThinkingWhenToolsActive(
    route: ScalarServingDecoderRoute,
    modelType: String?
) -> Bool {
    !servingThinksByDefault(route: route, modelType: modelType)
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
