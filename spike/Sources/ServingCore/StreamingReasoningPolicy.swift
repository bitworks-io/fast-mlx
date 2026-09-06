import Foundation

/// Model-type strings whose streamed reasoning shape has been LIVE-attested on the
/// `.nativeHeterogeneous` route: no leading `<think>` opener, reasoning emitted from token 0.
///
/// - `qwen3_5` (Qwen3.5/3.6/3.8 hybrid family): captured in this repo's own history at commit
///   `93e606a`.
/// - `qwen4_exp` (Flash Next): captured THIS cycle on the heavy host at harness SHA `c771346e`
///   against checkpoint `flashnext-oq4-mtp`. Non-streaming on that checkpoint split cleanly
///   (`content` = `"7pm"`, `reasoning_content` = the full reasoning, `finish_reason=stop`); the
///   defect this attestation fixes is that the STREAMING path never separated at all — zero
///   `reasoning_content` deltas were observed, the answer arrived as `content` deltas, and the
///   literal `</think>` marker leaked into visible content once. Do NOT reuse qwen3_5's "no leading
///   `<think>` opener" rationale for this entry; it is evidenced independently by the capture above,
///   not inherited from qwen3_5.
///
/// The route itself is NOT a reliable family classifier — it is the recurrent/hybrid *cache shape*,
/// and a second family can share it without sharing the attested streamed output shape. Keeping the
/// attested set as a named, documented constant (rather than a bare `route == .nativeHeterogeneous`
/// check) means admitting a new family onto this route is a reviewed decision to add it here, not an
/// implicit inheritance of another family's behavior. Membership here is NECESSARY but not
/// SUFFICIENT — see `servingThinksByDefault`, which also requires the loaded checkpoint's own chat
/// template to attest the `</think>` marker the streaming splitter hardcodes, rather than trusting
/// the model-type string alone.
private let attestedNativeHeterogeneousThinkingFamilies: Set<String> = ["qwen3_5", "qwen4_exp"]

/// Family-level classifier: does this model emit its reasoning block by DEFAULT, with no leading
/// `<think>` opener (reasoning starts at token 0)?
///
/// The gate is a THREE-way conjunction, all REQUIRED:
/// 1. `route == .nativeHeterogeneous` — the cache shape carrying every attested thinking family.
/// 2. `modelType` is in `attestedNativeHeterogeneousThinkingFamilies` — a reviewed, live-captured
///    family, not an inference from cache shape alone.
/// 3. `templateAttestsThinkMarkers` — the LOADED checkpoint's own chat template (not a hardcoded
///    assumption about the family) contains the `<think>`/`</think>` markers the streaming splitter
///    hardcodes. This is artifact-derived, not a compile-time constant: a future checkpoint that
///    reuses an attested `model_type` string but ships a template that never emits those markers
///    (or a corrupted/unreadable template) must NOT be routed through the splitter, because doing so
///    would separate a stream that structurally cannot produce the marker the splitter is looking
///    for — silently losing the answer into `reasoning_content` that never closes. There is no
///    default value for this parameter: a caller that forgets to resolve it is a compile error, not
///    a silent `true`.
///
/// `caller` note: an unreadable/absent template resolves `templateAttestsThinkMarkers` to `false`
/// (see `scalarServingChatTemplateAttestsThinkMarkers` in `MLXScalarServing.swift`), which degrades
/// this whole gate to `false` — today's byte-identical passthrough — never to separation. That is the
/// fail-closed direction: an unreadable artifact must never be treated as evidence FOR separation.
///
/// A future family admitted onto `.nativeHeterogeneous` (same cache shape, different model) must NOT
/// silently inherit an already-attested family's shape: an untested family that reasons with a
/// leading `<think>` opener, or does not reason by default at all, would have its answer mislabeled
/// as `reasoning_content` under a route-only or model-type-only check — the exact answer-loss class
/// this file's `servingSeparatesReasoning` doc already warns about. So any `modelType` not in
/// `attestedNativeHeterogeneousThinkingFamilies` (including `nil`, i.e. unknown) returns `false` —
/// today's byte-identical passthrough, zero regression, until that family is live-captured and added
/// to the attested set.
///
/// The `.compiled` (dense) route stays conservative `false` regardless of `modelType` or the template
/// probe: its streamed reasoning shape has not been live-captured, so it keeps today's byte-identical
/// passthrough until one attests it. Flipping it is a recorded handoff item, not a guess.
public func servingThinksByDefault(
    route: ScalarServingDecoderRoute,
    modelType: String?,
    templateAttestsThinkMarkers: Bool
) -> Bool {
    switch route {
    case .nativeHeterogeneous:
        guard let modelType else {
            return false
        }
        return attestedNativeHeterogeneousThinkingFamilies.contains(modelType)
            && templateAttestsThinkMarkers
    case .compiled:
        return false
    }
}

/// Legacy tool-thinking workaround gate: should this request force `enable_thinking:false` when
/// tools are attached and the client did not set it explicitly? Set ONLY for very old dense Qwen3
/// (QwenLM/Qwen3 #1817), where thinking-with-tools regressed reliability.
///
/// This is the exact negation of `servingThinksByDefault`, DELIBERATELY: every live-attested
/// thinking family (`.nativeHeterogeneous` + an attested `modelType` + an attesting template) is
/// trusted to think AND call tools without the legacy workaround, and every other combination keeps
/// it. For `qwen4_exp` (Flash Next) specifically, this coupling was live-verified this cycle: with
/// thinking ON and tools attached, a well-formed `get_weather{"city":"Paris"}` tool call was still
/// produced, `finish_reason=tool_calls`, in both streaming and non-streaming — so turning the legacy
/// workaround OFF for an attested `qwen4_exp` checkpoint is evidence-supported, not a guess. Every
/// other combination — `.compiled` (dense), any not-yet-attested `.nativeHeterogeneous` family, or an
/// attested family whose loaded template does NOT attest the think markers — keeps the legacy
/// workaround (`true`). Applying it conservatively to an unattested combination avoids the
/// thinking-on latency trap of assuming a model/checkpoint tolerates thinking-with-tools before that
/// is proven live. Do NOT decouple these two functions: the tools-active workaround and the
/// streaming-separation gate must always agree on which combinations are trusted.
public func servingDisablesThinkingWhenToolsActive(
    route: ScalarServingDecoderRoute,
    modelType: String?,
    templateAttestsThinkMarkers: Bool
) -> Bool {
    !servingThinksByDefault(
        route: route, modelType: modelType, templateAttestsThinkMarkers: templateAttestsThinkMarkers)
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
