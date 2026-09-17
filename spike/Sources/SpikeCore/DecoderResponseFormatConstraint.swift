import Foundation

/// Optional capability an injected `LogitProcessor` (see `MLXDecoder.setResponseFormatConstraint`)
/// may add to report a fatal mask violation from within `didSample`, which the vendored
/// `MLXLMCommon.LogitProcessor` protocol's `didSample` cannot itself throw.
///
/// `MLXDecoder` checks this after every `didSample` call on its `constraintProcessor` slot and
/// turns a recorded failure into a REAL thrown Swift error — surfacing through `Decoder.step`/
/// `.prefill` exactly like any other decode-time failure (see their doc comments), so a serving
/// adapter's existing generation-loop error handling picks it up unchanged, rather than silently
/// completing generation past a token the constraint said was disallowed.
///
/// A CLASS-CONSTRAINED protocol (not a value-type one): `MLXDecoder` stores its constraint
/// processor as `any LogitProcessor` (a `let`-like slot mutated only by `setResponseFormatConstraint`,
/// never copied out and back the way `processor`'s penalty struct is), so the SAME instance
/// `didSample` was just called on must be readable afterward via `as?` — a value-type conformer
/// would need its mutation threaded back out through the existential, which `LogitProcessor.
/// didSample`'s `mutating` requirement does not give this call site a way to do.
///
/// Deliberately generic: this file (SpikeCore) has no ServingCore/JSON-specific knowledge. The
/// concrete `json_object` masking processor lives in `SpikeServingAdapters`, which owns both the
/// tokenizer and the grammar table; SpikeCore only needs to know how to ask an opaque processor
/// "did you just fail?".
public protocol ConstraintProcessorFailureReporting: AnyObject {
    /// Non-nil once a sampled token violated the constraint (should be unreachable if the mask was
    /// applied correctly on the preceding `process(logits:)` call, but checked rather than trusted).
    var recordedFailure: Error? { get }
}
