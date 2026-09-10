import MLX
import MLXLMCommon
@testable import SpikeCore
import Testing

/// `repetitionPenalty` is an HF-style MULTIPLICATIVE penalty
/// (`RepetitionContext.process`: `x < 0 ? x * penalty : x / penalty`), whose neutral element is
/// `1`, not the additive-penalty neutral element `0`. Every "is a penalty requested?" check in
/// this repo must treat `repetitionPenalty == 1` the same as `nil`/`0`, or a request carrying the
/// documented HF no-op `repetition_penalty: 1.0` (both HF-recommended presets for the deployed
/// model use exactly this value) is wrongly treated as "a penalty is active" -- disabling sampled
/// speculative decoding and, on the continuous-batch route, getting rejected outright.
struct RepetitionPenaltyNeutralElementTests {

    // MARK: - 1. The load-bearing identity: `RepetitionContext` at penalty 1 is the identity.

    /// A vocab-8 logits vector with BOTH a positive and a negative value at the two penalized
    /// (prompt) indices, so `process(logits:)` exercises both arms of the `x < 0 ? ... : ...`
    /// branch. At `repetitionPenalty == 1`, `x * 1 == x` and `x / 1 == x` exactly (no rounding
    /// is possible for a multiply/divide by exactly 1), so the output must be bit-identical to
    /// the input. This is what justifies `processor()` skipping the `RepetitionContext`
    /// entirely at penalty 1 rather than merely returning something numerically close.
    @Test func repetitionContextAtPenaltyOneIsExactIdentity() {
        var context = RepetitionContext(repetitionPenalty: 1.0, repetitionContextSize: 20)
        // Prompt tokens 2 and 5 become the penalized indices; index 2 holds a positive logit
        // (3.0) and index 5 holds a negative logit (-6.0) below.
        context.prompt(MLXArray([Int32(2), Int32(5)]))

        let input = MLXArray(
            [Float(1.0), -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])[.newAxis, .ellipsis]
        let output = context.process(logits: input)

        // Rule: never `#expect` equality of a whole array. Compare shape/count and a scalar
        // max-absolute-difference instead.
        #expect(output.shape == input.shape)
        let maxAbsDiff = MLX.max(MLX.abs(output - input)).item(Float.self)
        #expect(maxAbsDiff == 0)
    }

    // MARK: - 2. `GenerateParameters.processor()` anti-vacuity.

    @Test func processorIsNilAtRepetitionPenaltyOne() {
        let parameters = GenerateParameters(repetitionPenalty: 1.0)
        #expect(parameters.processor() == nil)
    }

    /// Anti-vacuity: a REAL repetition penalty must still build a processor -- proves the guard
    /// wasn't loosened into always returning `nil`.
    @Test func processorIsNonNilAtRealRepetitionPenalty() {
        let parameters = GenerateParameters(repetitionPenalty: 1.1)
        #expect(parameters.processor() != nil)
    }

    // MARK: - 3. Sampled-MTP provider `supports(parameters:)` no longer distinguishes neutral
    // from non-neutral repetition penalty.

    /// `sharedSampledMTPSupportsPredicate` used to refuse any non-neutral `repetitionPenalty`;
    /// that clause was removed because penalties are now applied upstream, by
    /// `MTPSpeculativeTokenIterator` penalizing the target verify rows from a scratch copy of
    /// the request's `LogitProcessor` before handing them to this provider -- so the target law
    /// `p` this provider computes is already the penalized law, and `supports()` no longer needs
    /// (or gets) a say in whether the penalty is neutral.
    ///
    /// So `supports(parameters:)` is NOT the mechanism that still distinguishes
    /// `repetitionPenalty == 1` (neutral -- a documented HF no-op both deployed presets send)
    /// from a real penalty. What still is:
    ///   - `GenerateParameters.processor()` (vendored `Evaluate.swift`, via
    ///     `GenerateParameters.repetitionPenaltyIsNeutral`) -- this is exactly what
    ///     `MTPSpeculativeTokenIterator.init` calls (`let requestProcessor =
    ///     parameters.processor()`) to build the scratch `LogitProcessor` referenced above; it
    ///     decides whether a `RepetitionContext` is constructed at all, i.e. whether the target
    ///     rows get penalized, for the sampled-MTP path.
    ///   - `DecoderPenalties.isEmpty` / `DecoderPenalties.repetitionPenaltyIsNeutral`
    ///     (`InferenceActor.swift`) -- gates whether the compiled/greedy decode path installs a
    ///     logit processor at all.
    ///   - `ContinuousServingBackend.rejectUnsupportedDecodeControls`, via the same
    ///     `DecoderPenalties.repetitionPenaltyIsNeutral` -- the continuous-batch route has no
    ///     logit processor at all, so it still refuses any non-neutral repetition penalty
    ///     outright, independent of the sampled-MTP predicate.
    @Test func supportsNoLongerDistinguishesNeutralFromRealRepetitionPenalty() {
        let truncation = SampledMTPSamplingTruncation(
            temperature: 1, topP: 0.95, topK: 20, minP: 0)
        let provider = SeededSampledMTPBlockRuntimeProvider(seed: 1, truncation: truncation)

        let thinkingPreset = GenerateParameters(
            temperature: 1,
            topP: 0.95,
            topK: 20,
            minP: 0,
            repetitionPenalty: 1.0,
            presencePenalty: 0,
            frequencyPenalty: 0)
        #expect(provider.supports(parameters: thinkingPreset))

        // A real repetition penalty on the same otherwise-eligible request is now ADMITTED too,
        // not refused -- see this test's doc comment for why.
        let realPenalty = GenerateParameters(
            temperature: 1,
            topP: 0.95,
            topK: 20,
            minP: 0,
            repetitionPenalty: 1.1,
            presencePenalty: 0,
            frequencyPenalty: 0)
        #expect(provider.supports(parameters: realPenalty))

        // Anti-vacuity: pairing the now-admitted penalty with a truncation shape that is STILL
        // genuinely refused (`minP != 0`, unaffected by the penalty relaxation, and orthogonal
        // to what this test otherwise varies) proves `supports` is still discriminating on
        // something here, not merely returning `true` unconditionally now that repetition
        // penalty no longer excludes a request.
        let realPenaltyWithRefusedMinP = GenerateParameters(
            temperature: 1,
            topP: 0.95,
            topK: 20,
            minP: 0.05,
            repetitionPenalty: 1.1,
            presencePenalty: 0,
            frequencyPenalty: 0)
        #expect(!provider.supports(parameters: realPenaltyWithRefusedMinP))
    }

    // MARK: - 4. `DecoderPenalties.isEmpty`.

    @Test func decoderPenaltiesIsEmptyAtRepetitionPenaltyOne() {
        #expect(DecoderPenalties(repetitionPenalty: 1.0).isEmpty)
    }

    @Test func decoderPenaltiesIsNotEmptyAtRealRepetitionPenalty() {
        #expect(!DecoderPenalties(repetitionPenalty: 1.1).isEmpty)
    }

    // MARK: - 5. `ContinuousServingBackend` admission.
    //
    // NOT WRITTEN HERE: exercising `ContinuousServingBackend`'s admission path requires
    // `SpikeServingAdapters` (and its `ServingCore` request type), and this test target
    // (`SpikeCoreTests`) does not depend on either -- see `spike/Package.swift`'s
    // `SpikeCoreTests` target dependencies (`SpikeCore`, `HarnessCore`, and the vendored MLX
    // libraries only). `SpikeServingAdapters` itself depends on `SpikeCore`, so adding the
    // reverse test-only dependency, or adding a new test method to the existing
    // `SpikeServingAdaptersTests/ContinuousServingBackendTests.swift`, both require editing
    // files outside this task's write set (`spike/Package.swift` /
    // `ContinuousServingBackendTests.swift`). The admission-site source fix itself is covered by
    // the mutation check on `ContinuousServingBackend.swift` (see the task's verification
    // report), which is the narrowest available proof within this write set.
}
