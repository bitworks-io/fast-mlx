import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

/// Regression for a real defect found in independent review: `MTPSpeculativeDecoder
/// .buildParameters()` never set `GenerateParameters.prefillStepSize`, so it silently inherited
/// the vendored default of 512 (`Evaluate.swift:134`) instead of `MLXDecoder
/// .defaultPrefillChunkSize` (2048) — the value `CapacityModel.predictPeakBytes`'s fit check
/// already prices prefill at (see that constant's doc comment). On any prompt longer than 512
/// tokens the MTP serving route and the scalar serving route therefore took a monolithic-vs-
/// chunked prefill path difference unrelated to speculation; this repo has already measured a
/// max|Δ| of 0.546875 from sequence-chunking alone on real weights.
///
/// `buildParameters()` itself is `private`, so `@testable import` cannot reach it directly —
/// Swift access control keeps `private` file-scoped even across a testable import of the whole
/// module. This test instead drives the bridge's own PUBLIC `prefill()` with a weight-free
/// target/drafter pair and observes the `windowSize` the iterator threads down to
/// `LanguageModel.prepare(_:cache:windowSize:)`
/// (`MTPSpeculativeTokenIterator.prepare` passes `parameters.prefillStepSize` straight through as
/// `windowSize`, `MTPSpeculativeTokenIterator.swift:224` and `:292`) — the nearest observable
/// proxy for what `buildParameters()` actually set.
final class MTPSpeculativeDecoderPrefillChunkParityTests: XCTestCase {
    func testPrefillThreadsMLXDecoderDefaultPrefillChunkSizeAsThePrefillStepSize() throws {
        let target = PrefillChunkParityMockTargetModel(
            plannedTokens: [0, 0, 5, 6, 6, 6, 6, 6, 6])
        let drafter = PrefillChunkParityMockDrafter(draftedTokenValue: 6)
        var decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })

        _ = try decoder.prefill([1, 2, 3])

        let observedWindowSize = target.lastPrepareWindowSize
        XCTAssertEqual(
            observedWindowSize, MLXDecoder.defaultPrefillChunkSize,
            "MTPSpeculativeDecoder.buildParameters() must set prefillStepSize to "
                + "MLXDecoder.defaultPrefillChunkSize, or the two serving routes chunk prefill "
                + "at different sizes for reasons unrelated to speculation")
        // Discriminating on its own terms, not merely against the constant above: if this ever
        // regresses back to the vendored `GenerateParameters` default, the equality assertion
        // could still pass vacuously in the (hypothetical) case someone also changed
        // `defaultPrefillChunkSize` to 512 in the same change. Pinning against the vendored
        // default's literal value directly keeps this test meaningful even then.
        XCTAssertNotEqual(observedWindowSize, 512)
    }

    /// Kills the vacuity flagged in this file's header comment: the test above only observes
    /// that a `windowSize` value was threaded as a parameter — it never checks that anything
    /// downstream actually honors it. This drives `prefill()` with a prompt LONGER than the
    /// threaded `windowSize` (`MLXDecoder.defaultPrefillChunkSize`, 2048) against a target that
    /// conforms to `MTPPromptHiddenStatePreparingModel` with a genuine chunk loop (mirroring the
    /// real conformer's documented chunk schedule: repeated `windowSize`-token chunks, then one
    /// final forward over whatever remains), and asserts both that the loop actually iterated
    /// more than once AND that no single forward ever saw more than `windowSize` tokens. The
    /// second assertion is the detector for the real defect under investigation: a silent,
    /// unwindowed, `cache: nil` fallback forward on the drafter-priming path documented in
    /// `MTPSpeculativeTokenIterator.swift` (around its `reusablePromptHidden ??
    /// mainModel.evaluateThrowing(normalizedPrompt, cache: nil, ...)` line), which this file does
    /// not exercise directly but which this assertion shape is built to catch wherever a
    /// full-prompt forward slips through unwindowed.
    func testWindowedPrepareForMTPNeverForwardsMoreThanWindowSizeTokens() throws {
        let promptLength = 2500
        XCTAssertGreaterThan(
            promptLength, MLXDecoder.defaultPrefillChunkSize,
            "the prompt must exceed the threaded windowSize, or a monolithic forward would stay "
                + "within budget by coincidence rather than by actually chunking")
        let promptTokens = (0..<promptLength).map { Int($0 % 4) }
        let target = WindowedPrepareMockTargetModel()
        let drafter = WindowedPrepareMockDrafter()
        var decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })

        _ = try decoder.prefill(promptTokens)

        XCTAssertGreaterThan(
            target.recordedForwardTokenCounts.count, 1,
            "a mock that genuinely chunks over a prompt longer than windowSize must forward "
                + "more than once; a single forward here would mean the loop never iterated -- "
                + "recorded forwards: \(target.recordedForwardTokenCounts)")
        for tokenCount in target.recordedForwardTokenCounts {
            XCTAssertLessThanOrEqual(
                tokenCount, MLXDecoder.defaultPrefillChunkSize,
                "a forward received \(tokenCount) tokens, more than the threaded windowSize "
                    + "(\(MLXDecoder.defaultPrefillChunkSize)) -- this is exactly the shape of "
                    + "the unchunked full-prompt forward the quadratic prefill-memory defect "
                    + "predicts; recorded forwards: \(target.recordedForwardTokenCounts)")
        }
    }

    /// Mandatory anti-vacuity control for the assertion above. Without this, a defect in the
    /// assertion itself -- comparing against the wrong constant, never actually reading
    /// `recordedForwardTokenCounts`, or similar -- would be indistinguishable from a genuine
    /// pass, because both would show green. This drives the SAME >windowSize prompt against a
    /// target shaped like `PrefillChunkParityMockTargetModel` above: one that does NOT conform
    /// to `MTPPromptHiddenStatePreparingModel`, whose own `prepare()` hands the untouched full
    /// prompt back as `.tokens` rather than forwarding it in pieces. That is exactly the shape of
    /// an unchunked target, so this asserts a single forward DOES receive the full,
    /// unwindowed prompt length -- proving the detector above is capable of failing, not merely
    /// capable of passing.
    func testUnconformingTargetForwardsFullUnchunkedPromptAsAControl() throws {
        let promptLength = 2500
        let promptTokens = (0..<promptLength).map { Int($0 % 4) }
        let target = UnwindowedPrepareMockTargetModel()
        let drafter = PrefillChunkParityMockDrafter(draftedTokenValue: 0)
        var decoder = try MTPSpeculativeDecoder(
            target: target, drafter: drafter,
            cacheFactory: { target.newCache(parameters: nil) })

        _ = try decoder.prefill(promptTokens)

        XCTAssertEqual(
            target.recordedForwardTokenCounts, [promptLength],
            "a target whose prepare() does not chunk must forward the whole prompt in a single "
                + "call -- exactly the shape this file's chunk-loop assertion above must be able "
                + "to detect and fail on; recorded forwards: "
                + "\(target.recordedForwardTokenCounts)")
    }
}

// MARK: - Minimal weight-free fixtures (local to this file; not shared with
// SpikeServingAdaptersTests's InCheckpointMTPMockFixtures.swift, which lives in a different test
// target this one does not depend on)

/// Minimal `KVCache` mock: trimmable (so `MTPSpeculativeTokenIterator.init`'s
/// `canTrimPromptCache` gate is satisfied) and offset-tracking only. Modeled on
/// `InCheckpointMTPMockCountingKVCache` (`SpikeServingAdaptersTests/InCheckpointMTPMockFixtures.swift`).
private final class PrefillChunkParityMockKVCache: KVCache {
    var offset: Int = 0
    var maxSize: Int? { nil }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { (keys, values) }
    var state: [MLXArray] {
        get { [] }
        set {}
    }
    var metaState: [String] {
        get { [] }
        set {}
    }
    var isTrimmable: Bool { true }
    @discardableResult
    func trim(_ n: Int) -> Int {
        let removed = Swift.min(n, offset)
        offset -= removed
        return removed
    }
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }
    func copy() -> any KVCache {
        let copy = PrefillChunkParityMockKVCache()
        copy.offset = offset
        return copy
    }
    func innerState() -> [MLXArray] { [] }
}

/// Minimal `LanguageModel` mock: deterministic one-hot logits indexed by absolute cache position
/// (so an `ArgMaxSampler` reproduces `plannedTokens`), plus `lastPrepareWindowSize` — the single
/// value this test file exists to observe. Modeled on `InCheckpointMTPMockTargetModel`.
private final class PrefillChunkParityMockTargetModel: Module, LanguageModel {
    let plannedTokens: [Int32]
    private(set) var lastPrepareWindowSize: Int?

    init(plannedTokens: [Int32]) {
        self.plannedTokens = plannedTokens
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        lastPrepareWindowSize = windowSize
        return .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        makeLogits(startIndex: 0, positions: inputs.dim(-1))
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let positions = input.tokens.dim(-1)
        let countingCache = cache?.first as? PrefillChunkParityMockKVCache
        let startIndex = countingCache?.offset ?? 0
        let logits = makeLogits(startIndex: startIndex, positions: positions)
        countingCache?.offset = startIndex + positions

        guard state?[mtpEmitFlagKey] ?? false else {
            return LMOutput(logits: logits)
        }
        let kvSpan = countingCache?.offset ?? positions
        var out = state ?? LMOutput.State()
        out[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
        out[mtpSharedKVStatesKey] = [
            "full_attention": (
                MLXArray.zeros([1, 1, kvSpan, 4]),
                MLXArray.zeros([1, 1, kvSpan, 4])
            )
        ]
        return LMOutput(logits: logits, state: out)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [PrefillChunkParityMockKVCache()]
    }

    private func makeLogits(startIndex: Int, positions: Int) -> MLXArray {
        let vocab = 20
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0..<positions {
            let tokIdx = startIndex + i
            let tok = tokIdx < plannedTokens.count ? Int(plannedTokens[tokIdx]) : 0
            data[i * vocab + tok] = 100
        }
        return MLXArray(data, [1, positions, vocab])
    }
}

/// Minimal, stateless `MTPDrafterModel` mock: always proposes `draftedTokenValue`, `blockSize - 1`
/// times. Modeled on `InCheckpointMTPMockDrafter`; only what `prefill()`'s single call needs.
private final class PrefillChunkParityMockDrafter: Module, MTPDrafterModel {
    let draftedTokenValue: Int32

    init(draftedTokenValue: Int32) {
        self.draftedTokenValue = draftedTokenValue
        super.init()
    }

    func supportsSpeculation(for input: LMInput, target: any LanguageModel) -> Bool { true }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        let batch = lastToken.dim(0)
        let vals = Array(repeating: draftedTokenValue, count: (blockSize - 1) * batch)
        return MLXArray(vals, [batch, blockSize - 1])
    }
}

// MARK: - Chunk-loop fixtures (kill-the-vacuity test above, plus its mandatory anti-vacuity
// control). Kept separate from `PrefillChunkParityMockTargetModel`/`PrefillChunkParityMockDrafter`
// above so neither test's fixtures can accidentally couple to the other's shape.

/// A target that conforms to `MTPPromptHiddenStatePreparingModel` with a REAL chunk loop honoring
/// `windowSize`: repeatedly forwards a `windowSize`-token chunk, then exactly one final forward
/// over whatever remains. This mirrors the chunk schedule a real conformer documents for itself
/// (`windowSize ?? 512`, then chunks, then a final remainder forward) — a shape deliberately
/// generic here, not named after any specific model family (this file publishes to a sanitized
/// projection that excludes model-family names).
private final class WindowedPrepareMockTargetModel: Module, LanguageModel,
    MTPPromptHiddenStatePreparingModel
{
    /// Every forward this mock has actually executed, in call order, as the token count it
    /// received. This is the load-bearing observation both chunk-loop tests read.
    private(set) var recordedForwardTokenCounts: [Int] = []

    /// Should never be reached in this file's tests: the paired `WindowedPrepareMockDrafter` sets
    /// `requiresPromptPrefill`, which routes `MTPSpeculativeTokenIterator.prepare` to
    /// `prepareForMTP` below instead of this method. Implemented only because `LanguageModel`
    /// requires it (no default exists, unlike the no-argument `prepare()`).
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    /// Chunk loop under test: takes `windowSize`-token slices off the front of the prompt,
    /// forwarding each one (recording its size), until at most `windowSize` tokens remain, then
    /// forwards that remainder once. Every forward's hidden state is concatenated in token order,
    /// matching what a real conformer returns as `targetHidden`.
    func prepareForMTP(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> MTPPromptPreparation? {
        let prefillStepSize = windowSize ?? 512
        var y = input.text
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        var hiddenChunks: [MLXArray] = []
        while y.tokens.size > prefillStepSize {
            let chunk = y[text: .newAxis, ..<prefillStepSize]
            let output = try evaluateThrowing(chunk, cache: cache, state: state)
            // Safe to force-unwrap: this mock's own `evaluateThrowing` (below) always sets
            // `mtpLastHiddenStatesKey` on the state it returns.
            hiddenChunks.append(output.state![mtpLastHiddenStatesKey]!)
            state = output.state ?? state
            y = y[prefillStepSize...]
        }
        let finalOutput = try evaluateThrowing(y[text: .newAxis], cache: cache, state: state)
        hiddenChunks.append(finalOutput.state![mtpLastHiddenStatesKey]!)
        return MTPPromptPreparation(
            result: .logits(finalOutput),
            targetHidden: concatenated(hiddenChunks, axis: 1))
    }

    /// The single forward entry point `MTPSpeculativeTokenIterator` actually calls (confirmed by
    /// reading the vendored iterator: every one of its main-model calls goes through
    /// `evaluateThrowing`, never `callAsFunction` directly) — so recording here observes every
    /// forward this mock makes, both inside `prepareForMTP`'s chunk loop and any the iterator
    /// might make afterward.
    func evaluateThrowing(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) throws -> LMOutput {
        let positions = input.tokens.dim(-1)
        recordedForwardTokenCounts.append(positions)
        var out = state ?? LMOutput.State()
        out[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
        return LMOutput(logits: makeLogits(positions: positions), state: out)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        makeLogits(positions: inputs.dim(-1))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [PrefillChunkParityMockKVCache()]
    }

    private func makeLogits(positions: Int) -> MLXArray {
        let vocab = 4
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0..<positions { data[i * vocab] = 100 }
        return MLXArray(data, [1, positions, vocab])
    }
}

/// Drafter shape for the chunk-loop test: `requiresPromptPrefill` is `true` so
/// `MTPSpeculativeTokenIterator.prepare` actually attempts `MTPPromptHiddenStatePreparingModel`
/// prompt preparation on the target (with `requiresPromptPrefill` at its default `false`, as on
/// `PrefillChunkParityMockDrafter`, the iterator never even attempts the cast — see
/// `MTPSpeculativeTokenIterator.swift`'s `prepare`, every `prepareForMTP` branch is gated on
/// `drafter.requiresPromptPrefill`). Deliberately NOT `StatefulMTPDrafterModel`: that keeps this
/// test scoped to observing the prompt-preparation call itself, not the separate
/// stateful-drafter-priming fallback the iterator falls back to afterward for a drafter that
/// owns its own private cache. `requiresSharedTargetKV` is `false` so the iterator's post-preparation
/// state check does not also require this mock to fabricate a shared-KV dict.
private final class WindowedPrepareMockDrafter: Module, MTPDrafterModel {
    var requiresPromptPrefill: Bool { true }
    var requiresSharedTargetKV: Bool { false }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        let batch = lastToken.dim(0)
        let vals = Array(repeating: Int32(0), count: (blockSize - 1) * batch)
        return MLXArray(vals, [batch, blockSize - 1])
    }
}

/// A target that does NOT conform to `MTPPromptHiddenStatePreparingModel` — the same shape as
/// `PrefillChunkParityMockTargetModel` above in the one respect that matters here: `prepare()`
/// hands back the untouched full prompt as `.tokens` rather than forwarding any of it itself.
/// This is the anti-vacuity control for `WindowedPrepareMockTargetModel`'s chunk-loop assertion:
/// it proves that assertion is capable of failing on a target that genuinely does not chunk, not
/// merely capable of passing.
private final class UnwindowedPrepareMockTargetModel: Module, LanguageModel {
    private(set) var recordedForwardTokenCounts: [Int] = []

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func evaluateThrowing(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) throws -> LMOutput {
        let positions = input.tokens.dim(-1)
        recordedForwardTokenCounts.append(positions)
        var out = state ?? LMOutput.State()
        out[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
        return LMOutput(logits: makeLogits(positions: positions), state: out)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        makeLogits(positions: inputs.dim(-1))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [PrefillChunkParityMockKVCache()]
    }

    private func makeLogits(positions: Int) -> MLXArray {
        let vocab = 4
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0..<positions { data[i * vocab] = 100 }
        return MLXArray(data, [1, positions, vocab])
    }
}
