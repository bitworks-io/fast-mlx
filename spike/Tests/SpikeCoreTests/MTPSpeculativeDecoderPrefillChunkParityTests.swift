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
