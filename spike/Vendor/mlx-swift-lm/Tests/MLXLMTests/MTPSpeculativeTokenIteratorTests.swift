// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Testing) @testable import MLXLMCommon
import MLXNN
import Testing

// MARK: - Synthetic mocks for iterator plumbing

private let prepareContinuationStateKey = LMOutput.Key<Int>(
    "tests.mtp.prepareContinuation")

/// Records `draftBlock(...)` invocations and returns a fixed token pattern
/// so the iterator's draft/verify/accept flow can be exercised without a
/// real drafter.
private final class MockDrafter: Module, MTPDrafterModel {
    private(set) var draftBlockCallCount = 0
    var draftedTokenValue: Int32
    /// Per-call record of what the iterator handed `draftBlock`: the
    /// sequence-axis span of each sharedKV entry, and the query offset.
    /// Lets tests assert the state the drafter conditions on, not just the
    /// tokens that come out of it.
    private(set) var receivedSharedKVSpans: [[String: Int]] = []
    private(set) var receivedQueryOffsets: [Int] = []

    init(draftedTokenValue: Int32 = 7) {
        self.draftedTokenValue = draftedTokenValue
        super.init()
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlockCallCount += 1
        receivedSharedKVSpans.append(sharedKV.mapValues { $0.0.dim(-2) })
        receivedQueryOffsets.append(queryOffset)
        let batch = lastToken.dim(0)
        let vals = Array(repeating: draftedTokenValue, count: (blockSize - 1) * batch)
        return MLXArray(vals, [batch, blockSize - 1])
    }
}

private final class MockStatefulGreedyDrafter: Module, StatefulMTPDrafterModel {
    let maximumBlockSize: Int? = 2
    let requiresSharedTargetKV = false
    let requiresPromptPrefill = true
    let requiresGreedySampling = true
    var supportsInput = true

    private(set) var prepareCallCount = 0
    private(set) var draftCallCount = 0
    private(set) var commitCallCount = 0
    var sampleDuringCommit = false
    private(set) var preparedPromptShape: [Int]?
    private(set) var preparedHiddenShape: [Int]?

    func supportsSpeculation(for _: LMInput) -> Bool { supportsInput }

    func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: [CountingKVCache()])
    }

    func prepareDrafterState(
        target _: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus _: MLXArray,
        positionDeltas _: MLXArray?,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        prepareCallCount += 1
        preparedPromptShape = promptTokens.shape
        preparedHiddenShape = targetHidden.shape
    }

    func draftBlock(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        sampler _: any LogitSampler
    ) -> MLXArray {
        draftCallCount += 1
        return MLXArray.zeros([lastToken.dim(0), blockSize - 1], dtype: .int32)
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state _: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden,
            sharedKV: sharedKV, positionDeltas: positionDeltas,
            queryOffset: queryOffset, blockSize: blockSize, sampler: sampler)
    }

    func commitDrafterState(
        target _: any LanguageModel,
        targetHidden _: MLXArray,
        draftTokens _: MLXArray,
        acceptedCount _: Int,
        finalToken _: MLXArray,
        positionDeltas _: MLXArray?,
        state _: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        commitCallCount += 1
        if sampleDuringCommit {
            _ = sampler.sample(logits: MLXArray([Float(0), 1]))
        }
    }
}

private enum ForcedSampledBlockDecisionError: Error {
    case rejected
}

private final class ForcedSampledBlockDecisionProvider: SampledMTPBlockRuntimeDeciding {
    let proposalSampler: any LogitSampler
    var supported = true
    var shouldThrow = false
    var decision = SampledMTPBlockRuntimeDecision(
        outputTokens: [11], acceptedDraftCount: 0)
    private(set) var proposedTokens = [[Int]]()
    private(set) var targetRowCounts = [Int]()

    init(proposalSampler: any LogitSampler = ArgMaxSampler()) {
        self.proposalSampler = proposalSampler
    }

    func supports(parameters _: GenerateParameters) -> Bool { supported }

    func decide(
        proposedTokens: [Int],
        targetLogits: [MLXArray],
        bonusTargetLogits _: MLXArray
    ) throws -> SampledMTPBlockRuntimeDecision {
        self.proposedTokens.append(proposedTokens)
        targetRowCounts.append(targetLogits.count)
        if shouldThrow { throw ForcedSampledBlockDecisionError.rejected }
        return decision
    }
}

private final class CountingSampler: LogitSampler {
    private(set) var sampleCount = 0

    func sample(logits: MLXArray) -> MLXArray {
        sampleCount += 1
        return argMax(logits, axis: -1)
    }
}

/// Minimal `LanguageModel` mock that emits MTP state on every call when
/// `mtpEmitFlagKey` is true. Returns shaped logits and a trimmable KV cache.
private final class MockMainModel: Module, LanguageModel, KVCacheDimensionProvider,
    SpeculativeCacheRewindModel, MTPPromptHiddenStateEvaluationOrderPreparingModel
{
    var kvHeads: [Int] { [1] }
    var maximumNativeTargetCacheRewind = 0
    /// Sequence of token values returned in increasing position order. Length
    /// must cover all positions the iterator will sample across the run.
    var nextLogitTokens: [Int32]
    var perPositionIndex: Int = 0
    /// If `true`, omit the MTP state keys from the returned `LMOutput` so the
    /// iterator's passthrough fallback is exercised.
    var omitDrafterState: Bool = false
    /// Exercise the VLM-style prepare path that evaluates the prompt and
    /// returns logits plus continuation state.
    var returnLogitsFromPrepare: Bool = false
    var capturesPromptHiddenDuringPrepare = false

    private(set) var callCount: Int = 0
    private(set) var lastIncomingEmitFlag: Bool? = nil
    private(set) var incomingPrepareContinuationValues: [Int?] = []
    private(set) var receivedTokenShapes: [[Int]] = []
    private(set) var receivedPromptEvaluationOrders: [MTPPromptPreparationEvaluationOrder] = []
    /// Sequence-axis span of each emitted sharedKV snapshot, in emit order.
    /// Tests assert this against the mock cache offset to pin the mock's
    /// span fidelity (see the emit block below).
    private(set) var emittedSharedKVSpans: [Int] = []

    init(nextLogitTokens: [Int32]) {
        self.nextLogitTokens = nextLogitTokens
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        if returnLogitsFromPrepare {
            let positions = input.text.tokens.dim(-1)
            if let first = cache.first as? CountingKVCache {
                first.offset += positions
            }
            var state = LMOutput.State()
            state[prepareContinuationStateKey] = 41
            return .logits(LMOutput(logits: makeLogits(positions: positions), state: state))
        }
        // Return `.tokens(...)`; the iterator's `prepare` will follow up with
        // a one-position forward call that primes drafter state.
        return .tokens(input.text)
    }

    func prepareForMTP(
        _ input: LMInput,
        cache: [KVCache],
        windowSize _: Int?,
        collectTelemetry: Bool
    ) throws -> MTPPromptPreparation? {
        try prepareForMTP(
            input,
            cache: cache,
            windowSize: nil,
            collectTelemetry: collectTelemetry,
            evaluationOrder: .cacheFirst)
    }

    func prepareForMTP(
        _ input: LMInput,
        cache: [KVCache],
        windowSize _: Int?,
        collectTelemetry: Bool,
        evaluationOrder: MTPPromptPreparationEvaluationOrder
    ) throws -> MTPPromptPreparation? {
        guard capturesPromptHiddenDuringPrepare else { return nil }
        receivedPromptEvaluationOrders.append(evaluationOrder)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        let normalized = LMInput.Text(tokens: normalizedMTPTokenBatch(input.text.tokens))
        let output = callAsFunction(normalized, cache: cache, state: state)
        guard let hidden = output.state?[mtpLastHiddenStatesKey] else { return nil }
        return MTPPromptPreparation(
            result: .logits(output),
            targetHidden: hidden,
            telemetry: collectTelemetry
                ? .init(
                    promptTokenCount: normalized.tokens.size,
                    hiddenShape: hidden.shape,
                    hiddenByteCount: hidden.nbytes,
                    evaluationOrder: evaluationOrder,
                    chunks: [.init(
                        tokenOffset: 0,
                        tokenCount: normalized.tokens.size,
                        targetForwardSchedulingSeconds: 0)],
                    cacheHiddenEvaluationSeconds: 0,
                    hiddenConcatenationSeconds: 0)
                : nil)
    }

    /// Returns deterministic one-hot logits at each position so a `softmax/
    /// argmax` sampler picks the planned token sequence.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let positions = inputs.dim(-1)
        return makeLogits(positions: positions)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        callCount += 1
        receivedTokenShapes.append(input.tokens.shape)
        lastIncomingEmitFlag = state?[mtpEmitFlagKey]
        incomingPrepareContinuationValues.append(state?[prepareContinuationStateKey])

        let positions = input.tokens.dim(-1)
        let logits = makeLogits(positions: positions)

        // Update the mock cache to reflect that `positions` tokens were seen.
        if let cache {
            for entry in cache {
                if let counting = entry as? CountingKVCache {
                    counting.offset += positions
                } else if let recurrent = entry as? MambaCache {
                    updateMockMambaCache(
                        recurrent,
                        positions: positions,
                        checkpointAfter: state?[mtpCacheCheckpointIndexKey])
                }
            }
        }

        if !omitDrafterState, state?[mtpEmitFlagKey] ?? false {
            // Mirror the real emit hook's spans: sharedKV is captured from
            // what the attention layers consumed, so its sequence axis covers
            // the cache's full logical offset (everything processed so far);
            // lastHidden is the forward's own output, so it covers only the
            // current chunk. Span-accurate sharedKV emission is what lets a
            // test distinguish a trimmed snapshot from a stale one — with a
            // fixed-span mock those are indistinguishable.
            let kvSpan = (cache?.first as? CountingKVCache)?.offset ?? positions
            emittedSharedKVSpans.append(kvSpan)
            var out = state ?? LMOutput.State()
            out[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
            out[mtpSharedKVStatesKey] = [
                "full_attention": (
                    MLXArray.zeros([1, 1, kvSpan, 4]),
                    MLXArray.zeros([1, 1, kvSpan, 4])
                ),
                "sliding_attention": (
                    MLXArray.zeros([1, 1, kvSpan, 4]),
                    MLXArray.zeros([1, 1, kvSpan, 4])
                ),
            ]
            return LMOutput(logits: logits, state: out)
        }
        return LMOutput(logits: logits)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [CountingKVCache()]
    }

    private func makeLogits(positions: Int) -> MLXArray {
        // [B=1, positions, vocab=20]. One-hot at the planned token for each
        // position, taken from `nextLogitTokens[perPositionIndex...]`.
        let vocab = 20
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0 ..< positions {
            let tokIdx = perPositionIndex + i
            let tok = tokIdx < nextLogitTokens.count ? Int(nextLogitTokens[tokIdx]) : 0
            data[i * vocab + tok] = 100
        }
        perPositionIndex += positions
        return MLXArray(data, [1, positions, vocab])
    }

    private func updateMockMambaCache(
        _ cache: MambaCache,
        positions: Int,
        checkpointAfter: Int?
    ) {
        let currentState = cache.state
        if let first = currentState.first, first.size == 1 {
            eval(first)
        }
        let start = currentState.first.flatMap { first in
            first.size == 1 ? first.item(Int.self) : nil
        } ?? 0
        if let split = checkpointAfter, split > 0, split < positions {
            cache.saveSpeculativeCheckpoint(
                convState: mockMambaStateValue(start + split),
                recurrentState: mockMambaStateValue(start + split),
                advancedBy: split)
        }
        cache.advance(positions)
        cache[0] = mockMambaStateValue(start + positions)
        cache[1] = mockMambaStateValue(start + positions)
    }
}

private func mockMambaStateValue(_ value: Int) -> MLXArray {
    MLXArray([Int32(value)])
}

@Test
func testMTPPromptPrimingReusesNormalizedFullPromptHidden() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 0, 0, 0])
    main.capturesPromptHiddenDuringPrepare = true
    let drafter = MockStatefulGreedyDrafter()

    let iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        parameters: GenerateParameters(maxTokens: 1, temperature: 0),
        blockSize: 2,
        collectPhaseTelemetry: true)

    #expect(main.receivedTokenShapes == [[1, 3]])
    #expect(drafter.prepareCallCount == 1)
    #expect(drafter.preparedPromptShape == [1, 3])
    #expect(drafter.preparedHiddenShape == [1, 3, 4])
    let promptTelemetry = try #require(iterator.promptPreparationTelemetry)
    #expect(promptTelemetry.promptTokenCount == 3)
    #expect(promptTelemetry.hiddenShape == [1, 3, 4])
    #expect(promptTelemetry.evaluationOrder == .cacheFirst)
    #expect(promptTelemetry.chunks.map(\.tokenOffset) == [0])
    #expect(promptTelemetry.chunks.map(\.tokenCount) == [3])
    let phases = iterator.speculativeDecodingPhaseTelemetry
    #expect(
        phases.targetPrefillSeconds + phases.drafterPromptPrimingSeconds
            <= iterator.promptPrefillTime)
}

@Test
func testMTPPromptPrimingForwardsHiddenFirstEvaluationOrder() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 0, 0, 0])
    main.capturesPromptHiddenDuringPrepare = true
    let drafter = MockStatefulGreedyDrafter()

    let iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        parameters: GenerateParameters(maxTokens: 1, temperature: 0),
        blockSize: 2,
        collectPhaseTelemetry: true,
        promptPreparationEvaluationOrder: .hiddenFirst)

    #expect(main.receivedPromptEvaluationOrders == [.hiddenFirst])
    #expect(iterator.promptPreparationTelemetry?.evaluationOrder == .hiddenFirst)
}

@Test
func testMTPPromptPrimingForwardsCombinedEvaluationOrder() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 0, 0, 0])
    main.capturesPromptHiddenDuringPrepare = true
    let drafter = MockStatefulGreedyDrafter()

    let iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        parameters: GenerateParameters(maxTokens: 1, temperature: 0),
        blockSize: 2,
        collectPhaseTelemetry: true,
        promptPreparationEvaluationOrder: .combined)

    #expect(main.receivedPromptEvaluationOrders == [.combined])
    #expect(iterator.promptPreparationTelemetry?.evaluationOrder == .combined)
}

/// Minimal `KVCache` that satisfies the protocol's trimmable interface; the
/// mock model adjusts `offset` directly. Inherits the default
/// `ropeOffset = .scalar(offset)` from the `KVCache` protocol extension.
private final class CountingKVCache: KVCache {
    var offset: Int = 0
    var maxSize: Int? { nil }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }
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
        let c = CountingKVCache()
        c.offset = offset
        return c
    }
    func innerState() -> [MLXArray] { [] }
}

// MARK: - Smallest-unit-of-work smoke test

@Suite
struct MTPSpeculativeTokenIteratorTests {

@Test
func testMTPSpeculateRoundSmokeWithSynthetics() throws {
    // Plan: prompt of 3 tokens [1, 2, 3], main model is rigged so that it
    // samples bonus token 7 at prefill, then on the verify pass samples
    // [7, 7, 7, 9] (3 matching drafts, 1 correction). Drafter returns
    // [7, 7, 7] so all three drafts match. Total tokens yielded across
    // the bonus drain + one speculation round: [bonus=7, draft=7,
    // draft=7, draft=7, correction=9].

    let mainLogitTokens: [Int32] = [
        // Prefill follow-up call (length 3): only the final position is
        // sampled (iterator takes `logits[-1]`); positions 0/1 can hold any
        // value < vocab=20. Using 0 as an inert placeholder.
        0, 0, 7,
        // Verify pass (length 4): [7, 7, 7, 9]
        7, 7, 7, 9,
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    let drafter = MockDrafter(draftedTokenValue: 7)
    let promptTokens = MLXArray([Int32(1), 2, 3])
    let input = LMInput(tokens: promptTokens)

    var iter = try MTPSpeculativeTokenIterator(
        input: input,
        mainModel: main,
        drafter: drafter,
        mainCache: nil,
        parameters: GenerateParameters(maxTokens: 8),
        blockSize: 4,
        collectPhaseTelemetry: true
    )

    // First token drained from `next()` is the prepare-time bonus; the
    // speculation round runs on the second call.
    let t0 = iter.next()
    #expect(t0 == 7)

    // Drain the speculation round's pending buffer.
    let t1 = iter.next()
    let t2 = iter.next()
    let t3 = iter.next()
    let t4 = iter.next()
    #expect(t1 == 7)
    #expect(t2 == 7)
    #expect(t3 == 7)
    #expect(t4 == 9)

    // 1 prepare bonus + one round of 4 tokens (3 accepted + 1 correction).
    #expect(iter.tokenCount == 5)
    #expect(drafter.draftBlockCallCount == 1)
    // proposedCount = numDraft = 3; accepted = 3.
    #expect(iter.proposedCount == 3)
    #expect(iter.acceptedCount == 3)
    let telemetry = try #require(iter.speculativeDecodingTelemetry)
    #expect(telemetry.roundCount == 1)
    #expect(telemetry.draftTokenCount == 3)
    #expect(telemetry.acceptedDraftTokenCount == 3)
    #expect(telemetry.targetModelCallCount == 1)
    #expect(telemetry.draftModelCallCount == 1)
    #expect(telemetry.targetVerifiedTokenCount == 4)
    #expect(telemetry.emittedTokenCount == iter.tokenCount)
    let phases = iter.speculativeDecodingPhaseTelemetry
    #expect(phases.targetPrefillCount == 1)
    #expect(phases.drafterPromptPrimingCount == 0)
    #expect(phases.draftBlockCount == 1)
    #expect(phases.targetVerificationCount == 1)
    #expect(phases.hybridRewindReplayCount == 0)
    #expect(phases.finalizationCount == 0)
    #expect(phases.targetPrefillSeconds >= 0)
    #expect(phases.draftBlockSeconds >= 0)
    #expect(phases.targetVerificationSeconds >= 0)
    // Verify the main model received emit=true on every call after prefill.
    #expect(main.lastIncomingEmitFlag == true)
}

@Test
func testMTPGreedyVerificationBatchesTargetTokenReadbackWithoutChangingStream() throws {
    let main = MockMainModel(nextLogitTokens: [
        0, 0, 5,
        5, 5, 5, 9,
    ])
    let drafter = MockDrafter(draftedTokenValue: 5)
    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        parameters: GenerateParameters(maxTokens: 8, temperature: 0),
        blockSize: 4)

    let tokens = [
        iterator.next(), iterator.next(), iterator.next(), iterator.next(), iterator.next(),
    ]

    #expect(tokens == [5, 5, 5, 5, 9])
    #expect(iterator.proposedCount == 3)
    #expect(iterator.acceptedCount == 3)
    #expect(iterator.speculativeDecodingTelemetry?.targetVerifiedTokenCount == 4)
    #expect(iterator._verifierTokenReadbackCountForTesting == 1)
}

@Test
func testMTPFinalizationTrimsAcceptedLookaheadThatWasNeverEmitted() throws {
    let main = MockMainModel(nextLogitTokens: [
        0, 0, 7,
        7, 7, 7, 9,
    ])
    let drafter = MockDrafter(draftedTokenValue: 7)
    let cache = CountingKVCache()
    var iter = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 8),
        blockSize: 4)

    #expect(iter.next() == 7, "prepare-time bonus")
    #expect(iter.next() == 7, "first accepted draft")
    #expect(cache.offset == 7, "verify pass has cached all four candidate inputs")

    iter.finalizeGeneration()

    #expect(cache.offset == 5, "two accepted-but-unemitted draft inputs must be removed")
}

@Test
func testMTPFinalizationAlsoTrimsReturnedAcceptedStopTokenWhenCallerDiscardsIt() throws {
    let main = MockMainModel(nextLogitTokens: [
        0, 0, 7,
        7, 7, 7, 9,
    ])
    let drafter = MockDrafter(draftedTokenValue: 7)
    let cache = CountingKVCache()
    var iter = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 8),
        blockSize: 4)

    #expect(iter.next() == 7, "prepare-time bonus")
    #expect(iter.next() == 7, "accepted draft returned as a stop-token candidate")
    iter.discardGeneratedToken()
    iter.finalizeGeneration()

    #expect(
        cache.offset == 4,
        "discarded accepted token and all unreturned accepted lookahead must be removed")
}

@Test
func testMTPFinalizationCommitsRetainedPrepareBonusAtLengthLimit() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 7, 0])
    let drafter = MockDrafter(draftedTokenValue: 7)
    let cache = CountingKVCache()
    var iter = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 1),
        blockSize: 4)

    #expect(iter.next() == 7, "prepare-time bonus reaches the output limit")
    #expect(iter.next() == nil)
    #expect(cache.offset == 3, "the returned bonus has not been evaluated yet")

    iter.finalizeGeneration()

    #expect(cache.offset == 4, "every retained output token must be represented in cache")
}

@Test
func testMTPFinalizationCommitsRetainedVerifierBonusAtLengthLimit() throws {
    let main = MockMainModel(nextLogitTokens: [
        0, 0, 7,
        7, 7, 7, 9,
        0,
    ])
    let drafter = MockDrafter(draftedTokenValue: 7)
    let cache = CountingKVCache()
    var iter = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 5),
        blockSize: 4)

    var emitted: [Int] = []
    while let token = iter.next() {
        emitted.append(token)
    }
    #expect(emitted == [7, 7, 7, 7, 9])
    #expect(cache.offset == 7, "the verifier's final bonus has not been evaluated yet")

    iter.finalizeGeneration()

    #expect(cache.offset == 8, "prompt plus all five retained outputs must be cached")
}

@Test
func testMTPFinalizationDoesNotCommitDiscardedVerifierBonus() throws {
    let main = MockMainModel(nextLogitTokens: [
        0, 0, 7,
        7, 7, 7, 9,
        0,
    ])
    let drafter = MockDrafter(draftedTokenValue: 7)
    let cache = CountingKVCache()
    var iter = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 5),
        blockSize: 4)

    #expect(iter.next() == 7, "prepare-time bonus")
    #expect(iter.next() == 7, "first accepted draft")
    #expect(iter.next() == 7, "second accepted draft")
    #expect(iter.next() == 7, "third accepted draft")
    #expect(iter.next() == 9, "uncommitted verifier bonus returned as stop candidate")
    iter.discardGeneratedToken()

    iter.finalizeGeneration()

    #expect(cache.offset == 7, "a rejected terminal token must remain absent from cache")
}

@Test
func testMTPLogitsReprimeDiscardRestoresPromptCache() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 6, 0, 0, 0])
    main.returnLogitsFromPrepare = true
    let drafter = MockStatefulGreedyDrafter()
    let cache = CountingKVCache()
    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 4, temperature: 0),
        blockSize: 2)

    #expect(cache.offset == 4, "prompt plus the consumed re-prime token")
    #expect(iterator.next() == 5, "first re-prime pending token")
    iterator.discardGeneratedToken()
    iterator.finalizeGeneration()

    #expect(
        cache.offset == 3,
        "discarding the first re-prime token must restore the prompt-only cache")
}

@Test
func testMTPLogitsReprimeThreadsPrepareContinuationState() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 6, 0, 0, 0])
    main.returnLogitsFromPrepare = true
    let drafter = MockStatefulGreedyDrafter()

    _ = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        parameters: GenerateParameters(maxTokens: 4, temperature: 0),
        blockSize: 2)

    #expect(
        main.incomingPrepareContinuationValues.first == 41,
        "the re-prime forward must preserve model-specific prepare state")
}

@Test
func testMTPReprimeFallbackRetainsHybridCheckpointUntilDiscardFinalization() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 6, 0, 0, 0])
    main.returnLogitsFromPrepare = true
    main.omitDrafterState = true
    main.maximumNativeTargetCacheRewind = 1
    let drafter = MockStatefulGreedyDrafter()
    let attention = CountingKVCache()
    let recurrent = MambaCache()
    recurrent[0] = MLXArray.ones([1, 1, 4])
    recurrent[1] = MLXArray.ones([1, 2, 2, 2])
    let originalRecurrent = recurrent.state

    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [attention, recurrent],
        parameters: GenerateParameters(maxTokens: 4, temperature: 0),
        blockSize: 2)

    #expect(
        iterator.passthroughReason
            == "target did not emit drafter state for Qwen MTP prompt prefill")
    #expect(iterator.next() == 5)
    iterator.discardGeneratedToken()
    iterator.finalizeGeneration()

    #expect(attention.offset == 3)
    for (actual, expected) in zip(recurrent.state, originalRecurrent) {
        eval(actual, expected)
        #expect(allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self))
    }
}

// MARK: - Passthrough fallback when state is absent

@Test
func testMTPIteratorMissingStateFallsBackToPassthrough() throws {
    // Main model with `omitDrafterState=true` never populates the MTP keys,
    // so the iterator switches to passthrough on the first `speculateRound`
    // call. Drafter must not be invoked.
    //
    // With maxTokens=3, the iterator yields 1 prepare-time bonus + 2
    // passthrough tokens before hitting the budget. The third passthrough
    // position is never reached.
    let mainLogitTokens: [Int32] = [
        // Prefill follow-up: length 3, final position picks bonus 5
        // (positions 0/1 are not sampled and must be < vocab=20; using 0 as
        // a placeholder).
        0, 0, 5,
        // Passthrough single-token rounds (length-1 calls): the iterator
        // takes 2 of these inside the 3-token budget after yielding the
        // bonus first.
        11, 12,
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    main.omitDrafterState = true
    let drafter = MockDrafter()
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 3), blockSize: 4
    )

    let tokens = [iter.next(), iter.next(), iter.next(), iter.next()]
    // [bonus from prepare, 2 passthrough tokens, nil].
    #expect(tokens[0] == 5)
    #expect(tokens[1] == 11)
    #expect(tokens[2] == 12)
    #expect(tokens[3] == nil)
    // Drafter was never invoked for an actual round.
    #expect(drafter.draftBlockCallCount == 0)
}

@Test
func testGreedyOnlyDrafterFallsBackBeforeDrafterPrefill() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5])
    let drafter = MockStatefulGreedyDrafter()

    _ = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        parameters: GenerateParameters(maxTokens: 1, temperature: 0.7),
        blockSize: 2)

    #expect(drafter.prepareCallCount == 0)
    #expect(drafter.draftCallCount == 0)
    #expect(main.lastIncomingEmitFlag == false)
}

@Test
func testGreedyOnlyDrafterUsesExplicitSampledBlockDecisionProvider() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 7, 8])
    main.capturesPromptHiddenDuringPrepare = true
    let drafter = MockStatefulGreedyDrafter()
    let provider = ForcedSampledBlockDecisionProvider()
    let cache = CountingKVCache()

    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 4, temperature: 1),
        blockSize: 2,
        sampledBlockDecisionProvider: provider)

    #expect(iterator.next() == 5, "prepare-time target bonus")
    #expect(iterator.next() == 11, "provider-selected residual correction")
    #expect(provider.proposedTokens == [[0]])
    #expect(provider.targetRowCounts == [1])
    #expect(iterator.proposedCount == 1)
    #expect(iterator.acceptedCount == 0)
    #expect(drafter.draftCallCount == 1)
    #expect(iterator.passthroughReason == nil)
    #expect(cache.offset == 4, "only the selected input is committed")
}

@Test
func testSampledBlockDecisionFailureRollsBackAndFallsBackToTarget() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 7, 8, 9])
    main.capturesPromptHiddenDuringPrepare = true
    let drafter = MockStatefulGreedyDrafter()
    let provider = ForcedSampledBlockDecisionProvider()
    provider.shouldThrow = true
    let cache = CountingKVCache()

    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 4, temperature: 1, seed: 7),
        blockSize: 2,
        sampledBlockDecisionProvider: provider)

    #expect(iterator.next() == 5, "prepare-time target bonus")
    #expect(iterator.next() == 7, "ordinary target sampler correction")
    #expect(iterator.acceptedCount == 0)
    #expect(iterator.passthroughReason == "sampled MTP block decision failed")
    #expect(cache.offset == 4, "failed speculative draft must be rolled back")
}

@Test
func testSampledBlockDecisionFailureSkipsDrafterCommitAndProposalRNG() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 7, 8, 9])
    main.capturesPromptHiddenDuringPrepare = true
    let drafter = MockStatefulGreedyDrafter()
    drafter.sampleDuringCommit = true
    let proposalSampler = CountingSampler()
    let provider = ForcedSampledBlockDecisionProvider(proposalSampler: proposalSampler)
    provider.shouldThrow = true

    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        parameters: GenerateParameters(maxTokens: 4, temperature: 1, seed: 7),
        blockSize: 2,
        sampledBlockDecisionProvider: provider)

    #expect(iterator.next() == 5)
    #expect(iterator.next() == 7)
    #expect(drafter.commitCallCount == 0)
    #expect(proposalSampler.sampleCount == 0)
}

@Test
func testOutOfVocabularySampledDecisionFallsBackToTarget() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 7, 8, 9])
    main.capturesPromptHiddenDuringPrepare = true
    let drafter = MockStatefulGreedyDrafter()
    let provider = ForcedSampledBlockDecisionProvider()
    provider.decision = SampledMTPBlockRuntimeDecision(
        outputTokens: [20], acceptedDraftCount: 0)
    let cache = CountingKVCache()

    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 4, temperature: 1, seed: 7),
        blockSize: 2,
        sampledBlockDecisionProvider: provider)

    #expect(iterator.next() == 5)
    #expect(iterator.next() == 7)
    #expect(iterator.passthroughReason == "sampled MTP block decision failed")
    #expect(cache.offset == 4)
}

@Test(arguments: [
    (accepted: 1, output: [7, 11], expectedOffset: 5),
    (accepted: 2, output: [7, 7, 11], expectedOffset: 6),
])
func testSampledProviderPreservesPartialAndFullAcceptedPrefix(
    fixture: (accepted: Int, output: [Int], expectedOffset: Int)
) throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 7, 7, 9, 10])
    let drafter = MockDrafter(draftedTokenValue: 7)
    let provider = ForcedSampledBlockDecisionProvider()
    provider.decision = SampledMTPBlockRuntimeDecision(
        outputTokens: fixture.output,
        acceptedDraftCount: fixture.accepted)
    let cache = CountingKVCache()

    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 5, temperature: 1, seed: 7),
        blockSize: 3,
        sampledBlockDecisionProvider: provider)

    #expect(iterator.next() == 5)
    for token in fixture.output {
        #expect(iterator.next() == token)
    }
    #expect(iterator.acceptedCount == fixture.accepted)
    #expect(cache.offset == fixture.expectedOffset)
}

@Test
func testUnsupportedPromptInputFallsBackBeforeDrafterPrefill() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5])
    let drafter = MockStatefulGreedyDrafter()
    drafter.supportsInput = false
    let input = LMInput(
        text: .init(tokens: MLXArray([Int32(1), 2, 3])),
        image: .init(pixels: MLXArray.zeros([1, 1, 1, 3])))

    var iterator = try MTPSpeculativeTokenIterator(
        input: input,
        mainModel: main,
        drafter: drafter,
        parameters: GenerateParameters(maxTokens: 1, temperature: 0),
        blockSize: 2)

    #expect(drafter.prepareCallCount == 0)
    #expect(drafter.draftCallCount == 0)
    #expect(main.lastIncomingEmitFlag == false)
    #expect(iterator.next() == 5)
    #expect(iterator.passthroughReason == "drafter does not support this prompt input")
}

@Test
func testRecurrentCacheNeverFallsBackToGenericTrimWithoutNativeRewind() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 11])
    let drafter = MockDrafter(draftedTokenValue: 5)
    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        parameters: GenerateParameters(maxTokens: 3, temperature: 0),
        blockSize: 2)

    // Simulate a runtime cache-topology transition after otherwise-valid dense prefill.
    iterator.mainCache.append(MambaCache())
    #expect(iterator.next() == 5)
    #expect(iterator.next() == 11)
    #expect(drafter.draftBlockCallCount == 0)
    #expect(
        iterator.passthroughReason
            == "hybrid cache no longer supports native speculative rewind")
}

@Test
func testMTPHybridK2ZeroAcceptanceRestoresCommittedBonusBoundary() throws {
    try assertMTPHybridK2AcceptanceLeavesScalarEquivalentCache(
        verifyTokens: [7, 8, 9],
        expectedAccepted: 0,
        expectedOffsetAfterRound: 4,
        expectedEmitted: [5, 7])
}

@Test
func testMTPHybridK2PartialAcceptanceReplaysAcceptedPrefix() throws {
    try assertMTPHybridK2AcceptanceLeavesScalarEquivalentCache(
        verifyTokens: [5, 7, 9],
        expectedAccepted: 1,
        expectedOffsetAfterRound: 5,
        expectedEmitted: [5, 5, 7])
}

@Test
func testMTPHybridK2FullAcceptanceKeepsVerifiedPrefix() throws {
    try assertMTPHybridK2AcceptanceLeavesScalarEquivalentCache(
        verifyTokens: [5, 5, 9],
        expectedAccepted: 2,
        expectedOffsetAfterRound: 6,
        expectedEmitted: [5, 5, 5, 9])
}

@Test
func testMTPHybridK2FinalizationRewindsUnemittedAcceptedLookahead() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 5, 5, 9, 0])
    main.maximumNativeTargetCacheRewind = 2
    let drafter = MockDrafter(draftedTokenValue: 5)
    let attention = CountingKVCache()
    let recurrent = MambaCache()
    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [attention, recurrent],
        parameters: GenerateParameters(maxTokens: 8, temperature: 0),
        blockSize: 3)

    #expect(iterator.next() == 5, "prepare-time bonus")
    #expect(iterator.next() == 5, "first accepted draft")
    #expect(attention.offset == 6)
    #expect(recurrent.offset == 0)

    iterator.finalizeGeneration()

    #expect(attention.offset == 5)
    #expect(recurrent.offset == 0)
    for state in recurrent.state {
        eval(state)
        #expect(state.item(Int.self) == 5)
    }
}

@Test
func testMTPHybridK2DiscardedAcceptedStopRestoresCommittedBonusBoundary() throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5, 5, 5, 9, 0])
    main.maximumNativeTargetCacheRewind = 2
    let drafter = MockDrafter(draftedTokenValue: 5)
    let attention = CountingKVCache()
    let recurrent = MambaCache()
    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [attention, recurrent],
        parameters: GenerateParameters(maxTokens: 8, temperature: 0),
        blockSize: 3)

    #expect(iterator.next() == 5, "prepare-time bonus")
    #expect(iterator.next() == 5, "accepted draft returned as a stop candidate")
    iterator.discardGeneratedToken()
    iterator.finalizeGeneration()

    #expect(attention.offset == 4)
    #expect(recurrent.offset == 0)
    for state in recurrent.state {
        eval(state)
        #expect(state.item(Int.self) == 4)
    }
}

private func assertMTPHybridK2AcceptanceLeavesScalarEquivalentCache(
    verifyTokens: [Int32],
    expectedAccepted: Int,
    expectedOffsetAfterRound: Int,
    expectedEmitted: [Int]
) throws {
    let main = MockMainModel(nextLogitTokens: [0, 0, 5] + verifyTokens + [0])
    main.maximumNativeTargetCacheRewind = 2
    let drafter = MockDrafter(draftedTokenValue: 5)
    let attention = CountingKVCache()
    let recurrent = MambaCache()
    var iterator = try MTPSpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray([Int32(1), 2, 3])),
        mainModel: main,
        drafter: drafter,
        mainCache: [attention, recurrent],
        parameters: GenerateParameters(maxTokens: 8, temperature: 0),
        blockSize: 3)

    var emitted: [Int] = []
    while emitted.count < expectedEmitted.count, let token = iterator.next() {
        emitted.append(token)
    }

    #expect(emitted == expectedEmitted)
    #expect(iterator.proposedCount == 2)
    #expect(iterator.acceptedCount == expectedAccepted)
    #expect(attention.offset == expectedOffsetAfterRound)
    #expect(recurrent.offset == 0)
    let recurrentState = recurrent.state
    #expect(recurrentState.count == 2)
    for state in recurrentState {
        eval(state)
        #expect(state.item(Int.self) == expectedOffsetAfterRound)
    }
}

// MARK: - Pending buffer drain order

@Test
func testMTPIteratorPendingBufferDrainOrder() throws {
    // Drafter returns [5, 5, 5]; main verifies [5, 5, 7, 9].
    // After the prepare-time bonus (5) is yielded first, speculateRound's
    // accept-prefix is positions 0..1 → [5, 5], then correction at position
    // 2 = 7. The pendingTokens order inside the round should be [5, 5, 7] —
    // main-model sequence order — not the drafter's [5, 5, 5]. Total
    // stream: [bonus=5, draft=5, draft=5, correction=7].
    let mainLogitTokens: [Int32] = [
        0, 0, 5,  // prefill follow-up picks bonus 5 (positions 0/1 unused, < vocab=20)
        5, 5, 7, 9,  // verify positions
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    let drafter = MockDrafter(draftedTokenValue: 5)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
    let cache = CountingKVCache()

    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 8, temperature: 0), blockSize: 4
    )

    let t0 = iter.next()
    let t1 = iter.next()
    let t2 = iter.next()
    let t3 = iter.next()
    #expect(t0 == 5)  // bonus from prepare
    #expect(t1 == 5)  // first accepted draft
    #expect(t2 == 5)  // second accepted draft
    #expect(t3 == 7)  // correction at the rejected position
    // proposedCount = 3 (numDraft); accepted = 2.
    #expect(iter.proposedCount == 3)
    #expect(iter.acceptedCount == 2)
    #expect(iter._verifierTokenReadbackCountForTesting == 1)
    #expect(cache.offset == 6, "the correction remains the next uncommitted target input")

    iter.finalizeGeneration()

    #expect(cache.offset == 7, "finalization commits exactly the four retained outputs")
}

@Test
func testMTPIteratorUsesSingleStepWhenOnlyOneTokenRemains() throws {
    // With maxTokens=2, the prepare-time bonus consumes the first output slot.
    // Only one slot remains, so a speculative round would be unable to emit
    // both an accepted draft and the verifier's correction/bonus token. The
    // iterator must take one normal main-model step instead.
    let mainLogitTokens: [Int32] = [
        0, 0, 5,  // prefill follow-up picks bonus 5
        11,  // single-token tail step
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    let drafter = MockDrafter(draftedTokenValue: 11)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
    let cache = CountingKVCache()

    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: [cache],
        parameters: GenerateParameters(maxTokens: 2), blockSize: 4,
        collectPhaseTelemetry: true
    )

    let tokens = [iter.next(), iter.next(), iter.next()]

    #expect(tokens[0] == 5)
    #expect(tokens[1] == 11)
    #expect(tokens[2] == nil)
    #expect(iter.tokenCount == 2)
    #expect(drafter.draftBlockCallCount == 0)
    #expect(iter.proposedCount == 0)
    #expect(iter.acceptedCount == 0)
    #expect(cache.offset == 4)
    #expect(iter.speculativeDecodingPhaseTelemetry.targetTailCount == 1)
    #expect(iter.speculativeDecodingPhaseTelemetry.targetTailSeconds >= 0)
    #expect(iter.speculativeDecodingPhaseTelemetry.draftBlockCount == 0)
    #expect(iter.speculativeDecodingPhaseTelemetry.targetVerificationCount == 0)

    iter.finalizeGeneration()

    #expect(cache.offset == 5, "the retained single-step tail must be terminally committed")
    #expect(iter.speculativeDecodingPhaseTelemetry.finalizationCount == 1)
}

// MARK: - LogitProcessor emit-only invariant

/// Records `didSample(token:)` calls so a test can verify which tokens the
/// processor actually observed. Pure value semantics — Swift struct value
/// copies (e.g., `var verifyProcessorCopy = processor` in `speculateRound`)
/// produce a separate `recordedTokens` backing via array copy-on-write.
private struct EmissionLog: LogitProcessor {
    var recordedTokens: [Int] = []

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray { logits }

    mutating func didSample(token: MLXArray) {
        recordedTokens.append(token.item(Int.self))
    }
}

/// Locks in the value-semantics invariant of `speculateRound`'s verify
/// loop: `var verifyProcessorCopy = processor` makes a Swift struct copy,
/// so `verifyProcessorCopy.didSample(...)` calls mutate the local copy
/// and do NOT propagate back to `self.processor`. The canonical processor
/// state at `self.processor` is updated only by the accept loop, which
/// runs over the actually-emitted tokens (accepted drafts + correction).
///
/// Test scenario: bs=4 (numDraft=3), drafter proposes [5, 5, 5], main
/// verifies and samples [5, 9, 1, 2] — only position 0 matches the draft.
/// accepted=1, correction=9, emitted=[bonus=5, draft=5, correction=9].
/// Verify loop's `didSample` fires four times (on the copy) for [5, 9, 1, 2].
/// Self.processor's `didSample` should fire exactly twice (for emitted [5, 9])
/// — NOT four times. The probe processor is installed AFTER init so the
/// prepare-time bonus is not recorded; the test asserts on speculation-
/// round emissions only.
@Test
func testMTPVerifyLoopDidSampleStaysScopedToLocalCopy() throws {
    let mainLogitTokens: [Int32] = [
        0, 0, 5,  // prefill follow-up picks bonus 5 (positions 0/1 unused, < vocab=20)
        5, 9, 1, 2,  // verify positions: only position 0 matches draft
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    let drafter = MockDrafter(draftedTokenValue: 5)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    // maxTokens larger than the test's emit budget so `speculateRound`'s
    // `numDraft = min(remaining, blockSize - 1)` doesn't get capped — we
    // need the full numDraft=3 verify pass to exercise the invariant
    // (4 verify-position samples vs only 2 emitted tokens). Control the
    // round count by manual `next()` calls instead of draining.
    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 8), blockSize: 4
    )

    // Install probe AFTER init so the prepare-time bonus didSample (which
    // happens inside init's call to prepare()) hits the parameters-derived
    // processor (nil here, since no penalties were configured) rather than
    // the EmissionLog. The probe records speculation-round emissions only.
    iter._setProcessorForTesting(EmissionLog())

    // Manual drain — exactly 3 calls to cover prepare bonus + 1 accepted
    // draft + correction. Stopping here avoids triggering a second round
    // (which would need more `mainLogitTokens` data and would just retest
    // the same invariant redundantly).
    let t0 = iter.next()
    let t1 = iter.next()
    let t2 = iter.next()
    #expect(t0 == 5, "prepare bonus")
    #expect(t1 == 5, "first accepted draft")
    #expect(t2 == 9, "correction at the first rejected position")
    #expect(iter.proposedCount == 3, "numDraft=3 verify samples expected")
    #expect(iter.acceptedCount == 1, "only draft[0] matched")
    #expect(
        iter._verifierTokenReadbackCountForTesting == 2,
        "a non-nil processor must retain sequential target-token selection")

    let log = iter._processorForTesting as? EmissionLog
    #expect(log != nil, "probe processor lost between install and drain")

    // self.processor's didSample fired exactly twice — for the accepted
    // draft and the correction — NOT for the three other verify-position
    // samples (9, 1, 2) which happened on the local copy. If a regression
    // ever removes the local-copy idiom, log.recordedTokens would gain
    // entries [9, 1, 2] from the rejected verify positions.
    #expect(
        log?.recordedTokens == [5, 9],
        "self.processor.recordedTokens=\(log?.recordedTokens ?? []) — expected [5, 9] (1 accepted draft + 1 correction). Verify-loop didSample is leaking from the copy into the canonical processor."
    )
}

// MARK: - sharedKV span across partial acceptance

/// The drafter-conditioning observable that output-equivalence testing
/// cannot see: the verify pass emits `mtpSharedKVStatesKey` spanning the
/// full verify chunk, before acceptance is known. After a partial
/// acceptance the snapshot must be rewound in lockstep with the main
/// cache, or the next round's `draftBlock` cross-attends over K/V rows of
/// tokens the main model just rejected. Byte-identity suites verify the
/// target path only; the state the drafter consumes has to be asserted
/// directly. (PR #308 review: discussion_r3391133046,
/// discussion_r3391147261.)
@Test
func testMTPSharedKVSpanTrimmedAfterPartialAcceptance() throws {
    // Round 1: drafter proposes [5, 5, 5]; main verifies [5, 7, ...] —
    // draft_1 accepted, mismatch at draft_2, so rejected = 2. True
    // sequence after round 1: 3 prompt + 1 bonus + 1 accepted = 5.
    // Round 2's draftBlock must receive sharedKV spanning 5 (== cache
    // offset == queryOffset), not the stale verify-chunk span 7.
    let mainLogitTokens: [Int32] = [
        0, 0, 5,  // prefill follow-up picks bonus 5 (placeholders < vocab=20)
        5, 7, 0, 0,  // round-1 verify: accept draft_1, reject at draft_2
        0, 0, 0, 0,  // round-2 verify: placeholders; round 2 only needs to run
    ]
    let main = MockMainModel(nextLogitTokens: mainLogitTokens)
    let drafter = MockDrafter(draftedTokenValue: 5)
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))

    var iter = try MTPSpeculativeTokenIterator(
        input: input, mainModel: main, drafter: drafter, mainCache: nil,
        parameters: GenerateParameters(maxTokens: 12), blockSize: 4
    )

    // Drain the prepare bonus and round 1, then check round-1 accounting
    // before triggering round 2.
    let t0 = iter.next()
    let t1 = iter.next()
    let t2 = iter.next()
    #expect(t0 == 5, "prepare bonus")
    #expect(t1 == 5, "round-1 accepted draft")
    #expect(t2 == 7, "round-1 correction at the first rejected position")
    #expect(iter.acceptedCount == 1, "round 1 must accept exactly 1 of 3")

    // One more `next()` starts round 2 — the draftBlock call under test.
    _ = iter.next()

    // Mock-fidelity gate: every emitted sharedKV span equals the mock
    // cache offset at emit time (prefill 3, round-1 verify 7, round-2
    // verify 9). If this fails, the mock's emission lost span accuracy
    // and the assertions below would be vacuous.
    #expect(main.emittedSharedKVSpans == [3, 7, 9])

    #expect(drafter.draftBlockCallCount == 2)
    // Round 1 drafts from the prefill emission: span 3 at offset 3.
    #expect(
        drafter.receivedSharedKVSpans.first == [
            "full_attention": 3, "sliding_attention": 3,
        ])
    #expect(drafter.receivedQueryOffsets.first == 3)
    // Round 2 is the regression surface: the round-1 verify emission
    // spanned 7; after the rewind trims the 2 rejected rows the drafter
    // must see span 5 == cache offset == queryOffset.
    #expect(drafter.receivedQueryOffsets.last == 5)
    #expect(
        drafter.receivedSharedKVSpans.last == [
            "full_attention": 5, "sliding_attention": 5,
        ],
        "drafter received spans \(drafter.receivedSharedKVSpans.last ?? [:]) — expected 5 (= cache offset). Span 7 means the emitted snapshot crossed the cache rewind untrimmed."
    )
}

// MARK: - trimSharedKVState contract

/// Builds a state carrying a sharedKV snapshot with distinguishable content
/// (values encode their flat index) so prefix byte-identity is checkable
/// after a trim, plus a hidden entry that the trim must never touch. The
/// two layer types get different head/dim shapes to catch a slice applied
/// to the wrong axis.
private func makeSharedKVState(span: Int) -> LMOutput.State {
    func arange(_ shape: [Int]) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray(Array(Int32(0) ..< Int32(count)), shape)
    }
    var state = LMOutput.State()
    state[mtpLastHiddenStatesKey] = arange([1, 2, 4])
    state[mtpSharedKVStatesKey] = [
        "full_attention": (arange([1, 1, span, 4]), arange([1, 1, span, 4])),
        "sliding_attention": (arange([1, 2, span, 2]), arange([1, 2, span, 2])),
    ]
    return state
}

@Test
func testTrimSharedKVStateZeroTokensIsNoOp() throws {
    var state: LMOutput.State? = makeSharedKVState(span: 6)
    trimSharedKVState(&state, numTokens: 0)

    let kv = try #require(state?[mtpSharedKVStatesKey])
    let original = try #require(makeSharedKVState(span: 6)[mtpSharedKVStatesKey])
    #expect(Set(kv.keys) == Set(original.keys))
    for (layerType, pair) in kv {
        let origPair = try #require(original[layerType])
        #expect(pair.0.shape == origPair.0.shape)
        #expect(allClose(pair.0, origPair.0, rtol: 0, atol: 0).item(Bool.self))
        #expect(allClose(pair.1, origPair.1, rtol: 0, atol: 0).item(Bool.self))
    }
}

@Test
func testTrimSharedKVStateTrimsTrailingRowsPreservingPrefix() throws {
    var state: LMOutput.State? = makeSharedKVState(span: 6)
    trimSharedKVState(&state, numTokens: 2)

    let kv = try #require(state?[mtpSharedKVStatesKey])
    let original = try #require(makeSharedKVState(span: 6)[mtpSharedKVStatesKey])
    #expect(Set(kv.keys) == Set(original.keys))
    for (layerType, pair) in kv {
        let origPair = try #require(original[layerType])
        // Span reduced by exactly the trim on the sequence axis only.
        #expect(pair.0.dim(-2) == 4)
        #expect(pair.1.dim(-2) == 4)
        #expect(pair.0.shape.dropLast(2).elementsEqual(origPair.0.shape.dropLast(2)))
        // Retained prefix is byte-identical to the original's prefix.
        #expect(
            allClose(pair.0, origPair.0[.ellipsis, ..<4, 0...], rtol: 0, atol: 0)
                .item(Bool.self))
        #expect(
            allClose(pair.1, origPair.1[.ellipsis, ..<4, 0...], rtol: 0, atol: 0)
                .item(Bool.self))
    }

    // The hidden entry rides along untouched — it needs no analogous trim
    // (the accepted-index slice selects by position) and the helper must
    // not disturb it.
    let hidden = try #require(state?[mtpLastHiddenStatesKey])
    let origHidden = try #require(makeSharedKVState(span: 6)[mtpLastHiddenStatesKey])
    #expect(hidden.shape == origHidden.shape)
    #expect(allClose(hidden, origHidden, rtol: 0, atol: 0).item(Bool.self))
}

@Test
func testTrimSharedKVStateNoOpOnNilStateAndAbsentKey() {
    // Nil state: must not crash, must stay nil.
    var nilState: LMOutput.State? = nil
    trimSharedKVState(&nilState, numTokens: 3)
    #expect(nilState == nil)

    // Present state without the sharedKV key (e.g. the quantization-onset
    // round): must not crash, must not invent the key, must leave sibling
    // keys alone.
    var keylessState: LMOutput.State? = LMOutput.State()
    keylessState?[mtpEmitFlagKey] = true
    trimSharedKVState(&keylessState, numTokens: 3)
    #expect(keylessState?[mtpSharedKVStatesKey] == nil)
    #expect(keylessState?[mtpEmitFlagKey] == true)
}

}
