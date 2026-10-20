import Foundation
import HarnessCore
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

// Cycle 275: per-row sampling (`temperature > 0`) in the dense continuous-batch runtime.
//
// `FixedLogitsLanguageModel` returns the SAME non-degenerate logits at every position whatever the
// input, so a sampled row's token stream is a pure function of its sampler (temperature, filters,
// seed) and the draw index. The logits are `log([0.4, 0.25, ...])` in the first eight columns and a
// large finite negative elsewhere (the runtime's raw-logit finite guard rejects -inf).

private let fixedProbabilities: [Float] = [0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.015, 0.005]
private let vocabularyWidth = 2_048
private let fillerLogit: Float = -10_000

private func fixedLogitsRow() -> MLXArray {
    var values = [Float](repeating: fillerLogit, count: vocabularyWidth)
    for (index, probability) in fixedProbabilities.enumerated() {
        values[index] = Float(log(Double(probability)))
    }
    return MLXArray(values).reshaped([1, vocabularyWidth])
}

private final class FixedLogitsLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads = [1]
    private let row = fixedLogitsRow()

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let values = inputs.asType(.float32).reshaped([inputs.dim(0), 1, inputs.dim(1), 1])
        guard let cache = cache?.first else {
            preconditionFailure("fixed-logits model requires one cache")
        }
        _ = cache.update(keys: values, values: values)
        return broadcast(
            row.reshaped([1, 1, vocabularyWidth]),
            to: [inputs.dim(0), inputs.dim(1), vocabularyWidth])
    }
}

/// Cache-sensitive one-hot model (a copy of the greedy fixture in
/// `DenseContinuousBatchRuntimeTests`): the greedy next token is `prefix sum + 1`. When
/// `poisonInput` is set, the row at that input token is all-NaN (a full-shaped array: a scalar NaN
/// constant in a compiled step is inlined as an undeclared `nan` in the Metal source).
private final class OneHotPrefixSumLanguageModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads = [1]
    private let poisonInput: Int32?

    init(poisonInput: Int32? = nil) { self.poisonInput = poisonInput }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let values = inputs.asType(.float32).reshaped([inputs.dim(0), 1, inputs.dim(1), 1])
        guard let cache = cache?.first else {
            preconditionFailure("one-hot model requires one cache")
        }
        let (keys, _) = cache.update(keys: values, values: values)
        let flatInputs = inputs.asType(.int32)
        let historyBeforeChunk = keys.sum(axes: [1, 2, 3]).asType(.int32) - flatInputs.sum(axis: 1)
        let target = (
            historyBeforeChunk.reshaped([inputs.dim(0), 1]) + flatInputs.cumsum(axis: 1) + 1
        ).expandedDimensions(axis: -1)
        let vocabulary = MLXArray(Int32(0) ..< Int32(vocabularyWidth))
            .reshaped([1, 1, vocabularyWidth])
        let finite = (target .== vocabulary).asType(.float32) * 100
        guard let poisonInput else { return finite }
        let invalid = (flatInputs .== poisonInput).expandedDimensions(axis: -1)
        return which(
            invalid,
            MLXArray.full(
                [inputs.dim(0), inputs.dim(1), vocabularyWidth], values: MLXArray(Float.nan)),
            finite)
    }
}

/// Test double for `ContinuousBatchRowTokenMask`: the allowed ids for the row's NEXT token are
/// `sets[advanced.count % sets.count]` (so the allowed set changes every step), and every
/// `advance` is recorded in order.
private final class SteppedMask: ContinuousBatchRowTokenMask, @unchecked Sendable {
    private let lock = NSLock()
    let sets: [[Int]]
    private var recorded: [Int] = []

    init(sets: [[Int]]) { self.sets = sets }

    var advanced: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    func allowedTokenBitset() -> [UInt64]? {
        lock.lock()
        defer { lock.unlock() }
        let ids = sets[recorded.count % sets.count]
        var words = [UInt64](repeating: 0, count: (ids.max() ?? 0) / 64 + 1)
        for id in ids { words[id >> 6] |= UInt64(1) << UInt64(id & 63) }
        return words
    }

    func advance(token: Int) {
        lock.lock()
        defer { lock.unlock() }
        recorded.append(token)
    }
}

private struct EmptyDrafter: SpecDrafter {
    func propose(context _: [Int], maxDraft _: Int) -> [Int] { [] }
}

final class DenseContinuousBatchRuntimeSamplingTests: XCTestCase {
    // MARK: - Harness

    private func makeRuntime(
        model: any LanguageModel = FixedLogitsLanguageModel(),
        soloPLDConfiguration: SpecDecodeConfig? = nil
    ) throws -> DenseContinuousBatchRuntime {
        try DenseContinuousBatchRuntime(
            testing: model,
            allocationChunk: 4,
            maxContextTokens: 512,
            initialDecodeReserve: 384,
            soloPLDConfiguration: soloPLDConfiguration)
    }

    private func admit(
        _ runtime: DenseContinuousBatchRuntime,
        id: UInt64,
        prompt: [Int] = [10],
        sampling: ContinuousBatchRowSampling?,
        requestsSpeculation: Bool = false,
        maxOutputTokens: Int = 64,
        logitBias: [Int: Float] = [:],
        tokenMask: (any ContinuousBatchRowTokenMask)? = nil
    ) throws {
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(id),
                submission: ContinuousBatchSubmission(
                    promptTokens: prompt,
                    maxOutputTokens: maxOutputTokens,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: requestsSpeculation,
                    logitBias: logitBias,
                    tokenMask: tokenMask,
                    sampling: sampling)),
        ])
    }

    private func prefill(
        _ runtime: DenseContinuousBatchRuntime,
        id: UInt64,
        prompt: [Int] = [10],
        maxOutputTokens: Int = 64
    ) throws {
        try runtime.prefill(
            ContinuousBatchRuntimePrefill(
                id: BatchRequestID(id),
                startToken: 0,
                tokens: prompt,
                isFinal: true,
                totalPromptTokens: prompt.count,
                maxOutputTokens: maxOutputTokens))
    }

    private func admitAndPrefill(
        _ runtime: DenseContinuousBatchRuntime,
        id: UInt64,
        prompt: [Int] = [10],
        sampling: ContinuousBatchRowSampling?,
        logitBias: [Int: Float] = [:],
        tokenMask: (any ContinuousBatchRowTokenMask)? = nil
    ) throws {
        try admit(
            runtime, id: id, prompt: prompt, sampling: sampling, logitBias: logitBias,
            tokenMask: tokenMask)
        try prefill(runtime, id: id, prompt: prompt)
    }

    private func soloStep(
        _ runtime: DenseContinuousBatchRuntime, _ id: UInt64, index: Int
    ) throws -> Int {
        let identifier = BatchRequestID(id)
        if index % 4 == 3 {
            return try runtime.decode(.drainSoloPipeline(identifier))[0].tokens[0]
        }
        return try runtime.decode(.solo(identifier, speculationAllowed: false))[0].tokens[0]
    }

    private func soloTokens(
        sampling: ContinuousBatchRowSampling?,
        prompt: [Int] = [10],
        model: any LanguageModel = FixedLogitsLanguageModel(),
        logitBias: [Int: Float] = [:],
        tokenMask: (any ContinuousBatchRowTokenMask)? = nil,
        count: Int
    ) throws -> [Int] {
        let runtime = try makeRuntime(model: model)
        try admitAndPrefill(
            runtime, id: 1, prompt: prompt, sampling: sampling, logitBias: logitBias,
            tokenMask: tokenMask)
        return try (0 ..< count).map { try soloStep(runtime, 1, index: $0) }
    }

    /// Shared batch over ids 1...n (one prompt each); tokens per id over `count` shared steps.
    private func batchTokens(
        samplings: [ContinuousBatchRowSampling?],
        prompts: [[Int]]? = nil,
        biases: [[Int: Float]]? = nil,
        masks: [(any ContinuousBatchRowTokenMask)?]? = nil,
        model: any LanguageModel = FixedLogitsLanguageModel(),
        count: Int
    ) throws -> [UInt64: [Int]] {
        let runtime = try makeRuntime(model: model)
        for (index, sampling) in samplings.enumerated() {
            try admitAndPrefill(
                runtime, id: UInt64(index + 1), prompt: prompts?[index] ?? [10],
                sampling: sampling, logitBias: biases?[index] ?? [:],
                tokenMask: masks?[index] ?? nil)
        }
        let ids = samplings.indices.map { BatchRequestID(UInt64($0 + 1)) }
        var tokens: [UInt64: [Int]] = [:]
        for _ in 0 ..< count {
            for result in try runtime.decode(.batch(ids, speculationAllowed: false)) {
                tokens[result.id.rawValue, default: []].append(contentsOf: result.tokens)
            }
        }
        return tokens
    }

    /// `count` successive draws of a fresh sampler on the fixed logits: the reference sequence.
    private func referenceDraws(
        _ sampling: ContinuousBatchRowSampling, count: Int
    ) -> [Int] {
        let sampler = TopPSampler(
            temperature: sampling.temperature, topP: sampling.topP, topK: sampling.topK,
            minP: sampling.minP, seed: sampling.seed)
        let row = fixedLogitsRow()
        return (0 ..< count).map { _ in
            sampler.sample(logits: row).asType(.int32).reshaped([1]).item(Int.self)
        }
    }

    /// Chi-square critical values at p = 0.001, df 1...7.
    private static let criticalValue: [Int: Double] = [
        1: 10.828, 2: 13.816, 3: 16.266, 4: 18.467, 5: 20.515, 6: 22.458, 7: 24.322,
    ]

    private func chiSquare(counts: [Int], probabilities: [Double]) -> (statistic: Double, df: Int) {
        let total = Double(counts.reduce(0, +))
        var observed = counts.map(Double.init)
        var expected = probabilities.map { $0 * total }
        // Pool the tail until every cell expects at least 5.
        while expected.count > 2, expected[expected.count - 1] < 5 {
            expected[expected.count - 2] += expected.removeLast()
            observed[observed.count - 2] += observed.removeLast()
        }
        var statistic = 0.0
        for (o, e) in zip(observed, expected) { statistic += (o - e) * (o - e) / e }
        return (statistic, expected.count - 1)
    }

    /// Token counts over `groups * 4` seeded rows (seeds 1...), `steps` draws each, run as
    /// shared batches of four rows.
    private func sampledCounts(
        temperature: Float, topK: Int = 0, groups: Int, steps: Int = 50
    ) throws -> [Int] {
        var counts = [Int](repeating: 0, count: vocabularyWidth)
        for group in 0 ..< groups {
            let samplings = (0 ..< 4).map { member in
                ContinuousBatchRowSampling(
                    temperature: temperature, topK: topK,
                    seed: UInt64(group * 4 + member + 1))
            }
            let tokens = try batchTokens(samplings: samplings, count: steps)
            for row in tokens.values { for token in row { counts[token] += 1 } }
        }
        return counts
    }

    // MARK: - S1: distribution

    func testS1SampledTokensFollowSoftmaxOfLogitsOverTemperatureAtOneAndHalf() throws {
        for temperature in [Float(1), Float(0.5)] {
            let counts = try sampledCounts(temperature: temperature, groups: 10)
            XCTAssertEqual(counts.reduce(0, +), 2_000)
            XCTAssertEqual(
                counts[fixedProbabilities.count...].reduce(0, +), 0,
                "a token outside the eight real columns was drawn at T=\(temperature)")
            let weights = fixedProbabilities.map { pow(Double($0), 1 / Double(temperature)) }
            let normalizer = weights.reduce(0, +)
            let result = chiSquare(
                counts: Array(counts[0 ..< fixedProbabilities.count]),
                probabilities: weights.map { $0 / normalizer })
            let bound = try XCTUnwrap(Self.criticalValue[result.df])
            XCTAssertLessThan(
                result.statistic, bound,
                "chi-square \(result.statistic) (df \(result.df)) at T=\(temperature); "
                    + "counts \(counts[0 ..< fixedProbabilities.count])")
        }
    }

    func testS1TopKTwoOnlyEverDrawsTheTopTwoIds() throws {
        let counts = try sampledCounts(temperature: 1, topK: 2, groups: 3, steps: 50)
        XCTAssertEqual(counts.reduce(0, +), 600)
        XCTAssertEqual(counts[2...].reduce(0, +), 0, "top_k=2 drew an id outside {0, 1}")
        XCTAssertGreaterThan(counts[0], 0)
        XCTAssertGreaterThan(counts[1], 0)
    }

    // MARK: - S2: seeded reference

    func testS2SeededSoloRowEqualsAFreshSamplerOneCallPerToken() throws {
        let sampling = ContinuousBatchRowSampling(temperature: 0.8, topP: 0.95, seed: 42)
        let expected = referenceDraws(sampling, count: 12)
        XCTAssertGreaterThan(Set(expected).count, 1, "reference must not be degenerate")
        XCTAssertEqual(try soloTokens(sampling: sampling, count: 12), expected)
    }

    func testS2SeededBatchRowEqualsAFreshSamplerBesideAGreedyPartner() throws {
        let sampling = ContinuousBatchRowSampling(temperature: 1, seed: 5)
        let expected = referenceDraws(sampling, count: 10)
        let besideGreedy = try batchTokens(samplings: [sampling, nil], count: 10)
        XCTAssertEqual(besideGreedy[1], expected)
        XCTAssertEqual(besideGreedy[2], [Int](repeating: 0, count: 10))
        let sampledSecond = try batchTokens(samplings: [nil, sampling], count: 10)
        XCTAssertEqual(sampledSecond[2], expected)
    }

    func testS2SeededRowKeepsItsSamplerAcrossScalarToBatchToScalar() throws {
        let sampling = ContinuousBatchRowSampling(temperature: 1, seed: 11)
        let expected = referenceDraws(sampling, count: 12)
        let runtime = try makeRuntime()
        try admitAndPrefill(runtime, id: 1, sampling: sampling)
        try admitAndPrefill(runtime, id: 2, sampling: nil)
        var tokens: [Int] = []
        // Scalar: two pipelined steps and a drain (a batch refuses a pending lookahead).
        tokens.append(try soloStep(runtime, 1, index: 0))
        tokens.append(try soloStep(runtime, 1, index: 1))
        tokens.append(try runtime.decode(.drainSoloPipeline(BatchRequestID(1)))[0].tokens[0])
        // Batch beside a greedy partner.
        let ids = [BatchRequestID(1), BatchRequestID(2)]
        for _ in 0 ..< 4 {
            let results = try runtime.decode(.batch(ids, speculationAllowed: false))
            tokens.append(try XCTUnwrap(results.first { $0.id.rawValue == 1 }).tokens[0])
        }
        // Back to scalar once the partner leaves.
        runtime.remove(BatchRequestID(2))
        for index in 0 ..< 5 { tokens.append(try soloStep(runtime, 1, index: index)) }
        XCTAssertEqual(tokens, expected)
    }

    // MARK: - S3: partner independence

    func testS3SeededRowIsIdenticalSoloBesideGreedyAndBesideAnotherSampledRow() throws {
        let seven = ContinuousBatchRowSampling(temperature: 1, seed: 7)
        let eight = ContinuousBatchRowSampling(temperature: 1, seed: 8)
        let solo = try soloTokens(sampling: seven, count: 10)
        let besideGreedy = try batchTokens(samplings: [seven, nil], count: 10)
        let besideSampled = try batchTokens(samplings: [seven, eight], count: 10)
        XCTAssertEqual(besideGreedy[1], solo)
        XCTAssertEqual(besideSampled[1], solo)
        XCTAssertEqual(besideSampled[2], referenceDraws(eight, count: 10))
        XCTAssertNotEqual(besideSampled[2], solo, "seeds 7 and 8 must give different streams")
    }

    func testS3GreedyPartnerIsByteEqualToItsSoloControlBesideASampledRow() throws {
        let sampled = ContinuousBatchRowSampling(temperature: 1, seed: 7)
        // The cache-sensitive one-hot model makes the greedy stream input-dependent (11, 22, 44...).
        let control = try soloTokens(
            sampling: nil, prompt: [50], model: OneHotPrefixSumLanguageModel(), count: 5)
        XCTAssertEqual(control, [51, 102, 204, 408, 816])
        let beside = try batchTokens(
            samplings: [sampled, nil], prompts: [[10], [50]],
            model: OneHotPrefixSumLanguageModel(), count: 5)
        XCTAssertEqual(beside[2], control)
        let sampledFirstGreedySecond = try batchTokens(
            samplings: [nil, sampled], prompts: [[50], [10]],
            model: OneHotPrefixSumLanguageModel(), count: 5)
        XCTAssertEqual(sampledFirstGreedySecond[1], control)
        // And on the fixed-logits model the greedy partner is the argmax column throughout.
        let fixed = try batchTokens(samplings: [sampled, nil], count: 5)
        XCTAssertEqual(fixed[2], [0, 0, 0, 0, 0])
    }

    // MARK: - S4: speculation refused

    func testS4SamplingWithSpeculationIsRefusedAndAdmitsNothing() throws {
        let runtime = try makeRuntime(
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: EmptyDrafter(),
                maxDraft: 3,
                gate: PLDGate(window: 8, minimumSamples: 4, minAcceptPerStep: 0.5, cooldown: 8),
                lookback: 32,
                compiledVerify: true))
        XCTAssertThrowsError(
            try admit(
                runtime, id: 1,
                sampling: ContinuousBatchRowSampling(temperature: 1, seed: 1),
                requestsSpeculation: true)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .samplingWithSpeculation(BatchRequestID(1)))
        }
        let refused = runtime.resourceSnapshot()
        XCTAssertEqual(refused?.reservedKVBytes, 0)
        XCTAssertEqual(refused?.speculation?.requestedRequests, 0)
        XCTAssertEqual(refused?.speculation?.activeSessions, 0)

        // Nothing was left behind: the same id admits as an unsampled speculative row.
        try admit(runtime, id: 1, sampling: nil, requestsSpeculation: true)
        XCTAssertGreaterThan(runtime.resourceSnapshot()?.reservedKVBytes ?? 0, 0)
    }

    // MARK: - S5: poison

    private static let topOne = ContinuousBatchRowSampling(temperature: 1, topK: 1, seed: 3)

    /// Solo: input 11 poisons the row. The prefill-final token (prompt [10]) is 11 and sound; the
    /// following token reads back the -1 sentinel.
    private func soloPoisonError(sampling: ContinuousBatchRowSampling?) throws
        -> DenseContinuousBatchRuntimeError?
    {
        let runtime = try makeRuntime(model: OneHotPrefixSumLanguageModel(poisonInput: 11))
        try admitAndPrefill(runtime, id: 1, sampling: sampling)
        XCTAssertEqual(try soloStep(runtime, 1, index: 0), 11)
        do {
            _ = try soloStep(runtime, 1, index: 1)
            return nil
        } catch {
            return error as? DenseContinuousBatchRuntimeError
        }
    }

    private func batchPoisonError(sampling: ContinuousBatchRowSampling?) throws
        -> DenseContinuousBatchRuntimeError?
    {
        let runtime = try makeRuntime(model: OneHotPrefixSumLanguageModel(poisonInput: 11))
        try admitAndPrefill(runtime, id: 1, sampling: sampling)
        try admitAndPrefill(runtime, id: 2, sampling: nil)
        let ids = [BatchRequestID(1), BatchRequestID(2)]
        let first = try runtime.decode(.batch(ids, speculationAllowed: false))
        XCTAssertEqual(first.map { $0.tokens[0] }, [11, 11])
        do {
            _ = try runtime.decode(.batch(ids, speculationAllowed: false))
            return nil
        } catch {
            return error as? DenseContinuousBatchRuntimeError
        }
    }

    func testS5SampledRowFailsClosedOnNonFiniteRawLogitsExactlyLikeAGreedyRow() throws {
        let sentinel = DenseContinuousBatchRuntimeError.invalidTokenID(BatchRequestID(1), -1)
        XCTAssertEqual(try soloPoisonError(sampling: nil), sentinel, "greedy solo control")
        XCTAssertEqual(try batchPoisonError(sampling: nil), sentinel, "greedy batch control")
        XCTAssertEqual(try soloPoisonError(sampling: Self.topOne), sentinel, "sampled solo")
        XCTAssertEqual(try batchPoisonError(sampling: Self.topOne), sentinel, "sampled batch")
    }

    func testS5SampledPrefillFinalOnNonFiniteLogitsFailsLikeAGreedyPrefill() throws {
        func prefillError(sampling: ContinuousBatchRowSampling?) throws
            -> DenseContinuousBatchRuntimeError?
        {
            let runtime = try makeRuntime(model: OneHotPrefixSumLanguageModel(poisonInput: 11))
            try admit(runtime, id: 1, prompt: [11], sampling: sampling)
            do {
                try prefill(runtime, id: 1, prompt: [11])
                return nil
            } catch {
                return error as? DenseContinuousBatchRuntimeError
            }
        }
        let sentinel = DenseContinuousBatchRuntimeError.invalidTokenID(BatchRequestID(1), -1)
        XCTAssertEqual(try prefillError(sampling: nil), sentinel, "greedy control")
        XCTAssertEqual(try prefillError(sampling: Self.topOne), sentinel, "sampled")
    }

    // MARK: - S6: token mask and logit bias on sampled rows

    /// Four steps with a different allowed set each; none contains the argmax id 0, and every set
    /// holds several of the eight real columns, so a sampler that ignores the mask draws 0 (p 0.4).
    private static let maskSets: [[Int]] = [[1, 2, 3], [4, 5], [1, 3, 6], [2, 7, 4]]

    private func assertTokensFollowMask(
        _ tokens: [Int]?, _ mask: SteppedMask, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        guard let tokens else {
            XCTFail("\(label): no tokens", file: file, line: line)
            return
        }
        for (step, token) in tokens.enumerated() {
            XCTAssertTrue(
                Self.maskSets[step % Self.maskSets.count].contains(token),
                "\(label): step \(step) emitted \(token), outside \(Self.maskSets[step % 4])",
                file: file, line: line)
        }
        XCTAssertEqual(mask.advanced, tokens, "\(label): advance() must see the emitted tokens",
            file: file, line: line)
    }

    func testS6SampledSoloRowStaysInsideItsTokenMaskAndAdvancesWithTheEmittedTokens() throws {
        var distinct = Set<Int>()
        for seed in UInt64(1) ... 4 {
            let mask = SteppedMask(sets: Self.maskSets)
            let tokens = try soloTokens(
                sampling: ContinuousBatchRowSampling(temperature: 1, seed: seed),
                tokenMask: mask, count: 12)
            assertTokensFollowMask(tokens, mask, "solo seed \(seed)")
            distinct.formUnion(tokens)
        }
        XCTAssertGreaterThan(distinct.count, 3, "the masked draws must still be sampled")
    }

    func testS6SampledPrefillFinalTokenIsDrawnInsideTheMask() throws {
        var distinct = Set<Int>()
        for seed in UInt64(1) ... 16 {
            let mask = SteppedMask(sets: [[1, 2, 3]])
            let tokens = try soloTokens(
                sampling: ContinuousBatchRowSampling(temperature: 1, seed: seed),
                tokenMask: mask, count: 1)
            XCTAssertTrue(
                [1, 2, 3].contains(tokens[0]),
                "seed \(seed): prefill-final token \(tokens[0]) is outside the mask")
            XCTAssertEqual(mask.advanced, tokens)
            distinct.insert(tokens[0])
        }
        XCTAssertGreaterThan(distinct.count, 1)
    }

    func testS6SampledBatchRowsStayInsideTheirTokenMasksWhateverTheirPartner() throws {
        for seed in UInt64(1) ... 3 {
            let sampling = ContinuousBatchRowSampling(temperature: 1, seed: seed)
            // Masked sampled row first, unmasked greedy partner second.
            var mask = SteppedMask(sets: Self.maskSets)
            var tokens = try batchTokens(samplings: [sampling, nil], masks: [mask, nil], count: 12)
            assertTokensFollowMask(tokens[1], mask, "first beside greedy, seed \(seed)")
            XCTAssertEqual(tokens[2], [Int](repeating: 0, count: 12))
            // Masked sampled row second.
            mask = SteppedMask(sets: Self.maskSets)
            tokens = try batchTokens(samplings: [nil, sampling], masks: [nil, mask], count: 12)
            assertTokensFollowMask(tokens[2], mask, "second beside greedy, seed \(seed)")
            // Two masked sampled rows with different mask scripts.
            mask = SteppedMask(sets: Self.maskSets)
            let other = SteppedMask(sets: [[5, 6], [7]])
            tokens = try batchTokens(
                samplings: [sampling, ContinuousBatchRowSampling(temperature: 1, seed: seed + 100)],
                masks: [mask, other], count: 12)
            assertTokensFollowMask(tokens[1], mask, "first of two sampled, seed \(seed)")
            for (step, token) in (tokens[2] ?? []).enumerated() {
                XCTAssertEqual(
                    [[5, 6], [7]][step % 2].contains(token), true,
                    "second of two sampled, seed \(seed): step \(step) emitted \(token)")
            }
            XCTAssertEqual(other.advanced, tokens[2])
        }
    }

    func testS6LogitBiasOnTheArgmaxColumnKeepsItOutOfASampledRowsDraws() throws {
        let bias: [Int: Float] = [0: -100]
        var unbiased = 0
        var biased = 0
        var biasedBatch = 0
        for seed in UInt64(1) ... 4 {
            let sampling = ContinuousBatchRowSampling(temperature: 1, seed: seed)
            unbiased += try soloTokens(sampling: sampling, count: 40).filter { $0 == 0 }.count
            // The first token of each stream is the prefill-final token.
            biased += try soloTokens(sampling: sampling, logitBias: bias, count: 40)
                .filter { $0 == 0 }.count
        }
        let batch = try batchTokens(
            samplings: (1 ... 4).map { ContinuousBatchRowSampling(temperature: 1, seed: UInt64($0)) },
            biases: Array(repeating: bias, count: 4), count: 40)
        for row in batch.values { biasedBatch += row.filter { $0 == 0 }.count }
        XCTAssertEqual(batch.values.map(\.count), [40, 40, 40, 40])
        XCTAssertGreaterThan(unbiased, 0, "control: without the bias column 0 is drawn")
        XCTAssertEqual(biased, 0, "solo: column 0 was drawn despite logit_bias -100")
        XCTAssertEqual(biasedBatch, 0, "batch: column 0 was drawn despite logit_bias -100")
    }

    func testS6MaskedOrBiasedGreedyPartnerBesideASampledRowEqualsItsSoloControl() throws {
        let sampled = ContinuousBatchRowSampling(temperature: 1, seed: 9)

        // Masked greedy partner: the highest-probability allowed id per step.
        let maskedControl = try soloTokens(
            sampling: nil, tokenMask: SteppedMask(sets: Self.maskSets), count: 8)
        XCTAssertEqual(maskedControl, [1, 4, 1, 2, 1, 4, 1, 2])
        for position in [0, 1] {
            let partnerMask = SteppedMask(sets: Self.maskSets)
            let samplings: [ContinuousBatchRowSampling?] =
                position == 0 ? [nil, sampled] : [sampled, nil]
            let masks: [(any ContinuousBatchRowTokenMask)?] = [nil, nil]
            var arranged = masks
            arranged[position] = partnerMask
            let tokens = try batchTokens(samplings: samplings, masks: arranged, count: 8)
            XCTAssertEqual(tokens[UInt64(position + 1)], maskedControl, "masked partner at \(position)")
            XCTAssertEqual(partnerMask.advanced, maskedControl)
        }

        // Biased greedy partner: +10 on id 3 beats the argmax.
        let bias: [Int: Float] = [3: 10]
        let biasedControl = try soloTokens(sampling: nil, logitBias: bias, count: 8)
        XCTAssertEqual(biasedControl, [Int](repeating: 3, count: 8))
        let biasedFirst = try batchTokens(
            samplings: [nil, sampled], biases: [bias, [:]], count: 8)
        XCTAssertEqual(biasedFirst[1], biasedControl)
        let biasedSecond = try batchTokens(
            samplings: [sampled, nil], biases: [[:], bias], count: 8)
        XCTAssertEqual(biasedSecond[2], biasedControl)
    }
}
