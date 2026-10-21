import Foundation
import HarnessCore
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

// Cycle 277: presence / frequency / repetition penalties in the dense continuous-batch runtime.
//
// `FixedLogitsLanguageModel` returns the SAME logits at every position whatever the input, so a
// row's token stream is a pure function of its penalties, its prompt (which seeds the processor)
// and its sampler. The reference loop below is built independently from the vendored
// `GenerateParameters(...).processor()` plus `TopPSampler` / `argMax`, one call per token; it
// never touches runtime code. The logits are `log([0.4, 0.25, ...])` in the first eight columns
// (all negative) and a large finite negative elsewhere, so a repetition penalty (x2) or an
// additive penalty on a repeated token genuinely changes the argmax.

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
    private let row: MLXArray
    private let poisonInput: Int32?

    /// `row` overrides the default fixed row (any `[1, vocabularyWidth]` array, e.g. bf16). When
    /// `poisonInput` is set, the logits of a position whose INPUT token equals it are all-NaN.
    init(row: MLXArray? = nil, poisonInput: Int32? = nil) {
        self.row = row ?? fixedLogitsRow()
        self.poisonInput = poisonInput
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let values = inputs.asType(.float32).reshaped([inputs.dim(0), 1, inputs.dim(1), 1])
        guard let cache = cache?.first else {
            preconditionFailure("fixed-logits model requires one cache")
        }
        _ = cache.update(keys: values, values: values)
        let fixed = broadcast(
            row.reshaped([1, 1, vocabularyWidth]),
            to: [inputs.dim(0), inputs.dim(1), vocabularyWidth])
        guard let poisonInput else { return fixed }
        let invalid = (inputs.asType(.int32) .== poisonInput).expandedDimensions(axis: -1)
        return which(
            invalid,
            MLXArray.full(
                [inputs.dim(0), inputs.dim(1), vocabularyWidth],
                values: MLXArray(Float.nan).asType(fixed.dtype)),
            fixed)
    }
}

/// Cache-sensitive one-hot model (a copy of the fixture in the sampling tests): the greedy next
/// token is `prefix sum + 1`. When `poisonInput` is set, the row at that input token is all-NaN
/// (a full-shaped array: a scalar NaN constant in a compiled step is inlined as an undeclared
/// `nan` in the Metal source).
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

/// Fixed allowed set (the same for every step), recording every `advance`.
private final class FixedMask: ContinuousBatchRowTokenMask, @unchecked Sendable {
    private let lock = NSLock()
    let allowed: [Int]
    private var recorded: [Int] = []

    init(allowed: [Int]) { self.allowed = allowed }

    var advanced: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    func allowedTokenBitset() -> [UInt64]? {
        var words = [UInt64](repeating: 0, count: (allowed.max() ?? 0) / 64 + 1)
        for id in allowed { words[id >> 6] |= UInt64(1) << UInt64(id & 63) }
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

final class DenseContinuousBatchRuntimePenaltyTests: XCTestCase {
    // MARK: - Harness

    private static let repetition = ContinuousBatchRowPenalties(repetition: 2)
    private static let presence = ContinuousBatchRowPenalties(presence: 1.5)
    private static let frequency = ContinuousBatchRowPenalties(frequency: 1)

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
        penalties: ContinuousBatchRowPenalties?,
        sampling: ContinuousBatchRowSampling? = nil,
        requestsSpeculation: Bool = false,
        logitBias: [Int: Float] = [:],
        tokenMask: (any ContinuousBatchRowTokenMask)? = nil
    ) throws {
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(id),
                submission: ContinuousBatchSubmission(
                    promptTokens: prompt,
                    maxOutputTokens: 64,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: requestsSpeculation,
                    logitBias: logitBias,
                    tokenMask: tokenMask,
                    sampling: sampling,
                    penalties: penalties)),
        ])
    }

    private func prefill(
        _ runtime: DenseContinuousBatchRuntime, id: UInt64, prompt: [Int]
    ) throws {
        try runtime.prefill(
            ContinuousBatchRuntimePrefill(
                id: BatchRequestID(id),
                startToken: 0,
                tokens: prompt,
                isFinal: true,
                totalPromptTokens: prompt.count,
                maxOutputTokens: 64))
    }

    private func admitAndPrefill(
        _ runtime: DenseContinuousBatchRuntime,
        id: UInt64,
        prompt: [Int] = [10],
        penalties: ContinuousBatchRowPenalties?,
        sampling: ContinuousBatchRowSampling? = nil,
        logitBias: [Int: Float] = [:],
        tokenMask: (any ContinuousBatchRowTokenMask)? = nil
    ) throws {
        try admit(
            runtime, id: id, prompt: prompt, penalties: penalties, sampling: sampling,
            logitBias: logitBias, tokenMask: tokenMask)
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
        penalties: ContinuousBatchRowPenalties?,
        sampling: ContinuousBatchRowSampling? = nil,
        prompt: [Int] = [10],
        model: any LanguageModel = FixedLogitsLanguageModel(),
        logitBias: [Int: Float] = [:],
        tokenMask: (any ContinuousBatchRowTokenMask)? = nil,
        count: Int
    ) throws -> [Int] {
        let runtime = try makeRuntime(model: model)
        try admitAndPrefill(
            runtime, id: 1, prompt: prompt, penalties: penalties, sampling: sampling,
            logitBias: logitBias, tokenMask: tokenMask)
        return try (0 ..< count).map { try soloStep(runtime, 1, index: $0) }
    }

    /// Shared batch over ids 1...n; tokens per id over `count` shared steps.
    private func batchTokens(
        penalties: [ContinuousBatchRowPenalties?],
        samplings: [ContinuousBatchRowSampling?]? = nil,
        prompts: [[Int]]? = nil,
        biases: [[Int: Float]]? = nil,
        model: any LanguageModel = FixedLogitsLanguageModel(),
        count: Int
    ) throws -> [UInt64: [Int]] {
        let runtime = try makeRuntime(model: model)
        for (index, row) in penalties.enumerated() {
            try admitAndPrefill(
                runtime, id: UInt64(index + 1), prompt: prompts?[index] ?? [10],
                penalties: row, sampling: samplings?[index] ?? nil,
                logitBias: biases?[index] ?? [:])
        }
        let ids = penalties.indices.map { BatchRequestID(UInt64($0 + 1)) }
        var tokens: [UInt64: [Int]] = [:]
        for _ in 0 ..< count {
            for result in try runtime.decode(.batch(ids, speculationAllowed: false)) {
                tokens[result.id.rawValue, default: []].append(contentsOf: result.tokens)
            }
        }
        return tokens
    }

    // MARK: - Independent reference

    /// The scalar route's behavior on the fixed logits: a fresh vendored processor (as
    /// `MLXDecoder.setPenalties` builds it) seeded with the prompt, `process`, then the additive
    /// row, then a fresh sampler's single draw (or argmax), then `didSample`. Nothing here calls
    /// runtime code. `biasBeforeProcessor` builds the WRONG order (`process(raw + additive)`).
    private func referenceTokens(
        _ penalties: ContinuousBatchRowPenalties,
        prompt: [Int] = [10],
        sampling: ContinuousBatchRowSampling? = nil,
        additive: MLXArray? = nil,
        biasBeforeProcessor: Bool = false,
        count: Int
    ) -> [Int] {
        var processor = GenerateParameters(
            repetitionPenalty: penalties.repetition,
            presencePenalty: penalties.presence,
            frequencyPenalty: penalties.frequency
        ).processor()
        processor?.prompt(MLXArray(prompt))
        let sampler = sampling.map {
            TopPSampler(
                temperature: $0.temperature, topP: $0.topP, topK: $0.topK, minP: $0.minP,
                seed: $0.seed)
        }
        let raw = fixedLogitsRow()
        return (0 ..< count).map { _ in
            var logits = raw
            if biasBeforeProcessor, let additive { logits = logits + additive }
            logits = processor?.process(logits: logits) ?? logits
            if !biasBeforeProcessor, let additive { logits = logits + additive }
            let token = sampler?.sample(logits: logits) ?? argMax(logits, axis: -1)
            processor?.didSample(token: token)
            return token.asType(.int32).reshaped([1]).item(Int.self)
        }
    }

    private func biasRow(_ bias: [Int: Float]) -> MLXArray {
        var values = [Float](repeating: 0, count: vocabularyWidth)
        for (token, value) in bias { values[token] = value }
        return MLXArray(values).reshaped([1, vocabularyWidth])
    }

    private func maskRow(_ allowed: [Int]) -> MLXArray {
        var values = [Float](repeating: -Float.infinity, count: vocabularyWidth)
        for token in allowed { values[token] = 0 }
        return MLXArray(values).reshaped([1, vocabularyWidth])
    }

    private func unpenalizedTokens(
        sampling: ContinuousBatchRowSampling? = nil, count: Int
    ) -> [Int] {
        let none = ContinuousBatchRowPenalties()
        return referenceTokens(none, sampling: sampling, count: count)
    }

    // MARK: - P1: reference parity

    func testP1GreedySoloRowEqualsTheVendoredProcessorReferenceForEachPenaltyKind() throws {
        let count = 24  // past the default 20-token context window
        let plain = unpenalizedTokens(count: count)
        XCTAssertEqual(plain, [Int](repeating: 0, count: count), "fixture argmax is column 0")
        for (label, penalties) in [
            ("repetition", Self.repetition), ("presence", Self.presence),
            ("frequency", Self.frequency),
        ] {
            let expected = referenceTokens(penalties, count: count)
            XCTAssertNotEqual(expected, plain, "\(label): non-vacuity — the penalty must matter")
            XCTAssertGreaterThan(Set(expected).count, 1, "\(label): reference not degenerate")
            let actual = try soloTokens(penalties: penalties, count: count)
            XCTAssertEqual(actual, expected, label)
            XCTAssertNotEqual(actual, plain, "\(label): penalized tokens equal unpenalized")
        }
    }

    func testP1SeededSampledRowWithPresenceEqualsTheReference() throws {
        let sampling = ContinuousBatchRowSampling(temperature: 1, seed: 21)
        let expected = referenceTokens(Self.presence, sampling: sampling, count: 16)
        let unpenalized = referenceTokens(
            ContinuousBatchRowPenalties(), sampling: sampling, count: 16)
        XCTAssertNotEqual(expected, unpenalized, "non-vacuity: presence changes the seeded draws")
        XCTAssertEqual(
            try soloTokens(penalties: Self.presence, sampling: sampling, count: 16), expected)
        let beside = try batchTokens(
            penalties: [Self.presence, nil], samplings: [sampling, nil], count: 16)
        XCTAssertEqual(beside[1], expected)
    }

    func testP1GreedyBatchRowsEqualTheirReferencesWithDifferentPenaltiesPerRow() throws {
        let count = 12
        let tokens = try batchTokens(
            penalties: [Self.repetition, Self.presence, Self.frequency, nil], count: count)
        XCTAssertEqual(tokens[1], referenceTokens(Self.repetition, count: count))
        XCTAssertEqual(tokens[2], referenceTokens(Self.presence, count: count))
        XCTAssertEqual(tokens[3], referenceTokens(Self.frequency, count: count))
        XCTAssertEqual(tokens[4], [Int](repeating: 0, count: count))
    }

    func testP1PromptTokensSeedEachRowsOwnProcessor() throws {
        // Prompt [0] puts column 0 in the ring before the first draw; prompt [10] does not.
        let count = 6
        let seeded = try batchTokens(
            penalties: [Self.repetition, Self.repetition], prompts: [[0], [10]], count: count)
        XCTAssertEqual(seeded[1], referenceTokens(Self.repetition, prompt: [0], count: count))
        XCTAssertEqual(seeded[2], referenceTokens(Self.repetition, prompt: [10], count: count))
        XCTAssertNotEqual(seeded[1], seeded[2], "the two prompts must give different streams")
    }

    func testP1PenalizedRowKeepsItsProcessorAcrossScalarToBatchToScalar() throws {
        for (penalties, sampling) in [
            (Self.repetition, nil),
            (Self.presence, ContinuousBatchRowSampling(temperature: 1, seed: 11)),
        ] as [(ContinuousBatchRowPenalties, ContinuousBatchRowSampling?)] {
            let expected = referenceTokens(penalties, sampling: sampling, count: 14)
            let runtime = try makeRuntime()
            try admitAndPrefill(runtime, id: 1, penalties: penalties, sampling: sampling)
            try admitAndPrefill(runtime, id: 2, penalties: nil)
            var tokens: [Int] = []
            tokens.append(try soloStep(runtime, 1, index: 0))
            tokens.append(try soloStep(runtime, 1, index: 1))
            tokens.append(try runtime.decode(.drainSoloPipeline(BatchRequestID(1)))[0].tokens[0])
            let ids = [BatchRequestID(1), BatchRequestID(2)]
            for _ in 0 ..< 5 {
                let results = try runtime.decode(.batch(ids, speculationAllowed: false))
                tokens.append(try XCTUnwrap(results.first { $0.id.rawValue == 1 }).tokens[0])
            }
            runtime.remove(BatchRequestID(2))
            for index in 0 ..< 6 { tokens.append(try soloStep(runtime, 1, index: index)) }
            XCTAssertEqual(tokens, expected, "penalties \(penalties) sampling \(String(describing: sampling))")
        }
    }

    // MARK: - P2: partner independence

    func testP2UnpenalizedPartnerIsByteEqualToItsSoloControlAndPenalizedRowIsUnchanged() throws {
        let count = 10
        let partnerControl = try soloTokens(penalties: nil, count: count)
        XCTAssertEqual(partnerControl, [Int](repeating: 0, count: count))
        let penalizedControl = try soloTokens(penalties: Self.repetition, count: count)
        XCTAssertEqual(penalizedControl, referenceTokens(Self.repetition, count: count))
        XCTAssertNotEqual(penalizedControl, partnerControl, "non-vacuity: the penalty matters")
        for order in [[Self.repetition, nil], [nil, Self.repetition]]
            as [[ContinuousBatchRowPenalties?]]
        {
            let tokens = try batchTokens(penalties: order, count: count)
            let penalizedID: UInt64 = order[0] == nil ? 2 : 1
            let partnerID: UInt64 = order[0] == nil ? 1 : 2
            XCTAssertEqual(tokens[partnerID], partnerControl, "partner order \(order)")
            XCTAssertEqual(tokens[penalizedID], penalizedControl, "penalized order \(order)")
        }
    }

    func testP2CacheSensitiveUnpenalizedPartnerIsByteEqualBesideAPenalizedRow() throws {
        let control = try soloTokens(
            penalties: nil, prompt: [50], model: OneHotPrefixSumLanguageModel(), count: 5)
        XCTAssertEqual(control, [51, 102, 204, 408, 816])
        let penalizedControl = try soloTokens(
            penalties: Self.presence, prompt: [10], model: OneHotPrefixSumLanguageModel(),
            count: 5)
        let beside = try batchTokens(
            penalties: [Self.presence, nil], prompts: [[10], [50]],
            model: OneHotPrefixSumLanguageModel(), count: 5)
        XCTAssertEqual(beside[2], control)
        XCTAssertEqual(beside[1], penalizedControl)
    }

    // MARK: - P4: order

    func testP4LogitBiasIsAddedAfterTheProcessorNotBeforeIt() throws {
        // Prompt [1] seeds column 1 (raw -1.386). Bias +1 on it: process(raw)+bias leaves it at
        // -1.77 (< column 0's -0.916), while process(raw+bias) would put it at -0.77 (> -0.916).
        let bias: [Int: Float] = [1: 1]
        let count = 8
        let right = referenceTokens(
            Self.repetition, prompt: [1], additive: biasRow(bias), count: count)
        let wrong = referenceTokens(
            Self.repetition, prompt: [1], additive: biasRow(bias), biasBeforeProcessor: true,
            count: count)
        XCTAssertNotEqual(right, wrong, "fixture must make the order observable")
        XCTAssertEqual(
            try soloTokens(
                penalties: Self.repetition, prompt: [1], logitBias: bias, count: count),
            right)
        let beside = try batchTokens(
            penalties: [Self.repetition, nil], prompts: [[1], [1]], biases: [bias, [:]],
            count: count)
        XCTAssertEqual(beside[1], right)
        XCTAssertEqual(beside[2], [Int](repeating: 0, count: count))
    }

    func testP4TokenMaskIsAddedAfterTheProcessorAndAdvancesOnEmittedTokens() throws {
        let allowed = [0, 1, 2]
        let count = 8
        let expected = referenceTokens(
            Self.presence, additive: maskRow(allowed), count: count)
        XCTAssertNotEqual(
            expected, referenceTokens(ContinuousBatchRowPenalties(), additive: maskRow(allowed), count: count),
            "non-vacuity")
        let mask = FixedMask(allowed: allowed)
        let solo = try soloTokens(penalties: Self.presence, tokenMask: mask, count: count)
        XCTAssertEqual(solo, expected)
        XCTAssertEqual(mask.advanced, expected)
    }

    // MARK: - P4b: bias dtype (bf16 raw logits)

    /// bf16 logits whose columns 0 and 1 are a near-tie that the bias value 0.3 resolves one way in
    /// float32 (column 0 wins: `-0.1875 + 0.3f < 231/2048`) and the other way in bf16 (0.3 rounds
    /// to 0.30078125: column 1 wins). Every other column is a large finite negative. All values
    /// are bf16-exact, so only the BIAS rounding differs between the two references.
    private static let tieA: Float = 231.0 / 2048.0
    private static let tieB: Float = -0.1875

    private func bf16TieRow() -> MLXArray {
        var values = [Float](repeating: fillerLogit, count: vocabularyWidth)
        values[0] = Self.tieA
        values[1] = Self.tieB
        return MLXArray(values).reshaped([1, vocabularyWidth]).asType(.bfloat16)
    }

    /// Reference over the bf16 row: `process(raw)` (float32 after the histogram), then a bias row
    /// built in the PROCESSED dtype (the scalar route's `ScalarServingLogitBias.biasRow`), or in
    /// `biasDtype` when given (the WRONG bf16-rounded construction).
    private func bf16BiasReference(
        _ penalties: ContinuousBatchRowPenalties, bias: [Int: Float], biasDtype: DType? = nil,
        count: Int
    ) -> [Int] {
        var processor = GenerateParameters(
            repetitionPenalty: penalties.repetition,
            presencePenalty: penalties.presence,
            frequencyPenalty: penalties.frequency
        ).processor()
        processor?.prompt(MLXArray([10]))
        let raw = bf16TieRow()
        return (0 ..< count).map { _ in
            let processed = processor!.process(logits: raw)
            XCTAssertEqual(processed.dtype, .float32, "fixture must promote to float32")
            let row = biasRow(bias).asType(biasDtype ?? processed.dtype)
            let token = argMax(processed + row, axis: -1)
            processor!.didSample(token: token)
            return token.asType(.int32).reshaped([1]).item(Int.self)
        }
    }

    func testP4bBiasOnPenalizedRowIsBuiltInTheProcessedDtypeNotTheRawBf16Dtype() throws {
        let bias: [Int: Float] = [1: 0.3, 0: 0, 5: 0.3]
        let count = 8
        let exact = bf16BiasReference(Self.frequency, bias: bias, count: count)
        let rounded = bf16BiasReference(
            Self.frequency, bias: bias, biasDtype: .bfloat16, count: count)
        XCTAssertNotEqual(exact, rounded, "non-vacuity: bf16 rounding of the bias flips a token")
        XCTAssertEqual(exact.first, 0, "float32 bias keeps column 0 ahead at the first draw")
        XCTAssertEqual(rounded.first, 1, "bf16 bias flips the first draw to column 1")
        let model = { FixedLogitsLanguageModel(row: self.bf16TieRow()) }

        // Prefill-final (first token) and scalar steps.
        XCTAssertEqual(
            try soloTokens(
                penalties: Self.frequency, model: model(), logitBias: bias, count: count),
            exact, "solo")
        // Batch steps beside an unpenalized, unbiased partner.
        let beside = try batchTokens(
            penalties: [Self.frequency, nil], biases: [bias, [:]], model: model(), count: count)
        XCTAssertEqual(beside[1], exact, "batch")
        // Scalar -> batch -> scalar.
        let runtime = try makeRuntime(model: model())
        try admitAndPrefill(runtime, id: 1, penalties: Self.frequency, logitBias: bias)
        try admitAndPrefill(runtime, id: 2, penalties: nil)
        var tokens: [Int] = [try soloStep(runtime, 1, index: 0)]
        tokens.append(try runtime.decode(.drainSoloPipeline(BatchRequestID(1)))[0].tokens[0])
        let ids = [BatchRequestID(1), BatchRequestID(2)]
        for _ in 0 ..< 3 {
            let results = try runtime.decode(.batch(ids, speculationAllowed: false))
            tokens.append(try XCTUnwrap(results.first { $0.id.rawValue == 1 }).tokens[0])
        }
        runtime.remove(BatchRequestID(2))
        for index in 0 ..< 3 { tokens.append(try soloStep(runtime, 1, index: index)) }
        XCTAssertEqual(tokens, exact, "scalar -> batch -> scalar")
    }

    func testP4bMaskedAndBiasedPenalizedRowOnBf16LogitsStillUsesTheExactBias() throws {
        // Mask {0, 1, 2} keeps the tie columns; bias must still be float32-exact.
        let bias: [Int: Float] = [1: 0.3]
        let allowed = [0, 1, 2]
        let count = 6
        let exact = bf16BiasReference(Self.frequency, bias: bias, count: count)
        let mask = FixedMask(allowed: allowed)
        let tokens = try soloTokens(
            penalties: Self.frequency, model: FixedLogitsLanguageModel(row: bf16TieRow()),
            logitBias: bias, tokenMask: mask, count: count)
        XCTAssertEqual(tokens.first, exact.first)
        XCTAssertTrue(tokens.allSatisfy { allowed.contains($0) })
    }

    // MARK: - P4c: masked penalized rows in a batch

    func testP4cMaskedPenalizedRowBesideAnotherRowEqualsProcessThenMaskReference() throws {
        let allowed = [0, 1, 2]
        let count = 8
        for bias in [[Int: Float](), [1: 1]] {
            let additive = bias.isEmpty ? maskRow(allowed) : biasRow(bias) + maskRow(allowed)
            let expected = referenceTokens(Self.presence, additive: additive, count: count)
            XCTAssertNotEqual(
                expected,
                referenceTokens(ContinuousBatchRowPenalties(), additive: additive, count: count),
                "non-vacuity: the penalty matters under the mask")
            let mask = FixedMask(allowed: allowed)
            let runtime = try makeRuntime()
            try admitAndPrefill(
                runtime, id: 1, penalties: Self.presence, logitBias: bias, tokenMask: mask)
            try admitAndPrefill(runtime, id: 2, penalties: nil)
            let ids = [BatchRequestID(1), BatchRequestID(2)]
            var tokens: [UInt64: [Int]] = [:]
            for _ in 0 ..< count {
                for result in try runtime.decode(.batch(ids, speculationAllowed: false)) {
                    tokens[result.id.rawValue, default: []].append(contentsOf: result.tokens)
                }
            }
            XCTAssertEqual(tokens[1], expected, "bias \(bias)")
            XCTAssertEqual(mask.advanced, expected, "mask advances on every emitted token")
            XCTAssertEqual(tokens[2], [Int](repeating: 0, count: count), "partner unchanged")
        }
    }

    func testP4cMaskWithNoAllowedColumnInsideTheLogitsWidthIsUnconstrainedWhilePenalized() throws {
        let outside = [vocabularyWidth + 3_000]
        let count = 8
        for bias in [[Int: Float](), [1: 1]] {
            let additive: MLXArray? = bias.isEmpty ? nil : biasRow(bias)
            let expected = referenceTokens(Self.presence, additive: additive, count: count)
            XCTAssertNotEqual(
                expected,
                referenceTokens(Self.presence, additive: maskRow([0, 1, 2]), count: count),
                "non-vacuity: constrained and unconstrained streams differ")
            let solo = try soloTokens(
                penalties: Self.presence, logitBias: bias,
                tokenMask: FixedMask(allowed: outside), count: count)
            XCTAssertEqual(solo, expected, "solo bias \(bias)")
            let mask = FixedMask(allowed: outside)
            let runtime = try makeRuntime()
            try admitAndPrefill(
                runtime, id: 1, penalties: Self.presence, logitBias: bias, tokenMask: mask)
            try admitAndPrefill(runtime, id: 2, penalties: nil)
            let ids = [BatchRequestID(1), BatchRequestID(2)]
            var tokens: [Int] = []
            for _ in 0 ..< count {
                let results = try runtime.decode(.batch(ids, speculationAllowed: false))
                tokens.append(try XCTUnwrap(results.first { $0.id.rawValue == 1 }).tokens[0])
            }
            XCTAssertEqual(tokens, expected, "batch bias \(bias)")
            XCTAssertEqual(mask.advanced, expected)
        }
    }

    // MARK: - P5: speculation refused

    func testP5PenaltiesWithSpeculationAreRefusedAndAdmitNothing() throws {
        let runtime = try makeRuntime(
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: EmptyDrafter(),
                maxDraft: 3,
                gate: PLDGate(window: 8, minimumSamples: 4, minAcceptPerStep: 0.5, cooldown: 8),
                lookback: 32,
                compiledVerify: true))
        XCTAssertThrowsError(
            try admit(runtime, id: 1, penalties: Self.presence, requestsSpeculation: true)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .penaltiesWithSpeculation(BatchRequestID(1)))
        }
        let refused = runtime.resourceSnapshot()
        XCTAssertEqual(refused?.reservedKVBytes, 0)
        XCTAssertEqual(refused?.speculation?.requestedRequests, 0)
        // Nothing was left behind: the same id admits as an unpenalized speculative row.
        try admit(runtime, id: 1, penalties: nil, requestsSpeculation: true)
        XCTAssertGreaterThan(runtime.resourceSnapshot()?.reservedKVBytes ?? 0, 0)
    }

    // MARK: - P6: poison

    private func soloPoisonError(
        penalties: ContinuousBatchRowPenalties?, sampling: ContinuousBatchRowSampling? = nil
    ) throws -> DenseContinuousBatchRuntimeError? {
        let runtime = try makeRuntime(model: OneHotPrefixSumLanguageModel(poisonInput: 11))
        try admitAndPrefill(runtime, id: 1, penalties: penalties, sampling: sampling)
        XCTAssertEqual(try soloStep(runtime, 1, index: 0), 11)
        do {
            _ = try soloStep(runtime, 1, index: 1)
            return nil
        } catch {
            return error as? DenseContinuousBatchRuntimeError
        }
    }

    private func batchPoisonError(
        penalties: ContinuousBatchRowPenalties?, sampling: ContinuousBatchRowSampling? = nil
    ) throws -> DenseContinuousBatchRuntimeError? {
        let runtime = try makeRuntime(model: OneHotPrefixSumLanguageModel(poisonInput: 11))
        try admitAndPrefill(runtime, id: 1, penalties: penalties, sampling: sampling)
        try admitAndPrefill(runtime, id: 2, penalties: nil)
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

    func testP6PenalizedRowFailsClosedOnNonFiniteRawLogitsExactlyLikeAnUnpenalizedRow() throws {
        let sentinel = DenseContinuousBatchRuntimeError.invalidTokenID(BatchRequestID(1), -1)
        let sampled = ContinuousBatchRowSampling(temperature: 1, topK: 1, seed: 3)
        XCTAssertEqual(try soloPoisonError(penalties: nil), sentinel, "unpenalized solo control")
        XCTAssertEqual(try batchPoisonError(penalties: nil), sentinel, "unpenalized batch control")
        XCTAssertEqual(try soloPoisonError(penalties: Self.presence), sentinel, "greedy solo")
        XCTAssertEqual(try batchPoisonError(penalties: Self.presence), sentinel, "greedy batch")
        XCTAssertEqual(
            try soloPoisonError(penalties: Self.repetition, sampling: sampled), sentinel,
            "sampled solo")
        XCTAssertEqual(
            try batchPoisonError(penalties: Self.repetition, sampling: sampled), sentinel,
            "sampled batch")
    }

    func testP6HealthyPenalizedRowBesideAPoisonedPartnerKeepsItsReferenceTokens() throws {
        // The partner's bias forces token 7 first; the model returns all-NaN logits for input 7,
        // so the partner fails at the next batch step. The penalized row never feeds 7.
        let count = 6
        let expected = referenceTokens(Self.frequency, count: count)
        XCTAssertFalse(expected.contains(7), "fixture: the penalized row must never feed the poison")
        XCTAssertNotEqual(expected, referenceTokens(ContinuousBatchRowPenalties(), count: count))
        for sampling in [nil, ContinuousBatchRowSampling(temperature: 1, seed: 5)] {
            let reference = referenceTokens(Self.frequency, sampling: sampling, count: count)
            XCTAssertFalse(reference.contains(7))
            let runtime = try makeRuntime(model: FixedLogitsLanguageModel(poisonInput: 7))
            try admitAndPrefill(runtime, id: 1, penalties: Self.frequency, sampling: sampling)
            try admitAndPrefill(runtime, id: 2, penalties: nil, logitBias: [7: 1_000])
            let ids = [BatchRequestID(1), BatchRequestID(2)]
            let first = try runtime.decode(.batch(ids, speculationAllowed: false))
            XCTAssertEqual(first.map { $0.tokens[0] }, [reference[0], 7])
            XCTAssertThrowsError(try runtime.decode(.batch(ids, speculationAllowed: false))) {
                XCTAssertEqual(
                    $0 as? DenseContinuousBatchRuntimeError,
                    .invalidTokenID(BatchRequestID(2), -1), "the partner fails, not the penalized row")
            }
            // The failed partner leaves; the penalized row continues from exactly its reference
            // stream: no stray `didSample` (or draw) happened for it on the failed step.
            runtime.remove(BatchRequestID(2))
            var tokens = [first[0].tokens[0]]
            for index in 0 ..< count - 1 { tokens.append(try soloStep(runtime, 1, index: index)) }
            XCTAssertEqual(tokens, reference, "sampling \(String(describing: sampling))")
        }
    }

    func testP6PenalizedPrefillFinalOnNonFiniteLogitsFailsLikeAnUnpenalizedPrefill() throws {
        func prefillError(penalties: ContinuousBatchRowPenalties?) throws
            -> DenseContinuousBatchRuntimeError?
        {
            let runtime = try makeRuntime(model: OneHotPrefixSumLanguageModel(poisonInput: 11))
            try admit(runtime, id: 1, prompt: [11], penalties: penalties)
            do {
                try prefill(runtime, id: 1, prompt: [11])
                return nil
            } catch {
                return error as? DenseContinuousBatchRuntimeError
            }
        }
        let sentinel = DenseContinuousBatchRuntimeError.invalidTokenID(BatchRequestID(1), -1)
        XCTAssertEqual(try prefillError(penalties: nil), sentinel, "unpenalized control")
        XCTAssertEqual(try prefillError(penalties: Self.presence), sentinel, "penalized")
    }
}
