import Foundation
import HarnessCore
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

// Cycle 280: `logprobs` / `top_logprobs` in the dense continuous-batch runtime.
//
// A row whose submission carries a `ContinuousBatchRowLogprobSink` gets one entry per token it
// emits, built from THAT row's RAW logits for the step that selected the token (before penalties,
// `logit_bias`, the mask and the temperature) by the SAME helper the scalar route uses
// (`MLXDecoder.decodedTokenLogprob`). The reference below applies that helper to independently
// constructed raw rows: the fixed row of `FixedLogitsLanguageModel`, or the one-hot row of
// `OneHotPrefixSumLanguageModel` (whose argmax is `sum(prompt) + sum(fed tokens) + 1`, so a pairing
// error between a token and the wrong step's logits is visible). Nothing here calls runtime code
// except the runtime under test.

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

/// The `[1, width]` raw row of `OneHotPrefixSumLanguageModel`: 100 at `target`, 0 elsewhere.
private func oneHotRow(target: Int) -> MLXArray {
    var values = [Float](repeating: 0, count: vocabularyWidth)
    values[target] = 100
    return MLXArray(values).reshaped([1, vocabularyWidth])
}

private final class FixedLogitsLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads = [1]
    private let row: MLXArray
    private let poisonInput: Int32?

    init(poisonInput: Int32? = nil) {
        self.row = fixedLogitsRow()
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

/// Cache-sensitive one-hot model: the greedy next token is `prefix sum + 1`. When `poisonInput` is
/// set, the row at that input token is all-NaN (a full-shaped array: a scalar NaN constant in a
/// compiled step is inlined as an undeclared `nan` in the Metal source).
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

/// Fixed allowed set (the same for every step).
private final class FixedMask: ContinuousBatchRowTokenMask, @unchecked Sendable {
    let allowed: [Int]
    init(allowed: [Int]) { self.allowed = allowed }

    func allowedTokenBitset() -> [UInt64]? {
        var words = [UInt64](repeating: 0, count: (allowed.max() ?? 0) / 64 + 1)
        for id in allowed { words[id >> 6] |= UInt64(1) << UInt64(id & 63) }
        return words
    }

    func advance(token: Int) {}
}

private struct EmptyDrafter: SpecDrafter {
    func propose(context _: [Int], maxDraft _: Int) -> [Int] { [] }
}

private final class RecordingSink: ContinuousBatchRowLogprobSink, @unchecked Sendable {
    let topLogprobs: Int
    private let lock = NSLock()
    private var entries: [ContinuousBatchTokenLogprob] = []

    init(topLogprobs: Int) { self.topLogprobs = topLogprobs }

    var recorded: [ContinuousBatchTokenLogprob] {
        lock.lock()
        defer { lock.unlock() }
        return entries
    }

    func record(_ logprob: ContinuousBatchTokenLogprob) {
        lock.lock()
        defer { lock.unlock() }
        entries.append(logprob)
    }
}

/// One row of a test run. `topN == nil` means no sink.
private struct RowSpec {
    var prompt: [Int] = [10]
    var sampling: ContinuousBatchRowSampling?
    var penalties: ContinuousBatchRowPenalties?
    var bias: [Int: Float] = [:]
    var masked = false
    var topN: Int?
}

final class DenseContinuousBatchRuntimeLogprobTests: XCTestCase {
    private static let presence = ContinuousBatchRowPenalties(presence: 1.5)
    private static let repetition = ContinuousBatchRowPenalties(repetition: 2)
    private static let seeded = ContinuousBatchRowSampling(temperature: 1, seed: 21)
    private static let maskAllowed = [0, 1, 2]

    // MARK: - Harness

    private func makeRuntime(
        model: any LanguageModel, soloPLDConfiguration: SpecDecodeConfig? = nil
    ) throws -> DenseContinuousBatchRuntime {
        try DenseContinuousBatchRuntime(
            testing: model,
            allocationChunk: 4,
            maxContextTokens: 512,
            initialDecodeReserve: 384,
            soloPLDConfiguration: soloPLDConfiguration)
    }

    private func admit(
        _ runtime: DenseContinuousBatchRuntime, id: UInt64, spec: RowSpec,
        sink: RecordingSink?, requestsSpeculation: Bool = false
    ) throws {
        try runtime.admit([
            ContinuousBatchRuntimeAdmission(
                id: BatchRequestID(id),
                submission: ContinuousBatchSubmission(
                    promptTokens: spec.prompt,
                    maxOutputTokens: 64,
                    eosToken: 2,
                    architecture: .denseAttention,
                    requestsSpeculation: requestsSpeculation,
                    logitBias: spec.bias,
                    tokenMask: spec.masked ? FixedMask(allowed: Self.maskAllowed) : nil,
                    sampling: spec.sampling,
                    penalties: spec.penalties,
                    logprobs: sink)),
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

    @discardableResult
    private func admitAndPrefill(
        _ runtime: DenseContinuousBatchRuntime, id: UInt64, spec: RowSpec
    ) throws -> RecordingSink? {
        let sink = spec.topN.map { RecordingSink(topLogprobs: $0) }
        try admit(runtime, id: id, spec: spec, sink: sink)
        try prefill(runtime, id: id, prompt: spec.prompt)
        return sink
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

    private func soloRun(
        _ spec: RowSpec, model: any LanguageModel, count: Int
    ) throws -> (tokens: [Int], entries: [ContinuousBatchTokenLogprob]) {
        let runtime = try makeRuntime(model: model)
        let sink = try admitAndPrefill(runtime, id: 1, spec: spec)
        let tokens = try (0 ..< count).map { try soloStep(runtime, 1, index: $0) }
        return (tokens, sink?.recorded ?? [])
    }

    private func batchRun(
        _ specs: [RowSpec], model: any LanguageModel, count: Int
    ) throws -> (tokens: [UInt64: [Int]], sinks: [UInt64: RecordingSink]) {
        let runtime = try makeRuntime(model: model)
        var sinks: [UInt64: RecordingSink] = [:]
        for (index, spec) in specs.enumerated() {
            let id = UInt64(index + 1)
            sinks[id] = try admitAndPrefill(runtime, id: id, spec: spec)
        }
        let ids = specs.indices.map { BatchRequestID(UInt64($0 + 1)) }
        var tokens: [UInt64: [Int]] = [:]
        for _ in 0 ..< count {
            for result in try runtime.decode(.batch(ids, speculationAllowed: false)) {
                tokens[result.id.rawValue, default: []].append(contentsOf: result.tokens)
            }
        }
        return (tokens, sinks)
    }

    // MARK: - Reference

    /// The scalar route's helper applied to a given raw row, in the sink's value type.
    private func reference(
        token: Int, raw: MLXArray, topN: Int
    ) -> ContinuousBatchTokenLogprob {
        let decoded = MLXDecoder.decodedTokenLogprob(
            rawLogits: raw, sampledToken: token, topN: topN)
        return ContinuousBatchTokenLogprob(
            tokenID: decoded.tokenID,
            logprob: decoded.logprob,
            top: decoded.top.map {
                ContinuousBatchTokenLogprobCandidate(tokenID: $0.tokenID, logprob: $0.logprob)
            })
    }

    private func expectedFixed(_ tokens: [Int], topN: Int) -> [ContinuousBatchTokenLogprob] {
        tokens.map { reference(token: $0, raw: fixedLogitsRow(), topN: topN) }
    }

    /// Raw row of step `k` of the one-hot model: argmax `sum(prompt) + sum(tokens[..<k]) + 1`.
    private func expectedOneHot(
        prompt: [Int], tokens: [Int], topN: Int
    ) -> [ContinuousBatchTokenLogprob] {
        tokens.indices.map { k in
            let target = prompt.reduce(0, +) + tokens[..<k].reduce(0, +) + 1
            return reference(token: tokens[k], raw: oneHotRow(target: target), topN: topN)
        }
    }

    /// Tie-safe comparison for the one-hot rows (the 2,047 equal `-100` columns tie in the top
    /// list, so only their values and the unique top-1 id are comparable).
    private func assertEqualIgnoringTieIDs(
        _ actual: [ContinuousBatchTokenLogprob], _ expected: [ContinuousBatchTokenLogprob],
        _ message: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, message, file: file, line: line)
        for (index, (a, e)) in zip(actual, expected).enumerated() {
            XCTAssertEqual(a.tokenID, e.tokenID, "\(message) [\(index)] token", file: file, line: line)
            XCTAssertEqual(a.logprob, e.logprob, "\(message) [\(index)] logprob", file: file, line: line)
            XCTAssertEqual(
                a.top.map(\.logprob), e.top.map(\.logprob), "\(message) [\(index)] top values",
                file: file, line: line)
            XCTAssertEqual(
                a.top.first?.tokenID, e.top.first?.tokenID, "\(message) [\(index)] top-1 id",
                file: file, line: line)
        }
    }

    /// Logprob of `token` under the log-softmax of `adjusted` (a penalized and/or biased row).
    private func adjustedLogprob(token: Int, adjusted: MLXArray) -> Float {
        reference(token: token, raw: adjusted, topN: 0).logprob
    }

    private func biasRow(_ bias: [Int: Float]) -> MLXArray {
        var values = [Float](repeating: 0, count: vocabularyWidth)
        for (token, value) in bias { values[token] = value }
        return MLXArray(values).reshaped([1, vocabularyWidth])
    }

    // MARK: - L1: reference parity

    func testL1SoloEntriesEqualTheScalarHelperOnTheFixtureRawLogitsForEachRowKind() throws {
        let count = 10
        let kinds: [(String, RowSpec)] = [
            ("greedy", RowSpec()),
            ("seeded sampled", RowSpec(sampling: Self.seeded)),
            ("penalized", RowSpec(penalties: Self.presence)),
            ("biased", RowSpec(bias: [1: 5])),
            (
                "penalized+biased+sampled",
                RowSpec(sampling: Self.seeded, penalties: Self.repetition, bias: [2: 1])
            ),
        ]
        for (label, base) in kinds {
            for topN in [0, 5] {
                var spec = base
                spec.topN = topN
                let run = try soloRun(spec, model: FixedLogitsLanguageModel(), count: count)
                XCTAssertEqual(run.entries.count, count, "\(label) topN \(topN) count")
                XCTAssertEqual(run.entries.map(\.tokenID), run.tokens, "\(label) token pairing")
                XCTAssertEqual(
                    run.entries, expectedFixed(run.tokens, topN: topN), "\(label) topN \(topN)")
                XCTAssertTrue(run.entries.allSatisfy { $0.top.count == topN }, label)
                if topN == 5 {
                    // Independent of the helper: the fixture's probabilities are the first eight
                    // columns, so the raw top-5 are columns 0...4 with log(p) values.
                    for entry in run.entries {
                        XCTAssertEqual(entry.top.map(\.tokenID), [0, 1, 2, 3, 4], label)
                        for candidate in entry.top {
                            XCTAssertEqual(
                                candidate.logprob,
                                Float(log(Double(fixedProbabilities[candidate.tokenID]))),
                                accuracy: 1e-4, label)
                        }
                        XCTAssertEqual(
                            entry.logprob,
                            Float(log(Double(fixedProbabilities[entry.tokenID]))), accuracy: 1e-4,
                            label)
                    }
                }
            }
        }
    }

    func testL1NonVacuityTheRawLogprobDiffersFromTheAdjustedLogitsLogprob() throws {
        // Biased row: bias +5 on column 1 makes it the greedy token; the RAW logprob of column 1 is
        // log(0.25), while the log-softmax of raw+bias gives it a clearly larger value.
        let bias: [Int: Float] = [1: 5]
        var biased = RowSpec(bias: bias)
        biased.topN = 3
        let run = try soloRun(biased, model: FixedLogitsLanguageModel(), count: 4)
        XCTAssertEqual(run.tokens, [1, 1, 1, 1], "fixture: the bias decides the token")
        let adjusted = fixedLogitsRow() + biasRow(bias)
        for (token, entry) in zip(run.tokens, run.entries) {
            let wrong = adjustedLogprob(token: token, adjusted: adjusted)
            XCTAssertGreaterThan(abs(wrong - entry.logprob), 0.5, "adjusted differs from raw")
            XCTAssertEqual(entry.logprob, Float(log(0.25)), accuracy: 1e-4)
            XCTAssertEqual(entry.top.first?.tokenID, 0, "the raw top-1 is column 0, not the pick")
        }

        // Penalized row: the processed logits (the ring holds earlier tokens) give a different
        // logprob than the raw logits at every step after the first.
        var penalized = RowSpec(penalties: Self.presence)
        penalized.topN = 2
        let penalizedRun = try soloRun(penalized, model: FixedLogitsLanguageModel(), count: 6)
        XCTAssertNotEqual(Set(penalizedRun.tokens).count, 1, "fixture: the penalty moves tokens")
        var processor = GenerateParameters(presencePenalty: 1.5).processor()
        processor?.prompt(MLXArray([10]))
        var differing = 0
        for (token, entry) in zip(penalizedRun.tokens, penalizedRun.entries) {
            let processed = processor!.process(logits: fixedLogitsRow())
            if abs(adjustedLogprob(token: token, adjusted: processed) - entry.logprob) > 0.01 {
                differing += 1
            }
            processor!.didSample(token: MLXArray(Int32(token)))
        }
        XCTAssertGreaterThan(differing, 0, "non-vacuity: processed logprobs differ from raw")
    }

    func testL1BatchRowsEqualTheReferenceOnTheirOwnRawLogits() throws {
        let count = 6
        for topN in [0, 5] {
            // Two sink rows with different prompts: each entry must pair with ITS row's own step.
            let specs = [
                RowSpec(prompt: [10], topN: topN),
                RowSpec(prompt: [50], bias: [5: 1_000], topN: topN),
                RowSpec(prompt: [20]),
            ]
            let run = try batchRun(specs, model: OneHotPrefixSumLanguageModel(), count: count)
            for id in [UInt64(1), 2] {
                let spec = specs[Int(id) - 1]
                let tokens = try XCTUnwrap(run.tokens[id])
                let entries = try XCTUnwrap(run.sinks[id]).recorded
                assertEqualIgnoringTieIDs(
                    entries,
                    expectedOneHot(prompt: spec.prompt, tokens: tokens, topN: topN),
                    "row \(id) topN \(topN)")
            }
            XCTAssertEqual(run.tokens[1], [11, 22, 44, 88, 176, 352])
            XCTAssertEqual(run.tokens[2], [Int](repeating: 5, count: count), "bias decides row 2")
            // Row 2's chosen token is never the raw argmax, so its chosen logprob is -100.
            let biasedEntries = try XCTUnwrap(run.sinks[2]).recorded
            XCTAssertTrue(biasedEntries.allSatisfy { $0.logprob < -99 })
            XCTAssertTrue(biasedEntries.allSatisfy { $0.top.first.map { $0.logprob > -1e-3 } ?? true })

            // Sampled and penalized fixed-row batch members.
            let fixedSpecs = [
                RowSpec(sampling: Self.seeded, topN: topN),
                RowSpec(penalties: Self.presence, topN: topN),
                RowSpec(),
            ]
            let fixedRun = try batchRun(fixedSpecs, model: FixedLogitsLanguageModel(), count: count)
            for id in [UInt64(1), 2] {
                XCTAssertEqual(
                    try XCTUnwrap(fixedRun.sinks[id]).recorded,
                    expectedFixed(try XCTUnwrap(fixedRun.tokens[id]), topN: topN),
                    "fixed row \(id) topN \(topN)")
            }
        }
    }

    func testL1EntriesSurviveAScalarToBatchToScalarMove() throws {
        for (label, spec, model, oneHot) in [
            (
                "greedy one-hot", RowSpec(prompt: [0], topN: 5),
                { OneHotPrefixSumLanguageModel() as any LanguageModel }, true
            ),
            (
                "biased one-hot", RowSpec(prompt: [10], bias: [7: 1_000], topN: 3),
                { OneHotPrefixSumLanguageModel() as any LanguageModel }, true
            ),
            (
                "penalized sampled", RowSpec(sampling: Self.seeded, penalties: Self.presence, topN: 5),
                { FixedLogitsLanguageModel() as any LanguageModel }, false
            ),
            (
                "seeded sampled", RowSpec(sampling: Self.seeded, topN: 0),
                { FixedLogitsLanguageModel() as any LanguageModel }, false
            ),
        ] {
            let runtime = try makeRuntime(model: model())
            let sink = try XCTUnwrap(try admitAndPrefill(runtime, id: 1, spec: spec))
            var partner = RowSpec(prompt: [30])
            partner.topN = nil
            try admitAndPrefill(runtime, id: 2, spec: partner)
            var tokens: [Int] = []
            tokens.append(try soloStep(runtime, 1, index: 0))
            tokens.append(try soloStep(runtime, 1, index: 1))
            tokens.append(try runtime.decode(.drainSoloPipeline(BatchRequestID(1)))[0].tokens[0])
            XCTAssertEqual(sink.recorded.count, 3, "\(label): solo")
            let ids = [BatchRequestID(1), BatchRequestID(2)]
            for _ in 0 ..< 4 {
                let results = try runtime.decode(.batch(ids, speculationAllowed: false))
                tokens.append(try XCTUnwrap(results.first { $0.id.rawValue == 1 }).tokens[0])
            }
            XCTAssertEqual(sink.recorded.count, 7, "\(label): after batch")
            runtime.remove(BatchRequestID(2))
            // The prompt-[0] one-hot row doubles its token each step (1, 2, 4, ...), so 11 tokens
            // in total keep the target inside the 2,048-wide vocabulary.
            for index in 0 ..< 4 { tokens.append(try soloStep(runtime, 1, index: index)) }
            XCTAssertEqual(sink.recorded.count, 11, "\(label): after return to scalar")
            let topN = spec.topN ?? 0
            if oneHot {
                assertEqualIgnoringTieIDs(
                    sink.recorded,
                    expectedOneHot(prompt: spec.prompt, tokens: tokens, topN: topN), label)
            } else {
                XCTAssertEqual(sink.recorded, expectedFixed(tokens, topN: topN), label)
            }
            XCTAssertEqual(sink.recorded.map(\.tokenID), tokens, "\(label): pairing")
        }
    }

    // MARK: - L2: selection unchanged

    func testL2TokensAreByteEqualWithAndWithoutASink() throws {
        let kinds: [(String, RowSpec, () -> any LanguageModel)] = [
            ("greedy", RowSpec(prompt: [0]), { OneHotPrefixSumLanguageModel() }),
            ("greedy fixed", RowSpec(), { FixedLogitsLanguageModel() }),
            ("seeded sampled", RowSpec(sampling: Self.seeded), { FixedLogitsLanguageModel() }),
            ("penalized", RowSpec(penalties: Self.presence), { FixedLogitsLanguageModel() }),
            (
                "penalized sampled", RowSpec(sampling: Self.seeded, penalties: Self.presence),
                { FixedLogitsLanguageModel() }
            ),
            ("biased", RowSpec(bias: [1: 5]), { FixedLogitsLanguageModel() }),
            ("masked", RowSpec(masked: true), { FixedLogitsLanguageModel() }),
            (
                "masked penalized", RowSpec(penalties: Self.presence, masked: true),
                { FixedLogitsLanguageModel() }
            ),
        ]
        let count = 10
        for (label, base, model) in kinds {
            let control = try soloRun(base, model: model(), count: count)
            var withSink = base
            withSink.topN = 5
            let solo = try soloRun(withSink, model: model(), count: count)
            XCTAssertEqual(solo.tokens, control.tokens, "\(label) solo")
            XCTAssertEqual(solo.entries.count, count, "\(label): the sink saw every token")

            let besideControl = try batchRun(
                [base, RowSpec(prompt: [1])], model: model(), count: count)
            let besideSink = try batchRun(
                [withSink, RowSpec(prompt: [1])], model: model(), count: count)
            XCTAssertEqual(besideSink.tokens[1], besideControl.tokens[1], "\(label) batch")
            XCTAssertEqual(besideSink.tokens[1], control.tokens, "\(label) batch == solo")
            XCTAssertEqual(besideSink.tokens[2], besideControl.tokens[2], "\(label) partner")
        }
        let sampled = try soloRun(RowSpec(sampling: Self.seeded), model: FixedLogitsLanguageModel(), count: count)
        XCTAssertGreaterThan(Set(sampled.tokens).count, 1, "non-vacuity: the draws vary")
    }

    // MARK: - L3: partner independence

    func testL3PartnerWithoutASinkIsByteEqualToItsSoloControlAndRecordsNothing() throws {
        let count = 6
        let partnerSpec = RowSpec(prompt: [50])
        let partnerControl = try soloRun(partnerSpec, model: OneHotPrefixSumLanguageModel(), count: count)
        XCTAssertEqual(partnerControl.tokens, [51, 102, 204, 408, 816, 1_632])
        var sinkSpec = RowSpec(prompt: [10], topN: 5)
        let sinkSolo = try soloRun(sinkSpec, model: OneHotPrefixSumLanguageModel(), count: count)
        XCTAssertEqual(sinkSolo.entries.count, count)
        for order in [[sinkSpec, partnerSpec], [partnerSpec, sinkSpec]] {
            let sinkID: UInt64 = order[0].topN != nil ? 1 : 2
            let partnerID: UInt64 = sinkID == 1 ? 2 : 1
            let run = try batchRun(order, model: OneHotPrefixSumLanguageModel(), count: count)
            XCTAssertEqual(run.tokens[partnerID], partnerControl.tokens, "partner order \(sinkID)")
            XCTAssertEqual(run.tokens[sinkID], sinkSolo.tokens, "sink row tokens")
            XCTAssertNil(run.sinks[partnerID], "the partner has no sink")
            let entries = try XCTUnwrap(run.sinks[sinkID]).recorded
            XCTAssertEqual(entries.count, count, "the sink row's count is its own token count")
            XCTAssertEqual(entries, sinkSolo.entries, "entries identical solo and beside a partner")
            XCTAssertEqual(entries.map(\.tokenID), sinkSolo.tokens)
        }
        // Two sink rows keep separate streams.
        sinkSpec.prompt = [10]
        let two = try batchRun(
            [sinkSpec, RowSpec(prompt: [50], topN: 0)], model: OneHotPrefixSumLanguageModel(),
            count: count)
        XCTAssertEqual(try XCTUnwrap(two.sinks[1]).recorded.map(\.tokenID), sinkSolo.tokens)
        XCTAssertEqual(try XCTUnwrap(two.sinks[2]).recorded.map(\.tokenID), partnerControl.tokens)
        XCTAssertTrue(try XCTUnwrap(two.sinks[2]).recorded.allSatisfy { $0.top.isEmpty })
    }

    // MARK: - L4: count and order

    func testL4OneEntryPerTokenInOrderAndTheFirstIsThePrefillFinalToken() throws {
        let runtime = try makeRuntime(model: OneHotPrefixSumLanguageModel())
        let sink = RecordingSink(topLogprobs: 2)
        try admit(runtime, id: 1, spec: RowSpec(prompt: [50]), sink: sink)
        try prefill(runtime, id: 1, prompt: [50])
        XCTAssertEqual(sink.recorded.count, 0, "nothing is reported before a token is emitted")
        var tokens: [Int] = []
        for index in 0 ..< 6 {
            tokens.append(try soloStep(runtime, 1, index: index))
            XCTAssertEqual(sink.recorded.count, index + 1, "exactly one entry per emitted token")
            XCTAssertEqual(sink.recorded.last?.tokenID, tokens.last)
        }
        XCTAssertEqual(tokens.first, 51, "the prefill-final token (50 + 1)")
        XCTAssertEqual(sink.recorded.first?.tokenID, 51, "the first entry is the prefill-final token")
        XCTAssertEqual(sink.recorded.map(\.tokenID), tokens, "in order")
        XCTAssertEqual(sink.recorded.first?.logprob ?? 1, 0, accuracy: 1e-5)
        // Batch steps: one entry per row per step.
        let batchSink = RecordingSink(topLogprobs: 1)
        let batchRuntime = try makeRuntime(model: OneHotPrefixSumLanguageModel())
        try admit(batchRuntime, id: 1, spec: RowSpec(prompt: [50]), sink: batchSink)
        try prefill(batchRuntime, id: 1, prompt: [50])
        try admitAndPrefill(batchRuntime, id: 2, spec: RowSpec(prompt: [20]))
        let ids = [BatchRequestID(1), BatchRequestID(2)]
        for step in 0 ..< 4 {
            _ = try batchRuntime.decode(.batch(ids, speculationAllowed: false))
            XCTAssertEqual(batchSink.recorded.count, step + 1)
        }
        XCTAssertEqual(batchSink.recorded.map(\.tokenID), [51, 102, 204, 408])
    }

    // MARK: - L8: chunked prefill

    /// Prefills `prompt` in the given slice lengths (all but the last slice non-final).
    private func prefillInSlices(
        _ runtime: DenseContinuousBatchRuntime, id: UInt64, prompt: [Int], slices: [Int]
    ) throws {
        XCTAssertEqual(slices.reduce(0, +), prompt.count, "fixture: the slices cover the prompt")
        var start = 0
        for (index, length) in slices.enumerated() {
            try runtime.prefill(
                ContinuousBatchRuntimePrefill(
                    id: BatchRequestID(id),
                    startToken: start,
                    tokens: Array(prompt[start ..< start + length]),
                    isFinal: index == slices.count - 1,
                    totalPromptTokens: prompt.count,
                    maxOutputTokens: 64))
            start += length
        }
    }

    func testL4ChunkedPrefillRecordsOneEntryPerTokenAndTheFirstIsThePrefillFinalToken() throws {
        let count = 6
        let prompt = [50, 7, 3]
        let topN = 3
        // Control: the same prompt prefilled in ONE slice.
        let control = try soloRun(
            RowSpec(prompt: prompt, topN: topN), model: OneHotPrefixSumLanguageModel(), count: count)
        XCTAssertEqual(control.tokens, [61, 122, 244, 488, 976, 1_952], "fixture: sum(prompt) + 1")

        for slices in [[1, 1, 1], [2, 1], [1, 2]] {
            // Solo decode after a chunked prefill.
            let runtime = try makeRuntime(model: OneHotPrefixSumLanguageModel())
            let sink = RecordingSink(topLogprobs: topN)
            try admit(runtime, id: 1, spec: RowSpec(prompt: prompt), sink: sink)
            try prefillInSlices(runtime, id: 1, prompt: prompt, slices: slices)
            XCTAssertEqual(sink.recorded.count, 0, "\(slices): nothing before a token is emitted")
            var tokens: [Int] = []
            for index in 0 ..< count {
                tokens.append(try soloStep(runtime, 1, index: index))
                XCTAssertEqual(sink.recorded.count, index + 1, "\(slices): one entry per token")
            }
            XCTAssertEqual(tokens, control.tokens, "\(slices): chunking does not change tokens")
            XCTAssertEqual(sink.recorded.first?.tokenID, 61, "\(slices): first is the prefill-final token")
            XCTAssertEqual(sink.recorded.map(\.tokenID), tokens, "\(slices): in order")
            assertEqualIgnoringTieIDs(
                sink.recorded, expectedOneHot(prompt: prompt, tokens: tokens, topN: topN),
                "\(slices) solo vs reference")
            XCTAssertEqual(sink.recorded, control.entries, "\(slices): equals the one-slice run")

            // Batch decode beside a (chunk-prefilled) partner without a sink.
            let batch = try makeRuntime(model: OneHotPrefixSumLanguageModel())
            let batchSink = RecordingSink(topLogprobs: topN)
            try admit(batch, id: 1, spec: RowSpec(prompt: prompt), sink: batchSink)
            try prefillInSlices(batch, id: 1, prompt: prompt, slices: slices)
            let partnerPrompt = [20, 5]
            try admit(batch, id: 2, spec: RowSpec(prompt: partnerPrompt), sink: nil)
            try prefillInSlices(batch, id: 2, prompt: partnerPrompt, slices: [1, 1])
            let ids = [BatchRequestID(1), BatchRequestID(2)]
            var batchTokens: [Int] = []
            for step in 0 ..< count {
                let results = try batch.decode(.batch(ids, speculationAllowed: false))
                batchTokens.append(try XCTUnwrap(results.first { $0.id.rawValue == 1 }).tokens[0])
                XCTAssertEqual(batchSink.recorded.count, step + 1, "\(slices): batch entry per step")
            }
            XCTAssertEqual(batchTokens, control.tokens, "\(slices): batch tokens")
            XCTAssertEqual(batchSink.recorded, control.entries, "\(slices): batch entries")
        }

        // The fixed-row model: every raw row is the same, so the entries are the helper on it.
        let fixedPrompt = [10, 11, 12, 13]
        let runtime = try makeRuntime(model: FixedLogitsLanguageModel())
        let sink = RecordingSink(topLogprobs: 5)
        try admit(runtime, id: 1, spec: RowSpec(prompt: fixedPrompt), sink: sink)
        try prefillInSlices(runtime, id: 1, prompt: fixedPrompt, slices: [1, 2, 1])
        let tokens = try (0 ..< 5).map { try soloStep(runtime, 1, index: $0) }
        XCTAssertEqual(sink.recorded, expectedFixed(tokens, topN: 5))
        XCTAssertEqual(sink.recorded.count, tokens.count)
    }

    // MARK: - L9: membership change

    func testL3SinkRowIsUnchangedWhenItsBatchPartnerIsReplaced() throws {
        let topN = 3
        let sinkSolo = try soloRun(
            RowSpec(prompt: [10], topN: topN), model: OneHotPrefixSumLanguageModel(), count: 4)
        let partnerA = try soloRun(
            RowSpec(prompt: [50]), model: OneHotPrefixSumLanguageModel(), count: 2)
        let partnerB = try soloRun(
            RowSpec(prompt: [20]), model: OneHotPrefixSumLanguageModel(), count: 2)
        XCTAssertEqual(sinkSolo.tokens, [11, 22, 44, 88])

        let runtime = try makeRuntime(model: OneHotPrefixSumLanguageModel())
        let sink = try XCTUnwrap(try admitAndPrefill(runtime, id: 1, spec: RowSpec(prompt: [10], topN: topN)))
        try admitAndPrefill(runtime, id: 2, spec: RowSpec(prompt: [50]))
        var tokens: [UInt64: [Int]] = [:]
        func step(_ ids: [UInt64]) throws {
            let results = try runtime.decode(
                .batch(ids.map { BatchRequestID($0) }, speculationAllowed: false))
            for result in results {
                tokens[result.id.rawValue, default: []].append(contentsOf: result.tokens)
            }
        }
        try step([1, 2])
        try step([1, 2])
        XCTAssertEqual(sink.recorded.count, 2)
        // Partner A leaves, partner B joins: batch [1, 2] then [1, 3].
        runtime.remove(BatchRequestID(2))
        try admitAndPrefill(runtime, id: 3, spec: RowSpec(prompt: [20]))
        try step([1, 3])
        try step([1, 3])

        XCTAssertEqual(tokens[1], sinkSolo.tokens, "the sink row's tokens ignore the partner swap")
        XCTAssertEqual(tokens[2], partnerA.tokens, "partner A equals its solo control")
        XCTAssertEqual(tokens[3], partnerB.tokens, "partner B equals its solo control")
        XCTAssertEqual(sink.recorded.count, 4, "one entry per token, none for the partners")
        XCTAssertEqual(sink.recorded, sinkSolo.entries, "entries identical to the solo run")
        XCTAssertEqual(sink.recorded.map(\.tokenID), sinkSolo.tokens)
    }

    func testL3SinkRowLeavingTheBatchRecordsNothingMoreAndThePartnersAreUnchanged() throws {
        let topN = 3
        let sinkSolo = try soloRun(
            RowSpec(prompt: [10], topN: topN), model: OneHotPrefixSumLanguageModel(), count: 2)
        let partnerA = try soloRun(
            RowSpec(prompt: [50]), model: OneHotPrefixSumLanguageModel(), count: 5)
        let partnerB = try soloRun(
            RowSpec(prompt: [20]), model: OneHotPrefixSumLanguageModel(), count: 5)

        for removeSinkRow in [false, true] {
            let label = removeSinkRow ? "sink row removed" : "sink row only skipped"
            let runtime = try makeRuntime(model: OneHotPrefixSumLanguageModel())
            let sink = try XCTUnwrap(
                try admitAndPrefill(runtime, id: 1, spec: RowSpec(prompt: [10], topN: topN)))
            try admitAndPrefill(runtime, id: 2, spec: RowSpec(prompt: [50]))
            try admitAndPrefill(runtime, id: 3, spec: RowSpec(prompt: [20]))
            var tokens: [UInt64: [Int]] = [:]
            func step(_ ids: [UInt64]) throws {
                let results = try runtime.decode(
                    .batch(ids.map { BatchRequestID($0) }, speculationAllowed: false))
                for result in results {
                    tokens[result.id.rawValue, default: []].append(contentsOf: result.tokens)
                }
            }
            try step([1, 2, 3])
            try step([1, 2, 3])
            XCTAssertEqual(sink.recorded.count, 2, label)
            // The sink row leaves the batch; the partners stay batched together.
            if removeSinkRow { runtime.remove(BatchRequestID(1)) }
            try step([2, 3])
            try step([2, 3])
            try step([2, 3])

            XCTAssertEqual(tokens[1], sinkSolo.tokens, "\(label): sink row tokens")
            XCTAssertEqual(tokens[2], partnerA.tokens, "\(label): partner A equals solo")
            XCTAssertEqual(tokens[3], partnerB.tokens, "\(label): partner B equals solo")
            XCTAssertEqual(sink.recorded.count, 2, "\(label): nothing recorded after leaving")
            XCTAssertEqual(sink.recorded, sinkSolo.entries, "\(label): entries identical to solo")
        }
    }

    // MARK: - L5: admission refusal

    func testL5SinkWithSpeculationIsRefusedAndAdmitsNothing() throws {
        let runtime = try makeRuntime(
            model: FixedLogitsLanguageModel(),
            soloPLDConfiguration: SpecDecodeConfig(
                drafter: EmptyDrafter(),
                maxDraft: 3,
                gate: PLDGate(window: 8, minimumSamples: 4, minAcceptPerStep: 0.5, cooldown: 8),
                lookback: 32,
                compiledVerify: true))
        XCTAssertThrowsError(
            try admit(
                runtime, id: 1, spec: RowSpec(), sink: RecordingSink(topLogprobs: 1),
                requestsSpeculation: true)
        ) { error in
            XCTAssertEqual(
                error as? DenseContinuousBatchRuntimeError,
                .logprobsWithSpeculation(BatchRequestID(1)))
        }
        let refused = runtime.resourceSnapshot()
        XCTAssertEqual(refused?.reservedKVBytes, 0)
        XCTAssertEqual(refused?.speculation?.requestedRequests, 0)
        // Nothing was left behind: the same id admits as a speculative row without a sink, and a
        // sink row without speculation admits on the same runtime.
        try admit(runtime, id: 1, spec: RowSpec(), sink: nil, requestsSpeculation: true)
        XCTAssertGreaterThan(runtime.resourceSnapshot()?.reservedKVBytes ?? 0, 0)
        let sink = RecordingSink(topLogprobs: 0)
        try admit(runtime, id: 2, spec: RowSpec(), sink: sink)
        try prefill(runtime, id: 2, prompt: [10])
        _ = try runtime.decode(.solo(BatchRequestID(2), speculationAllowed: false))
        XCTAssertEqual(sink.recorded.count, 1)
    }

    // MARK: - L7: poison

    func testL7NonFiniteRawLogitsFailTheSinkRowAndRecordNoEntryForThatStep() throws {
        let sentinel = DenseContinuousBatchRuntimeError.invalidTokenID(BatchRequestID(1), -1)
        // Control: without the poison the second step records its entry.
        let healthy = try makeRuntime(model: OneHotPrefixSumLanguageModel())
        let healthySink = try XCTUnwrap(
            try admitAndPrefill(healthy, id: 1, spec: RowSpec(topN: 3)))
        XCTAssertEqual(try soloStep(healthy, 1, index: 0), 11)
        XCTAssertEqual(try soloStep(healthy, 1, index: 1), 22)
        XCTAssertEqual(healthySink.recorded.count, 2, "control: an entry per step")

        // Solo: the step that fed 11 produced NaN logits.
        let solo = try makeRuntime(model: OneHotPrefixSumLanguageModel(poisonInput: 11))
        let soloSink = try XCTUnwrap(try admitAndPrefill(solo, id: 1, spec: RowSpec(topN: 3)))
        XCTAssertEqual(try soloStep(solo, 1, index: 0), 11)
        XCTAssertEqual(soloSink.recorded.count, 1)
        XCTAssertThrowsError(try soloStep(solo, 1, index: 1)) {
            XCTAssertEqual($0 as? DenseContinuousBatchRuntimeError, sentinel)
        }
        XCTAssertEqual(soloSink.recorded.count, 1, "no entry for the poisoned step")
        XCTAssertEqual(soloSink.recorded.map(\.tokenID), [11])
        XCTAssertTrue(soloSink.recorded.allSatisfy { $0.logprob.isFinite })

        // Batch: the sink row fails; the partner has no sink.
        let batch = try makeRuntime(model: OneHotPrefixSumLanguageModel(poisonInput: 11))
        let batchSink = try XCTUnwrap(try admitAndPrefill(batch, id: 1, spec: RowSpec(topN: 3)))
        try admitAndPrefill(batch, id: 2, spec: RowSpec())
        let ids = [BatchRequestID(1), BatchRequestID(2)]
        XCTAssertEqual(
            try batch.decode(.batch(ids, speculationAllowed: false)).map { $0.tokens[0] }, [11, 11])
        XCTAssertEqual(batchSink.recorded.count, 1)
        XCTAssertThrowsError(try batch.decode(.batch(ids, speculationAllowed: false))) {
            XCTAssertEqual($0 as? DenseContinuousBatchRuntimeError, sentinel)
        }
        XCTAssertEqual(batchSink.recorded.count, 1, "no entry for the poisoned batch step")

        // Prefill-final: the sink row fails at prefill and records nothing.
        let early = try makeRuntime(model: OneHotPrefixSumLanguageModel(poisonInput: 11))
        let earlySink = RecordingSink(topLogprobs: 3)
        try admit(early, id: 1, spec: RowSpec(prompt: [11]), sink: earlySink)
        XCTAssertThrowsError(try prefill(early, id: 1, prompt: [11])) {
            XCTAssertEqual($0 as? DenseContinuousBatchRuntimeError, sentinel)
        }
        XCTAssertEqual(earlySink.recorded.count, 0)
    }
}
