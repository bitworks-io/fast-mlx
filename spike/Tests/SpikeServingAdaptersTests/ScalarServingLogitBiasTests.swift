import Foundation
import XCTest
import os

import MLX
import MLXLMCommon
import MLXNN
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

// OpenAI `logit_bias` on the plain SCALAR serving route.
//
// Layers covered here (decode + HTTP capability gate live in `OpenAILogitBiasTests` and
// `OpenAIChatCompletionsHTTPHandlerTests`):
//   1. `LogitBiasLogitProcessor` in isolation on small synthetic logits.
//   2. Composition with a response-format mask: the MASK WINS (bias first, mask last).
//   3. `ScalarServingBackend` admission: capability, vocabulary bound, and non-scalar refusal, all
//      BEFORE any decoder work starts.
//   4. End to end through the real `ScalarServingBackend` -> `InferenceActor` -> `MLXDecoder` with a
//      fixed-logits model: a -100 on the greedy argmax changes the first token, a +100 forces one.

// MARK: - Processor in isolation

final class LogitBiasLogitProcessorTests: XCTestCase {
    private func row(_ values: [Float]) -> MLXArray {
        MLXArray(values).reshaped([1, values.count])
    }

    func testAddsExactlyTheBiasToTargetedIdsAndLeavesOthersUnchanged() {
        let processor = LogitBiasLogitProcessor(bias: [1: 10, 3: -100, 4: 0.5])
        let out = processor.process(logits: row([1, 2, 3, 4, 5, 6])).asArray(Float.self)
        XCTAssertEqual(out, [1, 12, 3, -96, 5.5, 6])
    }

    func testPreservesShapeAndDtypeForHalfPrecisionLogits() {
        let processor = LogitBiasLogitProcessor(bias: [2: -100, 0: 100])
        let logits = row([1, 2, 3, 4]).asType(.float16)
        let out = processor.process(logits: logits)
        XCTAssertEqual(out.shape, [1, 4])
        XCTAssertEqual(out.dtype, .float16)
        XCTAssertEqual(out.asType(.float32).asArray(Float.self), [101, 2, -97, 4])
    }

    func testEveryStepIsBiasedWithTheSameRowAcrossRepeatedCalls() {
        let processor = LogitBiasLogitProcessor(bias: [0: 7])
        XCTAssertEqual(processor.process(logits: row([1, 1])).asArray(Float.self), [8, 1])
        XCTAssertEqual(processor.process(logits: row([5, 9])).asArray(Float.self), [12, 9])
        XCTAssertEqual(processor.process(logits: row([0, 0])).asArray(Float.self), [7, 0])
    }

    func testEmptyBiasIsTheIdentity() {
        let processor = LogitBiasLogitProcessor(bias: [:])
        XCTAssertEqual(processor.process(logits: row([3, 1, 2])).asArray(Float.self), [3, 1, 2])
    }

    func testHundredCanFlipTheArgmaxAndMinusHundredCanRemoveIt() {
        let base = row([0, 1, 5, 2])
        XCTAssertEqual(argMax(base, axis: -1).item(Int.self), 2)
        let down = LogitBiasLogitProcessor(bias: [2: -100]).process(logits: base)
        XCTAssertEqual(argMax(down, axis: -1).item(Int.self), 3)
        let up = LogitBiasLogitProcessor(bias: [0: 100]).process(logits: base)
        XCTAssertEqual(argMax(up, axis: -1).item(Int.self), 0)
    }

    /// Admission validates the vocabulary bound; if a bias id still reaches the processor beyond the
    /// real logits width, it must be a recorded failure (the decoder turns that into a thrown error)
    /// -- never an out-of-bounds MLX index and never a silent skip.
    func testOutOfRangeIdIsARecordedFailureNotACrash() {
        let processor = LogitBiasLogitProcessor(bias: [1: 5, 9: 5])
        let logits = row([1, 2, 3])
        let out = processor.process(logits: logits).asArray(Float.self)
        XCTAssertNotNil(processor.recordedFailure)
        XCTAssertEqual(out, [1, 2, 3], "no partial bias may be applied once the map is invalid for this vocab")
    }

    // MARK: Composition with a mask: the mask wins

    /// A replace-style mask (masked ids are set to a fixed finite floor) is the case where
    /// ordering matters: bias-then-mask keeps the floor, mask-then-bias would add +100 on top of it
    /// and resurrect the token.
    func testMaskBeatsAPlusHundredBiasOnAReplaceStyleMask() {
        let mask = ReplacingMaskProcessor(maskedIDs: [2], floor: -50)
        let processor = LogitBiasLogitProcessor(bias: [2: 100, 1: 1], then: mask)
        let out = processor.process(logits: row([0, 0, 0, 0])).asArray(Float.self)
        XCTAssertEqual(out, [0, 1, -50, 0], "the masked token must stay at the mask floor")
        XCTAssertEqual(argMax(MLXArray(out).reshaped([1, 4]), axis: -1).item(Int.self), 1)
    }

    /// The REAL json_object mask (additive -inf): at the document start only `{` (id 0) is legal,
    /// so a +100 on `}` and EOS must not resurrect them.
    func testRealJSONObjectMaskBeatsPlusHundredBias() async throws {
        let support = ScalarServingJSONObjectConstraintSupport(
            classifications: [.bytes([0x7B]), .bytes([0x7D]), .eos], thinkEndTokenID: nil)
        let mask = JSONObjectMaskingLogitProcessor(
            table: await support.table, activeFromStart: true, thinkEndTokenID: nil)
        let processor = LogitBiasLogitProcessor(bias: [1: 100, 2: 100], then: mask)
        let out = processor.process(logits: row([0, 0, 0])).asArray(Float.self)
        XCTAssertEqual(out, [0, -Float.infinity, -Float.infinity])
        XCTAssertEqual(argMax(MLXArray(out).reshaped([1, 3]), axis: -1).item(Int.self), 0)
    }

    func testPromptAndDidSampleAreForwardedToTheInnerProcessor() {
        let inner = ReplacingMaskProcessor(maskedIDs: [], floor: 0)
        let processor = LogitBiasLogitProcessor(bias: [0: 1], then: inner)
        processor.prompt(MLXArray([1, 2, 3]))
        processor.didSample(token: MLXArray([1]))
        processor.didSample(token: MLXArray([2]))
        XCTAssertEqual(inner.promptCalls, 1)
        XCTAssertEqual(inner.sampled, [1, 2])
    }

    func testInnerRecordedFailureSurfacesThroughTheComposite() {
        let inner = ReplacingMaskProcessor(maskedIDs: [], floor: 0)
        let processor = LogitBiasLogitProcessor(bias: [0: 1], then: inner)
        XCTAssertNil(processor.recordedFailure)
        inner.failWith(ReplacingMaskProcessor.Sentinel())
        XCTAssertTrue(processor.recordedFailure is ReplacingMaskProcessor.Sentinel)
    }
}

// MARK: - Backend admission + end to end

final class ScalarServingLogitBiasBackendTests: XCTestCase {
    private static let pieces: [Int: String] = [0: "a", 1: "b", 2: "c", 3: "d"]

    private func makeBackend(
        decoder: sending any Decoder,
        vocabularySize: Int? = 4,
        isNonSpeculativeScalarRoute: Bool = true,
        decoderSupportsResponseFormatConstraint: Bool = true,
        jsonObjectConstraintSupport: ScalarServingJSONObjectConstraintSupport? = nil,
        pieces: [Int: String] = ScalarServingLogitBiasBackendTests.pieces,
        stopTokenIDs: Set<Int> = [99]
    ) -> ScalarServingBackend {
        var configuration = ScalarServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            maximumQueuedRequests: 2,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 16, maxBytes: 4_096),
            jsonObjectConstraintSupport: jsonObjectConstraintSupport,
            isNonSpeculativeScalarRoute: isNonSpeculativeScalarRoute,
            decoderSupportsResponseFormatConstraint: decoderSupportsResponseFormatConstraint)
        configuration.logitBiasVocabularySize = vocabularySize
        return ScalarServingBackend(
            launchedModel: "fixture-model",
            inference: InferenceActor(decoder: decoder),
            codec: BiasFixtureCodec(pieces: pieces),
            stopTokenIDs: stopTokenIDs,
            modelStopStrings: [],
            configuration: configuration)
    }

    private func request(
        maxTokens: Int = 3,
        temperature: Double = 0,
        seed: Int64? = nil,
        logitBias: [Int: Float] = [:],
        responseFormat: ServingResponseFormat? = nil
    ) -> OpenAIChatCompletionRequest {
        OpenAIChatCompletionRequest(
            model: "fixture-model",
            messages: [OpenAIChatMessage(role: .user, text: "hi")],
            maxCompletionTokens: maxTokens,
            temperature: temperature,
            choiceCount: 1,
            stream: true,
            stop: [],
            seed: seed,
            responseFormat: responseFormat,
            logitBias: logitBias)
    }

    private func generate(_ backend: ScalarServingBackend, _ request: OpenAIChatCompletionRequest) async throws
        -> String
    {
        let handle = try await backend.start(request)
        var text = ""
        while let event = try await handle.mailbox.next() {
            if case .text(let piece) = event { text += piece }
        }
        return text
    }

    private func fixedLogitsBackend(
        _ logits: [Float], vocabularySize: Int? = nil
    ) -> ScalarServingBackend {
        makeBackend(
            decoder: MLXDecoder(model: BiasFixtureLogitsModel(logits: logits), cache: [KVCacheSimple()]),
            vocabularySize: vocabularySize ?? logits.count)
    }

    private func assertRefused(
        _ backend: ScalarServingBackend,
        _ request: OpenAIChatCompletionRequest,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        do {
            _ = try await backend.start(request)
            XCTFail("expected a 400 for logit_bias", file: file, line: line)
        } catch let error as OpenAIServingError {
            guard case .invalidRequest(_, let param) = error else {
                return XCTFail("expected .invalidRequest, got \(error)", file: file, line: line)
            }
            XCTAssertEqual(param, "logit_bias", file: file, line: line)
        } catch {
            XCTFail("unexpected error \(error)", file: file, line: line)
        }
    }

    // MARK: capability

    func testCapabilityRequiresScalarRouteDecoderCapabilityAndAResolvedVocabulary() {
        let decoder = { CountingDecoder(recorder: BiasStartRecorder()) }
        XCTAssertTrue(makeBackend(decoder: decoder()).supportsLogitBias)
        XCTAssertFalse(makeBackend(decoder: decoder(), isNonSpeculativeScalarRoute: false).supportsLogitBias)
        XCTAssertFalse(
            makeBackend(decoder: decoder(), decoderSupportsResponseFormatConstraint: false).supportsLogitBias)
        XCTAssertFalse(makeBackend(decoder: decoder(), vocabularySize: nil).supportsLogitBias)
    }

    // MARK: vocabulary bound (before generation)

    func testIdBeyondTheVocabularyIsA400BeforeAnyDecoderWork() async {
        let recorder = BiasStartRecorder()
        let backend = makeBackend(decoder: CountingDecoder(recorder: recorder), vocabularySize: 4)

        await assertRefused(backend, request(logitBias: [4: 1]))
        await assertRefused(backend, request(logitBias: [0: 1, 1_000_000: -1]))
        XCTAssertEqual(recorder.prefills, 0, "a refused request must not start any generation")
    }

    func testHighestInRangeIdIsAdmittedAndGenerates() async throws {
        let recorder = BiasStartRecorder()
        let backend = makeBackend(decoder: CountingDecoder(recorder: recorder), vocabularySize: 4)

        let handle = try await backend.start(request(maxTokens: 1, logitBias: [3: 1]))
        while try await handle.mailbox.next() != nil {}
        XCTAssertEqual(recorder.prefills, 1)
    }

    // MARK: non-scalar route refusal (zero backend generations)

    func testNonScalarRoutesRefuseANonEmptyBiasWithZeroStarts() async throws {
        let cases: [(String, Bool, Bool, Int?)] = [
            ("speculative/compiled route", false, true, 4),
            ("decoder without a constraint seam", true, false, 4),
            ("vocabulary unresolved", true, true, nil),
        ]
        for (label, scalarRoute, decoderSupports, vocab) in cases {
            let recorder = BiasStartRecorder()
            let backend = makeBackend(
                decoder: CountingDecoder(recorder: recorder),
                vocabularySize: vocab,
                isNonSpeculativeScalarRoute: scalarRoute,
                decoderSupportsResponseFormatConstraint: decoderSupports)
            await assertRefused(backend, request(logitBias: [1: 5]))
            XCTAssertEqual(recorder.prefills, 0, label)

            // Control: the neutral empty map is unaffected on the very same backend.
            let handle = try await backend.start(request(maxTokens: 1, logitBias: [:]))
            while try await handle.mailbox.next() != nil {}
            XCTAssertEqual(recorder.prefills, 1, label)
        }
    }

    // MARK: end to end through the real MLXDecoder

    func testBaselineGreedyPicksTheArgmaxToken() async throws {
        let text = try await generate(fixedLogitsBackend([0, 1, 5, 2]), request())
        XCTAssertEqual(text, "ccc")
    }

    func testMinusHundredOnTheGreedyArgmaxChangesTheFirstToken() async throws {
        let text = try await generate(fixedLogitsBackend([0, 1, 5, 2]), request(logitBias: [2: -100]))
        XCTAssertEqual(text, "ddd", "token 2 is suppressed, so the next-best token 3 must win")
    }

    func testPlusHundredOnAChosenIdForcesIt() async throws {
        let text = try await generate(fixedLogitsBackend([0, 1, 5, 2]), request(logitBias: [0: 100]))
        XCTAssertEqual(text, "aaa")
    }

    func testBiasAppliesToSampledGenerationToo() async throws {
        let forced = try await generate(
            fixedLogitsBackend([0, 1, 5, 2]),
            request(temperature: 1, seed: 7, logitBias: [1: 100]))
        XCTAssertEqual(forced, "bbb", "+100 must dominate the sampled distribution")

        let suppressed = try await generate(
            fixedLogitsBackend([0, 0, 50, 0]),
            request(maxTokens: 6, temperature: 1, seed: 7, logitBias: [2: -100]))
        XCTAssertEqual(suppressed.count, 6)
        XCTAssertFalse(suppressed.contains("c"), "token 2 dominates the raw logits but is suppressed by -100")
    }

    func testRequestWithoutBiasIsUnchangedAndABiasDoesNotLeakIntoTheNextRequest() async throws {
        let backend = fixedLogitsBackend([0, 1, 5, 2])
        let biased = try await generate(backend, request(logitBias: [2: -100]))
        XCTAssertEqual(biased, "ddd")
        let plain = try await generate(backend, request())
        XCTAssertEqual(plain, "ccc", "the processor slot must be cleared after the biased request")
    }

    /// The mask has the last word end to end: with json_object active, `{` is the only legal first
    /// token, so a +100 on `}` and on EOS must not change it; the output stays well-formed `{}`.
    func testJSONObjectMaskWinsOverPlusHundredBiasEndToEnd() async throws {
        let support = ScalarServingJSONObjectConstraintSupport(
            classifications: [.bytes([0x7B]), .bytes([0x7D]), .eos], thinkEndTokenID: nil)
        let backend = makeBackend(
            decoder: MLXDecoder(model: BiasFixtureLogitsModel(logits: [0, 0, 0]), cache: [KVCacheSimple()]),
            vocabularySize: 3,
            jsonObjectConstraintSupport: support,
            pieces: [0: "{", 1: "}"],
            stopTokenIDs: [2])  // EOS (id 2) ends generation here.
        let handle = try await backend.start(
            request(maxTokens: 6, logitBias: [1: 100, 2: 100], responseFormat: .jsonObject))
        var text = ""
        while let event = try await handle.mailbox.next() {
            if case .text(let piece) = event { text += piece }
        }
        XCTAssertEqual(text, "{}")
    }
}

// MARK: - Fixtures

private final class BiasStartRecorder: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock(initialState: 0)
    var prefills: Int { lock.withLock { $0 } }
    func recordPrefill() { lock.withLock { $0 += 1 } }
}

/// A decoder fixture that counts how many generations actually started and ends each at once.
private struct CountingDecoder: Decoder {
    let recorder: BiasStartRecorder
    var supportsResponseFormatConstraint: Bool { true }

    mutating func prefill(_ promptTokens: [Int]) throws -> Int {
        recorder.recordPrefill()
        return 99
    }
    mutating func step(last: Int) throws -> Int { 99 }
    mutating func reset() {}
}

/// Replace-style mask: ids in `maskedIDs` are forced to `floor` (finite). Also records what the
/// composite forwards to it.
private final class ReplacingMaskProcessor: LogitProcessor, ConstraintProcessorFailureReporting,
    @unchecked Sendable
{
    struct Sentinel: Error {}
    private let lock = OSAllocatedUnfairLock(initialState: State())
    private struct State {
        var promptCalls = 0
        var sampled: [Int] = []
        var failure: Error?
    }
    private let maskedIDs: Set<Int>
    private let floor: Float

    init(maskedIDs: Set<Int>, floor: Float) {
        self.maskedIDs = maskedIDs
        self.floor = floor
    }

    var promptCalls: Int { lock.withLock { $0.promptCalls } }
    var sampled: [Int] { lock.withLock { $0.sampled } }
    var recordedFailure: Error? { lock.withLock { $0.failure } }
    func failWith(_ error: Error) { lock.withLock { $0.failure = error } }

    func prompt(_ prompt: MLXArray) { lock.withLock { $0.promptCalls += 1 } }

    func process(logits: MLXArray) -> MLXArray {
        let width = logits.dim(-1)
        var keep = [Float](repeating: 1, count: width)
        var fill = [Float](repeating: 0, count: width)
        for id in maskedIDs where id < width {
            keep[id] = 0
            fill[id] = floor
        }
        let keepArray = MLXArray(keep).reshaped([1, width])
        let fillArray = MLXArray(fill).reshaped([1, width])
        return logits * keepArray.asType(logits.dtype) + fillArray.asType(logits.dtype)
    }

    func didSample(token: MLXArray) {
        let id = token.item(Int.self)
        lock.withLock { $0.sampled.append(id) }
    }
}

/// `LanguageModel` returning the same fixed logits at every position (mirrors
/// `MLXDecoderResponseFormatConstraintFailureTests.FixedLogitsModel`).
private final class BiasFixtureLogitsModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads: [Int] = [1]
    let logits: [Float]

    init(logits: [Float]) { self.logits = logits }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func evaluateThrowing(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) throws -> LMOutput {
        let seqLen = input.tokens.dim(1)
        let row = MLXArray(logits)
        let full = MLXArray.zeros([1, seqLen, logits.count]) + row
        return LMOutput(logits: full)
    }
}

private struct BiasFixtureCodec: ScalarServingTextCodec {
    let pieces: [Int: String]

    func render(
        messages: [OpenAIChatMessage], tools: [OpenAIToolSpec], enableThinking: Bool?,
        reasoningEffort: String?, addGenerationPrompt: Bool?
    ) throws -> [Int] {
        [10]
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        BiasFixtureDetokenizer(pieces: pieces)
    }
}

private struct BiasFixtureDetokenizer: ScalarServingDetokenizer {
    let pieces: [Int: String]
    private var pending: String?

    init(pieces: [Int: String]) { self.pieces = pieces }

    mutating func append(token: Int) { pending = pieces[token] }

    mutating func next() -> String? {
        defer { pending = nil }
        return pending
    }
}
