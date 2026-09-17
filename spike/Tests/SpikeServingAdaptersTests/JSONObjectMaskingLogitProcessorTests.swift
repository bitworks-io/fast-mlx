import Foundation
import XCTest

import MLX
import MLXLMCommon
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

/// A tiny synthetic byte-level vocab: ids `0...255` are the single-byte tokens (id N encodes raw
/// byte N — e.g. id 0x7B is `{`, id 0x78 is `x`), id `256` is a two-byte token (`"{}"`, unused by
/// these tests but present so a real multi-byte token exists in the trie alongside the single-byte
/// ones), and id `257` is EOS. No added/banned tokens — every id classifies as either `.bytes` or
/// `.eos`.
private enum SyntheticVocab {
    static let eosID = 257
    static let vocabSize = 258
    /// The adversarial scorer's favorite byte: a bare UTF-8 continuation byte (0x80), which
    /// `JSONObjectAutomaton` rejects in EVERY position — not just "not valid to start a document"
    /// (unlike an ordinary printable ASCII byte, which IS legal JSON STRING content and so would
    /// let an adversarial scorer stall forever inside an open string once masking forced it into
    /// one). This byte can never legally appear anywhere in the automaton's grammar, so masking it
    /// away can never itself introduce a way to stall.
    static let adversarialByte: UInt8 = 0x80

    /// Bytes given a secondary, still-arbitrary preference over the (0-logit) baseline — never
    /// preferred over the adversarial byte itself, only over ties among otherwise-equal allowed
    /// ids (e.g. every insignificant-whitespace byte also sits at the 0 baseline, and without SOME
    /// tie-breaking preference a greedy argmax naturally gravitates to the lowest allowed id, which
    /// is whitespace, and can loop on it up to the automaton's own whitespace cap before making any
    /// real progress). Each of these bytes is a legal, unremarkable move at every position where it
    /// is ever the trie's allowed alternative, so boosting them changes nothing about WHETHER the
    /// mask is doing its job — only how quickly this fixture's greedy walk reaches EOS.
    static let secondaryPreferenceBytes: [UInt8] = [0x7B, 0x22, 0x3A, 0x7D] // { " : }

    static let table: JSONObjectConstraintTable = {
        var classifications: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }
        classifications.append(.bytes([0x7B, 0x7D])) // id 256: "{}"
        classifications.append(.eos) // id 257
        return JSONObjectConstraintTable(classifications: classifications)
    }()

    /// Raw logits `[1, vocabSize]` that always score `adversarialByte`'s id far above every other
    /// id, including EOS — an adversarial scorer that, left unmasked, greedily repeats byte 0x80
    /// forever and never terminates, let alone produces JSON. `secondaryPreferenceBytes` and EOS
    /// get a smaller, uniform boost purely to break ties deterministically (see their doc comment);
    /// none of them ever outscores the adversarial byte itself.
    static func adversarialLogits() -> MLXArray {
        var values = [Float](repeating: 0, count: vocabSize)
        for byte in secondaryPreferenceBytes {
            values[Int(byte)] = 50
        }
        values[eosID] = 50
        values[Int(adversarialByte)] = 200
        return MLXArray(values).reshaped([1, vocabSize])
    }

    /// Bytes for a sampled id — `[UInt8(id)]` for a single-byte token, the multi-byte fixture's own
    /// bytes for id 256. Never called with `eosID`.
    static func bytes(forSampledID id: Int) -> [UInt8] {
        id == 256 ? [0x7B, 0x7D] : [UInt8(id)]
    }
}

/// Runs the adversarial scorer through `processor` (or, when `nil`, completely unmasked) for up to
/// `maxSteps` greedy decode steps, stopping early if EOS is sampled. Shared by the masked/unmasked
/// tests below so they can never drift in how a "decode step" is defined.
private func runGreedyLoop(
    through processor: JSONObjectMaskingLogitProcessor?, maxSteps: Int = 400
) -> (bytes: [UInt8], sampledEOS: Bool) {
    var produced: [UInt8] = []
    let raw = SyntheticVocab.adversarialLogits()
    for _ in 0..<maxSteps {
        let scored = processor?.process(logits: raw) ?? raw
        let tokenArray = argMax(scored, axis: -1)
        processor?.didSample(token: tokenArray)
        let id = tokenArray.item(Int.self)
        if id == SyntheticVocab.eosID {
            return (produced, true)
        }
        produced.append(contentsOf: SyntheticVocab.bytes(forSampledID: id))
    }
    return (produced, false)
}

final class JSONObjectMaskingLogitProcessorTests: XCTestCase {
    /// Acceptance test 1 (response-format design): an adversarial scorer that always prefers a
    /// non-JSON byte, run through the REAL masking processor, still produces a well-formed JSON
    /// object and terminates at EOS — the mask, not the scorer, decides the output shape.
    func testAdversarialScorerThroughMaskProducesValidJSONObjectEndingInEOS() throws {
        let processor = JSONObjectMaskingLogitProcessor(
            table: SyntheticVocab.table, activeFromStart: true, thinkEndTokenID: nil)

        let result = runGreedyLoop(through: processor)

        XCTAssertNil(processor.recordedFailure)
        XCTAssertTrue(result.sampledEOS, "expected the automaton to reach EOS")
        let text = String(decoding: result.bytes, as: UTF8.self)
        let parsed = try JSONSerialization.jsonObject(with: Data(result.bytes))
        XCTAssertTrue(parsed is [String: Any], "expected a JSON OBJECT, got: \(text)")
    }

    /// Control (and, per the response-format design, the same fact as the "disabling the mask"
    /// mutation): the IDENTICAL adversarial scorer, with NO masking processor at all, cannot
    /// produce JSON — it just repeats `x` until the step budget is exhausted, never emitting EOS.
    /// If a future change made `process(logits:)` a no-op (the mask "disabled"), this exact
    /// unmasked code path is what `testAdversarialScorerThroughMaskProducesValidJSONObjectEndingInEOS`
    /// would then be exercising too, and it would turn red for precisely the reason this test
    /// documents here.
    func testUnmaskedAdversarialScorerNeverProducesJSON() {
        let result = runGreedyLoop(through: nil, maxSteps: 20)

        XCTAssertFalse(result.sampledEOS)
        XCTAssertEqual(result.bytes, [UInt8](repeating: SyntheticVocab.adversarialByte, count: 20))
        XCTAssertThrowsError(try JSONSerialization.jsonObject(with: Data(result.bytes)))
    }

    // MARK: - Thinking-phase gate (response-format design item #4)

    /// With `activeFromStart: false`, the mask does nothing until `thinkEndTokenID` is sampled —
    /// the adversarial scorer wins freely during the "reasoning" phase, then the mask takes over
    /// once `</think>` (id 300 in this fixture) is sampled.
    func testMaskStaysInactiveUntilThinkEndTokenThenActivates() throws {
        let thinkEndTokenID = 300
        let processor = JSONObjectMaskingLogitProcessor(
            table: SyntheticVocab.table, activeFromStart: false, thinkEndTokenID: thinkEndTokenID)

        // Phase 1 ("reasoning"): the mask is inactive, so the adversarial scorer's favorite byte
        // passes straight through UNCHANGED — the defining, checkable behavior of "inactive".
        let raw = SyntheticVocab.adversarialLogits()
        for _ in 0..<5 {
            let scored = processor.process(logits: raw)
            XCTAssertEqual(
                scored.asArray(Float.self), raw.asArray(Float.self),
                "mask must be a no-op before the think-end token")
            let tokenArray = argMax(scored, axis: -1)
            processor.didSample(token: tokenArray)
            XCTAssertEqual(tokenArray.item(Int.self), Int(SyntheticVocab.adversarialByte))
        }

        // The think-end token itself is fed directly (it need not come from `raw`'s argmax) —
        // `didSample` is what flips `active`, not `process`.
        processor.didSample(token: MLXArray([Int32(thinkEndTokenID)]))

        // Phase 2 ("answer"): the mask is now active — the SAME adversarial scorer no longer wins;
        // the first legal byte is `{` (0x7B), not `x`.
        let postThinkScored = processor.process(logits: raw)
        let postThinkToken = argMax(postThinkScored, axis: -1).item(Int.self)
        XCTAssertEqual(postThinkToken, 0x7B, "mask must be active immediately after the think-end token")
        XCTAssertNil(processor.recordedFailure)
    }

    /// Mutation control: a processor built `activeFromStart: true` (the non-thinking configuration)
    /// masks from token 0 regardless of `thinkEndTokenID` — proving the PARAMETER, not some global
    /// flag, is what the phase gate actually keys on. If `activeFromStart` were ever hardcoded to
    /// `false`'s behavior (mask never turns on for a "no separation" request), this would go red.
    func testActiveFromStartMasksImmediatelyEvenWithAThinkEndTokenConfigured() {
        let processor = JSONObjectMaskingLogitProcessor(
            table: SyntheticVocab.table, activeFromStart: true, thinkEndTokenID: 300)

        let raw = SyntheticVocab.adversarialLogits()
        let scored = processor.process(logits: raw)
        let token = argMax(scored, axis: -1).item(Int.self)

        XCTAssertEqual(token, 0x7B, "mask must already be active with activeFromStart: true")
    }

    // MARK: - Failure path (response-format design: fail closed, never silently succeed)

    /// Forcing a disallowed token through `didSample` (bypassing `process` entirely — exactly what
    /// a masking bug or a rigged decoder would look like) records a failure rather than silently
    /// advancing as if nothing happened. `MLXDecoder.selectSampleAndAdvance` (SpikeCore) is what
    /// turns this into a real thrown error on the decode path — see
    /// `MLXDecoderResponseFormatConstraintFailureTests` (SpikeCoreTests) for that half of the
    /// contract; this test pins the processor's OWN half: it must actually notice.
    func testForcingADisallowedTokenRecordsAFailureInsteadOfSilentlyAdvancing() {
        let processor = JSONObjectMaskingLogitProcessor(
            table: SyntheticVocab.table, activeFromStart: true, thinkEndTokenID: nil)

        // `x` (0x78) is never allowed as the FIRST byte of a JSON object — the mask's own table
        // agrees (see the adversarial test above), so forcing it here is a genuine violation, not
        // an accident of this fixture.
        processor.didSample(token: MLXArray([Int32(SyntheticVocab.adversarialByte)]))

        XCTAssertNotNil(processor.recordedFailure)
        guard case .some(JSONObjectConstraintError.tokenDisallowed(let id)) = processor.recordedFailure else {
            return XCTFail("expected .tokenDisallowed, got \(String(describing: processor.recordedFailure))")
        }
        XCTAssertEqual(id, Int(SyntheticVocab.adversarialByte))

        // Once a failure is recorded, `process` becomes an inert passthrough (there is nothing
        // useful left to mask for a request that is already failing closed) rather than continuing
        // to do trie work or throwing again from a different call.
        let raw = SyntheticVocab.adversarialLogits()
        XCTAssertEqual(processor.process(logits: raw).asArray(Float.self), raw.asArray(Float.self))
    }

    // MARK: - Logits wider than the table (should-fix #9: padded lm_head output)

    /// `logits.dim(-1)` may exceed the table's own classified id range (a padded `lm_head` output —
    /// see `scalarServingModelVocabSize`'s doc comment). Every id at or beyond the table's range has
    /// no classification at all and must still end up masked to `-inf`, exactly like any other
    /// disallowed id, never left unmasked just because it falls outside the range the table was
    /// built for.
    func testLogitsWiderThanTableAreMasked() {
        let processor = JSONObjectMaskingLogitProcessor(
            table: SyntheticVocab.table, activeFromStart: true, thinkEndTokenID: nil)
        let widerVocabSize = SyntheticVocab.vocabSize + 40
        let raw = MLXArray([Float](repeating: 0, count: widerVocabSize)).reshaped([1, widerVocabSize])

        let masked = processor.process(logits: raw).asArray(Float.self)

        // A REAL in-table allowed id (the initial automaton state allows `{`, 0x7B) must stay 0 —
        // proves this test's masked/unmasked reading is the right way around.
        XCTAssertEqual(masked[0x7B], 0)
        for id in SyntheticVocab.vocabSize..<widerVocabSize {
            XCTAssertEqual(masked[id], -Float.infinity, "id \(id) beyond the table must be masked")
        }
        XCTAssertNil(processor.recordedFailure)
    }
}
