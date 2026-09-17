import Foundation
import XCTest

import MLX
import MLXLMCommon
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

/// Processor-level tests for `response_format: json_schema`'s masking processor
/// (`ConstraintMaskingLogitProcessor<JSONSchemaTokenConstraint>`, aliased
/// `JSONSchemaMaskingLogitProcessor`) — the stage 3b sibling of
/// `JSONObjectMaskingLogitProcessorTests`, mirrored test-for-test wherever the two formats share a
/// contract (adversarial-scorer correctness, the thinking-phase gate, failure reporting, padded
/// `lm_head` masking), plus one schema-specific test proving the generated instance actually
/// validates against the COMPILED schema, not merely "some JSON".
///
/// A tiny synthetic byte-level vocab: ids `0...255` are the single-byte tokens (id N encodes raw
/// byte N), id `256` is EOS. No added/banned tokens — every id classifies as either `.bytes` or
/// `.eos`. Mirrors `JSONObjectMaskingLogitProcessorTests.SyntheticVocab`'s shape, minus that file's
/// extra multi-byte fixture token (not needed here: this file's schemas never require a multi-byte
/// token to be reachable).
private enum SchemaSyntheticVocab {
    static let eosID = 256
    static let vocabSize = 257

    /// Never legal ANYWHERE in either schema fixture's grammar below (a bare UTF-8 continuation
    /// byte) — see `JSONObjectMaskingLogitProcessorTests.SyntheticVocab.adversarialByte`'s doc
    /// comment for why this specific byte is the right adversarial choice (it can never let a
    /// masked adversarial scorer "stall" inside some otherwise-legal state).
    static let adversarialByte: UInt8 = 0x80

    /// Bytes of the literal `true`, given a secondary preference over the 0-baseline so a masked
    /// greedy walk deterministically produces `true` (not `false`) and then EOS, rather than
    /// looping on whichever legal byte happens to tie-break lowest.
    static let trueLiteralPreferenceBytes: [UInt8] = Array("true".utf8)

    static let table: JSONSchemaConstraintTable = {
        var classifications: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }
        classifications.append(.eos) // id 256
        return JSONSchemaConstraintTable(classifications: classifications)
    }()

    static func adversarialLogits() -> MLXArray {
        var values = [Float](repeating: 0, count: vocabSize)
        for byte in trueLiteralPreferenceBytes {
            values[Int(byte)] = 50
        }
        values[eosID] = 50
        values[Int(adversarialByte)] = 200
        return MLXArray(values).reshaped([1, vocabSize])
    }

    /// Every byte the nested-object fixture's own content could ever need (structural punctuation,
    /// the `"true"` literal, the property names `"ok"`/`"count"`, and one digit for the integer
    /// value), each given the SAME secondary preference as `adversarialLogits()`'s literal bytes —
    /// see that function's doc comment for why this beats optional structural WHITESPACE (which
    /// stays at the 0 baseline) outright, with no tie-break needed: every real content byte here
    /// outscores whitespace at every state where both are legally allowed, so the masked greedy walk
    /// never dawdles in an optional-whitespace loop before making real progress.
    static let objectFixturePreferenceBytes: [UInt8] = Array(
        Set("{}\":,trueokcunt0".utf8))

    static func objectFixtureAdversarialLogits() -> MLXArray {
        var values = [Float](repeating: 0, count: vocabSize)
        for byte in objectFixturePreferenceBytes {
            values[Int(byte)] = 50
        }
        values[eosID] = 50
        values[Int(adversarialByte)] = 200
        return MLXArray(values).reshaped([1, vocabSize])
    }
}

/// Parses+compiles a `response_format.json_schema.schema` document the same way production
/// decoding does (`OpenAIChatCompletions.decodeChatResponseFormat`), so these fixtures exercise the
/// real compiler, not a hand-built `JSONSchemaNode`.
private func compileSchema(
    _ json: String, name: String = "test_schema", strict: Bool = true
) throws -> JSONSchemaResponseFormat {
    let raw = try JSONSchemaSubsetCompiler.parseOrdered(Data(json.utf8))
    return try JSONSchemaSubsetCompiler.compile(name: name, schema: raw, strict: strict)
}

/// Runs the adversarial scorer through `processor` (or, when `nil`, completely unmasked) for up to
/// `maxSteps` greedy decode steps, stopping early if EOS is sampled — mirrors
/// `JSONObjectMaskingLogitProcessorTests.runGreedyLoop`.
private func runGreedyLoop(
    through processor: JSONSchemaMaskingLogitProcessor?, maxSteps: Int = 20,
    logits: () -> MLXArray = SchemaSyntheticVocab.adversarialLogits
) -> (bytes: [UInt8], sampledEOS: Bool) {
    var produced: [UInt8] = []
    let raw = logits()
    for _ in 0..<maxSteps {
        let scored = processor?.process(logits: raw) ?? raw
        let tokenArray = argMax(scored, axis: -1)
        processor?.didSample(token: tokenArray)
        let id = tokenArray.item(Int.self)
        if id == SchemaSyntheticVocab.eosID {
            return (produced, true)
        }
        produced.append(UInt8(id))
    }
    return (produced, false)
}

final class JSONSchemaMaskingLogitProcessorTests: XCTestCase {
    /// Acceptance test (response-format design, schema variant of json_object's own adversarial
    /// test): an adversarial scorer that always prefers a byte no schema-legal instance could ever
    /// contain, run through the REAL masking processor against a `{"type":"boolean"}` schema, still
    /// produces a well-formed, SCHEMA-VALID instance and terminates at EOS.
    func testAdversarialScorerThroughMaskProducesSchemaValidBooleanEndingInEOS() throws {
        let format = try compileSchema(#"{"type":"boolean"}"#)
        let processor = JSONSchemaMaskingLogitProcessor(
            constraint: JSONSchemaTokenConstraint(table: SchemaSyntheticVocab.table, format: format),
            activeFromStart: true,
            thinkEndTokenID: nil)

        let result = runGreedyLoop(through: processor)

        XCTAssertNil(processor.recordedFailure)
        XCTAssertTrue(result.sampledEOS, "expected the automaton to reach EOS")
        let text = String(decoding: result.bytes, as: UTF8.self)
        XCTAssertEqual(text, "true", "the boosted literal's bytes should win the masked walk")
        // Independent proof, not just "the compiler's own automaton says so": the produced bytes
        // parse as a genuine JSON boolean.
        let parsed = try JSONSerialization.jsonObject(
            with: Data("[\(text)]".utf8), options: [.fragmentsAllowed])
        XCTAssertEqual((parsed as? [Bool])?.first, true)
    }

    /// A richer schema (nested object with a required + an optional property, `additionalProperties:
    /// false`) — proves the masking processor's output validates against a REAL multi-property
    /// object shape, not just a scalar literal.
    func testAdversarialScorerProducesASchemaValidNestedObject() throws {
        let format = try compileSchema(
            """
            {
              "type": "object",
              "properties": {
                "ok": {"type": "boolean"},
                "count": {"type": "integer"}
              },
              "required": ["ok", "count"],
              "additionalProperties": false
            }
            """)
        let processor = JSONSchemaMaskingLogitProcessor(
            constraint: JSONSchemaTokenConstraint(table: SchemaSyntheticVocab.table, format: format),
            activeFromStart: true,
            thinkEndTokenID: nil)

        let result = runGreedyLoop(
            through: processor, maxSteps: 60, logits: SchemaSyntheticVocab.objectFixtureAdversarialLogits)

        XCTAssertNil(processor.recordedFailure)
        XCTAssertTrue(result.sampledEOS, "expected the automaton to reach EOS")
        let text = String(decoding: result.bytes, as: UTF8.self)
        let parsed = try JSONSerialization.jsonObject(with: Data(result.bytes)) as? [String: Any]
        let object = try XCTUnwrap(parsed, "expected a JSON OBJECT, got: \(text)")
        XCTAssertEqual(Set(object.keys), ["ok", "count"], "expected exactly the two required keys")
        XCTAssertTrue(object["ok"] is Bool)
        XCTAssertTrue(object["count"] is NSNumber)
    }

    // MARK: - Thinking-phase gate (response-format design item #4, shared with json_object)

    func testMaskStaysInactiveUntilThinkEndTokenThenActivates() throws {
        let format = try compileSchema(#"{"type":"boolean"}"#)
        let thinkEndTokenID = 300
        let processor = JSONSchemaMaskingLogitProcessor(
            constraint: JSONSchemaTokenConstraint(table: SchemaSyntheticVocab.table, format: format),
            activeFromStart: false,
            thinkEndTokenID: thinkEndTokenID)

        let raw = SchemaSyntheticVocab.adversarialLogits()
        for _ in 0..<5 {
            let scored = processor.process(logits: raw)
            XCTAssertEqual(
                scored.asArray(Float.self), raw.asArray(Float.self),
                "mask must be a no-op before the think-end token")
            let tokenArray = argMax(scored, axis: -1)
            processor.didSample(token: tokenArray)
            XCTAssertEqual(tokenArray.item(Int.self), Int(SchemaSyntheticVocab.adversarialByte))
        }

        processor.didSample(token: MLXArray([Int32(thinkEndTokenID)]))

        let postThinkScored = processor.process(logits: raw)
        let postThinkToken = argMax(postThinkScored, axis: -1).item(Int.self)
        XCTAssertEqual(
            postThinkToken, Int(UInt8(ascii: "t")),
            "mask must be active immediately after the think-end token")
        XCTAssertNil(processor.recordedFailure)
    }

    func testActiveFromStartMasksImmediatelyEvenWithAThinkEndTokenConfigured() throws {
        let format = try compileSchema(#"{"type":"boolean"}"#)
        let processor = JSONSchemaMaskingLogitProcessor(
            constraint: JSONSchemaTokenConstraint(table: SchemaSyntheticVocab.table, format: format),
            activeFromStart: true,
            thinkEndTokenID: 300)

        let raw = SchemaSyntheticVocab.adversarialLogits()
        let scored = processor.process(logits: raw)
        let token = argMax(scored, axis: -1).item(Int.self)

        XCTAssertEqual(token, Int(UInt8(ascii: "t")), "mask must already be active with activeFromStart: true")
    }

    // MARK: - Failure path (response-format design: fail closed, never silently succeed)

    /// Mirrors `JSONObjectMaskingLogitProcessorTests
    /// .testForcingADisallowedTokenRecordsAFailureInsteadOfSilentlyAdvancing`.
    func testForcingADisallowedTokenRecordsAFailureInsteadOfSilentlyAdvancing() throws {
        let format = try compileSchema(#"{"type":"boolean"}"#)
        let processor = JSONSchemaMaskingLogitProcessor(
            constraint: JSONSchemaTokenConstraint(table: SchemaSyntheticVocab.table, format: format),
            activeFromStart: true,
            thinkEndTokenID: nil)

        // The adversarial byte is never a legal FIRST byte of a `{"type":"boolean"}` instance.
        processor.didSample(token: MLXArray([Int32(SchemaSyntheticVocab.adversarialByte)]))

        XCTAssertNotNil(processor.recordedFailure)
        guard case .some(JSONSchemaConstraintError.tokenDisallowed(let id)) = processor.recordedFailure else {
            return XCTFail("expected .tokenDisallowed, got \(String(describing: processor.recordedFailure))")
        }
        XCTAssertEqual(id, Int(SchemaSyntheticVocab.adversarialByte))

        let raw = SchemaSyntheticVocab.adversarialLogits()
        XCTAssertEqual(processor.process(logits: raw).asArray(Float.self), raw.asArray(Float.self))
    }

    // MARK: - Logits wider than the table (mirrors json_object's should-fix #9 coverage)

    func testLogitsWiderThanTableAreMasked() throws {
        let format = try compileSchema(#"{"type":"boolean"}"#)
        let processor = JSONSchemaMaskingLogitProcessor(
            constraint: JSONSchemaTokenConstraint(table: SchemaSyntheticVocab.table, format: format),
            activeFromStart: true,
            thinkEndTokenID: nil)
        let widerVocabSize = SchemaSyntheticVocab.vocabSize + 40
        let raw = MLXArray([Float](repeating: 0, count: widerVocabSize)).reshaped([1, widerVocabSize])

        let masked = processor.process(logits: raw).asArray(Float.self)

        XCTAssertEqual(masked[Int(UInt8(ascii: "t"))], 0)
        for id in SchemaSyntheticVocab.vocabSize..<widerVocabSize {
            XCTAssertEqual(masked[id], -Float.infinity, "id \(id) beyond the table must be masked")
        }
        XCTAssertNil(processor.recordedFailure)
    }
}
