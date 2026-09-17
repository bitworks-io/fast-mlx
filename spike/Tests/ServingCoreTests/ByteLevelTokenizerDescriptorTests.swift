import Foundation
import XCTest

@testable import ServingCore

/// Pure JSON-parsing tests for `ByteLevelTokenizerDescriptor` — no MLX, no `Tokenizers`, no real
/// checkpoint. Each fixture is the minimal `tokenizer.json`/`generation_config.json` shape needed
/// to exercise ONE decision this type makes.
final class ByteLevelTokenizerDescriptorTests: XCTestCase {
    private func json(_ object: [String: Any]) -> Data {
        try! JSONSerialization.data(withJSONObject: object)
    }

    // MARK: - Byte-level decoder detection

    func testPlainByteLevelDecoderIsByteLevel() throws {
        let data = json([
            "decoder": ["type": "ByteLevel"],
            "model": ["vocab": ["a": 0, "b": 1]],
        ])
        let descriptor = try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: data)
        XCTAssertTrue(descriptor.isByteLevel)
    }

    func testSequenceDecoderContainingByteLevelIsByteLevel() throws {
        let data = json([
            "decoder": [
                "type": "Sequence",
                "decoders": [["type": "Replace"], ["type": "ByteLevel"]],
            ],
            "model": ["vocab": ["a": 0]],
        ])
        let descriptor = try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: data)
        XCTAssertTrue(descriptor.isByteLevel)
    }

    func testSequenceDecoderWithoutByteLevelIsNotByteLevel() throws {
        let data = json([
            "decoder": [
                "type": "Sequence",
                "decoders": [["type": "Replace"], ["type": "Strip"]],
            ],
            "model": ["vocab": ["a": 0]],
        ])
        let descriptor = try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: data)
        XCTAssertFalse(descriptor.isByteLevel)
    }

    /// Metaspace is the canonical SentencePiece-family decoder, and the canonical non-byte-level
    /// case the response-format design's "a tokenizer whose decoder is not ByteLevel is refused"
    /// rule exists for.
    func testMetaspaceDecoderIsNotByteLevel() throws {
        let data = json([
            "decoder": ["type": "Metaspace"],
            "model": ["vocab": ["a": 0]],
        ])
        let descriptor = try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: data)
        XCTAssertFalse(descriptor.isByteLevel)
    }

    func testMissingDecoderKeyIsNotByteLevel() throws {
        let data = json(["model": ["vocab": ["a": 0]]])
        let descriptor = try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: data)
        XCTAssertFalse(descriptor.isByteLevel)
    }

    // MARK: - Vocab + added tokens

    func testVocabIsInvertedToIDKeyedStrings() throws {
        let data = json([
            "decoder": ["type": "ByteLevel"],
            "model": ["vocab": ["!": 0, "\"": 1, "Ġ": 2]],
        ])
        let descriptor = try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: data)
        XCTAssertEqual(descriptor.vocabStringsByID[0], "!")
        XCTAssertEqual(descriptor.vocabStringsByID[1], "\"")
        XCTAssertEqual(descriptor.vocabStringsByID[2], "Ġ")
    }

    func testAddedTokenIdsAreCollectedRegardlessOfSpecialFlag() throws {
        let data = json([
            "decoder": ["type": "ByteLevel"],
            "model": ["vocab": ["a": 0]],
            "added_tokens": [
                ["id": 100, "content": "<|im_end|>", "special": true],
                ["id": 101, "content": "custom_token", "special": false],
            ],
        ])
        let descriptor = try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: data)
        XCTAssertEqual(descriptor.addedTokenIds, [100, 101])
    }

    func testAddedTokenIDsByContentIsKeyedByEachEntrysOwnContentString() throws {
        let data = json([
            "decoder": ["type": "ByteLevel"],
            "model": ["vocab": ["a": 0]],
            "added_tokens": [
                ["id": 100, "content": "<|im_end|>", "special": true],
                ["id": 101, "content": "</think>", "special": true],
            ],
        ])
        let descriptor = try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: data)
        XCTAssertEqual(descriptor.addedTokenIDsByContent["<|im_end|>"], 100)
        XCTAssertEqual(descriptor.addedTokenIDsByContent["</think>"], 101)
    }

    func testMissingAddedTokensYieldsEmptySet() throws {
        let data = json([
            "decoder": ["type": "ByteLevel"],
            "model": ["vocab": ["a": 0]],
        ])
        let descriptor = try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: data)
        XCTAssertTrue(descriptor.addedTokenIds.isEmpty)
    }

    // MARK: - Malformed / missing vocab

    func testMissingModelVocabThrowsMalformedVocab() {
        let data = json(["decoder": ["type": "ByteLevel"]])
        XCTAssertThrowsError(try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: data)) {
            XCTAssertEqual($0 as? ByteLevelTokenizerDescriptorError, .malformedVocab)
        }
    }

    func testGarbageBytesThrowMalformedVocab() {
        let data = Data("not json at all".utf8)
        XCTAssertThrowsError(try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: data)) {
            XCTAssertEqual($0 as? ByteLevelTokenizerDescriptorError, .malformedVocab)
        }
    }
}

/// `GenerationConfigEOSIds` never throws: a missing file is represented upstream by the caller
/// simply never calling `parse` with real bytes, and an unparsable/absent field resolves to an
/// EMPTY set here — these tests pin exactly that "absent means empty, not an error" contract.
final class GenerationConfigEOSIdsTests: XCTestCase {
    private func json(_ object: [String: Any]) -> Data {
        try! JSONSerialization.data(withJSONObject: object)
    }

    func testSingleIntegerEOSTokenID() {
        let ids = GenerationConfigEOSIds.parse(generationConfigJSON: json(["eos_token_id": 151645]))
        XCTAssertEqual(ids, [151645])
    }

    func testArrayOfEOSTokenIDs() {
        let ids = GenerationConfigEOSIds.parse(
            generationConfigJSON: json(["eos_token_id": [151643, 151645]]))
        XCTAssertEqual(ids, [151643, 151645])
    }

    func testAbsentFieldYieldsEmptySet() {
        let ids = GenerationConfigEOSIds.parse(generationConfigJSON: json(["do_sample": false]))
        XCTAssertTrue(ids.isEmpty)
    }

    func testUnreadableBytesYieldEmptySetNotAnError() {
        let ids = GenerationConfigEOSIds.parse(generationConfigJSON: Data("garbage".utf8))
        XCTAssertTrue(ids.isEmpty)
    }
}
