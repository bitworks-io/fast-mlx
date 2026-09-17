import Foundation

/// GPT-2 / byte-level BPE `bytes_to_unicode` table and its inverse.
///
/// A byte-level BPE tokenizer's vocab strings are NOT UTF-8 text: every raw byte 0x00...0xFF is
/// remapped to one printable Unicode scalar so the vocab can round-trip through a text-based
/// tokenizer.json. Recovering the real bytes a token represents (as opposed to decoding it, which
/// can turn a partial UTF-8 sequence into U+FFFD) requires inverting that table over the vocab
/// string, never `decode([id])`: a single token's bytes may be one half of a multi-byte UTF-8
/// character (e.g. a split-CJK token), and `decode([id])` on such a token substitutes U+FFFD
/// rather than exposing the raw bytes a JSON-grammar constraint needs to walk.
public enum ByteLevelTokenBytes {
    /// Maps each raw byte value (0...255) to the printable Unicode scalar GPT-2's
    /// `bytes_to_unicode()` assigns it. Built once, at first use.
    public static let byteToUnicodeScalar: [UInt8: Unicode.Scalar] = buildByteToUnicodeScalar()

    /// Inverse of `byteToUnicodeScalar`: printable Unicode scalar -> raw byte value.
    public static let unicodeScalarToByte: [Unicode.Scalar: UInt8] = {
        var inverse: [Unicode.Scalar: UInt8] = [:]
        inverse.reserveCapacity(byteToUnicodeScalar.count)
        for (byte, scalar) in byteToUnicodeScalar {
            inverse[scalar] = byte
        }
        return inverse
    }()

    /// Recovers the raw bytes a byte-level BPE vocab string represents.
    ///
    /// Returns `nil` if any Unicode scalar in `string` falls outside the byte-level mapping table
    /// (that vocab string cannot be a byte-level token — e.g. an added/special token whose vocab
    /// string is plain text such as `<|im_end|>`).
    public static func bytes(forVocabString string: String) -> [UInt8]? {
        var out: [UInt8] = []
        out.reserveCapacity(string.unicodeScalars.count)
        for scalar in string.unicodeScalars {
            guard let byte = unicodeScalarToByte[scalar] else {
                return nil
            }
            out.append(byte)
        }
        return out
    }

    private static func buildByteToUnicodeScalar() -> [UInt8: Unicode.Scalar] {
        // Mirrors HF's canonical `bytes_to_unicode()`: printable ASCII/Latin-1 bytes map to
        // themselves; every other byte value (control chars, space, DEL, the 0x80...0xFF gap
        // bytes not already covered) gets an assigned codepoint starting at U+0100.
        var identityBytes: [Int] = []
        identityBytes.append(contentsOf: 33...126)
        identityBytes.append(contentsOf: 161...172)
        identityBytes.append(contentsOf: 174...255)
        let identitySet = Set(identityBytes)

        var allBytes = identityBytes
        var codepoints = identityBytes
        var nextAssigned = 256
        for byte in 0...255 where !identitySet.contains(byte) {
            allBytes.append(byte)
            codepoints.append(nextAssigned)
            nextAssigned += 1
        }

        var map: [UInt8: Unicode.Scalar] = [:]
        map.reserveCapacity(256)
        for (byte, codepoint) in zip(allBytes, codepoints) {
            guard let scalar = Unicode.Scalar(codepoint) else {
                preconditionFailure("ByteLevelTokenBytes: codepoint \(codepoint) is not a valid Unicode scalar")
            }
            map[UInt8(byte)] = scalar
        }
        precondition(map.count == 256, "ByteLevelTokenBytes: expected all 256 byte values to be mapped")
        return map
    }
}

/// Per-token-id classification produced by inverting a byte-level vocab against the EOS and
/// added-token sets. Drives both the JSON grammar's byte trie (`.bytes` ids only) and which ids
/// are always disallowed (`.banned`) or allowed exactly when the grammar is complete (`.eos`).
public enum TokenByteClassification: Sendable, Equatable, Hashable {
    /// A byte-level content token: `advance` walks the automaton through these raw bytes.
    case bytes([UInt8])
    /// A stop token: allowed only once `TokenConstraint.isComplete` is true.
    case eos
    /// An added/special token, or a vocab string that could not be inverted to bytes: never
    /// allowed by a JSON grammar constraint.
    case banned
}

extension ByteLevelTokenBytes {
    /// Classifies every id in `0..<vocabSize` for constrained decoding.
    ///
    /// - Parameters:
    ///   - vocabString: Looks up a token id's raw vocab string (the byte-level-encoded form, as
    ///     stored in tokenizer.json — NOT `decode([id])`). `nil` means the id is unused/invalid.
    ///   - addedTokenIds: Added/special token ids (from tokenizer.json's `added_tokens`). Banned
    ///     unless also in `eosTokenIds`.
    ///   - eosTokenIds: The tokenizer's EOS id(s) plus any `generation_config` eos ids. Checked
    ///     before `addedTokenIds`, so an id that is both added and EOS classifies as `.eos`.
    public static func classify(
        vocabSize: Int,
        vocabString: (Int) -> String?,
        addedTokenIds: Set<Int>,
        eosTokenIds: Set<Int>
    ) -> [TokenByteClassification] {
        var out: [TokenByteClassification] = []
        out.reserveCapacity(vocabSize)
        for id in 0..<vocabSize {
            if eosTokenIds.contains(id) {
                out.append(.eos)
                continue
            }
            if addedTokenIds.contains(id) {
                out.append(.banned)
                continue
            }
            guard let vocabString = vocabString(id), let bytes = bytes(forVocabString: vocabString),
                  !bytes.isEmpty
            else {
                // An empty byte sequence would classify as a content token that `advance` treats
                // as a no-op (its `for byte in bytes` loop never runs): a decoder could emit it
                // for free, unboundedly, without making any grammar progress. Ban it instead.
                out.append(.banned)
                continue
            }
            out.append(.bytes(bytes))
        }
        return out
    }
}

/// A `.bytes` id whose recovered bytes, decoded as UTF-8, disagree with the tokenizer's own
/// `decode([id])` — reported by `ByteLevelTokenBytes.selfCheckMismatches`, never thrown, so a
/// caller can log and refuse rather than silently trusting a wrong inversion.
public struct TokenByteMismatch: Sendable, Equatable {
    public let id: Int
    /// UTF-8 decoding of the recovered bytes (what the constraint's trie will treat this id as).
    public let recoveredUTF8: String
    /// What `decode([id])` returned (or `"<nil>"` if it returned nothing).
    public let decoded: String

    public init(id: Int, recoveredUTF8: String, decoded: String) {
        self.id = id
        self.recoveredUTF8 = recoveredUTF8
        self.decoded = decoded
    }
}

extension ByteLevelTokenBytes {
    /// Self-check: for every `.bytes` id whose recovered bytes are themselves valid UTF-8, verify
    /// `decode([id])` agrees. Ids whose bytes are valid UTF-8 only as part of a larger multi-token
    /// sequence (e.g. one half of a split CJK character) are skipped here, since `decode([id])` is
    /// expected to disagree for those by design (it substitutes U+FFFD) — that's exactly why the
    /// constraint must never call `decode([id])` on its own.
    public static func selfCheckMismatches(
        classifications: [TokenByteClassification],
        decode: (Int) -> String?
    ) -> [TokenByteMismatch] {
        var mismatches: [TokenByteMismatch] = []
        for (id, classification) in classifications.enumerated() {
            guard case .bytes(let bytes) = classification else { continue }
            guard let recoveredUTF8 = String(bytes: bytes, encoding: .utf8) else { continue }
            let decoded = decode(id) ?? "<nil>"
            if decoded != recoveredUTF8 {
                mismatches.append(
                    TokenByteMismatch(id: id, recoveredUTF8: recoveredUTF8, decoded: decoded))
            }
        }
        return mismatches
    }
}
