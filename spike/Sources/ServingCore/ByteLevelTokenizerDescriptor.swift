import Foundation

/// Parsed subset of a byte-level BPE `tokenizer.json` needed to build a `json_object` grammar
/// constraint's byte classification table (`ByteLevelTokenBytes.classify` — see that type's doc
/// comment). Parsing only, with NO `Tokenizers`/MLX dependency: the byte-level/non-byte-level
/// DECISION and the vocab/added-token extraction are pure JSON decoding, testable without ever
/// loading a real tokenizer or model.
public struct ByteLevelTokenizerDescriptor: Equatable, Sendable {
    /// True when the top-level `decoder.type == "ByteLevel"`, or `decoder.type == "Sequence"` and
    /// at least one entry of `decoder.decoders` has `type == "ByteLevel"`. Any other decoder (e.g.
    /// `Metaspace`, used by SentencePiece-family tokenizers), or a `tokenizer.json` with no
    /// top-level `decoder` at all, is NOT byte-level — per the response-format design, such a
    /// tokenizer refuses `json_object` rather than attempting the byte-inversion this constraint
    /// depends on.
    public let isByteLevel: Bool
    /// `model.vocab`, inverted from `vocab string -> id` to `id -> vocab string` — the shape
    /// `ByteLevelTokenBytes.classify`'s `vocabString` closure needs. This is the RAW byte-level-
    /// encoded form stored in `tokenizer.json`, never `decode([id])` (see `ByteLevelTokenBytes`'s
    /// doc comment for why those differ).
    public let vocabStringsByID: [Int: String]
    /// `added_tokens[].id` — every added/special token id, regardless of that entry's own
    /// `special` flag: both `special: true` control tokens (e.g. `<|im_end|>`) and `special: false`
    /// custom added-vocabulary entries are, per the response-format design, banned unless also EOS.
    public let addedTokenIds: Set<Int>
    /// `added_tokens[].content -> id`, the same entries as `addedTokenIds` but keyed by their own
    /// `content` string. Exists so a caller can resolve a SPECIFIC added token (e.g. a thinking
    /// model's end-of-reasoning marker) directly from the tokenizer.json's own added-token table,
    /// without going through `Tokenizer.convertTokenToId`, which returns the unknown-token id
    /// (never `nil`) for a token some BPE tokenizers do not otherwise resolve — see
    /// `loadScalarServingJSONObjectConstraintSupport`'s `</think>` resolution for why both sources
    /// are consulted rather than trusting `convertTokenToId` alone.
    public let addedTokenIDsByContent: [String: Int]

    public init(
        isByteLevel: Bool,
        vocabStringsByID: [Int: String],
        addedTokenIds: Set<Int>,
        addedTokenIDsByContent: [String: Int] = [:]
    ) {
        self.isByteLevel = isByteLevel
        self.vocabStringsByID = vocabStringsByID
        self.addedTokenIds = addedTokenIds
        self.addedTokenIDsByContent = addedTokenIDsByContent
    }
}

public enum ByteLevelTokenizerDescriptorError: Error, Equatable, Sendable {
    /// The JSON root is not an object, `model` is not an object, or `model.vocab` is not itself a
    /// JSON object of `string -> integer`. A `tokenizer.json` this malformed cannot support the
    /// grammar constraint at all (there is no vocabulary to classify) — a genuine parse failure,
    /// distinct from "well-formed but not byte-level" (which is `isByteLevel == false`, not a throw).
    case malformedVocab
}

extension ByteLevelTokenizerDescriptor {
    /// Parses a `tokenizer.json`'s bytes. Throws `.malformedVocab` only when the vocabulary itself
    /// cannot be recovered; every other shape question (byte-level or not, added tokens present or
    /// not) is reported as plain field values on the returned descriptor, never a throw.
    public static func parse(tokenizerJSON data: Data) throws -> ByteLevelTokenizerDescriptor {
        guard
            let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let model = root["model"] as? [String: Any],
            let rawVocab = model["vocab"] as? [String: Any]
        else {
            throw ByteLevelTokenizerDescriptorError.malformedVocab
        }

        var vocabStringsByID: [Int: String] = [:]
        vocabStringsByID.reserveCapacity(rawVocab.count)
        for (tokenString, rawID) in rawVocab {
            guard let id = (rawID as? NSNumber)?.intValue else { continue }
            vocabStringsByID[id] = tokenString
        }

        var addedTokenIds: Set<Int> = []
        var addedTokenIDsByContent: [String: Int] = [:]
        if let addedTokens = root["added_tokens"] as? [[String: Any]] {
            for entry in addedTokens {
                guard let id = (entry["id"] as? NSNumber)?.intValue else { continue }
                addedTokenIds.insert(id)
                if let content = entry["content"] as? String {
                    addedTokenIDsByContent[content] = id
                }
            }
        }

        return ByteLevelTokenizerDescriptor(
            isByteLevel: decoderIsByteLevel(root["decoder"]),
            vocabStringsByID: vocabStringsByID,
            addedTokenIds: addedTokenIds,
            addedTokenIDsByContent: addedTokenIDsByContent)
    }

    private static func decoderIsByteLevel(_ rawDecoder: Any?) -> Bool {
        guard let decoder = rawDecoder as? [String: Any] else {
            return false
        }
        if (decoder["type"] as? String) == "ByteLevel" {
            return true
        }
        guard (decoder["type"] as? String) == "Sequence",
            let subDecoders = decoder["decoders"] as? [[String: Any]]
        else {
            return false
        }
        return subDecoders.contains { ($0["type"] as? String) == "ByteLevel" }
    }
}

/// Parses `generation_config.json`'s `eos_token_id`, which Hugging Face ships as either a single
/// integer or an array of integers.
public enum GenerationConfigEOSIds {
    /// A missing file, unreadable/non-object JSON, or an absent `eos_token_id` field all yield an
    /// EMPTY set — never an error/throw. `generation_config.json` is optional, and its absence must
    /// not itself disable the `json_object` feature; only the CALLER'S combined EOS set (this
    /// result unioned with the tokenizer's own EOS id) ending up empty does that.
    public static func parse(generationConfigJSON data: Data) -> Set<Int> {
        guard let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return []
        }
        if let array = root["eos_token_id"] as? [Any] {
            return Set(array.compactMap { ($0 as? NSNumber)?.intValue })
        }
        if let single = (root["eos_token_id"] as? NSNumber)?.intValue {
            return [single]
        }
        return []
    }
}
