import CoreFoundation
import Foundation

/// OpenAI's own documented cap on the number of items accepted in one `/v1/embeddings` batch
/// (`input` as an array of strings, or an array of token-id arrays). A single flat array of
/// integers is ONE pre-tokenized input regardless of its length and is not subject to this cap —
/// see `OpenAIEmbeddingsRequest.decodeStrict`.
public let openAIEmbeddingsMaximumInputCount = 2048

/// OpenAI's documented per-input token-count ceiling for embedding models (8191 tokens for
/// `text-embedding-ada-002` and the `text-embedding-3-*` family). Applies to each pre-tokenized
/// `input` entry: a single flat array of token ids, or one row of an array-of-token-arrays batch.
/// Text `input` entries are tokenized downstream by the loaded model's tokenizer and are not
/// measured against this cap here.
public let openAIEmbeddingsMaximumTokensPerInput = 8191

/// OpenAI's documented aggregate token-count ceiling summed across every pre-tokenized input in
/// one `/v1/embeddings` request, independent of how that budget is split across individual
/// inputs. Like `openAIEmbeddingsMaximumTokensPerInput`, this only applies to pre-tokenized
/// (integer) inputs — text inputs are not counted here.
public let openAIEmbeddingsMaximumTotalTokens = 300_000

public enum OpenAIEmbeddingsEncodingFormat: String, Sendable, Equatable {
    case float
    case base64
}

/// One item of the decoded `input` field: either raw text (tokenized downstream by the loaded
/// model's tokenizer) or a pre-tokenized sequence of non-negative token ids supplied directly by
/// the caller. OpenAI's wire shape overloads `input` as EITHER a batch-of-strings OR a
/// batch-of-token-arrays OR a single flat array of integers (one pre-tokenized input) — never a
/// mix of shapes within one request.
public enum OpenAIEmbeddingsInput: Sendable, Equatable {
    case text(String)
    case tokens([Int])
}

/// The pure, MLX-free request contract for `POST /v1/embeddings`. No HTTP routing and no model
/// loading happens here — `decodeStrict` only parses and validates the JSON body.
public struct OpenAIEmbeddingsRequest: Sendable, Equatable {
    public var model: String
    public var inputs: [OpenAIEmbeddingsInput]
    public var encodingFormat: OpenAIEmbeddingsEncodingFormat
    public var dimensions: Int?
    /// Same contract as `OpenAIChatCompletionRequest.ignoredFields`: sorted, dedup-free field
    /// NAMES only (never values) for accepted-but-ignored top-level fields. Contains `"user"` when
    /// that field was present, plus one sanitized `"unknown:<key>"` entry per unrecognized
    /// top-level key (see `decodeStrict`'s unknown-key policy note and `rejectUnknownEmbeddingsKeys`).
    public var ignoredFields: [String]

    public init(
        model: String,
        inputs: [OpenAIEmbeddingsInput],
        encodingFormat: OpenAIEmbeddingsEncodingFormat = .float,
        dimensions: Int? = nil,
        ignoredFields: [String] = []
    ) {
        self.model = model
        self.inputs = inputs
        self.encodingFormat = encodingFormat
        self.dimensions = dimensions
        self.ignoredFields = ignoredFields
    }

    /// Unknown-top-level-key policy: mirrors `OpenAIChatCompletions.swift`'s lenient
    /// `rejectUnknownTopLevelKeys` policy (that helper is file-`private` and not reusable here
    /// without editing that file, which is out of scope for this slice, so the same
    /// sanitize-and-cap behaviour is replicated locally). Any top-level key outside `allowedKeys`
    /// is tolerated rather than a 400: it is recorded into `ignoredFields` as a sanitized,
    /// length-capped `"unknown:<key>"` warning (see `rejectUnknownEmbeddingsKeys`). Keys the
    /// embeddings API documents but this server cannot honour would still be a hard 400 via
    /// `semanticallyUnsupportedEmbeddingsKeys` — empty today because every documented
    /// `/v1/embeddings` field (`model`, `input`, `encoding_format`, `dimensions`, `user`) is
    /// already implemented below.
    public static func decodeStrict(
        from data: Data,
        limits: OpenAIChatRequestLimits = .productionDefault,
        maximumInputCount: Int = openAIEmbeddingsMaximumInputCount,
        maximumTokensPerInput: Int = openAIEmbeddingsMaximumTokensPerInput,
        maximumTotalTokens: Int = openAIEmbeddingsMaximumTotalTokens
    ) throws -> OpenAIEmbeddingsRequest {
        guard data.count <= limits.maximumBodyBytes else {
            throw OpenAIServingError.invalidRequest(
                "Request body exceeds the configured byte limit", param: nil)
        }

        let root = try decodeEmbeddingsJSONObject(data)
        let allowedKeys: Set<String> = [
            "model",
            "input",
            "encoding_format",
            "dimensions",
            "user",
        ]
        let unknownKeyWarnings = try rejectUnknownEmbeddingsKeys(in: root, allowed: allowedKeys)

        let model = try embeddingsRequiredString(root["model"], param: "model")
        guard !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OpenAIServingError.invalidRequest("model must be a non-empty string", param: "model")
        }

        let inputs = try decodeEmbeddingsInput(
            root["input"],
            maximumInputCount: maximumInputCount,
            maximumTokensPerInput: maximumTokensPerInput,
            maximumTotalTokens: maximumTotalTokens)
        let encodingFormat = try decodeEmbeddingsEncodingFormat(root["encoding_format"])
        let dimensions = try optionalPositiveEmbeddingsInt(root["dimensions"], param: "dimensions")

        let user = try embeddingsOptionalUser(root["user"])
        var ignoredFields: [String] = unknownKeyWarnings
        if user != nil { ignoredFields.append("user") }
        ignoredFields.sort()

        return OpenAIEmbeddingsRequest(
            model: model,
            inputs: inputs,
            encodingFormat: encodingFormat,
            dimensions: dimensions,
            ignoredFields: ignoredFields)
    }
}

// MARK: - Response

public struct OpenAIEmbeddingsUsage: Encodable, Sendable, Equatable {
    public var promptTokens: Int
    /// Embeddings requests never generate completion tokens, so `total_tokens` always equals
    /// `prompt_tokens` — matching OpenAI's own `/v1/embeddings` usage shape.
    public var totalTokens: Int { promptTokens }

    public init(promptTokens: Int) {
        self.promptTokens = promptTokens
    }

    private enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case totalTokens = "total_tokens"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(promptTokens, forKey: .promptTokens)
        try container.encode(totalTokens, forKey: .totalTokens)
    }
}

/// One encoded embedding vector, carrying whichever wire representation the request's
/// `encoding_format` selected. `float` encodes as a JSON number array; `base64` encodes as the
/// base64 of the vector's little-endian float32 bytes (the shape the OpenAI SDK itself decodes).
enum OpenAIEmbeddingsEncodedVector: Sendable, Equatable {
    case floats([Float])
    case base64(String)
}

public struct OpenAIEmbeddingsResponse: Encodable, Sendable, Equatable {
    public var object = "list"
    public var data: [Datum]
    public var model: String
    public var usage: OpenAIEmbeddingsUsage

    public struct Datum: Encodable, Sendable, Equatable {
        public var index: Int
        let embedding: OpenAIEmbeddingsEncodedVector

        private enum CodingKeys: String, CodingKey {
            case object
            case index
            case embedding
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode("embedding", forKey: .object)
            try container.encode(index, forKey: .index)
            switch embedding {
            case .floats(let values):
                try container.encode(values, forKey: .embedding)
            case .base64(let string):
                try container.encode(string, forKey: .embedding)
            }
        }
    }
}

/// Builds `OpenAIEmbeddingsResponse` values from raw embedding vectors. Kept separate from the
/// response type itself so the non-finite-value guard (JSON cannot carry NaN/Infinity) is
/// enforced at one single construction site.
public enum OpenAIEmbeddingsResponseBuilder {
    /// - Parameters:
    ///   - embeddings: One vector per input, in the same order as the request's `inputs`.
    ///   - promptTokens: Total input token count across every input in the batch.
    /// - Throws: `OpenAIServingError.server` (never `.invalidRequest` — a non-finite value here is
    ///   a backend defect, not a malformed client request) if any vector contains a NaN or
    ///   infinite value, since JSON cannot represent either.
    public static func encode(
        embeddings: [[Float]],
        model: String,
        encodingFormat: OpenAIEmbeddingsEncodingFormat,
        promptTokens: Int
    ) throws -> OpenAIEmbeddingsResponse {
        var data: [OpenAIEmbeddingsResponse.Datum] = []
        data.reserveCapacity(embeddings.count)
        for (index, vector) in embeddings.enumerated() {
            for value in vector where !value.isFinite {
                throw OpenAIServingError.server(
                    "embedding at index \(index) contains a non-finite value",
                    code: "non_finite_embedding")
            }
            let encoded: OpenAIEmbeddingsEncodedVector
            switch encodingFormat {
            case .float:
                encoded = .floats(vector)
            case .base64:
                encoded = .base64(base64Encode(vector))
            }
            data.append(OpenAIEmbeddingsResponse.Datum(index: index, embedding: encoded))
        }
        return OpenAIEmbeddingsResponse(
            data: data,
            model: model,
            usage: OpenAIEmbeddingsUsage(promptTokens: promptTokens))
    }

    /// Little-endian float32 bytes, base64-encoded — explicitly normalized to little-endian
    /// (rather than relying on the host's native byte order) so this is portable to a
    /// hypothetical big-endian host.
    private static func base64Encode(_ vector: [Float]) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(vector.count * 4)
        for value in vector {
            let bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: bits) { bytes.append(contentsOf: $0) }
        }
        return Data(bytes).base64EncodedString()
    }
}

// MARK: - Decoding helpers

private func decodeEmbeddingsJSONObject(_ data: Data) throws -> [String: Any] {
    do {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw OpenAIServingError.invalidRequest("Request body must be a JSON object", param: nil)
        }
        return object
    } catch let error as OpenAIServingError {
        throw error
    } catch {
        throw OpenAIServingError.invalidRequest("Request body is not valid JSON", param: nil)
    }
}

/// Top-level embeddings keys that OpenAI documents but this server cannot honor. Empty today:
/// every documented `/v1/embeddings` request field (`model`, `input`, `encoding_format`,
/// `dimensions`, `user`) is already implemented in `decodeStrict`. Kept as a named set (mirroring
/// `OpenAIChatCompletions.semanticallyUnsupportedTopLevelKeys`) so a future request field added to
/// `decodeStrict`'s `allowedKeys` without full support has an obvious place to land as a hard 400
/// instead of silently falling into the lenient unknown-key warning path below.
private let semanticallyUnsupportedEmbeddingsKeys: Set<String> = []

/// Lenient top-level-unknown-key policy, replicated from `OpenAIChatCompletions.rejectUnknownTopLevelKeys`
/// (that helper is file-`private` and not reusable here). An unrecognized top-level key is no
/// longer a 400 — it is tolerated and returned as one `"unknown:<key>"` entry per key for the
/// caller to fold into `ignoredFields`, EXCEPT `semanticallyUnsupportedEmbeddingsKeys`, which still
/// throws `Unsupported field: <key>` (fail-closed).
///
/// Never returns raw values, only key names — and only a sanitized shape of the name
/// (`^[A-Za-z0-9_.-]{1,64}$`); any key outside that shape collapses into a single
/// `"unknown:<invalid-key>"` sentinel instead of being echoed verbatim, so an attacker-controlled
/// key cannot inject unexpected bytes into any comma-joined diagnostic log line this ultimately
/// feeds. The result is capped at 16 recorded entries plus one trailing `"unknown:<more>"`
/// sentinel, so a request carrying hundreds of junk keys cannot inflate `ignoredFields`
/// unboundedly.
private func rejectUnknownEmbeddingsKeys(in object: [String: Any], allowed: Set<String>) throws -> [String] {
    var entries: [String] = []
    var hasInvalidKeyName = false
    for key in object.keys.sorted() where !allowed.contains(key) {
        if semanticallyUnsupportedEmbeddingsKeys.contains(key) {
            throw OpenAIServingError.invalidRequest("Unsupported field: \(key)", param: key)
        }
        if isValidUnknownEmbeddingsKeyName(key) {
            entries.append("unknown:\(key)")
        } else {
            hasInvalidKeyName = true
        }
    }
    if hasInvalidKeyName {
        entries.append("unknown:<invalid-key>")
    }
    if entries.count > 16 {
        entries = Array(entries.prefix(16))
        entries.append("unknown:<more>")
    }
    return entries
}

private func isValidUnknownEmbeddingsKeyName(_ key: String) -> Bool {
    guard (1...64).contains(key.count) else { return false }
    for scalar in key.unicodeScalars {
        switch scalar {
        case "A"..."Z", "a"..."z", "0"..."9", "_", ".", "-":
            continue
        default:
            return false
        }
    }
    return true
}

private func embeddingsRequiredString(_ raw: Any?, param: String) throws -> String {
    guard let string = raw as? String else {
        throw OpenAIServingError.invalidRequest("\(param) must be a string", param: param)
    }
    return string
}

private func embeddingsOptionalUser(_ raw: Any?) throws -> String? {
    guard let raw, !(raw is NSNull) else { return nil }
    guard let value = raw as? String else {
        throw OpenAIServingError.invalidRequest("user must be a string", param: "user")
    }
    return value
}

private func decodeEmbeddingsEncodingFormat(_ raw: Any?) throws -> OpenAIEmbeddingsEncodingFormat {
    guard let raw, !(raw is NSNull) else { return .float }
    guard let value = raw as? String, let format = OpenAIEmbeddingsEncodingFormat(rawValue: value) else {
        throw OpenAIServingError.invalidRequest(
            "encoding_format must be \"float\" or \"base64\"", param: "encoding_format")
    }
    return format
}

/// Strict JSON-integer decoder shared by `dimensions` and token ids: a value only passes if it
/// was written as a JSON integer literal (no decimal point, no exponent) — `CFNumberIsFloatType`
/// distinguishes `1` (stored by `JSONSerialization` as an integer `NSNumber`) from `1.0` or `1e2`
/// (stored as a `Double` `NSNumber`), so a float-typed literal is refused even when its value is
/// mathematically whole. Booleans are excluded by the existing `CFBooleanGetTypeID` check (JSON
/// `true`/`false` also decode to `NSNumber` under `JSONSerialization`). `typeErrorMessage` is
/// reported for every non-integer shape (float, bool, string, out-of-range) so callers can give a
/// single, unambiguous type error distinct from any value-range error they add afterward.
private func embeddingsExactInteger(_ raw: Any, typeErrorMessage: String, param: String) throws -> Int {
    guard let number = raw as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else {
        throw OpenAIServingError.invalidRequest(typeErrorMessage, param: param)
    }
    guard !CFNumberIsFloatType(number) else {
        throw OpenAIServingError.invalidRequest(typeErrorMessage, param: param)
    }
    let decimal = number.decimalValue
    guard decimal >= Decimal(Int.min), decimal <= Decimal(Int.max) else {
        throw OpenAIServingError.invalidRequest(typeErrorMessage, param: param)
    }
    guard let value = Int(NSDecimalNumber(decimal: decimal).stringValue) else {
        throw OpenAIServingError.invalidRequest(typeErrorMessage, param: param)
    }
    return value
}

private func optionalPositiveEmbeddingsInt(_ raw: Any?, param: String) throws -> Int? {
    guard let raw, !(raw is NSNull) else { return nil }
    let value = try embeddingsExactInteger(
        raw, typeErrorMessage: "\(param) must be a positive integer", param: param)
    guard value > 0 else {
        throw OpenAIServingError.invalidRequest("\(param) must be greater than zero", param: param)
    }
    return value
}

private enum EmbeddingsInputShape {
    case strings
    case tokenSequences
    case tokenIds
}

private func classifyEmbeddingsInputShape(_ first: Any) -> EmbeddingsInputShape? {
    if first is String { return .strings }
    if first is [Any] { return .tokenSequences }
    if let number = first as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() { return .tokenIds }
    return nil
}

/// `input` accepts: a non-empty string; a non-empty array of non-empty strings; a non-empty flat
/// array of non-negative integers (ONE pre-tokenized input); or a non-empty array of non-empty
/// integer arrays (multiple pre-tokenized inputs). The array's FIRST element's type determines
/// which of the three array shapes is expected — every subsequent element must match it, or the
/// whole request is rejected as mixed element types.
private func decodeEmbeddingsInput(
    _ raw: Any?,
    maximumInputCount: Int,
    maximumTokensPerInput: Int,
    maximumTotalTokens: Int
) throws -> [OpenAIEmbeddingsInput] {
    guard let raw, !(raw is NSNull) else {
        throw OpenAIServingError.invalidRequest("input is required", param: "input")
    }

    if let string = raw as? String {
        guard !string.isEmpty else {
            throw OpenAIServingError.invalidRequest("input must not be empty", param: "input")
        }
        return [.text(string)]
    }

    guard let array = raw as? [Any] else {
        throw OpenAIServingError.invalidRequest(
            "input must be a string, an array of strings, an array of integers, or an array of integer arrays",
            param: "input")
    }
    guard let first = array.first else {
        throw OpenAIServingError.invalidRequest("input must not be empty", param: "input")
    }
    guard let shape = classifyEmbeddingsInputShape(first) else {
        throw OpenAIServingError.invalidRequest(
            "input entries must be strings or integers", param: "input")
    }

    switch shape {
    case .strings:
        var texts: [String] = []
        texts.reserveCapacity(array.count)
        for element in array {
            guard let text = element as? String else {
                throw OpenAIServingError.invalidRequest(
                    "input entries must all be the same type", param: "input")
            }
            guard !text.isEmpty else {
                throw OpenAIServingError.invalidRequest("input entries must be non-empty strings", param: "input")
            }
            texts.append(text)
        }
        guard texts.count <= maximumInputCount else {
            throw OpenAIServingError.invalidRequest(
                "input must contain at most \(maximumInputCount) items", param: "input")
        }
        return texts.map { .text($0) }

    case .tokenSequences:
        var sequences: [[Int]] = []
        sequences.reserveCapacity(array.count)
        for element in array {
            guard let inner = element as? [Any] else {
                throw OpenAIServingError.invalidRequest(
                    "input entries must all be the same type", param: "input")
            }
            sequences.append(try decodeEmbeddingsTokenArray(inner, maximumTokensPerInput: maximumTokensPerInput))
        }
        guard sequences.count <= maximumInputCount else {
            throw OpenAIServingError.invalidRequest(
                "input must contain at most \(maximumInputCount) items", param: "input")
        }
        let totalTokens = sequences.reduce(0) { $0 + $1.count }
        guard totalTokens <= maximumTotalTokens else {
            throw OpenAIServingError.invalidRequest(
                "input must contain at most \(maximumTotalTokens) tokens total across all inputs",
                param: "input")
        }
        return sequences.map { .tokens($0) }

    case .tokenIds:
        // The whole array is ONE pre-tokenized input, not a batch — not subject to the item-count cap.
        let tokens = try decodeEmbeddingsTokenArray(array, maximumTokensPerInput: maximumTokensPerInput)
        guard tokens.count <= maximumTotalTokens else {
            throw OpenAIServingError.invalidRequest(
                "input must contain at most \(maximumTotalTokens) tokens total across all inputs",
                param: "input")
        }
        return [.tokens(tokens)]
    }
}

private func decodeEmbeddingsTokenArray(_ raw: [Any], maximumTokensPerInput: Int) throws -> [Int] {
    guard !raw.isEmpty else {
        throw OpenAIServingError.invalidRequest("input token arrays must not be empty", param: "input")
    }
    guard raw.count <= maximumTokensPerInput else {
        throw OpenAIServingError.invalidRequest(
            "input token arrays must contain at most \(maximumTokensPerInput) tokens", param: "input")
    }
    return try raw.map { try decodeEmbeddingsTokenId($0) }
}

private func decodeEmbeddingsTokenId(_ raw: Any) throws -> Int {
    let value = try embeddingsExactInteger(
        raw,
        typeErrorMessage: "input token ids must be non-negative integers",
        param: "input")
    guard value >= 0 else {
        throw OpenAIServingError.invalidRequest("input token ids must be non-negative integers", param: "input")
    }
    return value
}
