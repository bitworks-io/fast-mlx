import Foundation

/// Unified representation of a VALIDATED logprobs request, shared by both `/v1/chat/completions`
/// (`logprobs:true` + optional `top_logprobs`, OpenAI's own range 0...20) and the legacy
/// `/v1/completions` route (integer `logprobs`, this server's supported range 0...5 — `0` means
/// "the sampled token's logprob only, no alternatives"). `topLogprobs` is always in-range for the
/// case that produced it by the time this value exists: both decoders validate the range before
/// constructing one. `nil` (no `ServingLogprobsRequest` at all) means logprobs were not requested —
/// the ONLY state that must leave every existing response byte-for-byte unchanged.
public enum ServingLogprobsRequest: Sendable, Equatable {
    case chat(topLogprobs: Int)
    case completions(topLogprobs: Int)

    public var topLogprobs: Int {
        switch self {
        case .chat(let value), .completions(let value):
            return value
        }
    }
}

/// One alternative token a generation step considered, carried alongside the sampled token's own
/// `ServingTokenLogprob`. `logprob` is the same RAW (pre-sampling) log-probability described on
/// `ServingTokenLogprob`.
public struct ServingTokenLogprobCandidate: Equatable, Sendable {
    public let tokenText: String
    public let tokenBytes: [UInt8]
    public let logprob: Double

    public init(tokenText: String, logprob: Double) {
        self.tokenText = tokenText
        self.tokenBytes = Array(tokenText.utf8)
        self.logprob = logprob
    }

    /// Wire-safe `bytes` value for OpenAI's `top_logprobs[j].bytes`: `nil` when `tokenText`
    /// contains U+FFFD (REPLACEMENT CHARACTER) — this candidate is a byte-level BPE token that
    /// only covers part of a multi-byte UTF-8 character (e.g. one half of a split emoji), so
    /// decoding it alone cannot recover faithful bytes; `tokenBytes` in that case is the
    /// replacement character's OWN UTF-8 encoding, not the underlying token's real bytes.
    /// OpenAI's schema explicitly allows `bytes` to be null for exactly this case. Otherwise this
    /// is `tokenBytes` widened to `[Int]`, unchanged from today.
    public var wireBytes: [Int]? {
        tokenText.contains("\u{FFFD}") ? nil : tokenBytes.map(Int.init)
    }
}

/// One generated token's log-probability under the target model's RAW distribution: `log_softmax`
/// of the model's own output logits at that generation step, computed BEFORE temperature, top-p/
/// top-k, min-p, repetition/presence/frequency penalties, or any other logit processor runs. This
/// mirrors vLLM's own default `logprobs` semantics (the value most OpenAI-compatible servers
/// return) rather than the post-penalty/post-temperature distribution actually sampled from — see
/// README.md's logprobs section for the rationale and a worked example.
///
/// `topCandidates` holds the top-N tokens BY THAT SAME RAW LOG-PROBABILITY, sorted strictly
/// descending, with any `-infinity` candidate (a masked-out vocabulary entry — e.g. a media
/// sentinel or an unused special token) EXCLUDED — `-Infinity` is not valid JSON, so a candidate
/// list must never carry one. `topCandidates` may or may not include an entry equal to the sampled
/// token itself; this server does not deduplicate, matching OpenAI's own `top_logprobs` contract
/// (independent of `content[i]`/`tokens[i]`).
public struct ServingTokenLogprob: Equatable, Sendable {
    public let tokenText: String
    public let tokenBytes: [UInt8]
    public let logprob: Double
    public let topCandidates: [ServingTokenLogprobCandidate]

    public init(
        tokenText: String,
        logprob: Double,
        topCandidates: [ServingTokenLogprobCandidate] = []
    ) {
        self.tokenText = tokenText
        self.tokenBytes = Array(tokenText.utf8)
        self.logprob = logprob
        self.topCandidates = topCandidates
    }

    /// The value actually placed on the wire for `logprob`/`token_logprobs`: `-9999.0` in place of
    /// `-infinity` (see the type's doc comment), unchanged for every finite value. The generation
    /// loop should never actually select a token this server itself considers impossible, so this
    /// sentinel is a defensive fallback, not an expected path — but `JSONEncoder` cannot encode
    /// `-Infinity`/`NaN` at all, so an un-sanitized value here would crash response encoding rather
    /// than merely look wrong.
    public var sanitizedLogprob: Double {
        logprob.isFinite ? logprob : -9999.0
    }

    /// `topCandidates`, minus any `-infinity` entry — see the type's doc comment for why a
    /// candidate must never reach an encoded response. Producers are not required to pre-filter;
    /// every response builder below reads this instead of `topCandidates` directly.
    public var finiteTopCandidates: [ServingTokenLogprobCandidate] {
        topCandidates.filter { $0.logprob.isFinite }
    }

    /// Wire-safe `bytes` value for OpenAI's `content[i].bytes` — see
    /// `ServingTokenLogprobCandidate.wireBytes`'s doc comment for the U+FFFD rule this mirrors.
    public var wireBytes: [Int]? {
        tokenText.contains("\u{FFFD}") ? nil : tokenBytes.map(Int.init)
    }
}

/// Wire shape for `choices[0].logprobs` on both the non-streaming chat response and every
/// streaming chat chunk whose choice carries newly generated tokens. `logprobs` is a SIBLING of
/// `delta` on a streaming choice (never nested inside it) — matching OpenAI's actual
/// `chat.completion.chunk` shape. `refusal` is always encoded `null`: this server never marks
/// generated content as a refusal.
public struct OpenAIChatLogprobs: Encodable, Sendable, Equatable {
    public struct TopLogprob: Encodable, Sendable, Equatable {
        public var token: String
        public var logprob: Double
        /// `nil` (encoded JSON `null`, key always PRESENT — see `encode(to:)`) when the token has
        /// no faithful bytes representation: a byte-level BPE token that only covers part of a
        /// multi-byte UTF-8 character decodes alone to U+FFFD, and OpenAI's schema allows `bytes`
        /// to be null for exactly that case. See `ServingTokenLogprobCandidate.wireBytes`.
        public var bytes: [Int]?

        public init(token: String, logprob: Double, bytes: [Int]?) {
            self.token = token
            self.logprob = logprob
            self.bytes = bytes
        }

        private enum CodingKeys: String, CodingKey {
            case token
            case logprob
            case bytes
        }

        /// Custom `encode(to:)` because the synthesized `Encodable` conformance would use
        /// `encodeIfPresent` for the optional `bytes` and OMIT the key entirely when `nil` —
        /// OpenAI's contract requires the key to stay present with an explicit `null`.
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(token, forKey: .token)
            try container.encode(logprob, forKey: .logprob)
            if let bytes {
                try container.encode(bytes, forKey: .bytes)
            } else {
                try container.encodeNil(forKey: .bytes)
            }
        }
    }

    public struct Content: Encodable, Sendable, Equatable {
        public var token: String
        public var logprob: Double
        /// `nil` (encoded JSON `null`, key always PRESENT — see `encode(to:)`) under the same
        /// U+FFFD rule as `TopLogprob.bytes`.
        public var bytes: [Int]?
        public var topLogprobs: [TopLogprob]

        public init(token: String, logprob: Double, bytes: [Int]?, topLogprobs: [TopLogprob]) {
            self.token = token
            self.logprob = logprob
            self.bytes = bytes
            self.topLogprobs = topLogprobs
        }

        private enum CodingKeys: String, CodingKey {
            case token
            case logprob
            case bytes
            case topLogprobs = "top_logprobs"
        }

        /// Custom `encode(to:)` for the same reason as `TopLogprob.encode(to:)`: the synthesized
        /// conformance would omit `bytes` on `nil` instead of encoding an explicit `null`.
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(token, forKey: .token)
            try container.encode(logprob, forKey: .logprob)
            if let bytes {
                try container.encode(bytes, forKey: .bytes)
            } else {
                try container.encodeNil(forKey: .bytes)
            }
            try container.encode(topLogprobs, forKey: .topLogprobs)
        }
    }

    public var content: [Content]

    public init(content: [Content]) {
        self.content = content
    }

    /// Builds directly from generation-order tokens (see `ServingTokenLogprob`'s doc comment for
    /// what "generation order" covers — every generated token, including ones that rendered into
    /// `reasoning_content` or a tool call's arguments, never just the visible answer text).
    public init(tokens: [ServingTokenLogprob]) {
        self.init(
            content: tokens.map { token in
                Content(
                    token: token.tokenText,
                    logprob: token.sanitizedLogprob,
                    bytes: token.wireBytes,
                    topLogprobs: token.finiteTopCandidates.map {
                        TopLogprob(
                            token: $0.tokenText,
                            logprob: $0.logprob,
                            bytes: $0.wireBytes)
                    })
            })
    }

    private enum CodingKeys: String, CodingKey {
        case content
        case refusal
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(content, forKey: .content)
        try container.encodeNil(forKey: .refusal)
    }
}

/// Wire shape for the legacy `/v1/completions` `choices[0].logprobs` object, shared by the
/// non-streaming response and every streaming chunk that carries new tokens. `topLogprobs[i]` is
/// `nil` (encoded `null`) for every token when the caller's `logprobs` value was `0` — "the
/// sampled token's logprob only, no alternatives" — even though `tokens`/`token_logprobs` are
/// always fully populated. `textOffset[i]` is the Unicode scalar (code point) offset into the
/// FULL completion text where `tokens[i]` begins — matching the character-offset convention
/// OpenAI's own legacy completions API used, not a UTF-8 byte offset.
public struct OpenAICompletionLogprobs: Encodable, Sendable, Equatable {
    public var tokens: [String]
    public var tokenLogprobs: [Double]
    public var topLogprobs: [[String: Double]?]
    public var textOffset: [Int]

    public init(
        tokens: [String],
        tokenLogprobs: [Double],
        topLogprobs: [[String: Double]?],
        textOffset: [Int]
    ) {
        self.tokens = tokens
        self.tokenLogprobs = tokenLogprobs
        self.topLogprobs = topLogprobs
        self.textOffset = textOffset
    }

    private enum CodingKeys: String, CodingKey {
        case tokens
        case tokenLogprobs = "token_logprobs"
        case topLogprobs = "top_logprobs"
        case textOffset = "text_offset"
    }

    /// `requestedTopLogprobs` is the caller's original `logprobs` integer (0...5) — see the type's
    /// doc comment for why `0` forces every `topLogprobs[i]` entry to `nil` regardless of whether
    /// `tokens[i].topCandidates` is non-empty. `startingOffset` is the code-point offset into the
    /// OVERALL response text where this batch of tokens begins: `0` for the first batch of a
    /// response, and the running total of every earlier batch's token text lengths for a later
    /// streaming chunk or the final-chunk remainder.
    public static func build(
        tokens: [ServingTokenLogprob],
        requestedTopLogprobs: Int,
        startingOffset: Int
    ) -> OpenAICompletionLogprobs {
        var offsets: [Int] = []
        offsets.reserveCapacity(tokens.count)
        var offset = startingOffset
        for token in tokens {
            offsets.append(offset)
            offset += token.tokenText.count
        }
        return OpenAICompletionLogprobs(
            tokens: tokens.map(\.tokenText),
            tokenLogprobs: tokens.map(\.sanitizedLogprob),
            topLogprobs: tokens.map { token in
                guard requestedTopLogprobs > 0 else { return nil }
                var dict: [String: Double] = [:]
                for candidate in token.finiteTopCandidates {
                    dict[candidate.tokenText] = candidate.logprob
                }
                return dict
            },
            textOffset: offsets)
    }

    /// Total Unicode scalar count of every token's text, in order — the offset the NEXT batch
    /// (streaming chunk or final-chunk remainder) must start from.
    public static func endingOffset(startingOffset: Int, tokens: [ServingTokenLogprob]) -> Int {
        startingOffset + tokens.reduce(0) { $0 + $1.tokenText.count }
    }
}
