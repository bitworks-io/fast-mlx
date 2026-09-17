import XCTest

@testable import ServingCore

/// Differential correctness test for the 1e miss-path optimization (allocation-free trie walk +
/// truncated mask-cache key): proves `JSONObjectTokenConstraint.allowedTokenIds()` — the OPTIMIZED
/// path — agrees EXACTLY, at every compared automaton state, with a naive REFERENCE that copies a
/// fresh `JSONObjectAutomaton` per candidate token and feeds its bytes one at a time (the textbook
/// definition of "this token is allowed here"). The reference never touches the trie, the mask
/// cache, or the truncated cache key — it is the ground truth those exist to speed up, not
/// approximate.
///
/// States are reached via deterministic pseudo-random valid JSON-object PREFIXES (nesting up to
/// ~20, arrays and objects mixed, strings with escapes including `\uXXXX`, numbers with
/// fraction/exponent, literals, and structural whitespace runs), including the completed-document
/// state, against a synthetic vocab (~3,000 ids) whose longest `.bytes` token is deliberately SHORT
/// (`DiffVocab.maxTokenLength`), so `depth > maxTokenLength` states — the ones the truncated cache
/// key's `hasHiddenFrames` bit specifically exists to keep correct — are actually exercised, not
/// just theoretically possible. That coverage is asserted explicitly, not assumed (see
/// `testDifferentialAgreementAcrossManyStates`'s final assertion).
final class JSONObjectConstraintDifferentialTests: XCTestCase {
    // MARK: - Deterministic PRNG (same xorshift construction as the release micro-benchmark, for
    // reproducibility across test runs and easy bisection when a failure needs a fixed seed).

    private struct DeterministicRNG {
        var state: UInt64
        init(seed: UInt64) { self.state = seed }
        mutating func nextRaw() -> UInt64 {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return state
        }
        mutating func nextInt(_ bound: Int) -> Int { Int(nextRaw() % UInt64(bound)) }
        mutating func nextBool() -> Bool { nextRaw() % 2 == 0 }
    }

    // MARK: - Synthetic vocab

    /// All 256 single raw bytes (ids `0...255`) — guarantees ANY byte sequence can be exactly
    /// tokenized (greedy-longest-match with a single-byte fallback always terminates) — plus a
    /// curated set of multi-byte tokens (raw multi-byte UTF-8 content, multi-bracket closers like
    /// `"]}}"`, JSON literal fragments) and pseudo-random filler, all capped at `maxTokenLength`
    /// bytes so the trie's actual longest token stays small and deliberately exercises the
    /// mask-cache key's stack-truncation path at realistic nesting depths. Ends with one EOS id and
    /// two banned ids (mirroring `AdversarialVocab`'s added/special-token shape elsewhere in this
    /// target).
    private enum DiffVocab {
        static let maxTokenLength = 4

        private static let built = build()
        static var classifications: [TokenByteClassification] { built.classifications }
        static var eosId: Int { built.eosId }
        static var bannedIds: [Int] { built.bannedIds }

        private static func build() -> (
            classifications: [TokenByteClassification], eosId: Int, bannedIds: [Int]
        ) {
            var out: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }

            // Curated multi-byte tokens, every one <= maxTokenLength bytes.
            let curated: [[UInt8]] = [
                [0x5D, 0x7D],  // "]}"
                [0x7D, 0x7D],  // "}}"
                [0x5D, 0x5D],  // "]]"
                [0x5D, 0x7D, 0x7D],  // "]}}" — closes an array then two objects in one token
                [0x5D, 0x7D, 0x5D, 0x7D],  // "]}]}" — closes across two container kinds
                [0x22, 0x3A],  // "\":"
                [0x7B, 0x22],  // "{\""
                Array("true".utf8),
                Array("null".utf8),
                [0xC3, 0xA9],  // "é" as one raw 2-byte UTF-8 token
                [0xF0, 0x9F, 0x98, 0x80],  // "😀" as one raw 4-byte UTF-8 token
                [0xE4, 0xBD],  // split-CJK head (matches the head half of "你")
                [0xA0],  // split-CJK tail
                Array("1.5".utf8),
                Array("e+1".utf8),
            ]
            for bytes in curated {
                precondition(bytes.count <= maxTokenLength)
                out.append(.bytes(bytes))
            }

            // Pseudo-random printable-ASCII filler, length 1...maxTokenLength, to reach a few
            // thousand ids (a realistic-scale vocab, not just the hand-picked cases above).
            var rng = DeterministicRNG(seed: 0xD1FF_5EED_1234_5678)
            let printable: [UInt8] = Array(0x20...0x7E)
            let targetVocabSize = 3_000
            while out.count < targetVocabSize - 3 {
                let length = 1 + rng.nextInt(maxTokenLength)
                var bytes: [UInt8] = []
                bytes.reserveCapacity(length)
                for _ in 0..<length {
                    bytes.append(printable[rng.nextInt(printable.count)])
                }
                out.append(.bytes(bytes))
            }

            let eosId = out.count
            out.append(.eos)
            let banned1 = out.count
            out.append(.banned)
            let banned2 = out.count
            out.append(.banned)

            let actualMaxLength = out.compactMap { classification -> Int? in
                guard case .bytes(let bytes) = classification else { return nil }
                return bytes.count
            }.max() ?? 0
            precondition(
                actualMaxLength == maxTokenLength,
                "DiffVocab.maxTokenLength must equal the vocab's actual longest .bytes token "
                    + "(got \(actualMaxLength)) — this test's depth>K coverage assertion depends on it")

            return (out, eosId, [banned1, banned2])
        }
    }

    // MARK: - Greedy exact tokenizer over `DiffVocab` (longest-match-first, single-byte fallback
    // always available, so tokenization of ANY byte sequence the generator produces always
    // succeeds).

    private struct GreedyTokenizer {
        private let byFirstByte: [UInt8: [(bytes: [UInt8], id: Int)]]

        init(classifications: [TokenByteClassification]) {
            var grouped: [UInt8: [(bytes: [UInt8], id: Int)]] = [:]
            for (id, classification) in classifications.enumerated() {
                guard case .bytes(let bytes) = classification, let first = bytes.first else { continue }
                grouped[first, default: []].append((bytes, id))
            }
            for key in grouped.keys {
                grouped[key]!.sort { $0.bytes.count > $1.bytes.count }
            }
            self.byFirstByte = grouped
        }

        /// Tokenizes `bytes` into a sequence of vocab ids whose concatenated bytes exactly equal
        /// `bytes`. Always succeeds for `DiffVocab` (every byte value has a single-byte token).
        func tokenize(_ bytes: [UInt8]) -> [Int] {
            var ids: [Int] = []
            var index = 0
            while index < bytes.count {
                guard let candidates = byFirstByte[bytes[index]] else {
                    preconditionFailure("GreedyTokenizer: no token covers byte 0x\(String(bytes[index], radix: 16))")
                }
                guard
                    let match = candidates.first(where: { candidate in
                        let length = candidate.bytes.count
                        return index + length <= bytes.count
                            && Array(bytes[index..<(index + length)]) == candidate.bytes
                    })
                else {
                    preconditionFailure("GreedyTokenizer: failed to match at offset \(index)")
                }
                ids.append(match.id)
                index += match.bytes.count
            }
            return ids
        }
    }

    // MARK: - Deterministic valid-JSON-object prefix generator with depth-tracked checkpoints

    private struct Checkpoint {
        let prefix: [UInt8]
        let depth: Int
    }

    /// Generates ONE complete, valid top-level JSON object (random nesting up to `maxNesting`,
    /// mixing objects/arrays/strings-with-escapes/numbers/literals/whitespace runs), recording a
    /// `Checkpoint` at every point where `depth` (container nesting; matches
    /// `JSONObjectAutomaton`'s own "top-level object is depth 1" convention exactly, since `depth`
    /// is incremented/decremented in this function at the exact same byte boundaries the automaton
    /// pushes/pops its stack) is unambiguous — i.e. never mid-byte. The final checkpoint is always
    /// the completed document (`depth == 0`, automaton `isComplete`).
    private static func generateDocumentCheckpoints(seed: UInt64, maxNesting: Int) -> [Checkpoint] {
        var rng = DeterministicRNG(seed: seed)
        var out: [UInt8] = []
        var depth = 0
        var checkpoints: [Checkpoint] = []

        func record() { checkpoints.append(Checkpoint(prefix: out, depth: depth)) }
        func emitWhitespace() {
            let n = rng.nextInt(3)
            for _ in 0..<n { out.append([0x20, 0x09, 0x0A, 0x0D][rng.nextInt(4)]) }
            record()
        }
        func emitStringBytes() {
            out.append(0x22)
            let length = rng.nextInt(5)
            for _ in 0..<length {
                switch rng.nextInt(6) {
                case 0: out.append(UInt8(0x61 + rng.nextInt(26)))
                case 1: out.append(contentsOf: [0x5C, 0x6E])  // \n
                case 2: out.append(contentsOf: Array("\\u00e9".utf8))
                case 3: out.append(contentsOf: [0xC3, 0xA9])  // raw "é"
                case 4: out.append(contentsOf: [0xF0, 0x9F, 0x98, 0x80])  // raw "😀"
                default: out.append(0x20)
                }
            }
            out.append(0x22)
            record()
        }
        func emitNumber() {
            if rng.nextBool() { out.append(0x2D) }
            out.append(UInt8(0x30 + rng.nextInt(10)))
            if rng.nextInt(3) == 0 {
                out.append(0x2E)
                out.append(UInt8(0x30 + 1 + rng.nextInt(9)))
            }
            if rng.nextInt(3) == 0 {
                out.append(rng.nextBool() ? 0x65 : 0x45)
                if rng.nextBool() { out.append(rng.nextBool() ? 0x2B : 0x2D) }
                out.append(UInt8(0x30 + rng.nextInt(10)))
            }
            record()
        }
        func emitLiteral() {
            let words: [[UInt8]] = [Array("true".utf8), Array("false".utf8), Array("null".utf8)]
            out.append(contentsOf: words[rng.nextInt(3)])
            record()
        }
        // Forward-declared so emitValue/emitObjectBody/emitArrayBody can call each other.
        var emitObjectBody: () -> Void = {}
        var emitArrayBody: () -> Void = {}
        func emitValue() {
            emitWhitespace()
            let allowNest = depth < maxNesting
            let choice = allowNest ? rng.nextInt(5) : 2 + rng.nextInt(3)
            switch choice {
            case 0:
                out.append(0x7B)
                depth += 1
                record()
                emitObjectBody()
                out.append(0x7D)
                depth -= 1
                record()
            case 1:
                out.append(0x5B)
                depth += 1
                record()
                emitArrayBody()
                out.append(0x5D)
                depth -= 1
                record()
            case 2: emitStringBytes()
            case 3: emitNumber()
            default: emitLiteral()
            }
            emitWhitespace()
        }
        emitObjectBody = {
            emitWhitespace()
            if rng.nextBool() {
                let count = 1 + rng.nextInt(3)
                for i in 0..<count {
                    if i > 0 {
                        out.append(0x2C)
                        record()
                        emitWhitespace()
                    }
                    emitStringBytes()
                    emitWhitespace()
                    out.append(0x3A)
                    record()
                    emitWhitespace()
                    emitValue()
                }
            }
            emitWhitespace()
        }
        emitArrayBody = {
            emitWhitespace()
            if rng.nextBool() {
                let count = 1 + rng.nextInt(3)
                for i in 0..<count {
                    if i > 0 {
                        out.append(0x2C)
                        record()
                        emitWhitespace()
                    }
                    emitValue()
                }
            }
            emitWhitespace()
        }

        out.append(0x7B)
        depth += 1
        record()
        emitObjectBody()
        out.append(0x7D)
        depth -= 1
        record()  // completed-document checkpoint: depth == 0
        return checkpoints
    }

    /// Deterministically forces nesting straight down to `targetDepth` (mixing object and array
    /// containers — level 0 is always an object, since the top-level value must be one; every
    /// other level alternates/randomizes between object and array), then emits one leaf value and
    /// unwinds by closing every open container. Unlike `generateDocumentCheckpoints`, which decides
    /// whether to nest further with a per-level coin flip (so it overwhelmingly produces shallow
    /// documents — nesting to depth 5+ is geometrically unlikely, which is exactly why the original
    /// run of this test never reached `depth > DiffVocab.maxTokenLength`), this generator has no
    /// such escape hatch: it always reaches `targetDepth` and back, so every state along the way
    /// with `depth > DiffVocab.maxTokenLength` — where the mask cache key's `hasHiddenFrames`
    /// truncation is actually load-bearing — is guaranteed to be exercised, not merely possible.
    private static func generateDeepNestingCheckpoints(seed: UInt64, targetDepth: Int) -> [Checkpoint] {
        var rng = DeterministicRNG(seed: seed)
        var out: [UInt8] = []
        var depth = 0
        var checkpoints: [Checkpoint] = []
        var containerKinds: [UInt8] = []  // 0x7B (object) or 0x5B (array), one per open level

        func record() { checkpoints.append(Checkpoint(prefix: out, depth: depth)) }

        for level in 0..<targetDepth {
            let useObject = level == 0 ? true : rng.nextBool()
            if useObject {
                out.append(0x7B)
                depth += 1
                containerKinds.append(0x7B)
                record()
                out.append(contentsOf: Array(#""k""#.utf8))
                record()
                out.append(0x3A)
                record()
                if rng.nextBool() {
                    out.append(0x20)
                    record()
                }
            } else {
                out.append(0x5B)
                depth += 1
                containerKinds.append(0x5B)
                record()
            }
        }

        switch rng.nextInt(4) {
        case 0: out.append(contentsOf: Array(#""leaf""#.utf8))
        case 1: out.append(contentsOf: Array("123".utf8))
        case 2: out.append(contentsOf: Array("true".utf8))
        default: out.append(contentsOf: Array("null".utf8))
        }
        record()

        while let kind = containerKinds.popLast() {
            out.append(kind == 0x7B ? 0x7D : 0x5D)
            depth -= 1
            record()
        }
        return checkpoints
    }

    // MARK: - Reference (naive per-token) decision

    /// The textbook, per-token definition: `id` is allowed from `automaton`'s state iff EITHER it
    /// is EOS and the automaton is complete, OR its full byte sequence, fed one byte at a time
    /// through a COPY of `automaton`, never hits an invalid transition. Never touches the trie, the
    /// mask cache, or any truncated key — this is the ground truth the optimized path must match.
    private static func referenceAllowedIds(
        from automaton: JSONObjectAutomaton, classifications: [TokenByteClassification]
    ) -> Set<Int> {
        var allowed = Set<Int>()
        for (id, classification) in classifications.enumerated() {
            switch classification {
            case .eos:
                if automaton.isComplete { allowed.insert(id) }
            case .banned:
                continue
            case .bytes(let bytes):
                var probe = automaton
                var ok = true
                for byte in bytes {
                    guard probe.advance(byte: byte) else {
                        ok = false
                        break
                    }
                }
                if ok { allowed.insert(id) }
            }
        }
        return allowed
    }

    // MARK: - The differential test

    func testDifferentialAgreementAcrossManyStates() throws {
        let classifications = DiffVocab.classifications
        let tokenizer = GreedyTokenizer(classifications: classifications)
        let table = JSONObjectConstraintTable(classifications: classifications)

        // A mix of shallow (nesting capped small) and deep (up to ~20) documents: shallow ones
        // weight coverage toward ordinary depth<=K states and the completed-document state; deep
        // ones are what reaches depth>K, where the cache key's `hasHiddenFrames` bit is load-bearing.
        // `generateDocumentCheckpoints` nests probabilistically (a coin flip decides whether to go
        // deeper at each level), so on its own it essentially never reaches depth > K — see
        // `generateDeepNestingCheckpoints`'s doc comment. Raised from 30 to 40 documents.
        let documentSpecs: [(seed: UInt64, maxNesting: Int)] = (0..<40).map { i in
            let seed = 0x9E37_79B9_7F4A_7C15 &+ UInt64(i) &* 0xBF58_476D_1CE4_E5B9
            let maxNesting = i % 4 == 0 ? 2 + (i % 3) : 6 + (i % 15)  // occasional shallow doc, mostly deep
            return (seed, maxNesting)
        }

        // Deterministic deep-nesting documents that FORCE depth straight down to `targetDepth`
        // (no coin flip to bail out early), so `depth > DiffVocab.maxTokenLength` (the truncation
        // path) is actually and repeatedly exercised, not merely possible.
        let deepNestingSpecs: [(seed: UInt64, targetDepth: Int)] = (0..<10).map { i in
            let seed = 0xC2B2_AE3D_27D4_EB4F &+ UInt64(i) &* 0x1656_67B1_9E37_79F9
            let targetDepth = 6 + i * 2  // 6, 8, 10, ..., 24 — all well above maxTokenLength (4)
            return (seed, targetDepth)
        }

        var checkpointSets: [[Checkpoint]] = documentSpecs.map { spec in
            Self.generateDocumentCheckpoints(seed: spec.seed, maxNesting: spec.maxNesting)
        }
        checkpointSets.append(
            contentsOf: deepNestingSpecs.map { spec in
                Self.generateDeepNestingCheckpoints(seed: spec.seed, targetDepth: spec.targetDepth)
            })

        var comparedStates = 0
        var sawDepthAboveMaxTokenLength = false

        for checkpoints in checkpointSets {
            XCTAssertFalse(checkpoints.isEmpty)

            var constraint = JSONObjectTokenConstraint(table: table)
            var consumedLength = 0

            for checkpoint in checkpoints {
                // Advance the constraint by exactly the new bytes since the last checkpoint —
                // checkpoints within one document are strictly growing prefixes by construction.
                let deltaBytes = Array(checkpoint.prefix[consumedLength...])
                for id in tokenizer.tokenize(deltaBytes) {
                    try constraint.advance(token: id)
                }
                consumedLength = checkpoint.prefix.count

                // Independent reference automaton, replayed from scratch over the full prefix.
                var referenceAutomaton = JSONObjectAutomaton()
                for byte in checkpoint.prefix {
                    let ok = referenceAutomaton.advance(byte: byte)
                    XCTAssertTrue(ok, "generator produced a prefix the automaton itself rejects: \(checkpoint.prefix)")
                }

                let expected = Self.referenceAllowedIds(from: referenceAutomaton, classifications: classifications)
                let actual = Set(try constraint.allowedTokenIds())

                XCTAssertEqual(
                    actual, expected,
                    "mismatch at depth \(checkpoint.depth) for prefix \(String(decoding: checkpoint.prefix, as: UTF8.self)) "
                        + "— missing: \(expected.subtracting(actual)), extra: \(actual.subtracting(expected))")

                comparedStates += 1
                if checkpoint.depth > DiffVocab.maxTokenLength {
                    sawDepthAboveMaxTokenLength = true
                }
            }
        }

        XCTAssertGreaterThan(comparedStates, 500, "sanity: the generator must produce a meaningful number of states")
        XCTAssertTrue(
            sawDepthAboveMaxTokenLength,
            "no compared state reached depth > DiffVocab.maxTokenLength (\(DiffVocab.maxTokenLength)) — "
                + "the truncation path (hasHiddenFrames) would not actually be exercised by this run")
    }

    /// Narrow, fast companion focused specifically on EOS, banned ids, and multi-closer tokens at
    /// the completed-document state and at a deliberately shallow state, so a failure there is easy
    /// to isolate from the broad randomized sweep above.
    func testDifferentialAgreementAtCompletedDocumentAndShallowState() throws {
        let classifications = DiffVocab.classifications
        let tokenizer = GreedyTokenizer(classifications: classifications)
        let table = JSONObjectConstraintTable(classifications: classifications)

        for prefix in ["{}", #"{"a":[1,2]}"#, #"{"a":"café"}"#] {
            let bytes = Array(prefix.utf8)
            var constraint = JSONObjectTokenConstraint(table: table)
            for id in tokenizer.tokenize(bytes) {
                try constraint.advance(token: id)
            }
            var referenceAutomaton = JSONObjectAutomaton()
            for byte in bytes {
                XCTAssertTrue(referenceAutomaton.advance(byte: byte))
            }
            let expected = Self.referenceAllowedIds(from: referenceAutomaton, classifications: classifications)
            let actual = Set(try constraint.allowedTokenIds())
            XCTAssertEqual(actual, expected, "mismatch for completed/shallow prefix \(prefix)")
            XCTAssertTrue(expected.contains(DiffVocab.eosId) == referenceAutomaton.isComplete)
            for bannedId in DiffVocab.bannedIds {
                XCTAssertFalse(actual.contains(bannedId))
            }
        }
    }
}
