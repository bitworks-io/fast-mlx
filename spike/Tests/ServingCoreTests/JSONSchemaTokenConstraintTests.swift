import XCTest
import Foundation

@testable import ServingCore

/// Correctness tests for `JSONSchemaTokenConstraint`/`JSONSchemaConstraintTable` (response-format
/// slice 2c, stage 3a — wiring the shared trie/mask-cache machinery to the schema automaton).
///
/// Acceptance-criterion map (see the task's own lettered list):
/// - (a) `testBruteForceMaskDifferentialAcrossFiveSchemas`: cold-cache correctness against an
///   independent per-token reference, across 5 schemas, token-by-token.
/// - (a, warm half) `testWarmCacheAndCrossInstanceCacheSharing`: same walks replayed through a
///   SECOND, independently constructed `JSONSchemaTokenConstraint` over the SAME table, asserting
///   cache hits actually occur.
/// - (b) `testCrossSchemaIsolationOnAScaffoldPronetoCollisionWithoutTheFingerprint`.
/// - (c) folded into (a)'s walk (`XCTAssertFalse(expected.isEmpty, ...)` at every step) — a dead end
///   would ALSO show up as a differential mismatch, but this asserts the specific "never empty"
///   property by name.
/// - (d) `testGeneratedOutputValidatesAgainstAcceptsAndJSONSerialization`.
/// - (e) the full `ServingCoreTests` suite (this file included) — reported as run in the stage's
///   verification report, not a test function here.
/// - (f) `MUTATION CHECK` — performed as a one-time manual source-mutate/restore/shasum-verify
///   exercise (dropping the fingerprint from `JSONSchemaAutomaton.maskCacheKey(maxTokenBytes:)` and
///   confirming (b) — the test built specifically to be sensitive to that field — fails), reported
///   in the stage's verification report rather than checked into this file: the task's own framing
///   ("restore exactly, verify with shasum") describes a byte-for-byte source restore, not a
///   permanent runtime toggle.
/// - Timing: `testReleaseMicroBenchmarkMissLatency` (skipped by default, like
///   `JSONObjectTokenConstraintTests.testReleaseMicroBenchmark`) reports schema-state cache-MISS
///   median/p99 alongside the same measurement for json_object on a comparable synthetic vocab.
final class JSONSchemaTokenConstraintTests: XCTestCase {

    // MARK: - Deterministic PRNG (SplitMix64 — never SystemRandomNumberGenerator; matches
    // `JSONSchemaConstraintAutomatonTests`' own generator-side convention).

    private struct SplitMix64 {
        var state: UInt64
        init(seed: UInt64) { self.state = seed }
        mutating func next() -> UInt64 {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func nextInt(_ bound: Int) -> Int {
            precondition(bound > 0)
            return Int(next() % UInt64(bound))
        }
        mutating func nextBool() -> Bool { next() % 2 == 0 }
    }

    // MARK: - Synthetic vocab (mirrors `JSONObjectConstraintDifferentialTests.DiffVocab`'s shape:
    // every single byte 0...255 — guarantees ANY byte sequence tokenizes — plus curated multi-byte
    // tokens deliberately spanning structural boundaries, plus pseudo-random filler up to ~3,000
    // ids, plus EOS and two banned ids).

    private enum SchemaVocab {
        static let maxTokenLength = 10

        private static let built = build()
        static var classifications: [TokenByteClassification] { built.classifications }
        static var eosId: Int { built.eosId }
        static var bannedIds: [Int] { built.bannedIds }

        private static func build() -> (
            classifications: [TokenByteClassification], eosId: Int, bannedIds: [Int]
        ) {
            var out: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }

            // Curated multi-byte tokens spanning structural boundaries: key/colon, comma/brace
            // combinations, literal-plus-terminator, enum literals (whole and boundary-crossing),
            // property-name pieces for every fixture schema below, unicode content, and whitespace
            // runs. Every one <= maxTokenLength bytes.
            let curated: [String] = [
                "\":", "},{", "],[", "true,", "false,", "null,", "null}", "true}", "false}",
                "\"active\"", "\"inactive\"", "\"pending\"", "\"active\",", ":\"active\"",
                "\"id\"", "\"tags\"", "\"meta\"", "\"status\"", "\"nickname\"", "\"x\"", "\"y\"",
                "\"abc\"", "\"xyz\"", "\"ab", "\"xy", "  ", "\n  ", "[]", "{}",
            ]
            for text in curated {
                let bytes = Array(text.utf8)
                precondition(bytes.count <= maxTokenLength)
                out.append(.bytes(bytes))
            }
            out.append(.bytes([0xC3, 0xA9]))  // "é" as one raw 2-byte UTF-8 token
            out.append(.bytes([0xF0, 0x9F, 0x98, 0x80]))  // "😀" as one raw 4-byte UTF-8 token

            var rng = SplitMix64(seed: 0x5CDE_BA71_C0FF_EE42)
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
            return (out, eosId, [banned1, banned2])
        }
    }

    // MARK: - Greedy exact tokenizer (longest-match-first; single-byte fallback always available —
    // mirrors `JSONObjectConstraintDifferentialTests.GreedyTokenizer`).

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

    // MARK: - Schema fixtures (>= 5, per the task's own list: nested objects, enum, array of
    // objects, nullable anyOf, integer)

    private static func literal(_ jsonText: String) -> JSONSchemaLiteral {
        JSONSchemaLiteral(jsonText: Array(jsonText.utf8))
    }

    private struct Fixture {
        let name: String
        let node: JSONSchemaNode
        let fingerprint: String
    }

    private static let fixtures: [Fixture] = [
        Fixture(
            name: "nested",
            node: .object([
                JSONSchemaProperty(name: "id", required: true, value: .integer),
                JSONSchemaProperty(name: "tags", required: true, value: .array(items: .string)),
                JSONSchemaProperty(
                    name: "meta", required: false,
                    value: .object([JSONSchemaProperty(name: "active", required: true, value: .boolean)])),
            ]),
            fingerprint: "fp-nested-0001"),
        Fixture(
            name: "enum",
            node: .object([
                JSONSchemaProperty(
                    name: "status", required: true,
                    value: .enumeration([literal(#""active""#), literal(#""inactive""#), literal(#""pending""#)]))
            ]),
            fingerprint: "fp-enum-0002"),
        Fixture(
            name: "arrayOfObjects",
            node: .array(
                items: .object([
                    JSONSchemaProperty(name: "x", required: true, value: .integer),
                    JSONSchemaProperty(name: "y", required: true, value: .integer),
                ])),
            fingerprint: "fp-array-0003"),
        Fixture(
            name: "nullableAnyOf",
            node: .object([
                JSONSchemaProperty(name: "nickname", required: true, value: .anyOf([.string, .null]))
            ]),
            fingerprint: "fp-nullable-0004"),
        Fixture(name: "integer", node: .integer, fingerprint: "fp-integer-0005"),
    ]

    /// A single-required-property object per collision partner: SAME shape (one required top-level
    /// property, same value type), DIFFERENT property name — see
    /// `testCrossSchemaIsolationOnAScaffoldPronetoCollisionWithoutTheFingerprint`'s doc comment for
    /// why this shape is a genuine collision risk without the fingerprint.
    private static let collideSchemaA: JSONSchemaNode = .object([
        JSONSchemaProperty(name: "abc", required: true, value: .integer)
    ])
    private static let collideSchemaB: JSONSchemaNode = .object([
        JSONSchemaProperty(name: "xyz", required: true, value: .integer)
    ])
    private static let collideFingerprintA = "fp-collide-A"
    private static let collideFingerprintB = "fp-collide-B"

    // MARK: - Generic instance generator (byte-serialized directly; works for any `JSONSchemaNode`)

    private static func generateInstanceBytes(_ node: JSONSchemaNode, rng: inout SplitMix64) -> [UInt8] {
        switch node {
        case .string:
            return jsonStringBytes(randomAsciiString(rng: &rng))
        case .number:
            return Array(randomNumberText(rng: &rng, allowFraction: true).utf8)
        case .integer:
            return Array(randomNumberText(rng: &rng, allowFraction: false).utf8)
        case .boolean:
            return Array((rng.nextBool() ? "true" : "false").utf8)
        case .null:
            return Array("null".utf8)
        case .enumeration(let literals):
            return literals[rng.nextInt(literals.count)].jsonText
        case .object(let props):
            var out: [UInt8] = [0x7B]
            var first = true
            for prop in props {
                guard prop.required || rng.nextBool() else { continue }
                if !first { out.append(0x2C) }
                first = false
                out.append(contentsOf: jsonStringBytes(prop.name))
                out.append(0x3A)
                out.append(contentsOf: generateInstanceBytes(prop.value, rng: &rng))
            }
            out.append(0x7D)
            return out
        case .array(let items):
            var out: [UInt8] = [0x5B]
            let count = rng.nextInt(3)
            for index in 0..<count {
                if index > 0 { out.append(0x2C) }
                out.append(contentsOf: generateInstanceBytes(items, rng: &rng))
            }
            out.append(0x5D)
            return out
        case .anyOf(let branches):
            return generateInstanceBytes(branches[rng.nextInt(branches.count)], rng: &rng)
        }
    }

    private static func jsonStringBytes(_ string: String) -> [UInt8] {
        [0x22] + Array(string.utf8) + [0x22]
    }

    private static func randomAsciiString(rng: inout SplitMix64) -> String {
        let letters = Array("abcdefgh")
        let length = rng.nextInt(5)
        return String((0..<length).map { _ in letters[rng.nextInt(letters.count)] })
    }

    private static func randomNumberText(rng: inout SplitMix64, allowFraction: Bool) -> String {
        var text = ""
        if rng.nextBool() { text += "-" }
        text += String(1 + rng.nextInt(9))
        for _ in 0..<rng.nextInt(3) { text += String(rng.nextInt(10)) }
        if allowFraction, rng.nextBool() {
            text += "." + String(1 + rng.nextInt(9))
        }
        return text
    }

    // MARK: - Reference (naive per-token) decision — ground truth, mirrors
    // `JSONObjectConstraintDifferentialTests.referenceAllowedIds` exactly, over `JSONSchemaAutomaton`.

    private static func referenceAllowedIds(
        from automaton: JSONSchemaAutomaton, classifications: [TokenByteClassification]
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

    private static func idsSet(from bitset: [UInt64]) -> Set<Int> {
        var result = Set<Int>()
        for (wordIndex, word) in bitset.enumerated() where word != 0 {
            var bits = word
            while bits != 0 {
                let bit = bits.trailingZeroBitCount
                result.insert(wordIndex * 64 + bit)
                bits &= bits - 1
            }
        }
        return result
    }

    /// Never `XCTAssertEqual` on the raw sets directly (a mismatch at a state deep inside an
    /// unconstrained string can involve thousands of ids — see the memory note on never diffing
    /// large collections): compares scalar counts and reports only the FIRST divergent id.
    private func assertMaskMatches(
        _ actual: Set<Int>, _ expected: Set<Int>, _ context: @autoclosure () -> String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        guard actual != expected else { return }
        let missing = expected.subtracting(actual)
        let extra = actual.subtracting(expected)
        XCTFail(
            "\(context()) — missingCount=\(missing.count) extraCount=\(extra.count) "
                + "firstMissing=\(missing.min().map(String.init) ?? "none") "
                + "firstExtra=\(extra.min().map(String.init) ?? "none")",
            file: file, line: line)
    }

    // MARK: - (a) Brute-force mask differential, cold cache, across >= 5 schemas; also (c) no dead end

    func testBruteForceMaskDifferentialAcrossFiveSchemas() throws {
        let classifications = SchemaVocab.classifications
        let tokenizer = GreedyTokenizer(classifications: classifications)
        var totalSteps = 0
        var rng = SplitMix64(seed: 0x9E37_79B9_1234_5678)

        for fixture in Self.fixtures {
            let table = JSONSchemaConstraintTable(classifications: classifications)
            for _ in 0..<8 {
                let instanceBytes = Self.generateInstanceBytes(fixture.node, rng: &rng)
                let tokenIds = tokenizer.tokenize(instanceBytes)
                var automaton = JSONSchemaAutomaton(root: fixture.node, fingerprint: fixture.fingerprint)

                for id in tokenIds {
                    let expected = Self.referenceAllowedIds(from: automaton, classifications: classifications)
                    XCTAssertFalse(expected.isEmpty, "(c) dead end for schema \(fixture.name) before token \(id)")
                    let actual = Self.idsSet(from: try table.allowedTokenBitset(for: automaton))
                    assertMaskMatches(
                        actual, expected, "schema \(fixture.name) step \(totalSteps) before token \(id)")
                    totalSteps += 1

                    guard case .bytes(let tokenBytes) = classifications[id] else {
                        return XCTFail("generated token \(id) is not a .bytes token")
                    }
                    for byte in tokenBytes {
                        XCTAssertTrue(automaton.advance(byte: byte), "generator produced a byte the automaton rejects")
                    }
                }

                let finalExpected = Self.referenceAllowedIds(from: automaton, classifications: classifications)
                let finalActual = Self.idsSet(from: try table.allowedTokenBitset(for: automaton))
                assertMaskMatches(finalActual, finalExpected, "schema \(fixture.name) final state")
                XCTAssertTrue(automaton.isComplete, "schema \(fixture.name): generated instance did not complete")
            }
        }

        XCTAssertGreaterThan(totalSteps, 100, "sanity: the generator must produce a meaningful number of states")
    }

    // MARK: - (a, warm half) Warm cache + cache sharing across independently constructed constraints

    func testWarmCacheAndCrossInstanceCacheSharing() throws {
        let classifications = SchemaVocab.classifications
        let tokenizer = GreedyTokenizer(classifications: classifications)
        var rng = SplitMix64(seed: 0xABCD_1234_5678_9999)

        for fixture in Self.fixtures {
            let table = JSONSchemaConstraintTable(classifications: classifications)
            let instanceBytes = Self.generateInstanceBytes(fixture.node, rng: &rng)
            let tokenIds = tokenizer.tokenize(instanceBytes)

            // COLD pass: constraint #1, a fresh table (every state must miss at least once).
            var constraint1 = JSONSchemaTokenConstraint(
                table: table, format: JSONSchemaResponseFormat(name: "f", strict: false, root: fixture.node, fingerprint: fixture.fingerprint))
            for id in tokenIds {
                _ = try constraint1.allowedTokenIds()
                try constraint1.advance(token: id)
            }
            let missesAfterCold = table.cacheMissCount
            XCTAssertGreaterThan(missesAfterCold, 0, "schema \(fixture.name): cold pass must miss at least once")

            // WARM pass: constraint #2, a SEPARATE `JSONSchemaTokenConstraint`/`JSONSchemaAutomaton`
            // instance (different `init` calls — different `SchemaTable` object identity), replaying
            // the IDENTICAL token sequence, so every visited state is logically identical to the cold
            // pass. Cache reuse is the specific thing under test — see `MaskCacheKey`'s doc comment
            // for why this must be a HIT despite the different underlying automaton instance.
            let hitsBeforeWarm = table.cacheHitCount
            let missesBeforeWarm = table.cacheMissCount
            var constraint2 = JSONSchemaTokenConstraint(
                table: table, format: JSONSchemaResponseFormat(name: "f", strict: false, root: fixture.node, fingerprint: fixture.fingerprint))
            for id in tokenIds {
                // Per-step correctness is already covered by the brute-force differential test above;
                // this loop's own job is cache-hit accounting, asserted below. (Deliberately does NOT
                // call back into `constraint1` here: it has already been advanced past every token by
                // the cold pass above, to the completed-document state — a state the cold pass never
                // actually QUERIED `allowedTokenIds()` at, since that loop only queries BEFORE each
                // advance. Calling it here would therefore itself be a fresh miss on constraint1's
                // final state, silently inflating `table.cacheMissCount` by one and breaking the
                // "warm pass introduces zero new misses" assertion below for a reason unrelated to
                // what this test is supposed to prove.)
                _ = try constraint2.allowedTokenIds()
                try constraint2.advance(token: id)
            }
            XCTAssertGreaterThan(
                table.cacheHitCount, hitsBeforeWarm,
                "schema \(fixture.name): replaying the identical schema/state sequence through a second, "
                    + "independently constructed constraint over the SAME table must produce cache hits")
            XCTAssertEqual(
                table.cacheMissCount, missesBeforeWarm,
                "schema \(fixture.name): the warm pass must not introduce any NEW misses "
                    + "(every state it visits was already cached by the cold pass)")
        }
    }

    // MARK: - (b) Cross-schema isolation

    /// Two schemas with the SAME shape (one required top-level property, same value type) but
    /// DIFFERENT property names. Without the fingerprint, `JSONSchemaAutomaton.MaskCacheKey` reduces
    /// to `Frame`/`Lexeme` payloads carrying only INTEGER indices into each schema's own,
    /// independently-numbered `SchemaTable.properties` — and since `collideSchemaA`/`collideSchemaB`
    /// are structurally identical single-property objects, each compiles to property index 0 in its
    /// OWN table. At the state right after `{"` (0 key bytes matched), schema A's key candidates are
    /// `[0]` (matching `"abc"`) and schema B's are ALSO `[0]` (matching `"xyz"`) — an
    /// `.objectKey(matchedBytes: 0, candidates: [0])` lexeme in BOTH cases, with an empty `stack`
    /// suffix in both (just after `{`) — so EVERY field the key excludes fingerprint from would be
    /// pairwise equal between the two schemas at this position. Only `fingerprint` breaks the tie;
    /// this test proves it actually does, by running both schemas INTERLEAVED on one shared table and
    /// checking their masks (and eventual generated bytes) diverge as expected instead of one
    /// serving the other's cached entry.
    func testCrossSchemaIsolationOnAScaffoldPronetoCollisionWithoutTheFingerprint() throws {
        let classifications = SchemaVocab.classifications
        let table = JSONSchemaConstraintTable(classifications: classifications)

        var automatonA = JSONSchemaAutomaton(root: Self.collideSchemaA, fingerprint: Self.collideFingerprintA)
        var automatonB = JSONSchemaAutomaton(root: Self.collideSchemaB, fingerprint: Self.collideFingerprintB)

        // Drive both to "{\"" — right after the opening brace and key-opening quote — INTERLEAVED
        // (A's lookup first primes the cache; B's lookup immediately after is what a missing
        // fingerprint would incorrectly turn into a hit against A's entry).
        for byte in Array(#"{""#.utf8) {
            XCTAssertTrue(automatonA.advance(byte: byte))
            XCTAssertTrue(automatonB.advance(byte: byte))
        }

        let maskA = Self.idsSet(from: try table.allowedTokenBitset(for: automatonA))
        let maskB = Self.idsSet(from: try table.allowedTokenBitset(for: automatonB))

        // Schema A's key is "abc" (next legal byte 'a' = 0x61); schema B's is "xyz" ('x' = 0x78).
        // These masks MUST differ: the single-byte 'a' token is allowed for A, not for B, and
        // vice versa for 'x'.
        let byteAId = classifications.firstIndex { if case .bytes([0x61]) = $0 { return true }; return false }!
        let byteXId = classifications.firstIndex { if case .bytes([0x78]) = $0 { return true }; return false }!
        XCTAssertTrue(maskA.contains(byteAId), "schema A must allow 'a' right after its opening key quote")
        XCTAssertFalse(maskA.contains(byteXId), "schema A must NOT allow 'x' (that is schema B's key)")
        XCTAssertTrue(maskB.contains(byteXId), "schema B must allow 'x' right after its opening key quote")
        XCTAssertFalse(maskB.contains(byteAId), "schema B must NOT allow 'a' (that is schema A's key)")
        assertMaskMatches(maskA, maskA, "sanity: A vs itself must match")  // no-op, keeps helper exercised
        XCTAssertNotEqual(maskA, maskB, "cross-schema isolation: masks at a colliding-shape position must differ")

        // Independent reference confirms this is genuine schema semantics, not a fluke of caching
        // order: recompute both from scratch automatons via the brute-force reference.
        let referenceA = Self.referenceAllowedIds(from: automatonA, classifications: classifications)
        let referenceB = Self.referenceAllowedIds(from: automatonB, classifications: classifications)
        assertMaskMatches(maskA, referenceA, "schema A mask vs reference")
        assertMaskMatches(maskB, referenceB, "schema B mask vs reference")
    }

    // MARK: - Table-level regression: compiler-produced fingerprints for property-order-only variants

    /// Response-format bug regression (`JSONSchemaFingerprint.swift`): `{a:int,b:int}` vs.
    /// `{b:int,a:int}` (both required, `additionalProperties:false`) compiled through the REAL
    /// `JSONSchemaSubsetCompiler` (so this uses actual production fingerprints, not the synthetic
    /// per-fixture ones the rest of this file uses) must get DIFFERENT fingerprints, and therefore
    /// never share `JSONSchemaConstraintTable` mask-cache entries: interleaved on ONE shared table,
    /// right after `{"` schema A's mask allows only `a…` and schema B's only `b…`. This is the exact
    /// scenario the old sorted-keys-of-raw-JSON fingerprint got wrong: sorted keys are blind to
    /// `properties` declared order, but the automaton enforces it.
    func testTableLevelRegressionForPropertyOrderOnlyFingerprintCollision() throws {
        let schemaAJSON =
            #"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"additionalProperties":false}"#
        let schemaBJSON =
            #"{"type":"object","properties":{"b":{"type":"integer"},"a":{"type":"integer"}},"required":["a","b"],"additionalProperties":false}"#

        let rawA = try JSONSchemaSubsetCompiler.parseOrdered(Data(schemaAJSON.utf8))
        let rawB = try JSONSchemaSubsetCompiler.parseOrdered(Data(schemaBJSON.utf8))
        let compiledA = try JSONSchemaSubsetCompiler.compile(name: "a", schema: rawA, strict: true)
        let compiledB = try JSONSchemaSubsetCompiler.compile(name: "b", schema: rawB, strict: true)
        XCTAssertNotEqual(
            compiledA.fingerprint, compiledB.fingerprint,
            "property-order-only schemas must get different fingerprints (the bug this stage fixes)")

        let classifications = SchemaVocab.classifications
        let table = JSONSchemaConstraintTable(classifications: classifications)

        var automatonA = JSONSchemaAutomaton(root: compiledA.root, fingerprint: compiledA.fingerprint)
        var automatonB = JSONSchemaAutomaton(root: compiledB.root, fingerprint: compiledB.fingerprint)

        // Interleaved to `{"` — A primes the cache; B's lookup right after would incorrectly reuse
        // A's entry if the two schemas' fingerprints (wrongly) collided.
        for byte in Array(#"{""#.utf8) {
            XCTAssertTrue(automatonA.advance(byte: byte))
            XCTAssertTrue(automatonB.advance(byte: byte))
        }

        let maskA = Self.idsSet(from: try table.allowedTokenBitset(for: automatonA))
        let maskB = Self.idsSet(from: try table.allowedTokenBitset(for: automatonB))

        let byteAId = classifications.firstIndex { if case .bytes([0x61]) = $0 { return true }; return false }!
        let byteBId = classifications.firstIndex { if case .bytes([0x62]) = $0 { return true }; return false }!
        XCTAssertTrue(maskA.contains(byteAId), "schema A ({a,b} declared) must allow 'a' right after '{\"'")
        XCTAssertFalse(maskA.contains(byteBId), "schema A must NOT allow 'b' first — a is declared first and required")
        XCTAssertTrue(maskB.contains(byteBId), "schema B ({b,a} declared) must allow 'b' right after '{\"'")
        XCTAssertFalse(maskB.contains(byteAId), "schema B must NOT allow 'a' first — b is declared first and required")

        // Walk each schema's OWN valid instance token-by-token through a constraint built from its
        // real compiled format (name/strict/root/fingerprint) — must never throw.
        let tokenizer = GreedyTokenizer(classifications: classifications)
        let cases: [(label: String, format: JSONSchemaResponseFormat, instanceJSON: String)] = [
            ("A", compiledA, #"{"a":1,"b":2}"#),
            ("B", compiledB, #"{"b":2,"a":1}"#),
        ]
        for testCase in cases {
            var constraint = JSONSchemaTokenConstraint(table: table, format: testCase.format)
            let ids = tokenizer.tokenize(Array(testCase.instanceJSON.utf8))
            for id in ids {
                let allowed = Set(try constraint.allowedTokenIds())
                XCTAssertTrue(
                    allowed.contains(id),
                    "schema \(testCase.label): token \(id) for \(testCase.instanceJSON) must be allowed")
                try constraint.advance(token: id)
            }
            XCTAssertTrue(
                constraint.isComplete, "schema \(testCase.label): \(testCase.instanceJSON) should complete")
        }
    }

    // MARK: - Truncation soundness: `maskCacheKey(maxTokenBytes:)` beyond the vocab's max token length
    //
    // `JSONSchemaAutomaton.maskCacheKey(maxTokenBytes:)` truncates `stack` to its top `maxTokenBytes`
    // frames plus a `hasHiddenFrames` flag (see that function's doc comment, which mirrors
    // `JSONObjectAutomaton.maskCacheKey(maxTokenBytes:)`'s proof exactly). Every fixture/vocab above
    // nests shallower than `SchemaVocab.maxTokenLength` (10), so `hasHiddenFrames` is never `true`
    // there and this truncation path goes completely untested. `DeepVocab` below deliberately caps
    // the vocab's longest token at 4 bytes while `deepFixtures` nest 5 frames deep (an object frame
    // plus a 4-level array chain), forcing `stack.count > maxTokenBytes` well before a walk completes.

    private enum DeepVocab {
        static let maxTokenLength = 4

        private static let built = build()
        static var classifications: [TokenByteClassification] { built.classifications }
        static var eosId: Int { built.eosId }
        static var bannedIds: [Int] { built.bannedIds }

        private static func build() -> (
            classifications: [TokenByteClassification], eosId: Int, bannedIds: [Int]
        ) {
            var out: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }

            // Curated closer-run and content tokens spanning the deep-array structural boundaries —
            // per the task's own list, plus a 4-byte run matching this file's exact nesting depth.
            let curated: [String] = [
                "]]", "]]]", "]]]]", "}]", "}}", "1]]", "\"]}", "],", "]}", "},", ",\"",
                "\"p\"", "\"q\"", "\"r\"", "\":", "null", "true",
            ]
            for text in curated {
                let bytes = Array(text.utf8)
                precondition(bytes.count <= maxTokenLength)
                out.append(.bytes(bytes))
            }

            var rng = SplitMix64(seed: 0xDEAD_BEEF_C0FF_EE01)
            let printable: [UInt8] = Array(0x20...0x7E)
            let targetVocabSize = 1_200
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
            return (out, eosId, [banned1, banned2])
        }
    }

    private static func deepArray(depth: Int, leaf: JSONSchemaNode) -> JSONSchemaNode {
        var node = leaf
        for _ in 0..<depth { node = .array(items: node) }
        return node
    }

    private struct DeepFixture {
        let name: String
        let node: JSONSchemaNode
        let fingerprint: String
        /// A deterministic instance that reaches the FULL 4-level array depth — relying only on the
        /// random generator's `rng.nextInt(3)` element counts would make "did we ever exceed the
        /// vocab's max token length" a probabilistic sanity check instead of a guaranteed one.
        let maximalInstanceJSON: String
    }

    private static let deepFixtures: [DeepFixture] = [
        DeepFixture(
            name: "deepArraysPQR",
            // `p`/`q` deliberately have DIFFERENT leaf types (integer vs. string): a mask-cache key
            // that dropped/under-truncated frame identity would let a collision between "inside p's
            // innermost array" and "inside q's innermost array" go UNDETECTED if both leaves were the
            // same shape (same accepted bytes either way) — see the mutation-check note on
            // `JSONSchemaConstraintAutomaton.maskCacheKey(maxTokenBytes:)`.
            node: .object([
                JSONSchemaProperty(name: "p", required: true, value: deepArray(depth: 4, leaf: .integer)),
                JSONSchemaProperty(name: "q", required: false, value: deepArray(depth: 4, leaf: .string)),
                JSONSchemaProperty(name: "r", required: true, value: .integer),
            ]),
            fingerprint: "fp-deep-pqr-0001",
            maximalInstanceJSON: #"{"p":[[[[1]]]],"q":[[[["a"]]]],"r":3}"#),
        DeepFixture(
            name: "deepArraysAnyOfNullable",
            node: .object([
                JSONSchemaProperty(
                    name: "p", required: true, value: .anyOf([deepArray(depth: 4, leaf: .integer), .null]))
            ]),
            fingerprint: "fp-deep-anyof-0002",
            maximalInstanceJSON: #"{"p":[[[[1]]]]}"#),
    ]

    /// Drives each `deepFixtures` schema through MANY seeded valid instances — one deterministic
    /// maximal-depth instance plus 20 random ones (0-2 elements per array level, `q` sometimes
    /// skipped) — all sharing ONE warm `JSONSchemaConstraintTable`, so a mask cached from one
    /// instance's HIDDEN context can be looked up again from a DIFFERENT instance's state. If
    /// `maskCacheKey`'s truncation ever collided two states that a reachable token could tell apart,
    /// this would show up as a mismatch against the independent, untruncated `referenceAllowedIds`
    /// at the exact step it happens — see `assertMaskMatches`.
    func testMaskCacheTruncationSoundnessBeyondVocabMaxTokenLength() throws {
        let classifications = DeepVocab.classifications
        let actualMaxTokenBytes = JSONObjectConstraintTrie(classifications: classifications).maxTokenByteLength
        XCTAssertEqual(
            actualMaxTokenBytes, DeepVocab.maxTokenLength,
            "sanity: vocab's longest token must match the constant this test reasons about")

        let tokenizer = GreedyTokenizer(classifications: classifications)
        var rng = SplitMix64(seed: 0x2468_ACE0_1357_9BDF)
        var totalSteps = 0
        var sawHiddenFrames = false

        for fixture in Self.deepFixtures {
            let table = JSONSchemaConstraintTable(classifications: classifications)

            var instances: [[UInt8]] = [Array(fixture.maximalInstanceJSON.utf8)]
            for _ in 0..<20 {
                instances.append(Self.generateInstanceBytes(fixture.node, rng: &rng))
            }

            for instanceBytes in instances {
                let tokenIds = tokenizer.tokenize(instanceBytes)
                var automaton = JSONSchemaAutomaton(root: fixture.node, fingerprint: fixture.fingerprint)
                var depth = 0

                for id in tokenIds {
                    if depth > actualMaxTokenBytes { sawHiddenFrames = true }

                    let expected = Self.referenceAllowedIds(from: automaton, classifications: classifications)
                    XCTAssertFalse(expected.isEmpty, "deep schema \(fixture.name): dead end before token \(id)")
                    let actual = Self.idsSet(from: try table.allowedTokenBitset(for: automaton))
                    assertMaskMatches(
                        actual, expected, "deep schema \(fixture.name) step \(totalSteps) before token \(id)")
                    totalSteps += 1

                    guard case .bytes(let tokenBytes) = classifications[id] else {
                        return XCTFail("generated token \(id) is not a .bytes token")
                    }
                    for byte in tokenBytes {
                        XCTAssertTrue(automaton.advance(byte: byte), "generator produced a byte the automaton rejects")
                        switch byte {
                        case 0x7B, 0x5B: depth += 1  // '{' '['
                        case 0x7D, 0x5D: depth -= 1  // '}' ']'
                        default: break
                        }
                    }
                }

                let finalExpected = Self.referenceAllowedIds(from: automaton, classifications: classifications)
                let finalActual = Self.idsSet(from: try table.allowedTokenBitset(for: automaton))
                assertMaskMatches(finalActual, finalExpected, "deep schema \(fixture.name) final state")
                XCTAssertTrue(automaton.isComplete, "deep schema \(fixture.name): generated instance did not complete")
            }

            XCTAssertGreaterThan(
                table.cacheHitCount, 0,
                "deep schema \(fixture.name): 21 instances sharing one warm table must produce cache hits "
                    + "(a mask cached from one hidden context reused from another)")
        }

        XCTAssertTrue(
            sawHiddenFrames,
            "sanity: some walked state must exceed the vocab's max token length, or this test never "
                + "exercises `hasHiddenFrames` truncation at all")
        XCTAssertGreaterThan(totalSteps, 100, "sanity: the generator must produce a meaningful number of states")
    }

    /// Deterministically drives to FIVE stack frames deep (one object frame plus four array frames)
    /// — beyond `DeepVocab`'s 4-byte max token length, so the object frame starts HIDDEN — then
    /// closes the four array levels one byte at a time. The final closing byte pops through the
    /// entire visible window in one token-sized budget, revealing the previously-hidden object
    /// frame; the object frame's OWN `,`-vs-`}` decision (governed by whether a required property
    /// still remains) must match the untruncated reference at every one of those four steps, not
    /// just at the end — and must correctly differ between the two schemas below (`,` mandatory
    /// while `r` is still pending vs. `}` legal once `p` was the last required property).
    func testHiddenFrameGovernsCommaVsBraceDecision() throws {
        let classifications = DeepVocab.classifications
        let actualMaxTokenBytes = JSONObjectConstraintTrie(classifications: classifications).maxTokenByteLength

        struct Case {
            let label: String
            let node: JSONSchemaNode
            let fingerprint: String
            let prefix: String
            let commaRequired: Bool
        }
        let cases: [Case] = [
            Case(
                label: "commaRequired (r still pending after p)",
                node: .object([
                    JSONSchemaProperty(name: "p", required: true, value: Self.deepArray(depth: 4, leaf: .integer)),
                    JSONSchemaProperty(name: "r", required: true, value: .integer),
                ]),
                fingerprint: "fp-hf-comma-0001", prefix: #"{"p":[[[[1"#, commaRequired: true),
            Case(
                label: "braceAllowed (p is the last required property)",
                node: .object([
                    JSONSchemaProperty(name: "r", required: true, value: .integer),
                    JSONSchemaProperty(name: "p", required: true, value: Self.deepArray(depth: 4, leaf: .integer)),
                ]),
                fingerprint: "fp-hf-brace-0002", prefix: #"{"r":1,"p":[[[[1"#, commaRequired: false),
        ]

        // ONE shared warm table across both cases — mirrors the differential test's warm-cache setup.
        let table = JSONSchemaConstraintTable(classifications: classifications)

        for testCase in cases {
            var automaton = JSONSchemaAutomaton(root: testCase.node, fingerprint: testCase.fingerprint)
            var depth = 0
            for byte in Array(testCase.prefix.utf8) {
                XCTAssertTrue(
                    automaton.advance(byte: byte),
                    "\(testCase.label): prefix byte 0x\(String(byte, radix: 16)) rejected")
                switch byte {
                case 0x7B, 0x5B: depth += 1
                case 0x7D, 0x5D: depth -= 1
                default: break
                }
            }
            XCTAssertEqual(depth, 5, "\(testCase.label): sanity — object frame + 4 array frames")
            XCTAssertGreaterThan(
                depth, actualMaxTokenBytes, "\(testCase.label): sanity — must exceed the visible window")

            for closeIndex in 0..<4 {
                let expected = Self.referenceAllowedIds(from: automaton, classifications: classifications)
                XCTAssertFalse(expected.isEmpty, "\(testCase.label): dead end before close byte \(closeIndex)")
                let actual = Self.idsSet(from: try table.allowedTokenBitset(for: automaton))
                assertMaskMatches(actual, expected, "\(testCase.label) close step \(closeIndex)")
                XCTAssertTrue(
                    automaton.advance(byte: 0x5D), "\(testCase.label): ']' rejected at close step \(closeIndex)")
                depth -= 1
            }
            XCTAssertEqual(depth, 1, "\(testCase.label): sanity — only the object frame should remain")

            let finalExpected = Self.referenceAllowedIds(from: automaton, classifications: classifications)
            let finalActual = Self.idsSet(from: try table.allowedTokenBitset(for: automaton))
            assertMaskMatches(finalActual, finalExpected, "\(testCase.label) post-close mask")

            let commaId = 0x2C
            let braceId = 0x7D
            if testCase.commaRequired {
                XCTAssertTrue(
                    finalActual.contains(commaId), "\(testCase.label): ',' must be legal — a required property remains")
                XCTAssertFalse(
                    finalActual.contains(braceId), "\(testCase.label): '}' must be illegal — a required property remains")
            } else {
                XCTAssertTrue(finalActual.contains(braceId), "\(testCase.label): '}' must be legal — no properties remain")
                XCTAssertFalse(finalActual.contains(commaId), "\(testCase.label): ',' must be illegal — no properties remain")
            }
        }
    }

    // MARK: - (d) Generated output validates against `accepts` AND `JSONSerialization`

    /// Structural CLOSING single bytes — quote, comma, `}`, `]` — as token ids. Safe to hard-code as
    /// the raw byte VALUE: `SchemaVocab.classifications` begins with `(0...255).map { .bytes([byte]) }`,
    /// so byte `b`'s single-byte token id is always exactly `b`.
    private static let closerIds: Set<Int> = [0x22, 0x2C, 0x7D, 0x5D]
    /// Short-content single bytes (lowercase letters + digits) — covers every fixture's property
    /// names/enum literals/number digits without ever needing to enumerate the full vocab.
    private static let contentIds: Set<Int> = Set(0x61...0x7A).union(0x30...0x39)

    /// Picks ONE id from `allowed` (already EOS-filtered), biased toward quickly reaching a valid,
    /// SHORT instance rather than a uniform pick over `allowed` — which, at a state deep inside an
    /// open string, can hold thousands of ids (nearly the whole vocab) versus exactly one legal
    /// closing quote, making a uniform pick's expected steps-to-close scale with vocab SIZE, not with
    /// the instance's own grammar-shaped length. Mirrors the same "adversarial/preferring scorer"
    /// idiom `JSONObjectTokenConstraintTests` already uses (e.g.
    /// `testWhitespacePreferringScorerTerminatesWithCompleteObjectWithinBoundedSteps`), just with a
    /// probabilistic (not fully deterministic) preference so five samples per schema still vary.
    private static func biasedPick(from allowed: [Int], rng: inout SplitMix64) -> Int {
        let allowedSet = Set(allowed)
        let closers = allowedSet.intersection(closerIds).sorted()
        let contents = allowedSet.intersection(contentIds).sorted()
        if !closers.isEmpty, rng.nextInt(5) < 3 {
            return closers[rng.nextInt(closers.count)]
        }
        if !contents.isEmpty {
            return contents[rng.nextInt(contents.count)]
        }
        if !closers.isEmpty {
            return closers[rng.nextInt(closers.count)]
        }
        return allowed[rng.nextInt(allowed.count)]
    }

    func testGeneratedOutputValidatesAgainstAcceptsAndJSONSerialization() throws {
        let classifications = SchemaVocab.classifications
        var rng = SplitMix64(seed: 0x1357_9BDF_2468_ACE0)

        for fixture in Self.fixtures {
            let table = JSONSchemaConstraintTable(classifications: classifications)
            for _ in 0..<5 {
                var automaton = JSONSchemaAutomaton(root: fixture.node, fingerprint: fixture.fingerprint)
                var bytes: [UInt8] = []
                var steps = 0
                while steps < 400 {
                    steps += 1
                    if automaton.isComplete, rng.nextInt(4) == 0 { break }
                    let allowed = try table.allowedTokenIds(for: automaton).filter { $0 != SchemaVocab.eosId }
                    guard !allowed.isEmpty else {
                        XCTAssertTrue(automaton.isComplete, "schema \(fixture.name): stuck with no non-EOS token and incomplete")
                        break
                    }
                    let pick = Self.biasedPick(from: allowed, rng: &rng)
                    guard case .bytes(let tokenBytes) = classifications[pick] else {
                        return XCTFail("picked token \(pick) is not a .bytes token")
                    }
                    for byte in tokenBytes {
                        XCTAssertTrue(automaton.advance(byte: byte))
                    }
                    bytes.append(contentsOf: tokenBytes)
                }
                XCTAssertTrue(automaton.isComplete, "schema \(fixture.name): generation loop exhausted without completing")
                XCTAssertTrue(
                    JSONSchemaAutomaton.accepts(bytes: bytes, root: fixture.node),
                    "schema \(fixture.name): generated bytes must themselves round-trip through `accepts`: "
                        + String(decoding: bytes, as: UTF8.self))
                // `.fragmentsAllowed` so a bare top-level scalar (the `.integer` fixture) is accepted
                // by `JSONSerialization` too — its default `jsonObject(with:)` requires a top-level
                // array/object and would otherwise reject a legitimately valid bare-scalar instance.
                XCTAssertNoThrow(
                    try JSONSerialization.jsonObject(with: Data(bytes), options: [.fragmentsAllowed]),
                    "schema \(fixture.name): generated bytes must parse as JSON: " + String(decoding: bytes, as: UTF8.self))
            }
        }
    }

    // MARK: - json_object regression seam: proves this file's `@testable import` and vocab fixtures
    // don't interfere with the (separately, fully) exercised `JSONObjectConstraintTable` — the actual
    // regression evidence is "run the whole ServingCoreTests target", reported alongside this file's
    // own results, not re-asserted narrowly here.
    func testJSONObjectConstraintStillWorksAlongsideThisFile() throws {
        let table = JSONObjectConstraintTable(classifications: SchemaVocab.classifications)
        var constraint = JSONObjectTokenConstraint(table: table)
        for byte in Array(#"{"a":1}"#.utf8) {
            let allowed = Set(try constraint.allowedTokenIds())
            guard let id = SchemaVocab.classifications.firstIndex(where: {
                if case .bytes([byte]) = $0 { return true }
                return false
            }), allowed.contains(id) else {
                return XCTFail("expected a single-byte token for 0x\(String(byte, radix: 16)) to be allowed")
            }
            try constraint.advance(token: id)
        }
        XCTAssertTrue(constraint.isComplete)
    }

    // MARK: - Timing (skipped by default): schema-state cache-MISS latency vs json_object, same-scale vocab

    /// Informational-only: median/p99 latency of a schema-state cache MISS (`allowedTokenBitset` on
    /// a fresh, never-before-seen automaton state) on a ~150k-token synthetic vocab, alongside the
    /// SAME measurement for `JSONObjectConstraintTable` on an equal-size vocab (`warmCache: false`,
    /// so every visited json_object state also genuinely misses — an apples-to-apples first-miss
    /// comparison). Skipped unless `RUN_JSON_SCHEMA_CONSTRAINT_BENCH=1` — mirrors
    /// `JSONObjectTokenConstraintTests.testReleaseMicroBenchmark`'s own gating convention exactly
    /// (multi-second trie-build work that should not slow down the default `swift test` loop). Run
    /// explicitly with:
    ///   RUN_JSON_SCHEMA_CONSTRAINT_BENCH=1 swift test -c release --package-path spike \
    ///     --filter ServingCoreTests.JSONSchemaTokenConstraintTests/testReleaseMicroBenchmarkMissLatency
    func testReleaseMicroBenchmarkMissLatency() throws {
        guard ProcessInfo.processInfo.environment["RUN_JSON_SCHEMA_CONSTRAINT_BENCH"] == "1" else {
            throw XCTSkip("set RUN_JSON_SCHEMA_CONSTRAINT_BENCH=1 to run the perf micro-benchmark")
        }

        let vocabSize = 150_000
        func buildLargeVocab() -> [TokenByteClassification] {
            var classifications: [TokenByteClassification] = []
            classifications.reserveCapacity(vocabSize)
            for byte in 0...255 {
                classifications.append(.bytes([UInt8(byte)]))
            }
            var state: UInt64 = 0xC0FF_EE00_1234_5678
            func nextRandom() -> UInt64 {
                state ^= state << 13
                state ^= state >> 7
                state ^= state << 17
                return state
            }
            let printable: [UInt8] = Array(UInt8(0x20)...UInt8(0x7E))
            while classifications.count < vocabSize {
                let length = 1 + Int(nextRandom() % 12)
                var bytes: [UInt8] = []
                bytes.reserveCapacity(length)
                for _ in 0..<length {
                    bytes.append(printable[Int(nextRandom() % UInt64(printable.count))])
                }
                classifications.append(.bytes(bytes))
            }
            classifications.append(.eos)
            return classifications
        }

        func percentile(_ values: [Double], _ p: Double) -> Double {
            guard !values.isEmpty else { return .nan }
            let sorted = values.sorted()
            let index = min(sorted.count - 1, Int(Double(sorted.count - 1) * p))
            return sorted[index]
        }

        // Schema side: a moderately deep, moderately branchy schema (mirrors `nested` above), walked
        // via a long generated instance so most byte positions are genuinely distinct states.
        let schemaClassifications = buildLargeVocab()
        let schemaTable = JSONSchemaConstraintTable(classifications: schemaClassifications)
        let deepSchema: JSONSchemaNode = .object([
            JSONSchemaProperty(name: "id", required: true, value: .integer),
            JSONSchemaProperty(
                name: "items", required: true,
                value: .array(
                    items: .object([
                        JSONSchemaProperty(name: "k", required: true, value: .string),
                        JSONSchemaProperty(name: "v", required: true, value: .integer),
                    ]))),
        ])
        var schemaRNG = SplitMix64(seed: 0xF00D_BEEF_1357_2468)
        var schemaAutomaton = JSONSchemaAutomaton(root: deepSchema, fingerprint: "fp-bench")
        var schemaMissDurations: [Double] = []
        let schemaInstanceBytes = Self.generateInstanceBytes(deepSchema, rng: &schemaRNG)
        for byte in schemaInstanceBytes {
            let missesBefore = schemaTable.cacheMissCount
            let start = DispatchTime.now()
            _ = try schemaTable.allowedTokenBitset(for: schemaAutomaton)
            let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
            if schemaTable.cacheMissCount > missesBefore {
                schemaMissDurations.append(elapsedMs)
            }
            XCTAssertTrue(schemaAutomaton.advance(byte: byte))
        }

        // json_object side: same vocab scale, `warmCache: false` so every visited state misses once.
        let objectClassifications = buildLargeVocab()
        let objectTable = JSONObjectConstraintTable(classifications: objectClassifications, warmCache: false)
        var objectAutomaton = JSONObjectAutomaton()
        var objectMissDurations: [Double] = []
        let objectTraceBytes = Array(#"{"a":"x\n","b":[1,-2.5e+3],"c":123,"d":{"e":"f"}}"#.utf8)
        for byte in objectTraceBytes {
            let missesBefore = objectTable.cacheMissCount
            let start = DispatchTime.now()
            _ = try objectTable.allowedTokenBitset(for: objectAutomaton)
            let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
            if objectTable.cacheMissCount > missesBefore {
                objectMissDurations.append(elapsedMs)
            }
            XCTAssertTrue(objectAutomaton.advance(byte: byte))
        }

        print(
            """
            [json-schema-constraint-bench] vocabSize=\(vocabSize) \
            schemaMisses=\(schemaMissDurations.count) schemaMissP50ms=\(percentile(schemaMissDurations, 0.50)) \
            schemaMissP99ms=\(percentile(schemaMissDurations, 0.99)) \
            objectMisses=\(objectMissDurations.count) objectMissP50ms=\(percentile(objectMissDurations, 0.50)) \
            objectMissP99ms=\(percentile(objectMissDurations, 0.99))
            """)
    }
}
