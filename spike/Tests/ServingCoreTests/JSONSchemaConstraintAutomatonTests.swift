import XCTest

@testable import ServingCore

/// Correctness tests for `JSONSchemaAutomaton` (response-format slice 2b).
///
/// Three layers:
/// 1. Targeted accept/reject cases, one per documented rule in `JSONSchemaResponseFormat.swift`.
/// 2. A prefix-property sweep: every proper prefix of an accepted instance must itself advance
///    without failure, and `isComplete` must be false until the very last byte (root type here is
///    always `.object`/`.array`, so the "bare top-level number" `isComplete` quirk documented on
///    `JSONSchemaAutomaton.isComplete` never applies to this sweep — see
///    `testRootNumberIsCompleteAtEveryTerminatableDigitPosition` for that quirk in isolation).
/// 3. A differential test against an INDEPENDENT validator (`ReferenceJSONParser` + `validate`,
///    below): a hand-written recursive-descent JSON parser (not sharing any code with
///    `JSONSchemaAutomaton`, and using the Swift standard library's own UTF-8 decoder —
///    `String(bytes:encoding:)` — rather than reimplementing the automaton's continuation-byte
///    range table) that preserves object key order and duplicates, plus a recursive schema
///    validator over the resulting tree. Candidates are generated deterministically (SplitMix64,
///    never `SystemRandomNumberGenerator`) from several schemas, both as valid instances and as
///    mutations (byte insert/delete/replace, key swap, dropped-required, added-extra-key) of them.
final class JSONSchemaConstraintAutomatonTests: XCTestCase {

    // MARK: - Shared schema fixtures

    private static func literal(_ jsonText: String) -> JSONSchemaLiteral {
        JSONSchemaLiteral(jsonText: Array(jsonText.utf8))
    }

    private static let flatObjectProps: [JSONSchemaProperty] = [
        JSONSchemaProperty(name: "name", required: true, value: .string),
        JSONSchemaProperty(name: "age", required: false, value: .integer),
    ]
    private static let flatObjectSchema: JSONSchemaNode = .object(flatObjectProps)

    private static let nestedProps: [JSONSchemaProperty] = [
        JSONSchemaProperty(name: "id", required: true, value: .integer),
        JSONSchemaProperty(name: "tags", required: true, value: .array(items: .string)),
        JSONSchemaProperty(
            name: "meta", required: false,
            value: .object([JSONSchemaProperty(name: "active", required: true, value: .boolean)])),
    ]
    private static let nestedSchema: JSONSchemaNode = .object(nestedProps)

    private static let enumProps: [JSONSchemaProperty] = [
        JSONSchemaProperty(
            name: "status", required: true,
            value: .enumeration([literal(#""active""#), literal(#""inactive""#), literal(#""pending""#)]))
    ]
    private static let enumSchema: JSONSchemaNode = .object(enumProps)

    private static let nullableProps: [JSONSchemaProperty] = [
        JSONSchemaProperty(name: "nickname", required: true, value: .anyOf([.string, .null]))
    ]
    private static let nullableSchema: JSONSchemaNode = .object(nullableProps)

    private static let arrayItemProps: [JSONSchemaProperty] = [
        JSONSchemaProperty(name: "x", required: true, value: .integer),
        JSONSchemaProperty(name: "y", required: true, value: .integer),
    ]
    private static let arrayOfObjectsSchema: JSONSchemaNode = .array(items: .object(arrayItemProps))

    private static let numberIntegerProps: [JSONSchemaProperty] = [
        JSONSchemaProperty(name: "price", required: true, value: .number),
        JSONSchemaProperty(name: "count", required: true, value: .integer),
    ]
    private static let numberIntegerSchema: JSONSchemaNode = .object(numberIntegerProps)

    private static let anyOfThreeProps: [JSONSchemaProperty] = [
        JSONSchemaProperty(name: "value", required: true, value: .anyOf([.integer, .string, .boolean]))
    ]
    private static let anyOfThreeSchema: JSONSchemaNode = .object(anyOfThreeProps)

    private static let twoRequiredSchema: JSONSchemaNode = .object([
        JSONSchemaProperty(name: "a", required: true, value: .integer),
        JSONSchemaProperty(name: "b", required: true, value: .integer),
    ])

    private static let allOptionalSchema: JSONSchemaNode = .object([
        JSONSchemaProperty(name: "a", required: false, value: .integer),
        JSONSchemaProperty(name: "b", required: false, value: .integer),
    ])

    private static let stringOnlySchema: JSONSchemaNode = .object([
        JSONSchemaProperty(name: "s", required: true, value: .string)
    ])

    /// All differential/prefix fuzz schemas, paired with the property list a structural mutation
    /// (swap/drop/add) should target — the schema's own top-level properties for object-rooted
    /// schemas, or the item object's properties for the array-rooted schema.
    private struct FuzzSchema {
        let name: String
        let node: JSONSchemaNode
        let mutationProps: [JSONSchemaProperty]
    }
    private static let fuzzSchemas: [FuzzSchema] = [
        FuzzSchema(name: "flatObject", node: flatObjectSchema, mutationProps: flatObjectProps),
        FuzzSchema(name: "nestedObjectArray", node: nestedSchema, mutationProps: nestedProps),
        FuzzSchema(name: "enumStatus", node: enumSchema, mutationProps: enumProps),
        FuzzSchema(name: "nullableAnyOf", node: nullableSchema, mutationProps: nullableProps),
        FuzzSchema(name: "arrayOfObjects", node: arrayOfObjectsSchema, mutationProps: arrayItemProps),
        FuzzSchema(name: "numberVsInteger", node: numberIntegerSchema, mutationProps: numberIntegerProps),
        FuzzSchema(name: "anyOfThreeBranches", node: anyOfThreeSchema, mutationProps: anyOfThreeProps),
    ]

    // MARK: - Targeted accept/reject: object property order, duplicates, required/optional, extra keys

    func testAcceptsDeclaredOrder() {
        XCTAssertTrue(JSONSchemaAutomaton.accepts(bytes: Array(#"{"a":1,"b":2}"#.utf8), root: Self.twoRequiredSchema))
    }

    func testRejectsWrongOrder() {
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: Array(#"{"b":2,"a":1}"#.utf8), root: Self.twoRequiredSchema))
    }

    func testRejectsDuplicateKey() {
        XCTAssertFalse(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"a":1,"a":2}"#.utf8), root: Self.twoRequiredSchema))
    }

    func testRejectsMissingRequired() {
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: Array(#"{"a":1}"#.utf8), root: Self.twoRequiredSchema))
    }

    func testRejectsExtraKey() {
        let schema: JSONSchemaNode = .object([
            JSONSchemaProperty(name: "a", required: true, value: .integer),
            JSONSchemaProperty(name: "b", required: false, value: .integer),
        ])
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: Array(#"{"a":1,"z":2}"#.utf8), root: schema))
    }

    func testOptionalPropertyMaySkip() {
        XCTAssertTrue(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"name":"x"}"#.utf8), root: Self.flatObjectSchema))
        XCTAssertTrue(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"name":"x","age":5}"#.utf8), root: Self.flatObjectSchema))
    }

    func testRequiredPropertyMayNotSkip() {
        // `name` is required; an instance with only `age` is missing it, even though `age` alone
        // would otherwise be syntactically fine.
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: Array(#"{"age":5}"#.utf8), root: Self.flatObjectSchema))
    }

    func testEmptyObjectWhenAllOptional() {
        XCTAssertTrue(JSONSchemaAutomaton.accepts(bytes: Array("{}".utf8), root: Self.allOptionalSchema))
    }

    // MARK: - Targeted accept/reject: integer vs number

    func testIntegerRejectsFraction() {
        XCTAssertFalse(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"price":1,"count":1.0}"#.utf8), root: Self.numberIntegerSchema))
    }

    func testIntegerRejectsExponent() {
        XCTAssertFalse(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"price":1,"count":1e3}"#.utf8), root: Self.numberIntegerSchema))
    }

    func testIntegerAcceptsPlainDigits() {
        XCTAssertTrue(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"price":1,"count":3}"#.utf8), root: Self.numberIntegerSchema))
    }

    func testNumberAcceptsNegativeZero() {
        XCTAssertTrue(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"price":-0,"count":3}"#.utf8), root: Self.numberIntegerSchema))
    }

    func testNumberAcceptsFractionAndNegativeExponent() {
        XCTAssertTrue(
            JSONSchemaAutomaton.accepts(
                bytes: Array(#"{"price":0.5e-3,"count":3}"#.utf8), root: Self.numberIntegerSchema))
    }

    func testLeadingZeroRejected() {
        XCTAssertFalse(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"price":1,"count":01}"#.utf8), root: Self.numberIntegerSchema))
    }

    // MARK: - Targeted accept/reject: enumeration

    func testEnumAcceptsExactLiteral() {
        XCTAssertTrue(JSONSchemaAutomaton.accepts(bytes: Array(#"{"status":"active"}"#.utf8), root: Self.enumSchema))
    }

    func testEnumRejectsNearMissPrefix() {
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: Array(#"{"status":"activ"}"#.utf8), root: Self.enumSchema))
    }

    func testEnumRejectsSuperset() {
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: Array(#"{"status":"activee"}"#.utf8), root: Self.enumSchema))
    }

    func testEnumRejectsValueOutsideList() {
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: Array(#"{"status":"gone"}"#.utf8), root: Self.enumSchema))
    }

    // MARK: - Targeted accept/reject: anyOf (nullable, and three-way branch commit)

    func testAnyOfNullableAcceptsString() {
        XCTAssertTrue(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"nickname":"bob"}"#.utf8), root: Self.nullableSchema))
    }

    func testAnyOfNullableAcceptsNull() {
        XCTAssertTrue(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"nickname":null}"#.utf8), root: Self.nullableSchema))
    }

    func testAnyOfNullableRejectsOtherType() {
        XCTAssertFalse(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"nickname":false}"#.utf8), root: Self.nullableSchema))
    }

    func testAnyOfThreeBranchesCommitsOnFirstByte() {
        XCTAssertTrue(JSONSchemaAutomaton.accepts(bytes: Array(#"{"value":42}"#.utf8), root: Self.anyOfThreeSchema))
        XCTAssertTrue(JSONSchemaAutomaton.accepts(bytes: Array(#"{"value":"hi"}"#.utf8), root: Self.anyOfThreeSchema))
        XCTAssertTrue(JSONSchemaAutomaton.accepts(bytes: Array(#"{"value":true}"#.utf8), root: Self.anyOfThreeSchema))
        // A type outside the three branches (null) has no branch to route to.
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: Array(#"{"value":null}"#.utf8), root: Self.anyOfThreeSchema))
        // Committing to the integer branch on '4' then finding a non-digit is a hard reject, not a
        // fallback to another branch (no backtracking past the first byte).
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: Array(#"{"value":4x}"#.utf8), root: Self.anyOfThreeSchema))
    }

    // MARK: - Targeted accept/reject: arrays, including nested arrays of objects

    func testArrayOfObjectsAcceptsEmpty() {
        XCTAssertTrue(JSONSchemaAutomaton.accepts(bytes: Array("[]".utf8), root: Self.arrayOfObjectsSchema))
    }

    func testArrayOfObjectsAcceptsMultipleItems() {
        XCTAssertTrue(
            JSONSchemaAutomaton.accepts(
                bytes: Array(#"[{"x":1,"y":2},{"x":3,"y":4}]"#.utf8), root: Self.arrayOfObjectsSchema))
    }

    func testArrayOfObjectsRejectsItemMissingRequired() {
        XCTAssertFalse(
            JSONSchemaAutomaton.accepts(bytes: Array(#"[{"x":1}]"#.utf8), root: Self.arrayOfObjectsSchema))
    }

    func testNestedArraysOfObjects() {
        // An object property whose value is an array of arrays of objects — deeper nesting than the
        // root-level `arrayOfObjectsSchema` case above.
        let schema: JSONSchemaNode = .object([
            JSONSchemaProperty(
                name: "groups", required: true,
                value: .array(items: .array(items: .object([JSONSchemaProperty(name: "id", required: true, value: .integer)])))
            )
        ])
        XCTAssertTrue(JSONSchemaAutomaton.accepts(bytes: Array(#"{"groups":[[{"id":1}],[]]}"#.utf8), root: schema))
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: Array(#"{"groups":[[{}]]}"#.utf8), root: schema))
    }

    // MARK: - Targeted accept/reject: string escapes and UTF-8

    func testStringAcceptsUnicodeEscape() {
        XCTAssertTrue(
            JSONSchemaAutomaton.accepts(bytes: Array(#"{"s":"café"}"#.utf8), root: Self.stringOnlySchema))
    }

    func testStringRejectsBadEscape() {
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: Array(#"{"s":"\q"}"#.utf8), root: Self.stringOnlySchema))
    }

    func testStringRejectsRawControlCharacter() {
        var bytes = Array(#"{"s":""#.utf8)
        bytes.append(0x01)
        bytes.append(contentsOf: Array(#""}"#.utf8))
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: bytes, root: Self.stringOnlySchema))
    }

    func testStringAcceptsValidRawUTF8() {
        var bytes = Array(#"{"s":""#.utf8)
        bytes.append(contentsOf: [0xC3, 0xA9])  // "é"
        bytes.append(contentsOf: Array(#""}"#.utf8))
        XCTAssertTrue(JSONSchemaAutomaton.accepts(bytes: bytes, root: Self.stringOnlySchema))
    }

    func testStringRejectsInvalidUTF8LoneContinuationByte() {
        var bytes = Array(#"{"s":""#.utf8)
        bytes.append(0x80)  // a bare continuation byte can never start a UTF-8 sequence
        bytes.append(contentsOf: Array(#""}"#.utf8))
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: bytes, root: Self.stringOnlySchema))
    }

    func testStringRejectsInvalidUTF8OverlongLead() {
        var bytes = Array(#"{"s":""#.utf8)
        bytes.append(contentsOf: [0xC0, 0x80])  // overlong encoding of NUL; 0xC0 is never a valid lead byte
        bytes.append(contentsOf: Array(#""}"#.utf8))
        XCTAssertFalse(JSONSchemaAutomaton.accepts(bytes: bytes, root: Self.stringOnlySchema))
    }

    // MARK: - Targeted: structural whitespace bounded like `JSONObjectAutomaton`

    func testWhitespaceCapBetweenTokensLikeJSONObjectAutomaton() {
        let prefix = Array(#"{"a":1,"#.utf8)
        let suffix = Array(#""b":2}"#.utf8)

        func accepts(spaceCount: Int) -> Bool {
            var automaton = JSONSchemaAutomaton(root: Self.twoRequiredSchema)
            for byte in prefix {
                guard automaton.advance(byte: byte) else { return false }
            }
            for _ in 0..<spaceCount {
                guard automaton.advance(byte: 0x20) else { return false }
            }
            for byte in suffix {
                guard automaton.advance(byte: byte) else { return false }
            }
            return automaton.isComplete
        }

        XCTAssertTrue(accepts(spaceCount: JSONObjectAutomaton.defaultMaxConsecutiveWhitespace))
        XCTAssertFalse(accepts(spaceCount: JSONObjectAutomaton.defaultMaxConsecutiveWhitespace + 1))
    }

    // MARK: - `isComplete`: root-number semantics (documented on `JSONSchemaAutomaton.isComplete`)

    /// `JSONSchemaAutomaton.isComplete` mirrors `JSONObjectAutomaton`'s own definition: "would EOS
    /// be accepted right here", not "is there no pending suffix that could extend this value". A
    /// bare top-level number has no closing delimiter of its own, so `isComplete` legitimately
    /// flips true at EVERY digit position where the number could legally terminate — even mid-way
    /// through a longer valid literal like "123", where digit-by-digit `1`, `12`, and `123` are all
    /// individually valid complete top-level integers. This is intentional, not a bug: the
    /// automaton only answers "is stopping HERE valid"; the caller (mask/generation loop) decides
    /// whether to keep extending.
    func testRootNumberIsCompleteAtEveryTerminatableDigitPosition() {
        var automaton = JSONSchemaAutomaton(root: .integer)
        var completedAtEachPosition: [Bool] = []
        for byte in Array("123".utf8) {
            XCTAssertTrue(automaton.advance(byte: byte))
            completedAtEachPosition.append(automaton.isComplete)
        }
        XCTAssertEqual(completedAtEachPosition, [true, true, true])
    }

    // MARK: - Hashable / Equatable (mask cache key)

    func testEqualStatesCompareEqualAndHashEqual() {
        let base = JSONSchemaAutomaton(root: Self.twoRequiredSchema)
        var a = base
        var b = base
        for byte in Array(#"{"a":1"#.utf8) {
            XCTAssertTrue(a.advance(byte: byte))
            XCTAssertTrue(b.advance(byte: byte))
        }
        XCTAssertEqual(a, b)

        var hasherA = Hasher()
        a.hash(into: &hasherA)
        var hasherB = Hasher()
        b.hash(into: &hasherB)
        XCTAssertEqual(hasherA.finalize(), hasherB.finalize())
    }

    func testDifferentStatesCompareUnequal() {
        let base = JSONSchemaAutomaton(root: Self.twoRequiredSchema)
        var a = base
        for byte in Array(#"{"a":1"#.utf8) {
            XCTAssertTrue(a.advance(byte: byte))
        }
        // `b` is still required, so `,` (not `}`) is the valid next byte here — advancing it still
        // changes state (from "expect comma or end" to "awaiting the next key") without completing
        // the document, which is all this test needs to prove non-equality.
        var c = a
        XCTAssertTrue(c.advance(byte: UInt8(ascii: ",")))
        XCTAssertNotEqual(a, c)
    }

    func testSeparateInitCallsWithIdenticalSchemaCompareUnequal() {
        // Per the type doc on `Hashable` conformance: table IDENTITY (`===`), not structural
        // equality of the source `JSONSchemaNode`, is part of the key — this mirrors the
        // mask-cache layer, which keys on the caller's schema fingerprint, not structural equality.
        let x = JSONSchemaAutomaton(root: .integer)
        let y = JSONSchemaAutomaton(root: .integer)
        XCTAssertNotEqual(x, y)
    }

    func testEqualStatesUsableAsDictionaryKey() {
        let base = JSONSchemaAutomaton(root: Self.twoRequiredSchema)
        var a = base
        var b = base
        for byte in Array(#"{"a":1"#.utf8) {
            XCTAssertTrue(a.advance(byte: byte))
            XCTAssertTrue(b.advance(byte: byte))
        }
        var cache: [JSONSchemaAutomaton: String] = [:]
        cache[a] = "mask-for-state"
        XCTAssertEqual(cache[b], "mask-for-state")
    }

    // MARK: - Prefix property

    /// For every accepted instance of a (object/array-rooted) fuzz schema, every proper prefix must
    /// advance without failure, and `isComplete` must be false until the final byte. (Root type here
    /// is always `.object` or `.array` — see `fuzzSchemas` — so the bare-top-level-number quirk
    /// documented on `isComplete` and exercised by
    /// `testRootNumberIsCompleteAtEveryTerminatableDigitPosition` never applies to this sweep: the
    /// automaton's `stack` is non-empty for the entire prefix of any object/array-rooted instance
    /// until its very last closing byte.)
    func testPrefixesAdvanceAndIsCompleteOnlyAtFinalByte() {
        var rng = SplitMix64(seed: 0xABCD_EF01_2345_6789)
        var checkedInstances = 0
        for schema in Self.fuzzSchemas {
            for _ in 0..<15 {
                let instance = generateInstance(schema.node, rng: &rng)
                let bytes = serializeRefJSON(instance)
                guard bytes.count > 1 else { continue }
                checkedInstances += 1

                var automaton = JSONSchemaAutomaton(root: schema.node)
                for (index, byte) in bytes.enumerated() {
                    XCTAssertTrue(
                        automaton.advance(byte: byte),
                        "schema \(schema.name): prefix advance failed at index \(index) of "
                            + String(decoding: bytes, as: UTF8.self))
                    if index < bytes.count - 1 {
                        XCTAssertFalse(
                            automaton.isComplete,
                            "schema \(schema.name): isComplete true before the final byte (index \(index))")
                    } else {
                        XCTAssertTrue(automaton.isComplete, "schema \(schema.name): isComplete false at final byte")
                    }
                }
            }
        }
        XCTAssertGreaterThan(checkedInstances, 50, "sanity: the generator must produce enough instances to be meaningful")
    }

    // MARK: - Differential test against an independent validator

    func testDifferentialAgreementWithIndependentValidator() throws {
        var rng = SplitMix64(seed: 0xC0FF_EE12_3456_789A)
        var acceptedCount = 0
        var rejectedCount = 0
        var totalCandidates = 0
        var firstMismatch: (schema: String, bytes: [UInt8], automaton: Bool, validator: Bool)?

        // Curated non-whitespace mutation byte pool: printable ASCII (so structural/lexical bytes
        // like quote, backslash, braces, digits, minus, dot, e/E are all reachable) plus a few
        // bytes chosen to trigger invalid-UTF-8 rejections (a bare continuation byte, an overlong
        // lead byte, a surrogate-range lead byte, and an out-of-range byte). Whitespace bytes
        // (space/tab/CR/LF) are deliberately excluded: the differential's job here is schema/type
        // semantics, not the whitespace cap (covered separately by
        // `testWhitespaceCapBetweenTokensLikeJSONObjectAutomaton`) — including them would make
        // "automaton rejects, validator accepts" mismatches that are really just the (automaton-only)
        // whitespace cap, not a genuine schema-semantics bug.
        let mutationPool: [UInt8] = Array(0x21...0x7E) + [0x80, 0xFF, 0xC0, 0xED]

        func check(_ bytes: [UInt8], schemaName: String, node: JSONSchemaNode) {
            totalCandidates += 1
            let automatonAccepted = JSONSchemaAutomaton.accepts(bytes: bytes, root: node)
            let validatorAccepted = independentAccepts(bytes, schema: node)
            if automatonAccepted {
                acceptedCount += 1
            } else {
                rejectedCount += 1
            }
            if firstMismatch == nil && automatonAccepted != validatorAccepted {
                firstMismatch = (schemaName, bytes, automatonAccepted, validatorAccepted)
            }
        }

        for schema in Self.fuzzSchemas {
            for _ in 0..<50 {
                let instance = generateInstance(schema.node, rng: &rng)
                let validBytes = serializeRefJSON(instance)
                check(validBytes, schemaName: schema.name, node: schema.node)

                // Raw byte-level mutations of the valid instance.
                check(mutateInsert(validBytes, rng: &rng, pool: mutationPool), schemaName: schema.name, node: schema.node)
                check(mutateInsert(validBytes, rng: &rng, pool: mutationPool), schemaName: schema.name, node: schema.node)
                check(mutateDelete(validBytes, rng: &rng), schemaName: schema.name, node: schema.node)
                check(mutateReplace(validBytes, rng: &rng, pool: mutationPool), schemaName: schema.name, node: schema.node)
                check(mutateReplace(validBytes, rng: &rng, pool: mutationPool), schemaName: schema.name, node: schema.node)

                // Structural mutations (key swap / dropped required / added extra key), applied at
                // the root object (or, for the array-rooted schema, its first item object).
                for kind in [StructuralMutationKind.swapKeys, .dropRequired, .addExtraKey] {
                    if let mutated = structurallyMutatedRoot(
                        original: instance, props: schema.mutationProps, kind: kind, rng: &rng)
                    {
                        check(serializeRefJSON(mutated), schemaName: schema.name, node: schema.node)
                    }
                }
            }
        }

        if let mismatch = firstMismatch {
            XCTFail(
                """
                differential mismatch for schema \(mismatch.schema): automaton=\(mismatch.automaton) \
                validator=\(mismatch.validator)
                bytes(lossy-utf8)=\(String(decoding: mismatch.bytes, as: UTF8.self))
                bytes(hex)=\(mismatch.bytes.map { String(format: "%02x", $0) }.joined(separator: " "))
                """)
        }

        // Both classes populated, so this cannot pass vacuously.
        XCTAssertGreaterThanOrEqual(totalCandidates, 2000, "expected at least ~2000 generated candidates")
        XCTAssertGreaterThan(acceptedCount, 0, "differential corpus produced zero ACCEPTs")
        XCTAssertGreaterThan(rejectedCount, 0, "differential corpus produced zero REJECTs")
    }
}

// MARK: - Deterministic PRNG (SplitMix64; never SystemRandomNumberGenerator)

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

// MARK: - Generator-side tree (also used as the input to raw serialization for the automaton side)

/// A parsed/generated JSON value. Object keys preserve declaration order and duplicates (unlike
/// `Dictionary`), which is exactly what both the generator (declared-order property emission) and
/// the independent parser (order/duplicate-sensitive validation) need. `.rawLiteral` is
/// generator-only (never produced by `ReferenceJSONParser`): it lets the generator emit an enum's
/// exact declared literal bytes verbatim, without re-deriving/re-escaping them.
private enum RefJSON {
    case object([(key: String, value: RefJSON)])
    case array([RefJSON])
    case string(String)
    case number(String)  // raw JSON number text, preserved exactly (integer-vs-fraction matters)
    case bool(Bool)
    case null
    case rawLiteral([UInt8])
}

// MARK: - Generic instance generator (works for any `JSONSchemaNode`, not just the fuzz schemas)

private func generateInstance(_ node: JSONSchemaNode, rng: inout SplitMix64) -> RefJSON {
    switch node {
    case .string:
        return .string(randomString(rng: &rng))
    case .number:
        return .number(randomNumberText(rng: &rng, allowFraction: true))
    case .integer:
        return .number(randomNumberText(rng: &rng, allowFraction: false))
    case .boolean:
        return .bool(rng.nextBool())
    case .null:
        return .null
    case .enumeration(let literals):
        return .rawLiteral(literals[rng.nextInt(literals.count)].jsonText)
    case .object(let props):
        var pairs: [(key: String, value: RefJSON)] = []
        for prop in props {
            guard prop.required || rng.nextBool() else { continue }
            pairs.append((prop.name, generateInstance(prop.value, rng: &rng)))
        }
        return .object(pairs)
    case .array(let items):
        let count = rng.nextInt(4)  // 0...3
        return .array((0..<count).map { _ in generateInstance(items, rng: &rng) })
    case .anyOf(let branches):
        return generateInstance(branches[rng.nextInt(branches.count)], rng: &rng)
    }
}

private func randomString(rng: inout SplitMix64) -> String {
    let length = rng.nextInt(6)
    let letters = Array("abcdefghijklmnopqrstuvwxyzABCDEFG0123456789")
    var characters: [Character] = []
    for _ in 0..<length {
        switch rng.nextInt(10) {
        case 0: characters.append("é")
        case 1: characters.append(" ")
        default: characters.append(letters[rng.nextInt(letters.count)])
        }
    }
    return String(characters)
}

private func randomNumberText(rng: inout SplitMix64, allowFraction: Bool) -> String {
    var text = ""
    if rng.nextBool() { text += "-" }
    if rng.nextBool() {
        text += "0"
    } else {
        text += String(1 + rng.nextInt(9))
        for _ in 0..<rng.nextInt(4) { text += String(rng.nextInt(10)) }
    }
    if allowFraction {
        if rng.nextBool() {
            text += "."
            for _ in 0..<(1 + rng.nextInt(3)) { text += String(rng.nextInt(10)) }
        }
        if rng.nextBool() {
            text += rng.nextBool() ? "e" : "E"
            if rng.nextBool() { text += rng.nextBool() ? "+" : "-" }
            for _ in 0..<(1 + rng.nextInt(2)) { text += String(rng.nextInt(10)) }
        }
    }
    return text
}

// MARK: - Serializer (generator-side `RefJSON` -> compact JSON bytes)

private func serializeRefJSON(_ value: RefJSON) -> [UInt8] {
    switch value {
    case .object(let pairs):
        var out: [UInt8] = [0x7B]
        for (index, pair) in pairs.enumerated() {
            if index > 0 { out.append(0x2C) }
            out.append(contentsOf: serializeJSONString(pair.key))
            out.append(0x3A)
            out.append(contentsOf: serializeRefJSON(pair.value))
        }
        out.append(0x7D)
        return out
    case .array(let elements):
        var out: [UInt8] = [0x5B]
        for (index, element) in elements.enumerated() {
            if index > 0 { out.append(0x2C) }
            out.append(contentsOf: serializeRefJSON(element))
        }
        out.append(0x5D)
        return out
    case .string(let content):
        return serializeJSONString(content)
    case .number(let raw):
        return Array(raw.utf8)
    case .bool(let flag):
        return Array((flag ? "true" : "false").utf8)
    case .null:
        return Array("null".utf8)
    case .rawLiteral(let bytes):
        return bytes
    }
}

private func serializeJSONString(_ content: String) -> [UInt8] {
    var out: [UInt8] = [0x22]
    for scalar in content.unicodeScalars {
        switch scalar {
        case "\"": out.append(contentsOf: Array(#"\""#.utf8))
        case "\\": out.append(contentsOf: Array(#"\\"#.utf8))
        case "\n": out.append(contentsOf: Array(#"\n"#.utf8))
        case "\r": out.append(contentsOf: Array(#"\r"#.utf8))
        case "\t": out.append(contentsOf: Array(#"\t"#.utf8))
        default:
            if scalar.value < 0x20 {
                out.append(contentsOf: Array(String(format: "\\u%04x", scalar.value).utf8))
            } else {
                out.append(contentsOf: Array(String(scalar).utf8))
            }
        }
    }
    out.append(0x22)
    return out
}

// MARK: - Raw byte-level mutations

private func mutateInsert(_ bytes: [UInt8], rng: inout SplitMix64, pool: [UInt8]) -> [UInt8] {
    var out = bytes
    out.insert(pool[rng.nextInt(pool.count)], at: rng.nextInt(out.count + 1))
    return out
}

private func mutateDelete(_ bytes: [UInt8], rng: inout SplitMix64) -> [UInt8] {
    guard bytes.count > 1 else { return bytes }
    var out = bytes
    out.remove(at: rng.nextInt(out.count))
    return out
}

private func mutateReplace(_ bytes: [UInt8], rng: inout SplitMix64, pool: [UInt8]) -> [UInt8] {
    guard !bytes.isEmpty else { return bytes }
    var out = bytes
    out[rng.nextInt(out.count)] = pool[rng.nextInt(pool.count)]
    return out
}

// MARK: - Structural mutations (key swap / dropped-required / added-extra-key)

private enum StructuralMutationKind {
    case swapKeys
    case dropRequired
    case addExtraKey
}

private func mutatePairs(
    _ pairs: [(key: String, value: RefJSON)], props: [JSONSchemaProperty], kind: StructuralMutationKind,
    rng: inout SplitMix64
) -> [(key: String, value: RefJSON)]? {
    switch kind {
    case .swapKeys:
        guard pairs.count >= 2 else { return nil }
        var mutated = pairs
        let i = rng.nextInt(mutated.count)
        var j = rng.nextInt(mutated.count)
        while j == i { j = rng.nextInt(mutated.count) }
        mutated.swapAt(i, j)
        return mutated
    case .dropRequired:
        let requiredPresent = pairs.filter { pair in props.first(where: { $0.name == pair.key })?.required == true }
        guard !requiredPresent.isEmpty else { return nil }
        let victim = requiredPresent[rng.nextInt(requiredPresent.count)].key
        return pairs.filter { $0.key != victim }
    case .addExtraKey:
        var mutated = pairs
        mutated.insert(("__extra_mutated_key__", .string("x")), at: rng.nextInt(mutated.count + 1))
        return mutated
    }
}

/// Applies a structural mutation at the schema's object level: directly, for an object-rooted
/// schema, or to the first array element, for the (single) array-rooted fuzz schema. Returns `nil`
/// if the mutation doesn't apply to this particular instance (e.g. `swapKeys` needs >= 2 present
/// pairs) — the caller skips that candidate rather than mutating something irrelevant.
private func structurallyMutatedRoot(
    original: RefJSON, props: [JSONSchemaProperty], kind: StructuralMutationKind, rng: inout SplitMix64
) -> RefJSON? {
    switch original {
    case .object(let pairs):
        guard let mutated = mutatePairs(pairs, props: props, kind: kind, rng: &rng) else { return nil }
        return .object(mutated)
    case .array(let elements):
        guard let first = elements.first, case .object(let pairs) = first else { return nil }
        guard let mutated = mutatePairs(pairs, props: props, kind: kind, rng: &rng) else { return nil }
        var newElements = elements
        newElements[0] = .object(mutated)
        return .array(newElements)
    default:
        return nil
    }
}

// MARK: - Independent validator: hand-written recursive-descent JSON parser + schema validator

/// A JSON parser sharing no code with `JSONSchemaAutomaton`: recursive descent over an index into
/// `[UInt8]`, rather than a byte-at-a-time pushdown automaton, and using the Swift standard
/// library's own UTF-8 decoder (`String(bytes:encoding:)`) to validate string content instead of
/// reimplementing `JSONSchemaAutomaton`'s continuation-byte range table. Preserves object key order
/// and duplicates (unlike `JSONSerialization`, whose `NSDictionary` result loses both), which the
/// schema validator below depends on.
private struct ReferenceJSONParser {
    let bytes: [UInt8]
    var index: Int = 0

    init(_ bytes: [UInt8]) { self.bytes = bytes }

    /// Parses `bytes` as exactly one JSON value with no leading or trailing bytes (mirrors
    /// `JSONSchemaAutomaton`'s own "no whitespace before the first byte or after the last" rule at
    /// the top level — see `advanceStructural`). Returns `nil` on any syntax error.
    static func parseWhole(_ bytes: [UInt8]) -> RefJSON? {
        var parser = ReferenceJSONParser(bytes)
        guard let value = parser.parseValue() else { return nil }
        guard parser.index == bytes.count else { return nil }
        return value
    }

    private func peek() -> UInt8? { index < bytes.count ? bytes[index] : nil }

    private mutating func skipWhitespace() {
        while let b = peek(), b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D { index += 1 }
    }

    private mutating func parseValue() -> RefJSON? {
        skipWhitespace()
        guard let b = peek() else { return nil }
        switch b {
        case 0x7B: return parseObject()
        case 0x5B: return parseArray()
        case 0x22: return parseString().map { RefJSON.string($0) }
        case 0x74: return parseLiteral("true", .bool(true))
        case 0x66: return parseLiteral("false", .bool(false))
        case 0x6E: return parseLiteral("null", .null)
        case 0x2D, 0x30...0x39: return parseNumber()
        default: return nil
        }
    }

    private mutating func parseLiteral(_ word: String, _ value: RefJSON) -> RefJSON? {
        let wordBytes = Array(word.utf8)
        guard index + wordBytes.count <= bytes.count, Array(bytes[index..<(index + wordBytes.count)]) == wordBytes
        else { return nil }
        index += wordBytes.count
        return value
    }

    private mutating func parseObject() -> RefJSON? {
        index += 1  // consume '{'
        var pairs: [(key: String, value: RefJSON)] = []
        skipWhitespace()
        if peek() == 0x7D {
            index += 1
            return .object(pairs)
        }
        while true {
            skipWhitespace()
            guard peek() == 0x22, let key = parseString() else { return nil }
            skipWhitespace()
            guard peek() == 0x3A else { return nil }
            index += 1
            guard let value = parseValue() else { return nil }
            pairs.append((key, value))
            skipWhitespace()
            guard let b = peek() else { return nil }
            if b == 0x2C {
                index += 1
                continue
            }
            if b == 0x7D {
                index += 1
                break
            }
            return nil
        }
        return .object(pairs)
    }

    private mutating func parseArray() -> RefJSON? {
        index += 1  // consume '['
        var elements: [RefJSON] = []
        skipWhitespace()
        if peek() == 0x5D {
            index += 1
            return .array(elements)
        }
        while true {
            guard let value = parseValue() else { return nil }
            elements.append(value)
            skipWhitespace()
            guard let b = peek() else { return nil }
            if b == 0x2C {
                index += 1
                continue
            }
            if b == 0x5D {
                index += 1
                break
            }
            return nil
        }
        return .array(elements)
    }

    /// Parses a JSON string (assumes `peek() == '"'`). Validates escapes and rejects unescaped
    /// control bytes per RFC 8259; validates raw (unescaped) content bytes as well-formed UTF-8
    /// using the standard library's own decoder (`String(bytes:encoding:.utf8)`), independent of
    /// `JSONSchemaAutomaton`'s hand-rolled continuation-byte table. `\uXXXX` escapes are consumed
    /// (4 hex digits required) but not decoded into the content buffer — like
    /// `JSONSchemaAutomaton`, this parser does not validate surrogate pairing (a lone `\ud800` is
    /// accepted by both, matching each other, not full Unicode-scalar well-formedness), and the
    /// decoded value's exact content is never schema-significant (`.string` accepts any valid
    /// string) — only the raw bytes matter for the UTF-8-validity check.
    private mutating func parseString() -> String? {
        index += 1  // consume opening quote
        var rawContent: [UInt8] = []
        while true {
            guard let b = peek() else { return nil }
            if b == 0x22 {
                index += 1
                break
            }
            if b == 0x5C {
                index += 1
                guard let esc = peek() else { return nil }
                switch esc {
                case 0x22, 0x5C, 0x2F: rawContent.append(esc); index += 1
                case 0x62: rawContent.append(0x08); index += 1
                case 0x66: rawContent.append(0x0C); index += 1
                case 0x6E: rawContent.append(0x0A); index += 1
                case 0x72: rawContent.append(0x0D); index += 1
                case 0x74: rawContent.append(0x09); index += 1
                case 0x75:
                    index += 1
                    guard parseHex4() else { return nil }
                default:
                    return nil
                }
                continue
            }
            if b < 0x20 { return nil }  // unescaped control character
            rawContent.append(b)
            index += 1
        }
        return String(bytes: rawContent, encoding: .utf8)  // nil ⇒ invalid UTF-8 ⇒ parse failure
    }

    private mutating func parseHex4() -> Bool {
        for _ in 0..<4 {
            guard let b = peek(), isHexDigit(b) else { return false }
            index += 1
        }
        return true
    }

    private func isHexDigit(_ b: UInt8) -> Bool {
        (0x30...0x39).contains(b) || (0x41...0x46).contains(b) || (0x61...0x66).contains(b)
    }

    /// Standard JSON number grammar: optional `-`, an integer part with no leading zero unless it
    /// is exactly `0`, an optional `.` fraction (at least one digit required), and an optional
    /// `e`/`E` exponent (optional sign, at least one digit required). Raw source text is preserved
    /// verbatim so the schema validator can check integer-vs-fraction/exponent exactly.
    private mutating func parseNumber() -> RefJSON? {
        let start = index
        if peek() == 0x2D { index += 1 }
        guard let d0 = peek(), (0x30...0x39).contains(d0) else { return nil }
        if d0 == 0x30 {
            index += 1
        } else {
            while let b = peek(), (0x30...0x39).contains(b) { index += 1 }
        }
        if peek() == 0x2E {
            index += 1
            guard let f0 = peek(), (0x30...0x39).contains(f0) else { return nil }
            index += 1
            while let b = peek(), (0x30...0x39).contains(b) { index += 1 }
        }
        if let e = peek(), e == 0x65 || e == 0x45 {
            index += 1
            if let sign = peek(), sign == 0x2B || sign == 0x2D { index += 1 }
            guard let d1 = peek(), (0x30...0x39).contains(d1) else { return nil }
            index += 1
            while let b = peek(), (0x30...0x39).contains(b) { index += 1 }
        }
        return .number(String(decoding: bytes[start..<index], as: UTF8.self))
    }
}

/// Recursively validates a parsed `RefJSON` tree against the IR, per the semantics documented on
/// `JSONSchemaNode`/`JSONSchemaProperty` in `JSONSchemaResponseFormat.swift`: declared property
/// order with optional-skip/required-no-skip and no extra keys; integer as a fraction/exponent-free
/// number; enum as an exact literal-byte match; `anyOf` as "matches at least one branch" (sound
/// because the compiler guarantees branches are pairwise disjoint on their first byte, so at most
/// one branch can ever structurally match).
private func validate(_ value: RefJSON, against node: JSONSchemaNode) -> Bool {
    switch node {
    case .string:
        if case .string = value { return true }
        return false
    case .number:
        if case .number = value { return true }
        return false
    case .integer:
        guard case .number(let raw) = value else { return false }
        return !raw.contains(where: { $0 == "." || $0 == "e" || $0 == "E" })
    case .boolean:
        if case .bool = value { return true }
        return false
    case .null:
        if case .null = value { return true }
        return false
    case .enumeration(let literals):
        let candidateBytes: [UInt8]
        switch value {
        case .string(let s): candidateBytes = Array("\"\(s)\"".utf8)
        case .number(let raw): candidateBytes = Array(raw.utf8)
        case .bool(let b): candidateBytes = Array((b ? "true" : "false").utf8)
        case .null: candidateBytes = Array("null".utf8)
        case .object, .array, .rawLiteral: return false
        }
        return literals.contains { $0.jsonText == candidateBytes }
    case .object(let props):
        guard case .object(let pairs) = value else { return false }
        var i = 0
        var j = 0
        while i < props.count {
            if j < pairs.count && pairs[j].key == props[i].name {
                guard validate(pairs[j].value, against: props[i].value) else { return false }
                i += 1
                j += 1
            } else if props[i].required {
                return false  // required property missing (absent, or out of declared order)
            } else {
                i += 1  // optional property skipped
            }
        }
        return j == pairs.count  // no leftover (extra, duplicate, or out-of-order) keys
    case .array(let items):
        guard case .array(let elements) = value else { return false }
        return elements.allSatisfy { validate($0, against: items) }
    case .anyOf(let branches):
        return branches.contains { validate(value, against: $0) }
    }
}

private func independentAccepts(_ bytes: [UInt8], schema: JSONSchemaNode) -> Bool {
    guard let value = ReferenceJSONParser.parseWhole(bytes) else { return false }
    return validate(value, against: schema)
}
