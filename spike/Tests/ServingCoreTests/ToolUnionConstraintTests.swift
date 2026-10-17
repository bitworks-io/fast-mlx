import Foundation
import XCTest

@testable import ServingCore

/// `tool_choice: "required"` enforcement, ServingCore half (cycle 259): the union automaton that accepts
/// exactly `{"name": "<N>", "arguments": <A_N>}` (AC1), its token mask over a toy vocab through the
/// shared trie (AC2), and the request-time refusals that must precede any generation (AC5, format side).
final class ToolUnionConstraintTests: XCTestCase {

    // MARK: - Fixtures

    private func tool(_ name: String, properties: String = "", required: String = "") -> String {
        let params =
            properties.isEmpty
            ? #"{"type":"object","additionalProperties":false}"#
            : #"{"type":"object","properties":{\#(properties)},"required":[\#(required)],"additionalProperties":false}"#
        return #"{"type":"function","function":{"name":"\#(name)","description":"d","parameters":\#(params)}}"#
    }

    private let weather = #"{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}"#
    private let timeTool = #"{"type":"function","function":{"name":"lookup_time","description":"t","parameters":{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"],"additionalProperties":false}}}"#
    private let bare = #"{"type":"function","function":{"name":"ping"}}"#

    private func specs(_ tools: [String]) throws -> [OpenAIToolSpec] {
        let body = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[\#(tools.joined(separator: ","))]}"#
        return try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8)).tools
    }

    private func union(_ tools: [String]) throws -> ToolUnionFormat {
        try specs(tools).requiredToolUnionFormat()
    }

    private func automaton(_ tools: [String]) throws -> ToolUnionAutomaton {
        ToolUnionAutomaton(format: try union(tools))
    }

    /// Feeds `text`; returns the index of the first rejected byte, or `nil` when all were accepted.
    private func firstRejection(_ text: String, in start: ToolUnionAutomaton) -> Int? {
        var current = start
        for (index, byte) in text.utf8.enumerated() {
            guard current.advance(byte: byte) else { return index }
        }
        return nil
    }

    private func accepts(_ text: String, in start: ToolUnionAutomaton) -> Bool {
        var current = start
        for byte in text.utf8 { guard current.advance(byte: byte) else { return false } }
        return current.isComplete
    }

    // MARK: - AC1: automaton

    func testAcceptsTheFramedDocumentForEachDeclaredTool() throws {
        let start = try automaton([weather, timeTool, bare])
        XCTAssertTrue(accepts(#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#, in: start))
        XCTAssertTrue(accepts(#"{"name": "lookup_time", "arguments": {"zone": "UTC"}}"#, in: start))
        XCTAssertTrue(accepts(#"{"name": "ping", "arguments": {}}"#, in: start))
    }

    func testIsCompleteOnlyAfterTheFinalBrace() throws {
        let start = try automaton([weather])
        let document = #"{"name": "get_weather", "arguments": {"city": "Paris"}}"#
        var current = start
        XCTAssertFalse(current.isComplete)
        let bytes = Array(document.utf8)
        for (index, byte) in bytes.enumerated() {
            XCTAssertTrue(current.advance(byte: byte), "byte \(index)")
            XCTAssertEqual(current.isComplete, index == bytes.count - 1, "complete after byte \(index)")
        }
    }

    func testRejectsAnUndeclaredName() throws {
        let start = try automaton([weather, timeTool])
        XCTAssertEqual(firstRejection(#"{"name": "nope""#, in: start), 10)
        // Shares a prefix with a declared name, then diverges.
        XCTAssertEqual(firstRejection(#"{"name": "get_x"#, in: start), 14)
    }

    func testRejectsANameThatIsOnlyAStrictPrefixOfADeclaredName() throws {
        let start = try automaton([weather, timeTool])
        // `get` is the start of `get_weather`; the closing quote there is not a name.
        let text = #"{"name": "get""#
        XCTAssertEqual(firstRejection(text, in: start), text.utf8.count - 1)
    }

    func testRejectsAnotherToolsArgumentsUnderTheChosenName() throws {
        let start = try automaton([weather, timeTool])
        // Valid for lookup_time, but the name chosen is get_weather: `"zone"` is not a get_weather key.
        let text = #"{"name": "get_weather", "arguments": {"zone": "UTC"}}"#
        let rejected = try XCTUnwrap(firstRejection(text, in: start))
        XCTAssertEqual(rejected, #"{"name": "get_weather", "arguments": {"z"#.utf8.count - 1)
        // Control: the same arguments under their own tool are fine.
        XCTAssertTrue(accepts(#"{"name": "lookup_time", "arguments": {"zone": "UTC"}}"#, in: start))
    }

    func testRejectsAnyByteAfterTheClosingBrace() throws {
        let start = try automaton([weather])
        let document = #"{"name": "get_weather", "arguments": {"city": "Paris"}}"#
        for extra in ["}", " ", "\n", "x", "{", "\""] {
            XCTAssertEqual(firstRejection(document + extra, in: start), document.utf8.count, extra)
        }
    }

    func testRejectsAlteredFraming() throws {
        let start = try automaton([weather])
        let args = #"{"city": "Paris"}"#
        // (longest valid prefix, rest): the first rejected byte is the one right after the prefix.
        let head = #"{"name": "get_weather""#
        let variants: [(valid: String, bad: String)] = [
            (#"{"name":"#, #""get_weather", "arguments": \#(args)}"#),  // no space after the colon
            (#"{"#, #" "name": "get_weather", "arguments": \#(args)}"#),  // space after the brace
            (head + ",", #""arguments": \#(args)}"#),  // no space after the comma
            (head + #", "arguments":"#, args + "}"),  // no space before the arguments
            (#"{""#, #"arguments": \#(args), "name": "get_weather"}"#),  // reordered keys
            (head + #", "arguments": "# + args, " }"),  // space before the outer brace
        ]
        for variant in variants {
            XCTAssertEqual(
                firstRejection(variant.valid + variant.bad, in: start), variant.valid.utf8.count,
                variant.valid + variant.bad)
        }
    }

    func testOuterBraceIsRefusedUntilTheInnerObjectIsComplete() throws {
        let start = try automaton([weather])
        XCTAssertNotNil(firstRejection(#"{"name": "get_weather", "arguments": }"#, in: start))
        var open = start
        for byte in #"{"name": "get_weather", "arguments": {"city": "Paris""#.utf8 { XCTAssertTrue(open.advance(byte: byte)) }
        XCTAssertFalse(open.isComplete)
        XCTAssertNotNil(open.tryAdvanceForWalk(byte: UInt8(ascii: "}")))  // inner close
        XCTAssertFalse(open.isComplete, "inner close alone is not the end of the document")
        XCTAssertNotNil(open.tryAdvanceForWalk(byte: UInt8(ascii: "}")))  // outer close
        XCTAssertTrue(open.isComplete)
    }

    func testInnerWhitespaceFollowsTheInnerAutomatonRules() throws {
        let start = try automaton([weather])
        let pretty = "{\n  \"city\": \"Paris\"\n}"
        var inner = JSONSchemaAutomaton(root: try union([weather]).formats[0].root)
        for byte in pretty.utf8 { XCTAssertTrue(inner.advance(byte: byte), "inner reference") }
        XCTAssertTrue(accepts(#"{"name": "get_weather", "arguments": "# + pretty + "}", in: start))
    }

    func testWalkUndoRestoresTheExactState() throws {
        var current = try automaton([weather, timeTool])
        for byte in #"{"name": "lookup_time", "arguments": {"z"#.utf8 { XCTAssertTrue(current.advance(byte: byte)) }
        let before = current.maskCacheKey(maxTokenBytes: 8)
        let undo = try XCTUnwrap(current.tryAdvanceForWalk(byte: UInt8(ascii: "o")))
        XCTAssertNotEqual(current.maskCacheKey(maxTokenBytes: 8), before)
        current.undoForWalk(undo)
        XCTAssertEqual(current.maskCacheKey(maxTokenBytes: 8), before)
        // A failed trial leaves the state untouched too.
        XCTAssertNil(current.tryAdvanceForWalk(byte: UInt8(ascii: "}")))
        XCTAssertEqual(current.maskCacheKey(maxTokenBytes: 8), before)
        XCTAssertTrue(accepts(#"{"name": "lookup_time", "arguments": {"zone": "UTC"}}"#, in: try automaton([weather, timeTool])))
    }

    // MARK: - parse

    func testParseSplitsNameArgumentsAndCompletion() throws {
        let format = try union([weather, timeTool])
        let whole = try XCTUnwrap(
            format.parse(generated: #"{"name": "lookup_time", "arguments": {"zone": "UTC"}}"#))
        XCTAssertEqual(whole.name, "lookup_time")
        XCTAssertEqual(whole.toolIndex, 1)
        XCTAssertEqual(whole.arguments, #"{"zone": "UTC"}"#)
        XCTAssertTrue(whole.isComplete)

        let partial = try XCTUnwrap(format.parse(generated: #"{"name": "get_weather", "arguments": {"ci"#))
        XCTAssertEqual(partial.name, "get_weather")
        XCTAssertEqual(partial.arguments, #"{"ci"#)
        XCTAssertFalse(partial.isComplete)

        let trailing = try XCTUnwrap(format.parse(generated: #"{"name": "get_weather", "arguments": {"#+"\n  "))
        XCTAssertEqual(trailing.arguments, "{", "trailing inner whitespace is stripped")

        // An inner close without the outer brace keeps its own `}` as arguments.
        let innerClosed = try XCTUnwrap(
            format.parse(generated: #"{"name": "get_weather", "arguments": {"city": "P"}"#))
        XCTAssertEqual(innerClosed.arguments, #"{"city": "P"}"#)
        XCTAssertFalse(innerClosed.isComplete)

        let nothingYet = try XCTUnwrap(format.parse(generated: #"{"name": "get_weather", "arguments": "#))
        XCTAssertEqual(nothingYet.arguments, "")
    }

    func testParseReturnsNilWhenTheNameWasNotCompleted() throws {
        let format = try union([weather, timeTool])
        for text in ["", #"{"na"#, #"{"name": ""#, #"{"name": "get_wea"#, #"{"name": "get_weather"#, #"{"name": "get_weather", "arg"#] {
            XCTAssertNil(format.parse(generated: text), text)
        }
    }

    // MARK: - AC2: token mask over a toy vocab (shared trie)

    /// Every byte (so any text tokenizes), multi-byte tokens spanning every phase boundary, EOS, banned.
    private static let curated: [String] = [
        #"{""#, #"{"name": ""#, #"": "#, #"", "#, #"", "arguments": "#, #"arguments"#, #"": {"#,
        "get_weather", "get_", "lookup_time", "ping", "}}", "},", "{}", "{}}", #"""}"#, #""}}"#, #"":"#,
        #""zone""#, #""city""#, #""zone": "#, "UTC", "Paris", "\n", "  ",
    ]
    private static let eosID: Int = 256 + curated.count
    private static let bannedID: Int = eosID + 1
    private static let classifications: [TokenByteClassification] = {
        var out: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }
        out += curated.map { .bytes(Array($0.utf8)) }
        out.append(.eos)
        out.append(.banned)
        return out
    }()

    private func tokens(_ text: String) -> [Int] {
        // Longest match first; single bytes always exist.
        let bytes = Array(text.utf8)
        var ids: [Int] = []
        var index = 0
        while index < bytes.count {
            var best: (id: Int, length: Int)?
            for (id, classification) in Self.classifications.enumerated() {
                guard case .bytes(let candidate) = classification, !candidate.isEmpty,
                    index + candidate.count <= bytes.count,
                    Array(bytes[index..<(index + candidate.count)]) == candidate,
                    candidate.count > (best?.length ?? 0)
                else { continue }
                best = (id, candidate.count)
            }
            ids.append(best!.id)
            index += best!.length
        }
        return ids
    }

    private func reference(_ automaton: ToolUnionAutomaton) -> Set<Int> {
        var allowed = Set<Int>()
        for (id, classification) in Self.classifications.enumerated() {
            switch classification {
            case .eos:
                if automaton.isComplete { allowed.insert(id) }
            case .banned:
                break
            case .bytes(let bytes):
                var copy = automaton
                if bytes.allSatisfy({ copy.advance(byte: $0) }) { allowed.insert(id) }
            }
        }
        return allowed
    }

    private func allowed(_ constraint: ToolUnionTokenConstraint) throws -> Set<Int> {
        Set(try constraint.allowedTokenIds())
    }

    private func advance(_ constraint: inout ToolUnionTokenConstraint, _ automaton: inout ToolUnionAutomaton, _ text: String) throws {
        for byte in text.utf8 { XCTAssertTrue(automaton.advance(byte: byte), text) }
        for id in tokens(text) { try constraint.advance(token: id) }
    }

    func testMaskMatchesPerTokenReferenceAlongAWholeDocument() throws {
        let format = try union([weather, timeTool, bare])
        let table = JSONSchemaConstraintTable(classifications: Self.classifications)
        let document = #"{"name": "lookup_time", "arguments": {"zone": "UTC"}}"#
        var constraint = ToolUnionTokenConstraint(table: table, format: format)
        var reference = ToolUnionAutomaton(format: format)
        XCTAssertEqual(try allowed(constraint), self.reference(reference))
        // Walk byte by byte AND with straddling tokens: same ids either way.
        for byte in document.utf8 {
            XCTAssertTrue(reference.advance(byte: byte))
            try constraint.advance(token: Int(byte))
            XCTAssertEqual(try allowed(constraint), self.reference(reference), "after \(UnicodeScalar(byte))")
            XCTAssertFalse(try allowed(constraint).isEmpty)
        }
        XCTAssertTrue(constraint.isComplete)
        XCTAssertEqual(try allowed(constraint), [Self.eosID], "only EOS remains after the final brace")
    }

    func testTokenStraddlingTheInnerCloseAndOuterBraceIsAllowedExactlyWhenTheInnerCanClose() throws {
        let format = try union([weather, bare])
        let table = JSONSchemaConstraintTable(classifications: Self.classifications)
        let doubleBrace = tokens("}}")
        XCTAssertEqual(doubleBrace.count, 1, "`}}` must be one toy token")

        // Inner object `{}` for a tool with no properties: `}}` closes inner then outer.
        var constraint = ToolUnionTokenConstraint(table: table, format: format)
        var shadow = ToolUnionAutomaton(format: format)
        try advance(&constraint, &shadow, #"{"name": "ping", "arguments": {"#)
        XCTAssertTrue(try allowed(constraint).contains(doubleBrace[0]), "{ + }} closes inner and outer")

        // Inner cannot close yet (the required `city` is missing): `}}` is not allowed.
        constraint = ToolUnionTokenConstraint(table: table, format: format)
        shadow = ToolUnionAutomaton(format: format)
        try advance(&constraint, &shadow, #"{"name": "get_weather", "arguments": {"#)
        XCTAssertFalse(try allowed(constraint).contains(doubleBrace[0]))
        XCTAssertFalse(try allowed(constraint).contains(Int(UInt8(ascii: "}"))))

        // Inner can close after the property: `}}` allowed; once the inner is closed `}}` is too long.
        constraint = ToolUnionTokenConstraint(table: table, format: format)
        shadow = ToolUnionAutomaton(format: format)
        try advance(&constraint, &shadow, #"{"name": "get_weather", "arguments": {"city": "Paris""#)
        XCTAssertTrue(try allowed(constraint).contains(doubleBrace[0]))
        try advance(&constraint, &shadow, "}")
        XCTAssertFalse(try allowed(constraint).contains(doubleBrace[0]), "outer + extra brace")
        XCTAssertTrue(try allowed(constraint).contains(Int(UInt8(ascii: "}"))))
        XCTAssertFalse(constraint.isComplete)
    }

    func testTokensStraddlingNameAndFrameBoundariesFollowTheirBytes() throws {
        let format = try union([weather, timeTool])
        let table = JSONSchemaConstraintTable(classifications: Self.classifications)
        var constraint = ToolUnionTokenConstraint(table: table, format: format)
        var shadow = ToolUnionAutomaton(format: format)
        XCTAssertTrue(try allowed(constraint).contains(tokens(#"{"name": ""#)[0]))
        try advance(&constraint, &shadow, #"{"name": ""#)
        let ids = try allowed(constraint)
        XCTAssertTrue(ids.contains(tokens("get_weather")[0]))
        XCTAssertTrue(ids.contains(tokens("lookup_time")[0]))
        XCTAssertFalse(ids.contains(tokens("ping")[0]), "undeclared name")
        XCTAssertFalse(ids.contains(tokens(#"", ""#)[0]), "empty name")
        try advance(&constraint, &shadow, "get_weather")
        // The closing quote plus the middle literal in one token is allowed; the wrong frame is not.
        XCTAssertTrue(try allowed(constraint).contains(tokens(#"", "arguments": "#)[0]))
        XCTAssertFalse(try allowed(constraint).contains(tokens(#"": {"#)[0]))
        // Strict prefix `get_`: the closing quote is not allowed.
        constraint = ToolUnionTokenConstraint(table: table, format: format)
        shadow = ToolUnionAutomaton(format: format)
        try advance(&constraint, &shadow, #"{"name": "get_"#)
        XCTAssertFalse(try allowed(constraint).contains(Int(UInt8(ascii: "\""))))
    }

    func testEOSIsAllowedOnlyWhenCompleteAndBannedAlwaysThrows() throws {
        let format = try union([weather])
        let table = JSONSchemaConstraintTable(classifications: Self.classifications)
        var constraint = ToolUnionTokenConstraint(table: table, format: format)
        var shadow = ToolUnionAutomaton(format: format)
        for prefix in ["", #"{"name": "get_weather", "arguments": "#, #"{"city": "Paris""#, "}"] {
            try advance(&constraint, &shadow, prefix)
            XCTAssertFalse(try allowed(constraint).contains(Self.eosID), prefix)
            XCTAssertThrowsError(try { var copy = constraint; try copy.advance(token: Self.eosID) }()) { error in
                XCTAssertEqual(error as? JSONSchemaConstraintError, .tokenDisallowed(Self.eosID))
            }
        }
        XCTAssertThrowsError(try { var copy = constraint; try copy.advance(token: Self.bannedID) }())
        XCTAssertThrowsError(try { var copy = constraint; try copy.advance(token: Int(UInt8(ascii: "x"))) }())
        try advance(&constraint, &shadow, "}")
        XCTAssertTrue(constraint.isComplete)
        XCTAssertTrue(try allowed(constraint).contains(Self.eosID))
        try constraint.advance(token: Self.eosID)
    }

    func testCacheKeysOfTwoToolsAtTheSameInnerStateNeverCollide() throws {
        // `get_weather` and `lookup_time` both sit just inside `{` of their arguments object here: the
        // inner states have identical shape (their node indices overlap), only the tool differs.
        let format = try union([weather, timeTool])
        var a = ToolUnionAutomaton(format: format)
        var b = ToolUnionAutomaton(format: format)
        for byte in #"{"name": "get_weather", "arguments": {"#.utf8 { XCTAssertTrue(a.advance(byte: byte)) }
        for byte in #"{"name": "lookup_time", "arguments": {"#.utf8 { XCTAssertTrue(b.advance(byte: byte)) }
        XCTAssertNotEqual(a.maskCacheKey(maxTokenBytes: 8), b.maskCacheKey(maxTokenBytes: 8))

        // Through ONE shared table, in both orders: a cache hit never leaks tool A's mask into tool B.
        let quote = UInt8(ascii: "\"")
        for order in [[0, 1], [1, 0]] {
            let table = JSONSchemaConstraintTable(classifications: Self.classifications)
            var masks: [Set<Int>] = [[], []]
            for which in order {
                var constraint = ToolUnionTokenConstraint(table: table, format: format)
                var shadow = ToolUnionAutomaton(format: format)
                let name = which == 0 ? "get_weather" : "lookup_time"
                try advance(&constraint, &shadow, #"{"name": "\#(name)", "arguments": {"#)
                masks[which] = try allowed(constraint)
                XCTAssertEqual(masks[which], reference(shadow))
            }
            XCTAssertNotEqual(masks[0], masks[1])
            XCTAssertTrue(masks[0].contains(Int(quote)))
        }
    }

    func testRequestsWithTheSameToolListShareCacheEntriesAndDifferentListsDoNot() throws {
        let format = try union([weather, timeTool])
        let same = try union([weather, timeTool])
        XCTAssertEqual(format.fingerprint, same.fingerprint)
        let other = try union([weather, bare])
        XCTAssertNotEqual(format.fingerprint, other.fingerprint)
        XCTAssertNotEqual(format.fingerprint, try union([timeTool, weather]).fingerprint, "order is part of identity")

        let table = JSONSchemaConstraintTable(classifications: Self.classifications)
        let first = ToolUnionTokenConstraint(table: table, format: format)
        _ = try first.allowedTokenBitset()
        let misses = first.cacheMissCount
        let second = ToolUnionTokenConstraint(table: table, format: same)
        _ = try second.allowedTokenBitset()
        XCTAssertEqual(second.cacheMissCount, misses, "same tool list: a hit, not a new entry")
        XCTAssertGreaterThan(second.cacheHitCount, 0)

        let third = ToolUnionTokenConstraint(table: table, format: other)
        _ = try third.allowedTokenBitset()
        XCTAssertEqual(third.cacheMissCount, misses + 1, "different tool list: its own entry")
    }

    // MARK: - AC5 (format side): refusals

    private func assertRefusedOnToolChoice(
        _ tools: [String], file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let declared = try specs(tools)
        XCTAssertThrowsError(try declared.requiredToolUnionFormat(), file: file, line: line) { error in
            guard case OpenAIServingError.invalidRequest(_, let param) = error else {
                return XCTFail("expected invalidRequest, got \(error)", file: file, line: line)
            }
            XCTAssertEqual(param, "tool_choice", file: file, line: line)
        }
    }

    func testAnUnsupportedSchemaOnAnyDeclaredToolIsRefused() throws {
        let patterned =
            #"{"type":"function","function":{"name":"bad","parameters":{"type":"object","properties":{"a":{"type":"string","pattern":"^x"}}}}}"#
        try assertRefusedOnToolChoice([patterned, weather])
        try assertRefusedOnToolChoice([weather, patterned])  // not just the first
        try assertRefusedOnToolChoice([weather, timeTool, patterned])
        let notObject = #"{"type":"function","function":{"name":"bad","parameters":{"type":"string"}}}"#
        try assertRefusedOnToolChoice([weather, notObject])
    }

    func testANameWithAnyByteOutsideTheOpenAIGrammarIsRefused() throws {
        for bad in ["get weather", "get.weather", "get\\\"weather", "wé", "a/b", "a:b"] {
            try assertRefusedOnToolChoice([weather, tool(bad)])
        }
        // Control: the full grammar is accepted.
        let ok = try union([weather, tool("Az_09-x")])
        XCTAssertEqual(ok.names, ["get_weather", "Az_09-x"])
    }

    func testDuplicateNamesAreRefused() throws {
        try assertRefusedOnToolChoice([weather, timeTool, weather])
    }

    func testEveryToolIsCompiledInDeclarationOrder() throws {
        let format = try union([timeTool, weather, bare])
        XCTAssertEqual(format.names, ["lookup_time", "get_weather", "ping"])
        XCTAssertEqual(format.formats.count, 3)
    }
}
