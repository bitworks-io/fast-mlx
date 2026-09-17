import XCTest

@testable import ServingCore

/// Stage 2a acceptance tests (see
/// `docs/task-inbox/2026-09-17-DECISION-response-format-json-schema-subset.md`): the
/// `JSONSchemaSubsetCompiler` (compile-to-IR), `response_format: {"type":"json_schema", ...}`
/// request decoding, and the exhaustive capability gate that keeps `.jsonSchema` refused until the
/// scalar route is wired (stage 3). The frozen IR itself (`JSONSchemaNode` et al.) lives in
/// `JSONSchemaResponseFormat.swift` and is NOT modified by this stage.
final class ResponseFormatJSONSchemaRequestTests: XCTestCase {

    // MARK: - Compiler helpers

    private func compile(
        _ schemaJSON: String, name: String = "n", strict: Bool = false
    ) throws -> JSONSchemaResponseFormat {
        let schema = try JSONSchemaSubsetCompiler.parseOrdered(Data(schemaJSON.utf8))
        return try JSONSchemaSubsetCompiler.compile(name: name, schema: schema, strict: strict)
    }

    private func expectCompileError(
        _ schemaJSON: String, name: String = "n", strict: Bool = false,
        file: StaticString = #filePath, line: UInt = #line,
        _ assertions: (JSONSchemaCompileError) -> Void
    ) {
        XCTAssertThrowsError(
            try compile(schemaJSON, name: name, strict: strict), file: file, line: line
        ) { error in
            guard let compileError = error as? JSONSchemaCompileError else {
                return XCTFail("expected JSONSchemaCompileError, got \(error)", file: file, line: line)
            }
            assertions(compileError)
        }
    }

    // MARK: - Ordered parser nesting bound

    /// The ordered parser re-walks the whole request body, so nesting must be bounded (refused, not
    /// a stack overflow). 256 levels parse; 257 are refused.
    func testOrderedParserBoundsNestingDepth() throws {
        let atLimit = String(repeating: "[", count: 256) + String(repeating: "]", count: 256)
        XCTAssertNoThrow(try JSONSchemaSubsetCompiler.parseOrdered(Data(atLimit.utf8)))
        let overLimit = String(repeating: "[", count: 257) + String(repeating: "]", count: 257)
        XCTAssertThrowsError(try JSONSchemaSubsetCompiler.parseOrdered(Data(overLimit.utf8))) { error in
            XCTAssertTrue(
                (error as? JSONSchemaCompileError)?.message.contains("nesting") ?? false, "\(error)")
        }
        let deepObjects =
            String(repeating: #"{"a":"#, count: 300) + "1" + String(repeating: "}", count: 300)
        XCTAssertThrowsError(try JSONSchemaSubsetCompiler.parseOrdered(Data(deepObjects.utf8)))
    }

    // MARK: - Accepted subset: exact compiled IR

    func testCompilesEachScalarType() throws {
        XCTAssertEqual(try compile(#"{"type":"string"}"#).root, .string)
        XCTAssertEqual(try compile(#"{"type":"number"}"#).root, .number)
        XCTAssertEqual(try compile(#"{"type":"integer"}"#).root, .integer)
        XCTAssertEqual(try compile(#"{"type":"boolean"}"#).root, .boolean)
        XCTAssertEqual(try compile(#"{"type":"null"}"#).root, .null)
    }

    func testCompilesNullableType() throws {
        let format = try compile(#"{"type":["string","null"]}"#)
        XCTAssertEqual(format.root, .anyOf([.string, .null]))
    }

    func testCompilesNullableTypeRegardlessOfArrayOrder() throws {
        // "null" listed first must produce the identical IR to "null" listed second.
        let format = try compile(#"{"type":["null","integer"]}"#)
        XCTAssertEqual(format.root, .anyOf([.integer, .null]))
    }

    func testCompilesObjectWithPropertiesRequiredAdditionalPropertiesFalsePreservingDeclaredOrder() throws {
        let format = try compile(
            #"""
            {"type":"object","properties":{"zebra":{"type":"string"},"apple":{"type":"integer"}},
             "required":["zebra"],"additionalProperties":false}
            """#)
        XCTAssertEqual(
            format.root,
            .object([
                JSONSchemaProperty(name: "zebra", required: true, value: .string),
                JSONSchemaProperty(name: "apple", required: false, value: .integer),
            ]))
    }

    func testCompilesArrayItems() throws {
        let format = try compile(#"{"type":"array","items":{"type":"boolean"}}"#)
        XCTAssertEqual(format.root, .array(items: .boolean))
    }

    func testCompilesEnumOfScalarLiterals() throws {
        let format = try compile(#"{"enum":["red","green",3,true,null]}"#)
        XCTAssertEqual(
            format.root,
            .enumeration([
                JSONSchemaLiteral(jsonText: Array("\"red\"".utf8)),
                JSONSchemaLiteral(jsonText: Array("\"green\"".utf8)),
                JSONSchemaLiteral(jsonText: Array("3".utf8)),
                JSONSchemaLiteral(jsonText: Array("true".utf8)),
                JSONSchemaLiteral(jsonText: Array("null".utf8)),
            ]))
    }

    func testCompilesNesting() throws {
        let format = try compile(
            #"""
            {"type":"object","properties":{"items":{"type":"array","items":
              {"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],
               "additionalProperties":false}}},"required":["items"],"additionalProperties":false}
            """#)
        XCTAssertEqual(
            format.root,
            .object([
                JSONSchemaProperty(
                    name: "items", required: true,
                    value: .array(
                        items: .object([
                            JSONSchemaProperty(name: "id", required: true, value: .integer)
                        ])))
            ]))
    }

    func testCompilesAnyOfDisjointBranches() throws {
        let format = try compile(#"{"anyOf":[{"type":"string"},{"type":"integer"}]}"#)
        XCTAssertEqual(format.root, .anyOf([.string, .integer]))
    }

    func testCompilesLocalNonRecursiveRefToDefs() throws {
        let format = try compile(
            #"""
            {"$defs":{"Point":{"type":"object","properties":{"x":{"type":"integer"}},
              "required":["x"],"additionalProperties":false}},
             "type":"object","properties":{"p":{"$ref":"#/$defs/Point"}},
             "required":["p"],"additionalProperties":false}
            """#)
        XCTAssertEqual(
            format.root,
            .object([
                JSONSchemaProperty(
                    name: "p", required: true,
                    value: .object([
                        JSONSchemaProperty(name: "x", required: true, value: .integer)
                    ]))
            ]))
    }

    func testCompilesLocalNonRecursiveRefToDefinitions() throws {
        let format = try compile(
            #"""
            {"definitions":{"Point":{"type":"string"}},
             "type":"object","properties":{"p":{"$ref":"#/definitions/Point"}},
             "required":["p"],"additionalProperties":false}
            """#)
        XCTAssertEqual(
            format.root,
            .object([JSONSchemaProperty(name: "p", required: true, value: .string)]))
    }

    // MARK: - Ignored annotations

    func testIgnoredAnnotationsCompileIdenticallyToWithoutThem() throws {
        let plain = try compile(
            #"""
            {"type":"object","properties":{"a":{"type":"string"}},"required":["a"],
             "additionalProperties":false}
            """#)
        let annotated = try compile(
            #"""
            {"$schema":"https://json-schema.org/draft/2020-12/schema","title":"T","description":"D",
             "type":"object","properties":{"a":{"type":"string","title":"A","description":"the a",
             "default":"x","examples":["x","y"]}},"required":["a"],"additionalProperties":false,
             "default":{},"examples":[{"a":"x"}]}
            """#)
        XCTAssertEqual(plain.root, annotated.root)
    }

    // MARK: - Refusals: JSON-pointer path per unsupported keyword

    func testRefusesPattern() {
        expectCompileError(#"{"type":"string","pattern":"^a"}"#) { error in
            XCTAssertEqual(error.path, "")
        }
    }

    func testRefusesFormat() {
        expectCompileError(#"{"type":"string","format":"date-time"}"#) { error in
            XCTAssertEqual(error.path, "")
        }
    }

    func testRefusesMinLength() {
        expectCompileError(#"{"type":"string","minLength":1}"#) { error in
            XCTAssertEqual(error.path, "")
        }
    }

    func testRefusesMaximum() {
        expectCompileError(#"{"type":"integer","maximum":10}"#) { error in
            XCTAssertEqual(error.path, "")
        }
    }

    func testRefusesOneOf() {
        expectCompileError(#"{"oneOf":[{"type":"string"},{"type":"integer"}]}"#) { error in
            XCTAssertEqual(error.path, "")
        }
    }

    func testRefusesAllOf() {
        expectCompileError(#"{"allOf":[{"type":"string"}]}"#) { error in
            XCTAssertEqual(error.path, "")
        }
    }

    func testRefusesNot() {
        expectCompileError(#"{"not":{"type":"string"}}"#) { error in
            XCTAssertEqual(error.path, "")
        }
    }

    func testRefusesConst() {
        expectCompileError(#"{"const":"x"}"#) { error in
            XCTAssertEqual(error.path, "")
        }
    }

    func testRefusesUntypedNode() {
        expectCompileError(#"{}"#) { error in
            XCTAssertEqual(error.path, "")
        }
    }

    func testRefusesRecursiveRef() {
        expectCompileError(
            #"""
            {"$defs":{"Node":{"type":"object","properties":{"next":{"$ref":"#/$defs/Node"}},
              "required":[],"additionalProperties":false}},
             "$ref":"#/$defs/Node"}
            """#
        ) { error in
            XCTAssertEqual(error.path, "/properties/next/$ref")
        }
    }

    func testRefusesAdditionalPropertiesTrue() {
        expectCompileError(#"{"type":"object","properties":{},"additionalProperties":true}"#) { error in
            XCTAssertEqual(error.path, "/additionalProperties")
        }
    }

    func testRefusesAdditionalPropertiesAsSchema() {
        expectCompileError(
            #"{"type":"object","properties":{},"additionalProperties":{"type":"string"}}"#
        ) { error in
            XCTAssertEqual(error.path, "/additionalProperties")
        }
    }

    func testRefusesAnyOfWithOverlappingFirstBytesTwoObjectBranches() {
        expectCompileError(
            #"""
            {"anyOf":[{"type":"object","properties":{},"additionalProperties":false},
                      {"type":"object","properties":{"b":{"type":"string"}},"required":["b"],
                       "additionalProperties":false}]}
            """#
        ) { error in
            XCTAssertEqual(error.path, "/anyOf")
        }
    }

    func testRefusesAnyOfWithOverlappingFirstBytesIntegerAndNumber() {
        expectCompileError(#"{"anyOf":[{"type":"integer"},{"type":"number"}]}"#) { error in
            XCTAssertEqual(error.path, "/anyOf")
        }
    }

    func testRefusesEnumStringContainingDoubleQuote() {
        expectCompileError(#"{"enum":["a\"b"]}"#) { error in
            XCTAssertEqual(error.path, "/enum/0")
        }
    }

    func testRefusesEnumStringContainingBackslash() {
        expectCompileError(#"{"enum":["a\\b"]}"#) { error in
            XCTAssertEqual(error.path, "/enum/0")
        }
    }

    /// Precise nested-path assertion (required example from the task spec): an unsupported keyword
    /// on an array's `items` schema, nested inside an object property, names the exact JSON pointer
    /// to the offending node — not just the offending keyword's own name.
    func testUnsupportedKeywordInNestedArrayItemsNamesPreciseNestedPath() {
        expectCompileError(
            #"{"type":"object","properties":{"a":{"type":"array","items":{"pattern":"^x"}}}}"#
        ) { error in
            XCTAssertEqual(error.path, "/properties/a/items")
        }
    }

    // MARK: - `strict: true`

    func testStrictRefusesObjectMissingAdditionalPropertiesFalse() {
        expectCompileError(
            #"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#, strict: true
        ) { error in
            XCTAssertEqual(error.path, "")
        }
    }

    func testStrictRefusesPropertyNotListedInRequired() {
        expectCompileError(
            #"""
            {"type":"object","properties":{"a":{"type":"string"}},"required":[],
             "additionalProperties":false}
            """#, strict: true
        ) { error in
            XCTAssertEqual(error.path, "/required")
        }
    }

    func testStrictAcceptsConformingSchema() throws {
        let format = try compile(
            #"""
            {"type":"object","properties":{"a":{"type":"string"}},"required":["a"],
             "additionalProperties":false}
            """#, strict: true)
        XCTAssertEqual(
            format.root, .object([JSONSchemaProperty(name: "a", required: true, value: .string)]))
        XCTAssertTrue(format.strict)
    }

    // MARK: - Limits

    func testExceedsMaxDepthIsRefused() {
        // 40 levels of `{"type":"object","properties":{"a": <nested>},"additionalProperties":false}`
        // — well past `JSONSchemaLimits.maxDepth` (32) and well under `maxNodes` (512).
        var schema = #"{"type":"string"}"#
        for _ in 0..<40 {
            schema =
                #"{"type":"object","properties":{"a":\#(schema)},"additionalProperties":false}"#
        }
        expectCompileError(schema) { error in
            XCTAssertTrue(error.message.contains("maxDepth"), "message was: \(error.message)")
        }
    }

    func testExceedsMaxPropertiesPerObjectIsRefused() {
        var propertyPairs: [String] = []
        for index in 0...JSONSchemaLimits.maxPropertiesPerObject {  // one more than the limit
            propertyPairs.append(#""p\#(index)":{"type":"string"}"#)
        }
        let schema =
            #"{"type":"object","properties":{\#(propertyPairs.joined(separator: ","))},"#
            + #""additionalProperties":false}"#
        expectCompileError(schema) { error in
            XCTAssertTrue(
                error.message.contains("maxPropertiesPerObject"), "message was: \(error.message)")
        }
    }

    // MARK: - Request decoding

    private func decode(_ body: String) throws -> OpenAIChatCompletionRequest {
        try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
    }

    private func jsonSchemaBody(
        name: String = "\"weather\"", schema: String = #"{"type":"string"}"#, strict: String? = nil
    ) -> String {
        let strictField = strict.map { #","strict":\#($0)"# } ?? ""
        return #"""
            {"model":"m","messages":[{"role":"user","content":"Hi"}],
             "response_format":{"type":"json_schema","json_schema":{"name":\#(name),"schema":\#(schema)\#(strictField)}}}
            """#
    }

    func testDecodesToJSONSchemaCaseWithCompiledRootAndFingerprint() throws {
        let request = try decode(jsonSchemaBody())
        guard case .jsonSchema(let compiled)? = request.responseFormat else {
            return XCTFail("expected .jsonSchema, got \(String(describing: request.responseFormat))")
        }
        XCTAssertEqual(compiled.name, "weather")
        XCTAssertEqual(compiled.root, .string)
        XCTAssertFalse(compiled.strict)
        XCTAssertEqual(compiled.fingerprint.count, 64)
        XCTAssertTrue(compiled.fingerprint.allSatisfy(\.isHexDigit))
        XCTAssertEqual(compiled.fingerprint, compiled.fingerprint.lowercased())
        XCTAssertFalse(request.ignoredFields.contains("response_format"))
    }

    func testStrictAbsentDefaultsFalse() throws {
        let request = try decode(jsonSchemaBody())
        guard case .jsonSchema(let compiled)? = request.responseFormat else {
            return XCTFail("expected .jsonSchema")
        }
        XCTAssertFalse(compiled.strict)
    }

    func testStrictTrueIsThreadedThrough() throws {
        let request = try decode(
            jsonSchemaBody(
                schema: #"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}"#,
                strict: "true"))
        guard case .jsonSchema(let compiled)? = request.responseFormat else {
            return XCTFail("expected .jsonSchema")
        }
        XCTAssertTrue(compiled.strict)
    }

    func testEmptyNameIsRejected() {
        XCTAssertThrowsError(try decode(jsonSchemaBody(name: "\"\""))) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(_, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "response_format")
        }
    }

    func testTooLongNameIsRejected() {
        let name = String(repeating: "a", count: 65)
        XCTAssertThrowsError(try decode(jsonSchemaBody(name: "\"\(name)\""))) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(_, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "response_format")
        }
    }

    func testNameWithInvalidCharacterIsRejected() {
        XCTAssertThrowsError(try decode(jsonSchemaBody(name: "\"has space\""))) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(_, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "response_format")
        }
    }

    func testMissingSchemaIsRejected() {
        XCTAssertThrowsError(
            try decode(
                #"""
                {"model":"m","messages":[{"role":"user","content":"Hi"}],
                 "response_format":{"type":"json_schema","json_schema":{"name":"x"}}}
                """#)
        ) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(_, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "response_format")
        }
    }

    /// NON-properties key reordering (the root object's own `type`/`properties` sibling order) must
    /// still share a fingerprint: `properties`' OWN declared order ("a" then "b") is unchanged between
    /// A and B here, only the sibling key position of `type` vs. `properties` in the raw schema text
    /// moves — a difference the compiled `JSONSchemaNode` tree (what the fingerprint now hashes, see
    /// `JSONSchemaFingerprint.swift`) never observes. This is DELIBERATELY narrower than the old
    /// version of this test (which also swapped `properties`' own internal "a"/"b" order): swapping
    /// property declaration order is now a case that MUST differ — see
    /// `testFingerprintDiffersForDifferentPropertyDeclarationOrder` below.
    func testFingerprintIsIdenticalForSameSchemaWithDifferentKeyOrder() throws {
        let requestA = try decode(
            jsonSchemaBody(
                schema: #"{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"}}}"#))
        let requestB = try decode(
            jsonSchemaBody(
                schema: #"{"properties":{"a":{"type":"string"},"b":{"type":"integer"}},"type":"object"}"#))
        guard case .jsonSchema(let compiledA)? = requestA.responseFormat,
            case .jsonSchema(let compiledB)? = requestB.responseFormat
        else {
            return XCTFail("expected both requests to decode .jsonSchema")
        }
        XCTAssertEqual(compiledA.root, compiledB.root, "sanity: this pair must compile to the identical IR")
        XCTAssertEqual(compiledA.fingerprint, compiledB.fingerprint)
    }

    func testFingerprintDiffersForDifferentSchemas() throws {
        let requestA = try decode(jsonSchemaBody(schema: #"{"type":"string"}"#))
        let requestB = try decode(jsonSchemaBody(schema: #"{"type":"integer"}"#))
        guard case .jsonSchema(let compiledA)? = requestA.responseFormat,
            case .jsonSchema(let compiledB)? = requestB.responseFormat
        else {
            return XCTFail("expected both requests to decode .jsonSchema")
        }
        XCTAssertNotEqual(compiledA.fingerprint, compiledB.fingerprint)
    }

    /// The bug this stage fixes: two schemas identical except for `properties` DECLARATION order
    /// compile to structurally different `JSONSchemaNode.object` trees (the automaton enforces
    /// declared property order), and must therefore get DIFFERENT fingerprints — a sorted-keys
    /// fingerprint over the raw schema text (the old behavior) could not tell them apart, letting them
    /// wrongly share `JSONSchemaConstraintTable` mask-cache entries keyed by fingerprint.
    func testFingerprintDiffersForDifferentPropertyDeclarationOrder() throws {
        let requestA = try decode(
            jsonSchemaBody(
                schema: #"{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"}}}"#))
        let requestB = try decode(
            jsonSchemaBody(
                schema: #"{"type":"object","properties":{"b":{"type":"integer"},"a":{"type":"string"}}}"#))
        guard case .jsonSchema(let compiledA)? = requestA.responseFormat,
            case .jsonSchema(let compiledB)? = requestB.responseFormat
        else {
            return XCTFail("expected both requests to decode .jsonSchema")
        }
        XCTAssertNotEqual(
            compiledA.root, compiledB.root, "sanity: the compiled IR must actually differ in property order")
        XCTAssertNotEqual(compiledA.fingerprint, compiledB.fingerprint)
    }

    /// Raw schemas that compile to the IDENTICAL IR must share a fingerprint even when their surface
    /// syntax differs: pure annotations (`title`/`description`, ignored everywhere — see
    /// `JSONSchemaSubsetCompiler.ignoredAnnotationKeys`) never reach the compiled tree.
    func testFingerprintSameForRawSchemasThatCompileToIdenticalIRViaAnnotations() throws {
        let requestA = try decode(
            jsonSchemaBody(
                schema: #"""
                {"type":"object","title":"Foo","description":"bar",
                 "properties":{"a":{"type":"string","description":"the a field"}},
                 "required":["a"],"additionalProperties":false}
                """#,
                strict: "true"))
        let requestB = try decode(
            jsonSchemaBody(
                schema: #"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}"#,
                strict: "true"))
        guard case .jsonSchema(let compiledA)? = requestA.responseFormat,
            case .jsonSchema(let compiledB)? = requestB.responseFormat
        else {
            return XCTFail("expected both requests to decode .jsonSchema")
        }
        XCTAssertEqual(compiledA.root, compiledB.root)
        XCTAssertEqual(compiledA.fingerprint, compiledB.fingerprint)
    }

    /// Raw schemas that compile to the IDENTICAL IR must share a fingerprint even when one spells a
    /// sub-schema as an inlined `$ref` target and the other inlines it directly — `$ref` resolution
    /// happens entirely at compile time (`JSONSchemaSubsetCompiler.resolveRef`), so both reach the
    /// same `JSONSchemaNode`.
    func testFingerprintSameForRefVsInlinedEquivalentSchema() throws {
        let requestA = try decode(
            jsonSchemaBody(
                schema: #"""
                {"type":"object","properties":{"a":{"$ref":"#/$defs/Str"}},
                 "required":["a"],"additionalProperties":false,"$defs":{"Str":{"type":"string"}}}
                """#,
                strict: "true"))
        let requestB = try decode(
            jsonSchemaBody(
                schema: #"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}"#,
                strict: "true"))
        guard case .jsonSchema(let compiledA)? = requestA.responseFormat,
            case .jsonSchema(let compiledB)? = requestB.responseFormat
        else {
            return XCTFail("expected both requests to decode .jsonSchema")
        }
        XCTAssertEqual(compiledA.root, compiledB.root)
        XCTAssertEqual(compiledA.fingerprint, compiledB.fingerprint)
    }

    // MARK: - Canonical encoding injectivity spot checks (`JSONSchemaNode.canonicalEncoding()`)

    /// `enum(["ab"])` vs. `enum(["a","b"])`: without a literal-COUNT length prefix, `"ab"`'s single
    /// literal and `"a"`+`"b"`'s two literals could concatenate to related byte runs; the count prefix
    /// (1 vs. 2) makes them diverge before any literal byte is even compared — see
    /// `JSONSchemaNode.canonicalEncoding()`'s doc comment.
    func testCanonicalEncodingDistinguishesEnumLiteralSplitVsMerged() {
        let merged = JSONSchemaNode.enumeration([JSONSchemaLiteral(jsonText: Array(#""ab""#.utf8))])
        let split = JSONSchemaNode.enumeration([
            JSONSchemaLiteral(jsonText: Array(#""a""#.utf8)),
            JSONSchemaLiteral(jsonText: Array(#""b""#.utf8)),
        ])
        XCTAssertNotEqual(merged.canonicalEncoding(), split.canonicalEncoding())
    }

    /// Object property name `"a"` required vs. optional: same name bytes, only the fixed-width
    /// `required` byte differs.
    func testCanonicalEncodingDistinguishesRequiredVsOptionalProperty() {
        let required = JSONSchemaNode.object([JSONSchemaProperty(name: "a", required: true, value: .string)])
        let optional = JSONSchemaNode.object([JSONSchemaProperty(name: "a", required: false, value: .string)])
        XCTAssertNotEqual(required.canonicalEncoding(), optional.canonicalEncoding())
    }

    /// `array(items: .string)` vs. bare `.string`: distinguished by tag byte alone.
    func testCanonicalEncodingDistinguishesArrayOfStringVsString() {
        let arrayOfString = JSONSchemaNode.array(items: .string)
        XCTAssertNotEqual(arrayOfString.canonicalEncoding(), JSONSchemaNode.string.canonicalEncoding())
    }

    func testUnsupportedKeywordErrorMessageNamesThePath() {
        XCTAssertThrowsError(
            try decode(jsonSchemaBody(schema: #"{"type":"string","pattern":"^a"}"#))
        ) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(let message, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "response_format")
            XCTAssertTrue(
                message.hasPrefix("response_format.json_schema.schema"), "message was: \(message)")
        }
    }

    // MARK: - Fail-open guard: `.jsonSchema` refused wherever `.jsonObject` is gated

    /// Mirrors the HTTP-layer capability-gate tests in `OpenAIChatCompletionsHTTPHandlerTests`
    /// (`testJSONObjectResponseFormatIsRefusedByDefaultCapabilityFalseBackend`), scoped to the
    /// ServingCore-level gate itself: a backend that does not declare
    /// `supportsJSONSchemaResponseFormat` must have `.jsonSchema` refused, exactly like a backend
    /// that does not declare `supportsJSONObjectResponseFormat` has `.jsonObject` refused today.
    func testCapabilityGateRefusesJSONSchemaWhenBackendDoesNotSupportIt() throws {
        let request = try decode(jsonSchemaBody())
        XCTAssertThrowsError(
            try validateResponseFormatCapability(
                request: request,
                backendSupportsJSONObjectResponseFormat: false,
                backendSupportsJSONSchemaResponseFormat: false)
        ) { error in
            guard let servingError = error as? OpenAIServingError,
                case .invalidRequest(let message, let param) = servingError
            else {
                return XCTFail("expected invalidRequest, got \(error)")
            }
            XCTAssertEqual(param, "response_format")
            XCTAssertTrue(message.contains("json_schema"), "message was: \(message)")
        }
    }

    /// Even a backend that declares `supportsJSONObjectResponseFormat` (like today's
    /// `ScalarServingBackend`) must still refuse `.jsonSchema` — the two capabilities are
    /// independent flags, so opting into one must never silently opt into the other.
    func testCapabilityGateRefusesJSONSchemaEvenWhenBackendSupportsJSONObject() throws {
        let request = try decode(jsonSchemaBody())
        XCTAssertThrowsError(
            try validateResponseFormatCapability(
                request: request,
                backendSupportsJSONObjectResponseFormat: true,
                backendSupportsJSONSchemaResponseFormat: false))
    }

    /// Sanity check that the exhaustive `switch` did not invert the gate: once a backend DOES
    /// declare `supportsJSONSchemaResponseFormat`, the same request is admitted.
    func testCapabilityGateAdmitsJSONSchemaWhenBackendDeclaresSupport() throws {
        let request = try decode(jsonSchemaBody())
        XCTAssertNoThrow(
            try validateResponseFormatCapability(
                request: request,
                backendSupportsJSONObjectResponseFormat: false,
                backendSupportsJSONSchemaResponseFormat: true))
    }

    /// Control: the pre-existing `json_object` gate must still behave exactly as before the
    /// `switch` rewrite (regression guard for `validateResponseFormatCapability`'s refactor).
    func testCapabilityGateStillRefusesJSONObjectWhenUnsupported() throws {
        let request = try decode(
            #"""
            {"model":"m","messages":[{"role":"user","content":"Hi"}],
             "response_format":{"type":"json_object"}}
            """#)
        XCTAssertThrowsError(
            try validateResponseFormatCapability(
                request: request,
                backendSupportsJSONObjectResponseFormat: false,
                backendSupportsJSONSchemaResponseFormat: true))
    }

    /// Control: a request with no `response_format` at all must never be gated, regardless of
    /// either capability flag.
    func testCapabilityGateNeverThrowsWhenResponseFormatAbsent() throws {
        let request = try decode(#"{"model":"m","messages":[{"role":"user","content":"Hi"}]}"#)
        XCTAssertNoThrow(
            try validateResponseFormatCapability(
                request: request,
                backendSupportsJSONObjectResponseFormat: false,
                backendSupportsJSONSchemaResponseFormat: false))
    }
}
