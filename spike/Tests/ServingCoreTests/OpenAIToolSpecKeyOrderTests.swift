import XCTest

@testable import ServingCore

/// The chat template must see each tool in the CLIENT's JSON key order (as Python transformers
/// does), not the sorted order `JSONSerialization` + `Value(any:)` produce. Cycle 237.
final class OpenAIToolSpecKeyOrderTests: XCTestCase {
    private static let tool =
        #"{"type":"function","function":{"name":"search","description":"d","parameters":{"type":"object","properties":{"query":{"type":"string"},"opt":{"type":"string"}},"required":["query"]}}}"#

    private func keys(_ value: JSONSchemaSubsetCompiler.RawValue?) throws -> [String] {
        guard case .object(let pairs)? = value else {
            XCTFail("expected ordered object, got \(String(describing: value))")
            return []
        }
        return pairs.map { $0.key }
    }

    private func member(
        _ value: JSONSchemaSubsetCompiler.RawValue?, _ key: String
    ) throws -> JSONSchemaSubsetCompiler.RawValue {
        guard case .object(let pairs)? = value, let found = pairs.first(where: { $0.key == key }) else {
            XCTFail("missing key \(key)")
            return .null
        }
        return found.value
    }

    func testDecodeStrictKeepsToolKeyOrderForTemplate() throws {
        let body = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[\#(Self.tool)]}"#
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.tools.count, 1)
        let ordered = request.tools[0].orderedRaw
        XCTAssertEqual(try keys(ordered), ["type", "function"])
        let function = try member(ordered, "function")
        XCTAssertEqual(try keys(function), ["name", "description", "parameters"])
        let parameters = try member(function, "parameters")
        XCTAssertEqual(try keys(parameters), ["type", "properties", "required"])
        XCTAssertEqual(try keys(try member(parameters, "properties")), ["query", "opt"])
    }

    func testDuplicateKeyToolsBodyStillDecodesWithoutOrderedRaw() throws {
        let dupTool =
            #"{"type":"function","function":{"name":"search","name":"search","parameters":{"type":"object"}}}"#
        let body = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[\#(dupTool)]}"#
        let request = try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.tools.count, 1)
        XCTAssertEqual(request.tools[0].name, "search")
        XCTAssertNil(request.tools[0].orderedRaw)
    }
}
