import XCTest

@testable import ServingCore

final class OpenAIEmbeddingsTests: XCTestCase {
    // MARK: - Request decoding: accept cases

    func testAcceptsSingleStringInput() throws {
        let body = #"{"model":"embed-1","input":"hello world"}"#

        let request = try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8))

        XCTAssertEqual(request.model, "embed-1")
        XCTAssertEqual(request.inputs, [.text("hello world")])
        XCTAssertEqual(request.encodingFormat, .float)
        XCTAssertNil(request.dimensions)
        XCTAssertEqual(request.ignoredFields, [])
    }

    func testAcceptsArrayOfStrings() throws {
        let body = #"{"model":"embed-1","input":["a","b","c"]}"#

        let request = try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8))

        XCTAssertEqual(request.inputs, [.text("a"), .text("b"), .text("c")])
    }

    func testAcceptsArrayOfIntegersAsOnePreTokenizedInput() throws {
        let body = #"{"model":"embed-1","input":[1,2,3]}"#

        let request = try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8))

        XCTAssertEqual(request.inputs, [.tokens([1, 2, 3])])
    }

    func testAcceptsArrayOfIntegerArraysAsMultipleTokenizedInputs() throws {
        let body = #"{"model":"embed-1","input":[[1,2],[3,4,5]]}"#

        let request = try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8))

        XCTAssertEqual(request.inputs, [.tokens([1, 2]), .tokens([3, 4, 5])])
    }

    func testDefaultEncodingFormatIsFloat() throws {
        let body = #"{"model":"embed-1","input":"hi"}"#
        let request = try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.encodingFormat, .float)
    }

    func testExplicitEncodingFormats() throws {
        let floatBody = #"{"model":"embed-1","input":"hi","encoding_format":"float"}"#
        XCTAssertEqual(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(floatBody.utf8)).encodingFormat, .float)

        let base64Body = #"{"model":"embed-1","input":"hi","encoding_format":"base64"}"#
        XCTAssertEqual(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(base64Body.utf8)).encodingFormat, .base64)
    }

    func testDimensionsParsedWhenPositiveIntegerAndNilWhenAbsent() throws {
        let withDimensions = #"{"model":"embed-1","input":"hi","dimensions":256}"#
        XCTAssertEqual(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(withDimensions.utf8)).dimensions, 256)

        let withoutDimensions = #"{"model":"embed-1","input":"hi"}"#
        XCTAssertNil(try OpenAIEmbeddingsRequest.decodeStrict(from: Data(withoutDimensions.utf8)).dimensions)
    }

    func testUserFieldIsAcceptedAndIgnored() throws {
        let body = #"{"model":"embed-1","input":"hi","user":"user-123"}"#
        let request = try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.ignoredFields, ["user"])
    }

    func testMaximumInputCountConstantIsTwoThousandFortyEight() {
        XCTAssertEqual(openAIEmbeddingsMaximumInputCount, 2048)
    }

    func testAcceptsExactlyMaximumInputCount() throws {
        let inputs = (0..<openAIEmbeddingsMaximumInputCount).map { "\"item\($0)\"" }.joined(separator: ",")
        let body = #"{"model":"embed-1","input":["# + inputs + #"]}"#
        let request = try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.inputs.count, openAIEmbeddingsMaximumInputCount)
    }

    // MARK: - Request decoding: reject cases

    func testRejectsEmptyStringInput() {
        let body = #"{"model":"embed-1","input":""}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsEmptyArrayInput() {
        let body = #"{"model":"embed-1","input":[]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsEmptyInnerArray() {
        let body = #"{"model":"embed-1","input":[[1,2],[]]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsEmptyStringEntryInArray() {
        let body = #"{"model":"embed-1","input":["a",""]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsMixedStringAndArrayElementTypes() {
        let body = #"{"model":"embed-1","input":["a",[1,2]]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsMixedStringAndNumberElementTypes() {
        let body = #"{"model":"embed-1","input":["a",1]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsMixedNumberAndArrayElementTypes() {
        let body = #"{"model":"embed-1","input":[1,[2,3]]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsNonIntegerTokenId() {
        let body = #"{"model":"embed-1","input":[1,2.5,3]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsNegativeTokenId() {
        let body = #"{"model":"embed-1","input":[1,-2,3]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsNonIntegerTokenIdInsideInnerArray() {
        let body = #"{"model":"embed-1","input":[[1,2],[3,4.2]]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsMoreThanMaximumInputCount() {
        let inputs = (0...openAIEmbeddingsMaximumInputCount).map { "\"item\($0)\"" }.joined(separator: ",")
        let body = #"{"model":"embed-1","input":["# + inputs + #"]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsMoreThanMaximumTokenSequenceCount() {
        let inputs = (0...openAIEmbeddingsMaximumInputCount).map { _ in "[1,2]" }.joined(separator: ",")
        let body = #"{"model":"embed-1","input":["# + inputs + #"]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    // MARK: - Limits: per-input token count and total token count

    func testMaximumTokensPerInputConstantMatchesDocumentedLimit() {
        XCTAssertEqual(openAIEmbeddingsMaximumTokensPerInput, 8191)
    }

    func testMaximumTotalTokensConstantMatchesDocumentedLimit() {
        XCTAssertEqual(openAIEmbeddingsMaximumTotalTokens, 300_000)
    }

    func testAcceptsExactlyMaximumTokensPerInput() throws {
        let tokens = (0..<4).map(String.init).joined(separator: ",")
        let body = #"{"model":"embed-1","input":["# + "[" + tokens + "]" + #"]}"#
        let request = try OpenAIEmbeddingsRequest.decodeStrict(
            from: Data(body.utf8), maximumTokensPerInput: 4, maximumTotalTokens: 4)
        XCTAssertEqual(request.inputs, [.tokens([0, 1, 2, 3])])
    }

    func testRejectsOneOverMaximumTokensPerInput() {
        let tokens = (0..<5).map(String.init).joined(separator: ",")
        let body = #"{"model":"embed-1","input":["# + "[" + tokens + "]" + #"]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(
                from: Data(body.utf8), maximumTokensPerInput: 4, maximumTotalTokens: 5),
            type: .invalidRequest, param: "input")
    }

    func testAcceptsExactlyMaximumTotalTokens() throws {
        // Two sequences of 3 tokens each, summing exactly to the total-tokens cap.
        let body = #"{"model":"embed-1","input":[[1,2,3],[4,5,6]]}"#
        let request = try OpenAIEmbeddingsRequest.decodeStrict(
            from: Data(body.utf8), maximumTokensPerInput: 3, maximumTotalTokens: 6)
        XCTAssertEqual(request.inputs, [.tokens([1, 2, 3]), .tokens([4, 5, 6])])
    }

    func testRejectsOneOverMaximumTotalTokens() {
        let body = #"{"model":"embed-1","input":[[1,2,3],[4,5,6,7]]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(
                from: Data(body.utf8), maximumTokensPerInput: 4, maximumTotalTokens: 6),
            type: .invalidRequest, param: "input")
    }

    func testAcceptsExactlyMaximumTotalTokensForFlatTokenArray() throws {
        let tokens = (0..<6).map(String.init).joined(separator: ",")
        let body = #"{"model":"embed-1","input":["# + tokens + #"]}"#
        let request = try OpenAIEmbeddingsRequest.decodeStrict(
            from: Data(body.utf8), maximumTokensPerInput: 6, maximumTotalTokens: 6)
        XCTAssertEqual(request.inputs, [.tokens([0, 1, 2, 3, 4, 5])])
    }

    func testRejectsOneOverMaximumTotalTokensForFlatTokenArray() {
        let tokens = (0..<7).map(String.init).joined(separator: ",")
        let body = #"{"model":"embed-1","input":["# + tokens + #"]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(
                from: Data(body.utf8), maximumTokensPerInput: 7, maximumTotalTokens: 6),
            type: .invalidRequest, param: "input")
    }

    // MARK: - Integer strictness

    func testRejectsDimensionsAsFloatValue() {
        let body = #"{"model":"embed-1","input":"hi","dimensions":1.0}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "dimensions")
    }

    func testRejectsDimensionsAsNonIntegerFloat() {
        let body = #"{"model":"embed-1","input":"hi","dimensions":1.5}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "dimensions")
    }

    func testRejectsDimensionsAsBoolean() {
        let body = #"{"model":"embed-1","input":"hi","dimensions":true}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "dimensions")
    }

    func testRejectsDimensionsAsString() {
        let body = #"{"model":"embed-1","input":"hi","dimensions":"4"}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "dimensions")
    }

    func testNonIntegerDimensionsDoesNotGetGreaterThanZeroMessage() {
        let body = #"{"model":"embed-1","input":"hi","dimensions":1.5}"#
        do {
            _ = try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8))
            XCTFail("Expected OpenAIServingError")
        } catch let error as OpenAIServingError {
            XCTAssertFalse(error.openAIError.message.contains("greater than zero"))
            XCTAssertTrue(error.openAIError.message.contains("positive integer"))
        } catch {
            XCTFail("Expected OpenAIServingError, got \(error)")
        }
    }

    func testRejectsTokenIdAsFloatValue() {
        // 2.0 is mathematically integer-valued but was NOT written as a JSON integer literal.
        let body = #"{"model":"embed-1","input":[1,2.0,3]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsTokenIdAsBoolean() {
        let body = #"{"model":"embed-1","input":[1,true,3]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsTokenIdAsStringInsideTokenArray() {
        let body = #"{"model":"embed-1","input":[1,"2",3]}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsMissingModel() {
        let body = #"{"input":"hi"}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "model")
    }

    func testRejectsEmptyModel() {
        let body = #"{"model":"   ","input":"hi"}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "model")
    }

    func testRejectsMissingInput() {
        let body = #"{"model":"embed-1"}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "input")
    }

    func testRejectsUnsupportedEncodingFormat() {
        let body = #"{"model":"embed-1","input":"hi","encoding_format":"int8"}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8)),
            type: .invalidRequest, param: "encoding_format")
    }

    func testRejectsNonPositiveDimensions() {
        let zero = #"{"model":"embed-1","input":"hi","dimensions":0}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(zero.utf8)),
            type: .invalidRequest, param: "dimensions")

        let negative = #"{"model":"embed-1","input":"hi","dimensions":-4}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(negative.utf8)),
            type: .invalidRequest, param: "dimensions")
    }

    func testUnknownTopLevelKeyIsIgnoredAndWarned() throws {
        // Mirrors OpenAIChatCompletions' lenient-unknown-top-level-key policy: unrecognized keys
        // are tolerated, not a 400, and surfaced as a sanitized "unknown:<key>" warning.
        let body = #"{"model":"embed-1","input":"hi","totally_unknown_field":true}"#
        let request = try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.ignoredFields, ["unknown:totally_unknown_field"])
    }

    func testVeryLongUnknownKeyNameIsSanitizedNotEchoedVerbatim() throws {
        let longKey = String(repeating: "k", count: 10_000)
        let body = "{\"model\":\"embed-1\",\"input\":\"hi\",\"\(longKey)\":true}"
        let request = try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.ignoredFields, ["unknown:<invalid-key>"])
        for field in request.ignoredFields {
            XCTAssertLessThanOrEqual(field.count, 64)
            XCTAssertFalse(field.contains(longKey))
        }
    }

    func testManyUnknownKeysAreCapped() throws {
        var object = "\"model\":\"embed-1\",\"input\":\"hi\""
        for index in 0..<20 {
            object += ",\"unknown_field_\(index)\":true"
        }
        let body = "{\(object)}"
        let request = try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8))
        XCTAssertEqual(request.ignoredFields.count, 17)
        XCTAssertTrue(request.ignoredFields.contains("unknown:<more>"))
    }

    func testRejectsOversizedBody() {
        let limits = OpenAIChatRequestLimits(maximumBodyBytes: 32, maximumCompletionTokens: 8)
        let body = #"{"model":"embed-1","input":"\#(String(repeating: "x", count: 64))"}"#
        XCTAssertOpenAIError(
            try OpenAIEmbeddingsRequest.decodeStrict(from: Data(body.utf8), limits: limits),
            type: .invalidRequest, param: nil)
    }

    // MARK: - Response encoding

    func testResponseShapeAndIndexOrdering() throws {
        let response = try OpenAIEmbeddingsResponseBuilder.encode(
            embeddings: [[1.0, 2.0], [3.0, 4.0, 5.0]],
            model: "embed-1",
            encodingFormat: .float,
            promptTokens: 7)

        XCTAssertEqual(response.object, "list")
        XCTAssertEqual(response.model, "embed-1")
        XCTAssertEqual(response.data.map(\.index), [0, 1])
        XCTAssertEqual(response.usage.promptTokens, 7)
        XCTAssertEqual(response.usage.totalTokens, 7)

        let data = try JSONEncoder.openAI.encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["object"] as? String, "list")
        XCTAssertEqual(object["model"] as? String, "embed-1")
        let items = try XCTUnwrap(object["data"] as? [[String: Any]])
        XCTAssertEqual(items.count, 2)
        XCTAssertEqual(items[0]["object"] as? String, "embedding")
        XCTAssertEqual(items[0]["index"] as? Int, 0)
        XCTAssertEqual(items[0]["embedding"] as? [Double], [1.0, 2.0])
        XCTAssertEqual(items[1]["index"] as? Int, 1)
        XCTAssertEqual(items[1]["embedding"] as? [Double], [3.0, 4.0, 5.0])
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 7)
        XCTAssertEqual(usage["total_tokens"] as? Int, 7)
    }

    func testBase64EncodingRoundTripsToExactFloats() throws {
        let vector: [Float] = [1.5, -2.25, 0.0, 3.0e10, -1.0e-10]
        let response = try OpenAIEmbeddingsResponseBuilder.encode(
            embeddings: [vector],
            model: "embed-1",
            encodingFormat: .base64,
            promptTokens: 3)

        let data = try JSONEncoder.openAI.encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let items = try XCTUnwrap(object["data"] as? [[String: Any]])
        let base64String = try XCTUnwrap(items[0]["embedding"] as? String)

        let decodedBytes = try XCTUnwrap(Data(base64Encoded: base64String))
        XCTAssertEqual(decodedBytes.count, vector.count * 4)
        var decodedFloats: [Float] = []
        decodedBytes.withUnsafeBytes { raw in
            for i in 0..<vector.count {
                let bits = UInt32(littleEndian: raw.load(fromByteOffset: i * 4, as: UInt32.self))
                decodedFloats.append(Float(bitPattern: bits))
            }
        }
        XCTAssertEqual(decodedFloats, vector)
    }

    func testEncodingRejectsNonFiniteValues() {
        XCTAssertThrowsError(
            try OpenAIEmbeddingsResponseBuilder.encode(
                embeddings: [[1.0, Float.nan]],
                model: "embed-1",
                encodingFormat: .float,
                promptTokens: 1)
        ) { error in
            guard let servingError = error as? OpenAIServingError else {
                return XCTFail("Expected OpenAIServingError, got \(error)")
            }
            XCTAssertEqual(servingError.openAIError.type, .serverError)
        }

        XCTAssertThrowsError(
            try OpenAIEmbeddingsResponseBuilder.encode(
                embeddings: [[Float.infinity]],
                model: "embed-1",
                encodingFormat: .base64,
                promptTokens: 1)
        ) { error in
            guard let servingError = error as? OpenAIServingError else {
                return XCTFail("Expected OpenAIServingError, got \(error)")
            }
            XCTAssertEqual(servingError.openAIError.type, .serverError)
        }
    }

    func testUsageTotalTokensEqualsPromptTokens() throws {
        let response = try OpenAIEmbeddingsResponseBuilder.encode(
            embeddings: [[1.0]],
            model: "embed-1",
            encodingFormat: .float,
            promptTokens: 42)
        XCTAssertEqual(response.usage.totalTokens, response.usage.promptTokens)
    }
}

private func XCTAssertOpenAIError<T>(
    _ expression: @autoclosure () throws -> T,
    type: OpenAIErrorType,
    param: String?,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    do {
        _ = try expression()
        XCTFail("Expected OpenAIServingError", file: file, line: line)
    } catch let error as OpenAIServingError {
        XCTAssertEqual(error.openAIError.type, type, file: file, line: line)
        XCTAssertEqual(error.openAIError.param, param, file: file, line: line)
    } catch {
        XCTFail("Expected OpenAIServingError, got \(error)", file: file, line: line)
    }
}
