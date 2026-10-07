import XCTest

@testable import ServingCore

/// Decode + validation of the vLLM-compatible `ignore_eos` request field on both OpenAI routes.
/// Predeclared acceptance E1 (`docs/task-inbox/2026-10-07-PREDECLARATION-engine-ignore-eos-request-field.md`).
final class OpenAIIgnoreEOSTests: XCTestCase {
    private func chat(_ extra: String = "") throws -> OpenAIChatCompletionRequest {
        let body = #"{"model":"m","messages":[{"role":"user","content":"hi"}]"# + extra + "}"
        return try OpenAIChatCompletionRequest.decodeStrict(from: Data(body.utf8))
    }

    private func completion(_ extra: String = "") throws -> OpenAICompletionRequest {
        let body = #"{"model":"m","prompt":"hi""# + extra + "}"
        return try OpenAICompletionRequest.decodeStrict(from: Data(body.utf8))
    }

    private func assertRejectsNamingIgnoreEOS(
        _ body: () throws -> Any,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            _ = try body()
            XCTFail("expected a 400 for a non-boolean ignore_eos", file: file, line: line)
        } catch let error as OpenAIServingError {
            guard case .invalidRequest(let message, let param) = error else {
                XCTFail("expected invalidRequest, got \(error)", file: file, line: line)
                return
            }
            XCTAssertEqual(param, "ignore_eos", file: file, line: line)
            XCTAssertTrue(message.contains("ignore_eos"), message, file: file, line: line)
        } catch {
            XCTFail("unexpected error \(error)", file: file, line: line)
        }
    }

    // MARK: chat

    func testChatIgnoreEOSDecodesTrue() throws {
        let request = try chat(#","ignore_eos":true"#)
        XCTAssertTrue(request.ignoreEOS)
        XCTAssertEqual(request.ignoredFields, [], "a honored field must never be listed as ignored")
    }

    func testChatIgnoreEOSDecodesFalse() throws {
        let request = try chat(#","ignore_eos":false"#)
        XCTAssertFalse(request.ignoreEOS)
        XCTAssertEqual(request.ignoredFields, [])
    }

    func testChatIgnoreEOSAbsentDefaultsToFalse() throws {
        let request = try chat()
        XCTAssertFalse(request.ignoreEOS)
    }

    func testChatIgnoreEOSRejectsString() {
        assertRejectsNamingIgnoreEOS { try chat(#","ignore_eos":"true""#) }
    }

    func testChatIgnoreEOSRejectsNumber() {
        assertRejectsNamingIgnoreEOS { try chat(#","ignore_eos":1"#) }
        assertRejectsNamingIgnoreEOS { try chat(#","ignore_eos":0"#) }
    }

    func testChatIgnoreEOSRejectsNull() {
        // `NSNull` is not a Bool: same fail-closed behavior as every other typed boolean field.
        assertRejectsNamingIgnoreEOS { try chat(#","ignore_eos":null"#) }
    }

    // MARK: completions

    func testCompletionsIgnoreEOSDecodesTrue() throws {
        let request = try completion(#","ignore_eos":true"#)
        XCTAssertTrue(request.ignoreEOS)
        XCTAssertEqual(request.ignoredFields, [])
        XCTAssertTrue(request.asChatCompletionRequest().ignoreEOS)
    }

    func testCompletionsIgnoreEOSDecodesFalse() throws {
        let request = try completion(#","ignore_eos":false"#)
        XCTAssertFalse(request.ignoreEOS)
        XCTAssertEqual(request.ignoredFields, [])
        XCTAssertFalse(request.asChatCompletionRequest().ignoreEOS)
    }

    func testCompletionsIgnoreEOSAbsentDefaultsToFalse() throws {
        let request = try completion()
        XCTAssertFalse(request.ignoreEOS)
        XCTAssertFalse(request.asChatCompletionRequest().ignoreEOS)
    }

    func testCompletionsIgnoreEOSRejectsString() {
        assertRejectsNamingIgnoreEOS { try completion(#","ignore_eos":"yes""#) }
    }

    func testCompletionsIgnoreEOSRejectsNumber() {
        assertRejectsNamingIgnoreEOS { try completion(#","ignore_eos":1"#) }
    }

    // A sibling unknown key is still recorded as ignored while ignore_eos is not: the allowed-key
    // set, not a blanket skip, is what makes ignore_eos honored.
    func testIgnoreEOSIsNotAmongIgnoredUnknownKeysButSiblingsAre() throws {
        let chatRequest = try chat(#","ignore_eos":true,"vendor_extension":1"#)
        XCTAssertEqual(chatRequest.ignoredFields, ["unknown:vendor_extension"])
        let completionRequest = try completion(#","ignore_eos":true,"vendor_extension":1"#)
        XCTAssertEqual(completionRequest.ignoredFields, ["unknown:vendor_extension"])
    }
}
