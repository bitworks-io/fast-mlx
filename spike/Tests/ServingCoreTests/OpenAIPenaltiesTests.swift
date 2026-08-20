import XCTest

@testable import ServingCore

/// Decode + validation for the OpenAI/HF penalty knobs (presence/frequency in [-2, 2], repetition > 0).
/// These map to the decoder's logit processor on the scalar route; the continuous route rejects them.
final class OpenAIPenaltiesTests: XCTestCase {
    private func request(_ json: String) throws -> OpenAIChatCompletionRequest {
        try OpenAIChatCompletionRequest.decodeStrict(from: Data(json.utf8))
    }

    func testPenaltiesDecodeWhenPresent() throws {
        let r = try request(#"""
        {"model":"m","messages":[{"role":"user","content":"hi"}],
         "presence_penalty":1.5,"frequency_penalty":-0.5,"repetition_penalty":1.1}
        """#)
        XCTAssertEqual(r.presencePenalty, 1.5)
        XCTAssertEqual(r.frequencyPenalty, -0.5)
        XCTAssertEqual(r.repetitionPenalty, 1.1)
    }

    func testPenaltiesAreNilWhenAbsent() throws {
        let r = try request(#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
        XCTAssertNil(r.presencePenalty)
        XCTAssertNil(r.frequencyPenalty)
        XCTAssertNil(r.repetitionPenalty)
    }

    func testPresenceOutOfRangeRejected() {
        XCTAssertThrowsError(try request(#"{"model":"m","messages":[{"role":"user","content":"h"}],"presence_penalty":3}"#))
        XCTAssertThrowsError(try request(#"{"model":"m","messages":[{"role":"user","content":"h"}],"presence_penalty":-3}"#))
    }

    func testFrequencyOutOfRangeRejected() {
        XCTAssertThrowsError(try request(#"{"model":"m","messages":[{"role":"user","content":"h"}],"frequency_penalty":2.1}"#))
    }

    func testRepetitionMustBePositive() {
        XCTAssertThrowsError(try request(#"{"model":"m","messages":[{"role":"user","content":"h"}],"repetition_penalty":0}"#))
        XCTAssertThrowsError(try request(#"{"model":"m","messages":[{"role":"user","content":"h"}],"repetition_penalty":-1}"#))
    }
}
