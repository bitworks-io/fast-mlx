import XCTest

@testable import ServingCore

final class ServingLogprobsTests: XCTestCase {
    // MARK: - Sanitization (-infinity handling)

    func testSanitizedLogprobPassesThroughFiniteValues() {
        let token = ServingTokenLogprob(tokenText: "hi", logprob: -0.25)
        XCTAssertEqual(token.sanitizedLogprob, -0.25)
    }

    func testSanitizedLogprobReplacesNegativeInfinityWithSentinel() {
        let token = ServingTokenLogprob(tokenText: "hi", logprob: -Double.infinity)
        XCTAssertEqual(token.sanitizedLogprob, -9999.0)
    }

    func testFiniteTopCandidatesExcludesNegativeInfinityEntries() {
        let token = ServingTokenLogprob(
            tokenText: "hi",
            logprob: -0.1,
            topCandidates: [
                ServingTokenLogprobCandidate(tokenText: "hi", logprob: -0.1),
                ServingTokenLogprobCandidate(tokenText: "masked", logprob: -Double.infinity),
                ServingTokenLogprobCandidate(tokenText: "bye", logprob: -2.5),
            ])
        XCTAssertEqual(token.finiteTopCandidates.map(\.tokenText), ["hi", "bye"])
    }

    // MARK: - OpenAIChatLogprobs

    func testChatLogprobsEncodesContentAndAlwaysNullRefusal() throws {
        let logprobs = OpenAIChatLogprobs(tokens: [
            ServingTokenLogprob(
                tokenText: "Hi",
                logprob: -0.05,
                topCandidates: [
                    ServingTokenLogprobCandidate(tokenText: "Hi", logprob: -0.05),
                    ServingTokenLogprobCandidate(tokenText: "Hey", logprob: -3.2),
                ])
        ])
        let data = try JSONEncoder.openAI.encode(logprobs)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertTrue(object.keys.contains("refusal"))
        XCTAssertTrue(object["refusal"] is NSNull)
        let content = try XCTUnwrap(object["content"] as? [[String: Any]])
        XCTAssertEqual(content.count, 1)
        XCTAssertEqual(content[0]["token"] as? String, "Hi")
        XCTAssertEqual(content[0]["logprob"] as? Double, -0.05)
        XCTAssertEqual(content[0]["bytes"] as? [Int], Array("Hi".utf8).map(Int.init))
        let top = try XCTUnwrap(content[0]["top_logprobs"] as? [[String: Any]])
        XCTAssertEqual(top.count, 2)
        XCTAssertEqual(top[0]["token"] as? String, "Hi")
        XCTAssertEqual(top[1]["token"] as? String, "Hey")
    }

    func testChatLogprobsEncodesNullBytesForReplacementCharacterToken() throws {
        // A byte-level BPE token that only covers part of a multi-byte UTF-8 character (e.g. one
        // half of a split emoji) decodes alone to U+FFFD — its single-token `bytes` would be
        // fabricated ([239,191,189], the replacement character's own UTF-8), not the real
        // underlying bytes. OpenAI's schema allows `bytes: null` for exactly this case.
        let logprobs = OpenAIChatLogprobs(tokens: [
            ServingTokenLogprob(tokenText: "\u{FFFD}", logprob: -0.05)
        ])
        let data = try JSONEncoder.openAI.encode(logprobs)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let content = try XCTUnwrap(object["content"] as? [[String: Any]])
        XCTAssertTrue(content[0].keys.contains("bytes"), "the \"bytes\" key must still be present")
        XCTAssertTrue(content[0]["bytes"] is NSNull, "bytes must be JSON null, not omitted or fabricated")
    }

    func testChatLogprobsKeepsBytesForNormalToken() throws {
        let logprobs = OpenAIChatLogprobs(tokens: [
            ServingTokenLogprob(tokenText: " there", logprob: -0.05)
        ])
        let data = try JSONEncoder.openAI.encode(logprobs)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let content = try XCTUnwrap(object["content"] as? [[String: Any]])
        XCTAssertEqual(content[0]["bytes"] as? [Int], [32, 116, 104, 101, 114, 101])
    }

    func testChatLogprobsEncodesNullBytesForReplacementCharacterTopLogprobCandidate() throws {
        let logprobs = OpenAIChatLogprobs(tokens: [
            ServingTokenLogprob(
                tokenText: "hi",
                logprob: -0.05,
                topCandidates: [
                    ServingTokenLogprobCandidate(tokenText: "\u{FFFD}", logprob: -1.0),
                    ServingTokenLogprobCandidate(tokenText: " there", logprob: -2.0),
                ])
        ])
        let data = try JSONEncoder.openAI.encode(logprobs)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let content = try XCTUnwrap(object["content"] as? [[String: Any]])
        let top = try XCTUnwrap(content[0]["top_logprobs"] as? [[String: Any]])
        XCTAssertTrue(top[0].keys.contains("bytes"))
        XCTAssertTrue(top[0]["bytes"] is NSNull)
        XCTAssertEqual(top[1]["bytes"] as? [Int], [32, 116, 104, 101, 114, 101])
    }

    func testChatLogprobsExcludesNonFiniteCandidatesAndSanitizesSampledToken() throws {
        let logprobs = OpenAIChatLogprobs(tokens: [
            ServingTokenLogprob(
                tokenText: "masked",
                logprob: -Double.infinity,
                topCandidates: [
                    ServingTokenLogprobCandidate(tokenText: "masked", logprob: -Double.infinity),
                    ServingTokenLogprobCandidate(tokenText: "ok", logprob: -1.0),
                ])
        ])
        // Must not throw: JSONEncoder cannot encode -Infinity, so this alone proves sanitization ran.
        let data = try JSONEncoder.openAI.encode(logprobs)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let content = try XCTUnwrap(object["content"] as? [[String: Any]])
        XCTAssertEqual(content[0]["logprob"] as? Double, -9999.0)
        let top = try XCTUnwrap(content[0]["top_logprobs"] as? [[String: Any]])
        XCTAssertEqual(top.map { $0["token"] as? String }, ["ok"])
    }

    // MARK: - OpenAICompletionLogprobs

    func testCompletionLogprobsBuildComputesCodePointTextOffsets() {
        let tokens = [
            ServingTokenLogprob(tokenText: "Hel", logprob: -0.1),
            ServingTokenLogprob(tokenText: "lo", logprob: -0.2),
        ]
        let built = OpenAICompletionLogprobs.build(
            tokens: tokens, requestedTopLogprobs: 0, startingOffset: 0)
        XCTAssertEqual(built.tokens, ["Hel", "lo"])
        XCTAssertEqual(built.tokenLogprobs, [-0.1, -0.2])
        XCTAssertEqual(built.textOffset, [0, 3])
        XCTAssertEqual(built.topLogprobs, [nil, nil])
    }

    func testCompletionLogprobsZeroForcesNullTopLogprobsEvenWithCandidates() {
        let tokens = [
            ServingTokenLogprob(
                tokenText: "a",
                logprob: -0.1,
                topCandidates: [ServingTokenLogprobCandidate(tokenText: "a", logprob: -0.1)])
        ]
        let built = OpenAICompletionLogprobs.build(
            tokens: tokens, requestedTopLogprobs: 0, startingOffset: 0)
        XCTAssertEqual(built.topLogprobs, [nil])
    }

    func testCompletionLogprobsPositiveTopLogprobsPopulatesDictionaries() {
        let tokens = [
            ServingTokenLogprob(
                tokenText: "a",
                logprob: -0.1,
                topCandidates: [
                    ServingTokenLogprobCandidate(tokenText: "a", logprob: -0.1),
                    ServingTokenLogprobCandidate(tokenText: "b", logprob: -2.0),
                ])
        ]
        let built = OpenAICompletionLogprobs.build(
            tokens: tokens, requestedTopLogprobs: 2, startingOffset: 0)
        XCTAssertEqual(built.topLogprobs.count, 1)
        XCTAssertEqual(built.topLogprobs[0], ["a": -0.1, "b": -2.0])
    }

    func testCompletionLogprobsEndingOffsetContinuesAcrossBatches() {
        let firstBatch = [ServingTokenLogprob(tokenText: "Hel", logprob: -0.1)]
        let secondBatch = [ServingTokenLogprob(tokenText: "lo", logprob: -0.2)]
        let firstEnd = OpenAICompletionLogprobs.endingOffset(startingOffset: 0, tokens: firstBatch)
        XCTAssertEqual(firstEnd, 3)
        let secondBuilt = OpenAICompletionLogprobs.build(
            tokens: secondBatch, requestedTopLogprobs: 0, startingOffset: firstEnd)
        XCTAssertEqual(secondBuilt.textOffset, [3])
    }

    // MARK: - ServingLogprobsRequest

    func testServingLogprobsRequestTopLogprobsAccessor() {
        XCTAssertEqual(ServingLogprobsRequest.chat(topLogprobs: 5).topLogprobs, 5)
        XCTAssertEqual(ServingLogprobsRequest.completions(topLogprobs: 3).topLogprobs, 3)
    }
}
