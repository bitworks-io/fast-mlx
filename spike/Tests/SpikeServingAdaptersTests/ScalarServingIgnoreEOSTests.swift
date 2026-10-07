import XCTest

import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

/// `ignore_eos` on the scalar serving route (predeclared acceptance E2, E4, E5, E7 in
/// `docs/task-inbox/2026-10-07-PREDECLARATION-engine-ignore-eos-request-field.md`). A fake decoder
/// emits the model stop token (99) after N = 2 tokens and keeps going, so the flag's effect is
/// observable independent of MLX.
final class ScalarServingIgnoreEOSTests: XCTestCase {
    private static let script = [1, 2, 99, 5, 6, 7]
    private static let pieces: [Int: String] = [
        1: "a", 2: "b", 99: "</s>", 5: "c", 6: "d", 7: "e",
    ]

    private func logprobTokens(
        _ events: [ServingResponseDelta]
    ) -> [ServingTokenLogprob] {
        events.flatMap { event -> [ServingTokenLogprob] in
            if case .tokenLogprobs(let tokens) = event { return tokens }
            return []
        }
    }

    private func completion(_ events: [ServingResponseDelta]) -> ServingGenerationCompletion? {
        events.compactMap { event -> ServingGenerationCompletion? in
            if case .completion(let completion) = event { return completion }
            return nil
        }.first
    }

    private func text(_ events: [ServingResponseDelta]) -> String {
        events.compactMap { event -> String? in
            if case .text(let text) = event { return text }
            return nil
        }.joined()
    }

    // E2 + E5: N + 3 tokens come back, token N is the stop id with a logprob entry whose token
    // string is non-empty, and the run ends by `length`.
    func testIgnoreEOSGeneratesPastTheStopTokenAndEndsByLength() async throws {
        let backend = makeBackend()
        var ignoring = request(maxTokens: 5)
        ignoring.ignoreEOS = true
        ignoring.logprobsRequest = .chat(topLogprobs: 1)

        let handle = try await backend.start(ignoring)
        let events = try await collect(handle.mailbox)
        let tokens = logprobTokens(events)

        XCTAssertEqual(tokens.count, 5, "events: \(events)")
        guard tokens.count == 5 else { return }
        XCTAssertEqual(
            tokens.map(\.tokenText), ["tok1", "tok2", "tok99", "tok5", "tok6"],
            "the stop token (index 2) must carry its own logprob entry")
        XCTAssertFalse(tokens[2].tokenText.isEmpty)
        let done = completion(events)
        XCTAssertEqual(done?.finishReason, .length)
        XCTAssertEqual(done?.usage.completionTokens, 5)
        XCTAssertEqual(text(events), "ab</s>cd")
    }

    // E2 control: the identical request without the flag stops at the stop token exactly as before.
    func testWithoutIgnoreEOSTheStopTokenStillEndsGeneration() async throws {
        let backend = makeBackend()
        var plain = request(maxTokens: 5)
        plain.logprobsRequest = .chat(topLogprobs: 1)

        let handle = try await backend.start(plain)
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(logprobTokens(events).map(\.tokenText), ["tok1", "tok2"])
        let done = completion(events)
        XCTAssertEqual(done?.finishReason, .stop)
        XCTAssertEqual(done?.usage.completionTokens, 2)
        XCTAssertEqual(text(events), "ab")
    }

    // E2 control: an explicit `ignore_eos: false` is the same as absent.
    func testExplicitFalseBehavesLikeAbsent() async throws {
        let backend = makeBackend()
        var explicitFalse = request(maxTokens: 5)
        explicitFalse.ignoreEOS = false

        let handle = try await backend.start(explicitFalse)
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(text(events), "ab")
        XCTAssertEqual(completion(events)?.finishReason, .stop)
    }

    // E4: a user `stop` string still ends generation with `ignore_eos` on, and finish_reason is
    // `stop` (not `length`) even though the model stop token was generated through.
    func testUserStopStringStillEndsGenerationWithIgnoreEOS() async throws {
        let backend = makeBackend()
        var ignoring = request(maxTokens: 6, stop: ["c"])
        ignoring.ignoreEOS = true

        let handle = try await backend.start(ignoring)
        let events = try await collect(handle.mailbox)

        XCTAssertEqual(text(events), "ab</s>", "the stop string must cut the output before it")
        let done = completion(events)
        XCTAssertEqual(done?.finishReason, .stop)
        XCTAssertEqual(done?.usage.completionTokens, 4)
    }

    // E5: when the codec's token decode comes back empty for the stop token (a special token that
    // decodes to nothing), the logprob entry falls back to the token's literal vocabulary text so a
    // client can still tell it apart. Only the generated stop token gets that fallback.
    func testStopTokenLogprobFallsBackToVocabularyTextWhenDecodeIsEmpty() async throws {
        let backend = makeBackend(
            codec: EmptyDecodeCodec(pieces: Self.pieces, vocabulary: [99: "</s>"]))
        var ignoring = request(maxTokens: 4)
        ignoring.ignoreEOS = true
        ignoring.logprobsRequest = .chat(topLogprobs: 0)

        let handle = try await backend.start(ignoring)
        let events = try await collect(handle.mailbox)
        let tokens = logprobTokens(events)

        XCTAssertEqual(tokens.count, 4, "events: \(events)")
        guard tokens.count == 4 else { return }
        XCTAssertEqual(tokens[2].tokenText, "</s>")
        XCTAssertEqual(tokens[0].tokenText, "", "ordinary tokens keep today's decode result")
    }

    // E7: a speculative scalar route (in-checkpoint MTP, `isNonSpeculativeScalarRoute == false`)
    // refuses `ignore_eos: true` with a typed reason before any generation starts, instead of
    // silently ignoring it. `ignore_eos: false` stays admitted.
    func testSpeculativeScalarRouteRefusesIgnoreEOS() async throws {
        let backend = makeBackend(isNonSpeculativeScalarRoute: false)
        var ignoring = request(maxTokens: 5)
        ignoring.ignoreEOS = true

        do {
            _ = try await backend.start(ignoring)
            XCTFail("speculative route must refuse ignore_eos")
        } catch let error as OpenAIServingError {
            guard case .invalidRequestWithCode(_, let param, let code) = error else {
                XCTFail("expected invalidRequestWithCode, got \(error)")
                return
            }
            XCTAssertEqual(param, "ignore_eos")
            XCTAssertEqual(code, "ignore_eos_unsupported")
        }

        let handle = try await backend.start(request(maxTokens: 5))
        let events = try await collect(handle.mailbox)
        XCTAssertEqual(completion(events)?.finishReason, .stop)
    }

    // The same fake wired the other way (a plain non-speculative route) must be admitted: proves
    // the refusal above keys on the route, not on the flag alone.
    func testNonSpeculativeScalarRouteAdmitsIgnoreEOS() async throws {
        let backend = makeBackend(isNonSpeculativeScalarRoute: true)
        var ignoring = request(maxTokens: 4)
        ignoring.ignoreEOS = true

        let handle = try await backend.start(ignoring)
        let events = try await collect(handle.mailbox)
        XCTAssertEqual(completion(events)?.finishReason, .length)
    }
}

// MARK: - Fixtures

private func makeBackend(
    codec: (any ScalarServingTextCodec)? = nil,
    isNonSpeculativeScalarRoute: Bool = true
) -> ScalarServingBackend {
    ScalarServingBackend(
        launchedModel: "fixture-model",
        inference: InferenceActor(
            decoder: IgnoreEOSLogprobDecoder(
                script: [1, 2, 99, 5, 6, 7], candidateTokenIDs: [50, 60, 70, 80, 90, 91])),
        codec: codec
            ?? IgnoreEOSCodec(
                pieces: [1: "a", 2: "b", 99: "</s>", 5: "c", 6: "d", 7: "e"]),
        stopTokenIDs: [99],
        modelStopStrings: [],
        configuration: .init(
            defaultMaximumCompletionTokens: 8,
            maximumQueuedRequests: 2,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 16, maxBytes: 4_096),
            isNonSpeculativeScalarRoute: isNonSpeculativeScalarRoute))
}

private func request(
    maxTokens: Int,
    stop: [String] = []
) -> OpenAIChatCompletionRequest {
    OpenAIChatCompletionRequest(
        model: "fixture-model",
        messages: [OpenAIChatMessage(role: .user, text: "private prompt")],
        maxCompletionTokens: maxTokens,
        temperature: 0,
        choiceCount: 1,
        stream: true,
        stop: stop)
}

private func collect(
    _ mailbox: BoundedDeltaMailbox
) async throws -> [ServingResponseDelta] {
    var events: [ServingResponseDelta] = []
    while let event = try await mailbox.next() {
        events.append(event)
    }
    return events
}

/// Replays a fixed script and reports a synthetic logprob per generated token (never a real MLX
/// computation). The script is replayed by position, so a past-the-stop-token run keeps reading it.
private struct IgnoreEOSLogprobDecoder: LogprobDecoding {
    let script: [Int]
    let candidateTokenIDs: [Int]
    var i = 0

    mutating func prefill(_ promptTokens: [Int]) -> Int {
        defer { i += 1 }
        return script[i]
    }

    mutating func step(last: Int) -> Int {
        defer { i += 1 }
        return script[i]
    }

    mutating func reset() { i = 0 }

    mutating func prefillWithLogprob(
        _ promptTokens: [Int], topN: Int
    ) -> (token: Int, logprob: DecodedTokenLogprob) {
        let index = i
        let token = prefill(promptTokens)
        return (token, makeLogprob(token: token, index: index, topN: topN))
    }

    mutating func stepWithLogprob(
        last: Int, topN: Int
    ) -> (token: Int, logprob: DecodedTokenLogprob) {
        let index = i
        let token = step(last: last)
        return (token, makeLogprob(token: token, index: index, topN: topN))
    }

    private func makeLogprob(token: Int, index: Int, topN: Int) -> DecodedTokenLogprob {
        var top: [DecodedTokenLogprobCandidate] = []
        if topN > 0, index < candidateTokenIDs.count {
            top = [DecodedTokenLogprobCandidate(tokenID: candidateTokenIDs[index], logprob: -1.0)]
        }
        return DecodedTokenLogprob(tokenID: token, logprob: Float(-(index + 1)), top: top)
    }
}

private struct IgnoreEOSCodec: ScalarServingTextCodec {
    let pieces: [Int: String]

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?,
        addGenerationPrompt: Bool?
    ) throws -> [Int] {
        [10]
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        IgnoreEOSDetokenizer(pieces: pieces)
    }

    func decodeSingleToken(_ tokenID: Int) -> String {
        "tok\(tokenID)"
    }
}

/// A codec whose single-token decode is empty for every token (like a tokenizer that renders a
/// special token as nothing) but which can report a token's literal vocabulary text.
private struct EmptyDecodeCodec: ScalarServingTextCodec {
    let pieces: [Int: String]
    let vocabulary: [Int: String]

    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?,
        addGenerationPrompt: Bool?
    ) throws -> [Int] {
        [10]
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        IgnoreEOSDetokenizer(pieces: pieces)
    }

    func decodeSingleToken(_ tokenID: Int) -> String {
        ""
    }

    func vocabularyText(forTokenID tokenID: Int) -> String? {
        vocabulary[tokenID]
    }
}

private struct IgnoreEOSDetokenizer: ScalarServingDetokenizer {
    let pieces: [Int: String]
    private var pending: String?

    init(pieces: [Int: String]) {
        self.pieces = pieces
    }

    mutating func append(token: Int) {
        pending = pieces[token]
    }

    mutating func next() -> String? {
        defer { pending = nil }
        return pending
    }
}
