import XCTest
@testable import SpikeCore

final class InferenceActorTests: XCTestCase {
    func testStreamsExpectedGreedyTokensFromFakeModel() async throws {
        // Fake decoder returns a fixed script, proving the actor's loop/streaming
        // is correct independent of MLX. Real model wired in Task 5.
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 7, /*eos*/ 2], eos: 2))
        var got: [Int] = []
        for try await tok in await actor.submit(promptTokens: [1, 2, 3], maxTokens: 10) {
            got.append(tok)
        }
        XCTAssertEqual(got, [5, 6, 7]) // eos consumed, not emitted
    }

    func testMaxTokensStopsGenerationBeforeEOS() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 7, 8, 9], eos: 2))
        var got: [Int] = []
        for try await tok in await actor.submit(promptTokens: [1], maxTokens: 3) {
            got.append(tok)
        }
        XCTAssertEqual(got, [5, 6, 7])
    }

    func testResetForNewRunReplaysFromStartOfScript() async throws {
        let actor = InferenceActor(decoder: ScriptedDecoder(script: [5, 6, 2], eos: 2))
        var first: [Int] = []
        for try await tok in await actor.submit(promptTokens: [1], maxTokens: 10) {
            first.append(tok)
        }
        await actor.resetForNewRun()
        var second: [Int] = []
        for try await tok in await actor.submit(promptTokens: [1], maxTokens: 10) {
            second.append(tok)
        }
        XCTAssertEqual(first, [5, 6])
        XCTAssertEqual(second, [5, 6]) // same script from the top, not continuing past eos
    }
}
