import XCTest
@testable import HarnessCore

final class EngineDriverTests: XCTestCase {
    func testScriptedDriverReplaysTokensAndEngagement() async throws {
        let d = ScriptedDriver(tokens: [5, 6, 7], engagement: ["pld": 3], logprobs: [[-0.1, -2.0]])
        let r = try await d.generate(prompt: [1, 2], config: .greedy(maxTokens: 8))
        XCTAssertEqual(r.tokens, [5, 6, 7])
        XCTAssertEqual(r.engagement.counts["pld"], 3)
    }
}
