import XCTest
@testable import HarnessCore

/// PerplexityMetric — Layer 2 of the quality stack, the dial's "ppl delta <= 1%" instrument.
/// All rows are RAW LOGITS scripted as ln(p) so softmax reproduces p and every expected value
/// is hand-computable.
final class PerplexityTests: XCTestCase {
  // MARK: meanNLL math

  func testMeanNLLKnownValueUniform() {
    // softmax([0,0]) = [0.5,0.5]; NLL of token 0 = -ln(0.5) = ln 2.
    XCTAssertEqual(meanNLL(rows: [[0.0, 0.0]], tokens: [0]), log(2.0), accuracy: 1e-6)
  }

  func testMeanNLLKnownValueSkewed() {
    // softmax(ln[0.9,0.1]) = [0.9,0.1]; NLL of token 1 = -ln(0.1) ~= 2.3026.
    XCTAssertEqual(meanNLL(rows: [[logf(0.9), logf(0.1)]], tokens: [1]), 2.302585, accuracy: 1e-5)
  }

  func testMeanNLLInvariantToLogitShift() {
    // Raw logits are shift-invariant under softmax: [7,7] must equal [0,0].
    XCTAssertEqual(meanNLL(rows: [[7.0, 7.0]], tokens: [0]), log(2.0), accuracy: 1e-6)
  }

  func testMeanNLLAveragesAcrossPositions() {
    // Positions: NLL(-ln 0.9) + NLL(-ln 0.1), mean = -ln(sqrt(0.09)) -> ppl 1/0.3 = 3.3333.
    let row: [Float] = [logf(0.9), logf(0.1)]
    let mean = meanNLL(rows: [row, row], tokens: [0, 1])
    XCTAssertEqual(mean, -(log(0.9) + log(0.1)) / 2, accuracy: 1e-6)
    XCTAssertEqual(exp(mean), 1.0 / 0.3, accuracy: 1e-4)
  }

  // MARK: metric (teacher-forced, same pass as KL)

  func testPerplexityMetricZeroDeltaForIdenticalDrivers() async throws {
    let rows: [[Float]] = [[logf(0.9), logf(0.1)], [0.0, 0.0]]
    let d = ScriptedDriver(tokens: [0, 1], logprobs: rows)
    let delta = try await PerplexityMetric().measure(
      driver: d, reference: d, prompts: [[7]], config: .greedy(maxTokens: 8))
    XCTAssertEqual(delta, 0.0, accuracy: 1e-9)
  }

  func testPerplexityMetricKnownRelativeDelta() async throws {
    // Reference continuation [0]; reference assigns p=0.9 to it -> ppl 1/0.9 = 1.1111.
    // Candidate assigns p=0.5 -> ppl 2. Relative delta = (2 - 1.1111)/1.1111 = 0.8.
    let candidate = ScriptedDriver(tokens: [9], logprobs: [[0.0, 0.0]])
    let reference = ScriptedDriver(tokens: [0], logprobs: [[logf(0.9), logf(0.1)]])
    let delta = try await PerplexityMetric().measure(
      driver: candidate, reference: reference, prompts: [[7]], config: .greedy(maxTokens: 8))
    XCTAssertEqual(delta, 0.8, accuracy: 1e-4)
  }

  func testPerplexityPairPoolsAcrossPromptsAndExposesBothSides() async throws {
    // Two prompts, ScriptedDrivers replay the same rows each time: pooled ppl must equal the
    // single-prompt value (exp of pooled mean NLL, not a mean of per-prompt ppls).
    let candidate = ScriptedDriver(tokens: [9, 9], logprobs: [[0.0, 0.0], [0.0, 0.0]])
    let reference = ScriptedDriver(
      tokens: [0, 1], logprobs: [[logf(0.9), logf(0.1)], [logf(0.9), logf(0.1)]])
    let pair = try await teacherForcedPerplexities(
      driver: candidate, reference: reference, prompts: [[7], [8]], config: .greedy(maxTokens: 8))
    XCTAssertEqual(pair.candidate, 2.0, accuracy: 1e-4)          // p=0.5 every position
    XCTAssertEqual(pair.reference, 1.0 / 0.3, accuracy: 1e-3)    // sqrt(0.9*0.1) geometric mean
    XCTAssertEqual(pair.relativeDelta, (2.0 - 1.0 / 0.3) / (1.0 / 0.3), accuracy: 1e-4)
  }
}
