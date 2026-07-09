import XCTest
@testable import HarnessCore

final class KLTests: XCTestCase {
  func testKLZeroForIdentical() {
    let p: [Float] = [0.5,0.5]; XCTAssertEqual(klDivergence(reference: p, candidate: p), 0, accuracy: 1e-6)
  }
  func testKLPositiveAndKnown() {
    // KL(P=[0.9,0.1] || Q=[0.5,0.5]) ≈ 0.368 nats
    XCTAssertEqual(klDivergence(reference: [0.9,0.1], candidate: [0.5,0.5]), 0.368, accuracy: 1e-3)
  }
  func testKLClampsCandidateZeroToFinite() {
    // Candidate assigns ~0 probability where the reference assigns real mass: without the 1e-12
    // clamp this would be log(0) = -inf. Must stay finite: ~13.12 nats.
    let kl = klDivergence(reference: [0.5, 0.5], candidate: [1.0, 0.0])
    XCTAssertTrue(kl.isFinite, "expected finite KL, got \(kl)")
    XCTAssertEqual(kl, 13.12, accuracy: 1e-2)
  }
  func testKLSkipsZeroReferenceTerms() {
    // A reference entry of 0 contributes nothing (the `where p[i] > 0` guard) rather than a NaN
    // from 0 * log(0). Only the i=1 term contributes: 1.0 * (ln(1.0) - ln(0.5)) ≈ 0.6931.
    let kl = klDivergence(reference: [0.0, 1.0], candidate: [0.5, 0.5])
    XCTAssertTrue(kl.isFinite, "expected finite KL, got \(kl)")
    XCTAssertEqual(kl, 0.6931, accuracy: 1e-3)
  }
  func testKLDivergenceMetricMeasuresMedianAcrossScriptedDrivers() async throws {
    // End-to-end coverage of KLDivergenceMetric.measure, currently untested. logprobs are FULL-VOCAB,
    // index == token id (per the EngineDriver contract), so index-aligned KL is meaningful here.
    let candidate = ScriptedDriver(tokens: [1], logprobs: [
      [0.0, 0.0],      // position 0: uniform-ish after softmax -> some KL vs reference
      [10.0, 0.0],     // position 1: near one-hot on token 0
    ])
    let reference = ScriptedDriver(tokens: [1], logprobs: [
      [0.0, 0.0],      // position 0: identical to candidate -> KL ~= 0
      [0.0, 10.0],     // position 1: near one-hot on token 1 -> large KL vs candidate's token-0 spike
    ])
    let metric = KLDivergenceMetric()
    let median = try await metric.measure(driver: candidate, reference: reference, prompts: [[1, 2]], config: .greedy(maxTokens: 8))
    // Two positions: KL ~= 0 (position 0, identical distributions) and KL ~= 10 (position 1,
    // near-disjoint one-hot distributions). Proves the pipeline actually ran end to end and
    // produced a finite, non-trivially-large result — not that it picked a specific quantile.
    XCTAssertTrue(median.isFinite, "expected finite median KL, got \(median)")
    XCTAssertGreaterThan(median, 1.0)
  }
}
