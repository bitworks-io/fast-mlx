import XCTest
@testable import HarnessCore

/// Test driver whose teacher-forced logprobs are KEYED BY the forced continuation: it only
/// yields rows when handed the exact continuation it was scripted for, and throws otherwise.
/// This proves the metric passes the REFERENCE's greedy continuation to BOTH sides — not the
/// candidate's own path, and not free-running argmax.
private struct KeyedForcedDriver: EngineDriver {
    struct UnexpectedContinuation: Error { let got: [Int] }
    let tokens: [Int]
    let rowsByContinuation: [[Int]: [[Float]]]
    func generate(prompt: [Int], config: RunConfig) async throws -> RunResult {
        RunResult(tokens: tokens)
    }
    func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] { [] }
    func logprobs(prompt: [Int], forcedContinuation: [Int], config: RunConfig) async throws -> [[Float]] {
        guard let rows = rowsByContinuation[forcedContinuation] else {
            throw UnexpectedContinuation(got: forcedContinuation)
        }
        return rows
    }
}

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
  func testTeacherForcedTop1AgreementComparesCandidateAndReferenceRows() throws {
    let agreement = try teacherForcedTop1Agreement(
      candidate: [[0, 3, 1], [4, 0, 1], [0, 2, 1]],
      reference: [[0, 2, 1], [0, 4, 1], [0, 3, 1]])
    XCTAssertEqual(agreement.matches, 2)
    XCTAssertEqual(agreement.scoredPositions, 3)
    XCTAssertEqual(agreement.rate, 2.0 / 3.0, accuracy: 1e-12)
  }
  func testTeacherForcedTop1AgreementRejectsShapeMismatch() {
    XCTAssertThrowsError(try teacherForcedTop1Agreement(
      candidate: [[0, 1]], reference: [[0, 1], [1, 0]]))
    XCTAssertThrowsError(try teacherForcedTop1Agreement(
      candidate: [[0, 1]], reference: [[0, 1, 2]]))
  }
  func testTeacherForcedTop1AgreementRejectsNonFiniteLogits() {
    XCTAssertThrowsError(try teacherForcedTop1Agreement(
      candidate: [[0, .nan]], reference: [[0, 1]]))
    XCTAssertThrowsError(try teacherForcedTop1Agreement(
      candidate: [[0, 1]], reference: [[0, .infinity]]))
  }
  func testKLDivergenceMetricMeasuresMedianAcrossScriptedDrivers() async throws {
    // End-to-end coverage of KLDivergenceMetric.measure. logprobs are FULL-VOCAB, index ==
    // token id (per the EngineDriver contract), so index-aligned KL is meaningful here.
    // The metric is TEACHER-FORCED: the continuation length (reference.generate tokens, 2)
    // must match the scored row count (2) or the metric throws.
    let candidate = ScriptedDriver(tokens: [1, 2], logprobs: [
      [0.0, 0.0],      // position 0: uniform-ish after softmax -> some KL vs reference
      [10.0, 0.0],     // position 1: near one-hot on token 0
    ])
    let reference = ScriptedDriver(tokens: [1, 2], logprobs: [
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

  // MARK: teacher-forced metric definition (context-locked KL)

  func testTeacherForcedMetricFeedsReferenceContinuationToBothSidesAndScoresAllPositions() async throws {
    // The reference's greedy continuation is [3, 1]. BOTH drivers only yield rows for exactly
    // that continuation (KeyedForcedDriver throws on anything else), so a passing test proves:
    //  (a) the metric obtained the continuation from the REFERENCE (candidate.generate returns
    //      [9, 9] — using it would throw),
    //  (b) both sides were scored TEACHER-FORCED on it (free-running logprobs return [] — using
    //      them would fail the row-count contract),
    //  (c) KL is computed over ALL forced positions with a hand-computable value.
    let continuation = [3, 1]
    // Logit rows are ln(p): softmax reproduces p exactly, so KLs are hand-computable.
    let uniform: [Float] = [logf(0.5), logf(0.5)]
    let skewed: [Float] = [logf(0.9), logf(0.1)]
    let candidate = KeyedForcedDriver(
      tokens: [9, 9], // NOT the continuation — the metric must not use the candidate's path
      rowsByContinuation: [continuation: [uniform, uniform]])
    let reference = KeyedForcedDriver(
      tokens: continuation,
      rowsByContinuation: [continuation: [uniform, skewed]])
    let metric = KLDivergenceMetric()
    let median = try await metric.measure(
      driver: candidate, reference: reference, prompts: [[7]], config: .greedy(maxTokens: 8))
    // Position 0: identical distributions -> KL 0. Position 1: KL([0.9,0.1] || [0.5,0.5]) ~=
    // 0.368 nats. medianOf takes the upper-middle of 2 sorted values -> 0.368.
    XCTAssertEqual(median, 0.368, accuracy: 1e-3)
  }

  func testTeacherForcedMetricThrowsOnRowCountMismatch() async throws {
    // A driver that returns FEWER rows than the forced continuation is not fulfilling the
    // teacher-forced contract; silently scoring a truncated matrix would understate divergence.
    let continuation = [3, 1]
    let row: [Float] = [0.0, 0.0]
    let candidate = KeyedForcedDriver(
      tokens: [9], rowsByContinuation: [continuation: [row]]) // 1 row for a 2-token continuation
    let reference = KeyedForcedDriver(
      tokens: continuation, rowsByContinuation: [continuation: [row, row]])
    let metric = KLDivergenceMetric()
    do {
      _ = try await metric.measure(
        driver: candidate, reference: reference, prompts: [[7]], config: .greedy(maxTokens: 8))
      XCTFail("expected QualityMetricError.rowCountMismatch")
    } catch let error as QualityMetricError {
      guard case .rowCountMismatch = error else {
        return XCTFail("expected rowCountMismatch, got \(error)")
      }
    }
  }

  func testTeacherForcedMetricThrowsOnEmptyContinuation() async throws {
    // A reference that generates nothing gives zero scoreable positions — a "0.0 KL" here would
    // be a vacuous pass, so the metric must refuse.
    let empty = KeyedForcedDriver(tokens: [], rowsByContinuation: [:])
    let metric = KLDivergenceMetric()
    do {
      _ = try await metric.measure(
        driver: empty, reference: empty, prompts: [[7]], config: .greedy(maxTokens: 8))
      XCTFail("expected QualityMetricError.emptyContinuation")
    } catch let error as QualityMetricError {
      guard case .emptyContinuation = error else {
        return XCTFail("expected emptyContinuation, got \(error)")
      }
    }
  }
}
