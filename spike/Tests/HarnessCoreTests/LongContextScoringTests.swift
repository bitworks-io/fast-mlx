import XCTest
@testable import HarnessCore

/// A driver that only implements the base teacher-forced `logprobs` (not the sampled overload),
/// so it exercises the protocol extension's DEFAULT implementation: compute the full result, then
/// filter to the requested positions. Proves the default is behaviorally correct (not necessarily
/// memory-saving — real drivers override for that).
private struct FullOnlyDriver: EngineDriver {
    let rows: [[Float]]
    func generate(prompt: [Int], config: RunConfig) async throws -> RunResult { RunResult(tokens: []) }
    func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] { [] }
    func logprobs(prompt: [Int], forcedContinuation: [Int], config: RunConfig) async throws -> [[Float]] { rows }
}

/// A driver that overrides the sampled variant directly and asserts it is NEVER asked for the
/// full unsampled result — proving a memory-conscious override actually gets used instead of the
/// default filter-after-the-fact path.
private struct SampledOnlyDriver: EngineDriver {
    let rowsByPosition: [Int: [Float]]
    func generate(prompt: [Int], config: RunConfig) async throws -> RunResult { RunResult(tokens: []) }
    func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] { [] }
    func logprobs(prompt: [Int], forcedContinuation: [Int], config: RunConfig) async throws -> [[Float]] {
        XCTFail("must not call the full-materialization overload when the sampled overload is available")
        return []
    }
    func logprobs(prompt: [Int], forcedContinuation: [Int], atPositions positions: [Int], config: RunConfig) async throws -> [[Float]] {
        positions.map { rowsByPosition[$0]! }
    }
}

final class LongContextScoringTests: XCTestCase {
    func testDefaultSampledOverloadFiltersFullResult() async throws {
        let rows: [[Float]] = (0..<10).map { [Float($0), Float(-$0)] }
        let driver = FullOnlyDriver(rows: rows)
        let sampled = try await driver.logprobs(prompt: [0], forcedContinuation: Array(0..<10), atPositions: [0, 4, 9], config: .greedy(maxTokens: 10))
        XCTAssertEqual(sampled, [rows[0], rows[4], rows[9]])
    }

    func testSampledOverrideIsUsedInsteadOfDefault() async throws {
        let driver = SampledOnlyDriver(rowsByPosition: [0: [1, 0], 4: [2, 0], 9: [3, 0]])
        let sampled = try await driver.logprobs(prompt: [0], forcedContinuation: Array(0..<10), atPositions: [0, 4, 9], config: .greedy(maxTokens: 10))
        XCTAssertEqual(sampled, [[1, 0], [2, 0], [3, 0]])
    }

    func testTeacherForcedScoresAtSampledPositionsScoresOnlyRequestedPositions() async throws {
        // continuation of 6 tokens over a 2-token vocab (0/1), so forcedTokens are valid row indices.
        let continuation = [0, 1, 0, 1, 0, 1]
        let positions = evenlySpacedPositions(total: continuation.count, sampleSize: 3)
        XCTAssertEqual(positions, [0, 3, 5]) // ascending, endpoints included

        let candidateRows: [Int: [Float]] = [0: [1, 0], 3: [0, 1], 5: [1, 1]]
        let referenceRows: [Int: [Float]] = [0: [1, 0], 3: [1, 0], 5: [0, 1]]
        let candidate = SampledOnlyDriver(rowsByPosition: candidateRows)
        let reference = SampledOnlyDriver(rowsByPosition: referenceRows)

        let scored = try await teacherForcedScoresAtSampledPositions(
            driver: candidate, reference: reference,
            prompt: [1], continuation: continuation, positions: positions,
            config: .greedy(maxTokens: 8))

        XCTAssertEqual(scored.positions, positions)
        XCTAssertEqual(scored.forcedTokens, [0, 1, 1]) // continuation[p] at each sampled position
        XCTAssertEqual(scored.candidateRows.count, 3)
        XCTAssertEqual(scored.referenceRows.count, 3)

        // KL/perplexity math reuses the same per-position helpers as the short-prompt path.
        let kls = perPositionKLs(reference: scored.referenceRows, candidate: scored.candidateRows)
        XCTAssertEqual(kls.count, 3)
        XCTAssertTrue(kls.allSatisfy { $0.isFinite })
        let nll = meanNLL(rows: scored.candidateRows, tokens: scored.forcedTokens)
        XCTAssertTrue(nll.isFinite)
    }

    func testTeacherForcedScoresAtSampledPositionsThrowsOnEmptyContinuation() async throws {
        let driver = SampledOnlyDriver(rowsByPosition: [:])
        do {
            _ = try await teacherForcedScoresAtSampledPositions(
                driver: driver, reference: driver, prompt: [1], continuation: [], positions: [],
                config: .greedy(maxTokens: 8))
            XCTFail("expected QualityMetricError.emptyContinuation")
        } catch let error as QualityMetricError {
            guard case .emptyContinuation = error else { return XCTFail("expected emptyContinuation, got \(error)") }
        }
    }

    func testTeacherForcedScoresAtSampledPositionsThrowsOnRowCountMismatch() async throws {
        struct ShortDriver: EngineDriver {
            func generate(prompt: [Int], config: RunConfig) async throws -> RunResult { RunResult(tokens: []) }
            func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] { [] }
            func logprobs(prompt: [Int], forcedContinuation: [Int], config: RunConfig) async throws -> [[Float]] { [] }
            func logprobs(prompt: [Int], forcedContinuation: [Int], atPositions positions: [Int], config: RunConfig) async throws -> [[Float]] {
                [[0, 0]] // always 1 row, regardless of how many positions were asked for
            }
        }
        let driver = ShortDriver()
        do {
            _ = try await teacherForcedScoresAtSampledPositions(
                driver: driver, reference: driver, prompt: [1], continuation: [1, 2, 3], positions: [0, 1, 2],
                config: .greedy(maxTokens: 8))
            XCTFail("expected QualityMetricError.rowCountMismatch")
        } catch let error as QualityMetricError {
            guard case .rowCountMismatch = error else { return XCTFail("expected rowCountMismatch, got \(error)") }
        }
    }
}
