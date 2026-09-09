import XCTest

@testable import fastmlx_harness

/// Regression coverage for `rowsHaveNoCorruptValues` (extracted from the lossy-equivalence gate
/// at `Harness.swift:runVerify`, formerly an inline `allFinite` closure at line ~401). The gate
/// scans full-vocabulary LOGPROBS rows. Some checkpoints (e.g. `qwen4_exp` / Qwen3.8-Flash-Next)
/// mask unsupported media-sentinel vocabulary indices to `-Float.infinity` on every forward, so a
/// masked token's logprob is legitimately `-inf` (probability exactly 0) and must be ACCEPTED.
/// `NaN` and `+infinity` remain corruption and must be REJECTED. Every expected value below is
/// hand-derived in the adjacent comment, never produced by calling the function under test.
final class RowsHaveNoCorruptValuesTests: XCTestCase {

    // MARK: - case 1: regression test for the defect (-inf alongside ordinary finite values)

    func testRowWithNegativeInfinityAlongsideFiniteValuesIsAccepted() {
        // A masked-token row: three ordinary negative logprobs plus one masked (-inf) index.
        // None of the four values is NaN or +inf, so by the stated rule the row -- and the
        // single-row set containing it -- must be ACCEPTED (expected: true).
        let rows: [[Float]] = [[-0.5, -3.25, -Float.infinity, -1.0]]

        XCTAssertTrue(rowsHaveNoCorruptValues(rows))
    }

    // MARK: - case 2: NaN is rejected

    func testRowWithNaNIsRejected() {
        // A NaN anywhere in a row is corruption regardless of any other value in the row.
        // Expected: false.
        let rows: [[Float]] = [[-0.5, Float.nan, -1.0]]

        XCTAssertFalse(rowsHaveNoCorruptValues(rows))
    }

    // MARK: - case 3: +infinity is rejected

    func testRowWithPositiveInfinityIsRejected() {
        // A logprob can never exceed 0 (probability <= 1), so +infinity is impossible and must
        // be treated as corruption. Expected: false.
        let rows: [[Float]] = [[-0.5, Float.infinity, -1.0]]

        XCTAssertFalse(rowsHaveNoCorruptValues(rows))
    }

    // MARK: - case 4: happy-path control (all-finite ordinary row)

    func testAllFiniteOrdinaryRowIsAccepted() {
        // No NaN, no +inf, no -inf anywhere -- the unambiguous legitimate case. Expected: true.
        // This is the explicit success-path control: a bug that made the predicate always
        // return false would fail THIS test with the same symptom as a correct refusal, so it
        // must be asserted independently of the -inf regression case above.
        let rows: [[Float]] = [[-0.1, -2.0, -5.75]]

        XCTAssertTrue(rowsHaveNoCorruptValues(rows))
    }

    // MARK: - case 5: multi-row semantics

    func testAnyCorruptRowRejectsTheWholeSet() {
        // Row 0 is clean; row 1 contains a NaN. "Any row corrupt" must reject the whole set,
        // even though row 0 alone would pass. Expected: false.
        let rows: [[Float]] = [
            [-0.2, -1.0],
            [-0.3, Float.nan],
        ]

        XCTAssertFalse(rowsHaveNoCorruptValues(rows))
    }

    func testOnlyNegativeInfinityAcrossSeveralRowsIsAccepted() {
        // Three rows, each containing a mix of ordinary finite logprobs and masked (-inf)
        // entries, and nothing else. No row contains NaN or +inf, so the whole set must be
        // ACCEPTED. Expected: true.
        let rows: [[Float]] = [
            [-0.4, -Float.infinity],
            [-Float.infinity, -Float.infinity, -0.05],
            [-2.2, -0.9, -Float.infinity],
        ]

        XCTAssertTrue(rowsHaveNoCorruptValues(rows))
    }
}
