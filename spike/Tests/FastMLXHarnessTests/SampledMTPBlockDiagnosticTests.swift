import XCTest

@testable import MLXLMCommon

final class SampledMTPBlockDiagnosticTests: XCTestCase {
    func testFullVocabularyNormalizationClosesAccumulationResidual() {
        let raw = [Double](repeating: 1, count: 248_320)

        let normalized = normalizeSampledMTPBlockProbabilities(raw)

        XCTAssertEqual(normalized.count, raw.count)
        XCTAssertTrue(normalized.allSatisfy { $0.isFinite && $0 >= 0 })
        XCTAssertLessThanOrEqual(abs(normalized.reduce(0, +) - 1), 1e-12)
    }
}
