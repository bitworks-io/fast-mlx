import XCTest
@testable import HarnessCore

final class BenchMatrixTests: XCTestCase {
  func testDropsWarmupAndAverages() {
    let agg = aggregateRates([nil, 100.0, 102.0, 104.0]) // index 0 = warmup, dropped; nil = skipped
    XCTAssertEqual(agg.mean, 102.0, accuracy: 1e-9); XCTAssertEqual(agg.runs, 3)
  }
  func testDropsWarmupEvenWhenItIsARealValue() {
    // Index 0 must be dropped unconditionally as the warmup run, not just when it happens to be nil.
    let agg = aggregateRates([99.0, 100.0, 102.0, 104.0])
    XCTAssertEqual(agg.mean, 102.0, accuracy: 1e-9); XCTAssertEqual(agg.runs, 3)
  }
}
