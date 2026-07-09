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

  func testBenchRowCSVIncludesHardwareColumn() {
    // Spec §6.3's mandated durable-evidence dimensions: label, mode, concurrency, model, hardware.
    // label/mode/concurrency/model were already present; hardware was the gap (Task 5).
    let row = BenchRow(
      label: "harness", workload: .decode, mode: .none, model: "Qwen3-32B-4bit",
      decodeTokS: 42.5, ttftMs: 120.3, quant: "int4", concurrency: 1, hardware: "Apple M3 Ultra")
    XCTAssertTrue(BenchRow.csvHeader.hasSuffix(",hardware"))
    XCTAssertTrue(row.csvLine.hasSuffix(",Apple M3 Ultra"))
    XCTAssertEqual(row.csvLine, "harness,decode,none,Qwen3-32B-4bit,42.5,120.3,int4,1,Apple M3 Ultra")
  }
}
