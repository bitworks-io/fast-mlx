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

  func testServiceWorkloadIdentityPinsTheSamePromptAcrossPolicyProcesses() throws {
    let batchIdentity = try ServiceWorkloadIdentity(nonce: "frontier-20260714")
    let soloIdentity = try ServiceWorkloadIdentity(nonce: "frontier-20260714")

    XCTAssertEqual(
      batchIdentity.prompt(basePrompt: "serve", run: 2, request: 3),
      soloIdentity.prompt(basePrompt: "serve", run: 2, request: 3))
    XCTAssertEqual(
      batchIdentity.prompt(basePrompt: "serve", run: 2, request: 3),
      "serve [run=2 nonce=frontier-20260714] [request=3]")
    XCTAssertNotEqual(
      batchIdentity.prompt(basePrompt: "serve", run: 1, request: 3),
      batchIdentity.prompt(basePrompt: "serve", run: 2, request: 3))
    XCTAssertNotEqual(
      batchIdentity.prompt(basePrompt: "serve", run: 2, request: 2),
      batchIdentity.prompt(basePrompt: "serve", run: 2, request: 3))
  }

  func testServiceWorkloadIdentityRejectsUnstableOrPromptChangingNonceValues() {
    for nonce in [
      "", "   ", "contains space", "contains,comma", "-looks-like-a-flag",
      String(repeating: "x", count: 65),
    ] {
      XCTAssertThrowsError(try ServiceWorkloadIdentity(nonce: nonce)) {
        XCTAssertEqual($0 as? ServiceWorkloadIdentityError, .invalidNonce)
      }
    }
  }

  func testCLIFlagsFailClosedWhenAValueIsMissingOrFollowedByAnotherFlag() throws {
    let trailing = CLIFlags(["--workload-nonce"])
    XCTAssertThrowsError(
      try trailing.strictString("workload-nonce", default: "generated")) {
        XCTAssertEqual($0 as? FlagValueError, .missingValue(key: "workload-nonce"))
      }

    let followedByFlag = CLIFlags([
      "--workload-nonce", "--csv", "frontier.csv",
    ])
    XCTAssertThrowsError(
      try followedByFlag.strictString("workload-nonce", default: "generated")) {
        XCTAssertEqual($0 as? FlagValueError, .missingValue(key: "workload-nonce"))
      }
    XCTAssertEqual(followedByFlag.string("csv"), "frontier.csv")

    let explicit = CLIFlags(["--workload-nonce", "shared-identity"])
    XCTAssertEqual(
      try explicit.strictString("workload-nonce", default: "generated"),
      "shared-identity")
  }
}
