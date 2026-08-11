import XCTest
@testable import HarnessCore

final class TriadTests: XCTestCase {
  func testIdenticalPrefix() {
    XCTAssertEqual(identicalPrefix([1,2,3,9], [1,2,3,4]), 3)
    XCTAssertEqual(identicalPrefix([1,2], [1,2]), 2)
  }
  func testEngagementDeltaRequiresStrictIncrease() {
    XCTAssertTrue(EngagementCheck(marker: "pld", floor: 1).passed(before: 4, after: 6))
    XCTAssertFalse(EngagementCheck(marker: "pld", floor: 1).passed(before: 6, after: 6)) // presence != engagement
  }
  func testAcceptanceFloor() {
    XCTAssertTrue(AcceptanceCheck(floor: 0.5).passed(rate: 0.66))
    XCTAssertFalse(AcceptanceCheck(floor: 0.5).passed(rate: 0.11))
  }
  func testVerdictFailsWhenEquivalenceFailsEvenIfEngaged() {
    // Regression: equivalenceOK must actually gate `passed` — a prior bug used a vacuous
    // `equivalencePrefix >= 0` check that was always true for a token count.
    let failing = TriadVerdict(equivalenceOK: false, engaged: true, acceptanceOK: true)
    XCTAssertFalse(failing.passed)
    let passing = TriadVerdict(equivalenceOK: true, engaged: true, acceptanceOK: true)
    XCTAssertTrue(passing.passed)
  }
  func testEquivalenceCheckDrivesVerdict() {
    // The caller computes equivalenceOK via EquivalenceCheck.evaluate(...).passed, not a raw prefix.
    let check = EquivalenceCheck(minPrefix: 30)
    let (_, ok) = check.evaluate(candidate: [1, 2, 3], reference: [1, 2, 9])
    XCTAssertFalse(ok)
    let verdict = TriadVerdict(equivalenceOK: ok, engaged: true, acceptanceOK: nil)
    XCTAssertFalse(verdict.passed)
  }

  // MARK: - lossy-tier triad mode (Task 4)

  func testTriadModeSelectsExactForFP16OrNilTier() {
    XCTAssertEqual(triadMode(forKVQuantTier: nil), .exact)
    XCTAssertEqual(triadMode(forKVQuantTier: "fp16"), .exact)
  }

  func testTriadModeSelectsLossyForNamedQuantTiers() {
    XCTAssertEqual(triadMode(forKVQuantTier: "2bit"), .lossy)
    XCTAssertEqual(triadMode(forKVQuantTier: "turbo4"), .lossy)
    XCTAssertEqual(triadMode(forKVQuantTier: "8"), .lossy)
  }

  func testCoherenceCanaryPassesWhenSubstringPresent() {
    let canary = CoherenceCanary.capitalOfFrance
    XCTAssertTrue(canary.passed("The capital of France is Paris, a city on the Seine."))
    XCTAssertFalse(canary.passed("The capital of France is London."))
  }

  func testLossyEquivalencePassesWhenAllThreeConditionsHold() {
    let check = LossyEquivalenceCheck(minPrefix: 1)
    let (passed, reasons) = check.evaluate(prefix: 5, allFinite: true, canaryPassed: true)
    XCTAssertTrue(passed)
    XCTAssertTrue(reasons.isEmpty)
  }

  func testLossyEquivalenceFailsOnZeroPrefixCrashSignal() {
    let check = LossyEquivalenceCheck(minPrefix: 1)
    let (passed, reasons) = check.evaluate(prefix: 0, allFinite: true, canaryPassed: true)
    XCTAssertFalse(passed)
    XCTAssertTrue(reasons.contains { $0.contains("crashed or produced no tokens") })
  }

  func testLossyEquivalenceFailsOnNonFiniteLogits() {
    let check = LossyEquivalenceCheck(minPrefix: 1)
    let (passed, reasons) = check.evaluate(prefix: 5, allFinite: false, canaryPassed: true)
    XCTAssertFalse(passed)
    XCTAssertTrue(reasons.contains { $0.contains("non-finite") })
  }

  func testLossyEquivalenceFailsOnCanaryMiss() {
    let check = LossyEquivalenceCheck(minPrefix: 1)
    let (passed, reasons) = check.evaluate(prefix: 5, allFinite: true, canaryPassed: false)
    XCTAssertFalse(passed)
    XCTAssertTrue(reasons.contains { $0.contains("coherence canary failed") })
  }

  func testLossyEquivalenceAccumulatesMultipleFailureReasons() {
    // At 2-bit today, a short prefix is EXPECTED (not the crash signal) — the point of the lossy
    // gate is that a short prefix alone must NOT fail the tier; only crash/NaN/canary do.
    let check = LossyEquivalenceCheck(minPrefix: 1)
    let (passed, reasons) = check.evaluate(prefix: 3, allFinite: false, canaryPassed: false)
    XCTAssertFalse(passed)
    XCTAssertEqual(reasons.count, 2, "prefix=3 >= minPrefix=1 should NOT itself contribute a reason")
  }

  func testLossyEquivalenceAcceptsShortPrefixThatWouldFailExactGate() {
    // The core motivation: at 2-bit, `EquivalenceCheck(minPrefix: 30)` fails a run that only
    // matched the reference for 3 tokens — expected loss, not a bug. The lossy gate must accept
    // this run when nothing else is wrong.
    let exact = EquivalenceCheck(minPrefix: 30)
    XCTAssertFalse(exact.evaluate(candidate: [1, 2, 3, 99], reference: [1, 2, 3, 4]).passed)
    let lossy = LossyEquivalenceCheck(minPrefix: 1)
    let (passed, _) = lossy.evaluate(prefix: 3, allFinite: true, canaryPassed: true)
    XCTAssertTrue(passed)
  }
}
