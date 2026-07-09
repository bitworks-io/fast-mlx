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
}
