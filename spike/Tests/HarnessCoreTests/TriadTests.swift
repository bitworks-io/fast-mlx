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
}
