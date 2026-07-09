import XCTest
@testable import HarnessCore

final class KLTests: XCTestCase {
  func testKLZeroForIdentical() {
    let p: [Float] = [0.5,0.5]; XCTAssertEqual(klDivergence(reference: p, candidate: p), 0, accuracy: 1e-6)
  }
  func testKLPositiveAndKnown() {
    // KL(P=[0.9,0.1] || Q=[0.5,0.5]) ≈ 0.368 nats
    XCTAssertEqual(klDivergence(reference: [0.9,0.1], candidate: [0.5,0.5]), 0.368, accuracy: 1e-3)
  }
}
