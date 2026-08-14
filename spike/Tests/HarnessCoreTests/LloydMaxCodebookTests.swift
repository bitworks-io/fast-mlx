import XCTest
@testable import HarnessCore

final class LloydMaxCodebookTests: XCTestCase {
    func testB1MatchesSqrt2OverPi() {
        let c = LloydMaxCodebook.gaussian(bits: 1)
        XCTAssertEqual(c.count, 2)
        let expected = (2.0 / Double.pi).squareRoot() // 0.79788
        XCTAssertEqual(c[0], -expected, accuracy: 1e-4)
        XCTAssertEqual(c[1], expected, accuracy: 1e-4)
    }

    func testB2MatchesMax1960Levels() {
        let c = LloydMaxCodebook.gaussian(bits: 2)
        XCTAssertEqual(c.count, 4)
        let expected = [-1.5104, -0.4528, 0.4528, 1.5104]
        for (a, e) in zip(c, expected) { XCTAssertEqual(a, e, accuracy: 2e-3) }
    }

    func testMonotoneAndSymmetric() {
        let c = LloydMaxCodebook.gaussian(bits: 3)
        XCTAssertEqual(c.count, 8)
        XCTAssertEqual(c, c.sorted(), "centroids must be ascending")
        for i in 0..<4 { XCTAssertEqual(c[i], -c[7 - i], accuracy: 1e-3, "symmetric about 0") }
    }
}
