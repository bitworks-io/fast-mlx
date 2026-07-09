import XCTest
@testable import SpikeCore

final class DecodeMetricsTests: XCTestCase {
    func testTtftAndDecodeRateFromStreamEvents() {
        // token events at t = 0.10 (first token), 0.20, 0.30, 0.40 seconds; prompt submitted at t=0
        let m = DecodeMetrics(submitTime: 0.0, tokenTimes: [0.10, 0.20, 0.30, 0.40])
        XCTAssertEqual(m.ttftSeconds, 0.10, accuracy: 1e-9)
        // decode rate excludes prefill: 3 inter-token gaps over 0.30s => 10 tok/s
        let rate = try! XCTUnwrap(m.decodeTokensPerSecond)
        XCTAssertEqual(rate, 10.0, accuracy: 1e-9)
        XCTAssertEqual(m.generatedTokenCount, 4)
    }

    func testSingleTokenHasNoDecodeRate() {
        let m = DecodeMetrics(submitTime: 0.0, tokenTimes: [0.10])
        XCTAssertNil(m.decodeTokensPerSecond)
    }
}
