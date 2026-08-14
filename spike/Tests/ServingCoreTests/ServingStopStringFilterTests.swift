import XCTest

@testable import ServingCore

final class ServingStopStringFilterTests: XCTestCase {
    func testStopStringSplitAcrossChunksIsWithheldAndStops() {
        var filter = ServingStopStringFilter(stopStrings: ["<stop>"])

        XCTAssertEqual(
            filter.process("hello<"),
            .init(text: "hello", stopped: false))
        XCTAssertEqual(
            filter.process("stop"),
            .init(text: nil, stopped: false))
        XCTAssertEqual(
            filter.process(">hidden"),
            .init(text: nil, stopped: true))
        XCTAssertEqual(
            filter.process("tail"),
            .init(text: nil, stopped: true))
        XCTAssertNil(filter.finish())
    }

    func testEarliestStopWinsAndTextAfterStopIsNeverEmitted() {
        var filter = ServingStopStringFilter(stopStrings: ["END", "STOP"])

        XCTAssertEqual(
            filter.process("visibleENDhiddenSTOP"),
            .init(text: "visible", stopped: true))
        XCTAssertNil(filter.finish())
    }

    func testAmbiguousSuffixIsBoundedAndFlushedAtGenerationEnd() {
        var filter = ServingStopStringFilter(stopStrings: ["<stop>", "<stopped>"])
        let prefix = String(repeating: "x", count: 4_096)

        XCTAssertEqual(
            filter.process(prefix + "<sto"),
            .init(text: prefix, stopped: false))
        XCTAssertEqual(filter.bufferedCharacterCount, 4)
        XCTAssertEqual(filter.finish(), "<sto")
        XCTAssertEqual(filter.bufferedCharacterCount, 0)
    }

    func testNoStopsPassesChunksThroughWithoutBuffering() {
        var filter = ServingStopStringFilter(stopStrings: [])

        XCTAssertEqual(
            filter.process("visible"),
            .init(text: "visible", stopped: false))
        XCTAssertEqual(filter.bufferedCharacterCount, 0)
        XCTAssertNil(filter.finish())
    }
}
