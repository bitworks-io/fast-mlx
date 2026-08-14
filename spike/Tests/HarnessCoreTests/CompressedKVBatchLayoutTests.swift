import XCTest

@testable import HarnessCore

final class CompressedKVBatchLayoutTests: XCTestCase {
    func testUnequalMergePinsPhysicalEndAndEndAlignsRows() throws {
        let layout = try CompressedKVBatchLayout(
            logicalOffsets: [32, 48, 24],
            capacities: [48, 48, 48])

        XCTAssertEqual(layout.allocationCapacity, 48)
        XCTAssertEqual(layout.physicalWrittenEnd, 48)
        XCTAssertEqual(layout.logicalOffsets, [32, 48, 24])
        XCTAssertEqual(layout.leftPadding, [16, 0, 24])
        XCTAssertEqual(
            layout.physicalValidRanges,
            [16 ..< 48, 0 ..< 48, 24 ..< 48])
    }

    func testRemovingLongestBoundaryRowRetainsPhysicalEndAndPlansAppendAtIt() throws {
        let merged = try CompressedKVBatchLayout(
            logicalOffsets: [32, 48, 24],
            capacities: [48, 48, 48])

        let filtered = try merged.filtering(keeping: [0, 2])
        let append = try filtered.planAppend(tokenCount: 3)

        XCTAssertEqual(filtered.logicalOffsets, [32, 24])
        XCTAssertEqual(filtered.leftPadding, [16, 24])
        XCTAssertEqual(filtered.physicalWrittenEnd, 48)
        XCTAssertEqual(append.physicalRange, 48 ..< 51)
        XCTAssertEqual(append.requiredCapacity, 51)
        XCTAssertEqual(append.result.physicalWrittenEnd, 51)
        XCTAssertEqual(append.result.logicalOffsets, [35, 27])
        XCTAssertEqual(append.result.leftPadding, [16, 24])
    }

    func testMaskPlanUsesPhysicalEndPlusIncomingWidthAfterBoundaryRemoval() throws {
        let merged = try CompressedKVBatchLayout(
            logicalOffsets: [32, 48, 24],
            capacities: [48, 48, 48])
        let filtered = try merged.filtering(keeping: [0, 2])
        let grown = try filtered.growing(to: 51)
        let mask = try grown.maskPlan(incomingTokens: 3)

        XCTAssertEqual(mask.shape, [2, 1, 3, 51])
        XCTAssertEqual(mask.physicalWrittenExtent, 51)
        XCTAssertEqual(
            mask.allowedPhysicalRanges,
            [
                [16 ..< 49, 16 ..< 50, 16 ..< 51],
                [24 ..< 49, 24 ..< 50, 24 ..< 51],
            ])
    }

    func testInvalidBatchLayoutTransitionsFailWithoutMutation() throws {
        XCTAssertThrowsError(
            try CompressedKVBatchLayout(logicalOffsets: [], capacities: []))
        XCTAssertThrowsError(
            try CompressedKVBatchLayout(
                logicalOffsets: [1, 2], capacities: [2]))
        XCTAssertThrowsError(
            try CompressedKVBatchLayout(
                logicalOffsets: [-1], capacities: [2]))
        XCTAssertThrowsError(
            try CompressedKVBatchLayout(
                logicalOffsets: [3], capacities: [2]))

        let original = try CompressedKVBatchLayout(
            logicalOffsets: [32, 48, 24],
            capacities: [48, 48, 48])

        for selection in [[], [0, 0], [3], [-1]] {
            XCTAssertThrowsError(try original.filtering(keeping: selection))
            XCTAssertEqual(
                original,
                try CompressedKVBatchLayout(
                    logicalOffsets: [32, 48, 24],
                    capacities: [48, 48, 48]))
        }
        for tokenCount in [0, -1] {
            XCTAssertThrowsError(try original.planAppend(tokenCount: tokenCount))
            XCTAssertEqual(original.physicalWrittenEnd, 48)
        }
        XCTAssertThrowsError(try original.growing(to: 47))

        let int32Boundary = try CompressedKVBatchLayout(
            logicalOffsets: [Int(Int32.max)],
            capacities: [Int(Int32.max)])
        XCTAssertThrowsError(try int32Boundary.planAppend(tokenCount: 1))
        XCTAssertEqual(int32Boundary.physicalWrittenEnd, Int(Int32.max))
    }
}
