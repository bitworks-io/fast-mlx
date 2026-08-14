import Foundation
import XCTest
@testable import HarnessCore

final class KVTunerModelConfigPreflightTests: XCTestCase {
    func testLoadsPositiveNumHiddenLayersWithoutFilesystemIO() throws {
        let data = Data(
            #"{"model_type":"qwen3","num_hidden_layers":64}"#.utf8)

        XCTAssertEqual(
            try KVTunerModelConfigPreflight.load(from: data),
            64)
    }

    func testMissingMalformedOrNonIntegerLayerCountFailsClosed() {
        for data in [
            Data(#"{"model_type":"qwen3"}"#.utf8),
            Data(#"{"num_hidden_layers":null}"#.utf8),
            Data(#"{"num_hidden_layers":"64"}"#.utf8),
            Data(#"{"num_hidden_layers":64.5}"#.utf8),
            Data("{".utf8),
        ] {
            XCTAssertThrowsError(
                try KVTunerModelConfigPreflight.load(from: data))
        }
    }

    func testZeroOrNegativeLayerCountFailsClosed() {
        for count in [0, -1] {
            XCTAssertThrowsError(
                try KVTunerModelConfigPreflight.load(
                    from: Data(
                        "{\"num_hidden_layers\":\(count)}".utf8)))
        }
    }
}
