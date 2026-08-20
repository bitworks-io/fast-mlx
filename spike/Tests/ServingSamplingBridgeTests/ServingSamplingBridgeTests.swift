import HarnessCore
import ServingCore
import XCTest

@testable import ServingSamplingBridge

final class ServingSamplingBridgeTests: XCTestCase {
    func testGreedyPolicyBridgesToGreedyContract() throws {
        let contract = try ServingSamplingBridge.contractPolicy(for: .greedy)
        XCTAssertEqual(contract, .greedy())
    }

    func testSampledPolicyBridgesToSampledContractExactly() throws {
        let contract = try ServingSamplingBridge.contractPolicy(
            for: .sampled(temperature: 0.7, topP: 0.9, topK: nil, minP: nil, seed: 42))
        let expected = try SamplingPolicyV1.sampled(
            temperature: 0.7, topP: 0.9, seed: .callerSupplied(42))
        XCTAssertEqual(contract, expected)
    }

    func testSampledWithoutSeedFailsClosed() {
        XCTAssertThrowsError(
            try ServingSamplingBridge.contractPolicy(
                for: .sampled(temperature: 0.7, topP: 0.9, topK: nil, minP: nil, seed: nil))
        ) { error in
            XCTAssertEqual(
                error as? ServingSamplingBridgeError, .seedRequiredForSampledRoute)
        }
    }

    func testContractRejectionSurfacesFromBridge() {
        // A `.sampled` constructed directly (bypassing resolve's range guard) with an
        // out-of-range temperature must surface the contract's own typed failure,
        // not crash — the bridge is the last line before the oracle.
        XCTAssertThrowsError(
            try ServingSamplingBridge.contractPolicy(
                for: .sampled(temperature: 3.0, topP: 1.0, topK: nil, minP: nil, seed: 5))
        ) { error in
            XCTAssertEqual(
                error as? SamplingContractFailure, .temperatureOutOfRange(3.0))
        }
    }
}
