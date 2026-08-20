import XCTest

@testable import ServingCore

final class ServingSamplingPolicyTests: XCTestCase {
    func testResolutionIsGreedyWhenTemperatureAbsentOrZero() throws {
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(temperature: nil, topP: nil, seed: nil),
            .greedy)
        // topP/seed are ignored on the greedy branch.
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(temperature: 0, topP: 0.5, seed: 7),
            .greedy)
    }

    func testResolutionIsSampledWhenTemperaturePositive() throws {
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: 0.9, seed: 42),
            .sampled(temperature: 0.7, topP: 0.9, topK: nil, minP: nil, seed: 42))
    }

    func testSampledDefaultsTopPToOneAndSeedToUnsetWhenAbsent() throws {
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(temperature: 1.2, topP: nil, seed: nil),
            .sampled(temperature: 1.2, topP: 1.0, topK: nil, minP: nil, seed: nil))
    }

    func testSampledCarriesTopKAndMinPWhenPresent() throws {
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(
                temperature: 0.7, topP: 0.9, topK: 40, minP: 0.05, seed: 42),
            .sampled(temperature: 0.7, topP: 0.9, topK: 40, minP: 0.05, seed: 42))
    }

    func testResolutionRejectsInvalidTopK() {
        XCTAssertThrowsError(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: nil, topK: 0, seed: nil)
        ) { error in
            XCTAssertEqual(error as? ServingSamplingPolicyError, .topKOutOfRange(0))
        }
        XCTAssertThrowsError(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: nil, topK: -1, seed: nil)
        ) { error in
            XCTAssertEqual(error as? ServingSamplingPolicyError, .topKOutOfRange(-1))
        }
    }

    func testResolutionRejectsInvalidMinP() {
        XCTAssertThrowsError(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: nil, minP: -0.1, seed: nil)
        ) { error in
            XCTAssertEqual(error as? ServingSamplingPolicyError, .minPOutOfRange(-0.1))
        }
        XCTAssertThrowsError(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: nil, minP: 1.5, seed: nil)
        ) { error in
            XCTAssertEqual(error as? ServingSamplingPolicyError, .minPOutOfRange(1.5))
        }
        XCTAssertThrowsError(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: nil, minP: .nan, seed: nil)
        ) { error in
            XCTAssertEqual(error as? ServingSamplingPolicyError, .nonFiniteMinP)
        }
    }

    func testResolutionRejectsInvalidTemperature() {
        assertThrows(
            try ServingSamplingPolicy.resolve(temperature: -0.1, topP: nil, seed: nil),
            .temperatureOutOfRange(-0.1))
        assertThrows(
            try ServingSamplingPolicy.resolve(temperature: 2.5, topP: nil, seed: nil),
            .temperatureOutOfRange(2.5))
        assertThrows(
            try ServingSamplingPolicy.resolve(temperature: .nan, topP: nil, seed: nil),
            .nonFiniteTemperature)
        assertThrows(
            try ServingSamplingPolicy.resolve(temperature: .infinity, topP: nil, seed: nil),
            .nonFiniteTemperature)
    }

    func testResolutionRejectsInvalidTopP() {
        assertThrows(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: -0.1, seed: nil),
            .topPOutOfRange(-0.1))
        assertThrows(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: 1.5, seed: nil),
            .topPOutOfRange(1.5))
        assertThrows(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: .nan, seed: nil),
            .nonFiniteTopP)
    }

    func testResolveFromRequestReadsTemperature() throws {
        let sampledRequest = OpenAIChatCompletionRequest(
            model: "qwen3-32b",
            messages: [.init(role: .user, text: "Hi")],
            maxCompletionTokens: nil,
            temperature: 0.7,
            choiceCount: 1,
            stream: false,
            stop: [])
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(from: sampledRequest),
            .sampled(temperature: 0.7, topP: 1.0, topK: nil, minP: nil, seed: nil))

        let sampledRequestWithExtras = OpenAIChatCompletionRequest(
            model: "qwen3-32b",
            messages: [.init(role: .user, text: "Hi")],
            maxCompletionTokens: nil,
            temperature: 0.7,
            choiceCount: 1,
            stream: false,
            stop: [],
            topP: 0.9,
            topK: 40,
            minP: 0.05,
            seed: 42)
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(from: sampledRequestWithExtras),
            .sampled(temperature: 0.7, topP: 0.9, topK: 40, minP: 0.05, seed: 42))

        let greedyRequest = OpenAIChatCompletionRequest(
            model: "qwen3-32b",
            messages: [.init(role: .user, text: "Hi")],
            maxCompletionTokens: nil,
            temperature: nil,
            choiceCount: 1,
            stream: false,
            stop: [])
        XCTAssertEqual(try ServingSamplingPolicy.resolve(from: greedyRequest), .greedy)
    }

    private func assertThrows(
        _ expression: @autoclosure () throws -> ServingSamplingPolicy,
        _ expected: ServingSamplingPolicyError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try expression(), file: file, line: line) { error in
            XCTAssertEqual(error as? ServingSamplingPolicyError, expected, file: file, line: line)
        }
    }
}
