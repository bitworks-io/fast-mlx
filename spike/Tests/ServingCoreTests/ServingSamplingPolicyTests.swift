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

    func testResolutionRejectsZeroTopPAtPositiveTemperature() {
        // top_p == 0 must be refused, not silently widened: downstream
        // (GenerateParameters/TopPSampler) disables top-p truncation entirely
        // when topP == 0, which is the opposite of "narrowest possible nucleus"
        // that a caller asking for top_p: 0 actually means.
        assertThrows(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: 0, seed: nil),
            .topPOutOfRange(0))
    }

    func testResolutionRejectsTopPTooSmallToTruncate() {
        // 1e-9 is inside the measured-degenerate band at production vocabulary
        // width: the nucleus threshold (1 - topP) saturates to exactly 1.0 in
        // float32, so the top-p filter can never keep anything.
        assertThrows(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: 1e-9, seed: nil),
            .topPTooSmallToTruncate(1e-9))
    }

    func testResolutionAcceptsSmallestUsableTopPAndRejectsSaturatingBand() throws {
        // 1e-7 is measured usable at production vocabulary width (V=151936)
        // and must resolve, not be refused: it sits above the saturation
        // boundary and still discriminates a real nucleus.
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: 1e-7, seed: nil),
            .sampled(temperature: 0.7, topP: 1e-7, topK: nil, minP: nil, seed: nil))

        // A value in the saturating band (1 - topP rounds to exactly 1.0 in
        // float32) must be refused, proving the guard discriminates rather
        // than refusing everything small.
        assertThrows(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: 2e-8, seed: nil),
            .topPTooSmallToTruncate(2e-8))
    }

    func testResolutionKeepsDeployedPresetUnchanged() throws {
        // Regression that matters most: production traffic (temperature 1.0,
        // topP 0.95, topK 20, minP 0) must resolve exactly as before.
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(
                temperature: 1.0, topP: 0.95, topK: 20, minP: 0, seed: 42),
            .sampled(temperature: 1.0, topP: 0.95, topK: 20, minP: 0, seed: 42))
    }

    func testResolutionZeroTopPOnGreedyBranchDoesNotThrow() throws {
        // The greedy branch returns before any top_p validation runs.
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(temperature: nil, topP: 0, seed: nil),
            .greedy)
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(temperature: 0, topP: 0, seed: nil),
            .greedy)
    }

    func testResolutionTopPNilStillDefaultsToOne() throws {
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(temperature: 0.7, topP: nil, seed: nil),
            .sampled(temperature: 0.7, topP: 1.0, topK: nil, minP: nil, seed: nil))
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

    func testDefaultsFillEveryOmittedFieldWhenTemperatureAbsent() throws {
        // A param-less request plus artifact-sourced defaults resolves as
        // sampled, carrying exactly the default values.
        let defaults = ServingSamplingDefaults(temperature: 1.0, topP: 0.95, topK: 20, minP: 0)
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(
                temperature: nil, topP: nil, topK: nil, minP: nil, seed: nil, defaults: defaults),
            .sampled(temperature: 1.0, topP: 0.95, topK: 20, minP: 0, seed: nil))
    }

    func testExplicitZeroTemperatureStaysGreedyEvenWithDefaults() throws {
        // Explicit temperature: 0 is the client's escape hatch to the greedy
        // speculative-decoding path; defaults must never override it.
        let defaults = ServingSamplingDefaults(temperature: 1.0, topP: 0.95, topK: 20, minP: 0)
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(
                temperature: 0, topP: nil, topK: nil, minP: nil, seed: nil, defaults: defaults),
            .greedy)
    }

    func testNilDefaultsPreservesTodaysGreedyBehaviorOnParamlessRequest() throws {
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(
                temperature: nil, topP: nil, topK: nil, minP: nil, seed: nil, defaults: nil),
            .greedy)
    }

    func testExplicitFieldsWinOverDefaultsWhileOmittedFieldsAreDefaulted() throws {
        // Anti-vacuity control: defaults must not blindly overwrite fields the
        // request actually supplied.
        let defaults = ServingSamplingDefaults(temperature: 1.0, topP: 0.95, topK: 20, minP: 0)
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(
                temperature: nil, topP: 0.5, topK: nil, minP: nil, seed: nil, defaults: defaults),
            .sampled(temperature: 1.0, topP: 0.5, topK: 20, minP: 0, seed: nil))
    }

    func testExplicitNonzeroTemperatureWinsOverTheDefaultTemperature() throws {
        // The topP override is covered above, but `temperature` is the one field
        // with special nil-versus-explicit-0 handling, so its override path is
        // pinned separately rather than assumed to follow from topP's.
        let defaults = ServingSamplingDefaults(temperature: 1.0, topP: 0.95, topK: 20, minP: 0)
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(
                temperature: 0.7, topP: nil, topK: nil, minP: nil, seed: nil, defaults: defaults),
            .sampled(temperature: 0.7, topP: 0.95, topK: 20, minP: 0, seed: nil))
    }

    func testDefaultedValuesAreValidatedExactlyLikeClientSuppliedOnes() throws {
        // Defaults are artifact-sourced, so a malformed artifact must fail the
        // same way a malformed request does -- there must be no path that
        // reaches `.sampled` carrying an unvalidated value. Each case omits the
        // field from the request so ONLY the default can be the value under
        // test.
        assertThrows(
            try ServingSamplingPolicy.resolve(
                temperature: nil, topP: nil, topK: nil, minP: nil, seed: nil,
                defaults: ServingSamplingDefaults(
                    temperature: 2.5, topP: 0.95, topK: 20, minP: 0)),
            .temperatureOutOfRange(2.5))
        assertThrows(
            try ServingSamplingPolicy.resolve(
                temperature: nil, topP: nil, topK: nil, minP: nil, seed: nil,
                defaults: ServingSamplingDefaults(
                    temperature: 1.0, topP: 1.5, topK: 20, minP: 0)),
            .topPOutOfRange(1.5))
        assertThrows(
            try ServingSamplingPolicy.resolve(
                temperature: nil, topP: nil, topK: nil, minP: nil, seed: nil,
                defaults: ServingSamplingDefaults(
                    temperature: 1.0, topP: 0.95, topK: 0, minP: 0)),
            .topKOutOfRange(0))
        assertThrows(
            try ServingSamplingPolicy.resolve(
                temperature: nil, topP: nil, topK: nil, minP: nil, seed: nil,
                defaults: ServingSamplingDefaults(
                    temperature: 1.0, topP: 0.95, topK: 20, minP: 1.5)),
            .minPOutOfRange(1.5))
    }

    func testResolveFromRequestWithDefaultsFillsParamlessRequest() throws {
        let defaults = ServingSamplingDefaults(temperature: 1.0, topP: 0.95, topK: 20, minP: 0)
        let request = OpenAIChatCompletionRequest(
            model: "qwen3-32b",
            messages: [.init(role: .user, text: "Hi")],
            maxCompletionTokens: nil,
            temperature: nil,
            choiceCount: 1,
            stream: false,
            stop: [])
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(from: request, defaults: defaults),
            .sampled(temperature: 1.0, topP: 0.95, topK: 20, minP: 0, seed: nil))
    }

    func testResolveFromRequestWithDefaultsKeepsExplicitZeroTemperatureGreedy() throws {
        let defaults = ServingSamplingDefaults(temperature: 1.0, topP: 0.95, topK: 20, minP: 0)
        let request = OpenAIChatCompletionRequest(
            model: "qwen3-32b",
            messages: [.init(role: .user, text: "Hi")],
            maxCompletionTokens: nil,
            temperature: 0,
            choiceCount: 1,
            stream: false,
            stop: [])
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(from: request, defaults: defaults),
            .greedy)
    }

    func testResolveFromRequestWithoutDefaultsArgumentIsUnchanged() throws {
        let request = OpenAIChatCompletionRequest(
            model: "qwen3-32b",
            messages: [.init(role: .user, text: "Hi")],
            maxCompletionTokens: nil,
            temperature: nil,
            choiceCount: 1,
            stream: false,
            stop: [])
        XCTAssertEqual(try ServingSamplingPolicy.resolve(from: request), .greedy)
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
