import XCTest

@testable import ServingCore

final class GenerationConfigSamplingDefaultsTests: XCTestCase {
    private var tempDirectory: URL!

    override func setUpWithError() throws {
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("GenerationConfigSamplingDefaultsTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDirectory)
        tempDirectory = nil
    }

    private func write(_ contents: String, name: String = "generation_config.json") -> URL {
        let url = tempDirectory.appendingPathComponent(name)
        try! contents.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    func testRealDeployedShapeResolvesParamlessRequestToDeployedPreset() throws {
        // The acceptance case: the actual artifact shape the deployed model
        // ships must round-trip through the resolver to the deployed preset.
        let url = write(
            """
            {"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
            """)
        let defaults = try GenerationConfigSamplingDefaults.load(contentsOf: url)
        XCTAssertEqual(
            try ServingSamplingPolicy.resolve(
                temperature: nil, topP: nil, topK: nil, minP: nil, seed: nil, defaults: defaults),
            .sampled(temperature: 1.0, topP: 0.95, topK: 20, minP: nil, seed: nil))
    }

    func testDoSampleFalseThrowsSamplingNotEnabled() {
        let url = write(
            """
            {"do_sample": false, "temperature": 1.0}
            """)
        XCTAssertThrowsError(try GenerationConfigSamplingDefaults.load(contentsOf: url)) { error in
            XCTAssertEqual(
                error as? GenerationConfigSamplingDefaultsError, .samplingNotEnabled)
        }
    }

    func testDoSampleAbsentThrowsSamplingNotEnabled() {
        // Absence is not consent: HF's own default for `do_sample` is `false`.
        let url = write(
            """
            {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
            """)
        XCTAssertThrowsError(try GenerationConfigSamplingDefaults.load(contentsOf: url)) { error in
            XCTAssertEqual(
                error as? GenerationConfigSamplingDefaultsError, .samplingNotEnabled)
        }
    }

    func testMissingFileThrowsUnreadable() {
        let url = tempDirectory.appendingPathComponent("does-not-exist.json")
        XCTAssertThrowsError(try GenerationConfigSamplingDefaults.load(contentsOf: url)) { error in
            XCTAssertEqual(error as? GenerationConfigSamplingDefaultsError, .unreadable)
        }
    }

    func testMalformedJSONThrowsUnparseable() {
        let url = write("{not valid json")
        XCTAssertThrowsError(try GenerationConfigSamplingDefaults.load(contentsOf: url)) { error in
            XCTAssertEqual(error as? GenerationConfigSamplingDefaultsError, .unparseable)
        }
    }

    func testDoSampleTrueWithNoTemperatureThrowsNoUsableTemperature() {
        let url = write(
            """
            {"do_sample": true, "top_p": 0.95, "top_k": 20}
            """)
        XCTAssertThrowsError(try GenerationConfigSamplingDefaults.load(contentsOf: url)) { error in
            XCTAssertEqual(
                error as? GenerationConfigSamplingDefaultsError, .noUsableTemperature)
        }
    }

    func testTemperatureZeroThrowsNoUsableTemperature() {
        // A defaults object whose temperature is 0 resolves to `.greedy`,
        // i.e. a no-op default -- not worth building.
        let url = write(
            """
            {"do_sample": true, "temperature": 0}
            """)
        XCTAssertThrowsError(try GenerationConfigSamplingDefaults.load(contentsOf: url)) { error in
            XCTAssertEqual(
                error as? GenerationConfigSamplingDefaultsError, .noUsableTemperature)
        }
    }

    func testTopKZeroMapsToDisabledNotZero() throws {
        // HF's convention: `top_k: 0` means "disabled", not the literal 0
        // that `ServingSamplingPolicy.resolve` would refuse.
        let url = write(
            """
            {"do_sample": true, "temperature": 1.0, "top_k": 0}
            """)
        let defaults = try GenerationConfigSamplingDefaults.load(contentsOf: url)
        XCTAssertNil(defaults.topK)

        let policy = try ServingSamplingPolicy.resolve(
            temperature: nil, topP: nil, topK: nil, minP: nil, seed: nil, defaults: defaults)
        guard case .sampled(_, _, let topK, _, _) = policy else {
            return XCTFail("expected .sampled, got \(policy)")
        }
        XCTAssertNil(topK)
    }

    func testOutOfRangeTemperatureThrowsInvalidValueAtLoadTime() {
        // Fail-closed-at-startup proof: an artifact value the resolver would
        // reject on the first request instead blows up at load time.
        let url = write(
            """
            {"do_sample": true, "temperature": 5}
            """)
        XCTAssertThrowsError(try GenerationConfigSamplingDefaults.load(contentsOf: url)) { error in
            XCTAssertEqual(
                error as? GenerationConfigSamplingDefaultsError,
                .invalidValue(.temperatureOutOfRange(5)))
        }
    }
}
