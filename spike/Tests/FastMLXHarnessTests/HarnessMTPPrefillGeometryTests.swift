import Darwin
import Foundation
import XCTest

import SpikeCore

@testable import fastmlx_harness

/// Covers `HarnessMTPPrefillGeometry`, the single source of truth for the harness's MTP prefill
/// chunk size (task-inbox: seven harness-CLI `MTPSpeculativeTokenIterator` constructions were
/// silently inheriting the vendored `GenerateParameters` default of 512 instead of production's
/// `MLXDecoder.defaultPrefillChunkSize` of 2048). `testDefaultIsNotTheVendoredGenerateParametersDefault`
/// is the anti-vacuity check: it is the exact assertion that would have caught the original defect.
final class HarnessMTPPrefillGeometryTests: XCTestCase {

    private let variable = HarnessMTPPrefillGeometry.overrideEnvironmentVariable

    // MARK: - hermetic real-environment save/restore

    private var hadRealValue = false
    private var savedRealValue: String?

    override func setUp() {
        super.setUp()
        if let existing = ProcessInfo.processInfo.environment[variable] {
            hadRealValue = true
            savedRealValue = existing
        } else {
            hadRealValue = false
            savedRealValue = nil
        }
        unsetenv(variable)
    }

    override func tearDown() {
        if hadRealValue, let savedRealValue {
            setenv(variable, savedRealValue, 1)
        } else {
            unsetenv(variable)
        }
        super.tearDown()
    }

    // MARK: - default

    func testDefaultMatchesProductionConstant() throws {
        let result = try HarnessMTPPrefillGeometry.prefillChunkSize(environment: [:])
        XCTAssertEqual(result, MLXDecoder.defaultPrefillChunkSize)
    }

    func testProductionConstantIsCurrentlyTwoThousandFortyEight() {
        // Asserts the numeric anchor directly (not just equality against the symbol above) so a
        // future silent change to `MLXDecoder.defaultPrefillChunkSize` itself surfaces as a
        // FAILING test here, rather than this suite quietly tracking whatever it becomes.
        XCTAssertEqual(MLXDecoder.defaultPrefillChunkSize, 2048)
    }

    func testDefaultIsNotTheVendoredGenerateParametersDefault() throws {
        // ANTI-VACUITY: this is the exact check that would have caught the original defect --
        // every harness CLI silently inheriting the vendored `GenerateParameters` default of 512
        // instead of production's 2048. If this ever passes with `result == 512`, the helper has
        // drifted back into the bug it exists to prevent.
        let result = try HarnessMTPPrefillGeometry.prefillChunkSize(environment: [:])
        XCTAssertNotEqual(result, 512)
    }

    // MARK: - explicit override (injected environment)

    func testValidOverrideIsHonored() throws {
        let result = try HarnessMTPPrefillGeometry.prefillChunkSize(
            environment: [variable: "512"])
        XCTAssertEqual(result, 512)
    }

    func testZeroOverrideFailsLoudly() {
        XCTAssertThrowsError(
            try HarnessMTPPrefillGeometry.prefillChunkSize(environment: [variable: "0"])
        ) { error in
            guard
                let geometryError = error as? HarnessMTPPrefillGeometryError,
                case .invalidOverride(let reportedVariable, let reportedValue) = geometryError
            else {
                XCTFail("expected HarnessMTPPrefillGeometryError.invalidOverride, got \(error)")
                return
            }
            XCTAssertEqual(reportedVariable, variable)
            XCTAssertEqual(reportedValue, "0")
        }
    }

    func testNegativeOverrideFailsLoudly() {
        XCTAssertThrowsError(
            try HarnessMTPPrefillGeometry.prefillChunkSize(environment: [variable: "-1"])
        ) { error in
            guard
                let geometryError = error as? HarnessMTPPrefillGeometryError,
                case .invalidOverride(let reportedVariable, let reportedValue) = geometryError
            else {
                XCTFail("expected HarnessMTPPrefillGeometryError.invalidOverride, got \(error)")
                return
            }
            XCTAssertEqual(reportedVariable, variable)
            XCTAssertEqual(reportedValue, "-1")
        }
    }

    func testNonNumericOverrideFailsLoudly() {
        XCTAssertThrowsError(
            try HarnessMTPPrefillGeometry.prefillChunkSize(environment: [variable: "abc"])
        ) { error in
            guard
                let geometryError = error as? HarnessMTPPrefillGeometryError,
                case .invalidOverride(let reportedVariable, let reportedValue) = geometryError
            else {
                XCTFail("expected HarnessMTPPrefillGeometryError.invalidOverride, got \(error)")
                return
            }
            XCTAssertEqual(reportedVariable, variable)
            XCTAssertEqual(reportedValue, "abc")
        }
    }

    func testEmptyOverrideFailsLoudly() {
        XCTAssertThrowsError(
            try HarnessMTPPrefillGeometry.prefillChunkSize(environment: [variable: ""])
        ) { error in
            guard
                let geometryError = error as? HarnessMTPPrefillGeometryError,
                case .invalidOverride(let reportedVariable, let reportedValue) = geometryError
            else {
                XCTFail("expected HarnessMTPPrefillGeometryError.invalidOverride, got \(error)")
                return
            }
            XCTAssertEqual(reportedVariable, variable)
            XCTAssertEqual(reportedValue, "")
        }
    }

    // MARK: - explicit override (real process environment, end to end)

    func testRealProcessEnvironmentDefaultAppliesWhenUnset() throws {
        // `setUp` already unset the variable in the real environment.
        let result = try HarnessMTPPrefillGeometry.prefillChunkSize()
        XCTAssertEqual(result, MLXDecoder.defaultPrefillChunkSize)
    }

    func testRealProcessEnvironmentOverrideIsHonoredEndToEnd() throws {
        // Exercises the DEFAULT parameter value itself (`ProcessInfo.processInfo.environment`), not
        // just the injectable path above -- proves the wiring, not only the parsing logic.
        setenv(variable, "512", 1)
        let result = try HarnessMTPPrefillGeometry.prefillChunkSize()
        XCTAssertEqual(result, 512)
    }
}
