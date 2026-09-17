import XCTest

@testable import ServingCore

/// `--request-log`: parse-time flag selecting whether `ServingNIO` emits a structured per-request
/// JSON access log (default `off`, byte-identical to today; `json` opts a NEW production log
/// stream in). Mirrors `DefaultSamplingServeArgumentTests.swift`'s shape, in a NEW file (this
/// flag's write set does not include the existing, much larger `FastMLXServeArgumentsTests.swift`).
/// Unlike `--default-sampling`, this flag names no route/mode it would be silently dropped on --
/// every route (including `--scripted`) reaches the same handler finishing points -- so there are
/// no companion "requires"/"conflicts with" refusal tests here, only default/parse/duplicate/invalid.
final class RequestLogServeArgumentTests: XCTestCase {
    /// Acceptance criterion 1: the flag is absent -> requestLog == .off (the default), preserving
    /// today's production log volume byte-for-byte.
    func testRequestLogAbsentDefaultsToOff() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--scripted"
        ])

        XCTAssertEqual(arguments.requestLog, .off)
    }

    /// Acceptance criterion 2: --request-log off parses explicitly to .off.
    func testRequestLogExplicitOffParses() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--scripted",
            "--request-log", "off",
        ])

        XCTAssertEqual(arguments.requestLog, .off)
    }

    /// Acceptance criterion 3: --request-log json parses to .json on the scripted (transport-only)
    /// backend -- proving the flag is NOT scoped to a loaded-model route the way `--chat-template`/
    /// `--default-sampling`/etc are.
    func testRequestLogJSONParsesOnScriptedBackend() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--scripted",
            "--request-log", "json",
        ])

        XCTAssertEqual(arguments.requestLog, .json)
    }

    /// Acceptance criterion 4: --request-log json also parses on a plain loaded-model (scalar)
    /// serve command.
    func testRequestLogJSONParsesOnLoadedModelServe() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--request-log", "json",
        ])

        XCTAssertEqual(arguments.requestLog, .json)
    }

    /// Acceptance criterion 5: an unrecognized value throws the specific .invalidRequestLogMode
    /// error, fail closed rather than silently defaulting.
    func testRequestLogInvalidValueThrowsTheSpecificError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--request-log", "verbose",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .invalidRequestLogMode)
        }
    }

    /// Acceptance criterion 6: --request-log with no value throws .missingValue, matching every
    /// other valued flag's parser convention.
    func testRequestLogMissingValueThrowsMissingValue() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--request-log",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingValue("--request-log"))
        }
    }

    /// Acceptance criterion 7: passing --request-log twice throws .duplicateOption, fail closed
    /// like every other repeated flag in this parser rather than silently taking the last value.
    func testRequestLogDuplicateThrowsDuplicateOption() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--request-log", "off",
                "--request-log", "json",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .duplicateOption("--request-log"))
        }
    }

    /// Acceptance criterion 8: the refusal announce line names the flag and the specific violation,
    /// matching `fastMLXServeArgumentRefusalAnnounceLine`'s generic rendering for every other case.
    func testInvalidRequestLogModeAnnounceLineNamesTheFlag() {
        let line = fastMLXServeArgumentRefusalAnnounceLine(.invalidRequestLogMode)

        XCTAssertTrue(line.contains("--request-log"))
        XCTAssertTrue(line.contains("reason=invalid_arguments"))
    }
}
