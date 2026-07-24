import XCTest

@testable import ServingCore

final class FastMLXServeArgumentsTests: XCTestCase {
    func testScriptedModeRetainsTransportDefaults() throws {
        let arguments = try FastMLXServeArguments.parse(["--scripted"])

        XCTAssertEqual(arguments.backend, .scripted)
        XCTAssertEqual(arguments.host, "127.0.0.1")
        XCTAssertEqual(arguments.port, 8_080)
        XCTAssertEqual(arguments.model, "fastmlx-scripted")
        XCTAssertFalse(arguments.showHelp)
    }

    func testScalarModeRequiresAndPreservesExplicitIdentityAndMemoryLimits() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/source-locked",
            "--model", "source-locked-model",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--host", "127.0.0.2",
            "--port", "9000",
        ])

        XCTAssertEqual(
            arguments.backend,
            .scalar(
                modelDirectory: URL(
                    fileURLWithPath: "/models/source-locked",
                    isDirectory: true),
                memoryLimitBytes: 68_719_476_736,
                cacheLimitBytes: 8_589_934_592))
        XCTAssertEqual(arguments.model, "source-locked-model")
        XCTAssertEqual(arguments.host, "127.0.0.2")
        XCTAssertEqual(arguments.port, 9_000)
        XCTAssertFalse(arguments.showHelp)
    }

    func testContinuousBatchModeRequiresExplicitSelectorAndPreservesModelBundle() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--continuous-batch-no-spec",
            "--model-path", "/models/qwen3-source-locked",
            "--model", "qwen3-source-locked",
            "--memory-limit-bytes", "103079215104",
            "--cache-limit-bytes", "8589934592",
            "--max-reserved-kv-bytes", "17179869184",
            "--host", "127.0.0.2",
            "--port", "9001",
        ])

        XCTAssertEqual(
            arguments.backend,
            .continuousBatchNoSpec(
                modelDirectory: URL(
                    fileURLWithPath: "/models/qwen3-source-locked",
                    isDirectory: true),
                memoryLimitBytes: 103_079_215_104,
                cacheLimitBytes: 8_589_934_592,
                maxReservedKVBytes: 17_179_869_184))
        XCTAssertEqual(arguments.model, "qwen3-source-locked")
        XCTAssertEqual(arguments.host, "127.0.0.2")
        XCTAssertEqual(arguments.port, 9_001)
        XCTAssertFalse(arguments.showHelp)
    }

    func testHelpDoesNotRequireBackendConfiguration() throws {
        let arguments = try FastMLXServeArguments.parse(["--help"])

        XCTAssertTrue(arguments.showHelp)
        XCTAssertNil(arguments.backend)
    }

    func testNonPromotedDynamicPLDModeIsNotExposedByCLI() {
        XCTAssertFalse(
            FastMLXServeArguments.usage.contains("--continuous-dynamic-pld"))
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(["--continuous-dynamic-pld"])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .unknownArgument("--continuous-dynamic-pld"))
        }
    }

    func testMissingAndConflictingBackendModesFailClosed() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(["--model", "fixture"])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingBackendMode)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "4096",
                "--cache-limit-bytes", "1024",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .conflictingBackendModes)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--continuous-batch-no-spec",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .conflictingBackendModes)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--continuous-batch-no-spec",
                "--model-path", "/models/qwen3-source-locked",
                "--model", "qwen3-source-locked",
                "--memory-limit-bytes", "103079215104",
                "--cache-limit-bytes", "8589934592",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingRequiredOption("--max-reserved-kv-bytes"))
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/source-locked",
                "--model", "source-locked-model",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--max-reserved-kv-bytes", "17179869184",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .optionRequiresContinuousBatchMode(
                    "--max-reserved-kv-bytes"))
        }
    }

    func testScalarModeRejectsPartialRelativeOrUnsafeMemoryConfiguration() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "4096",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingRequiredOption("--cache-limit-bytes"))
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "4096",
                "--cache-limit-bytes", "1024",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .modelPathMustBeAbsolute)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "1024",
                "--cache-limit-bytes", "2048",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .cacheLimitExceedsMemoryLimit)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--continuous-batch-no-spec",
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "4096",
                "--cache-limit-bytes", "1024",
                "--max-reserved-kv-bytes", "8192",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .reservedKVLimitExceedsMemoryLimit)
        }
    }

    func testDuplicateUnknownAndInvalidNumericOptionsFailClosed() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--port", "8000",
                "--port", "9000",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .duplicateOption("--port"))
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(["--scripted", "--unknown"])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .unknownArgument("--unknown"))
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "not-a-number",
                "--cache-limit-bytes", "1024",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .invalidPositiveInteger("--memory-limit-bytes"))
        }
    }
}
