import XCTest

@testable import ServingCore

final class FastMLXServeArgumentsTests: XCTestCase {
    func testScriptedModeRetainsTransportDefaults() throws {
        let arguments = try FastMLXServeArguments.parse(["--scripted"])

        XCTAssertEqual(arguments.backend, .scripted)
        XCTAssertEqual(arguments.host, "127.0.0.1")
        XCTAssertEqual(arguments.port, 8_080)
        XCTAssertEqual(arguments.model, "fastmlx-scripted")
        XCTAssertEqual(arguments.maximumCompletionTokens, 4_096)
        XCTAssertNil(arguments.evidencePath)
        XCTAssertFalse(arguments.showHelp)
    }

    func testHostUseParsesExplicitOperatorIntentAndDefaultsNil() throws {
        let absent = try FastMLXServeArguments.parse(["--scripted"])
        XCTAssertNil(absent.requestedHostUse)

        let shared = try FastMLXServeArguments.parse([
            "--scripted",
            "--host-use", "shared",
        ])
        XCTAssertEqual(shared.requestedHostUse, .shared)

        let dedicated = try FastMLXServeArguments.parse([
            "--scripted",
            "--host-use", "dedicated-serving",
            "--os-service-reserve-bytes", "4294967296",
        ])
        XCTAssertEqual(dedicated.requestedHostUse, .dedicatedServing)
        XCTAssertEqual(dedicated.osServiceReserveBytes, 4_294_967_296)

        XCTAssertEqual(FastMLXServeHostUse.shared.rawValue, "shared")
        XCTAssertEqual(FastMLXServeHostUse.dedicatedServing.rawValue, "dedicated-serving")
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--host-use VALUE"))
        XCTAssertTrue(FastMLXServeArguments.usage.contains("shared|dedicated-serving"))
    }

    func testHostUseRejectsMissingDuplicateAndUnknownValues() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--host-use",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingValue("--host-use"))
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--host-use", "shared",
                "--host-use", "dedicated-serving",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .duplicateOption("--host-use"))
        }

        for rawValue in ["dedicated", "DEDICATED-SERVING", " shared", "shared "] {
            XCTAssertThrowsError(
                try FastMLXServeArguments.parse([
                    "--scripted",
                    "--host-use", rawValue,
                ]),
                "expected host-use rejection for \(rawValue.debugDescription)"
            ) { error in
                XCTAssertEqual(
                    error as? FastMLXServeArgumentError,
                    .invalidHostUse)
            }
        }
    }

    func testDedicatedHostUseRequiresAndPreservesDeclaredOSServiceReserve() throws {
        let loaded = [
            "--model-path", "/models/source-locked",
            "--model", "source-locked-model",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ]

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(
                loaded + ["--host-use", "dedicated-serving"])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingRequiredOption("--os-service-reserve-bytes"))
        }

        let dedicated = try FastMLXServeArguments.parse(
            loaded + [
                "--host-use", "dedicated-serving",
                "--os-service-reserve-bytes", "8589934592",
            ])
        XCTAssertEqual(dedicated.requestedHostUse, .dedicatedServing)
        XCTAssertEqual(dedicated.osServiceReserveBytes, 8_589_934_592)
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--os-service-reserve-bytes N"))
    }

    func testOSServiceReserveRejectsSharedMissingDuplicateAndInvalidValues() {
        let loaded = [
            "--model-path", "/models/source-locked",
            "--model", "source-locked-model",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ]

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(
                loaded + ["--os-service-reserve-bytes", "8589934592"])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .osServiceReserveRequiresDedicatedServing)
        }

        for invalid in ["0", "-1", "not-a-number"] {
            XCTAssertThrowsError(
                try FastMLXServeArguments.parse(
                    loaded + [
                        "--host-use", "dedicated-serving",
                        "--os-service-reserve-bytes", invalid,
                    ])
            ) { error in
                XCTAssertEqual(
                    error as? FastMLXServeArgumentError,
                    .invalidPositiveInteger("--os-service-reserve-bytes"))
            }
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(
                loaded + [
                    "--host-use", "dedicated-serving",
                    "--os-service-reserve-bytes", "1024",
                    "--os-service-reserve-bytes", "2048",
                ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .duplicateOption("--os-service-reserve-bytes"))
        }
    }

    func testMaxCompletionTokensAcceptsStrictPositiveIntegerAndAppearsInUsage() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--scripted",
            "--max-completion-tokens", "8192",
        ])

        XCTAssertEqual(arguments.maximumCompletionTokens, 8_192)
        XCTAssertTrue(arguments.maximumCompletionTokensWasExplicit)
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains("--max-completion-tokens N"))
    }

    func testCompletionBudgetDefaultsAndExplicitMaxIntentArePreserved() throws {
        let absent = try FastMLXServeArguments.parse(["--scripted"])
        XCTAssertEqual(absent.defaultCompletionTokens, 4_096)
        XCTAssertFalse(absent.defaultCompletionTokensWasExplicit)
        XCTAssertEqual(absent.maximumCompletionTokens, 4_096)
        XCTAssertFalse(absent.maximumCompletionTokensWasExplicit)
        XCTAssertEqual(absent.maximumNonStreamingCompletionTokens, 16_384)
        XCTAssertNil(absent.maximumRequestBodyBytes)
        XCTAssertEqual(absent.maximumNonStreamingResponseBytes, 16 * 1_048_576)
        XCTAssertEqual(absent.completionLimitPolicy, .reject)

        let explicitMax = try FastMLXServeArguments.parse([
            "--scripted",
            "--max-completion-tokens", "4096",
        ])
        XCTAssertEqual(explicitMax.maximumCompletionTokens, 4_096)
        XCTAssertTrue(
            explicitMax.maximumCompletionTokensWasExplicit,
            "explicit 4096 must remain distinguishable from the default so served model caps can be derived from the model when the operator omits the flag")
    }

    func testDefaultCompletionTokensParsesStrictPositiveInteger() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--scripted",
            "--default-completion-tokens", "8192",
        ])

        XCTAssertEqual(arguments.defaultCompletionTokens, 8_192)
        XCTAssertTrue(arguments.defaultCompletionTokensWasExplicit)
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains("--default-completion-tokens N"))
    }

    func testMaximumNonStreamingCompletionTokensParsesStrictPositiveInteger() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--scripted",
            "--max-non-streaming-completion-tokens", "32768",
        ])

        XCTAssertEqual(arguments.maximumNonStreamingCompletionTokens, 32_768)
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains(
                "--max-non-streaming-completion-tokens N"))
    }

    func testTransportByteLimitsParseStrictPositiveIntegersAndAppearInUsage() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--scripted",
            "--max-request-body-bytes", "67108864",
            "--max-non-streaming-response-bytes", "33554432",
        ])

        XCTAssertEqual(arguments.maximumRequestBodyBytes, 64 * 1_048_576)
        XCTAssertEqual(arguments.maximumNonStreamingResponseBytes, 32 * 1_048_576)
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--max-request-body-bytes N"))
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains(
                "--max-non-streaming-response-bytes N"))
    }

    func testCompletionLimitPolicyParsesRejectAndClamp() throws {
        let reject = try FastMLXServeArguments.parse([
            "--scripted",
            "--completion-limit-policy", "reject",
        ])
        XCTAssertEqual(reject.completionLimitPolicy, .reject)

        let clamp = try FastMLXServeArguments.parse([
            "--scripted",
            "--completion-limit-policy", "clamp",
        ])
        XCTAssertEqual(clamp.completionLimitPolicy, .clamp)
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains("--completion-limit-policy MODE"))
        XCTAssertTrue(FastMLXServeArguments.usage.contains("reject|clamp"))
    }

    func testCompletionBudgetFlagsRejectInvalidValues() {
        for option in [
            "--default-completion-tokens",
            "--max-non-streaming-completion-tokens",
            "--max-request-body-bytes",
            "--max-non-streaming-response-bytes",
        ] {
            XCTAssertThrowsError(
                try FastMLXServeArguments.parse([
                    "--scripted",
                    option, "0",
                ]),
                "expected zero rejection for \(option)"
            ) { error in
                XCTAssertEqual(
                    error as? FastMLXServeArgumentError,
                    .invalidPositiveInteger(option))
            }

            XCTAssertThrowsError(
                try FastMLXServeArguments.parse([
                    "--scripted",
                    option, "-1",
                ]),
                "expected negative rejection for \(option)"
            ) { error in
                XCTAssertEqual(
                    error as? FastMLXServeArgumentError,
                    .invalidPositiveInteger(option))
            }
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--completion-limit-policy", "truncate",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .invalidCompletionLimitPolicy)
        }
    }

    func testExplicitDefaultCompletionBudgetCannotExceedExplicitMaximum() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--default-completion-tokens", "8192",
                "--max-completion-tokens", "4096",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .defaultCompletionTokensExceedsMaximumCompletionTokens)
        }
    }

    func testMaxCompletionTokensRejectsMissingDuplicateAndNonStrictValues() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--max-completion-tokens",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingValue("--max-completion-tokens"))
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--max-completion-tokens", "8",
                "--max-completion-tokens", "9",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .duplicateOption("--max-completion-tokens"))
        }

        for rawValue in [
            "0", "-1", "+1", " 8", "8 ", "1.5", "18446744073709551616",
        ] {
            XCTAssertThrowsError(
                try FastMLXServeArguments.parse([
                    "--scripted",
                    "--max-completion-tokens", rawValue,
                ]),
                "expected strict positive-integer rejection for \(rawValue.debugDescription)"
            ) { error in
                XCTAssertEqual(
                    error as? FastMLXServeArgumentError,
                    .invalidPositiveInteger("--max-completion-tokens"))
            }
        }
    }

    func testEvidencePathIsExplicitAbsoluteAndFreshnessIsDeferredToStartup() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--scripted",
            "--evidence-path", "/tmp/fastmlx-serving-evidence.jsonl",
        ])

        XCTAssertEqual(
            arguments.evidencePath,
            URL(fileURLWithPath: "/tmp/fastmlx-serving-evidence.jsonl"))
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains("--evidence-path PATH"))

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--evidence-path", "relative/evidence.jsonl",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .evidencePathMustBeAbsolute)
        }
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

    func testDynamicPLDModeParsesAsExplicitContinuousRoute() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--continuous-dynamic-pld",
            "--model-path", "/models/qwen3-source-locked",
            "--model", "qwen3-source-locked",
            "--memory-limit-bytes", "103079215104",
            "--cache-limit-bytes", "8589934592",
            "--max-reserved-kv-bytes", "17179869184",
        ])

        XCTAssertEqual(
            arguments.backend,
            .continuousDynamicPLD(
                modelDirectory: URL(
                    fileURLWithPath: "/models/qwen3-source-locked",
                    isDirectory: true),
                memoryLimitBytes: 103_079_215_104,
                cacheLimitBytes: 8_589_934_592,
                maxReservedKVBytes: 17_179_869_184))
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains("--continuous-dynamic-pld"))
    }

    func testDynamicPLDModeRejectsConflictingOrUnprovenCombinations() {
        let loaded = [
            "--model-path", "/models/qwen3-source-locked",
            "--model", "qwen3-source-locked",
            "--memory-limit-bytes", "103079215104",
            "--cache-limit-bytes", "8589934592",
            "--max-reserved-kv-bytes", "17179869184",
        ]

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted", "--continuous-dynamic-pld",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .conflictingBackendModes)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(
                loaded + [
                    "--continuous-batch-no-spec",
                    "--continuous-dynamic-pld",
                ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .conflictingBackendModes)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(
                loaded + [
                    "--continuous-dynamic-pld",
                    "--allow-hybrid-qwen35",
                ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .dynamicPLDWithHybridQwen35)
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

    func testContextAndForceParseForTheFitCheck() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--continuous-batch-no-spec",
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--max-reserved-kv-bytes", "17179869184",
            "--context", "8192",
            "--force",
        ])
        XCTAssertEqual(arguments.requestedContext, 8192)
        XCTAssertTrue(arguments.forceServe)
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--context N"))
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--force"))
    }

    func testContextAndForceDefaultWhenAbsent() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])
        XCTAssertNil(arguments.requestedContext)
        XCTAssertFalse(arguments.forceServe)
    }

    func testContextRejectsNonPositiveInteger() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--context", "0",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .invalidPositiveInteger("--context"))
        }
    }

    // MARK: - quant auto-pick (--quant-candidates): supply several local checkpoint dirs; the
    // pre-load pick chooses which one loads. --model-path is replaced by the candidate list.

    func testQuantCandidatesParseAsAbsoluteDirsAndSeedTheScalarBackend() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-candidates", "/models/qwen3-4bit,/models/qwen3-8bit",
            "--model", "qwen3",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])

        XCTAssertEqual(
            arguments.quantCandidateDirectories,
            [URL(fileURLWithPath: "/models/qwen3-4bit", isDirectory: true),
             URL(fileURLWithPath: "/models/qwen3-8bit", isDirectory: true)])
        // The backend seeds its modelDirectory with the first candidate as a placeholder; the preflight
        // pick substitutes the actual winner before loading.
        XCTAssertEqual(
            arguments.backend,
            .scalar(
                modelDirectory: URL(fileURLWithPath: "/models/qwen3-4bit", isDirectory: true),
                memoryLimitBytes: 68_719_476_736,
                cacheLimitBytes: 8_589_934_592))
        XCTAssertEqual(arguments.model, "qwen3")
    }

    func testQuantCandidatesWorkOnTheContinuousRoute() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--continuous-batch-no-spec",
            "--quant-candidates", "/models/a,/models/b",
            "--model", "qwen3",
            "--memory-limit-bytes", "103079215104",
            "--cache-limit-bytes", "8589934592",
            "--max-reserved-kv-bytes", "17179869184",
        ])
        XCTAssertEqual(arguments.quantCandidateDirectories.count, 2)
        XCTAssertEqual(
            arguments.backend,
            .continuousBatchNoSpec(
                modelDirectory: URL(fileURLWithPath: "/models/a", isDirectory: true),
                memoryLimitBytes: 103_079_215_104,
                cacheLimitBytes: 8_589_934_592,
                maxReservedKVBytes: 17_179_869_184))
    }

    func testQuantCandidatesConflictWithModelPath() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-candidates", "/models/a,/models/b",
                "--model-path", "/models/c",
                "--model", "qwen3",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .quantCandidatesWithModelPath)
        }
    }

    func testQuantCandidatesRejectRelativeOrEmptyEntries() {
        // A relative entry, or a leading/trailing/doubled comma (which yields an empty entry), fails
        // closed rather than silently dropping a candidate.
        for list in ["/models/a,relative/b", "/models/a,", ",/models/b"] {
            XCTAssertThrowsError(
                try FastMLXServeArguments.parse([
                    "--quant-candidates", list,
                    "--model", "qwen3",
                    "--memory-limit-bytes", "68719476736",
                    "--cache-limit-bytes", "8589934592",
                ]),
                "expected \"\(list)\" to fail closed"
            ) { error in
                XCTAssertEqual(
                    error as? FastMLXServeArgumentError,
                    .quantCandidateMustBeAbsolute)
            }
        }
    }

    func testQuantCandidatesRejectEmptyValue() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-candidates", "",
                "--model", "qwen3",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingValue("--quant-candidates"))
        }
    }

    func testQuantCandidatesDefaultEmpty() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])
        XCTAssertTrue(arguments.quantCandidateDirectories.isEmpty)
    }

    // MARK: - --quant-pick-only: a dry-run that resolves which quant would load and exits, with NO
    // model load. It is its own early-return mode (like --help), so it needs ONLY --quant-candidates
    // (+ optional --context) — never the runtime load limits, because nothing is loaded.

    func testQuantPickOnlyParsesWithOnlyCandidates() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--quant-candidates", "/models/qwen3-4bit,/models/qwen3-8bit",
        ])
        XCTAssertTrue(arguments.quantPickOnly)
        XCTAssertNil(arguments.backend, "pick-only loads nothing, so it selects no backend")
        XCTAssertEqual(arguments.quantCandidateDirectories.count, 2)
        XCTAssertFalse(arguments.showHelp)
    }

    func testQuantPickOnlyHonorsContext() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--quant-candidates", "/models/a,/models/b",
            "--context", "16384",
        ])
        XCTAssertTrue(arguments.quantPickOnly)
        XCTAssertEqual(arguments.requestedContext, 16_384)
    }

    func testQuantPickOnlyRequiresCandidates() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(["--quant-pick-only"])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingRequiredOption("--quant-candidates"))
        }
    }

    /// Non-regression lock: adding pick-only must NOT weaken the load modes' required-option
    /// validation. A normal candidates-mode serve (no --quant-pick-only) still requires the runtime
    /// memory limit — the early-return mode is the ONLY path that skips it.
    func testQuantPickOnlyDoesNotWeakenLoadModeRequirements() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-candidates", "/models/a,/models/b",
                "--model", "qwen3",
                "--cache-limit-bytes", "8589934592",
            ]),
            "candidates mode without --quant-pick-only still requires --memory-limit-bytes"
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingRequiredOption("--memory-limit-bytes"))
        }
    }

    // MARK: - --quant-reliability: an ADVISORY artifact overlaid on the pick-only announce. It is an
    // absolute local path (mirroring --evidence-path) and defaults to nil.

    func testQuantReliabilityPathParsesAsAbsoluteOnPickOnly() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--quant-candidates", "/models/q4,/models/q8",
            "--quant-reliability", "/tmp/quant-reliability-v1-sample.json",
        ])
        XCTAssertTrue(arguments.quantPickOnly)
        XCTAssertEqual(
            arguments.quantReliabilityPath,
            URL(fileURLWithPath: "/tmp/quant-reliability-v1-sample.json"))
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains("--quant-reliability PATH"),
            "the flag must be documented in usage")
    }

    func testQuantReliabilityPathRejectsRelative() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-pick-only",
                "--quant-candidates", "/models/q4,/models/q8",
                "--quant-reliability", "relative/artifact.json",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .quantReliabilityPathMustBeAbsolute)
        }
    }

    func testQuantReliabilityPathDefaultsNil() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--quant-candidates", "/models/q4,/models/q8",
        ])
        XCTAssertNil(arguments.quantReliabilityPath)
    }

    // MARK: - --kv-quant: a requested KV-cache precision tier carried as a RAW string. ServingCore
    // deliberately does NOT validate the tier (that would force a HarnessCore dependency); it only
    // parses/threads the value and rejects it in scripted mode. Tier validation + the sizing preview
    // live in HarnessCore.KVQuantAdvisory, unit-tested there.

    func testKVQuantTierParsesAsRawStringOnScalarServe() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--kv-quant", "int8",
        ])
        XCTAssertEqual(arguments.kvQuantTier, "int8")
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--kv-quant TIER"))
    }

    func testKVQuantTierIsCarriedVerbatimEvenWhenUnrecognized() throws {
        // ServingCore does not know the tier vocabulary; an unknown value parses (HarnessCore fails
        // it closed at serve time). This proves ServingCore stays HarnessCore-free.
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--kv-quant", "int3",
        ])
        XCTAssertEqual(arguments.kvQuantTier, "int3")
    }

    func testKVQuantTierRequiresAValue() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--kv-quant",
            ])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .missingValue("--kv-quant"))
        }
    }

    func testKVQuantTierRejectedInScriptedMode() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(["--scripted", "--kv-quant", "int8"])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .kvQuantWithScripted)
        }
    }

    func testKVQuantTierDefaultsNil() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])
        XCTAssertNil(arguments.kvQuantTier)
    }

    // MARK: - --allow-hybrid-qwen35: opt-in admission of the qwen3_5 hybrid architecture onto the
    // continuous-batch serve route. Default OFF preserves today's silent scalar fallback for hybrid
    // families (fastmlx-serve continuous_fallback=scalar …). Continuous-only: rejected with --scripted,
    // which loads no model. See docs/task-inbox/2026-08-20-hybrid-continuous-serve-path-admission.md.

    func testAllowHybridQwen35ParsesOnContinuousRoute() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--continuous-batch-no-spec",
            "--model-path", "/models/qwen3_5-source-locked",
            "--model", "qwen3_5-source-locked",
            "--memory-limit-bytes", "103079215104",
            "--cache-limit-bytes", "8589934592",
            "--max-reserved-kv-bytes", "17179869184",
            "--allow-hybrid-qwen35",
        ])
        XCTAssertTrue(arguments.allowHybridQwen35)
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--allow-hybrid-qwen35"))
    }

    func testAllowHybridQwen35DefaultsFalseWhenAbsent() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--continuous-batch-no-spec",
            "--model-path", "/models/qwen3_5-source-locked",
            "--model", "qwen3_5-source-locked",
            "--memory-limit-bytes", "103079215104",
            "--cache-limit-bytes", "8589934592",
            "--max-reserved-kv-bytes", "17179869184",
        ])
        XCTAssertFalse(arguments.allowHybridQwen35)
    }

    func testAllowHybridQwen35RejectedInScriptedMode() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(["--scripted", "--allow-hybrid-qwen35"])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .allowHybridWithScripted)
        }
    }

    // MARK: - --exact-qwen35-mtp: explicit local target/drafter opt-in. Default OFF preserves the
    // existing scalar/continuous routes; the target remains --model-path and the drafter is a separate
    // absolute local snapshot directory.

    func testExactQwen35MTPParsesAsScalarRouteOptInWithSeparateDrafterPath() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/qwen35-target",
            "--model", "qwen35-exact",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--exact-qwen35-mtp",
            "--mtp-drafter-path", "/models/qwen35-drafter",
        ])

        XCTAssertEqual(
            arguments.backend,
            .scalar(
                modelDirectory: URL(fileURLWithPath: "/models/qwen35-target", isDirectory: true),
                memoryLimitBytes: 68_719_476_736,
                cacheLimitBytes: 8_589_934_592))
        XCTAssertTrue(arguments.exactQwen35MTP)
        XCTAssertEqual(
            arguments.mtpDrafterDirectory,
            URL(fileURLWithPath: "/models/qwen35-drafter", isDirectory: true))
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--exact-qwen35-mtp"))
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--mtp-drafter-path PATH"))
    }

    func testExactQwen35MTPParsesExplicitQwen38ArtifactSelection() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/qwen38-target",
            "--model", "qwen38-exact",
            "--memory-limit-bytes", "171798691840",
            "--cache-limit-bytes", "34359738368",
            "--exact-qwen35-mtp",
            "--exact-mtp-selection", "qwen38-27b-mxfp8-depth1",
            "--mtp-drafter-path", "/models/qwen38-drafter",
        ])

        XCTAssertEqual(arguments.exactMTPSelection, .qwen38_27BMXFP8Depth1)
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--exact-mtp-selection SELECTION"))
        XCTAssertTrue(FastMLXServeArguments.usage.contains("qwen38-27b-mxfp8-depth1"))
    }

    func testExactQwen35MTPDefaultsOffWhenAbsent() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])

        XCTAssertFalse(arguments.exactQwen35MTP)
        XCTAssertEqual(arguments.exactMTPSelection, .qwen35_9BDepth1)
        XCTAssertNil(arguments.mtpDrafterDirectory)
    }

    func testExactQwen35MTPRequiresAbsoluteDrafterPath() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/qwen35-target",
                "--model", "qwen35-exact",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--exact-qwen35-mtp",
            ])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .missingRequiredOption("--mtp-drafter-path"))
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/qwen35-target",
                "--model", "qwen35-exact",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--exact-qwen35-mtp",
                "--mtp-drafter-path", "relative/drafter",
            ])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .mtpDrafterPathMustBeAbsolute)
        }
    }

    func testExactMTPSelectionFailsClosedWhenInvalidOrMissingExactOptIn() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/qwen38-target",
                "--model", "qwen38-exact",
                "--memory-limit-bytes", "171798691840",
                "--cache-limit-bytes", "34359738368",
                "--exact-qwen35-mtp",
                "--exact-mtp-selection", "qwen38-latest",
                "--mtp-drafter-path", "/models/qwen38-drafter",
            ])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .invalidExactMTPSelection)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/qwen38-target",
                "--model", "qwen38-exact",
                "--memory-limit-bytes", "171798691840",
                "--cache-limit-bytes", "34359738368",
                "--exact-mtp-selection", "qwen38-27b-mxfp8-depth1",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .exactMTPSelectionRequiresExactQwen35MTP)
        }
    }

    func testMTPDrafterPathCannotBePassedWithoutExactQwen35MTP() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/qwen35-target",
                "--model", "qwen35-exact",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--mtp-drafter-path", "/models/qwen35-drafter",
            ])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .mtpDrafterRequiresExactQwen35MTP)
        }
    }

    func testExactQwen35MTPRejectsScriptedContinuousAndQuantSources() {
        let common = [
            "--model-path", "/models/qwen35-target",
            "--model", "qwen35-exact",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--exact-qwen35-mtp",
            "--mtp-drafter-path", "/models/qwen35-drafter",
        ]

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(["--scripted", "--exact-qwen35-mtp", "--mtp-drafter-path", "/models/qwen35-drafter"])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .exactQwen35MTPWithScripted)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(common + ["--continuous-batch-no-spec", "--max-reserved-kv-bytes", "17179869184"])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .exactQwen35MTPWithContinuousBatch)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse(common + ["--continuous-dynamic-pld", "--max-reserved-kv-bytes", "17179869184"])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .exactQwen35MTPWithContinuousBatch)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-candidates", "/models/q4,/models/q8",
                "--model", "qwen35-exact",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--exact-qwen35-mtp",
                "--mtp-drafter-path", "/models/qwen35-drafter",
            ])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .exactQwen35MTPWithQuantSource)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-pick-only",
                "--quant-candidates", "/models/q4,/models/q8",
                "--exact-qwen35-mtp",
                "--mtp-drafter-path", "/models/qwen35-drafter",
            ])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .exactQwen35MTPWithQuantSource)
        }

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-pick-only",
                "--auto-quant", "mlx-community/Qwen3-8B",
                "--exact-qwen35-mtp",
                "--mtp-drafter-path", "/models/qwen35-drafter",
            ])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .exactQwen35MTPWithQuantSource)
        }
    }

    // MARK: - --plan-concurrency: opt-in planning slot count. The fit-check verdict is computed at
    // concurrency=1 by default (the shipped, byte-identical behavior); an operator who will actually
    // run N concurrent decode streams can request the stricter, concurrency-aware verdict with this
    // flag. Absent → nil (default 1 at the call site). See
    // docs/task-inbox/2026-08-18-fit-check-concurrency-kv-undercount.md (option 2).

    func testPlanConcurrencyParsesForTheFitCheck() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--continuous-batch-no-spec",
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--max-reserved-kv-bytes", "17179869184",
            "--plan-concurrency", "4",
        ])
        XCTAssertEqual(arguments.planConcurrency, 4)
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--plan-concurrency N"))
    }

    func testPlanConcurrencyDefaultsNilWhenAbsent() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])
        XCTAssertNil(arguments.planConcurrency)
    }

    func testPlanConcurrencyRejectsNonPositiveInteger() {
        for bad in ["0", "-1", "abc"] {
            XCTAssertThrowsError(
                try FastMLXServeArguments.parse([
                    "--model-path", "/models/fixture",
                    "--model", "fixture",
                    "--memory-limit-bytes", "68719476736",
                    "--cache-limit-bytes", "8589934592",
                    "--plan-concurrency", bad,
                ]),
                "expected --plan-concurrency \(bad) to fail closed"
            ) { error in
                XCTAssertEqual(
                    error as? FastMLXServeArgumentError,
                    .invalidPositiveInteger("--plan-concurrency"))
            }
        }
    }

    /// The pick-only dry-run also honors --plan-concurrency so the resolved quant is evaluated at the
    /// concurrency the operator will actually serve, not always at 1.
    func testPlanConcurrencyHonoredOnQuantPickOnly() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--quant-candidates", "/models/q4,/models/q8",
            "--plan-concurrency", "3",
        ])
        XCTAssertTrue(arguments.quantPickOnly)
        XCTAssertEqual(arguments.planConcurrency, 3)
    }

    // MARK: - --tier: the operator-intent serve dial, carried as a RAW string (validated in
    // HarnessCore at the serve call site, same dependency-boundary idiom as --kv-quant).

    func testServeTierParsesAsRawStringOnQuantPickOnly() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--quant-candidates", "/models/q4,/models/q8",
            "--tier", "maxfit",
        ])
        XCTAssertEqual(arguments.serveTier, "maxfit")
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--tier TIER"))
    }

    /// Carried verbatim even when unrecognized — value validation is HarnessCore's job at the serve
    /// call site (fail-closed there), not the parser's.
    func testServeTierCarriedVerbatimEvenWhenUnrecognized() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--quant-candidates", "/models/q4",
            "--tier", "turbo",
        ])
        XCTAssertEqual(arguments.serveTier, "turbo")
    }

    func testServeTierRequiresAValue() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-pick-only",
                "--quant-candidates", "/models/q4",
                "--tier",
            ])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .missingValue("--tier"))
        }
    }

    func testServeTierDefaultsNil() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--quant-candidates", "/models/q4",
        ])
        XCTAssertNil(arguments.serveTier)
    }

    // MARK: - --prefer: the quant auto-pick ranking axis (context|quality), carried as a RAW string
    // (validated in HarnessCore.QuantPickPreference at the serve call site, same dependency-boundary
    // idiom as --tier/--kv-quant). Consumed only by the quant auto-pick; nil = context-first default.

    func testPreferModeParsesAsRawStringOnQuantPickOnly() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--quant-candidates", "/models/q4,/models/q8",
            "--prefer", "quality",
        ])
        XCTAssertEqual(arguments.preferMode, "quality")
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--prefer MODE"))
    }

    /// Carried verbatim even when unrecognized — value validation is HarnessCore's job at the serve
    /// call site (fail-closed there), not the parser's. Proves ServingCore stays HarnessCore-free.
    func testPreferModeIsCarriedVerbatimEvenWhenUnrecognized() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-candidates", "/models/q4",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--prefer", "speed",
        ])
        XCTAssertEqual(arguments.preferMode, "speed")
    }

    func testPreferModeRequiresAValue() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-pick-only",
                "--quant-candidates", "/models/q4",
                "--prefer",
            ])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .missingValue("--prefer"))
        }
    }

    func testPreferModeDefaultsNil() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--quant-candidates", "/models/q4",
        ])
        XCTAssertNil(arguments.preferMode)
    }

    // MARK: - --auto-quant: enumerate HF quant repo names for a base id (OFFLINE half). The network
    // probe/download half is not built, so it is an enumerate-only source — mutually exclusive with
    // the local --quant-candidates source and usable only under --quant-pick-only. Carried as a RAW
    // base string; HarnessCore.QuantCandidateSourcer enumerates at the serve call site.

    func testAutoQuantParsesAsRawBaseOnQuantPickOnly() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--auto-quant", "mlx-community/Qwen3-8B",
        ])
        XCTAssertEqual(arguments.autoQuantBase, "mlx-community/Qwen3-8B")
        XCTAssertTrue(arguments.quantPickOnly)
        XCTAssertTrue(arguments.quantCandidateDirectories.isEmpty)
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--auto-quant"))
    }

    func testAutoQuantDefaultsNil() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-pick-only",
            "--quant-candidates", "/models/q4",
        ])
        XCTAssertNil(arguments.autoQuantBase)
    }

    func testAutoQuantWithoutPickOnlyFailsClosed() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--auto-quant", "mlx-community/Qwen3-8B",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError, .autoQuantRequiresPickOnly)
        }
    }

    func testAutoQuantAndQuantCandidatesMutuallyExclusive() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-pick-only",
                "--auto-quant", "mlx-community/Qwen3-8B",
                "--quant-candidates", "/models/q4",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError, .autoQuantWithCandidates)
        }
    }

    func testAutoQuantWhitespaceOnlyBaseFailsClosed() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-pick-only",
                "--auto-quant", "   ",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError, .invalidAutoQuantBase)
        }
    }

    func testAutoQuantRequiresAValue() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-pick-only",
                "--auto-quant",
            ])
        ) { error in
            XCTAssertEqual(error as? FastMLXServeArgumentError, .missingValue("--auto-quant"))
        }
    }
}
