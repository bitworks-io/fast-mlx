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

    func testScalarModeRejectsPartialRelativeOrUnsafeMemoryConfiguration() throws {
        // Item 1: omitting --cache-limit-bytes is no longer a missing-required-option error — the
        // absent flag reaches the backend as `nil`, which is the explicitness signal itself.
        let cacheOmitted = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "4096",
        ])
        XCTAssertEqual(
            cacheOmitted.backend,
            .scalar(
                modelDirectory: URL(fileURLWithPath: "/models/fixture", isDirectory: true),
                memoryLimitBytes: 4_096,
                cacheLimitBytes: nil))

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

    /// Item 1, the headline case: omitting BOTH --memory-limit-bytes and --cache-limit-bytes now
    /// parses successfully instead of throwing `.missingRequiredOption("--memory-limit-bytes")` —
    /// the absent flags reach the top-level `memoryLimitBytes` field AND the backend's own payload
    /// as `nil`, which is the explicitness signal a downstream sizer-derived figure should apply.
    /// This is what lets a parsed `FastMLXServeArguments` reach the host-probe path at all; the old
    /// required-option guard refused before the host was ever probed.
    func testScalarModeParsesSuccessfullyWithBothMemoryLimitsOmitted() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
        ])
        XCTAssertNil(arguments.memoryLimitBytes,
            "an omitted --memory-limit-bytes must reach the top-level field as nil")
        XCTAssertEqual(
            arguments.backend,
            .scalar(
                modelDirectory: URL(fileURLWithPath: "/models/fixture", isDirectory: true),
                memoryLimitBytes: nil,
                cacheLimitBytes: nil))
    }

    /// Item 3: the `--max-reserved-kv-bytes <= --memory-limit-bytes` guard must be SKIPPED (not
    /// silently compared against a bogus stand-in) when `--memory-limit-bytes` is absent. A reserved
    /// KV figure that would obviously exceed any real memory limit must still parse successfully —
    /// proving the comparison did not fire against some fabricated default. The "WITH memory
    /// present" half of this validation is already locked immediately above
    /// (`.reservedKVLimitExceedsMemoryLimit`, both flags supplied).
    func testMaxReservedKVBytesValidationSkippedWhenMemoryLimitOmitted() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--continuous-batch-no-spec",
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--cache-limit-bytes", "1024",
            "--max-reserved-kv-bytes", "999999999999",
        ])
        XCTAssertEqual(
            arguments.backend,
            .continuousBatchNoSpec(
                modelDirectory: URL(fileURLWithPath: "/models/fixture", isDirectory: true),
                memoryLimitBytes: nil,
                cacheLimitBytes: 1_024,
                maxReservedKVBytes: 999_999_999_999))
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

    /// Non-regression lock, UPDATED for the operator-budget-envelope-shape decision:
    /// `--memory-limit-bytes` is no longer a required option on any path (item 1) — an omitted flag
    /// is the explicitness signal that no operator budget should bind, not a missing requirement the
    /// sizer needs filled in. A normal candidates-mode serve (no --quant-pick-only) without the flag
    /// now parses successfully, carrying `memoryLimitBytes == nil` all the way to the backend.
    func testCandidatesModeWithoutPickOnlyParsesWithOmittedMemoryLimit() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-candidates", "/models/a,/models/b",
            "--model", "qwen3",
            "--cache-limit-bytes", "8589934592",
        ])
        XCTAssertNil(arguments.memoryLimitBytes,
            "an omitted --memory-limit-bytes must reach the top-level field as nil")
        XCTAssertEqual(
            arguments.backend,
            .scalar(
                modelDirectory: URL(fileURLWithPath: "/models/a", isDirectory: true),
                memoryLimitBytes: nil,
                cacheLimitBytes: 8_589_934_592))
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

    func testExactQwen35MTPParsesExplicit4BitArtifactSelection() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/qwen38-target",
            "--model", "qwen38-exact",
            "--memory-limit-bytes", "171798691840",
            "--cache-limit-bytes", "34359738368",
            "--exact-qwen35-mtp",
            "--exact-mtp-selection", "qwen38-27b-4bit-depth1",
            "--mtp-drafter-path", "/models/qwen38-drafter",
        ])

        XCTAssertEqual(arguments.exactMTPSelection, .qwen38_27B4BitDepth1)
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--exact-mtp-selection SELECTION"))
        XCTAssertTrue(FastMLXServeArguments.usage.contains("qwen38-27b-4bit-depth1"))
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

    // MARK: - --ngram-offload-plan: an absolute local path to an offloaded n-gram serving plan,
    // selecting the offloaded load path for a qwen4_exp checkpoint at the scalar load call site.
    // Default nil preserves today's load path. Not supported on continuous batching or
    // --exact-qwen35-mtp, since neither route reaches the scalar-load seam that consumes it.

    func testNgramOffloadPlanParsesAsAbsolutePathOnScalarServe() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
        ])

        XCTAssertEqual(
            arguments.ngramOffloadPlanURL,
            URL(fileURLWithPath: "/abs/path/plan.json"))
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains("--ngram-offload-plan PATH"))
    }

    func testNgramOffloadPlanRejectsRelativePath() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--ngram-offload-plan", "relative/plan.json",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .ngramOffloadPlanMustBeAbsolute)
        }
    }

    func testNgramOffloadPlanRejectedWithQuantPickOnly() {
        // --quant-pick-only returns early WITHOUT threading ngramOffloadPlanURL, so without this
        // guard the flag would be silently dropped rather than merely unused. Unlike
        // --mtp-drafter-path, this flag has no transitive block against a quant source.
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-candidates", "/models/a,/models/b",
                "--quant-pick-only",
                "--ngram-offload-plan", "/abs/path/plan.json",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .ngramOffloadPlanWithQuantPickOnly)
        }
    }

    func testNgramOffloadPlanRejectedWithQuantCandidates() {
        // The plan is sealed against ONE specific on-disk artifact. `resolveServedDirectory` runs
        // the quant auto-pick (QuantCandidateResolver.resolve) BEFORE resolveServingLimits' offload-
        // aware fit adjustment ever sees the request, so on a host where the full-resident figure
        // alone would refuse, the pick fails closed on the unadjusted figure and never reaches the
        // adjustment the offload plan exists to unlock — and even if it didn't, the winning candidate
        // need not be the artifact the plan was sealed against. Reject the combination outright rather
        // than let auto-pick silently pair the plan with a mismatched checkpoint.
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-candidates", "/models/a,/models/b",
                "--model", "qwen3",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--ngram-offload-plan", "/abs/path/plan.json",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .ngramOffloadPlanWithQuantCandidates)
        }
    }

    func testNgramOffloadPlanAloneRemainsAcceptedWithoutQuantCandidates() throws {
        // Non-regression: the new guard above must not over-reach and reject the flag when it is
        // used on its own, as documented in testNgramOffloadPlanParsesAsAbsolutePathOnScalarServe.
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
        ])

        XCTAssertEqual(
            arguments.ngramOffloadPlanURL,
            URL(fileURLWithPath: "/abs/path/plan.json"))
        XCTAssertTrue(arguments.quantCandidateDirectories.isEmpty)
    }

    func testQuantCandidatesAloneRemainsAcceptedWithoutNgramOffloadPlan() throws {
        // Non-regression: the new guard above must not over-reach and reject --quant-candidates
        // when it is used on its own, without an offload plan in play.
        let arguments = try FastMLXServeArguments.parse([
            "--quant-candidates", "/models/a,/models/b",
            "--model", "qwen3",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])

        XCTAssertEqual(arguments.quantCandidateDirectories.count, 2)
        XCTAssertNil(arguments.ngramOffloadPlanURL)
    }

    func testNgramOffloadPlanRequiresAValue() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--ngram-offload-plan",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingValue("--ngram-offload-plan"))
        }
    }

    func testNgramOffloadPlanDefaultsNilAndDoesNotChangeOtherwiseValidParse() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])
        XCTAssertNil(arguments.ngramOffloadPlanURL)
        XCTAssertEqual(
            arguments.backend,
            .scalar(
                modelDirectory: URL(fileURLWithPath: "/models/fixture", isDirectory: true),
                memoryLimitBytes: 68_719_476_736,
                cacheLimitBytes: 8_589_934_592))
    }

    func testNgramOffloadPlanRejectedWithContinuousBatching() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--continuous-batch-no-spec",
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "103079215104",
                "--cache-limit-bytes", "8589934592",
                "--max-reserved-kv-bytes", "17179869184",
                "--ngram-offload-plan", "/abs/path/plan.json",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .ngramOffloadPlanWithContinuousBatch)
        }
    }

    func testNgramOffloadPlanRejectedWithExactQwen35MTP() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/qwen35-target",
                "--model", "qwen35-exact",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--exact-qwen35-mtp",
                "--mtp-drafter-path", "/models/qwen35-drafter",
                "--ngram-offload-plan", "/abs/path/plan.json",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .ngramOffloadPlanWithExactQwen35MTP)
        }
    }

    func testNgramOffloadPlanAppearsInHelpOutput() throws {
        let arguments = try FastMLXServeArguments.parse(["--help"])
        XCTAssertTrue(arguments.showHelp)
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains("--ngram-offload-plan PATH"))
    }

    // MARK: - --qwen4exp-mtp: a bare opt-in flag that loads the in-checkpoint Qwen4-Exp (Flash
    // Next) converted 4-bit MTP drafter from the served target's own checkpoint and gates it at
    // startup. Only one artifact is serve-eligible today, so it resolves to the single
    // FastMLXInCheckpointMTPSelection.converted4Bit case rather than taking a value. Requires
    // --ngram-offload-plan (the marker-family admission gate only admits this family when an
    // offloaded plan resolved) and cannot be combined with --scripted (which loads no model).
    // Every other conflicting mode is refused TRANSITIVELY through the --ngram-offload-plan
    // requirement itself.

    /// Acceptance criterion 1: --qwen4exp-mtp with --ngram-offload-plan and an otherwise valid
    /// loaded-model invocation parses to inCheckpointMTPSelection == .converted4Bit.
    func testInCheckpointMTPWithNgramOffloadPlanParsesAsConverted4Bit() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
        ])

        XCTAssertEqual(arguments.inCheckpointMTPSelection, .converted4Bit)
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--qwen4exp-mtp"))
    }

    /// Acceptance criterion 2: absent the flag, inCheckpointMTPSelection == nil — existing
    /// invocations (including the plain --ngram-offload-plan case) are unaffected.
    func testInCheckpointMTPDefaultsNilAndDoesNotChangeOtherwiseValidParse() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
        ])

        XCTAssertNil(arguments.inCheckpointMTPSelection)
    }

    /// Acceptance criterion 3: --qwen4exp-mtp without --ngram-offload-plan throws
    /// .qwen4ExpMTPRequiresNGramOffloadPlan.
    func testInCheckpointMTPRequiresNgramOffloadPlan() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--qwen4exp-mtp",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .qwen4ExpMTPRequiresNGramOffloadPlan)
        }
    }

    /// Acceptance criterion 4: --qwen4exp-mtp with --scripted throws .qwen4ExpMTPWithScripted.
    /// --ngram-offload-plan is included so the requires-check above does not fire first — this
    /// isolates the scripted-specific refusal.
    func testInCheckpointMTPWithScripted() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-mtp",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .qwen4ExpMTPWithScripted)
        }
    }

    /// Acceptance criterion 5a (transitive coverage): --qwen4exp-mtp + --ngram-offload-plan +
    /// continuous batching throws the EXISTING .ngramOffloadPlanWithContinuousBatch error, proving
    /// no separate --qwen4exp-mtp-specific continuous-batch refusal is needed.
    func testInCheckpointMTPWithNgramOffloadPlanAndContinuousBatchThrowsExistingNgramError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--continuous-batch-no-spec",
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "103079215104",
                "--cache-limit-bytes", "8589934592",
                "--max-reserved-kv-bytes", "17179869184",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-mtp",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .ngramOffloadPlanWithContinuousBatch)
        }
    }

    /// Acceptance criterion 5b (transitive coverage): --qwen4exp-mtp + --ngram-offload-plan +
    /// --exact-qwen35-mtp throws the EXISTING .ngramOffloadPlanWithExactQwen35MTP error.
    func testInCheckpointMTPWithNgramOffloadPlanAndExactQwen35MTPThrowsExistingNgramError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/qwen35-target",
                "--model", "qwen35-exact",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--exact-qwen35-mtp",
                "--mtp-drafter-path", "/models/qwen35-drafter",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-mtp",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .ngramOffloadPlanWithExactQwen35MTP)
        }
    }

    /// Acceptance criterion 5c (transitive coverage): --qwen4exp-mtp + --ngram-offload-plan +
    /// --quant-pick-only throws the EXISTING .ngramOffloadPlanWithQuantPickOnly error.
    func testInCheckpointMTPWithNgramOffloadPlanAndQuantPickOnlyThrowsExistingNgramError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-candidates", "/models/a,/models/b",
                "--quant-pick-only",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-mtp",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .ngramOffloadPlanWithQuantPickOnly)
        }
    }

    /// Acceptance criterion 5d (transitive coverage): --qwen4exp-mtp + --ngram-offload-plan +
    /// --quant-candidates throws the EXISTING .ngramOffloadPlanWithQuantCandidates error.
    func testInCheckpointMTPWithNgramOffloadPlanAndQuantCandidatesThrowsExistingNgramError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-candidates", "/models/a,/models/b",
                "--model", "qwen3",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-mtp",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .ngramOffloadPlanWithQuantCandidates)
        }
    }

    /// Acceptance criterion 6: --qwen4exp-mtp is a recognized supported option, not rejected as
    /// .unknownArgument — the happy-path parse above (criterion 1) succeeding is the positive
    /// proof; this asserts the negative directly against the specific failure mode a forgotten
    /// supportedOptions entry would produce.
    func testInCheckpointMTPIsNotRejectedAsUnknownArgument() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
        ])
        XCTAssertNotNil(arguments.inCheckpointMTPSelection)

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-mtp",
                "--qwen4exp-mtp",
            ])
        ) { error in
            // Duplicate detection also proves the option is registered: an unregistered option
            // would throw .unknownArgument on the FIRST occurrence, never reaching .duplicateOption.
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .duplicateOption("--qwen4exp-mtp"))
        }
    }

    /// Acceptance criterion: --ngram-offload-plan is consumed only at the scalar-load seam
    /// (`loadScalarServingBackend` → `ScalarServingModelLoadConfiguration`); the transport-only
    /// --scripted backend loads no model and never reaches that seam, so pairing the two would
    /// otherwise silently drop the operator's explicit plan. --scripted --ngram-offload-plan alone
    /// (no --qwen4exp-mtp) throws .ngramOffloadPlanWithScripted.
    func testNgramOffloadPlanRejectedWithScripted() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--ngram-offload-plan", "/abs/path/plan.json",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .ngramOffloadPlanWithScripted)
        }
    }

    /// Acceptance criterion: --quant-pick-only is its own early-return backend-mode (`backend:
    /// nil`) that never reaches the general --scripted conflict check further down. Without an
    /// explicit guard, --scripted --quant-pick-only would silently discard the requested
    /// --scripted transport-only backend and fall through into a quant-pick run instead. This
    /// asserts the pairing throws the same .conflictingBackendModes used for every other
    /// backend-mode conflict in this file.
    func testQuantPickOnlyRejectedWithScripted() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--quant-candidates", "/models/a,/models/b",
                "--quant-pick-only",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .conflictingBackendModes)
        }
    }

    /// Acceptance criterion: .ngramOffloadPlanWithContinuousBatch fires for BOTH continuous
    /// routes; testNgramOffloadPlanRejectedWithContinuousBatching above covers only
    /// --continuous-batch-no-spec, leaving the --continuous-dynamic-pld arm covered by
    /// construction only. This exercises that arm directly.
    func testNgramOffloadPlanRejectedWithContinuousDynamicPLD() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--continuous-dynamic-pld",
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "103079215104",
                "--cache-limit-bytes", "8589934592",
                "--max-reserved-kv-bytes", "17179869184",
                "--ngram-offload-plan", "/abs/path/plan.json",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .ngramOffloadPlanWithContinuousBatch)
        }
    }

    // MARK: - --chat-template: an absolute local path to a chat-template file that overrides the
    // served checkpoint's own resolved template for BOTH rendering and the boot attestation
    // probe. Default nil preserves today's resolution unchanged.

    func testChatTemplateParsesAsAbsolutePathOnScalarServe() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--chat-template", "/abs/path/chat_template.jinja",
        ])

        XCTAssertEqual(
            arguments.chatTemplateURL,
            URL(fileURLWithPath: "/abs/path/chat_template.jinja"))
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains("--chat-template PATH"))
    }

    func testChatTemplateRejectsRelativePath() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--chat-template", "relative/chat_template.jinja",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .chatTemplateMustBeAbsolute)
        }
    }

    func testChatTemplateRequiresAValue() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--chat-template",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .missingValue("--chat-template"))
        }
    }

    func testChatTemplateDefaultsNilAndDoesNotChangeOtherwiseValidParse() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])
        XCTAssertNil(arguments.chatTemplateURL)
        XCTAssertEqual(
            arguments.backend,
            .scalar(
                modelDirectory: URL(fileURLWithPath: "/models/fixture", isDirectory: true),
                memoryLimitBytes: 68_719_476_736,
                cacheLimitBytes: 8_589_934_592))
    }

    func testChatTemplateRejectedWithScripted() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--chat-template", "/abs/path/chat_template.jinja",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .chatTemplateWithScripted)
        }
    }

    func testChatTemplateRejectedWithContinuousBatchNoSpec() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--continuous-batch-no-spec",
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "103079215104",
                "--cache-limit-bytes", "8589934592",
                "--max-reserved-kv-bytes", "17179869184",
                "--chat-template", "/abs/path/chat_template.jinja",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .chatTemplateWithContinuousBatch)
        }
    }

    func testChatTemplateRejectedWithContinuousDynamicPLD() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--continuous-dynamic-pld",
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "103079215104",
                "--cache-limit-bytes", "8589934592",
                "--max-reserved-kv-bytes", "17179869184",
                "--chat-template", "/abs/path/chat_template.jinja",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .chatTemplateWithContinuousBatch)
        }
    }

    func testChatTemplateRejectedWithExactQwen35MTP() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/qwen35-target",
                "--model", "qwen35-exact",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--exact-qwen35-mtp",
                "--mtp-drafter-path", "/models/qwen35-drafter",
                "--chat-template", "/abs/path/chat_template.jinja",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .chatTemplateWithExactQwen35MTP)
        }
    }

    func testChatTemplateRejectedWithQuantPickOnly() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-candidates", "/models/a,/models/b",
                "--quant-pick-only",
                "--chat-template", "/abs/path/chat_template.jinja",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .chatTemplateWithQuantPickOnly)
        }
    }

    /// Non-regression: unlike --ngram-offload-plan, --chat-template is NOT refused when combined
    /// with the loaded (non-pick-only) --quant-candidates auto-pick route — the resolved winning
    /// directory still loads through the same scalar-load seam this flag targets.
    func testChatTemplateAcceptedWithLoadedQuantCandidates() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-candidates", "/models/a,/models/b",
            "--model", "qwen3",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--chat-template", "/abs/path/chat_template.jinja",
        ])

        XCTAssertEqual(
            arguments.chatTemplateURL,
            URL(fileURLWithPath: "/abs/path/chat_template.jinja"))
        XCTAssertEqual(arguments.quantCandidateDirectories.count, 2)
    }

    // MARK: - --fit-check-only: a dry-run flag that reports the `resolveServingLimits` verdict for
    // a single --model-path model directory (the SAME call the real serve makes, immediately before
    // the load that follows it in FastMLXServe.swift) and exits without loading weights. Refused
    // with --quant-pick-only (a different dry run), --force (which would suppress the verdict this
    // flag exists to learn), and --scripted (no model directory to check). Deliberately NOT threaded
    // through --quant-pick-only's early-return construction — it stays on the full loaded-model
    // parse path so it reaches the real load-adjacent seam unchanged.

    func testFitCheckOnlyParsesTrueAndDefaultsFalse() throws {
        let withFlag = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--fit-check-only",
        ])
        XCTAssertTrue(withFlag.fitCheckOnly)

        let withoutFlag = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])
        XCTAssertFalse(withoutFlag.fitCheckOnly)

        XCTAssertTrue(FastMLXServeArguments.usage.contains("--fit-check-only"))
    }

    /// Refusal 1: --fit-check-only with --quant-pick-only throws the specific
    /// .fitCheckOnlyWithQuantPickOnly case — asserting the REASON, not merely that parsing threw.
    func testFitCheckOnlyRejectedWithQuantPickOnly() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-candidates", "/models/a,/models/b",
                "--quant-pick-only",
                "--fit-check-only",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .fitCheckOnlyWithQuantPickOnly)
        }
    }

    /// Refusal 2: --fit-check-only with --force throws the specific .fitCheckOnlyWithForce case.
    /// --force exists to proceed past a red verdict; a dry run whose purpose is to LEARN the
    /// verdict must not also carry the flag that suppresses it.
    func testFitCheckOnlyRejectedWithForce() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--force",
                "--fit-check-only",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .fitCheckOnlyWithForce)
        }
    }

    /// Refusal 3: --fit-check-only with --scripted throws the specific .fitCheckOnlyWithScripted
    /// case. The transport-only scripted backend loads no model, so there is no fit to check.
    func testFitCheckOnlyRejectedWithScripted() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--fit-check-only",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .fitCheckOnlyWithScripted)
        }
    }

    /// Acceptance (anti-inertness): --fit-check-only together with --ngram-offload-plan AND
    /// --qwen4exp-mtp — the motivating cutover use case — parses successfully and preserves all
    /// three settings. If a future guard folds --fit-check-only into the --ngram-offload-plan
    /// refusal at :quantPickOnly's sibling check (see FastMLXServeArguments.swift's guard keyed on
    /// `ngramOffloadPlanURL != nil, quantPickOnly`), this test fails.
    func testFitCheckOnlyAcceptedWithNgramOffloadPlanAndInCheckpointMTP() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
            "--fit-check-only",
        ])

        XCTAssertTrue(arguments.fitCheckOnly)
        XCTAssertEqual(
            arguments.ngramOffloadPlanURL,
            URL(fileURLWithPath: "/abs/path/plan.json"))
        XCTAssertEqual(arguments.inCheckpointMTPSelection, .converted4Bit)
    }

    /// --fit-check-only must not perturb any other parsed field: compare every stored property
    /// (via `Mirror`, so a future field addition is covered automatically) between the same argv
    /// with and without the flag, excluding `fitCheckOnly` itself.
    func testFitCheckOnlyDoesNotPerturbOtherParsedFields() throws {
        let baseArguments: [String] = [
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
        ]
        let withoutFlag = try FastMLXServeArguments.parse(baseArguments)
        let withFlag = try FastMLXServeArguments.parse(baseArguments + ["--fit-check-only"])

        let withoutFields = Dictionary(
            uniqueKeysWithValues: Mirror(reflecting: withoutFlag).children.compactMap {
                child -> (String, String)? in
                guard let label = child.label else { return nil }
                return (label, String(describing: child.value))
            })
        let withFields = Dictionary(
            uniqueKeysWithValues: Mirror(reflecting: withFlag).children.compactMap {
                child -> (String, String)? in
                guard let label = child.label else { return nil }
                return (label, String(describing: child.value))
            })

        XCTAssertFalse(withoutFields.isEmpty)
        for (label, withoutValue) in withoutFields where label != "fitCheckOnly" {
            XCTAssertEqual(
                withFields[label], withoutValue,
                "field \(label) changed when --fit-check-only was added")
        }
        XCTAssertEqual(withoutFlag.fitCheckOnly, false)
        XCTAssertEqual(withFlag.fitCheckOnly, true)
    }

    // MARK: - --offload-plan-check-only: the SECOND dry run, with a different stop point.
    // --fit-check-only answers "does the declared budget arithmetic fit?" by reading the plan's
    // declared limits.maxResidentBytes and the safetensors headers -- it never touches the offload
    // artifacts, which is exactly why its attestation carries `offload_path_resolvable=unproven`.
    // --offload-plan-check-only answers the different question "is THIS host provisioned?": it
    // resolves the plan and runs the same pre-load verification the real load runs, then stops
    // before any weight load. Arithmetic fits != host provisioned, so the two are refused together
    // rather than given a silent precedence.

    func testOffloadPlanCheckOnlyParsesTrueAndDefaultsFalse() throws {
        let baseArguments: [String] = [
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
        ]

        let withFlag = try FastMLXServeArguments.parse(
            baseArguments + ["--offload-plan-check-only"])
        XCTAssertTrue(withFlag.offloadPlanCheckOnly)
        XCTAssertEqual(
            withFlag.ngramOffloadPlanURL,
            URL(fileURLWithPath: "/abs/path/plan.json"))
        XCTAssertEqual(withFlag.inCheckpointMTPSelection, .converted4Bit)

        // Unconditional control: the BYTE-IDENTICAL argv with only this flag removed parses and
        // leaves the field false, so the assertion above is attributable to the flag itself and
        // not to anything else on the line.
        let withoutFlag = try FastMLXServeArguments.parse(baseArguments)
        XCTAssertFalse(withoutFlag.offloadPlanCheckOnly)

        // An operator plans a production change window from --help. A flag that parses but is
        // undocumented is one the operator cannot discover at the moment it is needed.
        XCTAssertTrue(
            FastMLXServeArguments.usage.contains("--offload-plan-check-only"),
            "--offload-plan-check-only must be documented in the usage text")
    }

    /// Refusal 1: composing the two dry runs throws the specific
    /// .offloadPlanCheckOnlyWithFitCheckOnly case. They stop at different points and make
    /// non-confusable claims; silently honouring one and dropping the other is the trap.
    func testOffloadPlanCheckOnlyRejectedWithFitCheckOnly() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-mtp",
                "--fit-check-only",
                "--offload-plan-check-only",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .offloadPlanCheckOnlyWithFitCheckOnly)
        }
    }

    /// Refusal 2: --force suppresses the very refusal this dry run exists to surface, so a host
    /// with unresolvable offload artifacts could otherwise report "success". Mirrors
    /// testFitCheckOnlyRejectedWithForce's identical rationale.
    func testOffloadPlanCheckOnlyRejectedWithForce() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-mtp",
                "--force",
                "--offload-plan-check-only",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .offloadPlanCheckOnlyWithForce)
        }
    }

    /// Refusal 3: without a plan there is nothing to resolve, so the flag is meaningless rather
    /// than merely unused.
    func testOffloadPlanCheckOnlyRequiresNgramOffloadPlan() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--offload-plan-check-only",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .offloadPlanCheckOnlyRequiresNGramOffloadPlan)
        }
    }

    /// The TRANSITIVE-COVERAGE claim, proven rather than asserted in a comment.
    ///
    /// `offloadPlanCheckOnlyRequiresNGramOffloadPlan`'s doc comment justifies the ABSENCE of five
    /// --offload-plan-check-only-specific refusals by arguing they would be unreachable dead code:
    /// requiring --ngram-offload-plan means each conflicting mode is already refused by that
    /// flag's own guard. That argument is only sound if each combination actually refuses, and
    /// refuses for the REASON claimed -- a control asserting merely "it threw" would pass even if
    /// the refusal came from somewhere unrelated, which is exactly how a wrong-reason refusal
    /// slipped through this repository's live qualification once before. Each row below therefore
    /// pins the specific case. If any of these ever parses successfully, the doc comment is wrong
    /// and a real --offload-plan-check-only refusal is missing.
    func testOffloadPlanCheckOnlyConflictsAreCoveredTransitivelyByTheNgramPlanGuards() {
        let plan = ["--ngram-offload-plan", "/abs/path/plan.json"]
        let modelBase: [String] = [
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ]
        let flag = ["--offload-plan-check-only"]

        let cases: [(name: String, argv: [String], expected: FastMLXServeArgumentError)] = [
            (
                "continuous batching",
                ["--continuous-batch-no-spec"] + modelBase
                    + ["--max-reserved-kv-bytes", "17179869184"] + plan + flag,
                .ngramOffloadPlanWithContinuousBatch
            ),
            (
                "dynamic-PLD continuous batching",
                ["--continuous-dynamic-pld"] + modelBase
                    + ["--max-reserved-kv-bytes", "17179869184"] + plan + flag,
                .ngramOffloadPlanWithContinuousBatch
            ),
            (
                "--exact-qwen35-mtp",
                modelBase + ["--exact-qwen35-mtp"] + plan + flag,
                .ngramOffloadPlanWithExactQwen35MTP
            ),
            (
                "--quant-pick-only",
                modelBase + ["--quant-pick-only"] + plan + flag,
                .ngramOffloadPlanWithQuantPickOnly
            ),
            (
                "--quant-candidates",
                ["--quant-candidates", "/models/a,/models/b"] + modelBase + plan + flag,
                .ngramOffloadPlanWithQuantCandidates
            ),
            (
                "--scripted",
                ["--scripted"] + plan + flag,
                .ngramOffloadPlanWithScripted
            ),
        ]

        for testCase in cases {
            XCTAssertThrowsError(
                try FastMLXServeArguments.parse(testCase.argv),
                "\(testCase.name) with --offload-plan-check-only must refuse"
            ) { error in
                XCTAssertEqual(
                    error as? FastMLXServeArgumentError,
                    testCase.expected,
                    "\(testCase.name) refused for the wrong reason")
            }
        }
    }

    /// --offload-plan-check-only must not perturb any other parsed field. Mirrors
    /// testFitCheckOnlyDoesNotPerturbOtherParsedFields, including its Mirror-based sweep so a
    /// future field addition is covered without editing this test.
    func testOffloadPlanCheckOnlyDoesNotPerturbOtherParsedFields() throws {
        let baseArguments: [String] = [
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
        ]
        let withoutFlag = try FastMLXServeArguments.parse(baseArguments)
        let withFlag = try FastMLXServeArguments.parse(
            baseArguments + ["--offload-plan-check-only"])

        func fields(_ arguments: FastMLXServeArguments) -> [String: String] {
            Dictionary(
                uniqueKeysWithValues: Mirror(reflecting: arguments).children.compactMap {
                    child -> (String, String)? in
                    guard let label = child.label else { return nil }
                    return (label, String(describing: child.value))
                })
        }

        let withoutFields = fields(withoutFlag)
        let withFields = fields(withFlag)

        XCTAssertFalse(withoutFields.isEmpty)
        for (label, withoutValue) in withoutFields where label != "offloadPlanCheckOnly" {
            XCTAssertEqual(
                withFields[label], withoutValue,
                "field \(label) changed when --offload-plan-check-only was added")
        }
        XCTAssertNotEqual(
            withFields["offloadPlanCheckOnly"], withoutFields["offloadPlanCheckOnly"],
            "the flag itself must differ, or this test proves nothing")
    }

    // MARK: - fastMLXServeArgumentRefusalAnnounceLine: the machine-readable refusal line
    // `FastMLXServe.main`'s top-level `catch let error as FastMLXServeArgumentError` arm renders.
    // Without that arm, EVERY one of these 94 cases — thrown from the very first statement of
    // `run()` — survives to the top level unwrapped and traps via Swift's top-level fatalError
    // (exit 133, doubled message, no `reason=` token, unclassifiable by automation) instead of
    // exiting cleanly with exit(2). See `testFastMLXServeArgumentErrorCatchArmCallSitePinExists`
    // below for the structural pin on the arm itself — a passing renderer test here proves nothing
    // about whether the arm still calls it.

    func testRefusalLineRendersUnknownArgument() {
        let line = fastMLXServeArgumentRefusalAnnounceLine(.unknownArgument("--bogus"))

        XCTAssertEqual(
            line,
            "fastmlx-serve configuration=refused reason=invalid_arguments "
                + "detail=Unknown argument: --bogus")
        XCTAssertTrue(line.hasPrefix("fastmlx-serve configuration=refused reason=invalid_arguments"))
    }

    func testRefusalLineRendersDuplicateOption() {
        let line = fastMLXServeArgumentRefusalAnnounceLine(.duplicateOption("--port"))

        XCTAssertEqual(
            line,
            "fastmlx-serve configuration=refused reason=invalid_arguments "
                + "detail=--port may be specified only once")
    }

    func testRefusalLineRendersMissingValue() {
        let line = fastMLXServeArgumentRefusalAnnounceLine(.missingValue("--kv-quant"))

        XCTAssertEqual(
            line,
            "fastmlx-serve configuration=refused reason=invalid_arguments "
                + "detail=--kv-quant requires a value")
    }

    func testRefusalLineRendersInvalidPort() {
        let line = fastMLXServeArgumentRefusalAnnounceLine(.invalidPort)

        XCTAssertEqual(
            line,
            "fastmlx-serve configuration=refused reason=invalid_arguments "
                + "detail=--port must be an integer from 0 through 65535")
    }

    /// Multi-clause detail text (joined `"..." + "..."` in `description`), and the exact case that
    /// produced this task's live-host defect's sibling refusal shape (a flag rejected under a
    /// specific mode combination).
    func testRefusalLineRendersNgramOffloadPlanWithQuantCandidates() {
        let line = fastMLXServeArgumentRefusalAnnounceLine(.ngramOffloadPlanWithQuantCandidates)

        XCTAssertEqual(
            line,
            "fastmlx-serve configuration=refused reason=invalid_arguments "
                + "detail=--ngram-offload-plan is sealed against one specific artifact and cannot "
                + "be combined with --quant-candidates auto-pick across several candidate "
                + "directories; pass the single sealed model directory explicitly via --model-path "
                + "instead")
        XCTAssertTrue(line.contains("detail=--ngram-offload-plan"))
    }

    /// The other multi-clause case, and the actual live defect this task fixes: this is the case
    /// `--chat-template is not supported with continuous batching` belongs alongside (same "flag
    /// rejected under a mode combination" family) — proving the renderer covers that family, not
    /// just the single-value cases above.
    func testRefusalLineRendersChatTemplateWithContinuousBatch() {
        let line = fastMLXServeArgumentRefusalAnnounceLine(.chatTemplateWithContinuousBatch)

        XCTAssertEqual(
            line,
            "fastmlx-serve configuration=refused reason=invalid_arguments "
                + "detail=--chat-template is not supported with continuous batching")
    }

    /// Structural pin, precedent: `scripts/tests/test_serve_wrapper.py`'s
    /// `test_ngram_offload_plan_conflict_rule_still_exists_in_the_binary`. A renderer unit test
    /// above still passes if someone deletes the top-level `catch` arm in `FastMLXServe.swift` —
    /// the renderer would simply never run in production, and an argument-validation refusal would
    /// fall through to Swift's top-level fatalError trap (exit 133, doubled message) again. Read
    /// the ACTUAL source text of the call site and assert the arm still exists and its body still
    /// calls the renderer and `exit(2)`, rather than trusting the renderer alone.
    ///
    /// The path is resolved from `#filePath` (this test file's own compile-time absolute path),
    /// walking up to the repo root, never a hardcoded absolute path. If that resolution fails —
    /// read fails, arm text not found — this test FAILS with a named reason; it must never silently
    /// skip (this repo has been bitten before by a conditional assertion that skips instead of
    /// failing).
    func testFastMLXServeArgumentErrorCatchArmCallSitePinExists() {
        // Resolve the call site by SEARCHING ancestors for it, not by counting levels. The
        // repository layout is not the only one this file compiles under: the fleet sync script
        // deploys `spike/`'s CONTENTS as the package root on the fleet hosts, so the package sits
        // one directory shallower there and a fixed four-level walk lands on a path that does not
        // exist. `#filePath` is baked at compile time, so on the fleet that would be a spurious
        // FAILURE, not a skip -- and a pin that cries wolf gets muted, which is how a real pin dies.
        // Both layouts are accepted; neither found is still a hard failure.
        let candidateSuffixes = [
            ["spike", "Sources", "fastmlx-serve", "FastMLXServe.swift"],
            ["Sources", "fastmlx-serve", "FastMLXServe.swift"],
        ]
        var searchDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        var resolvedMainFile: URL?
        for _ in 0..<8 {
            for suffix in candidateSuffixes {
                let candidate = suffix.reduce(searchDirectory) { $0.appendingPathComponent($1) }
                if FileManager.default.fileExists(atPath: candidate.path) {
                    resolvedMainFile = candidate
                    break
                }
            }
            if resolvedMainFile != nil { break }
            searchDirectory.deleteLastPathComponent()
        }

        guard let mainFilePath = resolvedMainFile else {
            XCTFail(
                "could not locate FastMLXServe.swift by walking up from #filePath (\(#filePath)); "
                    + "the repo/package layout moved relative to this test file. Fix the search "
                    + "above rather than letting this pin go silent.")
            return
        }

        let source: String
        do {
            source = try String(contentsOf: mainFilePath, encoding: .utf8)
        } catch {
            XCTFail(
                "could not read FastMLXServe.swift from the #filePath-derived path "
                    + "\(mainFilePath.path) — repo layout moved relative to this test file, or "
                    + "#filePath resolution regressed; fix the path derivation above rather than "
                    + "letting this pin go silent: \(error)")
            return
        }

        let armMarker = "catch let error as FastMLXServeArgumentError"
        guard let armRange = source.range(of: armMarker) else {
            XCTFail(
                "FastMLXServe.swift no longer has a `\(armMarker)` catch arm; an "
                    + "argument-validation refusal thrown by FastMLXServeArguments.parse will fall "
                    + "through to Swift's top-level fatalError trap (exit 133, doubled message) "
                    + "instead of exiting cleanly with exit(2)")
            return
        }

        // Inspect only the text immediately following the arm's `catch` clause, not the whole
        // file, so this pin cannot be satisfied by an unrelated call to the renderer or to exit(2)
        // elsewhere in the file. 800 characters comfortably covers this arm's body (~542 chars
        // including its doc comment as of this writing) while staying well short of the next arm.
        let armBody = source[armRange.upperBound...].prefix(800)

        XCTAssertTrue(
            armBody.contains("fastMLXServeArgumentRefusalAnnounceLine"),
            "the FastMLXServeArgumentError catch arm no longer calls "
                + "fastMLXServeArgumentRefusalAnnounceLine; the refusal would go unrendered")
        XCTAssertTrue(
            armBody.contains("exit(2)"),
            "the FastMLXServeArgumentError catch arm no longer calls exit(2); the process would "
                + "fall through instead of exiting cleanly")
    }
}
