import XCTest

@testable import ServingCore

final class FastMLXServeArgumentsTests: XCTestCase {
    func testScriptedModeRetainsTransportDefaults() throws {
        let arguments = try FastMLXServeArguments.parse(["--scripted"])

        XCTAssertEqual(arguments.backend, .scripted)
        XCTAssertEqual(arguments.host, "127.0.0.1")
        XCTAssertEqual(arguments.port, 8_080)
        XCTAssertEqual(arguments.model, "fastmlx-scripted")
        XCTAssertNil(arguments.evidencePath)
        XCTAssertFalse(arguments.showHelp)
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
