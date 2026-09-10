import XCTest

@testable import ServingCore

/// `--default-sampling`: parse-time-only flag selecting whether a request omitting sampling
/// parameters (default `off`, greedy argmax) resolves the served checkpoint's own
/// `generation_config.json` sampling subset instead (`generation-config`). Nothing yet calls
/// `GenerationConfigSamplingDefaults.load(contentsOf:)` or threads a resolved default into a
/// serve route -- this increment is the operator-facing flag and its fail-closed refusals only.
/// Mirrors `SampledMTPServeArgumentTests.swift`'s shape, in a NEW file (this flag's write set
/// does not include the existing, much larger `FastMLXServeArgumentsTests.swift`).
final class DefaultSamplingServeArgumentTests: XCTestCase {
    /// Acceptance criterion 1: the flag is absent -> defaultSampling == .off (the default),
    /// preserving today's behavior byte-for-byte.
    func testDefaultSamplingAbsentDefaultsToOff() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
        ])

        XCTAssertEqual(arguments.defaultSampling, .off)
    }

    /// Acceptance criterion 2: --default-sampling off parses explicitly to .off.
    func testDefaultSamplingExplicitOffParses() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--default-sampling", "off",
        ])

        XCTAssertEqual(arguments.defaultSampling, .off)
    }

    /// Acceptance criterion 3 (the happy path): --default-sampling generation-config parses to
    /// .generationConfig on an otherwise plain scalar serve command -- proving the refusals below
    /// are NOT a blanket rejection of the flag.
    func testDefaultSamplingGenerationConfigParsesOnPlainScalarServe() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--default-sampling", "generation-config",
        ])

        XCTAssertEqual(arguments.defaultSampling, .generationConfig)
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--default-sampling"))
    }

    /// Acceptance criterion 4: an unrecognized value throws the specific .invalidDefaultSampling
    /// error, not merely "it threw".
    func testDefaultSamplingInvalidValueThrowsTheSpecificError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--default-sampling", "bogus",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .invalidDefaultSampling)
        }
    }

    // MARK: - Refusal 1: --scripted

    /// Acceptance criterion 5 (load-bearing): --default-sampling generation-config combined with
    /// --scripted throws the SPECIFIC .defaultSamplingWithScripted error.
    func testDefaultSamplingWithScriptedThrowsTheSpecificError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--default-sampling", "generation-config",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .defaultSamplingWithScripted)
        }
    }

    /// Control for refusal 1: the identical --scripted command line with --default-sampling off
    /// parses successfully -- proving the refusal above is caused by the flag's VALUE, not by
    /// some pre-existing incompatibility between --scripted and the rest of the command line.
    func testDefaultSamplingOffWithScriptedParsesSuccessfully() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--scripted",
            "--default-sampling", "off",
        ])

        XCTAssertEqual(arguments.defaultSampling, .off)
    }

    // MARK: - Refusal 2: --quant-pick-only

    /// Acceptance criterion 6 (load-bearing): --default-sampling generation-config combined with
    /// --quant-pick-only throws the SPECIFIC .defaultSamplingWithQuantPickOnly error.
    func testDefaultSamplingWithQuantPickOnlyThrowsTheSpecificError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--quant-candidates", "/models/a,/models/b",
                "--quant-pick-only",
                "--default-sampling", "generation-config",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .defaultSamplingWithQuantPickOnly)
        }
    }

    /// Control for refusal 2: the identical --quant-pick-only command line with
    /// --default-sampling off parses successfully.
    func testDefaultSamplingOffWithQuantPickOnlyParsesSuccessfully() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--quant-candidates", "/models/a,/models/b",
            "--quant-pick-only",
            "--default-sampling", "off",
        ])

        XCTAssertEqual(arguments.defaultSampling, .off)
        XCTAssertTrue(arguments.quantPickOnly)
    }

    // MARK: - Refusal 3: continuous batching

    /// Acceptance criterion 7 (load-bearing): --default-sampling generation-config combined with
    /// --continuous-batch-no-spec throws the SPECIFIC .defaultSamplingWithContinuousBatch error.
    func testDefaultSamplingWithContinuousBatchThrowsTheSpecificError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--continuous-batch-no-spec",
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--max-reserved-kv-bytes", "4294967296",
                "--default-sampling", "generation-config",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .defaultSamplingWithContinuousBatch)
        }
    }

    /// Control for refusal 3: the identical continuous-batch command line with
    /// --default-sampling off parses successfully.
    func testDefaultSamplingOffWithContinuousBatchParsesSuccessfully() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--continuous-batch-no-spec",
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--max-reserved-kv-bytes", "4294967296",
            "--default-sampling", "off",
        ])

        XCTAssertEqual(arguments.defaultSampling, .off)
    }

    // MARK: - Refusal 4: --exact-qwen35-mtp

    /// Acceptance criterion 8 (load-bearing): --default-sampling generation-config combined with
    /// --exact-qwen35-mtp throws the SPECIFIC .defaultSamplingWithExactQwen35MTP error.
    func testDefaultSamplingWithExactQwen35MTPThrowsTheSpecificError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--exact-qwen35-mtp",
                "--mtp-drafter-path", "/models/drafter",
                "--default-sampling", "generation-config",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .defaultSamplingWithExactQwen35MTP)
        }
    }

    /// Control for refusal 4: the identical --exact-qwen35-mtp command line with
    /// --default-sampling off parses successfully.
    func testDefaultSamplingOffWithExactQwen35MTPParsesSuccessfully() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--exact-qwen35-mtp",
            "--mtp-drafter-path", "/models/drafter",
            "--default-sampling", "off",
        ])

        XCTAssertEqual(arguments.defaultSampling, .off)
        XCTAssertTrue(arguments.exactQwen35MTP)
    }

    // MARK: - Refusal 5: --qwen4exp-mtp without --qwen4exp-sampled-mtp

    /// Acceptance criterion 9 (load-bearing): --default-sampling generation-config combined with
    /// --qwen4exp-mtp WITHOUT --qwen4exp-sampled-mtp throws the SPECIFIC
    /// .defaultSamplingWithInCheckpointMTPRequiresSampledMTP error -- the throughput-cliff case.
    func testDefaultSamplingWithInCheckpointMTPWithoutSampledMTPThrowsTheSpecificError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-mtp",
                "--default-sampling", "generation-config",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .defaultSamplingWithInCheckpointMTPRequiresSampledMTP)
        }
    }

    /// Control for refusal 5: the identical --qwen4exp-mtp (without sampled-mtp) command line
    /// with --default-sampling off parses successfully.
    func testDefaultSamplingOffWithInCheckpointMTPWithoutSampledMTPParsesSuccessfully() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
            "--default-sampling", "off",
        ])

        XCTAssertEqual(arguments.defaultSampling, .off)
        XCTAssertEqual(arguments.inCheckpointMTPSelection, .converted4Bit)
        XCTAssertFalse(arguments.sampledMTPBlockDecisionsEnabled)
    }

    // MARK: - The cliff does not exist when sampled MTP is on

    /// Acceptance criterion 10: --qwen4exp-mtp --qwen4exp-sampled-mtp --default-sampling
    /// generation-config is ACCEPTED -- the throughput cliff refusal above does not fire once
    /// sampled MTP acceleration is itself enabled.
    func testDefaultSamplingWithInCheckpointMTPAndSampledMTPIsAccepted() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
            "--qwen4exp-sampled-mtp",
            "--default-sampling", "generation-config",
        ])

        XCTAssertEqual(arguments.defaultSampling, .generationConfig)
        XCTAssertEqual(arguments.inCheckpointMTPSelection, .converted4Bit)
        XCTAssertTrue(arguments.sampledMTPBlockDecisionsEnabled)
    }
}
