import XCTest

@testable import ServingCore

/// `--qwen4exp-sampled-mtp`: opt-in to let the in-checkpoint Qwen4-Exp MTP drafter propose SAMPLED
/// block decisions through a `SampledMTPBlockRuntimeDeciding` provider, instead of accelerating
/// only greedy requests. Requires `--qwen4exp-mtp` -- the drafter iterator `--qwen4exp-mtp` loads
/// is what this flag's provider plugs into; without it there is nothing to hand the provider to.
/// Mirrors `FastMLXServeArgumentsTests.swift`'s own `--qwen4exp-mtp` section's acceptance-criterion
/// numbering style, in a NEW file (this flag's write set does not include the existing, much
/// larger `FastMLXServeArgumentsTests.swift`).
final class SampledMTPServeArgumentTests: XCTestCase {
    /// Acceptance criterion 1: --qwen4exp-sampled-mtp with --qwen4exp-mtp (and an otherwise valid
    /// loaded-model invocation) parses to sampledMTPBlockDecisionsEnabled == true.
    func testSampledMTPWithInCheckpointMTPParsesEnabled() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
            "--qwen4exp-sampled-mtp",
        ])

        XCTAssertTrue(arguments.sampledMTPBlockDecisionsEnabled)
        XCTAssertEqual(arguments.inCheckpointMTPSelection, .converted4Bit)
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--qwen4exp-sampled-mtp"))
    }

    /// Acceptance criterion 2: absent the flag, sampledMTPBlockDecisionsEnabled == false -- existing
    /// invocations (including the plain --qwen4exp-mtp case with no sampled opt-in) are unaffected.
    func testSampledMTPDefaultsFalseAndDoesNotChangeOtherwiseValidParse() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
        ])

        XCTAssertFalse(arguments.sampledMTPBlockDecisionsEnabled)
    }

    /// Acceptance criterion 3 (load-bearing): --qwen4exp-sampled-mtp WITHOUT --qwen4exp-mtp throws
    /// the SPECIFIC `.sampledMTPRequiresInCheckpointMTP` case -- not merely "parsing threw". A
    /// test that only asserted `XCTAssertThrowsError` would still pass if the command were refused
    /// for a completely unrelated reason (e.g. a missing --model-path), which would prove nothing
    /// about this flag's own requirement.
    func testSampledMTPWithoutInCheckpointMTPThrowsTheSpecificError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-sampled-mtp",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .sampledMTPRequiresInCheckpointMTP)
        }
    }

    /// Same acceptance criterion as above, isolated further: even a BARE --qwen4exp-sampled-mtp
    /// with no other flag at all still throws the specific requirement error, not some other
    /// missing-argument error that would fire first if the ordering were wrong.
    func testSampledMTPAloneThrowsTheSpecificError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--qwen4exp-sampled-mtp",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .sampledMTPRequiresInCheckpointMTP)
        }
    }

    /// Acceptance criterion 4: --qwen4exp-sampled-mtp is a recognized supported option, not
    /// rejected as .unknownArgument. The happy-path parse above (criterion 1) succeeding is the
    /// positive proof; this asserts the negative directly against the specific failure mode a
    /// forgotten `supportedOptions` entry would produce, mirroring
    /// `testInCheckpointMTPIsNotRejectedAsUnknownArgument`'s identical shape.
    func testSampledMTPIsNotRejectedAsUnknownArgument() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--ngram-offload-plan", "/abs/path/plan.json",
            "--qwen4exp-mtp",
            "--qwen4exp-sampled-mtp",
        ])
        XCTAssertTrue(arguments.sampledMTPBlockDecisionsEnabled)

        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--model-path", "/models/fixture",
                "--model", "fixture",
                "--memory-limit-bytes", "68719476736",
                "--cache-limit-bytes", "8589934592",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-mtp",
                "--qwen4exp-sampled-mtp",
                "--qwen4exp-sampled-mtp",
            ])
        ) { error in
            // Duplicate detection also proves the option is registered: an unregistered option
            // would throw .unknownArgument on the FIRST occurrence, never reaching
            // .duplicateOption.
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .duplicateOption("--qwen4exp-sampled-mtp"))
        }
    }

    /// Acceptance criterion 5 (transitive coverage): --qwen4exp-sampled-mtp requiring
    /// --qwen4exp-mtp already gives it the SAME transitive coverage --qwen4exp-mtp itself gets
    /// through --ngram-offload-plan -- e.g. pairing with --scripted throws the EXISTING, more
    /// specific `.qwen4ExpMTPWithScripted` (from the --qwen4exp-mtp / --scripted check, which runs
    /// before the sampled-specific requirement would even matter) rather than a redundant
    /// sampled-specific scripted refusal. --ngram-offload-plan is included so a REQUIRED-option
    /// check does not fire first and mask which refusal actually fired.
    func testSampledMTPWithScriptedThrowsExistingInCheckpointMTPError() {
        XCTAssertThrowsError(
            try FastMLXServeArguments.parse([
                "--scripted",
                "--ngram-offload-plan", "/abs/path/plan.json",
                "--qwen4exp-mtp",
                "--qwen4exp-sampled-mtp",
            ])
        ) { error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError,
                .qwen4ExpMTPWithScripted)
        }
    }
}
