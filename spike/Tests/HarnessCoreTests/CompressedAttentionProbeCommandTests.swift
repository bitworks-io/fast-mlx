import XCTest

@testable import HarnessCore

final class CompressedAttentionProbeCommandTests: XCTestCase {
    private let cleanSHA = String(repeating: "a", count: 40)

    func testParsesAuthenticatedAffinePromotionCommand() throws {
        let command = try CompressedAttentionProbeCommand(
            arguments: affineArguments(),
            harnessGitSHA: cleanSHA)

        XCTAssertEqual(command.modelPath, "/models/qwen3-32b")
        XCTAssertEqual(command.modelID, "mlx-community/Qwen3-32B-4bit")
        XCTAssertEqual(command.plan.operation, .swiftLMQuantizedAttention)
        XCTAssertEqual(
            command.plan.layout,
            .affine(
                keyBits: 4,
                valueBits: 4,
                keyGroupSize: 64,
                valueGroupSize: 64))
        XCTAssertEqual(command.plan.contextTokens, 32_768)
        XCTAssertEqual(command.plan.queryTokens, 1)
        XCTAssertEqual(command.plan.stopTokenIDs, [151_643, 151_645])
        XCTAssertTrue(command.plan.promotionEvidence)
        XCTAssertEqual(command.memoryLimitBytes, 32 << 30)
        XCTAssertEqual(command.cacheLimitBytes, 8 << 30)
        XCTAssertEqual(command.wiredLimitBytes, 0)
    }

    func testParsesFP16ExploratoryCommandWithoutLayoutOptions() throws {
        var arguments = baseArguments()
        arguments += [
            "--operation", "fp16-sdpa",
            "--layout", "fp16",
            "--context-tokens", "128",
            "--query-tokens", "16",
            "--prefill-chunk-tokens", "32",
        ]
        replaceFlag("measured-runs", with: "1", in: &arguments)
        replaceFlag("promotion-evidence", with: "false", in: &arguments)

        let command = try CompressedAttentionProbeCommand(
            arguments: arguments,
            harnessGitSHA: cleanSHA)

        XCTAssertEqual(command.plan.layout, .fp16)
        XCTAssertEqual(command.plan.mask, .causal)
        XCTAssertEqual(command.plan.dtype, .float16)
        XCTAssertEqual(command.plan.queryTokens, 16)
        XCTAssertFalse(command.plan.promotionEvidence)
    }

    func testUnknownDuplicatePositionalAndMissingValuesFailClosed() {
        assertCommandError(
            affineArguments() + ["--mystery", "value"],
            .unknownFlag("mystery"))
        assertCommandError(
            affineArguments() + ["--seed", "8"],
            .duplicateFlag("seed"))
        assertCommandError(
            affineArguments() + ["stray"],
            .unexpectedArgument("stray"))
        var missingValue = affineArguments()
        removeFlag("seed", from: &missingValue)
        missingValue.append("--seed")
        assertCommandError(missingValue, .missingFlagValue("seed"))
    }

    func testMissingRequiredAndMalformedScalarFlagsFailClosed() {
        var missing = affineArguments()
        removeFlag("model-id", from: &missing)
        assertCommandError(missing, .missingRequiredFlag("model-id"))

        var malformed = affineArguments()
        replaceFlag("context-tokens", with: "many", in: &malformed)
        assertCommandError(
            malformed,
            .invalidInteger(flag: "context-tokens", value: "many"))

        var invalidBoolean = affineArguments()
        replaceFlag("promotion-evidence", with: "yes", in: &invalidBoolean)
        assertCommandError(
            invalidBoolean,
            .invalidBoolean(flag: "promotion-evidence", value: "yes"))
    }

    func testModelIDStopIDsAndEnumsFailClosed() {
        var badModel = affineArguments()
        replaceFlag("model-id", with: "/models/qwen", in: &badModel)
        assertCommandError(badModel, .invalidModelID("/models/qwen"))

        var badStops = affineArguments()
        replaceFlag("stop-token-ids", with: "2,1,2", in: &badStops)
        assertCommandError(badStops, .invalidStopTokenIDs("2,1,2"))

        var badOperation = affineArguments()
        replaceFlag("operation", with: "magic", in: &badOperation)
        assertCommandError(badOperation, .invalidOperation("magic"))

        var badDType = affineArguments()
        replaceFlag("dtype", with: "float32", in: &badDType)
        assertCommandError(badDType, .invalidDType("float32"))

        var badMask = affineArguments()
        replaceFlag("mask", with: "sliding", in: &badMask)
        assertCommandError(badMask, .invalidMask("sliding"))
    }

    func testLayoutSpecificFlagsCannotBeOmittedOrSmuggled() {
        var missingAffine = affineArguments()
        removeFlag("key-bits", from: &missingAffine)
        assertCommandError(
            missingAffine,
            .missingRequiredFlag("key-bits"))

        var fp16 = baseArguments()
        fp16 += [
            "--operation", "fp16-sdpa",
            "--layout", "fp16",
            "--context-tokens", "128",
            "--query-tokens", "1",
            "--prefill-chunk-tokens", "32",
            "--key-bits", "4",
        ]
        replaceFlag("measured-runs", with: "1", in: &fp16)
        replaceFlag("promotion-evidence", with: "false", in: &fp16)
        assertCommandError(fp16, .unusedLayoutFlag("key-bits"))
    }

    func testKVarNCommandParsesOnlyItsDeclaredGeometry() throws {
        var arguments = baseArguments()
        arguments += [
            "--operation", "materialize-then-sdpa",
            "--layout", "kvarn",
            "--context-tokens", "8192",
            "--query-tokens", "1",
            "--prefill-chunk-tokens", "128",
            "--key-bits", "4",
            "--value-bits", "2",
            "--group-size", "128",
            "--sink-tokens", "128",
            "--iterations", "8",
        ]

        let command = try CompressedAttentionProbeCommand(
            arguments: arguments,
            harnessGitSHA: cleanSHA)
        XCTAssertEqual(
            command.plan.layout,
            .kvarn(
                keyBits: 4,
                valueBits: 2,
                groupSize: 128,
                sinkTokens: 128,
                iterations: 8))

        arguments += ["--key-group-size", "128"]
        assertCommandError(arguments, .unusedLayoutFlag("key-group-size"))
    }

    func testMemorySettingsMustBeExplicitAndInternallyConsistent() {
        var missing = affineArguments()
        removeFlag("cache-limit-bytes", from: &missing)
        assertCommandError(
            missing,
            .missingRequiredFlag("cache-limit-bytes"))

        var inconsistent = affineArguments()
        replaceFlag(
            "cache-limit-bytes",
            with: String(64 << 30),
            in: &inconsistent)
        assertCommandError(inconsistent, .invalidMemorySettings)

        var negativeWired = affineArguments()
        replaceFlag("wired-limit-bytes", with: "-1", in: &negativeWired)
        assertCommandError(negativeWired, .invalidMemorySettings)
    }

    private func baseArguments() -> [String] {
        [
            "--model", "/models/qwen3-32b",
            "--model-id", "mlx-community/Qwen3-32B-4bit",
            "--output-tokens", "16",
            "--stop-token-ids", "151643,151645",
            "--batch-size", "1",
            "--query-heads", "64",
            "--kv-heads", "8",
            "--head-dimension", "128",
            "--dtype", "float16",
            "--mask", "causal",
            "--warmup-runs", "1",
            "--measured-runs", "3",
            "--seed", "7",
            "--workload-nonce", "fused-kv-qwen3-v1",
            "--promotion-evidence", "true",
            "--evidence", "/tmp/probe.json",
            "--progress", "/tmp/probe.progress.json",
            "--memory-limit-bytes", String(32 << 30),
            "--cache-limit-bytes", String(8 << 30),
            "--wired-limit-bytes", "0",
        ]
    }

    private func affineArguments() -> [String] {
        baseArguments() + [
            "--operation", "swiftlm-quantized-attention",
            "--layout", "affine",
            "--context-tokens", "32768",
            "--query-tokens", "1",
            "--prefill-chunk-tokens", "128",
            "--key-bits", "4",
            "--value-bits", "4",
            "--key-group-size", "64",
            "--value-group-size", "64",
        ]
    }

    private func assertCommandError(
        _ arguments: [String],
        _ expected: CompressedAttentionProbeCommandError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try CompressedAttentionProbeCommand(
                arguments: arguments,
                harnessGitSHA: cleanSHA),
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(
                error as? CompressedAttentionProbeCommandError,
                expected,
                file: file,
                line: line)
        }
    }

    private func removeFlag(_ flag: String, from arguments: inout [String]) {
        guard let index = arguments.firstIndex(of: "--\(flag)") else {
            XCTFail("missing fixture flag --\(flag)")
            return
        }
        arguments.removeSubrange(index ... index + 1)
    }

    private func replaceFlag(
        _ flag: String,
        with value: String,
        in arguments: inout [String]
    ) {
        guard let index = arguments.firstIndex(of: "--\(flag)") else {
            XCTFail("missing fixture flag --\(flag)")
            return
        }
        arguments[index + 1] = value
    }
}
