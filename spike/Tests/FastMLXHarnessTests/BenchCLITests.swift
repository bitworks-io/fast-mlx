import Foundation
import HarnessCore
import XCTest
@testable import fastmlx_harness

final class BenchCLITests: XCTestCase {
    private let checkpointContentSHA256 = String(repeating: "d", count: 64)
    private let kvtunerArguments = [
        "--model", "/models/Qwen3-32B-4bit",
        "--kv-quant", "kvtuner-g128-b3.046875",
        "--cell-id", "kvtuner-g128-b3.046875",
        "--matrix-id", "kvarn-qwen3-32b-v1",
        "--kvtuner-schedule", "/evidence/qualification-bundle.json",
        "--workload-nonce", "kvarn-frontier-20260718",
    ]

    func testKVTunerBenchPlanRequiresFrozenScheduleMatrixAndWorkloadNonce() throws {
        let base = [
            "--model", "/models/Qwen3-32B-4bit",
            "--kv-quant", "kvtuner-g128-b3.046875",
            "--cell-id", "kvtuner-g128-b3.046875",
        ]

        XCTAssertThrowsError(try parseBenchPlan(Flags(base))) {
            XCTAssertEqual($0 as? KVTunerCLIError, .missingSchedule)
        }
        XCTAssertThrowsError(try parseBenchPlan(Flags(base + [
            "--kvtuner-schedule", "/evidence/qualification-bundle.json",
        ]))) {
            XCTAssertEqual($0 as? BenchCLIError, .missingMatrixID)
        }
        XCTAssertThrowsError(try parseBenchPlan(Flags(base + [
            "--kvtuner-schedule", "/evidence/qualification-bundle.json",
            "--matrix-id", "kvarn-qwen3-32b-v1",
        ]))) {
            XCTAssertEqual($0 as? BenchCLIError, .missingWorkloadNonce)
        }
    }

    func testKVTunerBenchPlanAcceptsAnExactAuditedWorkload() throws {
        let plan = try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--kv-quant", "kvtuner-g128-b3.046875",
            "--cell-id", "kvtuner-g128-b3.046875",
            "--matrix-id", "kvarn-qwen3-32b-v1",
            "--kvtuner-schedule", "/evidence/qualification-bundle.json",
            "--workload-nonce", "kvarn-frontier-20260718",
            "--runs", "3",
        ]))

        XCTAssertEqual(plan.kvQuantTier, "kvtuner-g128-b3.046875")
        XCTAssertEqual(plan.cellID, "kvtuner-g128-b3.046875")
        XCTAssertEqual(plan.matrixID, "kvarn-qwen3-32b-v1")
        XCTAssertEqual(
            plan.kvtunerSchedulePath,
            "/evidence/qualification-bundle.json")
        XCTAssertEqual(plan.workload.iterations, 4)
        XCTAssertEqual(plan.workload.nonce, "kvarn-frontier-20260718")
        XCTAssertEqual(plan.workload.basePrompt, defaultBenchPrompt)
        XCTAssertNil(plan.compressedKVAttention)
    }

    func testBenchPlanRequiresExplicitKnownAttentionRouteOnAffineBackedTiers() throws {
        let split = try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--kv-quant", "affine-k4v2-g64",
            "--kv-attention", "split-affine-quantized-mm",
            "--checkpoint-content-sha256", checkpointContentSHA256,
        ]))
        XCTAssertEqual(
            split.compressedKVAttention,
            .splitAffineQuantizedMM)
        XCTAssertEqual(
            split.compressedKVAttentionExpectedCheckpointContentSHA256,
            checkpointContentSHA256)

        let materialized = try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--kv-quant", "affine-k4v2-g64",
            "--kv-attention", "materialize",
            "--checkpoint-content-sha256", checkpointContentSHA256,
        ]))
        XCTAssertEqual(materialized.compressedKVAttention, .materialize)

        XCTAssertThrowsError(try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--kv-quant", "fp16",
            "--kv-attention", "split-affine-quantized-mm",
            "--checkpoint-content-sha256", checkpointContentSHA256,
        ]))) {
            XCTAssertEqual(
                $0 as? BenchCLIError,
                .unsupportedAttentionTier("fp16"))
        }
        XCTAssertThrowsError(try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--kv-quant", "affine-k4v2-g64",
            "--kv-attention", "custom-kernel",
            "--checkpoint-content-sha256", checkpointContentSHA256,
        ]))) {
            XCTAssertEqual(
                $0 as? BenchCLIError,
                .unknownAttentionOperation("custom-kernel"))
        }

        XCTAssertThrowsError(try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--kv-quant", "affine-k4v2-g64",
            "--kv-attention", "split-affine-quantized-mm",
        ]))) {
            XCTAssertEqual(
                $0 as? BenchCLIError,
                .invalidCheckpointContentSHA256(""))
        }
        XCTAssertThrowsError(try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--checkpoint-content-sha256", checkpointContentSHA256,
        ]))) {
            XCTAssertEqual(
                $0 as? BenchCLIError,
                .invalidCheckpointContentSHA256(
                    checkpointContentSHA256))
        }
    }

    func testKVTunerBenchPlanRejectsTierCellMismatchAndCustomPrompt() throws {
        let common = [
            "--model", "/models/Qwen3-32B-4bit",
            "--kv-quant", "kvtuner-g128-b3.046875",
            "--matrix-id", "kvarn-qwen3-32b-v1",
            "--kvtuner-schedule", "/evidence/qualification-bundle.json",
            "--workload-nonce", "kvarn-frontier-20260718",
        ]
        XCTAssertThrowsError(try parseBenchPlan(Flags(common + [
            "--cell-id", "kvtuner-g128-b3.125",
        ]))) {
            XCTAssertEqual(
                $0 as? KVTunerCLIError,
                .tierCellMismatch(
                    tier: "kvtuner-g128-b3.046875",
                    cell: "kvtuner-g128-b3.125"))
        }
        XCTAssertThrowsError(try parseBenchPlan(Flags(common + [
            "--cell-id", "kvtuner-g128-b3.046875",
            "--prompt", "A custom prompt",
        ]))) {
            XCTAssertEqual($0 as? BenchCLIError, .unauditedKVTunerPrompt)
        }
    }

    func testBenchPlanRejectsEveryInputOutputAlias() throws {
        XCTAssertThrowsError(try parseBenchPlan(Flags(kvtunerArguments + [
            "--evidence", "/evidence/qualification-bundle.json",
        ]))) {
            XCTAssertEqual(
                $0 as? BenchCLIError,
                .outputPathCollision("/evidence/qualification-bundle.json"))
        }
        XCTAssertThrowsError(try parseBenchPlan(Flags(kvtunerArguments + [
            "--csv", "/evidence/qualification-bundle.json",
        ]))) {
            XCTAssertEqual(
                $0 as? BenchCLIError,
                .outputPathCollision("/evidence/qualification-bundle.json"))
        }
        XCTAssertThrowsError(try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--evidence", "/evidence/runtime.jsonl",
            "--csv", "/evidence/runtime.jsonl",
        ]))) {
            XCTAssertEqual(
                $0 as? BenchCLIError,
                .outputPathCollision("/evidence/runtime.jsonl"))
        }
    }

    func testBenchPlanRejectsSymbolicLinkOutputs() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let target = directory.appendingPathComponent("target.csv")
        XCTAssertTrue(FileManager.default.createFile(
            atPath: target.path, contents: Data()))
        let link = directory.appendingPathComponent("alias.csv")
        try FileManager.default.createSymbolicLink(
            atPath: link.path, withDestinationPath: target.path)

        XCTAssertThrowsError(try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--csv", link.path,
        ]))) {
            XCTAssertEqual(
                $0 as? BenchCLIError,
                .symbolicLinkOutput(link.path))
        }
    }

    func testBenchPlanBoundsRunsBeforeMaterializingTheWorkload() throws {
        XCTAssertThrowsError(try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--runs", "101",
        ]))) {
            XCTAssertEqual($0 as? BenchCLIError, .invalidRuns(101))
        }
    }

    func testBenchPlanBuildsBoundedDeterministicLongPrompt() throws {
        let plan = try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--prompt-repeat", "3",
        ]))
        XCTAssertEqual(plan.promptRepeat, 3)
        XCTAssertEqual(
            plan.workload.basePrompt,
            Array(repeating: defaultBenchPrompt, count: 3)
                .joined(separator: "\n"))

        for value in [0, 4_097] {
            XCTAssertThrowsError(try parseBenchPlan(Flags([
                "--model", "/models/Qwen3-32B-4bit",
                "--prompt-repeat", String(value),
            ]))) {
                XCTAssertEqual(
                    $0 as? BenchCLIError,
                    .invalidPromptRepeat(value))
            }
        }
    }

    func testBenchContextWindowFailsClosedBeforeGeneration() throws {
        XCTAssertNoThrow(try validateBenchContextWindow(
            promptTokenCounts: [32_768, 32_770],
            maxTokens: 8,
            maxPositionEmbeddings: 40_960))

        XCTAssertThrowsError(try validateBenchContextWindow(
            promptTokenCounts: [40_953],
            maxTokens: 8,
            maxPositionEmbeddings: 40_960)) {
            XCTAssertEqual(
                $0 as? BenchCLIError,
                .contextLimitExceeded(
                    promptTokens: 40_953,
                    maxTokens: 8,
                    limit: 40_960))
        }
        XCTAssertThrowsError(try validateBenchContextWindow(
            promptTokenCounts: [Int.max],
            maxTokens: 8,
            maxPositionEmbeddings: Int.max))
    }

    func testBenchEngagementRejectsVacuousLossyRuns() throws {
        XCTAssertNoThrow(try validateBenchRuntimeEngagement(
            tier: "fp16", engagement: .init(),
            expectedKVTunerLayerCount: nil))
        XCTAssertThrowsError(try validateBenchRuntimeEngagement(
            tier: "affine-k4v2-g64",
            engagement: .init(["affine_tokens": 0]),
            expectedKVTunerLayerCount: nil))
        XCTAssertThrowsError(try validateBenchRuntimeEngagement(
            tier: "kvarn-k4v2-g128",
            engagement: .init([
                "kvarn_tokens": 256,
                "kvarn_completed_tiles": 0,
                "kvarn_compressed_tokens": 0,
                "kvarn_codec_iterations": 8,
            ]),
            expectedKVTunerLayerCount: nil))
        XCTAssertThrowsError(try validateBenchRuntimeEngagement(
            tier: "kvtuner-g128-b3.046875",
            engagement: .init([
                "kvtuner_tokens": 256,
                "kvtuner_layers": 63,
            ]),
            expectedKVTunerLayerCount: 64))
    }

    func testBenchEngagementAcceptsMeasuredLossyRuns() throws {
        XCTAssertNoThrow(try validateBenchRuntimeEngagement(
            tier: "affine-k4v2-g64",
            engagement: .init([
                "affine_tokens": 256,
                "affine_layers": 64,
                "affine_capacity_tokens": 512,
                "affine_payload_bytes": 1_000,
                "affine_metadata_bytes": 100,
                "affine_control_bytes": 4,
                "affine_workspace_bytes": 200,
            ]),
            expectedKVTunerLayerCount: nil))
        XCTAssertNoThrow(try validateBenchRuntimeEngagement(
            tier: "kvarn-k4v2-g128-i16",
            engagement: .init([
                "kvarn_tokens": 256,
                "kvarn_completed_tiles": 1,
                "kvarn_compressed_tokens": 128,
                "kvarn_codec_iterations": 16,
                "kvarn_layers": 64,
                "kvarn_capacity_tokens": 512,
                "kvarn_payload_bytes": 1_000,
                "kvarn_metadata_bytes": 100,
                "kvarn_control_bytes": 4,
                "kvarn_workspace_bytes": 200,
            ]),
            expectedKVTunerLayerCount: nil))
        XCTAssertNoThrow(try validateBenchRuntimeEngagement(
            tier: "kvtuner-g128-b3.046875",
            engagement: .init([
                "kvtuner_tokens": 256,
                "kvtuner_layers": 64,
                "kvtuner_capacity_tokens": 512,
                "kvtuner_payload_bytes": 1_000,
                "kvtuner_metadata_bytes": 100,
                "kvtuner_control_bytes": 256,
                "kvtuner_workspace_bytes": 200,
            ]),
            expectedKVTunerLayerCount: 64))

        XCTAssertThrowsError(try validateBenchRuntimeEngagement(
            tier: "affine-k4v2-g64",
            engagement: .init(["affine_tokens": 256]),
            expectedKVTunerLayerCount: nil))
    }

    func testBenchEngagementAuthenticatesSplitWorkspaceBreakdown() throws {
        let splitCounts = EngagementCounters([
            "affine_tokens": 256,
            "affine_layers": 64,
            "affine_capacity_tokens": 512,
            "affine_payload_bytes": 1_000,
            "affine_metadata_bytes": 100,
            "affine_control_bytes": 4,
            "affine_workspace_bytes": 256,
            "affine_materialization_bytes": 0,
            "affine_attention_workspace_bytes": 256,
            "affine_attention_split": 1,
            "affine_attention_materialized": 0,
        ])
        XCTAssertNoThrow(try validateBenchRuntimeEngagement(
            tier: "affine-k4v2-g64",
            engagement: splitCounts,
            expectedKVTunerLayerCount: nil,
            requestedCompressedKVAttention: .splitAffineQuantizedMM))
        XCTAssertThrowsError(try validateBenchRuntimeEngagement(
            tier: "affine-k4v2-g64",
            engagement: splitCounts,
            expectedKVTunerLayerCount: nil,
            requestedCompressedKVAttention: .materialize))

        var falseZero = splitCounts.counts
        falseZero["affine_workspace_bytes"] = 0
        falseZero["affine_attention_workspace_bytes"] = 0
        XCTAssertThrowsError(try validateBenchRuntimeEngagement(
            tier: "affine-k4v2-g64",
            engagement: .init(falseZero),
            expectedKVTunerLayerCount: nil,
            requestedCompressedKVAttention: .splitAffineQuantizedMM))
    }

    func testBenchCompressedAttentionWorkspaceMustFitRawMLXPeakReceipt() throws {
        let splitCounts = EngagementCounters([
            "affine_workspace_bytes": 256,
            "affine_materialization_bytes": 0,
            "affine_attention_workspace_bytes": 256,
            "affine_attention_split": 1,
            "affine_attention_materialized": 0,
        ])

        XCTAssertNoThrow(try validateBenchCompressedAttentionMemoryReceipt(
            tier: "affine-k4v2-g64",
            request: .splitAffineQuantizedMM,
            engagement: splitCounts,
            maxMLXPeakBytes: 256))
        XCTAssertThrowsError(try validateBenchCompressedAttentionMemoryReceipt(
            tier: "affine-k4v2-g64",
            request: .splitAffineQuantizedMM,
            engagement: splitCounts,
            maxMLXPeakBytes: 255)) {
                XCTAssertEqual(
                    $0 as? BenchCLIError,
                    .compressedAttentionWorkspaceExceedsMLXPeak(
                        workspaceBytes: 256,
                        peakBytes: 255))
            }
        XCTAssertNoThrow(try validateBenchCompressedAttentionMemoryReceipt(
            tier: "affine-k4v2-g64",
            request: nil,
            engagement: .init(),
            maxMLXPeakBytes: 0))
    }

    func testBenchBindingUsesObservedCountersRatherThanRequestedMode() throws {
        let config = Data(
            #"{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_size":5120,"num_hidden_layers":64,"num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,"max_position_embeddings":40960,"use_sliding_window":false}"#.utf8)
        let admission = try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: .load(
                exactModelConfigData: config,
                checkpointManifestHash: "0123456789abcdef",
                checkpointContentSHA256:
                    String(repeating: "d", count: 64),
                tokenizerSHA256:
                    String(repeating: "a", count: 64)))
        let split = EngagementCounters([
            "affine_attention_split": 1,
            "affine_attention_materialized": 0,
        ])

        let binding = try makeBenchCompressedKVAttentionRuntimeBinding(
            tier: "affine-k4v2-g64",
            request: .splitAffineQuantizedMM,
            admission: admission,
            engagement: split)
        XCTAssertEqual(
            binding?.observedOperation, .splitQuantizedMM)

        XCTAssertThrowsError(
            try makeBenchCompressedKVAttentionRuntimeBinding(
                tier: "affine-k4v2-g64",
                request: .materialize,
                admission: admission,
                engagement: split)
        ) {
            XCTAssertEqual(
                $0 as? BenchCLIError,
                .missingCompressedAttentionEvidence)
        }
    }

    func testNonKVTunerBenchKeepsCustomPromptAndRejectsSchedule() throws {
        let plan = try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--kv-quant", "affine-k4v2-g64",
            "--prompt", "A custom prompt",
        ]))
        XCTAssertEqual(plan.workload.basePrompt, "A custom prompt")
        XCTAssertEqual(plan.kvtunerSchedulePath, "")

        XCTAssertThrowsError(try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--kv-quant", "affine-k4v2-g64",
            "--cell-id", "affine-k8v2-g128",
        ]))) {
            XCTAssertEqual(
                $0 as? BenchCLIError,
                .tierCellMismatch(
                    tier: "affine-k4v2-g64",
                    cell: "affine-k8v2-g128"))
        }

        XCTAssertThrowsError(try parseBenchPlan(Flags([
            "--model", "/models/Qwen3-32B-4bit",
            "--kv-quant", "affine-k4v2-g64",
            "--kvtuner-schedule", "/evidence/qualification-bundle.json",
        ]))) {
            XCTAssertEqual(
                $0 as? KVTunerCLIError,
                .unexpectedSchedule(tier: "affine-k4v2-g64"))
        }
    }

    func testHistoricalBenchPayloadDecodesWithoutNewOptionalFields() throws {
        let data = try XCTUnwrap(
            #"{"label":"legacy","workload":"decode","mode":"none","decodeTokS":42.5,"ttftMs":120.3,"quant":"int4","kvQuantTier":"fp16","concurrency":1}"#
                .data(using: .utf8))
        let payload = try JSONDecoder().decode(BenchPayload.self, from: data)

        XCTAssertNil(payload.prefillTokS)
        XCTAssertNil(payload.prefillMs)
        XCTAssertNil(payload.promptTokensMin)
        XCTAssertNil(payload.promptTokensMax)
        XCTAssertNil(payload.kvtunerSchedule)
        XCTAssertNil(payload.engagementMax)
        XCTAssertNil(payload.workloadNonce)
        XCTAssertNil(payload.maxTokens)
        XCTAssertNil(payload.measuredRuns)
        XCTAssertNil(payload.promptTokenCountsByRun)
        XCTAssertNil(payload.prefillDurationSecondsByRun)
        XCTAssertNil(payload.prefillTokSByRun)
        XCTAssertNil(payload.decodeTokSByRun)
        XCTAssertNil(payload.ttftMsByRun)
        XCTAssertNil(payload.generatedTokenCountsByRun)
        XCTAssertNil(payload.memoryCacheLimitBytes)
        XCTAssertNil(payload.memoryRuns)
        XCTAssertNil(payload.maxSampledPhysicalFootprintBytes)
        XCTAssertNil(payload.maxMLXActiveBytes)
        XCTAssertNil(payload.maxMLXCacheBytes)
        XCTAssertNil(payload.maxMLXPeakBytes)
        XCTAssertNil(payload.compressedKVAttention)
        XCTAssertNil(payload.promptRepeat)
    }

    func testBenchCSVReadFailureIsNotTreatedAsAnEmptyFile() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let row = BenchRow(
            label: "runtime", workload: .decode, mode: .none,
            model: "Qwen3-32B", decodeTokS: 42, ttftMs: 100,
            quant: "int4", concurrency: 1, hardware: "M3 Ultra")
        XCTAssertThrowsError(
            try appendBenchCSVRow(row, to: directory.path))
    }
}
