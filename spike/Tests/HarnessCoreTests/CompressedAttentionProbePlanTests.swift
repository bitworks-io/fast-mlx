import XCTest

@testable import HarnessCore

final class CompressedAttentionProbePlanTests: XCTestCase {
    private let cleanSHA = String(repeating: "a", count: 40)

    func testPromotionPlanPinsTheAuthenticatedLongContextGeometry() throws {
        let plan = try CompressedAttentionProbePlan(
            operation: .swiftLMQuantizedAttention,
            contextTokens: 32_768,
            queryTokens: 1,
            prefillChunkTokens: 512,
            outputTokens: 128,
            stopTokenIDs: [151_643, 151_645],
            batchSize: 1,
            queryHeadCount: 64,
            kvHeadCount: 8,
            headDimension: 128,
            dtype: .float16,
            mask: .none,
            layout: .affine(
                keyBits: 4, valueBits: 4,
                keyGroupSize: 64, valueGroupSize: 64),
            warmupRuns: 1,
            measuredRuns: 3,
            seed: 2_026_071_800,
            workloadNonce: "fused-kv-qwen3-g64",
            harnessGitSHA: cleanSHA,
            promotionEvidence: true)

        XCTAssertTrue(plan.isPromotionContext)
        XCTAssertEqual(plan.gqaRepeatCount, 8)
        XCTAssertEqual(plan.totalKVScalarCount, 67_108_864)
        XCTAssertEqual(plan.operation.rawValue, "swiftlm-quantized-attention")
    }

    func testExploratoryPlanAllowsSmallFixturesButCannotClaimPromotion() throws {
        let plan = try CompressedAttentionProbePlan(
            operation: .materializeThenSDPA,
            contextTokens: 16,
            queryTokens: 1,
            prefillChunkTokens: 8,
            outputTokens: 4,
            stopTokenIDs: [],
            batchSize: 1,
            queryHeadCount: 8,
            kvHeadCount: 2,
            headDimension: 128,
            dtype: .bfloat16,
            mask: .causal,
            layout: .affine(
                keyBits: 4, valueBits: 2,
                keyGroupSize: 64, valueGroupSize: 64),
            warmupRuns: 1,
            measuredRuns: 1,
            seed: 7,
            workloadNonce: "small-fixture",
            harnessGitSHA: cleanSHA,
            promotionEvidence: false)

        XCTAssertFalse(plan.isPromotionContext)
        XCTAssertEqual(plan.gqaRepeatCount, 4)
    }

    func testPromotionRejectsUnapprovedContextAndTooFewMeasuredRuns() {
        XCTAssertThrowsError(try makePlan(
            contextTokens: 16_384,
            measuredRuns: 3,
            promotionEvidence: true)) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbePlanError,
                    .unapprovedPromotionContext(16_384))
            }

        XCTAssertThrowsError(try makePlan(
            contextTokens: 8_192,
            measuredRuns: 2,
            promotionEvidence: true)) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbePlanError,
                    .insufficientPromotionRuns(2))
            }
    }

    func testOperationAndLayoutMustDescribeTheExecutedPrimitive() {
        XCTAssertThrowsError(try makePlan(
            operation: .fp16SDPA,
            layout: .affine(
                keyBits: 4, valueBits: 4,
                keyGroupSize: 64, valueGroupSize: 64))) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbePlanError,
                    .operationLayoutMismatch)
            }

        XCTAssertThrowsError(try makePlan(
            operation: .swiftLMQuantizedAttention,
            layout: .affine(
                keyBits: 4, valueBits: 2,
                keyGroupSize: 64, valueGroupSize: 64))) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbePlanError,
                    .unsupportedStockQuantizedLayout)
            }

        XCTAssertNoThrow(try makePlan(
            operation: .materializeThenSDPA,
            layout: .kvarn(
                keyBits: 4, valueBits: 2, groupSize: 128,
                sinkTokens: 128, iterations: 8)))
    }

    func testGeometryAndQuantizationFailClosedBeforeAllocation() {
        XCTAssertThrowsError(try makePlan(
            queryHeadCount: 10,
            kvHeadCount: 3)) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbePlanError,
                    .invalidGQAGeometry)
            }

        XCTAssertThrowsError(try makePlan(
            headDimension: 96,
            layout: .affine(
                keyBits: 4, valueBits: 4,
                keyGroupSize: 64, valueGroupSize: 64))) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbePlanError,
                    .invalidLayoutGeometry)
            }

        XCTAssertThrowsError(try makePlan(batchSize: 0)) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbePlanError,
                .invalidTensorGeometry)
        }
    }

    func testProvenanceAndRunIdentityRejectUnstableValues() {
        XCTAssertThrowsError(try makePlan(workloadNonce: "contains space")) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbePlanError,
                .invalidWorkloadNonce)
        }

        XCTAssertThrowsError(try makePlan(harnessGitSHA: "dirty")) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbePlanError,
                .invalidHarnessGitSHA)
        }

        XCTAssertThrowsError(try makePlan(seed: -1)) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbePlanError,
                .invalidSeed)
        }

        XCTAssertThrowsError(try makePlan(queryTokens: 513)) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbePlanError,
                .invalidWorkloadShape)
        }

        XCTAssertThrowsError(try makePlan(stopTokenIDs: [2, 1, 1])) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbePlanError,
                .invalidStopTokenIDs)
        }
    }

    func testScalarCountOverflowIsRejectedWithoutTrapping() {
        XCTAssertThrowsError(try makePlan(
            contextTokens: Int.max,
            batchSize: Int.max,
            queryHeadCount: 2,
            kvHeadCount: 1,
            headDimension: 128)) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbePlanError,
                    .arithmeticOverflow)
        }

        XCTAssertThrowsError(try makePlan(measuredRuns: Int.max)) {
            XCTAssertEqual(
                $0 as? CompressedAttentionProbePlanError,
                .invalidRunCounts)
        }

        XCTAssertThrowsError(try makePlan(
            contextTokens: 1,
            queryTokens: 1,
            batchSize: 1,
            queryHeadCount: Int.max,
            kvHeadCount: 1,
            headDimension: 128)) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbePlanError,
                    .arithmeticOverflow)
            }
    }

    func testOutputBoundariesRejectAliasesAndSymbolicLinks() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let evidence = root.appendingPathComponent("probe.jsonl")
        let progressAlias = root.appendingPathComponent("./probe.jsonl")
        XCTAssertThrowsError(try makePlan(
            evidenceOutputPath: evidence.path,
            progressOutputPath: progressAlias.path)) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbePlanError,
                    .outputPathCollision(progressAlias.path))
            }

        let target = root.appendingPathComponent("target.jsonl")
        let link = root.appendingPathComponent("evidence-link.jsonl")
        try Data().write(to: target)
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: target)
        XCTAssertThrowsError(try makePlan(
            evidenceOutputPath: link.path,
            progressOutputPath: root.appendingPathComponent("progress.json").path)) {
                XCTAssertEqual(
                    $0 as? CompressedAttentionProbePlanError,
                    .symbolicLinkOutput(link.path))
            }
    }

    private func makePlan(
        operation: CompressedAttentionProbeOperation = .swiftLMQuantizedAttention,
        contextTokens: Int = 8_192,
        queryTokens: Int = 1,
        prefillChunkTokens: Int = 512,
        outputTokens: Int = 128,
        stopTokenIDs: [Int] = [151_643, 151_645],
        batchSize: Int = 1,
        queryHeadCount: Int = 8,
        kvHeadCount: Int = 2,
        headDimension: Int = 128,
        layout: CompressedAttentionProbeLayout = .affine(
            keyBits: 4, valueBits: 4,
            keyGroupSize: 64, valueGroupSize: 64),
        measuredRuns: Int = 3,
        seed: Int = 7,
        workloadNonce: String = "fused-kv-fixture",
        harnessGitSHA: String? = nil,
        promotionEvidence: Bool = false,
        evidenceOutputPath: String = "compressed-attention-probe.jsonl",
        progressOutputPath: String = "compressed-attention-probe.progress.json"
    ) throws -> CompressedAttentionProbePlan {
        try CompressedAttentionProbePlan(
            operation: operation,
            contextTokens: contextTokens,
            queryTokens: queryTokens,
            prefillChunkTokens: prefillChunkTokens,
            outputTokens: outputTokens,
            stopTokenIDs: stopTokenIDs,
            batchSize: batchSize,
            queryHeadCount: queryHeadCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            dtype: .float16,
            mask: .none,
            layout: layout,
            warmupRuns: 1,
            measuredRuns: measuredRuns,
            seed: seed,
            workloadNonce: workloadNonce,
            harnessGitSHA: harnessGitSHA ?? cleanSHA,
            promotionEvidence: promotionEvidence,
            evidenceOutputPath: evidenceOutputPath,
            progressOutputPath: progressOutputPath)
    }
}
