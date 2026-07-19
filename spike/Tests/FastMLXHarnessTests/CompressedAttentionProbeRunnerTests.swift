import HarnessCore
import XCTest

@testable import fastmlx_harness

final class CompressedAttentionProbeRunnerTests: XCTestCase {
    private let cleanSHA = String(repeating: "a", count: 40)

    func testStockQuantizedAttentionConsumesPackedStateAndMatchesItsOracle() async throws {
        let result = try await CompressedAttentionProbeRunner().runFixture(
            plan: makePlan(
                operation: .swiftLMQuantizedAttention,
                layout: .affine(
                    keyBits: 4, valueBits: 4,
                    keyGroupSize: 64, valueGroupSize: 64)))

        XCTAssertEqual(result.outputShape, [1, 8, 1, 128])
        XCTAssertTrue(result.valuesFinite)
        XCTAssertTrue(
            result.structuralEquivalent,
            "max=\(result.maxAbsoluteError) mean=\(result.meanAbsoluteError)")
        XCTAssertTrue(
            result.top1Matches,
            "max=\(result.maxAbsoluteError) mean=\(result.meanAbsoluteError)")
        XCTAssertLessThanOrEqual(result.maxAbsoluteError, 2e-3)
        XCTAssertGreaterThan(result.persistentBytes, 0)
        XCTAssertEqual(result.materializationWorkspaceBytes, 0)
    }

    func testMaterializeThenAttendReportsTheFullAffineWorkspace() async throws {
        let result = try await CompressedAttentionProbeRunner().runFixture(
            plan: makePlan(
                operation: .materializeThenSDPA,
                layout: .affine(
                    keyBits: 4, valueBits: 2,
                    keyGroupSize: 64, valueGroupSize: 64)))

        XCTAssertEqual(result.outputShape, [1, 8, 1, 128])
        XCTAssertTrue(result.valuesFinite)
        XCTAssertTrue(result.structuralEquivalent)
        XCTAssertTrue(result.top1Matches)
        XCTAssertGreaterThan(result.persistentBytes, 0)
        XCTAssertLessThan(result.persistentBytes, result.materializationWorkspaceBytes)
        XCTAssertEqual(result.materializationWorkspaceBytes, 65_536)
    }

    func testCausalPrefillShapeIsNotCollapsedIntoDecodeShape() async throws {
        let result = try await CompressedAttentionProbeRunner().runFixture(
            plan: makePlan(
                operation: .swiftLMQuantizedAttention,
                queryTokens: 16,
                mask: .causal,
                layout: .affine(
                    keyBits: 4, valueBits: 4,
                    keyGroupSize: 64, valueGroupSize: 64)))

        XCTAssertEqual(result.outputShape, [1, 8, 16, 128])
        XCTAssertTrue(
            result.structuralEquivalent,
            "max=\(result.maxAbsoluteError) mean=\(result.meanAbsoluteError)")
        XCTAssertTrue(
            result.top1Matches,
            "max=\(result.maxAbsoluteError) mean=\(result.meanAbsoluteError)")
    }

    func testCausalMaskIsLowerRightAligned() async throws {
        let values = try await CompressedAttentionProbeRunner()
            .runCausalAlignmentFixture()

        XCTAssertEqual(values.count, 2)
        XCTAssertEqual(values[0], 2, accuracy: 0.01)
        XCTAssertEqual(values[1], 26.5, accuracy: 0.01)
    }

    func testKVarNFixtureRefusesBeforeDispatch() async throws {
        let plan = try makePlan(
            operation: .materializeThenSDPA,
            layout: .kvarn(
                keyBits: 4,
                valueBits: 2,
                groupSize: 128,
                sinkTokens: 128,
                iterations: 8))

        do {
            _ = try await CompressedAttentionProbeRunner().runFixture(plan: plan)
            XCTFail("unsupported KVarN fixture must fail closed")
        } catch {
            XCTAssertEqual(
                error as? CompressedAttentionProbeRunnerError,
                .unsupportedLayout)
        }
    }

    private func makePlan(
        operation: CompressedAttentionProbeOperation,
        queryTokens: Int = 1,
        mask: CompressedAttentionProbeMask = .none,
        layout: CompressedAttentionProbeLayout
    ) throws -> CompressedAttentionProbePlan {
        try CompressedAttentionProbePlan(
            operation: operation,
            contextTokens: 64,
            queryTokens: queryTokens,
            prefillChunkTokens: 16,
            outputTokens: 8,
            stopTokenIDs: [],
            batchSize: 1,
            queryHeadCount: 8,
            kvHeadCount: 2,
            headDimension: 128,
            dtype: .float16,
            mask: mask,
            layout: layout,
            warmupRuns: 1,
            measuredRuns: 1,
            seed: 7,
            workloadNonce: "fused-kv-mlx-fixture",
            harnessGitSHA: cleanSHA,
            promotionEvidence: false)
    }
}
