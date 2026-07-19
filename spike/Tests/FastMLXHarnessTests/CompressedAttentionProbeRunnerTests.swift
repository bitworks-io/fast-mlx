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
        XCTAssertGreaterThan(result.payloadBytes, 0)
        XCTAssertGreaterThan(result.scaleBytes, 0)
        XCTAssertGreaterThan(result.biasBytes, 0)
        XCTAssertEqual(result.fp16ResidentBytes, 0)
        XCTAssertEqual(result.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(result.attentionSeconds, 0)
        XCTAssertEqual(result.outputTop1Index, result.oracleTop1Index)
        assertIsolatedAttentionMemory(result)
        assertSHA256(result.sourceKVTensorSHA256)
        assertSHA256(result.packedKVTensorSHA256)
        assertSHA256(result.queryTensorSHA256)
        assertSHA256(result.outputTensorSHA256)
    }

    func testSplitAffineQuantizedMMConsumesIndependentK4V2State()
        async throws
    {
        let result = try await CompressedAttentionProbeRunner().runFixture(
            plan: makePlan(
                operation: .splitAffineQuantizedMM,
                queryTokens: 16,
                mask: .causal,
                layout: .affine(
                    keyBits: 4, valueBits: 2,
                    keyGroupSize: 64, valueGroupSize: 32)))

        XCTAssertEqual(result.outputShape, [1, 8, 16, 128])
        XCTAssertTrue(result.valuesFinite)
        XCTAssertTrue(
            result.structuralEquivalent,
            "max=\(result.maxAbsoluteError) ratio=\(result.maximumToleranceRatio)")
        XCTAssertTrue(result.top1Matches)
        XCTAssertGreaterThan(result.payloadBytes, 0)
        XCTAssertGreaterThan(result.scaleBytes, 0)
        XCTAssertGreaterThan(result.biasBytes, 0)
        XCTAssertEqual(result.fp16ResidentBytes, 0)
        XCTAssertEqual(result.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(result.attentionSeconds, 0)
        XCTAssertEqual(result.outputTop1Index, result.oracleTop1Index)
        assertIsolatedAttentionMemory(result)
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
        XCTAssertGreaterThan(result.attentionSeconds, 0)
        assertIsolatedAttentionMemory(result)
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

    func testFP16FixtureReportsNativeStorageAndIndependentControl() async throws {
        let result = try await CompressedAttentionProbeRunner().runFixture(
            plan: makePlan(operation: .fp16SDPA, layout: .fp16))

        XCTAssertTrue(
            result.structuralEquivalent,
            "maxAbs=\(result.maxAbsoluteError) maxRel=\(result.maxRelativeError) "
                + "ratio=\(result.maximumToleranceRatio) "
                + "meanAbs=\(result.meanAbsoluteError) "
                + "top1=\(result.outputTop1Index)/\(result.oracleTop1Index)")
        XCTAssertTrue(result.top1Matches)
        XCTAssertLessThanOrEqual(
            result.maximumToleranceRatio,
            1,
            "maxAbs=\(result.maxAbsoluteError) maxRel=\(result.maxRelativeError)")
        XCTAssertEqual(result.payloadBytes, 0)
        XCTAssertEqual(result.scaleBytes, 0)
        XCTAssertEqual(result.biasBytes, 0)
        XCTAssertEqual(result.fp16ResidentBytes, 65_536)
        XCTAssertEqual(result.persistentBytes, result.fp16ResidentBytes)
        XCTAssertEqual(result.materializationWorkspaceBytes, 0)
        XCTAssertGreaterThan(result.attentionSeconds, 0)
        XCTAssertEqual(
            result.sourceKVTensorSHA256,
            result.packedKVTensorSHA256)
        assertIsolatedAttentionMemory(result)
    }

    func testCandidateAndReferenceBindTheSameSourceAndQueryProjections() async throws {
        let runner = CompressedAttentionProbeRunner()
        let candidate = try await runner.runFixture(
            plan: makePlan(
                operation: .swiftLMQuantizedAttention,
                layout: .affine(
                    keyBits: 4,
                    valueBits: 4,
                    keyGroupSize: 64,
                    valueGroupSize: 64)))
        let reference = try await runner.runFixture(
            plan: makePlan(operation: .fp16SDPA, layout: .fp16))

        XCTAssertEqual(
            candidate.sourceKVTensorSHA256,
            reference.sourceKVTensorSHA256)
        XCTAssertEqual(
            candidate.queryTensorSHA256,
            reference.queryTensorSHA256)
        XCTAssertNotEqual(
            candidate.packedKVTensorSHA256,
            reference.packedKVTensorSHA256)
    }

    func testTensorHashesCoverMiddleRowsAndNativePackedBits() async throws {
        let coverage = await CompressedAttentionProbeRunner()
            .runTensorHashCoverageFixture()

        XCTAssertNotEqual(coverage.baseTensorSHA256, coverage.middleMutationSHA256)
        XCTAssertNotEqual(coverage.packedWordASHA256, coverage.packedWordBSHA256)
        assertSHA256(coverage.baseTensorSHA256)
        assertSHA256(coverage.middleMutationSHA256)
        assertSHA256(coverage.packedWordASHA256)
        assertSHA256(coverage.packedWordBSHA256)
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
            qualificationEvidence: false)
    }

    private func assertSHA256(
        _ value: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(value.count, 64, file: file, line: line)
        XCTAssertTrue(
            value.allSatisfy { $0.isNumber || ("a" ... "f").contains($0) },
            file: file,
            line: line)
    }

    private func assertIsolatedAttentionMemory(
        _ result: CompressedAttentionProbeNumericResult,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            result.attentionMemoryBefore.peakBytes, 0,
            "peak must reset after fixture preparation",
            file: file, line: line)
        XCTAssertGreaterThan(
            result.attentionMemoryBefore.activeBytes, 0,
            file: file, line: line)
        XCTAssertGreaterThanOrEqual(
            result.attentionMemoryAfter.peakBytes,
            result.attentionMemoryBefore.activeBytes,
            file: file, line: line)
    }
}
