import HarnessCore
import MLX
import MLXLMCommon
@testable import SpikeCore
import XCTest

final class SampledMTPBlockRuntimeBridgeTests: XCTestCase {
    func testRejectFirstUsesResidualCorrectionContract() throws {
        let plan = SampledMTPBlockRuntimeDrawPlan(
            proposalUniforms: [0.25, 0.25],
            acceptanceUniforms: [0.9],
            terminalDraw: .residual(0.5))
        let bridge = SampledMTPBlockRuntimeBridge(plans: [plan])
        let proposals = sampleProposals(
            bridge,
            distributions: [[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])

        let decision = try bridge.decide(
            proposedTokens: proposals,
            targetLogits: [logits([0.2, 0.5, 0.3]), logits([0.2, 0.5, 0.3])],
            bonusTargetLogits: logits([0.2, 0.3, 0.5]))

        XCTAssertEqual(decision.acceptedDraftCount, 0)
        XCTAssertEqual(decision.outputTokens.count, 1)
        XCTAssertNotEqual(decision.outputTokens[0], proposals[0])
    }

    func testRejectSecondPreservesAcceptedPrefix() throws {
        let plan = SampledMTPBlockRuntimeDrawPlan(
            proposalUniforms: [0.25, 0.25],
            acceptanceUniforms: [0, 0.9],
            terminalDraw: .residual(0.5))
        let bridge = SampledMTPBlockRuntimeBridge(plans: [plan])
        let proposals = sampleProposals(
            bridge,
            distributions: [[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])

        let decision = try bridge.decide(
            proposedTokens: proposals,
            targetLogits: [logits([0.7, 0.2, 0.1]), logits([0.2, 0.5, 0.3])],
            bonusTargetLogits: logits([0.2, 0.3, 0.5]))

        XCTAssertEqual(decision.acceptedDraftCount, 1)
        XCTAssertEqual(decision.outputTokens.count, 2)
        XCTAssertEqual(decision.outputTokens[0], proposals[0])
        XCTAssertNotEqual(decision.outputTokens[1], proposals[1])
    }

    func testAcceptAllUsesBonusDistribution() throws {
        let plan = SampledMTPBlockRuntimeDrawPlan(
            proposalUniforms: [0.25, 0.25],
            acceptanceUniforms: [0, 0],
            terminalDraw: .bonus(0.75))
        let bridge = SampledMTPBlockRuntimeBridge(plans: [plan])
        let proposalDistributions = [[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]]
        let proposals = sampleProposals(bridge, distributions: proposalDistributions)

        let decision = try bridge.decide(
            proposedTokens: proposals,
            targetLogits: proposalDistributions.map(logits),
            bonusTargetLogits: logits([0.2, 0.3, 0.5]))

        XCTAssertEqual(decision.acceptedDraftCount, 2)
        XCTAssertEqual(Array(decision.outputTokens.prefix(2)), proposals)
        XCTAssertEqual(decision.outputTokens.count, 3)
    }

    func testUnsupportedSamplingParametersFailClosedBeforeProposal() {
        let bridge = SampledMTPBlockRuntimeBridge(plans: [
            SampledMTPBlockRuntimeDrawPlan(
                proposalUniforms: [0.25],
                acceptanceUniforms: [0],
                terminalDraw: .bonus(0.5))
        ])
        XCTAssertFalse(bridge.supports(parameters: GenerateParameters(temperature: 0)))
        XCTAssertFalse(bridge.supports(parameters: GenerateParameters(temperature: 0.7)))
        XCTAssertFalse(bridge.supports(parameters: GenerateParameters(temperature: 1, topP: 0.9)))
        XCTAssertTrue(bridge.supports(parameters: GenerateParameters(temperature: 1)))
    }

    func testMissingPlanAndInvalidProposalDrawFailClosed() {
        let missingPlan = SampledMTPBlockRuntimeBridge(plans: [])
        XCTAssertFalse(missingPlan.supports(parameters: GenerateParameters(temperature: 1)))

        let invalidDraw = SampledMTPBlockRuntimeBridge(plans: [
            SampledMTPBlockRuntimeDrawPlan(
                proposalUniforms: [1],
                acceptanceUniforms: [0],
                terminalDraw: .bonus(0.5))
        ])
        let proposed = sampleProposals(invalidDraw, distributions: [[0.6, 0.4]])
        XCTAssertThrowsError(try invalidDraw.decide(
            proposedTokens: proposed,
            targetLogits: [logits([0.6, 0.4])],
            bonusTargetLogits: logits([0.6, 0.4]))) {
                XCTAssertEqual(
                    $0 as? SampledMTPBlockRuntimeBridgeError,
                    .proposalCaptureFailed)
            }
    }

    private func sampleProposals(
        _ bridge: SampledMTPBlockRuntimeBridge,
        distributions: [[Double]]
    ) -> [Int] {
        distributions.map { distribution in
            bridge.proposalSampler.sample(logits: logits(distribution)).item(Int.self)
        }
    }

    private func logits(_ distribution: [Double]) -> MLXArray {
        MLXArray(distribution.map { Float(log($0)) })
    }
}
