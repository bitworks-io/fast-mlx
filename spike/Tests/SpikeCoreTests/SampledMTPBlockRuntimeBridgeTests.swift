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

final class SeededSampledMTPBlockRuntimeProviderTests: XCTestCase {
    func testSeededSampledMTPKnownVectorAndReproducibleTrace() throws {
        let first = SeededSampledMTPBlockRuntimeProvider(seed: 0x0123_4567_89ab_cdef)
        let second = SeededSampledMTPBlockRuntimeProvider(seed: 0x0123_4567_89ab_cdef)

        XCTAssertTrue(first.supports(parameters: GenerateParameters(temperature: 1, seed: 99)))
        XCTAssertFalse(first.supports(parameters: GenerateParameters(temperature: 0)))
        XCTAssertFalse(first.supports(parameters: GenerateParameters(temperature: 0.7)))
        XCTAssertFalse(first.supports(parameters: GenerateParameters(temperature: 1, topP: 0.95)))
        XCTAssertFalse(first.supports(parameters: GenerateParameters(temperature: 1, topK: 4)))
        XCTAssertFalse(first.supports(parameters: GenerateParameters(temperature: 1, minP: 0.05)))
        XCTAssertFalse(first.supports(parameters: GenerateParameters(
            temperature: 1,
            repetitionPenalty: 1.1)))
        XCTAssertFalse(first.supports(parameters: GenerateParameters(
            temperature: 1,
            presencePenalty: 0.1)))
        XCTAssertFalse(first.supports(parameters: GenerateParameters(
            temperature: 1,
            frequencyPenalty: 0.1)))

        let proposalDistributions = [[0.2, 0.3, 0.5], [0.4, 0.35, 0.25]]
        let targetDistributions = proposalDistributions

        let firstProposals = sampleProposals(first, distributions: proposalDistributions)
        let firstDecision = try first.decide(
            proposedTokens: firstProposals,
            targetLogits: targetDistributions.map(logits),
            bonusTargetLogits: logits([0.1, 0.2, 0.7]))

        let secondProposals = sampleProposals(second, distributions: proposalDistributions)
        let secondDecision = try second.decide(
            proposedTokens: secondProposals,
            targetLogits: targetDistributions.map(logits),
            bonusTargetLogits: logits([0.1, 0.2, 0.7]))

        XCTAssertEqual(firstProposals, secondProposals)
        XCTAssertEqual(firstDecision, secondDecision)
        XCTAssertEqual(first.drawTraces, second.drawTraces)
        XCTAssertEqual(first.drawTraces.count, 1)
        XCTAssertEqual(first.drawTraces[0].proposalUniforms, [
            0.15198413840918862,
            0.9481351609826048,
        ])
        XCTAssertEqual(first.drawTraces[0].acceptanceUniforms, [
            0.6430685803365146,
            0.2412293034472343,
        ])
        XCTAssertEqual(first.drawTraces[0].terminalDraw, .bonus(0.5096159962247693))
        XCTAssertEqual(first.drawTraces[0].acceptedDraftCount, 2)
        XCTAssertEqual(firstDecision.outputTokens, firstProposals + [2])

        let allUniforms = first.drawTraces[0].proposalUniforms
            + first.drawTraces[0].acceptanceUniforms
            + [first.drawTraces[0].terminalUniform]
        XCTAssertTrue(allUniforms.allSatisfy { $0 > 0 && $0 < 1 })
    }

    func testSeededSampledMTPDifferentSeedChangesTrace() throws {
        let first = SeededSampledMTPBlockRuntimeProvider(seed: 1)
        let second = SeededSampledMTPBlockRuntimeProvider(seed: 2)
        let proposalDistributions = [[0.2, 0.3, 0.5], [0.4, 0.35, 0.25]]

        let firstProposals = sampleProposals(first, distributions: proposalDistributions)
        _ = try first.decide(
            proposedTokens: firstProposals,
            targetLogits: proposalDistributions.map(logits),
            bonusTargetLogits: logits([0.1, 0.2, 0.7]))

        let secondProposals = sampleProposals(second, distributions: proposalDistributions)
        _ = try second.decide(
            proposedTokens: secondProposals,
            targetLogits: proposalDistributions.map(logits),
            bonusTargetLogits: logits([0.1, 0.2, 0.7]))

        XCTAssertNotEqual(first.drawTraces, second.drawTraces)
    }

    func testSeededSampledMTPUsesResidualTerminalAtFirstRejection() throws {
        let provider = SeededSampledMTPBlockRuntimeProvider(seed: 3)
        let proposalDistributions = [[0.999, 0.001], [0.5, 0.5]]
        let proposals = sampleProposals(provider, distributions: proposalDistributions)

        let decision = try provider.decide(
            proposedTokens: proposals,
            targetLogits: [logits([0.001, 0.999]), logits([0.5, 0.5])],
            bonusTargetLogits: logits([0.5, 0.5]))

        XCTAssertEqual(decision.acceptedDraftCount, 0)
        XCTAssertEqual(decision.outputTokens.count, 1)
        XCTAssertEqual(provider.drawTraces[0].acceptanceUniforms.count, 1)
        XCTAssertEqual(provider.drawTraces[0].terminalDraw.purpose, .residual)
    }

    func testSeededSampledMTPValidatesProposalCaptureAndMalformedData() throws {
        let mismatch = SeededSampledMTPBlockRuntimeProvider(seed: 4)
        let proposals = sampleProposals(mismatch, distributions: [[0.6, 0.4]])

        XCTAssertThrowsError(try mismatch.decide(
            proposedTokens: proposals + [0],
            targetLogits: [logits([0.6, 0.4])],
            bonusTargetLogits: logits([0.6, 0.4]))) {
                XCTAssertEqual(
                    $0 as? SeededSampledMTPBlockRuntimeProviderError,
                    .proposalCountMismatch(expected: 1, actual: 2))
            }

        let invalidToken = SeededSampledMTPBlockRuntimeProvider(seed: 4)
        _ = sampleProposals(invalidToken, distributions: [[0.6, 0.4]])
        XCTAssertThrowsError(try invalidToken.decide(
            proposedTokens: [2],
            targetLogits: [logits([0.6, 0.4])],
            bonusTargetLogits: logits([0.6, 0.4]))) {
                XCTAssertEqual(
                    $0 as? SeededSampledMTPBlockRuntimeProviderError,
                    .invalidProposalToken(token: 2, vocabularyCount: 2))
            }

        let widthMismatch = SeededSampledMTPBlockRuntimeProvider(seed: 4)
        let widthMismatchProposals = sampleProposals(widthMismatch, distributions: [[0.6, 0.4]])
        XCTAssertThrowsError(try widthMismatch.decide(
            proposedTokens: widthMismatchProposals,
            targetLogits: [logits([0.5, 0.25, 0.25])],
            bonusTargetLogits: logits([0.6, 0.4])))

        let invalidLogits = SeededSampledMTPBlockRuntimeProvider(seed: 4)
        _ = invalidLogits.proposalSampler.sample(logits: MLXArray([Float.nan, 0]))
        XCTAssertThrowsError(try invalidLogits.decide(
            proposedTokens: [0],
            targetLogits: [logits([0.6, 0.4])],
            bonusTargetLogits: logits([0.6, 0.4]))) {
                XCTAssertEqual(
                    $0 as? SeededSampledMTPBlockRuntimeProviderError,
                    .proposalCaptureFailed)
            }
    }

    func testSeededSampledMTPValidationFailureDoesNotConsumeCaptureOrAdvanceDraws() throws {
        let seed: UInt64 = 5
        let proposalDistributions = [[0.4, 0.6], [0.7, 0.3]]
        let targetDistributions = proposalDistributions
        let bonusDistribution = [0.2, 0.8]
        let provider = SeededSampledMTPBlockRuntimeProvider(seed: seed)
        let proposals = sampleProposals(provider, distributions: proposalDistributions)

        XCTAssertThrowsError(try provider.decide(
            proposedTokens: [2, proposals[1]],
            targetLogits: targetDistributions.map(logits),
            bonusTargetLogits: logits(bonusDistribution))) {
                XCTAssertEqual(
                    $0 as? SeededSampledMTPBlockRuntimeProviderError,
                    .invalidProposalToken(token: 2, vocabularyCount: 2))
            }
        XCTAssertTrue(provider.drawTraces.isEmpty)

        let decision = try provider.decide(
            proposedTokens: proposals,
            targetLogits: targetDistributions.map(logits),
            bonusTargetLogits: logits(bonusDistribution))

        let fresh = SeededSampledMTPBlockRuntimeProvider(seed: seed)
        let freshProposals = sampleProposals(fresh, distributions: proposalDistributions)
        let freshDecision = try fresh.decide(
            proposedTokens: freshProposals,
            targetLogits: targetDistributions.map(logits),
            bonusTargetLogits: logits(bonusDistribution))

        XCTAssertEqual(proposals, freshProposals)
        XCTAssertEqual(decision, freshDecision)
        XCTAssertEqual(provider.drawTraces, fresh.drawTraces)
    }

    func testSeededSampledMTPTwoSuccessiveBlocksReproduceDomainCounters() throws {
        let first = SeededSampledMTPBlockRuntimeProvider(seed: 6)
        let second = SeededSampledMTPBlockRuntimeProvider(seed: 6)

        let firstRun = try runTwoSeededBlocks(first)
        let secondRun = try runTwoSeededBlocks(second)

        XCTAssertEqual(firstRun.proposals, secondRun.proposals)
        XCTAssertEqual(firstRun.decisions, secondRun.decisions)
        XCTAssertEqual(first.drawTraces, second.drawTraces)
        XCTAssertEqual(first.drawTraces.map(\.blockIndex), [0, 1])
        XCTAssertEqual(first.drawTraces.map(\.terminalDraw.purpose), [.bonus, .bonus])
        XCTAssertTrue(first.drawTraces.flatMap(\.proposalUniforms).allSatisfy { $0 > 0 && $0 < 1 })
        XCTAssertTrue(first.drawTraces.flatMap(\.acceptanceUniforms).allSatisfy { $0 > 0 && $0 < 1 })
        XCTAssertTrue(first.drawTraces.map(\.terminalUniform).allSatisfy { $0 > 0 && $0 < 1 })
    }

    private func sampleProposals(
        _ provider: any SampledMTPBlockRuntimeDeciding,
        distributions: [[Double]]
    ) -> [Int] {
        distributions.map { distribution in
            provider.proposalSampler.sample(logits: logits(distribution)).item(Int.self)
        }
    }

    private func logits(_ distribution: [Double]) -> MLXArray {
        MLXArray(distribution.map { Float(log($0)) })
    }

    private func runTwoSeededBlocks(
        _ provider: SeededSampledMTPBlockRuntimeProvider
    ) throws -> (proposals: [[Int]], decisions: [SampledMTPBlockRuntimeDecision]) {
        let firstProposalDistributions = [[0.2, 0.3, 0.5], [0.4, 0.35, 0.25]]
        let firstProposals = sampleProposals(provider, distributions: firstProposalDistributions)
        let firstDecision = try provider.decide(
            proposedTokens: firstProposals,
            targetLogits: firstProposalDistributions.map(logits),
            bonusTargetLogits: logits([0.1, 0.2, 0.7]))

        let secondProposalDistributions = [[0.55, 0.45], [0.25, 0.75]]
        let secondProposals = sampleProposals(provider, distributions: secondProposalDistributions)
        let secondDecision = try provider.decide(
            proposedTokens: secondProposals,
            targetLogits: secondProposalDistributions.map(logits),
            bonusTargetLogits: logits([0.65, 0.35]))

        return (
            proposals: [firstProposals, secondProposals],
            decisions: [firstDecision, secondDecision])
    }
}

final class SampledMTPBlockRuntimeBridgeTestsNondeterministicProvider: XCTestCase {
    func testUsesLabeledEntropyDomainsAndFiniteTraceUniforms() throws {
        let entropy = RecordingRuntimeEntropy(draws: [
            .proposal: [0.1, 0.8],
            .acceptance: [0, 0],
            .bonus: [0.5],
        ])
        let provider = NondeterministicSampledMTPBlockRuntimeProvider(entropy: entropy.next)
        let proposalDistributions = [[0.2, 0.3, 0.5], [0.4, 0.35, 0.25]]
        let proposals = sampleProposals(provider, distributions: proposalDistributions)

        let decision = try provider.decide(
            proposedTokens: proposals,
            targetLogits: proposalDistributions.map(logits),
            bonusTargetLogits: logits([0.1, 0.2, 0.7]))

        XCTAssertEqual(entropy.domains, [.proposal, .proposal, .acceptance, .acceptance, .bonus])
        XCTAssertEqual(decision.acceptedDraftCount, 2)
        XCTAssertEqual(Array(decision.outputTokens.prefix(2)), proposals)
        XCTAssertEqual(provider.drawTraces.count, 1)
        XCTAssertEqual(provider.drawTraces[0].terminalDraw, .bonus(0.5))
        let allUniforms = provider.drawTraces[0].proposalUniforms
            + provider.drawTraces[0].acceptanceUniforms
            + [provider.drawTraces[0].terminalUniform]
        XCTAssertTrue(allUniforms.allSatisfy { $0.isFinite && (0 ..< 1).contains($0) })
    }

    func testDefaultEntropyDoesNotReplaySeededSequenceForFreshProviders() throws {
        let first = NondeterministicSampledMTPBlockRuntimeProvider()
        let second = NondeterministicSampledMTPBlockRuntimeProvider()
        let proposalDistributions = Array(
            repeating: [0.2, 0.3, 0.5],
            count: 6)

        let firstProposals = sampleProposals(first, distributions: proposalDistributions)
        _ = try first.decide(
            proposedTokens: firstProposals,
            targetLogits: proposalDistributions.map(logits),
            bonusTargetLogits: logits([0.1, 0.2, 0.7]))

        let secondProposals = sampleProposals(second, distributions: proposalDistributions)
        _ = try second.decide(
            proposedTokens: secondProposals,
            targetLogits: proposalDistributions.map(logits),
            bonusTargetLogits: logits([0.1, 0.2, 0.7]))

        XCTAssertNotEqual(
            first.drawTraces.flatMap(\.proposalUniforms),
            second.drawTraces.flatMap(\.proposalUniforms))
    }

    func testFirstRejectionUsesResidualEntropyDomain() throws {
        let entropy = RecordingRuntimeEntropy(draws: [
            .proposal: [0.1],
            .acceptance: [0.9],
            .residual: [0.4],
        ])
        let provider = NondeterministicSampledMTPBlockRuntimeProvider(entropy: entropy.next)
        let proposals = sampleProposals(provider, distributions: [[0.95, 0.05]])

        let decision = try provider.decide(
            proposedTokens: proposals,
            targetLogits: [logits([0.05, 0.95])],
            bonusTargetLogits: logits([0.6, 0.4]))

        XCTAssertEqual(entropy.domains, [.proposal, .acceptance, .residual])
        XCTAssertEqual(decision.acceptedDraftCount, 0)
        XCTAssertEqual(provider.drawTraces[0].terminalDraw, .residual(0.4))
    }

    func testValidationFailureDoesNotConsumeCaptureOrAdvanceTrace() throws {
        let entropy = RecordingRuntimeEntropy(draws: [
            .proposal: [0.25, 0.25],
            .acceptance: [0, 0],
            .bonus: [0.5],
        ])
        let provider = NondeterministicSampledMTPBlockRuntimeProvider(entropy: entropy.next)
        let proposalDistributions = [[0.6, 0.4], [0.7, 0.3]]
        let proposals = sampleProposals(provider, distributions: proposalDistributions)

        XCTAssertThrowsError(try provider.decide(
            proposedTokens: [2, proposals[1]],
            targetLogits: proposalDistributions.map(logits),
            bonusTargetLogits: logits([0.6, 0.4])))
        XCTAssertTrue(provider.drawTraces.isEmpty)

        let decision = try provider.decide(
            proposedTokens: proposals,
            targetLogits: proposalDistributions.map(logits),
            bonusTargetLogits: logits([0.6, 0.4]))

        let freshEntropy = RecordingRuntimeEntropy(draws: [
            .proposal: [0.25, 0.25],
            .acceptance: [0, 0],
            .bonus: [0.5],
        ])
        let fresh = NondeterministicSampledMTPBlockRuntimeProvider(entropy: freshEntropy.next)
        let freshProposals = sampleProposals(fresh, distributions: proposalDistributions)
        let freshDecision = try fresh.decide(
            proposedTokens: freshProposals,
            targetLogits: proposalDistributions.map(logits),
            bonusTargetLogits: logits([0.6, 0.4]))

        XCTAssertEqual(proposals, freshProposals)
        XCTAssertEqual(decision, freshDecision)
        XCTAssertEqual(provider.drawTraces, fresh.drawTraces)
    }

    func testUnsupportedSamplingParametersFailClosed() {
        let provider = NondeterministicSampledMTPBlockRuntimeProvider()

        XCTAssertTrue(provider.supports(parameters: GenerateParameters(temperature: 1, seed: 7)))
        XCTAssertFalse(provider.supports(parameters: GenerateParameters(temperature: 0)))
        XCTAssertFalse(provider.supports(parameters: GenerateParameters(temperature: 0.7)))
        XCTAssertFalse(provider.supports(parameters: GenerateParameters(temperature: 1, topP: 0.95)))
        XCTAssertFalse(provider.supports(parameters: GenerateParameters(temperature: 1, topK: 4)))
        XCTAssertFalse(provider.supports(parameters: GenerateParameters(temperature: 1, minP: 0.05)))
        XCTAssertFalse(provider.supports(parameters: GenerateParameters(
            temperature: 1,
            repetitionPenalty: 1.1)))
        XCTAssertFalse(provider.supports(parameters: GenerateParameters(
            temperature: 1,
            presencePenalty: 0.1)))
        XCTAssertFalse(provider.supports(parameters: GenerateParameters(
            temperature: 1,
            frequencyPenalty: 0.1)))
    }

    private func sampleProposals(
        _ provider: any SampledMTPBlockRuntimeDeciding,
        distributions: [[Double]]
    ) -> [Int] {
        distributions.map { distribution in
            provider.proposalSampler.sample(logits: logits(distribution)).item(Int.self)
        }
    }

    private func logits(_ distribution: [Double]) -> MLXArray {
        MLXArray(distribution.map { Float(log($0)) })
    }
}

private final class RecordingRuntimeEntropy: @unchecked Sendable {
    private var draws: [SampledMTPBlockRuntimeEntropyDomain: [Double]]
    private(set) var domains = [SampledMTPBlockRuntimeEntropyDomain]()

    init(draws: [SampledMTPBlockRuntimeEntropyDomain: [Double]]) {
        self.draws = draws
    }

    func next(_ domain: SampledMTPBlockRuntimeEntropyDomain) -> Double {
        domains.append(domain)
        guard var values = draws[domain], !values.isEmpty else { return .nan }
        let value = values.removeFirst()
        draws[domain] = values
        return value
    }
}
