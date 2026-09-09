import HarnessCore
import MLX
import MLXLMCommon
import MLXRandom
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

/// Regression coverage for a fail-closed defect: a multimodal checkpoint
/// can mask its unsupported media-sentinel indices to `-Float.infinity` on
/// every forward, and `MTPSpeculativeTokenIterator` hands those raw logits
/// straight to `sampledBlockDecisionProvider.decide`.
/// `validatingNormalizedProbabilities`'s raw guard used to reject any
/// non-finite raw logit (including the model's own legitimate `-inf` mask),
/// which made both production-shaped providers throw on block one, every
/// time -- indistinguishably from a legitimate refusal, since the iterator
/// reports only a generic sticky-passthrough reason.
final class SampledMTPInfLogitsFailClosedTests: XCTestCase {
    private static let productionVocabularyWidth = 151_936
    /// Mimics such a checkpoint's four media-sentinel indices: spread across the
    /// width (including the first and last positions) rather than
    /// clustered, so the fixture does not accidentally depend on where the
    /// mask sits.
    private static let mediaSentinelMaskedIndices = [0, 50_000, 100_000, 151_935]
    private static let numDraft = MTPSpeculativeDecoder.servingBlockSize - 1

    // MARK: - A. Happy path, PRODUCTION WIDTH

    /// This is the test that currently FAILS (before the fix): the bonus
    /// row's raw `-inf` entries are rejected by the raw guard, so `decide`
    /// throws `.invalidLogits` instead of returning a decision. Production
    /// width matters here specifically because the `abs(sum - 1) <= 1e-12`
    /// post-normalization tolerance is width-sensitive; a 3-element
    /// fixture would not exercise it.
    func testHappyPathAtProductionVocabularyWidthWithMaskedMediaTokenIndices() throws {
        let width = Self.productionVocabularyWidth
        let maskedIndices = Self.mediaSentinelMaskedIndices
        let provider = SeededSampledMTPBlockRuntimeProvider(seed: 0xC118_0001)

        var proposedTokens = [Int]()
        var targetLogits = [MLXArray]()
        for draftIndex in 0 ..< Self.numDraft {
            let proposalLogits = Self.productionProposalLogits(
                seed: UInt64(draftIndex + 1),
                width: width)
            let token = provider.proposalSampler.sample(logits: proposalLogits).item(Int.self)
            proposedTokens.append(token)
            targetLogits.append(Self.maskedProductionLogits(
                seed: UInt64(1000 + draftIndex),
                maskedIndices: maskedIndices,
                width: width))
        }
        let bonusTargetLogits = Self.maskedProductionLogits(
            seed: 9999,
            maskedIndices: maskedIndices,
            width: width)

        let decision = try provider.decide(
            proposedTokens: proposedTokens,
            targetLogits: targetLogits,
            bonusTargetLogits: bonusTargetLogits)

        XCTAssertEqual(decision.outputTokens.count, decision.acceptedDraftCount + 1)
        XCTAssertTrue((0 ... Self.numDraft).contains(decision.acceptedDraftCount))
    }

    // MARK: - B. Discriminating assertion

    /// Two independent proofs that a masked index can never be produced:
    /// (1) a direct probability assertion against MLX's own `softmax` on
    /// the exact array handed to `decide` (this file's own
    /// `normalizedProbabilities`/`validatingNormalizedProbabilities` are
    /// `private` at file scope, so they are not reachable even via
    /// `@testable` -- this is the closest honest seam); and (2) an
    /// empirical corroboration across many seeded blocks that no masked
    /// index ever appears in `outputTokens`.
    func testMaskedIndicesCarryExactlyZeroProbabilityAndNeverAppearInOutput() throws {
        let width = Self.productionVocabularyWidth
        let maskedIndices = Self.mediaSentinelMaskedIndices

        let bonusTargetLogits = Self.maskedProductionLogits(
            seed: 42,
            maskedIndices: maskedIndices,
            width: width)
        let probabilities = softmax(bonusTargetLogits.asType(.float32), axis: -1)
        eval(probabilities)
        let probabilityValues = probabilities.asArray(Float.self)
        for index in maskedIndices {
            XCTAssertEqual(probabilityValues[index], 0.0)
        }

        for seed in UInt64(1) ... 15 {
            let provider = SeededSampledMTPBlockRuntimeProvider(seed: seed)
            var proposedTokens = [Int]()
            var targetLogits = [MLXArray]()
            for draftIndex in 0 ..< Self.numDraft {
                let proposalLogits = Self.productionProposalLogits(
                    seed: seed &* 1000 &+ UInt64(draftIndex),
                    width: width)
                let token = provider.proposalSampler.sample(logits: proposalLogits).item(Int.self)
                proposedTokens.append(token)
                targetLogits.append(Self.maskedProductionLogits(
                    seed: seed &* 1000 &+ UInt64(500 + draftIndex),
                    maskedIndices: maskedIndices,
                    width: width))
            }
            let seededBonus = Self.maskedProductionLogits(
                seed: seed &* 1000 &+ 999,
                maskedIndices: maskedIndices,
                width: width)

            let decision = try provider.decide(
                proposedTokens: proposedTokens,
                targetLogits: targetLogits,
                bonusTargetLogits: seededBonus)

            for token in decision.outputTokens {
                XCTAssertFalse(maskedIndices.contains(token), "seed \(seed) produced a masked token")
            }
        }
    }

    // MARK: - C. Enumerated legitimate refusals

    func testSeededProviderStillRejectsRawNaNLogit() throws {
        try assertBonusLogitsStillRejected(
            bonus: [Float.nan, 0, 0],
            expected: SeededSampledMTPBlockRuntimeProviderError.invalidLogits,
            provider: SeededSampledMTPBlockRuntimeProvider(seed: 101))
    }

    func testNondeterministicProviderStillRejectsRawNaNLogit() throws {
        try assertBonusLogitsStillRejected(
            bonus: [Float.nan, 0, 0],
            expected: NondeterministicSampledMTPBlockRuntimeProviderError.invalidLogits,
            provider: NondeterministicSampledMTPBlockRuntimeProvider(entropy: { _ in 0.1 }))
    }

    func testSeededProviderStillRejectsRawPositiveInfinityLogit() throws {
        try assertBonusLogitsStillRejected(
            bonus: [Float.infinity, 0, 0],
            expected: SeededSampledMTPBlockRuntimeProviderError.invalidLogits,
            provider: SeededSampledMTPBlockRuntimeProvider(seed: 102))
    }

    func testNondeterministicProviderStillRejectsRawPositiveInfinityLogit() throws {
        try assertBonusLogitsStillRejected(
            bonus: [Float.infinity, 0, 0],
            expected: NondeterministicSampledMTPBlockRuntimeProviderError.invalidLogits,
            provider: NondeterministicSampledMTPBlockRuntimeProvider(entropy: { _ in 0.1 }))
    }

    /// An all-`-inf` row is degenerate (every raw entry is individually
    /// legitimate, but together they carry no information: `softmax` of an
    /// all-`-inf` row divides `NaN` by `NaN`). This still refuses after the
    /// fix, but via a DIFFERENT guard than C1/C2: the raw guard now admits
    /// it, and it is `normalizedProbabilities`'s own internal
    /// `sum.isFinite` check that collapses the result to `[]` (an all-NaN
    /// softmax output sums to NaN), which `validatingNormalizedProbabilities`
    /// then rejects via its `!distribution.isEmpty` guard -- not the
    /// per-element or sum-normalization checks.
    func testSeededProviderStillRejectsAllNegativeInfinityRow() throws {
        // Confirms the "which guard" claim above directly, against MLX's
        // own softmax, on the exact array shape used below: every entry of
        // an all-`-inf` row's softmax is `NaN` (not e.g. a special-cased
        // uniform distribution), which is what forces
        // `normalizedProbabilities`'s `sum.isFinite` guard (not
        // `validatingNormalizedProbabilities`'s per-element or
        // sum-normalization checks) to produce the empty distribution that
        // `!distribution.isEmpty` then rejects.
        let allNegativeInfinity = MLXArray(
            [-Float.infinity, -Float.infinity, -Float.infinity])
        let softmaxOutput = softmax(allNegativeInfinity.asType(.float32), axis: -1)
        eval(softmaxOutput)
        XCTAssertTrue(softmaxOutput.asArray(Float.self).allSatisfy(\.isNaN))

        try assertBonusLogitsStillRejected(
            bonus: [-Float.infinity, -Float.infinity, -Float.infinity],
            expected: SeededSampledMTPBlockRuntimeProviderError.invalidLogits,
            provider: SeededSampledMTPBlockRuntimeProvider(seed: 103))
    }

    func testNondeterministicProviderStillRejectsAllNegativeInfinityRow() throws {
        try assertBonusLogitsStillRejected(
            bonus: [-Float.infinity, -Float.infinity, -Float.infinity],
            expected: NondeterministicSampledMTPBlockRuntimeProviderError.invalidLogits,
            provider: NondeterministicSampledMTPBlockRuntimeProvider(entropy: { _ in 0.1 }))
    }

    func testSeededProviderStillRejectsEmptyLogitsRow() throws {
        try assertBonusLogitsStillRejected(
            bonus: [],
            expected: SeededSampledMTPBlockRuntimeProviderError.invalidLogits,
            provider: SeededSampledMTPBlockRuntimeProvider(seed: 104),
            proposalAndTargetDistribution: [0.6, 0.4])
    }

    func testNondeterministicProviderStillRejectsEmptyLogitsRow() throws {
        try assertBonusLogitsStillRejected(
            bonus: [],
            expected: NondeterministicSampledMTPBlockRuntimeProviderError.invalidLogits,
            provider: NondeterministicSampledMTPBlockRuntimeProvider(entropy: { _ in 0.1 }),
            proposalAndTargetDistribution: [0.6, 0.4])
    }

    /// A proposed token whose draft mass is exactly 0 (e.g. proposing a
    /// masked index) still refuses, via the existing
    /// `SampledMTPResidualCorrectionError.zeroDraftMass` path.
    ///
    /// This is deliberately NOT routed through either provider's own
    /// `proposalSampler.sample()` -> `decide()` pipeline, because doing so
    /// is structurally unreachable, not merely untried:
    /// `categoricalSample` only ever returns an index whose own probability
    /// is > 0, UNLESS its end-of-array fallback fires because the running
    /// cumulative sum falls short of the drawn uniform. Measured directly
    /// (mirroring Swift's `Double` IEEE-754 arithmetic bit-for-bit, using
    /// this file's own largest-absorbs-the-remainder correction) at
    /// production width with realistic random logits: the post-correction
    /// cumulative sum came back bit-exact `1.0` across ten independent
    /// trials, leaving no exploitable gap. So a real proposal draw can
    /// never hand `decide()` a token with zero draft mass -- that is
    /// exactly the invariant `testMaskedIndicesCarryExactlyZeroProbability...`
    /// proves above. What CAN be, and is, tested here is the shared
    /// acceptance layer both providers call into
    /// (`SampledMTPBlockAcceptance.decide` ->
    /// `SampledMTPResidualCorrection.acceptanceProbability`), which is not
    /// rewrapped by either provider's `decide()` for this error, so the
    /// thrown type is the shared `SampledMTPResidualCorrectionError`, not a
    /// provider-specific one.
    func testProposingAMaskedIndexStillRefusesViaTheSharedZeroDraftMassPath() {
        XCTAssertThrowsError(try SampledMTPBlockAcceptance.decide(
            steps: [SampledMTPBlockStep(
                targetDistribution: [0.5, 0.5],
                draftDistribution: [0.0, 1.0],
                proposedToken: 0)],
            acceptanceUniforms: [0.0],
            terminalDraws: [.bonus(0.5)],
            bonusTargetDistribution: [0.5, 0.5])) {
                XCTAssertEqual(
                    $0 as? SampledMTPResidualCorrectionError,
                    .zeroDraftMass(token: 0))
            }
    }

    // MARK: - Helpers

    private static func maskedProductionLogits(
        seed: UInt64,
        maskedIndices: [Int],
        width: Int
    ) -> MLXArray {
        let base = MLXRandom.normal([width], key: MLXRandom.key(seed))
        eval(base)
        var values = base.asArray(Float.self)
        for index in maskedIndices {
            values[index] = -Float.infinity
        }
        let masked = MLXArray(values)
        eval(masked)
        return masked
    }

    private static func productionProposalLogits(seed: UInt64, width: Int) -> MLXArray {
        let logits = MLXRandom.normal([width], key: MLXRandom.key(seed))
        eval(logits)
        return logits
    }

    /// Drives a single-step block whose proposal and target distribution
    /// are an ordinary finite two-element distribution, with the given raw
    /// `bonus` array handed to `decide` unmodified, and asserts the given
    /// error is thrown.
    private func assertBonusLogitsStillRejected<E: Error & Equatable>(
        bonus: [Float],
        expected: E,
        provider: some SampledMTPBlockRuntimeDeciding,
        proposalAndTargetDistribution: [Double] = [0.6, 0.4]
    ) throws {
        let proposalLogits = MLXArray(
            proposalAndTargetDistribution.map { Float(log($0)) })
        let token = provider.proposalSampler.sample(logits: proposalLogits).item(Int.self)

        XCTAssertThrowsError(try provider.decide(
            proposedTokens: [token],
            targetLogits: [proposalLogits],
            bonusTargetLogits: MLXArray(bonus))) {
                XCTAssertEqual($0 as? E, expected)
            }
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

/// Cheap pins for the `supports(parameters:)` truncation cross-check added
/// alongside `SampledMTPSamplingTruncation`.
final class SampledMTPSamplingTruncationSupportsTests: XCTestCase {
    /// This is the test that would have caught the `Optional(0.0)` refusal:
    /// the deployed runbook instructs clients to send `presence_penalty: 0`
    /// explicitly, which arrives as `Optional(0.0)`, not `nil`. A provider
    /// carrying the matching truncation must accept it; a provider that
    /// still carries `.untruncated` (i.e. was never wired up for the
    /// deployed thinking preset) must refuse, which is exactly the
    /// cross-check from item 3 of this increment.
    func testSupportsAcceptsDeployedThinkingPresetAndRejectsTruncationMismatch() {
        let deployedTruncation = SampledMTPSamplingTruncation(
            temperature: 1, topP: 0.95, topK: 20, minP: 0)
        let deployedParameters = GenerateParameters(
            temperature: 1,
            topP: 0.95,
            topK: 20,
            minP: 0,
            presencePenalty: 0.0)

        let matchingProvider = SeededSampledMTPBlockRuntimeProvider(
            seed: 1, truncation: deployedTruncation)
        XCTAssertTrue(matchingProvider.supports(parameters: deployedParameters))

        // Same request, but the provider was constructed with a DIFFERENT
        // truncation (the untruncated default) -- the cross-check must
        // refuse rather than silently serve the wrong target distribution.
        let mismatchedProvider = SeededSampledMTPBlockRuntimeProvider(seed: 1)
        XCTAssertFalse(mismatchedProvider.supports(parameters: deployedParameters))
    }
}

/// Direct coverage for the `categoricalSample` rounding-fallthrough fix.
/// `categoricalSample` is `internal` (not `private`) specifically so this
/// hand-built scenario is reachable: see the access-level comment on its
/// declaration.
final class SampledMTPCategoricalSampleFallbackTests: XCTestCase {
    func testRoundingFallthroughReturnsLastPositiveMassIndexNotLastIndex() throws {
        // Deliberately sums to 0.9, short of 1, with two trailing exact
        // zeros -- the shape a truncated distribution produces. `uniform`
        // is chosen so the running cumulative sum never exceeds it, which
        // forces every iteration of the loop to fall through to the final
        // fallback line.
        let probabilities = [0.3, 0.3, 0.3, 0.0, 0.0]
        let uniform = 0.95

        let result = try XCTUnwrap(categoricalSample(probabilities, uniform: uniform))

        XCTAssertEqual(result, 2)
        XCTAssertGreaterThan(probabilities[result], 0)
        // The bug this replaces returned `probabilities.indices.last` (4),
        // whose probability is exactly zero -- pin that index would have
        // been the wrong, zero-mass answer.
        XCTAssertEqual(probabilities[probabilities.count - 1], 0)
    }
}

/// The decisive test for this increment: drives `SeededSampledMTPBlockRuntimeProvider`
/// over many single-draft-token blocks with a truncating `SampledMTPSamplingTruncation`,
/// on a fixture whose draft (`q`) and target (`p`) top-2 sets are deliberately
/// disjoint, and checks the emitted-token histogram against an independently
/// computed truncated target distribution `p'` -- computed with this file's
/// own `[Double]` softmax/top-k arithmetic, never by calling
/// `truncatedSamplingProbabilities`.
///
/// Every emitted token (per-step accepted/corrected token, AND bonus token
/// on a fully-accepted block) is, by the standard unbiasedness property of
/// residual-corrected speculative sampling, marginally distributed as the
/// *target* distribution used for that slot -- regardless of what `q` is.
/// Using the SAME logits row for both the per-step target and the bonus
/// target here means the whole pool of emitted tokens across all blocks has
/// one shared theoretical marginal, `p'`, which is what makes a single
/// pooled histogram comparison meaningful.
final class SampledMTPTruncatedTargetDistributionTests: XCTestCase {
    private static let targetLogits: [Double] = [2.0, 1.0, 0.5, 0.0, -0.5, -1.0]
    // Chosen so the draft's own top-2 ({3, 4}) is disjoint from the
    // target's top-2 ({0, 1}), while still placing enough mass on tokens 0
    // and 1 that a meaningful fraction of blocks accept (residual
    // correction's acceptance probability is min(1, p'(x)/q(x)), which is
    // 1 whenever the drafted token is 0 or 1 here).
    private static let proposalLogits: [Double] = [2.0, 1.9, -3.0, 2.05, 2.02, -3.0]
    private static let topK = 2
    private static let blockCount = 3000

    func testEmittedHistogramMatchesIndependentlyComputedTruncatedTargetAndRejectsBothMisimplementations() throws {
        let fullTargetProbabilities = Self.referenceSoftmax(Self.targetLogits)
        let truncatedTargetProbabilities = Self.referenceTopKTruncated(
            fullTargetProbabilities, topK: Self.topK)
        // Structural precondition for the fixture itself, not the
        // production code: confirms the two top-2 sets really are disjoint,
        // so this test is actually exercising truncation and not an
        // accidentally-overlapping corner case.
        let targetTopTwo = Set(Self.topIndices(fullTargetProbabilities, count: 2))
        let proposalTopTwo = Set(Self.topIndices(Self.referenceSoftmax(Self.proposalLogits), count: 2))
        XCTAssertTrue(targetTopTwo.isDisjoint(with: proposalTopTwo))

        // Mass an UNTRUNCATED p would place outside the top-2 support --
        // the wrong-model yardstick controls (a) and (b) below compare
        // against, and the fraction the executed discrimination control
        // further down independently reproduces by actually running an
        // untruncated provider over this same fixture.
        let untruncatedLeakedMass = (Self.topK ..< fullTargetProbabilities.count)
            .reduce(0.0) { $0 + fullTargetProbabilities[$1] }

        let truncation = SampledMTPSamplingTruncation(
            temperature: 1, topP: 1, topK: Self.topK, minP: 0)
        let provider = SeededSampledMTPBlockRuntimeProvider(seed: 0xC0FF_EE01, truncation: truncation)
        let targetLogitsArray = MLXArray(Self.targetLogits.map(Float.init))
        let proposalLogitsArray = MLXArray(Self.proposalLogits.map(Float.init))

        let (histogram, acceptedBlocks) = try Self.runBlocks(
            provider: provider,
            proposalLogitsArray: proposalLogitsArray,
            targetLogitsArray: targetLogitsArray,
            blockCount: Self.blockCount)
        let totalTokens = histogram.reduce(0, +)

        // Anti-vacuity: both the residual-correction path (rejects) and the
        // bonus path (fully-accepted blocks) must actually have fired many
        // times, or this test would not cover the bonus-row truncation
        // site at all -- exactly the gap item 4 calls out.
        XCTAssertGreaterThan(acceptedBlocks, Self.blockCount / 5)
        XCTAssertLessThan(acceptedBlocks, Self.blockCount * 4 / 5)

        // --- Main assertion: matches the independently computed p' ---
        // p' has exactly zero mass outside the top-2 indices, and that
        // truncation is enforced by an exact `-infinity` masked logit
        // feeding an exact `0.0` softmax output, not by a small tolerance
        // -- so no correctly-truncated run can ever emit an out-of-set
        // token, and this is checked unconditionally rather than
        // statistically.
        let leakedOutsideTopK = (Self.topK ..< histogram.count).reduce(0) { $0 + histogram[$1] }
        XCTAssertEqual(leakedOutsideTopK, 0)

        // Within the top-2 support, check the token-0 fraction against p'(0)
        // with a 6-sigma normal-approximation bound on the observed
        // proportion (n = totalTokens, p = p'(0)): this is generous enough
        // to make a false failure astronomically unlikely while still
        // being far tighter than either control's expected gap below.
        let expectedToken0Fraction = truncatedTargetProbabilities[0]
        let observedToken0Fraction = Double(histogram[0]) / Double(totalTokens)
        let sixSigmaBound = 6 * (expectedToken0Fraction * (1 - expectedToken0Fraction)
            / Double(totalTokens)).squareRoot()
        XCTAssertLessThan(
            abs(observedToken0Fraction - expectedToken0Fraction), sixSigmaBound)

        // --- Control (a): analytic magnitude check against the untruncated-p model ---
        // `leakedOutsideTopK == 0` is an exact COUNT, not a proportion, so
        // comparing `untruncatedLeakedMass` (a MASS/fraction) against
        // `sixSigmaBound` (a bound on a PROPORTION estimate) compares two
        // different kinds of quantity and is not decisive at this sample
        // size: at `blockCount = 3000` (so at most 6000 emitted tokens)
        // `10 * sixSigmaBound` never drops below the fixed
        // `untruncatedLeakedMass` of this fixture, so that comparison can
        // never pass. Expressing the control as an expected leaked COUNT
        // under the wrong (untruncated) model fixes this: with this
        // fixture's numbers, an untruncated p would be expected to leak on
        // the order of a thousand tokens over a run this size, which is far
        // past `decisiveLeakCountThreshold` -- large enough that observing
        // exactly 0 leaked tokens (the real assertion above) is
        // astronomically unlikely to have happened by chance under this
        // wrong model. This and control (b) are analytic magnitude checks:
        // they establish that the exact-zero observation is decisive
        // evidence against the wrong models, not that either wrong model
        // was actually run -- see the executed discrimination control
        // below for that.
        let decisiveLeakCountThreshold = 30.0
        let untruncatedModelExpectedLeakedCount = untruncatedLeakedMass * Double(totalTokens)
        XCTAssertGreaterThan(untruncatedModelExpectedLeakedCount, decisiveLeakCountThreshold)

        // --- Control (b): analytic magnitude check against a bonus-untruncated model ---
        // If only the per-step sites were truncated and the bonus row were
        // left untruncated, every fully-accepted block would still emit its
        // bonus token from the FULL softmax, leaking `untruncatedLeakedMass`
        // outside the top-2 set on each of the `acceptedBlocks` accepted
        // blocks this run actually observed (using the run's own accept
        // count, so this is not a hypothetical count). Same fix as control
        // (a): express it as an expected COUNT compared against the same
        // decisive threshold, not a fraction compared against a proportion
        // bound. Like control (a), this is an analytic magnitude check, not
        // an executed mutation -- see the executed control below.
        let bonusUntruncatedModelExpectedLeakedCount =
            Double(acceptedBlocks) * untruncatedLeakedMass
        XCTAssertGreaterThan(bonusUntruncatedModelExpectedLeakedCount, decisiveLeakCountThreshold)

        // --- Executed discrimination control: an actually-untruncated provider ---
        // Controls (a) and (b) above never run a misconfigured provider --
        // they are pure arithmetic and would not change if, say, the bonus
        // site's `truncation:` argument were deleted from the production
        // code. This block closes that gap by actually constructing a
        // second `SeededSampledMTPBlockRuntimeProvider` with
        // `truncation: .untruncated` and driving it over the exact same
        // fixture via `decide` called directly. `decide` does not consult
        // `supports(parameters:)` at all -- confirmed by reading its body
        // above, the truncation cross-check lives only inside `supports` --
        // so calling `decide` directly bypasses that gate and lets a
        // genuinely mismatched provider run to completion instead of being
        // refused. Observing `leakedOutsideTopK > 0` here, at a fraction
        // close to the independently-computed `untruncatedLeakedMass`, is
        // what converts "would have leaked" into "did leak": it proves the
        // `leakedOutsideTopK == 0` assertion on the correctly-truncated
        // provider above is an actually discriminating measurement, not
        // trivially satisfied because nothing in this fixture could ever
        // produce a nonzero leak.
        let untruncatedProvider = SeededSampledMTPBlockRuntimeProvider(
            seed: 0xC0FF_EE02, truncation: .untruncated)
        let (untruncatedHistogram, _) = try Self.runBlocks(
            provider: untruncatedProvider,
            proposalLogitsArray: proposalLogitsArray,
            targetLogitsArray: targetLogitsArray,
            blockCount: Self.blockCount)
        let untruncatedTotalTokens = untruncatedHistogram.reduce(0, +)
        let untruncatedLeakedOutsideTopK = (Self.topK ..< untruncatedHistogram.count)
            .reduce(0) { $0 + untruncatedHistogram[$1] }
        XCTAssertGreaterThan(untruncatedLeakedOutsideTopK, 0)
        let untruncatedObservedLeakedFraction =
            Double(untruncatedLeakedOutsideTopK) / Double(untruncatedTotalTokens)
        XCTAssertEqual(untruncatedObservedLeakedFraction, untruncatedLeakedMass, accuracy: 0.05)
    }

    /// Shared driver for both the correctly-truncated provider and the
    /// executed-discrimination-control's deliberately untruncated provider:
    /// samples a proposal, calls `decide` for a single-draft-token block,
    /// and accumulates the emitted-token histogram and accept count.
    private static func runBlocks(
        provider: SeededSampledMTPBlockRuntimeProvider,
        proposalLogitsArray: MLXArray,
        targetLogitsArray: MLXArray,
        blockCount: Int
    ) throws -> (histogram: [Int], acceptedBlocks: Int) {
        var histogram = [Int](repeating: 0, count: Self.targetLogits.count)
        var acceptedBlocks = 0
        for _ in 0 ..< blockCount {
            let proposedToken = provider.proposalSampler.sample(logits: proposalLogitsArray)
                .item(Int.self)
            let decision = try provider.decide(
                proposedTokens: [proposedToken],
                targetLogits: [targetLogitsArray],
                bonusTargetLogits: targetLogitsArray)
            if decision.acceptedDraftCount == 1 {
                acceptedBlocks += 1
            }
            for token in decision.outputTokens {
                histogram[token] += 1
            }
        }
        return (histogram, acceptedBlocks)
    }

    private static func referenceSoftmax(_ logits: [Double]) -> [Double] {
        let maxLogit = logits.max() ?? 0
        let expValues = logits.map { exp($0 - maxLogit) }
        let sum = expValues.reduce(0, +)
        return expValues.map { $0 / sum }
    }

    /// Keeps only the `topK` highest-probability entries (ties broken by
    /// index), zeroes the rest, and renormalizes -- independently
    /// reimplementing (in `[Double]`, never by calling
    /// `truncatedSamplingProbabilities`) what the top-p/top-k/temperature-1
    /// filter chain reduces to when top-p and min-p are both no-ops.
    private static func referenceTopKTruncated(_ probabilities: [Double], topK: Int) -> [Double] {
        precondition(topK > 0 && topK < probabilities.count)
        let keptIndices = Set(topIndices(probabilities, count: topK))
        let masked = probabilities.enumerated().map { keptIndices.contains($0.offset) ? $0.element : 0 }
        let sum = masked.reduce(0, +)
        return masked.map { $0 / sum }
    }

    private static func topIndices(_ probabilities: [Double], count: Int) -> [Int] {
        Array(probabilities.indices.sorted { probabilities[$0] > probabilities[$1] }.prefix(count))
    }
}

/// Regression coverage for the production OOM risk: before the trim fix,
/// `SeededSampledMTPProposalSampler.commit` (and the equivalent methods on
/// `FixedUniformProposalSampler` and `NondeterministicSampledMTPProposalSampler`)
/// only advanced `consumedCaptureCount` and never shrank `captures`, so every
/// already-committed full-vocabulary probability row
/// (`CapturedProposal.probabilities`, ~1.2 MB at this model's ~151,936-wide
/// vocabulary) stayed retained in memory for the life of the sampler. This
/// provider is now reachable on the serving path
/// (`MTPSpeculativeDecoder.prefill` constructs one per request), so an
/// unbounded-retention regression here is a per-request multi-gigabyte leak,
/// not merely a harness inefficiency.
final class SeededSampledMTPProposalCaptureRetentionTests: XCTestCase {
    /// Drives `blockCount` two-draft-token blocks and asserts, after every
    /// single one, that nothing from a completed block is still retained.
    /// This is a concrete bound tied to the block shape (exactly 0
    /// outstanding captures immediately after a successful `decide()`
    /// commits and trims the whole block's captures), not merely "smaller
    /// than the naive total". The unfixed `commit` (`consumedCaptureCount +=
    /// count` with no `captures.removeFirst`) would instead have left
    /// `retainedProposalCaptureCount` growing by `proposalDistributions.count`
    /// (2) every block, reaching `blockCount * 2 == 1000` captures retained
    /// by the end of this run -- so this assertion would FAIL against the
    /// pre-fix code from the very first iteration onward.
    func testRetainedProposalCapturesStayBoundedAcrossManyBlocks() throws {
        let provider = SeededSampledMTPBlockRuntimeProvider(seed: 0xC0FF_EE03)
        let proposalDistributions = [[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]]
        let bonusDistribution = [0.2, 0.3, 0.5]
        let blockCount = 500

        func logits(_ distribution: [Double]) -> MLXArray {
            MLXArray(distribution.map { Float(log($0)) })
        }

        for _ in 0 ..< blockCount {
            let proposals = proposalDistributions.map { distribution in
                provider.proposalSampler.sample(logits: logits(distribution)).item(Int.self)
            }
            _ = try provider.decide(
                proposedTokens: proposals,
                targetLogits: proposalDistributions.map(logits),
                bonusTargetLogits: logits(bonusDistribution))

            // Immediately after each successful `decide()`, every capture
            // made for that block has been committed and trimmed: nothing
            // should carry over into the next block.
            XCTAssertEqual(provider.retainedProposalCaptureCount, 0)
        }

        // Final check, stated independently of the per-iteration loop
        // above: the pre-fix code would have accumulated
        // `blockCount * proposalDistributions.count` == 1000 retained
        // captures here; the fix keeps it at 0.
        XCTAssertEqual(provider.retainedProposalCaptureCount, 0)
        XCTAssertLessThan(
            provider.retainedProposalCaptureCount,
            blockCount * proposalDistributions.count)
    }
}
