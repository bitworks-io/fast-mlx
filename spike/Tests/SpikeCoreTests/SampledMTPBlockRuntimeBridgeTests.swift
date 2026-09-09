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

/// Closes a composed top-p AND top-k truncation gap that
/// `SampledMTPTruncatedTargetDistributionTests` above does not cover: that
/// class qualifies top-k ALONE (`topP: 1, topK: 2` -- top-p disabled). The
/// deployed thinking preset (`top_p: 0.95, top_k: 20`) runs BOTH filters, in
/// mlx-lm order top-p -> min-p -> top-k (`truncatedSamplingProbabilities`),
/// and that composed path -- which runs on every production request -- had
/// never been exercised.
///
/// `referenceComposedTruncation` and its helpers below are this class's own
/// independent `[Double]` reimplementation of the top-p/top-k filter chain,
/// derived from the semantics of `applyTopPFilter`/`applyTopKFilter`. They
/// are never built by calling `truncatedSamplingProbabilities` or any
/// vendored filter directly -- a defect in the vendored chain must not also
/// be able to corrupt the yardstick meant to catch it.
final class SampledMTPComposedTopPTopKTruncationTests: XCTestCase {
    private static let targetLogits: [Double] = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, -3.0, -4.0]
    private static let blockCount = 3000
    private static let seed: UInt64 = 0xBA5E_D000

    /// Draft distribution shared by both regimes below: half its mass sits
    /// INSIDE the composed support `{0, 1, 2}` -- in exactly the
    /// proportions of the independently computed `p'`, scaled down to a
    /// total of 0.5, so `p'(x) / q(x) == 2` for every in-support `x` and
    /// acceptance probability `min(1, 2)` is deterministically 1 -- and the
    /// other half sits on token 3, OUTSIDE the composed support, where
    /// `p'(3) == 0` makes acceptance probability exactly 0 (always
    /// rejected, forcing the residual-correction path). That 50/50 split
    /// lands `acceptedBlocks` near the middle of `(blockCount/5,
    /// blockCount*4/5)`, so both the residual-correction path and the
    /// fully-accepted bonus path fire many times on every run. Indices
    /// `4...7` carry only a negligible `1e-6` each so the row stays a
    /// valid, fully-supported distribution (no exact zero the proposal
    /// sampler could ever be asked to draw and reject as unsupported).
    private static let proposalLogits: [Double] = [
        -1.605049, -1.805049, -2.005049, -0.693155,
        -13.815511, -13.815511, -13.815511, -13.815511,
    ]

    /// Regime A: top-p (0.95) alone admits a six-token nucleus
    /// `{0,1,2,3,4,5}`, but top-k (3) then binds INSIDE that nucleus and
    /// cuts it down further to `{0,1,2}` -- `composedSupport !=
    /// nucleusSupport`. Discriminates against a wiring that drops top-k
    /// while keeping top-p.
    func testRegimeATopKBindsInsideTopPNucleus() throws {
        try runComposedTruncationRegime(
            topP: 0.95,
            topK: 3,
            expectedNucleus: [0, 1, 2, 3, 4, 5],
            expectedComposed: [0, 1, 2],
            expectedTopKOnly: [0, 1, 2],
            seedOffset: 1) { nucleusSupport, composedSupport, _ in
                XCTAssertNotEqual(composedSupport, nucleusSupport)
            }
    }

    /// Regime B, the production-shaped case: top-p (0.5) alone already
    /// narrows to `{0,1,2}`, the same set top-k (6) composes down to --
    /// `composedSupport == nucleusSupport` -- while top-k ALONE (ignoring
    /// top-p) would keep six tokens `{0,1,2,3,4,5}`: `composedSupport !=
    /// topKOnlySupport`. At the deployed preset the nucleus is usually
    /// smaller than `top_k: 20`, so top-p does the real work here and
    /// `applyTopKFilter` must be a no-op on an input that already carries
    /// `-inf` entries from the top-p pass -- that composition had never
    /// been tested before this method.
    func testRegimeBTopPBindsBeforeTopK() throws {
        try runComposedTruncationRegime(
            topP: 0.5,
            topK: 6,
            expectedNucleus: [0, 1, 2],
            expectedComposed: [0, 1, 2],
            expectedTopKOnly: [0, 1, 2, 3, 4, 5],
            seedOffset: 2) { _, composedSupport, topKOnlySupport in
                XCTAssertNotEqual(composedSupport, topKOnlySupport)
            }
    }

    // MARK: - Attribution fixture (30-token, real deployed topK: 20)

    /// Weight `w` in `1...30` sits at `index = (w - 1 + 13) % 30`; that
    /// token's logit is `ln(w / 465)` (465 == sum(1...30)), so `softmax` of
    /// this row is exactly `w / 465` at every index -- no rounding-
    /// sensitive normalization step to get wrong.
    private static let attributionVocabularySize = 30
    private static let attributionTopK = 20
    private static let attributionTargetLogits: [Double] = [
        -3.2516656477, -3.1975984264, -3.1463051320, -3.0975149679, -3.0509949522,
        -3.0065431897, -2.9639835752, -2.9231615807, -2.8839408676, -2.8462005396,
        -2.8098328954, -2.7747415756, -2.7408400239, -6.1420374056, -5.4488902250,
        -5.0434251169, -4.7557430445, -4.5325994932, -4.3502779364, -4.1961272565,
        -4.0625958639, -3.9448128283, -3.8394523126, -3.7441421328, -3.6571307558,
        -3.5770880481, -3.5029800760, -3.4339872045, -3.3694486833, -3.3088240615,
    ]

    /// Draft distribution for the attribution fixture: half its mass
    /// proportional to `p'` over `keptArmIndices` (weights `11...30`,
    /// scaled to sum 0.5 -- same `ratio == 2, always-accept-when-inside`
    /// construction as `proposalLogits` above), a quarter split uniformly
    /// over `topPArmIndices` (weights `1...6`), and a quarter split
    /// uniformly over `topKArmIndices` (weights `7...10`). The two arms
    /// getting nonzero draft mass is what lets a defect that leaks either
    /// arm actually surface in the histogram: a `q` with zero mass there
    /// could never propose those tokens in the first place.
    private static let attributionProposalLogits: [Double] = [
        -3.8189325824, -3.7648653611, -3.7135720667, -3.6647819025, -3.6182618869,
        -3.5738101243, -3.5312505099, -3.4904285154, -3.4512078022, -3.4134674743,
        -3.3770998301, -3.3420085103, -3.3081069586, -3.1780538303, -3.1780538303,
        -3.1780538303, -3.1780538303, -3.1780538303, -3.1780538303, -2.7725887222,
        -2.7725887222, -2.7725887222, -2.7725887222, -4.3114090675, -4.2243976905,
        -4.1443549828, -4.0702470106, -4.0012541392, -3.9367156180, -3.8760909962,
    ]

    /// Removed by top-p ALONE (weights `1...6`): still masked even with
    /// `topK: 0` -- see the executed control below.
    private static let topPArmIndices: Set<Int> = [13, 14, 15, 16, 17, 18]
    /// Removed by top-k ALONE (weights `7...10`): pass top-p's nucleus but
    /// do not survive the top-20 cut.
    private static let topKArmIndices: Set<Int> = [19, 20, 21, 22]
    /// Survive both filters (weights `11...30`).
    private static let keptArmIndices: Set<Int> = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 24, 25, 26, 27, 28, 29,
    ]

    /// The two indices a top-k-before-top-p filter-order swap removes that
    /// the correct top-p-then-top-k order keeps. Weights 11 and 12
    /// (indices 23 and 24) sit just inside the boundary top-p's `1 - topP`
    /// threshold draws against the FULL, unnormalized `465`-weight row.
    /// Swap the order -- run top-k (20) first -- and that boundary shifts:
    /// top-k alone keeps weights `11...30`, an unnormalized surviving mass
    /// of `410/465`, and the subsequent top-p pass then measures its
    /// threshold against that smaller mass instead, which cuts deeper and
    /// drops these two weights on top of the six the top-p arm (weights
    /// `1...6`) already drops under either order. Under the CORRECT order
    /// they instead survive both filters, carrying combined mass `23/410`
    /// of the composed support; under the SWAPPED order they carry exactly
    /// zero. These two indices exist specifically to catch that one
    /// mutation -- swapping the `if topP > 0 && topP < 1` and `if topK >
    /// 0` blocks in `truncatedSamplingProbabilities` -- which the
    /// `leakedTopPArm`/`leakedTopKArm` assertions above do not catch: both
    /// arms they check stay excluded under either filter order, so neither
    /// one moves when only the order changes.
    private static let orderDiscriminatorIndices: Set<Int> = [23, 24]

    /// Both `SampledMTPTruncatedTargetDistributionTests` above and
    /// `testRegimeATopKBindsInsideTopPNucleus`/`testRegimeBTopPBindsBeforeTopK`
    /// converge on the SAME composed support (`{0,1,2}` for both regimes
    /// above), so an out-of-support leak count there cannot say WHICH
    /// filter misbehaved. This fixture separates the two arms at the real
    /// deployed `topK: 20` over a 30-token vocabulary --
    /// `applyTopKFilter` no-ops whenever `topK >= vocabularySize`
    /// (Evaluate.swift:316), so anything narrower than 21 tokens would make
    /// a `topK: 20` assertion pass by construction rather than by actually
    /// exercising the filter, which is why this fixture is 30 wide.
    func testAttributesLeakToTopPOrTopKArmSeparately() throws {
        let vocabularySize = Self.attributionVocabularySize
        let topK = Self.attributionTopK
        let topP = 0.95

        let fullProbabilities = Self.referenceSoftmax(Self.attributionTargetLogits)
        let nucleusSupport = Self.referenceNucleusSupport(fullProbabilities, topP: topP)
        let (composedSupport, composedProbabilities) = Self.referenceComposedTruncation(
            fullProbabilities: fullProbabilities, topP: topP, topK: topK)

        // Anti-vacuity computed IN-TEST, not asserted only in a comment: a
        // later edit to this fixture (e.g. shrinking the vocabulary, or
        // raising topK) must not be able to silently make the top-k arm
        // inert without one of these four failing first.
        XCTAssertEqual(nucleusSupport.count, 24)
        XCTAssertEqual(topK, 20)
        XCTAssertGreaterThan(nucleusSupport.count, topK)
        XCTAssertLessThan(topK, vocabularySize)

        // Pins the fixture against the independently-verified index
        // mapping: if this fails, the fixture (or this reimplementation of
        // it) is wrong, not the production code under test -- stop and
        // report rather than pushing past it.
        XCTAssertEqual(nucleusSupport, Self.keptArmIndices.union(Self.topKArmIndices))
        XCTAssertEqual(composedSupport, Self.keptArmIndices)

        let truncation = SampledMTPSamplingTruncation(
            temperature: 1, topP: Float(topP), topK: topK, minP: 0)
        let provider = SeededSampledMTPBlockRuntimeProvider(
            seed: Self.seed &+ 3, truncation: truncation)
        let targetLogitsArray = MLXArray(Self.attributionTargetLogits.map(Float.init))
        let proposalLogitsArray = MLXArray(Self.attributionProposalLogits.map(Float.init))

        let (histogram, acceptedBlocks) = try Self.runBlocks(
            provider: provider,
            proposalLogitsArray: proposalLogitsArray,
            targetLogitsArray: targetLogitsArray,
            vocabularySize: vocabularySize,
            blockCount: Self.blockCount)
        let totalTokens = histogram.reduce(0, +)

        XCTAssertGreaterThan(acceptedBlocks, Self.blockCount / 5)
        XCTAssertLessThan(acceptedBlocks, Self.blockCount * 4 / 5)

        // The top-p arm: stays green even if top-k alone were dropped
        // (M2), since top-p already excludes it independently.
        let leakedTopPArm = Self.topPArmIndices.reduce(0) { $0 + histogram[$1] }
        XCTAssertEqual(leakedTopPArm, 0)
        // The top-k arm: THIS is the load-bearing assertion for
        // attribution. It stays green under a pure top-p mutation (M3
        // still excludes this arm independently, since top-p never touches
        // it) and goes red specifically under a top-k mutation (M2) --
        // that separation is the whole point of this fixture.
        let leakedTopKArm = Self.topKArmIndices.reduce(0) { $0 + histogram[$1] }
        XCTAssertEqual(leakedTopKArm, 0)

        // --- Order discriminator: catches a top-k-before-top-p filter
        // swap that the two arm checks above cannot. `leakedTopPArm` and
        // `leakedTopKArm` both stay green under a pure order swap -- their
        // arms are excluded by top-p and top-k respectively regardless of
        // which filter runs first -- so a mutation that swaps the order of
        // the `if topP > 0 && topP < 1` and `if topK > 0` blocks in
        // `truncatedSamplingProbabilities` would otherwise pass this whole
        // test undetected. `orderDiscriminatorIndices` (weights 11 and 12)
        // sit exactly on that order-dependent boundary: kept under the
        // correct order, dropped under the swap. See the field's doc
        // comment for the derivation of `410/465` and `23/410`.
        //
        // Anti-vacuity, checked against the independently-derived
        // reference support (`composedSupport`, never
        // `truncatedSamplingProbabilities`): a future edit to this fixture
        // that stopped keeping both discriminator indices in the composed
        // support would otherwise silently disarm the two assertions
        // below rather than failing loudly here first.
        XCTAssertTrue(Self.orderDiscriminatorIndices.isSubset(of: composedSupport))

        let orderDiscriminatorCount = Self.orderDiscriminatorIndices
            .reduce(0) { $0 + histogram[$1] }
        XCTAssertGreaterThan(orderDiscriminatorCount, 0)
        let orderDiscriminatorFraction = Double(orderDiscriminatorCount) / Double(totalTokens)
        XCTAssertEqual(orderDiscriminatorFraction, 23.0 / 410.0, accuracy: 0.02)

        // 6-sigma proportion check on one in-support index against the
        // independently computed p'.
        let probeIndex = 12 // weight 30, the largest surviving mass
        let expectedProbeFraction = composedProbabilities[probeIndex]
        let observedProbeFraction = Double(histogram[probeIndex]) / Double(totalTokens)
        let sixSigmaBound = 6 * (expectedProbeFraction * (1 - expectedProbeFraction)
            / Double(totalTokens)).squareRoot()
        XCTAssertLessThan(
            abs(observedProbeFraction - expectedProbeFraction), sixSigmaBound)

        // --- Executed control: topK: 0 on the SAME fixture and the SAME
        // draft, driven through `decide` directly (bypassing `supports`,
        // as the executed control in `SampledMTPTruncatedTargetDistributionTests`
        // does). If this leaked NOTHING on the top-k arm either, the
        // `leakedTopKArm == 0` assertion above would be trivially
        // satisfied by a fixture that could never leak there regardless of
        // truncation -- this proves it is actually reachable and that
        // disabling top-k alone (not top-p too) is what opens it.
        let controlTruncation = SampledMTPSamplingTruncation(
            temperature: 1, topP: Float(topP), topK: 0, minP: 0)
        let controlProvider = SeededSampledMTPBlockRuntimeProvider(
            seed: Self.seed &+ 4, truncation: controlTruncation)
        let (controlHistogram, _) = try Self.runBlocks(
            provider: controlProvider,
            proposalLogitsArray: proposalLogitsArray,
            targetLogitsArray: targetLogitsArray,
            vocabularySize: vocabularySize,
            blockCount: Self.blockCount)
        let controlTotalTokens = controlHistogram.reduce(0, +)

        // The control still excludes the top-p arm: proves the control
        // moved ONLY the top-k arm, not truncation as a whole.
        let controlLeakedTopPArm = Self.topPArmIndices.reduce(0) { $0 + controlHistogram[$1] }
        XCTAssertEqual(controlLeakedTopPArm, 0)

        let controlLeakedTopKArm = Self.topKArmIndices.reduce(0) { $0 + controlHistogram[$1] }
        XCTAssertGreaterThan(controlLeakedTopKArm, 0)
        // Independently computed expected mass: weights 7+8+9+10 == 34,
        // over the nucleus-only (topP: 0.95, topK: 0) total sum(7...30) ==
        // 444. At ~6000 emitted tokens that is ~460 expected, so observing
        // exactly 0 on the correctly-truncated provider above is decisive
        // evidence rather than luck.
        let controlObservedTopKArmFraction =
            Double(controlLeakedTopKArm) / Double(controlTotalTokens)
        XCTAssertEqual(controlObservedTopKArmFraction, 34.0 / 444.0, accuracy: 0.02)
    }

    // MARK: - Shared driver

    private func runComposedTruncationRegime(
        topP: Double,
        topK: Int,
        expectedNucleus: Set<Int>,
        expectedComposed: Set<Int>,
        expectedTopKOnly: Set<Int>,
        seedOffset: UInt64,
        additionalDiscrimination: (
            _ nucleusSupport: Set<Int>,
            _ composedSupport: Set<Int>,
            _ topKOnlySupport: Set<Int>
        ) -> Void
    ) throws {
        let fullProbabilities = Self.referenceSoftmax(Self.targetLogits)
        let nucleusSupport = Self.referenceNucleusSupport(fullProbabilities, topP: topP)
        let topKOnlySupport = Self.referenceTopKSupport(fullProbabilities, topK: topK)
        let (composedSupport, composedProbabilities) = Self.referenceComposedTruncation(
            fullProbabilities: fullProbabilities, topP: topP, topK: topK)

        // Pins the fixture against the table this increment was scoped
        // against: if either of these fails, the table (or this
        // reimplementation of it) is wrong, not the production code under
        // test -- stop and report rather than pushing past it.
        XCTAssertEqual(nucleusSupport, expectedNucleus)
        XCTAssertEqual(composedSupport, expectedComposed)
        XCTAssertEqual(topKOnlySupport, expectedTopKOnly)
        additionalDiscrimination(nucleusSupport, composedSupport, topKOnlySupport)

        let truncation = SampledMTPSamplingTruncation(
            temperature: 1, topP: Float(topP), topK: topK, minP: 0)
        let provider = SeededSampledMTPBlockRuntimeProvider(
            seed: Self.seed &+ seedOffset, truncation: truncation)
        let targetLogitsArray = MLXArray(Self.targetLogits.map(Float.init))
        let proposalLogitsArray = MLXArray(Self.proposalLogits.map(Float.init))

        let (histogram, acceptedBlocks) = try Self.runBlocks(
            provider: provider,
            proposalLogitsArray: proposalLogitsArray,
            targetLogitsArray: targetLogitsArray,
            vocabularySize: Self.targetLogits.count,
            blockCount: Self.blockCount)
        let totalTokens = histogram.reduce(0, +)

        // Anti-vacuity: both the residual-correction path (rejects) and the
        // bonus path (fully-accepted blocks) must actually have fired many
        // times.
        XCTAssertGreaterThan(acceptedBlocks, Self.blockCount / 5)
        XCTAssertLessThan(acceptedBlocks, Self.blockCount * 4 / 5)

        // Truncation is enforced by an exact `-infinity` -> exact `0.0`
        // softmax output, so any nonzero count outside the composed support
        // is a real defect, checked exactly rather than statistically.
        let leakedOutsideComposedSupport = histogram.indices
            .filter { !composedSupport.contains($0) }
            .reduce(0) { $0 + histogram[$1] }
        XCTAssertEqual(leakedOutsideComposedSupport, 0)

        // Within the composed support, check the token-0 fraction against
        // the independently computed p'(0) with a 6-sigma normal-
        // approximation bound on the observed proportion.
        let expectedToken0Fraction = composedProbabilities[0]
        let observedToken0Fraction = Double(histogram[0]) / Double(totalTokens)
        let sixSigmaBound = 6 * (expectedToken0Fraction * (1 - expectedToken0Fraction)
            / Double(totalTokens)).squareRoot()
        XCTAssertLessThan(
            abs(observedToken0Fraction - expectedToken0Fraction), sixSigmaBound)
    }

    /// Shared driver: samples a proposal, calls `decide` for a single-
    /// draft-token block, and accumulates the emitted-token histogram and
    /// accept count. Mirrors `SampledMTPTruncatedTargetDistributionTests
    /// .runBlocks`, generalized with an explicit `vocabularySize` so it can
    /// drive both this class's 8-token and 30-token fixtures.
    private static func runBlocks(
        provider: SeededSampledMTPBlockRuntimeProvider,
        proposalLogitsArray: MLXArray,
        targetLogitsArray: MLXArray,
        vocabularySize: Int,
        blockCount: Int
    ) throws -> (histogram: [Int], acceptedBlocks: Int) {
        var histogram = [Int](repeating: 0, count: vocabularySize)
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

    // MARK: - Independent reference implementation

    private static func referenceSoftmax(_ logits: [Double]) -> [Double] {
        let maxLogit = logits.max() ?? 0
        let expValues = logits.map { exp($0 - maxLogit) }
        let sum = expValues.reduce(0, +)
        return expValues.map { $0 / sum }
    }

    /// Nucleus-sampling support: sorts ascending, cumsums probabilities, and
    /// keeps indices whose cumulative mass exceeds `1 - topP` -- mirroring
    /// `applyTopPFilter`'s ascending-sort/cumsum/threshold shape exactly,
    /// including its `topP > 0 && topP < 1` no-op guard (`applyTopPFilter`
    /// is only ever called from inside that guard in
    /// `truncatedSamplingProbabilities`).
    private static func referenceNucleusSupport(
        _ probabilities: [Double], topP: Double
    ) -> Set<Int> {
        guard topP > 0, topP < 1 else { return Set(probabilities.indices) }
        let ascending = probabilities.indices.sorted { probabilities[$0] < probabilities[$1] }
        var cumulative = 0.0
        var kept = Set<Int>()
        for index in ascending {
            cumulative += probabilities[index]
            if cumulative > 1 - topP {
                kept.insert(index)
            }
        }
        return kept
    }

    /// Top-k support: the `topK` largest-probability indices, ties broken by
    /// index -- mirroring `applyTopKFilter`'s `guard topK < vocabularySize
    /// else { return logprobs }` no-op (and the caller's `if topK > 0`
    /// guard) exactly.
    private static func referenceTopKSupport(
        _ probabilities: [Double], topK: Int
    ) -> Set<Int> {
        guard topK > 0, topK < probabilities.count else { return Set(probabilities.indices) }
        let descending = probabilities.indices.sorted { probabilities[$0] > probabilities[$1] }
        return Set(descending.prefix(topK))
    }

    /// Composes top-p THEN top-k, in mlx-lm's order: masks everything
    /// outside the top-p nucleus to zero first, then selects the `topK`
    /// largest of what remains (which can only ever narrow the nucleus
    /// further, never re-admit anything top-p already dropped -- masked
    /// entries stay at exactly `0.0`, never renormalized back to life), and
    /// renormalizes the survivors to sum to 1.
    private static func referenceComposedTruncation(
        fullProbabilities: [Double], topP: Double, topK: Int
    ) -> (support: Set<Int>, probabilities: [Double]) {
        let nucleusSupport = referenceNucleusSupport(fullProbabilities, topP: topP)
        let afterTopP = fullProbabilities.enumerated().map {
            nucleusSupport.contains($0.offset) ? $0.element : 0.0
        }
        let topKRaw = referenceTopKSupport(afterTopP, topK: topK)
        let composedSupport = Set(topKRaw.filter { afterTopP[$0] > 0 })
        let masked = afterTopP.enumerated().map {
            composedSupport.contains($0.offset) ? $0.element : 0.0
        }
        let sum = masked.reduce(0, +)
        return (composedSupport, masked.map { $0 / sum })
    }
}
