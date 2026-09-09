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
