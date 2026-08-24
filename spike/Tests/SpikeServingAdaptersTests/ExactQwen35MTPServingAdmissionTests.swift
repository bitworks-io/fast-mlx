import XCTest

@testable import HarnessCore
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class ExactQwen35MTPServingAdmissionTests: XCTestCase {
    func testExactGreedySoloBindingIsEligible() {
        XCTAssertEqual(
            decide(),
            .eligible(
                ExactQwen35MTPServingDescriptor(
                    targetModelID: lock.targetIdentity.modelID,
                    drafterModelID: lock.drafterIdentity.modelID,
                    targetRevision: lock.targetIdentity.revision,
                    drafterRevision: lock.drafterIdentity.revision,
                    sourceRevision: lock.sourceRevision,
                    architecture: lock.architecture,
                    runtimeBlockSize: 3,
                    maximumAcceptedDraftTokens: 2)))
    }

    func testDisabledPolicyFallsBackBeforeBindingAdmission() {
        XCTAssertEqual(
            decide(enabled: false, binding: nil),
            .scalarFallback(.disabled))
    }

    func testMissingValidatedBindingFallsBack() {
        XCTAssertEqual(
            decide(binding: nil),
            .scalarFallback(.noValidatedBinding))
    }

    func testEveryExactBindingFieldDriftFallsBack() {
        let otherArchitecture = QwenMTPArchitecture(
            hiddenSize: lock.architecture.hiddenSize + 1,
            intermediateSize: lock.architecture.intermediateSize,
            vocabularySize: lock.architecture.vocabularySize,
            targetLayerCount: lock.architecture.targetLayerCount,
            fullAttentionInterval: lock.architecture.fullAttentionInterval,
            attentionHeadCount: lock.architecture.attentionHeadCount,
            keyValueHeadCount: lock.architecture.keyValueHeadCount,
            headDimension: lock.architecture.headDimension,
            usesDedicatedMTPEmbeddings: lock.architecture.usesDedicatedMTPEmbeddings)
        let drifts = [
            binding(targetModelID: "unknown/target"),
            binding(drafterModelID: "unknown/drafter"),
            binding(targetRevision: String(repeating: "1", count: 40)),
            binding(drafterRevision: String(repeating: "2", count: 40)),
            binding(sourceRevision: String(repeating: "3", count: 40)),
            binding(architecture: otherArchitecture),
            binding(runtimeBlockSize: 4),
            binding(maximumAcceptedDraftTokens: 1),
        ]

        for drift in drifts {
            XCTAssertEqual(
                decide(binding: drift),
                .scalarFallback(.bindingMismatch))
        }
    }

    func testSampledPolicyFallsBack() {
        XCTAssertEqual(
            decide(
                sampling: .sampled(
                    temperature: 0.7,
                    topP: 0.9,
                    topK: 40,
                    minP: 0.05,
                    seed: 7)),
            .scalarFallback(.sampledGeneration))
    }

    func testEveryNonzeroPenaltyFallsBackFromLogitProcessorPath() {
        for penalties in [
            DecoderPenalties(presencePenalty: 0.1),
            DecoderPenalties(frequencyPenalty: -0.1),
            DecoderPenalties(repetitionPenalty: 1.1),
        ] {
            XCTAssertEqual(
                decide(penalties: penalties),
                .scalarFallback(.logitProcessor))
        }
    }

    func testExplicitZeroPenaltiesRemainEligible() {
        XCTAssertEqual(
            decide(
                penalties: DecoderPenalties(
                    presencePenalty: 0,
                    frequencyPenalty: 0,
                    repetitionPenalty: 0)),
            decide())
    }

    func testActiveToolsFallBackUntilToolParityIsProven() {
        XCTAssertEqual(
            decide(hasActiveTools: true),
            .scalarFallback(.activeTools))
    }

    func testSpeculativeOverlapFallsBack() {
        XCTAssertEqual(
            decide(speculativeRequestCount: 1),
            .scalarFallback(.speculativeConcurrencyConflict))
    }

    func testNegativeSpeculativeRequestCountFailsClosed() {
        XCTAssertEqual(
            decide(speculativeRequestCount: -1),
            .scalarFallback(.invalidSpeculativeRequestCount))
    }
}

private let lock = QwenMTPKnownArtifactLocks.qwen35_9BDepth1

private func decide(
    enabled: Bool = true,
    binding: QwenMTPArtifactBinding? = binding(),
    sampling: ServingSamplingPolicy = .greedy,
    penalties: DecoderPenalties = .none,
    hasActiveTools: Bool = false,
    speculativeRequestCount: Int = 0
) -> ExactQwen35MTPServingAdmissionDecision {
    ExactQwen35MTPServingAdmissionPolicy.decide(
        enabled: enabled,
        binding: binding,
        sampling: sampling,
        penalties: penalties,
        hasActiveTools: hasActiveTools,
        speculativeRequestCount: speculativeRequestCount)
}

private func binding(
    targetModelID: String = lock.targetIdentity.modelID,
    drafterModelID: String = lock.drafterIdentity.modelID,
    targetRevision: String = lock.targetIdentity.revision,
    drafterRevision: String = lock.drafterIdentity.revision,
    sourceRevision: String = lock.sourceRevision,
    architecture: QwenMTPArchitecture = lock.architecture,
    runtimeBlockSize: Int = 3,
    maximumAcceptedDraftTokens: Int = 2
) -> QwenMTPArtifactBinding {
    QwenMTPArtifactBinding(
        targetModelID: targetModelID,
        drafterModelID: drafterModelID,
        targetRevision: targetRevision,
        drafterRevision: drafterRevision,
        sourceRevision: sourceRevision,
        architecture: architecture,
        runtimeBlockSize: runtimeBlockSize,
        maximumAcceptedDraftTokens: maximumAcceptedDraftTokens)
}
