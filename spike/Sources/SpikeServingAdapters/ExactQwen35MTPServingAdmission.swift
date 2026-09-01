import HarnessCore
import ServingCore
import SpikeCore

/// Immutable execution metadata for the one exact Qwen3.5 target/drafter row whose greedy corpus
/// passed the serving prerequisite gate. This descriptor is classification output only: it does not
/// contain model objects, load weights, construct an iterator, or authorize a live MTP executor.
public struct ExactQwen35MTPServingDescriptor: Equatable, Sendable {
    public let artifactSelection: Qwen35ExactMTPRuntimeSelection
    public let targetModelID: String
    public let drafterModelID: String
    public let targetRevision: String
    public let drafterRevision: String
    public let sourceRevision: String
    public let architecture: QwenMTPArchitecture
    public let runtimeBlockSize: Int
    public let maximumAcceptedDraftTokens: Int

    public init(
        artifactSelection: Qwen35ExactMTPRuntimeSelection = .qwen35_9BDepth1,
        targetModelID: String,
        drafterModelID: String,
        targetRevision: String,
        drafterRevision: String,
        sourceRevision: String,
        architecture: QwenMTPArchitecture,
        runtimeBlockSize: Int,
        maximumAcceptedDraftTokens: Int
    ) {
        self.artifactSelection = artifactSelection
        self.targetModelID = targetModelID
        self.drafterModelID = drafterModelID
        self.targetRevision = targetRevision
        self.drafterRevision = drafterRevision
        self.sourceRevision = sourceRevision
        self.architecture = architecture
        self.runtimeBlockSize = runtimeBlockSize
        self.maximumAcceptedDraftTokens = maximumAcceptedDraftTokens
    }
}

/// Stable reason an otherwise valid scalar request must not enter the future exact-Qwen MTP executor.
public enum ExactQwen35MTPServingScalarFallbackReason: String, Equatable, Sendable {
    case invalidSpeculativeRequestCount = "invalid_speculative_request_count"
    case disabled
    case noValidatedBinding = "no_validated_binding"
    case bindingMismatch = "binding_mismatch"
    case sampledGeneration = "sampled_generation"
    case logitProcessor = "logit_processor"
    case activeTools = "active_tools"
    case speculativeConcurrencyConflict = "speculative_concurrency_conflict"
}

public enum ExactQwen35MTPServingAdmissionDecision: Equatable, Sendable {
    case eligible(ExactQwen35MTPServingDescriptor)
    case scalarFallback(ExactQwen35MTPServingScalarFallbackReason)
}

/// Pure, fail-closed request classifier for a future solo-only exact-Qwen MTP serving route.
///
/// The caller must pass a binding returned by the existing artifact preflight. Even then, eligibility
/// is intentionally narrower than engine capability: only the measured processor-free greedy shape is
/// admitted, while sampling, penalties, tools, and overlapping speculative work stay on scalar serving.
/// This policy never loads a drafter and must not be treated as proof of streaming, cancellation, cache
/// finalization, or API parity; those remain separate live-executor gates.
public enum ExactQwen35MTPServingAdmissionPolicy {
    public static func decide(
        selection: Qwen35ExactMTPRuntimeSelection = .qwen35_9BDepth1,
        enabled: Bool,
        binding: QwenMTPArtifactBinding?,
        sampling: ServingSamplingPolicy,
        penalties: DecoderPenalties,
        hasActiveTools: Bool,
        speculativeRequestCount: Int
    ) -> ExactQwen35MTPServingAdmissionDecision {
        guard speculativeRequestCount >= 0 else {
            return .scalarFallback(.invalidSpeculativeRequestCount)
        }
        guard enabled else {
            return .scalarFallback(.disabled)
        }
        guard let binding else {
            return .scalarFallback(.noValidatedBinding)
        }
        guard matchesReviewedBinding(binding, selection: selection) else {
            return .scalarFallback(.bindingMismatch)
        }
        guard case .greedy = sampling else {
            return .scalarFallback(.sampledGeneration)
        }
        guard penalties.isEmpty else {
            return .scalarFallback(.logitProcessor)
        }
        guard !hasActiveTools else {
            return .scalarFallback(.activeTools)
        }
        guard speculativeRequestCount == 0 else {
            return .scalarFallback(.speculativeConcurrencyConflict)
        }

        return .eligible(
            ExactQwen35MTPServingDescriptor(
                artifactSelection: selection,
                targetModelID: binding.targetModelID,
                drafterModelID: binding.drafterModelID,
                targetRevision: binding.targetRevision,
                drafterRevision: binding.drafterRevision,
                sourceRevision: binding.sourceRevision,
                architecture: binding.architecture,
                runtimeBlockSize: binding.runtimeBlockSize,
                maximumAcceptedDraftTokens: binding.maximumAcceptedDraftTokens))
    }

    private static func matchesReviewedBinding(
        _ binding: QwenMTPArtifactBinding,
        selection: Qwen35ExactMTPRuntimeSelection
    ) -> Bool {
        let lock = reviewedLock(selection)
        return binding.targetModelID == lock.targetIdentity.modelID
            && binding.drafterModelID == lock.drafterIdentity.modelID
            && binding.targetRevision == lock.targetIdentity.revision
            && binding.drafterRevision == lock.drafterIdentity.revision
            && binding.sourceRevision == lock.sourceRevision
            && binding.architecture == lock.architecture
            && binding.runtimeBlockSize == 3
            && binding.maximumAcceptedDraftTokens == 2
    }

    private static func reviewedLock(
        _ selection: Qwen35ExactMTPRuntimeSelection
    ) -> QwenMTPArtifactLock {
        switch selection {
        case .qwen35_9BDepth1:
            QwenMTPKnownArtifactLocks.qwen35_9BDepth1
        case .qwen38_27BMXFP8Depth1:
            QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
        case .qwen38_27B4BitDepth1:
            // Identity mapping only; the 4-bit row has no ServingCore argument case, so serving
            // cannot select it until its own live-exactness and quality gates land.
            QwenMTPKnownArtifactLocks.qwen38_27B4BitDepth1
        }
    }
}
