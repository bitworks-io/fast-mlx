import Foundation
import HarnessCore
@_spi(FastMLXExactMTP) import MLXLLM
import MLXLMCommon

public enum Qwen35ExactMTPRuntimeAdmissionError: Error, Equatable, Sendable {
    case vendoredLockDrift
    case bindingDrift(field: String)
}

public enum Qwen35ExactMTPRuntimeSelection: Equatable, Sendable {
    case qwen35_9BDepth1
    case qwen38_27BMXFP8Depth1
}

/// Application composition boundary for explicitly selected, reviewed Qwen3 depth-one MTP pairs.
///
/// The vendored factory owns exact revision resolution and reconstructs artifact evidence from the
/// resolved directories. This wrapper makes the project's independent `HarnessCore` preflight a
/// mandatory second authorization before either model is constructed or any weights are loaded.
public enum Qwen35ExactMTPRuntimeFactory {
    public static func loadDepth1Pair(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending Qwen35ExactMTPLoadedPair {
        try await loadDepth1Pair(
            selection: .qwen35_9BDepth1,
            from: downloader,
            using: tokenizerLoader,
            progressHandler: progressHandler)
    }

    public static func loadDepth1Pair(
        selection: Qwen35ExactMTPRuntimeSelection,
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending Qwen35ExactMTPLoadedPair {
        try await Qwen35ExactMTPFactory.loadDepth1Pair(
            selection: vendoredSelection(selection),
            from: downloader,
            using: tokenizerLoader,
            authorizePreflight: { evidence in
                try authorize(evidence, selection: selection)
            },
            progressHandler: progressHandler)
    }

    /// Bridges the vendored binding carried by an admitted loaded pair into the independent
    /// HarnessCore admission domain. Callers must derive serving admission from the pair itself;
    /// accepting a separately supplied binding could authorize a different target/drafter pair.
    package static func servingBinding(
        for pair: borrowing Qwen35ExactMTPLoadedPair
    ) throws -> QwenMTPArtifactBinding {
        try validateKnownLockParity()
        let vendored = pair.binding
        let harness = QwenMTPArtifactBinding(
            targetModelID: vendored.targetModelID,
            drafterModelID: vendored.drafterModelID,
            targetRevision: vendored.targetRevision,
            drafterRevision: vendored.drafterRevision,
            sourceRevision: vendored.sourceRevision,
            architecture: harnessArchitecture(vendored.architecture),
            runtimeBlockSize: vendored.runtimeBlockSize,
            maximumAcceptedDraftTokens: vendored.maximumAcceptedDraftTokens)
        try requireEquivalentBinding(vendored: vendored, harness: harness)
        return harness
    }

    package static func validateKnownLockParity() throws {
        try validateKnownLockParity(selection: .qwen35_9BDepth1)
    }

    package static func validateKnownLockParity(
        selection: Qwen35ExactMTPRuntimeSelection
    ) throws {
        guard harnessLock(vendoredLock(selection)) == harnessLock(selection) else {
            throw Qwen35ExactMTPRuntimeAdmissionError.vendoredLockDrift
        }
    }

    package static func validateSelectedVendoredLock(
        _ lock: Qwen35ExactMTPArtifactLock,
        selection: Qwen35ExactMTPRuntimeSelection
    ) throws {
        guard lock == vendoredLock(selection) else {
            throw Qwen35ExactMTPRuntimeAdmissionError.vendoredLockDrift
        }
    }

    package static func authorize(
        _ evidence: Qwen35ExactMTPPreflightEvidence,
        selection: Qwen35ExactMTPRuntimeSelection = .qwen35_9BDepth1
    ) throws {
        try validateSelectedVendoredLock(evidence.lock, selection: selection)
        try validateKnownLockParity(selection: selection)

        let binding = try QwenMTPArtifactPreflight.validate(
            lock: harnessLock(selection),
            target: harnessCandidate(evidence.target),
            drafter: harnessCandidate(evidence.drafter))
        try requireEquivalentBinding(vendored: evidence.binding, harness: binding)
    }

    private static func vendoredSelection(
        _ selection: Qwen35ExactMTPRuntimeSelection
    ) -> Qwen35ExactMTPArtifactSelection {
        switch selection {
        case .qwen35_9BDepth1:
            .qwen35_9BDepth1
        case .qwen38_27BMXFP8Depth1:
            .qwen38_27BMXFP8Depth1
        }
    }

    private static func vendoredLock(
        _ selection: Qwen35ExactMTPRuntimeSelection
    ) -> Qwen35ExactMTPArtifactLock {
        switch selection {
        case .qwen35_9BDepth1:
            Qwen35ExactMTPKnownArtifactLocks.qwen35_9BDepth1
        case .qwen38_27BMXFP8Depth1:
            Qwen35ExactMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
        }
    }

    private static func harnessLock(
        _ selection: Qwen35ExactMTPRuntimeSelection
    ) -> QwenMTPArtifactLock {
        switch selection {
        case .qwen35_9BDepth1:
            QwenMTPKnownArtifactLocks.qwen35_9BDepth1
        case .qwen38_27BMXFP8Depth1:
            QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1
        }
    }

    private static func harnessLock(
        _ lock: Qwen35ExactMTPArtifactLock
    ) -> QwenMTPArtifactLock {
        QwenMTPArtifactLock(
            sourceRevision: lock.sourceRevision,
            targetIdentity: harnessIdentity(lock.target),
            drafterIdentity: harnessIdentity(lock.drafter),
            architecture: harnessArchitecture(lock.architecture),
            targetQuantization: harnessQuantization(lock.targetQuantization),
            drafterQuantization: harnessQuantization(lock.drafterQuantization),
            drafterTensors: lock.drafterTensors.map(harnessTensor))
    }

    private static func harnessCandidate(
        _ candidate: Qwen35ExactMTPArtifactCandidate
    ) -> QwenMTPArtifactCandidate {
        QwenMTPArtifactCandidate(
            identity: harnessIdentity(candidate.identity),
            configJSON: candidate.configJSON,
            tensors: candidate.tensors.map(harnessTensor))
    }

    private static func harnessIdentity(
        _ identity: Qwen35ExactMTPArtifactIdentity
    ) -> QwenMTPArtifactIdentity {
        QwenMTPArtifactIdentity(
            modelID: identity.modelID,
            revision: identity.revision,
            configSHA256: identity.configSHA256,
            tokenizerSHA256: identity.tokenizerSHA256,
            tensorManifestSHA256: identity.tensorManifestSHA256)
    }

    private static func harnessArchitecture(
        _ architecture: Qwen35ExactMTPArchitecture
    ) -> QwenMTPArchitecture {
        QwenMTPArchitecture(
            hiddenSize: architecture.hiddenSize,
            intermediateSize: architecture.intermediateSize,
            vocabularySize: architecture.vocabularySize,
            targetLayerCount: architecture.targetLayerCount,
            fullAttentionInterval: architecture.fullAttentionInterval,
            attentionHeadCount: architecture.attentionHeadCount,
            keyValueHeadCount: architecture.keyValueHeadCount,
            headDimension: architecture.headDimension,
            usesDedicatedMTPEmbeddings: architecture.usesDedicatedMTPEmbeddings)
    }

    private static func harnessQuantization(
        _ quantization: Qwen35ExactMTPQuantization
    ) -> QwenMTPQuantization {
        QwenMTPQuantization(
            bits: quantization.bits,
            groupSize: quantization.groupSize,
            mode: quantization.mode)
    }

    private static func harnessTensor(
        _ tensor: Qwen35ExactMTPTensorDescriptor
    ) -> QwenMTPTensorDescriptor {
        QwenMTPTensorDescriptor(name: tensor.name, shape: tensor.shape, dtype: tensor.dtype)
    }

    private static func requireEquivalentBinding(
        vendored: Qwen35ExactMTPBinding,
        harness: QwenMTPArtifactBinding
    ) throws {
        for (field, matches) in [
            ("targetModelID", vendored.targetModelID == harness.targetModelID),
            ("drafterModelID", vendored.drafterModelID == harness.drafterModelID),
            ("targetRevision", vendored.targetRevision == harness.targetRevision),
            ("drafterRevision", vendored.drafterRevision == harness.drafterRevision),
            ("sourceRevision", vendored.sourceRevision == harness.sourceRevision),
            ("architecture", harnessArchitecture(vendored.architecture) == harness.architecture),
            ("mtpDepth", vendored.architecture.mtpDepth == 1),
            ("runtimeBlockSize", vendored.runtimeBlockSize == harness.runtimeBlockSize),
            ("maximumAcceptedDraftTokens",
                vendored.maximumAcceptedDraftTokens == harness.maximumAcceptedDraftTokens),
        ] where !matches {
            throw Qwen35ExactMTPRuntimeAdmissionError.bindingDrift(field: field)
        }
    }
}
