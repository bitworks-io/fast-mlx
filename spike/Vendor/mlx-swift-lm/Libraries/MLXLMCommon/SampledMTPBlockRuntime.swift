// Copyright © 2026 Apple Inc.

import MLX

/// The caller-owned result of one sampled MTP verification block.
///
/// MLXLMCommon deliberately does not own probability-ratio acceptance policy.
/// A higher layer supplies that policy through ``SampledMTPBlockRuntimeDeciding``
/// and the iterator owns only the cache transaction described by this value.
public struct SampledMTPBlockRuntimeDecision: Sendable, Equatable {
    public let outputTokens: [Int]
    public let acceptedDraftCount: Int

    public init(outputTokens: [Int], acceptedDraftCount: Int) {
        self.outputTokens = outputTokens
        self.acceptedDraftCount = acceptedDraftCount
    }
}

/// Explicit, default-off sampled-block policy seam for MTP iteration.
///
/// The proposal sampler is distinct from the ordinary target sampler so draft
/// draws cannot perturb the target stream. Implementations may capture the
/// proposal distributions in that sampler, then combine them with the target
/// logits supplied to ``decide``. Throwing fails the current block closed; the
/// iterator rolls back every draft and continues on the ordinary target path.
public protocol SampledMTPBlockRuntimeDeciding: AnyObject {
    var proposalSampler: any LogitSampler { get }

    /// Return `true` only when the provider exactly implements the requested
    /// sampling policy. The iterator never infers compatibility.
    func supports(parameters: GenerateParameters) -> Bool

    /// Select the accepted prefix plus one residual correction or target bonus.
    /// `targetLogits` contains one row for each proposed token and
    /// `bonusTargetLogits` is the row after the complete proposal block.
    func decide(
        proposedTokens: [Int],
        targetLogits: [MLXArray],
        bonusTargetLogits: MLXArray
    ) throws -> SampledMTPBlockRuntimeDecision
}
