import HarnessCore
import MLX
import MLXLMCommon

public struct SampledMTPBlockRuntimeDrawPlan: Sendable, Equatable {
    public let proposalUniforms: [Double]
    public let acceptanceUniforms: [Double]
    public let terminalDraw: SampledMTPBlockTerminalDraw

    public init(
        proposalUniforms: [Double],
        acceptanceUniforms: [Double],
        terminalDraw: SampledMTPBlockTerminalDraw
    ) {
        self.proposalUniforms = proposalUniforms
        self.acceptanceUniforms = acceptanceUniforms
        self.terminalDraw = terminalDraw
    }
}

public enum SampledMTPBlockRuntimeBridgeError: Error, Sendable, Equatable {
    case missingPlan
    case proposalCaptureFailed
    case proposalCountMismatch
    case proposalTokenMismatch
    case targetRowCountMismatch
}

/// Default-off adapter from the MLX iterator callback to HarnessCore's
/// accepted ordered probability-ratio/residual-correction contract.
///
/// This first bounded runtime seam intentionally accepts only unfiltered,
/// temperature-one sampling with no penalties. Serving does not construct it.
public final class SampledMTPBlockRuntimeBridge: SampledMTPBlockRuntimeDeciding {
    private let plans: [SampledMTPBlockRuntimeDrawPlan]
    private let recorder: FixedUniformProposalSampler
    private var planIndex = 0

    public var proposalSampler: any LogitSampler { recorder }

    public init(plans: [SampledMTPBlockRuntimeDrawPlan]) {
        self.plans = plans
        self.recorder = FixedUniformProposalSampler(
            uniforms: plans.flatMap(\.proposalUniforms))
    }

    public func supports(parameters: GenerateParameters) -> Bool {
        !plans.isEmpty
            && parameters.temperature == 1
            && parameters.topP == 1
            && parameters.topK == 0
            && parameters.minP == 0
            && parameters.repetitionPenalty == nil
            && parameters.presencePenalty == nil
            && parameters.frequencyPenalty == nil
    }

    public func decide(
        proposedTokens: [Int],
        targetLogits: [MLXArray],
        bonusTargetLogits: MLXArray
    ) throws -> SampledMTPBlockRuntimeDecision {
        guard plans.indices.contains(planIndex) else {
            throw SampledMTPBlockRuntimeBridgeError.missingPlan
        }
        let plan = plans[planIndex]
        guard proposedTokens.count == plan.proposalUniforms.count else {
            throw SampledMTPBlockRuntimeBridgeError.proposalCountMismatch
        }
        guard targetLogits.count == proposedTokens.count else {
            throw SampledMTPBlockRuntimeBridgeError.targetRowCountMismatch
        }
        let proposals = try recorder.consume(count: proposedTokens.count)
        guard proposals.map(\.token) == proposedTokens else {
            throw SampledMTPBlockRuntimeBridgeError.proposalTokenMismatch
        }

        let steps = zip(zip(proposedTokens.indices, proposedTokens), proposals).map {
            indexed, proposal in
            let (index, token) = indexed
            return SampledMTPBlockStep(
                targetDistribution: normalizedProbabilities(targetLogits[index]),
                draftDistribution: proposal.probabilities,
                proposedToken: token,
            )
        }
        let decision = try SampledMTPBlockAcceptance.decide(
            steps: steps,
            acceptanceUniforms: plan.acceptanceUniforms,
            terminalDraws: [plan.terminalDraw],
            bonusTargetDistribution: normalizedProbabilities(bonusTargetLogits))
        planIndex += 1
        return SampledMTPBlockRuntimeDecision(
            outputTokens: decision.tokens,
            acceptedDraftCount: decision.acceptedDraftCount)
    }
}

private struct CapturedProposal {
    let token: Int
    let probabilities: [Double]
}

private final class FixedUniformProposalSampler: LogitSampler {
    private let uniforms: [Double]
    private var uniformIndex = 0
    private var consumedCaptureCount = 0
    private var captures = [CapturedProposal]()
    private var failed = false

    init(uniforms: [Double]) {
        self.uniforms = uniforms
    }

    func sample(logits: MLXArray) -> MLXArray {
        guard uniforms.indices.contains(uniformIndex) else {
            failed = true
            return MLXArray([Int32(0)])
        }
        let uniform = uniforms[uniformIndex]
        uniformIndex += 1
        let probabilities = normalizedProbabilities(logits)
        guard uniform.isFinite, (0 ..< 1).contains(uniform),
            let token = categoricalSample(probabilities, uniform: uniform)
        else {
            failed = true
            return MLXArray([Int32(0)])
        }
        captures.append(CapturedProposal(token: token, probabilities: probabilities))
        return MLXArray([Int32(token)])
    }

    func consume(count: Int) throws -> [CapturedProposal] {
        guard !failed, count >= 0,
            consumedCaptureCount + count <= captures.count
        else {
            throw SampledMTPBlockRuntimeBridgeError.proposalCaptureFailed
        }
        let result = Array(captures[consumedCaptureCount ..< consumedCaptureCount + count])
        consumedCaptureCount += count
        return result
    }
}

private func normalizedProbabilities(_ logits: MLXArray) -> [Double] {
    let probabilities = softmax(logits.asType(.float32), axis: -1).flattened()
    eval(probabilities)
    let raw = probabilities.asArray(Float.self).map(Double.init)
    let sum = raw.reduce(0, +)
    guard sum.isFinite, sum > 0 else { return [] }
    var normalized = raw.map { $0 / sum }
    guard let largest = normalized.indices.max(by: {
        normalized[$0] < normalized[$1]
    }) else { return normalized }
    normalized[largest] += 1 - normalized.reduce(0, +)
    return normalized
}

private func categoricalSample(_ probabilities: [Double], uniform: Double) -> Int? {
    guard !probabilities.isEmpty else { return nil }
    var cumulative = 0.0
    for (index, probability) in probabilities.enumerated() {
        cumulative += probability
        if uniform < cumulative { return index }
    }
    return probabilities.indices.last
}
