import Foundation
import HarnessCore

/// Configuration for the engine's speculative-decoding path (PLD first). The pure framework
/// pieces — the drafter, the accept-walk, the gate — live in `HarnessCore/SpecDecode`; this
/// binds them to `CompiledMLXDecoder`'s verify-forward + KV-rollback loop.
public struct SpecDecodeConfig: Sendable {
    /// Proposes continuation tokens from the context (PLD: n-gram prompt lookup).
    public var drafter: any SpecDrafter
    /// Max drafted tokens per verify forward (K).
    public var maxDraft: Int
    /// Self-managing enable/disable gate fed with accepted-per-step.
    public var gate: PLDGate
    /// The drafter only scans this many trailing context tokens. The PLD backward scan is
    /// O(scanned) per call; unbounded it would be O(context) per step — O(n²) per request
    /// at 32K context. 4K covers the recent repetition PLD exploits.
    public var lookback: Int
    /// Run the verify forward through a separate fixed-K compiled step (drafts padded to K
    /// so the trace replays without retracing). `false` = uncompiled verify forward
    /// (per-call graph construction; correctness identical).
    public var compiledVerify: Bool

    public init(
        drafter: any SpecDrafter = PromptLookupDrafter(ngram: 3),
        maxDraft: Int = 8,
        gate: PLDGate = PLDGate(),
        lookback: Int = 4096,
        compiledVerify: Bool = false
    ) {
        precondition(maxDraft > 0, "maxDraft must be positive")
        precondition(lookback > 0, "lookback must be positive")
        self.drafter = drafter
        self.maxDraft = maxDraft
        self.gate = gate
        self.lookback = lookback
        self.compiledVerify = compiledVerify
    }
}

/// Engagement + acceptance telemetry from one speculative-decoding run — the evidence the
/// equivalence gate (drafting actually happened) and the measurement verdict read.
public struct SpecDecodeStats: Sendable {
    /// Tokens proposed by the drafter and spent in verify forwards (includes fixed-K padding
    /// on the compiled-verify path — every counted token cost a verify position).
    public var drafted = 0
    /// Drafted tokens the accept-walk confirmed.
    public var accepted = 0
    /// Steps that ran a verify forward over a non-empty draft.
    public var verifySteps = 0
    /// Steps that ran the plain single-token compiled step (empty draft or gate disabled).
    public var normalSteps = 0
    /// Subset of `normalSteps` taken while the gate had PLD disabled.
    public var gateDisabledSteps = 0

    public init() {}

    /// accepted/drafted across the run; nil when nothing was drafted.
    public var acceptanceRate: Double? {
        drafted > 0 ? Double(accepted) / Double(drafted) : nil
    }

    /// Mean accepted drafts per verify forward (the yield the gate manages on).
    public var meanAcceptedPerVerify: Double {
        verifySteps > 0 ? Double(accepted) / Double(verifySteps) : 0
    }
}
