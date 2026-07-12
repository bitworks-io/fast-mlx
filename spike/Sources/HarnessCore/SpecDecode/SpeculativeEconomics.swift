import Foundation

/// Aggregate speculative-decoding counters with each metric's denominator made explicit.
///
/// Proposal acceptance (`accepted / proposed`) measures drafter accuracy. Accepted drafts per
/// verify round (`accepted / rounds`) measures economic yield. They answer different questions
/// and must not be substituted for one another. Published `acceptance_length` commonly includes
/// the target's always-emitted correction/bonus token, so it is one larger than round yield.
public struct SpeculativeAcceptanceSummary: Sendable, Equatable {
    public let proposedDraftTokens: Int
    public let acceptedDraftTokens: Int
    public let verifyRounds: Int

    public init(
        proposedDraftTokens: Int,
        acceptedDraftTokens: Int,
        verifyRounds: Int
    ) {
        precondition(proposedDraftTokens >= 0, "proposedDraftTokens must be nonnegative")
        precondition(acceptedDraftTokens >= 0, "acceptedDraftTokens must be nonnegative")
        precondition(verifyRounds >= 0, "verifyRounds must be nonnegative")
        precondition(
            acceptedDraftTokens <= proposedDraftTokens,
            "acceptedDraftTokens cannot exceed proposedDraftTokens")
        precondition(
            proposedDraftTokens == 0 || verifyRounds > 0,
            "proposed draft work must belong to at least one verify round")
        self.proposedDraftTokens = proposedDraftTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.verifyRounds = verifyRounds
    }

    /// Drafter accuracy across proposed token positions (`accepted / proposed`).
    public var proposalAcceptanceRate: Double? {
        proposedDraftTokens > 0
            ? Double(acceptedDraftTokens) / Double(proposedDraftTokens)
            : nil
    }

    /// Economic yield (`accepted draft tokens / target verify rounds`).
    public var acceptedDraftTokensPerRound: Double? {
        verifyRounds > 0
            ? Double(acceptedDraftTokens) / Double(verifyRounds)
            : nil
    }

    /// Yield including the target correction/bonus emitted by every greedy verify round.
    public var inclusiveAcceptanceLength: Double? {
        acceptedDraftTokensPerRound.map { 1 + $0 }
    }

    /// Converts a published inclusive acceptance length back to accepted draft tokens per round.
    public static func acceptedDraftTokensPerRound(
        inclusiveAcceptanceLength: Double
    ) -> Double {
        precondition(
            inclusiveAcceptanceLength.isFinite && inclusiveAcceptanceLength >= 1,
            "inclusiveAcceptanceLength must be finite and at least one")
        return inclusiveAcceptanceLength - 1
    }
}

/// Mean wall time of the mutually exclusive phases in one speculative verify round.
public struct SpeculativePhaseTiming: Sendable, Equatable {
    public let draftSeconds: Double
    public let verifySeconds: Double
    public let commitSeconds: Double

    public init(draftSeconds: Double, verifySeconds: Double, commitSeconds: Double) {
        precondition(
            draftSeconds.isFinite && draftSeconds >= 0,
            "draftSeconds must be finite and nonnegative")
        precondition(
            verifySeconds.isFinite && verifySeconds >= 0,
            "verifySeconds must be finite and nonnegative")
        precondition(
            commitSeconds.isFinite && commitSeconds >= 0,
            "commitSeconds must be finite and nonnegative")
        self.draftSeconds = draftSeconds
        self.verifySeconds = verifySeconds
        self.commitSeconds = commitSeconds
    }

    public var totalSeconds: Double {
        draftSeconds + verifySeconds + commitSeconds
    }
}

/// Pairing-specific speculative economics derived from measured phase costs.
///
/// This projection explains a result; observed end-to-end throughput remains the promotion gate.
public struct SpeculativeEconomics: Sendable, Equatable {
    public let baselineTokenSeconds: Double
    public let acceptedDraftTokensPerRound: Double
    public let timing: SpeculativePhaseTiming

    public init(
        baselineTokenSeconds: Double,
        acceptedDraftTokensPerRound: Double,
        timing: SpeculativePhaseTiming
    ) {
        precondition(
            baselineTokenSeconds.isFinite && baselineTokenSeconds > 0,
            "baselineTokenSeconds must be finite and positive")
        precondition(
            acceptedDraftTokensPerRound.isFinite && acceptedDraftTokensPerRound >= 0,
            "acceptedDraftTokensPerRound must be finite and nonnegative")
        self.baselineTokenSeconds = baselineTokenSeconds
        self.acceptedDraftTokensPerRound = acceptedDraftTokensPerRound
        self.timing = timing
    }

    public var speculativeRoundSeconds: Double {
        timing.totalSeconds
    }

    public var roundCostRatio: Double {
        speculativeRoundSeconds / baselineTokenSeconds
    }

    /// Draft yield required for the speculative round to emit as quickly as base decoding.
    public var breakEvenAcceptedDraftTokensPerRound: Double {
        roundCostRatio - 1
    }

    public var emittedTokensPerRound: Double {
        acceptedDraftTokensPerRound + 1
    }

    public var projectedSpeedup: Double {
        emittedTokensPerRound / roundCostRatio
    }
}

/// Observed speculative throughput normalized to the same target pairing's measured base loop.
public struct SpeculativeThroughputComparison: Sendable, Equatable {
    public let baselineTokensPerSecond: Double
    public let speculativeTokensPerSecond: Double

    public init(baselineTokensPerSecond: Double, speculativeTokensPerSecond: Double) {
        precondition(
            baselineTokensPerSecond.isFinite && baselineTokensPerSecond > 0,
            "baselineTokensPerSecond must be finite and positive")
        precondition(
            speculativeTokensPerSecond.isFinite && speculativeTokensPerSecond > 0,
            "speculativeTokensPerSecond must be finite and positive")
        self.baselineTokensPerSecond = baselineTokensPerSecond
        self.speculativeTokensPerSecond = speculativeTokensPerSecond
    }

    public var speedup: Double {
        speculativeTokensPerSecond / baselineTokensPerSecond
    }

    public var deltaPercent: Double {
        (speedup - 1) * 100
    }
}
