import Foundation

/// The cache invariant owned by one incremental solo-PLD request.
///
/// - `awaitingFirstToken`: prompt KV is committed; the first greedy token is staged but has not
///   been emitted or forwarded.
/// - `pipelined`: every emitted token is committed in KV and the next greedy token is staged.
/// - `speculative`: the last emitted PLD bonus token is staged but is not committed in KV.
public enum IncrementalPLDCacheInvariant: Sendable, Equatable {
    case awaitingFirstToken
    case pipelined
    case speculative
}

/// One pure planning decision for an actor-confined incremental PLD round.
public enum IncrementalPLDRoundPlan: Sendable, Equatable {
    case plainFromPipeline
    case fallbackFromSpeculative(lastEmittedToken: Int)
    case verifyPipelined(draft: [Int])
    case verifySpeculative(lastEmittedToken: Int, draft: [Int])
}

/// Pure result of accepting one target-model verification.
public struct IncrementalPLDCommit: Sendable, Equatable {
    public let acceptedDraftTokens: Int
    public let bonusToken: Int
    public let emittedTokens: [Int]

    public init(
        acceptedDraftTokens: Int,
        bonusToken: Int,
        emittedTokens: [Int]
    ) {
        self.acceptedDraftTokens = acceptedDraftTokens
        self.bonusToken = bonusToken
        self.emittedTokens = emittedTokens
    }
}

public enum IncrementalPLDSessionError: Error, Sendable, Equatable {
    case emptyPrompt
    case invalidConfiguration
    case invalidToken(Int)
    case invalidTransition
    case stalePlan
    case missingPrefetchedToken
    case unexpectedPrefetchedToken
    case invalidVerificationWidth(expected: Int, actual: Int)
    case invalidDraftWidth(maximum: Int, actual: Int)
    case invalidFixedDraftWidth(Int)
    case invalidDrainToken(expected: Int, actual: Int)
}

/// MLX-free session state for one incremental prompt-lookup decode request.
///
/// The model runtime owns arrays and cache mutation. This value owns only the bounded prompt
/// lookup context, gate, cache-invariant transition, accept walk, and engagement counts so the
/// risky transition rules remain testable off-box.
public struct IncrementalPLDSession: Sendable {
    private let drafter: any SpecDrafter
    private let maxDraft: Int
    private let lookback: Int
    private var gate: PLDGate

    public private(set) var context: [Int]
    public private(set) var cacheInvariant: IncrementalPLDCacheInvariant
    public private(set) var draftedTokens = 0
    public private(set) var acceptedDraftTokens = 0
    public private(set) var verificationRounds = 0
    public private(set) var fallbackRounds = 0

    public init(
        promptTokens: [Int],
        drafter: any SpecDrafter = PromptLookupDrafter(ngram: 3),
        maxDraft: Int = 8,
        lookback: Int = 4096,
        gate: PLDGate = PLDGate()
    ) throws {
        guard !promptTokens.isEmpty else {
            throw IncrementalPLDSessionError.emptyPrompt
        }
        guard maxDraft > 0, lookback > 0 else {
            throw IncrementalPLDSessionError.invalidConfiguration
        }
        if let invalid = promptTokens.first(where: { $0 < 0 }) {
            throw IncrementalPLDSessionError.invalidToken(invalid)
        }
        self.drafter = drafter
        self.maxDraft = maxDraft
        self.lookback = lookback
        self.gate = gate
        self.context = promptTokens
        self.cacheInvariant = .awaitingFirstToken
    }

    /// Commit the first greedy token after the runtime forwards it and arms the ordinary
    /// submit-first pipeline.
    public mutating func recordFirstToken(_ token: Int) throws {
        guard cacheInvariant == .awaitingFirstToken else {
            throw IncrementalPLDSessionError.invalidTransition
        }
        try validate(token)
        context.append(token)
        cacheInvariant = .pipelined
    }

    /// Plan the next bounded round without mutating the session.
    public func makeRoundPlan(
        maximumDraftTokens: Int? = nil,
        fixedDraftWidth: Int? = nil
    ) throws -> IncrementalPLDRoundPlan {
        guard cacheInvariant != .awaitingFirstToken else {
            throw IncrementalPLDSessionError.invalidTransition
        }
        guard let last = context.last else {
            throw IncrementalPLDSessionError.emptyPrompt
        }
        let requestedMaximum = maximumDraftTokens ?? maxDraft
        guard requestedMaximum >= 0 else {
            throw IncrementalPLDSessionError.invalidConfiguration
        }
        let draftLimit = min(maxDraft, requestedMaximum)
        if let fixedDraftWidth {
            guard fixedDraftWidth > 0, fixedDraftWidth <= draftLimit else {
                throw IncrementalPLDSessionError.invalidFixedDraftWidth(
                    fixedDraftWidth)
            }
        }

        var draft: [Int]
        if gate.isEnabled, draftLimit > 0 {
            draft = drafter.propose(
                context: Array(context.suffix(lookback)),
                maxDraft: draftLimit)
            guard draft.count <= draftLimit else {
                throw IncrementalPLDSessionError.invalidDraftWidth(
                    maximum: draftLimit,
                    actual: draft.count)
            }
            if let invalid = draft.first(where: { $0 < 0 }) {
                throw IncrementalPLDSessionError.invalidToken(invalid)
            }
            if let fixedDraftWidth, !draft.isEmpty, draft.count < fixedDraftWidth {
                draft += Array(
                    repeating: draft.last!,
                    count: fixedDraftWidth - draft.count)
            }
        } else {
            draft = []
        }

        switch (cacheInvariant, draft.isEmpty) {
        case (.pipelined, true):
            return .plainFromPipeline
        case (.speculative, true):
            return .fallbackFromSpeculative(lastEmittedToken: last)
        case (.pipelined, false):
            return .verifyPipelined(draft: draft)
        case (.speculative, false):
            return .verifySpeculative(
                lastEmittedToken: last,
                draft: draft)
        case (.awaitingFirstToken, _):
            throw IncrementalPLDSessionError.invalidTransition
        }
    }

    /// Commit target-model picks for the exact plan returned by `makeRoundPlan()`.
    ///
    /// Pipelined verification receives one prefetched target token plus one target pick after
    /// each forwarded draft token. Speculative verification forwards `[last] + draft`, so it
    /// receives `draft.count + 1` target picks and no prefetched token.
    public mutating func commitVerification(
        _ plan: IncrementalPLDRoundPlan,
        prefetchedToken: Int?,
        verifyArgmax: [Int]
    ) throws -> IncrementalPLDCommit {
        if let invalid = verifyArgmax.first(where: { $0 < 0 }) {
            throw IncrementalPLDSessionError.invalidToken(invalid)
        }

        let draft: [Int]
        let accepted: Int
        let bonus: Int
        switch plan {
        case .verifyPipelined(let proposed):
            guard cacheInvariant == .pipelined else {
                throw IncrementalPLDSessionError.stalePlan
            }
            guard let prefetchedToken else {
                throw IncrementalPLDSessionError.missingPrefetchedToken
            }
            try validate(prefetchedToken)
            guard verifyArgmax.count == proposed.count else {
                throw IncrementalPLDSessionError.invalidVerificationWidth(
                    expected: proposed.count,
                    actual: verifyArgmax.count)
            }
            draft = proposed
            (accepted, bonus) = SpecAccept.walk(
                draft: proposed,
                prefetched: prefetchedToken,
                verifyArgmaxAfterDraft: verifyArgmax)

        case .verifySpeculative(let lastEmittedToken, let proposed):
            guard cacheInvariant == .speculative,
                context.last == lastEmittedToken
            else {
                throw IncrementalPLDSessionError.stalePlan
            }
            guard prefetchedToken == nil else {
                throw IncrementalPLDSessionError.unexpectedPrefetchedToken
            }
            let expected = proposed.count + 1
            guard verifyArgmax.count == expected else {
                throw IncrementalPLDSessionError.invalidVerificationWidth(
                    expected: expected,
                    actual: verifyArgmax.count)
            }
            draft = proposed
            (accepted, bonus) = SpecAccept.walk(
                draft: proposed,
                verifyArgmax: verifyArgmax)

        case .plainFromPipeline, .fallbackFromSpeculative:
            throw IncrementalPLDSessionError.invalidTransition
        }

        try validate(bonus)
        let emitted = Array(draft.prefix(accepted)) + [bonus]
        context.append(contentsOf: emitted)
        draftedTokens += draft.count
        acceptedDraftTokens += accepted
        verificationRounds += 1
        gate.record(accepted: accepted)
        cacheInvariant = .speculative
        return IncrementalPLDCommit(
            acceptedDraftTokens: accepted,
            bonusToken: bonus,
            emittedTokens: emitted)
    }

    /// Commit the exact plain target token emitted by a cold or gate-disabled round.
    public mutating func commitFallback(emittedToken: Int) throws {
        try commitFallback(try makeRoundPlan(), emittedToken: emittedToken)
    }

    /// Commit a previously planned plain fallback without asking the drafter to run again.
    public mutating func commitFallback(
        _ plan: IncrementalPLDRoundPlan,
        emittedToken: Int
    ) throws {
        switch (cacheInvariant, plan) {
        case (.pipelined, .plainFromPipeline):
            break
        case (
            .speculative,
            .fallbackFromSpeculative(let lastEmittedToken)
        ) where context.last == lastEmittedToken:
            break
        case (.pipelined, .fallbackFromSpeculative),
            (.speculative, .plainFromPipeline):
            throw IncrementalPLDSessionError.stalePlan
        case (_, .verifyPipelined), (_, .verifySpeculative),
            (.awaitingFirstToken, _):
            throw IncrementalPLDSessionError.invalidTransition
        default:
            throw IncrementalPLDSessionError.stalePlan
        }
        try validate(emittedToken)
        context.append(emittedToken)
        fallbackRounds += 1
        gate.record(accepted: 0)
        cacheInvariant = .pipelined
    }

    /// Record one exact greedy token emitted by a shared batch or an explicitly non-PLD solo
    /// step. This keeps prompt-lookup context current if the request later returns to solo PLD.
    public mutating func recordCanonicalToken(_ token: Int) throws {
        switch cacheInvariant {
        case .awaitingFirstToken:
            try recordFirstToken(token)
        case .pipelined:
            try validate(token)
            context.append(token)
        case .speculative:
            throw IncrementalPLDSessionError.invalidTransition
        }
    }

    /// A speculative drain forwards the already-emitted bonus token without emitting another
    /// token. The runtime then has the canonical pipelined cache invariant needed for batching.
    public mutating func recordCanonicalDrain(
        forwardedToken: Int
    ) throws {
        guard cacheInvariant == .speculative else {
            throw IncrementalPLDSessionError.invalidTransition
        }
        try validate(forwardedToken)
        guard let expected = context.last, expected == forwardedToken else {
            throw IncrementalPLDSessionError.invalidDrainToken(
                expected: context.last ?? -1,
                actual: forwardedToken)
        }
        cacheInvariant = .pipelined
    }

    private func validate(_ token: Int) throws {
        guard token >= 0 else {
            throw IncrementalPLDSessionError.invalidToken(token)
        }
    }
}
