import Foundation

/// Proposes continuation tokens for speculative decoding, given the token context generated so
/// far. An empty proposal means "no draft available" — the engine falls back to a normal
/// single-token step. Pure logic: no notion of the model, the KV cache, or verification.
public protocol SpecDrafter: Sendable {
    /// Propose up to `maxDraft` continuation token ids following `context`.
    func propose(context: [Int], maxDraft: Int) -> [Int]
}

/// Prompt-lookup decoding (PLD): drafts by finding the most recent EARLIER occurrence of the
/// current trailing `ngram` tokens elsewhere in `context`, and proposing whatever followed that
/// occurrence. This exploits the common case where the model is about to repeat text it (or the
/// prompt) already produced — e.g. copying, quoting, or restating — without needing a second,
/// smaller model to draft from.
public struct PromptLookupDrafter: SpecDrafter {
    /// The suffix length used to find a prior match. Longer n-grams match less often but more
    /// reliably predict what follows; shorter n-grams match more often but more speculatively.
    public let ngram: Int

    public init(ngram: Int = 3) {
        precondition(ngram > 0, "ngram must be positive")
        self.ngram = ngram
    }

    public func propose(context: [Int], maxDraft: Int) -> [Int] {
        guard maxDraft > 0, context.count >= ngram else { return [] }
        let suffix = context.suffix(ngram)
        let suffixStart = context.count - ngram

        // Scan earlier candidate end-indices from most recent to oldest so the FIRST match found
        // is the most recent occurrence. A candidate match at end-index `i` covers
        // context[i-ngram+1...i] and must lie strictly before the current suffix (i < count - 1,
        // and the matched window must not overlap the live suffix itself).
        var i = suffixStart - 1
        while i >= ngram - 1 {
            let candidateStart = i - ngram + 1
            if context[candidateStart...i].elementsEqual(suffix) {
                let continuationStart = i + 1
                guard continuationStart < context.count else { return [] }
                let continuationEnd = min(continuationStart + maxDraft, context.count)
                return Array(context[continuationStart..<continuationEnd])
            }
            i -= 1
        }
        return []
    }
}
