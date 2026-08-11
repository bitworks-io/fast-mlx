import Foundation

/// Stopping rules for a multi-token emission. A verify forward emits `accepted + 1` tokens at
/// once, but the run's limits — the maxTokens budget and the terminal eos — were defined by the
/// plain greedy loop (`while tokens.count < maxTokens && tok != eos`), which applies them one
/// token at a time. Byte-identical output (the headline property) requires the batched path to
/// stop at EXACTLY the same token the plain loop would have stopped at, so this helper replays
/// those rules over the batch: cut at the remaining budget first, then at the first eos
/// (inclusive — the baseline stream includes its terminal eos).
public enum SpecEmit {
    /// - Parameters:
    ///   - emitted: The tokens one verify step produced, in order (accepted drafts + bonus).
    ///   - alreadyEmitted: How many tokens the run has produced before this batch.
    ///   - maxTokens: The run's total token budget.
    ///   - eos: The terminal token id.
    /// - Returns: The prefix of `emitted` the plain greedy loop would have produced, and
    ///   whether generation is finished (budget filled or eos emitted).
    public static func trim(
        emitted: [Int], alreadyEmitted: Int, maxTokens: Int, eos: Int
    ) -> (emit: [Int], done: Bool) {
        let budget = max(0, maxTokens - alreadyEmitted)
        var emit = Array(emitted.prefix(budget))
        var done = emit.count == budget
        if let eosIndex = emit.firstIndex(of: eos) {
            emit = Array(emit[...eosIndex])
            done = true
        }
        return (emit, done)
    }
}
