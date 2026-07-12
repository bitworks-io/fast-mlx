import Foundation

/// The greedy accept-walk: decides how many drafted tokens a single verify forward pass confirms.
///
/// This is the property that makes speculative decoding SAFE for greedy decoding: the emitted
/// tokens are byte-for-byte identical to what plain (non-speculative) greedy decoding would have
/// produced from the same target model. Speculation only changes HOW MANY tokens one forward
/// pass emits — never WHICH tokens. A drafted token is accepted exactly while it matches the
/// target's own greedy pick at that position; the walk stops at the first mismatch, and the
/// target's own pick at the stopping position (the "bonus" token) is emitted in its place. This
/// is why a single verify forward can emit `accepted + 1` tokens "for free": the bonus token was
/// already computed as a byproduct of verifying the draft.
public enum SpecAccept {
    /// - Parameters:
    ///   - draft: The drafted token ids, in order.
    ///   - verifyArgmax: The target's greedy (argmax) pick at each drafted position, PLUS one
    ///     extra entry for the position immediately after the last draft. Must have
    ///     `draft.count + 1` entries.
    /// - Returns: `accepted`, the number of leading drafted tokens confirmed (0...draft.count),
    ///   and `bonus`, the target's own next token at the first unconfirmed position — always
    ///   emitted, since it was computed for free during verification.
    public static func walk(draft: [Int], verifyArgmax: [Int]) -> (accepted: Int, bonus: Int) {
        precondition(
            verifyArgmax.count == draft.count + 1,
            "verifyArgmax must have draft.count + 1 entries (one per draft position plus the trailing position)")
        var accepted = 0
        while accepted < draft.count && draft[accepted] == verifyArgmax[accepted] {
            accepted += 1
        }
        return (accepted, verifyArgmax[accepted])
    }

    /// Pipelined variant: the plain submit-first step has already produced the target's pick
    /// immediately after the committed context, so the verify forward only consumes `draft`.
    /// Its row `i` therefore predicts the token AFTER `draft[i]`; prepending the prefetched pick
    /// restores the same K+1 target sequence consumed by the canonical accept walk above.
    public static func walk(
        draft: [Int],
        prefetched: Int,
        verifyArgmaxAfterDraft: [Int]
    ) -> (accepted: Int, bonus: Int) {
        precondition(
            verifyArgmaxAfterDraft.count == draft.count,
            "verifyArgmaxAfterDraft must have one entry per forwarded draft token")
        return walk(draft: draft, verifyArgmax: [prefetched] + verifyArgmaxAfterDraft)
    }
}
