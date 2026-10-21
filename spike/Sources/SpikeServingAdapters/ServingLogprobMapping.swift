import Foundation
import os

import HarnessCore
import ServingCore

/// The one place that turns a generated token's raw ids (the selected token plus its top
/// alternatives, as a decoder reports them) into the wire-level `ServingTokenLogprob`. Both the
/// scalar route (`ScalarServingBackend.publishLogprob`, fed `SpikeCore.DecodedTokenLogprob`) and the
/// continuous-batch route (`ContinuousServingBackend`, fed `ContinuousBatchTokenLogprob`) call it,
/// so the two routes cannot drift in how an id becomes text.
///
/// Every id is resolved with `codec.decodeSingleToken`, which decodes each token INDEPENDENTLY of any
/// other (never through the request's incremental detokenizer). Under `ignore_eos` a model stop token
/// is a generated token; it is usually a special token whose decode can come back empty, which would
/// leave a client unable to tell that entry apart, so the generated stop token (only) falls back to
/// its literal vocabulary text (e.g. "</s>").
enum ServingLogprobMapping {
    static func served(
        tokenID: Int,
        logprob: Float,
        top: [(tokenID: Int, logprob: Float)],
        codec: any ScalarServingTextCodec,
        ignoreEOS: Bool,
        stopTokenIDs: Set<Int>
    ) -> ServingTokenLogprob {
        let candidates = top.map { candidate in
            ServingTokenLogprobCandidate(
                tokenText: codec.decodeSingleToken(candidate.tokenID),
                logprob: Double(candidate.logprob))
        }
        var generatedText = codec.decodeSingleToken(tokenID)
        if ignoreEOS, generatedText.isEmpty,
            stopTokenIDs.contains(tokenID),
            let vocabularyText = codec.vocabularyText(forTokenID: tokenID)
        {
            generatedText = vocabularyText
        }
        return ServingTokenLogprob(
            tokenText: generatedText,
            logprob: Double(logprob),
            topCandidates: candidates)
    }
}

/// A request's logprob queue on the continuous-batch route. The runtime `record`s one entry per
/// SELECTED token in selection order (from its own thread); the backend's token consumer `pop`s the
/// head once per PUBLISHED token. The coordinator may trim a row's final tokens (a stop token, the
/// output budget), so trailing entries can be left unpopped: pairing is always from the head.
final class ContinuousServingLogprobSink: ContinuousBatchRowLogprobSink {
    private struct State: Sendable {
        var entries: [ContinuousBatchTokenLogprob] = []
        var head = 0
    }

    let topLogprobs: Int
    private let state = OSAllocatedUnfairLock(initialState: State())

    init(topLogprobs: Int) {
        self.topLogprobs = topLogprobs
    }

    func record(_ logprob: ContinuousBatchTokenLogprob) {
        state.withLock { $0.entries.append(logprob) }
    }

    /// The oldest entry not yet paired with a published token, or `nil` when none is queued.
    func popFirst() -> ContinuousBatchTokenLogprob? {
        state.withLock { state in
            guard state.head < state.entries.count else { return nil }
            let entry = state.entries[state.head]
            state.head += 1
            if state.head >= 256, state.head * 2 >= state.entries.count {
                state.entries.removeFirst(state.head)
                state.head = 0
            }
            return entry
        }
    }
}

/// A published token had no matching logprob entry (empty queue, or a different token id): the
/// request fails closed instead of emitting a wrong or misaligned logprob.
struct ContinuousServingLogprobPairingError: Error, Equatable, CustomStringConvertible {
    let publishedTokenID: Int
    let recordedTokenID: Int?

    var description: String {
        if let recordedTokenID {
            return "logprob entry is for token \(recordedTokenID) but token \(publishedTokenID) was published"
        }
        return "no logprob entry was recorded for published token \(publishedTokenID)"
    }
}
