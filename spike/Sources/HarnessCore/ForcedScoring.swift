import Foundation

/// One scored row inside a chunk: the model's output at `localIndex` (an offset into the
/// chunk's forward output) is the next-token distribution for forced position `position`.
public struct RowSelection: Sendable, Equatable {
    public let localIndex: Int
    public let position: Int
    public init(localIndex: Int, position: Int) {
        self.localIndex = localIndex; self.position = position
    }
}

/// The pure position bookkeeping behind CHUNKED teacher-forced scoring.
///
/// Why chunked: scoring a forced continuation with thousands of single-token forwards makes
/// every step's transient buffers (returned KV slices, attention intermediates) slightly LARGER
/// than the last step's, so MLX's buffer cache can never reuse a freed buffer — the cache grows
/// as the sum of all step sizes, i.e. O(context^2) bytes. Measured on Qwen3-32B-4bit: ~23GB of
/// dead cached buffers by position 5000, ~48GB extrapolated at 7200 — which, with the Python
/// reference process ballooning identically, is exactly the ~7K jetsam SIGKILL ceiling the
/// harness hit. Chunked prefill scoring replaces N single-token forwards with N/chunkSize
/// multi-token forwards whose transients are same-shaped chunk to chunk (so the cache reuses
/// them), and is prefill-fast instead of decode-slow.
///
/// Semantics: the full input is prompt + forced.dropLast() (the last forced token is never fed —
/// its ROW is produced by the token before it). The row for forced position i is the model's
/// output at input index promptCount-1+i. `wantedPositions` (ascending) selects a subset of
/// forced positions; nil means all of them.
public struct ForcedScoringPlan: Sendable, Equatable {
    public struct Chunk: Sendable, Equatable {
        /// Indices into the full input token array (prompt + forced.dropLast()).
        public let inputRange: Range<Int>
        /// Rows to extract from this chunk's forward output, ascending by position.
        public let rows: [RowSelection]
        public init(inputRange: Range<Int>, rows: [RowSelection]) {
            self.inputRange = inputRange; self.rows = rows
        }
    }
    public let chunks: [Chunk]
    public init(chunks: [Chunk]) { self.chunks = chunks }
}

public func forcedScoringPlan(
    promptCount: Int, forcedCount: Int, wantedPositions: [Int]?, chunkSize: Int
) -> ForcedScoringPlan {
    precondition(promptCount >= 1, "need at least one prompt token")
    precondition(forcedCount >= 1, "need at least one forced position")
    precondition(chunkSize >= 1, "chunk size must be positive")
    let inputLength = promptCount + forcedCount - 1
    let wanted = wantedPositions.map(Set.init)
    var chunks: [ForcedScoringPlan.Chunk] = []
    var start = 0
    while start < inputLength {
        let end = min(start + chunkSize, inputLength)
        var rows: [RowSelection] = []
        for inputIndex in start..<end {
            let position = inputIndex - (promptCount - 1)
            guard position >= 0, position < forcedCount else { continue }
            if let wanted, !wanted.contains(position) { continue }
            rows.append(RowSelection(localIndex: inputIndex - start, position: position))
        }
        chunks.append(.init(inputRange: start..<end, rows: rows))
        start = end
    }
    return ForcedScoringPlan(chunks: chunks)
}
