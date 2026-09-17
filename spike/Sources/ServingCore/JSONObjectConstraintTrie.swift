import Foundation

/// Minimal per-byte walk surface the shared trie DFS (and shared mask-cache table) need from ANY
/// grammar automaton — `JSONObjectAutomaton` (response-format slice 1) and `JSONSchemaAutomaton`
/// (response-format slice 2b) both conform. A PROTOCOL, not a class hierarchy or an existential
/// parameter: every call site below is a GENERIC function (`<A: ByteWalkAutomaton>`), so the
/// compiler specializes a dedicated copy of the DFS for each concrete conforming type at each
/// instantiation site — the json_object hot path pays no indirect (existential witness-table)
/// dispatch cost from this trie also supporting json_schema. See each conformer's own
/// `tryAdvanceForWalk`/`undoForWalk` doc comment for its specific undo strategy and cost (the two
/// conformers deliberately differ: `JSONObjectAutomaton` uses an O(1) field-diff restore,
/// `JSONSchemaAutomaton` uses a simpler but more expensive full-value snapshot — see that type's
/// doc comment for why).
protocol ByteWalkAutomaton {
    associatedtype WalkUndo
    associatedtype MaskCacheKey: Hashable

    mutating func tryAdvanceForWalk(byte: UInt8) -> WalkUndo?
    mutating func undoForWalk(_ undo: WalkUndo)
    var isComplete: Bool { get }
    func maskCacheKey(maxTokenBytes: Int) -> MaskCacheKey
}

/// An immutable, shared byte trie over every `.bytes` token in a vocab's classification table.
///
/// Built once per loaded model/tokenizer and shared read-only across requests — hence `final`
/// and `@unchecked Sendable`: nothing here mutates after `init`, so concurrent reads from multiple
/// requests are safe by construction. Only the per-request `JSONObjectAutomaton` STATE is a value
/// type; the trie itself is intentionally a reference type to avoid copying a (potentially large)
/// vocab-sized structure per request.
final class JSONObjectConstraintTrie: @unchecked Sendable {
    private struct Node {
        var children: [UInt8: Int32] = [:]
        var terminalIds: [Int] = []
    }

    private var nodes: [Node] = [Node()]

    /// The longest `.bytes` token in the vocab this trie was built from, in bytes (`0` if there are
    /// none). Equal to the trie's own maximum root-to-node depth, since every inserted path is
    /// exactly one edge per byte of its token — this is also the DFS's maximum possible depth, and
    /// therefore the `maxTokenBytes` bound `JSONObjectAutomaton.maskCacheKey(maxTokenBytes:)` needs
    /// to produce an exact (not approximate) truncated cache key.
    let maxTokenByteLength: Int

    init(classifications: [TokenByteClassification]) {
        // Computed before `insert` is ever called: `insert` is an instance method, and Swift
        // requires every stored property initialized before `self` is used in a method call.
        self.maxTokenByteLength = classifications.reduce(into: 0) { longest, classification in
            guard case .bytes(let bytes) = classification else { return }
            longest = max(longest, bytes.count)
        }
        for (id, classification) in classifications.enumerated() {
            guard case .bytes(let bytes) = classification, !bytes.isEmpty else { continue }
            insert(bytes: bytes, id: id)
        }
    }

    private func insert(bytes: [UInt8], id: Int) {
        var nodeIndex = 0
        for byte in bytes {
            if let next = nodes[nodeIndex].children[byte] {
                nodeIndex = Int(next)
            } else {
                nodes.append(Node())
                let newIndex = Int32(nodes.count - 1)
                nodes[nodeIndex].children[byte] = newIndex
                nodeIndex = Int(newIndex)
            }
        }
        nodes[nodeIndex].terminalIds.append(id)
    }

    /// DFS over the trie from the automaton's current state, pruning a subtree the instant a byte
    /// is rejected. Sets a bit for every `.bytes` token id whose COMPLETE byte sequence is valid
    /// from `automaton`'s current position (EOS and banned ids are layered on by the caller).
    ///
    /// Returns a compact bitset (`wordCount` `UInt64` words, `id`'s bit at `bitset[id/64] &
    /// (1 << id%64)`) rather than an `[Int]` id list: at a state deep inside a JSON string, nearly
    /// every vocab id is allowed, so an id list can run to hundreds of thousands of entries per
    /// state while the bitset's size is fixed by the vocab, not by how many ids happen to be
    /// allowed.
    ///
    /// Walks with a SINGLE mutable `JSONObjectAutomaton` (`walker`), advancing forward and undoing
    /// on backtrack via `tryAdvanceForWalk`/`undoForWalk`, rather than copying the automaton at
    /// every trie edge (as a naive `automaton.advancing(byte:)`-per-edge DFS would): a state deep
    /// inside a JSON string accepts nearly the entire vocab, so that naive walk performs one struct
    /// copy — with its own array-refcount churn, and a REAL array copy on any edge that also
    /// mutates `stack` — per trie edge visited, which is exactly the first-miss cost this exists to
    /// avoid. `walker` starts as a copy of the caller's `automaton` (one, up front — cheap until its
    /// first mutation, and the caller's own value is never touched), then every push/pop/replace
    /// from that point on mutates a uniquely-referenced array in place.
    ///
    /// Generic over `A: ByteWalkAutomaton` (specialized per conforming type at each call site, not
    /// existential-dispatched): the trie's own node/edge structure depends only on the vocab's
    /// classifications, never on which grammar automaton is walking it, so ONE trie instance (built
    /// once from `classifications`) serves both `JSONObjectAutomaton` and `JSONSchemaAutomaton`
    /// walks — see `SharedVocabConstraintResources` (`JSONSchemaTokenConstraint.swift`), which is
    /// how a future caller shares one trie build across both response-format kinds for the same
    /// loaded model.
    func allowedBitset<A: ByteWalkAutomaton>(from automaton: A, wordCount: Int) -> [UInt64] {
        var bitset = [UInt64](repeating: 0, count: wordCount)
        var walker = automaton
        visit(nodeIndex: 0, automaton: &walker, bitset: &bitset)
        return bitset
    }

    private func visit<A: ByteWalkAutomaton>(nodeIndex: Int, automaton: inout A, bitset: inout [UInt64]) {
        let node = nodes[nodeIndex]
        for id in node.terminalIds {
            bitset[id >> 6] |= (UInt64(1) << UInt64(id & 63))
        }
        for (byte, childIndex) in node.children {
            guard let undo = automaton.tryAdvanceForWalk(byte: byte) else { continue }
            visit(nodeIndex: Int(childIndex), automaton: &automaton, bitset: &bitset)
            automaton.undoForWalk(undo)
        }
    }

    // MARK: - String-run classification (slice 1f miss-path optimization)

    /// Classifies every `.bytes` token against `automaton`'s CURRENT in-string state (caller
    /// guarantees `automaton.stringLexicalKey != nil`) into two exact classes:
    ///
    /// - STAY: the token's full byte sequence is consumed while `automaton` remains inside the
    ///   SAME string throughout (no unescaped closing `"` reached). Depends only on the string-
    ///   internal lexical sub-state (see `JSONObjectAutomaton.StringLexicalKey`), so this result is
    ///   valid for ANY automaton sharing that key, regardless of stack/depth/key-vs-value context —
    ///   `JSONObjectConstraintTable` caches it keyed by that lexical key alone, computed once.
    /// - EXIT candidates: tokens where, walked from this state, some byte closes the string (an
    ///   unescaped `"` from `.normal`) partway through (or exactly at the last byte). Once a token's
    ///   walk crosses that boundary, the validity of any FURTHER bytes is structural and depends on
    ///   the REAL automaton's stack/key-vs-value context — this method does not attempt to decide
    ///   that; it only enumerates every token whose byte sequence closes the string at some point
    ///   (returning each one's FULL bytes so the caller can replay them against the real automaton,
    ///   exactly as `JSONObjectTokenConstraint.advance(token:)` already does for a single token). A
    ///   token that is lexically invalid inside the string before ever reaching a close (e.g. an
    ///   unescaped control byte, a bad `\u` hex digit, an out-of-range UTF-8 continuation byte) is
    ///   pruned immediately and appears in neither list — that verdict does not depend on context
    ///   either.
    func classifyStringRun(
        from automaton: JSONObjectAutomaton
    ) -> (stayIds: [Int], exitCandidates: [(id: Int, bytes: [UInt8])]) {
        var stayIds: [Int] = []
        var exitCandidates: [(id: Int, bytes: [UInt8])] = []
        var walker = automaton
        var path: [UInt8] = []
        visitStringRun(
            nodeIndex: 0, automaton: &walker, path: &path, stayIds: &stayIds,
            exitCandidates: &exitCandidates)
        return (stayIds, exitCandidates)
    }

    /// In-string phase: steps `automaton` forward for real (via `tryAdvanceForWalk`, same
    /// allocation-free walk `allowedBitset(from:wordCount:)` uses), pruning a byte that is
    /// lexically invalid inside the string. The moment a byte closes the string (automaton leaves
    /// `.string` state), this stops driving `automaton` further — the closing byte's own
    /// terminal ids (and everything below it in the trie) are handed to `collectExitSubtree`, a
    /// plain trie enumeration that makes no automaton-context assumption.
    private func visitStringRun(
        nodeIndex: Int, automaton: inout JSONObjectAutomaton, path: inout [UInt8],
        stayIds: inout [Int], exitCandidates: inout [(id: Int, bytes: [UInt8])]
    ) {
        let node = nodes[nodeIndex]
        for id in node.terminalIds {
            stayIds.append(id)  // token ended here; automaton is still inside the string
        }
        for (byte, childIndex) in node.children {
            guard let undo = automaton.tryAdvanceForWalk(byte: byte) else { continue }
            path.append(byte)
            if automaton.stringLexicalKey != nil {
                visitStringRun(
                    nodeIndex: Int(childIndex), automaton: &automaton, path: &path, stayIds: &stayIds,
                    exitCandidates: &exitCandidates)
            } else {
                collectExitSubtree(nodeIndex: Int(childIndex), path: &path, exitCandidates: &exitCandidates)
            }
            path.removeLast()
            automaton.undoForWalk(undo)
        }
    }

    /// Post-close phase: pure trie enumeration, no automaton stepping — every terminal id under
    /// this subtree becomes an EXIT candidate carrying its full accumulated byte path, since
    /// whether those post-close bytes are structurally valid depends on context this walk does not
    /// have (see `classifyStringRun`'s doc comment).
    private func collectExitSubtree(
        nodeIndex: Int, path: inout [UInt8], exitCandidates: inout [(id: Int, bytes: [UInt8])]
    ) {
        let node = nodes[nodeIndex]
        for id in node.terminalIds {
            exitCandidates.append((id, path))
        }
        for (byte, childIndex) in node.children {
            path.append(byte)
            collectExitSubtree(nodeIndex: Int(childIndex), path: &path, exitCandidates: &exitCandidates)
            path.removeLast()
        }
    }
}
