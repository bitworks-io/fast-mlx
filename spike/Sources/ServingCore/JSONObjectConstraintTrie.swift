import Foundation

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
    func allowedBitset(from automaton: JSONObjectAutomaton, wordCount: Int) -> [UInt64] {
        var bitset = [UInt64](repeating: 0, count: wordCount)
        var walker = automaton
        visit(nodeIndex: 0, automaton: &walker, bitset: &bitset)
        return bitset
    }

    private func visit(nodeIndex: Int, automaton: inout JSONObjectAutomaton, bitset: inout [UInt64]) {
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
}
