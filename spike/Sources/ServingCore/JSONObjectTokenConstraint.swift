import Foundation

/// A value-semantics, per-request grammar constraint over token ids.
///
/// A conformer's PER-REQUEST state must be a pure value type (e.g. `JSONObjectTokenConstraint`'s
/// `JSONObjectAutomaton`), so it can be copied, stashed, and restored (e.g. on speculative-
/// decoding rollback) without aliasing. A conformer MAY additionally hold a reference to a shared,
/// immutable, read-only table built once per (vocab, tokenizer) — e.g. `JSONObjectConstraintTable`
/// — since nothing ever mutates it through request state; only the per-request automaton needs
/// value semantics, not every stored property.
/// `allowedTokenIds()` returns a SORTED ASCENDING list of ids. It is materialized on demand from a
/// cached compact bitset rather than caching the `[Int]` list itself: at a state deep inside a
/// JSON string, nearly the entire vocab is allowed, so a per-state id list can run to hundreds of
/// thousands of entries (a bounded cache of thousands of such lists would need gigabytes), while a
/// bitset's size is fixed by the vocab (`vocabSize / 64` `UInt64` words, ~31 KiB at 248k ids)
/// regardless of how many ids are allowed. `allowedTokenBitset()` exposes that cached bitset
/// directly, so a logits processor can build a `-inf` mask without materializing the `[Int]` list
/// at all.
public protocol TokenConstraint: Sendable {
    /// Sorted ascending ids allowed at the current state, including EOS iff `isComplete`.
    /// Throws only in the (asserted-unreachable) case where no id is allowed.
    func allowedTokenIds() throws -> [Int]

    /// Advances the constraint by one sampled token. Throws if `token` is not currently allowed.
    mutating func advance(token: Int) throws

    /// True once the constraint would also accept an EOS token here.
    var isComplete: Bool { get }
}

public enum JSONObjectConstraintError: Error, Sendable, Equatable {
    /// `token` is not in the current `allowedTokenIds()` set (banned, EOS-before-complete, or the
    /// automaton rejects one of its bytes).
    case tokenDisallowed(Int)
    /// Invariant violated: the trie/classification produced zero allowed ids for a reachable
    /// automaton state. Always a bug (a JSON-object automaton state is never a dead end), but
    /// surfaced as a typed error in release rather than trusted blindly.
    case noAllowedTokens
}

/// Shared, immutable per-(vocab, tokenizer) table backing `JSONObjectTokenConstraint`: the byte
/// classification, the byte trie built from it, sorted EOS ids, and a byte-budgeted FIFO cache of
/// automaton-state -> allowed-id bitset, since many requests over the same model reuse the same
/// small set of automaton states (e.g. "just opened a string", "just closed a value").
///
/// Built once at model load and shared read-only across concurrent requests. The cache is the
/// only mutable state, and it is guarded by a lock — `@unchecked Sendable` is justified because
/// every stored/read value is itself a value type and every mutation happens under `lock`.
public final class JSONObjectConstraintTable: @unchecked Sendable {
    public let classifications: [TokenByteClassification]
    private let trie: JSONObjectConstraintTrie
    private let sortedEOSIds: [Int]
    /// Words needed to hold one bit per vocab id (`ceil(classifications.count / 64)`).
    private let wordCount: Int

    /// Test-only seam (internal, visible via `@testable import`): lets a mutation test confirm
    /// the EOS-completeness gate actually discriminates, by reading the real EOS id set without
    /// duplicating the classification's EOS scan in test code.
    var eosIdsForTesting: [Int] { sortedEOSIds }

    /// Default cache memory budget. A cached bitset entry costs `wordCount * 8` bytes regardless
    /// of how many ids it allows (~31 KiB at a 248k vocab), so bounding the cache by total bytes —
    /// not just entry count — keeps memory bounded even if a caller raises vocab size or the
    /// number of distinct automaton states seen in practice.
    public static let defaultCacheByteBudget = 64 * 1024 * 1024

    private let cacheByteBudget: Int

    private let lock = NSLock()
    private var cache: [JSONObjectAutomaton: [UInt64]] = [:]
    // FIFO eviction queue: `order[orderHead...]` are the live entries, oldest first. Eviction only
    // ever dequeues from `orderHead` (no scan), and lookup on a HIT never touches `order` at all —
    // this is a plain FIFO, not a strict LRU, specifically so a cache hit stays O(1) with no
    // per-hit reordering scan.
    private var order: [JSONObjectAutomaton] = []
    private var orderHead = 0
    private var cachedBytesUsed = 0
    private var evictionCount = 0
    private var hits = 0
    private var misses = 0

    public init(
        classifications: [TokenByteClassification],
        cacheByteBudget: Int = JSONObjectConstraintTable.defaultCacheByteBudget
    ) {
        self.classifications = classifications
        self.trie = JSONObjectConstraintTrie(classifications: classifications)
        self.sortedEOSIds = classifications.enumerated().compactMap { $1 == .eos ? $0 : nil }
        self.wordCount = (classifications.count + 63) / 64
        self.cacheByteBudget = cacheByteBudget
    }

    public func classification(for id: Int) -> TokenByteClassification {
        guard id >= 0, id < classifications.count else { return .banned }
        return classifications[id]
    }

    public var cacheHitCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return hits
    }

    public var cacheMissCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return misses
    }

    /// Total bytes currently held by cached bitset entries. Test-only seam (internal): proves the
    /// byte budget — not just an entry-count cap — is what bounds cache memory.
    var cachedBytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return cachedBytesUsed
    }

    /// Number of cache entries evicted so far. Test-only seam (internal): a mutation control that
    /// a tiny budget actually evicted something, not merely that it never overflowed by luck.
    var cacheEvictionCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return evictionCount
    }

    /// Compact bitset of allowed ids for `automaton`'s current state (cached by state; `id`'s bit
    /// lives at `bitset[id >> 6] & (1 << (id & 63))`). A logits processor can consult this
    /// directly to build a `-inf` mask without materializing a full `[Int]` id list.
    func allowedTokenBitset(for automaton: JSONObjectAutomaton) throws -> [UInt64] {
        if let cached = lookupCache(automaton) {
            return cached
        }
        var bitset = trie.allowedBitset(from: automaton, wordCount: wordCount)
        if automaton.isComplete {
            for id in sortedEOSIds {
                Self.setBit(&bitset, id)
            }
        }
        guard bitset.contains(where: { $0 != 0 }) else {
            assertionFailure(
                "JSONObjectConstraintTable: no tokens allowed for automaton state \(automaton)")
            throw JSONObjectConstraintError.noAllowedTokens
        }
        storeCache(automaton, bitset)
        return bitset
    }

    /// Sorted ascending allowed ids for `automaton`'s current state, materialized from the cached
    /// bitset (see `allowedTokenBitset(for:)`).
    func allowedTokenIds(for automaton: JSONObjectAutomaton) throws -> [Int] {
        Self.ids(from: try allowedTokenBitset(for: automaton))
    }

    /// True iff `id`'s bit is set in `bitset`.
    public static func isAllowed(id: Int, in bitset: [UInt64]) -> Bool {
        guard id >= 0 else { return false }
        let word = id >> 6
        guard word < bitset.count else { return false }
        return bitset[word] & (UInt64(1) << UInt64(id & 63)) != 0
    }

    private static func setBit(_ bitset: inout [UInt64], _ id: Int) {
        bitset[id >> 6] |= (UInt64(1) << UInt64(id & 63))
    }

    /// Ascending ids from a bitset's set bits (word order ascending, then bit order ascending
    /// within a word — so the result is already sorted with no separate `.sort()` needed).
    private static func ids(from bitset: [UInt64]) -> [Int] {
        var result: [Int] = []
        for (wordIndex, word) in bitset.enumerated() where word != 0 {
            var bits = word
            while bits != 0 {
                let bit = bits.trailingZeroBitCount
                result.append(wordIndex * 64 + bit)
                bits &= bits - 1
            }
        }
        return result
    }

    private func lookupCache(_ automaton: JSONObjectAutomaton) -> [UInt64]? {
        lock.lock()
        defer { lock.unlock() }
        guard let value = cache[automaton] else {
            misses += 1
            return nil
        }
        hits += 1
        return value
    }

    private func storeCache(_ automaton: JSONObjectAutomaton, _ bitset: [UInt64]) {
        lock.lock()
        defer { lock.unlock() }
        guard cache[automaton] == nil else { return }
        let entryBytes = bitset.count * MemoryLayout<UInt64>.stride
        while cachedBytesUsed + entryBytes > cacheByteBudget, orderHead < order.count {
            let oldest = order[orderHead]
            orderHead += 1
            if let removed = cache.removeValue(forKey: oldest) {
                cachedBytesUsed -= removed.count * MemoryLayout<UInt64>.stride
                evictionCount += 1
            }
        }
        if orderHead > 0, orderHead * 2 > order.count {
            order.removeFirst(orderHead)
            orderHead = 0
        }
        order.append(automaton)
        cache[automaton] = bitset
        cachedBytesUsed += entryBytes
    }
}

/// Constrains decoding to a well-formed top-level JSON object (`response_format: {"type":
/// "json_object"}`), backed by `JSONObjectAutomaton` (per-request value state) and a shared
/// `JSONObjectConstraintTable` (vocab-wide byte trie + classification + mask cache).
public struct JSONObjectTokenConstraint: TokenConstraint {
    private var automaton = JSONObjectAutomaton()
    private let table: JSONObjectConstraintTable

    /// When true, EOS is treated as always allowed regardless of `automaton.isComplete`. Only
    /// ever set by the mutation test in `JSONObjectTokenConstraintTests` — it exists to prove the
    /// EOS-completeness gate is load-bearing: with it flipped, the same adversarial scorer must
    /// produce invalid JSON. Internal (not `public`), so it is reachable only via `@testable
    /// import ServingCore`, never from production callers.
    private let eosAlwaysAllowedForTesting: Bool

    public init(table: JSONObjectConstraintTable) {
        self.table = table
        self.eosAlwaysAllowedForTesting = false
    }

    init(table: JSONObjectConstraintTable, eosAlwaysAllowedForTesting: Bool) {
        self.table = table
        self.eosAlwaysAllowedForTesting = eosAlwaysAllowedForTesting
    }

    public var isComplete: Bool { automaton.isComplete }

    public func allowedTokenIds() throws -> [Int] {
        var ids = try table.allowedTokenIds(for: automaton)
        if eosAlwaysAllowedForTesting {
            let merged = Set(ids).union(table.eosIdsForTesting)
            ids = merged.sorted()
        }
        return ids
    }

    /// Compact bitset accessor for `allowedTokenIds()`'s current state: a logits processor can
    /// consult `JSONObjectConstraintTable.isAllowed(id:in:)` directly to build a `-inf` mask
    /// without materializing the full `[Int]` list. Does not apply the test-only
    /// `eosAlwaysAllowedForTesting` override (that seam exists only to exercise the `[Int]` path).
    public func allowedTokenBitset() throws -> [UInt64] {
        try table.allowedTokenBitset(for: automaton)
    }

    public mutating func advance(token: Int) throws {
        switch table.classification(for: token) {
        case .eos:
            guard automaton.isComplete || eosAlwaysAllowedForTesting else {
                throw JSONObjectConstraintError.tokenDisallowed(token)
            }
            // EOS does not advance the byte automaton.
        case .banned:
            throw JSONObjectConstraintError.tokenDisallowed(token)
        case .bytes(let bytes):
            var next = automaton
            for byte in bytes {
                guard next.advance(byte: byte) else {
                    throw JSONObjectConstraintError.tokenDisallowed(token)
                }
            }
            automaton = next
        }
    }

    /// Cache hit count on the shared table, as of this call.
    public var cacheHitCount: Int { table.cacheHitCount }
    /// Cache miss count on the shared table, as of this call.
    public var cacheMissCount: Int { table.cacheMissCount }
}
