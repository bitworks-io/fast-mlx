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
    // Keyed by `JSONObjectAutomaton.MaskCacheKey`, NOT the automaton itself: the key truncates
    // `stack` to the top `min(depth, trie.maxTokenByteLength)` frames (see that type's doc
    // comment), so distinct automaton states that no reachable token could ever tell apart
    // deliberately collide onto one cache entry — cutting first-miss traffic on real, deeply
    // nested request traces without approximating the mask itself.
    private var cache: [JSONObjectAutomaton.MaskCacheKey: [UInt64]] = [:]
    // FIFO eviction queue: `order[orderHead...]` are the live entries, oldest first. Eviction only
    // ever dequeues from `orderHead` (no scan), and lookup on a HIT never touches `order` at all —
    // this is a plain FIFO, not a strict LRU, specifically so a cache hit stays O(1) with no
    // per-hit reordering scan.
    private var order: [JSONObjectAutomaton.MaskCacheKey] = []
    private var orderHead = 0
    private var cachedBytesUsed = 0
    private var evictionCount = 0
    private var hits = 0
    private var misses = 0

    public convenience init(
        classifications: [TokenByteClassification],
        cacheByteBudget: Int = JSONObjectConstraintTable.defaultCacheByteBudget
    ) {
        self.init(classifications: classifications, cacheByteBudget: cacheByteBudget, warmCache: true)
    }

    /// Test-only seam (internal, visible via `@testable import`): lets a cache-mechanics unit test
    /// observe RAW hit/miss behavior (a state's first lookup always misses, a repeat always hits)
    /// without `warmCommonStates()` pre-populating the exact canonical states those tests probe —
    /// that pre-population is real, correct, and tested separately (see
    /// `JSONObjectConstraintTests.testCommonStatesAreWarmedAtTableBuild`); it would otherwise turn
    /// "first call at a fresh state" into a hit for the small set of canonical states, which is
    /// not what this seam's callers are testing. Production code always goes through the public
    /// initializer above, which warms unconditionally.
    init(
        classifications: [TokenByteClassification],
        cacheByteBudget: Int = JSONObjectConstraintTable.defaultCacheByteBudget,
        warmCache: Bool
    ) {
        self.classifications = classifications
        self.trie = JSONObjectConstraintTrie(classifications: classifications)
        self.sortedEOSIds = classifications.enumerated().compactMap { $1 == .eos ? $0 : nil }
        self.wordCount = (classifications.count + 63) / 64
        self.cacheByteBudget = cacheByteBudget
        if warmCache {
            warmCommonStates()
        }
    }

    /// Pre-populates the mask cache for a curated set of canonical, SHALLOW (depth <= 2) automaton
    /// states at table-BUILD time. This is what closes the remaining miss-path gap after the
    /// allocation-free trie walk and truncated cache key: even with those in place, a state's
    /// FIRST occurrence still pays an O(vocab) trie DFS (hundreds of ms on a real ~248k-token
    /// vocab, since a state like "inside an open string" accepts nearly every token) — see defect
    /// 3. Every real request's early bytes revisit the same handful of shallow, common grammar
    /// states (an opened object, a key string, a colon, a value string, a comma, structural
    /// whitespace, a completed document, a number/literal/nested-container start, ...), so
    /// computing their masks ONCE here — amortized into model load, not any decode step, and
    /// reported separately from the per-step budget — turns what would otherwise be each state's
    /// first-request miss into a cache hit from the very first request onward. This is an exact
    /// cache-WARMING optimization, not an approximation: `allowedTokenBitset(for:)` below is the
    /// same call any request makes, so a warmed entry is byte-for-byte the value a real first miss
    /// would have computed. The prefixes are chosen to touch every `Frame` case, every `Lexeme`
    /// category (structural/string-normal/escape/unicode-escape/number sub-states/literal), and
    /// the completed-document state — not to match any single benchmark or request's exact trace.
    /// A prefix the automaton itself rejects (none should, but this stays defensive rather than
    /// asserting) is simply skipped.
    private func warmCommonStates() {
        let prefixes: [String] = [
            "",  // initial: beforeTopLevel, empty stack
            "{",  // objectExpectKeyOrEnd
            "{ ",  // structural whitespace after '{'
            "{\"",  // inside a KEY string (normal)
            "{\"k",  // inside a KEY string (normal), content byte consumed
            "{\"k\"",  // key string closed -> objectExpectColon
            "{\"k\":",  // colon consumed -> objectExpectValue
            "{\"k\": ",  // structural whitespace after colon
            "{\"k\":\"",  // inside a VALUE string (normal)
            "{\"k\":\"v",  // inside a VALUE string, content byte consumed
            "{\"k\":\"v\\",  // string escape started
            "{\"k\":\"v\\n",  // string escape resolved (simple escape) -> back to normal
            "{\"k\":\"v\\u",  // unicode escape started
            "{\"k\":\"v\\u00",  // unicode escape, hex digits partially consumed
            "{\"k\":\"v\\u00e9",  // unicode escape fully consumed -> back to normal
            "{\"k\":\"v\"",  // value string closed -> objectExpectCommaOrEnd
            "{\"k\":\"v\" ",  // structural whitespace after a closed value, before comma/close
            "{\"k\":\"v\",",  // comma -> objectExpectKey
            "{\"k\":\"v\", ",  // structural whitespace after comma
            "{}",  // completed document (empty object)
            "{} ",  // completed document, trailing whitespace
            "{\"k\":\"v\"}",  // completed document (non-empty object)
            "{\"k\":[",  // arrayExpectValueOrEnd
            "{\"k\":[1",  // number (intDigits), inside an array
            "{\"k\":[1,",  // arrayExpectValue (after comma)
            "{\"k\":[1]",  // array closed -> back to objectExpectCommaOrEnd
            "{\"k\":-",  // number afterMinus
            "{\"k\":0",  // number leadingZero
            "{\"k\":1.",  // number afterPoint
            "{\"k\":1.5",  // number fracDigits
            "{\"k\":1.5e",  // number expectExponentDigitsOrSign
            "{\"k\":1.5e+",  // number expectExponentDigits
            "{\"k\":1.5e+1",  // number exponentDigits
            "{\"k\":t",  // literal true, in progress
            "{\"k\":tr",
            "{\"k\":f",  // literal false, in progress
            "{\"k\":n",  // literal null, in progress
            "{\"k\":{",  // nested object (depth 2) — one level of real nesting
            "{\"k\":{\"k\":",  // nested object, depth 2, expecting value
        ]
        for prefix in prefixes {
            var automaton = JSONObjectAutomaton()
            var ok = true
            for byte in prefix.utf8 {
                guard automaton.advance(byte: byte) else {
                    ok = false
                    break
                }
            }
            guard ok else { continue }
            warmState(automaton)
        }
    }

    /// Best-effort single-state cache warm, used only by `warmCommonStates()`. Deliberately
    /// separate from `allowedTokenBitset(for:)`: that method's "no allowed tokens" case is a
    /// hard invariant violation for a REAL request on a real byte-level vocab (every reachable
    /// grammar state always has a continuation), so it asserts and throws. A warm-up prefix is
    /// only a grammar-shaped GUESS at a commonly-visited state — it can legitimately be
    /// unreachable-with-continuation against a small or intentionally-incomplete vocab (e.g. a
    /// unit-test fixture missing some raw byte's token), which is not that invariant violation.
    /// Also does not touch `hits`/`misses`: warming is not a real cache lookup, so it must not
    /// perturb the hit/miss counters a caller reads later.
    private func warmState(_ automaton: JSONObjectAutomaton) {
        let key = automaton.maskCacheKey(maxTokenBytes: trie.maxTokenByteLength)
        guard !isCached(key) else { return }
        var bitset = trie.allowedBitset(from: automaton, wordCount: wordCount)
        if automaton.isComplete {
            for id in sortedEOSIds {
                Self.setBit(&bitset, id)
            }
        }
        guard bitset.contains(where: { $0 != 0 }) else { return }
        storeCache(key, bitset)
    }

    private func isCached(_ key: JSONObjectAutomaton.MaskCacheKey) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cache[key] != nil
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
        let key = automaton.maskCacheKey(maxTokenBytes: trie.maxTokenByteLength)
        if let cached = lookupCache(key) {
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
        storeCache(key, bitset)
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

    private func lookupCache(_ key: JSONObjectAutomaton.MaskCacheKey) -> [UInt64]? {
        lock.lock()
        defer { lock.unlock() }
        guard let value = cache[key] else {
            misses += 1
            return nil
        }
        hits += 1
        return value
    }

    private func storeCache(_ key: JSONObjectAutomaton.MaskCacheKey, _ bitset: [UInt64]) {
        lock.lock()
        defer { lock.unlock() }
        guard cache[key] == nil else { return }
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
        order.append(key)
        cache[key] = bitset
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
