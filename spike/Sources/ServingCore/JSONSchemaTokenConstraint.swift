import Foundation

/// Shared, immutable per-(vocab, tokenizer) resources needed by ANY grammar's mask table: the byte
/// classification, the byte trie built from it, and the sorted EOS id list plus the vocab word
/// count derived from it.
///
/// Response-format design note (stage 3a, slice 2c) requirement: "one table per loaded model, not
/// one per format" — the trie walk over a real (~250k-token) vocab is the expensive part to build
/// (see `JSONObjectTokenConstraintTests.testReleaseMicroBenchmark`'s recorded build-time numbers),
/// and it does not depend on which grammar (json_object vs json_schema) later walks it — see
/// `JSONObjectConstraintTrie.allowedBitset(from:wordCount:)`'s generic-over-`ByteWalkAutomaton`
/// signature. Building ONE `SharedVocabConstraintResources` from a model's classifications and
/// handing it to BOTH `JSONObjectConstraintTable.init(resources:...)` and
/// `JSONSchemaConstraintTable.init(resources:...)` shares that build across both response-format
/// kinds for the same loaded model, without either table needing to know about the other's cache or
/// grammar. Wiring an actual model-load call site to build and share ONE instance this way is stage
/// 3b/3c work (touches `ScalarServingJSONObjectConstraint.swift`, out of this stage's scope) — this
/// type is the reusable piece that makes that wiring a small, additive change rather than a rewrite.
/// A plain `struct`, not a class: every field is itself `Sendable` (`JSONObjectConstraintTrie` is
/// `@unchecked Sendable`), so this needs no locking of its own — it is read-only after `init`.
struct SharedVocabConstraintResources: Sendable {
    let classifications: [TokenByteClassification]
    let trie: JSONObjectConstraintTrie
    let sortedEOSIds: [Int]
    /// Words needed to hold one bit per vocab id (`ceil(classifications.count / 64)`).
    let wordCount: Int

    init(classifications: [TokenByteClassification]) {
        self.classifications = classifications
        self.trie = JSONObjectConstraintTrie(classifications: classifications)
        self.sortedEOSIds = classifications.enumerated().compactMap { $1 == .eos ? $0 : nil }
        self.wordCount = (classifications.count + 63) / 64
    }

    func classification(for id: Int) -> TokenByteClassification {
        guard id >= 0, id < classifications.count else { return .banned }
        return classifications[id]
    }
}

/// Builds a `JSONObjectConstraintTable` and a `JSONSchemaConstraintTable` for the SAME loaded
/// model's vocabulary, sharing ONE `SharedVocabConstraintResources` build (one trie walk, one
/// classification array, one EOS-id scan) between them — see that type's doc comment for why this
/// matters (the trie walk over a real ~250k-token vocab is the expensive part to build). This is
/// ServingCore's only PUBLIC seam for the trie-sharing path: `SharedVocabConstraintResources`
/// itself, and each table's own `init(resources:...)`, stay non-public — a caller (stage 3b's
/// `SpikeServingAdapters` model-load wiring) never needs to know `SharedVocabConstraintResources`
/// (or `JSONObjectConstraintTrie`) exists at all, only that it gets back two tables that share one
/// build.
public func makeSharedVocabConstraintTables(
    classifications: [TokenByteClassification],
    objectCacheByteBudget: Int = JSONObjectConstraintTable.defaultCacheByteBudget,
    schemaCacheByteBudget: Int = JSONSchemaConstraintTable.defaultCacheByteBudget
) -> (object: JSONObjectConstraintTable, schema: JSONSchemaConstraintTable) {
    let resources = SharedVocabConstraintResources(classifications: classifications)
    return (
        object: JSONObjectConstraintTable(resources: resources, cacheByteBudget: objectCacheByteBudget),
        schema: JSONSchemaConstraintTable(resources: resources, cacheByteBudget: schemaCacheByteBudget)
    )
}

public enum JSONSchemaConstraintError: Error, Sendable, Equatable {
    /// `token` is not in the current `allowedTokenIds()` set (banned, EOS-before-complete, or the
    /// automaton rejects one of its bytes). Mirrors `JSONObjectConstraintError.tokenDisallowed`.
    case tokenDisallowed(Int)
    /// Invariant violated: the trie/classification produced zero allowed ids for a reachable
    /// automaton state. Mirrors `JSONObjectConstraintError.noAllowedTokens` — see that case's doc
    /// comment; the same "always a bug, but surfaced as a typed error rather than trusted blindly"
    /// rationale applies here (a json_schema automaton state, like a json_object one, is never a
    /// dead end for any schema the compiler accepted — see `testNoDeadEnd` in
    /// `JSONSchemaTokenConstraintTests`).
    case noAllowedTokens
}

/// Shared, immutable per-(vocab, tokenizer, SCHEMA) table backing `JSONSchemaTokenConstraint`.
/// Structurally parallel to `JSONObjectConstraintTable` (same FIFO byte-budgeted mask cache
/// mechanics, same cache-key-not-automaton-identity keying discipline), with two deliberate
/// differences documented at each divergence point below:
///
/// 1. The trie/classification/EOS-id/word-count resources are a `SharedVocabConstraintResources`
///    (constructible either fresh from `classifications`, matching `JSONObjectConstraintTable`'s own
///    default-init behavior, or from an already-built instance shared with a sibling
///    `JSONObjectConstraintTable` — see that type's doc comment).
/// 2. NO string-run fast path (the slice 1f miss-path optimization `JSONObjectConstraintTable` uses
///    for a state deep inside an open string). Design decision, not an oversight: applying it here
///    is very plausibly SAFE — unlike `JSONObjectAutomaton`, where every string (key or value) shares
///    ONE lexeme case (`.string`) and the fast path only avoids leaking through the `isKey`-gated
///    `completedValue()` side effect, `JSONSchemaAutomaton` already gives an UNCONSTRAINED string
///    VALUE (a `.string`-typed schema node) its own unambiguous `.string` lexeme case, while a
///    constrained property KEY uses `.objectKey` and a constrained enum STRING literal uses
///    `.enumMatch` — so a schema automaton's `.string` lexeme case, by construction, only ever means
///    "inside an unconstrained string value", with no keys-vs-values ambiguity to gate on. But
///    building it out (a `stringLexicalKey`-equivalent accessor, a `classifyStringRun`-equivalent
///    trie walk, and a matching STAY/EXIT-candidate cache) is real, untested new surface this stage's
///    time budget does not cover safely — left OFF here rather than adapted under time pressure. Cost
///    of leaving it off: a schema request generating a long unconstrained string value pays the full
///    O(vocab) trie DFS on ITS FIRST occurrence of each distinct in-string lexical sub-state (same
///    as json_object before slice 1f), rather than the ~13-keys-ever amortization slice 1f gives
///    json_object; the per-STATE mask cache below still turns any REPEAT of that exact state (byte
///    position, effectively) into a hit — only the very first miss per sub-state pays full price, not
///    every step. See `JSONSchemaTokenConstraintTests`'s informational timing comparison for the
///    measured gap this leaves on the table.
/// 3. No table-build-time cache warm-up (`JSONObjectConstraintTable.warmCommonStates()`): that
///    warm-up's curated prefix list is grammar-shaped for "any JSON object", not for an arbitrary
///    caller schema, so there is no schema-agnostic set of prefixes to warm here.
public final class JSONSchemaConstraintTable: @unchecked Sendable {
    private let resources: SharedVocabConstraintResources
    public var classifications: [TokenByteClassification] { resources.classifications }

    /// Same default as `JSONObjectConstraintTable.defaultCacheByteBudget` — one 64 MiB budget
    /// number for the project, not a format-specific tuning knob.
    public static let defaultCacheByteBudget = JSONObjectConstraintTable.defaultCacheByteBudget

    private let cacheByteBudget: Int

    private let lock = NSLock()
    // Keyed by `JSONSchemaAutomaton.MaskCacheKey` — see that type's doc comment for why it is safe
    // (and necessary) to key on the fingerprint-carrying, table-identity-EXCLUDING key rather than
    // the automaton itself: two `JSONSchemaTokenConstraint`s over the SAME schema (different
    // requests, different `JSONSchemaAutomaton` instances) must share entries here; two over
    // DIFFERENT schemas must never collide.
    private var cache: [JSONSchemaAutomaton.MaskCacheKey: [UInt64]] = [:]
    // FIFO eviction queue — same discipline as `JSONObjectConstraintTable.order`: eviction only ever
    // dequeues from `orderHead`, and a cache HIT never touches `order` at all.
    private var order: [JSONSchemaAutomaton.MaskCacheKey] = []
    private var orderHead = 0
    private var cachedBytesUsed = 0
    private var evictionCount = 0
    private var hits = 0
    private var misses = 0

    /// Test-only seam (internal, visible via `@testable import`) — mirrors
    /// `JSONObjectConstraintTable.eosIdsForTesting`.
    var eosIdsForTesting: [Int] { resources.sortedEOSIds }

    public convenience init(
        classifications: [TokenByteClassification],
        cacheByteBudget: Int = JSONSchemaConstraintTable.defaultCacheByteBudget
    ) {
        self.init(
            resources: SharedVocabConstraintResources(classifications: classifications),
            cacheByteBudget: cacheByteBudget)
    }

    /// Builds against an ALREADY-BUILT `SharedVocabConstraintResources` — the trie-sharing path (see
    /// that type's doc comment). The convenience initializer above is what every current call site
    /// (and this stage's own tests) uses; this is the seam a future model-load wiring stage uses to
    /// avoid a second trie build against the same classifications a sibling `JSONObjectConstraintTable`
    /// already built one for.
    init(
        resources: SharedVocabConstraintResources,
        cacheByteBudget: Int = JSONSchemaConstraintTable.defaultCacheByteBudget
    ) {
        self.resources = resources
        self.cacheByteBudget = cacheByteBudget
    }

    func classification(for id: Int) -> TokenByteClassification {
        resources.classification(for: id)
    }

    var cacheHitCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return hits
    }

    var cacheMissCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return misses
    }

    /// Test-only seam (internal): mirrors `JSONObjectConstraintTable.cachedBytes`.
    var cachedBytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return cachedBytesUsed
    }

    /// Test-only seam (internal): mirrors `JSONObjectConstraintTable.cacheEvictionCount`.
    var cacheEvictionCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return evictionCount
    }

    /// Compact bitset of allowed ids for `automaton`'s current state — same contract as
    /// `JSONObjectConstraintTable.allowedTokenBitset(for:)`.
    func allowedTokenBitset(for automaton: JSONSchemaAutomaton) throws -> [UInt64] {
        let key = automaton.maskCacheKey(maxTokenBytes: resources.trie.maxTokenByteLength)
        if let cached = lookupCache(key) {
            return cached
        }
        var bitset = resources.trie.allowedBitset(from: automaton, wordCount: resources.wordCount)
        if automaton.isComplete {
            for id in resources.sortedEOSIds {
                Self.setBit(&bitset, id)
            }
        }
        guard bitset.contains(where: { $0 != 0 }) else {
            assertionFailure(
                "JSONSchemaConstraintTable: no tokens allowed for automaton state \(automaton)")
            throw JSONSchemaConstraintError.noAllowedTokens
        }
        storeCache(key, bitset)
        return bitset
    }

    func allowedTokenIds(for automaton: JSONSchemaAutomaton) throws -> [Int] {
        Self.ids(from: try allowedTokenBitset(for: automaton))
    }

    /// True iff `id`'s bit is set in `bitset`. Delegates to `JSONObjectConstraintTable.isAllowed`
    /// (bitset layout is format-agnostic) rather than duplicating the bit-test arithmetic.
    public static func isAllowed(id: Int, in bitset: [UInt64]) -> Bool {
        JSONObjectConstraintTable.isAllowed(id: id, in: bitset)
    }

    private static func setBit(_ bitset: inout [UInt64], _ id: Int) {
        bitset[id >> 6] |= (UInt64(1) << UInt64(id & 63))
    }

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

    private func lookupCache(_ key: JSONSchemaAutomaton.MaskCacheKey) -> [UInt64]? {
        lock.lock()
        defer { lock.unlock() }
        guard let value = cache[key] else {
            misses += 1
            return nil
        }
        hits += 1
        return value
    }

    private func storeCache(_ key: JSONSchemaAutomaton.MaskCacheKey, _ bitset: [UInt64]) {
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

/// Constrains decoding to a well-formed instance of a compiled `response_format: {"type":
/// "json_schema", ...}` request (response-format slice 2, stage 3a wiring of the mask machinery),
/// backed by `JSONSchemaAutomaton` (per-request value state, carrying the caller's schema
/// fingerprint — see that type's `init(root:maxConsecutiveWhitespace:fingerprint:)`) and a shared
/// `JSONSchemaConstraintTable`. Mirrors `JSONObjectTokenConstraint` member-for-member.
public struct JSONSchemaTokenConstraint: TokenConstraint {
    private var automaton: JSONSchemaAutomaton
    private let table: JSONSchemaConstraintTable

    /// See `JSONObjectTokenConstraint.eosAlwaysAllowedForTesting`'s doc comment — same seam, same
    /// purpose, reachable only via `@testable import ServingCore`.
    private let eosAlwaysAllowedForTesting: Bool

    public init(table: JSONSchemaConstraintTable, format: JSONSchemaResponseFormat) {
        self.table = table
        self.automaton = JSONSchemaAutomaton(root: format.root, fingerprint: format.fingerprint)
        self.eosAlwaysAllowedForTesting = false
    }

    init(table: JSONSchemaConstraintTable, format: JSONSchemaResponseFormat, eosAlwaysAllowedForTesting: Bool) {
        self.table = table
        self.automaton = JSONSchemaAutomaton(root: format.root, fingerprint: format.fingerprint)
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

    /// Compact bitset accessor for `allowedTokenIds()`'s current state — see
    /// `JSONObjectTokenConstraint.allowedTokenBitset()`'s doc comment (same contract).
    public func allowedTokenBitset() throws -> [UInt64] {
        try table.allowedTokenBitset(for: automaton)
    }

    public mutating func advance(token: Int) throws {
        switch table.classification(for: token) {
        case .eos:
            guard automaton.isComplete || eosAlwaysAllowedForTesting else {
                throw JSONSchemaConstraintError.tokenDisallowed(token)
            }
            // EOS does not advance the byte automaton.
        case .banned:
            throw JSONSchemaConstraintError.tokenDisallowed(token)
        case .bytes(let bytes):
            var next = automaton
            for byte in bytes {
                guard next.advance(byte: byte) else {
                    throw JSONSchemaConstraintError.tokenDisallowed(token)
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
