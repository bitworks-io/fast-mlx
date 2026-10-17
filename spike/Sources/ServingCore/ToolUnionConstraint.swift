import CryptoKit
import Foundation

// `tool_choice: "required"` enforcement (cycle 259): a byte automaton that accepts exactly
//
//     {"name": "<N>", "arguments": <A_N>}
//
// where `<N>` is one declared function name and `<A_N>` is any document the CHOSEN tool's compiled
// arguments schema accepts. The framing bytes are fixed literals (exactly this spacing); whitespace
// freedom exists only inside `<A_N>`, under the inner `JSONSchemaAutomaton`'s own rules. The
// composite conforms to `ByteWalkAutomaton`, so the shared token trie computes its masks and a token
// that straddles a phase boundary (`",`, `}}`, ...) is allowed exactly when its bytes are.
//
// `JSONSchemaAutomaton` is NOT modified: the union is a sibling that owns one inner automaton at a
// time (the chosen tool's) and a small framing state machine around it. The subset compiler cannot
// express this as an `anyOf` (it needs disjoint first bytes, and every per-tool object starts `{`).

// MARK: - Format

/// The declared tools of one `tool_choice: "required"` request: their names and compiled arguments
/// formats, in declaration order.
public struct ToolUnionFormat: Sendable, Equatable {
    public let names: [String]
    public let formats: [JSONSchemaResponseFormat]
    /// Lowercase hex SHA-256 over the (name, format fingerprint) pairs in order. Mask-cache keys carry
    /// it, so two requests share cache entries only when their tool lists compile identically.
    public let fingerprint: String

    /// OpenAI's function-name grammar: non-empty, `[A-Za-z0-9_-]` only. Anything else would need JSON
    /// escaping inside the fixed `"name": "..."` frame.
    public static func isValidName(_ name: String) -> Bool {
        let bytes = Array(name.utf8)
        guard !bytes.isEmpty else { return false }
        return bytes.allSatisfy { byte in
            (0x30...0x39 ~= byte) || (0x41...0x5A ~= byte) || (0x61...0x7A ~= byte) || byte == 0x5F
                || byte == 0x2D
        }
    }

    /// `nil` when `entries` is empty, a name is not `isValidName`, or two names are equal.
    public init?(entries: [(name: String, format: JSONSchemaResponseFormat)]) {
        guard !entries.isEmpty else { return nil }
        var seen = Set<String>()
        var hasher = SHA256()
        hasher.update(data: Data("tool-union-v1".utf8))
        for entry in entries {
            guard Self.isValidName(entry.name), seen.insert(entry.name).inserted else { return nil }
            Self.update(&hasher, Array(entry.name.utf8))
            Self.update(&hasher, Array(entry.format.fingerprint.utf8))
        }
        self.names = entries.map(\.name)
        self.formats = entries.map(\.format)
        self.fingerprint = hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func update(_ hasher: inout SHA256, _ bytes: [UInt8]) {
        var length = UInt32(bytes.count).bigEndian
        withUnsafeBytes(of: &length) { hasher.update(bufferPointer: $0) }
        hasher.update(data: Data(bytes))
    }

    public static func == (lhs: ToolUnionFormat, rhs: ToolUnionFormat) -> Bool {
        lhs.fingerprint == rhs.fingerprint && lhs.names == rhs.names && lhs.formats == rhs.formats
    }

    /// What a generated document contains once the name is complete.
    public struct Parsed: Sendable, Equatable {
        public let toolIndex: Int
        public let name: String
        /// The bytes of `<A_N>` generated so far (trailing whitespace stripped).
        public let arguments: String
        /// True when the closing outer `}` was generated (the document is whole).
        public let isComplete: Bool
    }

    /// Reads the (possibly truncated) document `generated` produced under this union's mask. `nil`
    /// when the name was not completed (no `", "arguments": ` frame was finished), so no call can be
    /// named. Replays the bytes through the same automaton the mask used, so the split between name,
    /// arguments and closing brace cannot drift from what was enforced.
    public func parse(generated: String) -> Parsed? {
        var automaton = ToolUnionAutomaton(format: self)
        let bytes = Array(generated.utf8)
        var argumentsStart: Int?
        var consumed = 0
        for byte in bytes {
            guard automaton.advance(byte: byte) else { break }
            consumed += 1
            if argumentsStart == nil, automaton.isInArguments { argumentsStart = consumed }
        }
        guard let start = argumentsStart, let tool = automaton.chosenTool else { return nil }
        let complete = automaton.isComplete
        var end = complete ? consumed - 1 : consumed
        while end > start, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[end - 1]) { end -= 1 }
        return Parsed(
            toolIndex: tool, name: names[tool],
            arguments: String(decoding: bytes[start..<end], as: UTF8.self), isComplete: complete)
    }
}

// MARK: - Incremental argument stream

/// Incremental counterpart of `ToolUnionFormat.parse(generated:)` for streaming (cycle 260): fed the
/// generated document in pieces, it reports the chosen tool's name once the `"arguments": ` frame is
/// complete and then only the bytes that belong to the arguments value. The wrapper's closing `}` is
/// never reported, and trailing whitespace is held back until a later non-whitespace argument byte
/// proves it is interior, so the concatenation of every `arguments` piece equals
/// `parse(generated:)`'s `arguments` byte for byte at every cut point (a cut before the frame
/// completes reports no name, exactly as `parse` returns `nil`).
///
/// It replays the same `ToolUnionAutomaton` the mask used. A byte the automaton rejects ends the
/// stream (everything after it is ignored), mirroring `parse`'s `break`.
public struct ToolUnionArgumentStream: Sendable {
    public struct Output: Sendable, Equatable {
        /// The chosen tool's name; non-nil exactly once, on the piece whose bytes complete the frame.
        public var name: String?
        /// Argument bytes released by this piece (possibly empty).
        public var arguments: String
    }

    private var automaton: ToolUnionAutomaton
    private let names: [String]
    private var heldWhitespace: [UInt8] = []
    private var ended = false

    public init(format: ToolUnionFormat) {
        self.automaton = ToolUnionAutomaton(format: format)
        self.names = format.names
    }

    public mutating func consume(_ text: String) -> Output {
        var name: String?
        var released: [UInt8] = []
        for byte in text.utf8 {
            if ended { break }
            let wasInArguments = automaton.isInArguments
            guard automaton.advance(byte: byte) else {
                ended = true
                break
            }
            if !wasInArguments {
                if automaton.isInArguments, let tool = automaton.chosenTool { name = names[tool] }
                continue
            }
            if automaton.isComplete {
                // The wrapper's closing `}`: not an argument byte; held whitespace is trailing.
                heldWhitespace.removeAll()
                continue
            }
            if [0x20, 0x09, 0x0A, 0x0D].contains(byte) {
                heldWhitespace.append(byte)
            } else {
                released.append(contentsOf: heldWhitespace)
                heldWhitespace.removeAll()
                released.append(byte)
            }
        }
        return Output(name: name, arguments: String(decoding: released, as: UTF8.self))
    }
}

/// Immutable, shared (by reference) compiled state of one `ToolUnionFormat`; copies of an automaton
/// share it instead of each carrying the name tables and every tool's compiled schema.
final class ToolUnionSpec: Sendable {
    let fingerprint: String
    let nameBytes: [[UInt8]]
    /// Fresh inner automata, one per tool. Built with an EMPTY fingerprint on purpose: inside a union
    /// the discriminator between two tools' inner states is the tool index in the mask-cache key (plus
    /// the union fingerprint for the whole tool list), so that index is load-bearing rather than
    /// redundant with an inner fingerprint.
    let initialInner: [JSONSchemaAutomaton]

    init(format: ToolUnionFormat) {
        self.fingerprint = format.fingerprint
        self.nameBytes = format.names.map { Array($0.utf8) }
        self.initialInner = format.formats.map { JSONSchemaAutomaton(root: $0.root) }
    }
}

// MARK: - Automaton

struct ToolUnionAutomaton: ByteWalkAutomaton {
    static let prefixLiteral: [UInt8] = Array(#"{"name": ""#.utf8)
    /// Starts with the quote that closes the name.
    static let middleLiteral: [UInt8] = Array(#"", "arguments": "#.utf8)
    private static let quote: UInt8 = 0x22
    private static let closeBrace: UInt8 = 0x7D

    enum Phase: Hashable, Sendable {
        case prefix(matched: Int)
        /// `candidates`: indices of declared names whose first `position` bytes equal the bytes read.
        case name(candidates: [Int], position: Int)
        case middle(tool: Int, matched: Int)
        case arguments(tool: Int)
        case done
    }

    private let spec: ToolUnionSpec
    private(set) var phase: Phase = .prefix(matched: 0)
    private var inner: JSONSchemaAutomaton?
    /// Index of the tool whose name has been completed (set when the closing quote is read).
    private(set) var chosenTool: Int?

    init(format: ToolUnionFormat) {
        self.spec = ToolUnionSpec(format: format)
    }

    var isComplete: Bool { phase == .done }

    var isInArguments: Bool {
        if case .arguments = phase { return true }
        return false
    }

    /// Advances by `byte`; on rejection the automaton is left exactly as it was.
    @discardableResult
    mutating func advance(byte: UInt8) -> Bool {
        tryAdvanceForWalk(byte: byte) != nil
    }

    // MARK: ByteWalkAutomaton

    struct WalkUndo {
        fileprivate let snapshot: ToolUnionAutomaton
    }

    mutating func tryAdvanceForWalk(byte: UInt8) -> WalkUndo? {
        let undo = WalkUndo(snapshot: self)
        guard step(byte: byte) else {
            self = undo.snapshot
            return nil
        }
        return undo
    }

    mutating func undoForWalk(_ undo: WalkUndo) {
        self = undo.snapshot
    }

    private mutating func step(byte: UInt8) -> Bool {
        switch phase {
        case .prefix(let matched):
            guard byte == Self.prefixLiteral[matched] else { return false }
            let next = matched + 1
            phase =
                next == Self.prefixLiteral.count
                ? .name(candidates: Array(spec.nameBytes.indices), position: 0)
                : .prefix(matched: next)
            return true
        case .name(let candidates, let position):
            if byte == Self.quote {
                // A strict prefix of a declared name is not a name.
                guard let tool = candidates.first(where: { spec.nameBytes[$0].count == position }) else {
                    return false
                }
                chosenTool = tool
                phase = .middle(tool: tool, matched: 1)
                return true
            }
            let next = candidates.filter {
                position < spec.nameBytes[$0].count && spec.nameBytes[$0][position] == byte
            }
            guard !next.isEmpty else { return false }
            phase = .name(candidates: next, position: position + 1)
            return true
        case .middle(let tool, let matched):
            guard byte == Self.middleLiteral[matched] else { return false }
            let next = matched + 1
            if next == Self.middleLiteral.count {
                phase = .arguments(tool: tool)
                inner = spec.initialInner[tool]
            } else {
                phase = .middle(tool: tool, matched: next)
            }
            return true
        case .arguments:
            guard var current = inner else { return false }
            if current.tryAdvanceForWalk(byte: byte) != nil {
                inner = current
                return true
            }
            // The inner document can only be closed by the outer `}` once it is complete.
            if byte == Self.closeBrace, current.isComplete {
                inner = nil
                phase = .done
                return true
            }
            return false
        case .done:
            return false
        }
    }

    /// Mask-cache key: union fingerprint + framing phase (which carries the chosen tool index, or the
    /// name candidates and position) + the inner automaton's own key. Excludes table identity, so two
    /// requests with the same tool list share entries; the tool index keeps two tools' inner states
    /// (whose node indices overlap) from ever sharing one.
    struct MaskCacheKey: Hashable {
        fileprivate let fingerprint: String
        fileprivate let phase: Phase
        fileprivate let inner: JSONSchemaAutomaton.MaskCacheKey?
    }

    func maskCacheKey(maxTokenBytes: Int) -> MaskCacheKey {
        MaskCacheKey(
            fingerprint: spec.fingerprint, phase: phase,
            inner: inner?.maskCacheKey(maxTokenBytes: maxTokenBytes))
    }
}

// MARK: - Table

/// Mask table for `ToolUnionAutomaton`: shares the per-vocab trie / EOS ids of the owning
/// `JSONSchemaConstraintTable` and keeps its own byte-budgeted FIFO cache (same mechanics as that
/// table's), keyed by `ToolUnionAutomaton.MaskCacheKey`.
final class ToolUnionConstraintTable: @unchecked Sendable {
    private let resources: SharedVocabConstraintResources
    private let cacheByteBudget: Int
    private let lock = NSLock()
    private var cache: [ToolUnionAutomaton.MaskCacheKey: [UInt64]] = [:]
    private var order: [ToolUnionAutomaton.MaskCacheKey] = []
    private var orderHead = 0
    private var cachedBytesUsed = 0
    private var hits = 0
    private var misses = 0

    init(resources: SharedVocabConstraintResources, cacheByteBudget: Int) {
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

    var cachedEntryCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return cache.count
    }

    func allowedTokenBitset(for automaton: ToolUnionAutomaton) throws -> [UInt64] {
        let key = automaton.maskCacheKey(maxTokenBytes: resources.trie.maxTokenByteLength)
        if let cached = lookup(key) { return cached }
        var bitset = resources.trie.allowedBitset(from: automaton, wordCount: resources.wordCount)
        if automaton.isComplete {
            for id in resources.sortedEOSIds {
                bitset[id >> 6] |= (UInt64(1) << UInt64(id & 63))
            }
        }
        guard bitset.contains(where: { $0 != 0 }) else {
            assertionFailure("ToolUnionConstraintTable: no tokens allowed for state \(automaton.phase)")
            throw JSONSchemaConstraintError.noAllowedTokens
        }
        store(key, bitset)
        return bitset
    }

    private func lookup(_ key: ToolUnionAutomaton.MaskCacheKey) -> [UInt64]? {
        lock.lock()
        defer { lock.unlock() }
        guard let value = cache[key] else {
            misses += 1
            return nil
        }
        hits += 1
        return value
    }

    private func store(_ key: ToolUnionAutomaton.MaskCacheKey, _ bitset: [UInt64]) {
        lock.lock()
        defer { lock.unlock() }
        guard cache[key] == nil else { return }
        let entryBytes = bitset.count * MemoryLayout<UInt64>.stride
        while cachedBytesUsed + entryBytes > cacheByteBudget, orderHead < order.count {
            let oldest = order[orderHead]
            orderHead += 1
            if let removed = cache.removeValue(forKey: oldest) {
                cachedBytesUsed -= removed.count * MemoryLayout<UInt64>.stride
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

// MARK: - Token constraint

/// Constrains decoding to `{"name": "<N>", "arguments": <A_N>}` over the declared tools. Mirrors
/// `JSONSchemaTokenConstraint` (EOS allowed only when the document is complete; a banned token or a
/// byte the automaton rejects throws).
public struct ToolUnionTokenConstraint: TokenConstraint {
    private var automaton: ToolUnionAutomaton
    private let table: ToolUnionConstraintTable

    public init(table: JSONSchemaConstraintTable, format: ToolUnionFormat) {
        self.table = table.unionTable
        self.automaton = ToolUnionAutomaton(format: format)
    }

    public var isComplete: Bool { automaton.isComplete }

    public func allowedTokenIds() throws -> [Int] {
        var ids: [Int] = []
        for (wordIndex, word) in (try table.allowedTokenBitset(for: automaton)).enumerated() where word != 0 {
            var bits = word
            while bits != 0 {
                ids.append(wordIndex * 64 + bits.trailingZeroBitCount)
                bits &= bits - 1
            }
        }
        return ids
    }

    public func allowedTokenBitset() throws -> [UInt64] {
        try table.allowedTokenBitset(for: automaton)
    }

    public mutating func advance(token: Int) throws {
        switch table.classification(for: token) {
        case .eos:
            guard automaton.isComplete else { throw JSONSchemaConstraintError.tokenDisallowed(token) }
        case .banned:
            throw JSONSchemaConstraintError.tokenDisallowed(token)
        case .bytes(let bytes):
            var next = automaton
            for byte in bytes {
                guard next.advance(byte: byte) else { throw JSONSchemaConstraintError.tokenDisallowed(token) }
            }
            automaton = next
        }
    }

    public var cacheHitCount: Int { table.cacheHitCount }
    public var cacheMissCount: Int { table.cacheMissCount }
}
