import Foundation

/// A byte-level pushdown automaton that only accepts JSON instances of a compiled
/// `JSONSchemaNode` (response-format slice 2b). Mirrors `JSONObjectAutomaton`'s API, whitespace
/// policy, and lexical (string/number/UTF-8) sub-machines exactly — see that file for the
/// rationale behind each — but the value grammar at every position is driven by the schema tree
/// instead of being fixed to "any JSON value". Stage 3 makes the mask table generic over both
/// automata; this file does not depend on `JSONObjectAutomaton` at runtime, only mirrors it.
///
/// State cost: the `JSONSchemaNode` tree is interned ONCE per `init` into an immutable
/// `SchemaTable` (flattened node/property/literal arrays, children referenced by `Int` index) so
/// that per-byte automaton states — a `stack: [Frame]` of small value-type frames plus a scalar
/// `lexeme` — never copy the tree. `Hashable`/`Equatable` combine the table's `ObjectIdentifier`
/// with the frame stack and lexical state, so states from two different `init` calls (even with
/// structurally identical schemas) are never equal — see the type doc on `Hashable` conformance
/// below.
public struct JSONSchemaAutomaton: Hashable, Sendable {
    /// - Parameters:
    ///   - maxConsecutiveWhitespace: See `JSONObjectAutomaton.defaultMaxConsecutiveWhitespace`;
    ///     a test may override it to prove the cap (rather than some other detail) forces termination.
    ///   - fingerprint: The originating `JSONSchemaResponseFormat.fingerprint` (defaults to `""` so
    ///     every pre-stage-3a call site — direct `JSONSchemaAutomaton(root:)` construction throughout
    ///     `JSONSchemaConstraintAutomatonTests`, and `accepts(bytes:root:)` below — keeps compiling
    ///     and behaving identically). Carried ONLY into `maskCacheKey(maxTokenBytes:)`, never into this
    ///     type's own `==`/`hash(into:)` beyond the field already added there — see that key's doc
    ///     comment for why the mask cache needs it (and this type's own table-identity-keyed equality
    ///     does not, on its own, have to).
    public init(
        root: JSONSchemaNode,
        maxConsecutiveWhitespace: Int = JSONObjectAutomaton.defaultMaxConsecutiveWhitespace,
        fingerprint: String = ""
    ) {
        self.table = Self.buildTable(root: root)
        self.maxConsecutiveWhitespace = maxConsecutiveWhitespace
        self.fingerprint = fingerprint
    }

    private let table: SchemaTable
    private let maxConsecutiveWhitespace: Int
    private let fingerprint: String

    private var stack: [Frame] = []
    private var lexeme: Lexeme = .none
    private var beforeTopLevel = true
    private var afterTopLevel = false
    private var whitespaceRun = 0
    // Same "at most one line-break EVENT per structural whitespace run" rule as
    // `JSONObjectAutomaton` — see that file's field doc for `lineBreakSeenInRun`/`pendingCR`.
    private var lineBreakSeenInRun = false
    private var pendingCR = false
    private var rejected = false

    // MARK: - Hashable / Equatable (table identity + progress)

    /// Two states are equal only if they share the SAME compiled table (`===`, not structural
    /// equality of the source `JSONSchemaNode`) and agree on every progress field. This means two
    /// separately-`init`-ed automatons over structurally identical schemas compare unequal — by
    /// design (mirrors the mask-cache layer, which keys on the caller's schema fingerprint, not
    /// structural equality of the IR) — while copies derived from ONE automaton (e.g. via
    /// `advancing(byte:)`) compare equal exactly when their consumed bytes left them in the same
    /// state.
    public static func == (lhs: JSONSchemaAutomaton, rhs: JSONSchemaAutomaton) -> Bool {
        lhs.table === rhs.table
            && lhs.fingerprint == rhs.fingerprint
            && lhs.maxConsecutiveWhitespace == rhs.maxConsecutiveWhitespace
            && lhs.stack == rhs.stack
            && lhs.lexeme == rhs.lexeme
            && lhs.beforeTopLevel == rhs.beforeTopLevel
            && lhs.afterTopLevel == rhs.afterTopLevel
            && lhs.whitespaceRun == rhs.whitespaceRun
            && lhs.lineBreakSeenInRun == rhs.lineBreakSeenInRun
            && lhs.pendingCR == rhs.pendingCR
            && lhs.rejected == rhs.rejected
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(table))
        hasher.combine(fingerprint)
        hasher.combine(maxConsecutiveWhitespace)
        hasher.combine(stack)
        hasher.combine(lexeme)
        hasher.combine(beforeTopLevel)
        hasher.combine(afterTopLevel)
        hasher.combine(whitespaceRun)
        hasher.combine(lineBreakSeenInRun)
        hasher.combine(pendingCR)
        hasher.combine(rejected)
    }

    // MARK: - Public surface

    /// True once the top-level value has closed (or, for a bare top-level scalar with no closing
    /// delimiter of its own — a number, or a number-typed enum literal — once it has reached a
    /// byte position where EOS would be a legal terminator) and only... nothing else may follow:
    /// like `JSONObjectAutomaton`, NO trailing whitespace is accepted after the top-level value,
    /// so `isComplete` is exactly "would EOS be accepted here", not "is a suffix of whitespace
    /// still pending".
    public var isComplete: Bool {
        guard !rejected, !beforeTopLevel, stack.isEmpty else { return false }
        switch lexeme {
        case .none:
            return true
        case .number(let state, _):
            return Self.numberCanTerminate(state)
        case .enumMatch(let candidates, let position):
            return candidates.contains { table.literalBytes[$0].count == position }
        case .string, .literal, .objectKey:
            return false
        }
    }

    @discardableResult
    public mutating func advance(byte: UInt8) -> Bool {
        guard !rejected else { return false }
        guard advanceInner(byte: byte) else {
            rejected = true
            return false
        }
        return true
    }

    public func advancing(byte: UInt8) -> JSONSchemaAutomaton? {
        var copy = self
        return copy.advance(byte: byte) ? copy : nil
    }

    // MARK: - Allocation-free walk (trie DFS) — stage 3a, response-format slice 2c

    /// Opaque per-byte undo token for `tryAdvanceForWalk(byte:)`/`undoForWalk(_:)`.
    ///
    /// SNAPSHOT-based (holds a full copy of the pre-byte automaton), unlike
    /// `JSONObjectAutomaton.WalkUndo`'s O(1) field-diff restore: this automaton's `Frame` cases carry
    /// per-branch payload (`objectNode`, `matchedPropertyIndex`, `nextPropertyIndex`, `itemNode`) and
    /// its `Lexeme` cases additionally carry variable-length `[Int]` candidate arrays (`.objectKey`,
    /// `.enumMatch`) that `JSONObjectAutomaton.Lexeme` has no equivalent of — a precise "restore only
    /// what changed" diff would need its own bespoke undo logic for those candidate arrays too, on
    /// top of the stack-frame delta reasoning `JSONObjectAutomaton.WalkUndo`'s doc comment already
    /// spells out. The task's own design note calls a snapshot-based undo acceptable here specifically
    /// because it is simplest; the COST it trades away: any DFS byte that mutates `stack` (push, pop,
    /// or same-index top replacement) or replaces a `Lexeme` payload holding an array forces a REAL
    /// copy-on-write array copy on THIS call (`tryAdvanceForWalk`, not just `undoForWalk`) — the
    /// snapshot keeps the pre-call buffer's refcount above 1, so `self`'s own mutation right after
    /// cannot reuse it in place. Bytes that only change scalar fields (most in-string/in-number bytes,
    /// which is where nearly all of a mask's fanout lives) stay O(1) as usual. Net effect: a schema
    /// trie DFS's first-miss cost is higher than json_object's per byte that touches structural/array
    /// state, though still far cheaper than the naive "copy the whole automaton and replay per
    /// candidate TOKEN" reference `JSONSchemaTokenConstraintTests`'s differential test uses as ground
    /// truth — see that test file's informational timing comparison for the measured gap.
    struct WalkUndo {
        fileprivate let snapshot: JSONSchemaAutomaton
    }

    /// Trial advance for the trie DFS: mutates `self` in place, exactly like
    /// `JSONObjectAutomaton.tryAdvanceForWalk(byte:)` (same contract — never sets the permanent
    /// `rejected` latch on a live walker; success returns an undo token, failure leaves `self`
    /// untouched). See `WalkUndo`'s doc comment for why THIS conformer restores via a full snapshot.
    mutating func tryAdvanceForWalk(byte: UInt8) -> WalkUndo? {
        guard !rejected else { return nil }
        let undo = WalkUndo(snapshot: self)
        guard advanceInner(byte: byte) else {
            self = undo.snapshot
            return nil
        }
        return undo
    }

    /// Restores exactly what the matching `tryAdvanceForWalk(byte:)` call touched (the whole
    /// pre-call value — see `WalkUndo`'s doc comment).
    mutating func undoForWalk(_ undo: WalkUndo) {
        self = undo.snapshot
    }

    // MARK: - Mask cache key — stage 3a, response-format slice 2c

    /// Opaque, `Hashable` mask-cache key mirroring `JSONObjectAutomaton.MaskCacheKey` exactly in
    /// shape and truncation argument (see that type's doc comment for the full proof — it carries
    /// over unchanged here: a single byte transition still changes `stack` by at most one push, one
    /// pop, or one same-index top replacement — see `advanceStructural`/`beginValue` above, which
    /// mirror `JSONObjectAutomaton`'s own structural dispatch shape exactly), with ONE addition:
    /// `fingerprint`.
    ///
    /// Cache-key rules this type must satisfy (response-format slice 2c design note):
    /// - States from DIFFERENT schemas must NEVER share a key: `fingerprint` (the caller's
    ///   `JSONSchemaResponseFormat.fingerprint`, a SHA-256 of the canonical schema JSON) is part of
    ///   the key, so two different schemas — even ones that happen to reach structurally identical
    ///   `visibleFrames`/`lexeme` shapes — never collide.
    /// - The SAME schema at the same logical state must produce EQUAL keys across DIFFERENT
    ///   `JSONSchemaAutomaton` instances (so the shared mask table's cache is reused across separate
    ///   requests for the same schema, not just within one request's own automaton copies). This key
    ///   deliberately does NOT include `ObjectIdentifier(table)` — unlike this type's own `==`/
    ///   `hash(into:)` above, which intentionally treats two separate `init` calls as unequal (see
    ///   that doc comment) — because `table` is rebuilt fresh, per instance, by every `init(root:)`
    ///   call, even for the identical schema tree. What DOES transfer across instances is that
    ///   `buildTable`'s `intern` walk is a deterministic, purely structural depth-first traversal of
    ///   the SAME `JSONSchemaNode` tree: two separate `init` calls over an identical tree always
    ///   produce identically-numbered `SchemaTable.nodes`/`properties`/`literalBytes` indices. Since
    ///   every `Frame`/`Lexeme` payload this key touches (`objectNode`, `matchedPropertyIndex`,
    ///   `nextPropertyIndex`, `itemNode`, enum `candidates`, object-key `candidates`) is stored as
    ///   plain `Int`/`[Int]` INDICES into those arrays — never a reference back into `table` itself —
    ///   two instances over the same schema reach byte-for-byte equal keys at the same logical
    ///   position, with `fingerprint` (not table identity) as the only cross-schema discriminator.
    ///   This is the "build a key that excludes table identity and includes the fingerprint"
    ///   alternative from the design note (chosen over interning `SchemaTable` per fingerprint: no
    ///   bounded cache of compiled tables to size/evict, and it is a strictly smaller diff against
    ///   the existing per-instance `buildTable` call in `init`).
    struct MaskCacheKey: Hashable {
        fileprivate let fingerprint: String
        fileprivate let lexeme: Lexeme
        fileprivate let beforeTopLevel: Bool
        fileprivate let afterTopLevel: Bool
        fileprivate let whitespaceRun: Int
        fileprivate let lineBreakSeenInRun: Bool
        fileprivate let pendingCR: Bool
        fileprivate let rejected: Bool
        fileprivate let visibleFrames: [Frame]
        fileprivate let hasHiddenFrames: Bool
    }

    /// Builds this state's `MaskCacheKey` against a table whose longest `.bytes` token is
    /// `maxTokenBytes` bytes — see `JSONObjectConstraintTrie.maxTokenByteLength` (the SAME shared
    /// trie instance backs both automaton kinds) and `JSONObjectAutomaton.maskCacheKey(maxTokenBytes:)`
    /// for the truncation argument this mirrors exactly.
    func maskCacheKey(maxTokenBytes: Int) -> MaskCacheKey {
        let visibleFrames: [Frame]
        let hasHiddenFrames: Bool
        if stack.count > maxTokenBytes {
            visibleFrames = maxTokenBytes > 0 ? Array(stack.suffix(maxTokenBytes)) : []
            hasHiddenFrames = true
        } else {
            visibleFrames = stack
            hasHiddenFrames = false
        }
        return MaskCacheKey(
            fingerprint: fingerprint, lexeme: lexeme, beforeTopLevel: beforeTopLevel,
            afterTopLevel: afterTopLevel, whitespaceRun: whitespaceRun,
            lineBreakSeenInRun: lineBreakSeenInRun, pendingCR: pendingCR, rejected: rejected,
            visibleFrames: visibleFrames, hasHiddenFrames: hasHiddenFrames)
    }

    public static func accepts(bytes: [UInt8], root: JSONSchemaNode) -> Bool {
        var automaton = JSONSchemaAutomaton(root: root)
        for byte in bytes {
            guard automaton.advance(byte: byte) else { return false }
        }
        return automaton.isComplete
    }

    // MARK: - Interned schema table

    /// One flattened, index-referenced node. Mirrors `JSONSchemaNode`'s cases; children are `Int`
    /// indices into `SchemaTable.nodes` (or, for object properties / enum literals / anyOf
    /// branches, into the correspondingly-named flattened array) instead of nested payloads, so
    /// automaton states never carry a copy of the tree.
    fileprivate enum NodeRecord: Sendable {
        case string
        case number
        case integer
        case boolean
        case null
        case enumeration(literalStart: Int, literalCount: Int)
        case object(propStart: Int, propCount: Int)
        case array(itemNode: Int)
        case anyOf(branchStart: Int, branchCount: Int)
    }

    fileprivate struct PropertyRecord: Sendable {
        let nameBytes: [UInt8]
        let required: Bool
        let valueNodeIndex: Int
    }

    /// Immutable, reference-typed, built once per `init`. `anyOfRouting[node]` is a 256-entry
    /// byte→branch-node-index table precomputed at build time (not per state, not per byte): entry
    /// `-1` means no branch claims that byte, `-2` means more than one branch's first-byte set
    /// claims it (a compiler bug — branches are supposed to be pairwise disjoint on their first
    /// byte; the automaton fails closed rather than nondeterministically picking one — see
    /// `resolveAnyOfBranch`).
    fileprivate final class SchemaTable: Sendable {
        let nodes: [NodeRecord]
        let properties: [PropertyRecord]
        let literalBytes: [[UInt8]]
        let anyOfBranches: [Int]
        let rootIndex: Int
        let anyOfRouting: [Int: [Int16]]

        init(
            nodes: [NodeRecord], properties: [PropertyRecord], literalBytes: [[UInt8]],
            anyOfBranches: [Int], rootIndex: Int, anyOfRouting: [Int: [Int16]]
        ) {
            self.nodes = nodes
            self.properties = properties
            self.literalBytes = literalBytes
            self.anyOfBranches = anyOfBranches
            self.rootIndex = rootIndex
            self.anyOfRouting = anyOfRouting
        }
    }

    private static func buildTable(root: JSONSchemaNode) -> SchemaTable {
        var nodes: [NodeRecord] = []
        var properties: [PropertyRecord] = []
        var literalBytes: [[UInt8]] = []
        var anyOfBranches: [Int] = []

        func intern(_ node: JSONSchemaNode) -> Int {
            switch node {
            case .string:
                nodes.append(.string)
            case .number:
                nodes.append(.number)
            case .integer:
                nodes.append(.integer)
            case .boolean:
                nodes.append(.boolean)
            case .null:
                nodes.append(.null)
            case .enumeration(let literals):
                let start = literalBytes.count
                literalBytes.append(contentsOf: literals.map { $0.jsonText })
                nodes.append(.enumeration(literalStart: start, literalCount: literals.count))
            case .object(let props):
                var records: [PropertyRecord] = []
                records.reserveCapacity(props.count)
                for property in props {
                    let valueIndex = intern(property.value)
                    records.append(
                        PropertyRecord(
                            nameBytes: Array(property.name.utf8), required: property.required,
                            valueNodeIndex: valueIndex))
                }
                let start = properties.count
                properties.append(contentsOf: records)
                nodes.append(.object(propStart: start, propCount: records.count))
            case .array(let items):
                let itemIndex = intern(items)
                nodes.append(.array(itemNode: itemIndex))
            case .anyOf(let branches):
                var branchIndices: [Int] = []
                branchIndices.reserveCapacity(branches.count)
                for branch in branches {
                    branchIndices.append(intern(branch))
                }
                let start = anyOfBranches.count
                anyOfBranches.append(contentsOf: branchIndices)
                nodes.append(.anyOf(branchStart: start, branchCount: branchIndices.count))
            }
            return nodes.count - 1
        }

        let rootIndex = intern(root)

        var routing: [Int: [Int16]] = [:]
        for (index, record) in nodes.enumerated() {
            if case .anyOf(let branchStart, let branchCount) = record {
                routing[index] = buildAnyOfRouting(
                    nodes: nodes, literalBytes: literalBytes, anyOfBranches: anyOfBranches,
                    branchStart: branchStart, branchCount: branchCount)
            }
        }

        return SchemaTable(
            nodes: nodes, properties: properties, literalBytes: literalBytes,
            anyOfBranches: anyOfBranches, rootIndex: rootIndex, anyOfRouting: routing)
    }

    /// The set of bytes that could legally be the FIRST byte of a value matching `nodeIndex`.
    /// Used only at table-build time (to precompute `anyOfRouting`), never per automaton byte.
    private static func firstByteSet(
        nodes: [NodeRecord], literalBytes: [[UInt8]], anyOfBranches: [Int], nodeIndex: Int
    ) -> Set<UInt8> {
        switch nodes[nodeIndex] {
        case .string:
            return [ASCII.quote]
        case .number, .integer:
            var bytes: Set<UInt8> = [ASCII.minus]
            bytes.formUnion(ASCII.zero...ASCII.nine)
            return bytes
        case .boolean:
            return [ASCII.t, ASCII.f]
        case .null:
            return [ASCII.n]
        case .object:
            return [ASCII.openBrace]
        case .array:
            return [ASCII.openBracket]
        case .enumeration(let literalStart, let literalCount):
            return Set((literalStart..<(literalStart + literalCount)).map { literalBytes[$0][0] })
        case .anyOf(let branchStart, let branchCount):
            var bytes: Set<UInt8> = []
            for offset in 0..<branchCount {
                bytes.formUnion(
                    firstByteSet(
                        nodes: nodes, literalBytes: literalBytes, anyOfBranches: anyOfBranches,
                        nodeIndex: anyOfBranches[branchStart + offset]))
            }
            return bytes
        }
    }

    private static func buildAnyOfRouting(
        nodes: [NodeRecord], literalBytes: [[UInt8]], anyOfBranches: [Int], branchStart: Int,
        branchCount: Int
    ) -> [Int16] {
        var routing = [Int16](repeating: -1, count: 256)
        for offset in 0..<branchCount {
            let branchNode = anyOfBranches[branchStart + offset]
            let bytes = firstByteSet(
                nodes: nodes, literalBytes: literalBytes, anyOfBranches: anyOfBranches,
                nodeIndex: branchNode)
            for byte in bytes {
                let slot = Int(byte)
                if routing[slot] == -1 {
                    routing[slot] = Int16(branchNode)
                } else if routing[slot] != Int16(branchNode) {
                    routing[slot] = -2  // ambiguous first byte: compiler-disjointness violated
                }
            }
        }
        return routing
    }

    /// `nil` if `byte` cannot start any branch (unclaimed) or claims more than one branch
    /// (`-2`, a fail-closed rejection — see `SchemaTable.anyOfRouting`).
    private func resolveAnyOfBranch(nodeIndex: Int, byte: UInt8) -> Int? {
        guard let routing = table.anyOfRouting[nodeIndex] else { return nil }
        let branch = routing[Int(byte)]
        return branch >= 0 ? Int(branch) : nil
    }

    // MARK: - Frame stack (object/array structural progress)

    fileprivate enum Frame: Hashable, Sendable {
        // Just after `{` (allowEnd reflects whether no REQUIRED property remains from
        // nextPropertyIndex on, i.e. `}` may legally close here) or just after `,` (allowEnd always
        // false: a comma always implies at least one more property is still reachable, checked
        // before the comma is accepted — see `.objectExpectCommaOrEnd`).
        //
        // `nextPropertyIndex`/`matchedPropertyIndex` are ABSOLUTE indices into the flattened
        // `SchemaTable.properties` array (NOT relative to this object's own `propStart`): a nested
        // object's properties are interned — and so occupy a slice of `SchemaTable.properties` —
        // before its enclosing object's own slice is appended (see `buildTable`'s depth-first
        // `intern`), so `propStart` is only 0 for the very first object interned overall. Every
        // consumer of these indices (`allowedCandidates`, `hasRequiredFrom`, and the direct
        // `table.properties[matchedPropertyIndex]` lookup in `.objectExpectValue`) must agree on
        // this — see `allowedCandidates`'s doc comment.
        case objectAwaitingKey(objectNode: Int, nextPropertyIndex: Int, allowEnd: Bool)
        case objectExpectColon(objectNode: Int, matchedPropertyIndex: Int)
        case objectExpectValue(objectNode: Int, matchedPropertyIndex: Int)
        case objectExpectCommaOrEnd(objectNode: Int, nextPropertyIndex: Int)
        case arrayExpectValueOrEnd(itemNode: Int)
        case arrayExpectValue(itemNode: Int)
        case arrayExpectCommaOrEnd(itemNode: Int)
    }

    // MARK: - Lexical (in-progress value) state

    fileprivate enum StringState: Hashable, Sendable {
        case normal
        case escape
        case unicodeEscape(remaining: Int)
        case continuation(remaining: Int, min: UInt8, max: UInt8)
    }

    fileprivate enum NumberState: Hashable, Sendable {
        case afterMinus
        case leadingZero
        case intDigits
        case afterPoint
        case fracDigits
        case expectExponentDigitsOrSign
        case expectExponentDigits
        case exponentDigits
    }

    fileprivate enum Literal: Hashable, Sendable { case trueLiteral, falseLiteral, nullLiteral }

    fileprivate enum Lexeme: Hashable, Sendable {
        case none
        case string(StringState)
        case number(NumberState, allowFractionAndExponent: Bool)
        case literal(Literal, matched: Int)
        // `candidates`: absolute indices into `SchemaTable.literalBytes` still consistent with the
        // bytes consumed so far (`position` bytes of the value). Filtered every byte — see
        // `advanceEnum`.
        case enumMatch(candidates: [Int], position: Int)
        // `candidates`: absolute indices into `SchemaTable.properties`, restricted to the window
        // `allowedCandidates` computed when the key's opening quote was seen — see that function.
        case objectKey(matchedBytes: Int, candidates: [Int])
    }

    // MARK: - Byte dispatch

    private mutating func advanceInner(byte: UInt8) -> Bool {
        switch lexeme {
        case .string(let state):
            return advanceString(byte: byte, state: state)
        case .number(let state, let allowFractionAndExponent):
            return advanceNumber(byte: byte, state: state, allowFractionAndExponent: allowFractionAndExponent)
        case .literal(let literal, let matched):
            return advanceLiteral(byte: byte, literal: literal, matched: matched)
        case .enumMatch(let candidates, let position):
            return advanceEnum(byte: byte, candidates: candidates, position: position)
        case .objectKey(let matchedBytes, let candidates):
            return advanceObjectKey(byte: byte, matchedBytes: matchedBytes, candidates: candidates)
        case .none:
            return advanceStructural(byte: byte)
        }
    }

    private mutating func advanceStructural(byte: UInt8) -> Bool {
        // Whitespace policy parity with `JSONObjectAutomaton`: no whitespace is ever valid before
        // the first value byte or after the top-level value closes.
        if afterTopLevel {
            return false
        }
        if beforeTopLevel {
            beforeTopLevel = false
            return beginValue(nodeIndex: table.rootIndex, byte: byte)
        }

        if Self.isWhitespace(byte) {
            whitespaceRun += 1
            guard whitespaceRun <= maxConsecutiveWhitespace else { return false }
            switch byte {
            case 0x0A:
                if pendingCR {
                    pendingCR = false
                } else {
                    guard !lineBreakSeenInRun else { return false }
                    lineBreakSeenInRun = true
                }
            case 0x0D:
                guard !lineBreakSeenInRun else { return false }
                lineBreakSeenInRun = true
                pendingCR = true
            default:
                pendingCR = false
            }
            return true
        }
        whitespaceRun = 0
        lineBreakSeenInRun = false
        pendingCR = false

        guard let top = stack.last else { return false }
        switch top {
        case .objectAwaitingKey(let node, let next, let allowEnd):
            if byte == ASCII.closeBrace {
                guard allowEnd else { return false }
                stack.removeLast()
                completedValue()
                return true
            }
            if byte == ASCII.quote {
                let candidates = allowedCandidates(node, from: next)
                guard !candidates.isEmpty else { return false }
                lexeme = .objectKey(matchedBytes: 0, candidates: candidates)
                return true  // stack top unchanged: still `.objectAwaitingKey` until the key resolves
            }
            return false
        case .objectExpectColon(let node, let matchedPropertyIndex):
            guard byte == ASCII.colon else { return false }
            stack[stack.count - 1] = .objectExpectValue(objectNode: node, matchedPropertyIndex: matchedPropertyIndex)
            return true
        case .objectExpectValue(let node, let matchedPropertyIndex):
            let property = table.properties[matchedPropertyIndex]
            stack[stack.count - 1] = .objectExpectCommaOrEnd(objectNode: node, nextPropertyIndex: matchedPropertyIndex + 1)
            return beginValue(nodeIndex: property.valueNodeIndex, byte: byte)
        case .objectExpectCommaOrEnd(let node, let next):
            if byte == ASCII.comma {
                guard !allowedCandidates(node, from: next).isEmpty else { return false }  // no trailing comma
                stack[stack.count - 1] = .objectAwaitingKey(objectNode: node, nextPropertyIndex: next, allowEnd: false)
                return true
            }
            if byte == ASCII.closeBrace {
                guard !hasRequiredFrom(node, next) else { return false }
                stack.removeLast()
                completedValue()
                return true
            }
            return false
        case .arrayExpectValueOrEnd(let itemNode):
            if byte == ASCII.closeBracket {
                stack.removeLast()
                completedValue()
                return true
            }
            stack[stack.count - 1] = .arrayExpectCommaOrEnd(itemNode: itemNode)
            return beginValue(nodeIndex: itemNode, byte: byte)
        case .arrayExpectValue(let itemNode):
            stack[stack.count - 1] = .arrayExpectCommaOrEnd(itemNode: itemNode)
            return beginValue(nodeIndex: itemNode, byte: byte)
        case .arrayExpectCommaOrEnd(let itemNode):
            if byte == ASCII.comma {
                stack[stack.count - 1] = .arrayExpectValue(itemNode: itemNode)
                return true
            }
            if byte == ASCII.closeBracket {
                stack.removeLast()
                completedValue()
                return true
            }
            return false
        }
    }

    /// Consumes `byte` as the FIRST byte of a value matching `nodeIndex`. For `.object`/`.array`
    /// this is only the opening bracket (the rest of the value is driven by `Frame` dispatch in
    /// `advanceStructural`); for scalar kinds it starts (and may, for `.anyOf`/short enum matches,
    /// finish) the value's lexical sub-machine.
    private mutating func beginValue(nodeIndex: Int, byte: UInt8) -> Bool {
        switch table.nodes[nodeIndex] {
        case .string:
            guard byte == ASCII.quote else { return false }
            lexeme = .string(.normal)
            return true
        case .number:
            return beginNumber(byte: byte, allowFractionAndExponent: true)
        case .integer:
            return beginNumber(byte: byte, allowFractionAndExponent: false)
        case .boolean:
            if byte == ASCII.t {
                lexeme = .literal(.trueLiteral, matched: 1)
                return true
            }
            if byte == ASCII.f {
                lexeme = .literal(.falseLiteral, matched: 1)
                return true
            }
            return false
        case .null:
            guard byte == ASCII.n else { return false }
            lexeme = .literal(.nullLiteral, matched: 1)
            return true
        case .enumeration(let literalStart, let literalCount):
            return advanceEnum(byte: byte, candidates: Array(literalStart..<(literalStart + literalCount)), position: 0)
        case .object(let propStart, _):
            guard byte == ASCII.openBrace else { return false }
            guard stack.count < JSONSchemaLimits.maxDepth else { return false }
            stack.append(
                .objectAwaitingKey(
                    objectNode: nodeIndex, nextPropertyIndex: propStart,
                    allowEnd: !hasRequiredFrom(nodeIndex, propStart)))
            return true
        case .array(let itemNode):
            guard byte == ASCII.openBracket else { return false }
            guard stack.count < JSONSchemaLimits.maxDepth else { return false }
            stack.append(.arrayExpectValueOrEnd(itemNode: itemNode))
            return true
        case .anyOf(let branchStart, let branchCount):
            _ = branchStart
            _ = branchCount
            guard let branch = resolveAnyOfBranch(nodeIndex: nodeIndex, byte: byte) else { return false }
            return beginValue(nodeIndex: branch, byte: byte)
        }
    }

    /// Called whenever a value fully closes via its own terminating byte (`}`, `]`, the last
    /// letter of a literal, a string's closing quote, or a number's/enum's terminator
    /// reprocessing). Mirrors `JSONObjectAutomaton.completedValue()`.
    private mutating func completedValue() {
        if stack.isEmpty {
            afterTopLevel = true
            whitespaceRun = 0
            lineBreakSeenInRun = false
            pendingCR = false
        }
    }

    // MARK: - Object key matching (fixed-candidate trie, not a generic string)

    /// The property indices (absolute, into `SchemaTable.properties`) that may legally be the NEXT
    /// key at `nextPropertyIndex` (also an ABSOLUTE `SchemaTable.properties` index — matching every
    /// call site's `Frame` payload, which threads `matchedPropertyIndex + 1` and `closed[0]`
    /// unchanged, both already absolute): every property from there up to and including the first
    /// REQUIRED one (an optional property may be skipped, but scanning must stop at — and include
    /// — the first required one, since skipping past a required property is never legal). Bounded
    /// by this object's own slice of the flattened table (`propStart..<propStart+propCount`), so a
    /// `nextPropertyIndex` that has already walked off the end of THIS object's properties (but is
    /// still a valid index into some OTHER object's slice of the same flattened array) correctly
    /// yields no candidates. Computed fresh each time a key starts (properties per object ≤
    /// `JSONSchemaLimits.maxPropertiesPerObject`, so this is a small, bounded scan, not a per-byte
    /// cost).
    private func allowedCandidates(_ node: Int, from nextPropertyIndex: Int) -> [Int] {
        guard case .object(let propStart, let propCount) = table.nodes[node] else { return [] }
        let end = propStart + propCount
        guard nextPropertyIndex < end else { return [] }
        var result: [Int] = []
        var index = nextPropertyIndex
        while index < end {
            result.append(index)
            if table.properties[index].required { break }
            index += 1
        }
        return result
    }

    /// `nextPropertyIndex` is an ABSOLUTE `SchemaTable.properties` index — see `allowedCandidates`.
    private func hasRequiredFrom(_ node: Int, _ nextPropertyIndex: Int) -> Bool {
        guard case .object(let propStart, let propCount) = table.nodes[node] else { return false }
        let end = propStart + propCount
        var index = nextPropertyIndex
        while index < end {
            if table.properties[index].required { return true }
            index += 1
        }
        return false
    }

    /// A key's content bytes are matched directly against the candidate names (no escape handling
    /// needed: the compiler guarantees property names contain no `"`, `\`, or control characters —
    /// see `JSONSchemaResponseFormat.swift` — so an actual `\` or control byte in the input simply
    /// matches no candidate and is rejected by the same filter as any other wrong byte).
    private mutating func advanceObjectKey(byte: UInt8, matchedBytes: Int, candidates: [Int]) -> Bool {
        if byte == ASCII.quote {
            let closed = candidates.filter { table.properties[$0].nameBytes.count == matchedBytes }
            guard closed.count == 1 else { return false }  // 0 = no match; >1 = duplicate names (fail closed)
            guard case .objectAwaitingKey(let node, _, _) = stack.last else { return false }
            stack[stack.count - 1] = .objectExpectColon(objectNode: node, matchedPropertyIndex: closed[0])
            lexeme = .none
            return true
        }
        let matching = candidates.filter {
            matchedBytes < table.properties[$0].nameBytes.count && table.properties[$0].nameBytes[matchedBytes] == byte
        }
        guard !matching.isEmpty else { return false }
        lexeme = .objectKey(matchedBytes: matchedBytes + 1, candidates: matching)
        return true
    }

    // MARK: - Strings (identical sub-machine to `JSONObjectAutomaton`, minus key handling)

    private mutating func advanceString(byte: UInt8, state: StringState) -> Bool {
        switch state {
        case .normal:
            if byte == ASCII.quote {
                lexeme = .none
                completedValue()
                return true
            }
            if byte == ASCII.backslash {
                lexeme = .string(.escape)
                return true
            }
            if byte < 0x20 {
                return false
            }
            if byte < 0x80 {
                return true
            }
            guard let (remaining, min, max) = Self.utf8ContinuationSpec(leadByte: byte) else { return false }
            lexeme = .string(.continuation(remaining: remaining, min: min, max: max))
            return true
        case .escape:
            switch byte {
            case ASCII.quote, ASCII.backslash, ASCII.slash, ASCII.b, ASCII.f, ASCII.n, ASCII.r, ASCII.t:
                lexeme = .string(.normal)
                return true
            case ASCII.u:
                lexeme = .string(.unicodeEscape(remaining: 4))
                return true
            default:
                return false
            }
        case .unicodeEscape(let remaining):
            guard Self.isHexDigit(byte) else { return false }
            lexeme = remaining == 1 ? .string(.normal) : .string(.unicodeEscape(remaining: remaining - 1))
            return true
        case .continuation(let remaining, let min, let max):
            guard byte >= min && byte <= max else { return false }
            lexeme = remaining == 1 ? .string(.normal) : .string(.continuation(remaining: remaining - 1, min: 0x80, max: 0xBF))
            return true
        }
    }

    private static func utf8ContinuationSpec(leadByte: UInt8) -> (remaining: Int, min: UInt8, max: UInt8)? {
        switch leadByte {
        case 0xC2...0xDF:
            return (1, 0x80, 0xBF)
        case 0xE0:
            return (2, 0xA0, 0xBF)
        case 0xE1...0xEC:
            return (2, 0x80, 0xBF)
        case 0xED:
            return (2, 0x80, 0x9F)
        case 0xEE...0xEF:
            return (2, 0x80, 0xBF)
        case 0xF0:
            return (3, 0x90, 0xBF)
        case 0xF1...0xF3:
            return (3, 0x80, 0xBF)
        case 0xF4:
            return (3, 0x80, 0x8F)
        default:
            return nil
        }
    }

    // MARK: - Numbers (identical grammar to `JSONObjectAutomaton`; `.integer` disables fraction/exponent)

    private mutating func beginNumber(byte: UInt8, allowFractionAndExponent: Bool) -> Bool {
        switch byte {
        case ASCII.minus:
            lexeme = .number(.afterMinus, allowFractionAndExponent: allowFractionAndExponent)
            return true
        case ASCII.zero:
            lexeme = .number(.leadingZero, allowFractionAndExponent: allowFractionAndExponent)
            return true
        case ASCII.one...ASCII.nine:
            lexeme = .number(.intDigits, allowFractionAndExponent: allowFractionAndExponent)
            return true
        default:
            return false
        }
    }

    private mutating func advanceNumber(byte: UInt8, state: NumberState, allowFractionAndExponent: Bool) -> Bool {
        if let next = Self.numberTransition(state: state, byte: byte, allowFractionAndExponent: allowFractionAndExponent) {
            lexeme = .number(next, allowFractionAndExponent: allowFractionAndExponent)
            return true
        }
        guard Self.isNumberTerminator(byte), Self.numberCanTerminate(state) else { return false }
        lexeme = .none
        completedValue()
        return advanceStructural(byte: byte)  // the terminator byte was never part of the number
    }

    private static func numberTransition(state: NumberState, byte: UInt8, allowFractionAndExponent: Bool) -> NumberState? {
        switch state {
        case .afterMinus:
            if byte == ASCII.zero { return .leadingZero }
            if ASCII.one...ASCII.nine ~= byte { return .intDigits }
            return nil
        case .leadingZero:
            guard allowFractionAndExponent else { return nil }  // `.integer` rejects `.`/`e`/`E` here
            if byte == ASCII.dot { return .afterPoint }
            if byte == ASCII.e || byte == ASCII.E { return .expectExponentDigitsOrSign }
            return nil
        case .intDigits:
            if ASCII.zero...ASCII.nine ~= byte { return .intDigits }
            guard allowFractionAndExponent else { return nil }  // `.integer` rejects `.`/`e`/`E` here
            if byte == ASCII.dot { return .afterPoint }
            if byte == ASCII.e || byte == ASCII.E { return .expectExponentDigitsOrSign }
            return nil
        case .afterPoint:
            if ASCII.zero...ASCII.nine ~= byte { return .fracDigits }
            return nil
        case .fracDigits:
            if ASCII.zero...ASCII.nine ~= byte { return .fracDigits }
            if byte == ASCII.e || byte == ASCII.E { return .expectExponentDigitsOrSign }
            return nil
        case .expectExponentDigitsOrSign:
            if byte == ASCII.plus || byte == ASCII.minus { return .expectExponentDigits }
            if ASCII.zero...ASCII.nine ~= byte { return .exponentDigits }
            return nil
        case .expectExponentDigits:
            if ASCII.zero...ASCII.nine ~= byte { return .exponentDigits }
            return nil
        case .exponentDigits:
            if ASCII.zero...ASCII.nine ~= byte { return .exponentDigits }
            return nil
        }
    }

    private static func numberCanTerminate(_ state: NumberState) -> Bool {
        switch state {
        case .leadingZero, .intDigits, .fracDigits, .exponentDigits:
            return true
        case .afterMinus, .afterPoint, .expectExponentDigitsOrSign, .expectExponentDigits:
            return false
        }
    }

    private static func isNumberTerminator(_ byte: UInt8) -> Bool {
        isWhitespace(byte) || byte == ASCII.comma || byte == ASCII.closeBrace || byte == ASCII.closeBracket
    }

    // MARK: - Literals (`true`/`false`/`null`)

    private mutating func advanceLiteral(byte: UInt8, literal: Literal, matched: Int) -> Bool {
        let word: [UInt8]
        switch literal {
        case .trueLiteral: word = ASCII.trueBytes
        case .falseLiteral: word = ASCII.falseBytes
        case .nullLiteral: word = ASCII.nullBytes
        }
        guard matched < word.count, byte == word[matched] else { return false }
        let nextMatched = matched + 1
        if nextMatched == word.count {
            lexeme = .none
            completedValue()
        } else {
            lexeme = .literal(literal, matched: nextMatched)
        }
        return true
    }

    // MARK: - Enum literal matching

    /// Matches `byte` against every literal in `candidates` still consistent with the `position`
    /// bytes already consumed.
    ///
    /// String/boolean/null literals are SELF-TERMINATING (their JSON text includes its own closing
    /// delimiter — the closing `"`, or the last letter of `true`/`false`/`null`): because the
    /// compiler guarantees enum literal strings contain no internal `"`, no distinct literal's
    /// bytes can be a strict prefix of another's (a shared prefix followed by one literal's closing
    /// `"` forces the SAME position in any longer literal to also be `"`, which — since that
    /// longer literal's only unescaped `"` bytes are its own open/close — would require equal
    /// length, a contradiction). So whenever such a literal's final byte is matched, exactly one
    /// candidate survives and the value completes immediately.
    ///
    /// Number literals have no closing delimiter: matching a full number-literal's digits leaves it
    /// "closed but pending" (mirrors `advanceNumber`) until a genuine terminator byte confirms no
    /// further digit extends it to a DIFFERENT (longer) candidate in the same enum (e.g. `1` vs `12`).
    private mutating func advanceEnum(byte: UInt8, candidates: [Int], position: Int) -> Bool {
        let continuing = candidates.filter {
            position < table.literalBytes[$0].count && table.literalBytes[$0][position] == byte
        }
        if !continuing.isEmpty {
            let newPosition = position + 1
            let firstByte = table.literalBytes[continuing[0]][0]
            let selfTerminating =
                firstByte == ASCII.quote || firstByte == ASCII.t || firstByte == ASCII.f || firstByte == ASCII.n
            if selfTerminating {
                let closedNow = continuing.filter { table.literalBytes[$0].count == newPosition }
                if !closedNow.isEmpty {
                    guard closedNow.count == 1, continuing.count == 1 else { return false }  // fail closed
                    lexeme = .none
                    completedValue()
                    return true
                }
            }
            lexeme = .enumMatch(candidates: continuing, position: newPosition)
            return true
        }
        guard Self.isNumberTerminator(byte) else { return false }
        guard candidates.contains(where: { table.literalBytes[$0].count == position }) else { return false }
        lexeme = .none
        completedValue()
        return advanceStructural(byte: byte)  // the terminator byte was never part of the literal
    }

    // MARK: - Shared byte classes

    private static func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
    }

    private static func isHexDigit(_ byte: UInt8) -> Bool {
        (ASCII.zero...ASCII.nine ~= byte) || (0x41...0x46 ~= byte) || (0x61...0x66 ~= byte)
    }
}

/// ASCII byte constants — a private copy of `JSONObjectConstraintAutomaton.swift`'s `ASCII` enum
/// (that one is `fileprivate` to its own file, so it cannot be shared without changing the frozen
/// sibling file).
fileprivate enum ASCII {
    static let quote: UInt8 = 0x22
    static let backslash: UInt8 = 0x5C
    static let slash: UInt8 = 0x2F
    static let colon: UInt8 = 0x3A
    static let comma: UInt8 = 0x2C
    static let openBrace: UInt8 = 0x7B
    static let closeBrace: UInt8 = 0x7D
    static let openBracket: UInt8 = 0x5B
    static let closeBracket: UInt8 = 0x5D
    static let minus: UInt8 = 0x2D
    static let dot: UInt8 = 0x2E
    static let plus: UInt8 = 0x2B
    static let zero: UInt8 = 0x30
    static let nine: UInt8 = 0x39
    static let one: UInt8 = 0x31
    static let e: UInt8 = 0x65
    static let E: UInt8 = 0x45
    static let t: UInt8 = 0x74
    static let f: UInt8 = 0x66
    static let n: UInt8 = 0x6E
    static let b: UInt8 = 0x62
    static let r: UInt8 = 0x72
    static let u: UInt8 = 0x75
    static let trueBytes: [UInt8] = Array("true".utf8)
    static let falseBytes: [UInt8] = Array("false".utf8)
    static let nullBytes: [UInt8] = Array("null".utf8)
}

/// Stage 3a: lets the shared `JSONObjectConstraintTrie` DFS and `JSONSchemaConstraintTable` drive
/// `JSONSchemaAutomaton` through the same generic `ByteWalkAutomaton` surface `JSONObjectAutomaton`
/// conforms to (see that extension's doc comment).
extension JSONSchemaAutomaton: ByteWalkAutomaton {}
