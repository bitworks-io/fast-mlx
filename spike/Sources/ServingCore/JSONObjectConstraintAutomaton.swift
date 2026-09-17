import Foundation

/// A byte-level RFC 8259 JSON pushdown automaton constrained to a top-level object.
///
/// State is entirely value types (`Bool`, `Int`, and arrays/enums of those), so
/// `JSONObjectAutomaton` is cheap to copy, `Hashable` (for mask caching), and `Sendable`. Every
/// transition is driven one raw byte at a time via `advance(byte:)`, which never throws — it
/// returns `false` and moves the automaton to a permanent rejecting state on an invalid byte, so
/// callers that only want a yes/no probe (the trie DFS) avoid Swift error-handling overhead in a
/// hot path.
public struct JSONObjectAutomaton: Hashable, Sendable {
    /// Default consecutive-structural-whitespace cap (the production value): a degenerate
    /// generation cannot stall forever inside "always allowed" whitespace. Overridable per
    /// instance — see `init(maxConsecutiveWhitespace:)` — so a test can disable the cap (e.g.
    /// `Int.max`) to prove it, rather than some other automaton detail, is what forces
    /// termination.
    public static let defaultMaxConsecutiveWhitespace = 16
    /// Maximum object/array nesting depth (the top-level object itself counts as depth 1).
    public static let maxDepth = 64

    private let maxConsecutiveWhitespace: Int

    fileprivate enum Frame: Hashable, Sendable {
        case objectExpectKeyOrEnd  // just after '{': a string key, or '}' if the object is empty
        case objectExpectKey  // just after ',': a string key (NOT '}')
        case objectExpectColon  // just after a key string closed
        case objectExpectValue  // just after ':'
        case objectExpectCommaOrEnd  // just after a value
        case arrayExpectValueOrEnd  // just after '[': a value, or ']' if the array is empty
        case arrayExpectValue  // just after ','
        case arrayExpectCommaOrEnd  // just after a value
    }

    fileprivate enum StringState: Hashable, Sendable {
        case normal
        case escape
        case unicodeEscape(remaining: Int)  // \uXXXX: 4 hex digits remaining, then 3, 2, 1
        case continuation(remaining: Int, min: UInt8, max: UInt8)  // raw UTF-8 continuation bytes
    }

    fileprivate enum NumberState: Hashable, Sendable {
        case afterMinus
        case leadingZero
        case intDigits
        case afterPoint
        case fracDigits
        case expectExponentDigitsOrSign  // just consumed 'e'/'E'
        case expectExponentDigits  // just consumed the exponent's '+'/'-'
        case exponentDigits
    }

    fileprivate enum Literal: Hashable, Sendable { case trueLiteral, falseLiteral, nullLiteral }

    fileprivate enum Lexeme: Hashable, Sendable {
        case none
        case string(StringState, isKey: Bool)
        case number(NumberState)
        case literal(Literal, matched: Int)
    }

    private var stack: [Frame] = []
    private var lexeme: Lexeme = .none
    private var beforeTopLevel = true
    private var afterTopLevel = false
    private var whitespaceRun = 0
    // At most one line-break EVENT is allowed per structural whitespace run (defect B: unbounded
    // blank lines/indentation otherwise slip through even under the 16-byte cap). `\n`, a lone
    // `\r`, and an immediately-paired `\r\n` each count as exactly one event; `pendingCR` tracks
    // "the previous byte in this run was a bare `\r`, which may still merge with a following `\n`"
    // so that merge doesn't double-count. Both fields reset with `whitespaceRun` — see every site
    // that zeroes `whitespaceRun`.
    private var lineBreakSeenInRun = false
    private var pendingCR = false
    private var rejected = false

    /// - Parameter maxConsecutiveWhitespace: The consecutive-structural-whitespace cap. Defaults
    ///   to the production value (`defaultMaxConsecutiveWhitespace`); a test may override it (e.g.
    ///   `Int.max` to disable it) to prove the cap is load-bearing.
    public init(maxConsecutiveWhitespace: Int = JSONObjectAutomaton.defaultMaxConsecutiveWhitespace) {
        self.maxConsecutiveWhitespace = maxConsecutiveWhitespace
    }

    /// True once the top-level object has closed and only whitespace (or EOS) may follow.
    public var isComplete: Bool {
        !rejected && afterTopLevel && lexeme == .none
    }

    /// Advances the automaton by one raw byte. Returns `false` (and leaves the automaton
    /// permanently rejecting) if `byte` is not valid at the current position.
    @discardableResult
    public mutating func advance(byte: UInt8) -> Bool {
        guard !rejected else { return false }
        guard advanceInner(byte: byte) else {
            rejected = true
            return false
        }
        return true
    }

    /// Non-mutating probe: the automaton that would result from consuming `byte`, or `nil` if
    /// `byte` is invalid here.
    public func advancing(byte: UInt8) -> JSONObjectAutomaton? {
        var copy = self
        return copy.advance(byte: byte) ? copy : nil
    }

    // MARK: - Allocation-free walk (trie DFS)

    /// Opaque per-byte undo token for `tryAdvanceForWalk(byte:)`/`undoForWalk(_:)`: captures
    /// exactly the scalar fields plus the (at most one push, pop, or same-index top replacement)
    /// `stack` delta a single byte transition can make, so the caller can restore the automaton to
    /// its pre-call state in O(1) with no array copy. A naive "copy self, then mutate the copy"
    /// probe (as `advancing(byte:)` does) forces Swift's copy-on-write to materialize a real copy
    /// of `stack` the moment the copy's mutation diverges from the original — which is exactly the
    /// per-byte cost a trie DFS visiting many branches must avoid. This type intentionally exposes
    /// no public initializer or fields: it is meaningful only as the token `tryAdvanceForWalk`
    /// hands back to `undoForWalk`.
    struct WalkUndo {
        fileprivate let lexeme: Lexeme
        fileprivate let beforeTopLevel: Bool
        fileprivate let afterTopLevel: Bool
        fileprivate let whitespaceRun: Int
        fileprivate let lineBreakSeenInRun: Bool
        fileprivate let pendingCR: Bool
        fileprivate let rejected: Bool
        fileprivate let stackCountBefore: Int
        fileprivate let stackTopBefore: Frame?
    }

    /// Trial advance for the trie DFS: mutates `self` in place. On success, returns an undo token
    /// that `undoForWalk(_:)` later restores; on rejection, `self` is left exactly as it was
    /// (internally rolled back) and `nil` is returned. Unlike `advance(byte:)`, this never sets the
    /// permanent `rejected` latch on a live walker — it is meant to be called many times over the
    /// SAME automaton instance during a single DFS, alternating with `undoForWalk` as the walk
    /// backtracks, not once per terminal outcome.
    mutating func tryAdvanceForWalk(byte: UInt8) -> WalkUndo? {
        guard !rejected else { return nil }
        let undo = WalkUndo(
            lexeme: lexeme, beforeTopLevel: beforeTopLevel, afterTopLevel: afterTopLevel,
            whitespaceRun: whitespaceRun, lineBreakSeenInRun: lineBreakSeenInRun, pendingCR: pendingCR,
            rejected: rejected,
            stackCountBefore: stack.count, stackTopBefore: stack.last)
        guard advanceInner(byte: byte) else {
            restoreForWalk(undo)
            return nil
        }
        return undo
    }

    /// Restores exactly what the matching `tryAdvanceForWalk(byte:)` call touched.
    mutating func undoForWalk(_ undo: WalkUndo) {
        restoreForWalk(undo)
    }

    /// A single byte transition changes `stack` by at most one push, one pop, or one same-index
    /// top replacement — see `advanceStructural`/`beginValue`: the ONLY paths that touch `stack` at
    /// all either `append`/`removeLast` exactly once, or reassign `stack[stack.count - 1]` exactly
    /// once (sometimes immediately followed by a push from `beginValue`, e.g. `{"a":{` replaces the
    /// outer frame's `.objectExpectValue` with `.objectExpectCommaOrEnd` THEN pushes the inner
    /// object's frame — both in the same byte). That means the count delta alone doesn't always
    /// tell the whole story: a push can co-occur with the pre-existing top frame having been
    /// replaced, so undoing a push must restore both the removed frame's absence AND the
    /// possibly-mutated frame beneath it. A pop never co-occurs with a same-call top replacement
    /// (the removeLast/append pop sites never reassign `stack[stack.count - 1]` first), so undoing
    /// a pop only needs to put the popped frame back.
    private mutating func restoreForWalk(_ undo: WalkUndo) {
        lexeme = undo.lexeme
        beforeTopLevel = undo.beforeTopLevel
        afterTopLevel = undo.afterTopLevel
        whitespaceRun = undo.whitespaceRun
        lineBreakSeenInRun = undo.lineBreakSeenInRun
        pendingCR = undo.pendingCR
        rejected = undo.rejected
        if stack.count == undo.stackCountBefore + 1 {
            stack.removeLast()
            if let top = undo.stackTopBefore, !stack.isEmpty {
                stack[stack.count - 1] = top
            }
        } else if stack.count == undo.stackCountBefore - 1 {
            if let top = undo.stackTopBefore {
                stack.append(top)
            }
        } else if stack.count == undo.stackCountBefore {
            if let top = undo.stackTopBefore, !stack.isEmpty {
                stack[stack.count - 1] = top
            }
        }
    }

    // MARK: - String-run lexical key (slice 1f miss-path optimization)

    /// Opaque key for the string-internal lexical sub-state (escape progress, `\uXXXX` countdown,
    /// pending UTF-8 continuation expectation) — see `stringLexicalKey`. Two automaton states with
    /// equal keys accept EXACTLY the same set of "STAY" tokens (tokens fully consumed while
    /// remaining inside the string): `advanceString(byte:state:isKey:)` never reads or writes
    /// `stack`, and never branches its accept/reject decision on `isKey` (only the terminating
    /// quote's SIDE EFFECT — whether `completedValue()` runs — depends on `isKey`, not whether the
    /// quote byte itself is accepted). `StringState` is only ~13 distinct reachable values (1
    /// normal + 1 escape + 4 unicode-escape countdown steps + 7 distinct UTF-8 continuation specs —
    /// `remaining` decrements always reset `min`/`max` to `0x80`/`0xBF`, so the "mid-sequence"
    /// states collapse onto the same 7 specs reachable directly after a lead byte), which is why
    /// `JSONObjectConstraintTable` can afford to precompute each one's full STAY bitset once, ever.
    struct StringLexicalKey: Hashable, Sendable {
        fileprivate let state: StringState
    }

    /// Non-`nil` iff the automaton is currently inside a JSON string (any sub-state) — see
    /// `StringLexicalKey`.
    var stringLexicalKey: StringLexicalKey? {
        guard case .string(let state, _) = lexeme else { return nil }
        return StringLexicalKey(state: state)
    }

    // MARK: - Mask cache key

    /// Opaque, `Hashable` mask-cache key: the automaton state truncated to what a token of at most
    /// `maxTokenBytes` bytes can ever observe from here. A single token's bytes can pop at most
    /// `maxTokenBytes` stack frames (one pop per closing `}`/`]` byte, and the token has at most
    /// that many bytes total), and the trie DFS that computes a mask never walks deeper than
    /// `maxTokenBytes` edges from the root — so two automaton states that agree on the lexical
    /// state, the whitespace run, and the top `min(depth, maxTokenBytes)` stack frames produce the
    /// IDENTICAL allowed-mask, regardless of what (if anything) sits below that window. `hasHiddenFrames`
    /// distinguishes "the visible frames are the WHOLE stack" (so popping through all of them
    /// bottoms out at the empty stack, i.e. can complete the document) from "there is at least one
    /// more frame below the window" (so the deepest visible pop can never itself bottom out within
    /// a single token) — without that flag, two states with the same visible frames but different
    /// bottoming-out behavior would incorrectly collide. This is an EXACT reduction, not an
    /// approximation: every included field is either already used unchanged (lexeme, whitespaceRun,
    /// beforeTopLevel/afterTopLevel/rejected) or is the precise subset of `stack` any within-budget
    /// token walk could ever read.
    struct MaskCacheKey: Hashable {
        fileprivate let lexeme: Lexeme
        fileprivate let beforeTopLevel: Bool
        fileprivate let afterTopLevel: Bool
        fileprivate let whitespaceRun: Int
        // Like `whitespaceRun`, both decide whether a further line-break byte is accepted, so
        // states differing only in them must not share a cache entry (differential test + mutation).
        fileprivate let lineBreakSeenInRun: Bool
        fileprivate let pendingCR: Bool
        fileprivate let rejected: Bool
        fileprivate let visibleFrames: [Frame]
        fileprivate let hasHiddenFrames: Bool
    }

    /// Builds this state's `MaskCacheKey` against a table whose longest `.bytes` token is
    /// `maxTokenBytes` bytes (see `JSONObjectConstraintTrie.maxTokenByteLength`).
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
            lexeme: lexeme, beforeTopLevel: beforeTopLevel, afterTopLevel: afterTopLevel,
            whitespaceRun: whitespaceRun, lineBreakSeenInRun: lineBreakSeenInRun, pendingCR: pendingCR,
            rejected: rejected,
            visibleFrames: visibleFrames, hasHiddenFrames: hasHiddenFrames)
    }

    /// Convenience for whole-string acceptance tests: feeds `string`'s UTF-8 bytes through a
    /// fresh automaton and reports whether the result is both valid and complete.
    public static func accepts(_ string: String) -> Bool {
        accepts(bytes: Array(string.utf8))
    }

    /// Like `accepts(_:String)`, but over raw bytes — needed for fixtures that are not valid
    /// UTF-8 text (a lone/invalid continuation byte), which a Swift `String` literal cannot hold.
    public static func accepts(bytes: [UInt8]) -> Bool {
        var automaton = JSONObjectAutomaton()
        for byte in bytes {
            guard automaton.advance(byte: byte) else { return false }
        }
        return automaton.isComplete
    }

    // MARK: - Byte dispatch

    private mutating func advanceInner(byte: UInt8) -> Bool {
        switch lexeme {
        case .string(let state, let isKey):
            return advanceString(byte: byte, state: state, isKey: isKey)
        case .number(let state):
            return advanceNumber(byte: byte, state: state)
        case .literal(let literal, let matched):
            return advanceLiteral(byte: byte, literal: literal, matched: matched)
        case .none:
            return advanceStructural(byte: byte)
        }
    }

    private mutating func advanceStructural(byte: UInt8) -> Bool {
        // No whitespace is ever valid before the top-level `{` or after the top-level object
        // closes: only `{` (beforeTopLevel) or nothing but EOS (afterTopLevel) may appear there,
        // so whitespace handling below only ever runs strictly BETWEEN those two points.
        if afterTopLevel {
            return false  // only EOS may follow — see `isComplete`
        }
        if beforeTopLevel {
            guard byte == ASCII.openBrace else { return false }
            guard stack.count < Self.maxDepth else { return false }
            beforeTopLevel = false
            stack.append(.objectExpectKeyOrEnd)
            return true
        }

        if Self.isWhitespace(byte) {
            whitespaceRun += 1
            guard whitespaceRun <= maxConsecutiveWhitespace else { return false }
            switch byte {
            case 0x0A:  // '\n'
                if pendingCR {
                    pendingCR = false  // completes a '\r\n' pair: already counted by the '\r'
                } else {
                    guard !lineBreakSeenInRun else { return false }
                    lineBreakSeenInRun = true
                }
            case 0x0D:  // '\r'
                guard !lineBreakSeenInRun else { return false }
                lineBreakSeenInRun = true
                pendingCR = true
            default:  // space or tab: does not end a pending '\r's chance to pair, but a byte
                // between '\r' and '\n' means they are no longer adjacent, so no pairing.
                pendingCR = false
            }
            return true
        }
        whitespaceRun = 0
        lineBreakSeenInRun = false
        pendingCR = false

        guard let top = stack.last else { return false }
        switch top {
        case .objectExpectKeyOrEnd:
            if byte == ASCII.closeBrace {
                stack.removeLast()
                completedValue()
                return true
            }
            if byte == ASCII.quote {
                stack[stack.count - 1] = .objectExpectColon
                lexeme = .string(.normal, isKey: true)
                return true
            }
            return false
        case .objectExpectKey:
            guard byte == ASCII.quote else { return false }
            stack[stack.count - 1] = .objectExpectColon
            lexeme = .string(.normal, isKey: true)
            return true
        case .objectExpectColon:
            guard byte == ASCII.colon else { return false }
            stack[stack.count - 1] = .objectExpectValue
            return true
        case .objectExpectValue:
            stack[stack.count - 1] = .objectExpectCommaOrEnd
            return beginValue(byte: byte)
        case .objectExpectCommaOrEnd:
            if byte == ASCII.comma {
                stack[stack.count - 1] = .objectExpectKey
                return true
            }
            if byte == ASCII.closeBrace {
                stack.removeLast()
                completedValue()
                return true
            }
            return false
        case .arrayExpectValueOrEnd:
            if byte == ASCII.closeBracket {
                stack.removeLast()
                completedValue()
                return true
            }
            stack[stack.count - 1] = .arrayExpectCommaOrEnd
            return beginValue(byte: byte)
        case .arrayExpectValue:
            stack[stack.count - 1] = .arrayExpectCommaOrEnd
            return beginValue(byte: byte)
        case .arrayExpectCommaOrEnd:
            if byte == ASCII.comma {
                stack[stack.count - 1] = .arrayExpectValue
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

    private mutating func beginValue(byte: UInt8) -> Bool {
        switch byte {
        case ASCII.quote:
            lexeme = .string(.normal, isKey: false)
            return true
        case ASCII.openBrace:
            guard stack.count < Self.maxDepth else { return false }
            stack.append(.objectExpectKeyOrEnd)
            return true
        case ASCII.openBracket:
            guard stack.count < Self.maxDepth else { return false }
            stack.append(.arrayExpectValueOrEnd)
            return true
        case ASCII.t:
            lexeme = .literal(.trueLiteral, matched: 1)
            return true
        case ASCII.f:
            lexeme = .literal(.falseLiteral, matched: 1)
            return true
        case ASCII.n:
            lexeme = .literal(.nullLiteral, matched: 1)
            return true
        case ASCII.minus:
            lexeme = .number(.afterMinus)
            return true
        case ASCII.zero:
            lexeme = .number(.leadingZero)
            return true
        case ASCII.one...ASCII.nine:
            lexeme = .number(.intDigits)
            return true
        default:
            return false
        }
    }

    /// Called whenever a value fully closes via its own terminating byte (`}`, `]`, the last
    /// letter of a literal). Number completion is handled separately in `advanceNumber`, since a
    /// number's terminator byte is NOT part of the number and must be reprocessed structurally.
    private mutating func completedValue() {
        if stack.isEmpty {
            afterTopLevel = true
            whitespaceRun = 0
            lineBreakSeenInRun = false
            pendingCR = false
        }
    }

    // MARK: - Strings

    private mutating func advanceString(byte: UInt8, state: StringState, isKey: Bool) -> Bool {
        switch state {
        case .normal:
            if byte == ASCII.quote {
                lexeme = .none
                if !isKey {
                    completedValue()
                }
                return true
            }
            if byte == ASCII.backslash {
                lexeme = .string(.escape, isKey: isKey)
                return true
            }
            if byte < 0x20 {
                return false  // unescaped control character
            }
            if byte < 0x80 {
                return true  // ASCII content byte, stays normal
            }
            guard let (remaining, min, max) = Self.utf8ContinuationSpec(leadByte: byte) else {
                return false
            }
            lexeme = .string(.continuation(remaining: remaining, min: min, max: max), isKey: isKey)
            return true
        case .escape:
            switch byte {
            case ASCII.quote, ASCII.backslash, ASCII.slash, ASCII.b, ASCII.f, ASCII.n, ASCII.r, ASCII.t:
                lexeme = .string(.normal, isKey: isKey)
                return true
            case ASCII.u:
                lexeme = .string(.unicodeEscape(remaining: 4), isKey: isKey)
                return true
            default:
                return false
            }
        case .unicodeEscape(let remaining):
            guard Self.isHexDigit(byte) else { return false }
            lexeme = remaining == 1
                ? .string(.normal, isKey: isKey)
                : .string(.unicodeEscape(remaining: remaining - 1), isKey: isKey)
            return true
        case .continuation(let remaining, let min, let max):
            guard byte >= min && byte <= max else { return false }
            lexeme = remaining == 1
                ? .string(.normal, isKey: isKey)
                : .string(.continuation(remaining: remaining - 1, min: 0x80, max: 0xBF), isKey: isKey)
            return true
        }
    }

    /// Valid first-continuation-byte range for a UTF-8 lead byte, per the Unicode well-formedness
    /// table (excludes overlong encodings, surrogate code points, and codepoints above U+10FFFF).
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
            return nil  // 0x80...0xC1 (stray continuation / overlong), 0xF5...0xFF (out of range)
        }
    }

    // MARK: - Numbers

    private mutating func advanceNumber(byte: UInt8, state: NumberState) -> Bool {
        if let next = Self.numberTransition(state: state, byte: byte) {
            lexeme = .number(next)
            return true
        }
        guard Self.isNumberTerminator(byte), Self.numberCanTerminate(state) else {
            return false
        }
        lexeme = .none
        completedValue()
        return advanceStructural(byte: byte)  // the terminator byte was never part of the number
    }

    private static func numberTransition(state: NumberState, byte: UInt8) -> NumberState? {
        switch state {
        case .afterMinus:
            if byte == ASCII.zero { return .leadingZero }
            if ASCII.one...ASCII.nine ~= byte { return .intDigits }
            return nil
        case .leadingZero:
            if byte == ASCII.dot { return .afterPoint }
            if byte == ASCII.e || byte == ASCII.E { return .expectExponentDigitsOrSign }
            return nil  // no further digits after a leading zero (rejects "01")
        case .intDigits:
            if ASCII.zero...ASCII.nine ~= byte { return .intDigits }
            if byte == ASCII.dot { return .afterPoint }
            if byte == ASCII.e || byte == ASCII.E { return .expectExponentDigitsOrSign }
            return nil
        case .afterPoint:
            if ASCII.zero...ASCII.nine ~= byte { return .fracDigits }
            return nil  // at least one fraction digit is required
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

    // MARK: - Literals

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

    // MARK: - Shared byte classes

    fileprivate static func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
    }

    private static func isHexDigit(_ byte: UInt8) -> Bool {
        (ASCII.zero...ASCII.nine ~= byte) || (0x41...0x46 ~= byte) || (0x61...0x66 ~= byte)
    }
}

/// ASCII byte constants used by the JSON automaton, named for readability at call sites.
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
