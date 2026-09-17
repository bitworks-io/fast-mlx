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
        if Self.isWhitespace(byte) {
            whitespaceRun += 1
            return whitespaceRun <= maxConsecutiveWhitespace
        }
        whitespaceRun = 0

        if afterTopLevel {
            return false  // trailing non-whitespace content after the top-level object closed
        }

        if beforeTopLevel {
            guard byte == ASCII.openBrace else { return false }
            guard stack.count < Self.maxDepth else { return false }
            beforeTopLevel = false
            stack.append(.objectExpectKeyOrEnd)
            return true
        }

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
