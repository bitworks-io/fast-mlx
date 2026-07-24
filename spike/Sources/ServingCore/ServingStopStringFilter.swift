import Foundation

/// Incrementally suppresses configured stop strings without publishing an ambiguous suffix.
///
/// The retained buffer is bounded by the longest configured stop string minus one character.
/// Once a stop is observed, the stop string and all later text are permanently suppressed.
public struct ServingStopStringFilter: Sendable {
    public struct Output: Equatable, Sendable {
        public let text: String?
        public let stopped: Bool

        public init(text: String?, stopped: Bool) {
            self.text = text
            self.stopped = stopped
        }
    }

    private let stopStrings: [String]
    private var buffer = ""
    private var stopped = false

    public init(stopStrings: [String]) {
        self.stopStrings = stopStrings.filter { !$0.isEmpty }.sorted {
            if $0.count == $1.count {
                return $0 < $1
            }
            return $0.count > $1.count
        }
    }

    public init(stopStrings: Set<String>) {
        self.init(stopStrings: Array(stopStrings))
    }

    public var bufferedCharacterCount: Int {
        buffer.count
    }

    public mutating func process(_ chunk: String) -> Output {
        guard !stopped else {
            return Output(text: nil, stopped: true)
        }
        guard !stopStrings.isEmpty else {
            return Output(text: chunk.isEmpty ? nil : chunk, stopped: false)
        }

        buffer += chunk
        if let stopRange = earliestStopRange(in: buffer) {
            let text = String(buffer[..<stopRange.lowerBound])
            buffer = ""
            stopped = true
            return Output(text: text.isEmpty ? nil : text, stopped: true)
        }

        let suffixLength = longestStopPrefixSuffixLength(in: buffer)
        let emitEnd = buffer.index(buffer.endIndex, offsetBy: -suffixLength)
        let text = String(buffer[..<emitEnd])
        buffer = String(buffer[emitEnd...])
        return Output(text: text.isEmpty ? nil : text, stopped: false)
    }

    public mutating func finish() -> String? {
        guard !stopStrings.isEmpty, !stopped, !buffer.isEmpty else {
            return nil
        }
        let text = buffer
        buffer = ""
        return text
    }

    private func earliestStopRange(in text: String) -> Range<String.Index>? {
        var earliest: Range<String.Index>?
        for stopString in stopStrings {
            guard let range = text.range(of: stopString) else {
                continue
            }
            if let current = earliest {
                if range.lowerBound < current.lowerBound {
                    earliest = range
                }
            } else {
                earliest = range
            }
        }
        return earliest
    }

    private func longestStopPrefixSuffixLength(in text: String) -> Int {
        var longest = 0
        for stopString in stopStrings {
            let maximumLength = Swift.min(text.count, stopString.count - 1)
            guard maximumLength > longest else {
                continue
            }
            for length in stride(from: maximumLength, through: longest + 1, by: -1) {
                if text.suffix(length) == stopString.prefix(length) {
                    longest = length
                    break
                }
            }
        }
        return longest
    }
}
