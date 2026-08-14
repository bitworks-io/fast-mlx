import Foundation

/// Pure, MLX-free decode measurement from stream timestamps.
/// Decode rate deliberately EXCLUDES prefill (measured from first token onward),
/// matching the carry-forward bench methodology (rate from the live stream, not usage fields).
public struct DecodeMetrics: Sendable {
    public let ttftSeconds: Double
    public let generatedTokenCount: Int
    public let decodeTokensPerSecond: Double?

    public init(submitTime: Double, tokenTimes: [Double]) {
        precondition(!tokenTimes.isEmpty, "need at least one token")
        self.generatedTokenCount = tokenTimes.count
        self.ttftSeconds = tokenTimes[0] - submitTime
        if tokenTimes.count >= 2 {
            let decodeSpan = tokenTimes.last! - tokenTimes[0]
            let gaps = Double(tokenTimes.count - 1)
            self.decodeTokensPerSecond = decodeSpan > 0 ? gaps / decodeSpan : nil
        } else {
            self.decodeTokensPerSecond = nil
        }
    }
}
