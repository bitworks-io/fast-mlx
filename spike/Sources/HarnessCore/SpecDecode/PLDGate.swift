import Foundation

/// A self-managing on/off gate for prompt-lookup speculative decoding. PLD only pays off when
/// drafts tend to be accepted; on a workload where the drafter rarely matches (low acceptance),
/// every verify forward still costs the same but emits fewer tokens per pass than plain decoding
/// would have taken steps for, wasting the extra compute. The gate tracks a sliding window of
/// accepted-tokens-per-verify and disables PLD once the windowed mean drops too low, then
/// periodically re-enables it after a cooldown to probe whether the workload has become
/// repetitive again (e.g. the model started quoting a long passage).
///
/// Deterministic: driven entirely by `record(accepted:)` calls, no wall-clock or randomness.
public struct PLDGate: Sendable {
    /// Number of most-recent `record` samples the mean is computed over.
    public let window: Int
    /// Windowed mean accepted-per-verify below which the gate disables.
    public let minAcceptPerStep: Double
    /// Number of `record` calls while disabled before the gate re-enables to probe recovery.
    public let cooldown: Int

    public private(set) var isEnabled: Bool = true

    private var samples: [Int] = []
    private var cooldownCount: Int = 0

    public init(window: Int = 32, minAcceptPerStep: Double = 0.25, cooldown: Int = 16) {
        precondition(window > 0, "window must be positive")
        precondition(cooldown > 0, "cooldown must be positive")
        self.window = window
        self.minAcceptPerStep = minAcceptPerStep
        self.cooldown = cooldown
    }

    /// Feed the accepted-draft count from one verify (or, while disabled, one normal decode
    /// step — the count only matters when PLD is actively drafting).
    public mutating func record(accepted: Int) {
        if isEnabled {
            samples.append(accepted)
            if samples.count > window { samples.removeFirst() }
            // Only judge once a full window of evidence has accrued — a single low sample
            // shouldn't flip the gate; a sustained low windowed mean should.
            guard samples.count == window else { return }
            let mean = Double(samples.reduce(0, +)) / Double(samples.count)
            if mean < minAcceptPerStep {
                isEnabled = false
                samples.removeAll()
                cooldownCount = 0
            }
        } else {
            cooldownCount += 1
            if cooldownCount >= cooldown {
                isEnabled = true
                cooldownCount = 0
                samples.removeAll()
            }
        }
    }
}
