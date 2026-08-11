import Foundation

/// A self-managing on/off gate for prompt-lookup speculative decoding. PLD only pays off when
/// drafts tend to be accepted; on a workload where the drafter rarely matches (low acceptance),
/// every verify forward still costs the same but emits fewer tokens per pass than plain decoding
/// would have taken steps for, wasting the extra compute. The gate tracks a sliding window of
/// accepted-tokens-per-enabled-step and disables PLD once the windowed mean drops too low, then
/// periodically re-enables it after a cooldown to probe whether the workload has become
/// repetitive again (e.g. the model started quoting a long passage).
///
/// Deterministic: driven entirely by `record(accepted:)` calls, no wall-clock or randomness.
public struct PLDGate: Sendable {
    /// Number of most-recent `record` samples the mean is computed over.
    public let window: Int
    /// Number of samples required before a clearly bad partial window may disable PLD.
    public let minimumSamples: Int
    /// Windowed mean accepted-per-enabled-step below which the gate disables.
    public let minAcceptPerStep: Double
    /// Number of `record` calls while disabled before the gate re-enables to probe recovery.
    public let cooldown: Int

    public private(set) var isEnabled: Bool = true

    private var samples: [Int] = []
    private var cooldownCount: Int = 0

    public init(
        window: Int = 8,
        minimumSamples: Int? = nil,
        minAcceptPerStep: Double = 0.5,
        cooldown: Int = 32
    ) {
        precondition(window > 0, "window must be positive")
        precondition(cooldown > 0, "cooldown must be positive")
        precondition(minAcceptPerStep >= 0, "minAcceptPerStep must be nonnegative")
        let resolvedMinimumSamples = minimumSamples ?? min(window, 4)
        precondition(
            resolvedMinimumSamples > 0 && resolvedMinimumSamples <= window,
            "minimumSamples must be in 1...window")
        self.window = window
        self.minimumSamples = resolvedMinimumSamples
        self.minAcceptPerStep = minAcceptPerStep
        self.cooldown = cooldown
    }

    /// Feed the accepted-draft count from one enabled decode step (zero when lookup produced no
    /// draft). While disabled, the value is ignored and each call advances the cooldown clock.
    public mutating func record(accepted: Int) {
        precondition(accepted >= 0, "accepted must be nonnegative")
        if isEnabled {
            samples.append(accepted)
            if samples.count > window { samples.removeFirst() }
            // Avoid one-sample overreaction, but do not require a full long window before
            // stepping aside on an obviously cold request. Once full, this remains a normal
            // rolling-window mean over the newest `window` samples.
            guard samples.count >= minimumSamples else { return }
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
