/// Longest identical prefix of two token streams.
public func identicalPrefix(_ a: [Int], _ b: [Int]) -> Int {
    var i = 0; while i < a.count && i < b.count && a[i] == b[i] { i += 1 }; return i
}

/// Equivalence vs a reference at temp=0. First-N (not full) — INT4/MoE float-reduction order
/// legitimately flips near-tie argmax past a per-family horizon (backlog). `minPrefix` is the
/// documented, tunable gate; below it means a real bug, not numerics.
public struct EquivalenceCheck: Sendable {
    public let minPrefix: Int
    public init(minPrefix: Int = 30) { self.minPrefix = minPrefix }
    public func evaluate(candidate: [Int], reference: [Int]) -> (prefix: Int, passed: Bool) {
        let p = identicalPrefix(candidate, reference); return (p, p >= minPrefix)
    }
}

/// Engagement DELTA (not presence): the run's structured counter must strictly increase.
public struct EngagementCheck: Sendable {
    public let marker: String; public let floor: Int
    public init(marker: String, floor: Int = 1) { self.marker = marker; self.floor = floor }
    public func passed(before: Int, after: Int) -> Bool { (after - before) >= floor }
}

/// Effectiveness floor: a feature can engage every request yet accept ~0% and degenerately fall back.
public struct AcceptanceCheck: Sendable {
    public let floor: Double
    public init(floor: Double) { self.floor = floor }
    public func passed(rate: Double) -> Bool { rate >= floor }
}

// MARK: - lossy-tier triad variant (Task 4)

/// The equivalence bar an aggressive-quantization tier (e.g. a 2-bit KV cache) is actually held
/// to. `exact` is today's identical-prefix gate — appropriate while KV stays at fp16/int8, where a
/// short prefix signals a real bug. `lossy` accepts that a short prefix is EXPECTED at aggressive
/// tiers and substitutes a different floor: did it crash, did it emit non-finite logits, and does
/// a fixed known-answer canary still land.
public enum TriadMode: String, Sendable, Equatable, CaseIterable {
    case exact
    case lossy
}

/// Selects `exact` vs `lossy` by KV-quant tier. `nil`/`"fp16"` is today's only implemented tier
/// (exact-capable); any other named tier (a future 2-bit TurboQuant tier, say) uses the lossy bar
/// instead of failing `minPrefix` outright — which is exactly what happens today (`verify` at a
/// 2-bit tier just fails `minPrefix=30`, collapsing "expected quantization loss" and "real bug"
/// into the same failure).
public func triadMode(forKVQuantTier tier: String?) -> TriadMode {
    switch tier {
    case nil, "fp16": return .exact
    default: return .lossy
    }
}

/// A fixed prompt/expected-substring pair — the lossy tier's cheap "still basically working"
/// signal when a full identical-prefix match is not the right bar. Reuses the harness's existing
/// known-good prompt/answer from the equivalence work (`knownGoodPrompt` in the CLI) rather than
/// inventing a new one.
public struct CoherenceCanary: Sendable {
    public let prompt: String
    public let mustContain: String
    public init(prompt: String, mustContain: String) { self.prompt = prompt; self.mustContain = mustContain }
    public func passed(_ output: String) -> Bool { output.contains(mustContain) }
    public static let capitalOfFrance = CoherenceCanary(prompt: "The capital of France is", mustContain: "Paris")
}

/// Lossy-tier equivalence: non-crash (produced at least `minPrefix` tokens), non-NaN (no
/// non-finite value anywhere in the scored logits), and the coherence canary's answer still
/// contains the expected substring. All three must hold — any one failing is a real regression,
/// not expected quantization loss.
public struct LossyEquivalenceCheck: Sendable {
    public let minPrefix: Int
    public init(minPrefix: Int = 1) { self.minPrefix = minPrefix }
    public func evaluate(prefix: Int, allFinite: Bool, canaryPassed: Bool) -> (passed: Bool, reasons: [String]) {
        var reasons: [String] = []
        if prefix < minPrefix { reasons.append("prefix \(prefix) < minPrefix \(minPrefix) (crashed or produced no tokens)") }
        if !allFinite { reasons.append("non-finite (NaN/Inf) value in scored logits") }
        if !canaryPassed { reasons.append("coherence canary failed (answer missing expected substring)") }
        return (reasons.isEmpty, reasons)
    }
}

public struct TriadVerdict: Sendable {
    /// Caller computes this via `EquivalenceCheck.evaluate(...).passed` — the verdict never derives
    /// pass/fail from a raw prefix count itself (a token count is always >= 0, which made the prior
    /// `equivalencePrefix >= 0` gate vacuous).
    public let equivalenceOK: Bool; public let engaged: Bool; public let acceptanceOK: Bool?
    public var passed: Bool { equivalenceOK && engaged && (acceptanceOK ?? true) }
    public init(equivalenceOK: Bool, engaged: Bool, acceptanceOK: Bool?) {
        self.equivalenceOK = equivalenceOK; self.engaged = engaged; self.acceptanceOK = acceptanceOK
    }
}
