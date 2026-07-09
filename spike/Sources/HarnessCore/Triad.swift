/// Longest identical prefix of two token streams.
public func identicalPrefix(_ a: [Int], _ b: [Int]) -> Int {
    var i = 0; while i < a.count && i < b.count && a[i] == b[i] { i += 1 }; return i
}

/// Equivalence vs a reference at temp=0. First-N (not full) — INT4/MoE float-reduction order
/// legitimately flips near-tie argmax past a per-family horizon (backlog). `minPrefix` is the
/// documented, tunable gate; below it means a real bug, not numerics.
public struct EquivalenceCheck {
    public let minPrefix: Int
    public init(minPrefix: Int = 30) { self.minPrefix = minPrefix }
    public func evaluate(candidate: [Int], reference: [Int]) -> (prefix: Int, passed: Bool) {
        let p = identicalPrefix(candidate, reference); return (p, p >= minPrefix)
    }
}

/// Engagement DELTA (not presence): the run's structured counter must strictly increase.
public struct EngagementCheck {
    public let marker: String; public let floor: Int
    public init(marker: String, floor: Int = 1) { self.marker = marker; self.floor = floor }
    public func passed(before: Int, after: Int) -> Bool { (after - before) >= floor }
}

/// Effectiveness floor: a feature can engage every request yet accept ~0% and degenerately fall back.
public struct AcceptanceCheck {
    public let floor: Double
    public init(floor: Double) { self.floor = floor }
    public func passed(rate: Double) -> Bool { rate >= floor }
}

public struct TriadVerdict: Sendable {
    public let equivalencePrefix: Int; public let engaged: Bool; public let acceptanceOK: Bool?
    public var passed: Bool { equivalencePrefix >= 0 /* set by caller */ && engaged && (acceptanceOK ?? true) }
    public init(equivalencePrefix: Int, engaged: Bool, acceptanceOK: Bool?) {
        self.equivalencePrefix = equivalencePrefix; self.engaged = engaged; self.acceptanceOK = acceptanceOK
    }
}
