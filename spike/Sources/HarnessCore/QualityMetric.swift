import Foundation

/// KL(reference || candidate) in nats, over aligned probability vectors. Clamps for numerical safety.
public func klDivergence(reference p: [Float], candidate q: [Float]) -> Double {
    precondition(p.count == q.count)
    var kl = 0.0
    for i in p.indices where p[i] > 0 {
        let qi = max(Double(q[i]), 1e-12); kl += Double(p[i]) * (log(Double(p[i])) - log(qi))
    }
    return kl
}

public protocol QualityMetric: Sendable { var name: String { get }
    /// Measure candidate-vs-reference over a fixed corpus using the driver's logprobs. Lower = closer to reference.
    func measure(driver: EngineDriver, reference: EngineDriver, prompts: [[Int]], config: RunConfig) async throws -> Double }

/// Per-position KLs between two full-vocab RAW-LOGITS matrices (softmaxed here) — the metric's
/// inner loop, exposed so the CLI can report per-position detail (p95, aligned-prefix stats)
/// from the SAME computation the headline median comes from, not a reimplementation.
public func perPositionKLs(reference: [[Float]], candidate: [[Float]]) -> [Double] {
    (0..<min(candidate.count, reference.count)).map {
        klDivergence(reference: softmax(reference[$0]), candidate: softmax(candidate[$0]))
    }
}

/// The metric's median convention (sorted, upper-middle element). Exposed for the same reason.
public func medianOf(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted(); return sorted[sorted.count / 2]
}

public enum QualityMetricError: Error, CustomStringConvertible, Sendable {
    /// The reference generated no tokens for a prompt: zero scoreable positions. Reporting
    /// "0.0" here would be a vacuous pass, so the metric refuses instead.
    case emptyContinuation(prompt: [Int])
    /// A driver returned a different number of teacher-forced rows than forced positions —
    /// it is not fulfilling the contract, and a silently truncated score would lie.
    case rowCountMismatch(side: String, got: Int, expected: Int)
    public var description: String {
        switch self {
        case .emptyContinuation(let prompt):
            return "reference produced an empty continuation for prompt \(prompt) — no positions to score"
        case .rowCountMismatch(let side, let got, let expected):
            return "\(side) returned \(got) teacher-forced rows for \(expected) forced positions"
        }
    }
}

/// One teacher-forced scoring pass: the REFERENCE's greedy continuation is generated once,
/// then BOTH drivers score that same continuation position-by-position. Every row therefore
/// compares distributions over IDENTICAL context — the context-locked basis both the KL and
/// perplexity metrics are defined on. (Free-running comparison compared diverged contexts:
/// 33x distortion on the first real dial point, and no signal at all at 2-bit.)
public struct TeacherForcedScores: Sendable {
    public let continuation: [Int]        // the reference's greedy tokens (may end with eos)
    public let candidateRows: [[Float]]   // full-vocab raw logits, one row per forced position
    public let referenceRows: [[Float]]
}

public func teacherForcedScores(
    driver: EngineDriver, reference: EngineDriver, prompt: [Int], config: RunConfig
) async throws -> TeacherForcedScores {
    let continuation = try await reference.generate(prompt: prompt, config: config).tokens
    guard !continuation.isEmpty else { throw QualityMetricError.emptyContinuation(prompt: prompt) }
    let c = try await driver.logprobs(prompt: prompt, forcedContinuation: continuation, config: config)
    guard c.count == continuation.count else {
        throw QualityMetricError.rowCountMismatch(side: "candidate", got: c.count, expected: continuation.count)
    }
    let r = try await reference.logprobs(prompt: prompt, forcedContinuation: continuation, config: config)
    guard r.count == continuation.count else {
        throw QualityMetricError.rowCountMismatch(side: "reference", got: r.count, expected: continuation.count)
    }
    return TeacherForcedScores(continuation: continuation, candidateRows: c, referenceRows: r)
}

/// Median per-position KL of the candidate's next-token distribution vs the reference's,
/// TEACHER-FORCED on the reference's greedy continuation: all N positions are context-locked
/// by construction, so every position contributes a meaningful sample (no divergence
/// starvation, no diverged-context distortion).
public struct KLDivergenceMetric: QualityMetric {
    public let name = "kl_median"; public init() {}
    public func measure(driver: EngineDriver, reference: EngineDriver, prompts: [[Int]], config: RunConfig) async throws -> Double {
        var kls: [Double] = []
        for p in prompts {
            let s = try await teacherForcedScores(driver: driver, reference: reference, prompt: p, config: config)
            kls.append(contentsOf: perPositionKLs(reference: s.referenceRows, candidate: s.candidateRows))
        }
        return medianOf(kls)
    }
}
func softmax(_ x: [Float]) -> [Float] { let m = x.max() ?? 0; let e = x.map { expf($0 - m) }; let s = e.reduce(0,+); return e.map { $0/s } }
