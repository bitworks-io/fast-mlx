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

/// Like `TeacherForcedScores` but for an EXPLICIT continuation scored at a bounded, sampled
/// subset of positions — the long-context path. Instead of generating a short continuation and
/// forcing it, a long-context corpus entry is teacher-forced AGAINST ITSELF (its own tokenized
/// text is the continuation, wikitext-perplexity style): this is what makes a >=4K-token entry
/// meaningful — quantization loss accrues with context, so the signal lives deep in the
/// sequence, not in a handful of positions appended to a long prompt.
public struct TeacherForcedScoresAtPositions: Sendable {
    public let positions: [Int]           // ascending indices into `continuation`
    public let forcedTokens: [Int]        // continuation[p] for each scored position p
    public let candidateRows: [[Float]]   // full-vocab raw logits, one row per position
    public let referenceRows: [[Float]]
}

public func teacherForcedScoresAtSampledPositions(
    driver: EngineDriver, reference: EngineDriver,
    prompt: [Int], continuation: [Int], positions: [Int], config: RunConfig
) async throws -> TeacherForcedScoresAtPositions {
    guard !continuation.isEmpty else { throw QualityMetricError.emptyContinuation(prompt: prompt) }
    let c = try await driver.logprobs(prompt: prompt, forcedContinuation: continuation, atPositions: positions, config: config)
    guard c.count == positions.count else {
        throw QualityMetricError.rowCountMismatch(side: "candidate", got: c.count, expected: positions.count)
    }
    let r = try await reference.logprobs(prompt: prompt, forcedContinuation: continuation, atPositions: positions, config: config)
    guard r.count == positions.count else {
        throw QualityMetricError.rowCountMismatch(side: "reference", got: r.count, expected: positions.count)
    }
    return TeacherForcedScoresAtPositions(
        positions: positions,
        forcedTokens: positions.map { continuation[$0] },
        candidateRows: c, referenceRows: r)
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
// MARK: - Perplexity (Layer 2, same teacher-forced pass as KL)

/// Mean negative log-likelihood in NATS of `tokens[i]` under softmax(rows[i]).
/// Rows are raw logits (shift-invariant): NLL_i = logSumExp(row) - row[token_i].
public func meanNLL(rows: [[Float]], tokens: [Int]) -> Double {
    precondition(rows.count == tokens.count, "one forced token per scored row")
    precondition(!rows.isEmpty, "no positions to score")
    var total = 0.0
    for (row, tok) in zip(rows, tokens) {
        precondition(row.indices.contains(tok), "forced token id \(tok) outside vocab \(row.count)")
        let m = Double(row.max() ?? 0)
        let logZ = m + log(row.reduce(0.0) { $0 + exp(Double($1) - m) })
        total += logZ - Double(row[tok])
    }
    return total / Double(rows.count)
}

/// Pooled teacher-forced perplexities: exp(total NLL / total forced positions) across all
/// prompts, for candidate and reference from the SAME forward pass (same forced continuation).
public struct PerplexityPair: Sendable {
    public let candidate: Double
    public let reference: Double
    public init(candidate: Double, reference: Double) {
        self.candidate = candidate; self.reference = reference
    }
    /// The dial's instrument: relative ppl delta, gate "<= 1%" i.e. 0.01. 0 = parity;
    /// negative = candidate assigns the reference's continuation HIGHER likelihood.
    public var relativeDelta: Double { (candidate - reference) / reference }
}

public func teacherForcedPerplexities(
    driver: EngineDriver, reference: EngineDriver, prompts: [[Int]], config: RunConfig
) async throws -> PerplexityPair {
    var candTotal = 0.0, refTotal = 0.0, positions = 0
    for p in prompts {
        let s = try await teacherForcedScores(driver: driver, reference: reference, prompt: p, config: config)
        let n = Double(s.continuation.count)
        candTotal += meanNLL(rows: s.candidateRows, tokens: s.continuation) * n
        refTotal += meanNLL(rows: s.referenceRows, tokens: s.continuation) * n
        positions += s.continuation.count
    }
    return PerplexityPair(
        candidate: exp(candTotal / Double(positions)),
        reference: exp(refTotal / Double(positions)))
}

/// Relative perplexity delta of the candidate vs the reference over the reference's greedy
/// continuation (teacher-forced, pooled across prompts). Lower = closer; the dial gates on 1%.
public struct PerplexityMetric: QualityMetric {
    public let name = "ppl_delta"; public init() {}
    public func measure(driver: EngineDriver, reference: EngineDriver, prompts: [[Int]], config: RunConfig) async throws -> Double {
        try await teacherForcedPerplexities(driver: driver, reference: reference, prompts: prompts, config: config).relativeDelta
    }
}

func softmax(_ x: [Float]) -> [Float] { let m = x.max() ?? 0; let e = x.map { expf($0 - m) }; let s = e.reduce(0,+); return e.map { $0/s } }
