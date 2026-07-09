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

/// Median per-position KL of the candidate's next-token distribution vs the fp16 reference's.
public struct KLDivergenceMetric: QualityMetric {
    public let name = "kl_median"; public init() {}
    public func measure(driver: EngineDriver, reference: EngineDriver, prompts: [[Int]], config: RunConfig) async throws -> Double {
        var kls: [Double] = []
        for p in prompts {
            let c = try await driver.logprobs(prompt: p, config: config)
            let r = try await reference.logprobs(prompt: p, config: config)
            for i in 0..<min(c.count, r.count) { kls.append(klDivergence(reference: softmax(r[i]), candidate: softmax(c[i]))) }
        }
        kls.sort(); return kls.isEmpty ? 0 : kls[kls.count/2]
    }
}
func softmax(_ x: [Float]) -> [Float] { let m = x.max() ?? 0; let e = x.map { expf($0 - m) }; let s = e.reduce(0,+); return e.map { $0/s } }
