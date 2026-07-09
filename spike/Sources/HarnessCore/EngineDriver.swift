import Foundation

public struct RunConfig: Sendable, Hashable {
    public var temperature: Float; public var maxTokens: Int
    public var specDecode: String?   // "pld" | "dspark" | nil
    public var kvQuant: String?      // "fp16" | "8" | "turbo4" | nil
    public init(temperature: Float = 0, maxTokens: Int = 256, specDecode: String? = nil, kvQuant: String? = nil) {
        self.temperature = temperature; self.maxTokens = maxTokens; self.specDecode = specDecode; self.kvQuant = kvQuant
    }
    public static func greedy(maxTokens: Int) -> RunConfig { .init(temperature: 0, maxTokens: maxTokens) }
}

public struct EngagementCounters: Sendable { public var counts: [String: Int]; public init(_ c: [String: Int] = [:]) { counts = c } }

public struct RunResult: Sendable {
    public var tokens: [Int]
    public var engagement: EngagementCounters
    public var acceptanceRate: Double?     // for spec-decode runs; nil otherwise
    public var submitTime: Double
    public var tokenTimes: [Double]
    public init(tokens: [Int], engagement: EngagementCounters = .init(), acceptanceRate: Double? = nil,
                submitTime: Double = 0, tokenTimes: [Double] = []) {
        self.tokens = tokens; self.engagement = engagement; self.acceptanceRate = acceptanceRate
        self.submitTime = submitTime; self.tokenTimes = tokenTimes
    }
}

/// The seam. In-process (SwiftEngineDriver) now; HTTP/OpenAI later. Reference impl (mlx-lm) via ReferenceDriver.
public protocol EngineDriver: Sendable {
    func generate(prompt: [Int], config: RunConfig) async throws -> RunResult
    /// Top-k logprobs per generated position (temp=0), for KL. Empty if unsupported.
    func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]]
}

public struct ScriptedDriver: EngineDriver {
    let tokens: [Int]; let engagement: [String: Int]; let lp: [[Float]]
    public init(tokens: [Int], engagement: [String: Int] = [:], logprobs: [[Float]] = []) {
        self.tokens = tokens; self.engagement = engagement; self.lp = logprobs
    }
    public func generate(prompt: [Int], config: RunConfig) async throws -> RunResult {
        RunResult(tokens: tokens, engagement: .init(engagement))
    }
    public func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] { lp }
}
