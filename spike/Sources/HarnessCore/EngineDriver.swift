import Foundation

public struct RunConfig: Sendable, Hashable {
    public var temperature: Float; public var maxTokens: Int
    public var specDecode: String?   // "pld" | "dspark" | nil
    /// PLD match n-gram length (nil = the engine drafter's default).
    public var specNgram: Int?
    /// Max drafted tokens per verify forward, K (nil = the engine's default).
    public var specMaxDraft: Int?
    /// Verify-forward compile strategy: true = separate fixed-K compiled verify step (drafts
    /// padded to K so the trace replays), false/nil = uncompiled verify forward. The locked
    /// decision is "choose on-box — measure both", so the harness can select either.
    public var specCompiledVerify: Bool?
    /// Canonical engine KV-cache tier (`nil`/`fp16`, named affine cell, or TurboQuant tier).
    /// The engine parser owns the closed allowlist and rejects unknown spellings.
    public var kvQuant: String?
    public init(temperature: Float = 0, maxTokens: Int = 256, specDecode: String? = nil,
                specNgram: Int? = nil, specMaxDraft: Int? = nil, specCompiledVerify: Bool? = nil,
                kvQuant: String? = nil) {
        self.temperature = temperature; self.maxTokens = maxTokens; self.specDecode = specDecode
        self.specNgram = specNgram; self.specMaxDraft = specMaxDraft
        self.specCompiledVerify = specCompiledVerify; self.kvQuant = kvQuant
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
    /// Full-vocab logprobs per generated position (temp=0), for KL. Each inner array is ordered by
    /// token id — index == token id — so two drivers' outputs can be aligned by index directly.
    /// NOT top-k: under top-k, index i names a different token per model/run, which would make
    /// index-aligned KL meaningless. Empty if unsupported.
    ///
    /// FREE-RUNNING: the driver follows its own greedy path, so two drivers' rows share context
    /// only while their token streams still agree. For cross-driver quality metrics use the
    /// teacher-forced variant below — this one remains for single-driver introspection.
    func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]]
    /// TEACHER-FORCED variant: instead of following its own greedy path, the driver scores a
    /// fixed continuation. Row i is the full-vocab raw-logits next-token distribution given
    /// context = prompt + forcedContinuation[0..<i]; forcedContinuation[i] is then fed as the
    /// next input token REGARDLESS of argmax (and eos does not stop the loop — the continuation
    /// already encodes where its producer stopped). Returns exactly forcedContinuation.count
    /// rows, same index==token-id full-vocab ordering as `logprobs(prompt:config:)`.
    ///
    /// Two drivers scoring the SAME forced continuation therefore score IDENTICAL contexts at
    /// every position — the context-locked basis KL/perplexity require. Free-running paths
    /// diverge (at 2-bit after ~1 token) and would compare distributions over different
    /// contexts, which is not a quality signal.
    func logprobs(prompt: [Int], forcedContinuation: [Int], config: RunConfig) async throws -> [[Float]]
    /// Like `logprobs(prompt:forcedContinuation:config:)` but only materializes rows at
    /// `positions` (ascending indices into `forcedContinuation`) — for long sequences (a
    /// long-context corpus entry teacher-forced against itself can be thousands of positions)
    /// where materializing a full-vocab row at EVERY position would exhaust memory (~0.6MB/row x
    /// thousands of positions x 2 drivers). Returned rows are ordered to match `positions`;
    /// `rows.count == positions.count`. The default implementation (below) computes the full
    /// result and filters — correct but NOT memory-saving; drivers that can skip discarded rows
    /// inside their forward loop (SwiftEngineDriver, the Python reference) override this for the
    /// real saving.
    func logprobs(prompt: [Int], forcedContinuation: [Int], atPositions positions: [Int], config: RunConfig) async throws -> [[Float]]
}

public extension EngineDriver {
    func logprobs(prompt: [Int], forcedContinuation: [Int], atPositions positions: [Int], config: RunConfig) async throws -> [[Float]] {
        let full = try await logprobs(prompt: prompt, forcedContinuation: forcedContinuation, config: config)
        return positions.map { full[$0] }
    }
}

public struct ScriptedDriver: EngineDriver {
    let tokens: [Int]; let engagement: [String: Int]; let lp: [[Float]]; let forcedLp: [[Float]]?
    public init(tokens: [Int], engagement: [String: Int] = [:], logprobs: [[Float]] = [],
                forcedLogprobs: [[Float]]? = nil) {
        self.tokens = tokens; self.engagement = engagement; self.lp = logprobs; self.forcedLp = forcedLogprobs
    }
    public func generate(prompt: [Int], config: RunConfig) async throws -> RunResult {
        RunResult(tokens: tokens, engagement: .init(engagement))
    }
    public func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] { lp }
    public func logprobs(prompt: [Int], forcedContinuation: [Int], config: RunConfig) async throws -> [[Float]] {
        forcedLp ?? lp
    }
}
