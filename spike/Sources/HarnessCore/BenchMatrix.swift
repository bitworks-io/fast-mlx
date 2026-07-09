import Foundation

public enum Workload: String, Sendable, CaseIterable { case prefill, decode, echo, code }
public enum Mode: String, Sendable, CaseIterable { case none, pld, dspark }

public struct Cell: Sendable, Hashable {
    public let workload: Workload
    public let mode: Mode
    public let model: String
    public let quant: String
    public let concurrency: Int
    public init(workload: Workload, mode: Mode, model: String, quant: String = "fp16", concurrency: Int = 1) {
        self.workload = workload; self.mode = mode; self.model = model; self.quant = quant; self.concurrency = concurrency
    }
}

public struct RateAggregate: Sendable, Equatable { public let mean: Double; public let runs: Int
    public init(mean: Double, runs: Int) { self.mean = mean; self.runs = runs }
}

/// Drops run 0 (warmup) unconditionally, then averages the remaining non-nil (non-skipped) rates.
/// `nil` entries elsewhere in the array represent explicitly skipped runs and are excluded too.
public func aggregateRates(_ rates: [Double?]) -> RateAggregate {
    let postWarmup = rates.dropFirst()
    let valid = postWarmup.compactMap { $0 }
    guard !valid.isEmpty else { return RateAggregate(mean: 0, runs: 0) }
    return RateAggregate(mean: valid.reduce(0, +) / Double(valid.count), runs: valid.count)
}

/// Salts a prompt per run+nonce so repeated bench runs don't hit a KV/prefix cache that would
/// understate decode cost (backlog methodology: never bench a cached prefix as if it were cold).
public func saltPrompt(run: Int, nonce: String, _ basePrompt: String) -> String {
    "\(basePrompt) [run=\(run) nonce=\(nonce)]"
}

public enum BenchGuardError: Error, Sendable { case debugBuild }

/// Fails fast if this is a Debug/unoptimized build — perf numbers from a Debug build are
/// misleading (the ReleaseFast lesson from the spike).
public func assertReleaseBuild() throws {
    #if DEBUG
    throw BenchGuardError.debugBuild
    #endif
}

/// One CSV row of a bench run. Rate is expected to come from stream-timed metrics
/// (e.g. `DecodeMetrics` in the executable target), never usage/summary fields.
/// `hardware` is spec §6.3's mandated durable-evidence dimension ("enough dimensions... to diff
/// programmatically against a prior run"): label/mode/concurrency/model were already present;
/// hardware was the missing one (Task 5).
public struct BenchRow: Sendable {
    public let label: String
    public let workload: Workload
    public let mode: Mode
    public let model: String
    public let decodeTokS: Double
    public let ttftMs: Double
    public let quant: String
    public let concurrency: Int
    public let hardware: String

    public init(label: String, workload: Workload, mode: Mode, model: String, decodeTokS: Double, ttftMs: Double, quant: String, concurrency: Int, hardware: String) {
        self.label = label; self.workload = workload; self.mode = mode; self.model = model
        self.decodeTokS = decodeTokS; self.ttftMs = ttftMs; self.quant = quant; self.concurrency = concurrency
        self.hardware = hardware
    }

    public static let csvHeader = "label,workload,mode,model,decode_tok_s,ttft_ms,quant,concurrency,hardware"

    public var csvLine: String {
        "\(label),\(workload.rawValue),\(mode.rawValue),\(model),\(decodeTokS),\(ttftMs),\(quant),\(concurrency),\(hardware)"
    }
}

/// Runs a `Cell` for N iterations via a caller-supplied rate function — the MLX-touching timing
/// itself lives in the executable's `SwiftEngineDriver`, not here — applying warmup-drop and
/// prompt-salting per the bench methodology, and aggregating into a rate.
public struct BenchRunner: Sendable {
    public init() {}

    public func run(
        cell: Cell,
        iterations: Int,
        nonce: String,
        basePrompt: String,
        rate: (_ run: Int, _ saltedPrompt: String) async throws -> Double?
    ) async rethrows -> RateAggregate {
        var rates: [Double?] = []
        for i in 0..<iterations {
            let salted = saltPrompt(run: i, nonce: nonce, basePrompt)
            rates.append(try await rate(i, salted))
        }
        return aggregateRates(rates)
    }
}
