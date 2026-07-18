import Foundation

public enum Workload: String, Sendable, CaseIterable { case prefill, decode, echo, code }
public enum Mode: String, Sendable, CaseIterable { case none, pld, dspark }

/// Audited first-party prompt used by the batch-1 runtime frontier. KVTuner permits only this
/// source because an arbitrary prompt cannot truthfully assert disjoint upstream source rows.
public let defaultBenchPrompt =
    "Explain how continuous batching improves LLM serving throughput."

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

public enum BenchWorkloadIdentityError: Error, Sendable, Equatable {
    case invalidNonce
    case invalidIterations
}

/// Exact prompt set shared across separately launched KV-tier measurements. An explicit nonce is
/// part of durable evidence so every cell can replay identical cold, salted prompt bytes.
public struct BenchWorkloadIdentity: Sendable, Equatable {
    public let basePrompt: String
    public let nonce: String
    public let iterations: Int

    public init(basePrompt: String, nonce: String, iterations: Int) throws {
        guard iterations > 0 else {
            throw BenchWorkloadIdentityError.invalidIterations
        }
        do {
            _ = try ServiceWorkloadIdentity(nonce: nonce)
        } catch {
            throw BenchWorkloadIdentityError.invalidNonce
        }
        self.basePrompt = basePrompt
        self.nonce = nonce
        self.iterations = iterations
    }

    public func prompt(run: Int) -> String {
        precondition((0..<iterations).contains(run), "bench run outside workload identity")
        return saltPrompt(run: run, nonce: nonce, basePrompt)
    }

    public var prompts: [String] {
        (0..<iterations).map(prompt(run:))
    }
}

/// Direct prefill rate from the actor-timed `decoder.prefill` span. This never derives prefill
/// from TTFT, which also contains first-token bookkeeping and is a distinct latency metric.
public func prefillTokensPerSecond(
    promptTokens: Int,
    durationSeconds: Double
) -> Double? {
    guard promptTokens > 0, durationSeconds.isFinite, durationSeconds > 0 else {
        return nil
    }
    return Double(promptTokens) / durationSeconds
}

public enum ServiceWorkloadIdentityError: Error, Sendable, Equatable {
    case invalidNonce
}

/// Stable identity shared by every process in a service-policy frontier. Keeping the nonce
/// explicit prevents separate policy/concurrency invocations from benchmarking different
/// salted prompts while claiming a direct comparison.
public struct ServiceWorkloadIdentity: Sendable, Equatable {
    public let nonce: String

    public init(nonce: String) throws {
        let alphanumericByte: (UInt8) -> Bool = {
            (48 ... 57).contains($0) || (65 ... 90).contains($0)
                || (97 ... 122).contains($0)
        }
        let validByte: (UInt8) -> Bool = {
            alphanumericByte($0) || $0 == 45 || $0 == 46 || $0 == 95
        }
        guard let first = nonce.utf8.first, alphanumericByte(first),
            nonce.utf8.count <= 64, nonce.utf8.allSatisfy(validByte)
        else {
            throw ServiceWorkloadIdentityError.invalidNonce
        }
        self.nonce = nonce
    }

    public func prompt(basePrompt: String, run: Int, request: Int) -> String {
        "\(saltPrompt(run: run, nonce: nonce, basePrompt)) [request=\(request)]"
    }
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
    public let prefillTokS: Double?
    public let prefillMs: Double?
    public let promptTokensMin: Int?
    public let promptTokensMax: Int?
    public let kvQuantTier: String?
    public let matrixID: String?
    public let cellID: String?
    public let workloadNonce: String?
    public let kvtunerScheduleSHA256: String?
    public let kvtunerBundleSHA256: String?

    public init(
        label: String, workload: Workload, mode: Mode, model: String,
        decodeTokS: Double, ttftMs: Double, quant: String,
        concurrency: Int, hardware: String,
        prefillTokS: Double? = nil, prefillMs: Double? = nil,
        promptTokensMin: Int? = nil, promptTokensMax: Int? = nil,
        kvQuantTier: String? = nil, matrixID: String? = nil,
        cellID: String? = nil, workloadNonce: String? = nil,
        kvtunerScheduleSHA256: String? = nil,
        kvtunerBundleSHA256: String? = nil
    ) {
        self.label = label; self.workload = workload; self.mode = mode; self.model = model
        self.decodeTokS = decodeTokS; self.ttftMs = ttftMs; self.quant = quant; self.concurrency = concurrency
        self.hardware = hardware
        self.prefillTokS = prefillTokS; self.prefillMs = prefillMs
        self.promptTokensMin = promptTokensMin; self.promptTokensMax = promptTokensMax
        self.kvQuantTier = kvQuantTier; self.matrixID = matrixID
        self.cellID = cellID; self.workloadNonce = workloadNonce
        self.kvtunerScheduleSHA256 = kvtunerScheduleSHA256
        self.kvtunerBundleSHA256 = kvtunerBundleSHA256
    }

    public static let csvHeader =
        "label,workload,mode,model,decode_tok_s,ttft_ms,quant,concurrency,hardware"
        + ",prefill_tok_s,prefill_ms,prompt_tokens_min,prompt_tokens_max"
        + ",kv_quant_tier,matrix_id,cell_id,workload_nonce"
        + ",kvtuner_schedule_sha256,kvtuner_bundle_sha256"

    public var csvLine: String {
        let prefillTokSText = prefillTokS.map { String($0) } ?? ""
        let prefillMsText = prefillMs.map { String($0) } ?? ""
        let promptTokensMinText = promptTokensMin.map { String($0) } ?? ""
        let promptTokensMaxText = promptTokensMax.map { String($0) } ?? ""
        let directPrefill = [
            prefillTokSText,
            prefillMsText,
            promptTokensMinText,
            promptTokensMaxText,
        ].joined(separator: ",")
        let runtimeIdentity = [
            kvQuantTier ?? "",
            matrixID ?? "",
            cellID ?? "",
            workloadNonce ?? "",
            kvtunerScheduleSHA256 ?? "",
            kvtunerBundleSHA256 ?? "",
        ].joined(separator: ",")
        return "\(label),\(workload.rawValue),\(mode.rawValue),\(model),\(decodeTokS),\(ttftMs),\(quant),\(concurrency),\(hardware),\(directPrefill),\(runtimeIdentity)"
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
