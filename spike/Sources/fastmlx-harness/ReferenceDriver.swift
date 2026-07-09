import Foundation
import HarnessCore

/// `EngineDriver` that shells to `scripts/harness_reference.py` (Python mlx-lm) — the
/// equivalence + KL reference implementation.
///
/// Contract parity with every other driver: `logprobs` returns full-vocab RAW LOGITS in
/// token-id order (index == token id). Logits cross the process boundary as raw
/// little-endian float32 `[positions x vocab]` in a temp file — exact bytes, no text
/// round-trip precision loss — while tokens come back as small JSON on stdout.
/// Prompts are passed as TOKEN IDS (not text), so tokenizer differences between Swift and
/// Python cannot desynchronize the comparison; the Swift-side eos id is passed down for
/// identical stopping semantics.
struct ReferenceDriver: EngineDriver {
    var pythonPath: String
    var scriptPath: String
    var modelPath: String
    var eos: Int

    struct Header: Decodable {
        let tokens: [Int]
        let positions: Int
        let vocab: Int
        let mlxVersion: String?
        let mlxLmVersion: String?
        enum CodingKeys: String, CodingKey {
            case tokens, positions, vocab
            case mlxVersion = "mlx_version"
            case mlxLmVersion = "mlx_lm_version"
        }
    }

    struct ReferenceVersions: Sendable, Equatable { let mlx: String; let mlxLM: String }

    /// Records the reference's self-reported `mlx`/`mlx-lm` versions (Task 5 provenance) as they
    /// come back off successful calls — an `actor`, not a lock/box, so this stays Swift 6 clean
    /// with no `@unchecked Sendable` hatch. `ReferenceDriver` is a struct; the actor reference
    /// itself is Sendable, so every copy of a given driver shares the same sink.
    actor VersionSink {
        private(set) var versions: ReferenceVersions?
        func record(_ v: ReferenceVersions) { versions = v }
    }
    let versionSink = VersionSink()

    enum ReferenceError: Error, CustomStringConvertible {
        case unsupportedConfig(String)
        case processFailed(status: Int32, stderr: String)
        case badOutput(String)
        var description: String {
            switch self {
            case .unsupportedConfig(let what): return "ReferenceDriver: unsupported config: \(what)"
            case .processFailed(let status, let stderr): return "harness_reference.py exited \(status): \(stderr)"
            case .badOutput(let what): return "harness_reference.py bad output: \(what)"
            }
        }
    }

    func generate(prompt: [Int], config: RunConfig) async throws -> RunResult {
        let header = try run(prompt: prompt, config: config, logitsOut: nil)
        await recordVersions(from: header)
        return RunResult(tokens: header.tokens) // reference is untimed; engagement n/a
    }

    /// Stashes the reference's self-reported versions (Task 5) if this call's header carried
    /// them — every successful call updates the sink, so whichever call happens to run first in a
    /// subcommand is enough for the CLI to read `versionSink.versions` afterward.
    private func recordVersions(from header: Header) async {
        guard let mlx = header.mlxVersion, let mlxLM = header.mlxLmVersion else { return }
        await versionSink.record(ReferenceVersions(mlx: mlx, mlxLM: mlxLM))
    }

    func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] {
        try await readLogits(prompt: prompt, config: config, forceTokens: nil, samplePositions: nil)
    }

    /// Teacher-forced contract: `--force-tokens` makes the Python side feed each forced token
    /// instead of its own argmax, so row i's context is prompt + forcedContinuation[0..<i] —
    /// identical to what any other driver scores for the same continuation.
    func logprobs(prompt: [Int], forcedContinuation: [Int], config: RunConfig) async throws -> [[Float]] {
        try await readLogits(prompt: prompt, config: config, forceTokens: forcedContinuation, samplePositions: nil)
    }

    /// Sampled variant: `--sample-positions` tells `harness_reference.py` to only materialize+emit
    /// rows at the given (ascending) positions — the script still runs the full forward loop over
    /// `forcedContinuation` (causal decoding needs every intermediate token as context) but skips
    /// converting/writing the discarded rows, so this is a real memory saving on the Python side
    /// too, not just a post-hoc filter.
    func logprobs(prompt: [Int], forcedContinuation: [Int], atPositions positions: [Int], config: RunConfig) async throws -> [[Float]] {
        try await readLogits(prompt: prompt, config: config, forceTokens: forcedContinuation, samplePositions: positions)
    }

    private func readLogits(prompt: [Int], config: RunConfig, forceTokens: [Int]?, samplePositions: [Int]?) async throws -> [[Float]] {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("harness-ref-logits-\(UUID().uuidString).f32")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let header = try run(prompt: prompt, config: config, logitsOut: tmp.path, forceTokens: forceTokens, samplePositions: samplePositions)
        await recordVersions(from: header)
        let data = try Data(contentsOf: tmp)
        let expectedBytes = header.positions * header.vocab * MemoryLayout<Float32>.size
        guard data.count == expectedBytes else {
            throw ReferenceError.badOutput("logits file is \(data.count) bytes, expected \(expectedBytes)")
        }
        var flat = [Float](repeating: 0, count: header.positions * header.vocab)
        flat.withUnsafeMutableBytes { _ = data.copyBytes(to: $0) } // alignment-safe copy
        return (0..<header.positions).map { Array(flat[$0 * header.vocab ..< ($0 + 1) * header.vocab]) }
    }

    private func run(prompt: [Int], config: RunConfig, logitsOut: String?, forceTokens: [Int]? = nil, samplePositions: [Int]? = nil) throws -> Header {
        guard config.temperature == 0 else {
            throw ReferenceError.unsupportedConfig("temperature=\(config.temperature) (reference is greedy-only)")
        }
        let tokensJSON = "[" + prompt.map(String.init).joined(separator: ",") + "]"
        var args = [
            scriptPath,
            "--model", modelPath,
            "--tokens-json", tokensJSON,
            "--n", String(config.maxTokens),
            "--eos-id", String(eos),
        ]
        if let logitsOut { args += ["--logits-out", logitsOut] }
        if let forceTokens {
            args += ["--force-tokens", "[" + forceTokens.map(String.init).joined(separator: ",") + "]"]
        }
        if let samplePositions {
            args += ["--sample-positions", "[" + samplePositions.map(String.init).joined(separator: ",") + "]"]
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = args
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        try process.run()
        // Read both pipes to EOF before waiting, so a chatty child can't deadlock on a full pipe.
        let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ReferenceError.processFailed(
                status: process.terminationStatus,
                stderr: String(data: errData, encoding: .utf8) ?? "<non-utf8 stderr>")
        }
        // stdout may carry stray library warnings before the JSON; the header is the LAST non-empty line.
        guard let text = String(data: outData, encoding: .utf8),
              let line = text.split(separator: "\n").last(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }),
              let header = try? JSONDecoder().decode(Header.self, from: Data(line.utf8))
        else {
            throw ReferenceError.badOutput(String(data: outData, encoding: .utf8) ?? "<non-utf8 stdout>")
        }
        return header
    }
}
