import Foundation
import HarnessCore

#if canImport(Darwin)
import Darwin
#endif

/// CLI-side glue for Task 5 provenance: everything here is I/O (sysctl, git, file reads) that a
/// pure `HarnessCore` (Foundation-only, no shell-out) deliberately cannot do. `HarnessCore.
/// Provenance`/`ResultRecord` — the typed struct and its JSONL encoding — stay pure and TDD'd;
/// this file only GATHERS the values that get poured into them.
enum ProvenanceCLI {
    enum EvidenceIdentityError: Error, CustomStringConvertible {
        case unreadableModelConfig(String)
        case missingCheckpointWeights(String)
        case missingTokenizerFiles(String)
        case invalidTokenizerFile(String)

        var description: String {
            switch self {
            case .unreadableModelConfig(let path):
                return "cannot fingerprint model config at \(path)"
            case .missingCheckpointWeights(let path):
                return "checkpoint manifest at \(path) contains no safetensors files"
            case .missingTokenizerFiles(let path):
                return "tokenizer manifest at \(path) contains no recognized tokenizer files"
            case .invalidTokenizerFile(let path):
                return "tokenizer manifest entry is dangling or not a regular file: \(path)"
            }
        }
    }

    /// `machdep.cpu.brand_string` via sysctlbyname — the standard way to name the chip on Apple
    /// Silicon; there is no public Swift/Foundation API for it.
    static func chipBrand() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return "unknown" }
        var buffer = [CChar](repeating: 0, count: size)
        let result = sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
        guard result == 0 else { return "unknown" }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self).trimmingCharacters(in: .whitespaces)
    }

    static func ramBytes() -> UInt64 { ProcessInfo.processInfo.physicalMemory }

    static func osVersion() -> String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }

    /// `mlx-swift`'s pinned version, per `Package.swift` (`exact: "0.31.6"`). There is no runtime
    /// API to read a resolved SwiftPM dependency's version from inside the built binary, so this
    /// is a build-time constant that must be kept in sync with `Package.swift` by hand — cheaper
    /// and more honest than a fragile `Package.resolved` file-read that assumes a specific CWD.
    static let mlxSwiftVersion = "0.31.6"
    static let mlxSwiftLMRevision = "702e5a0eaf990e1f6d3db2b6e7d8872858a44055"

    /// Gathers the three SHA sources and applies the pure precedence (`resolveHarnessGitSHA`,
    /// TDD'd in HarnessCore): explicit `HARNESS_GIT_SHA` env override -> dirty-aware live Git ->
    /// the deploy-written `.harness-sha` file (`sync_llmbench.sh` writes it because the rsync'd
    /// bench tree has no `.git`) -> "unknown". Live Git must outrank a stale local stamp.
    static func harnessGitSHA() -> String {
        resolveHarnessGitSHA(
            env: ProcessInfo.processInfo.environment["HARNESS_GIT_SHA"],
            shaFile: try? String(contentsOfFile: ".harness-sha", encoding: .utf8),
            gitOutput: liveGitSHA())
    }

    private static func liveGitSHA() -> String? {
        guard let root = runGit(["rev-parse", "--show-toplevel"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !root.isEmpty,
            let sha = runGit(["-C", root, "rev-parse", "HEAD"])?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !sha.isEmpty,
            let status = runGit([
                "-C", root, "status", "--porcelain", "--untracked-files=normal",
                "--", "spike", "experiments",
            ])
        else { return nil }
        return status.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? sha : "\(sha)-dirty"
    }

    private static func runGit(_ arguments: [String]) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["git"] + arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { return nil }
            return String(data: data, encoding: .utf8)
        } catch {
            return nil
        }
    }

    /// Reads `<modelPath>/config.json`, hashes its raw bytes (Task 5: replaces the dirname-substring
    /// quant guess), and parses its `quantization` block. `nil` config bytes (file missing/unreadable)
    /// still returns a usable result — hash "unknown", quant defaulting to fp16 — rather than throwing.
    static func modelConfig(at modelPath: String) -> (hash: String, quant: ModelQuantInfo) {
        let configURL = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL) else {
            return ("unknown", ModelQuantInfo(bits: nil, groupSize: nil))
        }
        return (fnv1a64(data), ModelQuantInfoLoader.load(from: data))
    }

    /// Config/index bytes plus sorted shard names and sizes. This detects the common wrong-
    /// checkpoint/wrong-shard manifest case without reading tens of GiB of weights into the
    /// benchmark's timed process. It is explicitly a manifest fingerprint, not a content hash.
    static func checkpointManifestHash(at modelPath: String) throws -> String {
        let directory = URL(fileURLWithPath: modelPath)
        let fileManager = FileManager.default
        var bytes = Array(
            try Data(contentsOf: directory.appendingPathComponent("config.json")))
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        if fileManager.fileExists(atPath: index.path) {
            bytes.append(contentsOf: try Data(contentsOf: index))
        }
        let weights = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles])
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !weights.isEmpty else {
            throw EvidenceIdentityError.missingCheckpointWeights(modelPath)
        }
        for weight in weights {
            let size = try weight.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1
            bytes.append(contentsOf: "\(weight.lastPathComponent):\(size)\n".utf8)
        }
        return fnv1a64(bytes)
    }

    /// Promotion evidence uses both config bytes and a checkpoint manifest fingerprint. The
    /// latter is deliberately not described as a tensor-content hash: it covers config/index
    /// bytes plus sorted shard names and sizes so identity remains cheap enough to gather before
    /// a timed run.
    static func modelEvidenceIdentity(at modelPath: String) throws -> KVModelEvidenceIdentity {
        let configHash = modelConfig(at: modelPath).hash
        guard configHash != "unknown" else {
            throw EvidenceIdentityError.unreadableModelConfig(modelPath)
        }
        return KVModelEvidenceIdentity(
            configHash: configHash,
            checkpointManifestHash: try checkpointManifestHash(at: modelPath))
    }

    /// Cryptographic content manifest for every local file that can affect raw prompt encoding or
    /// generated-token decoding. Weight/config identity deliberately stays cheap and separate;
    /// task evidence needs this stronger tokenizer boundary because a mutable tokenizer.json can
    /// otherwise change both the scored prompt and structured output under identical model hashes.
    static func tokenizerManifestSHA256(at modelPath: String) throws -> String {
        let directory = URL(fileURLWithPath: modelPath)
        let recognizedNames: Set<String> = [
            "added_tokens.json", "merges.txt", "sentencepiece.bpe.model",
            "special_tokens_map.json", "spiece.model", "tokenizer.model",
            "vocab.json", "vocab.txt",
        ]
        let candidates = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])
            .filter { url in
                let name = url.lastPathComponent.lowercased()
                return name.hasPrefix("tokenizer")
                    || recognizedNames.contains(name)
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !candidates.isEmpty else {
            throw EvidenceIdentityError.missingTokenizerFiles(modelPath)
        }

        let files = try candidates.map { logicalURL -> (URL, URL) in
            let resolvedURL = logicalURL.resolvingSymlinksInPath()
            let values = try? resolvedURL.resourceValues(
                forKeys: [.isRegularFileKey])
            guard values?.isRegularFile == true else {
                throw EvidenceIdentityError.invalidTokenizerFile(
                    logicalURL.path)
            }
            return (logicalURL, resolvedURL)
        }

        var manifest = Data()
        func appendLength(_ value: Int) {
            var encoded = UInt64(value).bigEndian
            withUnsafeBytes(of: &encoded) {
                manifest.append(contentsOf: $0)
            }
        }
        for (logicalURL, resolvedURL) in files {
            let name = Data(logicalURL.lastPathComponent.utf8)
            let contents = try Data(contentsOf: resolvedURL)
            appendLength(name.count)
            manifest.append(name)
            appendLength(contents.count)
            manifest.append(contents)
        }
        return sha256Hex(manifest)
    }

    /// Manifest equality alone cannot prove tensor equality, so `sameWeights` also requires both
    /// CLI paths to resolve to the same filesystem location. Symlink aliases remain valid; copied
    /// lookalike checkpoints remain conservatively classified as different weights.
    static func sameResolvedModelPath(_ lhs: String, _ rhs: String) -> Bool {
        func resolved(_ path: String) -> String {
            URL(fileURLWithPath: path).standardizedFileURL
                .resolvingSymlinksInPath().path
        }
        return resolved(lhs) == resolved(rhs)
    }

    static func nonce() -> String { String(Int.random(in: 0..<1_000_000_000)) }

    static func nowISO8601() -> String {
        let formatter = ISO8601DateFormatter()
        return formatter.string(from: Date())
    }

    /// Assembles a full `Provenance` for one subcommand invocation. `referenceVersions` is read
    /// from `ReferenceDriver.versionSink` AFTER at least one reference call has completed (nil for
    /// runs that never invoke the Python reference, e.g. a same-weights-only bench).
    static func build(
        modelPath: String, referenceVersions: ReferenceDriver.ReferenceVersions?,
        corpus: MeasurementCorpus?, modelCheckpointManifestHash: String? = nil
    ) -> (provenance: Provenance, quant: ModelQuantInfo) {
        build(
            modelPath: modelPath, referenceVersions: referenceVersions,
            corpusID: corpus?.corpusId,
            corpusContentHash: corpus?.contentHash,
            modelCheckpointManifestHash: modelCheckpointManifestHash)
    }

    /// Task coherence uses a distinct frozen corpus type but the same provenance envelope. A
    /// dedicated overload prevents callers from passing nil and producing an artifact that only
    /// fails after an expensive model run.
    static func build(
        modelPath: String, referenceVersions: ReferenceDriver.ReferenceVersions?,
        taskCorpus: TaskCoherenceCorpus,
        modelCheckpointManifestHash: String
    ) -> (provenance: Provenance, quant: ModelQuantInfo) {
        build(
            modelPath: modelPath, referenceVersions: referenceVersions,
            corpusID: taskCorpus.id,
            corpusContentHash: taskCorpus.contentHash,
            modelCheckpointManifestHash: modelCheckpointManifestHash)
    }

    private static func build(
        modelPath: String, referenceVersions: ReferenceDriver.ReferenceVersions?,
        corpusID: String?, corpusContentHash: String?,
        modelCheckpointManifestHash: String?
    ) -> (provenance: Provenance, quant: ModelQuantInfo) {
        let (configHash, quant) = modelConfig(at: modelPath)
        let provenance = Provenance(
            date: nowISO8601(),
            hardwareChip: chipBrand(),
            hardwareRAMBytes: ramBytes(),
            hardwareOS: osVersion(),
            harnessGitSHA: harnessGitSHA(),
            mlxSwiftVersion: mlxSwiftVersion,
            referenceMLXVersion: referenceVersions?.mlx,
            referenceMLXLMVersion: referenceVersions?.mlxLM,
            modelPath: modelPath,
            modelConfigHash: configHash,
            modelCheckpointManifestHash: modelCheckpointManifestHash,
            modelQuant: quant,
            corpusId: corpusID,
            corpusContentHash: corpusContentHash,
            nonce: nonce())
        return (provenance, quant)
    }
}

/// Appends one JSONL line to `path`, creating the file (and its header-less, schema-less body) if
/// it doesn't exist yet. Best-effort: a provenance-recording failure must never fail the
/// subcommand it's recording — it prints a warning and continues.
func appendJSONLRecord<Payload: Codable & Sendable>(_ record: ResultRecord<Payload>, to path: String) {
    do {
        try appendRequiredJSONLRecord(record, to: path)
        print("# provenance: appended to \(path)")
    } catch {
        print("# provenance: WARNING failed to append JSONL record: \(error)")
    }
}

/// Promotion-gate measurements use a required writer: a result that cannot preserve its
/// provenance is a failed run, not a successful benchmark followed by a warning.
func appendRequiredJSONLRecord<Payload: Codable & Sendable>(
    _ record: ResultRecord<Payload>,
    to path: String
) throws {
    try RequiredJSONLWriter.append(record, to: URL(fileURLWithPath: path))
}
