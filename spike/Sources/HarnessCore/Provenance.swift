import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// FNV-1a 64-bit over raw bytes — the harness's one dependency-free (Foundation-only) content
/// fingerprint, shared by `MeasurementCorpusLoader.contentHash` and the model-config hash below.
/// Not cryptographic; the goal is stable "did this change" identity for provenance records.
public func fnv1a64(_ bytes: some Sequence<UInt8>) -> String {
    var h: UInt64 = 0xcbf2_9ce4_8422_2325
    let prime: UInt64 = 0x0000_0100_0000_01b3
    for byte in bytes { h ^= UInt64(byte); h = h &* prime }
    return String(format: "%016llx", h)
}

/// Parsed `quantization`/`quantization_config` block of an mlx model's `config.json` — replaces
/// the dirname-substring guess (`lower.contains("8bit") ? "int8" : ...`) with the model's own
/// declared bit-width, so a mislabeled checkpoint directory can no longer record the wrong tier.
public struct ModelQuantInfo: Sendable, Codable, Equatable {
    public let bits: Int?
    public let groupSize: Int?
    public init(bits: Int?, groupSize: Int?) { self.bits = bits; self.groupSize = groupSize }
    /// Display/CSV label: "int4"/"int8" when the config declares bits, "fp16" (the harness's
    /// only unquantized tier) when it doesn't.
    public var label: String { bits.map { "int\($0)" } ?? "fp16" }
}

public enum ModelQuantInfoLoader {
    private struct RawConfig: Decodable {
        struct Quantization: Decodable { let bits: Int?; let group_size: Int? }
        let quantization: Quantization?
    }

    /// Best-effort: a `config.json` with no `quantization` block (or that fails to parse as JSON
    /// at all) is treated as fp16 (`bits: nil`) rather than throwing — most of the fields this
    /// harness cares about are optional metadata, and a provenance record with an honest "unknown"
    /// is better than a hard failure blocking every other subcommand.
    public static func load(from data: Data) -> ModelQuantInfo {
        guard let raw = try? JSONDecoder().decode(RawConfig.self, from: data) else {
            return ModelQuantInfo(bits: nil, groupSize: nil)
        }
        return ModelQuantInfo(bits: raw.quantization?.bits, groupSize: raw.quantization?.group_size)
    }
}

/// Pure precedence for resolving the harness git SHA (the I/O lives in the CLI): an explicit
/// `HARNESS_GIT_SHA` env value wins; then live Git output (which can carry a `-dirty` suffix);
/// then the deploy-written `.harness-sha` file (rsync'd bench hosts have no `.git`, so the deploy
/// step captures the SHA at sync time); "unknown" if none yields a usable value. A usable value is a
/// single non-empty line after trimming — a multi-line value is an error message, not a SHA.
public func resolveHarnessGitSHA(env: String?, shaFile: String?, gitOutput: String?) -> String {
    for candidate in [env, gitOutput, shaFile] {
        guard let candidate else { continue }
        let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.contains("\n") else { continue }
        return trimmed
    }
    return "unknown"
}

/// Everything needed to trace a result back to the exact conditions that produced it — date,
/// hardware, code versions on both sides of the language boundary, the model actually measured,
/// the corpus actually used, and a nonce for byte-level replay identification. Every subcommand
/// (`verify`/`bench`/`kl`) attaches one of these to its JSONL record.
public struct Provenance: Sendable, Codable, Equatable {
    public let date: String                    // ISO 8601, UTC
    public let hardwareChip: String
    public let hardwareRAMBytes: UInt64
    public let hardwareOS: String
    public let harnessGitSHA: String
    public let mlxSwiftVersion: String
    public let referenceMLXVersion: String?     // nil when no Python reference was involved
    public let referenceMLXLMVersion: String?
    public let modelPath: String
    public let modelConfigHash: String          // fnv1a64 of the model's config.json bytes
    public let modelCheckpointManifestHash: String?
    public let modelQuant: ModelQuantInfo
    public let corpusId: String?                // nil for subcommands that don't use the corpus
    public let corpusContentHash: String?
    public let nonce: String

    public init(
        date: String, hardwareChip: String, hardwareRAMBytes: UInt64, hardwareOS: String,
        harnessGitSHA: String, mlxSwiftVersion: String,
        referenceMLXVersion: String?, referenceMLXLMVersion: String?,
        modelPath: String, modelConfigHash: String,
        modelCheckpointManifestHash: String? = nil,
        modelQuant: ModelQuantInfo,
        corpusId: String?, corpusContentHash: String?, nonce: String
    ) {
        self.date = date; self.hardwareChip = hardwareChip; self.hardwareRAMBytes = hardwareRAMBytes
        self.hardwareOS = hardwareOS; self.harnessGitSHA = harnessGitSHA; self.mlxSwiftVersion = mlxSwiftVersion
        self.referenceMLXVersion = referenceMLXVersion; self.referenceMLXLMVersion = referenceMLXLMVersion
        self.modelPath = modelPath; self.modelConfigHash = modelConfigHash
        self.modelCheckpointManifestHash = modelCheckpointManifestHash
        self.modelQuant = modelQuant
        self.corpusId = corpusId; self.corpusContentHash = corpusContentHash; self.nonce = nonce
    }
}

/// One append-only JSONL record: a subcommand's typed payload plus the full `Provenance` that
/// produced it. `subcommand` names which CLI verb wrote the record, so a single evidence file can
/// hold `verify`/`bench`/`kl` rows and still be filtered/parsed uniformly.
public struct ResultRecord<Payload: Codable & Sendable>: Sendable, Codable {
    public let subcommand: String
    public let provenance: Provenance
    public let payload: Payload
    public init(subcommand: String, provenance: Provenance, payload: Payload) {
        self.subcommand = subcommand; self.provenance = provenance; self.payload = payload
    }
}

public enum JSONLError: Error, CustomStringConvertible, Sendable {
    case notUTF8
    case containsNewline
    public var description: String {
        switch self {
        case .notUTF8: return "JSONL encode: encoder produced non-UTF8 bytes"
        case .containsNewline: return "JSONL encode: encoded record contains an embedded newline"
        }
    }
}

public extension ResultRecord {
    /// One line of valid JSON, no trailing newline (the caller appends it). Sorted keys make the
    /// output byte-diffable across runs when nothing semantic changed.
    func jsonLine() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        guard let text = String(data: data, encoding: .utf8) else { throw JSONLError.notUTF8 }
        guard !text.contains("\n") else { throw JSONLError.containsNewline }
        return text
    }
}

public enum RequiredJSONLWriterError: Error, Equatable, Sendable {
    case nonFileURL
    case destinationIsNotRegularFile
    case unterminatedExistingFile
    case malformedExistingLine(line: Int)
    case ioFailure(operation: String, code: Int32)
}

/// Required, append-only evidence writer. It refuses to append after a partial/corrupt line and
/// synchronizes successful writes before returning, so a promotion command cannot report success
/// when its typed evidence was not durably preserved.
public enum RequiredJSONLWriter {
    public static func append<Payload: Codable & Sendable>(
        _ record: ResultRecord<Payload>, to url: URL
    ) throws {
        guard url.isFileURL else { throw RequiredJSONLWriterError.nonFileURL }
        let data = Data((try record.jsonLine() + "\n").utf8)

        let descriptor = url.withUnsafeFileSystemRepresentation { path -> Int32 in
            guard let path else { return -1 }
            return open(
                path, O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC | O_NOFOLLOW,
                0o600)
        }
        guard descriptor >= 0 else {
            let code = errno
            if code == EISDIR || code == ELOOP {
                throw RequiredJSONLWriterError.destinationIsNotRegularFile
            }
            throw RequiredJSONLWriterError.ioFailure(operation: "open", code: code)
        }
        defer { _ = close(descriptor) }

        guard flock(descriptor, LOCK_EX) == 0 else {
            throw RequiredJSONLWriterError.ioFailure(
                operation: "flock", code: errno)
        }
        defer { _ = flock(descriptor, LOCK_UN) }

        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw RequiredJSONLWriterError.ioFailure(
                operation: "fstat", code: errno)
        }
        guard (status.st_mode & mode_t(S_IFMT)) == mode_t(S_IFREG) else {
            throw RequiredJSONLWriterError.destinationIsNotRegularFile
        }

        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        let end = try handle.seekToEnd()
        if end > 0 {
            try handle.seek(toOffset: end - 1)
            guard try handle.read(upToCount: 1) == Data([0x0a]) else {
                throw RequiredJSONLWriterError.unterminatedExistingFile
            }
            try handle.seek(toOffset: 0)
            let existing = try handle.readToEnd() ?? Data()
            let lines = existing.split(
                separator: 0x0a, omittingEmptySubsequences: false).dropLast()
            for (index, line) in lines.enumerated() {
                guard !line.isEmpty,
                    let object = try? JSONSerialization.jsonObject(with: Data(line)),
                    let envelope = object as? [String: Any],
                    envelope["subcommand"] is String,
                    envelope["provenance"] is [String: Any],
                    envelope.keys.contains("payload")
                else {
                    throw RequiredJSONLWriterError.malformedExistingLine(
                        line: index + 1)
                }
            }
        }
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
        try handle.synchronize()
    }
}
