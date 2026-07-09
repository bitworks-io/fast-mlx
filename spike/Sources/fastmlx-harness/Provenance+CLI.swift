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

    /// Prefers an explicit `HARNESS_GIT_SHA` (set by the caller when the binary runs somewhere
    /// without a `.git` directory, e.g. rsync'd to a bench host without history); falls back to
    /// `git rev-parse HEAD` in the current directory; "unknown" if neither works. A result record
    /// with an honest "unknown" is better than a hard failure blocking every subcommand.
    static func harnessGitSHA() -> String {
        if let env = ProcessInfo.processInfo.environment["HARNESS_GIT_SHA"], !env.isEmpty { return env }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["git", "rev-parse", "HEAD"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard process.terminationStatus == 0,
                  let sha = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !sha.isEmpty
            else { return "unknown" }
            return sha
        } catch {
            return "unknown"
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
        corpus: MeasurementCorpus?
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
            modelQuant: quant,
            corpusId: corpus?.corpusId,
            corpusContentHash: corpus?.contentHash,
            nonce: nonce())
        return (provenance, quant)
    }
}

/// Appends one JSONL line to `path`, creating the file (and its header-less, schema-less body) if
/// it doesn't exist yet. Best-effort: a provenance-recording failure must never fail the
/// subcommand it's recording — it prints a warning and continues.
func appendJSONLRecord<Payload: Codable & Sendable>(_ record: ResultRecord<Payload>, to path: String) {
    do {
        let line = try record.jsonLine() + "\n"
        let url = URL(fileURLWithPath: path)
        if let handle = FileHandle(forWritingAtPath: path) {
            defer { try? handle.close() }
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
        } else {
            try line.write(to: url, atomically: true, encoding: .utf8)
        }
        print("# provenance: appended to \(path)")
    } catch {
        print("# provenance: WARNING failed to append JSONL record: \(error)")
    }
}
