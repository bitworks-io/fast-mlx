import Darwin
import Foundation
import HarnessCore

struct Qwen38MTPPerformanceScorecardProducerArguments: Equatable, Sendable {
    let authorityPath: String
    let outputPath: String
}

enum Qwen38MTPPerformanceScorecardProducerCLIError:
    Error, Equatable, CustomStringConvertible, Sendable
{
    case missingFlag(String)
    case duplicateFlag(String)
    case unknownFlag
    case missingValue(String)
    case unexpectedPositional
    case authorityReadFailed
    case malformedAuthority
    case producerUnavailable
    case invalidProducerRecord
    case invalidOutput
    case outputExists
    case unsafeOutput
    case outputWriteFailed

    var description: String {
        switch self {
        case .missingFlag(let flag): return "missing required \(flag)"
        case .duplicateFlag(let flag): return "duplicate \(flag)"
        case .unknownFlag: return "unknown flag"
        case .missingValue(let flag): return "\(flag) requires a value"
        case .unexpectedPositional: return "unexpected positional argument"
        case .authorityReadFailed: return "failed to read --authority file"
        case .malformedAuthority: return "malformed authority JSON"
        case .producerUnavailable: return "authenticated measurement producer unavailable"
        case .invalidProducerRecord: return "measurement producer returned invalid evidence"
        case .invalidOutput: return "producer output is not one complete JSONL record"
        case .outputExists: return "output destination already exists"
        case .unsafeOutput: return "output destination is unsafe"
        case .outputWriteFailed: return "failed to publish output evidence"
        }
    }
}

typealias Qwen38MTPPerformanceScorecardRecordFactory =
    @Sendable (Qwen38MTPPerformanceScorecardAuthorityBundle) async throws
        -> ResultRecord<Qwen38MTPPerformanceScorecardEvidence>

func parseQwen38MTPPerformanceScorecardProducerArguments(
    _ arguments: [String]
) throws -> Qwen38MTPPerformanceScorecardProducerArguments {
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw Qwen38MTPPerformanceScorecardProducerCLIError.unexpectedPositional
        }
        guard flag == "--authority" || flag == "--output" else {
            throw Qwen38MTPPerformanceScorecardProducerCLIError.unknownFlag
        }
        guard values[flag] == nil else {
            throw Qwen38MTPPerformanceScorecardProducerCLIError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count,
            !arguments[index + 1].hasPrefix("--")
        else {
            throw Qwen38MTPPerformanceScorecardProducerCLIError.missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }
    guard let authorityPath = values["--authority"] else {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.missingFlag("--authority")
    }
    guard let outputPath = values["--output"] else {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.missingFlag("--output")
    }
    return Qwen38MTPPerformanceScorecardProducerArguments(
        authorityPath: authorityPath,
        outputPath: outputPath)
}

func qwen38MTPPerformanceScorecardProducerExternalDiagnostic(_ error: Error) -> String {
    if let cliError = error as? Qwen38MTPPerformanceScorecardProducerCLIError {
        return cliError.description
    }
    return "scorecard production failed"
}

func produceQwen38MTPPerformanceScorecard(
    arguments: [String],
    readFile: @escaping @Sendable (String) async throws -> Data = {
        try Data(contentsOf: URL(fileURLWithPath: $0))
    },
    makeRecord: Qwen38MTPPerformanceScorecardRecordFactory? = nil,
    writeFresh: @escaping @Sendable (Data, String) async throws -> Void = {
        try writeFreshQwen38MTPPerformanceScorecard($0, $1)
    }
) async throws -> String {
    let parsed = try parseQwen38MTPPerformanceScorecardProducerArguments(arguments)

    let authorityData: Data
    do {
        authorityData = try await readFile(parsed.authorityPath)
    } catch let error as Qwen38MTPPerformanceScorecardProducerCLIError {
        throw error
    } catch {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.authorityReadFailed
    }
    let authority = try decodeQwen38MTPPerformanceScorecardProducerAuthority(authorityData)
    guard let makeRecord else {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.producerUnavailable
    }

    let record = try await makeRecord(authority)
    let data = Data((try record.jsonLine() + "\n").utf8)
    do {
        try Qwen38MTPPerformanceScorecardGate.validatePreflight(
            authority: authority,
            provenance: record.provenance,
            releaseBuildObserved: record.payload.releaseBuildObserved)
        if record.payload.verdict.qualified {
            guard record.subcommand == Qwen38MTPPerformanceScorecardGate.subcommand else {
                throw Qwen38MTPPerformanceScorecardProducerCLIError.invalidProducerRecord
            }
            _ = try Qwen38MTPPerformanceScorecardGate.validateJSONL(
                data,
                authority: authority)
        } else {
            guard record.subcommand == Qwen38MTPPerformanceScorecardGate.rejectedSubcommand,
                try Qwen38MTPPerformanceScorecardGate.computeMetrics(
                    record.payload,
                    authority: authority) == record.payload.metrics,
                try Qwen38MTPPerformanceScorecardGate.evaluateCandidate(
                    record.payload,
                    authority: authority) == record.payload.verdict
            else {
                throw Qwen38MTPPerformanceScorecardProducerCLIError.invalidProducerRecord
            }
        }
    } catch let error as Qwen38MTPPerformanceScorecardProducerCLIError {
        throw error
    } catch {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.invalidProducerRecord
    }

    try await writeFresh(data, parsed.outputPath)
    return "qwen38-mtp-performance-scorecard-producer: WROTE qualified="
        + String(record.payload.verdict.qualified)
}

private func decodeQwen38MTPPerformanceScorecardProducerAuthority(
    _ data: Data
) throws -> Qwen38MTPPerformanceScorecardAuthorityBundle {
    guard (try? JSONSerialization.jsonObject(with: data)) is [String: Any] else {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.malformedAuthority
    }
    do {
        return try JSONDecoder().decode(
            Qwen38MTPPerformanceScorecardAuthorityBundle.self,
            from: data)
    } catch {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.malformedAuthority
    }
}

func writeFreshQwen38MTPPerformanceScorecard(_ data: Data, _ path: String) throws {
    guard !data.isEmpty, data.last == 0x0a,
        data.dropLast().firstIndex(of: 0x0a) == nil
    else {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.invalidOutput
    }

    let outputURL = URL(fileURLWithPath: path).standardizedFileURL
    let manager = FileManager.default
    guard !path.isEmpty, !outputPathIsSymbolicLink(outputURL.path) else {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.unsafeOutput
    }
    if manager.fileExists(atPath: outputURL.path) {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.outputExists
    }
    let parent = outputURL.deletingLastPathComponent()
    var isDirectory: ObjCBool = false
    guard manager.fileExists(atPath: parent.path, isDirectory: &isDirectory),
        isDirectory.boolValue
    else {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.unsafeOutput
    }

    let temporaryURL = parent.appendingPathComponent(
        ".\(outputURL.lastPathComponent).\(UUID().uuidString).tmp")
    let descriptor = temporaryURL.withUnsafeFileSystemRepresentation { pointer -> Int32 in
        guard let pointer else { return -1 }
        return open(pointer, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
    }
    guard descriptor >= 0 else {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.outputWriteFailed
    }
    var descriptorOpen = true
    defer {
        if descriptorOpen { _ = close(descriptor) }
        _ = temporaryURL.withUnsafeFileSystemRepresentation { pointer in
            pointer.map(unlink) ?? -1
        }
    }

    let wroteAll = data.withUnsafeBytes { rawBuffer -> Bool in
        guard let base = rawBuffer.baseAddress else { return false }
        var offset = 0
        while offset < rawBuffer.count {
            let result = Darwin.write(
                descriptor,
                base.advanced(by: offset),
                rawBuffer.count - offset)
            if result < 0 {
                if errno == EINTR { continue }
                return false
            }
            guard result > 0 else { return false }
            offset += result
        }
        return true
    }
    guard wroteAll, fsync(descriptor) == 0 else {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.outputWriteFailed
    }
    guard close(descriptor) == 0 else {
        descriptorOpen = false
        throw Qwen38MTPPerformanceScorecardProducerCLIError.outputWriteFailed
    }
    descriptorOpen = false

    let linked: Int32 = temporaryURL.withUnsafeFileSystemRepresentation {
        temporaryPointer -> Int32 in
        outputURL.withUnsafeFileSystemRepresentation { outputPointer -> Int32 in
            guard let temporaryPointer, let outputPointer else { return -1 }
            return Darwin.link(temporaryPointer, outputPointer)
        }
    }
    guard linked == 0 else {
        if errno == EEXIST {
            throw Qwen38MTPPerformanceScorecardProducerCLIError.outputExists
        }
        throw Qwen38MTPPerformanceScorecardProducerCLIError.outputWriteFailed
    }

    let directoryDescriptor = parent.withUnsafeFileSystemRepresentation { pointer -> Int32 in
        guard let pointer else { return -1 }
        return open(pointer, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
    }
    guard directoryDescriptor >= 0 else {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.outputWriteFailed
    }
    var directoryDescriptorOpen = true
    defer {
        if directoryDescriptorOpen { _ = close(directoryDescriptor) }
    }
    guard fsync(directoryDescriptor) == 0 else {
        throw Qwen38MTPPerformanceScorecardProducerCLIError.outputWriteFailed
    }
    guard close(directoryDescriptor) == 0 else {
        directoryDescriptorOpen = false
        throw Qwen38MTPPerformanceScorecardProducerCLIError.outputWriteFailed
    }
    directoryDescriptorOpen = false
}
