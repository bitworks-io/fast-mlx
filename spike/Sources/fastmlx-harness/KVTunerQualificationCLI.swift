import Foundation
import HarnessCore

#if canImport(Darwin)
import Darwin
#endif

enum KVTunerQualificationCLIError: Error, CustomStringConvertible {
    case missingFlag(String)
    case invalidPositiveInteger(flag: String, value: Int)
    case unsupportedGroupSize(Int)
    case invalidOrdinal(Int)
    case outputNotFresh(String)
    case outputPathCollision(String)
    case invalidHarnessGitSHA(String)
    case liveGitUnavailable
    case invalidArtifact(String)
    case missingEOSToken
    case releaseBuildRequired
    case sourceIdentityMismatch
    case invalidCandidateDirectory(String)
    case missingCandidateArtifact(Int)

    var description: String {
        switch self {
        case .missingFlag(let flag):
            return "missing required --\(flag)"
        case .invalidPositiveInteger(let flag, let value):
            return "--\(flag) must be greater than zero; got \(value)"
        case .unsupportedGroupSize(let value):
            return "--group-size must be 64 or 128; got \(value)"
        case .invalidOrdinal(let value):
            return "--candidate-ordinal must be zero or greater; got \(value)"
        case .outputNotFresh(let path):
            return "KVTuner artifact destination must be a new or empty writable regular file: \(path)"
        case .outputPathCollision(let path):
            return "KVTuner qualification paths must be distinct: \(path)"
        case .invalidHarnessGitSHA(let value):
            return "KVTuner qualification requires a clean 40-character git SHA; got \(value)"
        case .liveGitUnavailable:
            return "KVTuner qualification found live git metadata but could not authenticate the repository SHA and status"
        case .invalidArtifact(let path):
            return "invalid KVTuner qualification artifact: \(path)"
        case .missingEOSToken:
            return "KVTuner qualification requires a tokenizer EOS token"
        case .releaseBuildRequired:
            return "KVTuner qualification evidence must be produced by a Release build"
        case .sourceIdentityMismatch:
            return "KVTuner model, checkpoint, or tokenizer identity changed during qualification"
        case .invalidCandidateDirectory(let path):
            return "KVTuner candidate output must be a writable real directory: \(path)"
        case .missingCandidateArtifact(let ordinal):
            return "missing KVTuner candidate artifact at ordinal \(ordinal)"
        }
    }
}

struct KVTunerManifestPlan {
    let modelPath: String
    let promptFixturePath: String
    let normalizedTargetsPath: String
    let outputPath: String
}

struct KVTunerCandidatePlan {
    let modelPath: String
    let manifestPath: String
    let sensitivityPath: String
    let targetPairBitTotal: Int
    let maxCandidates: Int
    let outputDirectory: String
    let candidateOrdinal: Int?
}

struct KVTunerSensitivityPlan {
    let modelPath: String
    let manifestPath: String
    let matrixID: String
    let groupSize: Int
    let outputPath: String
}

struct KVTunerSearchPlan {
    let modelPath: String
    let manifestPath: String
    let sensitivityPath: String
    let targetPairBitTotal: Int
    let maxCandidates: Int
    let candidateDirectory: String
    let outputPath: String
    let scheduleOutputPath: String
}

struct KVTunerBundlePlan {
    let modelPath: String
    let manifestPath: String
    let sensitivityPath: String
    let searchPath: String
    let schedulePath: String
    let candidateDirectory: String
    let outputPath: String
}

private func requiredKVTunerFlag(
    _ flags: Flags,
    _ name: String
) throws -> String {
    let value = try flags.strictString(name, default: "")
    guard !value.isEmpty else {
        throw KVTunerQualificationCLIError.missingFlag(name)
    }
    return value
}

private func positiveKVTunerFlag(
    _ flags: Flags,
    _ name: String
) throws -> Int {
    let value = try flags.strictInt(name, default: 0)
    guard value > 0 else {
        throw KVTunerQualificationCLIError.invalidPositiveInteger(
            flag: name, value: value)
    }
    return value
}

private func requireDistinctKVTunerPaths(_ paths: [String]) throws {
    for leftIndex in paths.indices {
        for rightIndex in paths.indices where rightIndex > leftIndex {
            guard !outputPathsReferToSameFile(
                paths[leftIndex], paths[rightIndex])
            else {
                throw KVTunerQualificationCLIError.outputPathCollision(
                    paths[rightIndex])
            }
        }
    }
}

func parseKVTunerManifestPlan(_ flags: Flags) throws -> KVTunerManifestPlan {
    let modelPath = try requiredKVTunerFlag(flags, "model")
    let promptFixturePath = try requiredKVTunerFlag(flags, "prompt-fixture")
    let normalizedTargetsPath = try requiredKVTunerFlag(
        flags, "normalized-targets")
    let outputPath = try requiredKVTunerFlag(flags, "output")
    try requireDistinctKVTunerPaths([
        modelPath, promptFixturePath, normalizedTargetsPath, outputPath,
    ])
    return KVTunerManifestPlan(
        modelPath: modelPath,
        promptFixturePath: promptFixturePath,
        normalizedTargetsPath: normalizedTargetsPath,
        outputPath: outputPath)
}

func parseKVTunerCandidatePlan(_ flags: Flags) throws -> KVTunerCandidatePlan {
    let modelPath = try requiredKVTunerFlag(flags, "model")
    let manifestPath = try requiredKVTunerFlag(flags, "manifest")
    let sensitivityPath = try requiredKVTunerFlag(flags, "sensitivity")
    let targetPairBitTotal = try positiveKVTunerFlag(
        flags, "target-pair-bits")
    let maxCandidates = try positiveKVTunerFlag(flags, "max-candidates")
    let outputDirectory = try requiredKVTunerFlag(flags, "output-dir")
    let candidateOrdinal = try flags.optionalStrictInt("candidate-ordinal")
    if let candidateOrdinal, candidateOrdinal < 0 {
        throw KVTunerQualificationCLIError.invalidOrdinal(candidateOrdinal)
    }
    try requireDistinctKVTunerPaths([
        modelPath, manifestPath, sensitivityPath, outputDirectory,
    ])
    return KVTunerCandidatePlan(
        modelPath: modelPath,
        manifestPath: manifestPath,
        sensitivityPath: sensitivityPath,
        targetPairBitTotal: targetPairBitTotal,
        maxCandidates: maxCandidates,
        outputDirectory: outputDirectory,
        candidateOrdinal: candidateOrdinal)
}

func parseKVTunerSensitivityPlan(
    _ flags: Flags
) throws -> KVTunerSensitivityPlan {
    let modelPath = try requiredKVTunerFlag(flags, "model")
    let manifestPath = try requiredKVTunerFlag(flags, "manifest")
    let matrixID = try requiredKVTunerFlag(flags, "matrix-id")
    let groupSize = try positiveKVTunerFlag(flags, "group-size")
    let outputPath = try requiredKVTunerFlag(flags, "output")
    guard [64, 128].contains(groupSize) else {
        throw KVTunerQualificationCLIError.unsupportedGroupSize(groupSize)
    }
    try requireDistinctKVTunerPaths([
        modelPath, manifestPath, outputPath,
    ])
    return KVTunerSensitivityPlan(
        modelPath: modelPath,
        manifestPath: manifestPath,
        matrixID: matrixID,
        groupSize: groupSize,
        outputPath: outputPath)
}

func parseKVTunerSearchPlan(_ flags: Flags) throws -> KVTunerSearchPlan {
    let modelPath = try requiredKVTunerFlag(flags, "model")
    let manifestPath = try requiredKVTunerFlag(flags, "manifest")
    let sensitivityPath = try requiredKVTunerFlag(flags, "sensitivity")
    let targetPairBitTotal = try positiveKVTunerFlag(
        flags, "target-pair-bits")
    let maxCandidates = try positiveKVTunerFlag(flags, "max-candidates")
    let candidateDirectory = try requiredKVTunerFlag(flags, "candidate-dir")
    let outputPath = try requiredKVTunerFlag(flags, "output")
    let scheduleOutputPath = try requiredKVTunerFlag(
        flags, "schedule-output")
    try requireDistinctKVTunerPaths([
        modelPath, manifestPath, sensitivityPath, candidateDirectory,
        outputPath, scheduleOutputPath,
    ])
    return KVTunerSearchPlan(
        modelPath: modelPath,
        manifestPath: manifestPath,
        sensitivityPath: sensitivityPath,
        targetPairBitTotal: targetPairBitTotal,
        maxCandidates: maxCandidates,
        candidateDirectory: candidateDirectory,
        outputPath: outputPath,
        scheduleOutputPath: scheduleOutputPath)
}

func parseKVTunerBundlePlan(_ flags: Flags) throws -> KVTunerBundlePlan {
    let modelPath = try requiredKVTunerFlag(flags, "model")
    let manifestPath = try requiredKVTunerFlag(flags, "manifest")
    let sensitivityPath = try requiredKVTunerFlag(flags, "sensitivity")
    let searchPath = try requiredKVTunerFlag(flags, "search")
    let schedulePath = try requiredKVTunerFlag(flags, "schedule")
    let candidateDirectory = try requiredKVTunerFlag(flags, "candidate-dir")
    let outputPath = try requiredKVTunerFlag(flags, "output")
    try requireDistinctKVTunerPaths([
        modelPath, manifestPath, sensitivityPath, searchPath, schedulePath,
        candidateDirectory, outputPath,
    ])
    return KVTunerBundlePlan(
        modelPath: modelPath,
        manifestPath: manifestPath,
        sensitivityPath: sensitivityPath,
        searchPath: searchPath,
        schedulePath: schedulePath,
        candidateDirectory: candidateDirectory,
        outputPath: outputPath)
}

func kvtunerCandidateArtifactURL(
    directory: String,
    ordinal: Int
) -> URL {
    URL(fileURLWithPath: directory, isDirectory: true)
        .appendingPathComponent(
            String(format: "candidate-%05d.json", ordinal),
            isDirectory: false)
}

private func requireFreshKVTunerArtifactDestination(_ path: String) throws {
    guard !path.isEmpty, !outputPathIsSymbolicLink(path) else {
        throw KVTunerQualificationCLIError.outputNotFresh(path)
    }

    let manager = FileManager.default
    let url = URL(fileURLWithPath: path).standardizedFileURL
    var isDirectory: ObjCBool = false
    if manager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
        let attributes = try manager.attributesOfItem(atPath: url.path)
        let type = attributes[.type] as? FileAttributeType
        let size = (attributes[.size] as? NSNumber)?.uint64Value
        guard !isDirectory.boolValue, type == .typeRegular, size == 0,
            manager.isWritableFile(atPath: url.path)
        else {
            throw KVTunerQualificationCLIError.outputNotFresh(path)
        }
        return
    }

    let parent = url.deletingLastPathComponent()
    var parentIsDirectory: ObjCBool = false
    guard manager.fileExists(
        atPath: parent.path, isDirectory: &parentIsDirectory),
        parentIsDirectory.boolValue,
        manager.isWritableFile(atPath: parent.path)
    else {
        throw KVTunerQualificationCLIError.outputNotFresh(path)
    }
}

func kvtunerArtifactLockURL(for path: String) -> URL {
    let output = URL(fileURLWithPath: path).standardizedFileURL
    return output.deletingLastPathComponent().appendingPathComponent(
        ".\(output.lastPathComponent).fastmlx-kvtuner.lock")
}

private func acquireKVTunerArtifactLock(for path: String) throws -> (
    descriptor: Int32,
    url: URL
) {
    let lockURL = kvtunerArtifactLockURL(for: path)
    let descriptor = open(
        lockURL.path,
        O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW,
        S_IRUSR | S_IWUSR)
    guard descriptor >= 0 else {
        throw KVTunerQualificationCLIError.outputNotFresh(path)
    }
    return (descriptor, lockURL)
}

/// Writes one authenticated artifact without silently replacing prior evidence. A sibling
/// exclusive-create lock serializes qualification producers. Complete bytes are first written to
/// a unique sibling and then moved into the absent destination on the same volume; `FileManager`
/// refuses that move if another creator has populated the final path. The zero-byte recovery form
/// is removed only while the cooperating-producer lock is held.
func writeFreshKVTunerArtifact(_ data: Data, to path: String) throws {
    let lock = try acquireKVTunerArtifactLock(for: path)
    defer {
        close(lock.descriptor)
        unlink(lock.url.path)
    }
    try requireFreshKVTunerArtifactDestination(path)
    let outputURL = URL(fileURLWithPath: path).standardizedFileURL
    if FileManager.default.fileExists(atPath: outputURL.path) {
        // Freshness validation above proves that this is the permitted zero-byte regular-file
        // recovery case. Removing it lets the final write retain no-overwrite semantics.
        try FileManager.default.removeItem(at: outputURL)
    }
    let temporaryURL = outputURL.deletingLastPathComponent()
        .appendingPathComponent(
            ".\(outputURL.lastPathComponent).\(UUID().uuidString).tmp")
    defer { try? FileManager.default.removeItem(at: temporaryURL) }
    try data.write(to: temporaryURL, options: .withoutOverwriting)
    let temporaryHandle = try FileHandle(forWritingTo: temporaryURL)
    try temporaryHandle.synchronize()
    try temporaryHandle.close()
    try FileManager.default.moveItem(at: temporaryURL, to: outputURL)
    let directoryDescriptor = open(
        outputURL.deletingLastPathComponent().path,
        O_RDONLY | O_CLOEXEC)
    guard directoryDescriptor >= 0 else {
        throw KVTunerQualificationCLIError.outputNotFresh(path)
    }
    defer { close(directoryDescriptor) }
    guard fsync(directoryDescriptor) == 0 else {
        throw KVTunerQualificationCLIError.outputNotFresh(path)
    }
}

/// Idempotent completion boundary for multi-artifact or interrupted qualification stages. An
/// already-present output is accepted only when its exact bytes equal the canonical bytes this
/// invocation independently re-derived; different evidence is never replaced.
@discardableResult
func writeOrValidateExactKVTunerArtifact(
    _ data: Data,
    to path: String
) throws -> Bool {
    guard !outputPathIsSymbolicLink(path) else {
        throw KVTunerQualificationCLIError.outputNotFresh(path)
    }
    let url = URL(fileURLWithPath: path).standardizedFileURL
    var isDirectory: ObjCBool = false
    if FileManager.default.fileExists(
        atPath: url.path, isDirectory: &isDirectory)
    {
        let attributes = try FileManager.default.attributesOfItem(
            atPath: url.path)
        let type = attributes[.type] as? FileAttributeType
        let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        guard !isDirectory.boolValue, type == .typeRegular else {
            throw KVTunerQualificationCLIError.outputNotFresh(path)
        }
        if size > 0 {
            guard try Data(contentsOf: url) == data else {
                throw KVTunerQualificationCLIError.outputNotFresh(path)
            }
            return false
        }
    }
    try writeFreshKVTunerArtifact(data, to: path)
    guard try Data(contentsOf: url) == data else {
        throw KVTunerQualificationCLIError.outputNotFresh(path)
    }
    return true
}
