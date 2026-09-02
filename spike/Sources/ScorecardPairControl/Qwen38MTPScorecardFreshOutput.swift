import Darwin
import Foundation
import HarnessCore

package struct Qwen38MTPScorecardFreshOutputSet: Equatable, Sendable {
    package let scorecardPath: String
    package let authorityPath: String

    package init(scorecardPath: String, authorityPath: String) throws {
        guard scorecardPath != authorityPath else {
            throw Qwen38MTPScorecardLiveAdapterError.unsafeOutput
        }
        try Self.validateFreshPath(scorecardPath)
        try Self.validateFreshPath(authorityPath)
        self.scorecardPath = scorecardPath
        self.authorityPath = authorityPath
    }

    package static func validateFreshPath(_ path: String) throws {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let manager = FileManager.default
        guard !path.isEmpty, !outputPathIsSymbolicLink(url.path) else {
            throw Qwen38MTPScorecardLiveAdapterError.unsafeOutput
        }
        if manager.fileExists(atPath: url.path) {
            throw Qwen38MTPScorecardLiveAdapterError.outputExists
        }
        var isDirectory: ObjCBool = false
        guard manager.fileExists(atPath: url.deletingLastPathComponent().path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw Qwen38MTPScorecardLiveAdapterError.unsafeOutput
        }
    }
}

package func writeFreshQwen38MTPScorecardAuthority(_ data: Data, _ path: String) throws {
    guard !data.isEmpty else {
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
    let outputURL = URL(fileURLWithPath: path).standardizedFileURL
    try Qwen38MTPScorecardFreshOutputSet.validateFreshPath(outputURL.path)
    let manager = FileManager.default
    let parent = outputURL.deletingLastPathComponent()
    let temporaryURL = parent.appendingPathComponent(
        ".\(outputURL.lastPathComponent).\(UUID().uuidString).tmp")
    try data.write(to: temporaryURL, options: .withoutOverwriting)
    do {
        try manager.linkItem(at: temporaryURL, to: outputURL)
        try manager.removeItem(at: temporaryURL)
    } catch CocoaError.fileWriteFileExists {
        try? manager.removeItem(at: temporaryURL)
        throw Qwen38MTPScorecardLiveAdapterError.outputExists
    } catch {
        try? manager.removeItem(at: temporaryURL)
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
}

package func writeFreshQwen38MTPScorecardOutputSet(
    scorecardData: Data,
    authorityData: Data,
    outputs: Qwen38MTPScorecardFreshOutputSet,
    fsyncParents: ([URL]) throws -> Void = fsyncQwen38MTPScorecardParentDirectories
) throws {
    guard !scorecardData.isEmpty, scorecardData.last == 0x0a,
        scorecardData.dropLast().firstIndex(of: 0x0a) == nil,
        !authorityData.isEmpty
    else {
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
    try Qwen38MTPScorecardFreshOutputSet.validateFreshPath(outputs.scorecardPath)
    try Qwen38MTPScorecardFreshOutputSet.validateFreshPath(outputs.authorityPath)

    let scorecardURL = URL(fileURLWithPath: outputs.scorecardPath).standardizedFileURL
    let authorityURL = URL(fileURLWithPath: outputs.authorityPath).standardizedFileURL
    let scorecardTemporaryURL = try writeQwen38MTPScorecardTemporaryFile(
        scorecardData,
        near: scorecardURL)
    let authorityTemporaryURL = try writeQwen38MTPScorecardTemporaryFile(
        authorityData,
        near: authorityURL)
    var scorecardLinked = false
    var authorityLinked = false
    defer {
        _ = scorecardTemporaryURL.withUnsafeFileSystemRepresentation { pointer in
            pointer.map(unlink) ?? -1
        }
        _ = authorityTemporaryURL.withUnsafeFileSystemRepresentation { pointer in
            pointer.map(unlink) ?? -1
        }
    }

    do {
        try linkQwen38MTPScorecardTemporaryFile(scorecardTemporaryURL, to: scorecardURL)
        scorecardLinked = true
        try linkQwen38MTPScorecardTemporaryFile(authorityTemporaryURL, to: authorityURL)
        authorityLinked = true
        try fsyncParents(
            [scorecardURL.deletingLastPathComponent(), authorityURL.deletingLastPathComponent()])
    } catch {
        if authorityLinked {
            _ = authorityURL.withUnsafeFileSystemRepresentation { pointer in
                pointer.map(unlink) ?? -1
            }
        }
        if scorecardLinked {
            _ = scorecardURL.withUnsafeFileSystemRepresentation { pointer in
                pointer.map(unlink) ?? -1
            }
        }
        throw error
    }
}

private func writeQwen38MTPScorecardTemporaryFile(
    _ data: Data,
    near outputURL: URL
) throws -> URL {
    let temporaryURL = outputURL.deletingLastPathComponent().appendingPathComponent(
        ".\(outputURL.lastPathComponent).\(UUID().uuidString).tmp")
    let descriptor = temporaryURL.withUnsafeFileSystemRepresentation { pointer -> Int32 in
        guard let pointer else { return -1 }
        return open(pointer, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
    }
    guard descriptor >= 0 else {
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
    var descriptorOpen = true
    var removeTemporaryOnFailure = true
    defer {
        if descriptorOpen { _ = close(descriptor) }
        if removeTemporaryOnFailure {
            _ = temporaryURL.withUnsafeFileSystemRepresentation { pointer in
                pointer.map(unlink) ?? -1
            }
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
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
    guard close(descriptor) == 0 else {
        descriptorOpen = false
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
    descriptorOpen = false
    removeTemporaryOnFailure = false
    return temporaryURL
}

private func linkQwen38MTPScorecardTemporaryFile(_ temporaryURL: URL, to outputURL: URL) throws {
    let linked = temporaryURL.withUnsafeFileSystemRepresentation { temporaryPointer -> Int32 in
        outputURL.withUnsafeFileSystemRepresentation { outputPointer -> Int32 in
            guard let temporaryPointer, let outputPointer else { return -1 }
            return Darwin.link(temporaryPointer, outputPointer)
        }
    }
    guard linked == 0 else {
        if errno == EEXIST {
            throw Qwen38MTPScorecardLiveAdapterError.outputExists
        }
        throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
    }
}

private func fsyncQwen38MTPScorecardParentDirectories(_ directories: [URL]) throws {
    var seen: Set<String> = []
    for directory in directories where seen.insert(directory.path).inserted {
        let descriptor = directory.withUnsafeFileSystemRepresentation { pointer -> Int32 in
            guard let pointer else { return -1 }
            return open(pointer, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        }
        guard descriptor >= 0 else {
            throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
        }
        var descriptorOpen = true
        defer {
            if descriptorOpen { _ = close(descriptor) }
        }
        guard fsync(descriptor) == 0 else {
            throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
        }
        guard close(descriptor) == 0 else {
            descriptorOpen = false
            throw Qwen38MTPScorecardLiveAdapterError.outputWriteFailed
        }
        descriptorOpen = false
    }
}
