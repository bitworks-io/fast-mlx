import Darwin
import Foundation
import ServingCore

public actor ServingEvidenceJSONLSink {
    public enum Error: Swift.Error, Equatable, Sendable {
        case pathMustBeAbsolute
        case parentDirectoryMissing
        case outputAlreadyExists
        case openFailed
        case writeFailed
        case syncFailed
        case sinkClosed
    }

    private var descriptor: Int32
    private var terminalFailure: Error?
    private let simulatedWriteFailureAfterBytes: Int?

    public init(path: String) throws {
        try self.init(
            path: path,
            simulatedWriteFailureAfterBytes: nil)
    }

    init(
        path: String,
        simulatedWriteFailureAfterBytes: Int?
    ) throws {
        guard path.hasPrefix("/") else {
            throw Error.pathMustBeAbsolute
        }
        if let simulatedWriteFailureAfterBytes {
            precondition(simulatedWriteFailureAfterBytes >= 0)
        }
        let parent = URL(fileURLWithPath: path)
            .deletingLastPathComponent()
            .path
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: parent,
            isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw Error.parentDirectoryMissing
        }

        let parentDescriptor = parent.withCString { pointer in
            Darwin.open(pointer, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        }
        guard parentDescriptor >= 0 else {
            throw Error.openFailed
        }
        defer {
            Darwin.close(parentDescriptor)
        }

        errno = 0
        let flags =
            O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_NOFOLLOW | O_CLOEXEC
        let mode = mode_t(S_IRUSR | S_IWUSR)
        let opened = path.withCString { pointer in
            Darwin.open(pointer, flags, mode)
        }
        guard opened >= 0 else {
            if errno == EEXIST {
                throw Error.outputAlreadyExists
            }
            throw Error.openFailed
        }
        guard syncDescriptor(parentDescriptor) else {
            Darwin.close(opened)
            path.withCString { pointer in
                _ = Darwin.unlink(pointer)
            }
            throw Error.syncFailed
        }
        descriptor = opened
        terminalFailure = nil
        self.simulatedWriteFailureAfterBytes =
            simulatedWriteFailureAfterBytes
    }

    deinit {
        if descriptor >= 0 {
            Darwin.close(descriptor)
        }
    }

    public func record(_ evidence: ServingEvidence) throws {
        if let terminalFailure {
            throw terminalFailure
        }
        guard descriptor >= 0 else {
            throw Error.sinkClosed
        }
        var line = try evidence.canonicalJSONData()
        line.append(0x0A)
        let startingOffset = Darwin.lseek(descriptor, 0, SEEK_END)
        guard startingOffset >= 0 else {
            throw latch(.writeFailed, rollbackTo: nil)
        }
        do {
            try line.withUnsafeBytes { rawBuffer in
                guard let baseAddress = rawBuffer.baseAddress else {
                    throw Error.writeFailed
                }
                var offset = 0
                while offset < rawBuffer.count {
                    if let simulatedWriteFailureAfterBytes,
                        offset >= simulatedWriteFailureAfterBytes
                    {
                        throw Error.writeFailed
                    }
                    let remaining = rawBuffer.count - offset
                    let countToWrite: Int
                    if let simulatedWriteFailureAfterBytes {
                        countToWrite = min(
                            remaining,
                            simulatedWriteFailureAfterBytes - offset)
                    } else {
                        countToWrite = remaining
                    }
                    let count = Darwin.write(
                        descriptor,
                        baseAddress.advanced(by: offset),
                        countToWrite)
                    if count < 0, errno == EINTR {
                        continue
                    }
                    guard count > 0 else {
                        throw Error.writeFailed
                    }
                    offset += count
                }
            }
        } catch let error as Error {
            throw latch(error, rollbackTo: startingOffset)
        } catch {
            throw latch(.writeFailed, rollbackTo: startingOffset)
        }
        guard syncDescriptor(descriptor) else {
            throw latch(.syncFailed, rollbackTo: startingOffset)
        }
    }

    public func finish() throws {
        if let terminalFailure {
            throw terminalFailure
        }
        guard descriptor >= 0 else {
            return
        }
        guard syncDescriptor(descriptor) else {
            throw latch(.syncFailed, rollbackTo: nil)
        }
        let closing = descriptor
        descriptor = -1
        guard Darwin.close(closing) == 0 else {
            terminalFailure = .syncFailed
            throw Error.syncFailed
        }
    }

    private func latch(
        _ failure: Error,
        rollbackTo startingOffset: off_t?
    ) -> Error {
        var terminal = failure
        if let startingOffset {
            if !truncateDescriptor(descriptor, to: startingOffset)
                || !syncDescriptor(descriptor)
            {
                terminal = .syncFailed
            }
        }
        let closing = descriptor
        descriptor = -1
        if closing >= 0, Darwin.close(closing) != 0 {
            terminal = .syncFailed
        }
        terminalFailure = terminal
        return terminal
    }
}

private func syncDescriptor(_ descriptor: Int32) -> Bool {
    while true {
        if Darwin.fsync(descriptor) == 0 {
            return true
        }
        if errno != EINTR {
            return false
        }
    }
}

private func truncateDescriptor(
    _ descriptor: Int32,
    to offset: off_t
) -> Bool {
    while true {
        if Darwin.ftruncate(descriptor, offset) == 0 {
            return true
        }
        if errno != EINTR {
            return false
        }
    }
}
