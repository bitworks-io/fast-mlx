import CryptoKit
import Darwin
import Foundation

public enum AdmissionError: Error, Equatable, Sendable {
    case nonCanonicalAbsolutePath
    case invalidMaximumBytes
    case pathComponentRejected(component: String, code: Int32)
    case metadataReadFailed(code: Int32)
    case notRegularFile
    case unexpectedLinkCount(UInt64)
    case invalidFileSize
    case exceedsMaximumBytes(maximum: Int, actual: UInt64)
    case readFailed(code: Int32)
    case identityChanged
}

extension AdmissionError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .nonCanonicalAbsolutePath:
            "path must be absolute and contain no empty, dot, or dot-dot components"
        case .invalidMaximumBytes:
            "maximum byte count must be nonnegative"
        case .pathComponentRejected(let component, let code):
            "path component \(component) rejected: errno=\(code)"
        case .metadataReadFailed(let code):
            "fstat failed: errno=\(code)"
        case .notRegularFile:
            "admitted input is not a regular file"
        case .unexpectedLinkCount(let count):
            "admitted input has unexpected link count \(count)"
        case .invalidFileSize:
            "admitted input has an invalid file size"
        case .exceedsMaximumBytes(let maximum, let actual):
            "admitted input has \(actual) bytes, exceeding limit \(maximum)"
        case .readFailed(let code):
            "admitted input read failed: errno=\(code)"
        case .identityChanged:
            "admitted input identity changed while bytes were captured"
        }
    }
}

public struct AdmittedFileIdentity: Equatable, Sendable {
    public let device: UInt64
    public let inode: UInt64
    public let size: UInt64
    public let linkCount: UInt64
    public let mode: UInt16
    public let modificationSeconds: Int64
    public let modificationNanoseconds: Int64
    public let changeSeconds: Int64
    public let changeNanoseconds: Int64
}

public struct AdmittedFile: Equatable, Sendable {
    public let bytes: Data
    public let sha256: String
    public let identity: AdmittedFileIdentity

    public static func capture(
        absolutePath: String,
        maximumBytes: Int
    ) throws -> Self {
        guard maximumBytes >= 0 else {
            throw AdmissionError.invalidMaximumBytes
        }

        let components = try canonicalComponents(of: absolutePath)
        let descriptor = try openWithoutFollowingSymlinks(components: components)
        defer { Darwin.close(descriptor) }

        var metadataBefore = stat()
        guard fstat(descriptor, &metadataBefore) == 0 else {
            throw AdmissionError.metadataReadFailed(code: errno)
        }
        guard (metadataBefore.st_mode & S_IFMT) == S_IFREG else {
            throw AdmissionError.notRegularFile
        }

        let linkCount = UInt64(metadataBefore.st_nlink)
        guard linkCount == 1 else {
            throw AdmissionError.unexpectedLinkCount(linkCount)
        }
        guard metadataBefore.st_size >= 0 else {
            throw AdmissionError.invalidFileSize
        }

        let expectedSize = UInt64(metadataBefore.st_size)
        guard expectedSize <= UInt64(maximumBytes) else {
            throw AdmissionError.exceedsMaximumBytes(
                maximum: maximumBytes,
                actual: expectedSize
            )
        }

        let bytes = try readBytes(
            descriptor: descriptor,
            expectedSize: expectedSize,
            maximumBytes: maximumBytes
        )

        var metadataAfter = stat()
        guard fstat(descriptor, &metadataAfter) == 0 else {
            throw AdmissionError.metadataReadFailed(code: errno)
        }
        guard stableIdentity(metadataBefore) == stableIdentity(metadataAfter) else {
            throw AdmissionError.identityChanged
        }
        guard UInt64(bytes.count) == expectedSize else {
            throw AdmissionError.identityChanged
        }

        return Self(
            bytes: bytes,
            sha256: SHA256.hash(data: bytes)
                .map { String(format: "%02x", $0) }
                .joined(),
            identity: stableIdentity(metadataAfter)
        )
    }
}

private extension AdmittedFile {
    static func canonicalComponents(of absolutePath: String) throws -> [String] {
        guard
            !absolutePath.utf8.contains(0),
            absolutePath.hasPrefix("/")
        else {
            throw AdmissionError.nonCanonicalAbsolutePath
        }

        let rawComponents = absolutePath.split(
            separator: "/",
            omittingEmptySubsequences: false
        )
        guard
            rawComponents.first?.isEmpty == true,
            rawComponents.count > 1
        else {
            throw AdmissionError.nonCanonicalAbsolutePath
        }

        let components = rawComponents.dropFirst().map(String.init)
        guard components.allSatisfy({
            !$0.isEmpty && $0 != "." && $0 != ".."
        }) else {
            throw AdmissionError.nonCanonicalAbsolutePath
        }
        return components
    }

    static func openWithoutFollowingSymlinks(components: [String]) throws -> Int32 {
        let directoryFlags = O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
        var directoryDescriptor = Darwin.open("/", directoryFlags)
        guard directoryDescriptor >= 0 else {
            throw AdmissionError.pathComponentRejected(component: "/", code: errno)
        }

        for component in components.dropLast() {
            let nextDescriptor = component.withCString {
                Darwin.openat(directoryDescriptor, $0, directoryFlags)
            }
            guard nextDescriptor >= 0 else {
                let code = errno
                Darwin.close(directoryDescriptor)
                throw AdmissionError.pathComponentRejected(
                    component: component,
                    code: code
                )
            }
            Darwin.close(directoryDescriptor)
            directoryDescriptor = nextDescriptor
        }

        let finalComponent = components[components.index(before: components.endIndex)]
        // fstat validates the type after open; nonblocking prevents a FIFO from hanging first.
        let descriptor = finalComponent.withCString {
            Darwin.openat(
                directoryDescriptor,
                $0,
                O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC
            )
        }
        let openCode = errno
        Darwin.close(directoryDescriptor)
        guard descriptor >= 0 else {
            throw AdmissionError.pathComponentRejected(
                component: finalComponent,
                code: openCode
            )
        }
        return descriptor
    }

    static func readBytes(
        descriptor: Int32,
        expectedSize: UInt64,
        maximumBytes: Int
    ) throws -> Data {
        var result = Data()
        result.reserveCapacity(Int(expectedSize))
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)

        while true {
            let count = buffer.withUnsafeMutableBytes { storage in
                Darwin.read(descriptor, storage.baseAddress, storage.count)
            }
            if count == 0 {
                return result
            }
            if count < 0, errno == EINTR {
                continue
            }
            guard count > 0 else {
                throw AdmissionError.readFailed(code: errno)
            }
            result.append(contentsOf: buffer.prefix(Int(count)))
            guard result.count <= maximumBytes else {
                throw AdmissionError.exceedsMaximumBytes(
                    maximum: maximumBytes,
                    actual: UInt64(result.count)
                )
            }
        }
    }

    static func stableIdentity(_ metadata: stat) -> AdmittedFileIdentity {
        AdmittedFileIdentity(
            device: UInt64(metadata.st_dev),
            inode: UInt64(metadata.st_ino),
            size: UInt64(metadata.st_size),
            linkCount: UInt64(metadata.st_nlink),
            mode: UInt16(metadata.st_mode),
            modificationSeconds: Int64(metadata.st_mtimespec.tv_sec),
            modificationNanoseconds: Int64(metadata.st_mtimespec.tv_nsec),
            changeSeconds: Int64(metadata.st_ctimespec.tv_sec),
            changeNanoseconds: Int64(metadata.st_ctimespec.tv_nsec)
        )
    }
}
