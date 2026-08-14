import CryptoKit
import Foundation

struct SyntheticSharedCacheFileRecord:
    Equatable,
    Sendable
{
    let suffixBytes: UInt64
    let suffixBase64URL: String
    let fileSHA256: String
    let fileBytes: UInt64
    let headerUUID: String
}

enum SyntheticSharedCacheFileRecordField:
    Equatable,
    Sendable
{
    case suffixBytes
    case suffixBase64URL
    case fileSHA256
    case fileBytes
    case headerUUID
}

enum SyntheticSharedCacheSetIdentityFailure:
    Error,
    Equatable,
    Sendable
{
    case recordCountOutOfRange(Int)
    case invalidRecord(
        index: Int,
        field: SyntheticSharedCacheFileRecordField
    )
    case mainRecordMissing
    case mainRecordNotFirst(Int)
    case duplicateSuffix(Int)
    case duplicateHeaderUUID(Int)
    case recordsNotSorted(Int)
}

struct SharedCacheSetID:
    Equatable,
    Sendable
{
    let sha256: String

    fileprivate init(sha256: String) {
        self.sha256 = sha256
    }
}

/// Retained synthetic comparison facts only. This value neither locates nor
/// captures a shared cache and cannot override the runtime-policy denial.
struct SyntheticSharedCacheSetIdentityEvidence: Equatable {
    let records: [SyntheticSharedCacheFileRecord]
    let decodedSuffixes: [Data]
    let identityPreimage: Data
    let sharedCacheSetID: SharedCacheSetID

    let canExecute = false
    let canSpawn = false
    let canAccessNetwork = false
    let canConsumePack = false
    let canMutateFileSystem = false
    let canImportGitObjects = false
    let canBuild = false
    let canLoadModel = false
    let canReserveOutput = false
    let canPublish = false

    fileprivate init(
        records: [SyntheticSharedCacheFileRecord],
        decodedSuffixes: [Data],
        identityPreimage: Data
    ) {
        self.records = records
        self.decodedSuffixes = decodedSuffixes
        self.identityPreimage = identityPreimage
        self.sharedCacheSetID = SharedCacheSetID(
            sha256: Self.sha256Hex(identityPreimage)
        )
    }

    private static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}

enum SyntheticSharedCacheSetIdentityVerifier {
    static let identityDomain =
        "fast-mlx-proof-control-shared-cache-set-id-v1"
    static let maximumRecordCount = 64

    static func derive(
        records input: [SyntheticSharedCacheFileRecord]
    ) throws -> SyntheticSharedCacheSetIdentityEvidence {
        guard (1...maximumRecordCount).contains(input.count) else {
            throw SyntheticSharedCacheSetIdentityFailure
                .recordCountOutOfRange(input.count)
        }

        let records = Array(input)
        var decodedSuffixes: [Data] = []
        decodedSuffixes.reserveCapacity(records.count)

        for (index, record) in records.enumerated() {
            guard let decoded = decodeCanonicalBase64URL(
                record.suffixBase64URL
            ) else {
                throw SyntheticSharedCacheSetIdentityFailure
                    .invalidRecord(
                        index: index,
                        field: .suffixBase64URL
                    )
            }
            guard UInt64(decoded.count) == record.suffixBytes else {
                throw SyntheticSharedCacheSetIdentityFailure
                    .invalidRecord(
                        index: index,
                        field: .suffixBytes
                    )
            }
            guard isLowercaseHex(record.fileSHA256, count: 64) else {
                throw SyntheticSharedCacheSetIdentityFailure
                    .invalidRecord(
                        index: index,
                        field: .fileSHA256
                    )
            }
            guard record.fileBytes > 0 else {
                throw SyntheticSharedCacheSetIdentityFailure
                    .invalidRecord(
                        index: index,
                        field: .fileBytes
                    )
            }
            guard isLowercaseHex(record.headerUUID, count: 32) else {
                throw SyntheticSharedCacheSetIdentityFailure
                    .invalidRecord(
                        index: index,
                        field: .headerUUID
                    )
            }
            decodedSuffixes.append(decoded)
        }

        let mainIndices = decodedSuffixes.indices.filter {
            decodedSuffixes[$0].isEmpty
        }
        guard let firstMain = mainIndices.first else {
            throw SyntheticSharedCacheSetIdentityFailure
                .mainRecordMissing
        }
        guard firstMain == 0 else {
            throw SyntheticSharedCacheSetIdentityFailure
                .mainRecordNotFirst(firstMain)
        }
        if mainIndices.count > 1 {
            throw SyntheticSharedCacheSetIdentityFailure
                .duplicateSuffix(mainIndices[1])
        }

        var suffixes = Set<Data>()
        var headerUUIDs = Set<String>()
        for index in records.indices {
            guard suffixes.insert(decodedSuffixes[index]).inserted else {
                throw SyntheticSharedCacheSetIdentityFailure
                    .duplicateSuffix(index)
            }
            guard headerUUIDs.insert(records[index].headerUUID)
                .inserted
            else {
                throw SyntheticSharedCacheSetIdentityFailure
                    .duplicateHeaderUUID(index)
            }
            if index > 0,
               !decodedSuffixes[index - 1]
                   .lexicographicallyPrecedes(decodedSuffixes[index])
            {
                throw SyntheticSharedCacheSetIdentityFailure
                    .recordsNotSorted(index)
            }
        }

        let preimage = identityPreimage(records: records)
        return SyntheticSharedCacheSetIdentityEvidence(
            records: records,
            decodedSuffixes: decodedSuffixes,
            identityPreimage: preimage
        )
    }

    private static func identityPreimage(
        records: [SyntheticSharedCacheFileRecord]
    ) -> Data {
        var lines = [
            identityDomain,
            "shared_cache_file_count=\(records.count)",
        ]
        for (index, record) in records.enumerated() {
            let decimalIndex = String(index)
            let prefix = "shared_cache_file_" +
                String(
                    repeating: "0",
                    count: 4 - decimalIndex.count
                ) +
                decimalIndex
            lines.append(
                "\(prefix)_suffix_bytes=\(record.suffixBytes)"
            )
            lines.append(
                "\(prefix)_suffix_base64url=" +
                    record.suffixBase64URL
            )
            lines.append(
                "\(prefix)_sha256=\(record.fileSHA256)"
            )
            lines.append(
                "\(prefix)_bytes=\(record.fileBytes)"
            )
            lines.append(
                "\(prefix)_header_uuid=\(record.headerUUID)"
            )
        }
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    private static func decodeCanonicalBase64URL(
        _ encoded: String
    ) -> Data? {
        let bytes = Array(encoded.utf8)
        guard bytes.allSatisfy({
            (0x41...0x5a).contains($0) ||
                (0x61...0x7a).contains($0) ||
                (0x30...0x39).contains($0) ||
                $0 == 0x2d ||
                $0 == 0x5f
        }),
        bytes.count % 4 != 1
        else {
            return nil
        }

        let standard = encoded
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        let padded = standard + String(
            repeating: "=",
            count: (4 - standard.utf8.count % 4) % 4
        )
        guard let decoded = Data(base64Encoded: padded) else {
            return nil
        }

        let canonical = decoded.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
        guard canonical == encoded else {
            return nil
        }
        return decoded
    }

    private static func isLowercaseHex(
        _ value: String,
        count: Int
    ) -> Bool {
        let bytes = Array(value.utf8)
        return bytes.count == count &&
            bytes.allSatisfy {
                (0x30...0x39).contains($0) ||
                    (0x61...0x66).contains($0)
            }
    }
}
