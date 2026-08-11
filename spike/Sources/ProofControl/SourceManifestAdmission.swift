import CryptoKit
import Foundation

public enum SourceManifestFileMode: String, Equatable, Sendable {
    case regular = "100644"
    case executable = "100755"
}

public struct SourceManifestEntry: Equatable, Sendable {
    public let mode: SourceManifestFileMode
    public let gitBlobSHA1: String
    public let byteCount: UInt64
    public let sha256: String
    public let path: String

    public init(
        mode: SourceManifestFileMode,
        gitBlobSHA1: String,
        byteCount: UInt64,
        sha256: String,
        path: String
    ) {
        self.mode = mode
        self.gitBlobSHA1 = gitBlobSHA1
        self.byteCount = byteCount
        self.sha256 = sha256
        self.path = path
    }
}

public enum SourceManifestAdmissionError: Error, Equatable, Sendable {
    case unexpectedPurpose(
        expected: OperatorAuthorizationPurpose,
        actual: OperatorAuthorizationPurpose
    )
    case unexpectedRole(expected: RunSourceRole, actual: RunSourceRole)
    case nonCanonicalManifest
    case sourceAuthorizationIDMismatch(role: RunSourceRole)
    case sourceManifestDigestMismatch(role: RunSourceRole)
    case sourceManifestByteCountMismatch(role: RunSourceRole)
    case sourceRoleMismatch(role: RunSourceRole)
    case gitCommitMismatch(role: RunSourceRole)
    case gitTreeMismatch(role: RunSourceRole)
    case routeMismatch(role: RunSourceRole)
    case slotMismatch(role: RunSourceRole)
}

extension SourceManifestAdmissionError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .unexpectedPurpose(let expected, let actual):
            "source manifest purpose \(actual.rawValue) does not match \(expected.rawValue)"
        case .unexpectedRole(let expected, let actual):
            "source manifest role \(actual.rawValue) does not match \(expected.rawValue)"
        case .nonCanonicalManifest:
            "source manifest is not the canonical fixed-order UTF-8 form"
        case .sourceAuthorizationIDMismatch(let role):
            "\(role.rawValue) source authorization ID does not match the signed claim"
        case .sourceManifestDigestMismatch(let role):
            "\(role.rawValue) source manifest digest does not match the signed claim"
        case .sourceManifestByteCountMismatch(let role):
            "\(role.rawValue) source manifest byte count does not match the signed claim"
        case .sourceRoleMismatch(let role):
            "\(role.rawValue) source role does not match the signed claim"
        case .gitCommitMismatch(let role):
            "\(role.rawValue) source commit does not match the signed claim"
        case .gitTreeMismatch(let role):
            "\(role.rawValue) source tree does not match the signed claim"
        case .routeMismatch(let role):
            "\(role.rawValue) source route does not match the signed claim"
        case .slotMismatch(let role):
            "\(role.rawValue) source slot does not match the signed claim"
        }
    }
}

public struct AdmittedSourceManifestID:
    Equatable,
    Hashable,
    Sendable
{
    public let rawValue: String

    fileprivate init(rawValue: String) {
        self.rawValue = rawValue
    }
}

/// Canonical, purpose-authorized source-manifest evidence only.
///
/// This value does not admit the signing policy, import Git objects,
/// materialize source, build, spawn, load a model, or reserve output.
public struct AdmittedSourceManifest: Equatable, Sendable {
    public let manifestID: AdmittedSourceManifestID
    public let authorizedFile: OperatorAuthorizedFile
    public let role: RunSourceRole
    public let route: OperatorRunClaimRoute
    public let slot: OperatorRunClaimSlot
    public let gitCommitSHA1: String
    public let gitTreeSHA1: String
    public let entries: [SourceManifestEntry]

    fileprivate init(
        manifestID: AdmittedSourceManifestID,
        authorizedFile: OperatorAuthorizedFile,
        role: RunSourceRole,
        route: OperatorRunClaimRoute,
        slot: OperatorRunClaimSlot,
        gitCommitSHA1: String,
        gitTreeSHA1: String,
        entries: [SourceManifestEntry]
    ) {
        self.manifestID = manifestID
        self.authorizedFile = authorizedFile
        self.role = role
        self.route = route
        self.slot = slot
        self.gitCommitSHA1 = gitCommitSHA1
        self.gitTreeSHA1 = gitTreeSHA1
        self.entries = entries
    }
}

public struct ClaimMatchedSourceManifestID:
    Equatable,
    Hashable,
    Sendable
{
    public let rawValue: String

    fileprivate init(rawValue: String) {
        self.rawValue = rawValue
    }
}

/// Inert evidence that one admitted source manifest matches one signed claim.
public struct ClaimMatchedSourceManifest: Equatable, Sendable {
    public let matchID: ClaimMatchedSourceManifestID
    public let signedClaimID: OperatorSignedRunClaimID
    public let sourceManifest: AdmittedSourceManifest

    fileprivate init(
        matchID: ClaimMatchedSourceManifestID,
        signedClaimID: OperatorSignedRunClaimID,
        sourceManifest: AdmittedSourceManifest
    ) {
        self.matchID = matchID
        self.signedClaimID = signedClaimID
        self.sourceManifest = sourceManifest
    }
}

public enum SourceManifestAdmission {
    public static let manifestDomain =
        "fast-mlx-proof-control-source-manifest-v1"
    public static let manifestSubject =
        OperatorRunClaimSubject.absorbedMLALoadedResultPair.rawValue
    public static let admittedManifestIDDomain =
        "fast-mlx-proof-control-admitted-source-manifest-id-v1"
    public static let claimMatchIDDomain =
        "fast-mlx-proof-control-claim-matched-source-manifest-id-v1"
    public static let maximumEntryCount: UInt64 = 1_000_000
    public static let maximumPathByteCount = 4_096
    public static let maximumPathComponentByteCount = 255

    public static func manifestBytes(
        role: RunSourceRole,
        gitCommitSHA1: String,
        gitTreeSHA1: String,
        entries: [SourceManifestEntry]
    ) throws -> Data {
        guard
            isLowercaseHex(gitCommitSHA1, count: 40),
            isLowercaseHex(gitTreeSHA1, count: 40)
        else {
            throw SourceManifestAdmissionError.nonCanonicalManifest
        }

        let canonicalEntries = try canonicalEntries(entries)
        let binding = sourceBinding(for: role)
        var lines = [
            manifestDomain,
            "subject=\(manifestSubject)",
            "role=\(role.rawValue)",
            "route=\(binding.route.rawValue)",
            "slot=\(binding.slot.rawValue)",
            "git_commit_sha1=\(gitCommitSHA1)",
            "git_tree_sha1=\(gitTreeSHA1)",
            "entry_count=\(canonicalEntries.count)",
        ]
        lines.append(contentsOf: canonicalEntries.map {
            [
                "entry=\($0.mode.rawValue)",
                $0.gitBlobSHA1,
                String($0.byteCount),
                $0.sha256,
                $0.path,
            ].joined(separator: "\t")
        })
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    public static func admit(
        authorizedFile: OperatorAuthorizedFile,
        expectedRole: RunSourceRole
    ) throws -> AdmittedSourceManifest {
        guard authorizedFile.purpose == .sourceManifest else {
            throw SourceManifestAdmissionError.unexpectedPurpose(
                expected: .sourceManifest,
                actual: authorizedFile.purpose
            )
        }

        let parsed = try parseManifest(authorizedFile.file.bytes)
        guard parsed.role == expectedRole else {
            throw SourceManifestAdmissionError.unexpectedRole(
                expected: expectedRole,
                actual: parsed.role
            )
        }

        let manifestID = admittedManifestID(
            authorizedFile: authorizedFile,
            parsed: parsed
        )
        return AdmittedSourceManifest(
            manifestID: AdmittedSourceManifestID(rawValue: manifestID),
            authorizedFile: authorizedFile,
            role: parsed.role,
            route: parsed.route,
            slot: parsed.slot,
            gitCommitSHA1: parsed.gitCommitSHA1,
            gitTreeSHA1: parsed.gitTreeSHA1,
            entries: parsed.entries
        )
    }

    public static func match(
        _ sourceManifest: AdmittedSourceManifest,
        to signedClaim: OperatorSignedRunClaim
    ) throws -> ClaimMatchedSourceManifest {
        let reference: OperatorRunClaimSourceReference
        switch sourceManifest.role {
        case .baseline:
            reference = signedClaim.fields.baseline
        case .candidate:
            reference = signedClaim.fields.candidate
        }

        guard reference.sourceManifest.authorizationID ==
            sourceManifest.authorizedFile.authorizationID.rawValue
        else {
            throw SourceManifestAdmissionError
                .sourceAuthorizationIDMismatch(role: sourceManifest.role)
        }
        guard reference.sourceManifest.payload.sha256 ==
            sourceManifest.authorizedFile.file.sha256
        else {
            throw SourceManifestAdmissionError
                .sourceManifestDigestMismatch(role: sourceManifest.role)
        }
        guard reference.sourceManifest.payload.byteCount ==
            UInt64(sourceManifest.authorizedFile.file.bytes.count)
        else {
            throw SourceManifestAdmissionError
                .sourceManifestByteCountMismatch(role: sourceManifest.role)
        }
        guard reference.gitCommitSHA1 == sourceManifest.gitCommitSHA1 else {
            throw SourceManifestAdmissionError
                .gitCommitMismatch(role: sourceManifest.role)
        }
        guard reference.gitTreeSHA1 == sourceManifest.gitTreeSHA1 else {
            throw SourceManifestAdmissionError
                .gitTreeMismatch(role: sourceManifest.role)
        }

        let binding = sourceBinding(for: sourceManifest.role)
        guard claimRole(reference.role) == sourceManifest.role else {
            throw SourceManifestAdmissionError
                .sourceRoleMismatch(role: sourceManifest.role)
        }
        guard sourceManifest.route == binding.route else {
            throw SourceManifestAdmissionError
                .routeMismatch(role: sourceManifest.role)
        }
        guard
            reference.route == sourceManifest.route,
            reference.route == binding.route
        else {
            throw SourceManifestAdmissionError
                .routeMismatch(role: sourceManifest.role)
        }
        guard
            reference.slot == sourceManifest.slot,
            reference.slot == binding.slot
        else {
            throw SourceManifestAdmissionError
                .slotMismatch(role: sourceManifest.role)
        }

        let matchID = claimMatchID(
            signedClaimID: signedClaim.claimID,
            sourceManifest: sourceManifest
        )
        return ClaimMatchedSourceManifest(
            matchID: ClaimMatchedSourceManifestID(rawValue: matchID),
            signedClaimID: signedClaim.claimID,
            sourceManifest: sourceManifest
        )
    }
}

private extension SourceManifestAdmission {
    struct ParsedManifest {
        let role: RunSourceRole
        let route: OperatorRunClaimRoute
        let slot: OperatorRunClaimSlot
        let gitCommitSHA1: String
        let gitTreeSHA1: String
        let entries: [SourceManifestEntry]
    }

    struct SourceBinding {
        let route: OperatorRunClaimRoute
        let slot: OperatorRunClaimSlot
    }

    static func sourceBinding(for role: RunSourceRole) -> SourceBinding {
        switch role {
        case .baseline:
            SourceBinding(
                route: .decompressedDeepSeekV3,
                slot: .baseline
            )
        case .candidate:
            SourceBinding(
                route: .absorbedMLADeepSeekV3Explicit,
                slot: .candidate
            )
        }
    }

    static func claimRole(
        _ role: OperatorRunClaimSourceRole
    ) -> RunSourceRole {
        switch role {
        case .baseline:
            .baseline
        case .candidate:
            .candidate
        }
    }

    static func parseManifest(_ bytes: Data) throws -> ParsedManifest {
        guard
            let text = String(data: bytes, encoding: .utf8),
            Data(text.utf8) == bytes
        else {
            throw SourceManifestAdmissionError.nonCanonicalManifest
        }

        let lines = text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        )
        guard
            lines.count >= 10,
            lines.last?.isEmpty == true,
            lines[0] == Substring(manifestDomain),
            lines[1] == "subject=\(manifestSubject)",
            let roleText = value(in: lines[2], prefix: "role="),
            let role = RunSourceRole(rawValue: roleText)
        else {
            throw SourceManifestAdmissionError.nonCanonicalManifest
        }

        let binding = sourceBinding(for: role)
        guard
            lines[3] == "route=\(binding.route.rawValue)",
            lines[4] == "slot=\(binding.slot.rawValue)",
            let gitCommitSHA1 = value(
                in: lines[5],
                prefix: "git_commit_sha1="
            ),
            let gitTreeSHA1 = value(
                in: lines[6],
                prefix: "git_tree_sha1="
            ),
            isLowercaseHex(gitCommitSHA1, count: 40),
            isLowercaseHex(gitTreeSHA1, count: 40),
            let entryCountText = value(
                in: lines[7],
                prefix: "entry_count="
            ),
            isCanonicalDecimal(entryCountText),
            let entryCount = UInt64(entryCountText),
            entryCount > 0,
            entryCount <= maximumEntryCount,
            lines.count == Int(entryCount) + 9
        else {
            throw SourceManifestAdmissionError.nonCanonicalManifest
        }

        var entries: [SourceManifestEntry] = []
        entries.reserveCapacity(Int(entryCount))
        var previousPath: String?
        var caseFoldedPaths = Set<String>()
        for line in lines[8..<(lines.count - 1)] {
            let entry = try parseEntry(line)
            if let previousPath {
                guard pathBytesPrecede(previousPath, entry.path) else {
                    throw SourceManifestAdmissionError.nonCanonicalManifest
                }
            }
            guard caseFoldedPaths.insert(entry.path.lowercased()).inserted else {
                throw SourceManifestAdmissionError.nonCanonicalManifest
            }
            entries.append(entry)
            previousPath = entry.path
        }
        guard !hasCaseFoldedPathPrefixConflict(entries.map(\.path)) else {
            throw SourceManifestAdmissionError.nonCanonicalManifest
        }

        return ParsedManifest(
            role: role,
            route: binding.route,
            slot: binding.slot,
            gitCommitSHA1: gitCommitSHA1,
            gitTreeSHA1: gitTreeSHA1,
            entries: entries
        )
    }

    static func parseEntry(
        _ line: Substring
    ) throws -> SourceManifestEntry {
        guard line.hasPrefix("entry=") else {
            throw SourceManifestAdmissionError.nonCanonicalManifest
        }
        let fields = line.dropFirst("entry=".utf8.count).split(
            separator: "\t",
            omittingEmptySubsequences: false
        )
        guard
            fields.count == 5,
            let mode = SourceManifestFileMode(
                rawValue: String(fields[0])
            ),
            isLowercaseHex(String(fields[1]), count: 40),
            isCanonicalDecimal(String(fields[2])),
            let byteCount = UInt64(fields[2]),
            isLowercaseHex(String(fields[3]), count: 64),
            isCanonicalPath(String(fields[4]))
        else {
            throw SourceManifestAdmissionError.nonCanonicalManifest
        }

        return SourceManifestEntry(
            mode: mode,
            gitBlobSHA1: String(fields[1]),
            byteCount: byteCount,
            sha256: String(fields[3]),
            path: String(fields[4])
        )
    }

    static func canonicalEntries(
        _ entries: [SourceManifestEntry]
    ) throws -> [SourceManifestEntry] {
        guard
            !entries.isEmpty,
            UInt64(entries.count) <= maximumEntryCount
        else {
            throw SourceManifestAdmissionError.nonCanonicalManifest
        }

        for entry in entries {
            guard
                isLowercaseHex(entry.gitBlobSHA1, count: 40),
                isLowercaseHex(entry.sha256, count: 64),
                isCanonicalPath(entry.path)
            else {
                throw SourceManifestAdmissionError.nonCanonicalManifest
            }
        }

        let sorted = entries.sorted {
            pathBytesPrecede($0.path, $1.path)
        }
        var caseFoldedPaths = Set<String>()
        for index in sorted.indices.dropFirst() {
            guard sorted[index - 1].path != sorted[index].path else {
                throw SourceManifestAdmissionError.nonCanonicalManifest
            }
        }
        for entry in sorted {
            guard caseFoldedPaths.insert(entry.path.lowercased()).inserted else {
                throw SourceManifestAdmissionError.nonCanonicalManifest
            }
        }
        guard !hasCaseFoldedPathPrefixConflict(sorted.map(\.path)) else {
            throw SourceManifestAdmissionError.nonCanonicalManifest
        }
        return sorted
    }

    static func hasCaseFoldedPathPrefixConflict(_ paths: [String]) -> Bool {
        let caseFoldedPaths = paths.map { $0.lowercased() }.sorted {
            pathBytesPrecede($0, $1)
        }
        for possibleAncestor in caseFoldedPaths {
            let descendantPrefix = possibleAncestor + "/"
            var lowerBound = caseFoldedPaths.startIndex
            var upperBound = caseFoldedPaths.endIndex
            while lowerBound < upperBound {
                let midpoint = lowerBound
                    + caseFoldedPaths.distance(
                        from: lowerBound,
                        to: upperBound
                    ) / 2
                if pathBytesPrecede(
                    caseFoldedPaths[midpoint],
                    descendantPrefix
                ) {
                    lowerBound = caseFoldedPaths.index(after: midpoint)
                } else {
                    upperBound = midpoint
                }
            }
            if
                lowerBound < caseFoldedPaths.endIndex,
                caseFoldedPaths[lowerBound].hasPrefix(descendantPrefix)
            {
                return true
            }
        }
        return false
    }

    static func isCanonicalPath(_ path: String) -> Bool {
        let bytes = Array(path.utf8)
        guard
            !bytes.isEmpty,
            bytes.count <= maximumPathByteCount,
            bytes.first != UInt8(ascii: "/"),
            bytes.last != UInt8(ascii: "/"),
            bytes.allSatisfy(isAllowedPathByte)
        else {
            return false
        }

        let components = path.split(
            separator: "/",
            omittingEmptySubsequences: false
        )
        return components.allSatisfy {
            !$0.isEmpty &&
                $0 != "." &&
                $0 != ".." &&
                $0.lowercased() != ".git" &&
                $0.utf8.count <= maximumPathComponentByteCount
        }
    }

    static func isAllowedPathByte(_ byte: UInt8) -> Bool {
        (0x30...0x39).contains(byte) ||
            (0x41...0x5a).contains(byte) ||
            (0x61...0x7a).contains(byte) ||
            byte == UInt8(ascii: ".") ||
            byte == UInt8(ascii: "_") ||
            byte == UInt8(ascii: "+") ||
            byte == UInt8(ascii: "-") ||
            byte == UInt8(ascii: "/")
    }

    static func pathBytesPrecede(_ lhs: String, _ rhs: String) -> Bool {
        lhs.utf8.lexicographicallyPrecedes(rhs.utf8)
    }

    static func admittedManifestID(
        authorizedFile: OperatorAuthorizedFile,
        parsed: ParsedManifest
    ) -> String {
        let lines = [
            admittedManifestIDDomain,
            "source_authorization_id=\(authorizedFile.authorizationID.rawValue)",
            "manifest_sha256=\(authorizedFile.file.sha256)",
            "manifest_bytes=\(UInt64(authorizedFile.file.bytes.count))",
            "role=\(parsed.role.rawValue)",
            "route=\(parsed.route.rawValue)",
            "slot=\(parsed.slot.rawValue)",
            "git_commit_sha1=\(parsed.gitCommitSHA1)",
            "git_tree_sha1=\(parsed.gitTreeSHA1)",
            "entry_count=\(UInt64(parsed.entries.count))",
        ]
        return sha256Hex(Data((lines.joined(separator: "\n") + "\n").utf8))
    }

    static func claimMatchID(
        signedClaimID: OperatorSignedRunClaimID,
        sourceManifest: AdmittedSourceManifest
    ) -> String {
        let lines = [
            claimMatchIDDomain,
            "run_claim_id=\(signedClaimID.rawValue)",
            "admitted_source_manifest_id=\(sourceManifest.manifestID.rawValue)",
            "role=\(sourceManifest.role.rawValue)",
            "route=\(sourceManifest.route.rawValue)",
            "slot=\(sourceManifest.slot.rawValue)",
            "git_commit_sha1=\(sourceManifest.gitCommitSHA1)",
            "git_tree_sha1=\(sourceManifest.gitTreeSHA1)",
        ]
        return sha256Hex(Data((lines.joined(separator: "\n") + "\n").utf8))
    }

    static func value(
        in line: Substring,
        prefix: String
    ) -> String? {
        guard line.hasPrefix(prefix) else {
            return nil
        }
        return String(line.dropFirst(prefix.utf8.count))
    }

    static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        value.utf8.count == count && value.utf8.allSatisfy {
            (0x30...0x39).contains($0) || (0x61...0x66).contains($0)
        }
    }

    static func isCanonicalDecimal(_ value: String) -> Bool {
        guard !value.isEmpty, value.utf8.allSatisfy({
            (0x30...0x39).contains($0)
        }) else {
            return false
        }
        return value == "0" || value.utf8.first != 0x30
    }

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
