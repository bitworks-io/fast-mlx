import CryptoKit
import Foundation

/// Qwen38 scorecard chain key policy (chain Slice 2).
///
/// Root-signed operator key-policy admission for the Qwen38 scorecard
/// domain: the document a root key signs to name the currently trusted
/// operator run-signing key for the Slice 1 claim domain
/// (`Qwen38ScorecardRunClaimVerifier.claimDomain`). This is a NEW,
/// independent policy domain: own domain, subject, scope, and admitted-ID
/// domain strings, all distinct from the frozen absorbed-MLA
/// `OperatorKeyPolicyVerifier`, whose scope/subject stay locked to the
/// absorbed-MLA program. See docs/task-inbox/
/// 2026-09-01-qwen38-proof-runner-scope-and-chain-design.md (binding
/// review verdict, items 2 and 4).
///
/// Unlike Slice 1's structural-only claim envelope, `admit` here performs
/// the REAL Ed25519 root-signature verification over the exact policy
/// bytes, mirroring the frozen admission checks: externally pinned current
/// policy digest, generation-rollback floor, inclusive validity window
/// against a caller-supplied verification time, in-document revocation
/// list, and root/active key identity separation. Claim-signature
/// verification against the admitted operator key is chain Slice 3; runner
/// orchestration and scorecard integration are Slices 3/4.
///
/// Pure canonical-encoding helpers below are deliberately DUPLICATED from
/// the frozen `OperatorKeyPolicy.swift` (which keeps them file-private)
/// rather than extracted or shared, per binding item 4: the frozen file is
/// not modified, and each policy domain stays a self-contained auditable
/// unit. The Slice 1 claim file keeps its own copies for the same reason;
/// the golden-bytes oracles in each slice's tests pin the formats against
/// silent drift.
public enum Qwen38ScorecardKeyPolicyScope: String, Equatable, Sendable {
    case scorecardRunClaim = "qwen38-scorecard-run-claim"
}

public struct Qwen38ScorecardKeyPolicyFields: Equatable, Sendable {
    public let rootKeyID: String
    public let policyGeneration: UInt64
    public let validFromUnixSeconds: UInt64
    public let validUntilUnixSeconds: UInt64
    public let activeOperatorKeyID: String
    public let activeOperatorPublicKeyBase64: String
    public let activeOperatorScope: Qwen38ScorecardKeyPolicyScope
    public let allowedClaimSubject: Qwen38ScorecardRunClaimSubject
    public let revokedOperatorKeyIDs: [String]

    public init(
        rootKeyID: String,
        policyGeneration: UInt64,
        validFromUnixSeconds: UInt64,
        validUntilUnixSeconds: UInt64,
        activeOperatorKeyID: String,
        activeOperatorPublicKeyBase64: String,
        activeOperatorScope: Qwen38ScorecardKeyPolicyScope,
        allowedClaimSubject: Qwen38ScorecardRunClaimSubject,
        revokedOperatorKeyIDs: [String]
    ) {
        self.rootKeyID = rootKeyID
        self.policyGeneration = policyGeneration
        self.validFromUnixSeconds = validFromUnixSeconds
        self.validUntilUnixSeconds = validUntilUnixSeconds
        self.activeOperatorKeyID = activeOperatorKeyID
        self.activeOperatorPublicKeyBase64 =
            activeOperatorPublicKeyBase64
        self.activeOperatorScope = activeOperatorScope
        self.allowedClaimSubject = allowedClaimSubject
        self.revokedOperatorKeyIDs = revokedOperatorKeyIDs
    }
}

/// Immutable bootstrap context provisioned outside candidate and baseline
/// source trees. Canonical validation here cannot establish that
/// provenance; `host_admission_id`-style operator assertions stay operator
/// assertions.
public struct Qwen38ScorecardKeyPolicyTrustAnchor: Equatable, Sendable {
    public let rootPublicKeyBase64: String
    public let rootKeyID: String
    public let expectedCurrentPolicySHA256: String
    public let minimumPolicyGeneration: UInt64
    public let verificationUnixSeconds: UInt64

    public init(
        rootPublicKeyBase64: String,
        rootKeyID: String,
        expectedCurrentPolicySHA256: String,
        minimumPolicyGeneration: UInt64,
        verificationUnixSeconds: UInt64
    ) {
        self.rootPublicKeyBase64 = rootPublicKeyBase64
        self.rootKeyID = rootKeyID
        self.expectedCurrentPolicySHA256 =
            expectedCurrentPolicySHA256
        self.minimumPolicyGeneration = minimumPolicyGeneration
        self.verificationUnixSeconds = verificationUnixSeconds
    }
}

public enum Qwen38ScorecardKeyPolicyTrustAnchorField:
    String,
    Equatable,
    Sendable
{
    case rootKeyID = "root-key-id"
    case expectedCurrentPolicySHA256 =
        "expected-current-policy-sha256"
}

public enum Qwen38ScorecardKeyPolicyError: Error, Equatable, Sendable {
    case nonCanonicalPolicy
    case invalidTrustAnchor(Qwen38ScorecardKeyPolicyTrustAnchorField)
    case invalidRootPublicKeyEncoding
    case rootPublicKeyIDMismatch
    case policyDigestMismatch
    case rootKeyIDMismatch
    case policyGenerationRollback(minimum: UInt64, actual: UInt64)
    case policyNotYetValid
    case policyExpired
    case invalidActiveOperatorPublicKeyEncoding
    case activeOperatorKeyIDMismatch
    case rootAndActiveKeysMustDiffer
    case rootKeyRevocationUnsupported
    case activeOperatorKeyRevoked
    case invalidRootSignatureEncoding
    case rootSignatureRejected
}

extension Qwen38ScorecardKeyPolicyError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .nonCanonicalPolicy:
            "qwen38 scorecard key policy is not the canonical fixed-order UTF-8 form"
        case .invalidTrustAnchor(let field):
            "qwen38 scorecard key policy trust anchor \(field.rawValue) is not canonical"
        case .invalidRootPublicKeyEncoding:
            "qwen38 policy root public key is not canonical base64 for 32 Ed25519 bytes"
        case .rootPublicKeyIDMismatch:
            "qwen38 policy root public key does not match its externally pinned ID"
        case .policyDigestMismatch:
            "qwen38 policy bytes do not match the externally pinned current policy"
        case .rootKeyIDMismatch:
            "qwen38 policy root key ID does not match the external trust anchor"
        case .policyGenerationRollback(let minimum, let actual):
            "qwen38 policy generation \(actual) is below external floor \(minimum)"
        case .policyNotYetValid:
            "qwen38 policy is not valid at the external verification time"
        case .policyExpired:
            "qwen38 policy has expired at the external verification time"
        case .invalidActiveOperatorPublicKeyEncoding:
            "qwen38 active operator public key is not canonical base64 for 32 Ed25519 bytes"
        case .activeOperatorKeyIDMismatch:
            "qwen38 active operator public key does not match its policy key ID"
        case .rootAndActiveKeysMustDiffer:
            "qwen38 offline policy root and active run-signing key must differ"
        case .rootKeyRevocationUnsupported:
            "qwen38 root-key retirement must come from external trust configuration"
        case .activeOperatorKeyRevoked:
            "qwen38 active operator key is also listed as revoked"
        case .invalidRootSignatureEncoding:
            "qwen38 policy root signature is not canonical base64 for 64 Ed25519 bytes"
        case .rootSignatureRejected:
            "qwen38 policy root signature is not valid over the exact policy bytes"
        }
    }
}

public struct AdmittedQwen38ScorecardKeyPolicyID:
    Equatable,
    Hashable,
    Sendable
{
    public let rawValue: String

    fileprivate init(rawValue: String) {
        self.rawValue = rawValue
    }
}

/// Root-authenticated signer policy only. This does not authorize a run,
/// and it does not verify any claim signature — that is chain Slice 3.
public struct AdmittedQwen38ScorecardKeyPolicy: Equatable, Sendable {
    public let file: AdmittedFile
    public let admissionID: AdmittedQwen38ScorecardKeyPolicyID
    public let policySHA256: String
    public let policyGeneration: UInt64
    public let validFromUnixSeconds: UInt64
    public let validUntilUnixSeconds: UInt64
    public let rootKeyID: String
    public let activeOperatorKeyID: String
    public let activeOperatorPublicKeyBase64: String
    public let activeOperatorScope: Qwen38ScorecardKeyPolicyScope
    public let allowedClaimSubject: Qwen38ScorecardRunClaimSubject
    public let revokedOperatorKeyIDs: [String]
    public let rootSignatureSHA256: String

    fileprivate init(
        file: AdmittedFile,
        admissionID: AdmittedQwen38ScorecardKeyPolicyID,
        fields: Qwen38ScorecardKeyPolicyFields,
        rootSignatureSHA256: String
    ) {
        self.file = file
        self.admissionID = admissionID
        self.policySHA256 = file.sha256
        self.policyGeneration = fields.policyGeneration
        self.validFromUnixSeconds = fields.validFromUnixSeconds
        self.validUntilUnixSeconds = fields.validUntilUnixSeconds
        self.rootKeyID = fields.rootKeyID
        self.activeOperatorKeyID = fields.activeOperatorKeyID
        self.activeOperatorPublicKeyBase64 =
            fields.activeOperatorPublicKeyBase64
        self.activeOperatorScope = fields.activeOperatorScope
        self.allowedClaimSubject = fields.allowedClaimSubject
        self.revokedOperatorKeyIDs = fields.revokedOperatorKeyIDs
        self.rootSignatureSHA256 = rootSignatureSHA256
    }
}

public enum Qwen38ScorecardKeyPolicyVerifier {
    public static let policyDomain =
        "fastmlx-qwen38-scorecard-operator-key-policy-v1"
    public static let policySubject =
        "qwen38-scorecard-operator-run-signing-key-policy"
    public static let admittedPolicyIDDomain =
        "fastmlx-qwen38-admitted-scorecard-operator-key-policy-id-v1"
    public static let maximumRevokedKeyCount = 256

    public static func policyBytes(
        fields: Qwen38ScorecardKeyPolicyFields
    ) throws -> Data {
        try validate(fields)
        let revoked = fields.revokedOperatorKeyIDs.isEmpty
            ? "none"
            : fields.revokedOperatorKeyIDs.joined(separator: ",")
        let lines = [
            policyDomain,
            "subject=\(policySubject)",
            "root_key_id=\(fields.rootKeyID)",
            "policy_generation=\(fields.policyGeneration)",
            "valid_from_unix_seconds=\(fields.validFromUnixSeconds)",
            "valid_until_unix_seconds=\(fields.validUntilUnixSeconds)",
            "active_operator_key_id=\(fields.activeOperatorKeyID)",
            "active_operator_public_key_base64=\(fields.activeOperatorPublicKeyBase64)",
            "active_operator_scope=\(fields.activeOperatorScope.rawValue)",
            "allowed_claim_subject=\(fields.allowedClaimSubject.rawValue)",
            "revoked_operator_key_ids=\(revoked)",
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    public static func admit(
        policyFile: AdmittedFile,
        rootSignatureBase64: String,
        trustAnchor: Qwen38ScorecardKeyPolicyTrustAnchor
    ) throws -> AdmittedQwen38ScorecardKeyPolicy {
        let rootPublicKeyBytes = try validate(trustAnchor)
        guard policyFile.sha256 ==
            trustAnchor.expectedCurrentPolicySHA256
        else {
            throw Qwen38ScorecardKeyPolicyError.policyDigestMismatch
        }

        let fields = try parsePolicy(policyFile.bytes)
        guard fields.rootKeyID == trustAnchor.rootKeyID else {
            throw Qwen38ScorecardKeyPolicyError.rootKeyIDMismatch
        }
        guard fields.policyGeneration >=
            trustAnchor.minimumPolicyGeneration
        else {
            throw Qwen38ScorecardKeyPolicyError.policyGenerationRollback(
                minimum: trustAnchor.minimumPolicyGeneration,
                actual: fields.policyGeneration
            )
        }
        guard trustAnchor.verificationUnixSeconds >=
            fields.validFromUnixSeconds
        else {
            throw Qwen38ScorecardKeyPolicyError.policyNotYetValid
        }
        guard trustAnchor.verificationUnixSeconds <=
            fields.validUntilUnixSeconds
        else {
            throw Qwen38ScorecardKeyPolicyError.policyExpired
        }

        let rootPublicKey: Curve25519.Signing.PublicKey
        do {
            rootPublicKey = try Curve25519.Signing.PublicKey(
                rawRepresentation: rootPublicKeyBytes
            )
        } catch {
            throw Qwen38ScorecardKeyPolicyError
                .invalidRootPublicKeyEncoding
        }
        let signature = try decodeCanonicalBase64(
            rootSignatureBase64,
            expectedByteCount: 64,
            error: .invalidRootSignatureEncoding
        )
        guard rootPublicKey.isValidSignature(
            signature,
            for: policyFile.bytes
        ) else {
            throw Qwen38ScorecardKeyPolicyError.rootSignatureRejected
        }

        let signatureSHA256 = sha256Hex(signature)
        return AdmittedQwen38ScorecardKeyPolicy(
            file: policyFile,
            admissionID: AdmittedQwen38ScorecardKeyPolicyID(
                rawValue: admittedPolicyID(
                    rootKeyID: fields.rootKeyID,
                    policySHA256: policyFile.sha256,
                    signatureSHA256: signatureSHA256
                )
            ),
            fields: fields,
            rootSignatureSHA256: signatureSHA256
        )
    }
}

private extension Qwen38ScorecardKeyPolicyVerifier {
    static func parsePolicy(_ bytes: Data) throws
        -> Qwen38ScorecardKeyPolicyFields
    {
        guard
            let text = String(data: bytes, encoding: .utf8),
            Data(text.utf8) == bytes
        else {
            throw Qwen38ScorecardKeyPolicyError.nonCanonicalPolicy
        }

        let lines = text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        )
        guard
            lines.count == 12,
            lines[0] == Substring(policyDomain),
            lines[1] == "subject=\(policySubject)",
            lines[11].isEmpty,
            let rootKeyID = value(
                in: lines[2],
                prefix: "root_key_id="
            ),
            let generationText = value(
                in: lines[3],
                prefix: "policy_generation="
            ),
            let validFromText = value(
                in: lines[4],
                prefix: "valid_from_unix_seconds="
            ),
            let validUntilText = value(
                in: lines[5],
                prefix: "valid_until_unix_seconds="
            ),
            let activeKeyID = value(
                in: lines[6],
                prefix: "active_operator_key_id="
            ),
            let activePublicKey = value(
                in: lines[7],
                prefix: "active_operator_public_key_base64="
            ),
            lines[8] ==
                "active_operator_scope=\(Qwen38ScorecardKeyPolicyScope.scorecardRunClaim.rawValue)",
            lines[9] ==
                "allowed_claim_subject=\(Qwen38ScorecardRunClaimSubject.mtpScorecardResultPair.rawValue)",
            let revokedText = value(
                in: lines[10],
                prefix: "revoked_operator_key_ids="
            ),
            isCanonicalDecimal(generationText),
            isCanonicalDecimal(validFromText),
            isCanonicalDecimal(validUntilText),
            let generation = UInt64(generationText),
            let validFrom = UInt64(validFromText),
            let validUntil = UInt64(validUntilText)
        else {
            throw Qwen38ScorecardKeyPolicyError.nonCanonicalPolicy
        }

        let fields = Qwen38ScorecardKeyPolicyFields(
            rootKeyID: rootKeyID,
            policyGeneration: generation,
            validFromUnixSeconds: validFrom,
            validUntilUnixSeconds: validUntil,
            activeOperatorKeyID: activeKeyID,
            activeOperatorPublicKeyBase64: activePublicKey,
            activeOperatorScope: .scorecardRunClaim,
            allowedClaimSubject: .mtpScorecardResultPair,
            revokedOperatorKeyIDs: try parseRevokedKeyIDs(revokedText)
        )
        try validate(fields)
        return fields
    }

    static func parseRevokedKeyIDs(_ value: String) throws -> [String] {
        if value == "none" {
            return []
        }
        let ids = value.split(
            separator: ",",
            omittingEmptySubsequences: false
        ).map(String.init)
        guard
            !ids.isEmpty,
            ids.count <= maximumRevokedKeyCount,
            ids.allSatisfy(isCanonicalSHA256),
            zip(ids, ids.dropFirst()).allSatisfy(<)
        else {
            throw Qwen38ScorecardKeyPolicyError.nonCanonicalPolicy
        }
        return ids
    }

    static func validate(
        _ fields: Qwen38ScorecardKeyPolicyFields
    ) throws {
        guard
            isCanonicalSHA256(fields.rootKeyID),
            isCanonicalSHA256(fields.activeOperatorKeyID),
            fields.policyGeneration > 0,
            fields.validFromUnixSeconds <= fields.validUntilUnixSeconds,
            fields.activeOperatorScope == .scorecardRunClaim,
            fields.allowedClaimSubject == .mtpScorecardResultPair,
            fields.revokedOperatorKeyIDs.count <=
                maximumRevokedKeyCount,
            fields.revokedOperatorKeyIDs.allSatisfy(
                isCanonicalSHA256
            ),
            zip(
                fields.revokedOperatorKeyIDs,
                fields.revokedOperatorKeyIDs.dropFirst()
            ).allSatisfy(<)
        else {
            throw Qwen38ScorecardKeyPolicyError.nonCanonicalPolicy
        }

        let activePublicKey = try decodeCanonicalBase64(
            fields.activeOperatorPublicKeyBase64,
            expectedByteCount: 32,
            error: .invalidActiveOperatorPublicKeyEncoding
        )
        do {
            _ = try Curve25519.Signing.PublicKey(
                rawRepresentation: activePublicKey
            )
        } catch {
            throw Qwen38ScorecardKeyPolicyError
                .invalidActiveOperatorPublicKeyEncoding
        }
        guard sha256Hex(activePublicKey) ==
            fields.activeOperatorKeyID
        else {
            throw Qwen38ScorecardKeyPolicyError
                .activeOperatorKeyIDMismatch
        }
        guard fields.rootKeyID != fields.activeOperatorKeyID else {
            throw Qwen38ScorecardKeyPolicyError
                .rootAndActiveKeysMustDiffer
        }
        guard !fields.revokedOperatorKeyIDs.contains(
            fields.rootKeyID
        ) else {
            throw Qwen38ScorecardKeyPolicyError
                .rootKeyRevocationUnsupported
        }
        guard !fields.revokedOperatorKeyIDs.contains(
            fields.activeOperatorKeyID
        ) else {
            throw Qwen38ScorecardKeyPolicyError.activeOperatorKeyRevoked
        }
    }

    static func validate(
        _ trustAnchor: Qwen38ScorecardKeyPolicyTrustAnchor
    ) throws -> Data {
        guard isCanonicalSHA256(trustAnchor.rootKeyID) else {
            throw Qwen38ScorecardKeyPolicyError.invalidTrustAnchor(
                .rootKeyID
            )
        }
        guard isCanonicalSHA256(
            trustAnchor.expectedCurrentPolicySHA256
        ) else {
            throw Qwen38ScorecardKeyPolicyError.invalidTrustAnchor(
                .expectedCurrentPolicySHA256
            )
        }
        let rootPublicKey = try decodeCanonicalBase64(
            trustAnchor.rootPublicKeyBase64,
            expectedByteCount: 32,
            error: .invalidRootPublicKeyEncoding
        )
        guard sha256Hex(rootPublicKey) == trustAnchor.rootKeyID else {
            throw Qwen38ScorecardKeyPolicyError.rootPublicKeyIDMismatch
        }
        return rootPublicKey
    }

    // Pure canonical-encoding helpers duplicated from the frozen
    // `OperatorKeyPolicy.swift` (file-private there), per binding item 4.
    static func value(
        in line: Substring,
        prefix: String
    ) -> String? {
        guard line.hasPrefix(prefix) else {
            return nil
        }
        return String(line.dropFirst(prefix.utf8.count))
    }

    static func isCanonicalSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
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

    static func decodeCanonicalBase64(
        _ value: String,
        expectedByteCount: Int,
        error: Qwen38ScorecardKeyPolicyError
    ) throws -> Data {
        guard
            let decoded = Data(base64Encoded: value),
            decoded.count == expectedByteCount,
            decoded.base64EncodedString() == value
        else {
            throw error
        }
        return decoded
    }

    static func admittedPolicyID(
        rootKeyID: String,
        policySHA256: String,
        signatureSHA256: String
    ) -> String {
        let lines = [
            admittedPolicyIDDomain,
            "root_key_id=\(rootKeyID)",
            "policy_sha256=\(policySHA256)",
            "signature_sha256=\(signatureSHA256)",
        ]
        return sha256Hex(
            Data((lines.joined(separator: "\n") + "\n").utf8)
        )
    }

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
