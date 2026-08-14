import CryptoKit
import Foundation

public enum OperatorKeyPolicyScope: String, Equatable, Sendable {
    case runClaim = "run-claim"
}

public struct OperatorKeyPolicyFields: Equatable, Sendable {
    public let rootKeyID: String
    public let policyGeneration: UInt64
    public let validFromUnixSeconds: UInt64
    public let validUntilUnixSeconds: UInt64
    public let activeOperatorKeyID: String
    public let activeOperatorPublicKeyBase64: String
    public let activeOperatorScope: OperatorKeyPolicyScope
    public let allowedClaimSubject: OperatorRunClaimSubject
    public let revokedOperatorKeyIDs: [String]

    public init(
        rootKeyID: String,
        policyGeneration: UInt64,
        validFromUnixSeconds: UInt64,
        validUntilUnixSeconds: UInt64,
        activeOperatorKeyID: String,
        activeOperatorPublicKeyBase64: String,
        activeOperatorScope: OperatorKeyPolicyScope,
        allowedClaimSubject: OperatorRunClaimSubject,
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
/// source trees. Canonical validation here cannot establish that provenance.
public struct OperatorKeyPolicyTrustAnchor: Equatable, Sendable {
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

public enum OperatorKeyPolicyTrustAnchorField:
    String,
    Equatable,
    Sendable
{
    case rootKeyID = "root-key-id"
    case expectedCurrentPolicySHA256 =
        "expected-current-policy-sha256"
}

public enum OperatorKeyPolicyError: Error, Equatable, Sendable {
    case nonCanonicalPolicy
    case invalidTrustAnchor(OperatorKeyPolicyTrustAnchorField)
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

extension OperatorKeyPolicyError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .nonCanonicalPolicy:
            "operator key policy is not the canonical fixed-order UTF-8 form"
        case .invalidTrustAnchor(let field):
            "operator key policy trust anchor \(field.rawValue) is not canonical"
        case .invalidRootPublicKeyEncoding:
            "policy root public key is not canonical base64 for 32 Ed25519 bytes"
        case .rootPublicKeyIDMismatch:
            "policy root public key does not match its externally pinned ID"
        case .policyDigestMismatch:
            "policy bytes do not match the externally pinned current policy"
        case .rootKeyIDMismatch:
            "policy root key ID does not match the external trust anchor"
        case .policyGenerationRollback(let minimum, let actual):
            "policy generation \(actual) is below external floor \(minimum)"
        case .policyNotYetValid:
            "policy is not valid at the external verification time"
        case .policyExpired:
            "policy has expired at the external verification time"
        case .invalidActiveOperatorPublicKeyEncoding:
            "active operator public key is not canonical base64 for 32 Ed25519 bytes"
        case .activeOperatorKeyIDMismatch:
            "active operator public key does not match its policy key ID"
        case .rootAndActiveKeysMustDiffer:
            "offline policy root and active run-signing key must differ"
        case .rootKeyRevocationUnsupported:
            "root-key retirement must come from external trust configuration"
        case .activeOperatorKeyRevoked:
            "active operator key is also listed as revoked"
        case .invalidRootSignatureEncoding:
            "policy root signature is not canonical base64 for 64 Ed25519 bytes"
        case .rootSignatureRejected:
            "policy root signature is not valid over the exact policy bytes"
        }
    }
}

public struct AdmittedOperatorKeyPolicyID:
    Equatable,
    Hashable,
    Sendable
{
    public let rawValue: String

    fileprivate init(rawValue: String) {
        self.rawValue = rawValue
    }
}

/// Root-authenticated signer policy only. This does not authorize a run.
public struct AdmittedOperatorKeyPolicy: Equatable, Sendable {
    public let file: AdmittedFile
    public let admissionID: AdmittedOperatorKeyPolicyID
    public let policySHA256: String
    public let policyGeneration: UInt64
    public let validFromUnixSeconds: UInt64
    public let validUntilUnixSeconds: UInt64
    public let rootKeyID: String
    public let activeOperatorKeyID: String
    public let activeOperatorPublicKeyBase64: String
    public let activeOperatorScope: OperatorKeyPolicyScope
    public let allowedClaimSubject: OperatorRunClaimSubject
    public let revokedOperatorKeyIDs: [String]
    public let rootSignatureSHA256: String

    fileprivate init(
        file: AdmittedFile,
        admissionID: AdmittedOperatorKeyPolicyID,
        fields: OperatorKeyPolicyFields,
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

public enum OperatorKeyPolicyVerifier {
    public static let policyDomain =
        "fast-mlx-proof-control-operator-key-policy-v1"
    public static let policySubject =
        "operator-run-signing-key-policy"
    public static let admittedPolicyIDDomain =
        "fast-mlx-proof-control-admitted-operator-key-policy-id-v1"
    public static let maximumRevokedKeyCount = 256

    public static func policyBytes(
        fields: OperatorKeyPolicyFields
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
        trustAnchor: OperatorKeyPolicyTrustAnchor
    ) throws -> AdmittedOperatorKeyPolicy {
        let rootPublicKeyBytes = try validate(trustAnchor)
        guard policyFile.sha256 ==
            trustAnchor.expectedCurrentPolicySHA256
        else {
            throw OperatorKeyPolicyError.policyDigestMismatch
        }

        let fields = try parsePolicy(policyFile.bytes)
        guard fields.rootKeyID == trustAnchor.rootKeyID else {
            throw OperatorKeyPolicyError.rootKeyIDMismatch
        }
        guard fields.policyGeneration >=
            trustAnchor.minimumPolicyGeneration
        else {
            throw OperatorKeyPolicyError.policyGenerationRollback(
                minimum: trustAnchor.minimumPolicyGeneration,
                actual: fields.policyGeneration
            )
        }
        guard trustAnchor.verificationUnixSeconds >=
            fields.validFromUnixSeconds
        else {
            throw OperatorKeyPolicyError.policyNotYetValid
        }
        guard trustAnchor.verificationUnixSeconds <=
            fields.validUntilUnixSeconds
        else {
            throw OperatorKeyPolicyError.policyExpired
        }

        let rootPublicKey: Curve25519.Signing.PublicKey
        do {
            rootPublicKey = try Curve25519.Signing.PublicKey(
                rawRepresentation: rootPublicKeyBytes
            )
        } catch {
            throw OperatorKeyPolicyError.invalidRootPublicKeyEncoding
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
            throw OperatorKeyPolicyError.rootSignatureRejected
        }

        let signatureSHA256 = sha256Hex(signature)
        return AdmittedOperatorKeyPolicy(
            file: policyFile,
            admissionID: AdmittedOperatorKeyPolicyID(
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

private extension OperatorKeyPolicyVerifier {
    static func parsePolicy(_ bytes: Data) throws
        -> OperatorKeyPolicyFields
    {
        guard
            let text = String(data: bytes, encoding: .utf8),
            Data(text.utf8) == bytes
        else {
            throw OperatorKeyPolicyError.nonCanonicalPolicy
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
                "active_operator_scope=\(OperatorKeyPolicyScope.runClaim.rawValue)",
            lines[9] ==
                "allowed_claim_subject=\(OperatorRunClaimSubject.absorbedMLALoadedResultPair.rawValue)",
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
            throw OperatorKeyPolicyError.nonCanonicalPolicy
        }

        let fields = OperatorKeyPolicyFields(
            rootKeyID: rootKeyID,
            policyGeneration: generation,
            validFromUnixSeconds: validFrom,
            validUntilUnixSeconds: validUntil,
            activeOperatorKeyID: activeKeyID,
            activeOperatorPublicKeyBase64: activePublicKey,
            activeOperatorScope: .runClaim,
            allowedClaimSubject: .absorbedMLALoadedResultPair,
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
            throw OperatorKeyPolicyError.nonCanonicalPolicy
        }
        return ids
    }

    static func validate(_ fields: OperatorKeyPolicyFields) throws {
        guard
            isCanonicalSHA256(fields.rootKeyID),
            isCanonicalSHA256(fields.activeOperatorKeyID),
            fields.policyGeneration > 0,
            fields.validFromUnixSeconds <= fields.validUntilUnixSeconds,
            fields.activeOperatorScope == .runClaim,
            fields.allowedClaimSubject ==
                .absorbedMLALoadedResultPair,
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
            throw OperatorKeyPolicyError.nonCanonicalPolicy
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
            throw OperatorKeyPolicyError
                .invalidActiveOperatorPublicKeyEncoding
        }
        guard sha256Hex(activePublicKey) ==
            fields.activeOperatorKeyID
        else {
            throw OperatorKeyPolicyError.activeOperatorKeyIDMismatch
        }
        guard fields.rootKeyID != fields.activeOperatorKeyID else {
            throw OperatorKeyPolicyError.rootAndActiveKeysMustDiffer
        }
        guard !fields.revokedOperatorKeyIDs.contains(
            fields.rootKeyID
        ) else {
            throw OperatorKeyPolicyError.rootKeyRevocationUnsupported
        }
        guard !fields.revokedOperatorKeyIDs.contains(
            fields.activeOperatorKeyID
        ) else {
            throw OperatorKeyPolicyError.activeOperatorKeyRevoked
        }
    }

    static func validate(
        _ trustAnchor: OperatorKeyPolicyTrustAnchor
    ) throws -> Data {
        guard isCanonicalSHA256(trustAnchor.rootKeyID) else {
            throw OperatorKeyPolicyError.invalidTrustAnchor(.rootKeyID)
        }
        guard isCanonicalSHA256(
            trustAnchor.expectedCurrentPolicySHA256
        ) else {
            throw OperatorKeyPolicyError.invalidTrustAnchor(
                .expectedCurrentPolicySHA256
            )
        }
        let rootPublicKey = try decodeCanonicalBase64(
            trustAnchor.rootPublicKeyBase64,
            expectedByteCount: 32,
            error: .invalidRootPublicKeyEncoding
        )
        guard sha256Hex(rootPublicKey) == trustAnchor.rootKeyID else {
            throw OperatorKeyPolicyError.rootPublicKeyIDMismatch
        }
        return rootPublicKey
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
        error: OperatorKeyPolicyError
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
