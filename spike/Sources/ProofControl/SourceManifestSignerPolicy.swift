import CryptoKit
import Foundation

public enum SourceManifestSignerPolicyScope:
    String,
    Equatable,
    Sendable
{
    case sourceManifest = "source-manifest"
}

public struct SourceManifestSignerPolicyFields: Equatable, Sendable {
    public let rootKeyID: String
    public let policyGeneration: UInt64
    public let validFromUnixSeconds: UInt64
    public let validUntilUnixSeconds: UInt64
    public let activeSourceOperatorKeyID: String
    public let activeSourceOperatorPublicKeyBase64: String
    public let activeSourceOperatorScope: SourceManifestSignerPolicyScope
    public let allowedAuthorizationPurpose: OperatorAuthorizationPurpose
    public let allowedRunClaimSubject: OperatorRunClaimSubject
    public let allowedSourceRoles: [RunSourceRole]
    public let revokedSourceOperatorKeyIDs: [String]

    public init(
        rootKeyID: String,
        policyGeneration: UInt64,
        validFromUnixSeconds: UInt64,
        validUntilUnixSeconds: UInt64,
        activeSourceOperatorKeyID: String,
        activeSourceOperatorPublicKeyBase64: String,
        activeSourceOperatorScope: SourceManifestSignerPolicyScope,
        allowedAuthorizationPurpose: OperatorAuthorizationPurpose,
        allowedRunClaimSubject: OperatorRunClaimSubject,
        allowedSourceRoles: [RunSourceRole],
        revokedSourceOperatorKeyIDs: [String]
    ) {
        self.rootKeyID = rootKeyID
        self.policyGeneration = policyGeneration
        self.validFromUnixSeconds = validFromUnixSeconds
        self.validUntilUnixSeconds = validUntilUnixSeconds
        self.activeSourceOperatorKeyID = activeSourceOperatorKeyID
        self.activeSourceOperatorPublicKeyBase64 =
            activeSourceOperatorPublicKeyBase64
        self.activeSourceOperatorScope = activeSourceOperatorScope
        self.allowedAuthorizationPurpose = allowedAuthorizationPurpose
        self.allowedRunClaimSubject = allowedRunClaimSubject
        self.allowedSourceRoles = allowedSourceRoles
        self.revokedSourceOperatorKeyIDs = revokedSourceOperatorKeyIDs
    }
}

/// Immutable source-policy trust context supplied outside candidate and
/// baseline source trees. This value is not provenance for that context.
public struct SourceManifestSignerPolicyTrustAnchor:
    Equatable,
    Sendable
{
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

public enum SourceManifestSignerPolicyTrustAnchorField:
    String,
    Equatable,
    Sendable
{
    case rootKeyID = "root-key-id"
    case expectedCurrentPolicySHA256 =
        "expected-current-policy-sha256"
}

public enum SourceManifestSignerPolicyError:
    Error,
    Equatable,
    Sendable
{
    case nonCanonicalPolicy
    case invalidTrustAnchor(
        SourceManifestSignerPolicyTrustAnchorField
    )
    case invalidRootPublicKeyEncoding
    case rootPublicKeyIDMismatch
    case policyDigestMismatch
    case rootKeyIDMismatch
    case policyGenerationRollback(minimum: UInt64, actual: UInt64)
    case policyNotYetValid
    case policyExpired
    case invalidActiveSourceOperatorPublicKeyEncoding
    case activeSourceOperatorKeyIDMismatch
    case rootAndActiveSourceKeysMustDiffer
    case sourceAndRunActiveKeysMustDiffer
    case sourceRootAndRunActiveKeyMustDiffer
    case sourceActiveAndRunPolicyRootMustDiffer
    case rootKeyRevocationUnsupported
    case activeSourceOperatorKeyRevoked
    case invalidRootSignatureEncoding
    case rootSignatureRejected
    case signedClaimSourcePolicyMismatch
    case signedClaimRunKeyPolicyMismatch
    case unexpectedAuthorizationPurpose(
        expected: OperatorAuthorizationPurpose,
        actual: OperatorAuthorizationPurpose
    )
    case unsupportedInputRole(RunAuthorizedInputRole)
    case sourceAuthorizationKeyMismatch(role: RunSourceRole)
    case sourceRoleMismatch(expected: RunSourceRole, actual: RunSourceRole)
    case claimMatchedSourceManifestClaimMismatch(role: RunSourceRole)
    case missingRequiredSourceRole(RunSourceRole)
}

extension SourceManifestSignerPolicyError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .nonCanonicalPolicy:
            "source manifest signer policy is not the canonical fixed-order UTF-8 form"
        case .invalidTrustAnchor(let field):
            "source manifest signer policy trust anchor \(field.rawValue) is not canonical"
        case .invalidRootPublicKeyEncoding:
            "source policy root public key is not canonical base64 for 32 Ed25519 bytes"
        case .rootPublicKeyIDMismatch:
            "source policy root public key does not match its externally pinned ID"
        case .policyDigestMismatch:
            "source policy bytes do not match the externally pinned current policy"
        case .rootKeyIDMismatch:
            "source policy root key ID does not match the external trust anchor"
        case .policyGenerationRollback(let minimum, let actual):
            "source policy generation \(actual) is below external floor \(minimum)"
        case .policyNotYetValid:
            "source policy is not valid at the external verification time"
        case .policyExpired:
            "source policy has expired at the external verification time"
        case .invalidActiveSourceOperatorPublicKeyEncoding:
            "active source operator public key is not canonical base64 for 32 Ed25519 bytes"
        case .activeSourceOperatorKeyIDMismatch:
            "active source operator public key does not match its policy key ID"
        case .rootAndActiveSourceKeysMustDiffer:
            "offline source policy root and active source-manifest key must differ"
        case .sourceAndRunActiveKeysMustDiffer:
            "active source-manifest key and active run-claim key must differ"
        case .sourceRootAndRunActiveKeyMustDiffer:
            "source policy root and active run-claim key must differ"
        case .sourceActiveAndRunPolicyRootMustDiffer:
            "active source-manifest key and run policy root must differ"
        case .rootKeyRevocationUnsupported:
            "source root-key retirement must come from external trust configuration"
        case .activeSourceOperatorKeyRevoked:
            "active source operator key is also listed as revoked"
        case .invalidRootSignatureEncoding:
            "source policy root signature is not canonical base64 for 64 Ed25519 bytes"
        case .rootSignatureRejected:
            "source policy root signature is not valid over the exact policy bytes"
        case .signedClaimSourcePolicyMismatch:
            "signed run claim does not reference the admitted source policy"
        case .signedClaimRunKeyPolicyMismatch:
            "signed run claim does not reference the admitted run-key policy"
        case .unexpectedAuthorizationPurpose(let expected, let actual):
            "source authorization purpose \(actual.rawValue) does not match \(expected.rawValue)"
        case .unsupportedInputRole(let role):
            "source manifest signer policy does not authorize \(role.rawValue)"
        case .sourceAuthorizationKeyMismatch(let role):
            "\(role.rawValue) source manifest was not signed by the active source key"
        case .sourceRoleMismatch(let expected, let actual):
            "source role \(actual.rawValue) does not match expected \(expected.rawValue)"
        case .claimMatchedSourceManifestClaimMismatch(let role):
            "\(role.rawValue) source manifest was matched to a different run claim"
        case .missingRequiredSourceRole(let role):
            "missing required \(role.rawValue) source manifest policy match"
        }
    }
}

public struct AdmittedSourceManifestSignerPolicyID:
    Equatable,
    Hashable,
    Sendable
{
    public let rawValue: String

    fileprivate init(rawValue: String) {
        self.rawValue = rawValue
    }
}

/// Root-authenticated source-manifest signer policy only.
///
/// This does not authorize source import, build, spawn, model loading, or
/// output publication.
public struct AdmittedSourceManifestSignerPolicy:
    Equatable,
    Sendable
{
    public let file: AdmittedFile
    public let policyID: AdmittedSourceManifestSignerPolicyID
    public let policySHA256: String
    public let policyGeneration: UInt64
    public let validFromUnixSeconds: UInt64
    public let validUntilUnixSeconds: UInt64
    public let rootKeyID: String
    public let activeSourceOperatorKeyID: String
    public let activeSourceOperatorPublicKeyBase64: String
    public let activeSourceOperatorScope: SourceManifestSignerPolicyScope
    public let allowedAuthorizationPurpose: OperatorAuthorizationPurpose
    public let allowedRunClaimSubject: OperatorRunClaimSubject
    public let allowedSourceRoles: [RunSourceRole]
    public let revokedSourceOperatorKeyIDs: [String]
    public let rootSignatureSHA256: String

    fileprivate init(
        file: AdmittedFile,
        policyID: AdmittedSourceManifestSignerPolicyID,
        fields: SourceManifestSignerPolicyFields,
        rootSignatureSHA256: String
    ) {
        self.file = file
        self.policyID = policyID
        self.policySHA256 = file.sha256
        self.policyGeneration = fields.policyGeneration
        self.validFromUnixSeconds = fields.validFromUnixSeconds
        self.validUntilUnixSeconds = fields.validUntilUnixSeconds
        self.rootKeyID = fields.rootKeyID
        self.activeSourceOperatorKeyID =
            fields.activeSourceOperatorKeyID
        self.activeSourceOperatorPublicKeyBase64 =
            fields.activeSourceOperatorPublicKeyBase64
        self.activeSourceOperatorScope =
            fields.activeSourceOperatorScope
        self.allowedAuthorizationPurpose =
            fields.allowedAuthorizationPurpose
        self.allowedRunClaimSubject = fields.allowedRunClaimSubject
        self.allowedSourceRoles = fields.allowedSourceRoles
        self.revokedSourceOperatorKeyIDs =
            fields.revokedSourceOperatorKeyIDs
        self.rootSignatureSHA256 = rootSignatureSHA256
    }
}

public struct PolicyMatchedSourceManifestAuthorizationID:
    Equatable,
    Hashable,
    Sendable
{
    public let rawValue: String

    fileprivate init(rawValue: String) {
        self.rawValue = rawValue
    }
}

/// Inert evidence that one claim-matched source manifest also satisfies the
/// admitted source signer policy.
public struct PolicyMatchedSourceManifestAuthorization:
    Equatable,
    Sendable
{
    public let policyMatchID: PolicyMatchedSourceManifestAuthorizationID
    public let signedClaimID: OperatorSignedRunClaimID
    public let sourcePolicy: AdmittedSourceManifestSignerPolicy
    public let claimMatch: ClaimMatchedSourceManifest
    public let sourceManifest: AdmittedSourceManifest
    public let role: RunSourceRole

    fileprivate init(
        policyMatchID: PolicyMatchedSourceManifestAuthorizationID,
        signedClaimID: OperatorSignedRunClaimID,
        sourcePolicy: AdmittedSourceManifestSignerPolicy,
        claimMatch: ClaimMatchedSourceManifest
    ) {
        self.policyMatchID = policyMatchID
        self.signedClaimID = signedClaimID
        self.sourcePolicy = sourcePolicy
        self.claimMatch = claimMatch
        self.sourceManifest = claimMatch.sourceManifest
        self.role = claimMatch.sourceManifest.role
    }
}

public struct SourceInputsPolicyMatchedID:
    Equatable,
    Hashable,
    Sendable
{
    public let rawValue: String

    fileprivate init(rawValue: String) {
        self.rawValue = rawValue
    }
}

/// Inert aggregate evidence for the two role-bound source manifests named by
/// one signed run claim. All authority-bearing operations remain unavailable.
public struct SourceInputsPolicyMatched: Equatable, Sendable {
    public let sourceInputsPolicyMatchID: SourceInputsPolicyMatchedID
    public let signedClaimID: OperatorSignedRunClaimID
    public let runKeyPolicy: AdmittedOperatorKeyPolicy
    public let sourcePolicy: AdmittedSourceManifestSignerPolicy
    public let baseline: PolicyMatchedSourceManifestAuthorization
    public let candidate: PolicyMatchedSourceManifestAuthorization
    public let sourcePolicySHA256: String
    public let canImportGitObjects = false
    public let canBuild = false
    public let canSpawn = false
    public let canLoadModel = false
    public let canReserveOutput = false
    public let canPublish = false

    fileprivate init(
        sourceInputsPolicyMatchID: SourceInputsPolicyMatchedID,
        signedClaimID: OperatorSignedRunClaimID,
        runKeyPolicy: AdmittedOperatorKeyPolicy,
        sourcePolicy: AdmittedSourceManifestSignerPolicy,
        baseline: PolicyMatchedSourceManifestAuthorization,
        candidate: PolicyMatchedSourceManifestAuthorization
    ) {
        self.sourceInputsPolicyMatchID = sourceInputsPolicyMatchID
        self.signedClaimID = signedClaimID
        self.runKeyPolicy = runKeyPolicy
        self.sourcePolicy = sourcePolicy
        self.baseline = baseline
        self.candidate = candidate
        self.sourcePolicySHA256 = sourcePolicy.policySHA256
    }
}

public enum SourceManifestSignerPolicyVerifier {
    public static let policyDomain =
        "fast-mlx-proof-control-source-manifest-signer-policy-v1"
    public static let policySubject =
        "source-manifest-signing-key-policy"
    public static let admittedPolicyIDDomain =
        "fast-mlx-proof-control-admitted-source-manifest-signer-policy-id-v1"
    public static let policyMatchIDDomain =
        "fast-mlx-proof-control-policy-matched-source-manifest-authorization-id-v1"
    public static let maximumRevokedKeyCount = 256

    public static func policyBytes(
        fields: SourceManifestSignerPolicyFields
    ) throws -> Data {
        try validate(fields)
        let revoked = fields.revokedSourceOperatorKeyIDs.isEmpty
            ? "none"
            : fields.revokedSourceOperatorKeyIDs.joined(separator: ",")
        let roles = fields.allowedSourceRoles
            .map(\.rawValue)
            .joined(separator: ",")
        let lines = [
            policyDomain,
            "subject=\(policySubject)",
            "root_key_id=\(fields.rootKeyID)",
            "policy_generation=\(fields.policyGeneration)",
            "valid_from_unix_seconds=\(fields.validFromUnixSeconds)",
            "valid_until_unix_seconds=\(fields.validUntilUnixSeconds)",
            "active_source_operator_key_id=\(fields.activeSourceOperatorKeyID)",
            "active_source_operator_public_key_base64=\(fields.activeSourceOperatorPublicKeyBase64)",
            "active_source_operator_scope=\(fields.activeSourceOperatorScope.rawValue)",
            "allowed_authorization_purpose=\(fields.allowedAuthorizationPurpose.rawValue)",
            "allowed_run_claim_subject=\(fields.allowedRunClaimSubject.rawValue)",
            "allowed_source_roles=\(roles)",
            "revoked_source_operator_key_ids=\(revoked)",
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    public static func admit(
        policyFile: AdmittedFile,
        rootSignatureBase64: String,
        trustAnchor: SourceManifestSignerPolicyTrustAnchor,
        runKeyPolicy: AdmittedOperatorKeyPolicy
    ) throws -> AdmittedSourceManifestSignerPolicy {
        let rootPublicKeyBytes = try validate(trustAnchor)
        guard policyFile.sha256 ==
            trustAnchor.expectedCurrentPolicySHA256
        else {
            throw SourceManifestSignerPolicyError.policyDigestMismatch
        }

        let fields = try parsePolicy(policyFile.bytes)
        guard fields.rootKeyID == trustAnchor.rootKeyID else {
            throw SourceManifestSignerPolicyError.rootKeyIDMismatch
        }
        guard fields.policyGeneration >=
            trustAnchor.minimumPolicyGeneration
        else {
            throw SourceManifestSignerPolicyError
                .policyGenerationRollback(
                    minimum: trustAnchor.minimumPolicyGeneration,
                    actual: fields.policyGeneration
                )
        }
        guard trustAnchor.verificationUnixSeconds >=
            fields.validFromUnixSeconds
        else {
            throw SourceManifestSignerPolicyError.policyNotYetValid
        }
        guard trustAnchor.verificationUnixSeconds <=
            fields.validUntilUnixSeconds
        else {
            throw SourceManifestSignerPolicyError.policyExpired
        }
        try validateKeySeparation(fields, runKeyPolicy: runKeyPolicy)

        let rootPublicKey: Curve25519.Signing.PublicKey
        do {
            rootPublicKey = try Curve25519.Signing.PublicKey(
                rawRepresentation: rootPublicKeyBytes
            )
        } catch {
            throw SourceManifestSignerPolicyError
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
            throw SourceManifestSignerPolicyError.rootSignatureRejected
        }

        let signatureSHA256 = sha256Hex(signature)
        return AdmittedSourceManifestSignerPolicy(
            file: policyFile,
            policyID: AdmittedSourceManifestSignerPolicyID(
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

    public static func matchAuthorization(
        _ authorization: OperatorAuthorizedFile,
        sourcePolicy: AdmittedSourceManifestSignerPolicy,
        role: RunAuthorizedInputRole
    ) throws {
        guard authorization.purpose ==
            sourcePolicy.allowedAuthorizationPurpose
        else {
            throw SourceManifestSignerPolicyError
                .unexpectedAuthorizationPurpose(
                    expected: sourcePolicy.allowedAuthorizationPurpose,
                    actual: authorization.purpose
                )
        }
        let sourceRole = try sourceRole(for: role)
        try matchAuthorization(
            authorization,
            sourcePolicy: sourcePolicy,
            role: sourceRole
        )
    }
}

public enum SourceInputsPolicyResolver {
    public static let sourceInputsPolicyMatchIDDomain =
        "fast-mlx-proof-control-source-inputs-policy-matched-id-v1"

    public static func resolve(
        signedClaim: OperatorSignedRunClaim,
        runKeyPolicy: AdmittedOperatorKeyPolicy,
        sourcePolicy: AdmittedSourceManifestSignerPolicy,
        baseline: ClaimMatchedSourceManifest?,
        candidate: ClaimMatchedSourceManifest?
    ) throws -> SourceInputsPolicyMatched {
        guard signedClaim.fields.policies.sourceSHA256 ==
            sourcePolicy.policySHA256
        else {
            throw SourceManifestSignerPolicyError
                .signedClaimSourcePolicyMismatch
        }
        guard
            signedClaim.fields.operatorKeyPolicySHA256 ==
                runKeyPolicy.policySHA256,
            signedClaim.fields.operatorKeyID ==
                runKeyPolicy.activeOperatorKeyID
        else {
            throw SourceManifestSignerPolicyError
                .signedClaimRunKeyPolicyMismatch
        }

        let baseline = try require(
            baseline,
            role: .baseline,
            signedClaim: signedClaim,
            sourcePolicy: sourcePolicy
        )
        let candidate = try require(
            candidate,
            role: .candidate,
            signedClaim: signedClaim,
            sourcePolicy: sourcePolicy
        )
        let matchID = sourceInputsPolicyMatchID(
            sourcePolicy: sourcePolicy,
            signedClaim: signedClaim,
            runKeyPolicy: runKeyPolicy,
            baseline: baseline,
            candidate: candidate
        )
        return SourceInputsPolicyMatched(
            sourceInputsPolicyMatchID:
                SourceInputsPolicyMatchedID(rawValue: matchID),
            signedClaimID: signedClaim.claimID,
            runKeyPolicy: runKeyPolicy,
            sourcePolicy: sourcePolicy,
            baseline: baseline,
            candidate: candidate
        )
    }
}

private extension SourceManifestSignerPolicyVerifier {
    static func parsePolicy(_ bytes: Data) throws
        -> SourceManifestSignerPolicyFields
    {
        guard
            let text = String(data: bytes, encoding: .utf8),
            Data(text.utf8) == bytes
        else {
            throw SourceManifestSignerPolicyError.nonCanonicalPolicy
        }

        let lines = text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        )
        guard
            lines.count == 14,
            lines[0] == Substring(policyDomain),
            lines[1] == "subject=\(policySubject)",
            lines[13].isEmpty,
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
                prefix: "active_source_operator_key_id="
            ),
            let activePublicKey = value(
                in: lines[7],
                prefix: "active_source_operator_public_key_base64="
            ),
            lines[8] ==
                "active_source_operator_scope=\(SourceManifestSignerPolicyScope.sourceManifest.rawValue)",
            lines[9] ==
                "allowed_authorization_purpose=\(OperatorAuthorizationPurpose.sourceManifest.rawValue)",
            lines[10] ==
                "allowed_run_claim_subject=\(OperatorRunClaimSubject.absorbedMLALoadedResultPair.rawValue)",
            lines[11] == "allowed_source_roles=baseline,candidate",
            let revokedText = value(
                in: lines[12],
                prefix: "revoked_source_operator_key_ids="
            ),
            isCanonicalDecimal(generationText),
            isCanonicalDecimal(validFromText),
            isCanonicalDecimal(validUntilText),
            let generation = UInt64(generationText),
            let validFrom = UInt64(validFromText),
            let validUntil = UInt64(validUntilText)
        else {
            throw SourceManifestSignerPolicyError.nonCanonicalPolicy
        }

        let fields = SourceManifestSignerPolicyFields(
            rootKeyID: rootKeyID,
            policyGeneration: generation,
            validFromUnixSeconds: validFrom,
            validUntilUnixSeconds: validUntil,
            activeSourceOperatorKeyID: activeKeyID,
            activeSourceOperatorPublicKeyBase64: activePublicKey,
            activeSourceOperatorScope: .sourceManifest,
            allowedAuthorizationPurpose: .sourceManifest,
            allowedRunClaimSubject: .absorbedMLALoadedResultPair,
            allowedSourceRoles: [.baseline, .candidate],
            revokedSourceOperatorKeyIDs:
                try parseRevokedKeyIDs(revokedText)
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
            throw SourceManifestSignerPolicyError.nonCanonicalPolicy
        }
        return ids
    }

    static func validate(
        _ fields: SourceManifestSignerPolicyFields
    ) throws {
        guard
            isCanonicalSHA256(fields.rootKeyID),
            isCanonicalSHA256(fields.activeSourceOperatorKeyID),
            fields.policyGeneration > 0,
            fields.validFromUnixSeconds <= fields.validUntilUnixSeconds,
            fields.activeSourceOperatorScope == .sourceManifest,
            fields.allowedAuthorizationPurpose == .sourceManifest,
            fields.allowedRunClaimSubject ==
                .absorbedMLALoadedResultPair,
            fields.allowedSourceRoles == [.baseline, .candidate],
            fields.revokedSourceOperatorKeyIDs.count <=
                maximumRevokedKeyCount,
            fields.revokedSourceOperatorKeyIDs.allSatisfy(
                isCanonicalSHA256
            ),
            zip(
                fields.revokedSourceOperatorKeyIDs,
                fields.revokedSourceOperatorKeyIDs.dropFirst()
            ).allSatisfy(<)
        else {
            throw SourceManifestSignerPolicyError.nonCanonicalPolicy
        }

        let activePublicKey = try decodeCanonicalBase64(
            fields.activeSourceOperatorPublicKeyBase64,
            expectedByteCount: 32,
            error: .invalidActiveSourceOperatorPublicKeyEncoding
        )
        do {
            _ = try Curve25519.Signing.PublicKey(
                rawRepresentation: activePublicKey
            )
        } catch {
            throw SourceManifestSignerPolicyError
                .invalidActiveSourceOperatorPublicKeyEncoding
        }
        guard sha256Hex(activePublicKey) ==
            fields.activeSourceOperatorKeyID
        else {
            throw SourceManifestSignerPolicyError
                .activeSourceOperatorKeyIDMismatch
        }
        guard fields.rootKeyID != fields.activeSourceOperatorKeyID else {
            throw SourceManifestSignerPolicyError
                .rootAndActiveSourceKeysMustDiffer
        }
        guard !fields.revokedSourceOperatorKeyIDs.contains(
            fields.rootKeyID
        ) else {
            throw SourceManifestSignerPolicyError
                .rootKeyRevocationUnsupported
        }
        guard !fields.revokedSourceOperatorKeyIDs.contains(
            fields.activeSourceOperatorKeyID
        ) else {
            throw SourceManifestSignerPolicyError
                .activeSourceOperatorKeyRevoked
        }
    }

    static func validateKeySeparation(
        _ fields: SourceManifestSignerPolicyFields,
        runKeyPolicy: AdmittedOperatorKeyPolicy
    ) throws {
        guard fields.activeSourceOperatorKeyID !=
            runKeyPolicy.activeOperatorKeyID
        else {
            throw SourceManifestSignerPolicyError
                .sourceAndRunActiveKeysMustDiffer
        }
        guard fields.rootKeyID != runKeyPolicy.activeOperatorKeyID else {
            throw SourceManifestSignerPolicyError
                .sourceRootAndRunActiveKeyMustDiffer
        }
        guard fields.activeSourceOperatorKeyID !=
            runKeyPolicy.rootKeyID
        else {
            throw SourceManifestSignerPolicyError
                .sourceActiveAndRunPolicyRootMustDiffer
        }
    }

    static func validate(
        _ trustAnchor: SourceManifestSignerPolicyTrustAnchor
    ) throws -> Data {
        guard isCanonicalSHA256(trustAnchor.rootKeyID) else {
            throw SourceManifestSignerPolicyError
                .invalidTrustAnchor(.rootKeyID)
        }
        guard isCanonicalSHA256(
            trustAnchor.expectedCurrentPolicySHA256
        ) else {
            throw SourceManifestSignerPolicyError.invalidTrustAnchor(
                .expectedCurrentPolicySHA256
            )
        }
        let rootPublicKey = try decodeCanonicalBase64(
            trustAnchor.rootPublicKeyBase64,
            expectedByteCount: 32,
            error: .invalidRootPublicKeyEncoding
        )
        guard sha256Hex(rootPublicKey) == trustAnchor.rootKeyID else {
            throw SourceManifestSignerPolicyError
                .rootPublicKeyIDMismatch
        }
        return rootPublicKey
    }

    static func sourceRole(
        for role: RunAuthorizedInputRole
    ) throws -> RunSourceRole {
        switch role {
        case .baselineSource:
            .baseline
        case .candidateSource:
            .candidate
        case .worker:
            throw SourceManifestSignerPolicyError
                .unsupportedInputRole(role)
        }
    }

    static func matchAuthorization(
        _ authorization: OperatorAuthorizedFile,
        sourcePolicy: AdmittedSourceManifestSignerPolicy,
        role: RunSourceRole
    ) throws {
        guard sourcePolicy.allowedSourceRoles.contains(role) else {
            throw SourceManifestSignerPolicyError
                .sourceRoleMismatch(expected: role, actual: role)
        }
        guard authorization.operatorKeyID ==
            sourcePolicy.activeSourceOperatorKeyID
        else {
            throw SourceManifestSignerPolicyError
                .sourceAuthorizationKeyMismatch(role: role)
        }
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
        error: SourceManifestSignerPolicyError
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

    static func policyMatchID(
        sourcePolicy: AdmittedSourceManifestSignerPolicy,
        signedClaim: OperatorSignedRunClaim,
        claimMatch: ClaimMatchedSourceManifest
    ) -> String {
        let manifest = claimMatch.sourceManifest
        let authorization = manifest.authorizedFile
        let lines = [
            policyMatchIDDomain,
            "source_policy_id=\(sourcePolicy.policyID.rawValue)",
            "run_claim_id=\(signedClaim.claimID.rawValue)",
            "claim_matched_source_manifest_id=\(claimMatch.matchID.rawValue)",
            "source_authorization_id=\(authorization.authorizationID.rawValue)",
            "purpose=\(authorization.purpose.rawValue)",
            "role=\(manifest.role.rawValue)",
            "operator_key_id=\(authorization.operatorKeyID)",
            "payload_sha256=\(authorization.file.sha256)",
            "payload_bytes=\(UInt64(authorization.file.bytes.count))",
            "claim_sha256=\(authorization.claimSHA256)",
            "signature_sha256=\(authorization.signatureSHA256)",
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

private extension SourceInputsPolicyResolver {
    static func require(
        _ claimMatch: ClaimMatchedSourceManifest?,
        role: RunSourceRole,
        signedClaim: OperatorSignedRunClaim,
        sourcePolicy: AdmittedSourceManifestSignerPolicy
    ) throws -> PolicyMatchedSourceManifestAuthorization {
        guard let claimMatch else {
            throw SourceManifestSignerPolicyError
                .missingRequiredSourceRole(role)
        }
        guard claimMatch.signedClaimID == signedClaim.claimID else {
            throw SourceManifestSignerPolicyError
                .claimMatchedSourceManifestClaimMismatch(role: role)
        }
        guard claimMatch.sourceManifest.role == role else {
            throw SourceManifestSignerPolicyError.sourceRoleMismatch(
                expected: role,
                actual: claimMatch.sourceManifest.role
            )
        }
        guard claimMatch.sourceManifest.authorizedFile.purpose ==
            sourcePolicy.allowedAuthorizationPurpose
        else {
            throw SourceManifestSignerPolicyError
                .unexpectedAuthorizationPurpose(
                    expected: sourcePolicy.allowedAuthorizationPurpose,
                    actual: claimMatch.sourceManifest.authorizedFile.purpose
                )
        }
        try SourceManifestSignerPolicyVerifier.matchAuthorization(
            claimMatch.sourceManifest.authorizedFile,
            sourcePolicy: sourcePolicy,
            role: role
        )

        return PolicyMatchedSourceManifestAuthorization(
            policyMatchID: PolicyMatchedSourceManifestAuthorizationID(
                rawValue:
                    SourceManifestSignerPolicyVerifier.policyMatchID(
                        sourcePolicy: sourcePolicy,
                        signedClaim: signedClaim,
                        claimMatch: claimMatch
                    )
            ),
            signedClaimID: signedClaim.claimID,
            sourcePolicy: sourcePolicy,
            claimMatch: claimMatch
        )
    }

    static func sourceInputsPolicyMatchID(
        sourcePolicy: AdmittedSourceManifestSignerPolicy,
        signedClaim: OperatorSignedRunClaim,
        runKeyPolicy: AdmittedOperatorKeyPolicy,
        baseline: PolicyMatchedSourceManifestAuthorization,
        candidate: PolicyMatchedSourceManifestAuthorization
    ) -> String {
        let lines = [
            sourceInputsPolicyMatchIDDomain,
            "run_claim_id=\(signedClaim.claimID.rawValue)",
            "operator_key_policy_id=\(runKeyPolicy.admissionID.rawValue)",
            "source_policy_id=\(sourcePolicy.policyID.rawValue)",
            "baseline_policy_match_id=\(baseline.policyMatchID.rawValue)",
            "candidate_policy_match_id=\(candidate.policyMatchID.rawValue)",
            "source_policy_sha256=\(sourcePolicy.policySHA256)",
        ]
        return SourceManifestSignerPolicyVerifier.sha256Hex(
            Data((lines.joined(separator: "\n") + "\n").utf8)
        )
    }
}
