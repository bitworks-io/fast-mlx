import CryptoKit
import Foundation

/// Qwen38 scorecard chain resolved run authorization (chain Slice 3).
///
/// This is the step that turns Slice 1's STRUCTURAL claim envelope plus
/// Slice 2's root-admitted operator key policy into an actual
/// operator-authorized value: `resolve` performs the REAL Ed25519
/// verification of the operator's claim signature over the exact claim
/// bytes against the policy's admitted active operator key. See
/// docs/task-inbox/2026-09-01-qwen38-proof-runner-scope-and-chain-design.md
/// (Slice 3) and its 2026-09-02 security-review addendum.
///
/// Security invariants (binding, enforced structurally and by the
/// no-self-sign structural test in ProofControlTests):
///   - EXISTENCE of a `Qwen38ScorecardResolvedRunAuthorization` value IS
///     the authorization: the only way to construct one is through
///     `resolve`, whose signature check cannot be skipped. There is no
///     stored "promotable" flag to forge — a runner that does not hold a
///     resolved authorization stays `promotable:false`.
///   - This file must NEVER conform the resolved type to Swift's
///     serialization/decoding protocols: a decoding path would be a
///     public forgery constructor for the resolved value. The structural
///     test enforces the absence of those protocol tokens in this file.
///   - No signing API exists in ProofControl or the runner: this module
///     can only VERIFY operator signatures, never mint them. Real claim
///     signing happens outside, at the operator credential boundary.
///   - Replay is by design: the claim deliberately carries no nonce
///     (binding item 3 of the reviewed chain design), so one signed claim
///     authorizes unlimited re-runs of the same deterministic measurement
///     pair. The per-run nonce is minted post-signing by the trusted
///     parent (Slice 4); it is not part of the signed authorization.
///   - Apple CryptoKit Ed25519 signatures are randomized, so two valid
///     signatures over the same claim bytes differ; `authorizationID`
///     therefore identifies a SIGNATURE INSTANCE (a distinct signing
///     event), not merely the claim content. This is deliberate: audit
///     trails can distinguish separate operator signing events.
///
/// Pure canonical-encoding helpers below are deliberately DUPLICATED from
/// the sibling Qwen38 chain files rather than extracted or shared, per
/// binding item 4 of the reviewed slice design: each chain file stays a
/// self-contained auditable unit and the frozen absorbed-MLA files are
/// never modified.
public enum Qwen38ScorecardRunAuthorizationError: Error, Equatable, Sendable {
    case subjectNotAllowed
    case scopeNotAllowed
    case activeOperatorKeyRevoked
    case invalidActiveOperatorPublicKeyEncoding
    case activeOperatorKeyIDMismatch
    case invalidClaimSignatureEncoding
    case claimSignatureRejected
}

extension Qwen38ScorecardRunAuthorizationError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .subjectNotAllowed:
            "qwen38 resolved authorization claim subject is not allowed by the admitted policy"
        case .scopeNotAllowed:
            "qwen38 resolved authorization policy scope is not the scorecard run-claim scope"
        case .activeOperatorKeyRevoked:
            "qwen38 resolved authorization active operator key is revoked"
        case .invalidActiveOperatorPublicKeyEncoding:
            "qwen38 resolved authorization operator public key is not canonical base64 for 32 Ed25519 bytes"
        case .activeOperatorKeyIDMismatch:
            "qwen38 resolved authorization operator public key does not match its policy key ID"
        case .invalidClaimSignatureEncoding:
            "qwen38 resolved authorization claim signature is not canonical base64 for 64 Ed25519 bytes"
        case .claimSignatureRejected:
            "qwen38 resolved authorization claim signature is not valid over the exact claim bytes"
        }
    }
}

/// Operator-authorized run. Constructible only by
/// `Qwen38ScorecardRunAuthorizationResolver.resolve` after a REAL Ed25519
/// signature verification — see the invariants on the enclosing file
/// comment. Deliberately excluded from all serialization protocols.
public struct Qwen38ScorecardResolvedRunAuthorization: Equatable, Sendable {
    public let policyAdmissionID: AdmittedQwen38ScorecardKeyPolicyID
    public let operatorKeyID: String
    public let claim: Qwen38ScorecardVerifiedRunClaim
    public let claimSHA256: String
    public let signatureSHA256: String
    public let authorizationID: String

    fileprivate init(
        policyAdmissionID: AdmittedQwen38ScorecardKeyPolicyID,
        operatorKeyID: String,
        claim: Qwen38ScorecardVerifiedRunClaim,
        claimSHA256: String,
        signatureSHA256: String,
        authorizationID: String
    ) {
        self.policyAdmissionID = policyAdmissionID
        self.operatorKeyID = operatorKeyID
        self.claim = claim
        self.claimSHA256 = claimSHA256
        self.signatureSHA256 = signatureSHA256
        self.authorizationID = authorizationID
    }
}

public enum Qwen38ScorecardRunAuthorizationResolver {
    public static let authorizationIDDomain =
        "fastmlx-qwen38-scorecard-resolved-run-authorization-id-v1"

    /// Resolves an operator-signed claim into a run authorization.
    ///
    /// Fail-closed pipeline:
    ///   1. Slice 1 structural verification of the claim bytes (canonical
    ///      form, single pinned subject) and of the signature ENCODING.
    ///   2. Defensive re-guards over the admitted policy. These re-check
    ///      invariants that `Qwen38ScorecardKeyPolicyVerifier.admit`
    ///      already established and that `AdmittedQwen38ScorecardKeyPolicy`'s
    ///      fileprivate init makes unforgeable from outside; they are
    ///      defense-in-depth against future refactors, not reachable
    ///      states today.
    ///   3. REAL Ed25519 verification of the operator signature over the
    ///      exact claim bytes against the policy's admitted active key.
    public static func resolve(
        claimBytes: Data,
        claimSignatureBase64: String,
        policy: AdmittedQwen38ScorecardKeyPolicy
    ) throws -> Qwen38ScorecardResolvedRunAuthorization {
        let claim = try Qwen38ScorecardRunClaimVerifier.verify(
            claimBytes: claimBytes,
            signatureBase64: claimSignatureBase64
        )

        // Defensive re-guards: unforgeable-by-construction today (both
        // input types pin these invariants behind fileprivate inits), kept
        // as defense-in-depth against future refactors.
        guard claim.subject == policy.allowedClaimSubject else {
            throw Qwen38ScorecardRunAuthorizationError.subjectNotAllowed
        }
        guard policy.activeOperatorScope == .scorecardRunClaim else {
            throw Qwen38ScorecardRunAuthorizationError.scopeNotAllowed
        }
        guard !policy.revokedOperatorKeyIDs.contains(
            policy.activeOperatorKeyID
        ) else {
            throw Qwen38ScorecardRunAuthorizationError
                .activeOperatorKeyRevoked
        }

        let publicKeyBytes = try decodeCanonicalBase64(
            policy.activeOperatorPublicKeyBase64,
            expectedByteCount: 32,
            error: .invalidActiveOperatorPublicKeyEncoding
        )
        guard sha256Hex(publicKeyBytes) == policy.activeOperatorKeyID else {
            throw Qwen38ScorecardRunAuthorizationError
                .activeOperatorKeyIDMismatch
        }
        let publicKey: Curve25519.Signing.PublicKey
        do {
            publicKey = try Curve25519.Signing.PublicKey(
                rawRepresentation: publicKeyBytes
            )
        } catch {
            throw Qwen38ScorecardRunAuthorizationError
                .invalidActiveOperatorPublicKeyEncoding
        }

        let signature = try decodeCanonicalBase64(
            claimSignatureBase64,
            expectedByteCount: 64,
            error: .invalidClaimSignatureEncoding
        )
        guard publicKey.isValidSignature(signature, for: claimBytes) else {
            throw Qwen38ScorecardRunAuthorizationError.claimSignatureRejected
        }

        let signatureSHA256 = sha256Hex(signature)
        return Qwen38ScorecardResolvedRunAuthorization(
            policyAdmissionID: policy.admissionID,
            operatorKeyID: policy.activeOperatorKeyID,
            claim: claim,
            claimSHA256: claim.claimSHA256,
            signatureSHA256: signatureSHA256,
            authorizationID: authorizationID(
                policyAdmissionID: policy.admissionID.rawValue,
                operatorKeyID: policy.activeOperatorKeyID,
                claimSHA256: claim.claimSHA256,
                signatureSHA256: signatureSHA256
            )
        )
    }
}

private extension Qwen38ScorecardRunAuthorizationResolver {
    // Pure canonical-encoding helpers deliberately DUPLICATED from the
    // sibling Qwen38 chain files, per binding item 4 of the reviewed slice
    // design.
    static func decodeCanonicalBase64(
        _ value: String,
        expectedByteCount: Int,
        error: Qwen38ScorecardRunAuthorizationError
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

    static func authorizationID(
        policyAdmissionID: String,
        operatorKeyID: String,
        claimSHA256: String,
        signatureSHA256: String
    ) -> String {
        let lines = [
            authorizationIDDomain,
            "policy_admission_id=\(policyAdmissionID)",
            "operator_key_id=\(operatorKeyID)",
            "claim_sha256=\(claimSHA256)",
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
