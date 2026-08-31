import CryptoKit
import Foundation

struct Qwen38PerformanceAttributionControllerAdmission: Equatable, Sendable {
    let controllerSignatureVerified: Bool
    let eligibleForLiveCollection: Bool
    let promotionAuthorized: Bool
    let runtimeAuthorityGranted: Bool
}

struct Qwen38PerformanceAttributionTrustedControllerRoot:
    Equatable, Sendable
{
    let controllerPublicKeyBase64: String
    fileprivate let keyID: String

    fileprivate init(
        controllerPublicKeyBase64: String,
        keyID: String
    ) {
        self.controllerPublicKeyBase64 = controllerPublicKeyBase64
        self.keyID = keyID
    }
}

enum Qwen38PerformanceAttributionControllerSignatureError:
    Error, Equatable, CustomStringConvertible, Sendable
{
    case invalidDigest
    case trustedControllerRootUnconfigured
    case nonCanonicalClaim
    case invalidPublicKeyEncoding
    case invalidSignatureEncoding
    case signatureRejected

    var description: String {
        switch self {
        case .invalidDigest:
            return "invalid controller signature digest"
        case .trustedControllerRootUnconfigured:
            return "trusted controller root unconfigured"
        case .nonCanonicalClaim:
            return "non-canonical controller signature claim"
        case .invalidPublicKeyEncoding:
            return "invalid controller public key encoding"
        case .invalidSignatureEncoding:
            return "invalid controller signature encoding"
        case .signatureRejected:
            return "controller signature rejected"
        }
    }
}

enum Qwen38PerformanceAttributionControllerSignature {
    static let claimDomain =
        "fast-mlx.qwen3.8.performance-attribution.controller-signature"
    static let claimVersion = 1

    static func canonicalClaim(
        capture: Qwen38PerformanceAttributionUnsignedAuthorityCapture,
        trustedControllerRoot:
            Qwen38PerformanceAttributionTrustedControllerRoot
    ) throws -> Data {
        try validateDigests([
            capture.policyDocumentSHA256,
            capture.policyPinDocumentSHA256,
            capture.productionRouteReceiptDocumentSHA256,
            capture.productionRouteReceiptPinDocumentSHA256,
            capture.semanticPolicyDigest,
            capture.productionRouteReceiptDigest,
            capture.runIdentityDigest,
            capture.backendBuildIdentityDigest,
            capture.observationDigest,
        ])

        let lines = [
            claimDomain,
            "version=\(claimVersion)",
            "policy_document_sha256=\(capture.policyDocumentSHA256)",
            "policy_pin_document_sha256=\(capture.policyPinDocumentSHA256)",
            "production_route_receipt_document_sha256="
                + capture.productionRouteReceiptDocumentSHA256,
            "production_route_receipt_pin_document_sha256="
                + capture.productionRouteReceiptPinDocumentSHA256,
            "semantic_policy_digest=\(capture.semanticPolicyDigest)",
            "production_route_receipt_digest="
                + capture.productionRouteReceiptDigest,
            "run_identity_digest=\(capture.runIdentityDigest)",
            "backend_build_identity_digest="
                + capture.backendBuildIdentityDigest,
            "observation_digest=\(capture.observationDigest)",
            "controller_key_id=\(trustedControllerRoot.keyID)",
            "eligible_for_live_collection=true",
            "promotion_authorized=false",
            "runtime_authority_granted=false",
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func verify(
        capture: Qwen38PerformanceAttributionUnsignedAuthorityCapture,
        claimBytes: Data,
        trustedControllerRoot:
            Qwen38PerformanceAttributionTrustedControllerRoot,
        controllerSignatureBase64: String
    ) throws -> Qwen38PerformanceAttributionControllerAdmission {
        let expectedClaim = try canonicalClaim(
            capture: capture,
            trustedControllerRoot: trustedControllerRoot)
        guard let claimText = String(data: claimBytes, encoding: .utf8),
            Data(claimText.utf8) == claimBytes,
            claimBytes == expectedClaim
        else {
            throw Qwen38PerformanceAttributionControllerSignatureError
                .nonCanonicalClaim
        }

        let publicKeyBytes = try decodeCanonicalBase64(
            trustedControllerRoot.controllerPublicKeyBase64,
            expectedByteCount: 32,
            error: .invalidPublicKeyEncoding)

        let publicKey: Curve25519.Signing.PublicKey
        do {
            publicKey = try Curve25519.Signing.PublicKey(
                rawRepresentation: publicKeyBytes)
        } catch {
            throw Qwen38PerformanceAttributionControllerSignatureError
                .invalidPublicKeyEncoding
        }

        let signature = try decodeCanonicalBase64(
            controllerSignatureBase64,
            expectedByteCount: 64,
            error: .invalidSignatureEncoding)
        guard publicKey.isValidSignature(signature, for: claimBytes) else {
            throw Qwen38PerformanceAttributionControllerSignatureError
                .signatureRejected
        }

        return Qwen38PerformanceAttributionControllerAdmission(
            controllerSignatureVerified: true,
            eligibleForLiveCollection: true,
            promotionAuthorized: false,
            runtimeAuthorityGranted: false)
    }

    static func productionTrustedControllerRoot()
        throws -> Qwen38PerformanceAttributionTrustedControllerRoot
    {
        throw Qwen38PerformanceAttributionControllerSignatureError
            .trustedControllerRootUnconfigured
    }

#if DEBUG
    static func debugTrustedControllerRoot(
        controllerPublicKeyBase64: String
    ) throws -> Qwen38PerformanceAttributionTrustedControllerRoot {
        let publicKeyBytes = try decodeCanonicalBase64(
            controllerPublicKeyBase64,
            expectedByteCount: 32,
            error: .invalidPublicKeyEncoding)
        do {
            _ = try Curve25519.Signing.PublicKey(
                rawRepresentation: publicKeyBytes)
        } catch {
            throw Qwen38PerformanceAttributionControllerSignatureError
                .invalidPublicKeyEncoding
        }
        return Qwen38PerformanceAttributionTrustedControllerRoot(
            controllerPublicKeyBase64: controllerPublicKeyBase64,
            keyID: sha256Hex(publicKeyBytes))
    }
#endif

    private static func validateDigests(_ values: [String]) throws {
        guard values.allSatisfy(isCanonicalSHA256) else {
            throw Qwen38PerformanceAttributionControllerSignatureError
                .invalidDigest
        }
    }

    private static func isCanonicalSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (0x30...0x39).contains($0) || (0x61...0x66).contains($0)
        }
    }

    private static func decodeCanonicalBase64(
        _ value: String,
        expectedByteCount: Int,
        error: Qwen38PerformanceAttributionControllerSignatureError
    ) throws -> Data {
        guard let decoded = Data(base64Encoded: value),
            decoded.count == expectedByteCount,
            decoded.base64EncodedString() == value
        else {
            throw error
        }
        return decoded
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
