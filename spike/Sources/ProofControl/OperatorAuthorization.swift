import CryptoKit
import Foundation

public enum OperatorAuthorizationPurpose: String, Equatable, Sendable {
    case sourceManifest = "source-manifest"
    case workerBytes = "worker-bytes"
}

public enum OperatorAuthorizationError: Error, Equatable, Sendable {
    case nonCanonicalClaim
    case unsupportedPurpose
    case unexpectedPurpose(
        expected: OperatorAuthorizationPurpose,
        actual: OperatorAuthorizationPurpose
    )
    case payloadDigestMismatch
    case payloadByteCountMismatch
    case invalidAllowedKeyID
    case invalidPublicKeyEncoding
    case operatorKeyIDMismatch
    case invalidSignatureEncoding
    case signatureRejected
}

extension OperatorAuthorizationError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .nonCanonicalClaim:
            "operator claim is not the canonical five-line UTF-8 form"
        case .unsupportedPurpose:
            "operator claim purpose is not supported"
        case .unexpectedPurpose(let expected, let actual):
            "operator claim purpose \(actual.rawValue) does not match expected purpose \(expected.rawValue)"
        case .payloadDigestMismatch:
            "operator claim payload digest does not match admitted bytes"
        case .payloadByteCountMismatch:
            "operator claim payload byte count does not match admitted bytes"
        case .invalidAllowedKeyID:
            "allowed operator key ID is not canonical lowercase SHA-256"
        case .invalidPublicKeyEncoding:
            "operator public key is not canonical base64 for 32 raw Ed25519 bytes"
        case .operatorKeyIDMismatch:
            "operator public key does not match the allowed key ID"
        case .invalidSignatureEncoding:
            "operator signature is not canonical base64 for 64 raw Ed25519 bytes"
        case .signatureRejected:
            "operator signature is not valid for the canonical claim"
        }
    }
}

public struct OperatorAuthorizedFileID: Equatable, Hashable, Sendable {
    public let rawValue: String

    fileprivate init(rawValue: String) {
        self.rawValue = rawValue
    }
}

public struct OperatorAuthorizedFile: Equatable, Sendable {
    public let file: AdmittedFile
    public let purpose: OperatorAuthorizationPurpose
    public let authorizationID: OperatorAuthorizedFileID
    public let operatorKeyID: String
    public let claimSHA256: String
    public let signatureSHA256: String

    fileprivate init(
        file: AdmittedFile,
        purpose: OperatorAuthorizationPurpose,
        authorizationID: OperatorAuthorizedFileID,
        operatorKeyID: String,
        claimSHA256: String,
        signatureSHA256: String
    ) {
        self.file = file
        self.purpose = purpose
        self.authorizationID = authorizationID
        self.operatorKeyID = operatorKeyID
        self.claimSHA256 = claimSHA256
        self.signatureSHA256 = signatureSHA256
    }
}

public enum OperatorAuthorization {
    public static let claimDomain = "fast-mlx-proof-control-signature-v1"
    public static let claimSubject = "admitted-file"
    public static let authorizedFileIDDomain =
        "fast-mlx-proof-control-authorized-file-id-v1"

    public static func claimBytes(
        purpose: OperatorAuthorizationPurpose,
        payloadSHA256: String,
        payloadByteCount: UInt64
    ) throws -> Data {
        guard isCanonicalSHA256(payloadSHA256) else {
            throw OperatorAuthorizationError.nonCanonicalClaim
        }

        let lines = [
            claimDomain,
            "subject=\(claimSubject)",
            "purpose=\(purpose.rawValue)",
            "payload_sha256=\(payloadSHA256)",
            "payload_bytes=\(payloadByteCount)",
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    public static func verify(
        admittedFile: AdmittedFile,
        expectedPurpose: OperatorAuthorizationPurpose,
        claimBytes: Data,
        signatureBase64: String,
        publicKeyBase64: String,
        allowedKeyID: String
    ) throws -> OperatorAuthorizedFile {
        let claim = try parseClaim(claimBytes)

        guard claim.purpose == expectedPurpose else {
            throw OperatorAuthorizationError.unexpectedPurpose(
                expected: expectedPurpose,
                actual: claim.purpose
            )
        }
        guard claim.payloadSHA256 == admittedFile.sha256 else {
            throw OperatorAuthorizationError.payloadDigestMismatch
        }
        guard claim.payloadByteCount == UInt64(admittedFile.bytes.count) else {
            throw OperatorAuthorizationError.payloadByteCountMismatch
        }
        guard isCanonicalSHA256(allowedKeyID) else {
            throw OperatorAuthorizationError.invalidAllowedKeyID
        }

        let publicKeyBytes = try decodeCanonicalBase64(
            publicKeyBase64,
            expectedByteCount: 32,
            error: .invalidPublicKeyEncoding
        )
        let actualKeyID = sha256Hex(publicKeyBytes)
        guard actualKeyID == allowedKeyID else {
            throw OperatorAuthorizationError.operatorKeyIDMismatch
        }

        let publicKey: Curve25519.Signing.PublicKey
        do {
            publicKey = try Curve25519.Signing.PublicKey(
                rawRepresentation: publicKeyBytes
            )
        } catch {
            throw OperatorAuthorizationError.invalidPublicKeyEncoding
        }

        let signature = try decodeCanonicalBase64(
            signatureBase64,
            expectedByteCount: 64,
            error: .invalidSignatureEncoding
        )
        guard publicKey.isValidSignature(signature, for: claimBytes) else {
            throw OperatorAuthorizationError.signatureRejected
        }

        let claimSHA256 = sha256Hex(claimBytes)
        let signatureSHA256 = sha256Hex(signature)
        return OperatorAuthorizedFile(
            file: admittedFile,
            purpose: claim.purpose,
            authorizationID: OperatorAuthorizedFileID(
                rawValue: authorizedFileID(
                    purpose: claim.purpose,
                    operatorKeyID: actualKeyID,
                    claimSHA256: claimSHA256,
                    signatureSHA256: signatureSHA256
                )
            ),
            operatorKeyID: actualKeyID,
            claimSHA256: claimSHA256,
            signatureSHA256: signatureSHA256
        )
    }
}

private extension OperatorAuthorization {
    struct ParsedClaim {
        let purpose: OperatorAuthorizationPurpose
        let payloadSHA256: String
        let payloadByteCount: UInt64
    }

    static func parseClaim(_ bytes: Data) throws -> ParsedClaim {
        guard
            let text = String(data: bytes, encoding: .utf8),
            Data(text.utf8) == bytes
        else {
            throw OperatorAuthorizationError.nonCanonicalClaim
        }

        let lines = text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        )
        guard
            lines.count == 6,
            lines[0] == Substring(claimDomain),
            lines[1] == "subject=\(claimSubject)",
            lines[5].isEmpty,
            let purposeValue = value(
                in: lines[2],
                prefix: "purpose="
            ),
            let digest = value(
                in: lines[3],
                prefix: "payload_sha256="
            ),
            let byteCountText = value(
                in: lines[4],
                prefix: "payload_bytes="
            ),
            isCanonicalSHA256(digest),
            isCanonicalDecimal(byteCountText),
            let byteCount = UInt64(byteCountText)
        else {
            throw OperatorAuthorizationError.nonCanonicalClaim
        }
        guard let purpose = OperatorAuthorizationPurpose(rawValue: purposeValue) else {
            throw OperatorAuthorizationError.unsupportedPurpose
        }

        return ParsedClaim(
            purpose: purpose,
            payloadSHA256: digest,
            payloadByteCount: byteCount
        )
    }

    static func value(in line: Substring, prefix: String) -> String? {
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
        error: OperatorAuthorizationError
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

    static func authorizedFileID(
        purpose: OperatorAuthorizationPurpose,
        operatorKeyID: String,
        claimSHA256: String,
        signatureSHA256: String
    ) -> String {
        let lines = [
            authorizedFileIDDomain,
            "purpose=\(purpose.rawValue)",
            "operator_key_id=\(operatorKeyID)",
            "claim_sha256=\(claimSHA256)",
            "signature_sha256=\(signatureSHA256)",
        ]
        return sha256Hex(Data((lines.joined(separator: "\n") + "\n").utf8))
    }

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
