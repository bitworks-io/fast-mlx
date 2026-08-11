import Darwin
import Foundation
import XCTest
@testable import ProofControl

final class OperatorAuthorizationTests: XCTestCase {
    private static let payload = Data("admitted\n".utf8)
    private static let payloadSHA256 =
        "e7635c5d652fd35a6f2c259b32694b78d0aaefc90d611609e6401a76fcf31265"
    private static let publicKeyBase64 =
        "11qYAYKxCrfVS/7TyWQHOg7hcvPapiMlrwIaaPcHURo="
    private static let operatorKeyID =
        "21fe31dfa154a261626bf854046fd2271b7bed4b6abe45aa58877ef47f9721b9"
    private static let alternatePublicKeyBase64 =
        "O2onvM62pC1io6jQKm8Nc2UyFXcd4kOmOsBIoYtZ2ik="
    private static let alternateOperatorKeyID =
        "139e3940e64b5491722088d9a0d741628fc826e09475d341a780acde3c4b8070"
    private static let signatureBase64 =
        "9LRKVkmUV5DT9roMIHLZX/71u/fXurUVOF4S/N81li/4o4TkbGXpTlPipuPbVYEL7uuNqItA57/jh3l6dFrCCA=="
    private static let rawDigestSignatureBase64 =
        "i1VpIBLdYr6rGttV8Yg+h6as4nZpGkkG6OIaj0Sg+l4hRJP7Tgj5YgThL3a21izfevJgZ+VhrqisZ34/2fthCg=="
    private static let workerSignatureBase64 =
        "/TC4JwjI8rD9hECfEQ/Hl7RDhqEUqsQPzU+2SIbozm7TD7ittnvjWf0PAzVOeDIdWESlOtCu41k7wvGjeLHgAA=="
    private static let claimBytes = Data(
        """
        fast-mlx-proof-control-signature-v1
        subject=admitted-file
        purpose=source-manifest
        payload_sha256=e7635c5d652fd35a6f2c259b32694b78d0aaefc90d611609e6401a76fcf31265
        payload_bytes=9

        """.utf8
    )

    private var caseRoot: URL!

    override func setUpWithError() throws {
        let canonicalTemporaryPath = try XCTUnwrap(
            Darwin.realpath(NSTemporaryDirectory(), nil)
        )
        defer { Darwin.free(canonicalTemporaryPath) }

        caseRoot = URL(
            fileURLWithPath: String(cString: canonicalTemporaryPath),
            isDirectory: true
        )
        .appendingPathComponent("fast-mlx-proof-authorization-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: caseRoot,
            withIntermediateDirectories: false
        )
    }

    override func tearDownWithError() throws {
        if let caseRoot {
            try? FileManager.default.removeItem(at: caseRoot)
        }
    }

    func testPinnedEd25519VectorAuthorizesExactAdmittedBytes() throws {
        let admitted = try makeAdmittedPayload()

        XCTAssertEqual(
            try OperatorAuthorization.claimBytes(
                purpose: .sourceManifest,
                payloadSHA256: Self.payloadSHA256,
                payloadByteCount: 9
            ),
            Self.claimBytes
        )

        let authorized = try OperatorAuthorization.verify(
            admittedFile: admitted,
            expectedPurpose: .sourceManifest,
            claimBytes: Self.claimBytes,
            signatureBase64: Self.signatureBase64,
            publicKeyBase64: Self.publicKeyBase64,
            allowedKeyID: Self.operatorKeyID
        )

        XCTAssertEqual(authorized.file, admitted)
        XCTAssertEqual(authorized.purpose, OperatorAuthorizationPurpose.sourceManifest)
        XCTAssertEqual(authorized.operatorKeyID, Self.operatorKeyID)
        XCTAssertEqual(authorized.file.bytes, Self.payload)
        XCTAssertEqual(
            authorized.claimSHA256,
            "3f53254b5133285bffd46e069ec8639d3687b406a4fd55137f54d1e1eea240db"
        )
        XCTAssertEqual(
            authorized.signatureSHA256,
            "a4c63505ba9f01c159771343ea4aeb6e85cdd7a2642a3eed06ee7a26d112f630"
        )
    }

    func testAuthorizationIDIsDomainSeparatedAndPurposeBound() throws {
        let admitted = try makeAdmittedPayload()
        let sourceAuthorized = try verify(admitted)
        let workerClaim = replacing(
            "purpose=source-manifest",
            with: "purpose=worker-bytes"
        )
        let workerAuthorized = try verify(
            admitted,
            claimBytes: workerClaim,
            signatureBase64: Self.workerSignatureBase64,
            expectedPurpose: .workerBytes
        )

        XCTAssertEqual(
            sourceAuthorized.authorizationID.rawValue,
            "848cd6771d42ee71ccb6616eeb0a779b5c66f739398d88d350a66a55964538c9"
        )
        XCTAssertEqual(
            workerAuthorized.authorizationID.rawValue,
            "805ffd1af7f4ff87dfb8619f3f78cb1ed46dfd47589fa01f9695d94651bca364"
        )
        XCTAssertNotEqual(
            sourceAuthorized.authorizationID,
            workerAuthorized.authorizationID
        )
        XCTAssertNotEqual(
            sourceAuthorized.authorizationID.rawValue,
            sourceAuthorized.file.sha256
        )
        XCTAssertNotEqual(
            sourceAuthorized.authorizationID.rawValue,
            sourceAuthorized.operatorKeyID
        )
        XCTAssertNotEqual(
            sourceAuthorized.authorizationID.rawValue,
            sourceAuthorized.claimSHA256
        )
        XCTAssertNotEqual(
            sourceAuthorized.authorizationID.rawValue,
            sourceAuthorized.signatureSHA256
        )
    }

    func testClaimParserRejectsNonCanonicalAndMismatchedPayloadFields() throws {
        let admitted = try makeAdmittedPayload()
        let cases: [(Data, OperatorAuthorizationError)] = [
            (
                replacing(
                    "payload_sha256=\(Self.payloadSHA256)",
                    with: "payload_sha256=\(Self.payloadSHA256.uppercased())"
                ),
                .nonCanonicalClaim
            ),
            (
                replacing("payload_bytes=9", with: "payload_bytes=09"),
                .nonCanonicalClaim
            ),
            (
                replacing(
                    "payload_sha256=\(Self.payloadSHA256)",
                    with: "payload_sha256=\(String(repeating: "0", count: 64))"
                ),
                .payloadDigestMismatch
            ),
            (
                replacing("payload_bytes=9", with: "payload_bytes=10"),
                .payloadByteCountMismatch
            ),
            (
                Self.claimBytes + Data("extra=true\n".utf8),
                .nonCanonicalClaim
            ),
            (
                Data(Self.claimBytes.dropLast()),
                .nonCanonicalClaim
            ),
            (
                replacing(
                    "purpose=source-manifest\n",
                    with: "purpose=source-manifest\npurpose=source-manifest\n"
                ),
                .nonCanonicalClaim
            ),
            (
                replacing("subject=admitted-file\n", with: ""),
                .nonCanonicalClaim
            ),
            (
                Data(
                    String(decoding: Self.claimBytes, as: UTF8.self)
                        .replacingOccurrences(of: "\n", with: "\r\n")
                        .utf8
                ),
                .nonCanonicalClaim
            ),
            (
                Data([0xff]),
                .nonCanonicalClaim
            ),
        ]

        for (claim, expectedError) in cases {
            XCTAssertThrowsError(
                try verify(admitted, claimBytes: claim),
                "expected \(expectedError) for \(String(decoding: claim, as: UTF8.self))"
            ) { error in
                XCTAssertEqual(error as? OperatorAuthorizationError, expectedError)
            }
        }
    }

    func testClaimRejectsDomainSubjectPurposeAndRawDigestSignatureReplay() throws {
        let admitted = try makeAdmittedPayload()
        let malformedCases = [
            replacing(
                "fast-mlx-proof-control-signature-v1",
                with: "other-protocol-v1"
            ),
            replacing("subject=admitted-file", with: "subject=run"),
        ]

        for claim in malformedCases {
            XCTAssertThrowsError(try verify(admitted, claimBytes: claim)) { error in
                XCTAssertEqual(error as? OperatorAuthorizationError, .nonCanonicalClaim)
            }
        }

        let unknownPurposeClaim = replacing(
            "purpose=source-manifest",
            with: "purpose=unknown"
        )
        XCTAssertThrowsError(
            try verify(admitted, claimBytes: unknownPurposeClaim)
        ) { error in
            XCTAssertEqual(error as? OperatorAuthorizationError, .unsupportedPurpose)
        }

        let workerClaim = replacing(
            "purpose=source-manifest",
            with: "purpose=worker-bytes"
        )
        XCTAssertThrowsError(
            try verify(
                admitted,
                claimBytes: workerClaim,
                signatureBase64: Self.workerSignatureBase64
            )
        ) { error in
            XCTAssertEqual(
                error as? OperatorAuthorizationError,
                .unexpectedPurpose(expected: .sourceManifest, actual: .workerBytes)
            )
        }
        XCTAssertThrowsError(
            try verify(admitted, expectedPurpose: .workerBytes)
        ) { error in
            XCTAssertEqual(
                error as? OperatorAuthorizationError,
                .unexpectedPurpose(expected: .workerBytes, actual: .sourceManifest)
            )
        }
        let workerAuthorized = try verify(
            admitted,
            claimBytes: workerClaim,
            signatureBase64: Self.workerSignatureBase64,
            expectedPurpose: .workerBytes
        )
        XCTAssertEqual(workerAuthorized.purpose, .workerBytes)

        XCTAssertThrowsError(
            try verify(
                admitted,
                signatureBase64: Self.rawDigestSignatureBase64
            )
        ) { error in
            XCTAssertEqual(error as? OperatorAuthorizationError, .signatureRejected)
        }
    }

    func testRejectsNonCanonicalKeyAndSignatureConfiguration() throws {
        let admitted = try makeAdmittedPayload()
        let signatureBytes = try XCTUnwrap(
            Data(base64Encoded: Self.signatureBase64)
        )

        XCTAssertThrowsError(
            try verify(
                admitted,
                publicKeyBase64: String(Self.publicKeyBase64.dropLast())
            )
        ) { error in
            XCTAssertEqual(error as? OperatorAuthorizationError, .invalidPublicKeyEncoding)
        }
        XCTAssertThrowsError(
            try verify(
                admitted,
                publicKeyBase64: Self.publicKeyBase64 + "\n"
            )
        ) { error in
            XCTAssertEqual(error as? OperatorAuthorizationError, .invalidPublicKeyEncoding)
        }
        XCTAssertThrowsError(
            try verify(admitted, allowedKeyID: Self.operatorKeyID.uppercased())
        ) { error in
            XCTAssertEqual(error as? OperatorAuthorizationError, .invalidAllowedKeyID)
        }
        XCTAssertThrowsError(
            try verify(admitted, allowedKeyID: String(repeating: "0", count: 64))
        ) { error in
            XCTAssertEqual(error as? OperatorAuthorizationError, .operatorKeyIDMismatch)
        }
        XCTAssertThrowsError(
            try verify(
                admitted,
                publicKeyBase64: Self.alternatePublicKeyBase64,
                allowedKeyID: Self.alternateOperatorKeyID
            )
        ) { error in
            XCTAssertEqual(error as? OperatorAuthorizationError, .signatureRejected)
        }
        XCTAssertThrowsError(
            try verify(
                admitted,
                signatureBase64: String(Self.signatureBase64.dropLast())
            )
        ) { error in
            XCTAssertEqual(error as? OperatorAuthorizationError, .invalidSignatureEncoding)
        }
        XCTAssertThrowsError(
            try verify(
                admitted,
                signatureBase64: Self.signatureBase64 + "\n"
            )
        ) { error in
            XCTAssertEqual(error as? OperatorAuthorizationError, .invalidSignatureEncoding)
        }
        XCTAssertThrowsError(
            try verify(
                admitted,
                signatureBase64: Data(signatureBytes.dropLast()).base64EncodedString()
            )
        ) { error in
            XCTAssertEqual(error as? OperatorAuthorizationError, .invalidSignatureEncoding)
        }
    }

    func testAuthorizationUsesCapturedBytesAfterSameInodeRewrite() throws {
        let input = caseRoot.appendingPathComponent("source-manifest.json")
        try Self.payload.write(to: input)
        let admitted = try AdmittedFile.capture(
            absolutePath: input.path,
            maximumBytes: 64
        )

        let handle = try FileHandle(forWritingTo: input)
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Data("MALICIOUS\n".utf8))
        try handle.truncate(atOffset: 10)
        try handle.close()

        let authorized = try verify(admitted)
        XCTAssertEqual(authorized.file.bytes, Self.payload)
        XCTAssertNotEqual(authorized.file.bytes, try Data(contentsOf: input))
    }

    private func makeAdmittedPayload() throws -> AdmittedFile {
        let input = caseRoot.appendingPathComponent("source-manifest.json")
        try Self.payload.write(to: input)
        return try AdmittedFile.capture(
            absolutePath: input.path,
            maximumBytes: 64
        )
    }

    private func verify(
        _ admitted: AdmittedFile,
        claimBytes: Data = claimBytes,
        signatureBase64: String = signatureBase64,
        publicKeyBase64: String = publicKeyBase64,
        allowedKeyID: String = operatorKeyID,
        expectedPurpose: OperatorAuthorizationPurpose = .sourceManifest
    ) throws -> OperatorAuthorizedFile {
        try OperatorAuthorization.verify(
            admittedFile: admitted,
            expectedPurpose: expectedPurpose,
            claimBytes: claimBytes,
            signatureBase64: signatureBase64,
            publicKeyBase64: publicKeyBase64,
            allowedKeyID: allowedKeyID
        )
    }

    private func replacing(_ source: String, with replacement: String) -> Data {
        let text = String(decoding: Self.claimBytes, as: UTF8.self)
        return Data(text.replacingOccurrences(of: source, with: replacement).utf8)
    }
}
