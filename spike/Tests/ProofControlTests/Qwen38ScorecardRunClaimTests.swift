import Foundation
import XCTest

@testable import ProofControl

/// Slice 1: claim bytes + parser + structural verifier for the Qwen38
/// scorecard chain. This schema is deliberately independent of the frozen
/// absorbed-MLA `OperatorSignedRunClaim` — own domain string, own subject,
/// own policy domain (docs/task-inbox/
/// 2026-09-01-qwen38-proof-runner-scope-and-chain-design.md). There is no
/// key-policy admission and no cryptographic signature check in this slice;
/// `verify` only confirms the signature is structurally well-formed.
final class Qwen38ScorecardRunClaimTests: XCTestCase {
    private static let modelSHA256 =
        "2099a9daa28ddb34dda68fceb71fe9adfc572c3c9124cbea84d81f178433f868"
    private static let tokenizerSHA256 =
        "d3029454444784caf6a6bc5673a912268b2ca76922abdd5f0af88d3b0c2e4059"
    private static let tensorManifestSHA256 =
        "58a347d9ebdd20bb2e095a03de44e37e0e682d66bd2c981b6b07adaef11e0276"
    private static let chatTemplateSHA256 =
        "ad48c6767dfd64a422b84e29d9e848c542ed236e47efa9002344577850ebcfcb"
    private static let quantizationIdentity = "int4-group64-affine"
    private static let targetModelID = "mlx-community/Qwen3-8B-4bit-MTP-Target"
    private static let targetRevision = "8f2c9a1"
    private static let drafterModelID = "mlx-community/Qwen3-8B-4bit-MTP-Drafter"
    private static let drafterRevision = "3d7e0b4"
    private static let sourceID =
        "ffb0ab1879b67abe553270c8d274285efc4421522bfbd272e4a7b024dbbd5111"
    private static let hostAdmissionID =
        "d321d5223b8db7b66f767a027c3de9a7292d706d96122aca7725a6643fac3264"
    private static let harnessGitSHA1 =
        "beb7e461bd9d593202bb6e52c0dd808589f5fb65"
    // The REAL frozen scorecard workload identity (a name, not a digest) —
    // pins that the schema accepts the identity Slice 4 will actually sign.
    private static let corpusID = "qwen38-27b-frozen-scorecard-workload-v2"
    private static let corpusContentSHA256 =
        "961c672aa953ece83a8b8446675d8124afa27c2e5472403bcb7f3e531f948a9d"
    private static let resultPairID =
        "70956b78d9603040769843fee09052dffa1e9fef191addb51540d2e29b73a3dc"
    private static let differentSHA256 =
        "8e79f7c47e70a9941b8bb25335f63b90542dcd460147a94913017ef3cb614807"

    /// Structurally canonical 64 bytes; deliberately NOT a real signature
    /// over anything. Slice 1 has no key material and performs no
    /// cryptographic check, so a random-but-canonical blob must verify.
    private static let signatureBase64 =
        "2M2KU9W0Ho+bT0vGIiZihANeF0Y8oNvzstQlIPcxefvO+6H2eTBkgdNQXnQ47Lp0ZfwQWtEG2haIe/UfFo2Q7A=="
    private static let secondSignatureBase64 =
        "JXvhNXUi3fcglJejKTqvMBcSlOUhU0XGyuGx7/s9JI7eax9c7qZ1xrMW0dbuw/OUn2yDfHqLVTV5bNPO1cATVQ=="

    /// Independently computed via `shasum -a 256` over the hand-authored
    /// golden bytes below — a real oracle, not derived from the code under
    /// test.
    private static let expectedClaimSHA256 =
        "94808f9c1ffa8033bd494553c68249a74d0411fd92b8dc0c55c5e0c1479cfd1f"
    /// Independently computed via `shasum -a 256` over the raw 64
    /// signature bytes decoded from `signatureBase64`.
    private static let expectedSignatureSHA256 =
        "b0aebb3fc382245a68b4b39db9333916ab2def4348ca91ca0ef6ffbb65e154a0"

    private static let fields = Qwen38ScorecardRunClaimFields(
        subject: .mtpScorecardResultPair,
        modelSHA256: modelSHA256,
        tokenizerSHA256: tokenizerSHA256,
        tensorManifestSHA256: tensorManifestSHA256,
        chatTemplateSHA256: chatTemplateSHA256,
        quantizationIdentity: quantizationIdentity,
        target: Qwen38ScorecardModelIdentity(
            modelID: targetModelID,
            revision: targetRevision
        ),
        drafter: Qwen38ScorecardModelIdentity(
            modelID: drafterModelID,
            revision: drafterRevision
        ),
        sourceID: sourceID,
        hostAdmissionID: hostAdmissionID,
        harnessGitSHA1: harnessGitSHA1,
        gdnOnMode: .on,
        gdnOffMode: .off,
        corpusID: corpusID,
        corpusContentSHA256: corpusContentSHA256,
        resultPairID: resultPairID
    )

    /// Golden bytes, hand-authored to match the field order documented in
    /// the implementation. `wc -c`/`wc -l`/`xxd` verified out-of-band: 1089
    /// bytes, 19 content lines, single trailing LF, no CR.
    private static let claimBytes = Data(
        """
        fastmlx-qwen38-scorecard-run-claim-v1
        subject=qwen38-mtp-scorecard-result-pair
        model_sha256=\(modelSHA256)
        tokenizer_sha256=\(tokenizerSHA256)
        tensor_manifest_sha256=\(tensorManifestSHA256)
        chat_template_sha256=\(chatTemplateSHA256)
        quantization_identity=\(quantizationIdentity)
        target_model_id=\(targetModelID)
        target_revision=\(targetRevision)
        drafter_model_id=\(drafterModelID)
        drafter_revision=\(drafterRevision)
        source_id=\(sourceID)
        host_admission_id=\(hostAdmissionID)
        harness_git_sha1=\(harnessGitSHA1)
        gdn_on_mode=gdn-on
        gdn_off_mode=gdn-off
        corpus_id=\(corpusID)
        corpus_content_sha256=\(corpusContentSHA256)
        result_pair_id=\(resultPairID)

        """.utf8
    )

    // MARK: - Encode / verify round trip

    func testCanonicalBuilderProducesIndependentlyVerifiedGoldenBytes() throws {
        let built = try Qwen38ScorecardRunClaimVerifier.claimBytes(fields: Self.fields)
        XCTAssertEqual(built, Self.claimBytes)

        let verified = try Qwen38ScorecardRunClaimVerifier.verify(
            claimBytes: Self.claimBytes,
            signatureBase64: Self.signatureBase64
        )

        XCTAssertEqual(verified.subject, .mtpScorecardResultPair)
        XCTAssertEqual(verified.claimSHA256, Self.expectedClaimSHA256)
        XCTAssertEqual(verified.signatureSHA256, Self.expectedSignatureSHA256)
        XCTAssertEqual(verified.fields, Self.fields)
        XCTAssertNotEqual(verified.claimSHA256, verified.signatureSHA256)
    }

    func testClaimBytesIsDeterministic() throws {
        let first = try Qwen38ScorecardRunClaimVerifier.claimBytes(fields: Self.fields)
        let second = try Qwen38ScorecardRunClaimVerifier.claimBytes(fields: Self.fields)
        XCTAssertEqual(first, second)
    }

    /// Locks in that Slice 1 performs NO cryptographic verification: a
    /// second, unrelated random-but-canonical 64-byte signature must also
    /// verify successfully against the same claim bytes.
    func testVerifyPerformsNoCryptographicCheckOnAnyCanonicalSignature() throws {
        let verified = try Qwen38ScorecardRunClaimVerifier.verify(
            claimBytes: Self.claimBytes,
            signatureBase64: Self.secondSignatureBase64
        )
        XCTAssertEqual(verified.fields, Self.fields)
        XCTAssertNotEqual(verified.signatureSHA256, Self.expectedSignatureSHA256)
    }

    // MARK: - Refusal: non-canonical byte structure

    func testRejectsNonCanonicalStructureAndScalarEncoding() throws {
        var reorderedLines = String(decoding: Self.claimBytes, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: false)
        reorderedLines.swapAt(8, 9)

        let cases: [Data] = [
            replacing(
                "fastmlx-qwen38-scorecard-run-claim-v1",
                with: "other-protocol-v1"
            ),
            replacing(
                "model_sha256=\(Self.modelSHA256)",
                with: "model_sha256=\(Self.modelSHA256.uppercased())"
            ),
            replacing(
                "harness_git_sha1=\(Self.harnessGitSHA1)",
                with: "harness_git_sha1=\(Self.harnessGitSHA1.uppercased())"
            ),
            replacing("gdn_on_mode=gdn-on", with: "gdn_on_mode=gdn-off"),
            replacing("gdn_off_mode=gdn-off", with: "gdn_off_mode=gdn-on"),
            Self.claimBytes + Data("extra_field=true\n".utf8),
            Data(Self.claimBytes.dropLast()),
            replacing("model_sha256=\(Self.modelSHA256)\n", with: ""),
            replacing(
                "model_sha256=\(Self.modelSHA256)\n",
                with:
                    "model_sha256=\(Self.modelSHA256)\n" +
                    "model_sha256=\(Self.modelSHA256)\n"
            ),
            Data((reorderedLines.map(String.init).joined(separator: "\n")).utf8),
            Data(
                String(decoding: Self.claimBytes, as: UTF8.self)
                    .replacingOccurrences(of: "\n", with: "\r\n")
                    .utf8
            ),
            Data([0xff]),
        ]

        for claim in cases {
            XCTAssertThrowsError(
                try Qwen38ScorecardRunClaimVerifier.verify(
                    claimBytes: claim,
                    signatureBase64: Self.signatureBase64
                )
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardRunClaimError,
                    .nonCanonicalClaim
                )
            }
        }
    }

    func testRejectsUnexpectedSubjectIncludingAbsorbedMLASubject() throws {
        let cases: [(String, String)] = [
            ("subject=qwen38-mtp-scorecard-result-pair", "subject=other-result-pair"),
            (
                "subject=qwen38-mtp-scorecard-result-pair",
                "subject=absorbed-mla-loaded-result-pair"
            ),
        ]

        for (source, replacement) in cases {
            let claim = replacing(source, with: replacement)
            let expectedActual = String(replacement.dropFirst("subject=".count))
            XCTAssertThrowsError(
                try Qwen38ScorecardRunClaimVerifier.verify(
                    claimBytes: claim,
                    signatureBase64: Self.signatureBase64
                )
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardRunClaimError,
                    .unexpectedSubject(
                        expected: .mtpScorecardResultPair,
                        actual: expectedActual
                    )
                )
            }
        }
    }

    // MARK: - Refusal: signature encoding (structural only)

    func testRejectsMalformedSignatureEncoding() throws {
        let malformed = [
            Self.signatureBase64 + "\n",
            String(Self.signatureBase64.dropLast(2)),
            String(Self.signatureBase64.dropLast()) + "!",
            "",
            // Canonical base64 of the WRONG byte count (32 bytes): exercises
            // the length check distinctly from charset/padding refusals.
            Data(repeating: 0x5a, count: 32).base64EncodedString(),
        ]
        for signature in malformed {
            XCTAssertThrowsError(
                try Qwen38ScorecardRunClaimVerifier.verify(
                    claimBytes: Self.claimBytes,
                    signatureBase64: signature
                )
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardRunClaimError,
                    .invalidSignatureEncoding
                )
            }
        }
    }

    // MARK: - Refusal: field validation at construction (not just bytes)

    func testClaimBytesRejectsEmptyOrMalformedDigestFields() {
        XCTAssertThrowsError(
            try Qwen38ScorecardRunClaimVerifier.claimBytes(
                fields: makeFields(modelSHA256: "")
            )
        ) { error in
            XCTAssertEqual(error as? Qwen38ScorecardRunClaimError, .nonCanonicalClaim)
        }
        XCTAssertThrowsError(
            try Qwen38ScorecardRunClaimVerifier.claimBytes(
                fields: makeFields(modelSHA256: Self.modelSHA256.uppercased())
            )
        ) { error in
            XCTAssertEqual(error as? Qwen38ScorecardRunClaimError, .nonCanonicalClaim)
        }
        XCTAssertThrowsError(
            try Qwen38ScorecardRunClaimVerifier.claimBytes(
                fields: makeFields(resultPairID: Self.differentSHA256 + "0")
            )
        ) { error in
            XCTAssertEqual(error as? Qwen38ScorecardRunClaimError, .nonCanonicalClaim)
        }
    }

    func testClaimBytesRejectsTokenFieldsWithEmbeddedNewlineEmptyOrOverlong() {
        let badTokens = [
            "bad\nvalue",
            "",
            String(repeating: "x", count: 201),
            "bad value with space",
            // Non-newline control/edge bytes: prove the full 0x21-0x7e bound,
            // not just the newline special case.
            "bad\tvalue",
            "bad\u{7f}value",
        ]
        for token in badTokens {
            XCTAssertThrowsError(
                try Qwen38ScorecardRunClaimVerifier.claimBytes(
                    fields: makeFields(quantizationIdentity: token)
                ),
                "quantization_identity=\(token.debugDescription) should be refused"
            ) { error in
                XCTAssertEqual(error as? Qwen38ScorecardRunClaimError, .nonCanonicalClaim)
            }
            XCTAssertThrowsError(
                try Qwen38ScorecardRunClaimVerifier.claimBytes(
                    fields: makeFields(corpusID: token)
                ),
                "corpus_id=\(token.debugDescription) should be refused"
            ) { error in
                XCTAssertEqual(error as? Qwen38ScorecardRunClaimError, .nonCanonicalClaim)
            }
        }
    }

    func testClaimBytesRejectsGDNModeTampering() {
        XCTAssertThrowsError(
            try Qwen38ScorecardRunClaimVerifier.claimBytes(
                fields: makeFields(gdnOnMode: .off)
            )
        ) { error in
            XCTAssertEqual(error as? Qwen38ScorecardRunClaimError, .nonCanonicalClaim)
        }
        XCTAssertThrowsError(
            try Qwen38ScorecardRunClaimVerifier.claimBytes(
                fields: makeFields(gdnOffMode: .on)
            )
        ) { error in
            XCTAssertEqual(error as? Qwen38ScorecardRunClaimError, .nonCanonicalClaim)
        }
    }

    // MARK: - Helpers

    private func makeFields(
        modelSHA256: String = fields.modelSHA256,
        quantizationIdentity: String = fields.quantizationIdentity,
        resultPairID: String = fields.resultPairID,
        gdnOnMode: Qwen38ScorecardGDNMode = fields.gdnOnMode,
        gdnOffMode: Qwen38ScorecardGDNMode = fields.gdnOffMode,
        corpusID: String = fields.corpusID
    ) -> Qwen38ScorecardRunClaimFields {
        Qwen38ScorecardRunClaimFields(
            subject: .mtpScorecardResultPair,
            modelSHA256: modelSHA256,
            tokenizerSHA256: Self.tokenizerSHA256,
            tensorManifestSHA256: Self.tensorManifestSHA256,
            chatTemplateSHA256: Self.chatTemplateSHA256,
            quantizationIdentity: quantizationIdentity,
            target: Qwen38ScorecardModelIdentity(
                modelID: Self.targetModelID,
                revision: Self.targetRevision
            ),
            drafter: Qwen38ScorecardModelIdentity(
                modelID: Self.drafterModelID,
                revision: Self.drafterRevision
            ),
            sourceID: Self.sourceID,
            hostAdmissionID: Self.hostAdmissionID,
            harnessGitSHA1: Self.harnessGitSHA1,
            gdnOnMode: gdnOnMode,
            gdnOffMode: gdnOffMode,
            corpusID: corpusID,
            corpusContentSHA256: Self.corpusContentSHA256,
            resultPairID: resultPairID
        )
    }

    private func replacing(_ source: String, with replacement: String) -> Data {
        let text = String(decoding: Self.claimBytes, as: UTF8.self)
        return Data(text.replacingOccurrences(of: source, with: replacement).utf8)
    }
}
