import Darwin
import Foundation
import XCTest
@testable import ProofControl

final class OperatorSignedRunClaimTests: XCTestCase {
    private static let operatorKeyID =
        ProofControlKeyPolicyFixtures.activeKeyID
    private static let operatorKeyPolicySHA256 =
        ProofControlKeyPolicyFixtures.policySHA256
    private static let hostAdmissionID =
        "1c0c0ebdab94f49bf897b305568f9a1d73d10b91e1bb986d929c23c768272f5b"
    private static let runnerSHA256 =
        "ab49f3eb142e8e18c94fa79d2017f57b6141b1c4cdc042f1a0d0e3de5297090e"
    private static let resultPairID =
        "729e2531e1eecff3416fb970c6d3f71c1ca717601061263d63bbf55a9f69c307"
    private static let tokenizerManifestSHA256 =
        "4cb82e26fa573871ce4c0a532cb1bfb85082df0d7d04fc04a2e0d3303fae1076"
    private static let signatureBase64 =
        "EZ/V8V4zIRVxMBqq2r+7vSEO1QVPVAZUbLwVzPjeAwoEAEaubl/UE+1P+gEWNv8n76m2pqkSpPtInNx4V4BeDA=="
    private static let rawDigestSignatureBase64 =
        "zJvcRV1KdbEheKJmVu8EfBgyH0+iU5SGloIxHnHeFqU3U28BdZLeqURya9hIpcROWVOq/D79bvnG7r9MuXcmCA=="

    private static let sharedPayload = Data("admitted\n".utf8)
    private static let sharedPayloadSHA256 =
        "e7635c5d652fd35a6f2c259b32694b78d0aaefc90d611609e6401a76fcf31265"
    private static let sharedPublicKeyBase64 =
        "11qYAYKxCrfVS/7TyWQHOg7hcvPapiMlrwIaaPcHURo="
    private static let sharedOperatorKeyID =
        "21fe31dfa154a261626bf854046fd2271b7bed4b6abe45aa58877ef47f9721b9"
    private static let baselineSignatureBase64 =
        "9LRKVkmUV5DT9roMIHLZX/71u/fXurUVOF4S/N81li/4o4TkbGXpTlPipuPbVYEL7uuNqItA57/jh3l6dFrCCA=="
    private static let workerSignatureBase64 =
        "/TC4JwjI8rD9hECfEQ/Hl7RDhqEUqsQPzU+2SIbozm7TD7ittnvjWf0PAzVOeDIdWESlOtCu41k7wvGjeLHgAA=="

    private static let candidatePayload = Data("candidate\n".utf8)
    private static let candidatePayloadSHA256 =
        "1e81270f1a47dce22a2e4985250c74b2e3374443734f1492b03ea2cd2af4ec48"
    private static let candidatePublicKeyBase64 =
        "F6L2OWYoYrRGFVPJaxTTel/Tjd09wDms4zixO29EIzo="
    private static let candidateOperatorKeyID =
        "91d77da244dd24d006d312dacb3ea4776a17336aff532526c881d576a500fe0c"
    private static let candidateSignatureBase64 =
        "Ubz2lDZlh34OhY/FVHs71ApP6fPk4R1uf57FVGONNvTaqKoe7iFJSqOS5RFSRpCL1L9TJmBvluJStznH60dgAQ=="

    private static let claimBytes = Data(
        """
        fast-mlx-proof-control-run-claim-v1
        subject=absorbed-mla-loaded-result-pair
        operator_key_id=\(operatorKeyID)
        operator_key_policy_sha256=\(operatorKeyPolicySHA256)
        host_admission_id=1c0c0ebdab94f49bf897b305568f9a1d73d10b91e1bb986d929c23c768272f5b
        runner_sha256=ab49f3eb142e8e18c94fa79d2017f57b6141b1c4cdc042f1a0d0e3de5297090e
        runner_bytes=7
        worker_authorization_id=805ffd1af7f4ff87dfb8619f3f78cb1ed46dfd47589fa01f9695d94651bca364
        worker_sha256=e7635c5d652fd35a6f2c259b32694b78d0aaefc90d611609e6401a76fcf31265
        worker_bytes=9
        source_policy_sha256=b6112f60d206fe384a939d855db79c43eff1b3d9c4b63c7e6dc46cfb19e2cd16
        dependency_policy_sha256=d13bb64e180312bf25e6836c233c77f25d08cad1b412487351fd40e4e7a7709b
        build_policy_sha256=24da85a9a166870432b288bb029453d1ecefe2b560c49b1b91f37ecda1533df4
        runtime_policy_sha256=e542e26d7e3f8eadd6e9f2a9657575fdcb101b8ca77d60b24453f04bfd543743
        preflight_policy_sha256=6ad56cb0917f60bfcac3febeee1d82a87884addd53b5235c8cae591924348cda
        publication_policy_sha256=66ce1a93db2073f090ae4e05988bf939e5d40274ec0e00f02f3127615e54ab97
        tool_manifest_sha256=30aff620c42a9cc867729b8072557b2aae08b94e7e34014ca633c7241abd4c76
        tool_manifest_bytes=19
        baseline_source_authorization_id=848cd6771d42ee71ccb6616eeb0a779b5c66f739398d88d350a66a55964538c9
        baseline_source_manifest_sha256=e7635c5d652fd35a6f2c259b32694b78d0aaefc90d611609e6401a76fcf31265
        baseline_source_manifest_bytes=9
        baseline_git_commit_sha1=4400b8932df374945ebef2cc504782016297c0df
        baseline_git_tree_sha1=7c81bdf20225425e61efec24ce02835c9893fffa
        baseline_route=decompressed-deepseek-v3
        baseline_slot=baseline
        baseline_build_receipt_id=fc4f76d37973409fffe7a70a2676ec2c0709f8f2c7572fd3227a4f7bedc34ce4
        baseline_binary_sha256=c329127855982bcb8b8e7027af5a1039e9a27752097d740978b853fd89a56d2d
        baseline_binary_bytes=1001
        candidate_source_authorization_id=ef374a55c02714b86150e35ab821b5c5be29617bae826b4d237849c19a477629
        candidate_source_manifest_sha256=1e81270f1a47dce22a2e4985250c74b2e3374443734f1492b03ea2cd2af4ec48
        candidate_source_manifest_bytes=10
        candidate_git_commit_sha1=f8d86192e2c558605a8745c446598063aedaac36
        candidate_git_tree_sha1=f9ca3359d542ad621650fd968193e97051f56afb
        candidate_route=absorbed-mla-deepseek-v3-explicit
        candidate_slot=candidate
        candidate_build_receipt_id=47ba48101af7184e05ab39281fd126687706ee84c9813bbce81704b867393176
        candidate_binary_sha256=b75afc06019cb1f4b81851ad4d23fe7586c10564e26d06166ab124a9ff406233
        candidate_binary_bytes=1002
        model_authorization_id=5b61127fd5c206085681ee887ea38b825a06f270269e79cd9faada549442829d
        model_manifest_sha256=af862f7b1abfc56b313380b8680192247d54e5d8109d76babe1338f416c2e413
        model_manifest_bytes=55
        tokenizer_authorization_id=bd37e7f5ceab34c59c59ce4316630b3509d553f1981e45886b4bf9a473662e97
        tokenizer_manifest_sha256=\(tokenizerManifestSHA256)
        tokenizer_manifest_bytes=66
        workload_authorization_id=7d93688fc7ca3407bcd984e8c28a63bfbe671d0d7b0d05cb5d5a26dcba85e764
        workload_sha256=7f6c7360bf24fccf4b59cf9e9a9dfc99fd6a3b085fb2260ef707a34067bb911e
        workload_bytes=77
        result_pair_id=729e2531e1eecff3416fb970c6d3f71c1ca717601061263d63bbf55a9f69c307

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
        .appendingPathComponent("fast-mlx-signed-run-claim-\(UUID().uuidString)")
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

    func testCanonicalBuilderAndPinnedSignatureAdmitOnlyTypedClaim() throws {
        let inputs = try makeRunInputs()
        let runner = try makeRunner()
        let expectations = try makeExpectations(
            inputs: inputs,
            runner: runner
        )

        XCTAssertEqual(
            try OperatorSignedRunClaimVerifier.claimBytes(fields: Self.fields),
            Self.claimBytes
        )

        let signed = try OperatorSignedRunClaimVerifier.verify(
            claimBytes: Self.claimBytes,
            signatureBase64: Self.signatureBase64,
            expectations: expectations
        )

        XCTAssertEqual(signed.subject, .absorbedMLALoadedResultPair)
        XCTAssertEqual(signed.operatorKeyID, Self.operatorKeyID)
        XCTAssertEqual(
            signed.claimSHA256,
            "ea17f12d4fca33576bddf331b3701bd0feebf67067a5b83cf08e8e12cc152939"
        )
        XCTAssertEqual(
            signed.signatureSHA256,
            "19e5a4d0c3601e9bca990ceee99dc55105c74a79656d73ddf05ccf46cfd72f9b"
        )
        XCTAssertEqual(
            signed.claimID.rawValue,
            "dd72c4520dc3c0319678627288b2e1d6dbdcdc6b438216ef5cf1fb0977244677"
        )
        XCTAssertEqual(signed.fields, Self.fields)
        XCTAssertNotEqual(signed.claimID.rawValue, signed.claimSHA256)
        XCTAssertNotEqual(signed.claimID.rawValue, signed.signatureSHA256)
        XCTAssertNotEqual(signed.claimID.rawValue, Self.operatorKeyID)
        XCTAssertNotEqual(signed.claimID.rawValue, Self.resultPairID)
        XCTAssertNotEqual(
            signed.claimID.rawValue,
            inputs.worker.authorizationID.rawValue
        )
    }

    func testRejectsNonCanonicalStructureAndScalarEncoding() throws {
        let expectations = try makeExpectations()
        var reorderedLines = String(decoding: Self.claimBytes, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: false)
        reorderedLines.swapAt(10, 11)

        let cases: [Data] = [
            replacing(
                "fast-mlx-proof-control-run-claim-v1",
                with: "other-protocol-v1"
            ),
            replacing(
                "host_admission_id=\(Self.hostAdmissionID)",
                with: "host_admission_id=\(Self.hostAdmissionID.uppercased())"
            ),
            replacing("runner_bytes=7", with: "runner_bytes=07"),
            replacing(
                "runner_bytes=7",
                with: "runner_bytes=18446744073709551616"
            ),
            replacing(
                "baseline_git_commit_sha1=4400b8932df374945ebef2cc504782016297c0df",
                with: "baseline_git_commit_sha1=4400B8932DF374945EBEF2CC504782016297C0DF"
            ),
            replacing(
                "baseline_route=decompressed-deepseek-v3",
                with: "baseline_route=absorbed-mla-deepseek-v3-explicit"
            ),
            replacing(
                "candidate_slot=candidate",
                with: "candidate_slot=baseline"
            ),
            Self.claimBytes + Data("extra=true\n".utf8),
            Data(Self.claimBytes.dropLast()),
            replacing(
                "runner_sha256=\(Self.runnerSHA256)\n",
                with: ""
            ),
            replacing(
                "runner_sha256=\(Self.runnerSHA256)\n",
                with:
                    "runner_sha256=\(Self.runnerSHA256)\n" +
                    "runner_sha256=\(Self.runnerSHA256)\n"
            ),
            Data((reorderedLines.map(String.init).joined(separator: "\n")).utf8),
            Data(
                String(decoding: Self.claimBytes, as: UTF8.self)
                    .replacingOccurrences(of: "\n", with: "\r\n")
                    .utf8
            ),
            Data([0xff]),
            replacing(
                "candidate_source_authorization_id=ef374a55c02714b86150e35ab821b5c5be29617bae826b4d237849c19a477629",
                with:
                    "candidate_source_authorization_id=" +
                    "848cd6771d42ee71ccb6616eeb0a779b5c66f739398d88d350a66a55964538c9"
            ),
        ]

        for claim in cases {
            XCTAssertThrowsError(
                try OperatorSignedRunClaimVerifier.verify(
                    claimBytes: claim,
                    signatureBase64: Self.signatureBase64,
                    expectations: expectations
                )
            ) { error in
                XCTAssertEqual(
                    error as? OperatorSignedRunClaimError,
                    .nonCanonicalClaim
                )
            }
        }
    }

    func testRejectsUnexpectedSubjectAndEveryAvailableContextMismatch() throws {
        let expectations = try makeExpectations()
        XCTAssertThrowsError(
            try verify(
                replacing(
                    "subject=absorbed-mla-loaded-result-pair",
                    with: "subject=other-result-pair"
                ),
                expectations: expectations
            )
        ) { error in
            XCTAssertEqual(
                error as? OperatorSignedRunClaimError,
                .unexpectedSubject(
                    expected: .absorbedMLALoadedResultPair,
                    actual: "other-result-pair"
                )
            )
        }

        let mismatches: [(Data, OperatorSignedRunClaimError)] = [
            (
                replacing(
                    "operator_key_id=\(Self.operatorKeyID)",
                    with: "operator_key_id=\(Self.differentSHA256)"
                ),
                .operatorKeyIDMismatch
            ),
            (
                replacing(
                    "operator_key_policy_sha256=\(Self.operatorKeyPolicySHA256)",
                    with: "operator_key_policy_sha256=\(Self.differentSHA256)"
                ),
                .operatorKeyPolicyMismatch
            ),
            (
                replacing(
                    "host_admission_id=\(Self.hostAdmissionID)",
                    with: "host_admission_id=\(Self.differentSHA256)"
                ),
                .hostAdmissionIDMismatch
            ),
            (
                replacing(
                    "runner_sha256=\(Self.runnerSHA256)",
                    with: "runner_sha256=\(Self.differentSHA256)"
                ),
                .runnerDigestMismatch
            ),
            (
                replacing("runner_bytes=7", with: "runner_bytes=8"),
                .runnerByteCountMismatch
            ),
            (
                replacing(
                    "worker_authorization_id=805ffd1af7f4ff87dfb8619f3f78cb1ed46dfd47589fa01f9695d94651bca364",
                    with: "worker_authorization_id=\(Self.differentSHA256)"
                ),
                .inputAuthorizationIDMismatch(role: .worker)
            ),
            (
                replacing(
                    "worker_sha256=\(Self.sharedPayloadSHA256)",
                    with: "worker_sha256=\(Self.differentSHA256)"
                ),
                .inputDigestMismatch(role: .worker)
            ),
            (
                replacing("worker_bytes=9", with: "worker_bytes=8"),
                .inputByteCountMismatch(role: .worker)
            ),
            (
                replacing(
                    "baseline_source_authorization_id=848cd6771d42ee71ccb6616eeb0a779b5c66f739398d88d350a66a55964538c9",
                    with: "baseline_source_authorization_id=\(Self.differentSHA256)"
                ),
                .inputAuthorizationIDMismatch(role: .baselineSource)
            ),
            (
                replacing(
                    "baseline_source_manifest_sha256=\(Self.sharedPayloadSHA256)",
                    with: "baseline_source_manifest_sha256=\(Self.differentSHA256)"
                ),
                .inputDigestMismatch(role: .baselineSource)
            ),
            (
                replacing(
                    "baseline_source_manifest_bytes=9",
                    with: "baseline_source_manifest_bytes=8"
                ),
                .inputByteCountMismatch(role: .baselineSource)
            ),
            (
                replacing(
                    "candidate_source_authorization_id=ef374a55c02714b86150e35ab821b5c5be29617bae826b4d237849c19a477629",
                    with: "candidate_source_authorization_id=\(Self.differentSHA256)"
                ),
                .inputAuthorizationIDMismatch(role: .candidateSource)
            ),
            (
                replacing(
                    "candidate_source_manifest_sha256=\(Self.candidatePayloadSHA256)",
                    with: "candidate_source_manifest_sha256=\(Self.differentSHA256)"
                ),
                .inputDigestMismatch(role: .candidateSource)
            ),
            (
                replacing(
                    "candidate_source_manifest_bytes=10",
                    with: "candidate_source_manifest_bytes=11"
                ),
                .inputByteCountMismatch(role: .candidateSource)
            ),
            (
                replacing(
                    "result_pair_id=\(Self.resultPairID)",
                    with: "result_pair_id=\(Self.differentSHA256)"
                ),
                .resultPairIDMismatch
            ),
        ]

        for (claim, expectedError) in mismatches {
            XCTAssertThrowsError(
                try verify(claim, expectations: expectations)
            ) { error in
                XCTAssertEqual(
                    error as? OperatorSignedRunClaimError,
                    expectedError
                )
            }
        }
    }

    func testRejectsNonCanonicalContextAndSignatureReplay() throws {
        let inputs = try makeRunInputs()
        let runner = try makeRunner()
        let invalidExpectations: [
            (OperatorRunClaimAdmissionExpectations, OperatorRunClaimExpectationField)
        ] = [
            (
                try makeExpectations(
                    inputs: inputs,
                    runner: runner,
                    hostAdmissionID: Self.hostAdmissionID.uppercased()
                ),
                .hostAdmissionID
            ),
            (
                try makeExpectations(
                    inputs: inputs,
                    runner: runner,
                    resultPairID: Self.resultPairID.uppercased()
                ),
                .resultPairID
            ),
        ]

        for (expectations, field) in invalidExpectations {
            XCTAssertThrowsError(
                try verify(expectations: expectations)
            ) { error in
                XCTAssertEqual(
                    error as? OperatorSignedRunClaimError,
                    .invalidExpectation(field)
                )
            }
        }

        XCTAssertThrowsError(
            try verify(
                signatureBase64: Self.signatureBase64 + "\n",
                expectations: try makeExpectations(
                    inputs: inputs,
                    runner: runner
                )
            )
        ) { error in
            XCTAssertEqual(
                error as? OperatorSignedRunClaimError,
                .invalidSignatureEncoding
            )
        }
        XCTAssertThrowsError(
            try verify(
                signatureBase64: Self.rawDigestSignatureBase64,
                expectations: try makeExpectations(
                    inputs: inputs,
                    runner: runner
                )
            )
        ) { error in
            XCTAssertEqual(
                error as? OperatorSignedRunClaimError,
                .signatureRejected
            )
        }
    }

    private static let fields = OperatorRunClaimFields(
        subject: .absorbedMLALoadedResultPair,
        operatorKeyID: operatorKeyID,
        operatorKeyPolicySHA256: operatorKeyPolicySHA256,
        hostAdmissionID: hostAdmissionID,
        runner: OperatorRunClaimByteIdentity(
            sha256: runnerSHA256,
            byteCount: 7
        ),
        worker: OperatorRunClaimAuthorizedPayloadReference(
            authorizationID:
                "805ffd1af7f4ff87dfb8619f3f78cb1ed46dfd47589fa01f9695d94651bca364",
            payload: OperatorRunClaimByteIdentity(
                sha256: sharedPayloadSHA256,
                byteCount: 9
            )
        ),
        policies: OperatorRunClaimPolicyReferences(
            sourceSHA256:
                "b6112f60d206fe384a939d855db79c43eff1b3d9c4b63c7e6dc46cfb19e2cd16",
            dependencySHA256:
                "d13bb64e180312bf25e6836c233c77f25d08cad1b412487351fd40e4e7a7709b",
            buildSHA256:
                "24da85a9a166870432b288bb029453d1ecefe2b560c49b1b91f37ecda1533df4",
            runtimeSHA256:
                "e542e26d7e3f8eadd6e9f2a9657575fdcb101b8ca77d60b24453f04bfd543743",
            preflightSHA256:
                "6ad56cb0917f60bfcac3febeee1d82a87884addd53b5235c8cae591924348cda",
            publicationSHA256:
                "66ce1a93db2073f090ae4e05988bf939e5d40274ec0e00f02f3127615e54ab97"
        ),
        toolManifest: OperatorRunClaimByteIdentity(
            sha256:
                "30aff620c42a9cc867729b8072557b2aae08b94e7e34014ca633c7241abd4c76",
            byteCount: 19
        ),
        baseline: OperatorRunClaimSourceReference(
            role: .baseline,
            sourceManifest: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID:
                    "848cd6771d42ee71ccb6616eeb0a779b5c66f739398d88d350a66a55964538c9",
                payload: OperatorRunClaimByteIdentity(
                    sha256: sharedPayloadSHA256,
                    byteCount: 9
                )
            ),
            gitCommitSHA1: "4400b8932df374945ebef2cc504782016297c0df",
            gitTreeSHA1: "7c81bdf20225425e61efec24ce02835c9893fffa",
            route: .decompressedDeepSeekV3,
            slot: .baseline,
            buildReceiptID:
                "fc4f76d37973409fffe7a70a2676ec2c0709f8f2c7572fd3227a4f7bedc34ce4",
            binary: OperatorRunClaimByteIdentity(
                sha256:
                    "c329127855982bcb8b8e7027af5a1039e9a27752097d740978b853fd89a56d2d",
                byteCount: 1001
            )
        ),
        candidate: OperatorRunClaimSourceReference(
            role: .candidate,
            sourceManifest: OperatorRunClaimAuthorizedPayloadReference(
                authorizationID:
                    "ef374a55c02714b86150e35ab821b5c5be29617bae826b4d237849c19a477629",
                payload: OperatorRunClaimByteIdentity(
                    sha256: candidatePayloadSHA256,
                    byteCount: 10
                )
            ),
            gitCommitSHA1: "f8d86192e2c558605a8745c446598063aedaac36",
            gitTreeSHA1: "f9ca3359d542ad621650fd968193e97051f56afb",
            route: .absorbedMLADeepSeekV3Explicit,
            slot: .candidate,
            buildReceiptID:
                "47ba48101af7184e05ab39281fd126687706ee84c9813bbce81704b867393176",
            binary: OperatorRunClaimByteIdentity(
                sha256:
                    "b75afc06019cb1f4b81851ad4d23fe7586c10564e26d06166ab124a9ff406233",
                byteCount: 1002
            )
        ),
        model: OperatorRunClaimAuthorizedPayloadReference(
            authorizationID:
                "5b61127fd5c206085681ee887ea38b825a06f270269e79cd9faada549442829d",
            payload: OperatorRunClaimByteIdentity(
                sha256:
                    "af862f7b1abfc56b313380b8680192247d54e5d8109d76babe1338f416c2e413",
                byteCount: 55
            )
        ),
        tokenizer: OperatorRunClaimAuthorizedPayloadReference(
            authorizationID:
                "bd37e7f5ceab34c59c59ce4316630b3509d553f1981e45886b4bf9a473662e97",
            payload: OperatorRunClaimByteIdentity(
                sha256:
                    tokenizerManifestSHA256,
                byteCount: 66
            )
        ),
        workload: OperatorRunClaimAuthorizedPayloadReference(
            authorizationID:
                "7d93688fc7ca3407bcd984e8c28a63bfbe671d0d7b0d05cb5d5a26dcba85e764",
            payload: OperatorRunClaimByteIdentity(
                sha256:
                    "7f6c7360bf24fccf4b59cf9e9a9dfc99fd6a3b085fb2260ef707a34067bb911e",
                byteCount: 77
            )
        ),
        resultPairID: resultPairID
    )

    private func makeExpectations() throws
        -> OperatorRunClaimAdmissionExpectations
    {
        let inputs = try makeRunInputs()
        let runner = try makeRunner()
        return try makeExpectations(inputs: inputs, runner: runner)
    }

    private func makeExpectations(
        inputs: RunAuthorizedInputs,
        runner: AdmittedFile,
        hostAdmissionID: String = hostAdmissionID,
        resultPairID: String = resultPairID
    ) throws -> OperatorRunClaimAdmissionExpectations {
        let keyPolicy = try makeKeyPolicy()
        return OperatorRunClaimAdmissionExpectations(
            keyPolicy: keyPolicy,
            hostAdmissionID: hostAdmissionID,
            runner: runner,
            resultPairID: resultPairID,
            inputs: inputs
        )
    }

    private func makeKeyPolicy() throws -> AdmittedOperatorKeyPolicy {
        let url = caseRoot.appendingPathComponent(
            "\(UUID().uuidString).operator-key-policy"
        )
        try ProofControlKeyPolicyFixtures.policyBytes.write(to: url)
        let file = try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 4_096
        )
        let trustAnchor = OperatorKeyPolicyTrustAnchor(
            rootPublicKeyBase64:
                ProofControlKeyPolicyFixtures.rootPublicKeyBase64,
            rootKeyID: ProofControlKeyPolicyFixtures.rootKeyID,
            expectedCurrentPolicySHA256:
                ProofControlKeyPolicyFixtures.policySHA256,
            minimumPolicyGeneration: 7,
            verificationUnixSeconds: 2_000_000_000
        )
        return try OperatorKeyPolicyVerifier.admit(
            policyFile: file,
            rootSignatureBase64:
                ProofControlKeyPolicyFixtures.policySignatureBase64,
            trustAnchor: trustAnchor
        )
    }

    private func makeRunInputs() throws -> RunAuthorizedInputs {
        let worker = try makeAuthorizedFile(
            name: "worker.swift",
            payload: Self.sharedPayload,
            payloadSHA256: Self.sharedPayloadSHA256,
            purpose: .workerBytes,
            signatureBase64: Self.workerSignatureBase64,
            publicKeyBase64: Self.sharedPublicKeyBase64,
            operatorKeyID: Self.sharedOperatorKeyID
        )
        let baseline = try makeAuthorizedFile(
            name: "baseline-source.manifest",
            payload: Self.sharedPayload,
            payloadSHA256: Self.sharedPayloadSHA256,
            purpose: .sourceManifest,
            signatureBase64: Self.baselineSignatureBase64,
            publicKeyBase64: Self.sharedPublicKeyBase64,
            operatorKeyID: Self.sharedOperatorKeyID
        )
        let candidate = try makeAuthorizedFile(
            name: "candidate-source.manifest",
            payload: Self.candidatePayload,
            payloadSHA256: Self.candidatePayloadSHA256,
            purpose: .sourceManifest,
            signatureBase64: Self.candidateSignatureBase64,
            publicKeyBase64: Self.candidatePublicKeyBase64,
            operatorKeyID: Self.candidateOperatorKeyID
        )
        return try RunAuthorizedInputs(
            worker: worker,
            baselineSourceManifest: baseline,
            candidateSourceManifest: candidate
        )
    }

    private func makeRunner() throws -> AdmittedFile {
        let url = caseRoot.appendingPathComponent("fastmlx-proof-runner")
        try Data("runner\n".utf8).write(to: url)
        return try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 64
        )
    }

    private func makeAuthorizedFile(
        name: String,
        payload: Data,
        payloadSHA256: String,
        purpose: OperatorAuthorizationPurpose,
        signatureBase64: String,
        publicKeyBase64: String,
        operatorKeyID: String
    ) throws -> OperatorAuthorizedFile {
        let url = caseRoot.appendingPathComponent(name)
        try payload.write(to: url)
        let admitted = try AdmittedFile.capture(
            absolutePath: url.path,
            maximumBytes: 64
        )
        let claim = try OperatorAuthorization.claimBytes(
            purpose: purpose,
            payloadSHA256: payloadSHA256,
            payloadByteCount: UInt64(payload.count)
        )
        return try OperatorAuthorization.verify(
            admittedFile: admitted,
            expectedPurpose: purpose,
            claimBytes: claim,
            signatureBase64: signatureBase64,
            publicKeyBase64: publicKeyBase64,
            allowedKeyID: operatorKeyID
        )
    }

    private func verify(
        _ claimBytes: Data = claimBytes,
        signatureBase64: String = signatureBase64,
        expectations: OperatorRunClaimAdmissionExpectations
    ) throws -> OperatorSignedRunClaim {
        try OperatorSignedRunClaimVerifier.verify(
            claimBytes: claimBytes,
            signatureBase64: signatureBase64,
            expectations: expectations
        )
    }

    private func replacing(_ source: String, with replacement: String) -> Data {
        let text = String(decoding: Self.claimBytes, as: UTF8.self)
        return Data(text.replacingOccurrences(of: source, with: replacement).utf8)
    }

    private static let differentSHA256 =
        "0000000000000000000000000000000000000000000000000000000000000001"
}
