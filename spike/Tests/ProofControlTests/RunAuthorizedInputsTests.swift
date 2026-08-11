import Darwin
import Foundation
import XCTest
@testable import ProofControl

final class RunAuthorizedInputsTests: XCTestCase {
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
        .appendingPathComponent("fast-mlx-run-inputs-\(UUID().uuidString)")
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

    func testAdmitsExactWorkerAndDistinctRoleBoundSourceSnapshots() throws {
        let worker = try makeSharedAuthorizedFile(
            name: "worker.swift",
            purpose: .workerBytes,
            signatureBase64: Self.workerSignatureBase64
        )
        let baseline = try makeSharedAuthorizedFile(
            name: "baseline-source.manifest",
            purpose: .sourceManifest,
            signatureBase64: Self.baselineSignatureBase64
        )
        let candidate = try makeCandidateSource()

        let inputs = try RunAuthorizedInputs(
            worker: worker,
            baselineSourceManifest: baseline,
            candidateSourceManifest: candidate
        )

        XCTAssertEqual(inputs.worker, worker)
        XCTAssertEqual(inputs.baseline.role, .baseline)
        XCTAssertEqual(inputs.baseline.sourceManifest, baseline)
        XCTAssertEqual(inputs.candidate.role, .candidate)
        XCTAssertEqual(inputs.candidate.sourceManifest, candidate)
        XCTAssertEqual(
            inputs.worker.authorizationID.rawValue,
            "805ffd1af7f4ff87dfb8619f3f78cb1ed46dfd47589fa01f9695d94651bca364"
        )
        XCTAssertEqual(
            inputs.baseline.sourceManifest.authorizationID.rawValue,
            "848cd6771d42ee71ccb6616eeb0a779b5c66f739398d88d350a66a55964538c9"
        )
        XCTAssertEqual(
            inputs.candidate.sourceManifest.authorizationID.rawValue,
            "ef374a55c02714b86150e35ab821b5c5be29617bae826b4d237849c19a477629"
        )
    }

    func testRejectsPurposeConfusionBeforeConstructingRunInputs() throws {
        let worker = try makeSharedAuthorizedFile(
            name: "worker.swift",
            purpose: .workerBytes,
            signatureBase64: Self.workerSignatureBase64
        )
        let baseline = try makeSharedAuthorizedFile(
            name: "baseline-source.manifest",
            purpose: .sourceManifest,
            signatureBase64: Self.baselineSignatureBase64
        )
        let candidate = try makeCandidateSource()

        let cases: [
            (
                worker: OperatorAuthorizedFile,
                baseline: OperatorAuthorizedFile,
                candidate: OperatorAuthorizedFile,
                expected: RunAuthorizedInputsError
            )
        ] = [
            (
                baseline,
                baseline,
                candidate,
                .unexpectedPurpose(
                    role: .worker,
                    expected: .workerBytes,
                    actual: .sourceManifest
                )
            ),
            (
                worker,
                worker,
                candidate,
                .unexpectedPurpose(
                    role: .baselineSource,
                    expected: .sourceManifest,
                    actual: .workerBytes
                )
            ),
            (
                worker,
                baseline,
                worker,
                .unexpectedPurpose(
                    role: .candidateSource,
                    expected: .sourceManifest,
                    actual: .workerBytes
                )
            ),
        ]

        for item in cases {
            XCTAssertThrowsError(
                try RunAuthorizedInputs(
                    worker: item.worker,
                    baselineSourceManifest: item.baseline,
                    candidateSourceManifest: item.candidate
                )
            ) { error in
                XCTAssertEqual(error as? RunAuthorizedInputsError, item.expected)
            }
        }
    }

    func testRejectsDuplicateBaselineAndCandidateAuthorization() throws {
        let worker = try makeSharedAuthorizedFile(
            name: "worker.swift",
            purpose: .workerBytes,
            signatureBase64: Self.workerSignatureBase64
        )
        let baseline = try makeSharedAuthorizedFile(
            name: "baseline-source.manifest",
            purpose: .sourceManifest,
            signatureBase64: Self.baselineSignatureBase64
        )

        XCTAssertThrowsError(
            try RunAuthorizedInputs(
                worker: worker,
                baselineSourceManifest: baseline,
                candidateSourceManifest: baseline
            )
        ) { error in
            XCTAssertEqual(
                error as? RunAuthorizedInputsError,
                .duplicateAuthorizationID(
                    first: .baselineSource,
                    second: .candidateSource
                )
            )
        }
    }

    func testRetainsCapturedSnapshotsAfterBackingFilesChange() throws {
        let worker = try makeSharedAuthorizedFile(
            name: "worker.swift",
            purpose: .workerBytes,
            signatureBase64: Self.workerSignatureBase64
        )
        let baseline = try makeSharedAuthorizedFile(
            name: "baseline-source.manifest",
            purpose: .sourceManifest,
            signatureBase64: Self.baselineSignatureBase64
        )
        let candidate = try makeCandidateSource()
        let inputs = try RunAuthorizedInputs(
            worker: worker,
            baselineSourceManifest: baseline,
            candidateSourceManifest: candidate
        )

        try Data("changed\n".utf8).write(
            to: caseRoot.appendingPathComponent("worker.swift")
        )
        try Data("changed\n".utf8).write(
            to: caseRoot.appendingPathComponent("baseline-source.manifest")
        )
        try Data("changed\n".utf8).write(
            to: caseRoot.appendingPathComponent("candidate-source.manifest")
        )

        XCTAssertEqual(inputs.worker.file.bytes, Self.sharedPayload)
        XCTAssertEqual(inputs.baseline.sourceManifest.file.bytes, Self.sharedPayload)
        XCTAssertEqual(inputs.candidate.sourceManifest.file.bytes, Self.candidatePayload)
    }

    private func makeSharedAuthorizedFile(
        name: String,
        purpose: OperatorAuthorizationPurpose,
        signatureBase64: String
    ) throws -> OperatorAuthorizedFile {
        try makeAuthorizedFile(
            name: name,
            payload: Self.sharedPayload,
            payloadSHA256: Self.sharedPayloadSHA256,
            purpose: purpose,
            signatureBase64: signatureBase64,
            publicKeyBase64: Self.sharedPublicKeyBase64,
            operatorKeyID: Self.sharedOperatorKeyID
        )
    }

    private func makeCandidateSource() throws -> OperatorAuthorizedFile {
        try makeAuthorizedFile(
            name: "candidate-source.manifest",
            payload: Self.candidatePayload,
            payloadSHA256: Self.candidatePayloadSHA256,
            purpose: .sourceManifest,
            signatureBase64: Self.candidateSignatureBase64,
            publicKeyBase64: Self.candidatePublicKeyBase64,
            operatorKeyID: Self.candidateOperatorKeyID
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
}
