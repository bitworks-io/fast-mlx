import CryptoKit
import Foundation

public enum OperatorRunClaimSubject: String, Equatable, Sendable {
    case absorbedMLALoadedResultPair =
        "absorbed-mla-loaded-result-pair"
}

public struct OperatorRunClaimByteIdentity: Equatable, Sendable {
    public let sha256: String
    public let byteCount: UInt64

    public init(sha256: String, byteCount: UInt64) {
        self.sha256 = sha256
        self.byteCount = byteCount
    }
}

public struct OperatorRunClaimAuthorizedPayloadReference:
    Equatable,
    Sendable
{
    public let authorizationID: String
    public let payload: OperatorRunClaimByteIdentity

    public init(
        authorizationID: String,
        payload: OperatorRunClaimByteIdentity
    ) {
        self.authorizationID = authorizationID
        self.payload = payload
    }
}

public struct OperatorRunClaimPolicyReferences: Equatable, Sendable {
    public let sourceSHA256: String
    public let dependencySHA256: String
    public let buildSHA256: String
    public let runtimeSHA256: String
    public let preflightSHA256: String
    public let publicationSHA256: String

    public init(
        sourceSHA256: String,
        dependencySHA256: String,
        buildSHA256: String,
        runtimeSHA256: String,
        preflightSHA256: String,
        publicationSHA256: String
    ) {
        self.sourceSHA256 = sourceSHA256
        self.dependencySHA256 = dependencySHA256
        self.buildSHA256 = buildSHA256
        self.runtimeSHA256 = runtimeSHA256
        self.preflightSHA256 = preflightSHA256
        self.publicationSHA256 = publicationSHA256
    }
}

public enum OperatorRunClaimSourceRole: String, Equatable, Sendable {
    case baseline
    case candidate
}

public enum OperatorRunClaimRoute: String, Equatable, Sendable {
    case decompressedDeepSeekV3 = "decompressed-deepseek-v3"
    case absorbedMLADeepSeekV3Explicit =
        "absorbed-mla-deepseek-v3-explicit"
}

public enum OperatorRunClaimSlot: String, Equatable, Sendable {
    case baseline
    case candidate
}

public struct OperatorRunClaimSourceReference: Equatable, Sendable {
    public let role: OperatorRunClaimSourceRole
    public let sourceManifest: OperatorRunClaimAuthorizedPayloadReference
    public let gitCommitSHA1: String
    public let gitTreeSHA1: String
    public let route: OperatorRunClaimRoute
    public let slot: OperatorRunClaimSlot
    public let buildReceiptID: String
    public let binary: OperatorRunClaimByteIdentity

    public init(
        role: OperatorRunClaimSourceRole,
        sourceManifest: OperatorRunClaimAuthorizedPayloadReference,
        gitCommitSHA1: String,
        gitTreeSHA1: String,
        route: OperatorRunClaimRoute,
        slot: OperatorRunClaimSlot,
        buildReceiptID: String,
        binary: OperatorRunClaimByteIdentity
    ) {
        self.role = role
        self.sourceManifest = sourceManifest
        self.gitCommitSHA1 = gitCommitSHA1
        self.gitTreeSHA1 = gitTreeSHA1
        self.route = route
        self.slot = slot
        self.buildReceiptID = buildReceiptID
        self.binary = binary
    }
}

public struct OperatorRunClaimFields: Equatable, Sendable {
    public let subject: OperatorRunClaimSubject
    public let operatorKeyID: String
    public let operatorKeyPolicySHA256: String
    public let hostAdmissionID: String
    public let runner: OperatorRunClaimByteIdentity
    public let worker: OperatorRunClaimAuthorizedPayloadReference
    public let policies: OperatorRunClaimPolicyReferences
    public let toolManifest: OperatorRunClaimByteIdentity
    public let baseline: OperatorRunClaimSourceReference
    public let candidate: OperatorRunClaimSourceReference
    public let model: OperatorRunClaimAuthorizedPayloadReference
    public let tokenizer: OperatorRunClaimAuthorizedPayloadReference
    public let workload: OperatorRunClaimAuthorizedPayloadReference
    public let resultPairID: String

    public init(
        subject: OperatorRunClaimSubject,
        operatorKeyID: String,
        operatorKeyPolicySHA256: String,
        hostAdmissionID: String,
        runner: OperatorRunClaimByteIdentity,
        worker: OperatorRunClaimAuthorizedPayloadReference,
        policies: OperatorRunClaimPolicyReferences,
        toolManifest: OperatorRunClaimByteIdentity,
        baseline: OperatorRunClaimSourceReference,
        candidate: OperatorRunClaimSourceReference,
        model: OperatorRunClaimAuthorizedPayloadReference,
        tokenizer: OperatorRunClaimAuthorizedPayloadReference,
        workload: OperatorRunClaimAuthorizedPayloadReference,
        resultPairID: String
    ) {
        self.subject = subject
        self.operatorKeyID = operatorKeyID
        self.operatorKeyPolicySHA256 = operatorKeyPolicySHA256
        self.hostAdmissionID = hostAdmissionID
        self.runner = runner
        self.worker = worker
        self.policies = policies
        self.toolManifest = toolManifest
        self.baseline = baseline
        self.candidate = candidate
        self.model = model
        self.tokenizer = tokenizer
        self.workload = workload
        self.resultPairID = resultPairID
    }
}

/// Exact comparison context only. None of these values grants run authority.
public struct OperatorRunClaimAdmissionExpectations:
    Equatable,
    Sendable
{
    public let keyPolicy: AdmittedOperatorKeyPolicy
    public let hostAdmissionID: String
    public let runner: AdmittedFile
    public let resultPairID: String
    public let inputs: RunAuthorizedInputs

    public init(
        keyPolicy: AdmittedOperatorKeyPolicy,
        hostAdmissionID: String,
        runner: AdmittedFile,
        resultPairID: String,
        inputs: RunAuthorizedInputs
    ) {
        self.keyPolicy = keyPolicy
        self.hostAdmissionID = hostAdmissionID
        self.runner = runner
        self.resultPairID = resultPairID
        self.inputs = inputs
    }
}

public enum OperatorRunClaimExpectationField:
    String,
    Equatable,
    Sendable
{
    case hostAdmissionID = "host-admission-id"
    case resultPairID = "result-pair-id"
}

public enum OperatorSignedRunClaimError: Error, Equatable, Sendable {
    case invalidExpectation(OperatorRunClaimExpectationField)
    case nonCanonicalClaim
    case unexpectedSubject(
        expected: OperatorRunClaimSubject,
        actual: String
    )
    case operatorKeyIDMismatch
    case operatorKeyPolicyMismatch
    case hostAdmissionIDMismatch
    case runnerDigestMismatch
    case runnerByteCountMismatch
    case resultPairIDMismatch
    case inputAuthorizationIDMismatch(role: RunAuthorizedInputRole)
    case inputDigestMismatch(role: RunAuthorizedInputRole)
    case inputByteCountMismatch(role: RunAuthorizedInputRole)
    case invalidPublicKeyEncoding
    case invalidSignatureEncoding
    case signatureRejected
}

extension OperatorSignedRunClaimError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .invalidExpectation(let field):
            "run claim expectation \(field.rawValue) is not canonical"
        case .nonCanonicalClaim:
            "run claim is not the canonical fixed-order UTF-8 form"
        case .unexpectedSubject(let expected, let actual):
            "run claim subject \(actual) does not match \(expected.rawValue)"
        case .operatorKeyIDMismatch:
            "run claim operator key ID does not match expected key ID"
        case .operatorKeyPolicyMismatch:
            "run claim operator key policy does not match expectation"
        case .hostAdmissionIDMismatch:
            "run claim host admission ID does not match expectation"
        case .runnerDigestMismatch:
            "run claim runner digest does not match captured runner"
        case .runnerByteCountMismatch:
            "run claim runner byte count does not match captured runner"
        case .resultPairIDMismatch:
            "run claim result-pair ID does not match expectation"
        case .inputAuthorizationIDMismatch(let role):
            "run claim \(role.rawValue) authorization ID does not match input"
        case .inputDigestMismatch(let role):
            "run claim \(role.rawValue) digest does not match input"
        case .inputByteCountMismatch(let role):
            "run claim \(role.rawValue) byte count does not match input"
        case .invalidPublicKeyEncoding:
            "run claim public key is not canonical base64 for 32 Ed25519 bytes"
        case .invalidSignatureEncoding:
            "run claim signature is not canonical base64 for 64 Ed25519 bytes"
        case .signatureRejected:
            "run claim signature is not valid for the canonical claim"
        }
    }
}

public struct OperatorSignedRunClaimID: Equatable, Hashable, Sendable {
    public let rawValue: String

    fileprivate init(rawValue: String) {
        self.rawValue = rawValue
    }
}

/// Cryptographically admitted evidence only; this is not run authorization.
public struct OperatorSignedRunClaim: Equatable, Sendable {
    public let claimID: OperatorSignedRunClaimID
    public let subject: OperatorRunClaimSubject
    public let operatorKeyID: String
    public let claimSHA256: String
    public let signatureSHA256: String

    let fields: OperatorRunClaimFields

    fileprivate init(
        claimID: OperatorSignedRunClaimID,
        fields: OperatorRunClaimFields,
        claimSHA256: String,
        signatureSHA256: String
    ) {
        self.claimID = claimID
        self.subject = fields.subject
        self.operatorKeyID = fields.operatorKeyID
        self.claimSHA256 = claimSHA256
        self.signatureSHA256 = signatureSHA256
        self.fields = fields
    }
}

public enum OperatorSignedRunClaimVerifier {
    public static let claimDomain =
        "fast-mlx-proof-control-run-claim-v1"
    public static let claimIDDomain =
        "fast-mlx-proof-control-run-claim-id-v1"

    public static func claimBytes(
        fields: OperatorRunClaimFields
    ) throws -> Data {
        try validate(fields)

        let lines = [
            claimDomain,
            "subject=\(fields.subject.rawValue)",
            "operator_key_id=\(fields.operatorKeyID)",
            "operator_key_policy_sha256=\(fields.operatorKeyPolicySHA256)",
            "host_admission_id=\(fields.hostAdmissionID)",
            "runner_sha256=\(fields.runner.sha256)",
            "runner_bytes=\(fields.runner.byteCount)",
            "worker_authorization_id=\(fields.worker.authorizationID)",
            "worker_sha256=\(fields.worker.payload.sha256)",
            "worker_bytes=\(fields.worker.payload.byteCount)",
            "source_policy_sha256=\(fields.policies.sourceSHA256)",
            "dependency_policy_sha256=\(fields.policies.dependencySHA256)",
            "build_policy_sha256=\(fields.policies.buildSHA256)",
            "runtime_policy_sha256=\(fields.policies.runtimeSHA256)",
            "preflight_policy_sha256=\(fields.policies.preflightSHA256)",
            "publication_policy_sha256=\(fields.policies.publicationSHA256)",
            "tool_manifest_sha256=\(fields.toolManifest.sha256)",
            "tool_manifest_bytes=\(fields.toolManifest.byteCount)",
            "baseline_source_authorization_id=\(fields.baseline.sourceManifest.authorizationID)",
            "baseline_source_manifest_sha256=\(fields.baseline.sourceManifest.payload.sha256)",
            "baseline_source_manifest_bytes=\(fields.baseline.sourceManifest.payload.byteCount)",
            "baseline_git_commit_sha1=\(fields.baseline.gitCommitSHA1)",
            "baseline_git_tree_sha1=\(fields.baseline.gitTreeSHA1)",
            "baseline_route=\(fields.baseline.route.rawValue)",
            "baseline_slot=\(fields.baseline.slot.rawValue)",
            "baseline_build_receipt_id=\(fields.baseline.buildReceiptID)",
            "baseline_binary_sha256=\(fields.baseline.binary.sha256)",
            "baseline_binary_bytes=\(fields.baseline.binary.byteCount)",
            "candidate_source_authorization_id=\(fields.candidate.sourceManifest.authorizationID)",
            "candidate_source_manifest_sha256=\(fields.candidate.sourceManifest.payload.sha256)",
            "candidate_source_manifest_bytes=\(fields.candidate.sourceManifest.payload.byteCount)",
            "candidate_git_commit_sha1=\(fields.candidate.gitCommitSHA1)",
            "candidate_git_tree_sha1=\(fields.candidate.gitTreeSHA1)",
            "candidate_route=\(fields.candidate.route.rawValue)",
            "candidate_slot=\(fields.candidate.slot.rawValue)",
            "candidate_build_receipt_id=\(fields.candidate.buildReceiptID)",
            "candidate_binary_sha256=\(fields.candidate.binary.sha256)",
            "candidate_binary_bytes=\(fields.candidate.binary.byteCount)",
            "model_authorization_id=\(fields.model.authorizationID)",
            "model_manifest_sha256=\(fields.model.payload.sha256)",
            "model_manifest_bytes=\(fields.model.payload.byteCount)",
            "tokenizer_authorization_id=\(fields.tokenizer.authorizationID)",
            "tokenizer_manifest_sha256=\(fields.tokenizer.payload.sha256)",
            "tokenizer_manifest_bytes=\(fields.tokenizer.payload.byteCount)",
            "workload_authorization_id=\(fields.workload.authorizationID)",
            "workload_sha256=\(fields.workload.payload.sha256)",
            "workload_bytes=\(fields.workload.payload.byteCount)",
            "result_pair_id=\(fields.resultPairID)",
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    public static func verify(
        claimBytes: Data,
        signatureBase64: String,
        expectations: OperatorRunClaimAdmissionExpectations
    ) throws -> OperatorSignedRunClaim {
        try validate(expectations)
        let fields = try parseClaim(
            claimBytes,
            expectedSubject:
                expectations.keyPolicy.allowedClaimSubject
        )
        try match(fields, expectations: expectations)

        let publicKeyBytes = try decodeCanonicalBase64(
            expectations.keyPolicy.activeOperatorPublicKeyBase64,
            expectedByteCount: 32,
            error: .invalidPublicKeyEncoding
        )

        let publicKey: Curve25519.Signing.PublicKey
        do {
            publicKey = try Curve25519.Signing.PublicKey(
                rawRepresentation: publicKeyBytes
            )
        } catch {
            throw OperatorSignedRunClaimError.invalidPublicKeyEncoding
        }

        let signature = try decodeCanonicalBase64(
            signatureBase64,
            expectedByteCount: 64,
            error: .invalidSignatureEncoding
        )
        guard publicKey.isValidSignature(signature, for: claimBytes) else {
            throw OperatorSignedRunClaimError.signatureRejected
        }

        let claimSHA256 = sha256Hex(claimBytes)
        let signatureSHA256 = sha256Hex(signature)
        return OperatorSignedRunClaim(
            claimID: OperatorSignedRunClaimID(
                rawValue: claimID(
                    operatorKeyID: fields.operatorKeyID,
                    claimSHA256: claimSHA256,
                    signatureSHA256: signatureSHA256
                )
            ),
            fields: fields,
            claimSHA256: claimSHA256,
            signatureSHA256: signatureSHA256
        )
    }
}

private extension OperatorSignedRunClaimVerifier {
    struct CanonicalLineParser {
        let lines: [Substring]
        var index = 0

        mutating func require(_ expected: String) throws {
            guard index < lines.count, lines[index] == Substring(expected) else {
                throw OperatorSignedRunClaimError.nonCanonicalClaim
            }
            index += 1
        }

        mutating func value(prefix: String) throws -> String {
            guard index < lines.count, lines[index].hasPrefix(prefix) else {
                throw OperatorSignedRunClaimError.nonCanonicalClaim
            }
            let value = String(lines[index].dropFirst(prefix.utf8.count))
            index += 1
            return value
        }

        mutating func sha256(prefix: String) throws -> String {
            let value = try value(prefix: prefix)
            guard isLowercaseHex(value, count: 64) else {
                throw OperatorSignedRunClaimError.nonCanonicalClaim
            }
            return value
        }

        mutating func sha1(prefix: String) throws -> String {
            let value = try value(prefix: prefix)
            guard isLowercaseHex(value, count: 40) else {
                throw OperatorSignedRunClaimError.nonCanonicalClaim
            }
            return value
        }

        mutating func byteCount(prefix: String) throws -> UInt64 {
            let value = try value(prefix: prefix)
            guard
                isCanonicalDecimal(value),
                let result = UInt64(value)
            else {
                throw OperatorSignedRunClaimError.nonCanonicalClaim
            }
            return result
        }

        var isAtEnd: Bool {
            index == lines.count
        }
    }

    static func parseClaim(
        _ bytes: Data,
        expectedSubject: OperatorRunClaimSubject
    ) throws -> OperatorRunClaimFields {
        guard
            let text = String(data: bytes, encoding: .utf8),
            Data(text.utf8) == bytes
        else {
            throw OperatorSignedRunClaimError.nonCanonicalClaim
        }

        let splitLines = text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        )
        guard splitLines.count == 49, splitLines.last?.isEmpty == true else {
            throw OperatorSignedRunClaimError.nonCanonicalClaim
        }

        var parser = CanonicalLineParser(
            lines: Array(splitLines.dropLast())
        )
        try parser.require(claimDomain)
        let actualSubject = try parser.value(prefix: "subject=")
        guard actualSubject == expectedSubject.rawValue else {
            throw OperatorSignedRunClaimError.unexpectedSubject(
                expected: expectedSubject,
                actual: actualSubject
            )
        }

        let operatorKeyID = try parser.sha256(
            prefix: "operator_key_id="
        )
        let operatorKeyPolicySHA256 = try parser.sha256(
            prefix: "operator_key_policy_sha256="
        )
        let hostAdmissionID = try parser.sha256(
            prefix: "host_admission_id="
        )
        let runner = OperatorRunClaimByteIdentity(
            sha256: try parser.sha256(prefix: "runner_sha256="),
            byteCount: try parser.byteCount(prefix: "runner_bytes=")
        )
        let worker = OperatorRunClaimAuthorizedPayloadReference(
            authorizationID: try parser.sha256(
                prefix: "worker_authorization_id="
            ),
            payload: OperatorRunClaimByteIdentity(
                sha256: try parser.sha256(prefix: "worker_sha256="),
                byteCount: try parser.byteCount(prefix: "worker_bytes=")
            )
        )
        let policies = OperatorRunClaimPolicyReferences(
            sourceSHA256: try parser.sha256(
                prefix: "source_policy_sha256="
            ),
            dependencySHA256: try parser.sha256(
                prefix: "dependency_policy_sha256="
            ),
            buildSHA256: try parser.sha256(
                prefix: "build_policy_sha256="
            ),
            runtimeSHA256: try parser.sha256(
                prefix: "runtime_policy_sha256="
            ),
            preflightSHA256: try parser.sha256(
                prefix: "preflight_policy_sha256="
            ),
            publicationSHA256: try parser.sha256(
                prefix: "publication_policy_sha256="
            )
        )
        let toolManifest = OperatorRunClaimByteIdentity(
            sha256: try parser.sha256(
                prefix: "tool_manifest_sha256="
            ),
            byteCount: try parser.byteCount(
                prefix: "tool_manifest_bytes="
            )
        )
        let baseline = try parseSource(
            parser: &parser,
            role: .baseline
        )
        let candidate = try parseSource(
            parser: &parser,
            role: .candidate
        )
        let model = try parseAuthorizedPayload(
            parser: &parser,
            authorizationPrefix: "model_authorization_id=",
            digestPrefix: "model_manifest_sha256=",
            byteCountPrefix: "model_manifest_bytes="
        )
        let tokenizer = try parseAuthorizedPayload(
            parser: &parser,
            authorizationPrefix: "tokenizer_authorization_id=",
            digestPrefix: "tokenizer_manifest_sha256=",
            byteCountPrefix: "tokenizer_manifest_bytes="
        )
        let workload = try parseAuthorizedPayload(
            parser: &parser,
            authorizationPrefix: "workload_authorization_id=",
            digestPrefix: "workload_sha256=",
            byteCountPrefix: "workload_bytes="
        )
        let resultPairID = try parser.sha256(
            prefix: "result_pair_id="
        )
        guard parser.isAtEnd else {
            throw OperatorSignedRunClaimError.nonCanonicalClaim
        }

        let fields = OperatorRunClaimFields(
            subject: expectedSubject,
            operatorKeyID: operatorKeyID,
            operatorKeyPolicySHA256: operatorKeyPolicySHA256,
            hostAdmissionID: hostAdmissionID,
            runner: runner,
            worker: worker,
            policies: policies,
            toolManifest: toolManifest,
            baseline: baseline,
            candidate: candidate,
            model: model,
            tokenizer: tokenizer,
            workload: workload,
            resultPairID: resultPairID
        )
        try validate(fields)
        return fields
    }

    static func parseSource(
        parser: inout CanonicalLineParser,
        role: OperatorRunClaimSourceRole
    ) throws -> OperatorRunClaimSourceReference {
        let prefix = role.rawValue
        let sourceManifest = try parseAuthorizedPayload(
            parser: &parser,
            authorizationPrefix: "\(prefix)_source_authorization_id=",
            digestPrefix: "\(prefix)_source_manifest_sha256=",
            byteCountPrefix: "\(prefix)_source_manifest_bytes="
        )
        let gitCommitSHA1 = try parser.sha1(
            prefix: "\(prefix)_git_commit_sha1="
        )
        let gitTreeSHA1 = try parser.sha1(
            prefix: "\(prefix)_git_tree_sha1="
        )

        let route: OperatorRunClaimRoute
        let slot: OperatorRunClaimSlot
        switch role {
        case .baseline:
            route = .decompressedDeepSeekV3
            slot = .baseline
        case .candidate:
            route = .absorbedMLADeepSeekV3Explicit
            slot = .candidate
        }
        try parser.require("\(prefix)_route=\(route.rawValue)")
        try parser.require("\(prefix)_slot=\(slot.rawValue)")

        return OperatorRunClaimSourceReference(
            role: role,
            sourceManifest: sourceManifest,
            gitCommitSHA1: gitCommitSHA1,
            gitTreeSHA1: gitTreeSHA1,
            route: route,
            slot: slot,
            buildReceiptID: try parser.sha256(
                prefix: "\(prefix)_build_receipt_id="
            ),
            binary: OperatorRunClaimByteIdentity(
                sha256: try parser.sha256(
                    prefix: "\(prefix)_binary_sha256="
                ),
                byteCount: try parser.byteCount(
                    prefix: "\(prefix)_binary_bytes="
                )
            )
        )
    }

    static func parseAuthorizedPayload(
        parser: inout CanonicalLineParser,
        authorizationPrefix: String,
        digestPrefix: String,
        byteCountPrefix: String
    ) throws -> OperatorRunClaimAuthorizedPayloadReference {
        OperatorRunClaimAuthorizedPayloadReference(
            authorizationID: try parser.sha256(
                prefix: authorizationPrefix
            ),
            payload: OperatorRunClaimByteIdentity(
                sha256: try parser.sha256(prefix: digestPrefix),
                byteCount: try parser.byteCount(
                    prefix: byteCountPrefix
                )
            )
        )
    }

    static func validate(_ fields: OperatorRunClaimFields) throws {
        let sha256Values = [
            fields.operatorKeyID,
            fields.operatorKeyPolicySHA256,
            fields.hostAdmissionID,
            fields.runner.sha256,
            fields.worker.authorizationID,
            fields.worker.payload.sha256,
            fields.policies.sourceSHA256,
            fields.policies.dependencySHA256,
            fields.policies.buildSHA256,
            fields.policies.runtimeSHA256,
            fields.policies.preflightSHA256,
            fields.policies.publicationSHA256,
            fields.toolManifest.sha256,
            fields.baseline.sourceManifest.authorizationID,
            fields.baseline.sourceManifest.payload.sha256,
            fields.baseline.buildReceiptID,
            fields.baseline.binary.sha256,
            fields.candidate.sourceManifest.authorizationID,
            fields.candidate.sourceManifest.payload.sha256,
            fields.candidate.buildReceiptID,
            fields.candidate.binary.sha256,
            fields.model.authorizationID,
            fields.model.payload.sha256,
            fields.tokenizer.authorizationID,
            fields.tokenizer.payload.sha256,
            fields.workload.authorizationID,
            fields.workload.payload.sha256,
            fields.resultPairID,
        ]
        guard sha256Values.allSatisfy({
            isLowercaseHex($0, count: 64)
        }) else {
            throw OperatorSignedRunClaimError.nonCanonicalClaim
        }
        guard
            isLowercaseHex(fields.baseline.gitCommitSHA1, count: 40),
            isLowercaseHex(fields.baseline.gitTreeSHA1, count: 40),
            isLowercaseHex(fields.candidate.gitCommitSHA1, count: 40),
            isLowercaseHex(fields.candidate.gitTreeSHA1, count: 40),
            fields.baseline.role == .baseline,
            fields.baseline.route == .decompressedDeepSeekV3,
            fields.baseline.slot == .baseline,
            fields.candidate.role == .candidate,
            fields.candidate.route == .absorbedMLADeepSeekV3Explicit,
            fields.candidate.slot == .candidate
        else {
            throw OperatorSignedRunClaimError.nonCanonicalClaim
        }

        let authorizationIDs = [
            fields.worker.authorizationID,
            fields.baseline.sourceManifest.authorizationID,
            fields.candidate.sourceManifest.authorizationID,
            fields.model.authorizationID,
            fields.tokenizer.authorizationID,
            fields.workload.authorizationID,
        ]
        guard Set(authorizationIDs).count == authorizationIDs.count else {
            throw OperatorSignedRunClaimError.nonCanonicalClaim
        }
    }

    static func validate(
        _ expectations: OperatorRunClaimAdmissionExpectations
    ) throws {
        let fields: [
            (String, OperatorRunClaimExpectationField)
        ] = [
            (
                expectations.hostAdmissionID,
                .hostAdmissionID
            ),
            (
                expectations.resultPairID,
                .resultPairID
            ),
        ]
        for (value, field) in fields where
            !isLowercaseHex(value, count: 64)
        {
            throw OperatorSignedRunClaimError.invalidExpectation(field)
        }
    }

    static func match(
        _ fields: OperatorRunClaimFields,
        expectations: OperatorRunClaimAdmissionExpectations
    ) throws {
        guard fields.operatorKeyID ==
            expectations.keyPolicy.activeOperatorKeyID
        else {
            throw OperatorSignedRunClaimError.operatorKeyIDMismatch
        }
        guard fields.operatorKeyPolicySHA256 ==
            expectations.keyPolicy.policySHA256
        else {
            throw OperatorSignedRunClaimError.operatorKeyPolicyMismatch
        }
        guard fields.hostAdmissionID == expectations.hostAdmissionID else {
            throw OperatorSignedRunClaimError.hostAdmissionIDMismatch
        }
        guard fields.runner.sha256 == expectations.runner.sha256 else {
            throw OperatorSignedRunClaimError.runnerDigestMismatch
        }
        guard fields.runner.byteCount ==
            UInt64(expectations.runner.bytes.count)
        else {
            throw OperatorSignedRunClaimError.runnerByteCountMismatch
        }

        try match(
            fields.worker,
            input: expectations.inputs.worker,
            role: .worker
        )
        try match(
            fields.baseline.sourceManifest,
            input: expectations.inputs.baseline.sourceManifest,
            role: .baselineSource
        )
        try match(
            fields.candidate.sourceManifest,
            input: expectations.inputs.candidate.sourceManifest,
            role: .candidateSource
        )

        guard fields.resultPairID == expectations.resultPairID else {
            throw OperatorSignedRunClaimError.resultPairIDMismatch
        }
    }

    static func match(
        _ reference: OperatorRunClaimAuthorizedPayloadReference,
        input: OperatorAuthorizedFile,
        role: RunAuthorizedInputRole
    ) throws {
        guard reference.authorizationID ==
            input.authorizationID.rawValue
        else {
            throw OperatorSignedRunClaimError
                .inputAuthorizationIDMismatch(role: role)
        }
        guard reference.payload.sha256 == input.file.sha256 else {
            throw OperatorSignedRunClaimError
                .inputDigestMismatch(role: role)
        }
        guard reference.payload.byteCount ==
            UInt64(input.file.bytes.count)
        else {
            throw OperatorSignedRunClaimError
                .inputByteCountMismatch(role: role)
        }
    }

    static func decodeCanonicalBase64(
        _ value: String,
        expectedByteCount: Int,
        error: OperatorSignedRunClaimError
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

    static func claimID(
        operatorKeyID: String,
        claimSHA256: String,
        signatureSHA256: String
    ) -> String {
        let lines = [
            claimIDDomain,
            "operator_key_id=\(operatorKeyID)",
            "claim_sha256=\(claimSHA256)",
            "signature_sha256=\(signatureSHA256)",
        ]
        return sha256Hex(Data((lines.joined(separator: "\n") + "\n").utf8))
    }

    static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        value.utf8.count == count && value.utf8.allSatisfy {
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

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
