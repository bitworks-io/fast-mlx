import CryptoKit
import Foundation

/// Qwen38 scorecard chain claim schema (chain Slice 1).
///
/// This is a NEW, independent claim schema and policy domain for the Qwen38
/// MTP scorecard chain. It intentionally does not share code, a domain
/// string, or a subject with the frozen absorbed-MLA
/// `OperatorSignedRunClaim` in `OperatorSignedRunClaim.swift`, which stays
/// locked to the single subject `absorbed-mla-loaded-result-pair`. See
/// docs/task-inbox/2026-09-01-qwen38-proof-runner-scope-and-chain-design.md
/// (binding review verdict) for the design this file implements.
///
/// Pure canonical-encoding helpers below (`CanonicalLineParser`,
/// `isLowercaseHex`, `isCanonicalToken`, `decodeCanonicalBase64`,
/// `sha256Hex`) are deliberately DUPLICATED from `OperatorSignedRunClaim
/// .swift` rather than extracted into a shared file, per binding item 4 of
/// the reviewed slice design: the frozen absorbed-MLA parser file is not
/// modified or refactored by this change.
///
/// Scope: this slice covers claim bytes, parsing, and structural field
/// verification only.
///   - No key-policy admission (that is chain Slice 2 — it will name a
///     trusted signer for this domain).
///   - No runner spawn/observe orchestration or signing APIs (chain Slice
///     3).
///   - No scorecard integration (chain Slice 4).
///   - No cryptographic signature verification anywhere in this file:
///     `Qwen38ScorecardRunClaimVerifier.verify` mirrors the ENVELOPE SHAPE
///     of the frozen design (claim bytes + a separate signature parameter,
///     producing a typed verified value with a claim digest and a
///     signature digest) but only checks that the signature is
///     canonically encoded 64 bytes — never that it is a valid signature
///     over the claim. There is no admitted operator key in this slice to
///     check it against.
public enum Qwen38ScorecardRunClaimSubject: String, Equatable, Sendable {
    case mtpScorecardResultPair = "qwen38-mtp-scorecard-result-pair"
}

/// The Qwen38 scorecard result pair is always GDN-on vs. GDN-off of the
/// SAME engine build (never two different reference engines — see binding
/// item 3), so both mode labels are pinned constants rather than
/// caller-chosen values. This mirrors how the frozen absorbed-MLA schema
/// pins `baseline`/`candidate` route and slot to fixed values per role.
public enum Qwen38ScorecardGDNMode: String, Equatable, Sendable {
    case on = "gdn-on"
    case off = "gdn-off"
}

public struct Qwen38ScorecardModelIdentity: Equatable, Sendable {
    public let modelID: String
    public let revision: String

    public init(modelID: String, revision: String) {
        self.modelID = modelID
        self.revision = revision
    }
}

/// Signed fields are limited to static digests and identity strings per
/// binding item 3 ("trim the claim, do not expand"): no result-output
/// -reservation digest, no run nonce, no engine-version field (subsumed by
/// the worker executable sha, which is minted by the Slice 3 runner — not
/// present here), no reference-engine identity (the pair is GDN-on/off of
/// the same engine), and no measured host/memory/thermal facts.
public struct Qwen38ScorecardRunClaimFields: Equatable, Sendable {
    public let subject: Qwen38ScorecardRunClaimSubject
    public let modelSHA256: String
    public let tokenizerSHA256: String
    public let tensorManifestSHA256: String
    public let chatTemplateSHA256: String
    public let quantizationIdentity: String
    public let target: Qwen38ScorecardModelIdentity
    public let drafter: Qwen38ScorecardModelIdentity
    public let sourceID: String
    public let hostAdmissionID: String
    public let harnessGitSHA1: String
    public let gdnOnMode: Qwen38ScorecardGDNMode
    public let gdnOffMode: Qwen38ScorecardGDNMode
    public let corpusID: String
    public let corpusContentSHA256: String
    public let resultPairID: String

    public init(
        subject: Qwen38ScorecardRunClaimSubject,
        modelSHA256: String,
        tokenizerSHA256: String,
        tensorManifestSHA256: String,
        chatTemplateSHA256: String,
        quantizationIdentity: String,
        target: Qwen38ScorecardModelIdentity,
        drafter: Qwen38ScorecardModelIdentity,
        sourceID: String,
        hostAdmissionID: String,
        harnessGitSHA1: String,
        gdnOnMode: Qwen38ScorecardGDNMode,
        gdnOffMode: Qwen38ScorecardGDNMode,
        corpusID: String,
        corpusContentSHA256: String,
        resultPairID: String
    ) {
        self.subject = subject
        self.modelSHA256 = modelSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.tensorManifestSHA256 = tensorManifestSHA256
        self.chatTemplateSHA256 = chatTemplateSHA256
        self.quantizationIdentity = quantizationIdentity
        self.target = target
        self.drafter = drafter
        self.sourceID = sourceID
        self.hostAdmissionID = hostAdmissionID
        self.harnessGitSHA1 = harnessGitSHA1
        self.gdnOnMode = gdnOnMode
        self.gdnOffMode = gdnOffMode
        self.corpusID = corpusID
        self.corpusContentSHA256 = corpusContentSHA256
        self.resultPairID = resultPairID
    }
}

public enum Qwen38ScorecardRunClaimError: Error, Equatable, Sendable {
    case nonCanonicalClaim
    case unexpectedSubject(
        expected: Qwen38ScorecardRunClaimSubject,
        actual: String
    )
    case invalidSignatureEncoding
}

extension Qwen38ScorecardRunClaimError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .nonCanonicalClaim:
            "qwen38 scorecard run claim is not the canonical fixed-order UTF-8 form"
        case .unexpectedSubject(let expected, let actual):
            "qwen38 scorecard run claim subject \(actual) does not match \(expected.rawValue)"
        case .invalidSignatureEncoding:
            "qwen38 scorecard run claim signature is not canonical base64 for 64 bytes"
        }
    }
}

/// Structurally validated claim only — this is NOT cryptographic proof of
/// authorship. A later slice (Slice 3) adds the Ed25519 check that would
/// make it one; until then, do not treat a value of this type as run
/// authorization.
public struct Qwen38ScorecardVerifiedRunClaim: Equatable, Sendable {
    public let subject: Qwen38ScorecardRunClaimSubject
    public let claimSHA256: String
    public let signatureSHA256: String

    let fields: Qwen38ScorecardRunClaimFields

    fileprivate init(
        fields: Qwen38ScorecardRunClaimFields,
        claimSHA256: String,
        signatureSHA256: String
    ) {
        self.subject = fields.subject
        self.claimSHA256 = claimSHA256
        self.signatureSHA256 = signatureSHA256
        self.fields = fields
    }
}

public enum Qwen38ScorecardRunClaimVerifier {
    public static let claimDomain =
        "fastmlx-qwen38-scorecard-run-claim-v1"

    public static func claimBytes(
        fields: Qwen38ScorecardRunClaimFields
    ) throws -> Data {
        try validate(fields)

        let lines = [
            claimDomain,
            "subject=\(fields.subject.rawValue)",
            "model_sha256=\(fields.modelSHA256)",
            "tokenizer_sha256=\(fields.tokenizerSHA256)",
            "tensor_manifest_sha256=\(fields.tensorManifestSHA256)",
            "chat_template_sha256=\(fields.chatTemplateSHA256)",
            "quantization_identity=\(fields.quantizationIdentity)",
            "target_model_id=\(fields.target.modelID)",
            "target_revision=\(fields.target.revision)",
            "drafter_model_id=\(fields.drafter.modelID)",
            "drafter_revision=\(fields.drafter.revision)",
            "source_id=\(fields.sourceID)",
            "host_admission_id=\(fields.hostAdmissionID)",
            "harness_git_sha1=\(fields.harnessGitSHA1)",
            "gdn_on_mode=\(fields.gdnOnMode.rawValue)",
            "gdn_off_mode=\(fields.gdnOffMode.rawValue)",
            "corpus_id=\(fields.corpusID)",
            "corpus_content_sha256=\(fields.corpusContentSHA256)",
            "result_pair_id=\(fields.resultPairID)",
        ]
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    /// Structural verification only: parses and field-validates the claim
    /// (fail-closed on any malformed, foreign-domain, or wrong-subject
    /// input) and confirms `signatureBase64` is canonically encoded 64
    /// bytes. Does NOT check the signature cryptographically. Slice 2 adds
    /// the key policy that names a trusted signer for this domain; Slice 3
    /// adds the real Ed25519 verification against it.
    public static func verify(
        claimBytes: Data,
        signatureBase64: String
    ) throws -> Qwen38ScorecardVerifiedRunClaim {
        let fields = try parseClaim(claimBytes)

        let signature = try decodeCanonicalBase64(
            signatureBase64,
            expectedByteCount: 64,
            error: .invalidSignatureEncoding
        )

        return Qwen38ScorecardVerifiedRunClaim(
            fields: fields,
            claimSHA256: sha256Hex(claimBytes),
            signatureSHA256: sha256Hex(signature)
        )
    }
}

private extension Qwen38ScorecardRunClaimVerifier {
    struct CanonicalLineParser {
        let lines: [Substring]
        var index = 0

        mutating func require(_ expected: String) throws {
            guard index < lines.count, lines[index] == Substring(expected) else {
                throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
            }
            index += 1
        }

        mutating func value(prefix: String) throws -> String {
            guard index < lines.count, lines[index].hasPrefix(prefix) else {
                throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
            }
            let value = String(lines[index].dropFirst(prefix.utf8.count))
            index += 1
            return value
        }

        mutating func sha256(prefix: String) throws -> String {
            let value = try value(prefix: prefix)
            guard isLowercaseHex(value, count: 64) else {
                throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
            }
            return value
        }

        mutating func sha1(prefix: String) throws -> String {
            let value = try value(prefix: prefix)
            guard isLowercaseHex(value, count: 40) else {
                throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
            }
            return value
        }

        mutating func token(prefix: String) throws -> String {
            let value = try value(prefix: prefix)
            guard isCanonicalToken(value) else {
                throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
            }
            return value
        }

        var isAtEnd: Bool {
            index == lines.count
        }
    }

    static func parseClaim(
        _ bytes: Data
    ) throws -> Qwen38ScorecardRunClaimFields {
        guard
            let text = String(data: bytes, encoding: .utf8),
            Data(text.utf8) == bytes
        else {
            throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
        }

        let splitLines = text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        )
        guard splitLines.count == 20, splitLines.last?.isEmpty == true else {
            throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
        }

        var parser = CanonicalLineParser(
            lines: Array(splitLines.dropLast())
        )
        try parser.require(claimDomain)
        let actualSubject = try parser.value(prefix: "subject=")
        guard
            actualSubject ==
                Qwen38ScorecardRunClaimSubject.mtpScorecardResultPair.rawValue
        else {
            throw Qwen38ScorecardRunClaimError.unexpectedSubject(
                expected: .mtpScorecardResultPair,
                actual: actualSubject
            )
        }

        let modelSHA256 = try parser.sha256(prefix: "model_sha256=")
        let tokenizerSHA256 = try parser.sha256(prefix: "tokenizer_sha256=")
        let tensorManifestSHA256 = try parser.sha256(
            prefix: "tensor_manifest_sha256="
        )
        let chatTemplateSHA256 = try parser.sha256(
            prefix: "chat_template_sha256="
        )
        let quantizationIdentity = try parser.token(
            prefix: "quantization_identity="
        )
        let targetModelID = try parser.token(prefix: "target_model_id=")
        let targetRevision = try parser.token(prefix: "target_revision=")
        let drafterModelID = try parser.token(prefix: "drafter_model_id=")
        let drafterRevision = try parser.token(prefix: "drafter_revision=")
        let sourceID = try parser.sha256(prefix: "source_id=")
        let hostAdmissionID = try parser.sha256(prefix: "host_admission_id=")
        let harnessGitSHA1 = try parser.sha1(prefix: "harness_git_sha1=")
        try parser.require("gdn_on_mode=\(Qwen38ScorecardGDNMode.on.rawValue)")
        try parser.require(
            "gdn_off_mode=\(Qwen38ScorecardGDNMode.off.rawValue)"
        )
        // The corpus identity is a frozen workload NAME (e.g.
        // "qwen38-27b-frozen-scorecard-workload-v2" in
        // Qwen38MTPPerformanceScorecardGate.requiredWorkload), not a digest;
        // its bytes are pinned separately by corpus_content_sha256.
        let corpusID = try parser.token(prefix: "corpus_id=")
        let corpusContentSHA256 = try parser.sha256(
            prefix: "corpus_content_sha256="
        )
        let resultPairID = try parser.sha256(prefix: "result_pair_id=")
        guard parser.isAtEnd else {
            throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
        }

        let fields = Qwen38ScorecardRunClaimFields(
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
        try validate(fields)
        return fields
    }

    static func validate(_ fields: Qwen38ScorecardRunClaimFields) throws {
        let sha256Values = [
            fields.modelSHA256,
            fields.tokenizerSHA256,
            fields.tensorManifestSHA256,
            fields.chatTemplateSHA256,
            fields.sourceID,
            fields.hostAdmissionID,
            fields.corpusContentSHA256,
            fields.resultPairID,
        ]
        guard sha256Values.allSatisfy({ isLowercaseHex($0, count: 64) }) else {
            throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
        }
        guard isLowercaseHex(fields.harnessGitSHA1, count: 40) else {
            throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
        }

        let tokens = [
            fields.quantizationIdentity,
            fields.target.modelID,
            fields.target.revision,
            fields.drafter.modelID,
            fields.drafter.revision,
            fields.corpusID,
        ]
        guard tokens.allSatisfy(isCanonicalToken) else {
            throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
        }

        guard fields.gdnOnMode == .on, fields.gdnOffMode == .off else {
            throw Qwen38ScorecardRunClaimError.nonCanonicalClaim
        }
    }

    static func decodeCanonicalBase64(
        _ value: String,
        expectedByteCount: Int,
        error: Qwen38ScorecardRunClaimError
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

    static func isLowercaseHex(_ value: String, count: Int) -> Bool {
        value.utf8.count == count && value.utf8.allSatisfy {
            (0x30...0x39).contains($0) || (0x61...0x66).contains($0)
        }
    }

    /// Opaque identifier/descriptor fields (HF-style model IDs, revisions,
    /// quantization identity) are not hashes, but the canonical byte format
    /// is line-oriented (split on `\n`, fields matched via `hasPrefix`), so
    /// an embedded newline would silently desynchronize line parsing and
    /// widen the claim's effective content beyond what was validated.
    /// Restrict to non-empty, length-bounded, visible-ASCII-only (no space,
    /// no control characters, no newline) so every accepted value is
    /// reproducible as a single canonical text line.
    static func isCanonicalToken(_ value: String) -> Bool {
        !value.isEmpty && value.utf8.count <= 200
            && value.utf8.allSatisfy { (0x21...0x7e).contains($0) }
    }

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
