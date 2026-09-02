import Darwin
import Foundation
import HarnessCore
import ProofControl
import ScorecardPairControl

/// Chain Slice 4b (increment C): the trusted-parent pipeline pieces of the
/// Qwen38 scorecard proof runner. Everything here is deliberately free of
/// signing capability (the structural no-self-sign gate scans this target)
/// and free of caller-suppliable identity: worker identity inputs exist
/// only inside the operator-signed claim, resolved through ProofControl.
///
/// Trust-boundary notes (design verdict P7):
/// - The status JSON on STDOUT is the runner's durable audit witness: the
///   launch observations are Encodable-only, so the published status line
///   carries the claim SHA-256, the authorization ID (a signature
///   instance), and BOTH runner-minted process-isolation evidence IDs.
/// - The invoker chooses the trust anchor; `promotable:true` is only as
///   trustworthy as that anchor, so promotion review must check the anchor
///   file, not just the flag.
/// - The worker resolves its own harness git SHA from the runner's cwd
///   (live-git walk-up or `./.harness-sha`), so the runner preflights that
///   a deploy-tree root marker is reachable and fails with a typed error
///   instead of a distant evidence-ID mismatch.
enum Qwen38ScorecardRunnerError: Error, Equatable, CustomStringConvertible {
    case unknownFlag(String)
    case duplicateFlag(String)
    case missingValue(String)
    case missingFlag(String)
    case unexpectedPositional
    case invalidUnsignedInteger(String)
    case invalidHostUseClassification
    case deployTreeRootNotFound(String)
    case trustAnchorNotObject
    case trustAnchorFieldMissing(String)
    case trustAnchorFieldInvalid(String)
    case signatureFileNotSingleLine(String)
    case claimSourceIdentityMismatch

    var description: String {
        switch self {
        case .unknownFlag(let flag):
            return "unknown flag \(flag)"
        case .duplicateFlag(let flag):
            return "duplicate flag \(flag)"
        case .missingValue(let flag):
            return "missing value for \(flag)"
        case .missingFlag(let flag):
            return "missing required flag \(flag)"
        case .unexpectedPositional:
            return "unexpected positional argument"
        case .invalidUnsignedInteger(let flag):
            return "value for \(flag) is not an unsigned integer"
        case .invalidHostUseClassification:
            return "--host-use must be dedicated-serving and "
                + "--host-use-source must be operator-assertion"
        case .deployTreeRootNotFound(let cwd):
            return "no .harness-sha or .git reachable from \(cwd); "
                + "invoke the runner from the deploy tree root"
        case .trustAnchorNotObject:
            return "trust anchor is not a JSON object"
        case .trustAnchorFieldMissing(let field):
            return "trust anchor field missing: \(field)"
        case .trustAnchorFieldInvalid(let field):
            return "trust anchor field invalid: \(field)"
        case .signatureFileNotSingleLine(let label):
            return "\(label) signature file must be a single "
                + "newline-terminated base64 line"
        case .claimSourceIdentityMismatch:
            return "claim source_id does not match the harness source identity"
        }
    }
}

/// Input-file byte bounds for the hardened `AdmittedFile.capture` reads.
/// Anchors, policies, claims, and signatures are small canonical text
/// documents; anything larger is refused before parsing.
enum Qwen38ScorecardRunnerInputBounds {
    static let trustAnchorMaximumBytes = 16 * 1024
    static let policyMaximumBytes = 64 * 1024
    static let claimMaximumBytes = 64 * 1024
    static let signatureMaximumBytes = 4 * 1024
}

struct Qwen38ScorecardRunnerArguments: Equatable, Sendable {
    let trustAnchorPath: String
    let policyPath: String
    let policySignaturePath: String
    let claimPath: String
    let claimSignaturePath: String
    let workerExecutablePath: String
    let run: Qwen38MTPScorecardLiveRunArguments
}

func parseQwen38ScorecardRunnerArguments(
    _ arguments: [String]
) throws -> Qwen38ScorecardRunnerArguments {
    let allowed = Set([
        "--trust-anchor",
        "--policy",
        "--policy-signature",
        "--claim",
        "--claim-signature",
        "--worker",
        "--target",
        "--drafter",
        "--output",
        "--authority-output",
        "--host-use",
        "--host-use-source",
        "--expected-chip",
        "--memory-limit-bytes",
        "--cache-limit-bytes",
        "--reserved-kv-bytes",
        "--reserved-io-bytes",
        "--reserved-prefetch-bytes",
        "--os-service-reserve-bytes",
    ])
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw Qwen38ScorecardRunnerError.unexpectedPositional
        }
        guard allowed.contains(flag) else {
            throw Qwen38ScorecardRunnerError.unknownFlag(flag)
        }
        guard values[flag] == nil else {
            throw Qwen38ScorecardRunnerError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count,
            !arguments[index + 1].hasPrefix("--")
        else {
            throw Qwen38ScorecardRunnerError.missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }

    func require(_ flag: String) throws -> String {
        guard let value = values[flag], !value.isEmpty else {
            throw Qwen38ScorecardRunnerError.missingFlag(flag)
        }
        return value
    }
    // Mirrors the adapter parser's bounds (security review F1): zero and
    // Int.max-exceeding byte values fail with a typed error here instead
    // of trapping later in the budget's checked arithmetic.
    func requireUInt64(_ flag: String) throws -> UInt64 {
        guard let value = UInt64(try require(flag)),
            value > 0,
            value <= UInt64(Int.max)
        else {
            throw Qwen38ScorecardRunnerError.invalidUnsignedInteger(flag)
        }
        return value
    }

    // Trust inputs are validated first so a missing trust flag is always
    // the first reported error.
    let trustAnchorPath = try require("--trust-anchor")
    let policyPath = try require("--policy")
    let policySignaturePath = try require("--policy-signature")
    let claimPath = try require("--claim")
    let claimSignaturePath = try require("--claim-signature")
    let workerExecutablePath = try require("--worker")

    // Mirrors the adapter's host-use pins (security review F5): a wrong
    // classification fails fast here rather than deep in the worker
    // handshake; the workers still re-pin these fail-closed.
    let hostUse = try require("--host-use")
    let hostUseSource = try require("--host-use-source")
    guard hostUse == "dedicated-serving",
        hostUseSource == "operator-assertion"
    else {
        throw Qwen38ScorecardRunnerError.invalidHostUseClassification
    }

    return Qwen38ScorecardRunnerArguments(
        trustAnchorPath: trustAnchorPath,
        policyPath: policyPath,
        policySignaturePath: policySignaturePath,
        claimPath: claimPath,
        claimSignaturePath: claimSignaturePath,
        workerExecutablePath: workerExecutablePath,
        run: Qwen38MTPScorecardLiveRunArguments(
            targetPath: try require("--target"),
            drafterPath: try require("--drafter"),
            outputPath: try require("--output"),
            authorityOutputPath: try require("--authority-output"),
            hostUse: hostUse,
            hostUseSource: hostUseSource,
            expectedChip: try require("--expected-chip"),
            memoryBudget: try Qwen38MTPScorecardLiveMemoryBudget(
                memoryLimitBytes: try requireUInt64("--memory-limit-bytes"),
                cacheLimitBytes: try requireUInt64("--cache-limit-bytes"),
                reservedKVBytes: try requireUInt64("--reserved-kv-bytes"),
                reservedIOBytes: try requireUInt64("--reserved-io-bytes"),
                reservedPrefetchBytes: try requireUInt64(
                    "--reserved-prefetch-bytes"),
                osServiceReserveBytes: try requireUInt64(
                    "--os-service-reserve-bytes"))))
}

/// Parses the invoker-chosen trust-anchor JSON document; the verification
/// time is deliberately NOT a document field — the runner mints it from
/// its own clock so an anchor file cannot pin verification into a
/// policy-validity window it no longer occupies.
func parseQwen38ScorecardRunnerTrustAnchor(
    _ bytes: Data,
    verificationUnixSeconds: UInt64
) throws -> Qwen38ScorecardKeyPolicyTrustAnchor {
    let decoded = try? JSONSerialization.jsonObject(with: bytes)
    guard let object = decoded as? [String: Any] else {
        throw Qwen38ScorecardRunnerError.trustAnchorNotObject
    }
    func requireString(_ field: String) throws -> String {
        guard let raw = object[field] else {
            throw Qwen38ScorecardRunnerError.trustAnchorFieldMissing(field)
        }
        guard let value = raw as? String, !value.isEmpty else {
            throw Qwen38ScorecardRunnerError.trustAnchorFieldInvalid(field)
        }
        return value
    }
    func requireUInt64(_ field: String) throws -> UInt64 {
        guard let raw = object[field] else {
            throw Qwen38ScorecardRunnerError.trustAnchorFieldMissing(field)
        }
        guard let number = raw as? NSNumber,
            number.int64Value >= 0
        else {
            throw Qwen38ScorecardRunnerError.trustAnchorFieldInvalid(field)
        }
        return number.uint64Value
    }
    return Qwen38ScorecardKeyPolicyTrustAnchor(
        rootPublicKeyBase64: try requireString("root_public_key_base64"),
        rootKeyID: try requireString("root_key_id"),
        expectedCurrentPolicySHA256: try requireString(
            "expected_current_policy_sha256"),
        minimumPolicyGeneration: try requireUInt64(
            "minimum_policy_generation"),
        verificationUnixSeconds: verificationUnixSeconds)
}

/// A signature input is exactly one base64 line with a trailing newline.
func parseQwen38ScorecardRunnerSignatureLine(
    _ bytes: Data,
    label: String
) throws -> String {
    guard let text = String(data: bytes, encoding: .utf8),
        text.hasSuffix("\n")
    else {
        throw Qwen38ScorecardRunnerError.signatureFileNotSingleLine(label)
    }
    let line = String(text.dropLast())
    guard !line.isEmpty, !line.contains("\n") else {
        throw Qwen38ScorecardRunnerError.signatureFileNotSingleLine(label)
    }
    return line
}

/// P7 rider (c): the worker resolves its harness git SHA from the
/// inherited cwd, so require a deploy-tree root marker before spawning.
func qwen38ScorecardRunnerPreflightDeployTreeRoot(
    from directory: String = FileManager.default.currentDirectoryPath
) throws {
    var current = URL(fileURLWithPath: directory).standardizedFileURL
    let fileManager = FileManager.default
    for _ in 0 ..< 64 {
        let marker = current.appendingPathComponent(".harness-sha").path
        let git = current.appendingPathComponent(".git").path
        if fileManager.fileExists(atPath: marker)
            || fileManager.fileExists(atPath: git)
        {
            return
        }
        let parent = current.deletingLastPathComponent()
        guard parent.path != current.path else { break }
        current = parent
    }
    throw Qwen38ScorecardRunnerError.deployTreeRootNotFound(directory)
}

/// Runner-collected host facts, mirroring the worker-side string forms so
/// host digests stay comparable across historical records.
struct Qwen38ScorecardRunnerHostFacts: Equatable, Sendable {
    let chipBrand: String
    let ramBytes: UInt64
    let osBuild: String

    static func collect() -> Qwen38ScorecardRunnerHostFacts {
        Qwen38ScorecardRunnerHostFacts(
            chipBrand: sysctlChipBrand(),
            ramBytes: ProcessInfo.processInfo.physicalMemory,
            osBuild: {
                let v = ProcessInfo.processInfo.operatingSystemVersion
                return "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
            }())
    }

    private static func sysctlChipBrand() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return "unknown" }
        var buffer = [CChar](repeating: 0, count: size)
        let result = sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
        guard result == 0 else { return "unknown" }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
            .trimmingCharacters(in: .whitespaces)
    }
}

/// Design verdict P5: the run identity's model/corpus digests and the
/// harness git SHA are minted FROM THE OPERATOR-SIGNED CLAIM — the gate
/// pins them against its own constants in `validateAuthority`, so a claim
/// signing wrong digests fails closed instead of the signed digests being
/// decorative. Host facts come from the runner's own kernel;
/// `candidateMLXSwiftVersion` comes from the shared pinned constant.
func qwen38ScorecardRunnerTrustedRunIdentity(
    authorization: Qwen38ScorecardResolvedRunAuthorization,
    hostFacts: Qwen38ScorecardRunnerHostFacts
) -> Qwen38MTPPerformanceScorecardTrustedRunIdentity {
    Qwen38MTPPerformanceScorecardTrustedRunIdentity(
        measurementClass: Qwen38MTPPerformanceScorecardGate.measurementClass,
        hardwareChip: hostFacts.chipBrand,
        hardwareRAMBytes: hostFacts.ramBytes,
        hardwareOSBuild: hostFacts.osBuild,
        hostIdentityDigest: qwen38MTPScorecardSHA256Hex(
            Data(
                "\(hostFacts.chipBrand):\(hostFacts.ramBytes):\(hostFacts.osBuild)"
                    .utf8)),
        harnessGitSHA: authorization.authorizedHarnessGitSHA1,
        candidateMLXSwiftVersion: ScorecardPairControlVersions.mlxSwiftVersion,
        referenceMLXVersion: nil,
        referenceMLXLMVersion: nil,
        modelLabel: Qwen38MTPPerformanceScorecardGate.modelArtifactLabel,
        modelConfigHash: authorization.authorizedModelConfigSHA256,
        modelCheckpointManifestHash: authorization.authorizedTensorManifestSHA256,
        modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
        corpusID: authorization.authorizedCorpusID,
        corpusContentHash: authorization.authorizedCorpusContentSHA256)
}

/// The per-run nonce is minted post-signing by the trusted parent (design
/// note N6): replay of a signed claim is by design, and each run is
/// distinguished by this provenance nonce plus the authorization ID.
func qwen38ScorecardRunnerProvenance(
    identity: Qwen38MTPPerformanceScorecardTrustedRunIdentity
) -> Provenance {
    Provenance(
        date: ISO8601DateFormatter().string(from: Date()),
        hardwareChip: identity.hardwareChip,
        hardwareRAMBytes: identity.hardwareRAMBytes,
        hardwareOS: identity.hardwareOSBuild,
        harnessGitSHA: identity.harnessGitSHA,
        mlxSwiftVersion: identity.candidateMLXSwiftVersion,
        referenceMLXVersion: identity.referenceMLXVersion,
        referenceMLXLMVersion: identity.referenceMLXLMVersion,
        modelPath: identity.modelLabel,
        modelConfigHash: identity.modelConfigHash,
        modelCheckpointManifestHash: identity.modelCheckpointManifestHash,
        modelQuant: identity.modelQuant,
        corpusId: identity.corpusID,
        corpusContentHash: identity.corpusContentHash,
        nonce: UUID().uuidString)
}

/// The one durable audit witness of a runner invocation (design verdict
/// P7): emitted as a single sorted-keys JSON line on STDOUT for both
/// success and failure. Encodable-only observations mean these IDs appear
/// nowhere else durable.
struct Qwen38ScorecardRunnerStatus: Encodable, Equatable, Sendable {
    let schema: String
    let program: String
    let status: String
    let promotable: Bool
    let qualified: Bool?
    let claimSHA256: String?
    let authorizationID: String?
    let operatorKeyID: String?
    let policyAdmissionID: String?
    let candidateProcessIsolationEvidenceID: String?
    let referenceProcessIsolationEvidenceID: String?
    let error: String?

    static let programName = "qwen38-mtp-scorecard-proof-runner"

    static func failure(_ message: String) -> Qwen38ScorecardRunnerStatus {
        Qwen38ScorecardRunnerStatus(
            schema: ProofControl.schema,
            program: programName,
            status: "FAILED",
            promotable: false,
            qualified: nil,
            claimSHA256: nil,
            authorizationID: nil,
            operatorKeyID: nil,
            policyAdmissionID: nil,
            candidateProcessIsolationEvidenceID: nil,
            referenceProcessIsolationEvidenceID: nil,
            error: message)
    }

    func jsonLine() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }
}
