import CryptoKit
import Foundation

/// Per-case prompt geometry observed by the long-profile runner. Prompt token
/// counts are tokenizer-derived at run time, so they travel in the payload and
/// are cross-checked against every sample's prompt-preparation attribution.
public struct QwenMTPLongProfileCaseProfile: Codable, Equatable, Sendable {
    public var caseID: String
    public var promptTokenCount: Int

    public init(caseID: String, promptTokenCount: Int) {
        self.caseID = caseID
        self.promptTokenCount = promptTokenCount
    }
}

/// Non-authoritative long-decode measurement evidence for one target/drafter
/// pair. Reuses the frozen corpus profile-sample schema; identity, GDN mode,
/// and measurement class are self-describing so this evidence can never be
/// read back as the authoritative scorecard or the 9B promotion corpus.
public struct QwenMTPLongProfileEvidencePayload: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var corpusID: String
    public var corpusContentHash: String
    public var measurementClass: String
    public var binding: QwenMTPCorpusRuntimeBinding
    public var host: QwenMTPCorpusHostEvidence
    public var gdnMode: Qwen38MTPPerformanceScorecardGDNMode
    public var gdnObservedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv
    public var releaseBuildRequired: Bool
    public var releaseBuildObserved: Bool
    public var caseProfiles: [QwenMTPLongProfileCaseProfile]
    public var samples: [QwenMTPCorpusProfileSample]

    public init(
        schemaVersion: Int,
        corpusID: String,
        corpusContentHash: String,
        measurementClass: String,
        binding: QwenMTPCorpusRuntimeBinding,
        host: QwenMTPCorpusHostEvidence,
        gdnMode: Qwen38MTPPerformanceScorecardGDNMode,
        gdnObservedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv,
        releaseBuildRequired: Bool,
        releaseBuildObserved: Bool,
        caseProfiles: [QwenMTPLongProfileCaseProfile],
        samples: [QwenMTPCorpusProfileSample]
    ) {
        self.schemaVersion = schemaVersion
        self.corpusID = corpusID
        self.corpusContentHash = corpusContentHash
        self.measurementClass = measurementClass
        self.binding = binding
        self.host = host
        self.gdnMode = gdnMode
        self.gdnObservedEnv = gdnObservedEnv
        self.releaseBuildRequired = releaseBuildRequired
        self.releaseBuildObserved = releaseBuildObserved
        self.caseProfiles = caseProfiles
        self.samples = samples
    }
}

public struct QwenMTPLongProfileCaseSummary: Codable, Equatable, Sendable {
    public let caseID: String
    public let medianDecodeOnlyRatio: Double
    public let medianE2ERatio: Double
    public let medianScalarDecodeTokensPerSecond: Double
    public let medianMTPDecodeTokensPerSecond: Double
    public let meanDraftAcceptanceRate: Double

    public init(
        caseID: String,
        medianDecodeOnlyRatio: Double,
        medianE2ERatio: Double,
        medianScalarDecodeTokensPerSecond: Double,
        medianMTPDecodeTokensPerSecond: Double,
        meanDraftAcceptanceRate: Double
    ) {
        self.caseID = caseID
        self.medianDecodeOnlyRatio = medianDecodeOnlyRatio
        self.medianE2ERatio = medianE2ERatio
        self.medianScalarDecodeTokensPerSecond = medianScalarDecodeTokensPerSecond
        self.medianMTPDecodeTokensPerSecond = medianMTPDecodeTokensPerSecond
        self.meanDraftAcceptanceRate = meanDraftAcceptanceRate
    }
}

/// Descriptive medians only. This summary deliberately has no `qualified`
/// field: the long profile is an engineering pre-check, never a promotion
/// verdict.
public struct QwenMTPLongProfileSummary: Codable, Equatable, Sendable {
    public let aggregateMedianDecodeOnlyRatio: Double
    public let aggregateMedianE2ERatio: Double
    public let perCase: [QwenMTPLongProfileCaseSummary]

    public init(
        aggregateMedianDecodeOnlyRatio: Double,
        aggregateMedianE2ERatio: Double,
        perCase: [QwenMTPLongProfileCaseSummary]
    ) {
        self.aggregateMedianDecodeOnlyRatio = aggregateMedianDecodeOnlyRatio
        self.aggregateMedianE2ERatio = aggregateMedianE2ERatio
        self.perCase = perCase
    }
}

public enum QwenMTPLongProfileGateError: Error, Equatable, CustomStringConvertible, Sendable {
    case identityMismatch(String)
    case unknownBinding(String)
    case gdnModeNotPinned
    case releaseBuildRequired
    case invalidHost
    case invalidCaseProfiles(String)
    case invalidSample(index: Int, reason: String)
    case unterminatedJSONL
    case malformedJSONL(line: Int)
    case wrongSubcommand(String)
    case invalidProvenance(String)

    public var description: String {
        switch self {
        case .identityMismatch(let field):
            return "long-profile identity mismatch: \(field)"
        case .unknownBinding(let target):
            return "long-profile binding is not a known artifact lock row: \(target)"
        case .gdnModeNotPinned:
            return "long-profile requires gdnMode=gdn-on with the GDN environment observed enabled"
        case .releaseBuildRequired:
            return "long-profile measurement requires a Release build"
        case .invalidHost:
            return "long-profile host evidence is incomplete"
        case .invalidCaseProfiles(let reason):
            return "long-profile case profiles invalid: \(reason)"
        case .invalidSample(let index, let reason):
            return "long-profile sample[\(index)] invalid: \(reason)"
        case .unterminatedJSONL:
            return "long-profile evidence must be newline-terminated JSONL"
        case .malformedJSONL(let line):
            return "long-profile evidence line \(line) is malformed"
        case .wrongSubcommand(let subcommand):
            return "long-profile evidence has wrong subcommand: \(subcommand)"
        case .invalidProvenance(let reason):
            return "long-profile provenance invalid: \(reason)"
        }
    }
}

/// Long-decode (production-completion-shape) scalar-vs-MTP measurement corpus
/// and its structural gate. Exactness is enforced per pair exactly as in the
/// frozen 9B corpus; unlike that corpus this gate applies NO performance
/// qualification thresholds — it exists to measure, not to promote.
public enum QwenMTPLongProfileGate {
    public static let schemaVersion = 1
    public static let corpusID = "qwen38-mtp-long-decode-profile-v1"
    public static let corpusContentHash = "2257a871894869d4"
    public static let measurementClass = "engineering-precheck-non-authoritative"
    public static let subcommand = "qwen-mtp-long-profile"
    public static let rejectedSubcommand = "qwen-mtp-long-profile-rejected"

    /// Decision-grade generation length: the production default completion
    /// budget. Scalar and MTP arms always share it.
    public static let decisionMaxTokens = 4096
    /// Floor on tokenizer-observed prompt length; guards against a silently
    /// shortened prompt turning this back into a short-context artifact.
    public static let minimumPromptTokenCount = 3072
    /// Floor on generated tokens per measured arm; a run that stops this early
    /// is not a steady-state decode observation.
    public static let minimumGeneratedTokens = 256

    public static let cases: [QwenMTPCorpusCaseSpec] = [
        .init(
            id: "long-report-prose",
            kind: .fullGreedy,
            prompt: longReportPrompt,
            maxTokens: decisionMaxTokens),
        .init(
            id: "long-swift-code",
            kind: .fullGreedy,
            prompt: longSwiftCodePrompt,
            maxTokens: decisionMaxTokens),
        .init(
            id: "long-tool-json",
            kind: .fullGreedy,
            prompt: longToolJSONPrompt,
            maxTokens: decisionMaxTokens),
    ]

    public static let profilePlan = QwenMTPCorpusProfilePlan(
        caseIDs: ["long-report-prose", "long-swift-code", "long-tool-json"],
        droppedWarmupPairs: 1,
        measuredPairs: 3,
        orders: [.scalarThenMTP, .mtpThenScalar, .scalarThenMTP, .mtpThenScalar])

    /// Bindings this gate accepts, derived from the reviewed artifact locks so
    /// a lock revision update propagates here without a second pin.
    public static var allowedBindings: [QwenMTPCorpusRuntimeBinding] {
        [
            QwenMTPKnownArtifactLocks.qwen35_9BDepth1,
            QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1,
            QwenMTPKnownArtifactLocks.qwen38_27B4BitDepth1,
        ].map { lock in
            QwenMTPCorpusRuntimeBinding(
                targetModelID: lock.targetIdentity.modelID,
                drafterModelID: lock.drafterIdentity.modelID,
                targetRevision: lock.targetIdentity.revision,
                drafterRevision: lock.drafterIdentity.revision,
                sourceRevision: lock.sourceRevision,
                blockSize: 3,
                maxAcceptedDrafts: 2)
        }
    }

    public static func computedCorpusContentHash() -> String {
        var hasher = SHA256()
        func update(_ value: String) {
            hasher.update(data: Data(value.utf8))
            hasher.update(data: Data([0]))
        }
        update("corpus=\(corpusID)")
        update("schema=\(schemaVersion)")
        for spec in cases {
            update("case=\(spec.id)")
            update("kind=\(spec.kind.rawValue)")
            update("maxTokens=\(spec.maxTokens)")
            update("prompt=\(spec.prompt)")
        }
        update("plan=\(profilePlan.caseIDs.joined(separator: ","))")
        update("warmup=\(profilePlan.droppedWarmupPairs)")
        update("measured=\(profilePlan.measuredPairs)")
        update("orders=\(profilePlan.orders.map(\.rawValue).joined(separator: ","))")
        return hasher.finalize().prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    @discardableResult
    public static func validate(
        _ payload: QwenMTPLongProfileEvidencePayload
    ) throws -> QwenMTPLongProfileSummary {
        guard payload.schemaVersion == schemaVersion else {
            throw QwenMTPLongProfileGateError.identityMismatch("schemaVersion")
        }
        guard payload.corpusID == corpusID else {
            throw QwenMTPLongProfileGateError.identityMismatch("corpusID")
        }
        guard payload.corpusContentHash == corpusContentHash else {
            throw QwenMTPLongProfileGateError.identityMismatch("corpusContentHash")
        }
        guard payload.measurementClass == measurementClass else {
            throw QwenMTPLongProfileGateError.identityMismatch("measurementClass")
        }
        guard allowedBindings.contains(payload.binding) else {
            throw QwenMTPLongProfileGateError.unknownBinding(payload.binding.targetModelID)
        }
        guard payload.gdnMode == .gdnOn, payload.gdnObservedEnv == .enabled else {
            throw QwenMTPLongProfileGateError.gdnModeNotPinned
        }
        guard payload.releaseBuildRequired, payload.releaseBuildObserved else {
            throw QwenMTPLongProfileGateError.releaseBuildRequired
        }
        guard !payload.host.chip.isEmpty,
            payload.host.ramBytes > 0,
            !payload.host.os.isEmpty
        else {
            throw QwenMTPLongProfileGateError.invalidHost
        }

        guard payload.caseProfiles.map(\.caseID) == profilePlan.caseIDs else {
            throw QwenMTPLongProfileGateError.invalidCaseProfiles("case order")
        }
        var promptTokenCounts: [String: Int] = [:]
        for profile in payload.caseProfiles {
            guard profile.promptTokenCount >= minimumPromptTokenCount else {
                throw QwenMTPLongProfileGateError.invalidCaseProfiles(
                    "\(profile.caseID) promptTokenCount \(profile.promptTokenCount) below floor \(minimumPromptTokenCount)")
            }
            promptTokenCounts[profile.caseID] = profile.promptTokenCount
        }

        let expectedCount = profilePlan.caseIDs.count * profilePlan.totalPairsPerCase
        guard payload.samples.count == expectedCount else {
            throw QwenMTPLongProfileGateError.invalidSample(
                index: payload.samples.count,
                reason: "cardinality expected \(expectedCount)")
        }

        var sampleIndex = 0
        var measuredDecodeRatios: [Double] = []
        var measuredE2ERatios: [Double] = []
        var perCaseDecodeRatios: [String: [Double]] = [:]
        var perCaseE2ERatios: [String: [Double]] = [:]
        var perCaseScalarDecodeTPS: [String: [Double]] = [:]
        var perCaseMTPDecodeTPS: [String: [Double]] = [:]
        var perCaseAcceptance: [String: [Double]] = [:]

        for caseID in profilePlan.caseIDs {
            guard let spec = cases.first(where: { $0.id == caseID }) else {
                throw QwenMTPLongProfileGateError.invalidCaseProfiles("unknown case \(caseID)")
            }
            for pairIndex in 0..<profilePlan.totalPairsPerCase {
                let sample = payload.samples[sampleIndex]
                defer { sampleIndex += 1 }
                guard sample.caseID == caseID else {
                    throw QwenMTPLongProfileGateError.invalidSample(
                        index: sampleIndex, reason: "caseID")
                }
                guard sample.pairIndex == pairIndex else {
                    throw QwenMTPLongProfileGateError.invalidSample(
                        index: sampleIndex, reason: "pairIndex")
                }
                guard sample.order == profilePlan.orders[pairIndex] else {
                    throw QwenMTPLongProfileGateError.invalidSample(
                        index: sampleIndex, reason: "order")
                }
                guard sample.warmup == (pairIndex < profilePlan.droppedWarmupPairs) else {
                    throw QwenMTPLongProfileGateError.invalidSample(
                        index: sampleIndex, reason: "warmup")
                }
                do {
                    try QwenMTPCorpusGate.validateExactness(
                        sample.exactness,
                        context: "\(caseID) long-profile[\(pairIndex)]")
                    try QwenMTPCorpusGate.validateTelemetry(
                        sample.mtpTelemetry,
                        emittedTokenCount: sample.exactness.mtpTokenCount,
                        context: "\(caseID) long-profile[\(pairIndex)]")
                    try QwenMTPCorpusGate.validatePhaseAttribution(
                        sample.mtpPhaseAttribution,
                        telemetry: sample.mtpTelemetry,
                        timing: sample.mtpTiming,
                        expectsDrafterPriming: true,
                        expectedPromptTokenCount: promptTokenCounts[caseID] ?? 0,
                        context: "\(caseID) long-profile[\(pairIndex)]")
                } catch {
                    throw QwenMTPLongProfileGateError.invalidSample(
                        index: sampleIndex, reason: String(describing: error))
                }
                guard sample.exactness.mtpTokenCount >= minimumGeneratedTokens,
                    sample.exactness.mtpTokenCount <= spec.maxTokens
                else {
                    throw QwenMTPLongProfileGateError.invalidSample(
                        index: sampleIndex,
                        reason: "generated \(sample.exactness.mtpTokenCount) outside "
                            + "[\(minimumGeneratedTokens), \(spec.maxTokens)]")
                }
                guard sample.scalarTiming.allFinitePositive,
                    sample.mtpTiming.allFinitePositive,
                    sample.scalarTokensPerSecond.isFinite,
                    sample.scalarTokensPerSecond > 0,
                    sample.mtpTokensPerSecond.isFinite,
                    sample.mtpTokensPerSecond > 0,
                    sample.decodeOnlyRatio.isFinite,
                    sample.decodeOnlyRatio > 0,
                    sample.e2eRatio.isFinite,
                    sample.e2eRatio > 0
                else {
                    throw QwenMTPLongProfileGateError.invalidSample(
                        index: sampleIndex, reason: "timing")
                }
                let recomputedScalarTPS =
                    Double(sample.exactness.scalarTokenCount)
                    / sample.scalarTiming.e2eSeconds
                let recomputedMTPTPS =
                    Double(sample.exactness.mtpTokenCount)
                    / sample.mtpTiming.e2eSeconds
                let recomputedDecodeRatio =
                    sample.scalarTiming.generationSeconds
                    / sample.mtpTiming.generationSeconds
                let recomputedE2ERatio = recomputedMTPTPS / recomputedScalarTPS
                guard
                    QwenMTPCorpusGate.approximatelyEqual(
                        sample.scalarTokensPerSecond, recomputedScalarTPS),
                    QwenMTPCorpusGate.approximatelyEqual(
                        sample.mtpTokensPerSecond, recomputedMTPTPS),
                    QwenMTPCorpusGate.approximatelyEqual(
                        sample.decodeOnlyRatio, recomputedDecodeRatio),
                    QwenMTPCorpusGate.approximatelyEqual(
                        sample.e2eRatio, recomputedE2ERatio)
                else {
                    throw QwenMTPLongProfileGateError.invalidSample(
                        index: sampleIndex,
                        reason: "reported performance does not match counts and timings")
                }
                guard sample.mtpTelemetry.proposedDraftTokens > 0,
                    sample.passthroughReason == nil
                else {
                    throw QwenMTPLongProfileGateError.invalidSample(
                        index: sampleIndex, reason: "draft telemetry")
                }

                if !sample.warmup {
                    measuredDecodeRatios.append(recomputedDecodeRatio)
                    measuredE2ERatios.append(recomputedE2ERatio)
                    perCaseDecodeRatios[caseID, default: []].append(recomputedDecodeRatio)
                    perCaseE2ERatios[caseID, default: []].append(recomputedE2ERatio)
                    perCaseScalarDecodeTPS[caseID, default: []].append(
                        Double(sample.exactness.scalarTokenCount)
                            / sample.scalarTiming.generationSeconds)
                    perCaseMTPDecodeTPS[caseID, default: []].append(
                        Double(sample.exactness.mtpTokenCount)
                            / sample.mtpTiming.generationSeconds)
                    perCaseAcceptance[caseID, default: []].append(
                        Double(sample.mtpTelemetry.acceptedDraftTokens)
                            / Double(sample.mtpTelemetry.proposedDraftTokens))
                }
            }
        }

        let perCase = profilePlan.caseIDs.map { caseID in
            let acceptance = perCaseAcceptance[caseID] ?? []
            return QwenMTPLongProfileCaseSummary(
                caseID: caseID,
                medianDecodeOnlyRatio: QwenMTPCorpusGate.median(
                    perCaseDecodeRatios[caseID] ?? []),
                medianE2ERatio: QwenMTPCorpusGate.median(
                    perCaseE2ERatios[caseID] ?? []),
                medianScalarDecodeTokensPerSecond: QwenMTPCorpusGate.median(
                    perCaseScalarDecodeTPS[caseID] ?? []),
                medianMTPDecodeTokensPerSecond: QwenMTPCorpusGate.median(
                    perCaseMTPDecodeTPS[caseID] ?? []),
                meanDraftAcceptanceRate: acceptance.isEmpty
                    ? .nan
                    : acceptance.reduce(0, +) / Double(acceptance.count))
        }
        return QwenMTPLongProfileSummary(
            aggregateMedianDecodeOnlyRatio: QwenMTPCorpusGate.median(measuredDecodeRatios),
            aggregateMedianE2ERatio: QwenMTPCorpusGate.median(measuredE2ERatios),
            perCase: perCase)
    }

    public static func validateJSONL(_ data: Data) throws -> [QwenMTPLongProfileSummary] {
        guard data.last == 0x0a else {
            throw QwenMTPLongProfileGateError.unterminatedJSONL
        }
        let rows = data.split(separator: 0x0a, omittingEmptySubsequences: false).dropLast()
        var summaries: [QwenMTPLongProfileSummary] = []
        summaries.reserveCapacity(rows.count)
        let decoder = JSONDecoder()
        for (index, row) in rows.enumerated() {
            guard !row.isEmpty else {
                throw QwenMTPLongProfileGateError.malformedJSONL(line: index + 1)
            }
            let record: ResultRecord<QwenMTPLongProfileEvidencePayload>
            do {
                record = try decoder.decode(
                    ResultRecord<QwenMTPLongProfileEvidencePayload>.self,
                    from: Data(row))
            } catch {
                throw QwenMTPLongProfileGateError.malformedJSONL(line: index + 1)
            }
            guard record.subcommand == subcommand else {
                throw QwenMTPLongProfileGateError.wrongSubcommand(record.subcommand)
            }
            guard record.provenance.modelPath == record.payload.binding.targetModelID else {
                throw QwenMTPLongProfileGateError.invalidProvenance("model identity")
            }
            guard record.provenance.corpusId == corpusID,
                record.provenance.corpusContentHash == corpusContentHash
            else {
                throw QwenMTPLongProfileGateError.invalidProvenance("corpus identity")
            }
            summaries.append(try validate(record.payload))
        }
        return summaries
    }

    // MARK: - Deterministic long prompts

    private static let longReportPrompt: String = {
        let subsystems = [
            "cache eviction planner", "prefill scheduler", "tokenizer bridge",
            "KV quantizer", "draft verifier", "session router",
            "memory budget planner", "thermal governor",
        ]
        let outcomes = [
            "recovered without operator action",
            "required one bounded retry",
            "was resolved by an exact cache rollback",
            "was absorbed by the scalar fallback path",
            "cleared after a telemetry flush",
            "was confirmed benign by the audit trail",
        ]
        let entries = (1...60).map { index in
            "Entry \(index): At minute \(index * 7 % 1440), the \(subsystems[index % subsystems.count]) "
                + "reported a transient anomaly during a long structured decode. The on-call review found "
                + "the event \(outcomes[index % outcomes.count]), and the ledger records queue depth "
                + "\(index * 13 % 97), rollback count \(index * 5 % 7), and a p95 latency delta of "
                + "\(index * 29 % 450) microseconds."
        }.joined(separator: "\n")
        return entries
            + "\nContinue this operations ledger in exactly the same single-line entry format, "
            + "starting at Entry 61 and continuing through Entry 400. Do not stop early, do not "
            + "summarize, and do not add closing remarks."
    }()

    private static let longSwiftCodePrompt: String = {
        let functions = (1...55).map { index in
            """
            /// Returns the bounded moving average for probe \(index).
            func metricProbe\(index)(samples: [Double]) -> Double {
                guard !samples.isEmpty else { return 0 }
                let clamped = samples.map { min(max($0, 0), \(index % 50).0 + 1.0) }
                return clamped.reduce(0, +) / Double(clamped.count)
            }
            """
        }.joined(separator: "\n\n")
        return functions
            + "\n\n// Continue this file by implementing metricProbe56 through metricProbe240 in "
            + "exactly the same documented style: one doc comment, one guard, one clamp with the "
            + "bound cycling as above, one average. Emit Swift code only, with no commentary."
    }()

    private static let longToolJSONPrompt: String = {
        let tools = [
            "record_evidence", "verify_digest", "load_manifest",
            "publish_receipt", "roll_back_cache", "sample_telemetry",
        ]
        let statuses = ["ok", "retried", "queued", "verified", "flushed"]
        let rows = (1...75).map { index in
            "  {\"id\": \(index), \"tool\": \"\(tools[index % tools.count])\", "
                + "\"arguments\": {\"path\": \"/var/evidence/row-\(index).json\", "
                + "\"attempt\": \(index % 4), \"verified\": \(index % 3 == 0 ? "true" : "false")}, "
                + "\"status\": \"\(statuses[index % statuses.count])\"}"
        }.joined(separator: ",\n")
        return "Continue this JSON array of tool-call transcript rows from id 76 through id 400 "
            + "using exactly the same object shape and key order, strict JSON only, one object per "
            + "line, no prose before or after. Close the array only after id 400.\n\n[\n"
            + rows + ","
    }()
}
