import Foundation

public enum PrefixCacheRequestOutcome:
    String, Codable, Equatable, Sendable
{
    case disabled
    case miss
    case exactHit = "exact-hit"
    case partialHit = "partial-hit"
    case rejected
}

public enum RequestStartMetricsError:
    Error, Equatable, Sendable
{
    case invalidPromptTokenCount(Int)
    case invalidTokenCount(String)
    case promptTokenConservation
    case outcomeTokenMismatch
    case rejectionReasonMismatch
    case invalidDuration(String)
    case invalidRetainedBytes(Int)
    case invalidEntryCount(Int)
    case invalidEvictionCount(Int)
    case nonFiniteDerivedRate(String)
    case derivedRateMismatch(String)
}

/// Recomputable request-start evidence. Logical cache reads and physically evaluated prefill rows
/// are deliberately distinct so a hot hit cannot inflate a kernel throughput claim.
public struct RequestStartMetrics:
    Codable, Equatable, Sendable
{
    private enum CodingKeys: String, CodingKey {
        case promptTokenCount
        case cacheReadTokenCount
        case physicalPrefillTokenCount
        case prefixCacheOutcome
        case prefixCacheRejectionReason
        case templateTokenCacheHit
        case templateSeconds
        case tokenizeSeconds
        case lookupSeconds
        case restoreSeconds
        case prefillSeconds
        case retainedBytes
        case entryCount
        case evictionCount
        case eagerWarmupSeconds
        case apparentPrefillTokensPerSecond
        case physicalPrefillTokensPerSecond
    }

    public let promptTokenCount: Int
    public let cacheReadTokenCount: Int
    public let physicalPrefillTokenCount: Int
    public let prefixCacheOutcome: PrefixCacheRequestOutcome
    public let prefixCacheRejectionReason:
        ExactPrefixCommitSkipReason?
    public let templateTokenCacheHit: Bool
    public let templateSeconds: Double
    public let tokenizeSeconds: Double
    public let lookupSeconds: Double
    public let restoreSeconds: Double
    public let prefillSeconds: Double
    public let retainedBytes: Int
    public let entryCount: Int
    public let evictionCount: Int
    public let eagerWarmupSeconds: Double?
    public let apparentPrefillTokensPerSecond: Double
    public let physicalPrefillTokensPerSecond: Double

    public init(
        promptTokenCount: Int,
        cacheReadTokenCount: Int,
        physicalPrefillTokenCount: Int,
        prefixCacheOutcome: PrefixCacheRequestOutcome,
        prefixCacheRejectionReason:
            ExactPrefixCommitSkipReason? = nil,
        templateTokenCacheHit: Bool,
        templateSeconds: Double,
        tokenizeSeconds: Double,
        lookupSeconds: Double,
        restoreSeconds: Double,
        prefillSeconds: Double,
        retainedBytes: Int,
        entryCount: Int,
        evictionCount: Int,
        eagerWarmupSeconds: Double? = nil
    ) throws {
        guard promptTokenCount > 0 else {
            throw RequestStartMetricsError
                .invalidPromptTokenCount(promptTokenCount)
        }
        for (field, count) in [
            ("cacheReadTokenCount", cacheReadTokenCount),
            ("physicalPrefillTokenCount", physicalPrefillTokenCount),
        ] where count < 0 {
            throw RequestStartMetricsError.invalidTokenCount(field)
        }
        guard cacheReadTokenCount <= promptTokenCount,
            physicalPrefillTokenCount
                == promptTokenCount - cacheReadTokenCount
        else {
            throw RequestStartMetricsError.promptTokenConservation
        }
        let outcomeMatches: Bool
        switch prefixCacheOutcome {
        case .disabled, .miss, .rejected:
            outcomeMatches =
                cacheReadTokenCount == 0
                && physicalPrefillTokenCount == promptTokenCount
        case .exactHit:
            outcomeMatches =
                cacheReadTokenCount == promptTokenCount
                && physicalPrefillTokenCount == 0
        case .partialHit:
            outcomeMatches =
                cacheReadTokenCount > 0
                && cacheReadTokenCount < promptTokenCount
                && physicalPrefillTokenCount
                    == promptTokenCount - cacheReadTokenCount
        }
        guard outcomeMatches else {
            throw RequestStartMetricsError.outcomeTokenMismatch
        }
        let admissionRejectionReasons: Set<
            ExactPrefixCommitSkipReason
        > = [
            .prefixTooShort,
            .snapshotExceedsBudget,
            .reservationCapacityExhausted,
            .snapshotCaptureFailed,
            .snapshotRestoreFailed,
            .snapshotEvidenceMismatch,
            .padOnlyOutput,
        ]
        switch prefixCacheOutcome {
        case .rejected:
            guard let prefixCacheRejectionReason,
                admissionRejectionReasons.contains(
                    prefixCacheRejectionReason)
            else {
                throw RequestStartMetricsError
                    .rejectionReasonMismatch
            }
        case .disabled, .miss, .exactHit, .partialHit:
            guard prefixCacheRejectionReason == nil else {
                throw RequestStartMetricsError
                    .rejectionReasonMismatch
            }
        }

        for (field, duration) in [
            ("templateSeconds", templateSeconds),
            ("tokenizeSeconds", tokenizeSeconds),
            ("lookupSeconds", lookupSeconds),
            ("restoreSeconds", restoreSeconds),
            ("prefillSeconds", prefillSeconds),
        ] where !duration.isFinite || duration < 0 {
            throw RequestStartMetricsError.invalidDuration(field)
        }
        guard prefillSeconds > 0 else {
            throw RequestStartMetricsError.invalidDuration(
                "prefillSeconds")
        }
        if let eagerWarmupSeconds,
            !eagerWarmupSeconds.isFinite || eagerWarmupSeconds < 0
        {
            throw RequestStartMetricsError.invalidDuration(
                "eagerWarmupSeconds")
        }
        guard retainedBytes >= 0 else {
            throw RequestStartMetricsError
                .invalidRetainedBytes(retainedBytes)
        }
        guard entryCount >= 0 else {
            throw RequestStartMetricsError
                .invalidEntryCount(entryCount)
        }
        guard evictionCount >= 0 else {
            throw RequestStartMetricsError
                .invalidEvictionCount(evictionCount)
        }

        let apparentPrefillTokensPerSecond =
            Double(promptTokenCount) / prefillSeconds
        let physicalPrefillTokensPerSecond =
            Double(physicalPrefillTokenCount) / prefillSeconds
        guard apparentPrefillTokensPerSecond.isFinite else {
            throw RequestStartMetricsError.nonFiniteDerivedRate(
                "apparentPrefillTokensPerSecond")
        }
        guard physicalPrefillTokensPerSecond.isFinite else {
            throw RequestStartMetricsError.nonFiniteDerivedRate(
                "physicalPrefillTokensPerSecond")
        }

        self.promptTokenCount = promptTokenCount
        self.cacheReadTokenCount = cacheReadTokenCount
        self.physicalPrefillTokenCount =
            physicalPrefillTokenCount
        self.prefixCacheOutcome = prefixCacheOutcome
        self.prefixCacheRejectionReason =
            prefixCacheRejectionReason
        self.templateTokenCacheHit = templateTokenCacheHit
        self.templateSeconds = templateSeconds
        self.tokenizeSeconds = tokenizeSeconds
        self.lookupSeconds = lookupSeconds
        self.restoreSeconds = restoreSeconds
        self.prefillSeconds = prefillSeconds
        self.retainedBytes = retainedBytes
        self.entryCount = entryCount
        self.evictionCount = evictionCount
        self.eagerWarmupSeconds = eagerWarmupSeconds
        self.apparentPrefillTokensPerSecond =
            apparentPrefillTokensPerSecond
        self.physicalPrefillTokensPerSecond =
            physicalPrefillTokensPerSecond
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let apparentPrefillTokensPerSecond = try values.decode(
            Double.self,
            forKey: .apparentPrefillTokensPerSecond)
        let physicalPrefillTokensPerSecond = try values.decode(
            Double.self,
            forKey: .physicalPrefillTokensPerSecond)
        let decoded = try RequestStartMetrics(
            promptTokenCount: values.decode(
                Int.self,
                forKey: .promptTokenCount),
            cacheReadTokenCount: values.decode(
                Int.self,
                forKey: .cacheReadTokenCount),
            physicalPrefillTokenCount: values.decode(
                Int.self,
                forKey: .physicalPrefillTokenCount),
            prefixCacheOutcome: values.decode(
                PrefixCacheRequestOutcome.self,
                forKey: .prefixCacheOutcome),
            prefixCacheRejectionReason: values.decodeIfPresent(
                ExactPrefixCommitSkipReason.self,
                forKey: .prefixCacheRejectionReason),
            templateTokenCacheHit: values.decode(
                Bool.self,
                forKey: .templateTokenCacheHit),
            templateSeconds: values.decode(
                Double.self,
                forKey: .templateSeconds),
            tokenizeSeconds: values.decode(
                Double.self,
                forKey: .tokenizeSeconds),
            lookupSeconds: values.decode(
                Double.self,
                forKey: .lookupSeconds),
            restoreSeconds: values.decode(
                Double.self,
                forKey: .restoreSeconds),
            prefillSeconds: values.decode(
                Double.self,
                forKey: .prefillSeconds),
            retainedBytes: values.decode(
                Int.self,
                forKey: .retainedBytes),
            entryCount: values.decode(
                Int.self,
                forKey: .entryCount),
            evictionCount: values.decode(
                Int.self,
                forKey: .evictionCount),
            eagerWarmupSeconds: values.decodeIfPresent(
                Double.self,
                forKey: .eagerWarmupSeconds))
        guard decoded.apparentPrefillTokensPerSecond
            == apparentPrefillTokensPerSecond
        else {
            throw RequestStartMetricsError.derivedRateMismatch(
                "apparentPrefillTokensPerSecond")
        }
        guard decoded.physicalPrefillTokensPerSecond
            == physicalPrefillTokensPerSecond
        else {
            throw RequestStartMetricsError.derivedRateMismatch(
                "physicalPrefillTokensPerSecond")
        }
        self = decoded
    }

    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(
            promptTokenCount,
            forKey: .promptTokenCount)
        try values.encode(
            cacheReadTokenCount,
            forKey: .cacheReadTokenCount)
        try values.encode(
            physicalPrefillTokenCount,
            forKey: .physicalPrefillTokenCount)
        try values.encode(
            prefixCacheOutcome,
            forKey: .prefixCacheOutcome)
        try values.encodeIfPresent(
            prefixCacheRejectionReason,
            forKey: .prefixCacheRejectionReason)
        try values.encode(
            templateTokenCacheHit,
            forKey: .templateTokenCacheHit)
        try values.encode(templateSeconds, forKey: .templateSeconds)
        try values.encode(tokenizeSeconds, forKey: .tokenizeSeconds)
        try values.encode(lookupSeconds, forKey: .lookupSeconds)
        try values.encode(restoreSeconds, forKey: .restoreSeconds)
        try values.encode(prefillSeconds, forKey: .prefillSeconds)
        try values.encode(retainedBytes, forKey: .retainedBytes)
        try values.encode(entryCount, forKey: .entryCount)
        try values.encode(evictionCount, forKey: .evictionCount)
        try values.encodeIfPresent(
            eagerWarmupSeconds,
            forKey: .eagerWarmupSeconds)
        try values.encode(
            apparentPrefillTokensPerSecond,
            forKey: .apparentPrefillTokensPerSecond)
        try values.encode(
            physicalPrefillTokensPerSecond,
            forKey: .physicalPrefillTokensPerSecond)
    }
}
