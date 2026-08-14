import Foundation

public enum ExactPrefixProofCaseID:
    String, Codable, CaseIterable, Equatable, Hashable, Sendable
{
    case coldControlA = "cold-control-A"
    case coldCommitA = "cold-commit-A"
    case exactHitA = "exact-hit-A"
    case partialControl = "partial-control"
    case partialHit = "partial-hit"
    case coldCommitB = "cold-commit-B"
    case returnHitA = "return-hit-A"
    case pressureEvictedA = "pressure-evicted-A"
    case postWarmupControl = "post-warmup-control"
    case postWarmupMiss = "post-warmup-miss"
    case postWarmupHit = "post-warmup-hit"

    public static let requiredOrder: [ExactPrefixProofCaseID] = [
        .coldControlA,
        .coldCommitA,
        .exactHitA,
        .partialControl,
        .partialHit,
        .coldCommitB,
        .returnHitA,
        .pressureEvictedA,
        .postWarmupControl,
        .postWarmupMiss,
        .postWarmupHit,
    ]

    var isControl: Bool {
        switch self {
        case .coldControlA, .partialControl, .postWarmupControl:
            return true
        case .coldCommitA, .exactHitA, .partialHit, .coldCommitB,
            .returnHitA, .pressureEvictedA, .postWarmupMiss,
            .postWarmupHit:
            return false
        }
    }

    var expectedPrefixOutcome: PrefixCacheRequestOutcome? {
        switch self {
        case .coldControlA, .partialControl, .postWarmupControl:
            return nil
        case .coldCommitA, .coldCommitB, .pressureEvictedA,
            .postWarmupMiss:
            return .miss
        case .exactHitA, .returnHitA, .postWarmupHit:
            return .exactHit
        case .partialHit:
            return .partialHit
        }
    }

    var expectedTemplateReceipt: ExactPrefixProofCacheReceipt {
        switch self {
        case .coldControlA, .coldCommitA, .partialControl, .coldCommitB,
            .postWarmupControl, .postWarmupMiss:
            return .miss
        case .exactHitA, .partialHit, .returnHitA, .pressureEvictedA,
            .postWarmupHit:
            return .hit
        }
    }
}

public enum ExactPrefixProofCacheReceipt:
    String, Codable, Equatable, Hashable, Sendable
{
    case miss
    case hit

    var templateHit: Bool {
        switch self {
        case .miss:
            return false
        case .hit:
            return true
        }
    }
}

public enum ExactPrefixProofReusedPrefixKind:
    String, Codable, Equatable, Hashable, Sendable
{
    case promptOnly = "prompt-only"
    case finalContext = "final-context"
}

public enum ExactPrefixProofEvidenceError: Error, Equatable, Sendable {
    case unsupportedSchemaVersion(Int)
    case legacyPromotableEvidence
    case invalidModelID
    case invalidSourceRevision
    case sourceRevisionIdentityMismatch
    case invalidExpectedHarnessSHA
    case invalidExpectedExecutableSHA256
    case invalidWorkloadNonce
    case invalidMaxTokens
    case invalidPromptRepeat
    case invalidCachePolicy
    case invalidTemplateTokenCachePolicy
    case invalidMemoryLimits
    case invalidHash(ExactPrefixProofCaseID)
    case invalidPromptTokenCount(ExactPrefixProofCaseID)
    case invalidGeneratedTokenCount(ExactPrefixProofCaseID)
    case invalidFinalContext(ExactPrefixProofCaseID)
    case invalidTiming(ExactPrefixProofCaseID)
    case partialEvidence(ExactPrefixProofCaseID)
    case caseOrderMismatch
    case controlCarriesRequestMetrics(ExactPrefixProofCaseID)
    case templateTokenCacheMismatch(ExactPrefixProofCaseID)
    case referenceHashMismatch(ExactPrefixProofCaseID)
    case groupReferenceMismatch(ExactPrefixProofCaseID)
    case promptMetricMismatch(ExactPrefixProofCaseID)
    case warmupDurationMismatch(ExactPrefixProofCaseID)
    case reusedPrefixMismatch(ExactPrefixProofCaseID)
    case requestContextMismatch(ExactPrefixProofCaseID)
    case invalidWarmupDuration
    case warmupMutatedCache
    case derivedClaimMismatch
}

public enum ExactPrefixProofCommandPlanError:
    Error, Equatable, Sendable
{
    case invalidModelID
    case invalidSourceRevision
    case sourceRevisionIdentityMismatch
    case invalidExpectedHarnessSHA
    case invalidExpectedExecutableSHA256
    case invalidAdmission
    case invalidCheckpointContentSHA256
    case invalidTokenizerSHA256
    case checkpointIdentityMismatch
    case tokenizerIdentityMismatch
    case invalidWorkloadNonce
    case invalidMaxTokens
    case invalidPromptRepeat
    case invalidCachePolicy
    case invalidTemplateTokenCachePolicy
    case invalidMemoryLimits
    case invalidOutputPath
}

public struct ExactPrefixProofCaseTiming:
    Codable, Equatable, Sendable
{
    public let requestStartSeconds: Double
    public let ttftSeconds: Double
    public let decodeSeconds: Double
    public let totalSeconds: Double

    public init(
        requestStartSeconds: Double,
        ttftSeconds: Double,
        decodeSeconds: Double,
        totalSeconds: Double
    ) throws {
        self.requestStartSeconds = requestStartSeconds
        self.ttftSeconds = ttftSeconds
        self.decodeSeconds = decodeSeconds
        self.totalSeconds = totalSeconds
        guard Self.valid(requestStartSeconds), Self.valid(ttftSeconds),
            Self.valid(decodeSeconds), Self.valid(totalSeconds),
            ttftSeconds >= requestStartSeconds,
            totalSeconds >= ttftSeconds,
            totalSeconds >= decodeSeconds
        else {
            throw ExactPrefixProofEvidenceError.invalidTiming(
                .coldControlA)
        }
    }

    private enum CodingKeys: String, CodingKey {
        case requestStartSeconds
        case ttftSeconds
        case decodeSeconds
        case totalSeconds
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            requestStartSeconds: values.decode(
                Double.self,
                forKey: .requestStartSeconds),
            ttftSeconds: values.decode(Double.self, forKey: .ttftSeconds),
            decodeSeconds: values.decode(
                Double.self,
                forKey: .decodeSeconds),
            totalSeconds: values.decode(
                Double.self,
                forKey: .totalSeconds))
    }

    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(
            requestStartSeconds,
            forKey: .requestStartSeconds)
        try values.encode(ttftSeconds, forKey: .ttftSeconds)
        try values.encode(decodeSeconds, forKey: .decodeSeconds)
        try values.encode(totalSeconds, forKey: .totalSeconds)
    }

    fileprivate static func valid(_ value: Double) -> Bool {
        value.isFinite && value > 0
    }
}

public struct ExactPrefixProofCaseEvidence:
    Codable, Equatable, Sendable
{
    public let caseID: ExactPrefixProofCaseID
    public let promptTokenIDsSHA256: String
    public let promptTokenCount: Int
    public let generatedTokenIDsSHA256: String
    public let outputSHA256: String
    public let referenceGeneratedTokenIDsSHA256: String
    public let referenceOutputSHA256: String
    public let generatedTokenCount: Int
    public let finalContextTokenIDsSHA256: String?
    public let finalContextTokenCount: Int?
    public let timing: ExactPrefixProofCaseTiming
    public let requestStartMetrics: RequestStartMetrics?
    public let memoryEvidence: BenchRunMemoryEvidence?
    public let templateTokenCacheReceipt: ExactPrefixProofCacheReceipt
    public let reusedPrefixTokenIDsSHA256: String?
    public let reusedPrefixTokenCount: Int
    public let reusedPrefixSourceCaseID: ExactPrefixProofCaseID?
    public let reusedPrefixSourceKind:
        ExactPrefixProofReusedPrefixKind?
    public let requestContext: ExactPrefixRequestContext?

    public init(
        caseID: ExactPrefixProofCaseID,
        promptTokenIDsSHA256: String,
        promptTokenCount: Int,
        generatedTokenIDsSHA256: String,
        outputSHA256: String,
        referenceGeneratedTokenIDsSHA256: String,
        referenceOutputSHA256: String,
        generatedTokenCount: Int,
        finalContextTokenIDsSHA256: String? = nil,
        finalContextTokenCount: Int? = nil,
        timing: ExactPrefixProofCaseTiming,
        requestStartMetrics: RequestStartMetrics?,
        memoryEvidence: BenchRunMemoryEvidence?,
        templateTokenCacheReceipt: ExactPrefixProofCacheReceipt,
        reusedPrefixTokenIDsSHA256: String? = nil,
        reusedPrefixTokenCount: Int = 0,
        reusedPrefixSourceCaseID: ExactPrefixProofCaseID? = nil,
        reusedPrefixSourceKind:
            ExactPrefixProofReusedPrefixKind? = nil,
        requestContext: ExactPrefixRequestContext? = nil
    ) throws {
        self.init(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixTokenIDsSHA256 == nil
                    && reusedPrefixTokenCount == 0
                    ? nil : reusedPrefixSourceCaseID,
            reusedPrefixSourceKind:
                reusedPrefixTokenIDsSHA256 == nil
                    && reusedPrefixTokenCount == 0
                    ? nil : reusedPrefixSourceKind,
            requestContext: requestContext)
        try validate(
            memoryLimitBytes: Int.max,
            cacheLimitBytes: Int.max)
    }

    private init(
        uncheckedCaseID caseID: ExactPrefixProofCaseID,
        promptTokenIDsSHA256: String,
        promptTokenCount: Int,
        generatedTokenIDsSHA256: String,
        outputSHA256: String,
        referenceGeneratedTokenIDsSHA256: String,
        referenceOutputSHA256: String,
        generatedTokenCount: Int,
        finalContextTokenIDsSHA256: String?,
        finalContextTokenCount: Int?,
        timing: ExactPrefixProofCaseTiming,
        requestStartMetrics: RequestStartMetrics?,
        memoryEvidence: BenchRunMemoryEvidence?,
        templateTokenCacheReceipt: ExactPrefixProofCacheReceipt,
        reusedPrefixTokenIDsSHA256: String?,
        reusedPrefixTokenCount: Int,
        reusedPrefixSourceCaseID: ExactPrefixProofCaseID?,
        reusedPrefixSourceKind:
            ExactPrefixProofReusedPrefixKind?,
        requestContext: ExactPrefixRequestContext?
    ) {
        self.caseID = caseID
        self.promptTokenIDsSHA256 = promptTokenIDsSHA256
        self.promptTokenCount = promptTokenCount
        self.generatedTokenIDsSHA256 = generatedTokenIDsSHA256
        self.outputSHA256 = outputSHA256
        self.referenceGeneratedTokenIDsSHA256 =
            referenceGeneratedTokenIDsSHA256
        self.referenceOutputSHA256 = referenceOutputSHA256
        self.generatedTokenCount = generatedTokenCount
        self.finalContextTokenIDsSHA256 =
            finalContextTokenIDsSHA256
        self.finalContextTokenCount = finalContextTokenCount
        self.timing = timing
        self.requestStartMetrics = requestStartMetrics
        self.memoryEvidence = memoryEvidence
        self.templateTokenCacheReceipt = templateTokenCacheReceipt
        self.reusedPrefixTokenIDsSHA256 =
            reusedPrefixTokenIDsSHA256
        self.reusedPrefixTokenCount = reusedPrefixTokenCount
        self.reusedPrefixSourceCaseID =
            reusedPrefixSourceCaseID
        self.reusedPrefixSourceKind = reusedPrefixSourceKind
        self.requestContext = requestContext
    }

    @discardableResult
    fileprivate func validate(
        memoryLimitBytes: Int,
        cacheLimitBytes: Int
    ) throws -> Self {
        guard requestStartIsLowercaseSHA256(promptTokenIDsSHA256),
            requestStartIsLowercaseSHA256(generatedTokenIDsSHA256),
            requestStartIsLowercaseSHA256(outputSHA256),
            requestStartIsLowercaseSHA256(
                referenceGeneratedTokenIDsSHA256),
            requestStartIsLowercaseSHA256(referenceOutputSHA256)
        else {
            throw ExactPrefixProofEvidenceError.invalidHash(caseID)
        }
        guard promptTokenCount > 0 else {
            throw ExactPrefixProofEvidenceError
                .invalidPromptTokenCount(caseID)
        }
        guard generatedTokenCount > 0 else {
            throw ExactPrefixProofEvidenceError
                .invalidGeneratedTokenCount(caseID)
        }
        let (expectedFinalContextTokenCount, finalContextOverflow) =
            promptTokenCount.addingReportingOverflow(
                generatedTokenCount)
        switch (
            finalContextTokenIDsSHA256,
            finalContextTokenCount
        ) {
        case (nil, nil):
            break
        case let (sha256?, count?):
            guard !finalContextOverflow,
                requestStartIsLowercaseSHA256(sha256),
                count == expectedFinalContextTokenCount
            else {
                throw ExactPrefixProofEvidenceError
                    .invalidFinalContext(caseID)
            }
        case (.some, nil), (nil, .some):
            throw ExactPrefixProofEvidenceError
                .invalidFinalContext(caseID)
        }
        guard ExactPrefixProofCaseTiming.valid(
            timing.requestStartSeconds),
            ExactPrefixProofCaseTiming.valid(timing.ttftSeconds),
            ExactPrefixProofCaseTiming.valid(timing.decodeSeconds),
            ExactPrefixProofCaseTiming.valid(timing.totalSeconds),
            timing.ttftSeconds >= timing.requestStartSeconds,
            timing.totalSeconds >= timing.ttftSeconds,
            timing.totalSeconds >= timing.decodeSeconds
        else {
            throw ExactPrefixProofEvidenceError.invalidTiming(caseID)
        }
        guard memoryEvidence != nil else {
            throw ExactPrefixProofEvidenceError.partialEvidence(caseID)
        }
        if caseID.isControl {
            guard requestStartMetrics == nil,
                requestContext == nil,
                reusedPrefixTokenIDsSHA256 == nil,
                reusedPrefixTokenCount == 0,
                reusedPrefixSourceCaseID == nil,
                reusedPrefixSourceKind == nil
            else {
                throw ExactPrefixProofEvidenceError
                    .controlCarriesRequestMetrics(caseID)
            }
        } else {
            guard let requestStartMetrics,
                requestContext != nil
            else {
                throw ExactPrefixProofEvidenceError
                    .partialEvidence(caseID)
            }
            guard requestStartMetrics.promptTokenCount
                == promptTokenCount
            else {
                throw ExactPrefixProofEvidenceError
                    .promptMetricMismatch(caseID)
            }
            guard requestStartMetrics.templateTokenCacheHit
                == templateTokenCacheReceipt.templateHit
            else {
                throw ExactPrefixProofEvidenceError
                    .templateTokenCacheMismatch(caseID)
            }
            switch requestStartMetrics.prefixCacheOutcome {
            case .exactHit, .partialHit:
                guard let reusedPrefixTokenIDsSHA256,
                    requestStartIsLowercaseSHA256(
                        reusedPrefixTokenIDsSHA256),
                    reusedPrefixTokenCount > 0,
                    reusedPrefixTokenCount
                        == requestStartMetrics.cacheReadTokenCount,
                    (reusedPrefixSourceCaseID == nil)
                        == (reusedPrefixSourceKind == nil)
                else {
                    throw ExactPrefixProofEvidenceError
                        .reusedPrefixMismatch(caseID)
                }
            case .disabled, .miss, .rejected:
                guard reusedPrefixTokenIDsSHA256 == nil,
                    reusedPrefixTokenCount == 0,
                    reusedPrefixSourceCaseID == nil,
                    reusedPrefixSourceKind == nil
                else {
                    throw ExactPrefixProofEvidenceError
                        .reusedPrefixMismatch(caseID)
                }
            }
        }
        return self
    }

    func replacing(
        outputSHA256: String
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixSourceCaseID,
            reusedPrefixSourceKind: reusedPrefixSourceKind,
            requestContext: requestContext)
    }

    func replacing(
        referenceOutputSHA256: String
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixSourceCaseID,
            reusedPrefixSourceKind: reusedPrefixSourceKind,
            requestContext: requestContext)
    }

    func replacing(
        requestStartMetrics: RequestStartMetrics?
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixSourceCaseID,
            reusedPrefixSourceKind: reusedPrefixSourceKind,
            requestContext: requestContext)
    }

    func replacing(
        templateTokenCacheReceipt: ExactPrefixProofCacheReceipt
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixSourceCaseID,
            reusedPrefixSourceKind: reusedPrefixSourceKind,
            requestContext: requestContext)
    }

    func replacing(
        memoryEvidence: BenchRunMemoryEvidence?
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixSourceCaseID,
            reusedPrefixSourceKind: reusedPrefixSourceKind,
            requestContext: requestContext)
    }

    func replacing(
        promptTokenIDsSHA256: String
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixSourceCaseID,
            reusedPrefixSourceKind: reusedPrefixSourceKind,
            requestContext: requestContext)
    }

    func replacing(
        promptTokenCount: Int
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixSourceCaseID,
            reusedPrefixSourceKind: reusedPrefixSourceKind,
            requestContext: requestContext)
    }

    func replacing(
        timing: ExactPrefixProofCaseTiming
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixSourceCaseID,
            reusedPrefixSourceKind: reusedPrefixSourceKind,
            requestContext: requestContext)
    }

    func replacing(
        reusedPrefixTokenIDsSHA256: String?,
        reusedPrefixTokenCount: Int
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixTokenIDsSHA256 == nil
                    && reusedPrefixTokenCount == 0
                    ? nil : reusedPrefixSourceCaseID,
            reusedPrefixSourceKind:
                reusedPrefixTokenIDsSHA256 == nil
                    && reusedPrefixTokenCount == 0
                    ? nil : reusedPrefixSourceKind,
            requestContext: requestContext)
    }

    func replacing(
        reusedPrefixSourceCaseID: ExactPrefixProofCaseID?,
        reusedPrefixSourceKind:
            ExactPrefixProofReusedPrefixKind?
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixSourceCaseID,
            reusedPrefixSourceKind: reusedPrefixSourceKind,
            requestContext: requestContext)
    }

    func replacing(
        finalContextTokenIDsSHA256: String?,
        finalContextTokenCount: Int?
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixSourceCaseID,
            reusedPrefixSourceKind: reusedPrefixSourceKind,
            requestContext: requestContext)
    }

    func replacing(
        requestContext: ExactPrefixRequestContext?
    ) -> ExactPrefixProofCaseEvidence {
        Self(
            uncheckedCaseID: caseID,
            promptTokenIDsSHA256: promptTokenIDsSHA256,
            promptTokenCount: promptTokenCount,
            generatedTokenIDsSHA256: generatedTokenIDsSHA256,
            outputSHA256: outputSHA256,
            referenceGeneratedTokenIDsSHA256:
                referenceGeneratedTokenIDsSHA256,
            referenceOutputSHA256: referenceOutputSHA256,
            generatedTokenCount: generatedTokenCount,
            finalContextTokenIDsSHA256:
                finalContextTokenIDsSHA256,
            finalContextTokenCount: finalContextTokenCount,
            timing: timing,
            requestStartMetrics: requestStartMetrics,
            memoryEvidence: memoryEvidence,
            templateTokenCacheReceipt: templateTokenCacheReceipt,
            reusedPrefixTokenIDsSHA256:
                reusedPrefixTokenIDsSHA256,
            reusedPrefixTokenCount: reusedPrefixTokenCount,
            reusedPrefixSourceCaseID:
                reusedPrefixSourceCaseID,
            reusedPrefixSourceKind: reusedPrefixSourceKind,
            requestContext: requestContext)
    }
}

public struct ExactPrefixProofWarmupEvidence:
    Codable, Equatable, Sendable
{
    public let durationSeconds: Double
    public let before: ExactPrefixCacheSnapshot
    public let after: ExactPrefixCacheSnapshot

    public init(
        durationSeconds: Double,
        before: ExactPrefixCacheSnapshot,
        after: ExactPrefixCacheSnapshot
    ) {
        self.durationSeconds = durationSeconds
        self.before = before
        self.after = after
    }

    fileprivate func validate() throws {
        guard durationSeconds.isFinite, durationSeconds > 0 else {
            throw ExactPrefixProofEvidenceError.invalidWarmupDuration
        }
        guard before == after else {
            throw ExactPrefixProofEvidenceError.warmupMutatedCache
        }
    }
}

public struct ExactPrefixProofDerived:
    Codable, Equatable, Sendable
{
    public let byteIdentityPassed: Bool
    public let engagementPassed: Bool
    public let boundedPassed: Bool
    public let warmBenefitPassed: Bool
    public let promotable: Bool

    public init(
        byteIdentityPassed: Bool,
        engagementPassed: Bool,
        boundedPassed: Bool,
        warmBenefitPassed: Bool,
        promotable: Bool
    ) {
        self.byteIdentityPassed = byteIdentityPassed
        self.engagementPassed = engagementPassed
        self.boundedPassed = boundedPassed
        self.warmBenefitPassed = warmBenefitPassed
        self.promotable = promotable
    }
}

public struct ExactPrefixProofEvidence:
    Codable, Equatable, Sendable
{
    public static let currentSchemaVersion = 3

    public let schemaVersion: Int
    public let modelID: String
    public let sourceRevision: String
    public let admission: CompressedKVAttentionRuntimeAdmission
    public let expectedHarnessSHA: String
    public let expectedExecutableSHA256: String
    public let workloadNonce: String
    public let maxTokens: Int
    public let promptRepeat: Int
    public let exactPrefixCachePolicy: ExactPrefixCachePolicy
    public let templateTokenCachePolicy: TemplateTokenCachePolicy
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let requestContext: ExactPrefixRequestContext
    public let warmup: ExactPrefixProofWarmupEvidence
    public let cases: [ExactPrefixProofCaseEvidence]
    public let terminalNonPartial: Bool
    public let derived: ExactPrefixProofDerived

    public var byteIdentityPassed: Bool {
        derived.byteIdentityPassed
    }

    public var engagementPassed: Bool {
        derived.engagementPassed
    }

    public var boundedPassed: Bool {
        derived.boundedPassed
    }

    public var warmBenefitPassed: Bool {
        derived.warmBenefitPassed
    }

    public var promotable: Bool {
        derived.promotable
    }

    public init(
        modelID: String,
        sourceRevision: String,
        admission: CompressedKVAttentionRuntimeAdmission,
        expectedHarnessSHA: String,
        expectedExecutableSHA256: String,
        workloadNonce: String,
        maxTokens: Int,
        promptRepeat: Int,
        exactPrefixCachePolicy: ExactPrefixCachePolicy,
        templateTokenCachePolicy: TemplateTokenCachePolicy,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        requestContext: ExactPrefixRequestContext,
        warmup: ExactPrefixProofWarmupEvidence,
        cases: [ExactPrefixProofCaseEvidence],
        terminalNonPartial: Bool,
        derived: ExactPrefixProofDerived? = nil
    ) throws {
        self.schemaVersion = Self.currentSchemaVersion
        self.modelID = modelID
        self.sourceRevision = sourceRevision
        self.admission = admission
        self.expectedHarnessSHA = expectedHarnessSHA
        self.expectedExecutableSHA256 = expectedExecutableSHA256
        self.workloadNonce = workloadNonce
        self.maxTokens = maxTokens
        self.promptRepeat = promptRepeat
        self.exactPrefixCachePolicy = exactPrefixCachePolicy
        self.templateTokenCachePolicy = templateTokenCachePolicy
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.requestContext = requestContext
        self.warmup = warmup
        self.cases = cases
        self.terminalNonPartial = terminalNonPartial
        self.derived = try derived
            ?? Self.recomputeDerived(
                cases: cases,
                memoryLimitBytes: memoryLimitBytes,
                cacheLimitBytes: cacheLimitBytes,
                requiresRuntimeIdentity: true)
        try validated()
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case modelID
        case sourceRevision
        case admission
        case expectedHarnessSHA
        case expectedExecutableSHA256
        case workloadNonce
        case maxTokens
        case promptRepeat
        case exactPrefixCachePolicy
        case templateTokenCachePolicy
        case memoryLimitBytes
        case cacheLimitBytes
        case requestContext
        case warmup
        case cases
        case terminalNonPartial
        case derived
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try values.decode(
            Int.self,
            forKey: .schemaVersion)
        modelID = try values.decode(String.self, forKey: .modelID)
        sourceRevision = try values.decode(
            String.self,
            forKey: .sourceRevision)
        admission = try values.decode(
            CompressedKVAttentionRuntimeAdmission.self,
            forKey: .admission)
        expectedHarnessSHA = try values.decode(
            String.self,
            forKey: .expectedHarnessSHA)
        expectedExecutableSHA256 = try values.decode(
            String.self,
            forKey: .expectedExecutableSHA256)
        workloadNonce = try values.decode(
            String.self,
            forKey: .workloadNonce)
        maxTokens = try values.decode(Int.self, forKey: .maxTokens)
        promptRepeat = try values.decode(Int.self, forKey: .promptRepeat)
        exactPrefixCachePolicy = try values.decode(
            ExactPrefixCachePolicy.self,
            forKey: .exactPrefixCachePolicy)
        templateTokenCachePolicy = try values.decode(
            TemplateTokenCachePolicy.self,
            forKey: .templateTokenCachePolicy)
        memoryLimitBytes = try values.decode(
            Int.self,
            forKey: .memoryLimitBytes)
        cacheLimitBytes = try values.decode(
            Int.self,
            forKey: .cacheLimitBytes)
        requestContext = try values.decode(
            ExactPrefixRequestContext.self,
            forKey: .requestContext)
        warmup = try values.decode(
            ExactPrefixProofWarmupEvidence.self,
            forKey: .warmup)
        cases = try values.decode(
            [ExactPrefixProofCaseEvidence].self,
            forKey: .cases)
        terminalNonPartial = try values.decode(
            Bool.self,
            forKey: .terminalNonPartial)
        derived = try values.decode(
            ExactPrefixProofDerived.self,
            forKey: .derived)
        try validated()
    }

    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(schemaVersion, forKey: .schemaVersion)
        try values.encode(modelID, forKey: .modelID)
        try values.encode(sourceRevision, forKey: .sourceRevision)
        try values.encode(admission, forKey: .admission)
        try values.encode(expectedHarnessSHA, forKey: .expectedHarnessSHA)
        try values.encode(
            expectedExecutableSHA256,
            forKey: .expectedExecutableSHA256)
        try values.encode(workloadNonce, forKey: .workloadNonce)
        try values.encode(maxTokens, forKey: .maxTokens)
        try values.encode(promptRepeat, forKey: .promptRepeat)
        try values.encode(
            exactPrefixCachePolicy,
            forKey: .exactPrefixCachePolicy)
        try values.encode(
            templateTokenCachePolicy,
            forKey: .templateTokenCachePolicy)
        try values.encode(memoryLimitBytes, forKey: .memoryLimitBytes)
        try values.encode(cacheLimitBytes, forKey: .cacheLimitBytes)
        try values.encode(requestContext, forKey: .requestContext)
        try values.encode(warmup, forKey: .warmup)
        try values.encode(cases, forKey: .cases)
        try values.encode(
            terminalNonPartial,
            forKey: .terminalNonPartial)
        try values.encode(derived, forKey: .derived)
    }

    @discardableResult
    public func validated() throws -> Self {
        guard schemaVersion == 1
            || schemaVersion == 2
            || schemaVersion == Self.currentSchemaVersion
        else {
            throw ExactPrefixProofEvidenceError
                .unsupportedSchemaVersion(schemaVersion)
        }
        guard Self.pathFreeIdentifier(modelID) else {
            throw ExactPrefixProofEvidenceError.invalidModelID
        }
        guard Self.isLowercaseHex(sourceRevision, lengths: [64])
        else {
            throw ExactPrefixProofEvidenceError.invalidSourceRevision
        }
        guard Self.isLowercaseHex(expectedHarnessSHA, lengths: [40, 64])
        else {
            throw ExactPrefixProofEvidenceError.invalidExpectedHarnessSHA
        }
        guard requestStartIsLowercaseSHA256(expectedExecutableSHA256)
        else {
            throw ExactPrefixProofEvidenceError
                .invalidExpectedExecutableSHA256
        }
        do {
            _ = try admission.validatedForEvidence()
            _ = try ServiceWorkloadIdentity(nonce: workloadNonce)
        } catch let error as ExactPrefixProofEvidenceError {
            throw error
        } catch is ServiceWorkloadIdentityError {
            throw ExactPrefixProofEvidenceError.invalidWorkloadNonce
        } catch {
            throw error
        }
        guard sourceRevision == admission.checkpointContentSHA256 else {
            throw ExactPrefixProofEvidenceError
                .sourceRevisionIdentityMismatch
        }
        guard (2 ... 128).contains(maxTokens) else {
            throw ExactPrefixProofEvidenceError.invalidMaxTokens
        }
        guard (1 ... 256).contains(promptRepeat) else {
            throw ExactPrefixProofEvidenceError.invalidPromptRepeat
        }
        guard exactPrefixCachePolicy.isEnabled,
            (1 ... 64).contains(exactPrefixCachePolicy.maxEntries)
        else {
            throw ExactPrefixProofEvidenceError.invalidCachePolicy
        }
        guard templateTokenCachePolicy.isEnabled,
            (1 ... 64).contains(templateTokenCachePolicy.maxEntries)
        else {
            throw ExactPrefixProofEvidenceError
                .invalidTemplateTokenCachePolicy
        }
        let (configuredRetainedBytes, retainedBytesOverflow) =
            exactPrefixCachePolicy.maxRetainedBytes
            .addingReportingOverflow(
                templateTokenCachePolicy.maxRetainedBytes)
        guard memoryLimitBytes > 0,
            cacheLimitBytes > 0,
            cacheLimitBytes <= memoryLimitBytes,
            !retainedBytesOverflow,
            configuredRetainedBytes <= memoryLimitBytes
        else {
            throw ExactPrefixProofEvidenceError.invalidMemoryLimits
        }
        try warmup.validate()
        guard cases.map(\.caseID) == ExactPrefixProofCaseID.requiredOrder
        else {
            throw ExactPrefixProofEvidenceError.caseOrderMismatch
        }
        for row in cases {
            try row.validate(
                memoryLimitBytes: memoryLimitBytes,
                cacheLimitBytes: cacheLimitBytes)
        }
        if schemaVersion >= 3 {
            for row in cases {
                guard row.finalContextTokenIDsSHA256 != nil,
                    row.finalContextTokenCount != nil
                else {
                    throw ExactPrefixProofEvidenceError
                        .invalidFinalContext(row.caseID)
                }
            }
        }
        try validateReferenceGroups(cases)
        try validateCacheBindings(cases)
        try validateWarmupBindings(cases)
        guard terminalNonPartial else {
            throw ExactPrefixProofEvidenceError
                .partialEvidence(.postWarmupHit)
        }
        let recomputed = try Self.recomputeDerived(
            cases: cases,
            memoryLimitBytes: memoryLimitBytes,
            cacheLimitBytes: cacheLimitBytes,
            requiresRuntimeIdentity:
                schemaVersion >= 2)
        guard derived == recomputed else {
            throw ExactPrefixProofEvidenceError.derivedClaimMismatch
        }
        guard schemaVersion != 1 || !derived.promotable else {
            throw ExactPrefixProofEvidenceError
                .legacyPromotableEvidence
        }
        return self
    }

    private static func recomputeDerived(
        cases: [ExactPrefixProofCaseEvidence],
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        requiresRuntimeIdentity: Bool
    ) throws -> ExactPrefixProofDerived {
        let ordered = cases.map(\.caseID) == ExactPrefixProofCaseID.requiredOrder
        guard ordered else {
            return ExactPrefixProofDerived(
                byteIdentityPassed: false,
                engagementPassed: false,
                boundedPassed: false,
                warmBenefitPassed: false,
                promotable: false)
        }
        let byteIdentityPassed = ordered && cases.allSatisfy {
            $0.generatedTokenIDsSHA256
                == $0.referenceGeneratedTokenIDsSHA256
                && $0.outputSHA256 == $0.referenceOutputSHA256
        }
        let cacheRows = cases.filter { !$0.caseID.isControl }
        let runtimeIdentities = cacheRows.compactMap {
            $0.requestStartMetrics?.runtimeIdentity
        }
        let runtimeIdentityPassed =
            !requiresRuntimeIdentity
            || (runtimeIdentities.count == cacheRows.count
                && Set(runtimeIdentities).count == 1)
        let engagementPassed =
            ordered && runtimeIdentityPassed
            && cases.allSatisfy { row in
            guard row.templateTokenCacheReceipt
                == row.caseID.expectedTemplateReceipt
            else { return false }
            if row.caseID.isControl {
                return row.requestStartMetrics == nil
            }
            let requestStartMetrics = row.requestStartMetrics
            let expectedOutcome =
                requestStartMetrics?.prefixCacheOutcome
                == row.caseID.expectedPrefixOutcome
            let expectedTemplate =
                requestStartMetrics?.templateTokenCacheHit
                    == row.templateTokenCacheReceipt.templateHit
            let expectedEviction =
                row.caseID != .pressureEvictedA
                || (requestStartMetrics?.evictionCount ?? 0) > 0
            return expectedOutcome && expectedTemplate
                && expectedEviction
            }
        let boundedPassed = cases.allSatisfy { row in
            guard let memory = row.memoryEvidence else { return false }
            return memory.summary.maxMLXPeakBytes <= memoryLimitBytes
                && memory.summary.maxMLXCacheBytes <= cacheLimitBytes
                && memory.summary.maxSampledFootprintBytes
                    <= UInt64(memoryLimitBytes)
        }
        let warmBenefitPassed = try warmRequestStartsAreFaster(cases)
        return ExactPrefixProofDerived(
            byteIdentityPassed: byteIdentityPassed,
            engagementPassed: engagementPassed,
            boundedPassed: boundedPassed,
            warmBenefitPassed: warmBenefitPassed,
            promotable: byteIdentityPassed && engagementPassed
                && boundedPassed && warmBenefitPassed)
    }

    private func validateReferenceGroups(
        _ cases: [ExactPrefixProofCaseEvidence]
    ) throws {
        let rows = Dictionary(uniqueKeysWithValues: cases.map {
            ($0.caseID, $0)
        })
        let groups: [[ExactPrefixProofCaseID]] = [
            [
                .coldControlA,
                .coldCommitA,
                .exactHitA,
                .returnHitA,
                .pressureEvictedA,
            ],
            [.partialControl, .partialHit],
            [.coldCommitB],
            [.postWarmupControl, .postWarmupMiss, .postWarmupHit],
        ]
        for group in groups {
            guard let firstID = group.first, let first = rows[firstID]
            else { continue }
            guard first.referenceGeneratedTokenIDsSHA256
                == first.generatedTokenIDsSHA256,
                first.referenceOutputSHA256 == first.outputSHA256
            else {
                throw ExactPrefixProofEvidenceError
                    .referenceHashMismatch(firstID)
            }
            for id in group {
                guard let row = rows[id],
                    row.promptTokenIDsSHA256
                        == first.promptTokenIDsSHA256,
                    row.promptTokenCount == first.promptTokenCount
                else {
                    throw ExactPrefixProofEvidenceError
                        .groupReferenceMismatch(id)
                }
                guard row.referenceGeneratedTokenIDsSHA256
                    == first.generatedTokenIDsSHA256,
                    row.referenceOutputSHA256 == first.outputSHA256
                else {
                    throw ExactPrefixProofEvidenceError
                        .referenceHashMismatch(id)
                }
            }
        }
    }

    private func validateWarmupBindings(
        _ cases: [ExactPrefixProofCaseEvidence]
    ) throws {
        for row in cases where !row.caseID.isControl {
            let actual = row.requestStartMetrics?
                .eagerWarmupSeconds
            switch row.caseID {
            case .postWarmupMiss, .postWarmupHit:
                guard actual == warmup.durationSeconds else {
                    throw ExactPrefixProofEvidenceError
                        .warmupDurationMismatch(row.caseID)
                }
            case .coldCommitA, .exactHitA, .partialHit,
                .coldCommitB, .returnHitA, .pressureEvictedA:
                guard actual == nil else {
                    throw ExactPrefixProofEvidenceError
                        .warmupDurationMismatch(row.caseID)
                }
            case .coldControlA, .partialControl,
                .postWarmupControl:
                break
            }
        }
    }

    private func validateCacheBindings(
        _ cases: [ExactPrefixProofCaseEvidence]
    ) throws {
        let rows = Dictionary(uniqueKeysWithValues: cases.map {
            ($0.caseID, $0)
        })
        for row in cases where !row.caseID.isControl {
            guard row.requestContext == requestContext else {
                throw ExactPrefixProofEvidenceError
                    .requestContextMismatch(row.caseID)
            }
        }
        let expectedReusablePrefixes: [
            ExactPrefixProofCaseID: ExactPrefixProofCaseID
        ] = [
            .exactHitA: .coldCommitA,
            .partialHit: .coldCommitA,
            .returnHitA: .coldCommitA,
            .pressureEvictedA: .coldCommitA,
            .postWarmupHit: .postWarmupMiss,
        ]
        for (hitID, sourceID) in expectedReusablePrefixes {
            guard let hit = rows[hitID], let source = rows[sourceID]
            else {
                throw ExactPrefixProofEvidenceError
                    .reusedPrefixMismatch(hitID)
            }
            guard hit.requestStartMetrics?.prefixCacheOutcome
                == .exactHit
                || hit.requestStartMetrics?.prefixCacheOutcome
                    == .partialHit
            else {
                continue
            }
            if schemaVersion < 3 {
                guard hit.reusedPrefixTokenIDsSHA256
                    == source.promptTokenIDsSHA256,
                    hit.reusedPrefixTokenCount
                        == source.promptTokenCount
                else {
                    throw ExactPrefixProofEvidenceError
                        .reusedPrefixMismatch(hitID)
                }
                continue
            }
            guard hit.reusedPrefixSourceCaseID == sourceID,
                let reusedPrefixSourceKind =
                    hit.reusedPrefixSourceKind
            else {
                throw ExactPrefixProofEvidenceError
                    .reusedPrefixMismatch(hitID)
            }
            switch reusedPrefixSourceKind {
            case .promptOnly:
                guard hit.reusedPrefixTokenIDsSHA256
                    == source.promptTokenIDsSHA256,
                    hit.reusedPrefixTokenCount
                        == source.promptTokenCount
                else {
                    throw ExactPrefixProofEvidenceError
                        .reusedPrefixMismatch(hitID)
                }
            case .finalContext:
                guard hit.reusedPrefixTokenIDsSHA256
                    == source.finalContextTokenIDsSHA256,
                    hit.reusedPrefixTokenCount
                        == source.finalContextTokenCount
                else {
                    throw ExactPrefixProofEvidenceError
                        .reusedPrefixMismatch(hitID)
                }
            }
        }
    }

    private static func warmRequestStartsAreFaster(
        _ cases: [ExactPrefixProofCaseEvidence]
    ) throws -> Bool {
        let rows = Dictionary(uniqueKeysWithValues: cases.map {
            ($0.caseID, $0)
        })
        let pairs: [(
            control: ExactPrefixProofCaseID,
            warm: ExactPrefixProofCaseID
        )] = [
            (.coldControlA, .exactHitA),
            (.pressureEvictedA, .exactHitA),
            (.partialControl, .partialHit),
            (.coldControlA, .returnHitA),
            (.pressureEvictedA, .returnHitA),
            (.postWarmupControl, .postWarmupHit),
            (.postWarmupMiss, .postWarmupHit),
        ]
        for pair in pairs {
            guard let control = rows[pair.control]?
                .timing.requestStartSeconds,
                let warm = rows[pair.warm]?
                    .timing.requestStartSeconds
            else {
                throw ExactPrefixProofEvidenceError.caseOrderMismatch
            }
            guard warm < control else {
                return false
            }
        }
        return true
    }

    fileprivate static func isLowercaseHex(
        _ value: String,
        lengths: Set<Int>
    ) -> Bool {
        lengths.contains(value.utf8.count)
            && value.utf8.allSatisfy {
                (48 ... 57).contains($0) || (97 ... 102).contains($0)
            }
    }

    fileprivate static func pathFreeIdentifier(_ value: String) -> Bool {
        let bytes = value.utf8
        return !bytes.isEmpty && bytes.count <= 128
            && bytes.allSatisfy {
                (33 ... 126).contains($0) && $0 != 47 && $0 != 92
            }
    }
}

public struct ExactPrefixProofCommandPlan:
    Equatable, Sendable
{
    public let modelID: String
    public let sourceRevision: String
    public let expectedHarnessSHA: String
    public let expectedExecutableSHA256: String
    public let admission: CompressedKVAttentionRuntimeAdmission
    public let checkpointContentSHA256: String
    public let tokenizerSHA256: String
    public let workloadNonce: String
    public let maxTokens: Int
    public let promptRepeat: Int
    public let exactPrefixCachePolicy: ExactPrefixCachePolicy
    public let templateTokenCachePolicy: TemplateTokenCachePolicy
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let outputPath: String

    public init(
        modelID: String,
        sourceRevision: String,
        expectedHarnessSHA: String,
        expectedExecutableSHA256: String,
        admission: CompressedKVAttentionRuntimeAdmission,
        checkpointContentSHA256: String,
        tokenizerSHA256: String,
        workloadNonce: String,
        maxTokens: Int,
        promptRepeat: Int,
        exactPrefixCachePolicy: ExactPrefixCachePolicy,
        templateTokenCachePolicy: TemplateTokenCachePolicy,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        outputPath: String
    ) throws {
        guard ExactPrefixProofEvidence.pathFreeIdentifier(modelID) else {
            throw ExactPrefixProofCommandPlanError.invalidModelID
        }
        guard ExactPrefixProofEvidence.isLowercaseHex(
            sourceRevision,
            lengths: [64])
        else {
            throw ExactPrefixProofCommandPlanError.invalidSourceRevision
        }
        guard ExactPrefixProofEvidence.isLowercaseHex(
            expectedHarnessSHA,
            lengths: [40, 64])
        else {
            throw ExactPrefixProofCommandPlanError
                .invalidExpectedHarnessSHA
        }
        guard requestStartIsLowercaseSHA256(expectedExecutableSHA256)
        else {
            throw ExactPrefixProofCommandPlanError
                .invalidExpectedExecutableSHA256
        }
        do {
            _ = try admission.validatedForEvidence()
        } catch {
            throw ExactPrefixProofCommandPlanError.invalidAdmission
        }
        guard sourceRevision == admission.checkpointContentSHA256 else {
            throw ExactPrefixProofCommandPlanError
                .sourceRevisionIdentityMismatch
        }
        guard requestStartIsLowercaseSHA256(checkpointContentSHA256)
        else {
            throw ExactPrefixProofCommandPlanError
                .invalidCheckpointContentSHA256
        }
        guard requestStartIsLowercaseSHA256(tokenizerSHA256) else {
            throw ExactPrefixProofCommandPlanError
                .invalidTokenizerSHA256
        }
        guard checkpointContentSHA256 == admission.checkpointContentSHA256
        else {
            throw ExactPrefixProofCommandPlanError
                .checkpointIdentityMismatch
        }
        guard tokenizerSHA256 == admission.tokenizerSHA256 else {
            throw ExactPrefixProofCommandPlanError
                .tokenizerIdentityMismatch
        }
        do {
            _ = try ServiceWorkloadIdentity(nonce: workloadNonce)
        } catch {
            throw ExactPrefixProofCommandPlanError.invalidWorkloadNonce
        }
        guard (2 ... 128).contains(maxTokens) else {
            throw ExactPrefixProofCommandPlanError.invalidMaxTokens
        }
        guard (1 ... 256).contains(promptRepeat) else {
            throw ExactPrefixProofCommandPlanError.invalidPromptRepeat
        }
        guard exactPrefixCachePolicy.isEnabled,
            (1 ... 64).contains(exactPrefixCachePolicy.maxEntries)
        else {
            throw ExactPrefixProofCommandPlanError.invalidCachePolicy
        }
        guard templateTokenCachePolicy.isEnabled,
            (1 ... 64).contains(templateTokenCachePolicy.maxEntries)
        else {
            throw ExactPrefixProofCommandPlanError
                .invalidTemplateTokenCachePolicy
        }
        let (configuredRetainedBytes, retainedBytesOverflow) =
            exactPrefixCachePolicy.maxRetainedBytes
            .addingReportingOverflow(
                templateTokenCachePolicy.maxRetainedBytes)
        guard memoryLimitBytes > 0, cacheLimitBytes > 0,
            cacheLimitBytes <= memoryLimitBytes,
            !retainedBytesOverflow,
            configuredRetainedBytes <= memoryLimitBytes
        else {
            throw ExactPrefixProofCommandPlanError.invalidMemoryLimits
        }
        guard outputPath == outputPath.trimmingCharacters(
            in: .whitespacesAndNewlines),
            !outputPath.isEmpty
        else {
            throw ExactPrefixProofCommandPlanError.invalidOutputPath
        }

        self.modelID = modelID
        self.sourceRevision = sourceRevision
        self.expectedHarnessSHA = expectedHarnessSHA
        self.expectedExecutableSHA256 = expectedExecutableSHA256
        self.admission = admission
        self.checkpointContentSHA256 = checkpointContentSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.workloadNonce = workloadNonce
        self.maxTokens = maxTokens
        self.promptRepeat = promptRepeat
        self.exactPrefixCachePolicy = exactPrefixCachePolicy
        self.templateTokenCachePolicy = templateTokenCachePolicy
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.outputPath = outputPath
    }
}
