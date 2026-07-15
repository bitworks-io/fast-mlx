import Foundation

public enum KVComparisonBaseline: String, Codable, Equatable, Sendable {
    case sameWeightsFP16KV = "same-weights-fp16-kv"
    case differentWeightsFP16KV = "different-weights-fp16-kv"
}

public enum KVStorageFormatKind: String, Codable, Equatable, Sendable {
    case fp16
    case affine
    case kvarn
}

public struct KVModelEvidenceIdentity: Codable, Equatable, Sendable {
    public let configHash: String
    public let checkpointManifestHash: String

    public init(configHash: String, checkpointManifestHash: String) {
        self.configHash = configHash
        self.checkpointManifestHash = checkpointManifestHash
    }
}

public struct KVFormatGeometryEvidence: Codable, Equatable, Sendable {
    public let kind: KVStorageFormatKind
    public let tier: String
    public let keyBits: Int
    public let valueBits: Int
    public let groupSize: Int
    public let sinkTokens: Int
    public let layerCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let capacityTokens: Int
    public let sequences: Int
    public let metadataScalarBytes: Int
    public let recordAlignment: Int

    public init(
        kind: KVStorageFormatKind,
        tier: String, keyBits: Int, valueBits: Int, groupSize: Int,
        sinkTokens: Int, layerCount: Int, kvHeadCount: Int, headDimension: Int,
        capacityTokens: Int, sequences: Int, metadataScalarBytes: Int,
        recordAlignment: Int
    ) {
        self.kind = kind
        self.tier = tier
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.groupSize = groupSize
        self.sinkTokens = sinkTokens
        self.layerCount = layerCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.capacityTokens = capacityTokens
        self.sequences = sequences
        self.metadataScalarBytes = metadataScalarBytes
        self.recordAlignment = recordAlignment
    }
}

/// One storage view. `predicted` comes from `KVStorageFormat`; `actual` is the sum of real cache
/// arrays after allocation. Keeping the same terms on both sides makes hidden metadata, padding,
/// sink/tail state, and workspace visible instead of collapsing everything into a nominal bit label.
public struct KVStorageBreakdownEvidence: Codable, Equatable, Sendable {
    public let payloadBytes: Int
    public let metadataBytes: Int
    public let alignmentPaddingBytes: Int
    public let fp16SinkBytes: Int
    public let fp16TailBytes: Int
    public let workspaceBytes: Int
    public let totalBytes: Int

    public init(
        payloadBytes: Int, metadataBytes: Int, alignmentPaddingBytes: Int,
        fp16SinkBytes: Int, fp16TailBytes: Int, workspaceBytes: Int,
        totalBytes: Int
    ) {
        self.payloadBytes = payloadBytes
        self.metadataBytes = metadataBytes
        self.alignmentPaddingBytes = alignmentPaddingBytes
        self.fp16SinkBytes = fp16SinkBytes
        self.fp16TailBytes = fp16TailBytes
        self.workspaceBytes = workspaceBytes
        self.totalBytes = totalBytes
    }
}

public struct KVStorageEvidence: Codable, Equatable, Sendable {
    public let predicted: KVStorageBreakdownEvidence
    public let actual: KVStorageBreakdownEvidence

    public init(
        predicted: KVStorageBreakdownEvidence,
        actual: KVStorageBreakdownEvidence
    ) {
        self.predicted = predicted
        self.actual = actual
    }
}

/// Context that turns a general teacher-forced KL row into a matrix cell eligible for a verdict.
/// Geometry/storage remain optional for backward-compatible exploratory rows, but the promotion
/// validator requires both and refuses any predicted-vs-actual mismatch. `storage` describes the
/// format data arrays that reconcile with `KVStorageFormat`; engine bookkeeping arrays are kept
/// visible in `actualControlBytes` instead of being hidden inside a nominal bit rate.
public struct KVFrontierEvidence: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let matrixID: String
    public let cellID: String
    public let sameWeights: Bool
    public let comparisonBaseline: KVComparisonBaseline
    public let referenceKVQuantTier: String
    public let candidateModel: KVModelEvidenceIdentity
    public let referenceModel: KVModelEvidenceIdentity
    public let candidateFormat: KVFormatGeometryEvidence?
    public let storage: KVStorageEvidence?
    public let actualControlBytes: Int?

    public init(
        schemaVersion: Int, matrixID: String, cellID: String,
        sameWeights: Bool, comparisonBaseline: KVComparisonBaseline,
        referenceKVQuantTier: String,
        candidateModel: KVModelEvidenceIdentity,
        referenceModel: KVModelEvidenceIdentity,
        candidateFormat: KVFormatGeometryEvidence?,
        storage: KVStorageEvidence?,
        actualControlBytes: Int? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.matrixID = matrixID
        self.cellID = cellID
        self.sameWeights = sameWeights
        self.comparisonBaseline = comparisonBaseline
        self.referenceKVQuantTier = referenceKVQuantTier
        self.candidateModel = candidateModel
        self.referenceModel = referenceModel
        self.candidateFormat = candidateFormat
        self.storage = storage
        self.actualControlBytes = actualControlBytes
    }
}

/// Per-corpus-entry sample count. Aggregate cohort totals alone cannot prove that every document
/// was actually sampled deeply enough for a stable loss estimate.
public struct KVEntryScoringEvidence: Codable, Equatable, Sendable {
    public let entryID: String
    public let scoredPositions: Int

    public init(entryID: String, scoredPositions: Int) {
        self.entryID = entryID
        self.scoredPositions = scoredPositions
    }
}

/// Pure form of the `kl` JSONL payload. Optional additions preserve decoding of historical rows;
/// every newly written row is validated and supplies top-1 plus frontier identity, while promotion
/// additionally requires the full geometry/storage contract.
public struct KLPayload: Codable, Equatable, Sendable {
    public static let minimumPromotionLongContextTokens = 24_000
    public static let minimumPromotionShortPositionsPerEntry = 24
    public static let minimumPromotionLongContextPositionsPerEntry = 128

    public let kvQuantTier: String
    public let klMedianNats: Double
    public let klLongContextTailP95Nats: Double?
    public let klPooledMedianNats: Double
    public let klPooledP95Nats: Double
    public let pplCandidate: Double
    public let pplReference: Double
    public let pplDeltaPct: Double
    public let totalPositions: Int
    public let entryCount: Int
    public let teacherForcedTop1AgreementCount: Int?
    public let teacherForcedTop1ScoredPositions: Int?
    public let teacherForcedTop1AgreementRate: Double?
    public let frontier: KVFrontierEvidence?
    public let shortEntryCount: Int?
    public let shortScoredPositions: Int?
    public let longContextEntryCount: Int?
    public let longContextScoredPositions: Int?
    public let shortEntryScoring: [KVEntryScoringEvidence]?
    public let longContextEntryScoring: [KVEntryScoringEvidence]?
    /// Deepest tokenized long document and deepest context at which a distribution was actually
    /// scored. These are distinct because bounded sampling can inspect a shallow position in a
    /// long file; promotion is based on measured depth, not the corpus tag or document length.
    public let longContextMaxDocumentTokens: Int?
    public let longContextMaxScoredContextTokens: Int?

    public init(
        kvQuantTier: String,
        klMedianNats: Double, klLongContextTailP95Nats: Double?,
        klPooledMedianNats: Double, klPooledP95Nats: Double,
        pplCandidate: Double, pplReference: Double, pplDeltaPct: Double,
        totalPositions: Int, entryCount: Int,
        teacherForcedTop1AgreementCount: Int?,
        teacherForcedTop1ScoredPositions: Int?,
        teacherForcedTop1AgreementRate: Double?,
        frontier: KVFrontierEvidence?,
        shortEntryCount: Int? = nil,
        shortScoredPositions: Int? = nil,
        longContextEntryCount: Int? = nil,
        longContextScoredPositions: Int? = nil,
        shortEntryScoring: [KVEntryScoringEvidence]? = nil,
        longContextEntryScoring: [KVEntryScoringEvidence]? = nil,
        longContextMaxDocumentTokens: Int? = nil,
        longContextMaxScoredContextTokens: Int? = nil
    ) {
        self.kvQuantTier = kvQuantTier
        self.klMedianNats = klMedianNats
        self.klLongContextTailP95Nats = klLongContextTailP95Nats
        self.klPooledMedianNats = klPooledMedianNats
        self.klPooledP95Nats = klPooledP95Nats
        self.pplCandidate = pplCandidate
        self.pplReference = pplReference
        self.pplDeltaPct = pplDeltaPct
        self.totalPositions = totalPositions
        self.entryCount = entryCount
        self.teacherForcedTop1AgreementCount = teacherForcedTop1AgreementCount
        self.teacherForcedTop1ScoredPositions = teacherForcedTop1ScoredPositions
        self.teacherForcedTop1AgreementRate = teacherForcedTop1AgreementRate
        self.frontier = frontier
        self.shortEntryCount = shortEntryCount
        self.shortScoredPositions = shortScoredPositions
        self.longContextEntryCount = longContextEntryCount
        self.longContextScoredPositions = longContextScoredPositions
        self.shortEntryScoring = shortEntryScoring
        self.longContextEntryScoring = longContextEntryScoring
        self.longContextMaxDocumentTokens = longContextMaxDocumentTokens
        self.longContextMaxScoredContextTokens = longContextMaxScoredContextTokens
    }
}

public enum KVFrontierEvidenceError: Error, Equatable, Sendable {
    case unsupportedSchema(Int)
    case invalidIdentifier(String)
    case invalidModelIdentity
    case inconsistentBaseline
    case invalidGeometry
    case invalidStorageBreakdown
    case storageArithmeticOverflow
    case missingFormat
    case missingStorage
    case missingControlStorage
    case invalidControlStorage
    case storageMismatch
    case storagePredictionMismatch
    case invalidMetric(String)
    case missingTop1Evidence
    case missingCohortEvidence
    case missingRequiredCohort(String)
    case missingCohortEntryEvidence
    case insufficientEntryPositions(
        cohort: String, entryID: String, got: Int, required: Int)
    case missingLongContextDepthEvidence
    case insufficientLongContextDepth(got: Int, required: Int)
    case missingLongContextTail
    case missingFrontier
    case tierMismatch
    case promotionRequiresSameWeights
    case qualityFloorFailed(String)
    case invalidPromotionProvenance(String)
    case candidateProvenanceMismatch
}

public extension KLPayload {
    @discardableResult
    func validatedForRecord() throws -> KLPayload {
        guard Self.isIdentifier(kvQuantTier) else {
            throw KVFrontierEvidenceError.invalidIdentifier(kvQuantTier)
        }
        let nonnegativeMetrics = [
            klMedianNats, klPooledMedianNats, klPooledP95Nats,
        ]
        guard nonnegativeMetrics.allSatisfy({ $0.isFinite && $0 >= 0 }),
            klLongContextTailP95Nats.map({ $0.isFinite && $0 >= 0 }) ?? true,
            pplCandidate.isFinite, pplCandidate > 0,
            pplReference.isFinite, pplReference > 0,
            pplDeltaPct.isFinite, totalPositions > 0, entryCount > 0
        else { throw KVFrontierEvidenceError.invalidMetric("quality") }

        let expectedDelta = (pplCandidate - pplReference) / pplReference * 100
        guard abs(expectedDelta - pplDeltaPct) <= 1e-9 * max(1, abs(expectedDelta)) else {
            throw KVFrontierEvidenceError.invalidMetric("pplDeltaPct")
        }
        guard let matches = teacherForcedTop1AgreementCount,
            let scored = teacherForcedTop1ScoredPositions,
            let rate = teacherForcedTop1AgreementRate
        else { throw KVFrontierEvidenceError.missingTop1Evidence }
        guard scored == totalPositions, matches >= 0, matches <= scored,
            rate.isFinite, rate >= 0, rate <= 1,
            abs(rate - Double(matches) / Double(scored)) <= 1e-12
        else { throw KVFrontierEvidenceError.invalidMetric("teacherForcedTop1") }
        guard let shortEntries = shortEntryCount,
            let shortPositions = shortScoredPositions,
            let longEntries = longContextEntryCount,
            let longPositions = longContextScoredPositions
        else { throw KVFrontierEvidenceError.missingCohortEvidence }
        guard shortEntries >= 0, longEntries >= 0,
            shortEntries <= entryCount, longEntries <= entryCount,
            shortEntries == entryCount - longEntries,
            shortPositions >= 0, longPositions >= 0,
            shortPositions <= totalPositions, longPositions <= totalPositions,
            shortPositions == totalPositions - longPositions
        else { throw KVFrontierEvidenceError.invalidMetric("cohorts") }
        switch (shortEntryScoring, longContextEntryScoring) {
        case (.none, .none):
            break
        case (.some(let shortDetail), .some(let longDetail)):
            guard shortDetail.count == shortEntries,
                longDetail.count == longEntries
            else { throw KVFrontierEvidenceError.invalidMetric("cohortEntryEvidence") }
            var identifiers = Set<String>()
            var detailTotals = [0, 0]
            for (cohortIndex, detail) in [shortDetail, longDetail].enumerated() {
                for entry in detail {
                    guard Self.isIdentifier(entry.entryID), entry.scoredPositions >= 0,
                        identifiers.insert(entry.entryID).inserted
                    else {
                        throw KVFrontierEvidenceError.invalidMetric("cohortEntryEvidence")
                    }
                    let (total, overflow) = detailTotals[cohortIndex]
                        .addingReportingOverflow(entry.scoredPositions)
                    guard !overflow else {
                        throw KVFrontierEvidenceError.invalidMetric("cohortEntryEvidence")
                    }
                    detailTotals[cohortIndex] = total
                }
            }
            guard detailTotals == [shortPositions, longPositions] else {
                throw KVFrontierEvidenceError.invalidMetric("cohortEntryEvidence")
            }
        default:
            throw KVFrontierEvidenceError.missingCohortEntryEvidence
        }
        guard let maxDocumentTokens = longContextMaxDocumentTokens,
            let maxScoredContextTokens = longContextMaxScoredContextTokens
        else { throw KVFrontierEvidenceError.missingLongContextDepthEvidence }
        guard maxDocumentTokens > 1, maxScoredContextTokens > 0,
            maxScoredContextTokens < maxDocumentTokens
        else { throw KVFrontierEvidenceError.invalidMetric("longContextDepth") }

        guard let frontier else { throw KVFrontierEvidenceError.missingFrontier }
        try frontier.validateForRecord(candidateTier: kvQuantTier)
        return self
    }

    @discardableResult
    func validatedForPromotion() throws -> KLPayload {
        try validatedForRecord()
        guard let longContextTail = klLongContextTailP95Nats else {
            throw KVFrontierEvidenceError.missingLongContextTail
        }
        guard let shortEntries = shortEntryCount, shortEntries > 0,
            let shortPositions = shortScoredPositions, shortPositions > 0
        else { throw KVFrontierEvidenceError.missingRequiredCohort("short") }
        guard let longEntries = longContextEntryCount, longEntries > 0,
            let longPositions = longContextScoredPositions, longPositions > 0
        else { throw KVFrontierEvidenceError.missingRequiredCohort("longContext") }
        guard let shortDetail = shortEntryScoring,
            let longDetail = longContextEntryScoring
        else { throw KVFrontierEvidenceError.missingCohortEntryEvidence }
        for (cohort, detail, required) in [
            ("short", shortDetail, Self.minimumPromotionShortPositionsPerEntry),
            ("longContext", longDetail,
             Self.minimumPromotionLongContextPositionsPerEntry),
        ] {
            for entry in detail where entry.scoredPositions < required {
                throw KVFrontierEvidenceError.insufficientEntryPositions(
                    cohort: cohort, entryID: entry.entryID,
                    got: entry.scoredPositions, required: required)
            }
        }
        guard let maxScoredContextTokens = longContextMaxScoredContextTokens,
            maxScoredContextTokens >= Self.minimumPromotionLongContextTokens
        else {
            throw KVFrontierEvidenceError.insufficientLongContextDepth(
                got: longContextMaxScoredContextTokens ?? 0,
                required: Self.minimumPromotionLongContextTokens)
        }
        guard pplCandidate / pplReference < 2 else {
            throw KVFrontierEvidenceError.qualityFloorFailed("perplexity")
        }
        guard longContextTail < 5 else {
            throw KVFrontierEvidenceError.qualityFloorFailed("longContextTailP95")
        }
        guard let top1Rate = teacherForcedTop1AgreementRate, top1Rate >= 0.5 else {
            throw KVFrontierEvidenceError.qualityFloorFailed("teacherForcedTop1")
        }
        guard let frontier else { throw KVFrontierEvidenceError.missingFrontier }
        try frontier.validateForPromotion(candidateTier: kvQuantTier)
        return self
    }

    fileprivate static func isIdentifier(_ value: String) -> Bool {
        guard !value.isEmpty, value == value.trimmingCharacters(in: .whitespacesAndNewlines),
            !value.contains("\n"), !value.contains("\r")
        else { return false }
        let punctuation = CharacterSet(charactersIn: "-._:/@+")
        return value.unicodeScalars.allSatisfy {
            CharacterSet.alphanumerics.contains($0) || punctuation.contains($0)
        }
    }
}

private extension KVFrontierEvidence {
    func validateForRecord(candidateTier: String) throws {
        guard schemaVersion == 1 else {
            throw KVFrontierEvidenceError.unsupportedSchema(schemaVersion)
        }
        guard KLPayload.isIdentifier(matrixID), KLPayload.isIdentifier(cellID) else {
            throw KVFrontierEvidenceError.invalidIdentifier("\(matrixID)/\(cellID)")
        }
        guard Self.isIdentity(candidateModel), Self.isIdentity(referenceModel) else {
            throw KVFrontierEvidenceError.invalidModelIdentity
        }
        guard referenceKVQuantTier == "fp16" else {
            throw KVFrontierEvidenceError.inconsistentBaseline
        }
        if sameWeights {
            guard comparisonBaseline == .sameWeightsFP16KV,
                candidateModel == referenceModel
            else { throw KVFrontierEvidenceError.inconsistentBaseline }
        } else {
            guard comparisonBaseline == .differentWeightsFP16KV else {
                throw KVFrontierEvidenceError.inconsistentBaseline
            }
        }
        if let actualControlBytes, actualControlBytes < 0 {
            throw KVFrontierEvidenceError.invalidControlStorage
        }

        switch (candidateFormat, storage) {
        case (.none, .none):
            break
        case (.some(let format), .some(let storage)):
            try format.validate()
            guard format.tier == candidateTier else {
                throw KVFrontierEvidenceError.tierMismatch
            }
            try storage.validate(requireExactMatch: false)
            let expected = try format.predictedStorage(
                workspaceBytes: storage.predicted.workspaceBytes)
            guard expected == storage.predicted else {
                throw KVFrontierEvidenceError.storagePredictionMismatch
            }
        case (.none, .some):
            throw KVFrontierEvidenceError.missingFormat
        case (.some, .none):
            throw KVFrontierEvidenceError.missingStorage
        }
    }

    func validateForPromotion(candidateTier: String) throws {
        try validateForRecord(candidateTier: candidateTier)
        guard sameWeights, comparisonBaseline == .sameWeightsFP16KV else {
            throw KVFrontierEvidenceError.promotionRequiresSameWeights
        }
        guard candidateFormat != nil else { throw KVFrontierEvidenceError.missingFormat }
        guard let storage else { throw KVFrontierEvidenceError.missingStorage }
        guard actualControlBytes != nil else {
            throw KVFrontierEvidenceError.missingControlStorage
        }
        try storage.validate(requireExactMatch: true)
    }

    static func isIdentity(_ identity: KVModelEvidenceIdentity) -> Bool {
        [identity.configHash, identity.checkpointManifestHash].allSatisfy {
            !$0.isEmpty && $0 != "unknown"
                && $0 == $0.trimmingCharacters(in: .whitespacesAndNewlines)
                && !$0.contains("\n") && !$0.contains("\r")
        }
    }
}

public extension KVFormatGeometryEvidence {
    /// Pair real runtime array bytes with the accountant's prediction for this exact geometry.
    /// The caller supplies measured terms only; the prediction is always recomputed here so an
    /// evidence writer cannot accidentally bless a fabricated predicted breakdown.
    func storageEvidence(
        actual: KVStorageBreakdownEvidence
    ) throws -> KVStorageEvidence {
        try actual.validate()
        return KVStorageEvidence(
            predicted: try predictedStorage(
                workspaceBytes: actual.workspaceBytes),
            actual: actual)
    }
}

private extension KVFormatGeometryEvidence {
    func validate() throws {
        _ = try predictedStorage(workspaceBytes: 0)
    }

    func predictedStorage(workspaceBytes: Int) throws -> KVStorageBreakdownEvidence {
        guard KLPayload.isIdentifier(tier), workspaceBytes >= 0,
            layerCount > 0, kvHeadCount > 0, headDimension > 0,
            capacityTokens > 0, sinkTokens >= 0, sinkTokens <= capacityTokens,
            sequences > 0, metadataScalarBytes >= 0, recordAlignment > 0,
            (recordAlignment & (recordAlignment - 1)) == 0
        else { throw KVFrontierEvidenceError.invalidGeometry }

        let expectedTier: String
        let format: KVStorageFormat
        switch kind {
        case .fp16:
            expectedTier = "fp16"
            guard keyBits == 16, valueBits == 16, groupSize == 1,
                sinkTokens == 0, metadataScalarBytes == 0, recordAlignment == 1
            else { throw KVFrontierEvidenceError.invalidGeometry }
            format = .fp16
        case .affine:
            expectedTier = "affine-k\(keyBits)v\(valueBits)-g\(groupSize)"
            guard sinkTokens == 0, recordAlignment == 1 else {
                throw KVFrontierEvidenceError.invalidGeometry
            }
            format = .affine(
                keyBits: keyBits, valueBits: valueBits, groupSize: groupSize,
                metadataScalarBytes: metadataScalarBytes)
        case .kvarn:
            expectedTier = "kvarn-k\(keyBits)v\(valueBits)-g\(groupSize)"
            format = .kvarn(
                keyBits: keyBits, valueBits: valueBits, groupSize: groupSize,
                sinkTokens: sinkTokens, metadataScalarBytes: metadataScalarBytes,
                alignment: recordAlignment)
        }
        guard tier == expectedTier else { throw KVFrontierEvidenceError.invalidGeometry }

        do {
            let allocation = try format.allocation(
                geometry: KVStorageGeometry(
                    layerCount: layerCount, kvHeadCount: kvHeadCount,
                    headDimension: headDimension),
                capacityTokens: capacityTokens, sequences: sequences,
                workspaceBytes: workspaceBytes)
            return KVStorageBreakdownEvidence(
                payloadBytes: allocation.payloadBytes,
                metadataBytes: allocation.metadataBytes,
                alignmentPaddingBytes: allocation.alignmentPaddingBytes,
                fp16SinkBytes: allocation.fp16SinkBytes,
                fp16TailBytes: allocation.fp16TailBytes,
                workspaceBytes: allocation.workspaceBytes,
                totalBytes: allocation.totalBytes)
        } catch KVStorageFormatError.arithmeticOverflow {
            throw KVFrontierEvidenceError.storageArithmeticOverflow
        } catch {
            throw KVFrontierEvidenceError.invalidGeometry
        }
    }
}

private extension KVStorageEvidence {
    func validate(requireExactMatch: Bool) throws {
        try predicted.validate()
        try actual.validate()
        if requireExactMatch, predicted != actual {
            throw KVFrontierEvidenceError.storageMismatch
        }
    }
}

private extension KVStorageBreakdownEvidence {
    func validate() throws {
        let terms = [
            payloadBytes, metadataBytes, alignmentPaddingBytes,
            fp16SinkBytes, fp16TailBytes, workspaceBytes,
        ]
        guard terms.allSatisfy({ $0 >= 0 }), totalBytes > 0 else {
            throw KVFrontierEvidenceError.invalidStorageBreakdown
        }
        var sum = 0
        for term in terms {
            let (next, overflow) = sum.addingReportingOverflow(term)
            guard !overflow else { throw KVFrontierEvidenceError.storageArithmeticOverflow }
            sum = next
        }
        guard sum == totalBytes else {
            throw KVFrontierEvidenceError.invalidStorageBreakdown
        }
    }
}

public extension ResultRecord where Payload == KLPayload {
    /// Validates the payload together with the provenance fields that live in its JSONL envelope.
    /// Exploratory records may come from a dirty/local tree, but they still must bind the candidate
    /// config and the exact measurement corpus instead of preserving internally contradictory rows.
    @discardableResult
    func validatedForRecordEvidence() throws -> Self {
        try payload.validatedForRecord()
        // Optional decoding keeps historical rows readable, but every newly persisted envelope
        // must carry the detail needed to audit how deeply each corpus entry was scored.
        guard payload.shortEntryScoring != nil,
            payload.longContextEntryScoring != nil
        else { throw KVFrontierEvidenceError.missingCohortEntryEvidence }
        guard subcommand == "kl" else {
            throw KVFrontierEvidenceError.invalidPromotionProvenance("subcommand")
        }
        guard let frontier = payload.frontier,
            provenance.modelConfigHash == frontier.candidateModel.configHash,
            provenance.modelCheckpointManifestHash
                == frontier.candidateModel.checkpointManifestHash
        else { throw KVFrontierEvidenceError.candidateProvenanceMismatch }
        guard Self.isEvidenceValue(provenance.corpusId),
            Self.isEvidenceValue(provenance.corpusContentHash)
        else { throw KVFrontierEvidenceError.invalidPromotionProvenance("corpus") }
        return self
    }

    /// Promotion evidence additionally requires a clean committed tree and complete runtime
    /// identity. Task-domain coherence predicates remain a separate Phase 4 artifact, but the
    /// three predeclared teacher-forced hard-floor predicates are enforced by the payload.
    @discardableResult
    func validatedForPromotionEvidence() throws -> Self {
        try payload.validatedForPromotion()
        try validatedForRecordEvidence()
        guard Self.isCleanGitSHA(provenance.harnessGitSHA) else {
            throw KVFrontierEvidenceError.invalidPromotionProvenance("harnessGitSHA")
        }
        guard provenance.hardwareRAMBytes > 0,
            Self.isEvidenceValue(provenance.hardwareChip),
            Self.isEvidenceValue(provenance.hardwareOS),
            Self.isEvidenceValue(provenance.mlxSwiftVersion),
            Self.isEvidenceValue(provenance.referenceMLXVersion),
            Self.isEvidenceValue(provenance.referenceMLXLMVersion),
            Self.isEvidenceValue(provenance.modelPath),
            Self.isEvidenceValue(provenance.nonce),
            Self.isEvidenceValue(Optional(provenance.date))
        else { throw KVFrontierEvidenceError.invalidPromotionProvenance("runtime") }
        return self
    }

    private static func isEvidenceValue(_ value: String?) -> Bool {
        guard let value else { return false }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.isEmpty && trimmed == value && trimmed != "unknown"
            && !trimmed.contains("\n") && !trimmed.contains("\r")
    }

    private static func isCleanGitSHA(_ value: String) -> Bool {
        guard [40, 64].contains(value.count) else { return false }
        return value.unicodeScalars.allSatisfy {
            CharacterSet(charactersIn: "0123456789abcdefABCDEF").contains($0)
        }
    }
}

/// The KL-specific writer makes validation and persistence one operation. Callers cannot select
/// the promotion path and accidentally append the row with only the generic JSONL writer.
public enum RequiredKLEvidenceWriter {
    public static func append(
        _ record: ResultRecord<KLPayload>, to url: URL, promotion: Bool
    ) throws {
        if promotion {
            try record.validatedForPromotionEvidence()
        } else {
            try record.validatedForRecordEvidence()
        }
        try RequiredJSONLWriter.append(record, to: url)
    }
}
