import Foundation

public enum CompressedAttentionProbeEvidenceError:
    Error, Equatable, Sendable
{
    case unsupportedSchema(Int)
    case invalidPlanIdentity
    case invalidIdentity(String)
    case invalidArtifactID
    case invalidRunCount(expected: Int, actual: Int)
    case invalidRunPosition
    case duplicateRunPosition(block: Int, position: Int)
    case incompletePairedBlock(Int)
    case counterbalanceMismatch(
        block: Int, expectedCandidatePosition: Int)
    case operationMismatch
    case missingReceipts
    case invalidTiming
    case invalidByteAccounting
    case invalidMemoryCounters
    case invalidMemorySettings
    case invalidPowerState
    case invalidThermalState
    case pairedBlockIdentityMismatch(Int)
    case invalidHash(String)
    case invalidMetric(String)
    case packedStructuralMismatch
    case fp16ControlMismatch
}

public enum CompressedAttentionProbeLayoutKind:
    String, Codable, Equatable, Sendable
{
    case fp16
    case affine
    case kvarn
}

/// Codable projection of `CompressedAttentionProbeLayout`. It keeps the public evidence artifact
/// pure while validating through the same layout rules as `CompressedAttentionProbePlan`.
public struct CompressedAttentionProbeLayoutIdentity:
    Codable, Equatable, Sendable
{
    public let kind: CompressedAttentionProbeLayoutKind
    public let keyBits: Int?
    public let valueBits: Int?
    public let keyGroupSize: Int?
    public let valueGroupSize: Int?
    public let groupSize: Int?
    public let sinkTokens: Int?
    public let iterations: Int?

    public init(
        kind: CompressedAttentionProbeLayoutKind,
        keyBits: Int? = nil,
        valueBits: Int? = nil,
        keyGroupSize: Int? = nil,
        valueGroupSize: Int? = nil,
        groupSize: Int? = nil,
        sinkTokens: Int? = nil,
        iterations: Int? = nil
    ) {
        self.kind = kind
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.keyGroupSize = keyGroupSize
        self.valueGroupSize = valueGroupSize
        self.groupSize = groupSize
        self.sinkTokens = sinkTokens
        self.iterations = iterations
    }

    public init(layout: CompressedAttentionProbeLayout) {
        switch layout {
        case .fp16:
            self.init(kind: .fp16)
        case let .affine(keyBits, valueBits, keyGroupSize, valueGroupSize):
            self.init(
                kind: .affine,
                keyBits: keyBits,
                valueBits: valueBits,
                keyGroupSize: keyGroupSize,
                valueGroupSize: valueGroupSize)
        case let .kvarn(
            keyBits, valueBits, groupSize, sinkTokens, iterations):
            self.init(
                kind: .kvarn,
                keyBits: keyBits,
                valueBits: valueBits,
                groupSize: groupSize,
                sinkTokens: sinkTokens,
                iterations: iterations)
        }
    }

    public func materializedLayout() throws -> CompressedAttentionProbeLayout {
        switch kind {
        case .fp16:
            guard keyBits == nil, valueBits == nil, keyGroupSize == nil,
                valueGroupSize == nil, groupSize == nil,
                sinkTokens == nil, iterations == nil
            else { throw CompressedAttentionProbeEvidenceError.invalidPlanIdentity }
            return .fp16
        case .affine:
            guard let keyBits, let valueBits, let keyGroupSize,
                let valueGroupSize, groupSize == nil, sinkTokens == nil,
                iterations == nil
            else { throw CompressedAttentionProbeEvidenceError.invalidPlanIdentity }
            return .affine(
                keyBits: keyBits,
                valueBits: valueBits,
                keyGroupSize: keyGroupSize,
                valueGroupSize: valueGroupSize)
        case .kvarn:
            guard let keyBits, let valueBits, keyGroupSize == nil,
                valueGroupSize == nil, let groupSize, let sinkTokens,
                let iterations
            else { throw CompressedAttentionProbeEvidenceError.invalidPlanIdentity }
            return .kvarn(
                keyBits: keyBits,
                valueBits: valueBits,
                groupSize: groupSize,
                sinkTokens: sinkTokens,
                iterations: iterations)
        }
    }
}

public struct CompressedAttentionProbePlanIdentity:
    Codable, Equatable, Sendable
{
    public let operation: CompressedAttentionProbeOperation
    public let contextTokens: Int
    public let queryTokens: Int
    public let prefillChunkTokens: Int
    public let outputTokens: Int
    public let stopTokenIDs: [Int]
    public let batchSize: Int
    public let queryHeadCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let dtype: CompressedAttentionProbeDType
    public let mask: CompressedAttentionProbeMask
    public let layout: CompressedAttentionProbeLayoutIdentity
    public let warmupRuns: Int
    public let measuredRuns: Int
    public let seed: Int
    public let workloadNonce: String
    public let harnessGitSHA: String
    public let promotionEvidence: Bool
    public let evidenceOutputPath: String
    public let progressOutputPath: String

    public init(
        operation: CompressedAttentionProbeOperation,
        contextTokens: Int,
        queryTokens: Int,
        prefillChunkTokens: Int,
        outputTokens: Int,
        stopTokenIDs: [Int],
        batchSize: Int,
        queryHeadCount: Int,
        kvHeadCount: Int,
        headDimension: Int,
        dtype: CompressedAttentionProbeDType,
        mask: CompressedAttentionProbeMask,
        layout: CompressedAttentionProbeLayoutIdentity,
        warmupRuns: Int,
        measuredRuns: Int,
        seed: Int,
        workloadNonce: String,
        harnessGitSHA: String,
        promotionEvidence: Bool,
        evidenceOutputPath: String,
        progressOutputPath: String
    ) {
        self.operation = operation
        self.contextTokens = contextTokens
        self.queryTokens = queryTokens
        self.prefillChunkTokens = prefillChunkTokens
        self.outputTokens = outputTokens
        self.stopTokenIDs = stopTokenIDs
        self.batchSize = batchSize
        self.queryHeadCount = queryHeadCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.dtype = dtype
        self.mask = mask
        self.layout = layout
        self.warmupRuns = warmupRuns
        self.measuredRuns = measuredRuns
        self.seed = seed
        self.workloadNonce = workloadNonce
        self.harnessGitSHA = harnessGitSHA
        self.promotionEvidence = promotionEvidence
        self.evidenceOutputPath = evidenceOutputPath
        self.progressOutputPath = progressOutputPath
    }

    public init(plan: CompressedAttentionProbePlan) {
        self.init(
            operation: plan.operation,
            contextTokens: plan.contextTokens,
            queryTokens: plan.queryTokens,
            prefillChunkTokens: plan.prefillChunkTokens,
            outputTokens: plan.outputTokens,
            stopTokenIDs: plan.stopTokenIDs,
            batchSize: plan.batchSize,
            queryHeadCount: plan.queryHeadCount,
            kvHeadCount: plan.kvHeadCount,
            headDimension: plan.headDimension,
            dtype: plan.dtype,
            mask: plan.mask,
            layout: CompressedAttentionProbeLayoutIdentity(
                layout: plan.layout),
            warmupRuns: plan.warmupRuns,
            measuredRuns: plan.measuredRuns,
            seed: plan.seed,
            workloadNonce: plan.workloadNonce,
            harnessGitSHA: plan.harnessGitSHA,
            promotionEvidence: plan.promotionEvidence,
            evidenceOutputPath: plan.evidenceOutputPath,
            progressOutputPath: plan.progressOutputPath)
    }

    public func materializedPlan() throws -> CompressedAttentionProbePlan {
        do {
            return try CompressedAttentionProbePlan(
                operation: operation,
                contextTokens: contextTokens,
                queryTokens: queryTokens,
                prefillChunkTokens: prefillChunkTokens,
                outputTokens: outputTokens,
                stopTokenIDs: stopTokenIDs,
                batchSize: batchSize,
                queryHeadCount: queryHeadCount,
                kvHeadCount: kvHeadCount,
                headDimension: headDimension,
                dtype: dtype,
                mask: mask,
                layout: try layout.materializedLayout(),
                warmupRuns: warmupRuns,
                measuredRuns: measuredRuns,
                seed: seed,
                workloadNonce: workloadNonce,
                harnessGitSHA: harnessGitSHA,
                promotionEvidence: promotionEvidence,
                evidenceOutputPath: evidenceOutputPath,
                progressOutputPath: progressOutputPath)
        } catch {
            throw CompressedAttentionProbeEvidenceError.invalidPlanIdentity
        }
    }
}

public struct CompressedAttentionProbeModelIdentity:
    Codable, Equatable, Sendable
{
    public let modelID: String
    public let modelConfigSHA256: String
    public let checkpointManifestSHA256: String
    public let tokenizerSHA256: String
    public let tokenizerConfigSHA256: String

    public init(
        modelID: String,
        modelConfigSHA256: String,
        checkpointManifestSHA256: String,
        tokenizerSHA256: String,
        tokenizerConfigSHA256: String
    ) {
        self.modelID = modelID
        self.modelConfigSHA256 = modelConfigSHA256
        self.checkpointManifestSHA256 = checkpointManifestSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.tokenizerConfigSHA256 = tokenizerConfigSHA256
    }
}

public struct CompressedAttentionProbePackageIdentity:
    Codable, Equatable, Sendable
{
    public let mlxSwiftVersion: String
    public let mlxSwiftLMRevision: String
    public let swiftVersion: String
    public let harnessBuildConfiguration: String

    public init(
        mlxSwiftVersion: String,
        mlxSwiftLMRevision: String,
        swiftVersion: String,
        harnessBuildConfiguration: String
    ) {
        self.mlxSwiftVersion = mlxSwiftVersion
        self.mlxSwiftLMRevision = mlxSwiftLMRevision
        self.swiftVersion = swiftVersion
        self.harnessBuildConfiguration = harnessBuildConfiguration
    }
}

public enum CompressedAttentionProbeRunRole:
    String, Codable, Equatable, Sendable
{
    case candidate
    case fp16Reference = "fp16-reference"
}

public struct CompressedAttentionProbeRunPosition:
    Codable, Equatable, Hashable, Sendable
{
    public let pairedBlockIndex: Int
    public let runPosition: Int

    public init(pairedBlockIndex: Int, runPosition: Int) {
        self.pairedBlockIndex = pairedBlockIndex
        self.runPosition = runPosition
    }
}

public struct CompressedAttentionProbeTiming:
    Codable, Equatable, Sendable
{
    public let monotonicStartSeconds: Double
    public let monotonicEndSeconds: Double
    public let wallClockSeconds: Double
    public let attentionSeconds: Double

    public init(
        monotonicStartSeconds: Double,
        monotonicEndSeconds: Double,
        wallClockSeconds: Double,
        attentionSeconds: Double
    ) {
        self.monotonicStartSeconds = monotonicStartSeconds
        self.monotonicEndSeconds = monotonicEndSeconds
        self.wallClockSeconds = wallClockSeconds
        self.attentionSeconds = attentionSeconds
    }
}

public struct CompressedAttentionProbeByteReceipts:
    Codable, Equatable, Sendable
{
    public let payloadBytes: Int
    public let scaleBytes: Int
    public let biasBytes: Int
    public let controlBytes: Int
    public let alignmentPaddingBytes: Int
    public let fp16ResidentBytes: Int
    public let persistentKVBytes: Int
    public let materializationBytes: Int
    public let otherWorkspaceBytes: Int
    public let peakTemporaryBytes: Int
    public let totalBytes: Int

    public init(
        payloadBytes: Int,
        scaleBytes: Int,
        biasBytes: Int,
        controlBytes: Int,
        alignmentPaddingBytes: Int,
        fp16ResidentBytes: Int,
        persistentKVBytes: Int,
        materializationBytes: Int,
        otherWorkspaceBytes: Int,
        peakTemporaryBytes: Int,
        totalBytes: Int
    ) {
        self.payloadBytes = payloadBytes
        self.scaleBytes = scaleBytes
        self.biasBytes = biasBytes
        self.controlBytes = controlBytes
        self.alignmentPaddingBytes = alignmentPaddingBytes
        self.fp16ResidentBytes = fp16ResidentBytes
        self.persistentKVBytes = persistentKVBytes
        self.materializationBytes = materializationBytes
        self.otherWorkspaceBytes = otherWorkspaceBytes
        self.peakTemporaryBytes = peakTemporaryBytes
        self.totalBytes = totalBytes
    }
}

public struct CompressedAttentionProbeMLXMemorySnapshot:
    Codable, Equatable, Sendable
{
    public let activeBytes: Int
    public let cacheBytes: Int
    public let peakBytes: Int

    public init(activeBytes: Int, cacheBytes: Int, peakBytes: Int) {
        self.activeBytes = activeBytes
        self.cacheBytes = cacheBytes
        self.peakBytes = peakBytes
    }
}

public struct CompressedAttentionProbeMLXMemoryReceipts:
    Codable, Equatable, Sendable
{
    public let before: CompressedAttentionProbeMLXMemorySnapshot
    public let after: CompressedAttentionProbeMLXMemorySnapshot

    public init(
        before: CompressedAttentionProbeMLXMemorySnapshot,
        after: CompressedAttentionProbeMLXMemorySnapshot
    ) {
        self.before = before
        self.after = after
    }
}

public struct CompressedAttentionProbeProcessRSS:
    Codable, Equatable, Sendable
{
    public let residentSizeBeforeBytes: Int
    public let residentSizeAfterBytes: Int
    public let physicalFootprintBeforeBytes: Int
    public let physicalFootprintAfterBytes: Int

    public init(
        residentSizeBeforeBytes: Int,
        residentSizeAfterBytes: Int,
        physicalFootprintBeforeBytes: Int,
        physicalFootprintAfterBytes: Int
    ) {
        self.residentSizeBeforeBytes = residentSizeBeforeBytes
        self.residentSizeAfterBytes = residentSizeAfterBytes
        self.physicalFootprintBeforeBytes = physicalFootprintBeforeBytes
        self.physicalFootprintAfterBytes = physicalFootprintAfterBytes
    }
}

public enum CompressedAttentionProbeCacheResetPolicy:
    String, Codable, Equatable, Sendable
{
    case preserveAcrossPair = "preserve-across-pair"
    case clearBeforePair = "clear-before-pair"
}

public struct CompressedAttentionProbeMemorySettings:
    Codable, Equatable, Sendable
{
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let wiredLimitBytes: Int
    public let cacheResetPolicy: CompressedAttentionProbeCacheResetPolicy

    public init(
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        wiredLimitBytes: Int,
        cacheResetPolicy: CompressedAttentionProbeCacheResetPolicy
    ) {
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.wiredLimitBytes = wiredLimitBytes
        self.cacheResetPolicy = cacheResetPolicy
    }
}

public enum CompressedAttentionProbePowerSource:
    String, Codable, Equatable, Sendable
{
    case acPower = "ac-power"
    case battery
    case unavailable
}

public struct CompressedAttentionProbePowerReceipts:
    Codable, Equatable, Sendable
{
    public let lowPowerModeEnabledBefore: Bool
    public let lowPowerModeEnabledAfter: Bool
    public let powerSourceBefore: CompressedAttentionProbePowerSource
    public let powerSourceAfter: CompressedAttentionProbePowerSource

    public init(
        lowPowerModeEnabledBefore: Bool,
        lowPowerModeEnabledAfter: Bool,
        powerSourceBefore: CompressedAttentionProbePowerSource,
        powerSourceAfter: CompressedAttentionProbePowerSource
    ) {
        self.lowPowerModeEnabledBefore = lowPowerModeEnabledBefore
        self.lowPowerModeEnabledAfter = lowPowerModeEnabledAfter
        self.powerSourceBefore = powerSourceBefore
        self.powerSourceAfter = powerSourceAfter
    }
}

public enum CompressedAttentionProbeThermalState:
    String, Codable, Equatable, Sendable
{
    case nominal
    case fair
    case serious
    case critical
    case unknown
}

public struct CompressedAttentionProbeThermalReceipts:
    Codable, Equatable, Sendable
{
    public let before: CompressedAttentionProbeThermalState
    public let after: CompressedAttentionProbeThermalState

    public init(
        before: CompressedAttentionProbeThermalState,
        after: CompressedAttentionProbeThermalState
    ) {
        self.before = before
        self.after = after
    }
}

public struct CompressedAttentionProbeNumericControls:
    Codable, Equatable, Sendable
{
    public let packedMaxAbsoluteError: Double
    public let packedMaxRelativeError: Double
    public let packedTop1TokenID: Int
    public let unpackedTop1TokenID: Int
    public let fp16MaxAbsoluteError: Double
    public let fp16MaxRelativeError: Double
    public let fp16Top1TokenID: Int
    public let referenceTop1TokenID: Int

    public init(
        packedMaxAbsoluteError: Double,
        packedMaxRelativeError: Double,
        packedTop1TokenID: Int,
        unpackedTop1TokenID: Int,
        fp16MaxAbsoluteError: Double,
        fp16MaxRelativeError: Double,
        fp16Top1TokenID: Int,
        referenceTop1TokenID: Int
    ) {
        self.packedMaxAbsoluteError = packedMaxAbsoluteError
        self.packedMaxRelativeError = packedMaxRelativeError
        self.packedTop1TokenID = packedTop1TokenID
        self.unpackedTop1TokenID = unpackedTop1TokenID
        self.fp16MaxAbsoluteError = fp16MaxAbsoluteError
        self.fp16MaxRelativeError = fp16MaxRelativeError
        self.fp16Top1TokenID = fp16Top1TokenID
        self.referenceTop1TokenID = referenceTop1TokenID
    }
}

public struct CompressedAttentionProbeHashes:
    Codable, Equatable, Sendable
{
    public let sourceKVProjectionSHA256: String
    public let packedKVProjectionSHA256: String
    public let inputTokenIDsSHA256: String
    public let outputTokenIDsSHA256: String

    public init(
        sourceKVProjectionSHA256: String,
        packedKVProjectionSHA256: String,
        inputTokenIDsSHA256: String,
        outputTokenIDsSHA256: String
    ) {
        self.sourceKVProjectionSHA256 = sourceKVProjectionSHA256
        self.packedKVProjectionSHA256 = packedKVProjectionSHA256
        self.inputTokenIDsSHA256 = inputTokenIDsSHA256
        self.outputTokenIDsSHA256 = outputTokenIDsSHA256
    }
}

public struct CompressedAttentionProbeRunReceipts:
    Codable, Equatable, Sendable
{
    public let timing: CompressedAttentionProbeTiming
    public let bytes: CompressedAttentionProbeByteReceipts
    public let mlxMemory: CompressedAttentionProbeMLXMemoryReceipts
    public let processRSS: CompressedAttentionProbeProcessRSS
    public let memorySettings: CompressedAttentionProbeMemorySettings
    public let power: CompressedAttentionProbePowerReceipts
    public let thermal: CompressedAttentionProbeThermalReceipts
    public let numericControls: CompressedAttentionProbeNumericControls
    public let hashes: CompressedAttentionProbeHashes

    public init(
        timing: CompressedAttentionProbeTiming,
        bytes: CompressedAttentionProbeByteReceipts,
        mlxMemory: CompressedAttentionProbeMLXMemoryReceipts,
        processRSS: CompressedAttentionProbeProcessRSS,
        memorySettings: CompressedAttentionProbeMemorySettings,
        power: CompressedAttentionProbePowerReceipts,
        thermal: CompressedAttentionProbeThermalReceipts,
        numericControls: CompressedAttentionProbeNumericControls,
        hashes: CompressedAttentionProbeHashes
    ) {
        self.timing = timing
        self.bytes = bytes
        self.mlxMemory = mlxMemory
        self.processRSS = processRSS
        self.memorySettings = memorySettings
        self.power = power
        self.thermal = thermal
        self.numericControls = numericControls
        self.hashes = hashes
    }
}

public struct CompressedAttentionProbeRunRow:
    Codable, Equatable, Sendable
{
    public let role: CompressedAttentionProbeRunRole
    public let operation: CompressedAttentionProbeOperation
    public let position: CompressedAttentionProbeRunPosition
    public let receipts: CompressedAttentionProbeRunReceipts?

    public init(
        role: CompressedAttentionProbeRunRole,
        operation: CompressedAttentionProbeOperation,
        position: CompressedAttentionProbeRunPosition,
        receipts: CompressedAttentionProbeRunReceipts?
    ) {
        self.role = role
        self.operation = operation
        self.position = position
        self.receipts = receipts
    }
}

public struct CompressedAttentionProbeEvidence:
    Codable, Equatable, Sendable
{
    public static let schemaVersion = 1
    public static let packedRTolerance = 2e-3
    public static let packedATolerance = 2e-3
    public static let fp16RTolerance = 1e-4
    public static let fp16ATolerance = 1e-5

    public let schemaVersion: Int
    public let artifactID: String
    public let plan: CompressedAttentionProbePlanIdentity
    public let model: CompressedAttentionProbeModelIdentity
    public let package: CompressedAttentionProbePackageIdentity
    public let rows: [CompressedAttentionProbeRunRow]

    public init(
        schemaVersion: Int,
        artifactID: String,
        plan: CompressedAttentionProbePlanIdentity,
        model: CompressedAttentionProbeModelIdentity,
        package: CompressedAttentionProbePackageIdentity,
        rows: [CompressedAttentionProbeRunRow]
    ) {
        self.schemaVersion = schemaVersion
        self.artifactID = artifactID
        self.plan = plan
        self.model = model
        self.package = package
        self.rows = rows
    }

    @discardableResult
    public func validated() throws -> CompressedAttentionProbeEvidence {
        guard schemaVersion == Self.schemaVersion else {
            throw CompressedAttentionProbeEvidenceError
                .unsupportedSchema(schemaVersion)
        }
        guard Self.isSHA256(artifactID) else {
            throw CompressedAttentionProbeEvidenceError.invalidArtifactID
        }
        let materializedPlan = try plan.materializedPlan()
        try model.validate()
        try package.validate()

        let expectedRows = materializedPlan.measuredRuns * 2
        guard rows.count == expectedRows else {
            throw CompressedAttentionProbeEvidenceError.invalidRunCount(
                expected: expectedRows, actual: rows.count)
        }

        var seenPositions = Set<CompressedAttentionProbeRunPosition>()
        var blockRoles = Array(
            repeating: [Int: CompressedAttentionProbeRunRole](),
            count: materializedPlan.measuredRuns)
        var blockRows = Array(
            repeating: [Int: CompressedAttentionProbeRunRow](),
            count: materializedPlan.measuredRuns)
        var receiptHashes = Set<String>()

        for row in rows {
            let position = row.position
            guard position.pairedBlockIndex >= 0,
                position.pairedBlockIndex < materializedPlan.measuredRuns,
                position.runPosition == 0 || position.runPosition == 1
            else {
                throw CompressedAttentionProbeEvidenceError.invalidRunPosition
            }
            guard seenPositions.insert(position).inserted else {
                throw CompressedAttentionProbeEvidenceError
                    .duplicateRunPosition(
                        block: position.pairedBlockIndex,
                        position: position.runPosition)
            }
            blockRoles[position.pairedBlockIndex][position.runPosition] =
                row.role
            blockRows[position.pairedBlockIndex][position.runPosition] = row
            try row.validate(plan: materializedPlan)
            if let hashes = row.receipts?.hashes {
                receiptHashes.insert(hashes.sourceKVProjectionSHA256)
                receiptHashes.insert(hashes.packedKVProjectionSHA256)
                receiptHashes.insert(hashes.inputTokenIDsSHA256)
                receiptHashes.insert(hashes.outputTokenIDsSHA256)
            }
        }

        guard !receiptHashes.contains(artifactID) else {
            throw CompressedAttentionProbeEvidenceError.invalidArtifactID
        }

        for block in 0 ..< materializedPlan.measuredRuns {
            let roles = blockRoles[block]
            guard roles.count == 2,
                Set(roles.values) == Set([
                    CompressedAttentionProbeRunRole.candidate,
                    .fp16Reference,
                ])
            else {
                throw CompressedAttentionProbeEvidenceError
                    .incompletePairedBlock(block)
            }
            let expectedCandidatePosition = block % 2
            guard roles[expectedCandidatePosition] == .candidate else {
                throw CompressedAttentionProbeEvidenceError
                    .counterbalanceMismatch(
                        block: block,
                        expectedCandidatePosition:
                            expectedCandidatePosition)
            }
            try Self.validatePairedBlock(
                block, rows: blockRows[block])
        }
        return self
    }

    fileprivate static func isSHA256(_ value: String) -> Bool {
        isLowercaseHex(value, length: 64)
    }

    fileprivate static func isGitSHA(_ value: String) -> Bool {
        isLowercaseHex(value, lengths: [40, 64])
    }

    fileprivate static func isEvidenceValue(_ value: String) -> Bool {
        guard !value.isEmpty,
            value == value.trimmingCharacters(in: .whitespacesAndNewlines)
        else { return false }
        return value.unicodeScalars.allSatisfy {
            $0.value >= 32 && $0.value != 127 && $0 != "/" && $0 != "\\"
                && $0 != "~"
        }
    }

    fileprivate static func isModelID(_ value: String) -> Bool {
        let components = value.split(
            separator: "/", omittingEmptySubsequences: false)
        guard (1 ... 2).contains(components.count) else { return false }
        return components.allSatisfy { component in
            guard !component.isEmpty, component != ".", component != ".."
            else { return false }
            return component.utf8.allSatisfy {
                (48 ... 57).contains($0)
                    || (65 ... 90).contains($0)
                    || (97 ... 122).contains($0)
                    || $0 == 45 || $0 == 46 || $0 == 95
            }
        }
    }

    fileprivate static func checkedSum(_ values: [Int]) -> Int? {
        var result = 0
        for value in values {
            guard value >= 0 else { return nil }
            let (next, overflow) = result.addingReportingOverflow(value)
            guard !overflow else { return nil }
            result = next
        }
        return result
    }

    private static func validatePairedBlock(
        _ block: Int,
        rows: [Int: CompressedAttentionProbeRunRow]
    ) throws {
        guard let first = rows[0]?.receipts,
            let second = rows[1]?.receipts
        else {
            throw CompressedAttentionProbeEvidenceError
                .pairedBlockIdentityMismatch(block)
        }
        guard first.timing.monotonicEndSeconds
            <= second.timing.monotonicStartSeconds
        else {
            throw CompressedAttentionProbeEvidenceError.invalidTiming
        }
        guard
            first.hashes.sourceKVProjectionSHA256
                == second.hashes.sourceKVProjectionSHA256,
            first.hashes.inputTokenIDsSHA256
                == second.hashes.inputTokenIDsSHA256,
            first.memorySettings == second.memorySettings,
            first.power.lowPowerModeEnabledBefore
                == second.power.lowPowerModeEnabledBefore,
            first.power.lowPowerModeEnabledAfter
                == second.power.lowPowerModeEnabledAfter,
            first.power.powerSourceBefore == second.power.powerSourceBefore,
            first.power.powerSourceAfter == second.power.powerSourceAfter,
            first.thermal == second.thermal
        else {
            throw CompressedAttentionProbeEvidenceError
                .pairedBlockIdentityMismatch(block)
        }
    }

    private static func isLowercaseHex(
        _ value: String, length: Int
    ) -> Bool {
        isLowercaseHex(value, lengths: [length])
    }

    private static func isLowercaseHex(
        _ value: String, lengths: Set<Int>
    ) -> Bool {
        guard lengths.contains(value.utf8.count) else { return false }
        return value.utf8.allSatisfy {
            (48 ... 57).contains($0) || (97 ... 102).contains($0)
        }
    }
}

private extension CompressedAttentionProbeModelIdentity {
    func validate() throws {
        guard CompressedAttentionProbeEvidence.isModelID(modelID) else {
            throw CompressedAttentionProbeEvidenceError
                .invalidIdentity("modelID")
        }
        for (name, value) in [
            ("modelConfigSHA256", modelConfigSHA256),
            ("checkpointManifestSHA256", checkpointManifestSHA256),
            ("tokenizerSHA256", tokenizerSHA256),
            ("tokenizerConfigSHA256", tokenizerConfigSHA256),
        ] {
            guard CompressedAttentionProbeEvidence.isSHA256(value) else {
                throw CompressedAttentionProbeEvidenceError
                    .invalidIdentity(name)
            }
        }
    }
}

private extension CompressedAttentionProbePackageIdentity {
    func validate() throws {
        guard CompressedAttentionProbeEvidence.isEvidenceValue(
            mlxSwiftVersion)
        else {
            throw CompressedAttentionProbeEvidenceError
                .invalidIdentity("mlxSwiftVersion")
        }
        guard CompressedAttentionProbeEvidence.isGitSHA(
            mlxSwiftLMRevision)
        else {
            throw CompressedAttentionProbeEvidenceError
                .invalidIdentity("mlxSwiftLMRevision")
        }
        guard CompressedAttentionProbeEvidence.isEvidenceValue(swiftVersion)
        else {
            throw CompressedAttentionProbeEvidenceError
                .invalidIdentity("swiftVersion")
        }
        guard CompressedAttentionProbeEvidence.isEvidenceValue(
            harnessBuildConfiguration)
        else {
            throw CompressedAttentionProbeEvidenceError
                .invalidIdentity("harnessBuildConfiguration")
        }
    }
}

private extension CompressedAttentionProbeRunRow {
    func validate(plan: CompressedAttentionProbePlan) throws {
        switch role {
        case .candidate:
            guard operation == plan.operation else {
                throw CompressedAttentionProbeEvidenceError.operationMismatch
            }
        case .fp16Reference:
            guard operation == .fp16SDPA else {
                throw CompressedAttentionProbeEvidenceError.operationMismatch
            }
        }
        guard let receipts else {
            throw CompressedAttentionProbeEvidenceError.missingReceipts
        }
        try receipts.validate(
            role: role,
            operation: operation,
            plan: plan)
    }
}

private extension CompressedAttentionProbeRunReceipts {
    func validate(
        role: CompressedAttentionProbeRunRole,
        operation: CompressedAttentionProbeOperation,
        plan: CompressedAttentionProbePlan
    ) throws {
        try timing.validate()
        try bytes.validate(
            role: role,
            operation: operation,
            layout: plan.layout)
        try mlxMemory.validate()
        try processRSS.validate()
        try memorySettings.validate()
        try power.validate(promotionEvidence: plan.promotionEvidence)
        try thermal.validate(promotionEvidence: plan.promotionEvidence)
        try numericControls.validate()
        try hashes.validate()
    }
}

private extension CompressedAttentionProbeTiming {
    func validate() throws {
        guard monotonicStartSeconds.isFinite,
            monotonicEndSeconds.isFinite,
            monotonicStartSeconds >= 0,
            monotonicEndSeconds > monotonicStartSeconds,
            wallClockSeconds.isFinite, attentionSeconds.isFinite,
            wallClockSeconds > 0, attentionSeconds >= 0,
            attentionSeconds <= wallClockSeconds,
            abs(
                (monotonicEndSeconds - monotonicStartSeconds)
                    - wallClockSeconds)
                <= max(1e-6, wallClockSeconds * 1e-6)
        else { throw CompressedAttentionProbeEvidenceError.invalidTiming }
    }
}

private extension CompressedAttentionProbeByteReceipts {
    func validate(
        role: CompressedAttentionProbeRunRole,
        operation: CompressedAttentionProbeOperation,
        layout: CompressedAttentionProbeLayout
    ) throws {
        guard let expectedPersistent = CompressedAttentionProbeEvidence
            .checkedSum([
                payloadBytes,
                scaleBytes,
                biasBytes,
                controlBytes,
                alignmentPaddingBytes,
                fp16ResidentBytes,
            ]),
            let expectedPeakTemporary = CompressedAttentionProbeEvidence
                .checkedSum([
                    materializationBytes,
                    otherWorkspaceBytes,
                ]),
            let expectedTotal = CompressedAttentionProbeEvidence
            .checkedSum([
                persistentKVBytes,
                peakTemporaryBytes,
            ]),
            persistentKVBytes == expectedPersistent,
            peakTemporaryBytes == expectedPeakTemporary,
            totalBytes == expectedTotal,
            persistentKVBytes > 0
        else {
            throw CompressedAttentionProbeEvidenceError.invalidByteAccounting
        }
        switch role {
        case .fp16Reference:
            guard payloadBytes == 0,
                scaleBytes == 0,
                biasBytes == 0,
                controlBytes == 0,
                alignmentPaddingBytes == 0,
                fp16ResidentBytes > 0
            else {
                throw CompressedAttentionProbeEvidenceError
                    .invalidByteAccounting
            }
        case .candidate:
            switch layout {
            case .fp16:
                guard payloadBytes == 0,
                    scaleBytes == 0,
                    biasBytes == 0,
                    controlBytes == 0,
                    alignmentPaddingBytes == 0,
                    fp16ResidentBytes > 0
                else {
                    throw CompressedAttentionProbeEvidenceError
                        .invalidByteAccounting
                }
            case .affine:
                guard payloadBytes > 0,
                    scaleBytes > 0,
                    biasBytes > 0,
                    fp16ResidentBytes == 0
                else {
                    throw CompressedAttentionProbeEvidenceError
                        .invalidByteAccounting
                }
            case .kvarn:
                guard payloadBytes > 0,
                    scaleBytes > 0,
                    biasBytes > 0,
                    controlBytes > 0,
                    fp16ResidentBytes > 0
                else {
                    throw CompressedAttentionProbeEvidenceError
                        .invalidByteAccounting
                }
            }
        }
        switch (role, operation) {
        case (.candidate, .materializeThenSDPA):
            guard materializationBytes > 0 else {
                throw CompressedAttentionProbeEvidenceError
                    .invalidByteAccounting
            }
        case (_, .swiftLMQuantizedAttention), (_, .fp16SDPA):
            guard materializationBytes == 0 else {
                throw CompressedAttentionProbeEvidenceError
                    .invalidByteAccounting
            }
        default:
            break
        }
    }
}

private extension CompressedAttentionProbeMLXMemorySnapshot {
    func validate() throws {
        guard activeBytes >= 0, cacheBytes >= 0, peakBytes >= 0,
            peakBytes >= activeBytes
        else {
            throw CompressedAttentionProbeEvidenceError.invalidMemoryCounters
        }
    }
}

private extension CompressedAttentionProbeMLXMemoryReceipts {
    func validate() throws {
        try before.validate()
        try after.validate()
        guard after.peakBytes >= before.peakBytes else {
            throw CompressedAttentionProbeEvidenceError.invalidMemoryCounters
        }
    }
}

private extension CompressedAttentionProbeProcessRSS {
    func validate() throws {
        guard residentSizeBeforeBytes >= 0,
            residentSizeAfterBytes >= 0,
            physicalFootprintBeforeBytes >= 0,
            physicalFootprintAfterBytes >= 0
        else {
            throw CompressedAttentionProbeEvidenceError.invalidMemoryCounters
        }
    }
}

private extension CompressedAttentionProbeMemorySettings {
    func validate() throws {
        guard memoryLimitBytes > 0, cacheLimitBytes > 0,
            memoryLimitBytes >= cacheLimitBytes, wiredLimitBytes >= 0
        else {
            throw CompressedAttentionProbeEvidenceError.invalidMemorySettings
        }
    }
}

private extension CompressedAttentionProbePowerReceipts {
    func validate(promotionEvidence: Bool) throws {
        guard !promotionEvidence
            || (lowPowerModeEnabledBefore == lowPowerModeEnabledAfter
                && powerSourceBefore == powerSourceAfter
                && powerSourceBefore != .unavailable)
        else {
            throw CompressedAttentionProbeEvidenceError.invalidPowerState
        }
    }
}

private extension CompressedAttentionProbeThermalReceipts {
    func validate(promotionEvidence: Bool) throws {
        guard !promotionEvidence
            || (before == after && before != .unknown)
        else {
            throw CompressedAttentionProbeEvidenceError.invalidThermalState
        }
    }
}

private extension CompressedAttentionProbeNumericControls {
    func validate() throws {
        let metrics = [
            packedMaxAbsoluteError,
            packedMaxRelativeError,
            fp16MaxAbsoluteError,
            fp16MaxRelativeError,
        ]
        guard metrics.allSatisfy({ $0.isFinite && $0 >= 0 }),
            packedTop1TokenID >= 0,
            unpackedTop1TokenID >= 0,
            fp16Top1TokenID >= 0,
            referenceTop1TokenID >= 0
        else {
            throw CompressedAttentionProbeEvidenceError
                .invalidMetric("numericControls")
        }
        guard packedTop1TokenID == unpackedTop1TokenID,
            packedMaxRelativeError
                <= CompressedAttentionProbeEvidence.packedRTolerance,
            packedMaxAbsoluteError
                <= CompressedAttentionProbeEvidence.packedATolerance
        else {
            throw CompressedAttentionProbeEvidenceError
                .packedStructuralMismatch
        }
        guard fp16Top1TokenID == referenceTop1TokenID,
            fp16MaxRelativeError
                <= CompressedAttentionProbeEvidence.fp16RTolerance,
            fp16MaxAbsoluteError
                <= CompressedAttentionProbeEvidence.fp16ATolerance
        else {
            throw CompressedAttentionProbeEvidenceError.fp16ControlMismatch
        }
    }
}

private extension CompressedAttentionProbeHashes {
    func validate() throws {
        for (name, value) in [
            ("sourceKVProjectionSHA256", sourceKVProjectionSHA256),
            ("packedKVProjectionSHA256", packedKVProjectionSHA256),
            ("inputTokenIDsSHA256", inputTokenIDsSHA256),
            ("outputTokenIDsSHA256", outputTokenIDsSHA256),
        ] {
            guard CompressedAttentionProbeEvidence.isSHA256(value) else {
                throw CompressedAttentionProbeEvidenceError.invalidHash(name)
            }
        }
    }
}
