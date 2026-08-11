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
    case candidateStructuralMismatch
    case referenceControlMismatch
}

public enum CompressedAttentionProbeLayoutKind:
    String, Codable, Equatable, Sendable
{
    case fp16
    case affine
    case kvarn
}

/// Identifies what this artifact actually measured. The Phase-0 probe authenticates a
/// checkpoint and constrains tensors to its declared attention geometry, but deliberately does
/// not load or execute model weights. Keeping that boundary in the signed payload prevents a
/// synthetic kernel profile from being mistaken for loaded-model or dial-promotion evidence.
public enum CompressedAttentionProbeEvidenceKind:
    String, Codable, Equatable, Sendable
{
    case checkpointAuthenticatedSyntheticGeometry =
        "checkpoint-authenticated-synthetic-geometry"
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
    public let qualificationEvidence: Bool
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
        qualificationEvidence: Bool,
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
        self.qualificationEvidence = qualificationEvidence
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
            qualificationEvidence: plan.qualificationEvidence,
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
                qualificationEvidence: qualificationEvidence,
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
    public static let qualifiedMLXSwiftVersion = "0.31.6"
    public static let qualifiedMLXSwiftLMRevision =
        "702e5a0eaf990e1f6d3db2b6e7d8872858a44055"
    public static let qualifiedBuildConfiguration = "Release"

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

/// Workspace totals derived from the raw MLX high-water receipt and the independently known
/// materialization allocation. MLX resets the peak counter to zero, so an allocation-free
/// measured section can report a post-run peak below the already-resident active baseline. The
/// effective peak therefore floors the raw counter at that baseline instead of inventing a
/// baseline-inclusive MLX counter value.
public struct CompressedAttentionProbeWorkspaceBytes:
    Equatable, Sendable
{
    public let otherWorkspaceBytes: Int
    public let peakTemporaryBytes: Int
    public let totalBytes: Int

    public static func derive(
        persistentKVBytes: Int,
        materializationBytes: Int,
        mlxMemory: CompressedAttentionProbeMLXMemoryReceipts
    ) throws -> Self {
        guard persistentKVBytes >= 0,
            materializationBytes >= 0,
            mlxMemory.before.activeBytes >= 0,
            mlxMemory.after.peakBytes >= 0
        else {
            throw CompressedAttentionProbeEvidenceError.invalidByteAccounting
        }
        let effectivePeakBytes = max(
            mlxMemory.after.peakBytes,
            mlxMemory.before.activeBytes)
        let observedPeakAboveBaseline = effectivePeakBytes
            - mlxMemory.before.activeBytes
        let otherWorkspaceBytes = max(
            0,
            observedPeakAboveBaseline - materializationBytes)
        guard let peakTemporaryBytes = checkedSum(
            materializationBytes, otherWorkspaceBytes),
            let totalBytes = checkedSum(
                persistentKVBytes, peakTemporaryBytes)
        else {
            throw CompressedAttentionProbeEvidenceError.invalidByteAccounting
        }
        return Self(
            otherWorkspaceBytes: otherWorkspaceBytes,
            peakTemporaryBytes: peakTemporaryBytes,
            totalBytes: totalBytes)
    }

    private static func checkedSum(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? nil : value
    }
}

/// Geometry-derived persistent and materialization bytes for one authenticated probe row.
/// These values are independent of allocator telemetry: evidence records the actual array byte
/// receipts, then validation requires them to equal this plan-derived contract exactly.
public struct CompressedAttentionProbeExpectedByteGeometry:
    Equatable, Sendable
{
    public let payloadBytes: Int
    public let scaleBytes: Int
    public let biasBytes: Int
    public let controlBytes: Int
    public let alignmentPaddingBytes: Int
    public let fp16ResidentBytes: Int
    public let persistentKVBytes: Int
    public let materializationBytes: Int

    public static func derive(
        plan: CompressedAttentionProbePlan,
        role: CompressedAttentionProbeRunRole,
        operation: CompressedAttentionProbeOperation
    ) throws -> Self {
        let scalarBytes = 2
        let fp16Bytes = try product([plan.totalKVScalarCount, scalarBytes])
        let payload: Int
        let scales: Int
        let biases: Int
        let control: Int
        let padding: Int
        let fp16Resident: Int

        if role == .fp16Reference || plan.layout == .fp16 {
            payload = 0
            scales = 0
            biases = 0
            control = 0
            padding = 0
            fp16Resident = fp16Bytes
        } else {
            switch plan.layout {
            case .fp16:
                throw CompressedAttentionProbeEvidenceError
                    .invalidByteAccounting
            case let .affine(
                keyBits, valueBits, keyGroupSize, valueGroupSize):
                let oneSideScalars = try product([
                    plan.batchSize,
                    plan.kvHeadCount,
                    plan.contextTokens,
                    plan.headDimension,
                ])
                payload = try sum([
                    try product([oneSideScalars, keyBits]) / 8,
                    try product([oneSideScalars, valueBits]) / 8,
                ])
                let metadataScalars = try product([
                    plan.batchSize,
                    plan.kvHeadCount,
                    plan.contextTokens,
                ])
                scales = try product([
                    metadataScalars,
                    plan.headDimension / keyGroupSize
                        + plan.headDimension / valueGroupSize,
                    scalarBytes,
                ])
                biases = scales
                control = 0
                padding = 0
                fp16Resident = 0
            case let .kvarn(
                keyBits, valueBits, groupSize, sinkTokens, _):
                let headSequences = try product([
                    plan.batchSize, plan.kvHeadCount,
                ])
                let postSinkTokens = max(0, plan.contextTokens - sinkTokens)
                let slots = (postSinkTokens + groupSize - 1) / groupSize
                let keyPayloadUnit = try product([
                    plan.headDimension, groupSize, keyBits,
                ]) / 8
                let valuePayloadUnit = try product([
                    groupSize, plan.headDimension, valueBits,
                ]) / 8
                let payloadUnit = try sum([
                    keyPayloadUnit, valuePayloadUnit,
                ])
                let scaleUnit = try product([
                    2 * (plan.headDimension + groupSize), scalarBytes,
                ])
                let biasUnit = try product([
                    plan.headDimension + groupSize, scalarBytes,
                ])
                let storedUnits = try product([headSequences, slots])
                payload = try product([storedUnits, payloadUnit])
                scales = try product([storedUnits, scaleUnit])
                biases = try product([storedUnits, biasUnit])
                control = MemoryLayout<Int32>.size
                let rawUnit = try sum([payloadUnit, scaleUnit, biasUnit])
                let alignment = 8
                let alignedUnit = ((rawUnit + alignment - 1) / alignment)
                    * alignment
                padding = try product([
                    storedUnits, alignedUnit - rawUnit,
                ])
                fp16Resident = try product([
                    headSequences,
                    sinkTokens + groupSize,
                    plan.headDimension,
                    2,
                    scalarBytes,
                ])
            }
        }

        let persistent = try sum([
            payload, scales, biases, control, padding, fp16Resident,
        ])
        let materialization = role == .candidate
            && operation == .materializeThenSDPA
            ? fp16Bytes
            : 0
        return Self(
            payloadBytes: payload,
            scaleBytes: scales,
            biasBytes: biases,
            controlBytes: control,
            alignmentPaddingBytes: padding,
            fp16ResidentBytes: fp16Resident,
            persistentKVBytes: persistent,
            materializationBytes: materialization)
    }

    private static func product(_ values: [Int]) throws -> Int {
        var result = 1
        for value in values {
            guard value >= 0 else {
                throw CompressedAttentionProbeEvidenceError
                    .invalidByteAccounting
            }
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            guard !overflow else {
                throw CompressedAttentionProbeEvidenceError
                    .invalidByteAccounting
            }
            result = next
        }
        return result
    }

    private static func sum(_ values: [Int]) throws -> Int {
        guard let result = CompressedAttentionProbeEvidence.checkedSum(values)
        else {
            throw CompressedAttentionProbeEvidenceError.invalidByteAccounting
        }
        return result
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
    case preserveAcrossRun = "preserve-across-run"
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
    public let candidateMaxAbsoluteError: Double
    public let candidateMaxRelativeError: Double
    /// Maximum per-element `abs(error) / (atol + rtol * abs(reference))`.
    /// Values at or below one satisfy the frozen mixed tolerance even when the raw relative
    /// error is large for a reference value near zero.
    public let candidateMaximumToleranceRatio: Double
    public let candidateTop1Index: Int
    public let candidateOracleTop1Index: Int
    public let referenceMaxAbsoluteError: Double
    public let referenceMaxRelativeError: Double
    public let referenceMaximumToleranceRatio: Double
    public let referenceTop1Index: Int
    public let referenceOracleTop1Index: Int

    public init(
        candidateMaxAbsoluteError: Double,
        candidateMaxRelativeError: Double,
        candidateMaximumToleranceRatio: Double,
        candidateTop1Index: Int,
        candidateOracleTop1Index: Int,
        referenceMaxAbsoluteError: Double,
        referenceMaxRelativeError: Double,
        referenceMaximumToleranceRatio: Double,
        referenceTop1Index: Int,
        referenceOracleTop1Index: Int
    ) {
        self.candidateMaxAbsoluteError = candidateMaxAbsoluteError
        self.candidateMaxRelativeError = candidateMaxRelativeError
        self.candidateMaximumToleranceRatio = candidateMaximumToleranceRatio
        self.candidateTop1Index = candidateTop1Index
        self.candidateOracleTop1Index = candidateOracleTop1Index
        self.referenceMaxAbsoluteError = referenceMaxAbsoluteError
        self.referenceMaxRelativeError = referenceMaxRelativeError
        self.referenceMaximumToleranceRatio = referenceMaximumToleranceRatio
        self.referenceTop1Index = referenceTop1Index
        self.referenceOracleTop1Index = referenceOracleTop1Index
    }
}

public struct CompressedAttentionProbeHashes:
    Codable, Equatable, Sendable
{
    public let sourceKVTensorSHA256: String
    public let packedKVTensorSHA256: String
    public let queryTensorSHA256: String
    public let outputTensorSHA256: String

    public init(
        sourceKVTensorSHA256: String,
        packedKVTensorSHA256: String,
        queryTensorSHA256: String,
        outputTensorSHA256: String
    ) {
        self.sourceKVTensorSHA256 = sourceKVTensorSHA256
        self.packedKVTensorSHA256 = packedKVTensorSHA256
        self.queryTensorSHA256 = queryTensorSHA256
        self.outputTensorSHA256 = outputTensorSHA256
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
    public static let schemaVersion = 2
    public static let packedRTolerance = 2e-3
    public static let packedATolerance = 2e-3
    /// Matches the pinned MLX float16 SDPA qualification envelope in
    /// `python/tests/test_fast_sdpa.py`.
    public static let fp16RTolerance = 3e-4
    public static let fp16ATolerance = 3e-4

    public let schemaVersion: Int
    public let evidenceKind: CompressedAttentionProbeEvidenceKind
    public let artifactID: String
    public let plan: CompressedAttentionProbePlanIdentity
    public let model: CompressedAttentionProbeModelIdentity
    public let package: CompressedAttentionProbePackageIdentity
    public let rows: [CompressedAttentionProbeRunRow]

    public init(
        schemaVersion: Int,
        evidenceKind: CompressedAttentionProbeEvidenceKind,
        artifactID: String,
        plan: CompressedAttentionProbePlanIdentity,
        model: CompressedAttentionProbeModelIdentity,
        package: CompressedAttentionProbePackageIdentity,
        rows: [CompressedAttentionProbeRunRow]
    ) {
        self.schemaVersion = schemaVersion
        self.evidenceKind = evidenceKind
        self.artifactID = artifactID
        self.plan = plan
        self.model = model
        self.package = package
        self.rows = rows
    }

    /// Derives the artifact identity from the complete validated payload while deliberately
    /// excluding `artifactID` itself. This avoids a recursive self-hash while authenticating
    /// every plan, model, package, row, and receipt field in one domain-separated transcript.
    public static func deriveArtifactID(
        plan: CompressedAttentionProbePlanIdentity,
        model: CompressedAttentionProbeModelIdentity,
        package: CompressedAttentionProbePackageIdentity,
        rows: [CompressedAttentionProbeRunRow]
    ) throws -> String {
        let placeholder = String(repeating: "0", count: 64)
        let evidence = Self(
            schemaVersion: schemaVersion,
            evidenceKind: .checkpointAuthenticatedSyntheticGeometry,
            artifactID: placeholder,
            plan: plan,
            model: model,
            package: package,
            rows: rows)
        _ = try evidence.validatePayload()
        return try computeArtifactID(
            plan: plan,
            model: model,
            package: package,
            rows: rows)
    }

    private static func computeArtifactID(
        plan: CompressedAttentionProbePlanIdentity,
        model: CompressedAttentionProbeModelIdentity,
        package: CompressedAttentionProbePackageIdentity,
        rows: [CompressedAttentionProbeRunRow]
    ) throws -> String {
        let payload = ArtifactIdentityPayload(
            schemaVersion: schemaVersion,
            evidenceKind: .checkpointAuthenticatedSyntheticGeometry,
            plan: plan,
            model: model,
            package: package,
            rows: rows)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var transcript = Data(
            "fastmlx-compressed-attention-artifact-v2\n".utf8)
        transcript.append(try encoder.encode(payload))
        return sha256Hex(transcript)
    }

    @discardableResult
    public func validated() throws -> CompressedAttentionProbeEvidence {
        guard schemaVersion == Self.schemaVersion else {
            throw CompressedAttentionProbeEvidenceError
                .unsupportedSchema(schemaVersion)
        }
        guard evidenceKind == .checkpointAuthenticatedSyntheticGeometry else {
            throw CompressedAttentionProbeEvidenceError.invalidPlanIdentity
        }
        guard Self.isSHA256(artifactID) else {
            throw CompressedAttentionProbeEvidenceError.invalidArtifactID
        }
        let receiptHashes = try validatePayload()
        let expectedArtifactID = try Self.computeArtifactID(
            plan: plan,
            model: model,
            package: package,
            rows: rows)
        guard artifactID == expectedArtifactID,
            !receiptHashes.contains(artifactID)
        else {
            throw CompressedAttentionProbeEvidenceError.invalidArtifactID
        }
        return self
    }

    private func validatePayload() throws -> Set<String> {
        guard schemaVersion == Self.schemaVersion else {
            throw CompressedAttentionProbeEvidenceError
                .unsupportedSchema(schemaVersion)
        }
        let materializedPlan = try plan.materializedPlan()
        try model.validate()
        try package.validate(
            qualificationEvidence: materializedPlan.qualificationEvidence)

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
                receiptHashes.insert(hashes.sourceKVTensorSHA256)
                receiptHashes.insert(hashes.packedKVTensorSHA256)
                receiptHashes.insert(hashes.queryTensorSHA256)
                receiptHashes.insert(hashes.outputTensorSHA256)
            }
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
        return receiptHashes
    }

    private struct ArtifactIdentityPayload: Codable {
        let schemaVersion: Int
        let evidenceKind: CompressedAttentionProbeEvidenceKind
        let plan: CompressedAttentionProbePlanIdentity
        let model: CompressedAttentionProbeModelIdentity
        let package: CompressedAttentionProbePackageIdentity
        let rows: [CompressedAttentionProbeRunRow]
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
            first.hashes.sourceKVTensorSHA256
                == second.hashes.sourceKVTensorSHA256,
            first.hashes.queryTensorSHA256
                == second.hashes.queryTensorSHA256,
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
    func validate(qualificationEvidence: Bool) throws {
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
        guard !qualificationEvidence
            || mlxSwiftVersion == Self.qualifiedMLXSwiftVersion
        else {
            throw CompressedAttentionProbeEvidenceError
                .invalidIdentity("mlxSwiftVersion")
        }
        guard !qualificationEvidence
            || mlxSwiftLMRevision == Self.qualifiedMLXSwiftLMRevision
        else {
            throw CompressedAttentionProbeEvidenceError
                .invalidIdentity("mlxSwiftLMRevision")
        }
        guard !qualificationEvidence
            || harnessBuildConfiguration
                == Self.qualifiedBuildConfiguration
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
            plan: plan)
        try mlxMemory.validate()
        let expectedWorkspace = try CompressedAttentionProbeWorkspaceBytes
            .derive(
                persistentKVBytes: bytes.persistentKVBytes,
                materializationBytes: bytes.materializationBytes,
                mlxMemory: mlxMemory)
        guard bytes.otherWorkspaceBytes
                == expectedWorkspace.otherWorkspaceBytes,
            bytes.peakTemporaryBytes == expectedWorkspace.peakTemporaryBytes,
            bytes.totalBytes == expectedWorkspace.totalBytes
        else {
            throw CompressedAttentionProbeEvidenceError.invalidByteAccounting
        }
        try processRSS.validate()
        try memorySettings.validate()
        try power.validate(qualificationEvidence: plan.qualificationEvidence)
        try thermal.validate(qualificationEvidence: plan.qualificationEvidence)
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
        plan: CompressedAttentionProbePlan
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
        let geometry = try CompressedAttentionProbeExpectedByteGeometry.derive(
            plan: plan, role: role, operation: operation)
        guard payloadBytes == geometry.payloadBytes,
            scaleBytes == geometry.scaleBytes,
            biasBytes == geometry.biasBytes,
            controlBytes == geometry.controlBytes,
            alignmentPaddingBytes == geometry.alignmentPaddingBytes,
            fp16ResidentBytes == geometry.fp16ResidentBytes,
            persistentKVBytes == geometry.persistentKVBytes,
            materializationBytes == geometry.materializationBytes
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
            switch plan.layout {
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
        case (_, .swiftLMQuantizedAttention),
            (_, .splitAffineQuantizedMM),
            (_, .fp16SDPA):
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
    func validateNonnegative() throws {
        guard activeBytes >= 0, cacheBytes >= 0, peakBytes >= 0
        else {
            throw CompressedAttentionProbeEvidenceError.invalidMemoryCounters
        }
    }
}

private extension CompressedAttentionProbeMLXMemoryReceipts {
    func validate() throws {
        try before.validateNonnegative()
        try after.validateNonnegative()
        // The probe resets MLX's peak counter only after the prepared cache/query state is
        // resident. A zero pre-run peak is the authenticated boundary marker. The raw post-run
        // counter only records allocations after that reset and is not guaranteed to include the
        // resident baseline; workspace derivation accounts for that documented raw behavior.
        let rawCounterIsPlausible = after.peakBytes == 0
            ? after.activeBytes <= before.activeBytes
            : after.peakBytes >= max(
                before.activeBytes,
                after.activeBytes)
        guard before.peakBytes == 0, rawCounterIsPlausible else {
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
    func validate(qualificationEvidence: Bool) throws {
        guard !qualificationEvidence
            || (lowPowerModeEnabledBefore == lowPowerModeEnabledAfter
                && powerSourceBefore == powerSourceAfter
                && powerSourceBefore != .unavailable)
        else {
            throw CompressedAttentionProbeEvidenceError.invalidPowerState
        }
    }
}

private extension CompressedAttentionProbeThermalReceipts {
    func validate(qualificationEvidence: Bool) throws {
        guard !qualificationEvidence
            || (before == after && before != .unknown)
        else {
            throw CompressedAttentionProbeEvidenceError.invalidThermalState
        }
    }
}

private extension CompressedAttentionProbeNumericControls {
    func validate() throws {
        let metrics = [
            candidateMaxAbsoluteError,
            candidateMaxRelativeError,
            candidateMaximumToleranceRatio,
            referenceMaxAbsoluteError,
            referenceMaxRelativeError,
            referenceMaximumToleranceRatio,
        ]
        guard metrics.allSatisfy({ $0.isFinite && $0 >= 0 }),
            candidateTop1Index >= 0,
            candidateOracleTop1Index >= 0,
            referenceTop1Index >= 0,
            referenceOracleTop1Index >= 0
        else {
            throw CompressedAttentionProbeEvidenceError
                .invalidMetric("numericControls")
        }
        guard candidateTop1Index == candidateOracleTop1Index,
            candidateMaximumToleranceRatio <= 1
        else {
            throw CompressedAttentionProbeEvidenceError
                .candidateStructuralMismatch
        }
        guard referenceTop1Index == referenceOracleTop1Index,
            referenceMaximumToleranceRatio <= 1
        else {
            throw CompressedAttentionProbeEvidenceError.referenceControlMismatch
        }
    }
}

private extension CompressedAttentionProbeHashes {
    func validate() throws {
        for (name, value) in [
            ("sourceKVTensorSHA256", sourceKVTensorSHA256),
            ("packedKVTensorSHA256", packedKVTensorSHA256),
            ("queryTensorSHA256", queryTensorSHA256),
            ("outputTensorSHA256", outputTensorSHA256),
        ] {
            guard CompressedAttentionProbeEvidence.isSHA256(value) else {
                throw CompressedAttentionProbeEvidenceError.invalidHash(name)
            }
        }
    }
}
