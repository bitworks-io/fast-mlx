import Foundation

public enum CompressedAttentionProbeOperation: String, Codable, Sendable, CaseIterable {
    case fp16SDPA = "fp16-sdpa"
    case swiftLMQuantizedAttention = "swiftlm-quantized-attention"
    case splitAffineQuantizedMM = "split-affine-quantized-mm"
    case materializeThenSDPA = "materialize-then-sdpa"
}

public enum CompressedAttentionProbeDType: String, Codable, Sendable, CaseIterable {
    case float16
    case bfloat16
}

public enum CompressedAttentionProbeMask: String, Codable, Sendable, CaseIterable {
    case none
    case causal
}

/// The persistent KV representation consumed by a probe operation. Keeping K and V geometry
/// independent prevents a symmetric control from being mislabeled as K4V2 or KVTuner evidence.
public enum CompressedAttentionProbeLayout: Equatable, Sendable {
    case fp16
    case affine(
        keyBits: Int,
        valueBits: Int,
        keyGroupSize: Int,
        valueGroupSize: Int)
    case kvarn(
        keyBits: Int,
        valueBits: Int,
        groupSize: Int,
        sinkTokens: Int,
        iterations: Int)
}

public enum CompressedAttentionProbePlanError: Error, Equatable, Sendable {
    case invalidTensorGeometry
    case invalidGQAGeometry
    case invalidLayoutGeometry
    case invalidWorkloadShape
    case invalidStopTokenIDs
    case invalidRunCounts
    case invalidSeed
    case invalidWorkloadNonce
    case invalidHarnessGitSHA
    case operationLayoutMismatch
    case unsupportedStockQuantizedLayout
    case unapprovedQualificationWindow(
        contextTokens: Int, outputTokens: Int)
    case insufficientQualificationRuns(Int)
    case invalidOutputPath(String)
    case symbolicLinkOutput(String)
    case outputPathCollision(String)
    case arithmeticOverflow
}

/// Pure, allocation-free validation for a compressed-attention profiling cell. MLX-coupled code
/// must accept one of these plans rather than inferring geometry or qualification status from
/// flags.
public struct CompressedAttentionProbePlan: Equatable, Sendable {
    public static let qualificationContextTokens: Set<Int> = [8_192, 32_768]
    public static let near128KQualificationContextTokens = 130_944
    public static let near128KQualificationOutputTokens = 128
    public static let qualificationTotalWindowTokens = 131_072
    public static let minimumQualificationRuns = 3

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
    public let layout: CompressedAttentionProbeLayout
    public let warmupRuns: Int
    public let measuredRuns: Int
    public let seed: Int
    public let workloadNonce: String
    public let harnessGitSHA: String
    /// Enables the strict long-context capture contract. Passing this gate is necessary but never
    /// sufficient for dial promotion: this probe measures synthetic attention geometry, not a
    /// loaded model or user workload.
    public let qualificationEvidence: Bool
    public let evidenceOutputPath: String
    public let progressOutputPath: String
    public let isQualificationContext: Bool
    public let gqaRepeatCount: Int
    public let totalKVScalarCount: Int
    public let totalQueryScalarCount: Int

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
        layout: CompressedAttentionProbeLayout,
        warmupRuns: Int,
        measuredRuns: Int,
        seed: Int,
        workloadNonce: String,
        harnessGitSHA: String,
        qualificationEvidence: Bool,
        evidenceOutputPath: String = "compressed-attention-probe.jsonl",
        progressOutputPath: String = "compressed-attention-probe.progress.json"
    ) throws {
        guard contextTokens > 0, batchSize > 0, queryHeadCount > 0,
            kvHeadCount > 0, headDimension > 0
        else {
            throw CompressedAttentionProbePlanError.invalidTensorGeometry
        }
        guard queryHeadCount.isMultiple(of: kvHeadCount) else {
            throw CompressedAttentionProbePlanError.invalidGQAGeometry
        }
        guard queryTokens > 0, prefillChunkTokens > 0,
            queryTokens <= prefillChunkTokens,
            queryTokens <= contextTokens,
            outputTokens > 0
        else {
            throw CompressedAttentionProbePlanError.invalidWorkloadShape
        }
        guard stopTokenIDs.allSatisfy({ $0 >= 0 }),
            Set(stopTokenIDs).count == stopTokenIDs.count,
            stopTokenIDs == stopTokenIDs.sorted()
        else {
            throw CompressedAttentionProbePlanError.invalidStopTokenIDs
        }
        guard (1 ... 100).contains(warmupRuns),
            (1 ... 100).contains(measuredRuns)
        else {
            throw CompressedAttentionProbePlanError.invalidRunCounts
        }
        guard seed >= 0 else {
            throw CompressedAttentionProbePlanError.invalidSeed
        }
        do {
            _ = try ServiceWorkloadIdentity(nonce: workloadNonce)
        } catch {
            throw CompressedAttentionProbePlanError.invalidWorkloadNonce
        }
        guard Self.isLowercaseHex(harnessGitSHA, lengths: [40, 64]) else {
            throw CompressedAttentionProbePlanError.invalidHarnessGitSHA
        }
        for outputPath in [evidenceOutputPath, progressOutputPath] {
            guard !outputPath.isEmpty,
                outputPath == outputPath.trimmingCharacters(
                    in: .whitespacesAndNewlines)
            else {
                throw CompressedAttentionProbePlanError.invalidOutputPath(
                    outputPath)
            }
            guard !outputPathIsSymbolicLink(outputPath) else {
                throw CompressedAttentionProbePlanError.symbolicLinkOutput(
                    outputPath)
            }
        }
        guard !outputPathsReferToSameFile(
            evidenceOutputPath, progressOutputPath)
        else {
            throw CompressedAttentionProbePlanError.outputPathCollision(
                progressOutputPath)
        }

        try Self.validate(layout: layout, headDimension: headDimension)
        try Self.validate(operation: operation, layout: layout)

        let scalarCount = try Self.checkedProduct([
            batchSize,
            kvHeadCount,
            contextTokens,
            headDimension,
            2,
        ])
        let queryScalarCount = try Self.checkedProduct([
            batchSize,
            queryHeadCount,
            queryTokens,
            headDimension,
        ])

        let (totalWindowTokens, windowOverflow) = contextTokens
            .addingReportingOverflow(outputTokens)
        guard !windowOverflow else {
            throw CompressedAttentionProbePlanError.arithmeticOverflow
        }
        let isQualificationContext = Self.qualificationContextTokens
            .contains(contextTokens)
            || (contextTokens == Self.near128KQualificationContextTokens
                && outputTokens == Self.near128KQualificationOutputTokens
                && totalWindowTokens
                    == Self.qualificationTotalWindowTokens)

        if qualificationEvidence {
            guard isQualificationContext else {
                throw CompressedAttentionProbePlanError
                    .unapprovedQualificationWindow(
                        contextTokens: contextTokens,
                        outputTokens: outputTokens)
            }
            guard measuredRuns >= Self.minimumQualificationRuns else {
                throw CompressedAttentionProbePlanError
                    .insufficientQualificationRuns(measuredRuns)
            }
        }

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
        self.isQualificationContext = qualificationEvidence
            && isQualificationContext
        self.gqaRepeatCount = queryHeadCount / kvHeadCount
        self.totalKVScalarCount = scalarCount
        self.totalQueryScalarCount = queryScalarCount
    }

    private static func validate(
        layout: CompressedAttentionProbeLayout,
        headDimension: Int
    ) throws {
        switch layout {
        case .fp16:
            return

        case let .affine(keyBits, valueBits, keyGroupSize, valueGroupSize):
            let supportedBits = [2, 4, 8]
            let supportedGroups = [32, 64, 128]
            guard supportedBits.contains(keyBits), supportedBits.contains(valueBits),
                supportedGroups.contains(keyGroupSize),
                supportedGroups.contains(valueGroupSize),
                headDimension.isMultiple(of: keyGroupSize),
                headDimension.isMultiple(of: valueGroupSize)
            else {
                throw CompressedAttentionProbePlanError.invalidLayoutGeometry
            }

        case let .kvarn(
            keyBits, valueBits, groupSize, sinkTokens, iterations):
            guard keyBits == 4, valueBits == 2, groupSize == 128,
                sinkTokens == groupSize, [8, 16].contains(iterations),
                headDimension.isMultiple(of: groupSize)
            else {
                throw CompressedAttentionProbePlanError.invalidLayoutGeometry
            }
        }
    }

    private static func validate(
        operation: CompressedAttentionProbeOperation,
        layout: CompressedAttentionProbeLayout
    ) throws {
        switch (operation, layout) {
        case (.fp16SDPA, .fp16):
            return
        case (.fp16SDPA, _), (.materializeThenSDPA, .fp16):
            throw CompressedAttentionProbePlanError.operationLayoutMismatch
        case let (
            .swiftLMQuantizedAttention,
            .affine(keyBits, valueBits, keyGroupSize, valueGroupSize)):
            guard keyBits == valueBits, keyGroupSize == valueGroupSize else {
                throw CompressedAttentionProbePlanError
                    .unsupportedStockQuantizedLayout
            }
        case (.swiftLMQuantizedAttention, _):
            throw CompressedAttentionProbePlanError
                .unsupportedStockQuantizedLayout
        case (.splitAffineQuantizedMM, .affine):
            return
        case (.splitAffineQuantizedMM, _):
            throw CompressedAttentionProbePlanError.operationLayoutMismatch
        case (.materializeThenSDPA, .affine),
            (.materializeThenSDPA, .kvarn):
            return
        }
    }

    private static func checkedProduct(_ factors: [Int]) throws -> Int {
        var result = 1
        for factor in factors {
            let (next, overflow) = result.multipliedReportingOverflow(by: factor)
            guard !overflow else {
                throw CompressedAttentionProbePlanError.arithmeticOverflow
            }
            result = next
        }
        return result
    }

    private static func isLowercaseHex(
        _ value: String,
        lengths: Set<Int>
    ) -> Bool {
        guard lengths.contains(value.utf8.count) else { return false }
        return value.utf8.allSatisfy {
            (48 ... 57).contains($0) || (97 ... 102).contains($0)
        }
    }
}
