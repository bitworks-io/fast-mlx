import Darwin
import Foundation
import HarnessCore
import IOKit.ps

enum CompressedAttentionProbeCLIError: Error, Equatable {
    case outputNotFresh(String)
    case outputPathCollision(String)
    case evidenceAlreadyWritten(String)
    case lockNotHeld
    case ioFailure(operation: String, code: Int32)
    case unreadableModelFile(String)
    case missingCheckpointWeights(String)
    case invalidCheckpointWeight(String)
    case unsupportedModelConfig(String)
    case modelGeometryMismatch(String)
    case unsupportedOperationLayout
    case releaseBuildRequired
}

enum CompressedAttentionProbeProgressStatus:
    String, Codable, Equatable, Sendable
{
    case starting
    case warming
    case running
    case complete
    case failed
}

struct CompressedAttentionProbeProgress: Codable, Equatable, Sendable {
    let schemaVersion: Int
    let status: CompressedAttentionProbeProgressStatus
    let completedWarmupRuns: Int
    let completedMeasuredRows: Int
    let totalMeasuredRows: Int
    let activeBlockIndex: Int?
    let activeRole: CompressedAttentionProbeRunRole?
    let elapsedSeconds: Double
    let processResidentBytes: Int
    let processPhysicalFootprintBytes: Int
    let harnessGitSHA: String
    let workloadNonce: String
}

/// A process-scoped lease for one fresh evidence/progress pair. Each destination has its own
/// sibling lock, acquired in canonical path order, so writers that share either destination
/// serialize even when the other destination differs. Evidence is immutable; progress is the
/// only replaceable artifact under both held locks.
final class CompressedAttentionProbeOutputLease {
    let evidenceURL: URL
    let progressURL: URL
    let lockURL: URL
    let lockURLs: [URL]
    private var lockDescriptors: [Int32]

    private init(
        evidenceURL: URL,
        progressURL: URL,
        lockURL: URL,
        lockURLs: [URL],
        lockDescriptors: [Int32]
    ) {
        self.evidenceURL = evidenceURL
        self.progressURL = progressURL
        self.lockURL = lockURL
        self.lockURLs = lockURLs
        self.lockDescriptors = lockDescriptors
    }

    static func acquire(
        evidencePath: String,
        progressPath: String
    ) throws -> CompressedAttentionProbeOutputLease {
        guard !outputPathsReferToSameFile(evidencePath, progressPath) else {
            throw CompressedAttentionProbeCLIError
                .outputPathCollision(progressPath)
        }
        let evidenceURL = URL(fileURLWithPath: evidencePath)
            .standardizedFileURL
        let progressURL = URL(fileURLWithPath: progressPath)
            .standardizedFileURL
        let evidenceLockURL = siblingLockURL(for: evidenceURL)
        let progressLockURL = siblingLockURL(for: progressURL)
        let lockURLs = [evidenceLockURL, progressLockURL]
            .sorted { $0.path < $1.path }
        guard Set(lockURLs.map(\.path)).count == 2 else {
            throw CompressedAttentionProbeCLIError
                .outputPathCollision(evidenceLockURL.path)
        }
        for lockURL in lockURLs {
            guard !outputPathsReferToSameFile(evidencePath, lockURL.path),
                !outputPathsReferToSameFile(progressPath, lockURL.path)
            else {
                throw CompressedAttentionProbeCLIError
                    .outputPathCollision(lockURL.path)
            }
        }
        try requireAbsentRegularDestination(evidenceURL)
        try requireAbsentRegularDestination(progressURL)

        var descriptors: [Int32] = []
        do {
            for lockURL in lockURLs {
                let descriptor = open(
                    lockURL.path,
                    O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                    S_IRUSR | S_IWUSR)
                guard descriptor >= 0 else {
                    throw CompressedAttentionProbeCLIError
                        .outputNotFresh(evidencePath)
                }
                descriptors.append(descriptor)
                try writeLockPID(to: descriptor)
            }
            // Close the preflight-to-lock race for cooperating producers and fail closed if an
            // unrelated writer populated either destination while the lock was being acquired.
            try requireAbsentRegularDestination(evidenceURL)
            try requireAbsentRegularDestination(progressURL)
            return CompressedAttentionProbeOutputLease(
                evidenceURL: evidenceURL,
                progressURL: progressURL,
                lockURL: evidenceLockURL,
                lockURLs: lockURLs,
                lockDescriptors: descriptors)
        } catch {
            for (descriptor, lockURL) in zip(descriptors, lockURLs) {
                _ = close(descriptor)
                _ = unlink(lockURL.path)
            }
            throw error
        }
    }

    func writeProgress(_ progress: CompressedAttentionProbeProgress) throws {
        try requireHeldLock()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        try writeAtomicReplace(
            try encoder.encode(progress),
            to: progressURL)
    }

    func writeEvidence(_ data: Data) throws {
        try requireHeldLock()
        guard !outputPathIsSymbolicLink(evidenceURL.path),
            !FileManager.default.fileExists(atPath: evidenceURL.path)
        else {
            throw CompressedAttentionProbeCLIError
                .evidenceAlreadyWritten(evidenceURL.path)
        }
        let temporaryURL = siblingTemporaryURL(for: evidenceURL)
        defer { try? FileManager.default.removeItem(at: temporaryURL) }
        do {
            try data.write(to: temporaryURL, options: .withoutOverwriting)
            try synchronizeFile(at: temporaryURL)
            try FileManager.default.moveItem(
                at: temporaryURL, to: evidenceURL)
            try synchronizeDirectory(containing: evidenceURL)
        } catch let error as CompressedAttentionProbeCLIError {
            throw error
        } catch {
            throw CompressedAttentionProbeCLIError
                .evidenceAlreadyWritten(evidenceURL.path)
        }
    }

    func release() {
        guard !lockDescriptors.isEmpty else { return }
        let descriptors = lockDescriptors
        lockDescriptors.removeAll(keepingCapacity: false)
        for (descriptor, lockURL) in zip(descriptors, lockURLs) {
            _ = close(descriptor)
            _ = unlink(lockURL.path)
        }
    }

    deinit { release() }

    private func requireHeldLock() throws {
        guard lockDescriptors.count == lockURLs.count else {
            throw CompressedAttentionProbeCLIError.lockNotHeld
        }
    }

    private static func siblingLockURL(for outputURL: URL) -> URL {
        outputURL.deletingLastPathComponent().appendingPathComponent(
            ".\(outputURL.lastPathComponent).fastmlx-compressed-attention.lock")
    }

    private static func requireAbsentRegularDestination(
        _ url: URL
    ) throws {
        guard !outputPathIsSymbolicLink(url.path),
            !FileManager.default.fileExists(atPath: url.path)
        else {
            throw CompressedAttentionProbeCLIError.outputNotFresh(url.path)
        }
        let parent = url.deletingLastPathComponent()
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: parent.path, isDirectory: &isDirectory),
            isDirectory.boolValue,
            FileManager.default.isWritableFile(atPath: parent.path)
        else {
            throw CompressedAttentionProbeCLIError.outputNotFresh(url.path)
        }
    }

    private static func writeLockPID(to descriptor: Int32) throws {
        let data = Data(
            "\(ProcessInfo.processInfo.processIdentifier)\n".utf8)
        try data.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            var offset = 0
            while offset < rawBuffer.count {
                let written = Darwin.write(
                    descriptor,
                    baseAddress.advanced(by: offset),
                    rawBuffer.count - offset)
                if written < 0, errno == EINTR { continue }
                guard written > 0 else {
                    throw CompressedAttentionProbeCLIError.ioFailure(
                        operation: "write-lock-pid", code: errno)
                }
                offset += written
            }
        }
        guard fsync(descriptor) == 0 else {
            throw CompressedAttentionProbeCLIError.ioFailure(
                operation: "fsync-lock-pid", code: errno)
        }
    }

    private func writeAtomicReplace(_ data: Data, to url: URL) throws {
        guard !outputPathIsSymbolicLink(url.path) else {
            throw CompressedAttentionProbeCLIError.outputNotFresh(url.path)
        }
        if FileManager.default.fileExists(atPath: url.path) {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true else {
                throw CompressedAttentionProbeCLIError.outputNotFresh(url.path)
            }
        }
        let temporaryURL = siblingTemporaryURL(for: url)
        defer { try? FileManager.default.removeItem(at: temporaryURL) }
        try data.write(to: temporaryURL, options: .withoutOverwriting)
        try synchronizeFile(at: temporaryURL)
        guard rename(temporaryURL.path, url.path) == 0 else {
            throw CompressedAttentionProbeCLIError.ioFailure(
                operation: "rename-progress", code: errno)
        }
        try synchronizeDirectory(containing: url)
    }

    private func siblingTemporaryURL(for url: URL) -> URL {
        url.deletingLastPathComponent().appendingPathComponent(
            ".\(url.lastPathComponent).\(UUID().uuidString).tmp")
    }

    private func synchronizeFile(at url: URL) throws {
        let handle = try FileHandle(forWritingTo: url)
        do {
            try handle.synchronize()
            try handle.close()
        } catch {
            try? handle.close()
            throw error
        }
    }

    private func synchronizeDirectory(containing url: URL) throws {
        let descriptor = open(
            url.deletingLastPathComponent().path,
            O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw CompressedAttentionProbeCLIError.ioFailure(
                operation: "open-output-directory", code: errno)
        }
        defer { _ = close(descriptor) }
        guard fsync(descriptor) == 0 else {
            throw CompressedAttentionProbeCLIError.ioFailure(
                operation: "fsync-output-directory", code: errno)
        }
    }
}

/// Strong identities for every mutable model-side input used to label the synthetic geometry
/// probe. Checkpoint identity authenticates the exact config/index bytes plus each sorted shard's
/// logical name, size, and complete contents. Qualification therefore cannot silently retain an
/// identity after a same-size checkpoint replacement.
func compressedAttentionProbeModelIdentity(
    modelID: String,
    modelPath: String
) throws -> CompressedAttentionProbeModelIdentity {
    let directory = URL(fileURLWithPath: modelPath)
        .standardizedFileURL
    let configURL = directory.appendingPathComponent("config.json")
    let tokenizerConfigURL = directory.appendingPathComponent(
        "tokenizer_config.json")
    guard let configData = try? Data(contentsOf: configURL) else {
        throw CompressedAttentionProbeCLIError
            .unreadableModelFile(configURL.path)
    }
    guard let tokenizerConfigData = try? Data(
        contentsOf: tokenizerConfigURL)
    else {
        throw CompressedAttentionProbeCLIError
            .unreadableModelFile(tokenizerConfigURL.path)
    }

    let checkpointManifestSHA256 =
        try compressedAttentionProbeCheckpointManifestSHA256(
            modelPath: modelPath,
            exactConfigData: configData)

    return CompressedAttentionProbeModelIdentity(
        modelID: modelID,
        modelConfigSHA256: sha256Hex(configData),
        checkpointManifestSHA256: checkpointManifestSHA256,
        tokenizerSHA256: try ProvenanceCLI.tokenizerManifestSHA256(
            at: directory.path),
        tokenizerConfigSHA256: sha256Hex(tokenizerConfigData))
}

/// Binds operator-supplied synthetic tensor geometry to the authenticated model config. The
/// shared top-level/text-config fields cover the dense Qwen and Llama families selected for the
/// Phase 0 matrix; architectures without an exact mapping are refused instead of guessed.
func validateCompressedAttentionProbeModelGeometry(
    plan: CompressedAttentionProbePlan,
    modelPath: String
) throws {
    let configURL = URL(fileURLWithPath: modelPath)
        .standardizedFileURL.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
        let root = try? JSONSerialization.jsonObject(with: data)
            as? [String: Any]
    else {
        throw CompressedAttentionProbeCLIError
            .unsupportedModelConfig(configURL.path)
    }
    let config = (root["text_config"] as? [String: Any]) ?? root
    guard let queryHeads = compressedAttentionProbeJSONInteger(
        config["num_attention_heads"]),
        let hiddenSize = compressedAttentionProbeJSONInteger(
            config["hidden_size"]),
        queryHeads > 0,
        let kvHeads = compressedAttentionProbeJSONInteger(
            config["num_key_value_heads"] ?? queryHeads),
        kvHeads > 0,
        let maxContext = compressedAttentionProbeJSONInteger(
            config["max_position_embeddings"]
                ?? config["max_sequence_length"]),
        maxContext > 0
    else {
        throw CompressedAttentionProbeCLIError
            .unsupportedModelConfig(configURL.path)
    }
    let headDimension: Int
    if let explicit = compressedAttentionProbeJSONInteger(
        config["head_dim"]), explicit > 0
    {
        headDimension = explicit
    } else {
        guard hiddenSize > 0, hiddenSize.isMultiple(of: queryHeads) else {
            throw CompressedAttentionProbeCLIError
                .unsupportedModelConfig(configURL.path)
        }
        headDimension = hiddenSize / queryHeads
    }
    guard plan.queryHeadCount == queryHeads else {
        throw CompressedAttentionProbeCLIError
            .modelGeometryMismatch("queryHeadCount")
    }
    guard plan.kvHeadCount == kvHeads else {
        throw CompressedAttentionProbeCLIError
            .modelGeometryMismatch("kvHeadCount")
    }
    guard plan.headDimension == headDimension else {
        throw CompressedAttentionProbeCLIError
            .modelGeometryMismatch("headDimension")
    }
    let (contextWindowTokens, contextWindowOverflow) = plan.contextTokens
        .addingReportingOverflow(plan.outputTokens)
    guard !contextWindowOverflow, contextWindowTokens <= maxContext else {
        throw CompressedAttentionProbeCLIError
            .modelGeometryMismatch("contextWindowTokens")
    }
}

private func compressedAttentionProbeJSONInteger(_ value: Any?) -> Int? {
    guard let number = value as? NSNumber,
        CFGetTypeID(number) != CFBooleanGetTypeID()
    else { return nil }
    let double = number.doubleValue
    guard double.isFinite, double.rounded() == double,
        double >= Double(Int.min), double <= Double(Int.max)
    else { return nil }
    return Int(double)
}

private struct CompressedAttentionProbeSystemSnapshot: Sendable {
    let residentBytes: Int
    let physicalFootprintBytes: Int
    let lowPowerModeEnabled: Bool
    let powerSource: CompressedAttentionProbePowerSource
    let thermalState: CompressedAttentionProbeThermalState
}

private struct CompressedAttentionProbeMeasuredRun: Sendable {
    let role: CompressedAttentionProbeRunRole
    let operation: CompressedAttentionProbeOperation
    let position: CompressedAttentionProbeRunPosition
    let result: CompressedAttentionProbeNumericResult
    let timing: CompressedAttentionProbeTiming
    let memoryBefore: CompressedAttentionProbeMLXMemorySnapshot
    let memoryAfter: CompressedAttentionProbeMLXMemorySnapshot
    let systemBefore: CompressedAttentionProbeSystemSnapshot
    let systemAfter: CompressedAttentionProbeSystemSnapshot
}

func executeCompressedAttentionProbe(
    command: CompressedAttentionProbeCommand,
    packageIdentity: CompressedAttentionProbePackageIdentity
) async throws -> CompressedAttentionProbeEvidence {
    let lease = try CompressedAttentionProbeOutputLease.acquire(
        evidencePath: command.plan.evidenceOutputPath,
        progressPath: command.plan.progressOutputPath)
    defer { lease.release() }

    let startedAt = ProcessInfo.processInfo.systemUptime
    var completedWarmupRuns = 0
    var completedMeasuredRows = 0
    let totalMeasuredRows = command.plan.measuredRuns * 2

    func progress(
        status: CompressedAttentionProbeProgressStatus,
        block: Int? = nil,
        role: CompressedAttentionProbeRunRole? = nil
    ) throws {
        let memory = compressedAttentionProbeProcessMemory()
        try lease.writeProgress(CompressedAttentionProbeProgress(
            schemaVersion: 1,
            status: status,
            completedWarmupRuns: completedWarmupRuns,
            completedMeasuredRows: completedMeasuredRows,
            totalMeasuredRows: totalMeasuredRows,
            activeBlockIndex: block,
            activeRole: role,
            elapsedSeconds: max(
                0, ProcessInfo.processInfo.systemUptime - startedAt),
            processResidentBytes: memory.residentBytes,
            processPhysicalFootprintBytes: memory.physicalFootprintBytes,
            harnessGitSHA: command.plan.harnessGitSHA,
            workloadNonce: command.plan.workloadNonce))
    }

    do {
        guard compressedAttentionProbeSupports(command.plan) else {
            throw CompressedAttentionProbeCLIError
                .unsupportedOperationLayout
        }
        #if DEBUG
        if command.plan.qualificationEvidence {
            throw CompressedAttentionProbeCLIError.releaseBuildRequired
        }
        #endif

        try progress(status: .starting)
        let modelIdentity = try compressedAttentionProbeModelIdentity(
            modelID: command.modelID,
            modelPath: command.modelPath)
        try validateCompressedAttentionProbeModelGeometry(
            plan: command.plan,
            modelPath: command.modelPath)
        let referencePlan = try compressedAttentionProbeReferencePlan(
            for: command.plan)

        // The actor owns every MLX mutation and array. Output freshness and the exclusive lock
        // have already succeeded before this first MLX call.
        let runner = CompressedAttentionProbeRunner()
        await runner.configureMemory(
            memoryLimitBytes: command.memoryLimitBytes,
            cacheLimitBytes: command.cacheLimitBytes)

        for warmup in 0 ..< command.plan.warmupRuns {
            let orderedRoles = compressedAttentionProbeOrder(block: warmup)
            for role in orderedRoles {
                try progress(status: .warming, block: warmup, role: role)
                let plan = role == .candidate
                    ? command.plan : referencePlan
                _ = try await runner.runFixture(plan: plan)
                completedWarmupRuns += 1
            }
        }

        let memorySettings = CompressedAttentionProbeMemorySettings(
            memoryLimitBytes: command.memoryLimitBytes,
            cacheLimitBytes: command.cacheLimitBytes,
            wiredLimitBytes: command.wiredLimitBytes,
            cacheResetPolicy: .preserveAcrossRun)
        var rows: [CompressedAttentionProbeRunRow] = []
        rows.reserveCapacity(totalMeasuredRows)

        for block in 0 ..< command.plan.measuredRuns {
            var paired: [CompressedAttentionProbeMeasuredRun] = []
            for (position, role) in compressedAttentionProbeOrder(
                block: block).enumerated()
            {
                try progress(status: .running, block: block, role: role)
                let plan = role == .candidate
                    ? command.plan : referencePlan
                let measured = try await compressedAttentionProbeMeasure(
                    runner: runner,
                    plan: plan,
                    role: role,
                    position: CompressedAttentionProbeRunPosition(
                        pairedBlockIndex: block,
                        runPosition: position))
                paired.append(measured)
                completedMeasuredRows += 1
                try progress(status: .running, block: block, role: role)
            }
            guard let candidate = paired.first(where: {
                $0.role == .candidate
            }), let reference = paired.first(where: {
                $0.role == .fp16Reference
            }) else {
                throw CompressedAttentionProbeCLIError
                    .unsupportedOperationLayout
            }
            let numeric = CompressedAttentionProbeNumericControls(
                candidateMaxAbsoluteError:
                    candidate.result.maxAbsoluteError,
                candidateMaxRelativeError:
                    candidate.result.maxRelativeError,
                candidateMaximumToleranceRatio:
                    candidate.result.maximumToleranceRatio,
                candidateTop1Index: candidate.result.outputTop1Index,
                candidateOracleTop1Index:
                    candidate.result.oracleTop1Index,
                referenceMaxAbsoluteError:
                    reference.result.maxAbsoluteError,
                referenceMaxRelativeError:
                    reference.result.maxRelativeError,
                referenceMaximumToleranceRatio:
                    reference.result.maximumToleranceRatio,
                referenceTop1Index: reference.result.outputTop1Index,
                referenceOracleTop1Index:
                    reference.result.oracleTop1Index)
            rows.append(contentsOf: try paired.map {
                try compressedAttentionProbeRow(
                    measured: $0,
                    numericControls: numeric,
                    memorySettings: memorySettings)
            })
        }

        let planIdentity = CompressedAttentionProbePlanIdentity(
            plan: command.plan)
        let artifactID = try CompressedAttentionProbeEvidence
            .deriveArtifactID(
                plan: planIdentity,
                model: modelIdentity,
                package: packageIdentity,
                rows: rows)
        let evidence = try CompressedAttentionProbeEvidence(
            schemaVersion: CompressedAttentionProbeEvidence.schemaVersion,
            evidenceKind: .checkpointAuthenticatedSyntheticGeometry,
            artifactID: artifactID,
            plan: planIdentity,
            model: modelIdentity,
            package: packageIdentity,
            rows: rows).validated()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var evidenceData = try encoder.encode(evidence)
        evidenceData.append(0x0a)
        try lease.writeEvidence(evidenceData)
        try progress(status: .complete)
        return evidence
    } catch {
        try? progress(status: .failed)
        throw error
    }
}

private func compressedAttentionProbeSupports(
    _ plan: CompressedAttentionProbePlan
) -> Bool {
    switch (plan.operation, plan.layout) {
    case (.fp16SDPA, .fp16),
        (.swiftLMQuantizedAttention, .affine),
        (.splitAffineQuantizedMM, .affine),
        (.materializeThenSDPA, .affine):
        true
    default:
        false
    }
}

private func compressedAttentionProbeReferencePlan(
    for plan: CompressedAttentionProbePlan
) throws -> CompressedAttentionProbePlan {
    try CompressedAttentionProbePlan(
        operation: .fp16SDPA,
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
        layout: .fp16,
        warmupRuns: plan.warmupRuns,
        measuredRuns: plan.measuredRuns,
        seed: plan.seed,
        workloadNonce: plan.workloadNonce,
        harnessGitSHA: plan.harnessGitSHA,
        qualificationEvidence: plan.qualificationEvidence,
        evidenceOutputPath: plan.evidenceOutputPath,
        progressOutputPath: plan.progressOutputPath)
}

private func compressedAttentionProbeOrder(
    block: Int
) -> [CompressedAttentionProbeRunRole] {
    block.isMultiple(of: 2)
        ? [.candidate, .fp16Reference]
        : [.fp16Reference, .candidate]
}

private func compressedAttentionProbeMeasure(
    runner: CompressedAttentionProbeRunner,
    plan: CompressedAttentionProbePlan,
    role: CompressedAttentionProbeRunRole,
    position: CompressedAttentionProbeRunPosition
) async throws -> CompressedAttentionProbeMeasuredRun {
    let systemBefore = compressedAttentionProbeSystemSnapshot()
    let startedAt = ProcessInfo.processInfo.systemUptime
    let result = try await runner.runFixture(plan: plan)
    let endedAt = ProcessInfo.processInfo.systemUptime
    let systemAfter = compressedAttentionProbeSystemSnapshot()
    return CompressedAttentionProbeMeasuredRun(
        role: role,
        operation: role == .candidate ? plan.operation : .fp16SDPA,
        position: position,
        result: result,
        timing: CompressedAttentionProbeTiming(
            monotonicStartSeconds: startedAt,
            monotonicEndSeconds: endedAt,
            wallClockSeconds: endedAt - startedAt,
            attentionSeconds: result.attentionSeconds),
        memoryBefore: result.attentionMemoryBefore,
        memoryAfter: result.attentionMemoryAfter,
        systemBefore: systemBefore,
        systemAfter: systemAfter)
}

private func compressedAttentionProbeRow(
    measured: CompressedAttentionProbeMeasuredRun,
    numericControls: CompressedAttentionProbeNumericControls,
    memorySettings: CompressedAttentionProbeMemorySettings
) throws -> CompressedAttentionProbeRunRow {
    let result = measured.result
    let mlxMemory = CompressedAttentionProbeMLXMemoryReceipts(
        before: measured.memoryBefore,
        after: measured.memoryAfter)
    let workspace = try CompressedAttentionProbeWorkspaceBytes.derive(
        persistentKVBytes: result.persistentBytes,
        materializationBytes: result.materializationWorkspaceBytes,
        mlxMemory: mlxMemory)
    let bytes = CompressedAttentionProbeByteReceipts(
        payloadBytes: result.payloadBytes,
        scaleBytes: result.scaleBytes,
        biasBytes: result.biasBytes,
        controlBytes: result.controlBytes,
        alignmentPaddingBytes: result.alignmentPaddingBytes,
        fp16ResidentBytes: result.fp16ResidentBytes,
        persistentKVBytes: result.persistentBytes,
        materializationBytes: result.materializationWorkspaceBytes,
        otherWorkspaceBytes: workspace.otherWorkspaceBytes,
        peakTemporaryBytes: workspace.peakTemporaryBytes,
        totalBytes: workspace.totalBytes)
    return CompressedAttentionProbeRunRow(
        role: measured.role,
        operation: measured.operation,
        position: measured.position,
        receipts: CompressedAttentionProbeRunReceipts(
            timing: measured.timing,
            bytes: bytes,
            mlxMemory: mlxMemory,
            processRSS: CompressedAttentionProbeProcessRSS(
                residentSizeBeforeBytes:
                    measured.systemBefore.residentBytes,
                residentSizeAfterBytes:
                    measured.systemAfter.residentBytes,
                physicalFootprintBeforeBytes:
                    measured.systemBefore.physicalFootprintBytes,
                physicalFootprintAfterBytes:
                    measured.systemAfter.physicalFootprintBytes),
            memorySettings: memorySettings,
            power: CompressedAttentionProbePowerReceipts(
                lowPowerModeEnabledBefore:
                    measured.systemBefore.lowPowerModeEnabled,
                lowPowerModeEnabledAfter:
                    measured.systemAfter.lowPowerModeEnabled,
                powerSourceBefore: measured.systemBefore.powerSource,
                powerSourceAfter: measured.systemAfter.powerSource),
            thermal: CompressedAttentionProbeThermalReceipts(
                before: measured.systemBefore.thermalState,
                after: measured.systemAfter.thermalState),
            numericControls: numericControls,
            hashes: CompressedAttentionProbeHashes(
                sourceKVTensorSHA256: result.sourceKVTensorSHA256,
                packedKVTensorSHA256: result.packedKVTensorSHA256,
                queryTensorSHA256: result.queryTensorSHA256,
                outputTensorSHA256: result.outputTensorSHA256)))
}

private func compressedAttentionProbeSystemSnapshot()
    -> CompressedAttentionProbeSystemSnapshot
{
    let process = ProcessInfo.processInfo
    let memory = compressedAttentionProbeProcessMemory()
    return CompressedAttentionProbeSystemSnapshot(
        residentBytes: memory.residentBytes,
        physicalFootprintBytes: memory.physicalFootprintBytes,
        lowPowerModeEnabled: process.isLowPowerModeEnabled,
        powerSource: compressedAttentionProbePowerSource(),
        thermalState: compressedAttentionProbeThermalState(
            process.thermalState))
}

private func compressedAttentionProbeProcessMemory() -> (
    residentBytes: Int, physicalFootprintBytes: Int
) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size
            / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(
            to: integer_t.self, capacity: Int(count)
        ) {
            task_info(
                mach_task_self_,
                task_flavor_t(TASK_VM_INFO),
                $0,
                &count)
        }
    }
    guard result == KERN_SUCCESS else { return (0, 0) }
    return (
        residentBytes: Int(info.resident_size),
        physicalFootprintBytes: Int(info.phys_footprint))
}

private func compressedAttentionProbePowerSource()
    -> CompressedAttentionProbePowerSource
{
    guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
        let source = IOPSGetProvidingPowerSourceType(snapshot)?
            .takeUnretainedValue()
    else { return .unavailable }
    let value = ((source as NSString) as String).lowercased()
    if value.contains("battery") { return .battery }
    if value.contains("ac") || value.contains("ups") { return .acPower }
    return .unavailable
}

private func compressedAttentionProbeThermalState(
    _ state: ProcessInfo.ThermalState
) -> CompressedAttentionProbeThermalState {
    switch state {
    case .nominal: .nominal
    case .fair: .fair
    case .serious: .serious
    case .critical: .critical
    @unknown default: .unknown
    }
}

func runCompressedAttentionProbe(_ arguments: [String]) async {
    do {
        let harnessGitSHA = try ProvenanceCLI.qualificationHarnessGitSHA()
        let command = try CompressedAttentionProbeCommand(
            arguments: arguments,
            harnessGitSHA: harnessGitSHA)
        let evidence = try await executeCompressedAttentionProbe(
            command: command,
            packageIdentity: compressedAttentionProbePackageIdentity())
        print(
            "# compressed-attention-probe: complete "
                + "artifact=\(evidence.artifactID) "
                + "rows=\(evidence.rows.count) "
                + "evidence=\(command.plan.evidenceOutputPath)")
    } catch {
        let message = "compressed-attention-probe failed: \(error)\n"
        FileHandle.standardError.write(Data(message.utf8))
        exit(1)
    }
}

private func compressedAttentionProbePackageIdentity()
    -> CompressedAttentionProbePackageIdentity
{
    #if DEBUG
    let buildConfiguration = "Debug"
    #else
    let buildConfiguration = "Release"
    #endif
    return CompressedAttentionProbePackageIdentity(
        mlxSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
        mlxSwiftLMRevision: ProvenanceCLI.mlxSwiftLMRevision,
        swiftVersion: compressedAttentionProbeSwiftCompilerVersion(),
        harnessBuildConfiguration: buildConfiguration)
}

private func compressedAttentionProbeSwiftCompilerVersion() -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
    process.arguments = ["swiftc", "--version"]
    let output = Pipe()
    process.standardOutput = output
    process.standardError = Pipe()
    do {
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0,
            let raw = String(data: data, encoding: .utf8)
        else { return "Swift 6" }
        let normalized = raw.split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
        return normalized.isEmpty ? "Swift 6" : normalized
    } catch {
        return "Swift 6"
    }
}

private func compressedAttentionProbeCheckpointManifestSHA256(
    modelPath: String,
    exactConfigData: Data
) throws -> String {
    do {
        return try ProvenanceCLI.fullContentCheckpointManifestSHA256(
            at: modelPath,
            exactConfigData: exactConfigData)
    } catch let error as ProvenanceCLI.EvidenceIdentityError {
        switch error {
        case .missingCheckpointWeights(let path):
            throw CompressedAttentionProbeCLIError
                .missingCheckpointWeights(path)
        case .invalidCheckpointWeight(let path):
            throw CompressedAttentionProbeCLIError
                .invalidCheckpointWeight(path)
        case .unreadableCheckpointManifestFile(let path),
             .unreadableModelConfig(let path):
            throw CompressedAttentionProbeCLIError.unreadableModelFile(path)
        case .missingTokenizerFiles(_),
             .invalidTokenizerFile(_):
            throw error
        }
    }
}
