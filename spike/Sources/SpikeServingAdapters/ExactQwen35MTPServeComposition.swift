import Foundation

import HarnessCore
import MLXHuggingFace
import MLXLMCommon
import ServingCore
import SpikeCore
import Tokenizers

public struct ExactQwen35MTPServeCompositionConfiguration: Sendable {
    public let launchedModel: String
    public let targetDirectory: URL
    public let drafterDirectory: URL
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let scalarBackendConfiguration: ScalarServingBackendConfiguration
    public let scalarStartupMessages: [OpenAIChatMessage]
    public let kvQuantTier: KVQuantTier

    public init(
        launchedModel: String,
        targetDirectory: URL,
        drafterDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        scalarBackendConfiguration: ScalarServingBackendConfiguration,
        scalarStartupMessages: [OpenAIChatMessage] = ScalarServingModelLoadConfiguration
            .defaultStartupMessages,
        kvQuantTier: KVQuantTier = .fp16
    ) {
        self.launchedModel = launchedModel
        self.targetDirectory = targetDirectory
        self.drafterDirectory = drafterDirectory
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.scalarBackendConfiguration = scalarBackendConfiguration
        self.scalarStartupMessages = scalarStartupMessages
        self.kvQuantTier = kvQuantTier
    }

    var scalarConfiguration: ScalarServingModelLoadConfiguration {
        ScalarServingModelLoadConfiguration(
            launchedModel: launchedModel,
            modelDirectory: targetDirectory,
            memoryLimitBytes: memoryLimitBytes,
            cacheLimitBytes: cacheLimitBytes,
            backendConfiguration: scalarBackendConfiguration,
            startupMessages: scalarStartupMessages,
            kvQuantTier: kvQuantTier)
    }

    var runtimeConfiguration: ExactQwen35MTPRuntimeLoadConfiguration {
        ExactQwen35MTPRuntimeLoadConfiguration(
            targetDirectory: targetDirectory,
            drafterDirectory: drafterDirectory)
    }
}

public struct ExactQwen35MTPRuntimeLoadConfiguration: Sendable {
    public let targetDirectory: URL
    public let drafterDirectory: URL

    public init(targetDirectory: URL, drafterDirectory: URL) {
        self.targetDirectory = targetDirectory
        self.drafterDirectory = drafterDirectory
    }
}

public enum ExactQwen35MTPCompositeFitProfileError: Error, Equatable, Sendable {
    case drafterWeightsUnavailable
    case residentWeightsOverflow
}

/// Builds the fit-check profile for the actual exact-MTP process residency: the scalar fallback's
/// target, the exact runner's separate target, and the drafter. Geometry/KV behavior comes from the
/// target profile because each request executes on one route; the weight term counts all three
/// co-resident artifacts. This is deliberately separate from the ordinary single-model planner so
/// exact mode cannot make its default-off double-load change the shipped scalar/continuous contract.
public enum ExactQwen35MTPCompositeFitProfile {
    public static func make(
        target: ParsedModelArch,
        drafterDirectory: URL
    ) throws -> ParsedModelArch {
        let drafterShards = ModelConfigDecoder.sumSafetensorsBytes(in: drafterDirectory)
        let drafterDeclared = ModelConfigDecoder.declaredSafetensorsBytes(in: drafterDirectory)

        let drafterBytes: Int
        let drafterMeasured: Bool
        let drafterIsDeclared: Bool
        if let declared = drafterDeclared, declared > drafterShards {
            drafterBytes = declared
            drafterMeasured = false
            drafterIsDeclared = true
        } else if drafterShards > 0 {
            drafterBytes = drafterShards
            drafterMeasured = true
            drafterIsDeclared = false
        } else {
            throw ExactQwen35MTPCompositeFitProfileError.drafterWeightsUnavailable
        }

        let (twoTargets, targetOverflow) = target.profile.weightsBytes4bitEstimate
            .multipliedReportingOverflow(by: 2)
        let (residentWeights, totalOverflow) = twoTargets.addingReportingOverflow(drafterBytes)
        guard !targetOverflow, !totalOverflow else {
            throw ExactQwen35MTPCompositeFitProfileError.residentWeightsOverflow
        }

        let targetHasSizedProvenance = target.weightsAreMeasured || target.weightsAreDeclared
        let drafterHasSizedProvenance = drafterMeasured || drafterIsDeclared
        let allMeasured = target.weightsAreMeasured && drafterMeasured
        let allMeasuredOrDeclared = targetHasSizedProvenance && drafterHasSizedProvenance
        let profile = target.profile
        return ParsedModelArch(
            profile: ModelArchProfile(
                id: "\(profile.id)+exact-qwen35-mtp-composition",
                modelType: profile.modelType,
                nLayers: profile.nLayers,
                nAttnLayers: profile.nAttnLayers,
                nKVHeads: profile.nKVHeads,
                headDim: profile.headDim,
                slidingWindow: profile.slidingWindow,
                fixedStateBytes: profile.fixedStateBytes,
                nativeMaxContext: profile.nativeMaxContext,
                weightsBytes4bitEstimate: residentWeights,
                license: profile.license,
                mlaHeads: profile.mlaHeads,
                mlaRopeDim: profile.mlaRopeDim,
                mlaNopeDim: profile.mlaNopeDim,
                mlaVDim: profile.mlaVDim,
                swaKVHeads: profile.swaKVHeads,
                swaHeadDim: profile.swaHeadDim,
                vHeadDim: profile.vHeadDim,
                swaVHeadDim: profile.swaVHeadDim),
            weightsAreMeasured: allMeasured,
            weightsAreDeclared: !allMeasured && allMeasuredOrDeclared,
            quantBits: target.quantBits)
    }
}

public struct ExactQwen35MTPServingRuntimeComponents: Sendable {
    public let runner: any ExactQwen35MTPServingRunner
    public let codec: any ScalarServingTextCodec
    public let descriptor: ExactQwen35MTPServingDescriptor

    public init(
        runner: any ExactQwen35MTPServingRunner,
        codec: sending any ScalarServingTextCodec,
        descriptor: ExactQwen35MTPServingDescriptor
    ) {
        self.runner = runner
        self.codec = codec
        self.descriptor = descriptor
    }
}

public enum ExactQwen35MTPServeScalarFallbackReason: String, Equatable, Sendable {
    case exactRuntimeUnavailable = "exact_runtime_unavailable"
}

public enum ExactQwen35MTPServeStartupStatus: Equatable, Sendable {
    case exactSuccess
    case scalarFallback(reason: ExactQwen35MTPServeScalarFallbackReason)
}

public struct ExactQwen35MTPServeStartupReport: Equatable, Sendable {
    public let status: ExactQwen35MTPServeStartupStatus
    public let descriptor: ExactQwen35MTPServingDescriptor?

    public init(
        status: ExactQwen35MTPServeStartupStatus,
        descriptor: ExactQwen35MTPServingDescriptor?
    ) {
        self.status = status
        self.descriptor = descriptor
    }

    public func machineReadableFields() -> String {
        switch status {
        case .exactSuccess:
            guard let descriptor else {
                return "exact_qwen35_mtp_status=exact_success"
            }
            return [
                "exact_qwen35_mtp_status=exact_success",
                "exact_qwen35_mtp_target_id=\(descriptor.targetModelID)",
                "exact_qwen35_mtp_drafter_id=\(descriptor.drafterModelID)",
                "exact_qwen35_mtp_target_revision=\(descriptor.targetRevision)",
                "exact_qwen35_mtp_drafter_revision=\(descriptor.drafterRevision)",
                "exact_qwen35_mtp_source_revision=\(descriptor.sourceRevision)",
                "exact_qwen35_mtp_block_size=\(descriptor.runtimeBlockSize)",
                "exact_qwen35_mtp_max_draft_tokens=\(descriptor.maximumAcceptedDraftTokens)",
            ].joined(separator: " ")
        case .scalarFallback(let reason):
            return "exact_qwen35_mtp_status=scalar_fallback exact_qwen35_mtp_reason=\(reason.rawValue)"
        }
    }
}

public struct LoadedExactQwen35MTPServeComposition: Sendable {
    public let backend: any ServingGenerationBackend
    public let scalarStartupReport: ScalarServingModelStartupReport
    public let exactStartupReport: ExactQwen35MTPServeStartupReport

    public init(
        backend: any ServingGenerationBackend,
        scalarStartupReport: ScalarServingModelStartupReport,
        exactStartupReport: ExactQwen35MTPServeStartupReport
    ) {
        self.backend = backend
        self.scalarStartupReport = scalarStartupReport
        self.exactStartupReport = exactStartupReport
    }
}

public struct ExactQwen35MTPLoadedScalarFallback: Sendable {
    public let backend: any ServingGenerationBackend
    public let startupReport: ScalarServingModelStartupReport

    public init(
        backend: any ServingGenerationBackend,
        startupReport: ScalarServingModelStartupReport
    ) {
        self.backend = backend
        self.startupReport = startupReport
    }
}

public typealias ExactQwen35MTPScalarLoader =
    @Sendable (ScalarServingModelLoadConfiguration) async throws
        -> ExactQwen35MTPLoadedScalarFallback

public typealias ExactQwen35MTPRuntimeComponentsLoader =
    @Sendable (ExactQwen35MTPRuntimeLoadConfiguration) async throws
        -> ExactQwen35MTPServingRuntimeComponents

public enum ExactQwen35MTPServeCompositionError: Error, Equatable, Sendable {
    case invalidRuntimeBinding
}

public func loadExactQwen35MTPServeComposition(
    configuration: ExactQwen35MTPServeCompositionConfiguration,
    scalarLoader: ExactQwen35MTPScalarLoader = { configuration in
        let loaded = try await loadScalarServingModel(configuration: configuration)
        return ExactQwen35MTPLoadedScalarFallback(
            backend: loaded.backend,
            startupReport: loaded.startupReport)
    },
    runtimeLoader: ExactQwen35MTPRuntimeComponentsLoader = { configuration in
        try await loadExactQwen35MTPRuntimeComponents(configuration: configuration)
    }
) async throws -> LoadedExactQwen35MTPServeComposition {
    let scalar = try await scalarLoader(configuration.scalarConfiguration)
    do {
        let runtime = try await runtimeLoader(configuration.runtimeConfiguration)
        let descriptor = try validatedExactDescriptor(
            runtimeDescriptor: runtime.descriptor,
            runnerBinding: runtime.runner.binding)
        let backend = try ExactQwen35MTPServingBackend(
            launchedModel: configuration.launchedModel,
            enabled: true,
            runner: runtime.runner,
            scalarFallback: scalar.backend,
            scalarFallbackIsolation: .strictlySeparateRawTarget,
            codec: runtime.codec,
            configuration: ExactQwen35MTPServingBackendConfiguration(
                defaultMaximumCompletionTokens: configuration
                    .scalarBackendConfiguration.defaultMaximumCompletionTokens,
                mailboxCapacity: configuration.scalarBackendConfiguration.mailboxCapacity,
                disableThinkingWhenToolsActive: configuration
                    .scalarBackendConfiguration.disableThinkingWhenToolsActive,
                thinksByDefault: configuration.scalarBackendConfiguration.thinksByDefault))
        return LoadedExactQwen35MTPServeComposition(
            backend: backend,
            scalarStartupReport: scalar.startupReport,
            exactStartupReport: ExactQwen35MTPServeStartupReport(
                status: .exactSuccess,
                descriptor: descriptor))
    } catch is CancellationError {
        await scalar.backend.shutdown()
        throw CancellationError()
    } catch {
        return LoadedExactQwen35MTPServeComposition(
            backend: scalar.backend,
            scalarStartupReport: scalar.startupReport,
            exactStartupReport: ExactQwen35MTPServeStartupReport(
                status: .scalarFallback(reason: .exactRuntimeUnavailable),
                descriptor: nil))
    }
}

public func loadExactQwen35MTPRuntimeComponents(
    configuration: ExactQwen35MTPRuntimeLoadConfiguration
) async throws -> ExactQwen35MTPServingRuntimeComponents {
    let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
        from: ExactQwen35MTPLocalSnapshotDownloader(
            targetDirectory: configuration.targetDirectory,
            drafterDirectory: configuration.drafterDirectory),
        using: #huggingFaceTokenizerLoader())
    let codec = MLXScalarTextCodec(tokenizer: pair.target.tokenizer)
    let runner = try ExactQwen35MTPMLXServingRunner(pair: pair)
    let descriptor = try validatedExactDescriptor(
        runtimeDescriptor: nil,
        runnerBinding: runner.binding)
    return ExactQwen35MTPServingRuntimeComponents(
        runner: runner,
        codec: codec,
        descriptor: descriptor)
}

public enum ExactQwen35MTPLocalSnapshotDownloaderError: Error, Equatable, Sendable {
    case unexpectedRequest
    case localDirectoryUnavailable
}

public struct ExactQwen35MTPLocalSnapshotDownloader: Downloader {
    public let targetDirectory: URL
    public let drafterDirectory: URL

    public init(targetDirectory: URL, drafterDirectory: URL) {
        self.targetDirectory = targetDirectory
        self.drafterDirectory = drafterDirectory
    }

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard !useLatest, patterns == ["*.safetensors", "*.json", "*.jinja"] else {
            throw ExactQwen35MTPLocalSnapshotDownloaderError.unexpectedRequest
        }
        let lock = QwenMTPKnownArtifactLocks.qwen35_9BDepth1
        let directory: URL
        switch (id, revision) {
        case (lock.targetIdentity.modelID, lock.targetIdentity.revision):
            directory = targetDirectory
        case (lock.drafterIdentity.modelID, lock.drafterIdentity.revision):
            directory = drafterDirectory
        default:
            throw ExactQwen35MTPLocalSnapshotDownloaderError.unexpectedRequest
        }
        guard directory.isFileURL, directory.path.hasPrefix("/") else {
            throw ExactQwen35MTPLocalSnapshotDownloaderError.localDirectoryUnavailable
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw ExactQwen35MTPLocalSnapshotDownloaderError.localDirectoryUnavailable
        }
        progressHandler(Progress(totalUnitCount: 1))
        return directory
    }
}

private func validatedExactDescriptor(
    runtimeDescriptor: ExactQwen35MTPServingDescriptor?,
    runnerBinding: QwenMTPArtifactBinding?
) throws -> ExactQwen35MTPServingDescriptor {
    guard let binding = runnerBinding else {
        throw ExactQwen35MTPServeCompositionError.invalidRuntimeBinding
    }
    let decision = ExactQwen35MTPServingAdmissionPolicy.decide(
        enabled: true,
        binding: binding,
        sampling: .greedy,
        penalties: .none,
        hasActiveTools: false,
        speculativeRequestCount: 0)
    guard case .eligible(let descriptor) = decision,
        runtimeDescriptor == nil || runtimeDescriptor == descriptor
    else {
        throw ExactQwen35MTPServeCompositionError.invalidRuntimeBinding
    }
    return descriptor
}
