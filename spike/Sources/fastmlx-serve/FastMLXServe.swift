import Darwin
import Dispatch
import Foundation
import HarnessCore
import ServingCore
import ServingNIO
import SpikeServingAdapters

@main
struct FastMLXServe {
    static func main() async throws {
        let arguments = try FastMLXServeArguments.parse(
            CommandLine.arguments.dropFirst())
        if arguments.showHelp {
            print(FastMLXServeArguments.usage)
            return
        }

        let apiKey = ProcessInfo.processInfo.environment["FASTMLX_API_KEY"].flatMap {
            $0.isEmpty ? nil : $0
        }
        let prepared = try await prepareBackend(arguments)
        let evidenceSink: ServingEvidenceJSONLSink?
        do {
            evidenceSink = try arguments.evidencePath.map {
                try ServingEvidenceJSONLSink(path: $0.path)
            }
        } catch {
            await prepared.backend.shutdown()
            throw error
        }
        let evidenceConfiguration = evidenceSink.map { sink in
            ServingHTTPEvidenceConfiguration(
                snapshot: prepared.evidenceSnapshot,
                record: { evidence in
                    try await sink.record(evidence)
                },
                reportFailure: { message in
                    FileHandle.standardError.write(
                        Data("fastmlx-serve evidence_failure=\(message)\n".utf8))
                })
        }
        let configuration = ServingHTTPConfiguration(
            launchedModel: arguments.model,
            requestLimits: .productionDefault,
            requiredBearerToken: apiKey,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(5),
            evidence: evidenceConfiguration)
        let server: ServingHTTPServer
        do {
            server = try await ServingHTTPServer.start(
                bind: .init(host: arguments.host, port: arguments.port),
                configuration: configuration,
                backend: prepared.backend)
        } catch {
            try? await evidenceSink?.finish()
            await prepared.backend.shutdown()
            throw error
        }

        print(prepared.startupLine(localAddress: server.localAddress.description))
        print("fastmlx-serve ready=true; press Control-C to stop.")

        await waitForShutdownSignal()
        do {
            try await server.shutdown(gracePeriod: .seconds(5))
            try await evidenceSink?.finish()
        } catch {
            try? await evidenceSink?.finish()
            throw error
        }
        print("fastmlx-serve shutdown=complete")
    }
}

private enum PreparedServingStartupReport {
    case scalar(ScalarServingModelStartupReport)
    case continuous(ContinuousServingModelStartupReport)
}

private struct PreparedServingBackend {
    let backend: any ServingGenerationBackend
    let evidenceSnapshot:
        ServingHTTPEvidenceConfiguration.SnapshotProvider?
    let mode: String
    let launchedModel: String
    let startupReport: PreparedServingStartupReport?

    func startupLine(localAddress: String) -> String {
        guard let startupReport else {
            return """
                fastmlx-serve mode=\(mode) transport_only=true \
                model=\(launchedModel) listening=\(localAddress)
                """
        }
        switch startupReport {
        case .scalar(let report):
            let nativeCacheKinds = Set(
                report.nativeCacheKinds.map(\.rawValue)
            ).sorted().joined(separator: ",")
            return """
                fastmlx-serve mode=\(mode) route=\(report.route.rawValue) \
                model=\(report.launchedModel) \
                memory_limit_bytes=\(report.memoryLimitBytes) \
                cache_limit_bytes=\(report.cacheLimitBytes) \
                native_cache_layers=\(report.nativeCacheKinds.count) \
                native_cache_kinds=\(nativeCacheKinds) \
                stop_token_count=\(report.stopTokenCount) \
                stop_string_count=\(report.stopStringCount) \
                startup_prompt_token_count=\(report.startupPromptTokenCount) \
                startup_generated_token_count=\(report.startupGeneratedTokenCount) \
                reset_parity_verified=\(report.resetParityVerified) \
                listening=\(localAddress)
                """
        case .continuous(let report):
            let nativeCacheKinds = Set(
                report.nativeCacheKinds.map(\.rawValue)
            ).sorted().joined(separator: ",")
            let soloPLD = report.soloPLDPolicy
            return """
                fastmlx-serve mode=\(mode) route=\(report.route.rawValue) \
                model=\(report.launchedModel) \
                memory_limit_bytes=\(report.memoryLimitBytes) \
                cache_limit_bytes=\(report.cacheLimitBytes) \
                max_reserved_kv_bytes=\(report.maxReservedKVBytes) \
                max_context_tokens=\(report.maxContextTokens) \
                max_reserved_context_tokens=\(report.maxReservedContextTokens) \
                model_family=\(report.modelFamily.rawValue) \
                model_config_sha256=\(report.modelConfigurationSHA256) \
                model_layers=\(report.layerCount) \
                kv_heads=\(report.keyValueHeadCount) \
                head_dimension=\(report.headDimension) \
                native_cache_layers=\(report.nativeCacheKinds.count) \
                native_cache_kinds=\(nativeCacheKinds) \
                stop_token_count=\(report.stopTokenCount) \
                stop_string_count=\(report.stopStringCount) \
                startup_prompt_token_count=\(report.startupPromptTokenCount) \
                startup_generated_token_count=\(report.startupGeneratedTokenCount) \
                max_active_slots=\(report.maxActiveSlots) \
                max_prefill_slots=\(report.maxPrefillSlots) \
                prefill_chunk_size=\(report.prefillChunkSize) \
                max_queued_requests=\(report.maxQueuedRequests) \
                publication_capacity=\(report.publicationCapacity) \
                solo_pld_policy_configured=\(soloPLD != nil) \
                solo_pld_ngram=\(soloPLD?.ngram ?? 0) \
                solo_pld_max_draft=\(soloPLD?.maxDraft ?? 0) \
                solo_pld_lookback=\(soloPLD?.lookback ?? 0) \
                solo_pld_compiled_verify=\(soloPLD?.compiledVerify ?? false) \
                model_proof_verified=\(report.modelProofVerified) \
                listening=\(localAddress)
                """
        }
    }
}

private func prepareBackend(
    _ arguments: FastMLXServeArguments
) async throws -> PreparedServingBackend {
    switch arguments.backend {
    case .scripted:
        return PreparedServingBackend(
            backend: ScriptedTransportBackend(),
            evidenceSnapshot: nil,
            mode: "scripted",
            launchedModel: arguments.model,
            startupReport: nil)
    case .scalar(
        let modelDirectory,
        let memoryLimitBytes,
        let cacheLimitBytes
    ):
        let loaded = try await loadScalarServingModel(
            configuration: ScalarServingModelLoadConfiguration(
                launchedModel: arguments.model,
                modelDirectory: modelDirectory,
                memoryLimitBytes: memoryLimitBytes,
                cacheLimitBytes: cacheLimitBytes,
                backendConfiguration: ScalarServingBackendConfiguration(
                    defaultMaximumCompletionTokens: 512,
                    maximumQueuedRequests: 2,
                    queueRetryAfterSeconds: 1,
                    mailboxCapacity: .init(
                        maxDeltas: 8,
                        maxBytes: 32 * 1_024))))
        return PreparedServingBackend(
            backend: loaded.backend,
            evidenceSnapshot: nil,
            mode: "scalar",
            launchedModel: arguments.model,
            startupReport: .scalar(loaded.startupReport))
    case .continuousBatchNoSpec(
        let modelDirectory,
        let memoryLimitBytes,
        let cacheLimitBytes,
        let maxReservedKVBytes
    ):
        let loaded = try await loadContinuousServingModel(
            configuration: ContinuousServingModelLoadConfiguration(
                launchedModel: arguments.model,
                modelDirectory: modelDirectory,
                memoryLimitBytes: memoryLimitBytes,
                cacheLimitBytes: cacheLimitBytes,
                maxReservedKVBytes: maxReservedKVBytes,
                coordinatorConfiguration: try ContinuousBatchConfiguration(
                    maxActiveSlots: 4,
                    maxPrefillSlots: 2,
                    prefillChunkSize: 512,
                    maxQueuedRequests: 8),
                publicationCapacity: 1,
                backendConfiguration: ContinuousServingBackendConfiguration(
                    defaultMaximumCompletionTokens: 512,
                    queueRetryAfterSeconds: 1,
                    mailboxCapacity: .init(
                        maxDeltas: 8,
                        maxBytes: 32 * 1_024))))
        return PreparedServingBackend(
            backend: loaded.backend,
            evidenceSnapshot: {
                let snapshot = await loaded.backend.snapshot()
                return try ServingEvidence.ResourceSnapshot(
                    activeRequests: snapshot.activeRequests,
                    coordinatorSlots: snapshot.coordinatorSlots,
                    reservedKVBytes: snapshot.reservedKVBytes,
                    maxReservedKVBytes: snapshot.maxReservedKVBytes,
                    mlxActiveBytes: snapshot.mlxActiveBytes,
                    mlxCacheBytes: snapshot.mlxCacheBytes,
                    mlxPeakBytes: snapshot.mlxPeakBytes)
            },
            mode: "continuous-batch-no-spec",
            launchedModel: arguments.model,
            startupReport: .continuous(loaded.startupReport))
    case nil:
        preconditionFailure("help is the only invocation without a backend")
    }
}

private final class ScriptedTransportBackend: ServingGenerationBackend, Sendable {
    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let mailbox = BoundedDeltaMailbox(
            capacity: .init(maxDeltas: 2, maxBytes: 32 * 1_024))
        let lease = ServingRequestLease(
            id: ServingRequestID("scripted-\(UUID().uuidString)"),
            onCancel: { [mailbox] in
                await mailbox.cancel(.clientDisconnected)
            })
        let handle = ServingGenerationHandle(
            responseID: "chatcmpl-scripted-\(UUID().uuidString)",
            created: Int(Date().timeIntervalSince1970),
            model: request.model,
            route: .scriptedTransport,
            mailbox: mailbox,
            lease: lease)

        Task {
            do {
                try await mailbox.send(
                    .text("fast-mlx scripted transport is ready; no model is loaded."))
                try await mailbox.send(
                    .completion(
                        ServingGenerationCompletion(
                            finishReason: .stop,
                            usage: OpenAIChatUsage(
                                promptTokens: 0,
                                completionTokens: 0))))
                await mailbox.finish()
            } catch {
                // The request lease owns terminal and cancellation state.
            }
        }
        return handle
    }
}

private func waitForShutdownSignal() async {
    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)
    signal(SIGPIPE, SIG_IGN)

    let (signals, continuation) = AsyncStream<Int32>.makeStream(
        bufferingPolicy: .bufferingNewest(1))
    let interruptSource = DispatchSource.makeSignalSource(signal: SIGINT)
    let terminateSource = DispatchSource.makeSignalSource(signal: SIGTERM)
    interruptSource.setEventHandler {
        continuation.yield(SIGINT)
    }
    terminateSource.setEventHandler {
        continuation.yield(SIGTERM)
    }
    interruptSource.resume()
    terminateSource.resume()

    for await _ in signals {
        break
    }
    continuation.finish()
    interruptSource.cancel()
    terminateSource.cancel()
}
