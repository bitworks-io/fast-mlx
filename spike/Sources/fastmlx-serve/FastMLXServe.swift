import Darwin
import Dispatch
import Foundation
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
        let configuration = ServingHTTPConfiguration(
            launchedModel: arguments.model,
            requestLimits: .productionDefault,
            requiredBearerToken: apiKey,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(5))
        let server: ServingHTTPServer
        do {
            server = try await ServingHTTPServer.start(
                bind: .init(host: arguments.host, port: arguments.port),
                configuration: configuration,
                backend: prepared.backend)
        } catch {
            await prepared.backend.shutdown()
            throw error
        }

        print(prepared.startupLine(localAddress: server.localAddress.description))
        print("fastmlx-serve ready=true; press Control-C to stop.")

        await waitForShutdownSignal()
        try await server.shutdown(gracePeriod: .seconds(5))
        print("fastmlx-serve shutdown=complete")
    }
}

private struct PreparedServingBackend {
    let backend: any ServingGenerationBackend
    let mode: String
    let launchedModel: String
    let startupReport: ScalarServingModelStartupReport?

    func startupLine(localAddress: String) -> String {
        guard let startupReport else {
            return """
                fastmlx-serve mode=\(mode) transport_only=true \
                model=\(launchedModel) listening=\(localAddress)
                """
        }
        let nativeCacheKinds = Set(
            startupReport.nativeCacheKinds.map(\.rawValue)
        ).sorted().joined(separator: ",")
        return """
            fastmlx-serve mode=\(mode) route=\(startupReport.route.rawValue) \
            model=\(startupReport.launchedModel) \
            memory_limit_bytes=\(startupReport.memoryLimitBytes) \
            cache_limit_bytes=\(startupReport.cacheLimitBytes) \
            native_cache_layers=\(startupReport.nativeCacheKinds.count) \
            native_cache_kinds=\(nativeCacheKinds) \
            stop_token_count=\(startupReport.stopTokenCount) \
            stop_string_count=\(startupReport.stopStringCount) \
            startup_prompt_token_count=\(startupReport.startupPromptTokenCount) \
            startup_generated_token_count=\(startupReport.startupGeneratedTokenCount) \
            reset_parity_verified=\(startupReport.resetParityVerified) \
            listening=\(localAddress)
            """
    }
}

private func prepareBackend(
    _ arguments: FastMLXServeArguments
) async throws -> PreparedServingBackend {
    switch arguments.backend {
    case .scripted:
        return PreparedServingBackend(
            backend: ScriptedTransportBackend(),
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
            mode: "scalar",
            launchedModel: arguments.model,
            startupReport: loaded.startupReport)
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
