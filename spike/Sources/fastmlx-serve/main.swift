import Darwin
import Dispatch
import Foundation
import ServingCore
import ServingNIO

@main
struct FastMLXServe {
    static func main() async throws {
        let arguments = try Arguments.parse(CommandLine.arguments.dropFirst())
        if arguments.showHelp {
            print(Arguments.usage)
            return
        }
        guard arguments.scripted else {
            throw CLIError(
                "Phase 1 requires --scripted; no model-serving backend is wired yet")
        }

        let apiKey = ProcessInfo.processInfo.environment["FASTMLX_API_KEY"].flatMap {
            $0.isEmpty ? nil : $0
        }
        let configuration = ServingHTTPConfiguration(
            launchedModel: arguments.model,
            requestLimits: .productionDefault,
            requiredBearerToken: apiKey,
            maximumNonStreamingResponseBytes: 1_048_576,
            backpressureStallTimeout: .seconds(5))
        let server = try await ServingHTTPServer.start(
            bind: .init(host: arguments.host, port: arguments.port),
            configuration: configuration,
            backend: ScriptedTransportBackend())

        print(
            "fastmlx-serve mode=scripted transport_only=true listening=\(server.localAddress)")
        print("No model is loaded; press Control-C to stop.")

        await waitForShutdownSignal()
        try await server.shutdown(gracePeriod: .seconds(5))
        print("fastmlx-serve shutdown=complete")
    }
}

private struct Arguments {
    static let usage = """
    Usage: fastmlx-serve --scripted [--host HOST] [--port PORT] [--model MODEL]

      --scripted     Required Phase 1 transport-only backend.
      --host HOST    Bind host (default: 127.0.0.1).
      --port PORT    Bind port (default: 8080; use 0 for an ephemeral port).
      --model MODEL  OpenAI request model identifier (default: fastmlx-scripted).
      --help         Show this help.

    Set FASTMLX_API_KEY to require Bearer authentication. A non-loopback host is
    rejected unless that environment variable is non-empty.
    """

    var scripted = false
    var host = "127.0.0.1"
    var port = 8_080
    var model = "fastmlx-scripted"
    var showHelp = false

    static func parse<S: Sequence>(_ rawArguments: S) throws -> Arguments
    where S.Element == String {
        var parsed = Arguments()
        let arguments = Array(rawArguments)
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--scripted":
                parsed.scripted = true
            case "--help", "-h":
                parsed.showHelp = true
            case "--host":
                index += 1
                parsed.host = try value(at: index, in: arguments, for: argument)
            case "--port":
                index += 1
                let rawPort = try value(at: index, in: arguments, for: argument)
                guard let port = Int(rawPort), (0...65_535).contains(port) else {
                    throw CLIError("--port must be an integer from 0 through 65535")
                }
                parsed.port = port
            case "--model":
                index += 1
                parsed.model = try value(at: index, in: arguments, for: argument)
            default:
                throw CLIError("Unknown argument: \(argument)")
            }
            index += 1
        }
        return parsed
    }

    private static func value(
        at index: Int,
        in arguments: [String],
        for option: String
    ) throws -> String {
        guard arguments.indices.contains(index), !arguments[index].isEmpty else {
            throw CLIError("\(option) requires a value")
        }
        return arguments[index]
    }
}

private struct CLIError: Error, CustomStringConvertible {
    let description: String

    init(_ description: String) {
        self.description = description
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
