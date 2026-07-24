import Foundation

public enum FastMLXServeBackend: Equatable, Sendable {
    case scripted
    case scalar(
        modelDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int)
    case continuousBatchNoSpec(
        modelDirectory: URL,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        maxReservedKVBytes: Int)
}

public enum FastMLXServeArgumentError:
    Error, Equatable, CustomStringConvertible, Sendable
{
    case unknownArgument(String)
    case duplicateOption(String)
    case missingValue(String)
    case invalidPort
    case invalidPositiveInteger(String)
    case missingBackendMode
    case conflictingBackendModes
    case missingRequiredOption(String)
    case invalidModelIdentifier
    case modelPathMustBeAbsolute
    case evidencePathMustBeAbsolute
    case cacheLimitExceedsMemoryLimit
    case reservedKVLimitExceedsMemoryLimit
    case optionRequiresContinuousBatchMode(String)

    public var description: String {
        switch self {
        case .unknownArgument(let argument):
            "Unknown argument: \(argument)"
        case .duplicateOption(let option):
            "\(option) may be specified only once"
        case .missingValue(let option):
            "\(option) requires a value"
        case .invalidPort:
            "--port must be an integer from 0 through 65535"
        case .invalidPositiveInteger(let option):
            "\(option) must be a positive integer"
        case .missingBackendMode:
            "Choose --scripted or provide --model-path with explicit model limits"
        case .conflictingBackendModes:
            "--scripted cannot be combined with loaded-model options"
        case .missingRequiredOption(let option):
            "Loaded model serving requires \(option)"
        case .invalidModelIdentifier:
            "--model must be a non-empty identifier"
        case .modelPathMustBeAbsolute:
            "--model-path must be an absolute local path"
        case .evidencePathMustBeAbsolute:
            "--evidence-path must be an absolute local path"
        case .cacheLimitExceedsMemoryLimit:
            "--cache-limit-bytes cannot exceed --memory-limit-bytes"
        case .reservedKVLimitExceedsMemoryLimit:
            "--max-reserved-kv-bytes cannot exceed --memory-limit-bytes"
        case .optionRequiresContinuousBatchMode(let option):
            "\(option) requires --continuous-batch-no-spec"
        }
    }
}

public struct FastMLXServeArguments: Equatable, Sendable {
    public static let usage = """
        Usage:
          fastmlx-serve --scripted [--host HOST] [--port PORT] [--model MODEL]
          fastmlx-serve [--continuous-batch-no-spec]
            --model-path PATH --model MODEL
            --memory-limit-bytes N --cache-limit-bytes N
            [--max-reserved-kv-bytes N]
            [--host HOST] [--port PORT] [--evidence-path PATH]

          --scripted                  Transport-only backend; no model is loaded.
          --continuous-batch-no-spec  Explicit dense continuous-batch route.
          --model-path PATH           Absolute local source-locked model directory.
          --model MODEL               Exact OpenAI request model identifier.
          --memory-limit-bytes N      Explicit positive MLX memory limit.
          --cache-limit-bytes N       Explicit positive MLX cache limit.
          --max-reserved-kv-bytes N   Required continuous-route aggregate KV cap.
          --host HOST                 Bind host (default: 127.0.0.1).
          --port PORT                 Bind port (default: 8080; 0 is ephemeral).
          --evidence-path PATH        Fresh append-only canonical evidence output.
          --help                      Show this help.

        Set FASTMLX_API_KEY to require Bearer authentication. A non-loopback host
        is rejected unless that environment variable is non-empty.
        """

    public let backend: FastMLXServeBackend?
    public let host: String
    public let port: Int
    public let model: String
    public let evidencePath: URL?
    public let showHelp: Bool

    private init(
        backend: FastMLXServeBackend?,
        host: String,
        port: Int,
        model: String,
        evidencePath: URL?,
        showHelp: Bool
    ) {
        self.backend = backend
        self.host = host
        self.port = port
        self.model = model
        self.evidencePath = evidencePath
        self.showHelp = showHelp
    }

    public static func parse<S: Sequence>(
        _ rawArguments: S
    ) throws -> FastMLXServeArguments where S.Element == String {
        let arguments = Array(rawArguments)
        var seen: Set<String> = []
        var scripted = false
        var continuousBatchNoSpec = false
        var showHelp = false
        var host = "127.0.0.1"
        var port = 8_080
        var model: String?
        var modelPath: String?
        var memoryLimitBytes: Int?
        var cacheLimitBytes: Int?
        var maxReservedKVBytes: Int?
        var evidencePath: URL?

        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            guard Self.supportedOptions.contains(argument) else {
                throw FastMLXServeArgumentError.unknownArgument(argument)
            }
            guard seen.insert(argument).inserted else {
                throw FastMLXServeArgumentError.duplicateOption(argument)
            }

            switch argument {
            case "--scripted":
                scripted = true
            case "--continuous-batch-no-spec":
                continuousBatchNoSpec = true
            case "--help", "-h":
                showHelp = true
            case "--host":
                index += 1
                host = try value(at: index, in: arguments, for: argument)
            case "--port":
                index += 1
                let rawPort = try value(
                    at: index, in: arguments, for: argument)
                guard let parsedPort = Int(rawPort),
                    (0...65_535).contains(parsedPort)
                else {
                    throw FastMLXServeArgumentError.invalidPort
                }
                port = parsedPort
            case "--model":
                index += 1
                model = try value(at: index, in: arguments, for: argument)
            case "--model-path":
                index += 1
                modelPath = try value(
                    at: index, in: arguments, for: argument)
            case "--evidence-path":
                index += 1
                let path = try value(
                    at: index, in: arguments, for: argument)
                guard path.hasPrefix("/") else {
                    throw FastMLXServeArgumentError
                        .evidencePathMustBeAbsolute
                }
                evidencePath = URL(fileURLWithPath: path)
            case "--memory-limit-bytes":
                index += 1
                memoryLimitBytes = try positiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--cache-limit-bytes":
                index += 1
                cacheLimitBytes = try positiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            case "--max-reserved-kv-bytes":
                index += 1
                maxReservedKVBytes = try positiveInteger(
                    try value(at: index, in: arguments, for: argument),
                    option: argument)
            default:
                preconditionFailure("supported option was not handled")
            }
            index += 1
        }

        if showHelp {
            return FastMLXServeArguments(
                backend: nil,
                host: host,
                port: port,
                model: model ?? "fastmlx-scripted",
                evidencePath: evidencePath,
                showHelp: true)
        }

        let hasLoadedModelOptions =
            modelPath != nil || memoryLimitBytes != nil || cacheLimitBytes != nil
                || maxReservedKVBytes != nil
        if scripted, continuousBatchNoSpec || hasLoadedModelOptions {
            throw FastMLXServeArgumentError.conflictingBackendModes
        }
        if scripted {
            let launchedModel = try validatedModel(
                model ?? "fastmlx-scripted")
            return FastMLXServeArguments(
                backend: .scripted,
                host: host,
                port: port,
                model: launchedModel,
                evidencePath: evidencePath,
                showHelp: false)
        }

        guard continuousBatchNoSpec || hasLoadedModelOptions else {
            throw FastMLXServeArgumentError.missingBackendMode
        }
        if !continuousBatchNoSpec, maxReservedKVBytes != nil {
            throw FastMLXServeArgumentError.optionRequiresContinuousBatchMode(
                "--max-reserved-kv-bytes")
        }
        guard let modelPath else {
            throw FastMLXServeArgumentError.missingRequiredOption(
                "--model-path")
        }
        guard modelPath.hasPrefix("/") else {
            throw FastMLXServeArgumentError.modelPathMustBeAbsolute
        }
        guard let model else {
            throw FastMLXServeArgumentError.missingRequiredOption("--model")
        }
        let launchedModel = try validatedModel(model)
        guard let memoryLimitBytes else {
            throw FastMLXServeArgumentError.missingRequiredOption(
                "--memory-limit-bytes")
        }
        guard let cacheLimitBytes else {
            throw FastMLXServeArgumentError.missingRequiredOption(
                "--cache-limit-bytes")
        }
        guard cacheLimitBytes <= memoryLimitBytes else {
            throw FastMLXServeArgumentError.cacheLimitExceedsMemoryLimit
        }
        let resolvedMaxReservedKVBytes: Int?
        if continuousBatchNoSpec {
            guard let maxReservedKVBytes else {
                throw FastMLXServeArgumentError.missingRequiredOption(
                    "--max-reserved-kv-bytes")
            }
            guard maxReservedKVBytes <= memoryLimitBytes else {
                throw FastMLXServeArgumentError
                    .reservedKVLimitExceedsMemoryLimit
            }
            resolvedMaxReservedKVBytes = maxReservedKVBytes
        } else {
            resolvedMaxReservedKVBytes = nil
        }

        return FastMLXServeArguments(
            backend: continuousBatchNoSpec
                ? .continuousBatchNoSpec(
                    modelDirectory: URL(
                        fileURLWithPath: modelPath,
                        isDirectory: true),
                    memoryLimitBytes: memoryLimitBytes,
                    cacheLimitBytes: cacheLimitBytes,
                    maxReservedKVBytes: resolvedMaxReservedKVBytes!)
                : .scalar(
                    modelDirectory: URL(
                        fileURLWithPath: modelPath,
                        isDirectory: true),
                    memoryLimitBytes: memoryLimitBytes,
                    cacheLimitBytes: cacheLimitBytes),
            host: host,
            port: port,
            model: launchedModel,
            evidencePath: evidencePath,
            showHelp: false)
    }

    private static let supportedOptions: Set<String> = [
        "--scripted",
        "--continuous-batch-no-spec",
        "--help",
        "-h",
        "--host",
        "--port",
        "--model",
        "--model-path",
        "--evidence-path",
        "--memory-limit-bytes",
        "--cache-limit-bytes",
        "--max-reserved-kv-bytes",
    ]

    private static func value(
        at index: Int,
        in arguments: [String],
        for option: String
    ) throws -> String {
        guard arguments.indices.contains(index),
            !arguments[index].isEmpty,
            !arguments[index].hasPrefix("--")
        else {
            throw FastMLXServeArgumentError.missingValue(option)
        }
        return arguments[index]
    }

    private static func positiveInteger(
        _ rawValue: String,
        option: String
    ) throws -> Int {
        guard let value = Int(rawValue), value > 0 else {
            throw FastMLXServeArgumentError.invalidPositiveInteger(option)
        }
        return value
    }

    private static func validatedModel(_ rawValue: String) throws -> String {
        guard !rawValue.trimmingCharacters(
            in: .whitespacesAndNewlines
        ).isEmpty else {
            throw FastMLXServeArgumentError.invalidModelIdentifier
        }
        return rawValue
    }
}
