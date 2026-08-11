import Foundation

public enum CompressedAttentionProbeCommandError:
    Error, Equatable, Sendable
{
    case unexpectedArgument(String)
    case unknownFlag(String)
    case duplicateFlag(String)
    case missingFlagValue(String)
    case missingRequiredFlag(String)
    case invalidInteger(flag: String, value: String)
    case invalidBoolean(flag: String, value: String)
    case invalidOperation(String)
    case invalidLayout(String)
    case invalidDType(String)
    case invalidMask(String)
    case invalidModelID(String)
    case invalidStopTokenIDs(String)
    case unusedLayoutFlag(String)
    case invalidMemorySettings
}

/// Allocation-free CLI contract for one authenticated compressed-attention probe process.
/// The executable must construct this value and acquire its output lock before touching MLX.
public struct CompressedAttentionProbeCommand: Equatable, Sendable {
    public let modelPath: String
    public let modelID: String
    public let plan: CompressedAttentionProbePlan
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let wiredLimitBytes: Int

    public init(
        arguments: [String],
        harnessGitSHA: String
    ) throws {
        let values = try Self.parse(arguments)
        let modelPath = try Self.required("model", in: values)
        let modelID = try Self.required("model-id", in: values)
        guard !modelPath.isEmpty,
            modelPath == modelPath.trimmingCharacters(
                in: .whitespacesAndNewlines)
        else {
            throw CompressedAttentionProbeCommandError
                .missingRequiredFlag("model")
        }
        guard Self.isModelID(modelID) else {
            throw CompressedAttentionProbeCommandError.invalidModelID(modelID)
        }

        let operationRaw = try Self.required("operation", in: values)
        guard let operation = CompressedAttentionProbeOperation(
            rawValue: operationRaw)
        else {
            throw CompressedAttentionProbeCommandError
                .invalidOperation(operationRaw)
        }
        let layoutRaw = try Self.required("layout", in: values)
        let layout = try Self.layout(raw: layoutRaw, values: values)
        let dtypeRaw = values["dtype"] ?? "float16"
        guard let dtype = CompressedAttentionProbeDType(rawValue: dtypeRaw)
        else {
            throw CompressedAttentionProbeCommandError.invalidDType(dtypeRaw)
        }
        let maskRaw = values["mask"] ?? "causal"
        guard let mask = CompressedAttentionProbeMask(rawValue: maskRaw)
        else {
            throw CompressedAttentionProbeCommandError.invalidMask(maskRaw)
        }
        let stopTokenIDs = try Self.stopTokenIDs(
            values["stop-token-ids"] ?? "")
        let qualificationEvidence = try Self.boolean(
            "qualification-evidence", in: values, default: false)

        let memoryLimitBytes = try Self.integer(
            "memory-limit-bytes", in: values)
        let cacheLimitBytes = try Self.integer(
            "cache-limit-bytes", in: values)
        let wiredLimitBytes = try Self.integer(
            "wired-limit-bytes", in: values)
        guard memoryLimitBytes > 0,
            cacheLimitBytes > 0,
            memoryLimitBytes >= cacheLimitBytes,
            wiredLimitBytes >= 0
        else {
            throw CompressedAttentionProbeCommandError.invalidMemorySettings
        }

        let plan = try CompressedAttentionProbePlan(
            operation: operation,
            contextTokens: try Self.integer("context-tokens", in: values),
            queryTokens: try Self.integer("query-tokens", in: values),
            prefillChunkTokens: try Self.integer(
                "prefill-chunk-tokens", in: values),
            outputTokens: try Self.integer(
                "output-tokens", in: values, default: 16),
            stopTokenIDs: stopTokenIDs,
            batchSize: try Self.integer(
                "batch-size", in: values, default: 1),
            queryHeadCount: try Self.integer("query-heads", in: values),
            kvHeadCount: try Self.integer("kv-heads", in: values),
            headDimension: try Self.integer("head-dimension", in: values),
            dtype: dtype,
            mask: mask,
            layout: layout,
            warmupRuns: try Self.integer(
                "warmup-runs", in: values, default: 1),
            measuredRuns: try Self.integer(
                "measured-runs", in: values, default: 3),
            seed: try Self.integer("seed", in: values, default: 7),
            workloadNonce: try Self.required("workload-nonce", in: values),
            harnessGitSHA: harnessGitSHA,
            qualificationEvidence: qualificationEvidence,
            evidenceOutputPath: try Self.required("evidence", in: values),
            progressOutputPath: try Self.required("progress", in: values))

        self.modelPath = modelPath
        self.modelID = modelID
        self.plan = plan
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.wiredLimitBytes = wiredLimitBytes
    }

    private static let layoutFlags: Set<String> = [
        "key-bits", "value-bits", "key-group-size", "value-group-size",
        "group-size", "sink-tokens", "iterations",
    ]

    private static let allowedFlags: Set<String> = layoutFlags.union([
        "model", "model-id", "operation", "layout", "context-tokens",
        "query-tokens", "prefill-chunk-tokens", "output-tokens",
        "stop-token-ids", "batch-size", "query-heads", "kv-heads",
        "head-dimension", "dtype", "mask", "warmup-runs",
        "measured-runs", "seed", "workload-nonce", "qualification-evidence",
        "evidence", "progress", "memory-limit-bytes", "cache-limit-bytes",
        "wired-limit-bytes",
    ])

    private static func parse(
        _ arguments: [String]
    ) throws -> [String: String] {
        var result: [String: String] = [:]
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            guard argument.hasPrefix("--"), argument.count > 2 else {
                throw CompressedAttentionProbeCommandError
                    .unexpectedArgument(argument)
            }
            let flag = String(argument.dropFirst(2))
            guard allowedFlags.contains(flag) else {
                throw CompressedAttentionProbeCommandError.unknownFlag(flag)
            }
            guard result[flag] == nil else {
                throw CompressedAttentionProbeCommandError.duplicateFlag(flag)
            }
            guard index + 1 < arguments.count,
                !arguments[index + 1].hasPrefix("--")
            else {
                throw CompressedAttentionProbeCommandError
                    .missingFlagValue(flag)
            }
            result[flag] = arguments[index + 1]
            index += 2
        }
        return result
    }

    private static func required(
        _ flag: String,
        in values: [String: String]
    ) throws -> String {
        guard let value = values[flag], !value.isEmpty else {
            throw CompressedAttentionProbeCommandError
                .missingRequiredFlag(flag)
        }
        return value
    }

    private static func integer(
        _ flag: String,
        in values: [String: String],
        default defaultValue: Int? = nil
    ) throws -> Int {
        guard let raw = values[flag] else {
            guard let defaultValue else {
                throw CompressedAttentionProbeCommandError
                    .missingRequiredFlag(flag)
            }
            return defaultValue
        }
        guard let value = Int(raw) else {
            throw CompressedAttentionProbeCommandError
                .invalidInteger(flag: flag, value: raw)
        }
        return value
    }

    private static func boolean(
        _ flag: String,
        in values: [String: String],
        default defaultValue: Bool
    ) throws -> Bool {
        guard let raw = values[flag] else { return defaultValue }
        switch raw {
        case "true": return true
        case "false": return false
        default:
            throw CompressedAttentionProbeCommandError
                .invalidBoolean(flag: flag, value: raw)
        }
    }

    private static func layout(
        raw: String,
        values: [String: String]
    ) throws -> CompressedAttentionProbeLayout {
        let used: Set<String>
        let layout: CompressedAttentionProbeLayout
        switch raw {
        case "fp16":
            used = []
            layout = .fp16
        case "affine":
            used = [
                "key-bits", "value-bits", "key-group-size",
                "value-group-size",
            ]
            layout = try .affine(
                keyBits: integer("key-bits", in: values),
                valueBits: integer("value-bits", in: values),
                keyGroupSize: integer("key-group-size", in: values),
                valueGroupSize: integer("value-group-size", in: values))
        case "kvarn":
            used = [
                "key-bits", "value-bits", "group-size", "sink-tokens",
                "iterations",
            ]
            layout = try .kvarn(
                keyBits: integer("key-bits", in: values),
                valueBits: integer("value-bits", in: values),
                groupSize: integer("group-size", in: values),
                sinkTokens: integer("sink-tokens", in: values),
                iterations: integer("iterations", in: values))
        default:
            throw CompressedAttentionProbeCommandError.invalidLayout(raw)
        }
        if let unused = layoutFlags.subtracting(used)
            .intersection(values.keys).sorted().first
        {
            throw CompressedAttentionProbeCommandError
                .unusedLayoutFlag(unused)
        }
        return layout
    }

    private static func stopTokenIDs(_ raw: String) throws -> [Int] {
        guard !raw.isEmpty else { return [] }
        let components = raw.split(
            separator: ",", omittingEmptySubsequences: false)
        let values = components.compactMap { Int($0) }
        guard values.count == components.count,
            values.allSatisfy({ $0 >= 0 }),
            values == values.sorted(),
            Set(values).count == values.count
        else {
            throw CompressedAttentionProbeCommandError
                .invalidStopTokenIDs(raw)
        }
        return values
    }

    private static func isModelID(_ value: String) -> Bool {
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
}
