import ScorecardPairControl

func parseQwen38MTPScorecardLiveRunArguments(
    _ arguments: [String]
) throws -> Qwen38MTPScorecardLiveRunArguments {
    let allowed = Set([
        "--target",
        "--drafter",
        "--output",
        "--authority-output",
        "--host-use",
        "--host-use-source",
        "--expected-chip",
        "--memory-limit-bytes",
        "--cache-limit-bytes",
        "--reserved-kv-bytes",
        "--reserved-io-bytes",
        "--reserved-prefetch-bytes",
        "--os-service-reserve-bytes",
    ])
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw Qwen38MTPScorecardLiveAdapterError.unexpectedPositional
        }
        guard allowed.contains(flag) else {
            throw Qwen38MTPScorecardLiveAdapterError.unknownFlag
        }
        guard values[flag] == nil else {
            throw Qwen38MTPScorecardLiveAdapterError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count,
            !arguments[index + 1].hasPrefix("--")
        else {
            throw Qwen38MTPScorecardLiveAdapterError.missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }

    func require(_ flag: String) throws -> String {
        guard let value = values[flag], !value.isEmpty else {
            throw Qwen38MTPScorecardLiveAdapterError.missingFlag(flag)
        }
        return value
    }
    func requireUInt64(_ flag: String) throws -> UInt64 {
        let raw = try require(flag)
        guard let value = UInt64(raw), value > 0, value <= UInt64(Int.max) else {
            throw Qwen38MTPScorecardLiveAdapterError.invalidInteger(flag)
        }
        return value
    }

    let hostUse = try require("--host-use")
    let hostUseSource = try require("--host-use-source")
    let expectedChip = try require("--expected-chip")
    guard hostUse == "dedicated-serving",
        hostUseSource == "operator-assertion"
    else {
        throw Qwen38MTPScorecardLiveAdapterError.invalidMemoryBudget
    }

    return try Qwen38MTPScorecardLiveRunArguments(
        targetPath: require("--target"),
        drafterPath: require("--drafter"),
        outputPath: require("--output"),
        authorityOutputPath: require("--authority-output"),
        hostUse: hostUse,
        hostUseSource: hostUseSource,
        expectedChip: expectedChip,
        memoryBudget: Qwen38MTPScorecardLiveMemoryBudget(
            memoryLimitBytes: requireUInt64("--memory-limit-bytes"),
            cacheLimitBytes: requireUInt64("--cache-limit-bytes"),
            reservedKVBytes: requireUInt64("--reserved-kv-bytes"),
            reservedIOBytes: requireUInt64("--reserved-io-bytes"),
            reservedPrefetchBytes: requireUInt64("--reserved-prefetch-bytes"),
            osServiceReserveBytes: requireUInt64("--os-service-reserve-bytes")))
}

func qwen38MTPScorecardLiveAdapterExternalDiagnostic(_ error: Error) -> String {
    if let error = error as? Qwen38MTPScorecardLiveAdapterError {
        return error.description
    }
    if error is CancellationError {
        return "scorecard production cancelled"
    }
    return "scorecard production failed"
}
