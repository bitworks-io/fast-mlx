import Foundation
import HarnessCore

struct Qwen38HeavyHostTrustReadinessArguments: Equatable, Sendable {
    let inventoryPath: String
    let observedIdentityPath: String
}

enum Qwen38HeavyHostTrustReadinessDisposition: Equatable, Sendable {
    case blocked
    case invalid

    var label: String {
        switch self {
        case .blocked: return "BLOCKED"
        case .invalid: return "INVALID"
        }
    }

    var exitCode: Int32 {
        switch self {
        case .blocked: return 3
        case .invalid: return 2
        }
    }
}

enum Qwen38HeavyHostTrustReadinessCLIError:
    Error, Equatable, CustomStringConvertible, Sendable
{
    case missingFlag(String)
    case duplicateFlag(String)
    case unknownFlag
    case missingValue(String)
    case unexpectedPositional
    case inventoryUnavailable
    case fileReadFailed(String)
    case sourceIdentityUnavailable

    var description: String {
        switch self {
        case .missingFlag(let flag): return "missing required \(flag)"
        case .duplicateFlag(let flag): return "duplicate \(flag)"
        case .unknownFlag: return "unknown flag"
        case .missingValue(let flag): return "\(flag) requires a value"
        case .unexpectedPositional: return "unexpected positional argument"
        case .inventoryUnavailable: return "trusted inventory file unavailable"
        case .fileReadFailed(let role): return "failed to read \(role) file"
        case .sourceIdentityUnavailable: return "clean source identity unavailable"
        }
    }
}

func parseQwen38HeavyHostTrustReadinessArguments(
    _ arguments: [String]
) throws -> Qwen38HeavyHostTrustReadinessArguments {
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw Qwen38HeavyHostTrustReadinessCLIError.unexpectedPositional
        }
        guard flag == "--inventory" || flag == "--observed" else {
            throw Qwen38HeavyHostTrustReadinessCLIError.unknownFlag
        }
        guard values[flag] == nil else {
            throw Qwen38HeavyHostTrustReadinessCLIError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count,
            !arguments[index + 1].hasPrefix("--")
        else {
            throw Qwen38HeavyHostTrustReadinessCLIError.missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }
    guard let inventoryPath = values["--inventory"] else {
        throw Qwen38HeavyHostTrustReadinessCLIError.missingFlag("--inventory")
    }
    guard let observedIdentityPath = values["--observed"] else {
        throw Qwen38HeavyHostTrustReadinessCLIError.missingFlag("--observed")
    }
    return Qwen38HeavyHostTrustReadinessArguments(
        inventoryPath: inventoryPath,
        observedIdentityPath: observedIdentityPath)
}

func qwen38HeavyHostTrustReadinessDisposition(
    _ error: Error
) -> Qwen38HeavyHostTrustReadinessDisposition {
    if let cliError = error as? Qwen38HeavyHostTrustReadinessCLIError,
        cliError == .inventoryUnavailable
    {
        return .blocked
    }
    if let gateError = error as? Qwen38HeavyHostTrustReadinessGateError,
        gateError == .inventoryAuthorityNotPromoted
    {
        return .blocked
    }
    return .invalid
}

func qwen38HeavyHostTrustReadinessExternalDiagnostic(_ error: Error) -> String {
    if let cliError = error as? Qwen38HeavyHostTrustReadinessCLIError {
        return cliError.description
    }
    if let gateError = error as? Qwen38HeavyHostTrustReadinessGateError,
        gateError == .inventoryAuthorityNotPromoted
    {
        return gateError.description
    }
    return "trust readiness validation failed"
}

func validateQwen38HeavyHostTrustReadiness(
    arguments: [String],
    trustedInventoryAuthority: Qwen38HeavyHostTrustInventoryAuthority? =
        Qwen38HeavyHostTrustReadinessGate.requiredInventoryAuthority,
    readFile: (String) throws -> Data = {
        try Data(contentsOf: URL(fileURLWithPath: $0))
    },
    sourceRevision: () throws -> String = {
        try ProvenanceCLI.qualificationHarnessGitSHA()
    }
) throws -> String {
    let parsed = try parseQwen38HeavyHostTrustReadinessArguments(arguments)
    guard let trustedInventoryAuthority else {
        throw Qwen38HeavyHostTrustReadinessGateError.inventoryAuthorityNotPromoted
    }

    let inventoryData: Data
    do {
        inventoryData = try readFile(parsed.inventoryPath)
    } catch {
        throw Qwen38HeavyHostTrustReadinessCLIError.inventoryUnavailable
    }

    let observedIdentityData: Data
    do {
        observedIdentityData = try readFile(parsed.observedIdentityPath)
    } catch let error as Qwen38HeavyHostTrustReadinessCLIError {
        throw error
    } catch {
        throw Qwen38HeavyHostTrustReadinessCLIError.fileReadFailed("--observed")
    }

    let revision: String
    do {
        revision = try sourceRevision()
    } catch {
        throw Qwen38HeavyHostTrustReadinessCLIError.sourceIdentityUnavailable
    }
    let record = try Qwen38HeavyHostTrustReadinessGate.validate(
        inventoryData: inventoryData,
        observedIdentityData: observedIdentityData,
        sourceRevision: revision,
        trustedInventoryAuthority: trustedInventoryAuthority)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    guard let output = String(data: try encoder.encode(record), encoding: .utf8) else {
        throw Qwen38HeavyHostTrustReadinessCLIError.sourceIdentityUnavailable
    }
    return output
}
