import Foundation
import HarnessCore

struct Qwen38MTPPerformanceScorecardValidationArguments: Equatable, Sendable {
    let evidencePath: String
    let authorityPath: String
}

enum Qwen38MTPPerformanceScorecardCLIError: Error, Equatable, CustomStringConvertible, Sendable {
    case missingFlag(String)
    case duplicateFlag(String)
    case unknownFlag
    case missingValue(String)
    case unexpectedPositional
    case fileReadFailed(String)
    case malformedAuthority
    case invalidVerdictCount(Int)

    var description: String {
        switch self {
        case .missingFlag(let flag): return "missing required \(flag)"
        case .duplicateFlag(let flag): return "duplicate \(flag)"
        case .unknownFlag: return "unknown flag"
        case .missingValue(let flag): return "\(flag) requires a value"
        case .unexpectedPositional: return "unexpected positional argument"
        case .fileReadFailed(let role): return "failed to read \(role) file"
        case .malformedAuthority: return "malformed authority JSON"
        case .invalidVerdictCount(let count): return "invalid verdict count \(count)"
        }
    }
}

func parseQwen38MTPPerformanceScorecardValidationArguments(
    _ arguments: [String]
) throws -> Qwen38MTPPerformanceScorecardValidationArguments {
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw Qwen38MTPPerformanceScorecardCLIError.unexpectedPositional
        }
        guard flag == "--evidence" || flag == "--authority" else {
            throw Qwen38MTPPerformanceScorecardCLIError.unknownFlag
        }
        guard values[flag] == nil else {
            throw Qwen38MTPPerformanceScorecardCLIError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count,
            !arguments[index + 1].hasPrefix("--")
        else {
            throw Qwen38MTPPerformanceScorecardCLIError.missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }
    guard let evidencePath = values["--evidence"] else {
        throw Qwen38MTPPerformanceScorecardCLIError.missingFlag("--evidence")
    }
    guard let authorityPath = values["--authority"] else {
        throw Qwen38MTPPerformanceScorecardCLIError.missingFlag("--authority")
    }
    return Qwen38MTPPerformanceScorecardValidationArguments(
        evidencePath: evidencePath,
        authorityPath: authorityPath)
}

func qwen38MTPPerformanceScorecardExternalDiagnostic(_ error: Error) -> String {
    if let cliError = error as? Qwen38MTPPerformanceScorecardCLIError {
        return cliError.description
    }
    return "evidence or authority validation failed"
}

func validateQwen38MTPPerformanceScorecard(
    arguments: [String],
    readFile: (String) throws -> Data = {
        try Data(contentsOf: URL(fileURLWithPath: $0))
    }
) throws -> String {
    let parsed = try parseQwen38MTPPerformanceScorecardValidationArguments(arguments)
    let evidenceData: Data
    let authorityData: Data
    do {
        evidenceData = try readFile(parsed.evidencePath)
    } catch let error as Qwen38MTPPerformanceScorecardCLIError {
        throw error
    } catch {
        throw Qwen38MTPPerformanceScorecardCLIError.fileReadFailed("--evidence")
    }
    do {
        authorityData = try readFile(parsed.authorityPath)
    } catch let error as Qwen38MTPPerformanceScorecardCLIError {
        throw error
    } catch {
        throw Qwen38MTPPerformanceScorecardCLIError.fileReadFailed("--authority")
    }
    let verdict = try validateQwen38MTPPerformanceScorecardData(
        evidenceData: evidenceData,
        authorityData: authorityData)
    return "qwen38-mtp-performance-scorecard: VALID qualified=\(verdict.qualified)"
}

func validateQwen38MTPPerformanceScorecardData(
    evidenceData: Data,
    authorityData: Data
) throws -> Qwen38MTPPerformanceScorecardVerdict {
    guard (try? JSONSerialization.jsonObject(with: authorityData)) is [String: Any] else {
        throw Qwen38MTPPerformanceScorecardCLIError.malformedAuthority
    }
    let authority: Qwen38MTPPerformanceScorecardAuthorityBundle
    do {
        authority = try JSONDecoder().decode(
            Qwen38MTPPerformanceScorecardAuthorityBundle.self,
            from: authorityData)
    } catch {
        throw Qwen38MTPPerformanceScorecardCLIError.malformedAuthority
    }
    let verdicts = try Qwen38MTPPerformanceScorecardGate.validateJSONL(
        evidenceData,
        authority: authority)
    guard verdicts.count == 1, let verdict = verdicts.first else {
        throw Qwen38MTPPerformanceScorecardCLIError.invalidVerdictCount(verdicts.count)
    }
    return verdict
}
