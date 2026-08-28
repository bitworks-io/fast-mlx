import Foundation
import HarnessCore

struct Qwen38MTPLiveExactnessValidationArguments: Equatable, Sendable {
    let evidencePath: String
}

enum Qwen38MTPLiveExactnessCLIError: Error, Equatable, CustomStringConvertible, Sendable {
    case missingFlag(String)
    case duplicateFlag(String)
    case unknownFlag
    case missingValue(String)
    case unexpectedPositional
    case fileReadFailed(String)

    var description: String {
        switch self {
        case .missingFlag(let flag): return "missing required \(flag)"
        case .duplicateFlag(let flag): return "duplicate \(flag)"
        case .unknownFlag: return "unknown flag"
        case .missingValue(let flag): return "\(flag) requires a value"
        case .unexpectedPositional: return "unexpected positional argument"
        case .fileReadFailed(let role): return "failed to read \(role) file"
        }
    }
}

func parseQwen38MTPLiveExactnessValidationArguments(
    _ arguments: [String]
) throws -> Qwen38MTPLiveExactnessValidationArguments {
    var evidencePath: String?
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw Qwen38MTPLiveExactnessCLIError.unexpectedPositional
        }
        guard flag == "--evidence" else {
            throw Qwen38MTPLiveExactnessCLIError.unknownFlag
        }
        guard evidencePath == nil else {
            throw Qwen38MTPLiveExactnessCLIError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count,
            !arguments[index + 1].hasPrefix("--")
        else {
            throw Qwen38MTPLiveExactnessCLIError.missingValue(flag)
        }
        evidencePath = arguments[index + 1]
        index += 2
    }
    guard let evidencePath else {
        throw Qwen38MTPLiveExactnessCLIError.missingFlag("--evidence")
    }
    return Qwen38MTPLiveExactnessValidationArguments(evidencePath: evidencePath)
}

func qwen38MTPLiveExactnessExternalDiagnostic(_ error: Error) -> String {
    if let cliError = error as? Qwen38MTPLiveExactnessCLIError {
        return cliError.description
    }
    return "live exactness evidence validation failed"
}

func validateQwen38MTPLiveExactness(
    arguments: [String],
    readFile: (String) throws -> Data = {
        try Data(contentsOf: URL(fileURLWithPath: $0))
    }
) throws -> String {
    let parsed = try parseQwen38MTPLiveExactnessValidationArguments(arguments)
    let evidenceData: Data
    do {
        evidenceData = try readFile(parsed.evidencePath)
    } catch let error as Qwen38MTPLiveExactnessCLIError {
        throw error
    } catch {
        throw Qwen38MTPLiveExactnessCLIError.fileReadFailed("--evidence")
    }
    let proof = try Qwen38MTPLiveExactnessGate.validateJSONL(evidenceData)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    guard let text = String(data: try encoder.encode(proof), encoding: .utf8) else {
        throw Qwen38MTPLiveExactnessCLIError.fileReadFailed("--evidence")
    }
    return text
}
