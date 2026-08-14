public struct CLIFlags: Sendable {
    private var values: [String: String] = [:]
    private var missingValueKeys: Set<String> = []

    public init(_ arguments: [String]) {
        var index = 0
        while index < arguments.count {
            guard arguments[index].hasPrefix("--") else {
                index += 1
                continue
            }
            let key = String(arguments[index].dropFirst(2))
            guard !key.isEmpty else {
                index += 1
                continue
            }
            guard index + 1 < arguments.count,
                !arguments[index + 1].hasPrefix("--")
            else {
                missingValueKeys.insert(key)
                index += 1
                continue
            }
            values[key] = arguments[index + 1]
            index += 2
        }
    }

    public func string(_ key: String, default defaultValue: String) -> String {
        values[key] ?? defaultValue
    }

    public func string(_ key: String) -> String? { values[key] }

    public func strictString(
        _ key: String,
        default defaultValue: @autoclosure () -> String
    ) throws -> String {
        try requireValueIfPresent(key)
        return values[key] ?? defaultValue()
    }

    public func int(_ key: String, default defaultValue: Int) -> Int {
        values[key].flatMap(Int.init) ?? defaultValue
    }

    public func strictInt(_ key: String, default defaultValue: Int) throws -> Int {
        try requireValueIfPresent(key)
        guard let raw = values[key] else { return defaultValue }
        guard let value = Int(raw) else {
            throw FlagValueError.invalidInteger(key: key, value: raw)
        }
        return value
    }

    public func optionalStrictInt(_ key: String) throws -> Int? {
        try requireValueIfPresent(key)
        guard let raw = values[key] else { return nil }
        guard let value = Int(raw) else {
            throw FlagValueError.invalidInteger(key: key, value: raw)
        }
        return value
    }

    public func strictBool(_ key: String, default defaultValue: Bool) throws -> Bool {
        try requireValueIfPresent(key)
        guard let raw = values[key] else { return defaultValue }
        switch raw {
        case "true": return true
        case "false": return false
        default: throw FlagValueError.invalidBoolean(key: key, value: raw)
        }
    }

    private func requireValueIfPresent(_ key: String) throws {
        if missingValueKeys.contains(key) {
            throw FlagValueError.missingValue(key: key)
        }
    }
}

public enum FlagValueError: Error, Sendable, Equatable, CustomStringConvertible {
    case missingValue(key: String)
    case invalidInteger(key: String, value: String)
    case invalidBoolean(key: String, value: String)

    public var description: String {
        switch self {
        case .missingValue(let key):
            return "--\(key) requires a value"
        case .invalidInteger(let key, let value):
            return "--\(key) requires an integer; actual=\(value)"
        case .invalidBoolean(let key, let value):
            return "--\(key) requires true or false; actual=\(value)"
        }
    }
}
