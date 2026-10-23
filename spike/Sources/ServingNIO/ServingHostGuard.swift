import Foundation
import ServingCore

/// DNS-rebinding guard for a server that has NO bearer token configured (cycle 291).
///
/// A tokenless server only ever binds loopback, but a web page the operator visits can rebind its
/// own hostname to 127.0.0.1 and then talk to the server same-origin. Such a request still carries
/// the attacker's hostname in `Host`, so a tokenless server refuses any `Host` that is not a
/// loopback name (or an operator-supplied `--allow-host` name).
///
/// This is deliberately NOT built on `ServingHTTPBind.isLoopback`, whose `Int()` octet parse
/// accepts `127.000.0.1` and `+1`. Everything here is a strict, pure function of the header values,
/// the request target and the allow list so it is unit-testable without a socket.
public struct ServingHostGuard: Equatable, Sendable {
    public enum Decision: Equatable, Sendable {
        case allow
        /// Missing, duplicated or malformed `Host` -> 400 `invalid_host_header`.
        case rejectInvalidHost
        /// Well-formed but not an allowed name, or a non-origin-form request target -> 421
        /// `host_not_allowed`.
        case rejectNotAllowed
    }

    public static let invalidHostCode = "invalid_host_header"
    public static let notAllowedCode = "host_not_allowed"

    /// Fixed messages: the `Host` value is never echoed back to the (possibly hostile) client.
    public static let invalidHostMessage =
        "Exactly one well-formed Host header is required"
    public static let notAllowedMessage =
        "The Host header is not allowed by this server"

    /// True for the two errors this guard produces (they alone force `Connection: close`).
    static func isGuardRejection(_ error: OpenAIServingError) -> Bool {
        guard case .invalidRequestWithCode(_, _, let code) = error else {
            return false
        }
        return code == invalidHostCode || code == notAllowedCode
    }

    /// Lowercased operator-supplied names (`--allow-host`), already validated by the argument parser.
    public let allowedHosts: [String]

    public init(allowedHosts: [String] = []) {
        self.allowedHosts = allowedHosts.map { $0.lowercased() }
    }

    /// `hostHeaders` is every `Host` header value in arrival order; `requestTarget` is the request
    /// line target exactly as received.
    public func evaluate(hostHeaders: [String], requestTarget: String) -> Decision {
        guard hostHeaders.count == 1 else {
            return .rejectInvalidHost
        }
        let parsed: ParsedHost
        switch Self.parse(hostHeaders[0]) {
        case .malformed:
            return .rejectInvalidHost
        case .host(let value):
            parsed = value
        }
        // Absolute-form (`GET http://evil/ ...`) and asterisk-form (`OPTIONS *`) targets are never
        // served by this API; refuse them rather than reason about which authority wins.
        guard requestTarget.hasPrefix("/") else {
            return .rejectNotAllowed
        }
        if Self.isLoopbackName(parsed.name) || allowedHosts.contains(parsed.name) {
            return .allow
        }
        return .rejectNotAllowed
    }

    // MARK: - Parsing

    struct ParsedHost: Equatable {
        /// Lowercased, port removed, one trailing dot removed. A bracketed IPv6 literal keeps its
        /// brackets so it can never collide with a registered name.
        let name: String
    }

    enum ParseResult: Equatable {
        case malformed
        case host(ParsedHost)
    }

    static func parse(_ raw: String) -> ParseResult {
        let trimmed = raw.trimmingCharacters(in: CharacterSet(charactersIn: " \t"))
        guard !trimmed.isEmpty else {
            return .malformed
        }
        let value = trimmed.lowercased()
        // Reject anything that cannot be part of a host[:port] authority: whitespace/control
        // characters, userinfo, path/query/fragment delimiters, backslashes.
        for scalar in value.unicodeScalars {
            if scalar.value <= 0x20 || scalar.value == 0x7F {
                return .malformed
            }
            switch scalar {
            case "/", "\\", "@", "?", "#":
                return .malformed
            default:
                break
            }
        }

        var hostPart: Substring
        if value.hasPrefix("[") {
            guard let close = value.firstIndex(of: "]") else {
                return .malformed
            }
            let inner = value[value.index(after: value.startIndex)..<close]
            let trailer = value[value.index(after: close)...]
            guard !inner.isEmpty else {
                return .malformed
            }
            if trailer.isEmpty {
                // fallthrough with no port
            } else if trailer.hasPrefix(":") {
                guard isPort(trailer.dropFirst()) else {
                    return .malformed
                }
            } else {
                // `[::1].evil`: a lookalike, not a loopback literal. Well-formed enough to be
                // refused as "not allowed" (421), never accepted.
                return .host(ParsedHost(name: "[\(inner)]\(trailer)"))
            }
            hostPart = value[value.startIndex...close]
            return .host(ParsedHost(name: String(hostPart)))
        }

        let pieces = value.split(separator: ":", maxSplits: .max, omittingEmptySubsequences: false)
        switch pieces.count {
        case 1:
            hostPart = pieces[0]
        case 2:
            guard isPort(pieces[1]) else {
                return .malformed
            }
            hostPart = pieces[0]
        default:
            // A bare IPv6 address (several colons) is malformed: it must be bracketed.
            return .malformed
        }
        if hostPart.hasSuffix(".") {
            hostPart = hostPart.dropLast()
        }
        guard !hostPart.isEmpty else {
            return .malformed
        }
        return .host(ParsedHost(name: String(hostPart)))
    }

    private static func isPort(_ digits: Substring) -> Bool {
        digits.count <= 5 && digits.unicodeScalars.allSatisfy { $0.value >= 0x30 && $0.value <= 0x39 }
    }

    // MARK: - Loopback names

    /// `localhost`, a canonical dotted-quad IPv4 whose first octet is 127, or `[::1]`.
    /// Not `*.localhost`, not `127.1`, not `2130706433`, not `0x7f.1`, not `127.000.000.001`.
    static func isLoopbackName(_ name: String) -> Bool {
        if name == "localhost" || name == "[::1]" {
            return true
        }
        guard let octets = canonicalIPv4Octets(name) else {
            return false
        }
        return octets[0] == 127
    }

    /// Four decimal octets, each 0-255 with no sign, no leading zeros and 1-3 ASCII digits.
    static func canonicalIPv4Octets(_ text: String) -> [Int]? {
        let parts = text.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else {
            return nil
        }
        var octets: [Int] = []
        for part in parts {
            guard (1...3).contains(part.count),
                part.unicodeScalars.allSatisfy({ $0.value >= 0x30 && $0.value <= 0x39 })
            else {
                return nil
            }
            if part.count > 1, part.first == "0" {
                return nil
            }
            guard let octet = Int(part), octet <= 255 else {
                return nil
            }
            octets.append(octet)
        }
        return octets
    }
}
