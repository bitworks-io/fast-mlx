import Foundation

struct SyntheticRuntimeClosureInstallName:
    Equatable,
    Sendable
{
    let bytes: UInt64
    let base64URL: String
}

enum SyntheticRuntimeClosureInstallNameFailure:
    Error,
    Equatable,
    Sendable
{
    case bytes
    case base64URL
    case syntax
}

enum SyntheticRuntimeClosureInstallNameVerifier {
    static let maximumInstallNameBytes = 4_096
    static let maximumEncodedInstallNameBytes =
        (maximumInstallNameBytes * 4 + 2) / 3

    static func validate(
        _ value: SyntheticRuntimeClosureInstallName
    ) throws -> Data {
        guard hasBoundedEncodedInstallNameLength(
            value.base64URL
        ) else {
            throw SyntheticRuntimeClosureInstallNameFailure.bytes
        }
        guard
            let decoded = decodeCanonicalBase64URL(
                value.base64URL
            )
        else {
            throw SyntheticRuntimeClosureInstallNameFailure
                .base64URL
        }
        guard
            UInt64(decoded.count) == value.bytes,
            (2...maximumInstallNameBytes).contains(
                decoded.count
            )
        else {
            throw SyntheticRuntimeClosureInstallNameFailure.bytes
        }
        guard isCanonicalInstallName(decoded) else {
            throw SyntheticRuntimeClosureInstallNameFailure.syntax
        }
        return decoded
    }
}

private extension SyntheticRuntimeClosureInstallNameVerifier {
    static func hasBoundedEncodedInstallNameLength(
        _ encoded: String
    ) -> Bool {
        var count = 0
        for _ in encoded.utf8 {
            guard count < maximumEncodedInstallNameBytes else {
                return false
            }
            count += 1
        }
        return true
    }

    static func decodeCanonicalBase64URL(
        _ encoded: String
    ) -> Data? {
        let bytes = Array(encoded.utf8)
        guard
            bytes.allSatisfy({
                (0x41...0x5a).contains($0) ||
                    (0x61...0x7a).contains($0) ||
                    (0x30...0x39).contains($0) ||
                    $0 == 0x2d ||
                    $0 == 0x5f
            }),
            bytes.count % 4 != 1
        else {
            return nil
        }

        let standard = encoded
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        let padded = standard + String(
            repeating: "=",
            count: (4 - standard.utf8.count % 4) % 4
        )
        guard let decoded = Data(base64Encoded: padded) else {
            return nil
        }
        let canonical = decoded.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
        guard canonical == encoded else {
            return nil
        }
        return decoded
    }

    static func isCanonicalInstallName(_ value: Data) -> Bool {
        let bytes = Array(value)
        guard bytes.first == 0x2f else {
            return false
        }
        guard bytes.dropFirst().allSatisfy({
            (0x21...0x7e).contains($0) &&
                $0 != 0x40 &&
                $0 != 0x5c
        }) else {
            return false
        }

        var componentStart = 1
        for index in 1...bytes.count {
            guard index == bytes.count || bytes[index] == 0x2f else {
                continue
            }
            let component = bytes[componentStart..<index]
            guard
                !component.isEmpty,
                component != [0x2e],
                component != [0x2e, 0x2e]
            else {
                return false
            }
            componentStart = index + 1
        }
        return true
    }
}
