import CryptoKit
import Foundation

public struct Qwen38HeavyHostTrustInventory: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var hostLabel: String
    public var expectedFingerprintSHA256: String

    public init(
        schemaVersion: Int,
        hostLabel: String,
        expectedFingerprintSHA256: String
    ) {
        self.schemaVersion = schemaVersion
        self.hostLabel = hostLabel
        self.expectedFingerprintSHA256 = expectedFingerprintSHA256
    }
}

public struct Qwen38HeavyHostObservedIdentity: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var hostLabel: String
    public var observedFingerprintSHA256: String

    public init(
        schemaVersion: Int,
        hostLabel: String,
        observedFingerprintSHA256: String
    ) {
        self.schemaVersion = schemaVersion
        self.hostLabel = hostLabel
        self.observedFingerprintSHA256 = observedFingerprintSHA256
    }
}

/// Caller-issued trust root for one immutable inventory record. The production default is nil;
/// matching user-supplied files alone can never promote a host identity.
public struct Qwen38HeavyHostTrustInventoryAuthority: Codable, Equatable, Sendable {
    public var inventory: Qwen38HeavyHostTrustInventory
    public var inventoryRecordSHA256: String

    public init(
        inventory: Qwen38HeavyHostTrustInventory,
        inventoryRecordSHA256: String
    ) {
        self.inventory = inventory
        self.inventoryRecordSHA256 = inventoryRecordSHA256
    }
}

public struct Qwen38HeavyHostTrustReadinessRecord: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var status: String
    public var hostLabel: String
    public var expectedFingerprintSHA256: String
    public var observedFingerprintSHA256: String
    public var inventoryRecordSHA256: String
    public var sourceRevision: String
    public var noRemoteActionPerformed: Bool
    public var issuesExactnessAuthority: Bool
    public var issuesPerformanceAuthority: Bool

    public init(
        schemaVersion: Int,
        status: String,
        hostLabel: String,
        expectedFingerprintSHA256: String,
        observedFingerprintSHA256: String,
        inventoryRecordSHA256: String,
        sourceRevision: String,
        noRemoteActionPerformed: Bool,
        issuesExactnessAuthority: Bool,
        issuesPerformanceAuthority: Bool
    ) {
        self.schemaVersion = schemaVersion
        self.status = status
        self.hostLabel = hostLabel
        self.expectedFingerprintSHA256 = expectedFingerprintSHA256
        self.observedFingerprintSHA256 = observedFingerprintSHA256
        self.inventoryRecordSHA256 = inventoryRecordSHA256
        self.sourceRevision = sourceRevision
        self.noRemoteActionPerformed = noRemoteActionPerformed
        self.issuesExactnessAuthority = issuesExactnessAuthority
        self.issuesPerformanceAuthority = issuesPerformanceAuthority
    }
}

public enum Qwen38HeavyHostTrustReadinessGateError:
    Error, Equatable, CustomStringConvertible, Sendable
{
    case inventoryAuthorityNotPromoted
    case malformedInventory
    case nonCanonicalInventory
    case inventoryAuthorityMismatch
    case invalidInventoryIdentity
    case malformedObservedIdentity
    case nonCanonicalObservedIdentity
    case invalidObservedIdentity
    case hostLabelMismatch
    case fingerprintMismatch
    case invalidSourceRevision

    public var description: String {
        switch self {
        case .inventoryAuthorityNotPromoted:
            return "trusted inventory authority is not promoted"
        case .malformedInventory:
            return "malformed inventory"
        case .nonCanonicalInventory:
            return "non-canonical inventory"
        case .inventoryAuthorityMismatch:
            return "inventory authority mismatch"
        case .invalidInventoryIdentity:
            return "invalid inventory identity"
        case .malformedObservedIdentity:
            return "malformed observed identity"
        case .nonCanonicalObservedIdentity:
            return "non-canonical observed identity"
        case .invalidObservedIdentity:
            return "invalid observed identity"
        case .hostLabelMismatch:
            return "host label mismatch"
        case .fingerprintMismatch:
            return "host fingerprint mismatch"
        case .invalidSourceRevision:
            return "invalid source revision"
        }
    }
}

public enum Qwen38HeavyHostTrustReadinessGate {
    public static let schemaVersion = 1

    /// Deliberately nil until an inventory fingerprint has been authenticated outside this tool.
    /// Observed network keys and caller-authored fixture files are not trust anchors.
    public static let requiredInventoryAuthority:
        Qwen38HeavyHostTrustInventoryAuthority? = nil

    public static func validate(
        inventoryData: Data,
        observedIdentityData: Data,
        sourceRevision: String
    ) throws -> Qwen38HeavyHostTrustReadinessRecord {
        guard let requiredInventoryAuthority else {
            throw Qwen38HeavyHostTrustReadinessGateError.inventoryAuthorityNotPromoted
        }
        return try validate(
            inventoryData: inventoryData,
            observedIdentityData: observedIdentityData,
            sourceRevision: sourceRevision,
            trustedInventoryAuthority: requiredInventoryAuthority)
    }

    public static func validate(
        inventoryData: Data,
        observedIdentityData: Data,
        sourceRevision: String,
        trustedInventoryAuthority: Qwen38HeavyHostTrustInventoryAuthority
    ) throws -> Qwen38HeavyHostTrustReadinessRecord {
        let inventory = try decodeInventory(inventoryData)
        try validateInventory(inventory)

        let canonicalInventory = canonicalData(inventory)
        guard inventoryData == canonicalInventory else {
            throw Qwen38HeavyHostTrustReadinessGateError.nonCanonicalInventory
        }
        let inventoryRecordSHA256 = sha256Hex(canonicalInventory)
        guard trustedInventoryAuthority.inventory == inventory,
            trustedInventoryAuthority.inventoryRecordSHA256 == inventoryRecordSHA256,
            isLowerHex(trustedInventoryAuthority.inventoryRecordSHA256, count: 64)
        else {
            throw Qwen38HeavyHostTrustReadinessGateError.inventoryAuthorityMismatch
        }

        let observed = try decodeObservedIdentity(observedIdentityData)
        try validateObservedIdentity(observed)
        guard observedIdentityData == canonicalData(observed) else {
            throw Qwen38HeavyHostTrustReadinessGateError.nonCanonicalObservedIdentity
        }
        guard observed.hostLabel == inventory.hostLabel else {
            throw Qwen38HeavyHostTrustReadinessGateError.hostLabelMismatch
        }
        guard observed.observedFingerprintSHA256
            == inventory.expectedFingerprintSHA256
        else {
            throw Qwen38HeavyHostTrustReadinessGateError.fingerprintMismatch
        }
        guard isLowerHex(sourceRevision, count: 40) else {
            throw Qwen38HeavyHostTrustReadinessGateError.invalidSourceRevision
        }

        return Qwen38HeavyHostTrustReadinessRecord(
            schemaVersion: schemaVersion,
            status: "ready",
            hostLabel: inventory.hostLabel,
            expectedFingerprintSHA256: inventory.expectedFingerprintSHA256,
            observedFingerprintSHA256: observed.observedFingerprintSHA256,
            inventoryRecordSHA256: inventoryRecordSHA256,
            sourceRevision: sourceRevision,
            noRemoteActionPerformed: true,
            issuesExactnessAuthority: false,
            issuesPerformanceAuthority: false)
    }

    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func decodeInventory(
        _ data: Data
    ) throws -> Qwen38HeavyHostTrustInventory {
        do {
            return try JSONDecoder().decode(
                Qwen38HeavyHostTrustInventory.self,
                from: data)
        } catch {
            throw Qwen38HeavyHostTrustReadinessGateError.malformedInventory
        }
    }

    private static func decodeObservedIdentity(
        _ data: Data
    ) throws -> Qwen38HeavyHostObservedIdentity {
        do {
            return try JSONDecoder().decode(
                Qwen38HeavyHostObservedIdentity.self,
                from: data)
        } catch {
            throw Qwen38HeavyHostTrustReadinessGateError.malformedObservedIdentity
        }
    }

    private static func validateInventory(
        _ inventory: Qwen38HeavyHostTrustInventory
    ) throws {
        guard inventory.schemaVersion == schemaVersion,
            isHostLabel(inventory.hostLabel),
            isLowerHex(inventory.expectedFingerprintSHA256, count: 64)
        else {
            throw Qwen38HeavyHostTrustReadinessGateError.invalidInventoryIdentity
        }
    }

    private static func validateObservedIdentity(
        _ observed: Qwen38HeavyHostObservedIdentity
    ) throws {
        guard observed.schemaVersion == schemaVersion,
            isHostLabel(observed.hostLabel),
            isLowerHex(observed.observedFingerprintSHA256, count: 64)
        else {
            throw Qwen38HeavyHostTrustReadinessGateError.invalidObservedIdentity
        }
    }

    private static func canonicalData<T: Encodable>(_ value: T) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try! encoder.encode(value)
    }

    private static func isHostLabel(_ value: String) -> Bool {
        let bytes = Array(value.utf8)
        guard (3 ... 64).contains(bytes.count),
            bytes.first != 45,
            bytes.last != 45
        else { return false }
        return bytes.allSatisfy {
            (97 ... 122).contains($0) || (48 ... 57).contains($0) || $0 == 45
        }
    }

    private static func isLowerHex(_ value: String, count: Int) -> Bool {
        value.utf8.count == count && value.utf8.allSatisfy {
            (48 ... 57).contains($0) || (97 ... 102).contains($0)
        }
    }
}
