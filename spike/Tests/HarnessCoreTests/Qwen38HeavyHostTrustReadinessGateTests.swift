import Foundation
import HarnessCore
import XCTest

final class Qwen38HeavyHostTrustReadinessGateTests: XCTestCase {
    private typealias Gate = Qwen38HeavyHostTrustReadinessGate
    private typealias GateError = Qwen38HeavyHostTrustReadinessGateError

    func testDefaultValidationBlocksWithoutPromotedInventoryAuthority() throws {
        XCTAssertNil(Gate.requiredInventoryAuthority)

        XCTAssertThrowsError(
            try Gate.validate(
                inventoryData: canonicalData(inventory()),
                observedIdentityData: canonicalData(observed()),
                sourceRevision: sourceRevision)
        ) { error in
            XCTAssertEqual(error as? GateError, .inventoryAuthorityNotPromoted)
        }
    }

    func testTrustedMatchingInventoryProducesCanonicalReadyRecord() throws {
        let inventory = inventory()
        let authority = authority(for: inventory)

        let record = try Gate.validate(
            inventoryData: canonicalData(inventory),
            observedIdentityData: canonicalData(observed()),
            sourceRevision: sourceRevision,
            trustedInventoryAuthority: authority)

        XCTAssertEqual(record.schemaVersion, Gate.schemaVersion)
        XCTAssertEqual(record.status, "ready")
        XCTAssertEqual(record.hostLabel, inventory.hostLabel)
        XCTAssertEqual(
            record.expectedFingerprintSHA256,
            inventory.expectedFingerprintSHA256)
        XCTAssertEqual(
            record.observedFingerprintSHA256,
            inventory.expectedFingerprintSHA256)
        XCTAssertEqual(
            record.inventoryRecordSHA256,
            authority.inventoryRecordSHA256)
        XCTAssertEqual(record.sourceRevision, sourceRevision)
        XCTAssertTrue(record.noRemoteActionPerformed)
        XCTAssertFalse(record.issuesExactnessAuthority)
        XCTAssertFalse(record.issuesPerformanceAuthority)

        let encoded = canonicalData(record)
        XCTAssertEqual(
            try JSONDecoder().decode(
                Qwen38HeavyHostTrustReadinessRecord.self,
                from: encoded),
            record)
    }

    func testUnpromotedInventorySubstitutionCannotSelfAttestTrust() throws {
        let trusted = inventory()
        var substituted = trusted
        substituted.expectedFingerprintSHA256 = hex("b")

        XCTAssertThrowsError(
            try Gate.validate(
                inventoryData: canonicalData(substituted),
                observedIdentityData: canonicalData(observed(fingerprint: hex("b"))),
                sourceRevision: sourceRevision,
                trustedInventoryAuthority: authority(for: trusted))
        ) { error in
            XCTAssertEqual(error as? GateError, .inventoryAuthorityMismatch)
        }
    }

    func testHostOrFingerprintMismatchFailsClosed() throws {
        let inventory = inventory()
        let authority = authority(for: inventory)

        XCTAssertThrowsError(
            try Gate.validate(
                inventoryData: canonicalData(inventory),
                observedIdentityData: canonicalData(observed(hostLabel: "different-heavy-host")),
                sourceRevision: sourceRevision,
                trustedInventoryAuthority: authority)
        ) { error in
            XCTAssertEqual(error as? GateError, .hostLabelMismatch)
        }

        XCTAssertThrowsError(
            try Gate.validate(
                inventoryData: canonicalData(inventory),
                observedIdentityData: canonicalData(observed(fingerprint: hex("b"))),
                sourceRevision: sourceRevision,
                trustedInventoryAuthority: authority)
        ) { error in
            XCTAssertEqual(error as? GateError, .fingerprintMismatch)
        }
    }

    func testMalformedAndNonCanonicalInputsFailClosed() throws {
        let inventory = inventory()
        let authority = authority(for: inventory)

        XCTAssertThrowsError(
            try Gate.validate(
                inventoryData: Data("{".utf8),
                observedIdentityData: canonicalData(observed()),
                sourceRevision: sourceRevision,
                trustedInventoryAuthority: authority)
        ) { error in
            XCTAssertEqual(error as? GateError, .malformedInventory)
        }

        let nonCanonicalInventory = canonicalData(inventory) + Data("\n".utf8)
        XCTAssertThrowsError(
            try Gate.validate(
                inventoryData: nonCanonicalInventory,
                observedIdentityData: canonicalData(observed()),
                sourceRevision: sourceRevision,
                trustedInventoryAuthority: authority)
        ) { error in
            XCTAssertEqual(error as? GateError, .nonCanonicalInventory)
        }

        XCTAssertThrowsError(
            try Gate.validate(
                inventoryData: canonicalData(inventory),
                observedIdentityData: Data("{}".utf8),
                sourceRevision: sourceRevision,
                trustedInventoryAuthority: authority)
        ) { error in
            XCTAssertEqual(error as? GateError, .malformedObservedIdentity)
        }
    }

    func testDirtyUnpinnedAndMalformedIdentityFieldsFailClosed() throws {
        let inventory = inventory()
        let trustedAuthority = authority(for: inventory)

        for invalidSource in [
            sourceRevision + "-dirty",
            "unknown",
            String(repeating: "A", count: 40),
        ] {
            XCTAssertThrowsError(
                try Gate.validate(
                    inventoryData: canonicalData(inventory),
                    observedIdentityData: canonicalData(observed()),
                    sourceRevision: invalidSource,
                    trustedInventoryAuthority: trustedAuthority)
            ) { error in
                XCTAssertEqual(error as? GateError, .invalidSourceRevision)
            }
        }

        var invalidLabel = inventory
        invalidLabel.hostLabel = "invalid.host.label"
        XCTAssertThrowsError(
            try Gate.validate(
                inventoryData: canonicalData(invalidLabel),
                observedIdentityData: canonicalData(observed()),
                sourceRevision: sourceRevision,
                trustedInventoryAuthority: authority(for: invalidLabel))
        ) { error in
            XCTAssertEqual(error as? GateError, .invalidInventoryIdentity)
        }
    }

    private let sourceRevision = String(repeating: "a", count: 40)

    private func inventory() -> Qwen38HeavyHostTrustInventory {
        Qwen38HeavyHostTrustInventory(
            schemaVersion: Gate.schemaVersion,
            hostLabel: "dedicated-heavy-256gib",
            expectedFingerprintSHA256: hex("a"))
    }

    private func observed(
        hostLabel: String = "dedicated-heavy-256gib",
        fingerprint: String? = nil
    ) -> Qwen38HeavyHostObservedIdentity {
        Qwen38HeavyHostObservedIdentity(
            schemaVersion: Gate.schemaVersion,
            hostLabel: hostLabel,
            observedFingerprintSHA256: fingerprint ?? hex("a"))
    }

    private func authority(
        for inventory: Qwen38HeavyHostTrustInventory
    ) -> Qwen38HeavyHostTrustInventoryAuthority {
        Qwen38HeavyHostTrustInventoryAuthority(
            inventory: inventory,
            inventoryRecordSHA256: Gate.sha256Hex(canonicalData(inventory)))
    }

    private func canonicalData<T: Encodable>(_ value: T) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try! encoder.encode(value)
    }

    private func hex(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }
}
