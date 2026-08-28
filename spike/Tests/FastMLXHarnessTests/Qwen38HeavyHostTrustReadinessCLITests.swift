import Foundation
import HarnessCore
import XCTest

@testable import fastmlx_harness

final class Qwen38HeavyHostTrustReadinessCLITests: XCTestCase {
    private typealias Gate = Qwen38HeavyHostTrustReadinessGate
    private typealias GateError = Qwen38HeavyHostTrustReadinessGateError
    private typealias CLIError = Qwen38HeavyHostTrustReadinessCLIError

    func testStrictArgumentsRequireInventoryAndObservedFilesExactlyOnce() throws {
        let parsed = try parseQwen38HeavyHostTrustReadinessArguments([
            "--inventory", "inventory.json",
            "--observed", "observed.json",
        ])
        XCTAssertEqual(parsed.inventoryPath, "inventory.json")
        XCTAssertEqual(parsed.observedIdentityPath, "observed.json")

        XCTAssertThrowsError(try parseQwen38HeavyHostTrustReadinessArguments([])) { error in
            XCTAssertEqual(error as? CLIError, .missingFlag("--inventory"))
        }
        XCTAssertThrowsError(try parseQwen38HeavyHostTrustReadinessArguments([
            "--inventory", "inventory.json",
        ])) { error in
            XCTAssertEqual(error as? CLIError, .missingFlag("--observed"))
        }
        XCTAssertThrowsError(try parseQwen38HeavyHostTrustReadinessArguments([
            "--inventory", "inventory.json",
            "--observed", "observed.json",
            "--inventory", "other.json",
        ])) { error in
            XCTAssertEqual(error as? CLIError, .duplicateFlag("--inventory"))
        }
        XCTAssertThrowsError(try parseQwen38HeavyHostTrustReadinessArguments([
            "private/operator/inventory.json",
        ])) { error in
            XCTAssertEqual(error as? CLIError, .unexpectedPositional)
            XCTAssertFalse(externalDiagnostic(error).contains("private/operator"))
        }
    }

    func testUnpromotedAuthorityBlocksBeforeAnyFixtureOrSourceRead() throws {
        var readPaths: [String] = []
        var sourceRead = false

        XCTAssertThrowsError(
            try validateQwen38HeavyHostTrustReadiness(
                arguments: [
                    "--inventory", "inventory.json",
                    "--observed", "observed.json",
                ],
                readFile: { path in
                    readPaths.append(path)
                    return Data()
                },
                sourceRevision: {
                    sourceRead = true
                    return self.sourceRevision
                })
        ) { error in
            XCTAssertEqual(error as? GateError, .inventoryAuthorityNotPromoted)
            XCTAssertEqual(qwen38HeavyHostTrustReadinessDisposition(error), .blocked)
        }
        XCTAssertTrue(readPaths.isEmpty)
        XCTAssertFalse(sourceRead)
    }

    func testMissingTrustedInventoryBlocksBeforeObservedOrSourceRead() throws {
        var readPaths: [String] = []
        var sourceRead = false

        XCTAssertThrowsError(
            try validateQwen38HeavyHostTrustReadiness(
                arguments: [
                    "--inventory", "private/operator/inventory.json",
                    "--observed", "private/operator/observed.json",
                ],
                trustedInventoryAuthority: authority(),
                readFile: { path in
                    readPaths.append(path)
                    throw CocoaError(.fileReadNoSuchFile)
                },
                sourceRevision: {
                    sourceRead = true
                    return self.sourceRevision
                })
        ) { error in
            XCTAssertEqual(error as? CLIError, .inventoryUnavailable)
            XCTAssertEqual(qwen38HeavyHostTrustReadinessDisposition(error), .blocked)
            XCTAssertFalse(externalDiagnostic(error).contains("private/operator"))
        }
        XCTAssertEqual(readPaths, ["private/operator/inventory.json"])
        XCTAssertFalse(sourceRead)
    }

    func testTrustedMatchEmitsOnlyCanonicalReadinessJSON() throws {
        let output = try validateQwen38HeavyHostTrustReadiness(
            arguments: [
                "--inventory", "inventory.json",
                "--observed", "observed.json",
            ],
            trustedInventoryAuthority: authority(),
            readFile: { path in
                switch path {
                case "inventory.json": return self.canonicalData(self.inventory())
                case "observed.json": return self.canonicalData(self.observed())
                default: throw CocoaError(.fileReadNoSuchFile)
                }
            },
            sourceRevision: { self.sourceRevision })

        let record = try JSONDecoder().decode(
            Qwen38HeavyHostTrustReadinessRecord.self,
            from: Data(output.utf8))
        XCTAssertEqual(record.status, "ready")
        XCTAssertTrue(record.noRemoteActionPerformed)
        XCTAssertFalse(record.issuesExactnessAuthority)
        XCTAssertFalse(record.issuesPerformanceAuthority)
        XCTAssertEqual(output, String(data: canonicalData(record), encoding: .utf8))
        XCTAssertFalse(output.contains("inventory.json"))
        XCTAssertFalse(output.contains("VALID"))
    }

    func testMismatchAndObservedReadFailuresAreInvalidAndRedacted() throws {
        XCTAssertThrowsError(
            try validateQwen38HeavyHostTrustReadiness(
                arguments: [
                    "--inventory", "inventory.json",
                    "--observed", "private/operator/observed.json",
                ],
                trustedInventoryAuthority: authority(),
                readFile: { path in
                    if path == "inventory.json" {
                        return self.canonicalData(self.inventory())
                    }
                    throw CocoaError(.fileReadNoSuchFile)
                },
                sourceRevision: { self.sourceRevision })
        ) { error in
            XCTAssertEqual(error as? CLIError, .fileReadFailed("--observed"))
            XCTAssertEqual(qwen38HeavyHostTrustReadinessDisposition(error), .invalid)
            XCTAssertFalse(externalDiagnostic(error).contains("private/operator"))
        }

        XCTAssertThrowsError(
            try validateQwen38HeavyHostTrustReadiness(
                arguments: [
                    "--inventory", "inventory.json",
                    "--observed", "observed.json",
                ],
                trustedInventoryAuthority: authority(),
                readFile: { path in
                    path == "inventory.json"
                        ? self.canonicalData(self.inventory())
                        : self.canonicalData(self.observed(fingerprint: self.hex("b")))
                },
                sourceRevision: { self.sourceRevision })
        ) { error in
            XCTAssertEqual(error as? GateError, .fingerprintMismatch)
            XCTAssertEqual(qwen38HeavyHostTrustReadinessDisposition(error), .invalid)
            XCTAssertEqual(externalDiagnostic(error), "trust readiness validation failed")
        }
    }

    private let sourceRevision = String(repeating: "a", count: 40)

    private func externalDiagnostic(_ error: Error) -> String {
        qwen38HeavyHostTrustReadinessExternalDiagnostic(error)
    }

    private func inventory() -> Qwen38HeavyHostTrustInventory {
        Qwen38HeavyHostTrustInventory(
            schemaVersion: Gate.schemaVersion,
            hostLabel: "dedicated-heavy-256gib",
            expectedFingerprintSHA256: hex("a"))
    }

    private func observed(
        fingerprint: String? = nil
    ) -> Qwen38HeavyHostObservedIdentity {
        Qwen38HeavyHostObservedIdentity(
            schemaVersion: Gate.schemaVersion,
            hostLabel: "dedicated-heavy-256gib",
            observedFingerprintSHA256: fingerprint ?? hex("a"))
    }

    private func authority() -> Qwen38HeavyHostTrustInventoryAuthority {
        let inventory = inventory()
        return Qwen38HeavyHostTrustInventoryAuthority(
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
