import Foundation
import XCTest
@testable import HarnessCore

/// Swift side of the shared pack-identity conformance table
/// (`Fixtures/pack-identity-conformance-v1.json`). The same table is checked against the Python
/// `fastmlx_launch.derive_pack_identity` by `scripts/tests/test_pack_identity_conformance.py`.
/// This engine is the authority; if the two drift, `fastmlx recommend` vouches for a pack the
/// engine refuses.
final class PackIdentityConformanceTests: XCTestCase {
    private struct Table: Decodable {
        struct Receipt: Decodable {
            let text: String?
            let base64: String?
        }
        struct Symlink: Decodable {
            let link: String
            let target: String
        }
        struct Conflict: Decodable {
            let receiptRepo: String
            let pathRepo: String
        }
        struct Expect: Decodable {
            let repo: String?
            let revision: String?
            let conflict: Conflict?
        }
        struct Case: Decodable {
            let id: String
            let why: String
            let modelDir: String
            let createModelDir: Bool?
            let receipt: Receipt?
            let receiptBase: String?
            let symlinks: [Symlink]?
            let extraDirs: [String]?
            let inputPath: String?
            let expect: Expect?
        }
        let schema: String
        let cases: [Case]
    }

    private func table() throws -> Table {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "pack-identity-conformance-v1", withExtension: "json"))
        return try JSONDecoder().decode(Table.self, from: Data(contentsOf: url))
    }

    /// Materializes one case under `root` and returns the input path string.
    private func build(_ c: Table.Case, root: String) throws -> String {
        let fm = FileManager.default
        let modelDir = "\(root)/\(c.modelDir)"
        if c.createModelDir ?? true {
            try fm.createDirectory(atPath: modelDir, withIntermediateDirectories: true)
        }
        try fm.createDirectory(atPath: "\(root)/other-dir", withIntermediateDirectories: true)
        for extra in c.extraDirs ?? [] {
            try fm.createDirectory(atPath: "\(root)/\(extra)", withIntermediateDirectories: true)
        }
        for link in c.symlinks ?? [] {
            let linkPath = "\(root)/\(link.link)"
            try fm.createDirectory(
                atPath: (linkPath as NSString).deletingLastPathComponent,
                withIntermediateDirectories: true)
            try fm.createSymbolicLink(atPath: linkPath, withDestinationPath: "\(root)/\(link.target)")
        }
        if let receipt = c.receipt {
            let data: Data
            if let b64 = receipt.base64 {
                data = try XCTUnwrap(Data(base64Encoded: b64), "\(c.id): bad base64")
            } else {
                let resolved = URL(fileURLWithPath: modelDir).resolvingSymlinksInPath().path
                let text = try XCTUnwrap(receipt.text, "\(c.id): receipt needs text or base64")
                    .replacingOccurrences(of: "{DIR_RESOLVED}", with: resolved)
                    .replacingOccurrences(of: "{DIR}", with: modelDir)
                    .replacingOccurrences(of: "{OTHER}", with: "\(root)/other-dir")
                    .replacingOccurrences(of: "{ROOT}", with: root)
                data = Data(text.utf8)
            }
            let base = "\(root)/\(c.receiptBase ?? c.modelDir)"
            try data.write(to: URL(fileURLWithPath: "\(base).pull-receipt.json"))
        }
        return "\(root)/\(c.inputPath ?? c.modelDir)"
    }

    func testEngineMatchesTheSharedTable() throws {
        let table = try table()
        XCTAssertEqual(table.schema, "pack-identity-conformance-v1")
        XCTAssertGreaterThan(table.cases.count, 50, "the table must not be silently truncated")
        let ids = table.cases.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count, "duplicate case ids")
        var failures: [String] = []
        for c in table.cases {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("packid-\(UUID().uuidString)").path
            defer { try? FileManager.default.removeItem(atPath: root) }
            try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
            let input = try build(c, root: root)
            do {
                let actual = try PackIdentity.derive(directory: URL(fileURLWithPath: input))
                if let conflict = c.expect?.conflict {
                    failures.append("\(c.id): expected conflict \(conflict) but got \(String(describing: actual))")
                } else if let expect = c.expect {
                    let want = PackIdentity(repo: expect.repo, revision: expect.revision)
                    if actual != want {
                        failures.append("\(c.id): expected \(want) but got \(String(describing: actual)) -- \(c.why)")
                    }
                } else if actual != nil {
                    failures.append("\(c.id): expected nil but got \(String(describing: actual)) -- \(c.why)")
                }
            } catch let error as PackIdentityConflict {
                if let conflict = c.expect?.conflict {
                    if error.receiptRepo != conflict.receiptRepo || error.pathRepo != conflict.pathRepo {
                        failures.append("\(c.id): conflict mismatch \(error.receiptRepo) / \(error.pathRepo)")
                    }
                } else {
                    failures.append("\(c.id): unexpected conflict \(error) -- \(c.why)")
                }
            }
        }
        XCTAssertTrue(failures.isEmpty, "\(failures.count) case(s) diverge:\n" + failures.joined(separator: "\n"))
    }

    func testTableCoversEveryOutcomeKind() throws {
        let cases = try table().cases
        XCTAssertTrue(cases.contains { $0.expect == nil }, "no nil-identity case")
        XCTAssertTrue(cases.contains { $0.expect?.conflict != nil }, "no conflict case")
        XCTAssertTrue(cases.contains { $0.expect?.repo != nil && $0.expect?.revision != nil }, "no full-identity case")
        XCTAssertTrue(cases.contains { $0.expect?.repo != nil && $0.expect?.revision == nil }, "no repo-only case")
        XCTAssertTrue(cases.contains { $0.expect?.repo == nil && $0.expect?.revision != nil }, "no revision-only case")
    }
}
