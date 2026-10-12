import CryptoKit
import Foundation
import XCTest
@testable import HarnessCore

/// Swift side of the shared receipt-binding conformance table
/// (`Fixtures/receipt-binding-conformance-v1.json`). The same table is checked against the Python
/// `fastmlx_launch.verify_receipt_binding` by `scripts/tests/test_receipt_binding_conformance.py`.
/// The two must agree on the status and, for a mismatch, on the first difference's kind and path.
final class ReceiptBindingConformanceTests: XCTestCase {
    private func table() throws -> [String: Any] {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "receipt-binding-conformance-v1", withExtension: "json"))
        return try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    private func cases() throws -> [[String: Any]] {
        try XCTUnwrap(try table()["cases"] as? [[String: Any]])
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func buildTree(_ spec: [String: Any], at root: URL) throws {
        let fm = FileManager.default
        for (relative, value) in (spec["files"] as? [String: [String: Any]]) ?? [:] {
            let url = root.appendingPathComponent(relative)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if let text = value["text"] as? String {
                try Data(text.utf8).write(to: url)
            } else if let size = value["size"] as? Int {
                try Data(repeating: 0x78, count: size).write(to: url)
            }
        }
        for relative in (spec["dirs"] as? [String]) ?? [] {
            try fm.createDirectory(
                at: root.appendingPathComponent(relative), withIntermediateDirectories: true)
        }
        for link in (spec["symlinks"] as? [[String: String]]) ?? [] {
            try fm.createSymbolicLink(
                atPath: root.appendingPathComponent(link["link"]!).path,
                withDestinationPath: link["target"]!)
        }
    }

    private func treeEntry(_ relative: String, at root: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: root.appendingPathComponent(relative))
        return ["size": data.count, "sha256": sha256Hex(data)]
    }

    private func buildReceipt(_ spec: [String: Any], at root: URL) throws -> [String: Any] {
        var receipt: [String: Any] = ["repo_id": "example-org/Pack", "revision": String(repeating: "a", count: 40)]
        guard let receiptFiles = spec["receiptFiles"] else { return receipt }
        if let mode = receiptFiles as? String, mode == "fromTree" {
            let exclude = Set((spec["receiptExclude"] as? [String]) ?? [])
            var files: [String: [String: Any]] = [:]
            for relative in ((spec["files"] as? [String: Any]) ?? [:]).keys where !exclude.contains(relative) {
                files[relative] = try treeEntry(relative, at: root)
            }
            for (relative, edits) in (spec["receiptOverrides"] as? [String: [String: Any]]) ?? [:] {
                var entry = try XCTUnwrap(files[relative])
                for (key, value) in edits {
                    if key == "upperSha256" {
                        entry["sha256"] = (entry["sha256"] as? String)?.uppercased()
                    } else if value is NSNull {
                        entry.removeValue(forKey: key)
                    } else {
                        entry[key] = value
                    }
                }
                files[relative] = entry
            }
            for (relative, extra) in (spec["receiptExtra"] as? [String: [String: Any]]) ?? [:] {
                var built: [String: Any] = [:]
                if let source = extra["copyFrom"] as? String { built = try treeEntry(source, at: root) }
                for (key, value) in extra where key != "copyFrom" { built[key] = value }
                files[relative] = built
            }
            receipt["files"] = files
        } else {
            receipt["files"] = receiptFiles
        }
        return receipt
    }

    func testFixtureIsWellFormed() throws {
        XCTAssertEqual(try table()["schema"] as? String, "receipt-binding-conformance-v1")
        let all = try cases()
        let ids = all.compactMap { $0["id"] as? String }
        XCTAssertEqual(ids.count, all.count)
        XCTAssertEqual(Set(ids).count, ids.count, "duplicate case ids")
        let statuses = Set(all.compactMap { ($0["expect"] as? [String: Any])?["status"] as? String })
        XCTAssertEqual(statuses, ["ok", "mismatch", "uncheckable"])
        for required in [
            "ok-matching-tree", "size-differs", "sha256-differs", "missing-listed-file",
            "unlisted-top-level-file", "unlisted-nested-file", "dot-components-are-ignored",
            "nested-listed-file-ok", "listed-prefix-is-a-file", "size-only-above-hash-threshold",
            "hash-threshold-is-inclusive", "uncheckable-no-files-key", "uncheckable-name-absolute",
            "uncheckable-entry-is-a-string", "uncheckable-size-bool", "uncheckable-sha256-short",
        ] {
            XCTAssertTrue(ids.contains(required), "missing required case \(required)")
        }
        XCTAssertTrue(all.contains { $0["hashMaxBytes"] != nil })
    }

    func testSwiftVerifyMatchesTheSharedTable() throws {
        let all = try cases()
        XCTAssertGreaterThan(all.count, 30)
        for spec in all {
            let id = try XCTUnwrap(spec["id"] as? String)
            let expect = try XCTUnwrap(spec["expect"] as? [String: Any])
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("receipt-conformance-\(UUID().uuidString)", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let pack = root.appendingPathComponent("pack", isDirectory: true)
            try FileManager.default.createDirectory(at: pack, withIntermediateDirectories: true)
            try buildTree(spec, at: pack)
            let receipt = try buildReceipt(spec, at: pack)
            let threshold = (spec["hashMaxBytes"] as? Int).map(Int64.init) ?? PackAdmission.defaultReceiptHashMaxBytes
            let outcome = PackReceiptBinding.verify(
                directory: pack, receipt: receipt, hashMaxBytes: threshold)
            let why = "\(id): \(spec["why"] as? String ?? "")"
            switch (expect["status"] as? String, outcome) {
            case ("ok", .ok):
                break
            case ("uncheckable", .uncheckable(let reason)):
                let needle = try XCTUnwrap(expect["reasonContains"] as? String)
                XCTAssertTrue(reason.contains(needle), "\(why) -> \(reason)")
            case ("mismatch", .mismatch(let difference)):
                XCTAssertEqual(difference.kind.rawValue, expect["kind"] as? String, why)
                XCTAssertEqual(difference.path, expect["path"] as? String, why)
            default:
                XCTFail("\(why): expected \(String(describing: expect["status"])), got \(outcome)")
            }
        }
    }
}
