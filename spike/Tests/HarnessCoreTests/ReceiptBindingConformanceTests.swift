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

    private func changeTime(_ url: URL) -> PackReceiptBinding.ChangeTime? {
        PackReceiptBinding.statChangeTime(url.path)
    }

    private func bumpChangeTime(_ url: URL) throws {
        var info = stat()
        XCTAssertEqual(stat(url.path, &info), 0)
        XCTAssertEqual(chmod(url.path, info.st_mode & 0o7777), 0)
    }

    /// Every regular file (symlinks resolved) under `directory`; a dangling symlink is skipped.
    private func resolvedFiles(under directory: URL) -> [URL] {
        var found: [URL] = []
        let fm = FileManager.default
        guard let names = fm.enumerator(atPath: directory.path)?.allObjects as? [String] else {
            return []
        }
        for name in names {
            let url = directory.appendingPathComponent(name)
            var info = stat()
            if stat(url.path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG { found.append(url) }
        }
        return found
    }

    /// Writes the sibling receipt and makes its ctime STRICTLY greater than every created file's.
    private func writeReceiptAndOrder(_ receipt: [String: Any], pack: URL) throws -> URL {
        let receiptURL = URL(fileURLWithPath: PackReceipt.receiptPath(for: pack))
        XCTAssertNotEqual(receiptURL.path, pack.path)
        try JSONSerialization.data(withJSONObject: receipt).write(to: receiptURL)
        let files = resolvedFiles(under: pack)
        for _ in 0..<500 {
            if let receiptTime = changeTime(receiptURL),
                files.allSatisfy({ (changeTime($0).map { $0 < receiptTime }) ?? false })
            {
                return receiptURL
            }
            Thread.sleep(forTimeInterval: 0.002)
            try bumpChangeTime(receiptURL)
        }
        XCTFail("could not order the receipt ctime after the pack files")
        return receiptURL
    }

    private func orderNotBefore(_ url: URL, receipt receiptURL: URL) throws {
        for _ in 0..<500 {
            if let receiptTime = changeTime(receiptURL), let time = changeTime(url),
                time >= receiptTime
            {
                return
            }
            Thread.sleep(forTimeInterval: 0.002)
            try bumpChangeTime(url)
        }
        XCTFail("could not order the file ctime at or after the receipt")
    }

    /// `editedAfterReceipt` (one byte changed in place, same size) and `touchedAfterReceipt`
    /// (ctime bump only), both applied after the receipt exists.
    private func applyPostReceiptEdits(_ spec: [String: Any], pack: URL, receipt receiptURL: URL) throws {
        let files = (spec["files"] as? [String: [String: Any]]) ?? [:]
        for name in (spec["editedAfterReceipt"] as? [String]) ?? [] {
            let url = pack.appendingPathComponent(name)
            var data = try Data(contentsOf: url)
            if files[name]?["size"] != nil {
                XCTAssertEqual(data.first, 0x78)
                data[0] = 0x79
            } else {
                data[0] ^= 0x01
            }
            let handle = try FileHandle(forWritingTo: url)  // in place, no truncate
            try handle.write(contentsOf: data)
            try handle.close()
            try orderNotBefore(url, receipt: receiptURL)
        }
        for name in (spec["touchedAfterReceipt"] as? [String]) ?? [] {
            try orderNotBefore(pack.appendingPathComponent(name), receipt: receiptURL)
        }
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
            "large-file-edited-after-receipt-is-hashed", "large-file-restored-after-receipt-is-ok",
            "receipt-unstatable-hashes-large-files",
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
            if (spec["receiptMissing"] as? Bool) != true {
                let receiptURL = try writeReceiptAndOrder(receipt, pack: pack)
                try applyPostReceiptEdits(spec, pack: pack, receipt: receiptURL)
            }
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
