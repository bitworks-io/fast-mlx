import CryptoKit
import Foundation
import XCTest
@testable import HarnessCore

/// The engine refuses a pack whose bytes no longer match its `fastmlx pull` receipt, exactly as the
/// Python launcher does (`verify_receipt_binding` / `judge_receipt_binding`). See
/// docs/task-inbox/2026-10-12-PREDECLARATION-engine-refuses-a-pack-that-drifted-from-its-receipt.md.
final class PackReceiptBindingTests: XCTestCase {
    private static let passRepo = "example-org/Pack-8bit"
    private static let noGoRepo = "example-org/Pack-4bit"
    private static let revA = String(repeating: "a1", count: 20)

    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("receipt-binding-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: scratch)
    }

    // MARK: fixtures

    private var cards: [QualityCard] {
        func make(_ id: String, _ repo: String, _ verdict: QualityVerdict) -> QualityCard {
            QualityCard(
                id: id, model: .init(repo: repo, hfPin: nil), verdict: verdict,
                admission: .init(default: false, optIn: true, reason: "fixture"),
                legible: .init(tier: "Noticeable", headline: "About 1 word in 12 differs."))
        }
        return [
            make("pack-8bit@host", Self.passRepo, .pass),
            make("pack-4bit@host", Self.noGoRepo, .noGo),
            make("alias-ok@host", "alias-org/Alias", .pass),
        ]
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private struct Pack {
        let directory: URL
        let receiptURL: URL
    }

    private static let layout: [(String, String)] = [
        ("config.json", "{\"hidden_size\": 64}\n"),
        ("model.safetensors", "weights-weights-weights-weights"),
        ("tokenizer/tokenizer.json", "{\"version\": \"1.0\"}\n"),
    ]

    /// Writes a small pack and a sibling receipt describing exactly those bytes.
    private func makePack(
        name: String = "pack", repo: String = PackReceiptBindingTests.passRepo,
        dest: String? = nil, withFilesMap: Bool = true
    ) throws -> Pack {
        let fm = FileManager.default
        let directory = scratch.appendingPathComponent(name, isDirectory: true)
        var files: [String: Any] = [:]
        for (relative, text) in Self.layout {
            let url = directory.appendingPathComponent(relative)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = Data(text.utf8)
            try data.write(to: url)
            files[relative] = ["size": data.count, "sha256": sha256Hex(data)]
        }
        var receipt: [String: Any] = [
            "repo_id": repo, "revision": Self.revA, "dest": dest ?? directory.path,
        ]
        if withFilesMap { receipt["files"] = files }
        let receiptURL = URL(fileURLWithPath: directory.path + ".pull-receipt.json")
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys]).write(to: receiptURL)
        return Pack(directory: directory, receiptURL: receiptURL)
    }

    private func judge(
        _ directory: URL?, alias: String = "my-model", accept: [String] = [],
        using cardSet: [QualityCard]? = nil
    ) -> PackAdmission.Judgement {
        PackAdmission.judge(
            alias: alias, aliasRevision: nil, directory: directory, hostHardwareClass: nil,
            cards: cardSet ?? cards, optIn: QualityOptIn(acceptedIDs: Set(accept)))
    }

    /// A PASS card identified ONLY by `hfPin` (and optionally a different repo).
    private func pinCard(repo: String?, hfPin: String) -> QualityCard {
        QualityCard(
            id: "pin-only@host", model: .init(repo: repo, hfPin: hfPin), verdict: .pass,
            admission: .init(default: false, optIn: true, reason: "fixture"),
            legible: .init(tier: "Noticeable", headline: "About 1 word in 12 differs."))
    }

    /// Replaces the receipt with hand-built JSON text so sizes and names Swift cannot serialize
    /// (integers beyond 64 bits, floats, NFC and NFD siblings) reach the loader as written.
    private func writeRawReceipt(_ pack: Pack, filesJSON: String) throws {
        let text =
            "{\"repo_id\": \"\(Self.passRepo)\", \"revision\": \"\(Self.revA)\", \"files\": {\(filesJSON)}}"
        try Data(text.utf8).write(to: pack.receiptURL)
    }

    private func entryJSON(_ relative: String, sizeLiteral: String? = nil) throws -> String {
        let data = try XCTUnwrap(
            Self.layout.first { $0.0 == relative }.map { Data($0.1.utf8) })
        return "\"\(relative)\": {\"size\": \(sizeLiteral ?? String(data.count)), \"sha256\": \"\(sha256Hex(data))\"}"
    }

    private func layoutFilesJSON(configSizeLiteral: String) throws -> String {
        [
            try entryJSON("config.json", sizeLiteral: configSizeLiteral),
            try entryJSON("model.safetensors"),
            try entryJSON("tokenizer/tokenizer.json"),
        ].joined(separator: ", ")
    }

    private func append(_ byte: UInt8, to url: URL) throws {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data([byte]))
    }

    private func assertReceiptRefusal(
        _ judgement: PackAdmission.Judgement, pack: Pack, contains fragments: [String],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        guard case .refuse(let card, let notices, let reason, let message) = judgement else {
            return XCTFail("expected refusal, got \(judgement)", file: file, line: line)
        }
        XCTAssertNil(card, file: file, line: line)
        XCTAssertEqual(notices, [], file: file, line: line)
        XCTAssertEqual(reason, "quality_card_receipt_mismatch", file: file, line: line)
        XCTAssertTrue(
            message.hasPrefix(
                "fastmlx-serve configuration=refused reason=quality_card_receipt_mismatch detail=the pack at \(pack.directory.path) no longer matches its pull receipt \(pack.receiptURL.path): "),
            message, file: file, line: line)
        XCTAssertTrue(
            message.hasSuffix(
                "; its quality card describes the bytes that were pulled, not these -- re-run `fastmlx pull` for this directory, or remove the stale receipt so the pack is judged as unidentified"),
            message, file: file, line: line)
        XCTAssertFalse(message.contains("\n"), file: file, line: line)
        for fragment in fragments {
            XCTAssertTrue(message.contains(fragment), "\(fragment) not in \(message)", file: file, line: line)
        }
    }

    // MARK: AC1 -- drift refuses when the card matches the receipt identity

    func testT1AppendedByteRefusesWithSizeDifference() throws {
        let pack = try makePack()
        try append(0x21, to: pack.directory.appendingPathComponent("model.safetensors"))
        assertReceiptRefusal(
            judge(pack.directory), pack: pack,
            contains: ["'model.safetensors' size differs (receipt 31, found 32)"])
    }

    func testT2SameSizeByteFlipRefusesWithShaDifference() throws {
        let pack = try makePack()
        let url = pack.directory.appendingPathComponent("model.safetensors")
        var bytes = try Data(contentsOf: url)
        bytes[0] ^= 0x01
        try bytes.write(to: url)
        assertReceiptRefusal(
            judge(pack.directory), pack: pack, contains: ["'model.safetensors' sha256 differs (receipt "])
    }

    func testT3UnlistedFileAndUnlistedSubdirectoryRefuseNamingTheFirst() throws {
        let pack = try makePack()
        try Data("x".utf8).write(to: pack.directory.appendingPathComponent("extra.bin"))
        assertReceiptRefusal(
            judge(pack.directory), pack: pack,
            contains: ["'extra.bin' is in the pack but not listed in the receipt"])
        try FileManager.default.removeItem(at: pack.directory.appendingPathComponent("extra.bin"))
        let nested = pack.directory.appendingPathComponent("adapters", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: nested.appendingPathComponent("lora.bin"))
        assertReceiptRefusal(
            judge(pack.directory), pack: pack,
            contains: ["'adapters/lora.bin' is in the pack but not listed in the receipt"])
    }

    func testT4DeletedListedFileRefusesAsMissing() throws {
        let pack = try makePack()
        try FileManager.default.removeItem(at: pack.directory.appendingPathComponent("config.json"))
        assertReceiptRefusal(
            judge(pack.directory), pack: pack,
            contains: ["'config.json' is listed in the receipt but missing"])
    }

    // MARK: AC2 -- notice only

    func testT5DriftWhoseSelectedCardDoesNotMatchTheReceiptIsOneNoticeFirst() throws {
        // The receipt names a repo no card describes; the alias resolves a PASS card that is NOT
        // the receipt's evidence, so the drift is a notice and the card still decides.
        let pack = try makePack(repo: "unknown-org/Nothing")
        try append(0x21, to: pack.directory.appendingPathComponent("config.json"))
        guard case .admit(let card, let notices, let flag) = judge(pack.directory, alias: "alias-org/Alias")
        else { return XCTFail("expected admit") }
        XCTAssertEqual(card?.id, "alias-ok@host")
        XCTAssertNil(flag)
        XCTAssertEqual(notices.count, 1, "\(notices)")
        XCTAssertEqual(
            notices.first,
            "fastmlx-serve: pull receipt \(pack.receiptURL.path) does not match the pack's bytes ('config.json' size differs (receipt 20, found 21)); the receipt is not evidence for any quality card here")
    }

    func testT5DriftWithNoCardAtAllIsAlsoANoticeOnly() throws {
        let pack = try makePack(repo: "unknown-org/Nothing")
        try Data("x".utf8).write(to: pack.directory.appendingPathComponent("extra.bin"))
        guard case .admit(let card, let notices, _) = judge(pack.directory) else {
            return XCTFail("expected admit")
        }
        XCTAssertNil(card)
        XCTAssertEqual(notices.count, 1)
        guard let notice = notices.first else { return }
        XCTAssertTrue(notice.hasPrefix("fastmlx-serve: pull receipt "), notice)
        XCTAssertTrue(notice.contains("'extra.bin' is in the pack but not listed in the receipt"))
    }

    func testT6ReceiptWithoutAFilesMapIsANoticeOnly() throws {
        let pack = try makePack(withFilesMap: false)
        guard case .admit(let card, let notices, _) = judge(pack.directory) else {
            return XCTFail("expected admit")
        }
        XCTAssertEqual(card?.id, "pack-8bit@host")
        XCTAssertEqual(
            notices,
            ["fastmlx-serve: pull receipt \(pack.receiptURL.path) cannot be checked against the pack's bytes (it records no files map)"])
    }

    // MARK: AC3 -- positive controls (unchanged judgement, no new notice)

    func testT7MatchingPackAdmitsExactlyAsBefore() throws {
        let pack = try makePack()
        XCTAssertEqual(
            judge(pack.directory),
            .admit(card: cards[0], notices: [], flagMessage: nil))
    }

    func testT7NoReceiptAndNilDirectoryAreUnchanged() throws {
        let pack = try makePack()
        try FileManager.default.removeItem(at: pack.receiptURL)
        try append(0x21, to: pack.directory.appendingPathComponent("config.json"))
        XCTAssertEqual(judge(pack.directory), .admit(card: nil, notices: [], flagMessage: nil))
        XCTAssertEqual(
            judge(nil, alias: Self.passRepo), .admit(card: cards[0], notices: [], flagMessage: nil))
        XCTAssertEqual(
            judge(nil, alias: "alias-org/Alias"), .admit(card: cards[2], notices: [], flagMessage: nil))
    }

    func testT7ReceiptWhoseDestNamesAnotherDirectoryIsIgnored() throws {
        let other = scratch.appendingPathComponent("elsewhere", isDirectory: true)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        let pack = try makePack(dest: other.path)
        try append(0x21, to: pack.directory.appendingPathComponent("config.json"))
        XCTAssertEqual(judge(pack.directory), .admit(card: nil, notices: [], flagMessage: nil))
    }

    // MARK: AC5 -- ordering against the quality decision

    func testT9DriftWithAMatchingNoGoCardRefusesForTheDriftNotTheNoGo() throws {
        let pack = try makePack(repo: Self.noGoRepo)
        // Undrifted: the NO_GO card refuses with its own (reason-less) refusal.
        guard case .refuse(let card, _, let plainReason, _) = judge(pack.directory) else {
            return XCTFail("expected the NO_GO refusal")
        }
        XCTAssertEqual(card?.id, "pack-4bit@host")
        XCTAssertNil(plainReason)
        try append(0x21, to: pack.directory.appendingPathComponent("config.json"))
        assertReceiptRefusal(
            judge(pack.directory), pack: pack, contains: ["'config.json' size differs"])
        // An opt-in does not wave a drifted pack through either.
        assertReceiptRefusal(
            judge(pack.directory, accept: ["pack-4bit@host"]), pack: pack,
            contains: ["'config.json' size differs"])
    }

    // MARK: the accepted limit -- size only above the hash threshold

    func testT10SameSizeFlipInAFileLargerThanTheThresholdIsNotDetectedButAnAppendIs() throws {
        let pack = try makePack()
        let url = pack.directory.appendingPathComponent("model.safetensors")  // 31 bytes
        var bytes = try Data(contentsOf: url)
        bytes[0] ^= 0x01
        try bytes.write(to: url)
        func judged(threshold: Int64) -> PackAdmission.Judgement {
            PackAdmission.judge(
                alias: "my-model", aliasRevision: nil, directory: pack.directory,
                hostHardwareClass: nil, cards: cards, optIn: QualityOptIn(),
                receiptHashMaxBytes: threshold)
        }
        // Threshold below the 31-byte file but above the other two: the flip is accepted ...
        XCTAssertEqual(
            judged(threshold: 30), .admit(card: cards[0], notices: [], flagMessage: nil))
        // ... at the file's own size it is hashed (the bound is inclusive) and refused ...
        assertReceiptRefusal(
            judged(threshold: 31), pack: pack, contains: ["'model.safetensors' sha256 differs"])
        // ... and an appended byte is still a size difference above the threshold.
        try append(0x21, to: url)
        assertReceiptRefusal(
            judged(threshold: 30), pack: pack, contains: ["'model.safetensors' size differs"])
    }

    func testDefaultHashThresholdIs64MiB() {
        XCTAssertEqual(PackAdmission.defaultReceiptHashMaxBytes, 64 * 1024 * 1024)
        XCTAssertEqual(PackAdmission.receiptMismatchReason, "quality_card_receipt_mismatch")
    }

    func testReceiptNoticeRepeatedByTheSecondQuantCandidatesJudgementIsEmittedOnce() throws {
        let pack = try makePack(repo: "unknown-org/Nothing")
        try append(0x21, to: pack.directory.appendingPathComponent("config.json"))
        guard case .admit(_, let first, _) = judge(pack.directory),
            case .admit(_, let second, _) = judge(pack.directory)
        else { return XCTFail("expected admit twice") }
        XCTAssertEqual(first, second)
        var ledger = AdmissionEmissionLedger()
        XCTAssertEqual(ledger.unseen(first), first)
        XCTAssertEqual(ledger.unseen(second), [])
    }

    // MARK: identity through hfPin alone

    func testDriftRefusesWhenTheCardMatchesTheReceiptOnlyThroughHfPin() throws {
        let hfPin = String(Self.revA.prefix(8))
        for repo in [nil, "other-org/Unrelated"] {
            let set = [pinCard(repo: repo, hfPin: hfPin)]
            let pack = try makePack(name: "pinned-\(repo == nil ? "nil" : "other")", repo: "unknown-org/Nothing")
            // The undrifted pack proves the card is really selected by the pack's revision.
            XCTAssertEqual(
                judge(pack.directory, using: set), .admit(card: set[0], notices: [], flagMessage: nil))
            try append(0x21, to: pack.directory.appendingPathComponent("model.safetensors"))
            assertReceiptRefusal(
                judge(pack.directory, using: set), pack: pack,
                contains: ["'model.safetensors' size differs (receipt 31, found 32)"])
        }
    }

    // MARK: a listed directory that is a symlink

    func testSymlinkedListedPrefixDirectoryIsNotUnlisted() throws {
        let pack = try makePack()
        let real = scratch.appendingPathComponent("real-tokenizer", isDirectory: true)
        let fm = FileManager.default
        try fm.moveItem(at: pack.directory.appendingPathComponent("tokenizer"), to: real)
        try fm.createSymbolicLink(
            atPath: pack.directory.appendingPathComponent("tokenizer").path,
            withDestinationPath: real.path)
        XCTAssertEqual(
            judge(pack.directory), .admit(card: cards[0], notices: [], flagMessage: nil))
    }

    // MARK: names compare by code point (Python str ==), not canonical equivalence

    private func createFile(named name: String, in directory: URL, bytes: [UInt8]) throws {
        let descriptor = open(directory.path + "/" + name, O_CREAT | O_WRONLY, 0o644)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        defer { close(descriptor) }
        XCTAssertEqual(write(descriptor, bytes, bytes.count), bytes.count)
    }

    func testNFCListedNameDoesNotCoverTheNFDNameOnDisk() throws {
        let nfc = "caf\u{E9}.txt"
        let nfd = "cafe\u{301}.txt"
        XCTAssertEqual(nfc, nfd, "Swift String equality is canonical; the code must not rely on it")
        let pack = try makePack()
        try createFile(named: nfd, in: pack.directory, bytes: [0x68, 0x69])
        let hi = sha256Hex(Data("hi".utf8))
        let files = try layoutFilesJSON(configSizeLiteral: "20")
            + ", \"\(nfc)\": {\"size\": 2, \"sha256\": \"\(hi)\"}"
        try writeRawReceipt(pack, filesJSON: files)
        let receipt = try XCTUnwrap(PackReceipt.load(for: pack.directory))
        // The NFC name resolves on APFS (normalization-insensitive), so the listed-file pass is
        // clean; the walk then sees the NFD spelling, which is a different str in Python.
        guard case .mismatch(let difference) = PackReceiptBinding.verify(
            directory: pack.directory, receipt: receipt.object)
        else { return XCTFail("expected a mismatch") }
        XCTAssertEqual(difference.kind, .unlisted)
        XCTAssertEqual(Array(difference.path.utf8), Array(nfd.utf8))
    }

    func testNFCAndNFDEntriesInTheReceiptStayDistinct() throws {
        let nfc = "caf\u{E9}.txt"
        let nfd = "cafe\u{301}.txt"
        let pack = try makePack()
        try createFile(named: nfd, in: pack.directory, bytes: [0x68, 0x69])
        let hi = sha256Hex(Data("hi".utf8))
        // One of the two spellings records a wrong size; whichever it is, the check must reach it
        // (a map that collapses canonically equivalent keys keeps only one entry). The listed pass
        // runs in code-point order, NFD first.
        for wrong in [nfd, nfc] {
            let nfcSize = wrong.utf8.elementsEqual(nfc.utf8) ? 99 : 2
            let nfdSize = wrong.utf8.elementsEqual(nfd.utf8) ? 99 : 2
            let files = try layoutFilesJSON(configSizeLiteral: "20")
                + ", \"\(nfc)\": {\"size\": \(nfcSize), \"sha256\": \"\(hi)\"}"
                + ", \"\(nfd)\": {\"size\": \(nfdSize), \"sha256\": \"\(hi)\"}"
            try writeRawReceipt(pack, filesJSON: files)
            let receipt = try XCTUnwrap(PackReceipt.load(for: pack.directory))
            guard case .mismatch(let difference) = PackReceiptBinding.verify(
                directory: pack.directory, receipt: receipt.object)
            else { return XCTFail("expected a mismatch when \(Array(wrong.utf8)) is wrong") }
            XCTAssertEqual(difference.kind, .size)
            XCTAssertEqual(Array(difference.path.utf8), Array(wrong.utf8))
            XCTAssertTrue(difference.detail.contains("(receipt 99, found 2)"), difference.detail)
        }
    }

    // MARK: sizes beyond 64 bits, and floats

    func testSizeBeyondSixtyFourBitsIsASizeDifferenceNotUncheckable() throws {
        let big = "100000000000000000000000000000"
        let pack = try makePack()
        try writeRawReceipt(pack, filesJSON: try layoutFilesJSON(configSizeLiteral: big))
        assertReceiptRefusal(
            judge(pack.directory), pack: pack,
            contains: ["'config.json' size differs (receipt \(big), found 20)"])
    }

    func testSizeJustAboveInt64MaxIsASizeDifference() throws {
        let pack = try makePack()
        try writeRawReceipt(
            pack, filesJSON: try layoutFilesJSON(configSizeLiteral: "9223372036854775808"))
        assertReceiptRefusal(
            judge(pack.directory), pack: pack,
            contains: ["'config.json' size differs (receipt 9223372036854775808, found 20)"])
    }

    func testFloatAndNegativeSizesStayUncheckable() throws {
        let pack = try makePack()
        for literal in ["1.0", "1e3", "1.5", "20.0", "-5", "-100000000000000000000000000000", "true"] {
            try writeRawReceipt(pack, filesJSON: try layoutFilesJSON(configSizeLiteral: literal))
            let receipt = try XCTUnwrap(PackReceipt.load(for: pack.directory))
            XCTAssertEqual(
                PackReceiptBinding.verify(directory: pack.directory, receipt: receipt.object),
                .uncheckable("its entry for 'config.json' has no integer size"), literal)
        }
    }

    // MARK: an unlistable directory is described with strerror, on one line

    func testUnreadableListedSubdirectoryReportsPosixStrerrorOnOneLine() throws {
        try XCTSkipIf(geteuid() == 0, "root ignores directory permissions")
        let pack = try makePack()
        let sub = pack.directory.appendingPathComponent("sub", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        let data = Data("payload".utf8)
        try data.write(to: sub.appendingPathComponent("w.bin"))
        // Search-only (x without r): the listed file still stats and hashes, the walk cannot list.
        defer { chmod(sub.path, 0o755) }
        var files = try layoutFilesJSON(configSizeLiteral: "20")
        files += ", \"sub/w.bin\": {\"size\": \(data.count), \"sha256\": \"\(sha256Hex(data))\"}"
        try writeRawReceipt(pack, filesJSON: files)
        XCTAssertEqual(chmod(sub.path, 0o311), 0)
        assertReceiptRefusal(
            judge(pack.directory), pack: pack, contains: ["sub could not be listed (Permission denied)"])
    }
}
