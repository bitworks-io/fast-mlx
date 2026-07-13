import Foundation
import XCTest

@testable import HarnessCore

final class OutputPathSafetyTests: XCTestCase {
    func testDetectsCaseVariantAliasesAccordingToVolumeSemantics() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let caseSensitive = try XCTUnwrap(
            directory.resourceValues(
                forKeys: [.volumeSupportsCaseSensitiveNamesKey]
            ).volumeSupportsCaseSensitiveNames)
        let upper = directory.appendingPathComponent("Evidence.JSONL").path
        let lower = directory.appendingPathComponent("evidence.jsonl").path

        XCTAssertEqual(
            outputPathsReferToSameFile(upper, lower),
            !caseSensitive)
    }

    func testDetectsExistingHardLinkAliases() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let original = directory.appendingPathComponent("original")
        let alias = directory.appendingPathComponent("alias")
        try Data("evidence".utf8).write(to: original)
        try FileManager.default.linkItem(at: original, to: alias)

        XCTAssertTrue(outputPathsReferToSameFile(original.path, alias.path))
    }

    func testDetectsDanglingFinalSymlinkBeforeItsTargetExists() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let target = directory.appendingPathComponent("future-progress.json")
        let link = directory.appendingPathComponent("evidence-link.jsonl")
        try FileManager.default.createSymbolicLink(
            at: link,
            withDestinationURL: target)

        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
        XCTAssertTrue(outputPathIsSymbolicLink(link.path))
    }

    func testDistinctPathsOnTheSameVolumeRemainDistinct() {
        XCTAssertFalse(
            outputPathsReferToSameFile(
                "/tmp/continuous-soak-evidence.jsonl",
                "/tmp/continuous-soak-progress.json"))
    }
}
