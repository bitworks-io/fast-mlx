import Foundation
import HarnessCore
import XCTest
@testable import fastmlx_harness

final class ProvenanceCLITests: XCTestCase {
    func testCheckpointManifestIsInvariantToSafeShardSymlinkMaterialization() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("target.safetensors")
        let regular = root.appendingPathComponent("regular", isDirectory: true)
        let alias = root.appendingPathComponent("alias", isDirectory: true)
        try FileManager.default.createDirectory(at: regular, withIntermediateDirectories: false)
        try FileManager.default.createDirectory(at: alias, withIntermediateDirectories: false)
        for model in [regular, alias] {
            try Data("{\"model_type\":\"qwen3_5\"}".utf8).write(
                to: model.appendingPathComponent("config.json"))
        }
        let shard = Data([9, 8, 7, 6])
        try shard.write(to: target)
        try shard.write(
            to: regular.appendingPathComponent("model-00001-of-00001.safetensors"))
        try FileManager.default.createSymbolicLink(
            at: alias.appendingPathComponent("model-00001-of-00001.safetensors"),
            withDestinationURL: target)

        XCTAssertEqual(
            try ProvenanceCLI.checkpointManifestHash(at: regular.path),
            try ProvenanceCLI.checkpointManifestHash(at: alias.path))
    }

    func testCheckpointManifestRejectsNonRegularShardAlias() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("model", isDirectory: true)
        let directoryTarget = root.appendingPathComponent("target", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: false)
        try FileManager.default.createDirectory(
            at: directoryTarget,
            withIntermediateDirectories: false)
        try Data("{}".utf8).write(to: model.appendingPathComponent("config.json"))
        let logicalShard = model.appendingPathComponent("model.safetensors")
        try FileManager.default.createSymbolicLink(
            at: logicalShard,
            withDestinationURL: directoryTarget)

        XCTAssertThrowsError(
            try ProvenanceCLI.checkpointManifestHash(at: model.path)
        ) { error in
            guard case let ProvenanceCLI.EvidenceIdentityError
                .invalidCheckpointWeight(path) = error,
                URL(fileURLWithPath: path).lastPathComponent
                    == logicalShard.lastPathComponent
            else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "provenance-cli-\(UUID().uuidString)",
            isDirectory: true)
        try FileManager.default.createDirectory(
            at: url,
            withIntermediateDirectories: false)
        return url
    }
}
