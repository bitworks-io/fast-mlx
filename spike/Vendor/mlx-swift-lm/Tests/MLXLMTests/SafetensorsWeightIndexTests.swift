// Copyright © 2026 Apple Inc.

import Foundation
@testable import MLXLMCommon
import XCTest

final class SafetensorsWeightIndexTests: XCTestCase {

    // MARK: - 1. Decodes weight_map, ignores metadata

    /// Pins: a real index file also carries a `metadata` object alongside
    /// `weight_map`; decoding must accept that shape and only use `weight_map`.
    func testDecodesWeightMapAndIgnoresMetadata() throws {
        let json: [String: Any] = [
            "metadata": ["total_size": 123_456],
            "weight_map": [
                "model.layers.0.weight": "shard-1.safetensors",
                "model.layers.1.weight": "shard-1.safetensors",
                "model.layers.2.weight": "shard-2.safetensors",
                "model.layers.3.weight": "shard-2.safetensors",
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: json, options: [])

        let index = try SafetensorsWeightIndex(decoding: data)

        XCTAssertEqual(
            index.weightKeys,
            [
                "model.layers.0.weight", "model.layers.1.weight",
                "model.layers.2.weight", "model.layers.3.weight",
            ])
        XCTAssertEqual(index.shardFileNames, ["shard-1.safetensors", "shard-2.safetensors"])
    }

    // MARK: - 2. Whole-artifact key set with no tensor file present at all (load-bearing)

    /// Pins: the whole point of this type is that `weightKeys` is obtainable
    /// from the index alone, without opening or even having a tensor file on
    /// disk. If some code path ever needed to peek at a shard to build the
    /// key set, this test would fail because no shard file exists here.
    func testBuildsWholeArtifactKeySetWithNoTensorFilePresent() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let weightMap = [
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.10.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        ]
        try Self.writeIndex(weightMap: weightMap, to: directory)

        // Explicitly confirm no shard file exists on disk before loading.
        for shardName in Set(weightMap.values) {
            XCTAssertFalse(
                FileManager.default.fileExists(
                    atPath: directory.appendingPathComponent(shardName).path(percentEncoded: false)),
                "expected no shard file on disk for \(shardName)")
        }

        let index = try XCTUnwrap(try SafetensorsWeightIndex.load(modelDirectory: directory))

        XCTAssertEqual(index.weightKeys, Set(weightMap.keys))
    }

    // MARK: - 3. shardFileName(forKey:) and keys(inShard:) partition the key set

    func testShardFileNameForKeyAndKeysInShardPartitionTheKeySet() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let weightMap = [
            "a": "shard-b.safetensors",
            "c": "shard-a.safetensors",
            "b": "shard-b.safetensors",
            "d": "shard-a.safetensors",
            "e": "shard-c.safetensors",
        ]
        try Self.writeIndex(weightMap: weightMap, to: directory)
        let index = try XCTUnwrap(try SafetensorsWeightIndex.load(modelDirectory: directory))

        for (key, shard) in weightMap {
            XCTAssertEqual(index.shardFileName(forKey: key), shard)
        }
        XCTAssertNil(index.shardFileName(forKey: "not-a-key"))

        // `weightMap`'s keys are already unique dictionary keys, so a bare
        // cross-shard disjointness check on `keys(inShard:)` output is
        // vacuously true no matter what `keys(inShard:)` actually returns —
        // it cannot fail even for a buggy implementation that returns an
        // empty list, a superset, or keys routed to the wrong shard. Assert
        // each shard's exact expected contents instead, which can actually
        // fail on those bugs.
        XCTAssertEqual(index.keys(inShard: "shard-a.safetensors"), ["c", "d"])
        XCTAssertEqual(index.keys(inShard: "shard-b.safetensors"), ["a", "b"])
        XCTAssertEqual(index.keys(inShard: "shard-c.safetensors"), ["e"])

        var unionOfKeys = Set<String>()
        for shardFileName in index.shardFileNames {
            let keysInShard = index.keys(inShard: shardFileName)
            XCTAssertEqual(keysInShard, keysInShard.sorted(), "keys(inShard:) must be sorted")
            unionOfKeys.formUnion(keysInShard)
        }
        XCTAssertEqual(unionOfKeys, index.weightKeys)
    }

    // MARK: - 4. shardURLs throws invalidConfiguration naming the missing shard

    func testShardURLsThrowsInvalidConfigurationNamingTheMissingShard() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let weightMap = [
            "a": "present.safetensors",
            "b": "missing.safetensors",
        ]
        try Self.writeIndex(weightMap: weightMap, to: directory)
        try Self.writeEmptyFile(named: "present.safetensors", in: directory)
        // "missing.safetensors" is deliberately never written.

        let index = try XCTUnwrap(try SafetensorsWeightIndex.load(modelDirectory: directory))

        XCTAssertThrowsError(try index.shardURLs(in: directory)) { error in
            guard case ModelFactoryError.invalidConfiguration(let message) = error else {
                return XCTFail("expected ModelFactoryError.invalidConfiguration, got \(error)")
            }
            XCTAssertTrue(
                message.contains("missing.safetensors"),
                "expected error message to name the missing shard, got: \(message)")
        }
    }

    // MARK: - 5. shardURLs excludes an index-external safetensors sidecar

    /// Pins the hazard `Load.swift`'s doc comment describes: repositories can
    /// ship additional index-external safetensors (drafter/vision sidecars,
    /// calibration extras) whose foreign keys must never reach the model.
    func testShardURLsExcludesIndexExternalSafetensorsSidecar() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let weightMap = [
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        ]
        try Self.writeIndex(weightMap: weightMap, to: directory)
        try Self.writeEmptyFile(named: "model-00001-of-00002.safetensors", in: directory)
        try Self.writeEmptyFile(named: "model-00002-of-00002.safetensors", in: directory)
        // Unmapped sidecar, not referenced by the index at all.
        try Self.writeEmptyFile(named: "drafter.safetensors", in: directory)

        let index = try XCTUnwrap(try SafetensorsWeightIndex.load(modelDirectory: directory))
        let urls = try index.shardURLs(in: directory)

        XCTAssertEqual(
            Set(urls.map { $0.lastPathComponent }),
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    }

    // MARK: - 6. load returns nil when there is no index file

    func testLoadReturnsNilWhenNoIndexFile() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try Self.writeEmptyFile(named: "model.safetensors", in: directory)

        XCTAssertNil(try SafetensorsWeightIndex.load(modelDirectory: directory))
    }

    // MARK: - 7. load throws when the index file is malformed

    func testLoadThrowsWhenIndexIsMalformed() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let indexURL = directory.appendingPathComponent(SafetensorsWeightIndex.indexFileName)
        try Data("{".utf8).write(to: indexURL)

        XCTAssertThrowsError(try SafetensorsWeightIndex.load(modelDirectory: directory))
    }

    // MARK: - 8. load throws when weight_map key is absent

    func testLoadThrowsWhenWeightMapKeyIsAbsent() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let json: [String: Any] = ["metadata": ["total_size": 1]]
        let data = try JSONSerialization.data(withJSONObject: json, options: [])
        let indexURL = directory.appendingPathComponent(SafetensorsWeightIndex.indexFileName)
        try data.write(to: indexURL)

        XCTAssertThrowsError(try SafetensorsWeightIndex.load(modelDirectory: directory))
    }

    // MARK: - 9-11. modelWeightFileURLs(modelDirectory:) — the production model-loading path
    //
    // `modelWeightFileURLs` (Load.swift) had no direct test at all, before or
    // after this change, despite being on the production model-loading path
    // that every `loadWeights` call goes through. It is internal, but
    // reachable via this file's existing `@testable import MLXLMCommon`.

    /// Index present: `modelWeightFileURLs` must return exactly the index's
    /// mapped shard files, excluding an index-external sidecar `.safetensors`
    /// file in the same directory (the same hazard
    /// `testShardURLsExcludesIndexExternalSafetensorsSidecar` pins for
    /// `SafetensorsWeightIndex.shardURLs(in:)` directly — this pins it for the
    /// production entry point that actually wraps that call).
    func testModelWeightFileURLsWithIndexReturnsOnlyMappedShardsAndExcludesSidecar() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let weightMap = [
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        ]
        try Self.writeIndex(weightMap: weightMap, to: directory)
        try Self.writeEmptyFile(named: "model-00001-of-00002.safetensors", in: directory)
        try Self.writeEmptyFile(named: "model-00002-of-00002.safetensors", in: directory)
        // Index-external sidecar: never referenced by weight_map.
        try Self.writeEmptyFile(named: "drafter.safetensors", in: directory)

        let urls = try modelWeightFileURLs(modelDirectory: directory)

        XCTAssertEqual(
            Set(urls.map { $0.lastPathComponent }),
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    }

    /// Index absent: `modelWeightFileURLs` must fall back to enumerating
    /// `*.safetensors` RECURSIVELY. Pinned with a shard placed in a
    /// subdirectory, which a non-recursive (top-level-only) enumeration would
    /// miss.
    func testModelWeightFileURLsWithoutIndexFallsBackToRecursiveEnumeration() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try Self.writeEmptyFile(named: "model-00001-of-00001.safetensors", in: directory)

        let subdirectory = directory.appendingPathComponent("nested", isDirectory: true)
        try FileManager.default.createDirectory(
            at: subdirectory, withIntermediateDirectories: true)
        try Self.writeEmptyFile(named: "extra.safetensors", in: subdirectory)

        // Confirm there is genuinely no index for this fallback path.
        XCTAssertNil(try SafetensorsWeightIndex.load(modelDirectory: directory))

        let urls = try modelWeightFileURLs(modelDirectory: directory)

        XCTAssertEqual(
            Set(urls.map { $0.lastPathComponent }),
            ["model-00001-of-00001.safetensors", "extra.safetensors"],
            "expected the fallback to recurse into subdirectories, not just enumerate the top level"
        )
    }

    /// Index present but mapping a shard file that does not exist on disk:
    /// `modelWeightFileURLs` must throw, naming the missing shard, rather
    /// than silently loading a partial weight set.
    func testModelWeightFileURLsWithIndexThrowsNamingMissingShard() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let weightMap = [
            "a": "present.safetensors",
            "b": "missing.safetensors",
        ]
        try Self.writeIndex(weightMap: weightMap, to: directory)
        try Self.writeEmptyFile(named: "present.safetensors", in: directory)
        // "missing.safetensors" is deliberately never written.

        XCTAssertThrowsError(try modelWeightFileURLs(modelDirectory: directory)) { error in
            guard case ModelFactoryError.invalidConfiguration(let message) = error else {
                return XCTFail("expected ModelFactoryError.invalidConfiguration, got \(error)")
            }
            XCTAssertTrue(
                message.contains("missing.safetensors"),
                "expected error message to name the missing shard, got: \(message)")
        }
    }

    // MARK: - 12-17. weight_map shard-name validation (B5: untrusted index, path traversal)
    //
    // `weight_map` values come from a downloaded, untrusted
    // `model.safetensors.index.json`. Before this change,
    // `shardURLs(in:)` built a URL straight from an unvalidated value via
    // `appendingPathComponent`, so a shard name like `"../../etc/passwd"`
    // produced a path outside the model directory that `fileExists` would
    // happily resolve. These tests pin that a hostile shard name is
    // rejected, and that rejection happens as early as decode time (not
    // deferred to `shardURLs`), because `init(decoding:)` /
    // `SafetensorsWeightIndex.load(modelDirectory:)` is the seam every
    // caller — including whole-artifact namespace resolution via
    // `weightKeys`, not only `shardURLs` — goes through.

    /// A shard name containing `../` is rejected, naming the offending value.
    func testInitDecodingRejectsShardNameContainingParentDirectoryTraversal() throws {
        let json: [String: Any] = [
            "weight_map": [
                "a": "../../etc/passwd"
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: json, options: [])

        XCTAssertThrowsError(try SafetensorsWeightIndex(decoding: data)) { error in
            guard case ModelFactoryError.invalidConfiguration(let message) = error else {
                return XCTFail("expected ModelFactoryError.invalidConfiguration, got \(error)")
            }
            XCTAssertTrue(
                message.contains("../../etc/passwd"),
                "expected error message to name the offending shard name, got: \(message)")
        }
    }

    /// A shard name containing a nested subdirectory is rejected.
    func testInitDecodingRejectsShardNameContainingNestedSubdirectory() throws {
        let json: [String: Any] = [
            "weight_map": [
                "a": "sub/dir/x.safetensors"
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: json, options: [])

        XCTAssertThrowsError(try SafetensorsWeightIndex(decoding: data)) { error in
            guard case ModelFactoryError.invalidConfiguration(let message) = error else {
                return XCTFail("expected ModelFactoryError.invalidConfiguration, got \(error)")
            }
            XCTAssertTrue(
                message.contains("sub/dir/x.safetensors"),
                "expected error message to name the offending shard name, got: \(message)")
        }
    }

    /// An absolute-path shard name is rejected.
    func testInitDecodingRejectsAbsolutePathShardName() throws {
        let json: [String: Any] = [
            "weight_map": [
                "a": "/etc/passwd"
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: json, options: [])

        XCTAssertThrowsError(try SafetensorsWeightIndex(decoding: data)) { error in
            guard case ModelFactoryError.invalidConfiguration(let message) = error else {
                return XCTFail("expected ModelFactoryError.invalidConfiguration, got \(error)")
            }
            XCTAssertTrue(
                message.contains("/etc/passwd"),
                "expected error message to name the offending shard name, got: \(message)")
        }
    }

    /// An empty-string shard name is rejected.
    func testInitDecodingRejectsEmptyShardName() throws {
        let json: [String: Any] = [
            "weight_map": [
                "a": ""
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: json, options: [])

        XCTAssertThrowsError(try SafetensorsWeightIndex(decoding: data)) { error in
            guard case ModelFactoryError.invalidConfiguration(let message) = error else {
                return XCTFail("expected ModelFactoryError.invalidConfiguration, got \(error)")
            }
            XCTAssertTrue(
                message.contains("empty"),
                "expected error message to describe an empty shard name, got: \(message)")
        }
    }

    /// A `..` component that is not a leading prefix is still rejected —
    /// the check must not be satisfied merely by scanning for a `../` prefix.
    func testInitDecodingRejectsShardNameWithNonPrefixParentDirectoryComponent() throws {
        let json: [String: Any] = [
            "weight_map": [
                "a": "a/../../b.safetensors"
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: json, options: [])

        XCTAssertThrowsError(try SafetensorsWeightIndex(decoding: data)) { error in
            guard case ModelFactoryError.invalidConfiguration(let message) = error else {
                return XCTFail("expected ModelFactoryError.invalidConfiguration, got \(error)")
            }
            XCTAssertTrue(
                message.contains("a/../../b.safetensors"),
                "expected error message to name the offending shard name, got: \(message)")
        }
    }

    /// Decode-time rejection: a hostile index throws from `init(decoding:)`
    /// itself (and therefore from `load(modelDirectory:)`), not only from a
    /// later `shardURLs(in:)` call. This is what keeps `weightKeys` /
    /// `keys(inShard:)` / `shardFileName(forKey:)` safe too, since a
    /// `SafetensorsWeightIndex` value can never exist with an unvalidated
    /// shard name in its `weightMap`.
    func testLoadThrowsAtDecodeTimeForHostileIndexBeforeShardURLsIsEverCalled() throws {
        let directory = try Self.makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let weightMap = [
            "a": "../../etc/passwd"
        ]
        try Self.writeIndex(weightMap: weightMap, to: directory)

        XCTAssertThrowsError(try SafetensorsWeightIndex.load(modelDirectory: directory)) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                return XCTFail("expected ModelFactoryError.invalidConfiguration, got \(error)")
            }
        }
    }

    // MARK: - Test fixtures

    private static func makeTempDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(
            path: "safetensors-weight-index-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func writeIndex(weightMap: [String: String], to directory: URL) throws {
        let json: [String: Any] = ["metadata": ["total_size": 0], "weight_map": weightMap]
        let data = try JSONSerialization.data(withJSONObject: json, options: [])
        let indexURL = directory.appendingPathComponent(indexFileName)
        try data.write(to: indexURL)
    }

    private static var indexFileName: String { SafetensorsWeightIndex.indexFileName }

    private static func writeEmptyFile(named name: String, in directory: URL) throws {
        let url = directory.appendingPathComponent(name)
        try Data().write(to: url)
    }
}
