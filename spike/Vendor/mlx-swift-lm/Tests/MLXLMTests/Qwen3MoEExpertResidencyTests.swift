import Foundation
import XCTest

@testable import MLXLLM

final class Qwen3MoEExpertResidencyTests: XCTestCase {
    func testManifestBuildsExactRangesForFirstAndLastExpert() throws {
        let fixture = try Fixture.make(numLayers: 2, numExperts: 3)

        let manifest = try fixture.makeManifest()

        XCTAssertEqual(manifest.expectedMoELayers, [0, 1])
        XCTAssertEqual(manifest.numExperts, 3)
        XCTAssertEqual(manifest.bytesPerExpert, 12)
        XCTAssertEqual(manifest.identity.modelID, "fixture/Qwen3MoE")
        XCTAssertEqual(manifest.identity.resolvedRevision, String(repeating: "a", count: 40))
        XCTAssertEqual(manifest.identity.configSHA256.count, 64)
        XCTAssertEqual(manifest.identity.rangeManifestSHA256.count, 64)
        XCTAssertFalse(manifest.identity.rangeManifestSHA256.contains("/tmp"))

        for expert in [0, 2] {
            let ranges = try manifest.ranges(layer: 0, expert: expert)
            XCTAssertEqual(ranges.count, 3)
            XCTAssertEqual(Set(ranges.map(\.projection)), Set(Qwen3MoEExpertProjection.allCases))
            for range in ranges {
                let tensor = fixture.tensors[range.tensorName]!
                let expectedLower = fixture.dataBase + tensor.offsets.lowerBound + expert * 4
                XCTAssertEqual(range.absoluteRange, expectedLower ..< expectedLower + 4)
                XCTAssertEqual(range.relativeFile, "model.safetensors")
                XCTAssertEqual(range.component, "weight")
                XCTAssertEqual(range.dtype, "U8")
                XCTAssertEqual(range.tensorShape, [3, 2, 2])
                XCTAssertEqual(range.expertShape, [2, 2])
            }
        }
    }

    func testManifestDerivesSparseLayersUsingModelRule() throws {
        let fixture = try Fixture.make(
            numLayers: 6,
            numExperts: 2,
            decoderSparseStep: 2,
            mlpOnlyLayers: [3]
        )

        let manifest = try fixture.makeManifest()

        XCTAssertEqual(manifest.expectedMoELayers, [1, 5])
    }

    func testManifestIdentityBindsDTypeAndShapeEvenWhenRangesAreUnchanged() throws {
        let original = try Fixture.make(numLayers: 1, numExperts: 3)
        var changed = original
        let first = changed.tensors.keys.sorted().first!
        changed.tensors[first]!.dtype = "I8"
        changed.tensors[first]!.shape = [3, 1, 4]

        let originalManifest = try original.makeManifest()
        let changedManifest = try changed.makeManifest()

        XCTAssertNotEqual(
            originalManifest.identity.rangeManifestSHA256,
            changedManifest.identity.rangeManifestSHA256
        )
    }

    func testManifestRejectsUnsupportedOrEmptySnapshots() throws {
        var fixture = try Fixture.make()
        fixture.config["model_type"] = "qwen3_5"
        XCTAssertThrowsError(try fixture.makeManifest()) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .unsupportedModelType("qwen3_5")
            )
        }

        fixture = try Fixture.make()
        XCTAssertThrowsError(
            try Qwen3MoEExpertManifestBuilder.build(
                modelID: fixture.modelID,
                resolvedRevision: fixture.resolvedRevision,
                configData: fixture.configData,
                indexData: fixture.indexData,
                shards: []
            )
        ) {
            XCTAssertEqual($0 as? Qwen3MoEExpertResidencyError, .noSafetensors)
        }
    }

    func testManifestRejectsFloatingPointIntegersAndUnboundedGeometry() throws {
        let floatingConfig = try Fixture.make()
        let validConfigData = try floatingConfig.configData
        let validConfigText = String(decoding: validConfigData, as: UTF8.self)
        let floatingConfigData = Data(
            validConfigText.replacingOccurrences(
                of: "\"num_hidden_layers\":2",
                with: "\"num_hidden_layers\":2.0"
            ).utf8
        )
        XCTAssertNotEqual(floatingConfigData, validConfigData)
        XCTAssertThrowsError(
            try Qwen3MoEExpertManifestBuilder.build(
                modelID: floatingConfig.modelID,
                resolvedRevision: floatingConfig.resolvedRevision,
                configData: floatingConfigData,
                indexData: floatingConfig.indexData,
                shards: floatingConfig.shards
            )
        ) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .invalidConfiguration("num_hidden_layers")
            )
        }

        let floatingShape = try Fixture.make()
        let validShard = floatingShape.shards[0]
        let validHeader = validShard.prefixData.subdata(in: 8 ..< floatingShape.dataBase)
        let validHeaderText = String(decoding: validHeader, as: UTF8.self)
        let floatingHeader = Data(
            validHeaderText.replacingOccurrences(
                of: "\"shape\":[3,2,2]",
                with: "\"shape\":[3.0,2,2]"
            ).utf8
        )
        XCTAssertNotEqual(floatingHeader, validHeader)
        let bodyBytes = validShard.prefixData.count - floatingShape.dataBase
        let floatingShardData = safetensorData(header: floatingHeader, bodyBytes: bodyBytes)
        let floatingShard = Qwen3MoESnapshotShard(
            relativeFile: validShard.relativeFile,
            identity: .init(
                byteCount: floatingShardData.count,
                contentDigest: validShard.identity.contentDigest
            ),
            prefixData: floatingShardData
        )
        XCTAssertThrowsError(
            try Qwen3MoEExpertManifestBuilder.build(
                modelID: floatingShape.modelID,
                resolvedRevision: floatingShape.resolvedRevision,
                configData: floatingShape.configData,
                indexData: floatingShape.indexData,
                shards: [floatingShard]
            )
        ) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .malformedHeader("model.safetensors")
            )
        }

        var excessiveLayers = try Fixture.make()
        excessiveLayers.config["num_hidden_layers"] = 257
        XCTAssertThrowsError(try excessiveLayers.makeManifest()) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .invalidConfiguration("num_hidden_layers exceeds contract maximum")
            )
        }

        var excessiveExperts = try Fixture.make()
        excessiveExperts.config["num_experts"] = 513
        XCTAssertThrowsError(try excessiveExperts.makeManifest()) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .invalidConfiguration("num_experts exceeds contract maximum")
            )
        }
    }

    func testManifestRejectsPathEscapeAndIndexMismatch() throws {
        var fixture = try Fixture.make()
        fixture.shards[0].relativeFile = "../model.safetensors"
        fixture.weightMap = fixture.weightMap.mapValues { _ in "../model.safetensors" }
        XCTAssertThrowsError(try fixture.makeManifest()) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .invalidRelativePath("../model.safetensors")
            )
        }

        fixture = try Fixture.make()
        let firstName = fixture.weightMap.keys.sorted().first!
        fixture.weightMap[firstName] = "other.safetensors"
        XCTAssertThrowsError(try fixture.makeManifest()) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .indexMismatch(firstName)
            )
        }
    }

    func testManifestRejectsDuplicateTensorNameBeforeJSONCollapsesIt() throws {
        let name = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        let entry = #"{"dtype":"U8","shape":[2,2,2],"data_offsets":[0,8]}"#
        let header = Data("{\"\(name)\":\(entry),\"\(name)\":\(entry)}".utf8)
        let shard = Qwen3MoESnapshotShard(
            relativeFile: "model.safetensors",
            identity: .init(byteCount: 8 + header.count + 8, contentDigest: "duplicate"),
            prefixData: safetensorData(header: header, bodyBytes: 8)
        )
        let fixture = try Fixture.make(numLayers: 1, numExperts: 2)

        XCTAssertThrowsError(
            try Qwen3MoEExpertManifestBuilder.build(
                modelID: fixture.modelID,
                resolvedRevision: fixture.resolvedRevision,
                configData: fixture.configData,
                indexData: fixture.indexData,
                shards: [shard]
            )
        ) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .duplicateTensorName(name)
            )
        }
    }

    func testManifestRejectsDuplicateTensorNameInsideIndex() throws {
        let fixture = try Fixture.make(numLayers: 1, numExperts: 2)
        let name = fixture.weightMap.keys.sorted().first!
        let file = fixture.weightMap[name]!
        let duplicateIndex = Data(
            "{\"weight_map\":{\"\(name)\":\"\(file)\",\"\(name)\":\"\(file)\"}}".utf8
        )

        XCTAssertThrowsError(
            try Qwen3MoEExpertManifestBuilder.build(
                modelID: fixture.modelID,
                resolvedRevision: fixture.resolvedRevision,
                configData: fixture.configData,
                indexData: duplicateIndex,
                shards: fixture.shards
            )
        ) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .duplicateTensorName(name)
            )
        }
    }

    func testManifestRejectsMissingExtraAndNoExpectedMoELayers() throws {
        var missing = try Fixture.make(numLayers: 2, numExperts: 2)
        for name in missing.tensors.keys where name.contains("layers.0") {
            missing.tensors[name] = nil
            missing.weightMap[name] = nil
        }
        XCTAssertThrowsError(try missing.makeManifest()) {
            XCTAssertEqual($0 as? Qwen3MoEExpertResidencyError, .missingMoELayer(0))
        }

        var extra = try Fixture.make(numLayers: 2, numExperts: 2)
        extra.config["num_hidden_layers"] = 1
        XCTAssertThrowsError(try extra.makeManifest()) {
            XCTAssertEqual($0 as? Qwen3MoEExpertResidencyError, .extraMoELayer(1))
        }

        var none = try Fixture.make(numLayers: 1, numExperts: 2)
        none.config["mlp_only_layers"] = [0]
        XCTAssertThrowsError(try none.makeManifest()) {
            XCTAssertEqual($0 as? Qwen3MoEExpertResidencyError, .noExpectedMoELayers)
        }
    }

    func testManifestRejectsShapeDTypeOverlapAndMissingProjection() throws {
        var fixture = try Fixture.make()
        let first = fixture.tensors.keys.sorted().first!
        fixture.tensors[first]!.shape = [4, 1, 3]
        XCTAssertThrowsError(try fixture.makeManifest()) {
            guard case .expertDimensionMismatch(let name, expected: 3, actual: 4) =
                $0 as? Qwen3MoEExpertResidencyError
            else { return XCTFail("unexpected error: \($0)") }
            XCTAssertEqual(name, first)
        }

        fixture = try Fixture.make()
        fixture.tensors[first]!.dtype = "F8_E4M3"
        XCTAssertThrowsError(try fixture.makeManifest()) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .unsupportedDType("F8_E4M3")
            )
        }

        fixture = try Fixture.make()
        let names = fixture.tensors.keys.sorted()
        fixture.tensors[names[1]]!.offsets = fixture.tensors[names[0]]!.offsets
        XCTAssertThrowsError(try fixture.makeManifest()) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .overlappingRanges("model.safetensors")
            )
        }

        fixture = try Fixture.make()
        let missing = fixture.tensors.keys.first { $0.contains("layers.0") && $0.contains("up_proj") }!
        fixture.tensors[missing] = nil
        fixture.weightMap[missing] = nil
        XCTAssertThrowsError(try fixture.makeManifest()) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .missingProjection(layer: 0, projection: .up)
            )
        }
    }

    func testFetchDeduplicatesHitsAndReadsOnlyColdRanges() throws {
        let fixture = try Fixture.make(numLayers: 1, numExperts: 3)
        let manifest = try fixture.makeManifest()
        let reader = TestRangeReader(manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)

        let cold = try residency.fetch(layer: 0, routedExperts: [1, 1], reader: reader)
        XCTAssertEqual(
            cold.metrics,
            .init(
                hits: 0,
                misses: 1,
                bytesRead: 12,
                readCount: 3,
                readNanoseconds: cold.metrics.readNanoseconds,
                evictedExperts: []
            )
        )
        XCTAssertGreaterThan(cold.metrics.readNanoseconds, 0)
        XCTAssertEqual(cold.payloads.keys.map(\.expert), [1])
        XCTAssertEqual(cold.payloads.values.first?.count, 3)
        XCTAssertEqual(reader.reads.count, 3)

        let hit = try residency.fetch(layer: 0, routedExperts: [1], reader: reader)
        XCTAssertEqual(hit.metrics, .init(hits: 1, misses: 0, bytesRead: 0, readCount: 0, evictedExperts: []))
        XCTAssertEqual(reader.reads.count, 3)
        XCTAssertEqual(
            residency.snapshot().counters,
            .init(
                transactions: 2,
                hits: 1,
                misses: 1,
                bytesRead: 12,
                readCount: 3,
                readNanoseconds: residency.snapshot().counters.readNanoseconds,
                evictions: 0
            )
        )
    }

    func testLRUEvictionIsPerLayerDeterministicAndAtomic() throws {
        let fixture = try Fixture.make(numLayers: 2, numExperts: 3)
        let manifest = try fixture.makeManifest()
        let reader = TestRangeReader(manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)

        _ = try residency.fetch(layer: 0, routedExperts: [0, 1], reader: reader)
        _ = try residency.fetch(layer: 0, routedExperts: [1], reader: reader)
        let third = try residency.fetch(layer: 0, routedExperts: [2], reader: reader)

        XCTAssertEqual(third.metrics.evictedExperts, [0])
        XCTAssertEqual(residency.snapshot().residentExpertsByLayer[0], [1, 2])
        XCTAssertEqual(residency.snapshot().residentExpertsByLayer[1], [])
    }

    func testShortReadAndFileIdentityDriftDoNotMutateAndRetryRecovers() throws {
        let fixture = try Fixture.make(numLayers: 1, numExperts: 3)
        let manifest = try fixture.makeManifest()
        let reader = TestRangeReader(manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)
        let before = residency.snapshot()

        reader.shortReadAtCall = 2
        XCTAssertThrowsError(try residency.fetch(layer: 0, routedExperts: [0], reader: reader)) {
            guard case .shortRead(expected: 4, actual: 3) =
                $0 as? Qwen3MoEExpertResidencyError
            else { return XCTFail("unexpected error: \($0)") }
        }
        XCTAssertEqual(residency.snapshot(), before)

        reader.shortReadAtCall = nil
        reader.resetCalls()
        reader.identities["model.safetensors"] = .init(
            byteCount: fixture.shards[0].identity.byteCount,
            contentDigest: "changed"
        )
        XCTAssertThrowsError(try residency.fetch(layer: 0, routedExperts: [0], reader: reader)) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .fileIdentityChanged("model.safetensors")
            )
        }
        XCTAssertEqual(residency.snapshot(), before)

        reader.identities = manifest.shardIdentities
        let recovered = try residency.fetch(layer: 0, routedExperts: [0], reader: reader)
        XCTAssertEqual(recovered.metrics.misses, 1)
        XCTAssertEqual(residency.snapshot().counters.transactions, 1)

        let beforeHitDrift = residency.snapshot()
        reader.identities["model.safetensors"] = .init(
            byteCount: fixture.shards[0].identity.byteCount,
            contentDigest: "changed-again"
        )
        XCTAssertThrowsError(try residency.fetch(layer: 0, routedExperts: [0], reader: reader)) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .fileIdentityChanged("model.safetensors")
            )
        }
        XCTAssertEqual(residency.snapshot(), beforeHitDrift)
    }

    func testCancellationBeforeAndDuringReadIsTransactional() throws {
        let fixture = try Fixture.make(numLayers: 1, numExperts: 3)
        let manifest = try fixture.makeManifest()
        let reader = TestRangeReader(manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)
        let before = residency.snapshot()

        XCTAssertThrowsError(
            try residency.fetch(
                layer: 0,
                routedExperts: [0],
                reader: reader,
                cancellationCheck: { throw CancellationError() }
            )
        ) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertEqual(residency.snapshot(), before)

        reader.cancelAtCall = 2
        XCTAssertThrowsError(try residency.fetch(layer: 0, routedExperts: [0], reader: reader)) {
            XCTAssertTrue($0 is CancellationError)
        }
        XCTAssertEqual(residency.snapshot(), before)

        reader.cancelAtCall = nil
        reader.resetCalls()
        XCTAssertNoThrow(try residency.fetch(layer: 0, routedExperts: [0], reader: reader))
        XCTAssertEqual(residency.snapshot().counters.transactions, 1)
    }

    func testEmptyAndOverCapacityRequestsDoNotChurnState() throws {
        let fixture = try Fixture.make(numLayers: 1, numExperts: 3)
        let manifest = try fixture.makeManifest()
        let reader = TestRangeReader(manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 1)
        let before = residency.snapshot()

        let empty = try residency.fetch(layer: 0, routedExperts: [], reader: reader)
        XCTAssertEqual(empty.metrics, .zero)
        XCTAssertTrue(empty.payloads.isEmpty)
        XCTAssertEqual(residency.snapshot(), before)

        XCTAssertThrowsError(try residency.fetch(layer: 0, routedExperts: [0, 1], reader: reader)) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .requestExceedsCapacity(requested: 2, capacity: 1)
            )
        }
        XCTAssertEqual(residency.snapshot(), before)
        XCTAssertTrue(reader.reads.isEmpty)

        XCTAssertThrowsError(try residency.fetch(layer: 0, routedExperts: [3], reader: reader)) {
            XCTAssertEqual($0 as? Qwen3MoEExpertResidencyError, .invalidExpert(3))
        }
        XCTAssertEqual(residency.snapshot(), before)
        XCTAssertTrue(reader.reads.isEmpty)
    }
}

private final class TestRangeReader: Qwen3MoEExpertRangeReading {
    var identities: [String: Qwen3MoEShardIdentity]
    var reads: [Qwen3MoEExpertRange] = []
    var shortReadAtCall: Int?
    var cancelAtCall: Int?

    init(manifest: Qwen3MoEExpertManifest) {
        identities = manifest.shardIdentities
    }

    func identity(relativeFile: String) throws -> Qwen3MoEShardIdentity {
        identities[relativeFile]!
    }

    func read(_ range: Qwen3MoEExpertRange) throws -> Data {
        reads.append(range)
        if cancelAtCall == reads.count { throw CancellationError() }
        let count = shortReadAtCall == reads.count ? range.byteCount - 1 : range.byteCount
        return Data(repeating: UInt8(range.absoluteRange.lowerBound % 251), count: count)
    }

    func resetCalls() {
        reads = []
    }
}

private struct FixtureTensor {
    var dtype = "U8"
    var shape: [Int]
    var offsets: Range<Int>
}

private struct Fixture {
    let modelID = "fixture/Qwen3MoE"
    let resolvedRevision = String(repeating: "a", count: 40)
    var config: [String: Any]
    var weightMap: [String: String]
    var tensors: [String: FixtureTensor]
    var shards: [Qwen3MoESnapshotShard]
    var dataBase: Int

    var configData: Data {
        get throws { try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]) }
    }

    var indexData: Data {
        get throws {
            try JSONSerialization.data(
                withJSONObject: ["weight_map": weightMap],
                options: [.sortedKeys]
            )
        }
    }

    static func make(
        numLayers: Int = 2,
        numExperts: Int = 3,
        decoderSparseStep: Int = 1,
        mlpOnlyLayers: [Int] = []
    ) throws -> Fixture {
        let sparseLayers = (0 ..< numLayers).filter {
            !mlpOnlyLayers.contains($0) && ($0 + 1) % decoderSparseStep == 0
        }
        var tensors: [String: FixtureTensor] = [:]
        var weightMap: [String: String] = [:]
        var offset = 0
        for layer in sparseLayers {
            for projection in Qwen3MoEExpertProjection.allCases {
                let name = "model.layers.\(layer).mlp.switch_mlp.\(projection.tensorComponent).weight"
                let byteCount = numExperts * 4
                tensors[name] = FixtureTensor(
                    shape: [numExperts, 2, 2],
                    offsets: offset ..< offset + byteCount
                )
                weightMap[name] = "model.safetensors"
                offset += byteCount
            }
        }
        var fixture = Fixture(
            config: [
                "model_type": "qwen3_moe",
                "num_hidden_layers": numLayers,
                "num_experts": numExperts,
                "decoder_sparse_step": decoderSparseStep,
                "mlp_only_layers": mlpOnlyLayers,
            ],
            weightMap: weightMap,
            tensors: tensors,
            shards: [],
            dataBase: 0
        )
        fixture.rebuildShard()
        return fixture
    }

    mutating func rebuildShard() {
        var headerObject: [String: Any] = [:]
        for (name, tensor) in tensors {
            headerObject[name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [tensor.offsets.lowerBound, tensor.offsets.upperBound],
            ]
        }
        let header = try! JSONSerialization.data(withJSONObject: headerObject, options: [.sortedKeys])
        dataBase = 8 + header.count
        let bodyBytes = tensors.values.map(\.offsets.upperBound).max() ?? 0
        let data = safetensorData(header: header, bodyBytes: bodyBytes)
        shards = [
            Qwen3MoESnapshotShard(
                relativeFile: "model.safetensors",
                identity: .init(byteCount: data.count, contentDigest: "fixture-v1"),
                prefixData: data
            )
        ]
    }

    func makeManifest() throws -> Qwen3MoEExpertManifest {
        var current = self
        current.rebuildShard()
        return try Qwen3MoEExpertManifestBuilder.build(
            modelID: current.modelID,
            resolvedRevision: current.resolvedRevision,
            configData: current.configData,
            indexData: current.indexData,
            shards: current.shards
        )
    }
}

private func safetensorData(header: Data, bodyBytes: Int) -> Data {
    var length = UInt64(header.count).littleEndian
    var data = withUnsafeBytes(of: &length) { Data($0) }
    data.append(header)
    data.append(Data(repeating: 0x5a, count: bodyBytes))
    return data
}
