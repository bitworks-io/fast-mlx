import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import MLXNN
import XCTest

final class Qwen3MoEPagedSwitchTests: XCTestCase {
    func testPagedSwitchMatchesEagerSwitchGLUForDuplicateUnsortedBatch1Experts() throws {
        let inputDims = 4
        let hiddenDims = 3
        let numExperts = 4
        let layer = 0
        let weights = TestSwitchWeights(
            numExperts: numExperts,
            inputDims: inputDims,
            hiddenDims: hiddenDims
        )
        let fixture = try PagedSwitchFixture(layer: layer, weights: weights, dtype: .float16)
        let manifest = try fixture.makeManifest()
        let reader = PagedSwitchReader(shard: fixture.shard, manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)

        let x = MLXArray([Float](arrayLiteral: 0.25, -0.5, 0.75, 1.0)).reshaped(1, 1, inputDims)
        let routedExperts = [3, 1, 3]
        let indices = MLXArray(routedExperts.map(Int32.init)).reshaped(1, 1, routedExperts.count)

        let eager = SwitchGLU(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            bias: false
        )
        try eager.update(parameters: ModuleParameters.unflattened(weights.moduleParameters(dtype: .float16)), verify: [])
        let expected = eager(x, indices)

        let fetched = try residency.fetchPagedSwitch(
            layer: layer,
            routedExperts: routedExperts,
            reader: reader,
            inputDims: inputDims,
            hiddenDims: hiddenDims
        )
        let actual = try fetched.switchGLU(x, indices)

        eval(expected, actual)
        XCTAssertTrue(allClose(actual, expected, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        XCTAssertEqual(fetched.page.globalExpertIDs, [1, 3])
        XCTAssertEqual(
            fetched.metrics,
            .init(
                hits: 0,
                misses: 2,
                bytesRead: 144,
                readCount: 6,
                readNanoseconds: fetched.metrics.readNanoseconds,
                evictedExperts: []
            )
        )
        XCTAssertGreaterThan(fetched.metrics.readNanoseconds, 0)
        XCTAssertEqual(reader.reads.map(\.address.expert), [1, 1, 1, 3, 3, 3])
        XCTAssertEqual(
            residency.snapshot().counters,
            .init(
                transactions: 1,
                hits: 0,
                misses: 2,
                bytesRead: 144,
                readCount: 6,
                readNanoseconds: residency.snapshot().counters.readNanoseconds,
                evictions: 0
            )
        )
    }

    func testMaterializedFetchRejectsUnsupportedMetadataTransactionallyAndRetryRecovers() throws {
        let weights = TestSwitchWeights(numExperts: 3, inputDims: 2, hiddenDims: 2)
        let fixture = try PagedSwitchFixture(layer: 0, weights: weights, dtype: .float32)
        let manifest = try fixture.makeManifest()
        let reader = PagedSwitchReader(shard: fixture.shard, manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)
        let before = residency.snapshot()

        XCTAssertThrowsError(
            try residency.fetchMaterializedPage(layer: 0, routedExperts: [1], reader: reader)
        ) {
            XCTAssertEqual($0 as? Qwen3MoEExpertResidencyError, .unsupportedDType("F32"))
        }
        XCTAssertEqual(residency.snapshot(), before)

        let recoveredFixture = try PagedSwitchFixture(
            layer: 0,
            weights: weights,
            dtype: .float16
        )
        let recoveredManifest = try recoveredFixture.makeManifest()
        let recoveredReader = PagedSwitchReader(
            shard: recoveredFixture.shard,
            manifest: recoveredManifest
        )
        let recoveredResidency = try Qwen3MoEExpertResidency(
            manifest: recoveredManifest,
            capacityPerLayer: 2
        )
        let recovered = try recoveredResidency.fetchMaterializedPage(
            layer: 0,
            routedExperts: [1],
            reader: recoveredReader
        )
        XCTAssertEqual(recovered.metrics.misses, 1)
        XCTAssertEqual(recoveredResidency.snapshot().counters.transactions, 1)
    }

    func testBF16PagedSwitchMaterializesSelectedExpertsAndMatchesEager() throws {
        let inputDims = 4
        let hiddenDims = 3
        let numExperts = 4
        let weights = TestSwitchWeights(
            numExperts: numExperts,
            inputDims: inputDims,
            hiddenDims: hiddenDims
        )
        let fixture = try PagedSwitchFixture(layer: 0, weights: weights, dtype: .bfloat16)
        let manifest = try fixture.makeManifest()
        let reader = PagedSwitchReader(shard: fixture.shard, manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)
        let routedExperts = [3, 1, 3]
        let indices = MLXArray(routedExperts.map(Int32.init)).reshaped(1, 1, routedExperts.count)
        let x = MLXArray([Float](arrayLiteral: 0.25, -0.5, 0.75, 1.0)).reshaped(1, 1, inputDims)

        let eager = SwitchGLU(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            bias: false
        )
        try eager.update(
            parameters: ModuleParameters.unflattened(weights.moduleParameters(dtype: .bfloat16)),
            verify: []
        )
        let expected = eager(x, indices)

        let fetched = try residency.fetchPagedSwitch(
            layer: 0,
            routedExperts: routedExperts,
            reader: reader,
            inputDims: inputDims,
            hiddenDims: hiddenDims
        )
        let actual = try fetched.switchGLU(x, indices)

        eval(expected, actual)
        XCTAssertTrue(allClose(actual, expected, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        XCTAssertEqual(fetched.page.globalExpertIDs, [1, 3])
        XCTAssertEqual(try fetched.page.array(projection: .gate).dtype, .bfloat16)
        XCTAssertEqual(try fetched.page.array(projection: .gate).shape, [2, hiddenDims, inputDims])
        XCTAssertEqual(try fetched.page.array(projection: .down).shape, [2, inputDims, hiddenDims])
        XCTAssertEqual(fetched.metrics.bytesRead, manifest.bytesPerExpert * 2)
        XCTAssertEqual(reader.reads.map(\.address.expert), [1, 1, 1, 3, 3, 3])
    }

    func testU8WeightOnlyMaterializedPageAcceptsSortedExpertSlices() throws {
        let weights = TestSwitchWeights(numExperts: 4, inputDims: 2, hiddenDims: 3)
        var fixture = try PagedSwitchFixture(layer: 0, weights: weights, dtype: .uint8)
        for (projectionIndex, projection) in Qwen3MoEExpertProjection.allCases.enumerated() {
            let name = fixture.tensorName(projection)
            let elementCount = fixture.tensors[name]!.shape.reduce(1, *)
            fixture.tensors[name]!.data = Data(
                (0 ..< elementCount).map { UInt8((projectionIndex + 1) * 32 + $0) }
            )
        }
        fixture.rebuildShard()
        let manifest = try fixture.makeManifest()
        let reader = PagedSwitchReader(shard: fixture.shard, manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)

        let fetched = try residency.fetchMaterializedPage(
            layer: 0,
            routedExperts: [3, 1, 3],
            reader: reader
        )

        XCTAssertEqual(fetched.page.globalExpertIDs, [1, 3])
        XCTAssertEqual(fetched.metrics.bytesRead, manifest.bytesPerExpert * 2)
        XCTAssertEqual(fetched.metrics.readCount, 6)
        XCTAssertEqual(
            reader.reads.map { "\($0.address.expert):\($0.projection.tensorComponent)" },
            ["1:gate_proj", "1:up_proj", "1:down_proj", "3:gate_proj", "3:up_proj", "3:down_proj"]
        )
        for (projectionIndex, projection) in Qwen3MoEExpertProjection.allCases.enumerated() {
            let array = try fetched.page.array(projection: projection)
            let expertShape = fixture.tensors[fixture.tensorName(projection)]!.shape.dropFirst()
            let sliceCount = expertShape.reduce(1, *)
            let tensorBytes = (0 ..< fixture.tensors[fixture.tensorName(projection)]!.shape.reduce(1, *))
                .map { UInt8((projectionIndex + 1) * 32 + $0) }
            let expectedBytes = Array(tensorBytes[sliceCount ..< sliceCount * 2])
                + Array(tensorBytes[sliceCount * 3 ..< sliceCount * 4])

            eval(array)
            XCTAssertEqual(array.dtype, .uint8)
            XCTAssertEqual(array.shape, [2] + Array(expertShape))
            XCTAssertEqual(array.asArray(UInt8.self), expectedBytes)
        }
    }

    func testMaterializedFetchRejectsUnsupportedComponentAndShapeWithoutStateMutation() throws {
        let weights = TestSwitchWeights(numExperts: 3, inputDims: 2, hiddenDims: 2)
        var componentFixture = try PagedSwitchFixture(layer: 0, weights: weights, dtype: .float16)
        componentFixture.addUnsupportedComponent(named: "zeros")
        let componentManifest = try componentFixture.makeManifest()
        let componentReader = PagedSwitchReader(
            shard: componentFixture.shard,
            manifest: componentManifest
        )
        let componentResidency = try Qwen3MoEExpertResidency(
            manifest: componentManifest,
            capacityPerLayer: 2
        )
        let beforeComponent = componentResidency.snapshot()
        XCTAssertThrowsError(
            try componentResidency.fetchMaterializedPage(layer: 0, routedExperts: [0], reader: componentReader)
        ) {
            guard case .unsupportedComponent(let tensor, let component) =
                $0 as? Qwen3MoEExpertResidencyError
            else { return XCTFail("unexpected error: \($0)") }
            XCTAssertTrue(tensor.hasSuffix(".zeros"))
            XCTAssertEqual(component, "zeros")
        }
        XCTAssertEqual(componentResidency.snapshot(), beforeComponent)

        var shapeFixture = try PagedSwitchFixture(layer: 0, weights: weights, dtype: .float16)
        shapeFixture.tensors[shapeFixture.tensorName(.down)]!.shape = [3, 1, 4]
        shapeFixture.rebuildShard()
        let shapeManifest = try shapeFixture.makeManifest()
        let shapeReader = PagedSwitchReader(shard: shapeFixture.shard, manifest: shapeManifest)
        let shapeResidency = try Qwen3MoEExpertResidency(manifest: shapeManifest, capacityPerLayer: 2)
        let beforeShape = shapeResidency.snapshot()
        XCTAssertThrowsError(
            try shapeResidency.fetchMaterializedPage(layer: 0, routedExperts: [0], reader: shapeReader)
        ) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .invalidShape(shapeFixture.tensorName(.down))
            )
        }
        XCTAssertEqual(shapeResidency.snapshot(), beforeShape)
    }

    func testAffineQuantizedPagedSwitchMatchesEagerForSelectedExperts() throws {
        let inputDims = 32
        let hiddenDims = 32
        let numExperts = 4
        let groupSize = 32
        let bits = 4
        let weights = TestSwitchWeights(
            numExperts: numExperts,
            inputDims: inputDims,
            hiddenDims: hiddenDims
        )
        let parameters = weights.quantizedModuleParameters(groupSize: groupSize, bits: bits)
        var fixture = try PagedSwitchFixture(
            layer: 0,
            weights: weights,
            dtype: .float16
        )
        fixture.replaceTensors(with: parameters)
        let manifest = try fixture.makeManifest()
        let reader = PagedSwitchReader(shard: fixture.shard, manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)
        let routedExperts = [2, 0, 2]
        let indices = MLXArray(routedExperts.map(Int32.init)).reshaped(1, 1, routedExperts.count)
        let x = MLXArray((0 ..< inputDims).map { Float($0) * 0.125 - 0.25 })
            .reshaped(1, 1, inputDims)

        let eager = SwitchGLU(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            bias: false
        )
        quantize(model: eager, groupSize: groupSize, bits: bits, mode: .affine)
        try eager.update(parameters: ModuleParameters.unflattened(parameters), verify: [])
        let expected = eager(x, indices)

        let fetched = try residency.fetchPagedSwitch(
            layer: 0,
            routedExperts: routedExperts,
            reader: reader,
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            quantization: .init(groupSize: groupSize, bits: bits, mode: .affine)
        )
        let actual = try fetched.switchGLU(x, indices)

        eval(expected, actual)
        XCTAssertTrue(allClose(actual, expected, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        XCTAssertEqual(fetched.page.globalExpertIDs, [0, 2])
        XCTAssertEqual(fetched.metrics.bytesRead, manifest.bytesPerExpert * 2)
        XCTAssertEqual(fetched.metrics.readCount, 18)
    }

    func testPagedFetchFailuresStayAtomicAndValidRetryPreservesLRUCounters() throws {
        let weights = TestSwitchWeights(numExperts: 3, inputDims: 2, hiddenDims: 2)
        let fixture = try PagedSwitchFixture(layer: 0, weights: weights, dtype: .float16)
        let manifest = try fixture.makeManifest()
        let reader = PagedSwitchReader(shard: fixture.shard, manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)
        let empty = residency.snapshot()

        reader.shortReadAtCall = 2
        XCTAssertThrowsError(
            try residency.fetchMaterializedPage(layer: 0, routedExperts: [0], reader: reader)
        ) {
            guard case .shortRead = $0 as? Qwen3MoEExpertResidencyError else {
                return XCTFail("unexpected error: \($0)")
            }
        }
        XCTAssertEqual(residency.snapshot(), empty)

        reader.shortReadAtCall = nil
        reader.reads = []
        reader.identities["model.safetensors"] = .init(
            byteCount: fixture.shard.count,
            contentDigest: "drifted"
        )
        XCTAssertThrowsError(
            try residency.fetchMaterializedPage(layer: 0, routedExperts: [0], reader: reader)
        ) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .fileIdentityChanged("model.safetensors")
            )
        }
        XCTAssertEqual(residency.snapshot(), empty)

        reader.identities = manifest.shardIdentities
        XCTAssertThrowsError(
            try residency.fetchMaterializedPage(
                layer: 0,
                routedExperts: [0],
                reader: reader,
                cancellationCheck: { throw CancellationError() }
            )
        ) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertEqual(residency.snapshot(), empty)

        XCTAssertThrowsError(
            try residency.fetchMaterializedPage(layer: 0, routedExperts: [0, 1, 2], reader: reader)
        ) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .requestExceedsCapacity(requested: 3, capacity: 2)
            )
        }
        XCTAssertEqual(residency.snapshot(), empty)

        let first = try residency.fetchMaterializedPage(
            layer: 0,
            routedExperts: [0, 1],
            reader: reader
        )
        XCTAssertEqual(first.metrics.misses, 2)
        let second = try residency.fetchMaterializedPage(
            layer: 0,
            routedExperts: [1, 2],
            reader: reader
        )
        XCTAssertEqual(
            second.metrics,
            .init(
                hits: 1,
                misses: 1,
                bytesRead: manifest.bytesPerExpert,
                readCount: 3,
                readNanoseconds: second.metrics.readNanoseconds,
                evictedExperts: [0]
            )
        )
        XCTAssertEqual(second.page.globalExpertIDs, [1, 2])
        XCTAssertEqual(
            residency.snapshot().counters,
            .init(
                transactions: 2,
                hits: 1,
                misses: 3,
                bytesRead: manifest.bytesPerExpert * 3,
                readCount: 9,
                readNanoseconds: residency.snapshot().counters.readNanoseconds,
                evictions: 1
            )
        )
    }

    func testQuantizedSwitchShapeValidationOccursBeforeResidencyCommit() throws {
        let dimensions = 32
        let weights = TestSwitchWeights(
            numExperts: 3,
            inputDims: dimensions,
            hiddenDims: dimensions
        )
        var fixture = try PagedSwitchFixture(layer: 0, weights: weights, dtype: .float16)
        fixture.replaceTensors(
            with: weights.quantizedModuleParameters(groupSize: 32, bits: 4)
        )
        let scalesName = fixture.tensorName(.down, component: "scales")
        fixture.tensors[scalesName]!.shape = [3, 16, 2]
        fixture.rebuildShard()
        let manifest = try fixture.makeManifest()
        let reader = PagedSwitchReader(shard: fixture.shard, manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)
        let before = residency.snapshot()

        XCTAssertThrowsError(
            try residency.fetchPagedSwitch(
                layer: 0,
                routedExperts: [1],
                reader: reader,
                inputDims: dimensions,
                hiddenDims: dimensions,
                quantization: .init(groupSize: 32, bits: 4, mode: .affine)
            )
        ) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .invalidShape("qwen3_moe paged down_proj")
            )
        }
        XCTAssertEqual(residency.snapshot(), before)
    }

    func testQuantizedSwitchDTypeValidationOccursBeforeResidencyCommit() throws {
        let dimensions = 32
        let weights = TestSwitchWeights(
            numExperts: 3,
            inputDims: dimensions,
            hiddenDims: dimensions
        )
        var fixture = try PagedSwitchFixture(layer: 0, weights: weights, dtype: .float16)
        fixture.replaceTensors(
            with: weights.quantizedModuleParameters(groupSize: 32, bits: 4)
        )
        let weightName = fixture.tensorName(.gate)
        let weightShape = fixture.tensors[weightName]!.shape
        fixture.tensors[weightName]!.dtype = "F16"
        fixture.tensors[weightName]!.data = Data(
            count: weightShape.reduce(1, *) * MemoryLayout<Float16>.stride
        )
        fixture.rebuildShard()
        let manifest = try fixture.makeManifest()
        let reader = PagedSwitchReader(shard: fixture.shard, manifest: manifest)
        let residency = try Qwen3MoEExpertResidency(manifest: manifest, capacityPerLayer: 2)
        let before = residency.snapshot()

        XCTAssertThrowsError(
            try residency.fetchPagedSwitch(
                layer: 0,
                routedExperts: [1],
                reader: reader,
                inputDims: dimensions,
                hiddenDims: dimensions,
                quantization: .init(groupSize: 32, bits: 4, mode: .affine)
            )
        ) {
            XCTAssertEqual(
                $0 as? Qwen3MoEExpertResidencyError,
                .componentDTypeMismatch(
                    tensor: "qwen3_moe paged gate_proj.weight",
                    expected: "U32",
                    actual: "F16"
                )
            )
        }
        XCTAssertEqual(residency.snapshot(), before)
    }

}

private final class PagedSwitchReader: Qwen3MoEExpertRangeReading {
    let shard: Data
    var identities: [String: Qwen3MoEShardIdentity]
    var reads: [Qwen3MoEExpertRange] = []
    var shortReadAtCall: Int?

    init(shard: Data, manifest: Qwen3MoEExpertManifest) {
        self.shard = shard
        self.identities = manifest.shardIdentities
    }

    func identity(relativeFile: String) throws -> Qwen3MoEShardIdentity {
        identities[relativeFile]!
    }

    func read(_ range: Qwen3MoEExpertRange) throws -> Data {
        reads.append(range)
        let data = shard.subdata(in: range.absoluteRange)
        if shortReadAtCall == reads.count {
            return data.dropLast()
        }
        return data
    }
}

private struct PagedSwitchTensor {
    var dtype: String
    var shape: [Int]
    var data: Data
}

private struct PagedSwitchFixture {
    let layer: Int
    let numExperts: Int
    var tensors: [String: PagedSwitchTensor]
    private(set) var weightMap: [String: String]
    private(set) var shard: Data
    private(set) var snapshotShard: Qwen3MoESnapshotShard

    init(layer: Int, weights: TestSwitchWeights, dtype: DType) throws {
        self.layer = layer
        self.numExperts = weights.numExperts
        self.tensors = [:]
        self.weightMap = [:]
        self.shard = Data()
        self.snapshotShard = Qwen3MoESnapshotShard(
            relativeFile: "model.safetensors",
            identity: .init(byteCount: 0, contentDigest: "fixture-v1"),
            prefixData: Data()
        )
        for projection in Qwen3MoEExpertProjection.allCases {
            let array = weights.array(for: projection, dtype: dtype)
            tensors[tensorName(projection)] = PagedSwitchTensor(
                dtype: safetensorsDType(dtype),
                shape: array.shape,
                data: array.asData().data
            )
        }
        rebuildShard()
    }

    func tensorName(_ projection: Qwen3MoEExpertProjection, component: String = "weight") -> String {
        "model.layers.\(layer).mlp.switch_mlp.\(projection.tensorComponent).\(component)"
    }

    mutating func addUnsupportedComponent(named component: String) {
        for projection in Qwen3MoEExpertProjection.allCases {
            let source = tensors[tensorName(projection)]!
            tensors[tensorName(projection, component: component)] = source
        }
        rebuildShard()
    }

    mutating func replaceTensors(with parameters: [String: MLXArray]) {
        tensors = [:]
        for (path, array) in parameters {
            let name = "model.layers.\(layer).mlp.switch_mlp.\(path)"
            tensors[name] = PagedSwitchTensor(
                dtype: safetensorsDType(array.dtype),
                shape: array.shape,
                data: array.asData().data
            )
        }
        rebuildShard()
    }

    mutating func rebuildShard() {
        var body = Data()
        var headerObject: [String: Any] = [:]
        weightMap = [:]
        for name in tensors.keys.sorted() {
            let lower = body.count
            body.append(tensors[name]!.data)
            let upper = body.count
            headerObject[name] = [
                "dtype": tensors[name]!.dtype,
                "shape": tensors[name]!.shape,
                "data_offsets": [lower, upper],
            ]
            weightMap[name] = "model.safetensors"
        }
        let header = try! JSONSerialization.data(withJSONObject: headerObject, options: [.sortedKeys])
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header)
        data.append(body)
        shard = data
        snapshotShard = Qwen3MoESnapshotShard(
            relativeFile: "model.safetensors",
            identity: .init(byteCount: data.count, contentDigest: "fixture-v1"),
            prefixData: data
        )
    }

    func makeManifest() throws -> Qwen3MoEExpertManifest {
        try Qwen3MoEExpertManifestBuilder.build(
            modelID: "fixture/Qwen3MoE",
            resolvedRevision: String(repeating: "b", count: 40),
            configData: try JSONSerialization.data(
                withJSONObject: [
                    "model_type": "qwen3_moe",
                    "num_hidden_layers": 1,
                    "num_experts": numExperts,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                ],
                options: [.sortedKeys]
            ),
            indexData: try JSONSerialization.data(
                withJSONObject: ["weight_map": weightMap],
                options: [.sortedKeys]
            ),
            shards: [snapshotShard]
        )
    }
}

private struct TestSwitchWeights {
    let numExperts: Int
    let inputDims: Int
    let hiddenDims: Int
    let gate: [Float]
    let up: [Float]
    let down: [Float]

    init(numExperts: Int, inputDims: Int, hiddenDims: Int) {
        self.numExperts = numExperts
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.gate = Self.values(count: numExperts * hiddenDims * inputDims, scale: 0.031, offset: -0.37)
        self.up = Self.values(count: numExperts * hiddenDims * inputDims, scale: -0.027, offset: 0.29)
        self.down = Self.values(count: numExperts * inputDims * hiddenDims, scale: 0.019, offset: -0.21)
    }

    func array(for projection: Qwen3MoEExpertProjection, dtype: DType) -> MLXArray {
        switch projection {
        case .gate:
            MLXArray(gate, [numExperts, hiddenDims, inputDims]).asType(dtype)
        case .up:
            MLXArray(up, [numExperts, hiddenDims, inputDims]).asType(dtype)
        case .down:
            MLXArray(down, [numExperts, inputDims, hiddenDims]).asType(dtype)
        }
    }

    func moduleParameters(dtype: DType) -> [String: MLXArray] {
        [
            "gate_proj.weight": array(for: .gate, dtype: dtype),
            "up_proj.weight": array(for: .up, dtype: dtype),
            "down_proj.weight": array(for: .down, dtype: dtype),
        ]
    }

    func quantizedModuleParameters(groupSize: Int, bits: Int) -> [String: MLXArray] {
        var parameters: [String: MLXArray] = [:]
        for projection in Qwen3MoEExpertProjection.allCases {
            let quantized = MLX.quantized(
                array(for: projection, dtype: .float16),
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
            let prefix = "\(projection.tensorComponent)"
            parameters["\(prefix).weight"] = quantized.wq
            parameters["\(prefix).scales"] = quantized.scales
            parameters["\(prefix).biases"] = quantized.biases!
        }
        return parameters
    }

    private static func values(count: Int, scale: Float, offset: Float) -> [Float] {
        (0 ..< count).map { Float($0 % 17) * scale + offset }
    }
}

private func safetensorsDType(_ dtype: DType) -> String {
    switch dtype {
    case .float16: "F16"
    case .float32: "F32"
    case .bfloat16: "BF16"
    case .uint32: "U32"
    case .uint8: "U8"
    default: fatalError("unsupported fixture dtype \(dtype)")
    }
}
