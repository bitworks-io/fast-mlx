import CryptoKit
import CoreFoundation
import Foundation
import MLX

enum Qwen3MoEExpertProjection: String, CaseIterable, Hashable, Sendable {
    case gate
    case up
    case down

    var tensorComponent: String {
        switch self {
        case .gate: "gate_proj"
        case .up: "up_proj"
        case .down: "down_proj"
        }
    }

    fileprivate var sortOrder: Int {
        switch self {
        case .gate: 0
        case .up: 1
        case .down: 2
        }
    }

    fileprivate init?(tensorComponent: String) {
        switch tensorComponent {
        case "gate_proj": self = .gate
        case "up_proj": self = .up
        case "down_proj": self = .down
        default: return nil
        }
    }
}

enum Qwen3MoEExpertResidencyError: Error, Equatable, Sendable {
    case unsupportedModelType(String?)
    case invalidConfiguration(String)
    case noSafetensors
    case noExpectedMoELayers
    case invalidRelativePath(String)
    case malformedHeader(String)
    case duplicateTensorName(String)
    case indexMismatch(String)
    case missingMoELayer(Int)
    case extraMoELayer(Int)
    case missingProjection(layer: Int, projection: Qwen3MoEExpertProjection)
    case componentSetMismatch(layer: Int, projection: Qwen3MoEExpertProjection)
    case unsupportedDType(String)
    case invalidShape(String)
    case expertDimensionMismatch(tensor: String, expected: Int, actual: Int)
    case spanMismatch(String)
    case rangeOverflow(String)
    case rangeOutOfBounds(String)
    case overlappingRanges(String)
    case nonUniformExpertBytes
    case invalidCapacity(Int)
    case unknownLayer(Int)
    case invalidExpert(Int)
    case requestExceedsCapacity(requested: Int, capacity: Int)
    case fileIdentityChanged(String)
    case shortRead(expected: Int, actual: Int)
    case unsupportedComponent(tensor: String, component: String)
    case duplicateComponent(tensor: String, component: String)
    case componentDTypeMismatch(tensor: String, expected: String, actual: String)
    case emptyExpertPage
    case expertNotResident(Int)
}

struct Qwen3MoEShardIdentity: Equatable, Hashable, Sendable {
    let byteCount: Int
    let contentDigest: String
}

struct Qwen3MoESnapshotShard: Equatable, Sendable {
    var relativeFile: String
    var identity: Qwen3MoEShardIdentity
    var prefixData: Data
}

struct Qwen3MoEExpertManifestIdentity: Equatable, Sendable {
    let modelID: String
    let resolvedRevision: String
    let configSHA256: String
    let rangeManifestSHA256: String
}

struct Qwen3MoEExpertAddress: Equatable, Hashable, Sendable {
    let layer: Int
    let expert: Int
}

struct Qwen3MoEExpertRange: Equatable, Hashable, Sendable {
    let address: Qwen3MoEExpertAddress
    let projection: Qwen3MoEExpertProjection
    let component: String
    let tensorName: String
    let dtype: String
    let tensorShape: [Int]
    let relativeFile: String
    let absoluteRange: Range<Int>

    var byteCount: Int { absoluteRange.count }
    var expertShape: [Int] { Array(tensorShape.dropFirst()) }
}

struct Qwen3MoEExpertManifest: Sendable {
    let identity: Qwen3MoEExpertManifestIdentity
    let expectedMoELayers: [Int]
    let numExperts: Int
    let bytesPerExpert: Int
    let shardIdentities: [String: Qwen3MoEShardIdentity]
    let expertTensorNames: Set<String>
    fileprivate let rangesByAddress: [Qwen3MoEExpertAddress: [Qwen3MoEExpertRange]]

    func ranges(layer: Int, expert: Int) throws -> [Qwen3MoEExpertRange] {
        guard expectedMoELayers.contains(layer) else {
            throw Qwen3MoEExpertResidencyError.unknownLayer(layer)
        }
        guard 0 ..< numExperts ~= expert else {
            throw Qwen3MoEExpertResidencyError.invalidExpert(expert)
        }
        let address = Qwen3MoEExpertAddress(layer: layer, expert: expert)
        guard let ranges = rangesByAddress[address] else {
            throw Qwen3MoEExpertResidencyError.invalidExpert(expert)
        }
        return ranges
    }
}

enum Qwen3MoEExpertManifestBuilder {
    static func build(
        modelID: String,
        resolvedRevision: String,
        configData: Data,
        indexData: Data,
        shards: [Qwen3MoESnapshotShard]
    ) throws -> Qwen3MoEExpertManifest {
        guard !modelID.isEmpty else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration("empty model id")
        }
        guard resolvedRevision.count == 40,
            resolvedRevision.allSatisfy({ $0.isHexDigit })
        else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration("revision is not an immutable SHA")
        }
        guard !shards.isEmpty else {
            throw Qwen3MoEExpertResidencyError.noSafetensors
        }

        do {
            try DuplicateJSONKeyValidator.validate(
                configData,
                integerValueKeys: [
                    "num_hidden_layers",
                    "num_experts",
                    "decoder_sparse_step",
                    "mlp_only_layers",
                ]
            )
        } catch let error as DuplicateJSONKeyError {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration("duplicate key: \(error.key)")
        } catch let error as NonIntegerJSONValueError {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration(error.key)
        }
        let configObject = try jsonObject(configData, context: "config")
        let modelType = configObject["model_type"] as? String
        guard modelType == "qwen3_moe" else {
            throw Qwen3MoEExpertResidencyError.unsupportedModelType(modelType)
        }
        let hiddenLayers = try requiredPositiveInt(configObject["num_hidden_layers"], name: "num_hidden_layers")
        let numExperts = try requiredPositiveInt(configObject["num_experts"], name: "num_experts")
        guard hiddenLayers <= 256 else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration(
                "num_hidden_layers exceeds contract maximum"
            )
        }
        guard numExperts <= 512 else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration(
                "num_experts exceeds contract maximum"
            )
        }
        let sparseStep = try requiredPositiveInt(configObject["decoder_sparse_step"], name: "decoder_sparse_step")
        let mlpOnlyLayers = try intArray(configObject["mlp_only_layers"], name: "mlp_only_layers")
        guard mlpOnlyLayers.allSatisfy({ 0 ..< hiddenLayers ~= $0 }) else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration("mlp_only_layers out of range")
        }
        let mlpOnlySet = Set(mlpOnlyLayers)
        let expectedLayers = (0 ..< hiddenLayers).filter {
            !mlpOnlySet.contains($0) && ($0 + 1) % sparseStep == 0
        }
        guard !expectedLayers.isEmpty else {
            throw Qwen3MoEExpertResidencyError.noExpectedMoELayers
        }

        do {
            try DuplicateJSONKeyValidator.validate(indexData)
        } catch let error as DuplicateJSONKeyError {
            throw Qwen3MoEExpertResidencyError.duplicateTensorName(error.key)
        }
        let indexObject = try jsonObject(indexData, context: "index")
        guard let rawWeightMap = indexObject["weight_map"] as? [String: Any] else {
            throw Qwen3MoEExpertResidencyError.indexMismatch("weight_map")
        }
        var weightMap: [String: String] = [:]
        for (name, value) in rawWeightMap {
            guard let relativeFile = value as? String else {
                throw Qwen3MoEExpertResidencyError.indexMismatch(name)
            }
            try validateRelativeFile(relativeFile)
            weightMap[name] = relativeFile
        }

        var shardIdentities: [String: Qwen3MoEShardIdentity] = [:]
        var parsedTensors: [String: ParsedTensor] = [:]
        for shard in shards {
            try validateRelativeFile(shard.relativeFile)
            guard shard.identity.byteCount > 0, !shard.identity.contentDigest.isEmpty else {
                throw Qwen3MoEExpertResidencyError.invalidConfiguration("invalid shard identity")
            }
            guard shardIdentities.updateValue(shard.identity, forKey: shard.relativeFile) == nil else {
                throw Qwen3MoEExpertResidencyError.invalidRelativePath(shard.relativeFile)
            }
            let tensors = try parseShard(shard)
            for tensor in tensors {
                guard parsedTensors.updateValue(tensor, forKey: tensor.name) == nil else {
                    throw Qwen3MoEExpertResidencyError.duplicateTensorName(tensor.name)
                }
            }
        }

        let headerNames = Set(parsedTensors.keys)
        let indexNames = Set(weightMap.keys)
        if headerNames != indexNames {
            let mismatch = headerNames.symmetricDifference(indexNames).sorted().first ?? "weight_map"
            throw Qwen3MoEExpertResidencyError.indexMismatch(mismatch)
        }
        for (name, tensor) in parsedTensors {
            guard weightMap[name] == tensor.relativeFile else {
                throw Qwen3MoEExpertResidencyError.indexMismatch(name)
            }
        }

        let expectedLayerSet = Set(expectedLayers)
        var expertTensors: [Int: [Qwen3MoEExpertProjection: [String: ParsedTensor]]] = [:]
        for tensor in parsedTensors.values {
            guard let parsedName = ExpertTensorName(tensor.name) else { continue }
            guard expectedLayerSet.contains(parsedName.layer) else {
                throw Qwen3MoEExpertResidencyError.extraMoELayer(parsedName.layer)
            }
            guard let actualExperts = tensor.shape.first else {
                throw Qwen3MoEExpertResidencyError.invalidShape(tensor.name)
            }
            guard actualExperts == numExperts else {
                throw Qwen3MoEExpertResidencyError.expertDimensionMismatch(
                    tensor: tensor.name,
                    expected: numExperts,
                    actual: actualExperts
                )
            }
            guard tensor.byteCount % numExperts == 0 else {
                throw Qwen3MoEExpertResidencyError.spanMismatch(tensor.name)
            }
            var projections = expertTensors[parsedName.layer, default: [:]]
            var components = projections[parsedName.projection, default: [:]]
            guard components.updateValue(tensor, forKey: parsedName.component) == nil else {
                throw Qwen3MoEExpertResidencyError.duplicateTensorName(tensor.name)
            }
            projections[parsedName.projection] = components
            expertTensors[parsedName.layer] = projections
        }

        var canonicalComponents: Set<String>?
        var bytesPerExpert: Int?
        var rangesByAddress: [Qwen3MoEExpertAddress: [Qwen3MoEExpertRange]] = [:]
        for layer in expectedLayers {
            guard let projections = expertTensors[layer] else {
                throw Qwen3MoEExpertResidencyError.missingMoELayer(layer)
            }
            var layerBytes = 0
            for projection in Qwen3MoEExpertProjection.allCases {
                guard let components = projections[projection], components["weight"] != nil else {
                    throw Qwen3MoEExpertResidencyError.missingProjection(
                        layer: layer,
                        projection: projection
                    )
                }
                let componentSet = Set(components.keys)
                if let canonicalComponents, canonicalComponents != componentSet {
                    throw Qwen3MoEExpertResidencyError.componentSetMismatch(
                        layer: layer,
                        projection: projection
                    )
                } else if canonicalComponents == nil {
                    canonicalComponents = componentSet
                }
                for tensor in components.values {
                    let sliceBytes = tensor.byteCount / numExperts
                    let (newLayerBytes, overflow) = layerBytes.addingReportingOverflow(sliceBytes)
                    guard !overflow else {
                        throw Qwen3MoEExpertResidencyError.rangeOverflow(tensor.name)
                    }
                    layerBytes = newLayerBytes
                    for expert in 0 ..< numExperts {
                        let (expertOffset, multiplyOverflow) = expert.multipliedReportingOverflow(by: sliceBytes)
                        let (lower, addOverflow) = tensor.absoluteRange.lowerBound.addingReportingOverflow(expertOffset)
                        let (upper, endOverflow) = lower.addingReportingOverflow(sliceBytes)
                        guard !multiplyOverflow, !addOverflow, !endOverflow,
                            upper <= tensor.absoluteRange.upperBound
                        else {
                            throw Qwen3MoEExpertResidencyError.rangeOverflow(tensor.name)
                        }
                        let address = Qwen3MoEExpertAddress(layer: layer, expert: expert)
                        rangesByAddress[address, default: []].append(
                            Qwen3MoEExpertRange(
                                address: address,
                                projection: projection,
                                component: ExpertTensorName(tensor.name)!.component,
                                tensorName: tensor.name,
                                dtype: tensor.dtype,
                                tensorShape: tensor.shape,
                                relativeFile: tensor.relativeFile,
                                absoluteRange: lower ..< upper
                            )
                        )
                    }
                }
            }
            if let bytesPerExpert, bytesPerExpert != layerBytes {
                throw Qwen3MoEExpertResidencyError.nonUniformExpertBytes
            }
            bytesPerExpert = layerBytes
        }

        for address in rangesByAddress.keys {
            rangesByAddress[address]!.sort {
                if $0.projection.sortOrder != $1.projection.sortOrder {
                    return $0.projection.sortOrder < $1.projection.sortOrder
                }
                if $0.component != $1.component { return $0.component < $1.component }
                return $0.tensorName < $1.tensorName
            }
        }

        let rangeIdentityData = try canonicalRangeIdentityData(
            ranges: rangesByAddress.values.flatMap { $0 },
            tensors: Array(parsedTensors.values),
            shardIdentities: shardIdentities
        )
        return Qwen3MoEExpertManifest(
            identity: Qwen3MoEExpertManifestIdentity(
                modelID: modelID,
                resolvedRevision: resolvedRevision.lowercased(),
                configSHA256: sha256(configData),
                rangeManifestSHA256: sha256(rangeIdentityData)
            ),
            expectedMoELayers: expectedLayers,
            numExperts: numExperts,
            bytesPerExpert: bytesPerExpert!,
            shardIdentities: shardIdentities,
            expertTensorNames: Set(rangesByAddress.values.flatMap { $0.map(\.tensorName) }),
            rangesByAddress: rangesByAddress
        )
    }

    private static func parseShard(_ shard: Qwen3MoESnapshotShard) throws -> [ParsedTensor] {
        guard shard.prefixData.count >= 8 else {
            throw Qwen3MoEExpertResidencyError.malformedHeader(shard.relativeFile)
        }
        var headerLength: UInt64 = 0
        for index in 0 ..< 8 {
            headerLength |= UInt64(shard.prefixData[index]) << UInt64(index * 8)
        }
        guard headerLength > 0, headerLength <= UInt64(Int.max) else {
            throw Qwen3MoEExpertResidencyError.malformedHeader(shard.relativeFile)
        }
        let (dataBase, baseOverflow) = 8.addingReportingOverflow(Int(headerLength))
        guard !baseOverflow, dataBase <= shard.prefixData.count,
            dataBase <= shard.identity.byteCount
        else {
            throw Qwen3MoEExpertResidencyError.malformedHeader(shard.relativeFile)
        }
        let header = shard.prefixData.subdata(in: 8 ..< dataBase)
        do {
            try DuplicateJSONKeyValidator.validate(
                header,
                integerValueKeys: ["shape", "data_offsets"]
            )
        } catch let error as DuplicateJSONKeyError {
            throw Qwen3MoEExpertResidencyError.duplicateTensorName(error.key)
        } catch is NonIntegerJSONValueError {
            throw Qwen3MoEExpertResidencyError.malformedHeader(shard.relativeFile)
        }
        let headerObject = try jsonObject(header, context: shard.relativeFile)

        var parsed: [ParsedTensor] = []
        for (name, rawValue) in headerObject where name != "__metadata__" {
            guard let value = rawValue as? [String: Any],
                let dtype = value["dtype"] as? String,
                let rawShape = value["shape"] as? [Any],
                let rawOffsets = value["data_offsets"] as? [Any], rawOffsets.count == 2
            else {
                throw Qwen3MoEExpertResidencyError.malformedHeader(name)
            }
            guard let width = dtypeWidths[dtype] else {
                throw Qwen3MoEExpertResidencyError.unsupportedDType(dtype)
            }
            let shape = try rawShape.map {
                guard let value = jsonInt($0), value > 0 else {
                    throw Qwen3MoEExpertResidencyError.invalidShape(name)
                }
                return value
            }
            guard !shape.isEmpty else {
                throw Qwen3MoEExpertResidencyError.invalidShape(name)
            }
            guard let start = jsonInt(rawOffsets[0]), let end = jsonInt(rawOffsets[1]),
                start >= 0, end > start
            else {
                throw Qwen3MoEExpertResidencyError.malformedHeader(name)
            }
            var expectedBytes = width
            for dimension in shape {
                let (product, overflow) = expectedBytes.multipliedReportingOverflow(by: dimension)
                guard !overflow else {
                    throw Qwen3MoEExpertResidencyError.rangeOverflow(name)
                }
                expectedBytes = product
            }
            guard end - start == expectedBytes else {
                throw Qwen3MoEExpertResidencyError.spanMismatch(name)
            }
            let (absoluteStart, startOverflow) = dataBase.addingReportingOverflow(start)
            let (absoluteEnd, endOverflow) = dataBase.addingReportingOverflow(end)
            guard !startOverflow, !endOverflow else {
                throw Qwen3MoEExpertResidencyError.rangeOverflow(name)
            }
            guard absoluteEnd <= shard.identity.byteCount else {
                throw Qwen3MoEExpertResidencyError.rangeOutOfBounds(name)
            }
            parsed.append(
                ParsedTensor(
                    name: name,
                    dtype: dtype,
                    shape: shape,
                    relativeFile: shard.relativeFile,
                    headerLength: Int(headerLength),
                    dataOffsets: start ..< end,
                    absoluteRange: absoluteStart ..< absoluteEnd,
                    fileByteCount: shard.identity.byteCount,
                    fileContentDigest: shard.identity.contentDigest
                )
            )
        }
        guard !parsed.isEmpty else {
            throw Qwen3MoEExpertResidencyError.malformedHeader(shard.relativeFile)
        }
        let sorted = parsed.sorted { $0.absoluteRange.lowerBound < $1.absoluteRange.lowerBound }
        for (left, right) in zip(sorted, sorted.dropFirst())
        where left.absoluteRange.upperBound > right.absoluteRange.lowerBound {
            throw Qwen3MoEExpertResidencyError.overlappingRanges(shard.relativeFile)
        }
        return parsed
    }

    private static func canonicalRangeIdentityData(
        ranges: [Qwen3MoEExpertRange],
        tensors: [ParsedTensor],
        shardIdentities: [String: Qwen3MoEShardIdentity]
    ) throws -> Data {
        let rangeObjects: [[String: Any]] = ranges.sorted(by: rangeSort).map {
            [
                "layer": $0.address.layer,
                "expert": $0.address.expert,
                "projection": $0.projection.tensorComponent,
                "component": $0.component,
                "tensor": $0.tensorName,
                "dtype": $0.dtype,
                "shape": $0.tensorShape,
                "file": $0.relativeFile,
                "lower": $0.absoluteRange.lowerBound,
                "upper": $0.absoluteRange.upperBound,
            ]
        }
        let shardObjects: [[String: Any]] = shardIdentities.keys.sorted().map {
            [
                "file": $0,
                "bytes": shardIdentities[$0]!.byteCount,
                "content": shardIdentities[$0]!.contentDigest,
            ]
        }
        let tensorObjects: [[String: Any]] = tensors.sorted { $0.name < $1.name }.map {
            [
                "name": $0.name,
                "dtype": $0.dtype,
                "shape": $0.shape,
                "file": $0.relativeFile,
                "headerLength": $0.headerLength,
                "dataLower": $0.dataOffsets.lowerBound,
                "dataUpper": $0.dataOffsets.upperBound,
                "fileBytes": $0.fileByteCount,
                "fileContent": $0.fileContentDigest,
            ]
        }
        return try JSONSerialization.data(
            withJSONObject: [
                "ranges": rangeObjects,
                "shards": shardObjects,
                "tensors": tensorObjects,
            ],
            options: [.sortedKeys]
        )
    }

    private static func rangeSort(_ lhs: Qwen3MoEExpertRange, _ rhs: Qwen3MoEExpertRange) -> Bool {
        if lhs.address.layer != rhs.address.layer { return lhs.address.layer < rhs.address.layer }
        if lhs.address.expert != rhs.address.expert { return lhs.address.expert < rhs.address.expert }
        if lhs.projection.sortOrder != rhs.projection.sortOrder {
            return lhs.projection.sortOrder < rhs.projection.sortOrder
        }
        if lhs.component != rhs.component { return lhs.component < rhs.component }
        return lhs.tensorName < rhs.tensorName
    }

    private static func validateRelativeFile(_ path: String) throws {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"), !path.contains(":"),
            components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
        else {
            throw Qwen3MoEExpertResidencyError.invalidRelativePath(path)
        }
    }

    private static func jsonObject(_ data: Data, context: String) throws -> [String: Any] {
        do {
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw Qwen3MoEExpertResidencyError.malformedHeader(context)
            }
            return object
        } catch let error as Qwen3MoEExpertResidencyError {
            throw error
        } catch {
            throw Qwen3MoEExpertResidencyError.malformedHeader(context)
        }
    }

    private static func requiredPositiveInt(_ raw: Any?, name: String) throws -> Int {
        guard let value = jsonInt(raw), value > 0 else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration(name)
        }
        return value
    }

    private static func intArray(_ raw: Any?, name: String) throws -> [Int] {
        guard let raw = raw as? [Any] else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration(name)
        }
        return try raw.map {
            guard let value = jsonInt($0), value >= 0 else {
                throw Qwen3MoEExpertResidencyError.invalidConfiguration(name)
            }
            return value
        }
    }

    private static func jsonInt(_ raw: Any?) -> Int? {
        guard let raw, let number = raw as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID(),
            !["f", "d"].contains(String(cString: number.objCType))
        else { return nil }
        return Int(number.stringValue)
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static let dtypeWidths: [String: Int] = [
        "BOOL": 1, "I8": 1, "U8": 1,
        "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
        "I32": 4, "U32": 4, "F32": 4,
        "I64": 8, "U64": 8, "F64": 8,
    ]
}

protocol Qwen3MoEExpertRangeReading: AnyObject {
    func identity(relativeFile: String) throws -> Qwen3MoEShardIdentity
    func read(_ range: Qwen3MoEExpertRange) throws -> Data
}

struct Qwen3MoEExpertPayload: Equatable, Sendable {
    let range: Qwen3MoEExpertRange
    let data: Data
}

struct Qwen3MoEExpertFetchMetrics: Equatable, Sendable {
    let hits: Int
    let misses: Int
    let bytesRead: Int
    let readCount: Int
    let readNanoseconds: Int
    let evictedExperts: [Int]

    init(
        hits: Int,
        misses: Int,
        bytesRead: Int,
        readCount: Int,
        readNanoseconds: Int = 0,
        evictedExperts: [Int]
    ) {
        self.hits = hits
        self.misses = misses
        self.bytesRead = bytesRead
        self.readCount = readCount
        self.readNanoseconds = readNanoseconds
        self.evictedExperts = evictedExperts
    }

    static let zero = Qwen3MoEExpertFetchMetrics(
        hits: 0,
        misses: 0,
        bytesRead: 0,
        readCount: 0,
        readNanoseconds: 0,
        evictedExperts: []
    )
}

struct Qwen3MoEExpertFetchResult: Equatable, Sendable {
    let metrics: Qwen3MoEExpertFetchMetrics
    let payloads: [Qwen3MoEExpertAddress: [Qwen3MoEExpertPayload]]
}

struct Qwen3MoEExpertPageFetchResult {
    let metrics: Qwen3MoEExpertFetchMetrics
    let page: Qwen3MoEExpertPage
}

struct Qwen3MoEExpertPage {
    let layer: Int
    let globalExpertIDs: [Int]
    let arrays: [Qwen3MoEExpertProjection: [String: MLXArray]]

    func array(
        projection: Qwen3MoEExpertProjection,
        component: String = "weight"
    ) throws -> MLXArray {
        guard let array = arrays[projection]?[component] else {
            throw Qwen3MoEExpertResidencyError.missingProjection(
                layer: layer,
                projection: projection
            )
        }
        return array
    }
}

extension Qwen3MoEExpertPage {
    static func materialize(
        layer: Int,
        fetchResult: Qwen3MoEExpertFetchResult
    ) throws -> Qwen3MoEExpertPage {
        let addresses = fetchResult.payloads.keys.sorted {
            if $0.layer != $1.layer { return $0.layer < $1.layer }
            return $0.expert < $1.expert
        }
        guard !addresses.isEmpty else {
            throw Qwen3MoEExpertResidencyError.emptyExpertPage
        }
        guard addresses.allSatisfy({ $0.layer == layer }) else {
            throw Qwen3MoEExpertResidencyError.unknownLayer(layer)
        }

        typealias Components = [Qwen3MoEExpertProjection: [String: MLXArray]]
        var materializedByAddress: [Qwen3MoEExpertAddress: Components] = [:]
        var rangesByAddress: [Qwen3MoEExpertAddress: [Qwen3MoEExpertPayload]] = [:]

        for address in addresses {
            guard let payloads = fetchResult.payloads[address], !payloads.isEmpty else {
                throw Qwen3MoEExpertResidencyError.invalidExpert(address.expert)
            }
            var projections: Components = [:]
            for payload in payloads {
                let range = payload.range
                guard range.address == address else {
                    throw Qwen3MoEExpertResidencyError.invalidExpert(address.expert)
                }
                guard ["weight", "scales", "biases"].contains(range.component) else {
                    throw Qwen3MoEExpertResidencyError.unsupportedComponent(
                        tensor: range.tensorName,
                        component: range.component
                    )
                }
                guard let dtype = materializationDType(range.dtype) else {
                    throw Qwen3MoEExpertResidencyError.unsupportedDType(range.dtype)
                }
                guard !range.expertShape.isEmpty else {
                    throw Qwen3MoEExpertResidencyError.invalidShape(range.tensorName)
                }
                guard projections[range.projection]?[range.component] == nil else {
                    throw Qwen3MoEExpertResidencyError.duplicateComponent(
                        tensor: range.tensorName,
                        component: range.component
                    )
                }
                let array = MLXArray(payload.data, range.expertShape, dtype: dtype)
                projections[range.projection, default: [:]][range.component] = array
            }
            materializedByAddress[address] = projections
            rangesByAddress[address] = payloads
        }

        let expectedProjections = Set(Qwen3MoEExpertProjection.allCases)
        let firstAddress = addresses[0]
        guard let first = materializedByAddress[firstAddress],
            Set(first.keys) == expectedProjections,
            let firstComponents = first[.gate].map({ Set($0.keys) }),
            firstComponents.contains("weight")
        else {
            throw Qwen3MoEExpertResidencyError.missingProjection(layer: layer, projection: .gate)
        }
        for address in addresses {
            guard let projections = materializedByAddress[address],
                Set(projections.keys) == expectedProjections
            else {
                throw Qwen3MoEExpertResidencyError.missingProjection(layer: layer, projection: .gate)
            }
            for projection in Qwen3MoEExpertProjection.allCases {
                guard let components = projections[projection],
                    Set(components.keys) == firstComponents
                else {
                    throw Qwen3MoEExpertResidencyError.componentSetMismatch(
                        layer: layer,
                        projection: projection
                    )
                }
            }
        }

        if firstComponents == ["weight"] {
            let shapes = try Dictionary(
                uniqueKeysWithValues: Qwen3MoEExpertProjection.allCases.map { projection in
                    guard let payload = rangesByAddress[firstAddress]?.first(where: {
                        $0.range.projection == projection && $0.range.component == "weight"
                    }) else {
                        throw Qwen3MoEExpertResidencyError.missingProjection(
                            layer: layer,
                            projection: projection
                        )
                    }
                    return (projection, payload.range.expertShape)
                }
            )
            guard let gate = shapes[.gate], let up = shapes[.up], let down = shapes[.down],
                gate.count == 2, gate == up, down == [gate[1], gate[0]]
            else {
                let tensor = rangesByAddress[firstAddress]?.first(where: {
                    $0.range.projection == .down && $0.range.component == "weight"
                })?.range.tensorName ?? "qwen3_moe expert weight"
                throw Qwen3MoEExpertResidencyError.invalidShape(tensor)
            }
        }

        var stacked: [Qwen3MoEExpertProjection: [String: MLXArray]] = [:]
        for projection in Qwen3MoEExpertProjection.allCases {
            for component in firstComponents.sorted() {
                let arrays = addresses.map { materializedByAddress[$0]![projection]![component]! }
                stacked[projection, default: [:]][component] = MLX.stacked(arrays)
            }
        }
        return Qwen3MoEExpertPage(
            layer: layer,
            globalExpertIDs: addresses.map(\.expert),
            arrays: stacked
        )
    }

    private static func materializationDType(_ dtype: String) -> DType? {
        switch dtype {
        case "F16": .float16
        case "BF16": .bfloat16
        case "U32": .uint32
        case "U8": .uint8
        default: nil
        }
    }
}

struct Qwen3MoEExpertResidencyCounters: Equatable, Sendable {
    let transactions: Int
    let hits: Int
    let misses: Int
    let bytesRead: Int
    let readCount: Int
    let readNanoseconds: Int
    let evictions: Int

    init(
        transactions: Int,
        hits: Int,
        misses: Int,
        bytesRead: Int,
        readCount: Int,
        readNanoseconds: Int = 0,
        evictions: Int
    ) {
        self.transactions = transactions
        self.hits = hits
        self.misses = misses
        self.bytesRead = bytesRead
        self.readCount = readCount
        self.readNanoseconds = readNanoseconds
        self.evictions = evictions
    }

    static let zero = Qwen3MoEExpertResidencyCounters(
        transactions: 0,
        hits: 0,
        misses: 0,
        bytesRead: 0,
        readCount: 0,
        readNanoseconds: 0,
        evictions: 0
    )
}

struct Qwen3MoEExpertResidencySnapshot: Equatable, Sendable {
    let residentExpertsByLayer: [Int: [Int]]
    let counters: Qwen3MoEExpertResidencyCounters
}

final class Qwen3MoEExpertResidency {
    private struct ResidentExpert {
        var lastUsedTick: UInt64
        let payloads: [Qwen3MoEExpertPayload]
    }

    private let manifest: Qwen3MoEExpertManifest
    private let capacityPerLayer: Int
    private var residents: [Int: [Int: ResidentExpert]]
    private var counters = Qwen3MoEExpertResidencyCounters.zero
    private var tick: UInt64 = 0

    init(manifest: Qwen3MoEExpertManifest, capacityPerLayer: Int) throws {
        guard capacityPerLayer > 0, capacityPerLayer <= manifest.numExperts else {
            throw Qwen3MoEExpertResidencyError.invalidCapacity(capacityPerLayer)
        }
        self.manifest = manifest
        self.capacityPerLayer = capacityPerLayer
        self.residents = Dictionary(
            uniqueKeysWithValues: manifest.expectedMoELayers.map { ($0, [:]) }
        )
    }

    func snapshot() -> Qwen3MoEExpertResidencySnapshot {
        Qwen3MoEExpertResidencySnapshot(
            residentExpertsByLayer: residents.mapValues { $0.keys.sorted() },
            counters: counters
        )
    }

    func fetch(
        layer: Int,
        routedExperts: [Int],
        reader: Qwen3MoEExpertRangeReading,
        cancellationCheck: () throws -> Void = {},
        materializationCheck: ((Qwen3MoEExpertFetchResult) throws -> Void)? = nil
    ) throws -> Qwen3MoEExpertFetchResult {
        guard let currentResidents = residents[layer] else {
            throw Qwen3MoEExpertResidencyError.unknownLayer(layer)
        }
        let requested = Array(Set(routedExperts)).sorted()
        for expert in requested where !(0 ..< manifest.numExperts ~= expert) {
            throw Qwen3MoEExpertResidencyError.invalidExpert(expert)
        }
        guard requested.count <= capacityPerLayer else {
            throw Qwen3MoEExpertResidencyError.requestExceedsCapacity(
                requested: requested.count,
                capacity: capacityPerLayer
            )
        }
        guard !requested.isEmpty else {
            return Qwen3MoEExpertFetchResult(metrics: .zero, payloads: [:])
        }

        try cancellationCheck()
        let requestedSet = Set(requested)
        let hits = requested.filter { currentResidents[$0] != nil }
        let misses = requested.filter { currentResidents[$0] == nil }
        let evictionCount = max(0, currentResidents.count + misses.count - capacityPerLayer)
        let evicted = currentResidents
            .filter { !requestedSet.contains($0.key) }
            .sorted {
                if $0.value.lastUsedTick != $1.value.lastUsedTick {
                    return $0.value.lastUsedTick < $1.value.lastUsedTick
                }
                return $0.key < $1.key
            }
            .prefix(evictionCount)
            .map(\.key)
        guard evicted.count == evictionCount else {
            throw Qwen3MoEExpertResidencyError.requestExceedsCapacity(
                requested: requested.count,
                capacity: capacityPerLayer
            )
        }

        var coldPayloads: [Int: [Qwen3MoEExpertPayload]] = [:]
        var bytesRead = 0
        var readCount = 0
        var readNanoseconds = 0
        let requestedRanges = try requested.flatMap {
            try manifest.ranges(layer: layer, expert: $0)
        }
        let missSet = Set(misses)
        let coldRanges = requestedRanges.filter { missSet.contains($0.address.expert) }
        let referencedFiles = Set(requestedRanges.map(\.relativeFile)).sorted()
        try validateIdentities(referencedFiles, reader: reader, cancellationCheck: cancellationCheck)
        for range in coldRanges {
            try cancellationCheck()
            let readStart = DispatchTime.now().uptimeNanoseconds
            let data = try reader.read(range)
            let elapsed = DispatchTime.now().uptimeNanoseconds - readStart
            guard data.count == range.byteCount else {
                throw Qwen3MoEExpertResidencyError.shortRead(
                    expected: range.byteCount,
                    actual: data.count
                )
            }
            let (nextBytes, bytesOverflow) = bytesRead.addingReportingOverflow(range.byteCount)
            let (nextReads, readsOverflow) = readCount.addingReportingOverflow(1)
            guard let elapsedNanoseconds = Int(exactly: elapsed) else {
                throw Qwen3MoEExpertResidencyError.rangeOverflow("read duration")
            }
            let (nextReadNanoseconds, readTimeOverflow) = readNanoseconds.addingReportingOverflow(
                elapsedNanoseconds
            )
            guard !bytesOverflow, !readsOverflow, !readTimeOverflow else {
                throw Qwen3MoEExpertResidencyError.rangeOverflow("transaction counters")
            }
            bytesRead = nextBytes
            readCount = nextReads
            readNanoseconds = nextReadNanoseconds
            coldPayloads[range.address.expert, default: []].append(
                Qwen3MoEExpertPayload(range: range, data: data)
            )
            try cancellationCheck()
        }
        try validateIdentities(referencedFiles, reader: reader, cancellationCheck: cancellationCheck)

        let (nextTick, tickOverflow) = tick.addingReportingOverflow(1)
        guard !tickOverflow else {
            throw Qwen3MoEExpertResidencyError.rangeOverflow("LRU tick")
        }
        let transactionMetrics = Qwen3MoEExpertFetchMetrics(
            hits: hits.count,
            misses: misses.count,
            bytesRead: bytesRead,
            readCount: readCount,
            readNanoseconds: readNanoseconds,
            evictedExperts: evicted
        )
        let nextCounters = try counters.adding(transactionMetrics)

        var nextResidents = currentResidents
        for expert in evicted { nextResidents[expert] = nil }
        for expert in misses {
            nextResidents[expert] = ResidentExpert(
                lastUsedTick: nextTick,
                payloads: coldPayloads[expert]!
            )
        }
        for expert in hits { nextResidents[expert]!.lastUsedTick = nextTick }

        let resultPayloads = Dictionary(
            uniqueKeysWithValues: requested.map {
                (
                    Qwen3MoEExpertAddress(layer: layer, expert: $0),
                    nextResidents[$0]!.payloads
                )
            }
        )
        let result = Qwen3MoEExpertFetchResult(metrics: transactionMetrics, payloads: resultPayloads)
        try cancellationCheck()
        try materializationCheck?(result)
        try cancellationCheck()

        residents[layer] = nextResidents
        counters = nextCounters
        tick = nextTick
        return result
    }

    func fetchMaterializedPage(
        layer: Int,
        routedExperts: [Int],
        reader: Qwen3MoEExpertRangeReading,
        cancellationCheck: () throws -> Void = {}
    ) throws -> Qwen3MoEExpertPageFetchResult {
        guard !routedExperts.isEmpty else {
            throw Qwen3MoEExpertResidencyError.emptyExpertPage
        }
        var page: Qwen3MoEExpertPage?
        let result = try fetch(
            layer: layer,
            routedExperts: routedExperts,
            reader: reader,
            cancellationCheck: cancellationCheck,
            materializationCheck: {
                page = try Qwen3MoEExpertPage.materialize(layer: layer, fetchResult: $0)
            }
        )
        guard let page else {
            throw Qwen3MoEExpertResidencyError.emptyExpertPage
        }
        return Qwen3MoEExpertPageFetchResult(metrics: result.metrics, page: page)
    }

    private func validateIdentities(
        _ relativeFiles: [String],
        reader: Qwen3MoEExpertRangeReading,
        cancellationCheck: () throws -> Void
    ) throws {
        for relativeFile in relativeFiles {
            try cancellationCheck()
            guard try reader.identity(relativeFile: relativeFile) == manifest.shardIdentities[relativeFile] else {
                throw Qwen3MoEExpertResidencyError.fileIdentityChanged(relativeFile)
            }
        }
    }
}

private extension Qwen3MoEExpertResidencyCounters {
    func adding(_ metrics: Qwen3MoEExpertFetchMetrics) throws -> Qwen3MoEExpertResidencyCounters {
        func add(_ lhs: Int, _ rhs: Int) throws -> Int {
            let (value, overflow) = lhs.addingReportingOverflow(rhs)
            guard !overflow else {
                throw Qwen3MoEExpertResidencyError.rangeOverflow("residency counters")
            }
            return value
        }
        return Qwen3MoEExpertResidencyCounters(
            transactions: try add(transactions, 1),
            hits: try add(hits, metrics.hits),
            misses: try add(misses, metrics.misses),
            bytesRead: try add(bytesRead, metrics.bytesRead),
            readCount: try add(readCount, metrics.readCount),
            readNanoseconds: try add(readNanoseconds, metrics.readNanoseconds),
            evictions: try add(evictions, metrics.evictedExperts.count)
        )
    }
}

private struct ParsedTensor {
    let name: String
    let dtype: String
    let shape: [Int]
    let relativeFile: String
    let headerLength: Int
    let dataOffsets: Range<Int>
    let absoluteRange: Range<Int>
    let fileByteCount: Int
    let fileContentDigest: String

    var byteCount: Int { absoluteRange.count }
}

private struct ExpertTensorName {
    let layer: Int
    let projection: Qwen3MoEExpertProjection
    let component: String

    init?(_ name: String) {
        let parts = name.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count >= 7,
            parts[0] == "model", parts[1] == "layers", let layer = Int(parts[2]),
            parts[3] == "mlp", parts[4] == "switch_mlp",
            let projection = Qwen3MoEExpertProjection(tensorComponent: String(parts[5]))
        else { return nil }
        let component = parts[6...].joined(separator: ".")
        guard !component.isEmpty else { return nil }
        self.layer = layer
        self.projection = projection
        self.component = component
    }
}

private struct DuplicateJSONKeyError: Error {
    let key: String
}

private struct NonIntegerJSONValueError: Error {
    let key: String
}

private struct DuplicateJSONKeyValidator {
    private let bytes: [UInt8]
    private let integerValueKeys: Set<String>
    private var index = 0

    static func validate(_ data: Data, integerValueKeys: Set<String> = []) throws {
        var parser = DuplicateJSONKeyValidator(
            bytes: Array(data),
            integerValueKeys: integerValueKeys
        )
        try parser.parseValue(integerOnly: false, key: nil)
        parser.skipWhitespace()
        guard parser.index == parser.bytes.count else {
            throw Qwen3MoEExpertResidencyError.malformedHeader("trailing JSON data")
        }
    }

    private mutating func parseValue(integerOnly: Bool, key: String?) throws {
        skipWhitespace()
        guard index < bytes.count else { throw malformed() }
        switch bytes[index] {
        case 0x7b:
            guard !integerOnly else { throw NonIntegerJSONValueError(key: key ?? "value") }
            try parseObject()
        case 0x5b: try parseArray(integerOnly: integerOnly, key: key)
        case 0x22:
            guard !integerOnly else { throw NonIntegerJSONValueError(key: key ?? "value") }
            _ = try parseString()
        default: try parsePrimitive(integerOnly: integerOnly, key: key)
        }
    }

    private mutating func parseObject() throws {
        index += 1
        skipWhitespace()
        if consume(0x7d) { return }
        var keys: Set<String> = []
        while true {
            skipWhitespace()
            let key = try parseString()
            guard keys.insert(key).inserted else { throw DuplicateJSONKeyError(key: key) }
            skipWhitespace()
            guard consume(0x3a) else { throw malformed() }
            try parseValue(integerOnly: integerValueKeys.contains(key), key: key)
            skipWhitespace()
            if consume(0x7d) { return }
            guard consume(0x2c) else { throw malformed() }
        }
    }

    private mutating func parseArray(integerOnly: Bool, key: String?) throws {
        index += 1
        skipWhitespace()
        if consume(0x5d) { return }
        while true {
            try parseValue(integerOnly: integerOnly, key: key)
            skipWhitespace()
            if consume(0x5d) { return }
            guard consume(0x2c) else { throw malformed() }
        }
    }

    private mutating func parseString() throws -> String {
        guard consume(0x22) else { throw malformed() }
        let start = index - 1
        var escaped = false
        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            if escaped {
                escaped = false
            } else if byte == 0x5c {
                escaped = true
            } else if byte == 0x22 {
                let data = Data(bytes[start ..< index])
                do { return try JSONDecoder().decode(String.self, from: data) }
                catch { throw malformed() }
            } else if byte < 0x20 {
                throw malformed()
            }
        }
        throw malformed()
    }

    private mutating func parsePrimitive(integerOnly: Bool, key: String?) throws {
        let start = index
        while index < bytes.count,
            ![0x2c, 0x5d, 0x7d, 0x20, 0x09, 0x0a, 0x0d].contains(bytes[index])
        {
            index += 1
        }
        guard index > start else { throw malformed() }
        if integerOnly {
            let token = bytes[start ..< index]
            var digits = token[...]
            if digits.first == 0x2d { digits = digits.dropFirst() }
            guard !digits.isEmpty, digits.allSatisfy({ 0x30 ... 0x39 ~= $0 }),
                digits.count == 1 || digits.first != 0x30
            else {
                throw NonIntegerJSONValueError(key: key ?? "value")
            }
        }
    }

    private mutating func skipWhitespace() {
        while index < bytes.count, [0x20, 0x09, 0x0a, 0x0d].contains(bytes[index]) {
            index += 1
        }
    }

    private mutating func consume(_ byte: UInt8) -> Bool {
        guard index < bytes.count, bytes[index] == byte else { return false }
        index += 1
        return true
    }

    private func malformed() -> Qwen3MoEExpertResidencyError {
        .malformedHeader("invalid JSON")
    }
}
