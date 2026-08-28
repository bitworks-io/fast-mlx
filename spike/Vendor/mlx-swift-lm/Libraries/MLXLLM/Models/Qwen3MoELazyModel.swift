import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Throwing, batch-one research runtime that owns Qwen3MoE expert pages outside the
/// public non-throwing `LLMModel` call chain. The runtime is single-caller only.
final class Qwen3MoELazyModelRuntime {
    let identity: Qwen3MoEExpertManifestIdentity
    let capacityPerLayer: Int
    let loadedParameterNames: [String]

    private let model: Qwen3MoEModel
    private let experts: Qwen3MoELazyExpertRuntime

    init(
        model: Qwen3MoEModel,
        experts: Qwen3MoELazyExpertRuntime,
        identity: Qwen3MoEExpertManifestIdentity,
        capacityPerLayer: Int
    ) {
        self.model = model
        self.experts = experts
        self.identity = identity
        self.capacityPerLayer = capacityPerLayer
        self.loadedParameterNames = model.parameters().flattened().map(\.0).sorted()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) throws -> MLXArray {
        guard inputs.dim(0) == 1 else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration(
                "lazy qwen3_moe supports batch one only"
            )
        }
        guard let cache else {
            return try model.callPaged(inputs, cache: nil, runtime: experts)
        }

        // Expert I/O can fail after attention has updated a layer cache. Run on
        // independent copies and commit only after every layer succeeds so a retry
        // with the caller-owned cache is well-defined.
        let workingCache = cache.map { $0.copy() }
        let output = try model.callPaged(inputs, cache: workingCache, runtime: experts)
        for index in cache.indices {
            var destination = cache[index]
            destination.state = workingCache[index].state
            destination.metaState = workingCache[index].metaState
        }
        return output
    }

    func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        model.newCache(parameters: parameters)
    }

    func snapshot() -> Qwen3MoEExpertResidencySnapshot {
        experts.snapshot()
    }
}

final class Qwen3MoELazyExpertRuntime {
    private let residency: Qwen3MoEExpertResidency
    private let reader: Qwen3MoEExpertRangeReading
    private let inputDims: Int
    private let hiddenDims: Int
    private let capacityPerLayer: Int
    private let quantizationByLayer: [Int: Qwen3MoEPagedSwitchQuantization]

    init(
        residency: Qwen3MoEExpertResidency,
        reader: Qwen3MoEExpertRangeReading,
        inputDims: Int,
        hiddenDims: Int,
        capacityPerLayer: Int,
        quantizationByLayer: [Int: Qwen3MoEPagedSwitchQuantization]
    ) {
        self.residency = residency
        self.reader = reader
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.capacityPerLayer = capacityPerLayer
        self.quantizationByLayer = quantizationByLayer
    }

    func switchOutput(
        layer: Int,
        input: MLXArray,
        globalIndices: MLXArray
    ) throws -> MLXArray {
        eval(globalIndices)
        let routedExperts = globalIndices.asType(.int32).asArray(Int32.self).map(Int.init)
        let topK = globalIndices.dim(-1)
        guard topK > 0, routedExperts.count.isMultiple(of: topK) else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration(
                "paged routing indices have an invalid top-k shape"
            )
        }
        let tokenCount = routedExperts.count / topK
        guard tokenCount == input.dim(-2) else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration(
                "paged routing indices do not match the input token count"
            )
        }

        var outputs: [MLXArray] = []
        var start = 0
        while start < tokenCount {
            var selected = Set<Int>()
            var end = start
            while end < tokenCount {
                let offset = end * topK
                let tokenExperts = Set(routedExperts[offset ..< (offset + topK)])
                if end > start, selected.union(tokenExperts).count > capacityPerLayer {
                    break
                }
                selected.formUnion(tokenExperts)
                end += 1
            }

            let inputChunk = input[0..., start ..< end, 0...]
            let indexChunk = globalIndices[0..., start ..< end, 0...]
            let fetched = try residency.fetchPagedSwitch(
                layer: layer,
                routedExperts: Array(selected),
                reader: reader,
                inputDims: inputDims,
                hiddenDims: hiddenDims,
                quantization: quantizationByLayer[layer]
            )
            let output = try fetched.switchGLU(inputChunk, indexChunk)
            eval(output)
            outputs.append(output)
            start = end
        }
        return concatenated(outputs, axis: 1)
    }

    func snapshot() -> Qwen3MoEExpertResidencySnapshot {
        residency.snapshot()
    }
}

func loadQwen3MoELazyModel(
    from downloader: any Downloader,
    modelID: String,
    revision: String,
    capacityFraction: Double,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> Qwen3MoELazyModelRuntime {
    try validateQwen3MoEImmutableIdentity(modelID: modelID, revision: revision)
    let directory = try await downloader.download(
        id: modelID,
        revision: revision,
        matching: modelDownloadPatterns,
        useLatest: false,
        progressHandler: progressHandler
    )
    return try loadQwen3MoELazyModel(
        modelDirectory: directory,
        modelID: modelID,
        resolvedRevision: revision,
        capacityFraction: capacityFraction
    )
}

func loadQwen3MoELazyModel(
    modelDirectory: URL,
    modelID: String,
    resolvedRevision: String,
    capacityFraction: Double
) throws -> Qwen3MoELazyModelRuntime {
    try validateQwen3MoEImmutableIdentity(modelID: modelID, revision: resolvedRevision)
    guard capacityFraction.isFinite, capacityFraction > 0, capacityFraction <= 1 else {
        throw Qwen3MoEExpertResidencyError.invalidConfiguration("invalid capacity fraction")
    }

    let configURL = modelDirectory.appending(path: "config.json")
    let indexURL = modelDirectory.appending(path: "model.safetensors.index.json")
    let configData = try Data(contentsOf: configURL)
    let indexData = try Data(contentsOf: indexURL)
    let configuration = try JSONDecoder.json5().decode(Qwen3MoEConfiguration.self, from: configData)
    let baseConfiguration = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
    let disk = try Qwen3MoEDiskExpertSource(modelDirectory: modelDirectory)
    let manifest = try Qwen3MoEExpertManifestBuilder.build(
        modelID: modelID,
        resolvedRevision: resolvedRevision,
        configData: configData,
        indexData: indexData,
        shards: disk.snapshotShards
    )

    let capacity = Int((Double(manifest.numExperts) * capacityFraction).rounded(.down))
    guard capacity >= configuration.numExpertsPerToken else {
        throw Qwen3MoEExpertResidencyError.invalidCapacity(capacity)
    }
    let quantizationByLayer = try qwen3MoEExpertQuantization(
        manifest: manifest,
        perLayerQuantization: baseConfiguration.perLayerQuantization
    )
    let residency = try Qwen3MoEExpertResidency(
        manifest: manifest,
        capacityPerLayer: capacity
    )
    let expertRuntime = Qwen3MoELazyExpertRuntime(
        residency: residency,
        reader: disk,
        inputDims: configuration.hiddenSize,
        hiddenDims: configuration.moeIntermediateSize,
        capacityPerLayer: capacity,
        quantizationByLayer: quantizationByLayer
    )
    let model = Qwen3MoEModel(configuration, pagedExperts: true)
    try loadWeights(
        modelDirectory: modelDirectory,
        model: model,
        perLayerQuantization: baseConfiguration.perLayerQuantization,
        weightFilter: { !manifest.expertTensorNames.contains($0) }
    )
    return Qwen3MoELazyModelRuntime(
        model: model,
        experts: expertRuntime,
        identity: manifest.identity,
        capacityPerLayer: capacity
    )
}

private func validateQwen3MoEImmutableIdentity(modelID: String, revision: String) throws {
    guard !modelID.isEmpty else {
        throw Qwen3MoEExpertResidencyError.invalidConfiguration("empty model id")
    }
    guard revision.count == 40, revision.allSatisfy(\.isHexDigit) else {
        throw Qwen3MoEExpertResidencyError.invalidConfiguration(
            "revision is not an immutable SHA"
        )
    }
}

private func qwen3MoEExpertQuantization(
    manifest: Qwen3MoEExpertManifest,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization?
) throws -> [Int: Qwen3MoEPagedSwitchQuantization] {
    let isQuantized = manifest.expertTensorNames.contains { $0.hasSuffix(".scales") }
    guard isQuantized else { return [:] }
    guard let perLayerQuantization else {
        throw Qwen3MoEExpertResidencyError.invalidConfiguration(
            "quantized expert pages require quantization configuration"
        )
    }

    var result: [Int: Qwen3MoEPagedSwitchQuantization] = [:]
    for layer in manifest.expectedMoELayers {
        var selected: BaseConfiguration.Quantization?
        for projection in Qwen3MoEExpertProjection.allCases {
            let path = "model.layers.\(layer).mlp.switch_mlp.\(projection.tensorComponent)"
            guard let candidate = perLayerQuantization.quantization(layer: path),
                case .affine = candidate.mode
            else {
                throw Qwen3MoEExpertResidencyError.invalidConfiguration(
                    "unsupported paged expert quantization"
                )
            }
            if let selected {
                guard selected.groupSize == candidate.groupSize, selected.bits == candidate.bits else {
                    throw Qwen3MoEExpertResidencyError.invalidConfiguration(
                        "mixed paged expert quantization"
                    )
                }
            } else {
                selected = candidate
            }
        }
        guard let selected else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration(
                "paged expert quantization was not resolved"
            )
        }
        result[layer] = Qwen3MoEPagedSwitchQuantization(
            groupSize: selected.groupSize,
            bits: selected.bits,
            mode: selected.mode
        )
    }
    return result
}

private final class Qwen3MoEDiskExpertSource: Qwen3MoEExpertRangeReading {
    private struct Fingerprint: Equatable {
        let byteCount: Int
        let modificationNanoseconds: Int64
        let fileIdentifier: String
    }

    private struct Shard {
        let url: URL
        let handle: FileHandle
        let expectedIdentity: Qwen3MoEShardIdentity
        var acceptedFingerprint: Fingerprint
    }

    let snapshotShards: [Qwen3MoESnapshotShard]
    private var shards: [String: Shard]

    init(modelDirectory: URL) throws {
        let urls = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !urls.isEmpty else {
            throw Qwen3MoEExpertResidencyError.noSafetensors
        }

        var snapshots: [Qwen3MoESnapshotShard] = []
        var opened: [String: Shard] = [:]
        for url in urls {
            let relativeFile = url.lastPathComponent
            let handle = try FileHandle(forReadingFrom: url.resolvingSymlinksInPath())
            let fingerprint = try Self.fingerprint(url)
            let byteCount = try Self.byteCount(handle)
            let digest = try Self.sha256(handle)
            let identity = Qwen3MoEShardIdentity(
                byteCount: byteCount,
                contentDigest: digest
            )
            snapshots.append(
                Qwen3MoESnapshotShard(
                    relativeFile: relativeFile,
                    identity: identity,
                    prefixData: try Self.headerPrefix(
                        handle,
                        byteCount: byteCount,
                        fileName: relativeFile
                    )
                )
            )
            opened[relativeFile] = Shard(
                url: url,
                handle: handle,
                expectedIdentity: identity,
                acceptedFingerprint: fingerprint
            )
        }
        self.snapshotShards = snapshots
        self.shards = opened
    }

    func identity(relativeFile: String) throws -> Qwen3MoEShardIdentity {
        guard var shard = shards[relativeFile] else {
            throw Qwen3MoEExpertResidencyError.invalidRelativePath(relativeFile)
        }
        let current = try Self.fingerprint(shard.url)
        if current == shard.acceptedFingerprint {
            return shard.expectedIdentity
        }
        let observed = Qwen3MoEShardIdentity(
            byteCount: current.byteCount,
            contentDigest: try Self.sha256(shard.url)
        )
        if observed == shard.expectedIdentity {
            shard.acceptedFingerprint = current
            shards[relativeFile] = shard
        }
        return observed
    }

    func read(_ range: Qwen3MoEExpertRange) throws -> Data {
        guard let shard = shards[range.relativeFile] else {
            throw Qwen3MoEExpertResidencyError.invalidRelativePath(range.relativeFile)
        }
        // The retained descriptor binds validation and reads to the immutable file
        // opened during manifest construction, even if a symlink is retargeted.
        try shard.handle.seek(toOffset: UInt64(range.absoluteRange.lowerBound))
        return try shard.handle.read(upToCount: range.byteCount) ?? Data()
    }

    private static func fingerprint(_ url: URL) throws -> Fingerprint {
        let resolvedURL = url.resolvingSymlinksInPath()
        let attributes = try FileManager.default.attributesOfItem(atPath: resolvedURL.path)
        guard let byteCount = (attributes[.size] as? NSNumber)?.intValue, byteCount > 0 else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration("invalid shard size")
        }
        let modificationNanoseconds = Int64(
            ((attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
                * 1_000_000_000
        )
        let fileIdentifier = [
            attributes[.systemNumber], attributes[.systemFileNumber],
        ]
        .map { String(describing: $0) }
        .joined(separator: ":")
        return Fingerprint(
            byteCount: byteCount,
            modificationNanoseconds: modificationNanoseconds,
            fileIdentifier: fileIdentifier
        )
    }

    private static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        return try sha256(handle)
    }

    private static func sha256(_ handle: FileHandle) throws -> String {
        try handle.seek(toOffset: 0)
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func byteCount(_ handle: FileHandle) throws -> Int {
        let end = try handle.seekToEnd()
        guard end > 0, let result = Int(exactly: end) else {
            throw Qwen3MoEExpertResidencyError.invalidConfiguration("invalid shard size")
        }
        return result
    }

    private static func headerPrefix(
        _ handle: FileHandle,
        byteCount: Int,
        fileName: String
    ) throws -> Data {
        try handle.seek(toOffset: 0)
        let prefix = try handle.read(upToCount: 8) ?? Data()
        guard prefix.count == 8 else {
            throw Qwen3MoEExpertResidencyError.malformedHeader(fileName)
        }
        var headerLength: UInt64 = 0
        for index in 0 ..< 8 {
            headerLength |= UInt64(prefix[index]) << UInt64(index * 8)
        }
        guard headerLength > 0, headerLength <= 100_000_000,
            headerLength <= UInt64(Int.max), Int(headerLength) <= byteCount - 8
        else {
            throw Qwen3MoEExpertResidencyError.malformedHeader(fileName)
        }
        let header = try handle.read(upToCount: Int(headerLength)) ?? Data()
        guard header.count == Int(headerLength) else {
            throw Qwen3MoEExpertResidencyError.malformedHeader(fileName)
        }
        var data = prefix
        data.append(header)
        return data
    }
}
