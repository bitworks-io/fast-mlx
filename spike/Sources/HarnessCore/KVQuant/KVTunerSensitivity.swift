import Foundation

public enum KVTunerSensitivityError: Error, Equatable, Sendable {
    case unsupportedSchema(Int)
    case invalidIdentifier(String)
    case invalidDigest(String)
    case invalidPromptDigests
    case invalidPromptCount(expected: Int, actual: Int)
    case unsupportedGroupSize(Int)
    case invalidLayerCount
    case invalidPrecisionPairs
    case invalidProtocol(String)
    case sampleCountOverflow
    case incompleteSamples(expected: Int, actual: Int)
    case nonCanonicalSample(position: Int)
    case invalidMetric(sample: Int)
    case invalidDBSCANPoint
    case calibrationManifestMismatch
}

/// One native affine precision pair admitted by the fast-mlx KVTuner runtime. The pair-bit cost
/// is twice KVTuner's per-layer nominal mean: `(keyBits + valueBits) / 2`.
public struct KVTunerPrecisionPair:
    Codable, Equatable, Hashable, Sendable
{
    public let keyBits: Int
    public let valueBits: Int

    public init(keyBits: Int, valueBits: Int) {
        self.keyBits = keyBits
        self.valueBits = valueBits
    }

    public var pairBitCost: Int { keyBits + valueBits }
}

/// Scalar result of applying one precision pair to one layer for one calibration prompt. The
/// MLX capture path reduces arrays before crossing its actor boundary; this pure artifact never
/// owns MLX state.
public struct KVTunerSensitivitySample: Codable, Equatable, Sendable {
    public var promptIndex: Int
    public var layer: Int
    public var keyBits: Int
    public var valueBits: Int
    public var relativeKeyError: Double
    public var relativeValueError: Double
    public var attentionScoreError: Double
    public var relativeAttentionOutputError: Double

    public init(
        promptIndex: Int,
        layer: Int,
        keyBits: Int,
        valueBits: Int,
        relativeKeyError: Double,
        relativeValueError: Double,
        attentionScoreError: Double,
        relativeAttentionOutputError: Double
    ) {
        self.promptIndex = promptIndex
        self.layer = layer
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.relativeKeyError = relativeKeyError
        self.relativeValueError = relativeValueError
        self.attentionScoreError = attentionScoreError
        self.relativeAttentionOutputError = relativeAttentionOutputError
    }
}

/// Immutable-on-disk input to deterministic Pareto pruning and grouping. It intentionally binds
/// group size because MLX affine sensitivities at g64 and g128 are different experiments.
public struct KVTunerSensitivityArtifact: Codable, Equatable, Sendable {
    public static let requiredSensitivityPromptCount = 20
    public static let canonicalPrecisionPairs = [
        KVTunerPrecisionPair(keyBits: 8, valueBits: 4),
        KVTunerPrecisionPair(keyBits: 8, valueBits: 2),
        KVTunerPrecisionPair(keyBits: 4, valueBits: 2),
    ]

    public var schemaVersion: Int
    public var matrixID: String
    public var modelConfigHash: String
    public var modelConfigSHA256: String
    public var checkpointManifestHash: String
    public var tokenizerSHA256: String
    public var calibrationCorpusID: String
    public var calibrationCorpusHash: String
    public var promptManifestSHA256: String
    public var promptDigests: [String]
    public var quantizerID: String
    public var captureMode: String
    public var groupSize: Int
    public var layerCount: Int
    public var precisionPairs: [KVTunerPrecisionPair]
    /// Pins the paper-v5 elementwise metrics: mean(abs(X-Xq) / max(abs(X), eps))
    /// for K, V, and attention output, plus mean(abs(A-Aq)) after softmax for attention.
    public var metricProtocolID: String
    /// Reduction and softmax accumulation precision used by the MLX capture oracle.
    public var metricAccumulationDType: String
    public var denominatorEpsilon: Double
    public var aggregationID: String
    public var dbscanEpsilon: Double
    public var dbscanMinSamples: Int
    public var samples: [KVTunerSensitivitySample]

    public init(
        schemaVersion: Int,
        matrixID: String,
        modelConfigHash: String,
        modelConfigSHA256: String,
        checkpointManifestHash: String,
        tokenizerSHA256: String,
        calibrationCorpusID: String,
        calibrationCorpusHash: String,
        promptManifestSHA256: String,
        promptDigests: [String],
        quantizerID: String,
        captureMode: String,
        groupSize: Int,
        layerCount: Int,
        precisionPairs: [KVTunerPrecisionPair],
        metricProtocolID: String,
        metricAccumulationDType: String,
        denominatorEpsilon: Double,
        aggregationID: String,
        dbscanEpsilon: Double,
        dbscanMinSamples: Int,
        samples: [KVTunerSensitivitySample]
    ) {
        self.schemaVersion = schemaVersion
        self.matrixID = matrixID
        self.modelConfigHash = modelConfigHash
        self.modelConfigSHA256 = modelConfigSHA256
        self.checkpointManifestHash = checkpointManifestHash
        self.tokenizerSHA256 = tokenizerSHA256
        self.calibrationCorpusID = calibrationCorpusID
        self.calibrationCorpusHash = calibrationCorpusHash
        self.promptManifestSHA256 = promptManifestSHA256
        self.promptDigests = promptDigests
        self.quantizerID = quantizerID
        self.captureMode = captureMode
        self.groupSize = groupSize
        self.layerCount = layerCount
        self.precisionPairs = precisionPairs
        self.metricProtocolID = metricProtocolID
        self.metricAccumulationDType = metricAccumulationDType
        self.denominatorEpsilon = denominatorEpsilon
        self.aggregationID = aggregationID
        self.dbscanEpsilon = dbscanEpsilon
        self.dbscanMinSamples = dbscanMinSamples
        self.samples = samples
    }

    @discardableResult
    public func validated() throws -> KVTunerSensitivityArtifact {
        guard schemaVersion == 1 else {
            throw KVTunerSensitivityError.unsupportedSchema(schemaVersion)
        }
        for identifier in [
            matrixID,
            calibrationCorpusID,
            quantizerID,
            captureMode,
            aggregationID,
        ] {
            guard Self.isIdentifier(identifier) else {
                throw KVTunerSensitivityError.invalidIdentifier(identifier)
            }
        }
        for digest in [modelConfigHash, checkpointManifestHash,
                       calibrationCorpusHash] {
            guard Self.isIdentityDigest(digest) else {
                throw KVTunerSensitivityError.invalidDigest(digest)
            }
        }
        for digest in [
            modelConfigSHA256, tokenizerSHA256, promptManifestSHA256,
        ] {
            guard Self.isLowercaseHex(digest, length: 64) else {
                throw KVTunerSensitivityError.invalidDigest(digest)
            }
        }
        guard !promptDigests.isEmpty,
            Set(promptDigests).count == promptDigests.count,
            promptDigests.allSatisfy(KVTunerPromptDigest.isCanonical)
        else {
            throw KVTunerSensitivityError.invalidPromptDigests
        }
        guard promptDigests.count == Self.requiredSensitivityPromptCount else {
            throw KVTunerSensitivityError.invalidPromptCount(
                expected: Self.requiredSensitivityPromptCount,
                actual: promptDigests.count)
        }
        guard [64, 128].contains(groupSize) else {
            throw KVTunerSensitivityError.unsupportedGroupSize(groupSize)
        }
        guard layerCount > 0 else {
            throw KVTunerSensitivityError.invalidLayerCount
        }
        guard precisionPairs == Self.canonicalPrecisionPairs else {
            throw KVTunerSensitivityError.invalidPrecisionPairs
        }
        guard quantizerID == "mlx-affine-asymmetric-v1" else {
            throw KVTunerSensitivityError.invalidProtocol("quantizerID")
        }
        guard captureMode == "single-prefill-no-error-propagation-v1" else {
            throw KVTunerSensitivityError.invalidProtocol("captureMode")
        }
        guard metricProtocolID
            == "kvtuner-v5-elementwise-mean-absolute-v1"
        else {
            throw KVTunerSensitivityError.invalidProtocol(
                "metricProtocolID")
        }
        guard metricAccumulationDType == "float32" else {
            throw KVTunerSensitivityError.invalidProtocol(
                "metricAccumulationDType")
        }
        guard denominatorEpsilon == 1e-8 else {
            throw KVTunerSensitivityError.invalidProtocol(
                "denominatorEpsilon")
        }
        guard aggregationID == "ordered-incremental-mean-v1" else {
            throw KVTunerSensitivityError.invalidProtocol("aggregationID")
        }
        guard dbscanEpsilon == 0.05 else {
            throw KVTunerSensitivityError.invalidProtocol("dbscanEpsilon")
        }
        guard dbscanMinSamples == 2 else {
            throw KVTunerSensitivityError.invalidProtocol(
                "dbscanMinSamples")
        }

        let promptLayerCount = promptDigests.count.multipliedReportingOverflow(
            by: layerCount)
        guard !promptLayerCount.overflow else {
            throw KVTunerSensitivityError.sampleCountOverflow
        }
        let expectedResult = promptLayerCount.partialValue
            .multipliedReportingOverflow(by: precisionPairs.count)
        guard !expectedResult.overflow else {
            throw KVTunerSensitivityError.sampleCountOverflow
        }
        let expected = expectedResult.partialValue
        guard samples.count == expected else {
            throw KVTunerSensitivityError.incompleteSamples(
                expected: expected, actual: samples.count)
        }

        var position = 0
        for promptIndex in promptDigests.indices {
            for layer in 0..<layerCount {
                for pair in precisionPairs {
                    let sample = samples[position]
                    guard sample.promptIndex == promptIndex,
                        sample.layer == layer,
                        sample.keyBits == pair.keyBits,
                        sample.valueBits == pair.valueBits
                    else {
                        throw KVTunerSensitivityError.nonCanonicalSample(
                            position: position)
                    }
                    let metrics = [
                        sample.relativeKeyError,
                        sample.relativeValueError,
                        sample.attentionScoreError,
                        sample.relativeAttentionOutputError,
                    ]
                    guard metrics.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                        throw KVTunerSensitivityError.invalidMetric(
                            sample: position)
                    }
                    position += 1
                }
            }
        }
        return self
    }

    /// Authenticates the compact source/tokenizer manifest at the same boundary as sensitivity
    /// rows. Structural validity alone is insufficient: the exact manifest bytes, all model
    /// identities, and the ordered first-20 prompt fingerprints must agree.
    @discardableResult
    public func validated(
        calibrationManifest: KVTunerCalibrationManifest,
        exactCalibrationManifestData: Data
    ) throws -> KVTunerSensitivityArtifact {
        let decoded: KVTunerCalibrationManifest
        do {
            decoded = try JSONDecoder().decode(
                KVTunerCalibrationManifest.self,
                from: exactCalibrationManifestData)
            _ = try decoded.validated()
        } catch {
            throw KVTunerSensitivityError.calibrationManifestMismatch
        }
        guard decoded == calibrationManifest else {
            throw KVTunerSensitivityError.calibrationManifestMismatch
        }
        let manifestSHA256 = sha256Hex(exactCalibrationManifestData)
        guard modelConfigHash == decoded.modelConfigHash,
            modelConfigSHA256 == decoded.modelConfigSHA256,
            checkpointManifestHash == decoded.checkpointManifestHash,
            tokenizerSHA256 == decoded.tokenizerSHA256,
            calibrationCorpusID == decoded.corpusID,
            calibrationCorpusHash == manifestSHA256,
            promptManifestSHA256 == manifestSHA256,
            promptDigests == decoded.sensitivityPrompts.map(\.promptDigest)
        else {
            throw KVTunerSensitivityError.calibrationManifestMismatch
        }
        return try validated()
    }

    public func analyzed() throws -> KVTunerSensitivityAnalysis {
        let artifact = try validated()
        let promptCount = artifact.promptDigests.count
        let pairCount = artifact.precisionPairs.count
        var layers: [KVTunerLayerSensitivity] = []
        layers.reserveCapacity(artifact.layerCount)

        for layer in 0..<artifact.layerCount {
            var aggregates: [KVTunerPairSensitivity] = []
            aggregates.reserveCapacity(pairCount)
            for (pairIndex, pair) in artifact.precisionPairs.enumerated() {
                var keyError = 0.0
                var valueError = 0.0
                var scoreError = 0.0
                var outputError = 0.0
                for promptIndex in 0..<promptCount {
                    let sampleIndex = promptIndex * artifact.layerCount
                        * pairCount + layer * pairCount + pairIndex
                    let sample = artifact.samples[sampleIndex]
                    let count = Double(promptIndex + 1)
                    keyError += (sample.relativeKeyError - keyError) / count
                    valueError +=
                        (sample.relativeValueError - valueError) / count
                    scoreError +=
                        (sample.attentionScoreError - scoreError) / count
                    outputError +=
                        (sample.relativeAttentionOutputError - outputError)
                            / count
                }
                guard keyError.isFinite, valueError.isFinite,
                    scoreError.isFinite, outputError.isFinite
                else {
                    throw KVTunerSensitivityError.invalidMetric(
                        sample: layer * pairCount + pairIndex)
                }
                aggregates.append(KVTunerPairSensitivity(
                    pair: pair,
                    relativeKeyError: keyError,
                    relativeValueError: valueError,
                    attentionScoreError: scoreError,
                    relativeAttentionOutputError: outputError))
            }
            let survivors = Self.paretoSurvivors(aggregates)
            layers.append(KVTunerLayerSensitivity(
                layer: layer,
                aggregates: aggregates,
                paretoPairs: survivors))
        }

        var partitions: [[KVTunerPrecisionPair]: [KVTunerLayerSensitivity]] = [:]
        for layer in layers {
            partitions[layer.paretoPairs, default: []].append(layer)
        }

        var rawGroups: [([Int], [KVTunerPrecisionPair])] = []
        for (allowedPairs, partitionLayers) in partitions {
            let points = partitionLayers.map { layer in
                let errorsByPair = Dictionary(uniqueKeysWithValues:
                    layer.aggregates.map {
                        ($0.pair, $0.relativeAttentionOutputError)
                    })
                return KVTunerDBSCANPoint(
                    layer: layer.layer,
                    features: allowedPairs.map { pair in
                        // Analysis constructs each layer from the artifact's complete canonical
                        // pair list, so every Pareto survivor is present in this dictionary.
                        errorsByPair[pair] ?? .infinity
                    })
            }
            let clusters = try KVTunerDeterministicDBSCAN.cluster(
                points: points,
                epsilon: artifact.dbscanEpsilon,
                minSamples: artifact.dbscanMinSamples)
            rawGroups.append(contentsOf: clusters.map { ($0, allowedPairs) })
        }
        rawGroups.sort {
            ($0.0.first ?? Int.max) < ($1.0.first ?? Int.max)
        }
        let groups = rawGroups.enumerated().map { index, raw in
            KVTunerSensitivityGroup(
                id: index,
                layers: raw.0,
                allowedPairs: raw.1)
        }
        return KVTunerSensitivityAnalysis(layers: layers, groups: groups)
    }

    static func paretoSurvivors(
        _ aggregates: [KVTunerPairSensitivity]
    ) -> [KVTunerPrecisionPair] {
        aggregates.compactMap { candidate in
            let dominated = aggregates.contains { other in
                guard other.pair != candidate.pair else { return false }
                let noWorseCost = other.pair.pairBitCost
                    <= candidate.pair.pairBitCost
                let noWorseError = other.relativeAttentionOutputError
                    <= candidate.relativeAttentionOutputError
                let strictlyBetter = other.pair.pairBitCost
                    < candidate.pair.pairBitCost
                    || other.relativeAttentionOutputError
                        < candidate.relativeAttentionOutputError
                return noWorseCost && noWorseError && strictlyBetter
            }
            return dominated ? nil : candidate.pair
        }
    }

    private static func isIdentifier(_ value: String) -> Bool {
        guard !value.isEmpty, value != "unknown",
            value == value.trimmingCharacters(in: .whitespacesAndNewlines),
            !value.contains("\n"), !value.contains("\r")
        else { return false }
        let punctuation = CharacterSet(charactersIn: "-._:/@+")
        return value.unicodeScalars.allSatisfy {
            CharacterSet.alphanumerics.contains($0)
                || punctuation.contains($0)
        }
    }

    private static func isIdentityDigest(_ value: String) -> Bool {
        isLowercaseHex(value, length: 16)
            || isLowercaseHex(value, length: 64)
    }

    private static func isLowercaseHex(_ value: String, length: Int) -> Bool {
        guard value.count == length else { return false }
        return value.unicodeScalars.allSatisfy {
            CharacterSet(charactersIn: "0123456789abcdef").contains($0)
        }
    }
}

public struct KVTunerPairSensitivity: Codable, Equatable, Sendable {
    public let pair: KVTunerPrecisionPair
    public let relativeKeyError: Double
    public let relativeValueError: Double
    public let attentionScoreError: Double
    public let relativeAttentionOutputError: Double
}

public struct KVTunerLayerSensitivity: Codable, Equatable, Sendable {
    public let layer: Int
    public let aggregates: [KVTunerPairSensitivity]
    public let paretoPairs: [KVTunerPrecisionPair]
}

public struct KVTunerSensitivityGroup: Codable, Equatable, Sendable {
    public let id: Int
    public let layers: [Int]
    public let allowedPairs: [KVTunerPrecisionPair]
}

public struct KVTunerSensitivityAnalysis: Codable, Equatable, Sendable {
    public let layers: [KVTunerLayerSensitivity]
    public let groups: [KVTunerSensitivityGroup]
}

struct KVTunerDBSCANPoint: Equatable, Sendable {
    let layer: Int
    let features: [Double]
}

enum KVTunerDeterministicDBSCAN {
    static func cluster(
        points: [KVTunerDBSCANPoint],
        epsilon: Double,
        minSamples: Int
    ) throws -> [[Int]] {
        guard epsilon.isFinite, epsilon > 0, minSamples > 0 else {
            throw KVTunerSensitivityError.invalidDBSCANPoint
        }
        guard !points.isEmpty else { return [] }

        let ordered = points.sorted { $0.layer < $1.layer }
        let featureCount = ordered[0].features.count
        guard featureCount > 0,
            Set(ordered.map(\.layer)).count == ordered.count,
            ordered.allSatisfy({ point in
                point.features.count == featureCount
                    && point.features.allSatisfy { $0.isFinite }
            })
        else {
            throw KVTunerSensitivityError.invalidDBSCANPoint
        }

        let epsilonSquared = epsilon * epsilon
        func neighbors(of index: Int) -> [Int] {
            ordered.indices.filter { candidate in
                var distanceSquared = 0.0
                for feature in 0..<featureCount {
                    let delta = ordered[index].features[feature]
                        - ordered[candidate].features[feature]
                    distanceSquared += delta * delta
                }
                return distanceSquared <= epsilonSquared
            }
        }

        var visited = Array(repeating: false, count: ordered.count)
        var labels = Array<Int?>(repeating: nil, count: ordered.count)
        var nextCluster = 0

        for index in ordered.indices {
            guard !visited[index] else { continue }
            visited[index] = true
            var seeds = neighbors(of: index)
            if seeds.count < minSamples {
                labels[index] = -1
                continue
            }

            let cluster = nextCluster
            nextCluster += 1
            labels[index] = cluster
            var queued = Set(seeds)
            var cursor = 0
            while cursor < seeds.count {
                let neighbor = seeds[cursor]
                if !visited[neighbor] {
                    visited[neighbor] = true
                    let expanded = neighbors(of: neighbor)
                    if expanded.count >= minSamples {
                        for candidate in expanded where queued.insert(candidate).inserted {
                            seeds.append(candidate)
                        }
                    }
                }
                if labels[neighbor] == nil || labels[neighbor] == -1 {
                    labels[neighbor] = cluster
                }
                cursor += 1
            }
        }

        var clusters: [[Int]] = []
        for cluster in 0..<nextCluster {
            clusters.append(ordered.indices.compactMap { index in
                labels[index] == cluster ? ordered[index].layer : nil
            }.sorted())
        }
        for index in ordered.indices where labels[index] == -1 {
            clusters.append([ordered[index].layer])
        }
        clusters.sort { ($0.first ?? Int.max) < ($1.first ?? Int.max) }
        return clusters
    }
}
