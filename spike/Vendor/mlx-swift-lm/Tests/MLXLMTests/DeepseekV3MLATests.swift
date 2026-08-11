import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import MLXNN
import Testing

struct DeepseekV3MLATests {
    @Test("DeepseekV3 affine projection conversion admits only MLX-supported bit widths")
    func deepseekV3AffineProjectionConversionAdmitsOnlySupportedBitWidths() {
        for bits in [2, 3, 4, 5, 6, 8] {
            #expect(deepseekV3MLASupportsAffineBits(bits))
        }
        for bits in [0, 1, 7, 9, 16] {
            #expect(!deepseekV3MLASupportsAffineBits(bits))
        }
    }

    @Test("Derived MLA projections inherit kv_b_proj mixed-bit quantization metadata")
    func derivedMLAProjectionQuantizationPathInheritance() {
        let prefix = "model.layers.4.self_attn"
        let sourcePath = "\(prefix).kv_b_proj"
        let embedPath = "\(prefix).embed_q"
        let unembedPath = "\(prefix).unembed_out"
        let sourceQuantization = BaseConfiguration.Quantization(
            groupSize: 32, bits: 2)
        let explicitDerivedQuantization = BaseConfiguration.Quantization(
            groupSize: 64, bits: 4)
        let perLayer = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 64, bits: 4),
            perLayerQuantization: [
                sourcePath: .quantize(sourceQuantization),
                unembedPath: .quantize(explicitDerivedQuantization),
            ])

        #expect(
            deepseekV3MLAWeightQuantizationSourcePath(for: embedPath)
                == sourcePath)
        #expect(
            deepseekV3MLAWeightQuantizationSourcePath(for: unembedPath)
                == sourcePath)
        #expect(
            deepseekV3MLAWeightQuantizationSourcePath(
                for: "\(prefix).o_proj") == nil)

        let resolvedEmbed = resolvedWeightQuantizationPath(
            sanitizedPath: embedPath,
            sourcePath: sourcePath,
            perLayerQuantization: perLayer)
        let resolvedUnembed = resolvedWeightQuantizationPath(
            sanitizedPath: unembedPath,
            sourcePath: sourcePath,
            perLayerQuantization: perLayer)

        #expect(resolvedEmbed == sourcePath)
        #expect(perLayer.quantization(layer: resolvedEmbed) == sourceQuantization)
        #expect(resolvedUnembed == unembedPath)
        #expect(
            perLayer.quantization(layer: resolvedUnembed)
                == explicitDerivedQuantization)
    }

    @Test("MultiLinear transpose modes match explicit per-head matmul")
    func multiLinearTransposeModesMatchExplicitMatmul() throws {
        let weight = MLXArray(Array(stride(from: Float(-1.1), through: Float(1.2), by: 0.1)))
            .reshaped(2, 3, 4)
        let layer = MultiLinear(inputDims: 4, outputDims: 3, numHeads: 2)
        try layer.update(
            parameters: ModuleParameters.unflattened(["weight": weight]),
            verify: [])

        let forwardInput = MLXArray(Array(stride(from: Float(-0.9), through: Float(0.6), by: 0.1)))
            .reshaped(1, 2, 2, 4)
        let expectedForward = forwardInput.matmul(weight.swappedAxes(-1, -2))
        let actualForward = layer(forwardInput, transpose: true)

        #expect(allClose(actualForward, expectedForward, rtol: 0, atol: 1e-6).item(Bool.self))

        let transposeFalseInput = MLXArray(
            Array(stride(from: Float(-0.8), through: Float(0.3), by: 0.1))
        )
        .reshaped(1, 2, 2, 3)
        let expectedTransposeFalse = transposeFalseInput.matmul(weight)
        let actualTransposeFalse = layer(transposeFalseInput, transpose: false)

        #expect(
            allClose(actualTransposeFalse, expectedTransposeFalse, rtol: 0, atol: 1e-6)
                .item(Bool.self))
    }

    @Test("QuantizedMultiLinear transpose false matches dequantized reference")
    func quantizedMultiLinearTransposeFalseMatchesDequantizedReference() {
        let sourceValues: [Float] = (0 ..< (2 * 3 * 32)).map {
            Float($0 % 31) / 16 - 0.9
        }
        let sourceWeight = MLXArray(sourceValues).reshaped(2, 3, 32)
        let layer = QuantizedMultiLinear(weight: sourceWeight, groupSize: 32, bits: 4)
        let dequantizedWeight = dequantized(
            layer.weight,
            scales: layer.scales,
            biases: layer.biases,
            groupSize: layer.groupSize,
            bits: layer.bits
        )

        let transposeFalseInput = MLXArray(
            Array(stride(from: Float(-0.6), through: Float(0.5), by: 0.1))
        )
        .reshaped(1, 2, 2, 3)
        let expected = transposeFalseInput.matmul(dequantizedWeight)
        let actual = layer(transposeFalseInput, transpose: false)

        #expect(allClose(actual, expected, rtol: 0.05, atol: 0.05).item(Bool.self))
    }

    @Test("DeepseekV3Model creates one MLA cache per decoded hidden layer")
    func deepseekV3ModelNewCacheMatchesDecodedLayerCount() throws {
        let config = try tinyDeepseekV3Configuration(numHiddenLayers: 3)
        let model = DeepseekV3Model(config)

        #expect(model.kvHeads == [1, 1, 1])

        let cache = model.newCache(parameters: nil)
        #expect(cache.count == config.numHiddenLayers)
        #expect(cache.allSatisfy { $0 is KVCacheSimple })
    }

    @Test("DeepseekV3 scalar cache admission rejects rotating and quantized policies")
    func deepseekV3ScalarCacheAdmissionIsFP16OrBF16Only() throws {
        try validateDeepseekV3MLACacheParameters(nil)
        try validateDeepseekV3MLACacheStorage(
            latent: MLXArray.zeros([1, 1, 1, 4], dtype: .float16),
            rotary: MLXArray.zeros([1, 1, 1, 2], dtype: .float16))
        try validateDeepseekV3MLACacheStorage(
            latent: MLXArray.zeros([1, 1, 1, 4], dtype: .bfloat16),
            rotary: MLXArray.zeros([1, 1, 1, 2], dtype: .bfloat16))

        #expect(throws: DeepseekV3MLACachePolicyError.self) {
            try validateDeepseekV3MLACacheParameters(
                GenerateParameters(maxKVSize: 128))
        }
        #expect(throws: DeepseekV3MLACachePolicyError.self) {
            try validateDeepseekV3MLACacheParameters(
                GenerateParameters(kvBits: 4))
        }
        #expect(throws: DeepseekV3MLACachePolicyError.self) {
            try validateDeepseekV3MLACacheParameters(
                GenerateParameters(kvScheme: "affine4"))
        }
        #expect(throws: DeepseekV3MLACachePolicyError.self) {
            try validateDeepseekV3MLACacheStorage(
                latent: MLXArray.zeros([1, 1, 1, 4], dtype: .float32),
                rotary: MLXArray.zeros([1, 1, 1, 2], dtype: .float32))
        }
        #expect(throws: DeepseekV3MLACachePolicyError.self) {
            try validateDeepseekV3MLACacheStorage(
                latent: MLXArray.zeros([1, 1, 1, 4], dtype: .float16),
                rotary: MLXArray.zeros([1, 1, 1, 2], dtype: .bfloat16))
        }
    }

    @Test("DeepseekV3 direct cache admission rejects rotating state before mutation")
    func deepseekV3DirectCacheAdmissionRejectsRotatingBeforeMutation() throws {
        let ordinary = KVCacheSimple()
        try validateDeepseekV3MLACacheInstance(ordinary)
        try validateDeepseekV3MLACacheInstances(
            [ordinary, KVCacheSimple()],
            expectedLayerCount: 2)

        let rotating = RotatingKVCache(maxSize: 16, keep: 4)
        #expect(rotating.offset == 0)
        #expect(throws: DeepseekV3MLACachePolicyError.self) {
            try validateDeepseekV3MLACacheInstance(rotating)
        }
        #expect(throws: DeepseekV3MLACachePolicyError.self) {
            try validateDeepseekV3MLACacheInstances(
                [ordinary, rotating],
                expectedLayerCount: 2)
        }
        #expect(throws: DeepseekV3MLACachePolicyError.self) {
            try validateDeepseekV3MLACacheInstances(
                [ordinary],
                expectedLayerCount: 2)
        }
        #expect(ordinary.offset == 0)
        #expect(ordinary.innerState().isEmpty)
        #expect(rotating.offset == 0)
        #expect(rotating.innerState().isEmpty)
    }

    @Test("Float kv_b projection converts to exact absorbed per-head layouts")
    func floatKVProjectionConversionMatchesExactLayout() throws {
        let config = try tinyDeepseekV3Configuration(numHiddenLayers: 1)
        let prefix = "model.layers.0.self_attn"
        let source = MLXArray((0 ..< 32).map(Float.init)).reshaped(8, 4)

        let converted = try convertDeepseekV3MLAProjectionWeights(
            ["\(prefix).kv_b_proj.weight": source],
            configuration: config)

        #expect(converted["\(prefix).kv_b_proj.weight"] == nil)
        let embed = try #require(converted["\(prefix).embed_q.weight"])
        let unembed = try #require(converted["\(prefix).unembed_out.weight"])
        #expect(embed.shape == [2, 4, 2])
        #expect(unembed.shape == [2, 2, 4])

        let perHead = source.reshaped(2, 4, 4)
        let expectedEmbed = perHead[0..., ..<2, 0...].swappedAxes(-1, -2)
        let expectedUnembed = perHead[0..., 2..., 0...]
        #expect(allClose(embed, expectedEmbed, rtol: 0, atol: 0).item(Bool.self))
        #expect(allClose(unembed, expectedUnembed, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test("Affine kv_b projection preserves absorbed values and metadata")
    func affineKVProjectionConversionMatchesDequantizedLayout() throws {
        let config = try affineDeepseekV3Configuration()
        let prefix = "model.layers.0.self_attn"
        let sourceValues: [Float] = (0 ..< (128 * 32)).map {
            Float($0 % 37) / 21 - 0.8
        }
        let source = MLXArray(sourceValues).reshaped(128, 32)
        let (packed, scales, optionalBiases) = MLX.quantized(
            source, groupSize: 32, bits: 4)
        let biases = try #require(optionalBiases)

        let converted = try convertDeepseekV3MLAProjectionWeights(
            [
                "\(prefix).kv_b_proj.weight": packed,
                "\(prefix).kv_b_proj.scales": scales,
                "\(prefix).kv_b_proj.biases": biases,
            ],
            configuration: config)

        let embedWeight = try #require(converted["\(prefix).embed_q.weight"])
        let embedScales = try #require(converted["\(prefix).embed_q.scales"])
        let embedBiases = try #require(converted["\(prefix).embed_q.biases"])
        let unembedWeight = try #require(converted["\(prefix).unembed_out.weight"])
        let unembedScales = try #require(converted["\(prefix).unembed_out.scales"])
        let unembedBiases = try #require(converted["\(prefix).unembed_out.biases"])
        let actualEmbed = dequantized(
            embedWeight,
            scales: embedScales,
            biases: embedBiases,
            groupSize: 32,
            bits: 4)
        let actualUnembed = dequantized(
            unembedWeight,
            scales: unembedScales,
            biases: unembedBiases,
            groupSize: 32,
            bits: 4)
        let sourceDequantized = dequantized(
            packed,
            scales: scales,
            biases: biases,
            groupSize: 32,
            bits: 4)
        let perHead = sourceDequantized.reshaped(2, 64, 32)
        let expectedEmbed = perHead[0..., ..<32, 0...].swappedAxes(-1, -2)
        let expectedUnembed = perHead[0..., 32..., 0...]

        #expect(actualEmbed.shape == [2, 32, 32])
        #expect(actualUnembed.shape == [2, 32, 32])
        #expect(allClose(actualEmbed, expectedEmbed, rtol: 0.1, atol: 0.1).item(Bool.self))
        #expect(allClose(actualUnembed, expectedUnembed, rtol: 0.1, atol: 0.1).item(Bool.self))
    }

    @Test("Converted projections are idempotent and malformed representations fail closed")
    func convertedProjectionIdempotenceAndErrors() throws {
        let config = try tinyDeepseekV3Configuration(numHiddenLayers: 1)
        let prefix = "model.layers.0.self_attn"
        let embed = MLXArray.zeros([2, 4, 2])
        let unembed = MLXArray.zeros([2, 2, 4])
        let alreadyConverted = [
            "\(prefix).embed_q.weight": embed,
            "\(prefix).unembed_out.weight": unembed,
        ]

        let idempotent = try convertDeepseekV3MLAProjectionWeights(
            alreadyConverted, configuration: config)
        #expect(idempotent.keys.sorted() == alreadyConverted.keys.sorted())
        #expect(idempotent["\(prefix).embed_q.weight"] === embed)
        #expect(idempotent["\(prefix).unembed_out.weight"] === unembed)

        #expect(throws: DeepseekV3MLAWeightConversionError.self) {
            _ = try convertDeepseekV3MLAProjectionWeights(
                [:], configuration: config)
        }
        #expect(throws: DeepseekV3MLAWeightConversionError.self) {
            _ = try convertDeepseekV3MLAProjectionWeights(
                [
                    "\(prefix).kv_b_proj.weight": MLXArray.zeros([8, 4]),
                    "\(prefix).kv_b_proj.scales": MLXArray.ones([8, 1]),
                ],
                configuration: config)
        }
        #expect(throws: DeepseekV3MLAWeightConversionError.self) {
            _ = try convertDeepseekV3MLAProjectionWeights(
                alreadyConverted.merging(
                    ["\(prefix).kv_b_proj.weight": MLXArray.zeros([8, 4])],
                    uniquingKeysWith: { current, _ in current }),
                configuration: config)
        }
    }

    @Test("Converted affine projections require one shared quantization geometry")
    func convertedAffineProjectionGeometryMustMatch() throws {
        let config = try affineDeepseekV3Configuration()
        let prefix = "model.layers.0.self_attn"
        let scales = MLXArray.ones([2, 32, 1], dtype: .float16)
        let biases = MLXArray.zeros([2, 32, 1], dtype: .float16)
        let mixedGeometry = [
            "\(prefix).embed_q.weight":
                MLXArray.zeros([2, 32, 4], dtype: .uint32),
            "\(prefix).embed_q.scales": scales,
            "\(prefix).embed_q.biases": biases,
            "\(prefix).unembed_out.weight":
                MLXArray.zeros([2, 32, 2], dtype: .uint32),
            "\(prefix).unembed_out.scales": scales,
            "\(prefix).unembed_out.biases": biases,
        ]

        #expect(throws: DeepseekV3MLAWeightConversionError.self) {
            _ = try convertDeepseekV3MLAProjectionWeights(
                mixedGeometry, configuration: config)
        }
    }

    @Test("Unsupported affine bit widths fail closed before projection conversion")
    func unsupportedAffineBitWidthsFailClosedBeforeProjectionConversion() throws {
        let config = try affineDeepseekV3Configuration()
        let prefix = "model.layers.0.self_attn"
        let sourceScales = MLXArray.ones([128, 1], dtype: .float16)
        let sourceBiases = MLXArray.zeros([128, 1], dtype: .float16)
        #expect(throws: DeepseekV3MLAWeightConversionError.self) {
            _ = try convertDeepseekV3MLAProjectionWeights(
                [
                    "\(prefix).kv_b_proj.weight":
                        MLXArray.zeros([128, 1], dtype: .uint32),
                    "\(prefix).kv_b_proj.scales": sourceScales,
                    "\(prefix).kv_b_proj.biases": sourceBiases,
                ],
                configuration: config)
        }

        let convertedScales = MLXArray.ones([2, 32, 1], dtype: .float16)
        let convertedBiases = MLXArray.zeros([2, 32, 1], dtype: .float16)
        #expect(throws: DeepseekV3MLAWeightConversionError.self) {
            _ = try convertDeepseekV3MLAProjectionWeights(
                [
                    "\(prefix).embed_q.weight":
                        MLXArray.zeros([2, 32, 1], dtype: .uint32),
                    "\(prefix).embed_q.scales": convertedScales,
                    "\(prefix).embed_q.biases": convertedBiases,
                    "\(prefix).unembed_out.weight":
                        MLXArray.zeros([2, 32, 1], dtype: .uint32),
                    "\(prefix).unembed_out.scales": convertedScales,
                    "\(prefix).unembed_out.biases": convertedBiases,
                ],
                configuration: config)
        }
    }

    @Test("Converted affine tensors must match the loader-selected tuple")
    func convertedAffineProjectionMustMatchLoaderQuantization() throws {
        let config = try affineDeepseekV3Configuration()
        let prefix = "model.layers.0.self_attn"
        let sourcePath = "\(prefix).kv_b_proj"
        let scales = MLXArray.ones([2, 32, 1], dtype: .float16)
        let biases = MLXArray.zeros([2, 32, 1], dtype: .float16)
        let converted = [
            "\(prefix).embed_q.weight":
                MLXArray.zeros([2, 32, 4], dtype: .uint32),
            "\(prefix).embed_q.scales": scales,
            "\(prefix).embed_q.biases": biases,
            "\(prefix).unembed_out.weight":
                MLXArray.zeros([2, 32, 4], dtype: .uint32),
            "\(prefix).unembed_out.scales": scales,
            "\(prefix).unembed_out.biases": biases,
        ]
        let mismatchedLoaderPolicy = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 32, bits: 4),
            perLayerQuantization: [
                sourcePath: .quantize(.init(groupSize: 32, bits: 2))
            ])
        let matchingLoaderPolicy = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 64, bits: 2),
            perLayerQuantization: [
                sourcePath: .quantize(.init(groupSize: 32, bits: 4))
            ])

        #expect(throws: DeepseekV3MLAWeightConversionError.self) {
            try validateDeepseekV3MLALoaderQuantization(
                weights: converted,
                configuration: config,
                quantization: nil,
                perLayerQuantization: mismatchedLoaderPolicy)
        }
        try validateDeepseekV3MLALoaderQuantization(
            weights: converted,
            configuration: config,
            quantization: nil,
            perLayerQuantization: matchingLoaderPolicy)
    }

    @Test("Packed int4 wrappers normalize completely and reject partial input")
    func packedInt4NormalizationIsCompleteAndFailClosed() throws {
        let base = "model.layers.0.self_attn.kv_b_proj."
        let packed = MLXArray([UInt8](repeating: 0x12, count: 16))
        let shape = MLXArray([Int32(8), Int32(4)])
        let scales = MLXArray.ones([8, 1])

        let normalized = try normalizeDeepseekV3PackedInt4Weights([
            base + "weight_shape": shape,
            base + "weight_packed": packed,
            base + "weight_scale": scales,
        ])

        #expect(normalized[base + "weight_shape"] == nil)
        #expect(normalized[base + "weight_packed"] == nil)
        #expect(normalized[base + "weight_scale"] == nil)
        #expect(normalized[base + "weight"]?.dtype == .uint32)
        #expect(normalized[base + "scales"] === scales)
        let biases = try #require(normalized[base + "biases"])
        #expect(allClose(biases, scales * -8, rtol: 0, atol: 0).item(Bool.self))

        #expect(throws: DeepseekV3MLAWeightConversionError.self) {
            _ = try normalizeDeepseekV3PackedInt4Weights([
                base + "weight_packed": packed,
                base + "weight_scale": scales,
            ])
        }
    }

    @Test("Absorbed MLA initial prefill matches decompressed K/V oracle and caches compact state")
    func absorbedMLAInitialPrefillMatchesDecompressedOracleAndCachesCompactState() throws {
        let fixture = try deterministicMLAFixture()
        let x = deterministicInput(batch: 2, length: 3, hiddenSize: fixture.config.hiddenSize)
        let cache = KVCacheSimple()
        cache.step = 1

        #expect(cache.offset == 0)
        let actual = fixture.attention(x, mask: .causal, cache: cache)
        let expected = decompressedMLAOracle(fixture: fixture, x: x)
        assertAllClose(actual, expected, MLATolerance.float16)
        assertCompactMLACacheState(cache, batch: 2, tokens: 3, fixture: fixture)
    }

    @Test("Absorbed MLA chunked prefill then decode matches decompressed K/V oracle")
    func absorbedMLAChunkedPrefillThenDecodeMatchesDecompressedOracle() throws {
        let fixture = try deterministicMLAFixture()
        let x = deterministicInput(batch: 2, length: 5, hiddenSize: fixture.config.hiddenSize)
        let expectedPrefix2 = decompressedMLAOracle(fixture: fixture, x: tokenSlice(x, 0 ..< 2))
        let expectedPrefix4 = decompressedMLAOracle(fixture: fixture, x: tokenSlice(x, 0 ..< 4))
        let expectedPrefix5 = decompressedMLAOracle(fixture: fixture, x: x)
        let cache = KVCacheSimple()
        cache.step = 1

        let chunk0 = tokenSlice(x, 0 ..< 2)
        let actualChunk0 = fixture.attention(chunk0, mask: .causal, cache: cache)
        assertAllClose(actualChunk0, expectedPrefix2, MLATolerance.float16)
        assertCompactMLACacheState(cache, batch: 2, tokens: 2, fixture: fixture)

        let chunk1 = tokenSlice(x, 2 ..< 4)
        let actualChunk1 = fixture.attention(chunk1, mask: .causal, cache: cache)
        assertAllClose(actualChunk1, tokenSlice(expectedPrefix4, 2 ..< 4), MLATolerance.float16)
        assertCompactMLACacheState(cache, batch: 2, tokens: 4, fixture: fixture)

        let decode = tokenSlice(x, 4 ..< 5)
        let actualDecode = fixture.attention(decode, mask: .none, cache: cache)
        assertAllClose(actualDecode, tokenSlice(expectedPrefix5, 4 ..< 5), MLATolerance.float16)
        assertCompactMLACacheState(cache, batch: 2, tokens: 5, fixture: fixture)
    }

    @Test("Absorbed MLA one-token decode matches decompressed K/V oracle")
    func absorbedMLAOneTokenDecodeMatchesDecompressedOracle() throws {
        let fixture = try deterministicMLAFixture()
        let x = deterministicInput(batch: 2, length: 4, hiddenSize: fixture.config.hiddenSize)
        let cache = KVCacheSimple()
        cache.step = 1
        _ = fixture.attention(tokenSlice(x, 0 ..< 3), mask: .causal, cache: cache)
        assertCompactMLACacheState(cache, batch: 2, tokens: 3, fixture: fixture)

        let actual = fixture.attention(tokenSlice(x, 3 ..< 4), mask: .none, cache: cache)
        let expected = tokenSlice(decompressedMLAOracle(fixture: fixture, x: x), 3 ..< 4)
        assertAllClose(actual, expected, MLATolerance.float16)
        assertCompactMLACacheState(cache, batch: 2, tokens: 4, fixture: fixture)
    }

    @Test("DeepseekV3 model logits and greedy IDs match decompressed MLA oracle")
    func deepseekV3ModelLogitsAndGreedyIDsMatchDecompressedOracle() throws {
        let fixture = try deterministicModelFixture()
        let tokenIds = MLXArray([Int32(1), Int32(5), Int32(9), Int32(3)]).reshaped(1, 4)
        let expectedFull = decompressedModelLogitOracle(fixture: fixture, tokenIds: tokenIds)

        let actualFull = fixture.model(tokenIds, cache: nil)
        assertAllClose(actualFull, expectedFull, MLATolerance.float16)
        assertGreedyIDsEqual(actualFull, expectedFull)

        let cache = fixture.model.newCache(parameters: nil)
        cache.forEach { ($0 as? KVCacheSimple)?.step = 1 }

        let prefillIds = tokenIdSlice(tokenIds, 0 ..< 3)
        let actualPrefill = fixture.model(prefillIds, cache: cache)
        let expectedPrefill = tokenSlice(expectedFull, 0 ..< 3)
        assertAllClose(actualPrefill, expectedPrefill, MLATolerance.float16)
        assertGreedyIDsEqual(actualPrefill, expectedPrefill)
        #expect(cache.first?.offset == 3)

        let decodeIds = tokenIdSlice(tokenIds, 3 ..< 4)
        let actualDecode = fixture.model(decodeIds, cache: cache)
        let expectedDecode = tokenSlice(expectedFull, 3 ..< 4)
        assertAllClose(actualDecode, expectedDecode, MLATolerance.float16)
        assertGreedyIDsEqual(actualDecode, expectedDecode)
        #expect(cache.first?.offset == 4)
    }

    @Test("Deepseek-V3 absorbed MLA capacity formula is locked")
    func deepseekV3AbsorbedMLACapacityFormulaIsLocked() {
        let decodedLayers = 61
        let kvLoraRank = 512
        let qkRopeHeadDim = 64
        let attentionHeads = 128
        let qkNopeHeadDim = 128
        let vHeadDim = 128
        let bytesPerElement = 2
        let contextTokens = 32_768
        let bytesPerKiB = 1024.0
        let bytesPerGiB = 1024.0 * 1024.0 * 1024.0

        let absorbedBytesPerLayerPerToken = (kvLoraRank + qkRopeHeadDim) * bytesPerElement
        let absorbedBytesPerToken = decodedLayers * absorbedBytesPerLayerPerToken
        let absorbedKiBPerToken = Double(absorbedBytesPerToken) / bytesPerKiB
        let absorbedGiBAt32K = Double(absorbedBytesPerToken * contextTokens) / bytesPerGiB

        let expandedKeyDimsPerLayer = attentionHeads * (qkNopeHeadDim + qkRopeHeadDim)
        let expandedValueDimsPerLayer = attentionHeads * vHeadDim
        let expandedBytesPerToken =
            decodedLayers * (expandedKeyDimsPerLayer + expandedValueDimsPerLayer)
            * bytesPerElement
        let expandedGiBAt32K = Double(expandedBytesPerToken * contextTokens) / bytesPerGiB
        let expansionRatio = expandedGiBAt32K / absorbedGiBAt32K

        #expect(absorbedKiBPerToken == 68.625)
        #expect(absorbedGiBAt32K == 2.14453125)
        #expect(expandedGiBAt32K == 152.5)
        #expect(abs(expansionRatio - 71.11111111111111) < 0.000001)
    }

    private func tinyDeepseekV3Configuration(numHiddenLayers: Int) throws -> DeepseekV3Configuration {
        let json = """
            {
              "model_type": "deepseek_v3",
              "vocab_size": 32,
              "hidden_size": 8,
              "intermediate_size": 16,
              "moe_intermediate_size": 8,
              "num_hidden_layers": \(numHiddenLayers),
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "routed_scaling_factor": 1.0,
              "kv_lora_rank": 4,
              "q_lora_rank": 4,
              "qk_rope_head_dim": 2,
              "v_head_dim": 2,
              "qk_nope_head_dim": 2,
              "norm_topk_prob": false,
              "moe_layer_freq": 1,
              "first_k_dense_replace": 100,
              "max_position_embeddings": 16,
              "rms_norm_eps": 0.000001,
              "rope_theta": 10000.0,
              "attention_bias": false
            }
            """

        return try JSONDecoder().decode(DeepseekV3Configuration.self, from: Data(json.utf8))
    }

    private struct MLATolerance {
        let dtype: DType
        let rtol: Double
        let atol: Double

        static let float16 = MLATolerance(dtype: .float16, rtol: 0.02, atol: 0.02)
    }

    private struct DeterministicMLAFixture {
        let config: DeepseekV3Configuration
        let attention: DeepseekV3Attention
        let qAWeight: MLXArray
        let qALayerNormWeight: MLXArray
        let qBWeight: MLXArray
        let kvAWeight: MLXArray
        let kvALayerNormWeight: MLXArray
        let embedQWeight: MLXArray
        let unembedOutWeight: MLXArray
        let oWeight: MLXArray
        let qALayerNormEps: Float
        let kvALayerNormEps: Float
    }

    private struct DeterministicModelFixture {
        let config: DeepseekV3Configuration
        let model: DeepseekV3Model
        let attentionFixture: DeterministicMLAFixture
        let embeddingWeight: MLXArray
        let inputLayerNormWeight: MLXArray
        let finalNormWeight: MLXArray
        let lmHeadWeight: MLXArray
    }

    private final class ZeroUnaryLayer: Module, UnaryLayer {
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            MLXArray.zeros(x.shape, dtype: x.dtype)
        }
    }

    private func deterministicMLAFixture() throws -> DeterministicMLAFixture {
        let config = try tinyDeepseekV3Configuration(numHiddenLayers: 1)
        let attention = DeepseekV3Attention(config: config)
        let qAWeight = deterministicArray(
            shape: [config.qLoraRank, config.hiddenSize], start: -0.35, step: 0.025)
        let qALayerNormWeight = deterministicArray(
            shape: [config.qLoraRank], start: 0.85, step: 0.03)
        let qBWeight = deterministicArray(
            shape: [config.numAttentionHeads * attention.qHeadDim, config.qLoraRank],
            start: -0.28,
            step: 0.021)
        let kvAWeight = deterministicArray(
            shape: [config.kvLoraRank + config.qkRopeHeadDim, config.hiddenSize],
            start: -0.32,
            step: 0.019)
        let kvALayerNormWeight = deterministicArray(
            shape: [config.kvLoraRank], start: 0.9, step: 0.02)
        let embedQWeight = deterministicArray(
            shape: [config.numAttentionHeads, config.kvLoraRank, config.qkNopeHeadDim],
            start: -0.24,
            step: 0.017)
        let unembedOutWeight = deterministicArray(
            shape: [config.numAttentionHeads, config.vHeadDim, config.kvLoraRank],
            start: -0.22,
            step: 0.015)
        let oWeight = deterministicArray(
            shape: [config.hiddenSize, config.numAttentionHeads * config.vHeadDim],
            start: -0.3,
            step: 0.023)

        try attention.update(
            parameters: ModuleParameters.unflattened([
                "q_a_proj.weight": qAWeight,
                "q_a_layernorm.weight": qALayerNormWeight,
                "q_b_proj.weight": qBWeight,
                "kv_a_proj_with_mqa.weight": kvAWeight,
                "kv_a_layernorm.weight": kvALayerNormWeight,
                "embed_q.weight": embedQWeight,
                "unembed_out.weight": unembedOutWeight,
                "o_proj.weight": oWeight,
            ]),
            verify: [])
        eval(attention)

        return DeterministicMLAFixture(
            config: config,
            attention: attention,
            qAWeight: qAWeight,
            qALayerNormWeight: qALayerNormWeight,
            qBWeight: qBWeight,
            kvAWeight: kvAWeight,
            kvALayerNormWeight: kvALayerNormWeight,
            embedQWeight: embedQWeight,
            unembedOutWeight: unembedOutWeight,
            oWeight: oWeight,
            qALayerNormEps: attention.qALayerNorm!.eps,
            kvALayerNormEps: attention.kvALayerNorm.eps)
    }

    private func deterministicModelFixture() throws -> DeterministicModelFixture {
        let attentionFixture = try deterministicMLAFixture()
        let config = attentionFixture.config
        let model = DeepseekV3Model(config)
        model.model.layers[0].mlp = ZeroUnaryLayer()

        let embeddingWeight = deterministicArray(
            shape: [config.vocabSize, config.hiddenSize], start: -0.27, step: 0.011)
        let inputLayerNormWeight = deterministicArray(
            shape: [config.hiddenSize], start: 0.94, step: 0.017)
        let finalNormWeight = deterministicArray(
            shape: [config.hiddenSize], start: 0.88, step: 0.019)
        let lmHeadWeight = deterministicArray(
            shape: [config.vocabSize, config.hiddenSize], start: -0.31, step: 0.029)

        try model.model.update(
            parameters: ModuleParameters.unflattened([
                "embed_tokens.weight": embeddingWeight,
                "layers.0.input_layernorm.weight": inputLayerNormWeight,
                "layers.0.post_attention_layernorm.weight":
                    MLXArray.ones([config.hiddenSize], dtype: .float16),
                "layers.0.self_attn.q_a_proj.weight": attentionFixture.qAWeight,
                "layers.0.self_attn.q_a_layernorm.weight": attentionFixture.qALayerNormWeight,
                "layers.0.self_attn.q_b_proj.weight": attentionFixture.qBWeight,
                "layers.0.self_attn.kv_a_proj_with_mqa.weight": attentionFixture.kvAWeight,
                "layers.0.self_attn.kv_a_layernorm.weight": attentionFixture.kvALayerNormWeight,
                "layers.0.self_attn.embed_q.weight": attentionFixture.embedQWeight,
                "layers.0.self_attn.unembed_out.weight": attentionFixture.unembedOutWeight,
                "layers.0.self_attn.o_proj.weight": attentionFixture.oWeight,
                "norm.weight": finalNormWeight,
            ]),
            verify: [])
        try model.lmHead.update(
            parameters: ModuleParameters.unflattened([
                "weight": lmHeadWeight
            ]),
            verify: [])
        eval(model)

        return DeterministicModelFixture(
            config: config,
            model: model,
            attentionFixture: attentionFixture,
            embeddingWeight: embeddingWeight,
            inputLayerNormWeight: inputLayerNormWeight,
            finalNormWeight: finalNormWeight,
            lmHeadWeight: lmHeadWeight)
    }

    private func decompressedMLAOracle(fixture: DeterministicMLAFixture, x: MLXArray) -> MLXArray {
        let attention = fixture.attention
        let batch = x.dim(0)
        let length = x.dim(1)

        let qALatent = linear(x, weight: fixture.qAWeight)
        let qANormalized = MLXFast.rmsNorm(
            qALatent, weight: fixture.qALayerNormWeight, eps: fixture.qALayerNormEps)
        var q = linear(qANormalized, weight: fixture.qBWeight)
        q = q.reshaped(batch, length, attention.numHeads, attention.qHeadDim)
            .transposed(0, 2, 1, 3)

        let splitQ = split(q, indices: [attention.qkNopeHeadDim], axis: -1)
        let qNope = splitQ[0]
        let qPe = attention.rope(splitQ[1], offset: 0)

        let compressedKvAndRotary = linear(x, weight: fixture.kvAWeight)
        let splitCompressedKv = split(
            compressedKvAndRotary, indices: [attention.kvLoraRank], axis: -1)
        let kvLatent = MLXFast.rmsNorm(
            splitCompressedKv[0],
            weight: fixture.kvALayerNormWeight,
            eps: fixture.kvALayerNormEps)
        let kvLatentForHeads = expandedDimensions(kvLatent, axis: 1)
        var kPe = splitCompressedKv[1].reshaped(
            batch, length, 1, attention.qkRopeHeadDim
        )
        .transposed(0, 2, 1, 3)
        kPe = attention.rope(kPe, offset: 0)
        let kPeForHeads = broadcast(
            kPe,
            to: [batch, attention.numHeads, length, attention.qkRopeHeadDim])

        let kNope = multiLinear(kvLatentForHeads, weight: fixture.embedQWeight, transpose: false)
        let values = multiLinear(kvLatentForHeads, weight: fixture.unembedOutWeight)
        let queries = concatenated([qNope, qPe], axis: -1)
        let keys = concatenated([kNope, kPeForHeads], axis: -1)
        let scores = matmul(queries * attention.scale, keys.swappedAxes(-1, -2))
        let mask = createCausalMask(n: length, offset: 0)
        let negativeInfinity = MLXArray(-Float.infinity).asType(scores.dtype)
        let maskedScores = MLX.where(mask, scores, negativeInfinity)
        let probabilities = softmax(maskedScores.asType(.float32), axis: -1)
            .asType(maskedScores.dtype)
        var output = matmul(probabilities, values)
        output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return linear(output, weight: fixture.oWeight)
    }

    private func decompressedModelLogitOracle(
        fixture: DeterministicModelFixture, tokenIds: MLXArray
    ) -> MLXArray {
        let hidden = fixture.embeddingWeight[tokenIds]
        let attentionInput = MLXFast.rmsNorm(
            hidden,
            weight: fixture.inputLayerNormWeight,
            eps: fixture.config.rmsNormEps)
        let attentionOutput = decompressedMLAOracle(
            fixture: fixture.attentionFixture,
            x: attentionInput)
        let residual = hidden + attentionOutput
        let normalized = MLXFast.rmsNorm(
            residual,
            weight: fixture.finalNormWeight,
            eps: fixture.config.rmsNormEps)
        return linear(normalized, weight: fixture.lmHeadWeight)
    }

    private func linear(_ x: MLXArray, weight: MLXArray) -> MLXArray {
        matmul(x, weight.T)
    }

    private func multiLinear(
        _ x: MLXArray, weight: MLXArray, transpose: Bool = true
    ) -> MLXArray {
        let rhs = transpose ? weight.swappedAxes(-1, -2) : weight
        return x.matmul(rhs)
    }

    private func deterministicInput(batch: Int, length: Int, hiddenSize: Int) -> MLXArray {
        deterministicArray(shape: [batch, length, hiddenSize], start: -0.42, step: 0.013)
    }

    private func deterministicArray(shape: [Int], start: Float, step: Float) -> MLXArray {
        let count = shape.reduce(1, *)
        let values = (0 ..< count).map { index in
            start + Float(index % 29) * step
        }
        return MLXArray(values).reshaped(shape).asType(.float16)
    }

    private func tokenSlice(_ x: MLXArray, _ range: Range<Int>) -> MLXArray {
        x[0..., range, 0...]
    }

    private func tokenIdSlice(_ x: MLXArray, _ range: Range<Int>) -> MLXArray {
        x[0..., range]
    }

    private func assertAllClose(
        _ actual: MLXArray, _ expected: MLXArray, _ tolerance: MLATolerance
    ) {
        #expect(actual.shape == expected.shape)
        #expect(actual.dtype == tolerance.dtype)
        #expect(expected.dtype == tolerance.dtype)
        #expect(
            allClose(actual, expected, rtol: tolerance.rtol, atol: tolerance.atol)
                .item(Bool.self))
    }

    private func assertGreedyIDsEqual(_ actual: MLXArray, _ expected: MLXArray) {
        let actualIDs = actual.argMax(axis: -1)
        let expectedIDs = expected.argMax(axis: -1)
        #expect(actualIDs.shape == expectedIDs.shape)
        #expect((actualIDs .== expectedIDs).all().item(Bool.self))
    }

    private func assertCompactMLACacheState(
        _ cache: KVCacheSimple, batch: Int, tokens: Int, fixture: DeterministicMLAFixture
    ) {
        #expect(cache.offset == tokens)
        let state = cache.innerState()
        #expect(state.count == 2)
        #expect(state[0].shape == [batch, 1, tokens, fixture.config.kvLoraRank])
        #expect(state[1].shape == [batch, 1, tokens, fixture.config.qkRopeHeadDim])
        #expect(state[0].dtype == MLATolerance.float16.dtype)
        #expect(state[1].dtype == MLATolerance.float16.dtype)
    }

    private func affineDeepseekV3Configuration() throws -> DeepseekV3Configuration {
        let json = """
            {
              "model_type": "deepseek_v3",
              "vocab_size": 32,
              "hidden_size": 8,
              "intermediate_size": 16,
              "moe_intermediate_size": 8,
              "num_hidden_layers": 1,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "routed_scaling_factor": 1.0,
              "kv_lora_rank": 32,
              "q_lora_rank": 4,
              "qk_rope_head_dim": 2,
              "v_head_dim": 32,
              "qk_nope_head_dim": 32,
              "norm_topk_prob": false,
              "moe_layer_freq": 1,
              "first_k_dense_replace": 100,
              "max_position_embeddings": 16,
              "rms_norm_eps": 0.000001,
              "rope_theta": 10000.0,
              "attention_bias": false
            }
            """

        return try JSONDecoder().decode(DeepseekV3Configuration.self, from: Data(json.utf8))
    }
}
