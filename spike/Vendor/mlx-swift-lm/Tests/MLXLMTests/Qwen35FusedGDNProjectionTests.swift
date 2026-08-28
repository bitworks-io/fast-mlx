// Copyright © 2026 Apple Inc.

import Foundation
import Metal
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXVLM

final class Qwen35FusedGDNProjectionTests: XCTestCase {

    private func configurationJSON(
        hiddenLayers: Int = 2,
        fullAttentionInterval: Int = 2
    ) -> String {
        """
        {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "num_hidden_layers": \(hiddenLayers),
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 32,
            "full_attention_interval": \(fullAttentionInterval),
            "num_experts": 0,
            "num_experts_per_tok": 0
        }
        """
    }

    private func llmConfiguration(
        hiddenLayers: Int = 2,
        fullAttentionInterval: Int = 2
    ) throws -> Qwen35TextConfiguration {
        try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                configurationJSON(
                    hiddenLayers: hiddenLayers,
                    fullAttentionInterval: fullAttentionInterval
                ).utf8))
    }

    private func vlmConfiguration(
        hiddenLayers: Int = 2,
        fullAttentionInterval: Int = 2
    ) throws -> MLXVLM.Qwen35Configuration.TextConfiguration {
        try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(
                configurationJSON(
                    hiddenLayers: hiddenLayers,
                    fullAttentionInterval: fullAttentionInterval
                ).utf8))
    }

    private func withGPU<R>(_ body: () throws -> R) throws -> R {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Qwen GDN fusion parity requires an available Metal device")
        }
        return try Device.withDefaultDevice(.gpu) {
            try body()
        }
    }

    private func assertBitIdentical(
        _ actual: MLXArray,
        _ expected: MLXArray,
        _ label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.dtype, expected.dtype, "\(label): dtype", file: file, line: line)
        XCTAssertEqual(actual.shape, expected.shape, "\(label): shape", file: file, line: line)
        let actualValues = actual.asType(.float32).asArray(Float.self)
        let expectedValues = expected.asType(.float32).asArray(Float.self)
        let mismatches = zip(actualValues, expectedValues).filter {
            $0.bitPattern != $1.bitPattern
        }.count
        XCTAssertEqual(
            mismatches,
            0,
            "\(label): \(mismatches)/\(actualValues.count) values differ",
            file: file,
            line: line)
    }

    private func assertParameterTopologyPreserved(
        _ layer: Qwen35GatedDeltaNet,
        before keysBefore: Set<String>,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let keysAfter = Set(layer.parameters().flattened().map(\.0))
        XCTAssertEqual(keysAfter, keysBefore, file: file, line: line)
        XCTAssertTrue(keysAfter.contains("in_proj_qkv.weight"), file: file, line: line)
        XCTAssertTrue(keysAfter.contains("in_proj_z.weight"), file: file, line: line)
        XCTAssertTrue(keysAfter.contains("in_proj_b.weight"), file: file, line: line)
        XCTAssertTrue(keysAfter.contains("in_proj_a.weight"), file: file, line: line)
        XCTAssertFalse(keysAfter.contains { $0.contains("fused") }, file: file, line: line)
    }

    private func quantize(_ layer: Qwen35GatedDeltaNet) throws {
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(
                    QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(
                    QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(
                    QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(
                    QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]),
            verify: [])
    }

    private func quantize(_ layer: Qwen35Language.GatedDeltaNet) throws {
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(
                    QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(
                    QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(
                    QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(
                    QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]),
            verify: [])
    }

    private func assertTextFusionFallsBackAfterReplacingProjection(
        _ key: String,
        replacement: (Qwen35GatedDeltaNet, QuantizedLinear) throws -> Linear,
        _ label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let layer = Qwen35GatedDeltaNet(try llmConfiguration())
        try quantize(layer)
        let source = try XCTUnwrap(layer.inProjZ as? QuantizedLinear, file: file, line: line)
        try layer.update(
            modules: ModuleChildren(values: [
                key: .value(try replacement(layer, source))
            ]),
            verify: [])

        layer.fusedInputProjectionEnabled = true
        XCTAssertFalse(
            try layer.prepareFusedInputProjection(), label, file: file, line: line)
        XCTAssertFalse(layer.hasFusedInputProjection, label, file: file, line: line)
    }

    func testDefaultOffForwardDoesNotPrepareOrMutateTextTopology() throws {
        try withGPU {
            let layer = Qwen35GatedDeltaNet(try llmConfiguration())
            try quantize(layer)
            let modulesBefore = [
                ObjectIdentifier(layer.inProjQKV),
                ObjectIdentifier(layer.inProjZ),
                ObjectIdentifier(layer.inProjB),
                ObjectIdentifier(layer.inProjA),
            ]

            let output = layer(MLXRandom.normal([1, 3, 64]).asType(.bfloat16))
            eval(output)

            XCTAssertFalse(layer.fusedInputProjectionEnabled)
            XCTAssertFalse(layer.hasFusedInputProjection)
            XCTAssertEqual(
                modulesBefore,
                [
                    ObjectIdentifier(layer.inProjQKV),
                    ObjectIdentifier(layer.inProjZ),
                    ObjectIdentifier(layer.inProjB),
                    ObjectIdentifier(layer.inProjA),
                ])
        }
    }

    func testOptInTextFusionMatchesUnfusedProjectionForwardAndCheckpointCache() throws {
        try withGPU {
            MLXRandom.seed(31)
            let layer = Qwen35GatedDeltaNet(try llmConfiguration())
            try quantize(layer)
            let keysBefore = Set(layer.parameters().flattened().map(\.0))
            let input = MLXRandom.normal([1, 5, 64]).asType(.bfloat16)

            layer.fusedInputProjectionEnabled = false
            let referenceProjections = layer.projectInputs(input, batch: 1, sequence: 5)
            let referenceCache = MambaCache()
            let referenceOutput = layer(input, cache: referenceCache, checkpointAfter: 2)
            eval(
                referenceProjections.qkv,
                referenceProjections.z,
                referenceProjections.b,
                referenceProjections.a,
                referenceOutput)

            layer.fusedInputProjectionEnabled = true
            XCTAssertTrue(try layer.prepareFusedInputProjection())
            let fusedProjections = layer.projectInputs(input, batch: 1, sequence: 5)
            let fusedCache = MambaCache()
            let fusedOutput = layer(input, cache: fusedCache, checkpointAfter: 2)
            eval(
                fusedProjections.qkv,
                fusedProjections.z,
                fusedProjections.b,
                fusedProjections.a,
                fusedOutput)

            XCTAssertTrue(layer.hasFusedInputProjection)
            assertParameterTopologyPreserved(layer, before: keysBefore)
            assertBitIdentical(fusedProjections.qkv, referenceProjections.qkv, "text qkv")
            assertBitIdentical(fusedProjections.z, referenceProjections.z, "text z")
            assertBitIdentical(fusedProjections.b, referenceProjections.b, "text b")
            assertBitIdentical(fusedProjections.a, referenceProjections.a, "text a")
            assertBitIdentical(fusedOutput, referenceOutput, "text forward")
            XCTAssertEqual(
                fusedCache.hasSpeculativeCheckpoint,
                referenceCache.hasSpeculativeCheckpoint)
            XCTAssertEqual(
                fusedCache.restoreSpeculativeCheckpoint(),
                referenceCache.restoreSpeculativeCheckpoint())
            for (actual, expected) in zip(fusedCache.state, referenceCache.state) {
                assertBitIdentical(actual, expected, "text checkpoint state")
            }
        }
    }

    func testOptInVLMFusionMatchesUnfusedProjectionForwardAndCheckpointCache() throws {
        try withGPU {
            MLXRandom.seed(32)
            let layer = Qwen35Language.GatedDeltaNet(try vlmConfiguration())
            try quantize(layer)
            let input = MLXRandom.normal([1, 5, 64]).asType(.bfloat16)

            layer.fusedInputProjectionEnabled = false
            let referenceProjections = layer.projectInputs(input, batch: 1, sequence: 5)
            let referenceCache = MambaCache()
            let referenceOutput = layer(input, cache: referenceCache, checkpointAfter: 2)
            eval(
                referenceProjections.qkv,
                referenceProjections.z,
                referenceProjections.b,
                referenceProjections.a,
                referenceOutput)

            layer.fusedInputProjectionEnabled = true
            XCTAssertTrue(try layer.prepareFusedInputProjection())
            let fusedProjections = layer.projectInputs(input, batch: 1, sequence: 5)
            let fusedCache = MambaCache()
            let fusedOutput = layer(input, cache: fusedCache, checkpointAfter: 2)
            eval(
                fusedProjections.qkv,
                fusedProjections.z,
                fusedProjections.b,
                fusedProjections.a,
                fusedOutput)

            XCTAssertTrue(layer.hasFusedInputProjection)
            assertBitIdentical(fusedProjections.qkv, referenceProjections.qkv, "vlm qkv")
            assertBitIdentical(fusedProjections.z, referenceProjections.z, "vlm z")
            assertBitIdentical(fusedProjections.b, referenceProjections.b, "vlm b")
            assertBitIdentical(fusedProjections.a, referenceProjections.a, "vlm a")
            assertBitIdentical(fusedOutput, referenceOutput, "vlm forward")
            XCTAssertEqual(
                fusedCache.restoreSpeculativeCheckpoint(),
                referenceCache.restoreSpeculativeCheckpoint())
            for (actual, expected) in zip(fusedCache.state, referenceCache.state) {
                assertBitIdentical(actual, expected, "vlm checkpoint state")
            }
        }
    }

    func testIncompatibleCustomAdapterAndMixedProjectionPoliciesFallBack() throws {
        try withGPU {
            let layer = Qwen35GatedDeltaNet(try llmConfiguration())
            try quantize(layer)
            let adapted = try XCTUnwrap(
                LoRALinear.from(linear: layer.inProjQKV, rank: 4, scale: 1) as? Linear)
            try layer.update(
                modules: ModuleChildren(values: ["in_proj_qkv": .value(adapted)]),
                verify: [])

            layer.fusedInputProjectionEnabled = true
            XCTAssertFalse(try layer.prepareFusedInputProjection())
            XCTAssertFalse(layer.hasFusedInputProjection)
            XCTAssertTrue(layer.inProjQKV is QLoRALinear)

            let mixed = Qwen35GatedDeltaNet(try llmConfiguration())
            try mixed.update(
                modules: ModuleChildren(values: [
                    "in_proj_qkv": .value(
                        QuantizedLinear(mixed.inProjQKV, groupSize: 32, bits: 4)),
                    "in_proj_z": .value(mixed.inProjZ),
                    "in_proj_b": .value(
                        QuantizedLinear(mixed.inProjB, groupSize: 32, bits: 4)),
                    "in_proj_a": .value(
                        QuantizedLinear(mixed.inProjA, groupSize: 32, bits: 4)),
                ]),
                verify: [])
            mixed.fusedInputProjectionEnabled = true
            XCTAssertFalse(try mixed.prepareFusedInputProjection())
            XCTAssertFalse(mixed.hasFusedInputProjection)

            let incompatible = Qwen35GatedDeltaNet(try llmConfiguration())
            try incompatible.update(
                modules: ModuleChildren(values: [
                    "in_proj_qkv": .value(
                        QuantizedLinear(incompatible.inProjQKV, groupSize: 32, bits: 4)),
                    "in_proj_z": .value(
                        QuantizedLinear(incompatible.inProjZ, groupSize: 32, bits: 4)),
                    "in_proj_b": .value(
                        QuantizedLinear(incompatible.inProjB, groupSize: 32, bits: 8)),
                    "in_proj_a": .value(
                        QuantizedLinear(incompatible.inProjA, groupSize: 32, bits: 4)),
                ]),
                verify: [])
            incompatible.fusedInputProjectionEnabled = true
            XCTAssertFalse(try incompatible.prepareFusedInputProjection())
            XCTAssertFalse(incompatible.hasFusedInputProjection)
        }
    }

    func testQuantizedProjectionCompatibilityMismatchesFallBack() throws {
        try withGPU {
            try assertTextFusionFallsBackAfterReplacingProjection(
                "in_proj_z",
                replacement: { _, source in
                    QuantizedLinear(
                        Linear(64, source.shape.0, bias: false),
                        groupSize: 64,
                        bits: 4)
                },
                "group size mismatch")

            try assertTextFusionFallsBackAfterReplacingProjection(
                "in_proj_z",
                replacement: { _, source in
                    QuantizedLinear(
                        weight: source.weight,
                        bias: nil,
                        scales: source.scales,
                        biases: source.biases,
                        groupSize: source.groupSize,
                        bits: source.bits,
                        mode: .mxfp4)
                },
                "mode mismatch")

            try assertTextFusionFallsBackAfterReplacingProjection(
                "in_proj_z",
                replacement: { _, source in
                    QuantizedLinear(
                        weight: source.weight.asType(.float32),
                        bias: nil,
                        scales: source.scales,
                        biases: source.biases,
                        groupSize: source.groupSize,
                        bits: source.bits,
                        mode: source.mode)
                },
                "weight dtype mismatch")

            try assertTextFusionFallsBackAfterReplacingProjection(
                "in_proj_z",
                replacement: { _, source in
                    QuantizedLinear(
                        weight: source.weight,
                        bias: nil,
                        scales: source.scales.asType(.float16),
                        biases: source.biases,
                        groupSize: source.groupSize,
                        bits: source.bits,
                        mode: source.mode)
                },
                "scales dtype mismatch")

            try assertTextFusionFallsBackAfterReplacingProjection(
                "in_proj_z",
                replacement: { _, source in
                    let biases = try XCTUnwrap(source.biases)
                    return QuantizedLinear(
                        weight: source.weight,
                        bias: nil,
                        scales: source.scales,
                        biases: biases.asType(.float16),
                        groupSize: source.groupSize,
                        bits: source.bits,
                        mode: source.mode)
                },
                "quantization biases dtype mismatch")

            try assertTextFusionFallsBackAfterReplacingProjection(
                "in_proj_z",
                replacement: { _, source in
                    QuantizedLinear(
                        weight: source.weight,
                        bias: nil,
                        scales: source.scales,
                        biases: nil,
                        groupSize: source.groupSize,
                        bits: source.bits,
                        mode: source.mode)
                },
                "quantization bias presence mismatch")

            try assertTextFusionFallsBackAfterReplacingProjection(
                "in_proj_z",
                replacement: { _, source in
                    QuantizedLinear(
                        weight: source.weight,
                        bias: MLXArray.zeros([source.shape.0]),
                        scales: source.scales,
                        biases: source.biases,
                        groupSize: source.groupSize,
                        bits: source.bits,
                        mode: source.mode)
                },
                "regular bias presence mismatch")

            try assertTextFusionFallsBackAfterReplacingProjection(
                "in_proj_b",
                replacement: { _, _ in
                    QuantizedLinear(Linear(64, 5, bias: false), groupSize: 32, bits: 4)
                },
                "wrong output row geometry")
        }
    }

    func testPreparationFailureRollsBackAndDoesNotRetryUntilInvalidated() throws {
        try withGPU {
            enum ExpectedFailure: Error { case install }

            let layer = Qwen35GatedDeltaNet(try llmConfiguration())
            try quantize(layer)
            let linears = [layer.inProjQKV, layer.inProjZ, layer.inProjB, layer.inProjA]
            let cache = FusedQuantizedLinearProjectionCache()
            var installAttempts = 0
            var installedModules: [[Linear]] = []

            XCTAssertThrowsError(
                try cache.prepare(enabled: true, linears: linears) { modules in
                    installAttempts += 1
                    installedModules.append(modules)
                    if installAttempts == 1 {
                        throw ExpectedFailure.install
                    }
                }
            ) { error in
                let preparationError = error as? FusedQuantizedLinearPreparationError
                XCTAssertNotNil(preparationError)
                XCTAssertNil(preparationError?.rollbackError)
            }
            XCTAssertEqual(installAttempts, 2)
            XCTAssertEqual(
                installedModules[1].map(ObjectIdentifier.init),
                linears.map(ObjectIdentifier.init))

            XCTAssertFalse(
                try cache.prepare(enabled: true, linears: linears) { _ in
                    installAttempts += 1
                })
            XCTAssertEqual(installAttempts, 2)

            cache.invalidate()
            XCTAssertTrue(
                try cache.prepare(enabled: true, linears: linears) { _ in
                    installAttempts += 1
                })
            XCTAssertEqual(installAttempts, 3)
        }
    }

    func testParameterAndModuleMutationInvalidateFusion() throws {
        try withGPU {
            let model = Qwen35TextModel(try llmConfiguration())
            let layer = try XCTUnwrap(
                model.modules().compactMap { $0 as? Qwen35GatedDeltaNet }.first)
            try quantize(layer)
            layer.fusedInputProjectionEnabled = true
            try model.prepare()
            XCTAssertTrue(layer.hasFusedInputProjection)

            let replacement =
                layer.inProjQKV.weight
                + MLXArray.zeros(
                    layer.inProjQKV.weight.shape, dtype: layer.inProjQKV.weight.dtype)
            try layer.update(
                parameters: ModuleParameters.unflattened(["in_proj_qkv.weight": replacement]),
                verify: [])
            XCTAssertFalse(layer.hasFusedInputProjection)
            XCTAssertTrue(try layer.prepareFusedInputProjection())

            let adapted = try XCTUnwrap(
                LoRALinear.from(linear: layer.inProjQKV, rank: 4, scale: 1) as? Linear)
            try layer.update(
                modules: ModuleChildren(values: ["in_proj_qkv": .value(adapted)]),
                verify: [])
            XCTAssertFalse(layer.hasFusedInputProjection)
            XCTAssertFalse(try layer.prepareFusedInputProjection())
        }
    }

    func testModelPrepareFailureRestoresEarlierPreparedLayers() throws {
        try withGPU {
            enum ExpectedFailure: Error { case prepare }

            let model = Qwen35TextModel(
                try llmConfiguration(hiddenLayers: 3, fullAttentionInterval: 4))
            let layers = model.linearAttentionLayers
            guard layers.count >= 2 else {
                XCTFail("expected at least two linear-attention layers, found \(layers.count)")
                return
            }
            for layer in layers {
                try quantize(layer)
                layer.fusedInputProjectionEnabled = true
            }
            let firstLayerOriginalModules = layers[0].inputProjectionLinears
                .map(ObjectIdentifier.init)

            XCTAssertThrowsError(
                try prepareQwen35GatedDeltaNetLayers(Array(layers.prefix(2))) { layer in
                    if layer === layers[1] {
                        throw ExpectedFailure.prepare
                    }
                    return try layer.prepareFusedInputProjectionResult()
                }
            ) { error in
                XCTAssertTrue(error is ExpectedFailure)
            }

            XCTAssertFalse(layers[0].hasFusedInputProjection)
            XCTAssertFalse(layers[1].hasFusedInputProjection)
            XCTAssertEqual(
                layers[0].inputProjectionLinears.map(ObjectIdentifier.init),
                firstLayerOriginalModules)
        }
    }

    func testModelPrepareFailureDoesNotDeoptimizePreviouslyReadyLayer() throws {
        try withGPU {
            enum ExpectedFailure: Error { case prepare }

            let model = Qwen35TextModel(
                try llmConfiguration(hiddenLayers: 3, fullAttentionInterval: 4))
            let layers = model.linearAttentionLayers
            guard layers.count >= 2 else {
                XCTFail("expected at least two linear-attention layers, found \(layers.count)")
                return
            }
            for layer in layers {
                try quantize(layer)
                layer.fusedInputProjectionEnabled = true
            }

            XCTAssertTrue(try layers[0].prepareFusedInputProjection())
            XCTAssertTrue(layers[0].hasFusedInputProjection)
            let readyLayerModules = layers[0].inputProjectionLinears.map(ObjectIdentifier.init)

            XCTAssertThrowsError(
                try prepareQwen35GatedDeltaNetLayers(Array(layers.prefix(2))) { layer in
                    if layer === layers[1] {
                        throw ExpectedFailure.prepare
                    }
                    return try layer.prepareFusedInputProjectionResult()
                }
            ) { error in
                XCTAssertTrue(error is ExpectedFailure)
            }

            XCTAssertTrue(layers[0].hasFusedInputProjection)
            XCTAssertFalse(layers[1].hasFusedInputProjection)
            XCTAssertEqual(
                layers[0].inputProjectionLinears.map(ObjectIdentifier.init),
                readyLayerModules)
        }
    }
}
