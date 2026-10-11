// Copyright © 2026 Apple Inc.
// Adapted from mlx-swift-lm c4084528 (MIT); adds the old-formula control.
//
// The Qwen3.5 and Qwen3-Next gated delta nets must normalize q and k
// like the reference l2norm, x / sqrt(sum(x^2) + 1e-6). Each test builds the
// layer output again from the layer's own modules with that l2norm and compares.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXVLM

final class GatedDeltaQKNormTests: XCTestCase {

    // One key head keeps q, k and v contiguous in every input projection.
    let hiddenSize = 32
    let numVHeads = 2
    let headKDim = 64
    let headVDim = 16
    let convKernelSize = 4

    private var configurationJSON: Data {
        Data(
            """
            {
                "model_type": "qwen3_5",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 1,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "vocab_size": 32,
                "linear_num_value_heads": \(numVHeads),
                "linear_num_key_heads": 1,
                "linear_key_head_dim": \(headKDim),
                "linear_value_head_dim": \(headVDim),
                "linear_conv_kernel_dim": \(convKernelSize),
                "rms_norm_eps": 1e-6,
                "num_experts": 0, "num_experts_per_tok": 0, "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 64, "mlp_only_layers": [],
                "moe_intermediate_size": 64,
                "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                "max_position_embeddings": 128
            }
            """.utf8)
    }

    /// Token scales from 1 to 1e-6 put sum(q^2) and sum(k^2) above, near and
    /// below the eps.
    func makeInputs() -> MLXArray {
        let scales = (0 ..< 28).map { Float(pow(10, -Double($0 / 4))) }
        let x = MLXRandom.normal([1, scales.count, hiddenSize])
        return x * MLXArray(scales).reshaped(1, -1, 1)
    }

    /// Eight tokens at scale 1, for the old-formula control.
    private func makeUnitScaleInputs() -> MLXArray {
        MLXRandom.normal([1, 8, hiddenSize])
    }

    func l2norm(_ x: MLXArray) -> MLXArray {
        x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6)
    }

    /// The layer output with a given q/k normalization. `mixedQKV` is the conv
    /// input, laid out as [q, k, v]. `normalizeKey` returns the k-scaled
    /// normalization; q gets one more factor of invScale, as in the layers.
    func referenceOutput(
        mixedQKV: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray,
        conv1d: Conv1d, aLog: MLXArray, dtBias: MLXArray,
        norm: (MLXArray, MLXArray) -> MLXArray, outProj: Linear,
        normalizeKey: ((MLXArray) -> MLXArray)? = nil
    ) -> MLXArray {
        let B = mixedQKV.dim(0)
        let S = mixedQKV.dim(1)
        let invScale = pow(Float(headKDim), -0.5)
        let normalizeKey = normalizeKey ?? { [self] in l2norm($0) }
        let convState = MLXArray.zeros([B, convKernelSize - 1, mixedQKV.dim(-1)])
        let convOut = silu(conv1d(concatenated([convState, mixedQKV], axis: 1)))
        let parts = MLX.split(convOut, indices: [headKDim, 2 * headKDim], axis: -1)
        let q = normalizeKey(parts[0].reshaped(B, S, 1, headKDim)) * invScale
        let k = normalizeKey(parts[1].reshaped(B, S, 1, headKDim))
        let v = parts[2].reshaped(B, S, numVHeads, headVDim)
        let (out, _) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias)
        let gated = norm(out, z.reshaped(B, S, numVHeads, headVDim))
        return outProj(gated.reshaped(B, S, -1))
    }

    /// Compares per token, because the small tokens have small outputs.
    func assertMatches(
        _ name: String, _ output: MLXArray, _ reference: MLXArray, tolerance: Float = 1e-4,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let error = MLX.abs(output - reference).max(axis: -1) / MLX.abs(reference).max(axis: -1)
        let worst = error.max().item(Float.self)
        print("GatedDeltaQKNormTests \(name): worst per-token relative error = \(worst)")
        XCTAssertLessThan(worst, tolerance, "max per-token relative error", file: file, line: line)
    }

    func testQwen35() throws {
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: configurationJSON)
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let gdn = Qwen35GatedDeltaNet(config)
            let x = makeInputs()
            let reference = referenceOutput(
                mixedQKV: gdn.inProjQKV(x), z: gdn.inProjZ(x),
                b: gdn.inProjB(x), a: gdn.inProjA(x),
                conv1d: gdn.conv1d, aLog: gdn.aLog, dtBias: gdn.dtBias,
                norm: { gdn.norm($0, gate: $1) }, outProj: gdn.outProj)
            assertMatches("Qwen35", gdn(x), reference)
        }
    }

    /// C2 positive control. At token scale 1 the fixed layer must still match the
    /// old formula, invScale * rmsNorm(x, eps: 1e-6), which the fix moves only
    /// where sum(x^2) is comparable to the eps.
    func testQwen35AtUnitScaleMatchesOldFormula() throws {
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: configurationJSON)
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let gdn = Qwen35GatedDeltaNet(config)
            // The default conv init is small (about 0.04), which leaves sum(q^2) and
            // sum(k^2) of a few 1e-3 and is not the regime of a trained checkpoint.
            // Scale it so the control runs where the eps is negligible.
            try? gdn.update(
                parameters: ModuleParameters.unflattened([
                    "conv1d.weight": gdn.conv1d.weight * 100
                ]),
                verify: [])
            let x = makeUnitScaleInputs()
            let invScale = pow(Float(headKDim), -0.5)
            // Anti-vacuity: q and k must sit far above the eps in this control.
            let convInput = concatenated(
                [MLXArray.zeros([1, convKernelSize - 1, gdn.convDim]), gdn.inProjQKV(x)], axis: 1)
            let convOut = silu(gdn.conv1d(convInput))
            let qk = convOut[.ellipsis, ..<(2 * headKDim)].reshaped(1, -1, 2, headKDim)
            let minSumSquares = (qk * qk).sum(axis: -1).min().item(Float.self)
            XCTAssertGreaterThan(minSumSquares, 1.0, "control must run far above the eps")
            let reference = referenceOutput(
                mixedQKV: gdn.inProjQKV(x), z: gdn.inProjZ(x),
                b: gdn.inProjB(x), a: gdn.inProjA(x),
                conv1d: gdn.conv1d, aLog: gdn.aLog, dtBias: gdn.dtBias,
                norm: { gdn.norm($0, gate: $1) }, outProj: gdn.outProj,
                normalizeKey: {
                    MLXArray(invScale) * MLXFast.rmsNorm($0, weight: MLXArray.mlxNone, eps: 1e-6)
                })
            assertMatches("Qwen35 unit scale vs old formula", gdn(x), reference, tolerance: 1e-5)
        }
    }

    func testQwen35VLM() throws {
        let config = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self, from: configurationJSON)
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let gdn = Qwen35Language.GatedDeltaNet(config)
            let x = makeInputs()
            let reference = referenceOutput(
                mixedQKV: gdn.inProjQKV(x), z: gdn.inProjZ(x),
                b: gdn.inProjB(x), a: gdn.inProjA(x),
                conv1d: gdn.conv1d, aLog: gdn.aLog, dtBias: gdn.dtBias,
                norm: { gdn.norm($0, gate: $1) }, outProj: gdn.outProj)
            assertMatches("Qwen35VLM", gdn(x), reference)
        }
    }

    func testQwen3Next() throws {
        let config = try JSONDecoder().decode(
            Qwen3NextConfiguration.self, from: configurationJSON)
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let gdn = Qwen3NextGatedDeltaNet(config)
            let x = makeInputs()
            // With one key head, in_proj_qkvz is [q, k, v, z] and in_proj_ba is [b, a].
            let qkvz = gdn.inProjQKVZ(x)
            let ba = gdn.inProjBA(x)
            let qkvDim = 2 * headKDim + numVHeads * headVDim
            let reference = referenceOutput(
                mixedQKV: qkvz[.ellipsis, ..<qkvDim], z: qkvz[.ellipsis, qkvDim...],
                b: ba[.ellipsis, ..<numVHeads], a: ba[.ellipsis, numVHeads...],
                conv1d: gdn.conv1d, aLog: gdn.aLog, dtBias: gdn.dtBias,
                norm: { gdn.norm($0, gate: $1) }, outProj: gdn.outProj)
            assertMatches("Qwen3Next", gdn(x), reference)
        }
    }
}
