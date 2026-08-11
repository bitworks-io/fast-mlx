import Foundation
import MLX

/// TurboQuant quantize/dequantize (arXiv:2504.19874) over batched row-vectors `[n, d]`.
///
/// `_mse` (Algorithm 1): rotate by the fixed Haar Π, then snap each coordinate to the nearest
/// 1/√d-scaled Lloyd-Max centroid. Optimal for reconstruction MSE, but its inner-product estimate
/// is biased — the `_prod` variant (Task 4) adds the QJL sign-residual to fix that.
public enum TurboQuantCodec {
    /// Per-element nearest-centroid indices of the rotated vectors. `x` is `[n, d]`;
    /// returns `[n, d]` centroid indices (values in `0..<2^baseBits`).
    public static func quantizeMSE(_ x: MLXArray, params p: TurboQuantParams) -> MLXArray {
        let y = x.matmul(p.rotation.transposed())  // rows: y = Π·x  → [n, d]
        // |y[..., None] − centroids| → argmin over the centroid axis.
        let diff = (y.expandedDimensions(axis: -1) - p.scaledCentroids).abs()
        return diff.argMin(axis: -1)
    }

    /// Reconstruct `[n, d]` vectors from centroid indices: gather the LUT, undo the rotation
    /// (Π orthogonal ⇒ inverse = Πᵀ; rows multiply by Π on the right).
    public static func dequantizeMSE(_ idx: MLXArray, params p: TurboQuantParams) -> MLXArray {
        let yq = p.scaledCentroids[idx]  // gather → [n, d]
        return yq.matmul(p.rotation)
    }
}

/// Per-vector `_prod` stored state (TurboQuant Algorithm 2): base-bit LUT indices, the QJL
/// sign-residual, and the residual norm γ. No per-group scale/zero — this IS the metadata.
public struct TurboQuantCode {
    public let idx: MLXArray  // [n, d] base-bit centroid indices
    public let signs: MLXArray  // [n, d] ±1 (1 bit/element)
    public let norms: MLXArray  // [n, 1] γ = ‖r‖₂ per row
    /// [n, 1] per-row input L2 norm ‖x‖ (paper §1.1). Set by `quantizeProdNormalized`;
    /// nil for raw unit-norm codes from `quantizeProd`. Distinct from `norms` (the residual γ).
    public let xNorm: MLXArray?

    public init(idx: MLXArray, signs: MLXArray, norms: MLXArray, xNorm: MLXArray? = nil) {
        self.idx = idx
        self.signs = signs
        self.norms = norms
        self.xNorm = xNorm
    }
}

extension TurboQuantCodec {
    /// Algorithm 2 quantize: base `_mse` indices, then the QJL 1-bit sketch of the residual
    /// `r = x − dequantMSE(idx)`: `signs = sign(S·r)` per row, `γ = ‖r‖₂`.
    public static func quantizeProd(_ x: MLXArray, params p: TurboQuantParams) -> TurboQuantCode {
        let idx = quantizeMSE(x, params: p)
        let r = x - dequantizeMSE(idx, params: p)  // residual in original space
        let signs = MLX.sign(r.matmul(p.qjl.transposed()))  // rows: sign(S·r) → [n, d]
        let norms = MLX.sqrt((r * r).sum(axis: -1, keepDims: true))  // [n, 1]
        return TurboQuantCode(idx: idx, signs: signs, norms: norms)
    }

    /// Algorithm 2 dequantize: base reconstruction + the unbiased QJL residual estimate
    /// `r̂ = (√(π/2)/d)·γ·(Sᵀ·signs)` (E[s·sign⟨s,r⟩] = √(2/π)·r/‖r‖ over the d rows of S).
    public static func dequantizeProd(_ c: TurboQuantCode, params p: TurboQuantParams) -> MLXArray {
        let base = dequantizeMSE(c.idx, params: p)
        let scale = Float((Double.pi / 2).squareRoot() / Double(p.headDim))
        let qjlTerm = c.signs.matmul(p.qjl) * c.norms * scale  // rows: (√(π/2)/d)·γ·(Sᵀ·signs)
        return base + qjlTerm
    }
}

extension TurboQuantCodec {
    /// Non-unit-norm wrapper (paper §1.1): the codebook and Theorem-2 guarantees hold for unit
    /// vectors, so store ‖x‖ per row, quantize the direction x/‖x‖ with the proven `_prod`
    /// codec, and rescale on dequant. Zero rows are safe: the clamped divisor avoids NaN and
    /// the stored ‖x‖ = 0 reconstructs them exactly to zero.
    public static func quantizeProdNormalized(_ x: MLXArray, params p: TurboQuantParams)
        -> TurboQuantCode
    {
        let xNorm = MLX.sqrt((x * x).sum(axis: -1, keepDims: true))  // [n, 1] true input norm
        let u = x / MLX.maximum(xNorm, MLXArray(Float.leastNormalMagnitude))
        let c = quantizeProd(u, params: p)
        return TurboQuantCode(idx: c.idx, signs: c.signs, norms: c.norms, xNorm: xNorm)
    }

    /// Inverse of `quantizeProdNormalized`: unit-sphere reconstruction rescaled by the stored ‖x‖.
    public static func dequantizeProdNormalized(_ c: TurboQuantCode, params p: TurboQuantParams)
        -> MLXArray
    {
        guard let xNorm = c.xNorm else {
            preconditionFailure("dequantizeProdNormalized requires a code from quantizeProdNormalized (xNorm set)")
        }
        return xNorm * dequantizeProd(c, params: p)
    }
}
