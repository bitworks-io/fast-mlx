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
