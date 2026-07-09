import Foundation
import HarnessCore
import MLX
import MLXRandom

/// Fixed, once-generated TurboQuant parameters for a given head_dim (TurboQuant §"Setup",
/// arXiv:2504.19874): a dense Haar-random orthogonal rotation Π, the QJL Gaussian S, and the
/// 1/√d-scaled Lloyd-Max codebook. Generated once and reused for every K/V vector (the paper's
/// "global parameters" — not per token); seeded for reproducibility.
///
/// Not `Sendable`: `MLXArray` is a non-Sendable class in mlx-swift, so these params stay confined
/// to the inference actor like every other MLX value in SpikeCore (no `@unchecked` escape hatches).
public struct TurboQuantParams {
    public let headDim: Int
    public let baseBits: Int
    public let rotation: MLXArray  // Π  [d, d], Haar-orthogonal (QR of an i.i.d. Gaussian)
    public let qjl: MLXArray  // S  [d, d], i.i.d. N(0,1)
    public let scaledCentroids: MLXArray  // [2^baseBits], ascending Lloyd-Max levels × (1/√d)

    public init(headDim d: Int, baseBits: Int, seed: UInt64) {
        self.headDim = d
        self.baseBits = baseBits
        let (k1, k2) = MLXRandom.split(key: MLXRandom.key(seed))
        // Π = Q from QR of a Gaussian d×d (Haar measure). MLX's QR runs on the CPU backend;
        // this is a one-time setup cost per head_dim, not a per-token op.
        let g = MLXRandom.normal([d, d], key: k1)
        let (q, _) = MLXLinalg.qr(g, stream: .cpu)
        self.rotation = q
        self.qjl = MLXRandom.normal([d, d], key: k2)
        // Post-rotation coordinates of a unit vector are ≈ N(0, 1/d), so the N(0,1) Lloyd-Max
        // levels rescale by 1/√d (reference doc §"The algorithm").
        let c = LloydMaxCodebook.gaussian(bits: baseBits)
            .map { Float($0 / Double(d).squareRoot()) }
        self.scaledCentroids = MLXArray(c)
    }
}
