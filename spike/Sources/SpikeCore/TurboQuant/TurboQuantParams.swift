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

/// The v1 TurboQuant KV tiers: **uniform** `baseBits` Lloyd-Max bits + 1 QJL bit per element,
/// plus the fp16 residual norm γ amortized over head_dim. Honest naming (plan §"Design
/// decisions"): the paper's sub-integer "2.5-bit"/"3.5-bit" labels require the deferred
/// outlier-channel mixing, so the tiers are named by their base bits. They occupy the harness's
/// `tq2.5`/`tq3.5` recording slots — **never report a tqB3 result under a "2.5-bit" label**;
/// tqB2 is ~3 bits/element and tqB3 ~4 bits/element until outlier channels land.
public enum TurboQuantTier: String, Sendable, CaseIterable {
    case tqB2  // 2 base + 1 QJL — occupies the harness "tq2.5" slot (uniform-bit v1)
    case tqB3  // 3 base + 1 QJL — occupies the harness "tq3.5" slot (uniform-bit v1)

    public var baseBits: Int {
        switch self {
        case .tqB2: 2
        case .tqB3: 3
        }
    }

    /// The harness `kvQuant` recording slot this tier fills (v1 is uniform-bit; the sub-integer
    /// slot names await outlier channels — see type doc).
    public var harnessSlot: String {
        switch self {
        case .tqB2: "tq2.5"
        case .tqB3: "tq3.5"
        }
    }

    /// Honest storage cost per KV element: base bits + 1 QJL sign bit + TWO per-row fp16
    /// scalars amortized over head_dim — γ = ‖r‖₂ (Algorithm 2) and ‖x‖ (the Task-6a
    /// non-unit-norm handling, paper §1.1). This is the FORMAT's design width; the v1
    /// in-memory cache stores byte-aligned codes (uint8 idx + int8 sign), so it does not
    /// realize this footprint yet — bit-packing is deferred engineering.
    public func bitsPerElement(headDim: Int) -> Double {
        Double(baseBits) + 1.0 + 32.0 / Double(headDim)
    }
}
